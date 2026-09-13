use super::config::{CONTEXT, EPS, HEADS, PER_LAYER, VOCAB};
use super::session::{Gemma4Session, KvLayer};
use super::weights::kv_source_layer;
use crate::core::prefill::prefill_chunks;
use crate::core::tensor::GGMLType;
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{PreparedRows, Weight};
use crate::ops::{
    bf16_to_f32, dot_f32, f16_to_f32, f32_to_bf16, f32_to_f16, quantize_q8_0_into, rms_norm,
    rms_norm_inplace, rms_unit_inplace, rope_neox_inplace, softmax_inplace,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Gemma4InputRow {
    Token(u32),
    Raw {
        values: Vec<f32>,
        per_layer_token: u32,
    },
}

#[derive(Debug)]
enum InputValues {
    Token(u32),
    Raw(Vec<f32>),
}

#[derive(Debug)]
pub(super) struct AssembledInputRow {
    values: InputValues,
    pub(super) scale_token_embedding: bool,
    pub(super) per_layer_token: u32,
}

impl Gemma4Session<'_> {
    pub fn forward_rows(&mut self, rows: &[Gemma4InputRow]) -> Result<Vec<f32>, String> {
        let end = self
            .seq_len
            .checked_add(rows.len())
            .ok_or_else(|| "Gemma4 input length overflow".to_string())?;
        if end > CONTEXT {
            return Err(format!(
                "Gemma4 input length {end} exceeds context {CONTEXT}"
            ));
        }
        validate_input_rows(rows, self.model.config.embd)?;

        let mut chunks = prefill_chunks(rows.len(), self.prefill_batch_size).peekable();
        while let Some(range) = chunks.next() {
            let chunk_rows = assemble_validated_input_rows(&rows[range.clone()]);
            let kv_lengths = self
                .kv
                .iter()
                .map(|layer| (layer.keys.len(), layer.values.len()))
                .collect::<Vec<_>>();
            if let Err(error) = self.forward_chunk_inner(&chunk_rows, chunks.peek().is_none()) {
                for (layer, (key_len, value_len)) in self.kv.iter_mut().zip(kv_lengths) {
                    layer.keys.truncate(key_len);
                    layer.values.truncate(value_len);
                }
                return Err(error);
            }
            self.seq_len += range.len();
        }
        Ok(self.scratch.logits.clone())
    }

    pub(super) fn forward_chunk(&mut self, rows: &[AssembledInputRow]) -> Result<(), String> {
        self.forward_chunk_inner(rows, true)
    }

    fn forward_chunk_inner(
        &mut self,
        rows: &[AssembledInputRow],
        project_logits: bool,
    ) -> Result<(), String> {
        if rows.is_empty() || rows.len() > self.prefill_batch_size {
            return Err("Invalid Gemma4 prefill chunk size".into());
        }

        let model = self.model;
        let cfg = &model.config;
        let scratch = &mut self.scratch;
        let row_count = rows.len();
        let embd = cfg.embd;
        let x_len = row_count * embd;
        let per_layer_all = cfg.per_layer_all();
        let per_layer_len = row_count * per_layer_all;

        for (index, row) in rows.iter().enumerate() {
            let x = &mut scratch.x[index * embd..(index + 1) * embd];
            match &row.values {
                InputValues::Token(token) => model.token_embedding.embedding_lookup(*token, x),
                InputValues::Raw(values) => x.copy_from_slice(values),
            }
            if row.scale_token_embedding {
                let scale = (embd as f32).sqrt();
                for value in x {
                    *value *= scale;
                }
            }
            model.per_layer_token_embedding.embedding_lookup(
                row.per_layer_token,
                &mut scratch.per_layer[index * per_layer_all..(index + 1) * per_layer_all],
            );
        }
        ensure_finite("gemma4.input", &scratch.x[..x_len])?;

        let token_scale = (PER_LAYER as f32).sqrt();
        for value in &mut scratch.per_layer[..per_layer_len] {
            *value *= token_scale;
        }
        matmul_rows(
            "per_layer_model_proj.weight",
            &model.per_layer_model_proj,
            &scratch.x[..x_len],
            &mut scratch.per_layer_projected[..per_layer_len],
            row_count,
            model.pool(),
            &mut scratch.prepared,
            &mut scratch.q8,
            &mut scratch.scales,
        )?;
        let projection_scale = 1.0 / (embd as f32).sqrt();
        let merge_scale = 1.0 / 2.0_f32.sqrt();
        for row in 0..row_count {
            for layer in 0..cfg.layers {
                let start = row * per_layer_all + layer * PER_LAYER;
                let end = start + PER_LAYER;
                let projected = &mut scratch.per_layer_projected[start..end];
                for value in projected.iter_mut() {
                    *value *= projection_scale;
                }
                rms_norm_inplace(projected, &model.per_layer_proj_norm, EPS);
                for (target, projected) in scratch.per_layer[start..end].iter_mut().zip(projected) {
                    *target = (*target + *projected) * merge_scale;
                }
            }
        }
        ensure_finite(
            "gemma4.per_layer_input",
            &scratch.per_layer[..per_layer_len],
        )?;

        let base_position = self.seq_len;
        let base_kv = cfg.base_kv_layers();
        for layer_index in 0..cfg.layers {
            let layer = &model.layers[layer_index];
            let dim = layer.head_dim;
            let kv_width = cfg.kv_heads * dim;
            let q_width = HEADS * dim;
            let ffn = layer.ffn_gate.n_out;

            for (input, output) in scratch.x[..x_len]
                .chunks_exact(embd)
                .zip(scratch.normed[..x_len].chunks_exact_mut(embd))
            {
                checked_rms_norm(
                    &format!("blk.{layer_index}.attn_norm.weight"),
                    input,
                    &layer.attn_norm,
                    output,
                )?;
            }
            let q_len = row_count * q_width;
            let kv_len = row_count * kv_width;
            if layer_index < base_kv {
                matmul_group_rows(
                    [
                        (
                            &format!("blk.{layer_index}.attn_q.weight"),
                            &layer.attn_q,
                            &mut scratch.q[..q_len],
                        ),
                        (
                            &format!("blk.{layer_index}.attn_k.weight"),
                            &layer.attn_k,
                            &mut scratch.k[..kv_len],
                        ),
                        (
                            &format!("blk.{layer_index}.attn_v.weight"),
                            &layer.attn_v,
                            &mut scratch.v[..kv_len],
                        ),
                    ],
                    &scratch.normed[..x_len],
                    row_count,
                    model.pool(),
                    &mut scratch.prepared,
                    &mut scratch.q8,
                    &mut scratch.scales,
                )?;
                ensure_finite(
                    &format!("blk.{layer_index}.attn_q.weight"),
                    &scratch.q[..q_len],
                )?;
                ensure_finite(
                    &format!("blk.{layer_index}.attn_k.weight"),
                    &scratch.k[..kv_len],
                )?;
                ensure_finite(
                    &format!("blk.{layer_index}.attn_v.weight"),
                    &scratch.v[..kv_len],
                )?;
            } else {
                matmul_rows(
                    &format!("blk.{layer_index}.attn_q.weight"),
                    &layer.attn_q,
                    &scratch.normed[..x_len],
                    &mut scratch.q[..q_len],
                    row_count,
                    model.pool(),
                    &mut scratch.prepared,
                    &mut scratch.q8,
                    &mut scratch.scales,
                )?;
            }

            for row in 0..row_count {
                let position = base_position + row;
                let query = &mut scratch.q[row * q_width..(row + 1) * q_width];
                for head in query.chunks_exact_mut(dim) {
                    rms_norm_inplace(head, &layer.attn_q_norm, EPS);
                }
                apply_rope(
                    query,
                    position,
                    dim,
                    layer_index,
                    cfg.is_swa(layer_index),
                    &model.rope_freqs,
                )?;
            }

            if layer_index < base_kv {
                for row in 0..row_count {
                    let position = base_position + row;
                    let key = &mut scratch.k[row * kv_width..(row + 1) * kv_width];
                    let value = &mut scratch.v[row * kv_width..(row + 1) * kv_width];
                    for kv_head in 0..cfg.kv_heads {
                        let offset = kv_head * dim;
                        rms_norm_inplace(&mut key[offset..offset + dim], &layer.attn_k_norm, EPS);
                        rms_unit_inplace(&mut value[offset..offset + dim], EPS);
                    }
                    apply_rope(
                        key,
                        position,
                        dim,
                        layer_index,
                        cfg.is_swa(layer_index),
                        &model.rope_freqs,
                    )?;
                    self.kv[layer_index].append(layer_index, position, key, value)?;
                }
            }

            let cache_layer = kv_source_layer(cfg, layer_index);
            for row in 0..row_count {
                let position = base_position + row;
                attend(
                    layer_index,
                    position,
                    &scratch.q[row * q_width..(row + 1) * q_width],
                    &self.kv[cache_layer],
                    cfg.is_swa(layer_index),
                    &mut scratch.attn[row * q_width..(row + 1) * q_width],
                    &mut scratch.scores,
                    &mut scratch.attention_values,
                    model.pool(),
                )?;
            }
            matmul_rows(
                &format!("blk.{layer_index}.attn_output.weight"),
                &layer.attn_output,
                &scratch.attn[..q_len],
                &mut scratch.projected[..x_len],
                row_count,
                model.pool(),
                &mut scratch.prepared,
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            for row in 0..row_count {
                let projected = &scratch.projected[row * embd..(row + 1) * embd];
                let down = &mut scratch.down[row * embd..(row + 1) * embd];
                checked_rms_norm(
                    &format!("blk.{layer_index}.post_attention_norm.weight"),
                    projected,
                    &layer.post_attention_norm,
                    down,
                )?;
                let hidden = &mut scratch.x[row * embd..(row + 1) * embd];
                for (hidden, attention) in hidden.iter_mut().zip(down) {
                    *hidden += *attention;
                }
                ensure_finite(&format!("gemma4.layer.{layer_index}.attn_out"), hidden)?;
                trace_layer("attn_out", layer_index, hidden);
            }

            for (input, output) in scratch.x[..x_len]
                .chunks_exact(embd)
                .zip(scratch.normed[..x_len].chunks_exact_mut(embd))
            {
                checked_rms_norm(
                    &format!("blk.{layer_index}.ffn_norm.weight"),
                    input,
                    &layer.ffn_norm,
                    output,
                )?;
            }
            let ffn_len = row_count * ffn;
            matmul_group_rows(
                [
                    (
                        &format!("blk.{layer_index}.ffn_gate.weight"),
                        &layer.ffn_gate,
                        &mut scratch.gate[..ffn_len],
                    ),
                    (
                        &format!("blk.{layer_index}.ffn_up.weight"),
                        &layer.ffn_up,
                        &mut scratch.up[..ffn_len],
                    ),
                ],
                &scratch.normed[..x_len],
                row_count,
                model.pool(),
                &mut scratch.prepared,
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            ensure_finite(
                &format!("blk.{layer_index}.ffn_gate.weight"),
                &scratch.gate[..ffn_len],
            )?;
            ensure_finite(
                &format!("blk.{layer_index}.ffn_up.weight"),
                &scratch.up[..ffn_len],
            )?;
            for (gate, up) in scratch.gate[..ffn_len]
                .chunks_exact_mut(ffn)
                .zip(scratch.up[..ffn_len].chunks_exact(ffn))
            {
                ggml_geglu_fp16_inplace(gate, up);
            }
            matmul_rows(
                &format!("blk.{layer_index}.ffn_down.weight"),
                &layer.ffn_down,
                &scratch.gate[..ffn_len],
                &mut scratch.down[..x_len],
                row_count,
                model.pool(),
                &mut scratch.prepared,
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            for row in 0..row_count {
                let down = &scratch.down[row * embd..(row + 1) * embd];
                let projected = &mut scratch.projected[row * embd..(row + 1) * embd];
                checked_rms_norm(
                    &format!("blk.{layer_index}.post_ffw_norm.weight"),
                    down,
                    &layer.post_ffw_norm,
                    projected,
                )?;
                let hidden = &mut scratch.x[row * embd..(row + 1) * embd];
                for (hidden, ffn) in hidden.iter_mut().zip(projected) {
                    *hidden += *ffn;
                }
                ensure_finite(&format!("gemma4.layer.{layer_index}.ffn_out"), hidden)?;
                trace_layer("ffn_out", layer_index, hidden);
            }

            matmul_rows(
                &format!("blk.{layer_index}.inp_gate.weight"),
                &layer.inp_gate,
                &scratch.x[..x_len],
                &mut scratch.per_layer_gate[..row_count * PER_LAYER],
                row_count,
                model.pool(),
                &mut scratch.prepared,
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            for row in 0..row_count {
                let start = row * per_layer_all + layer_index * PER_LAYER;
                ggml_geglu_fp16_inplace(
                    &mut scratch.per_layer_gate[row * PER_LAYER..(row + 1) * PER_LAYER],
                    &scratch.per_layer[start..start + PER_LAYER],
                );
            }
            matmul_rows(
                &format!("blk.{layer_index}.proj.weight"),
                &layer.proj,
                &scratch.per_layer_gate[..row_count * PER_LAYER],
                &mut scratch.down[..x_len],
                row_count,
                model.pool(),
                &mut scratch.prepared,
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            for row in 0..row_count {
                let down = &scratch.down[row * embd..(row + 1) * embd];
                let projected = &mut scratch.projected[row * embd..(row + 1) * embd];
                checked_rms_norm(
                    &format!("blk.{layer_index}.post_norm.weight"),
                    down,
                    &layer.post_norm,
                    projected,
                )?;
                let hidden = &mut scratch.x[row * embd..(row + 1) * embd];
                for (hidden, per_layer) in hidden.iter_mut().zip(projected) {
                    *hidden = (*hidden + *per_layer) * layer.output_scale;
                }
                ensure_finite(&format!("gemma4.layer.{layer_index}.per_layer_out"), hidden)?;
                trace_layer("per_layer_out", layer_index, hidden);
            }
        }

        if project_logits {
            let last = &scratch.x[(row_count - 1) * embd..row_count * embd];
            checked_rms_norm(
                "output_norm.weight",
                last,
                &model.output_norm,
                &mut scratch.normed[..embd],
            )?;
            ensure_finite("gemma4.final.norm", &scratch.normed[..embd])?;
            trace("gemma4.final.norm", None, &scratch.normed[..embd]);
            matmul(
                "token_embd.weight (tied output)",
                &model.token_embedding,
                &scratch.normed[..embd],
                &mut scratch.logits,
                model.pool(),
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            for logit in &mut scratch.logits {
                *logit = softcap(*logit, model.config.logit_softcap);
            }
            ensure_finite("gemma4.logits", &scratch.logits)?;
            trace("gemma4.logits", None, &scratch.logits);
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn matmul_rows(
    name: &str,
    weight: &Weight<'_>,
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    pool: &ComputePool,
    prepared: &mut PreparedRows,
    q8: &mut [u8],
    scales: &mut [f32],
) -> Result<(), String> {
    if input.len() != rows * weight.n_in || output.len() != rows * weight.n_out {
        return Err(format!("Invalid {name} batched matmul lengths"));
    }
    if name == "per_layer_model_proj.weight" || weight.ggml_type == GGMLType::F32 {
        for (input, output) in input
            .chunks_exact(weight.n_in)
            .zip(output.chunks_exact_mut(weight.n_out))
        {
            matmul(name, weight, input, output, pool, q8, scales)?;
        }
        return Ok(());
    }
    prepared.prepare(
        input,
        rows,
        weight.n_in,
        weight.needs_q8_0_activation(),
        weight.uses_q8_k(),
    )?;
    prepared.matmul(weight, input, output, pool)?;
    ensure_finite(name, output)
}

#[allow(clippy::too_many_arguments)]
fn matmul_group_rows<const N: usize>(
    projections: [(&str, &Weight<'_>, &mut [f32]); N],
    input: &[f32],
    rows: usize,
    pool: &ComputePool,
    prepared: &mut PreparedRows,
    q8: &mut [u8],
    scales: &mut [f32],
) -> Result<(), String> {
    if projections.iter().any(|(_, weight, _)| {
        weight.ggml_type == GGMLType::F32 || weight.ggml_type == GGMLType::BF16
    }) {
        for (name, weight, output) in projections {
            matmul_rows(
                name, weight, input, output, rows, pool, prepared, q8, scales,
            )?;
        }
        return Ok(());
    }
    let need_q8 = projections
        .iter()
        .any(|(_, weight, _)| weight.needs_q8_0_activation());
    let need_q8k = projections.iter().any(|(_, weight, _)| weight.uses_q8_k());
    prepared.prepare(input, rows, projections[0].1.n_in, need_q8, need_q8k)?;
    prepared.matmul_group(
        input,
        projections.map(|(_, weight, output)| (weight, output)),
        pool,
    )
}
pub(super) fn assemble_input_rows(
    rows: &[Gemma4InputRow],
    embd: usize,
) -> Result<Vec<AssembledInputRow>, String> {
    validate_input_rows(rows, embd)?;
    Ok(assemble_validated_input_rows(rows))
}

fn validate_input_rows(rows: &[Gemma4InputRow], embd: usize) -> Result<(), String> {
    if rows.is_empty() {
        return Err("Gemma4 input rows are empty".into());
    }
    for (index, row) in rows.iter().enumerate() {
        match row {
            Gemma4InputRow::Token(token) => {
                validate_token(index, "token", *token)?;
            }
            Gemma4InputRow::Raw {
                values,
                per_layer_token,
            } => {
                if values.len() != embd {
                    return Err(format!(
                        "Gemma4 raw row {index} has length {}; expected {embd}",
                        values.len()
                    ));
                }
                if let Some((value_index, value)) = values
                    .iter()
                    .enumerate()
                    .find(|(_, value)| !value.is_finite())
                {
                    return Err(format!(
                        "Gemma4 raw row {index} has non-finite value {value:?} at index {value_index}"
                    ));
                }
                validate_token(index, "per-layer token", *per_layer_token)?;
            }
        }
    }
    Ok(())
}

fn assemble_validated_input_rows(rows: &[Gemma4InputRow]) -> Vec<AssembledInputRow> {
    rows.iter()
        .map(|row| match row {
            Gemma4InputRow::Token(token) => AssembledInputRow {
                values: InputValues::Token(*token),
                scale_token_embedding: true,
                per_layer_token: *token,
            },
            Gemma4InputRow::Raw {
                values,
                per_layer_token,
            } => AssembledInputRow {
                values: InputValues::Raw(values.clone()),
                scale_token_embedding: false,
                per_layer_token: *per_layer_token,
            },
        })
        .collect()
}

fn validate_token(row: usize, kind: &str, token: u32) -> Result<(), String> {
    if token as usize >= VOCAB {
        return Err(format!(
            "Gemma4 row {row} {kind} ID {token} exceeds vocabulary {VOCAB}"
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn matmul(
    name: &str,
    weight: &Weight<'_>,
    input: &[f32],
    output: &mut [f32],
    pool: &ComputePool,
    q8: &mut [u8],
    scales: &mut [f32],
) -> Result<(), String> {
    if input.len() != weight.n_in || output.len() != weight.n_out {
        return Err(format!(
            "Invalid {name} matmul lengths: input {}, output {}; expected {}, {}",
            input.len(),
            output.len(),
            weight.n_in,
            weight.n_out
        ));
    }
    if name == "per_layer_model_proj.weight" {
        if weight.ggml_type != GGMLType::BF16 {
            return Err(format!("{name} requires BF16 weight"));
        }
        let bytes = weight
            .kernel
            .bf16_bytes()
            .ok_or_else(|| format!("Invalid {name} BF16 kernel"))?;
        gemma4_bf16_projection_matmul(bytes, input, output, pool, q8)?;
    } else if weight.ggml_type == GGMLType::F32 {
        let values = weight
            .kernel
            .f32_slice()
            .ok_or_else(|| format!("Invalid {name} F32 kernel"))?;
        let expected = weight
            .n_in
            .checked_mul(weight.n_out)
            .ok_or_else(|| format!("Invalid {name} F32 weight size"))?;
        if values.len() != expected {
            return Err(format!(
                "Invalid {name} F32 weight length: expected {expected}, got {}",
                values.len()
            ));
        }
        for (result, row) in output.iter_mut().zip(values.chunks_exact(weight.n_in)) {
            *result = dot_f32(row, input, weight.n_in);
        }
    } else {
        let blocks = weight.n_in.div_ceil(32);
        if q8.len() < weight.n_in || scales.len() < blocks {
            return Err(format!("Invalid {name} activation scratch length"));
        }
        quantize_q8_0_into(
            input,
            weight.n_in,
            &mut q8[..weight.n_in],
            &mut scales[..blocks],
        );
        let input_ptr = input.as_ptr();
        let q8_ptr = q8.as_ptr();
        let scales_ptr = scales.as_ptr();
        let output_ptr = output.as_mut_ptr();
        pool.compute(|thread, threads| unsafe {
            weight.kernel.forward_prepared(
                std::slice::from_raw_parts(input_ptr, weight.n_in),
                std::slice::from_raw_parts(q8_ptr, weight.n_in),
                std::slice::from_raw_parts(scales_ptr, blocks),
                None,
                std::slice::from_raw_parts_mut(output_ptr, weight.n_out),
                weight.n_in,
                weight.n_out,
                thread,
                threads,
            );
        });
    }
    ensure_finite(name, output)
}

fn gemma4_bf16_projection_matmul(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    pool: &ComputePool,
    input_bf16: &mut [u8],
) -> Result<(), String> {
    let input_bytes = input
        .len()
        .checked_mul(2)
        .ok_or_else(|| "Gemma4 BF16 projection input byte size overflow".to_owned())?;
    let weight_bytes = input_bytes
        .checked_mul(output.len())
        .ok_or_else(|| "Gemma4 BF16 projection weight byte size overflow".to_owned())?;
    if input_bf16.len() < input_bytes {
        return Err("Invalid Gemma4 BF16 projection storage length".to_owned());
    }
    if weight.len() != weight_bytes {
        return Err(format!(
            "Invalid Gemma4 BF16 projection weight storage length: expected {weight_bytes} bytes, got {}",
            weight.len()
        ));
    }

    for (bytes, value) in input_bf16[..input_bytes].chunks_exact_mut(2).zip(input) {
        bytes.copy_from_slice(&f32_to_bf16(*value).to_le_bytes());
    }

    let n_in = input.len();
    let n_out = output.len();
    let weight_ptr = weight.as_ptr();
    let input_ptr = input_bf16.as_ptr();
    let output_ptr = output.as_mut_ptr();
    pool.compute(|thread, threads| unsafe {
        let start = n_out * thread / threads;
        let end = n_out * (thread + 1) / threads;
        for row in start..end {
            let mut sum = 0.0f64;
            for column in 0..n_in {
                let weight_offset = (row * n_in + column) * 2;
                let input_offset = column * 2;
                let weight_bits = u16::from_le_bytes([
                    *weight_ptr.add(weight_offset),
                    *weight_ptr.add(weight_offset + 1),
                ]);
                let input_bits = u16::from_le_bytes([
                    *input_ptr.add(input_offset),
                    *input_ptr.add(input_offset + 1),
                ]);
                let product = bf16_to_f32(weight_bits) * bf16_to_f32(input_bits);
                sum += f64::from(product);
            }
            *output_ptr.add(row) = sum as f32;
        }
    });
    Ok(())
}

fn checked_rms_norm(
    name: &str,
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    if input.len() != weight.len() || input.len() != output.len() {
        return Err(format!(
            "Invalid {name} RMS lengths: input {}, weight {}, output {}",
            input.len(),
            weight.len(),
            output.len()
        ));
    }
    rms_norm(input, weight, output, EPS);
    ensure_finite(name, output)
}

fn apply_rope(
    values: &mut [f32],
    position: usize,
    dim: usize,
    layer: usize,
    sliding: bool,
    full_freq_factors: &[f32],
) -> Result<(), String> {
    if values.len() % dim != 0 {
        return Err(format!(
            "blk.{layer} RoPE length {} is not divisible by head width {dim}",
            values.len()
        ));
    }
    if sliding {
        // `rope_neox_inplace` internally loops over `n_heads = x.len() / dim`
        // with one cached sin/cos table, so passing the full buffer is
        // strictly cheaper than calling it once per head (which would
        // rebuild the same table 8 times for E2B/E4B).
        rope_neox_inplace(values, position, dim, 10_000.0);
        return Ok(());
    }
    if full_freq_factors.len() != dim / 2 {
        return Err(format!(
            "rope_freqs.weight length {}; expected {} for blk.{layer}",
            full_freq_factors.len(),
            dim / 2
        ));
    }
    apply_rope_full(values, position, dim, full_freq_factors);
    Ok(())
}

/// Full-attention RoPE: pre-compute the sin/cos table once from
/// `factors` (which already encodes `θ_i = 1 / freq_base^(2i/dim)`)
/// and apply the standard neox rotation to every head in one pass.
fn apply_rope_full(values: &mut [f32], position: usize, dim: usize, factors: &[f32]) {
    let half = dim / 2;
    let n_heads = values.len() / dim;
    if half == 0 || n_heads == 0 {
        return;
    }
    let pos_f = position as f32;
    let mut cos_table = vec![0.0f32; half];
    let mut sin_table = vec![0.0f32; half];
    for (i, factor) in factors[..half].iter().enumerate() {
        let angle = pos_f / factor;
        let (c, s) = crate::ops::rope::rope_sin_cos(angle);
        cos_table[i] = c;
        sin_table[i] = s;
    }
    // Same rotation formula as the scalar path; AVX2-equivalent of the
    // inner FMA pattern will be picked up by the compiler when
    // targeting AVX2. The key win is one sin/cos table per call
    // (vs `n_heads × half` calls in the old loop).
    for h in 0..n_heads {
        let base = h * dim;
        // split_at_mut avoids the borrow-conflict that the slice
        // pair `(values[..base+half], values[base+half..])` would
        // create (overlap at base+half).
        let (lo, hi) = values[base..base + dim].split_at_mut(half);
        for i in 0..half {
            let (c, s) = (cos_table[i], sin_table[i]);
            let first = lo[i];
            let second = hi[i];
            lo[i] = first.mul_add(c, second * -s);
            hi[i] = first.mul_add(s, second * c);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attend(
    layer: usize,
    position: usize,
    query: &[f32],
    cache: &KvLayer,
    sliding: bool,
    output: &mut [f32],
    scores: &mut Vec<f32>,
    values: &mut Vec<f32>,
    pool: &ComputePool,
) -> Result<(), String> {
    let dim = cache.head_dim;
    let row_width = cache.row_width;
    let group_size = cache.group_size;
    let q_width = HEADS * dim;
    if query.len() != q_width || output.len() != q_width {
        return Err(format!(
            "blk.{layer} attention length mismatch: query {}, output {}, expected {}",
            query.len(),
            output.len(),
            q_width
        ));
    }
    let rows = position + 1;
    let expected = rows
        .checked_mul(row_width)
        .ok_or_else(|| format!("blk.{layer} KV context length overflow"))?;
    if cache.keys.len() != cache.values.len()
        || cache.keys.len() < expected
        || !cache.keys.len().is_multiple_of(row_width)
    {
        return Err(format!(
            "blk.{layer} shared KV context mismatch: key {}, value {}, expected at least {expected}",
            cache.keys.len(),
            cache.values.len()
        ));
    }
    let first = if sliding { rows.saturating_sub(512) } else { 0 };
    let cached = rows - first;
    let padded = cached.div_ceil(256) * 256;
    scores.resize(padded, f32::NEG_INFINITY);
    values.resize(padded, 0.0);

    // Parallelize per-head over the compute pool. The V pass iterates
    // `token` outer / `dimension` inner: V at fixed token is contiguous in
    // memory (stride 1 in `dim`), so each token-load is cache-friendly, and
    // the inner dim loop is FMA-fused (AVX2 `_mm256_fmadd_ps`) over
    // `score[token] * V[token, dim]`. This drops the per-dim `head_values`
    // gather + `dot_f32` and avoids the strided V reads of the old code.
    let query_ptr = query.as_ptr();
    let keys_ptr = cache.keys.as_ptr();
    let values_ptr = cache.values.as_ptr();
    let output_ptr = output.as_mut_ptr();
    pool.compute(move |ith, nth| {
        let h_step = (HEADS + nth - 1) / nth;
        let h_start = (ith * h_step).min(HEADS);
        let h_end = (h_start + h_step).min(HEADS);
        let mut head_scores = vec![f32::NEG_INFINITY; padded];
        for head in h_start..h_end {
            let query_head = unsafe { std::slice::from_raw_parts(query_ptr.add(head * dim), dim) };
            let kv_head = head / group_size;
            let kv_offset = kv_head * dim;
            head_scores.fill(f32::NEG_INFINITY);
            for (score, token) in head_scores[..cached].iter_mut().zip(first..rows) {
                let offset = token * row_width + kv_offset;
                let key = unsafe { std::slice::from_raw_parts(keys_ptr.add(offset), dim) };
                *score = dot_f32(query_head, key, dim);
            }
            softmax_inplace(&mut head_scores);
            let head_output =
                unsafe { std::slice::from_raw_parts_mut(output_ptr.add(head * dim), dim) };
            unsafe {
                attend_v(
                    values_ptr,
                    head_scores.as_ptr(),
                    row_width,
                    kv_offset,
                    first,
                    cached,
                    dim,
                    head_output,
                );
            }
        }
    });
    ensure_finite(&format!("blk.{layer} attention"), output)
}

/// Fused `output[d] += Σ_t scores[t] * V[t, d]` over `[first, first+cached)`.
/// Outer loop iterates `t` so each V load is a contiguous `dim`-length chunk
/// (stride 1 in `dim`). Inner loop uses the SIMD FMA of the host to fuse
/// `output[d] += score * V[t, d]`.
#[inline]
unsafe fn attend_v(
    values_ptr: *const f32,
    scores_ptr: *const f32,
    row_width: usize,
    kv_offset: usize,
    first: usize,
    cached: usize,
    dim: usize,
    head_output: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return attend_v_avx2(
                values_ptr,
                scores_ptr,
                row_width,
                kv_offset,
                first,
                cached,
                dim,
                head_output,
            );
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return attend_v_neon(
                values_ptr,
                scores_ptr,
                row_width,
                kv_offset,
                first,
                cached,
                dim,
                head_output,
            );
        }
    }
    attend_v_scalar(
        values_ptr,
        scores_ptr,
        row_width,
        kv_offset,
        first,
        cached,
        dim,
        head_output,
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn attend_v_avx2(
    values_ptr: *const f32,
    scores_ptr: *const f32,
    row_width: usize,
    kv_offset: usize,
    first: usize,
    cached: usize,
    dim: usize,
    head_output: &mut [f32],
) {
    use std::arch::x86_64::*;
    head_output.fill(0.0);
    for i in 0..cached {
        let token = first + i;
        let v_base = values_ptr.add(token * row_width + kv_offset);
        let score = *scores_ptr.add(i);
        let vscore = _mm256_set1_ps(score);
        let mut d = 0;
        while d + 8 <= dim {
            let vo = _mm256_loadu_ps(head_output.as_ptr().add(d));
            let vd = _mm256_loadu_ps(v_base.add(d));
            _mm256_storeu_ps(
                head_output.as_mut_ptr().add(d),
                _mm256_fmadd_ps(vscore, vd, vo),
            );
            d += 8;
        }
        while d < dim {
            *head_output.as_mut_ptr().add(d) += score * *v_base.add(d);
            d += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn attend_v_neon(
    values_ptr: *const f32,
    scores_ptr: *const f32,
    row_width: usize,
    kv_offset: usize,
    first: usize,
    cached: usize,
    dim: usize,
    head_output: &mut [f32],
) {
    use std::arch::aarch64::*;
    head_output.fill(0.0);
    for i in 0..cached {
        let token = first + i;
        let v_base = values_ptr.add(token * row_width + kv_offset);
        let score = *scores_ptr.add(i);
        let vscore = vdupq_n_f32(score);
        let mut d = 0;
        while d + 4 <= dim {
            let vo = vld1q_f32(head_output.as_ptr().add(d));
            let vd = vld1q_f32(v_base.add(d));
            vst1q_f32(head_output.as_mut_ptr().add(d), vfmaq_f32(vo, vscore, vd));
            d += 4;
        }
        while d < dim {
            *head_output.as_mut_ptr().add(d) += score * *v_base.add(d);
            d += 1;
        }
    }
}

unsafe fn attend_v_scalar(
    values_ptr: *const f32,
    scores_ptr: *const f32,
    row_width: usize,
    kv_offset: usize,
    first: usize,
    cached: usize,
    dim: usize,
    head_output: &mut [f32],
) {
    head_output.fill(0.0);
    for i in 0..cached {
        let token = first + i;
        let v_base = values_ptr.add(token * row_width + kv_offset);
        let score = *scores_ptr.add(i);
        for d in 0..dim {
            *head_output.as_mut_ptr().add(d) += score * *v_base.add(d);
        }
    }
}

pub(super) fn softcap(value: f32, cap: f32) -> f32 {
    cap * (value * (1.0 / cap)).tanh()
}

pub(super) fn ggml_geglu_fp16_inplace(gate: &mut [f32], up: &[f32]) {
    // Scalar reference; matches llama.cpp GEGLU with f16 intermediate.
    //
    // === TODO-002: SIMD GeGLU (`tanh_approx`) ===
    //
    // An AVX2+F16C SIMD version was prototyped (see git history) but
    // reverted because tanh lacks native SIMD on x86 and the
    // Padé [3/2] rational `tanh(x) ≈ x*(27+x²)/(27+9x²)` diverges for
    // |x| > 5. Three mitigation strategies were tried, none ideal:
    //
    //   1. Clamp arg to ±5 before Padé — works for typical gelu inputs
    //      (arg stays in ~(-3.6, 3.6) when x ∈ (-3, 3)) but loses
    //      ~0.5% accuracy once clamped because tanh saturates near ±1
    //      and the rational diverges; clamping trades divergence for
    //      a hard saturation step.
    //   2. Padé [5/4] / [7/6] higher-order — more accurate but adds
    //      4-6 extra FMA per lane, eroding the SIMD win.
    //   3. Schraudolph-style fast exp via bit manipulation — too
    //      imprecise (5-10% error) for gate values outside (-2, 2).
    //
    // The SIMD version showed ~0.5% drift on the gelu output (after
    // f16 round-trip) and caused occasional top-K token flips on a few
    // gemma4 reference prompts. The scalar path remains the safe
    // reference. See `docs/TODO.md` TODO-002 for full analysis and
    // recovery plan (likely a `[7/6]` padé + per-call opt-in feature
    // flag once precision is characterised against the llama.cpp
    // pinned Oracle).
    const GELU_COEF_A: f32 = 0.044715;
    const SQRT_2_OVER_PI: f32 = 0.79788456080286535587989211986876;

    assert_eq!(gate.len(), up.len());
    for (gate, up) in gate.iter_mut().zip(up) {
        let x = *gate;
        *gate = if x <= -10.0 {
            0.0
        } else if x >= 10.0 {
            x * up
        } else {
            let x = f16_to_f32(f32_to_f16(x));
            let gelu =
                0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * x.mul_add(GELU_COEF_A * x, 1.0)).tanh());
            f16_to_f32(f32_to_f16(gelu)) * up
        };
    }
}

#[cfg_attr(
    not(debug_assertions),
    allow(dead_code, unused_variables),
    inline(always)
)]
fn ensure_finite(name: &str, values: &[f32]) -> Result<(), String> {
    #[cfg(debug_assertions)]
    {
        if let Some((index, value)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "{name} produced non-finite value {value:?} at index {index}"
            ));
        }
    }
    let _ = (name, values);
    Ok(())
}

fn trace_layer(stage: &str, layer: usize, values: &[f32]) {
    trace(
        &format!("gemma4.layer.{layer}.{stage}"),
        Some(layer),
        values,
    );
}

#[cfg(feature = "parity-trace")]
fn trace(name: &str, layer: Option<usize>, values: &[f32]) {
    crate::parity_trace::report(crate::parity_trace::checkpoint(
        name,
        layer,
        &[1, values.len()],
        values,
    ));
}

#[cfg(not(feature = "parity-trace"))]
fn trace(_name: &str, _layer: Option<usize>, _values: &[f32]) {}
