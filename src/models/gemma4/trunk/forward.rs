use super::config::{BASE_KV_LAYERS, CONTEXT, EMBED, EPS, HEADS, LAYERS, PER_LAYER, VOCAB};
use super::session::{Gemma4Session, KvLayer};
use super::weights::{is_swa, kv_source_layer};
use crate::core::tensor::GGMLType;
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::Weight;
use crate::ops::{
    attention_value_f32, bf16_to_f32, dot_f32, f16_to_f32, f32_to_bf16, f32_to_f16,
    quantize_q8_0_into, rms_norm, rms_norm_inplace, rope_neox, softmax_inplace,
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
        let rows = assemble_input_rows(rows)?;
        let end = self
            .seq_len
            .checked_add(rows.len())
            .ok_or_else(|| "Gemma4 input length overflow".to_string())?;
        if end > CONTEXT {
            return Err(format!(
                "Gemma4 input length {end} exceeds context {CONTEXT}"
            ));
        }

        for row in &rows {
            let kv_lengths = self
                .kv
                .iter()
                .map(|layer| (layer.keys.len(), layer.values.len()))
                .collect::<Vec<_>>();
            if let Err(error) = self.forward_row(row) {
                for (layer, (key_len, value_len)) in self.kv.iter_mut().zip(kv_lengths) {
                    layer.keys.truncate(key_len);
                    layer.values.truncate(value_len);
                }
                return Err(error);
            }
            self.seq_len += 1;
        }
        Ok(self.scratch.logits.clone())
    }

    fn forward_row(&mut self, row: &AssembledInputRow) -> Result<(), String> {
        let model = self.model;
        let scratch = &mut self.scratch;
        match &row.values {
            InputValues::Token(token) => {
                model
                    .token_embedding
                    .embedding_lookup(*token, &mut scratch.x);
            }
            InputValues::Raw(values) => scratch.x.copy_from_slice(values),
        }
        if row.scale_token_embedding {
            let scale = (EMBED as f32).sqrt();
            for value in &mut scratch.x {
                *value *= scale;
            }
        }
        ensure_finite("gemma4.input", &scratch.x)?;

        model
            .per_layer_token_embedding
            .embedding_lookup(row.per_layer_token, &mut scratch.per_layer);
        let token_scale = (PER_LAYER as f32).sqrt();
        for value in &mut scratch.per_layer {
            *value *= token_scale;
        }
        matmul(
            "per_layer_model_proj.weight",
            &model.per_layer_model_proj,
            &scratch.x,
            &mut scratch.per_layer_projected,
            model.pool(),
            &mut scratch.q8,
            &mut scratch.scales,
        )?;
        let projection_scale = 1.0 / (EMBED as f32).sqrt();
        let merge_scale = 1.0 / 2.0_f32.sqrt();
        for layer in 0..LAYERS {
            let start = layer * PER_LAYER;
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
        ensure_finite("gemma4.per_layer_input", &scratch.per_layer)?;

        let position = self.seq_len;
        for layer_index in 0..LAYERS {
            let layer = &model.layers[layer_index];
            let dim = layer.head_dim;
            let q_width = HEADS * dim;
            let ffn = layer.ffn_gate.n_out;

            checked_rms_norm(
                &format!("blk.{layer_index}.attn_norm.weight"),
                &scratch.x,
                &layer.attn_norm,
                &mut scratch.normed,
            )?;
            matmul(
                &format!("blk.{layer_index}.attn_q.weight"),
                &layer.attn_q,
                &scratch.normed,
                &mut scratch.q[..q_width],
                model.pool(),
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            for query in scratch.q[..q_width].chunks_exact_mut(dim) {
                rms_norm_inplace(query, &layer.attn_q_norm, EPS);
                apply_rope(query, position, dim, layer_index, &model.rope_freqs)?;
            }

            if layer_index < BASE_KV_LAYERS {
                matmul(
                    &format!("blk.{layer_index}.attn_k.weight"),
                    &layer.attn_k,
                    &scratch.normed,
                    &mut scratch.k[..dim],
                    model.pool(),
                    &mut scratch.q8,
                    &mut scratch.scales,
                )?;
                matmul(
                    &format!("blk.{layer_index}.attn_v.weight"),
                    &layer.attn_v,
                    &scratch.normed,
                    &mut scratch.v[..dim],
                    model.pool(),
                    &mut scratch.q8,
                    &mut scratch.scales,
                )?;
                rms_norm_inplace(&mut scratch.k[..dim], &layer.attn_k_norm, EPS);
                rms_norm_inplace(&mut scratch.v[..dim], &scratch.v_norm_weight[..dim], EPS);
                apply_rope(
                    &mut scratch.k[..dim],
                    position,
                    dim,
                    layer_index,
                    &model.rope_freqs,
                )?;
                self.kv[layer_index].append(
                    layer_index,
                    position,
                    &scratch.k[..dim],
                    &scratch.v[..dim],
                )?;
            }

            let cache_layer = kv_source_layer(layer_index);
            attend(
                layer_index,
                position,
                &scratch.q[..q_width],
                &self.kv[cache_layer],
                is_swa(layer_index),
                &mut scratch.attn[..q_width],
                &mut scratch.scores,
                &mut scratch.attention_values,
            )?;
            matmul(
                &format!("blk.{layer_index}.attn_output.weight"),
                &layer.attn_output,
                &scratch.attn[..q_width],
                &mut scratch.projected,
                model.pool(),
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            checked_rms_norm(
                &format!("blk.{layer_index}.post_attention_norm.weight"),
                &scratch.projected,
                &layer.post_attention_norm,
                &mut scratch.down,
            )?;
            for (hidden, attention) in scratch.x.iter_mut().zip(&scratch.down) {
                *hidden += *attention;
            }
            ensure_finite(&format!("gemma4.layer.{layer_index}.attn_out"), &scratch.x)?;
            trace_layer("attn_out", layer_index, &scratch.x);

            checked_rms_norm(
                &format!("blk.{layer_index}.ffn_norm.weight"),
                &scratch.x,
                &layer.ffn_norm,
                &mut scratch.normed,
            )?;
            matmul(
                &format!("blk.{layer_index}.ffn_gate.weight"),
                &layer.ffn_gate,
                &scratch.normed,
                &mut scratch.gate[..ffn],
                model.pool(),
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            matmul(
                &format!("blk.{layer_index}.ffn_up.weight"),
                &layer.ffn_up,
                &scratch.normed,
                &mut scratch.up[..ffn],
                model.pool(),
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            ggml_geglu_fp16_inplace(&mut scratch.gate[..ffn], &scratch.up[..ffn]);
            matmul(
                &format!("blk.{layer_index}.ffn_down.weight"),
                &layer.ffn_down,
                &scratch.gate[..ffn],
                &mut scratch.down,
                model.pool(),
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            checked_rms_norm(
                &format!("blk.{layer_index}.post_ffw_norm.weight"),
                &scratch.down,
                &layer.post_ffw_norm,
                &mut scratch.projected,
            )?;
            for (hidden, ffn) in scratch.x.iter_mut().zip(&scratch.projected) {
                *hidden += *ffn;
            }
            ensure_finite(&format!("gemma4.layer.{layer_index}.ffn_out"), &scratch.x)?;
            trace_layer("ffn_out", layer_index, &scratch.x);

            matmul(
                &format!("blk.{layer_index}.inp_gate.weight"),
                &layer.inp_gate,
                &scratch.x,
                &mut scratch.per_layer_gate,
                model.pool(),
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            let per_start = layer_index * PER_LAYER;
            ggml_geglu_fp16_inplace(
                &mut scratch.per_layer_gate,
                &scratch.per_layer[per_start..per_start + PER_LAYER],
            );
            matmul(
                &format!("blk.{layer_index}.proj.weight"),
                &layer.proj,
                &scratch.per_layer_gate,
                &mut scratch.down,
                model.pool(),
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            checked_rms_norm(
                &format!("blk.{layer_index}.post_norm.weight"),
                &scratch.down,
                &layer.post_norm,
                &mut scratch.projected,
            )?;
            for (hidden, per_layer) in scratch.x.iter_mut().zip(&scratch.projected) {
                *hidden = (*hidden + *per_layer) * layer.output_scale;
            }
            ensure_finite(
                &format!("gemma4.layer.{layer_index}.per_layer_out"),
                &scratch.x,
            )?;
            trace_layer("per_layer_out", layer_index, &scratch.x);
        }

        checked_rms_norm(
            "output_norm.weight",
            &scratch.x,
            &model.output_norm,
            &mut scratch.normed,
        )?;
        ensure_finite("gemma4.final.norm", &scratch.normed)?;
        trace("gemma4.final.norm", None, &scratch.normed);
        matmul(
            "token_embd.weight (tied output)",
            &model.token_embedding,
            &scratch.normed,
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
        Ok(())
    }
}
pub(super) fn assemble_input_rows(
    rows: &[Gemma4InputRow],
) -> Result<Vec<AssembledInputRow>, String> {
    if rows.is_empty() {
        return Err("Gemma4 input rows are empty".into());
    }
    rows.iter()
        .enumerate()
        .map(|(index, row)| match row {
            Gemma4InputRow::Token(token) => {
                validate_token(index, "token", *token)?;
                Ok(AssembledInputRow {
                    values: InputValues::Token(*token),
                    scale_token_embedding: true,
                    per_layer_token: *token,
                })
            }
            Gemma4InputRow::Raw {
                values,
                per_layer_token,
            } => {
                if values.len() != EMBED {
                    return Err(format!(
                        "Gemma4 raw row {index} has length {}; expected {EMBED}",
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
                Ok(AssembledInputRow {
                    values: InputValues::Raw(values.clone()),
                    scale_token_embedding: false,
                    per_layer_token: *per_layer_token,
                })
            }
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
        #[cfg(target_arch = "aarch64")]
        {
            for (result, row) in output.iter_mut().zip(values.chunks_exact(weight.n_in)) {
                *result = dot_f32(row, input, weight.n_in);
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            weight
                .kernel
                .forward(input, output, weight.n_in, weight.n_out);
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
    full_freq_factors: &[f32],
) -> Result<(), String> {
    if values.len() % dim != 0 {
        return Err(format!(
            "blk.{layer} RoPE length {} is not divisible by head width {dim}",
            values.len()
        ));
    }
    if is_swa(layer) {
        rope_neox(values, position, dim, 10_000.0);
        return Ok(());
    }
    if full_freq_factors.len() != dim / 2 {
        return Err(format!(
            "rope_freqs.weight length {}; expected {} for blk.{layer}",
            full_freq_factors.len(),
            dim / 2
        ));
    }
    let theta_scale = 1_000_000.0_f32.powf(-2.0 / dim as f32);
    for head in values.chunks_exact_mut(dim) {
        let mut theta = position as f32;
        for pair in 0..dim / 2 {
            let angle = theta / full_freq_factors[pair];
            let (cosine, sine) = crate::ops::rope::rope_sin_cos(angle);
            let first = head[pair];
            let second = head[pair + dim / 2];
            head[pair] = first.mul_add(cosine, second * -sine);
            head[pair + dim / 2] = first.mul_add(sine, second * cosine);
            theta *= theta_scale;
        }
    }
    Ok(())
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
) -> Result<(), String> {
    let dim = cache.head_dim;
    if query.len() != HEADS * dim || output.len() != HEADS * dim {
        return Err(format!(
            "blk.{layer} attention length mismatch: query {}, output {}, expected {}",
            query.len(),
            output.len(),
            HEADS * dim
        ));
    }
    let rows = position + 1;
    let expected = rows
        .checked_mul(dim)
        .ok_or_else(|| format!("blk.{layer} KV context length overflow"))?;
    if cache.keys.len() != expected || cache.values.len() != expected {
        return Err(format!(
            "blk.{layer} shared KV context mismatch: key {}, value {}, expected {expected}",
            cache.keys.len(),
            cache.values.len()
        ));
    }
    let first = if sliding { rows.saturating_sub(512) } else { 0 };
    let cached = rows - first;
    let padded = cached.div_ceil(256) * 256;
    scores.resize(padded, f32::NEG_INFINITY);
    values.resize(padded, 0.0);

    for (head, query) in query.chunks_exact(dim).enumerate() {
        scores.fill(f32::NEG_INFINITY);
        for (score, token) in scores[..cached].iter_mut().zip(first..rows) {
            let offset = token * dim;
            *score = dot_f32(query, &cache.keys[offset..offset + dim], dim);
        }
        softmax_inplace(scores);
        for dimension in 0..dim {
            values.fill(0.0);
            for (slot, token) in values[..cached].iter_mut().zip(first..rows) {
                *slot = cache.values[token * dim + dimension];
            }
            output[head * dim + dimension] = attention_value_f32(values, scores, cached, padded);
        }
    }
    ensure_finite(&format!("blk.{layer} attention"), output)
}

pub(super) fn softcap(value: f32, cap: f32) -> f32 {
    cap * (value * (1.0 / cap)).tanh()
}

pub(super) fn ggml_geglu_fp16_inplace(gate: &mut [f32], up: &[f32]) {
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

fn ensure_finite(name: &str, values: &[f32]) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{name} produced non-finite value {value:?} at index {index}"
        ));
    }
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
