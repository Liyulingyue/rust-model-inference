use super::config::{EPS, PER_LAYER, VOCAB};
use super::session::{Gemma4PrefillLinear, Gemma4Session, KvLayer};
use super::weights::{kv_source_layer, Gemma4Model};
use crate::core::prefill::prefill_chunks;
use crate::core::tensor::GGMLType;
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{PreparedRows, Weight};
use crate::ops::{
    dot_f32, quantize_q8_0_into, rms_norm, rms_norm_inplace, rms_unit_inplace,
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
        let n_ctx = self.model.config.n_ctx;
        let end = self
            .seq_len
            .checked_add(rows.len())
            .ok_or_else(|| "Gemma4 input length overflow".to_string())?;
        if end > n_ctx {
            return Err(format!("Gemma4 input length {end} exceeds context {n_ctx}"));
        }
        validate_input_rows(rows, self.model.config.embd)?;

        // The app uses subsequent singleton calls for decode. Keep all chunks
        // of the initial prompt on the same backend, including batch size one.
        let prefill = self.seq_len == 0 || rows.len() != 1;
        let mut chunks = prefill_chunks(rows.len(), self.prefill_batch_size).peekable();
        while let Some(range) = chunks.next() {
            let chunk_rows = assemble_validated_input_rows(&rows[range.clone()]);
            let kv_lengths = self
                .kv
                .iter()
                .map(|layer| (layer.keys.len(), layer.values.len()))
                .collect::<Vec<_>>();
            if let Err(error) =
                self.forward_chunk_inner(&chunk_rows, chunks.peek().is_none(), prefill)
            {
                for (layer, (key_len, value_len)) in self.kv.iter_mut().zip(kv_lengths) {
                    layer.keys.truncate(key_len);
                    layer.values.truncate(value_len);
                }
                return Err(error);
            }
            self.seq_len += range.len();
        }
        #[cfg(feature = "parity-trace")]
        if crate::parity_trace::enabled("gemma4.kv") {
            for (layer, kv) in self.kv.iter().enumerate() {
                trace(0, &format!("gemma4.kv.{layer}.keys"), Some(layer), &kv.keys);
                trace(
                    0,
                    &format!("gemma4.kv.{layer}.values"),
                    Some(layer),
                    &kv.values,
                );
            }
        }
        Ok(self.scratch.logits.clone())
    }

    pub(super) fn forward_chunk(&mut self, rows: &[AssembledInputRow]) -> Result<(), String> {
        self.forward_chunk_inner(rows, true, true)
    }

    fn forward_chunk_inner(
        &mut self,
        rows: &[AssembledInputRow],
        project_logits: bool,
        prefill: bool,
    ) -> Result<(), String> {
        if rows.is_empty() || rows.len() > self.prefill_batch_size {
            return Err("Invalid Gemma4 prefill chunk size".into());
        }

        let model = self.model;
        let cfg = &model.config;
        let scratch = &mut self.scratch;
        let mut cpu_linear = Gemma4PrefillLinear::default();
        let linear = if prefill {
            &mut self.prefill_linear
        } else {
            &mut cpu_linear
        };
        let row_count = rows.len();
        #[cfg(feature = "parity-trace")]
        let _trace = crate::parity_trace::TokenMajorTrace::new(row_count);
        let embd = cfg.embd;
        let x_len = row_count * embd;
        let per_layer_all = cfg.per_layer_all();
        let per_layer_len = row_count * per_layer_all;

        let use_per_layer = cfg.use_per_layer_projection();
        let per_layer_width = if use_per_layer { PER_LAYER } else { 0 };
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
            if use_per_layer {
                let pe = model
                    .per_layer_token_embedding
                    .as_ref()
                    .expect("per-layer projection enabled but tensor missing");
                pe.embedding_lookup(
                    row.per_layer_token,
                    &mut scratch.per_layer[index * per_layer_all..(index + 1) * per_layer_all],
                );
            }
        }
        ensure_finite("gemma4.input", &scratch.x[..x_len])?;
        trace_rows("gemma4.input", None, &scratch.x[..x_len], row_count, embd);

        if use_per_layer {
            let token_scale = (PER_LAYER as f32).sqrt();
            for value in &mut scratch.per_layer[..per_layer_len] {
                *value *= token_scale;
            }
            let pm = model
                .per_layer_model_proj
                .as_ref()
                .expect("per-layer projection enabled but tensor missing");
            prefill_matmul_rows(
                "per_layer_model_proj.weight",
                pm,
                &scratch.x[..x_len],
                &mut scratch.per_layer_projected[..per_layer_len],
                row_count,
                model,
                linear,
                model.pool(),
                &mut scratch.prepared,
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            let projection_scale = 1.0 / (embd as f32).sqrt();
            let merge_scale = 1.0 / 2.0_f32.sqrt();
            let pn = model
                .per_layer_proj_norm
                .as_ref()
                .expect("per-layer projection enabled but norm missing");
            for row in 0..row_count {
                for layer in 0..cfg.layers {
                    let start = row * per_layer_all + layer * PER_LAYER;
                    let end = start + PER_LAYER;
                    let projected = &mut scratch.per_layer_projected[start..end];
                    for value in projected.iter_mut() {
                        *value *= projection_scale;
                    }
                    rms_norm_inplace(projected, pn, EPS);
                    for (target, projected) in
                        scratch.per_layer[start..end].iter_mut().zip(projected)
                    {
                        *target = (*target + *projected) * merge_scale;
                    }
                }
            }
            ensure_finite(
                "gemma4.per_layer_input",
                &scratch.per_layer[..per_layer_len],
            )?;
        }

        let base_position = self.seq_len;
        let base_kv = cfg.base_kv_layers();
        for layer_index in 0..cfg.layers {
            let layer = &model.layers[layer_index];
            let dim = layer.head_dim;
            let kv_width = layer.kv_heads * dim;
            let q_width = layer.q_heads * dim;
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
            trace_layer_rows(
                "attn_norm",
                layer_index,
                &scratch.normed[..x_len],
                row_count,
                embd,
            );
            let q_len = row_count * q_width;
            let kv_len = row_count * kv_width;
            if layer_index < base_kv {
                if layer.kv_shared_with_k {
                    // 12B MQA fallback: V is shared with K. Compute Q
                    // and K in parallel, then copy K into V.
                    matmul_group_rows(
                        [
                            (
                                &format!("blk.{layer_index}.attn_q.weight"),
                                &layer.attn_q,
                                &mut scratch.q[..q_len],
                            ),
                            (
                                &format!("blk.{layer_index}.attn_k.weight"),
                                layer
                                    .attn_k
                                    .as_ref()
                                    .expect("attn_k present for base kv layer"),
                                &mut scratch.k[..kv_len],
                            ),
                        ],
                        &scratch.normed[..x_len],
                        row_count,
                        model,
                        linear,
                        model.pool(),
                        &mut scratch.prepared,
                        &mut scratch.q8,
                        &mut scratch.scales,
                    )?;
                    scratch.v[..kv_len].copy_from_slice(&scratch.k[..kv_len]);
                } else {
                    matmul_group_rows(
                        [
                            (
                                &format!("blk.{layer_index}.attn_q.weight"),
                                &layer.attn_q,
                                &mut scratch.q[..q_len],
                            ),
                            (
                                &format!("blk.{layer_index}.attn_k.weight"),
                                layer
                                    .attn_k
                                    .as_ref()
                                    .expect("attn_k present for base kv layer"),
                                &mut scratch.k[..kv_len],
                            ),
                            (
                                &format!("blk.{layer_index}.attn_v.weight"),
                                layer
                                    .attn_v
                                    .as_ref()
                                    .expect("attn_v present when not shared"),
                                &mut scratch.v[..kv_len],
                            ),
                        ],
                        &scratch.normed[..x_len],
                        row_count,
                        model,
                        linear,
                        model.pool(),
                        &mut scratch.prepared,
                        &mut scratch.q8,
                        &mut scratch.scales,
                    )?;
                }
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
                prefill_matmul_rows(
                    &format!("blk.{layer_index}.attn_q.weight"),
                    &layer.attn_q,
                    &scratch.normed[..x_len],
                    &mut scratch.q[..q_len],
                    row_count,
                    model,
                    linear,
                    model.pool(),
                    &mut scratch.prepared,
                    &mut scratch.q8,
                    &mut scratch.scales,
                )?;
            }
            for row in 0..row_count {
                let position = base_position + row;
                let query = &mut scratch.q[row * q_width..(row + 1) * q_width];
                trace_layer(row, "q", layer_index, query);
                for head in query.chunks_exact_mut(dim) {
                    rms_norm_inplace(head, &layer.attn_q_norm, EPS);
                }
                trace_layer(row, "q_norm", layer_index, query);
                apply_rope(
                    query,
                    position,
                    dim,
                    layer_index,
                    cfg.is_swa(layer_index),
                    cfg.rope_freq_base_swa,
                    cfg.rope_freq_base,
                    &model.rope_freqs,
                )?;
                trace_layer(row, "q_rope", layer_index, query);
            }

            if layer_index < base_kv {
                let k_norm = layer.attn_k_norm.as_deref();
                for row in 0..row_count {
                    let position = base_position + row;
                    let key = &mut scratch.k[row * kv_width..(row + 1) * kv_width];
                    let value = &mut scratch.v[row * kv_width..(row + 1) * kv_width];
                    trace_layer(row, "k", layer_index, key);
                    for kv_head in 0..layer.kv_heads {
                        let offset = kv_head * dim;
                        if let Some(kn) = k_norm {
                            rms_norm_inplace(&mut key[offset..offset + dim], kn, EPS);
                        }
                    }
                    trace_layer(row, "k_norm", layer_index, key);
                    apply_rope(
                        key,
                        position,
                        dim,
                        layer_index,
                        cfg.is_swa(layer_index),
                        cfg.rope_freq_base_swa,
                        cfg.rope_freq_base,
                        &model.rope_freqs,
                    )?;
                    trace_layer(row, "k_rope", layer_index, key);
                    trace_layer(row, "v", layer_index, value);
                    for kv_head in 0..layer.kv_heads {
                        let offset = kv_head * dim;
                        rms_unit_inplace(&mut value[offset..offset + dim], EPS);
                    }
                    trace_layer(row, "v_norm", layer_index, value);
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
                    cfg.sliding_window,
                    &mut scratch.attn[row * q_width..(row + 1) * q_width],
                    &mut scratch.scores,
                    &mut scratch.attention_values,
                    model.pool(),
                )?;
            }
            trace_layer_rows(
                "attention",
                layer_index,
                &scratch.attn[..q_len],
                row_count,
                q_width,
            );
            prefill_matmul_rows(
                &format!("blk.{layer_index}.attn_output.weight"),
                &layer.attn_output,
                &scratch.attn[..q_len],
                &mut scratch.projected[..x_len],
                row_count,
                model,
                linear,
                model.pool(),
                &mut scratch.prepared,
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            trace_layer_rows(
                "attention_projected",
                layer_index,
                &scratch.projected[..x_len],
                row_count,
                embd,
            );
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
                trace_layer(row, "attn_out", layer_index, hidden);
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
            trace_layer_rows(
                "ffn_norm",
                layer_index,
                &scratch.normed[..x_len],
                row_count,
                embd,
            );
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
                model,
                linear,
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
            trace_layer_rows(
                "ffn_gate",
                layer_index,
                &scratch.gate[..ffn_len],
                row_count,
                ffn,
            );
            trace_layer_rows(
                "ffn_up",
                layer_index,
                &scratch.up[..ffn_len],
                row_count,
                ffn,
            );
            for (gate, up) in scratch.gate[..ffn_len]
                .chunks_exact_mut(ffn)
                .zip(scratch.up[..ffn_len].chunks_exact(ffn))
            {
                ggml_geglu_fp16_inplace(gate, up);
            }
            trace_layer_rows(
                "ffn_activated",
                layer_index,
                &scratch.gate[..ffn_len],
                row_count,
                ffn,
            );
            prefill_matmul_rows(
                &format!("blk.{layer_index}.ffn_down.weight"),
                &layer.ffn_down,
                &scratch.gate[..ffn_len],
                &mut scratch.down[..x_len],
                row_count,
                model,
                linear,
                model.pool(),
                &mut scratch.prepared,
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            trace_layer_rows(
                "ffn_down",
                layer_index,
                &scratch.down[..x_len],
                row_count,
                embd,
            );
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
                trace_layer(row, "ffn_out", layer_index, hidden);
            }

            if use_per_layer {
                let ig = layer
                    .inp_gate
                    .as_ref()
                    .expect("per-layer projection enabled but inp_gate missing");
                let pj = layer
                    .proj
                    .as_ref()
                    .expect("per-layer projection enabled but proj missing");
                let pn = layer
                    .post_norm
                    .as_ref()
                    .expect("per-layer projection enabled but post_norm missing");
                prefill_matmul_rows(
                    &format!("blk.{layer_index}.inp_gate.weight"),
                    ig,
                    &scratch.x[..x_len],
                    &mut scratch.per_layer_gate[..row_count * per_layer_width],
                    row_count,
                    model,
                    linear,
                    model.pool(),
                    &mut scratch.prepared,
                    &mut scratch.q8,
                    &mut scratch.scales,
                )?;
                for row in 0..row_count {
                    let start = row * per_layer_all + layer_index * per_layer_width;
                    ggml_geglu_fp16_inplace(
                        &mut scratch.per_layer_gate
                            [row * per_layer_width..(row + 1) * per_layer_width],
                        &scratch.per_layer[start..start + per_layer_width],
                    );
                }
                prefill_matmul_rows(
                    &format!("blk.{layer_index}.proj.weight"),
                    pj,
                    &scratch.per_layer_gate[..row_count * per_layer_width],
                    &mut scratch.down[..x_len],
                    row_count,
                    model,
                    linear,
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
                        pn,
                        projected,
                    )?;
                    let hidden = &mut scratch.x[row * embd..(row + 1) * embd];
                    for (hidden, per_layer) in hidden.iter_mut().zip(projected) {
                        *hidden = (*hidden + *per_layer) * layer.output_scale;
                    }
                    ensure_finite(&format!("gemma4.layer.{layer_index}.per_layer_out"), hidden)?;
                    trace_layer(row, "per_layer_out", layer_index, hidden);
                }
            } else {
                // Per-layer projection disabled (12B). Apply the
                // residual scale directly to x and continue.
                for row in 0..row_count {
                    let hidden = &mut scratch.x[row * embd..(row + 1) * embd];
                    for value in hidden.iter_mut() {
                        *value *= layer.output_scale;
                    }
                    ensure_finite(&format!("gemma4.layer.{layer_index}.residual_out"), hidden)?;
                }
            }
            trace_layer_rows(
                "layer_output",
                layer_index,
                &scratch.x[..x_len],
                row_count,
                embd,
            );
        }

        #[cfg(feature = "parity-trace")]
        let trace_all = crate::parity_trace::enabled("gemma4.logits")
            || crate::parity_trace::enabled("gemma4.final.norm");
        #[cfg(not(feature = "parity-trace"))]
        let trace_all = false;
        if project_logits || trace_all {
            for row in if trace_all {
                0..row_count
            } else {
                row_count - 1..row_count
            } {
                let last = &scratch.x[row * embd..(row + 1) * embd];
                checked_rms_norm(
                    "output_norm.weight",
                    last,
                    &model.output_norm,
                    &mut scratch.normed[..embd],
                )?;
                ensure_finite("gemma4.final.norm", &scratch.normed[..embd])?;
                trace(row, "gemma4.final.norm", None, &scratch.normed[..embd]);
                matmul(
                    "token_embd.weight (tied output)",
                    &model.token_embedding,
                    &scratch.normed[..embd],
                    &mut scratch.logits,
                    model.pool(),
                    &mut scratch.q8,
                    &mut scratch.scales,
                )?;
                trace(row, "gemma4.logits.raw", None, &scratch.logits);
                for logit in &mut scratch.logits {
                    *logit = softcap(*logit, model.config.logit_softcap);
                }
                ensure_finite("gemma4.logits", &scratch.logits)?;
                trace(row, "gemma4.logits", None, &scratch.logits);
            }
        }
        Ok(())
    }
}

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
fn try_vulkan_rows(
    linear: &mut Gemma4PrefillLinear,
    model: &Gemma4Model,
    name: &str,
    weight: &Weight<'_>,
    input: &[f32],
    output: &mut [f32],
    rows: usize,
) -> bool {
    use crate::vulkan::ops::GpuWeightFormat;
    if crate::vulkan::gpu_broken() {
        linear.runtime = None;
    }
    if !linear.active() {
        return false;
    }
    let Ok(format) = GpuWeightFormat::from_ggml_type(weight.ggml_type) else {
        return false;
    };
    #[cfg(test)]
    if let Some(dispatcher) = &linear.dispatcher {
        let result = dispatcher.lock().unwrap().dispatch(name, rows);
        return linear.finish_dispatch(result);
    }
    let Some(bytes) = model._source.tensor_slice(name) else {
        return false;
    };
    let result = linear.runtime.as_mut().unwrap().matmul_rows(
        bytes,
        format,
        input,
        rows,
        weight.n_in,
        weight.n_out,
        output,
    );
    linear.finish_dispatch(result)
}

#[allow(clippy::too_many_arguments)]
fn prefill_matmul_rows(
    name: &str,
    weight: &Weight<'_>,
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    model: &Gemma4Model,
    linear: &mut Gemma4PrefillLinear,
    pool: &ComputePool,
    prepared: &mut PreparedRows,
    q8: &mut [u8],
    scales: &mut [f32],
) -> Result<(), String> {
    if input.len() != rows * weight.n_in || output.len() != rows * weight.n_out {
        return Err(format!("Invalid {name} batched matmul lengths"));
    }
    #[cfg(feature = "vulkan")]
    if try_vulkan_rows(linear, model, name, weight, input, output, rows) {
        return ensure_finite(name, output);
    }
    #[cfg(not(feature = "vulkan"))]
    let _ = (model, linear);
    #[cfg(feature = "vulkan")]
    let _cpu_scope = ComputePool::disable_gpu_matmul_for_scope();
    if weight.ggml_type == GGMLType::F32 {
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
    model: &Gemma4Model,
    linear: &mut Gemma4PrefillLinear,
    pool: &ComputePool,
    prepared: &mut PreparedRows,
    q8: &mut [u8],
    scales: &mut [f32],
) -> Result<(), String> {
    if linear.active()
        || projections.iter().any(|(_, weight, _)| {
            weight.ggml_type == GGMLType::F32 || weight.ggml_type == GGMLType::BF16
        })
    {
        for (name, weight, output) in projections {
            prefill_matmul_rows(
                name, weight, input, output, rows, model, linear, pool, prepared, q8, scales,
            )?;
        }
        return Ok(());
    }
    #[cfg(feature = "vulkan")]
    let _cpu_scope = ComputePool::disable_gpu_matmul_for_scope();
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
    #[cfg(feature = "vulkan")]
    let _cpu_scope = ComputePool::disable_gpu_matmul_for_scope();
    if input.len() != weight.n_in || output.len() != weight.n_out {
        return Err(format!(
            "Invalid {name} matmul lengths: input {}, output {}; expected {}, {}",
            input.len(),
            output.len(),
            weight.n_in,
            weight.n_out
        ));
    }
    if weight.ggml_type == GGMLType::F32 {
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
    swa_freq_base: f32,
    full_freq_base: f32,
    full_freq_factors: &[f32],
) -> Result<(), String> {
    if values.len() % dim != 0 {
        return Err(format!(
            "blk.{layer} RoPE length {} is not divisible by head width {dim}",
            values.len()
        ));
    }
    if sliding {
        apply_rope_ggml(values, position, dim, swa_freq_base, None);
        return Ok(());
    }
    if full_freq_factors.len() != dim / 2 {
        return Err(format!(
            "rope_freqs.weight length {}; expected {} for blk.{layer} (freq_base={full_freq_base})",
            full_freq_factors.len(),
            dim / 2
        ));
    }
    apply_rope_ggml(
        values,
        position,
        dim,
        full_freq_base,
        Some(full_freq_factors),
    );
    Ok(())
}

/// Gemma 4 RoPE matching ggml's `ggml_rope_cache_init` and fused rotation.
fn apply_rope_ggml(
    values: &mut [f32],
    position: usize,
    dim: usize,
    freq_base: f32,
    factors: Option<&[f32]>,
) {
    let half = dim / 2;
    let n_heads = values.len() / dim;
    if half == 0 || n_heads == 0 {
        return;
    }
    let theta_scale = freq_base.powf(-2.0 / dim as f32);
    let mut theta = position as f32;
    let mut cos_table = vec![0.0f32; half];
    let mut sin_table = vec![0.0f32; half];
    for i in 0..half {
        let angle = theta / factors.map_or(1.0, |values| values[i]);
        let (c, s) = crate::ops::rope::rope_sin_cos(angle);
        cos_table[i] = c;
        sin_table[i] = s;
        theta *= theta_scale;
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

#[cfg(target_arch = "x86_64")]
#[inline]
pub(super) fn ggml_attention_dot(left: &[f32], right: &[f32], len: usize) -> f32 {
    use std::arch::x86_64::*;

    debug_assert!(left.len() >= len && right.len() >= len);
    let mut sums = unsafe { [_mm_setzero_ps(); 8] };
    let use_fma = std::is_x86_feature_detected!("fma");
    let aligned = len & !31;
    let mut index = 0;
    while index < aligned {
        for lane in 0..8 {
            let offset = index + lane * 4;
            unsafe {
                sums[lane] = if use_fma {
                    _mm_fmadd_ps(
                        _mm_loadu_ps(left.as_ptr().add(offset)),
                        _mm_loadu_ps(right.as_ptr().add(offset)),
                        sums[lane],
                    )
                } else {
                    _mm_add_ps(
                        sums[lane],
                        _mm_mul_ps(
                            _mm_loadu_ps(left.as_ptr().add(offset)),
                            _mm_loadu_ps(right.as_ptr().add(offset)),
                        ),
                    )
                };
            }
        }
        index += 32;
    }
    for lane in 0..4 {
        unsafe { sums[lane] = _mm_add_ps(sums[lane], sums[lane + 4]) };
    }
    for lane in 0..2 {
        unsafe { sums[lane] = _mm_add_ps(sums[lane], sums[lane + 2]) };
    }
    unsafe { sums[0] = _mm_add_ps(sums[0], sums[1]) };
    let mut lanes = [0.0f32; 4];
    unsafe { _mm_storeu_ps(lanes.as_mut_ptr(), sums[0]) };
    let mut sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    while index < len {
        sum += left[index] * right[index];
        index += 1;
    }
    sum
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(super) fn ggml_attention_dot(left: &[f32], right: &[f32], len: usize) -> f32 {
    crate::ops::dot_f32(left, right, len)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn ggml_exp_sse2(x: std::arch::x86_64::__m128, use_fma: bool) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;

    let magic = _mm_set1_ps(f32::from_bits(0x4b40_0000));
    let z = if use_fma {
        _mm_fmadd_ps(x, _mm_set1_ps(f32::from_bits(0x3fb8_aa3b)), magic)
    } else {
        _mm_add_ps(
            _mm_mul_ps(x, _mm_set1_ps(f32::from_bits(0x3fb8_aa3b))),
            magic,
        )
    };
    let n = _mm_sub_ps(z, magic);
    let b = if use_fma {
        _mm_fnmadd_ps(
            n,
            _mm_set1_ps(f32::from_bits(0x35bf_be8e)),
            _mm_fnmadd_ps(n, _mm_set1_ps(f32::from_bits(0x3f31_7200)), x),
        )
    } else {
        _mm_sub_ps(
            _mm_sub_ps(x, _mm_mul_ps(n, _mm_set1_ps(f32::from_bits(0x3f31_7200)))),
            _mm_mul_ps(n, _mm_set1_ps(f32::from_bits(0x35bf_be8e))),
        )
    };
    let exponent = _mm_slli_epi32(_mm_castps_si128(z), 23);
    let scale = _mm_castsi128_ps(_mm_add_epi32(exponent, _mm_set1_epi32(0x3f80_0000)));
    let out_of_range = _mm_castps_si128(_mm_cmpgt_ps(
        _mm_andnot_ps(_mm_set1_ps(-0.0), n),
        _mm_set1_ps(126.0),
    ));
    let squared = _mm_mul_ps(b, b);
    let (low, high) = if use_fma {
        (
            _mm_fmadd_ps(
                _mm_set1_ps(f32::from_bits(0x3c07_2010)),
                b,
                _mm_set1_ps(f32::from_bits(0x3d2b_9f17)),
            ),
            _mm_fmadd_ps(
                _mm_set1_ps(f32::from_bits(0x3e2a_af33)),
                b,
                _mm_set1_ps(f32::from_bits(0x3eff_fedb)),
            ),
        )
    } else {
        (
            _mm_add_ps(
                _mm_mul_ps(_mm_set1_ps(f32::from_bits(0x3c07_2010)), b),
                _mm_set1_ps(f32::from_bits(0x3d2b_9f17)),
            ),
            _mm_add_ps(
                _mm_mul_ps(_mm_set1_ps(f32::from_bits(0x3e2a_af33)), b),
                _mm_set1_ps(f32::from_bits(0x3eff_fedb)),
            ),
        )
    };
    let linear = _mm_mul_ps(_mm_set1_ps(f32::from_bits(0x3f7f_fff6)), b);
    let polynomial = if use_fma {
        _mm_fmadd_ps(_mm_fmadd_ps(low, squared, high), squared, linear)
    } else {
        _mm_add_ps(
            _mm_mul_ps(_mm_add_ps(_mm_mul_ps(low, squared), high), squared),
            linear,
        )
    };
    if _mm_movemask_epi8(out_of_range) == 0 {
        return if use_fma {
            _mm_fmadd_ps(polynomial, scale, scale)
        } else {
            _mm_add_ps(_mm_mul_ps(polynomial, scale), scale)
        };
    }

    let adjustment = _mm_and_si128(
        _mm_castps_si128(_mm_cmple_ps(n, _mm_setzero_ps())),
        _mm_set1_epi32(0x8200_0000u32 as i32),
    );
    let scale1 = _mm_castsi128_ps(_mm_add_epi32(adjustment, _mm_set1_epi32(0x7f00_0000)));
    let scale2 = _mm_castsi128_ps(_mm_sub_epi32(exponent, adjustment));
    let extreme = _mm_castps_si128(_mm_cmpgt_ps(
        _mm_andnot_ps(_mm_set1_ps(-0.0), n),
        _mm_set1_ps(192.0),
    ));
    let scaled = if use_fma {
        _mm_fmadd_ps(scale, polynomial, scale)
    } else {
        _mm_add_ps(_mm_mul_ps(scale, polynomial), scale)
    };
    let ranged_scaled = if use_fma {
        _mm_mul_ps(_mm_fmadd_ps(scale2, polynomial, scale2), scale1)
    } else {
        _mm_mul_ps(_mm_add_ps(_mm_mul_ps(scale2, polynomial), scale2), scale1)
    };
    let ranged = _mm_or_ps(
        _mm_and_ps(_mm_castsi128_ps(out_of_range), ranged_scaled),
        _mm_andnot_ps(_mm_castsi128_ps(out_of_range), scaled),
    );
    _mm_or_ps(
        _mm_and_ps(_mm_castsi128_ps(extreme), _mm_mul_ps(scale1, scale1)),
        _mm_andnot_ps(_mm_castsi128_ps(extreme), ranged),
    )
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn ggml_attention_softmax(values: &mut [f32]) {
    use std::arch::x86_64::*;

    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let use_fma = std::is_x86_feature_detected!("fma");
    let mut sum = 0.0f64;
    let mut index = 0;
    while index + 8 <= values.len() {
        unsafe {
            let low = ggml_exp_sse2(
                _mm_sub_ps(_mm_loadu_ps(values.as_ptr().add(index)), _mm_set1_ps(max)),
                use_fma,
            );
            let high = ggml_exp_sse2(
                _mm_sub_ps(
                    _mm_loadu_ps(values.as_ptr().add(index + 4)),
                    _mm_set1_ps(max),
                ),
                use_fma,
            );
            _mm_storeu_ps(values.as_mut_ptr().add(index), low);
            _mm_storeu_ps(values.as_mut_ptr().add(index + 4), high);
            let mut reduced = _mm_add_ps(high, low);
            reduced = _mm_add_ps(reduced, _mm_movehl_ps(reduced, reduced));
            reduced = _mm_add_ss(reduced, _mm_shuffle_ps(reduced, reduced, 0xf5));
            sum += f64::from(_mm_cvtss_f32(reduced));
        }
        index += 8;
    }
    while index < values.len() {
        values[index] = (values[index] - max).exp();
        sum += f64::from(values[index]);
        index += 1;
    }

    let scale = (1.0 / sum) as f32;
    for value in values {
        *value *= scale;
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn ggml_attention_softmax(values: &mut [f32]) {
    crate::ops::softmax_approx_inplace(values);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attend(
    layer: usize,
    position: usize,
    query: &[f32],
    cache: &KvLayer,
    sliding: bool,
    sliding_window: usize,
    output: &mut [f32],
    scores: &mut Vec<f32>,
    values: &mut Vec<f32>,
    pool: &ComputePool,
) -> Result<(), String> {
    let dim = cache.head_dim;
    let row_width = cache.row_width;
    let group_size = cache.group_size;
    // `cache.q_heads` is `group_size * cache.kv_heads`. We derive
    // q_heads from `row_width` and `group_size` indirectly: the
    // largest head index we visit is `q_heads - 1` where
    // `q_heads = group_size * kv_heads`. Easiest: compute q_heads as
    // `query.len() / dim`, since the caller sized the buffer exactly.
    let q_heads = query.len() / dim;
    let q_width = q_heads * dim;
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
    let first = if sliding {
        rows.saturating_sub(sliding_window)
    } else {
        0
    };
    let cached = rows - first;
    let padded = cached.div_ceil(256) * 256;
    let workspace = padded
        .checked_mul(pool.n_threads())
        .ok_or_else(|| format!("blk.{layer} attention scratch length overflow"))?;
    scores.resize(workspace, f32::NEG_INFINITY);
    values.resize(workspace, 0.0);
    values.fill(0.0);
    let query_ptr = query.as_ptr();
    let keys_ptr = cache.keys.as_ptr();
    let values_ptr = cache.values.as_ptr();
    let output_ptr = output.as_mut_ptr();
    let scores_ptr = scores.as_mut_ptr();
    let scratch_values_ptr = values.as_mut_ptr();
    pool.compute(move |ith, nth| {
        let h_step = (q_heads + nth - 1) / nth;
        let h_start = (ith * h_step).min(q_heads);
        let h_end = (h_start + h_step).min(q_heads);
        let head_scores =
            unsafe { std::slice::from_raw_parts_mut(scores_ptr.add(ith * padded), padded) };
        let head_values =
            unsafe { std::slice::from_raw_parts_mut(scratch_values_ptr.add(ith * padded), padded) };
        for head in h_start..h_end {
            let query_head = unsafe { std::slice::from_raw_parts(query_ptr.add(head * dim), dim) };
            let kv_head = head / group_size;
            let kv_offset = kv_head * dim;
            head_scores.fill(f32::NEG_INFINITY);
            for (score, token) in head_scores[..cached].iter_mut().zip(first..rows) {
                let offset = token * row_width + kv_offset;
                let key = unsafe { std::slice::from_raw_parts(keys_ptr.add(offset), dim) };
                *score = ggml_attention_dot(query_head, key, dim);
            }
            ggml_attention_softmax(head_scores);
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
                    head_values,
                    head_output,
                );
            }
        }
    });
    ensure_finite(&format!("blk.{layer} attention"), output)
}

/// Match ggml's dimension-major `MUL_MAT` by gathering each V column and using
/// the same padded SIMD dot reduction as the Q·K pass.
#[inline]
unsafe fn attend_v(
    values_ptr: *const f32,
    scores_ptr: *const f32,
    row_width: usize,
    kv_offset: usize,
    first: usize,
    cached: usize,
    dim: usize,
    head_values: &mut [f32],
    head_output: &mut [f32],
) {
    let padded = cached.div_ceil(256) * 256;
    let scores = std::slice::from_raw_parts(scores_ptr, padded);
    for (d, output) in head_output[..dim].iter_mut().enumerate() {
        for (value, token) in head_values[..cached].iter_mut().zip(first..) {
            *value = *values_ptr.add(token * row_width + kv_offset + d);
        }
        *output = ggml_attention_dot(scores, &head_values, padded);
    }
}

pub(super) fn softcap(value: f32, cap: f32) -> f32 {
    cap * (value * (1.0 / cap)).tanh()
}

pub(super) fn ggml_geglu_fp16_inplace(gate: &mut [f32], up: &[f32]) {
    // Scalar reference; matches llama.cpp GEGLU with f16 GELU table entries.
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
    assert_eq!(gate.len(), up.len());
    for (gate, up) in gate.iter_mut().zip(up) {
        let x = *gate;
        *gate = if x <= -10.0 {
            0.0
        } else {
            crate::ops::gelu_ggml_f16(x) * up
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

#[cfg(feature = "parity-trace")]
fn trace_layer(row: usize, stage: &str, layer: usize, values: &[f32]) {
    trace(
        row,
        &format!("gemma4.layer.{layer}.{stage}"),
        Some(layer),
        values,
    );
}

#[cfg(not(feature = "parity-trace"))]
#[inline(always)]
fn trace_layer(_row: usize, _stage: &str, _layer: usize, _values: &[f32]) {}

#[cfg(feature = "parity-trace")]
fn trace_layer_rows(stage: &str, layer: usize, values: &[f32], rows: usize, width: usize) {
    let name = format!("gemma4.layer.{layer}.{stage}");
    let _ = crate::parity_trace::checkpoint_rows(&name, Some(layer), &[rows, width], values);
}

#[cfg(not(feature = "parity-trace"))]
#[inline(always)]
fn trace_layer_rows(_stage: &str, _layer: usize, _values: &[f32], _rows: usize, _width: usize) {}

#[cfg(feature = "parity-trace")]
fn trace_rows(name: &str, layer: Option<usize>, values: &[f32], rows: usize, width: usize) {
    let _ = crate::parity_trace::checkpoint_rows(name, layer, &[rows, width], values);
}

#[cfg(not(feature = "parity-trace"))]
#[inline(always)]
fn trace_rows(_name: &str, _layer: Option<usize>, _values: &[f32], _rows: usize, _width: usize) {}

#[cfg(feature = "parity-trace")]
fn trace(row: usize, name: &str, layer: Option<usize>, values: &[f32]) {
    crate::parity_trace::report(crate::parity_trace::checkpoint_row(
        row,
        name,
        layer,
        &[1, values.len()],
        values,
    ));
}

#[cfg(not(feature = "parity-trace"))]
fn trace(_row: usize, _name: &str, _layer: Option<usize>, _values: &[f32]) {}

#[cfg(test)]
mod tests {
    use super::{apply_rope, attend, attend_v, ggml_attention_softmax, KvLayer};
    use crate::core::thread_pool::ComputePool;

    #[test]
    fn full_attention_rope_applies_base_before_frequency_factors() {
        let mut values = [1.25, -2.0, 0.75, 3.5];

        apply_rope(&mut values, 1, 4, 0, false, 10_000.0, 16.0, &[1.0, 2.0]).unwrap();

        assert_eq!(
            values.map(f32::to_bits),
            [0x3d35_5950, 0xc01a_edae, 0x3fba_811e, 0x404e_4b3f]
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn attention_softmax_matches_llama_avx2_raw_bits() {
        let mut scores = vec![f32::NEG_INFINITY; 256];
        scores[..3].copy_from_slice(&[0x3fcd_edd6, 0x40e4_80e5, 0x4150_be72].map(f32::from_bits));

        ggml_attention_softmax(&mut scores);

        assert_eq!(
            scores[..3]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![0x3734_641c, 0x3b32_039e, 0x3f7f_4d49],
        );

        let mut scores = vec![f32::NEG_INFINITY; 256];
        scores[..5].copy_from_slice(
            &[
                0x3fd0_6948,
                0xc003_7c6a,
                0xbf45_a7b2,
                0x4118_5013,
                0x4015_ff08,
            ]
            .map(f32::from_bits),
        );
        ggml_attention_softmax(&mut scores);
        assert_eq!(
            scores[..5]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![
                0x39c3_d611,
                0x371d_a495,
                0x380e_1575,
                0x3f7f_b29f,
                0x3a48_4231
            ],
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn attention_softmax_matches_llama_neon_raw_bits() {
        let cache = KvLayer {
            head_dim: 1,
            row_width: 1,
            group_size: 1,
            keys: [0x410d_e8ceu32, 0x410e_45e2, 0x4110_5392]
                .map(f32::from_bits)
                .to_vec(),
            values: vec![1.0, 0.0, 0.0],
        };
        let mut output = [0.0];

        attend(
            16,
            2,
            &[1.0],
            &cache,
            false,
            0,
            &mut output,
            &mut Vec::new(),
            &mut Vec::new(),
            &ComputePool::new(1),
        )
        .unwrap();

        assert_eq!(output[0].to_bits(), 0x3ea0_b33e);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn attention_values_match_llama_neon_dot_reduction() {
        let mut scores = [0.0; 256];
        scores[0] = f32::from_bits(0x3efa_b5d9);
        scores[1] = f32::from_bits(0x3f02_a513);
        let mut values = [0.0; 8];
        values[0] = f32::from_bits(0x3cb9_1aee);
        values[4] = f32::from_bits(0x3c9f_4180);
        let mut output = [0.0; 4];
        let mut scratch = [0.0; 256];

        unsafe {
            attend_v(
                values.as_ptr(),
                scores.as_ptr(),
                4,
                0,
                0,
                2,
                4,
                &mut scratch,
                &mut output,
            );
        }

        assert_eq!(output[0].to_bits(), 0x3cab_e9d8);
    }

    #[test]
    fn attention_reuses_worker_partitioned_scratch() {
        let cache = KvLayer {
            head_dim: 1,
            row_width: 1,
            group_size: 2,
            keys: vec![0.0, 1.0],
            values: vec![1.0, 2.0],
        };
        let mut output = [0.0; 2];
        let mut scores = Vec::new();
        let mut values = Vec::new();

        attend(
            0,
            1,
            &[1.0; 2],
            &cache,
            false,
            0,
            &mut output,
            &mut scores,
            &mut values,
            &ComputePool::new(2),
        )
        .unwrap();

        assert_eq!(scores.len(), 2 * 256);
        assert_eq!(values.len(), 2 * 256);
        assert_eq!(output[0].to_bits(), output[1].to_bits());
    }
}
