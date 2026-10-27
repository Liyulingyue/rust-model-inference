//! Free functions that operate on a fully-loaded [`Qwen3Model`] but do
//! not belong to the model's `impl` block.
//!
//! - [`text_encode`]: the canonical "give me a hidden-state vector for
//!   this token sequence" entry point. Used by VL, ASR, TTS, and the
//!   Z-Image text encoder.
//! - [`run_shared_inference`]: end-to-end CLI helper used by the VL /
//!   ASR / TTS entry points. Builds the prompt, loads the model, runs
//!   `Qwen3Model::generate`, prints timings.
//!
//! Also defines the forward-loop input/output structs (`Qwen3Input`,
//! `Qwen3GenerateOptions`, `Qwen3Generation`) and the `Qwen3Model::generate` /
//! `Qwen3Model::text_encode` method wrappers.

use super::config::{Qwen3Config, Qwen3Rope};
use super::positions::qwen_text_positions;
use super::session::Qwen3Session;
use super::util::{
    checked_product, checked_session_capacity, validate_generation, validate_token_ids,
};
use super::weights::{Qwen3LayerWeights, Qwen3Model};
use crate::app::cli::resolve_thread_count;
use crate::compute::{
    linear::{LinearBinding, LinearExecutor, LinearMode},
    ComputePolicy,
};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::ops::rope::rope_neox_inplace;
use crate::ops::*;
use crate::prompt::{build_qwen_chat_prompt, QwenMessage};
#[cfg(feature = "vulkan")]
use crate::vulkan::qwen3::Qwen3VulkanSession;
use std::io::{self, Write};

use std::sync::Arc;
use std::time::{Duration, Instant};

pub(super) fn forward_moe_token(
    input: &[f32],
    layer: &Qwen3LayerWeights<'_>,
    config: &Qwen3Config,
    output: &mut [f32],
) -> Result<(), String> {
    let router = layer.moe_router.as_deref().ok_or("missing MoE router")?;
    let gate = layer
        .moe_gate
        .as_deref()
        .ok_or("missing MoE gate experts")?;
    let up = layer.moe_up.as_deref().ok_or("missing MoE up experts")?;
    let down = layer
        .moe_down
        .as_deref()
        .ok_or("missing MoE down experts")?;
    let expert_count = gate.len();
    let used = config.moe.map(|value| value.expert_used_count).unwrap_or(0);
    if expert_count == 0
        || used == 0
        || used > expert_count
        || router.len() != input.len() * expert_count
    {
        return Err("invalid Qwen3VL-MoE router shape".into());
    }
    let mut logits: Vec<(usize, f32)> = (0..expert_count)
        .map(|expert| {
            let row = &router[expert * input.len()..(expert + 1) * input.len()];
            (expert, dot_f32(input, row, input.len()))
        })
        .collect();
    logits.sort_by(|(a, va), (b, vb)| {
        vb.partial_cmp(va)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(b))
    });
    let max = logits[..used]
        .iter()
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits[..used]
        .iter()
        .map(|(_, value)| (*value - max).exp())
        .collect();
    let norm = probs.iter().sum::<f32>();
    for value in &mut probs {
        *value /= norm;
    }
    output.fill(0.0);
    let mut gate_buf = vec![
        0.0;
        config
            .moe
            .map(|value| value.expert_ffn)
            .unwrap_or(config.n_ff)
    ];
    let mut up_buf = vec![0.0; gate_buf.len()];
    let mut down_buf = vec![0.0; output.len()];
    let expert_ffn = gate_buf.len();
    for ((expert, _), probability) in logits[..used].iter().zip(probs) {
        gate[*expert]
            .kernel
            .forward(input, &mut gate_buf, input.len(), expert_ffn);
        up[*expert]
            .kernel
            .forward(input, &mut up_buf, input.len(), expert_ffn);
        for (gate_value, up_value) in gate_buf.iter_mut().zip(&up_buf) {
            *gate_value = silu(*gate_value) * *up_value;
        }
        down[*expert]
            .kernel
            .forward(&gate_buf, &mut down_buf, expert_ffn, output.len());
        for (value, expert_value) in output.iter_mut().zip(&down_buf) {
            *value += probability * *expert_value;
        }
    }
    Ok(())
}

// =============================================================================
// Forward-loop input/output types
// =============================================================================

#[derive(Clone)]
pub struct Qwen3Input<'a> {
    pub token_ids: &'a [u32],
    pub positions: &'a [[usize; 4]],
    pub embeddings: Option<&'a [f32]>,
    pub deepstack_embeddings: Option<&'a [f32]>,
}

#[derive(Clone)]
pub struct Qwen3GenerateOptions {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub prefill_batch_size: usize,
}

#[derive(Clone)]
pub struct Qwen3Generation {
    pub text: String,
    pub rendered_tokens: Vec<String>,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub prompt_duration: Duration,
    pub decode_duration: Duration,
    #[cfg(feature = "vulkan")]
    pub prompt_submissions: u64,
    #[cfg(feature = "vulkan")]
    pub decode_submissions: u64,
}

// =============================================================================
// Qwen3Model generate / text_encode method wrappers
// =============================================================================

impl Qwen3Model {
    pub fn generate(
        &self,
        input: Qwen3Input<'_>,
        options: Qwen3GenerateOptions,
    ) -> Result<Qwen3Generation, String> {
        self.generate_with_asr_trace(input, options, false, 1.0)
    }

    pub(crate) fn generate_asr(
        &self,
        input: Qwen3Input<'_>,
        options: Qwen3GenerateOptions,
        repetition_penalty: f32,
    ) -> Result<Qwen3Generation, String> {
        self.generate_with_asr_trace(input, options, true, repetition_penalty)
    }

    fn generate_with_asr_trace(
        &self,
        input: Qwen3Input<'_>,
        options: Qwen3GenerateOptions,
        asr_trace: bool,
        repetition_penalty: f32,
    ) -> Result<Qwen3Generation, String> {
        validate_generation(self, &input, &options)?;
        let capacity = checked_session_capacity(
            input.token_ids.len(),
            options.max_new_tokens,
            self.config.n_ctx,
        )?;
        Qwen3Session::new(self, capacity)?.generate_with_asr_trace(
            input,
            options,
            asr_trace,
            repetition_penalty,
        )
    }

    /// Hidden-state extraction entry point. Used by VL/ASR/TTS/Z-Image
    /// encoder paths that need token-sequence embeddings rather than
    /// generated text.
    pub fn text_encode(
        &self,
        token_ids: &[u32],
        positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        text_encode(self, token_ids, positions)
    }

    pub fn text_encode_embeddings(
        &self,
        embeddings: Vec<f32>,
        positions: &[[usize; 4]],
        key_mask: &[bool],
    ) -> Result<Vec<f32>, String> {
        if positions.len() != key_mask.len()
            || embeddings.len() != positions.len().saturating_mul(self.config.n_embd)
            || key_mask.first() != Some(&true)
        {
            return Err("Invalid Qwen text embeddings or attention mask".into());
        }
        let n_tokens = positions.len();
        text_encode_forward(self, &embeddings, n_tokens, positions, Some(key_mask), &[])
    }

    /// Token-major concatenation of selected pre-output-norm hidden layers.
    /// Layer numbers follow Transformers' `hidden_states` (embedding is zero).
    pub(crate) fn text_encode_hidden_layers(
        &self,
        token_ids: &[u32],
        positions: &[[usize; 4]],
        key_mask: &[bool],
        layers: &[usize],
    ) -> Result<Vec<f32>, String> {
        if token_ids.len() != positions.len()
            || token_ids.len() != key_mask.len()
            || key_mask.first() != Some(&true)
            || layers.is_empty()
            || layers[0] == 0
            || layers.windows(2).any(|w| w[0] >= w[1])
            || *layers.last().unwrap() >= self.config.n_layer
        {
            return Err("Invalid Qwen hidden-layer extraction request".into());
        }
        let embeddings = self.embed_tokens(token_ids)?;
        text_encode_forward(
            self,
            &embeddings,
            token_ids.len(),
            positions,
            Some(key_mask),
            layers,
        )
    }
}

pub fn text_encode(
    model: &Qwen3Model,
    token_ids: &[u32],
    positions: &[[usize; 4]],
) -> Result<Vec<f32>, String> {
    let n_tokens = token_ids.len();
    if n_tokens == 0 {
        return Ok(Vec::new());
    }
    let embeddings = model.embed_tokens(token_ids)?;
    text_encode_forward(model, &embeddings, n_tokens, positions, None, &[])
}

/// Qwen2.5-Omni CFMEdit extension: same as [`text_encode`] but with audio
/// embeddings substituted for specific token positions.
///
/// `audio_replacements` is a list of `(position_index, audio_embedding_2048d)`
/// pairs. For each pair, the embedding at that token position is overwritten
/// with the corresponding audio tower output (also 2048-dim). The transformer
/// then runs over the substituted sequence, so text and audio tokens attend
/// to each other inside the Qwen trunk -- matching audio.cpp's
/// `conditioning.cpp::build_text_conditioning` behavior.
pub fn text_encode_with_audio(
    model: &Qwen3Model,
    token_ids: &[u32],
    positions: &[[usize; 4]],
    audio_replacements: &[(usize, &[f32])],
) -> Result<Vec<f32>, String> {
    validate_token_ids(token_ids, model.config.vocab)?;
    let n_tokens = token_ids.len();
    if positions.len() != n_tokens {
        return Err(format!(
            "positions length {} != token_ids length {}",
            positions.len(),
            n_tokens
        ));
    }
    if n_tokens == 0 {
        return Ok(Vec::new());
    }
    let n_embd = model.config.n_embd;
    let mut embeddings = model.embed_tokens(token_ids)?;
    for (pos, audio_embed) in audio_replacements {
        if *pos >= n_tokens {
            return Err(format!(
                "audio_replacement position {pos} out of bounds (n_tokens={n_tokens})"
            ));
        }
        if audio_embed.len() != n_embd {
            return Err(format!(
                "audio embedding length {} != n_embd={n_embd}",
                audio_embed.len()
            ));
        }
        let off = *pos * n_embd;
        embeddings[off..off + n_embd].copy_from_slice(audio_embed);
    }
    text_encode_forward(model, &embeddings, n_tokens, positions, None, &[])
}

const PROJECTION_ROWS: usize = 128;

/// Keep weight uploads across layers and let the shared executor select the backend.
fn project_tokens<const N: usize>(
    executor: &mut LinearExecutor<'_, '_>,
    input: &[f32],
    mut projections: [(&crate::ops::kernel::Weight<'_>, &mut [f32], Option<&[f32]>); N],
) -> Result<(), String> {
    let n_in = projections[0].0.n_in;
    let rows = input.len() / n_in;
    let mut ids = [executor
        .id_for(projections[0].0)
        .map_err(|e| e.to_string())?; N];
    for (id, (weight, _, _)) in ids.iter_mut().zip(&projections) {
        *id = executor.id_for(weight).map_err(|e| e.to_string())?;
    }
    for start in (0..rows).step_by(PROJECTION_ROWS) {
        let end = (start + PROJECTION_ROWS).min(rows);
        executor
            .run_group(
                ids,
                &input[start * n_in..end * n_in],
                end - start,
                projections.each_mut().map(|(weight, output, _)| {
                    &mut output[start * weight.n_out..end * weight.n_out]
                }),
            )
            .map_err(|e| e.to_string())?;
    }
    for (weight, output, bias) in projections {
        if let Some(bias) = bias {
            for row in output.chunks_exact_mut(weight.n_out) {
                for (value, bias) in row.iter_mut().zip(bias) {
                    *value += *bias;
                }
            }
        }
    }
    Ok(())
}

/// Shared forward path for [`text_encode`] and [`text_encode_with_audio`].
/// Operates on pre-built per-token embeddings of shape `[n_tokens, n_embd]`.
///
/// `key_mask` masks out padded keys; causal attention always hides future keys.
fn text_encode_forward(
    model: &Qwen3Model,
    embeddings: &[f32],
    n_tokens: usize,
    positions: &[[usize; 4]],
    key_mask: Option<&[bool]>,
    layers: &[usize],
) -> Result<Vec<f32>, String> {
    #[cfg(feature = "vulkan")]
    let mut full_model_gpu_failed = false;
    #[cfg(feature = "vulkan")]
    if layers.is_empty()
        && key_mask.is_none()
        && positions
            .iter()
            .enumerate()
            .all(|(index, position)| position[0] == index)
    {
        if let Some(context) = crate::ops::get_vulkan_context() {
            match text_encode_vulkan(model, &embeddings, n_tokens, context) {
                Ok(Some(hidden)) => return Ok(hidden),
                Ok(None) => {}
                Err(error) => {
                    eprintln!(
                        "[GPU] Qwen3 Vulkan embedding executor disabled after error: {error}. Falling back to CPU."
                    );
                    full_model_gpu_failed = true;
                }
            }
        }
    }

    #[cfg(feature = "vulkan")]
    let _gpu_matmul_scope = full_model_gpu_failed.then(ComputePool::disable_gpu_matmul_for_scope);

    let cfg = &model.config;
    let n_embd_q = checked_product("query width", cfg.n_head, cfg.n_embd_head_k)?;
    let n_embd_k = checked_product("key width", cfg.n_head_kv, cfg.n_embd_head_k)?;
    let n_embd_v = checked_product("value width", cfg.n_head_kv, cfg.n_embd_head_v)?;
    let n_attn = checked_product("attn width", cfg.n_head, cfg.n_embd_head_v)?;
    let group_size = cfg.n_head / cfg.n_head_kv;
    let kq_scale = 1.0 / (cfg.n_embd_head_k as f32).sqrt();

    let mut hidden = embeddings.to_vec();
    let mut collected = if layers.is_empty() {
        Vec::new()
    } else {
        vec![0.0; checked_product("hidden-layer output", hidden.len(), layers.len())?]
    };

    let depth = layers.last().copied().unwrap_or(cfg.n_layer);
    let bindings = model.layers[..depth]
        .iter()
        .flat_map(|layer| {
            let mut weights = vec![&layer.wq, &layer.wk, &layer.wv, &layer.wo];
            if layer.moe_router.is_none() {
                weights.extend([&layer.w_gate, &layer.w_up, &layer.w_down]);
            }
            weights.into_iter().map(|weight| LinearBinding {
                weight,
                mode: LinearMode::Prepared,
            })
        })
        .collect();
    let mut executor = LinearExecutor::new(
        ComputePolicy::legacy(),
        bindings,
        PROJECTION_ROWS,
        model.pool(),
    )
    .map_err(|e| e.to_string())?;

    for layer_idx in 0..depth {
        let layer = &model.layers[layer_idx];

        let mut normed = vec![0.0; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            rms_norm(
                &hidden[off..off + cfg.n_embd],
                &layer.attn_norm,
                &mut normed[off..off + cfg.n_embd],
                cfg.eps,
            );
        }

        let mut q_all = vec![0.0; n_tokens * n_embd_q];
        let mut k_all = vec![0.0; n_tokens * n_embd_k];
        let mut v_all = vec![0.0; n_tokens * n_embd_v];
        project_tokens(
            &mut executor,
            &normed,
            [
                (&layer.wq, &mut q_all, layer.q_bias.as_deref()),
                (&layer.wk, &mut k_all, layer.k_bias.as_deref()),
                (&layer.wv, &mut v_all, layer.v_bias.as_deref()),
            ],
        )?;

        if let (Some(q_norm), Some(k_norm)) = (layer.q_norm.as_deref(), layer.k_norm.as_deref()) {
            apply_qk_norms(
                &mut q_all,
                &mut k_all,
                n_tokens,
                cfg.n_head,
                cfg.n_head_kv,
                cfg.n_embd_head_k,
                q_norm,
                k_norm,
                cfg.eps,
            );
        }

        for tok in 0..n_tokens {
            let pos = positions[tok];
            for head in 0..cfg.n_head {
                let off = tok * n_embd_q + head * cfg.n_embd_head_k;
                let q_slice = &mut q_all[off..off + cfg.n_embd_head_k];
                match cfg.rope {
                    Qwen3Rope::Neox => {
                        rope_neox_inplace(q_slice, pos[0], cfg.n_embd_head_k, cfg.freq_base);
                    }
                    Qwen3Rope::Mrope { sections } => {
                        rope_mrope(q_slice, pos, sections, cfg.n_embd_head_k, cfg.freq_base);
                    }
                    Qwen3Rope::Interleaved { sections, n_dims } => {
                        rope_mrope_interleaved(
                            q_slice,
                            pos,
                            sections,
                            cfg.n_embd_head_k,
                            cfg.freq_base,
                            n_dims,
                        );
                    }
                }
            }
            for head in 0..cfg.n_head_kv {
                let off = tok * n_embd_k + head * cfg.n_embd_head_k;
                let k_slice = &mut k_all[off..off + cfg.n_embd_head_k];
                match cfg.rope {
                    Qwen3Rope::Neox => {
                        rope_neox_inplace(k_slice, pos[0], cfg.n_embd_head_k, cfg.freq_base);
                    }
                    Qwen3Rope::Mrope { sections } => {
                        rope_mrope(k_slice, pos, sections, cfg.n_embd_head_k, cfg.freq_base);
                    }
                    Qwen3Rope::Interleaved { sections, n_dims } => {
                        rope_mrope_interleaved(
                            k_slice,
                            pos,
                            sections,
                            cfg.n_embd_head_k,
                            cfg.freq_base,
                            n_dims,
                        );
                    }
                }
            }
        }

        let mut attn_out = vec![0.0; n_tokens * n_attn];
        let scalar_qwen2vl = cfg.architecture == "qwen2vl" && crate::ops::scalar_mode();
        for head in 0..cfg.n_head {
            let kv_head = head / group_size;
            let q_off = head * cfg.n_embd_head_k;
            let k_off = kv_head * cfg.n_embd_head_k;
            let v_off = kv_head * cfg.n_embd_head_v;
            let attn_off = head * cfg.n_embd_head_v;

            for i in 0..n_tokens {
                let mut max_val = f32::NEG_INFINITY;
                let mut scores = vec![0.0; n_tokens];
                for j in 0..=i {
                    if key_mask.is_some_and(|mask| !mask[j]) {
                        continue;
                    }
                    let q_row =
                        &q_all[i * n_embd_q + q_off..i * n_embd_q + q_off + cfg.n_embd_head_k];
                    let k_row =
                        &k_all[j * n_embd_k + k_off..j * n_embd_k + k_off + cfg.n_embd_head_k];
                    scores[j] = dot_f32(q_row, k_row, cfg.n_embd_head_k) * kq_scale;
                    if scores[j] > max_val {
                        max_val = scores[j];
                    }
                }
                let mut exp_sum = 0.0f32;
                let mut exp_sum_exact = 0.0f64;
                for j in 0..=i {
                    if key_mask.is_some_and(|mask| !mask[j]) {
                        continue;
                    }
                    scores[j] = (scores[j] - max_val).exp();
                    if scalar_qwen2vl {
                        exp_sum_exact += scores[j] as f64;
                    } else {
                        exp_sum += scores[j];
                    }
                }
                let inv_sum = if scalar_qwen2vl {
                    (1.0f64 / exp_sum_exact) as f32
                } else {
                    0.0
                };
                for j in 0..=i {
                    if key_mask.is_some_and(|mask| !mask[j]) {
                        continue;
                    }
                    if scalar_qwen2vl {
                        scores[j] *= inv_sum;
                    } else {
                        scores[j] /= exp_sum;
                    }
                }
                for dim in 0..cfg.n_embd_head_v {
                    let mut sum = 0.0f32;
                    let mut sum_exact = 0.0f64;
                    for j in 0..=i {
                        if key_mask.is_some_and(|mask| !mask[j]) {
                            continue;
                        }
                        let v_row =
                            &v_all[j * n_embd_v + v_off..j * n_embd_v + v_off + cfg.n_embd_head_v];
                        let product = scores[j] * v_row[dim];
                        if scalar_qwen2vl {
                            sum_exact += product as f64;
                        } else {
                            sum += product;
                        }
                    }
                    attn_out[i * n_attn + attn_off + dim] = if scalar_qwen2vl {
                        sum_exact as f32
                    } else {
                        sum
                    };
                }
            }
        }

        let mut attn_proj_out = vec![0.0; n_tokens * cfg.n_embd];
        project_tokens(
            &mut executor,
            &attn_out,
            [(&layer.wo, &mut attn_proj_out, None)],
        )?;

        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            for j in 0..cfg.n_embd {
                hidden[off + j] += attn_proj_out[off + j];
            }
        }

        let mut ffn_normed = vec![0.0; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            rms_norm(
                &hidden[off..off + cfg.n_embd],
                &layer.ffn_norm,
                &mut ffn_normed[off..off + cfg.n_embd],
                cfg.eps,
            );
        }

        if layer.moe_router.is_some() {
            let mut moe_out = vec![0.0; cfg.n_embd];
            for tok in 0..n_tokens {
                let off = tok * cfg.n_embd;
                forward_moe_token(&ffn_normed[off..off + cfg.n_embd], layer, cfg, &mut moe_out)?;
                hidden[off..off + cfg.n_embd]
                    .iter_mut()
                    .zip(&moe_out)
                    .for_each(|(value, update)| *value += *update);
            }
        } else {
            let mut gate_buf = vec![0.0; n_tokens * cfg.n_ff];
            let mut up_buf = vec![0.0; n_tokens * cfg.n_ff];
            project_tokens(
                &mut executor,
                &ffn_normed,
                [
                    (&layer.w_gate, &mut gate_buf, None),
                    (&layer.w_up, &mut up_buf, None),
                ],
            )?;

            for tok in 0..n_tokens {
                let off = tok * cfg.n_ff;
                for i in 0..cfg.n_ff {
                    gate_buf[off + i] = silu(gate_buf[off + i]) * up_buf[off + i];
                }
            }

            // The SwiGLU above wrote the activated product back into
            // `gate_buf`, so that buffer is the down projection's input.
            let mut down_buf = vec![0.0; n_tokens * cfg.n_embd];
            project_tokens(
                &mut executor,
                &gate_buf,
                [(&layer.w_down, &mut down_buf, None)],
            )?;

            for tok in 0..n_tokens {
                let off = tok * cfg.n_embd;
                for i in 0..cfg.n_embd {
                    hidden[off + i] += down_buf[off + i];
                }
            }
        }
        if let Ok(index) = layers.binary_search(&(layer_idx + 1)) {
            for tok in 0..n_tokens {
                let dst = (tok * layers.len() + index) * cfg.n_embd;
                collected[dst..dst + cfg.n_embd]
                    .copy_from_slice(&hidden[tok * cfg.n_embd..(tok + 1) * cfg.n_embd]);
            }
        }
    }

    if !layers.is_empty() {
        return Ok(collected);
    }
    let mut output = vec![0.0; n_tokens * cfg.n_embd];
    for tok in 0..n_tokens {
        let off = tok * cfg.n_embd;
        rms_norm(
            &hidden[off..off + cfg.n_embd],
            &model.output_norm,
            &mut output[off..off + cfg.n_embd],
            cfg.eps,
        );
    }

    Ok(output)
}

#[cfg(feature = "vulkan")]
fn text_encode_vulkan(
    model: &Qwen3Model,
    embeddings: &[f32],
    n_tokens: usize,
    context: &'static crate::vulkan::VulkanContext,
) -> Result<Option<Vec<f32>>, crate::vulkan::VulkanError> {
    let Some(mut session) = Qwen3VulkanSession::try_new(model, n_tokens, context)? else {
        return Ok(None);
    };
    let width = model.config.n_embd;
    let mut hidden = Vec::with_capacity(n_tokens * width);
    for (position, input) in embeddings.chunks_exact(width).enumerate() {
        hidden.extend_from_slice(session.forward_hidden_token(input, position)?);
        session.commit_token();
    }
    Ok(Some(hidden))
}

fn apply_qk_norms(
    q_all: &mut [f32],
    k_all: &mut [f32],
    n_tokens: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    q_norm: &[f32],
    k_norm: &[f32],
    eps: f32,
) {
    let q_width = n_head * head_dim;
    let k_width = n_head_kv * head_dim;
    for tok in 0..n_tokens {
        let q_row = &mut q_all[tok * q_width..(tok + 1) * q_width];
        for head in q_row.chunks_exact_mut(head_dim) {
            rms_norm_inplace(head, q_norm, eps);
        }
        let k_row = &mut k_all[tok * k_width..(tok + 1) * k_width];
        for head in k_row.chunks_exact_mut(head_dim) {
            rms_norm_inplace(head, k_norm, eps);
        }
    }
}

pub fn run_shared_inference(
    source: Arc<dyn TensorSource>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    thinking: bool,
    prefill_batch_size: usize,
) -> Result<(), String> {
    let started = Instant::now();
    let tokenizer = Arc::new(
        BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?,
    );
    let available_threads = std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(4);
    let pool = Arc::new(ComputePool::new(resolve_thread_count(
        n_threads_arg,
        available_threads,
    )));
    let model = Qwen3Model::from_source(source, Arc::clone(&tokenizer), pool)?;
    let input_tokens = build_qwen_chat_prompt(
        &tokenizer,
        &[QwenMessage {
            role: "user",
            content: prompt,
        }],
        thinking,
    )?;
    let positions = qwen_text_positions(input_tokens.len());
    println!(
        "Model: {} | n_embd={} n_layer={} n_head={} n_head_kv={} n_ff={} | loaded in {}ms",
        model.config().architecture,
        model.config().n_embd,
        model.config().n_layer,
        model.config().n_head,
        model.config().n_head_kv,
        model.config().n_ff,
        started.elapsed().as_millis(),
    );
    eprintln!("compute pool: {} threads", model.pool().n_threads());
    println!("Prompt: {} ({} tokens)", prompt, input_tokens.len());
    print!("Output: ");
    io::stdout().flush().map_err(|error| error.to_string())?;
    let inference_started = Instant::now();
    let generation = model.generate(
        Qwen3Input {
            token_ids: &input_tokens,
            positions: &positions,
            embeddings: None,
            deepstack_embeddings: None,
        },
        Qwen3GenerateOptions {
            max_new_tokens: max_tokens,
            temperature,
            prefill_batch_size,
        },
    )?;
    print!("{}", generation.text);
    io::stdout().flush().map_err(|error| error.to_string())?;
    let elapsed_ms = inference_started.elapsed().as_millis();
    let tokens_per_second = if elapsed_ms > 0 {
        generation.token_ids.len() as f64 / elapsed_ms as f64 * 1000.0
    } else {
        0.0
    };
    println!();
    println!(
        "[end-to-end: {} output tokens in {}ms | {:.1} tok/s]",
        generation.token_ids.len(),
        elapsed_ms,
        tokens_per_second,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::apply_qk_norms;

    #[test]
    fn qk_norms_are_applied_to_every_token_and_head() {
        let mut q = vec![3.0, 4.0, 5.0, 12.0];
        let mut k = vec![3.0, 4.0, 5.0, 12.0];
        apply_qk_norms(&mut q, &mut k, 2, 1, 1, 2, &[1.0, 1.0], &[1.0, 1.0], 0.0);
        assert_eq!(q, [0.84852815, 1.1313709, 0.54392827, 1.3054278]);
        assert_eq!(k, q);
    }
}
