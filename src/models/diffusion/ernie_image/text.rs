//! ERNIE-Image text encoder: forward Ministral-3 (mistral3 arch, llama
//! trunk) up to layer 35 and return the per-token hidden states at the
//! hidden_size boundary (3072).
//!
//! Reference for arch sizes:
//! - `tests/ministral3_3b_instruct_q4_k_m.rs` (`pick(&loader, "mistral3.embedding_length") == 3072`)
//!
//! The forward mirrors `src/models/diffusion/z_image/text.rs` (Qwen3 encoder
//! for Z-Image) with the model dimensions swapped: this module drives 35
//! transformer layers with hidden=3072, q_heads=32, kv_heads=8, head_dim=128,
//! FFN=9216, vocab=65536. We stop after layer 35 because that's where the
//! LLM's last hidden state lives — running `model.norm` afterward would mix
//! in a final RMSNorm that the ERNIE-Image text encoder does not need.

use std::sync::Arc;

use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::ops::{
    attention_value_reduce, embedding_lookup, rms_norm, rms_norm_inplace, rope_neox_inplace,
    silu_mul_inplace,
};

use super::{linear_into, validate_component, Component, Q8Scratch};

const HIDDEN: usize = super::dit::TEXT_IN_DIM; // 3072
const QUERY_HEADS: usize = super::dit::TEXT_NUM_HEADS; // 32
const KV_HEADS: usize = super::dit::TEXT_NUM_KV_HEADS; // 8
const HEAD_WIDTH: usize = super::dit::TEXT_HEAD_DIM; // 128
const QUERY_WIDTH: usize = QUERY_HEADS * HEAD_WIDTH;
const KV_WIDTH: usize = KV_HEADS * HEAD_WIDTH;
const FFN_WIDTH: usize = super::dit::TEXT_FFN; // 9216
const LAYERS: usize = super::dit::TEXT_NUM_LAYERS; // 35
const STOP_LAYER: usize = LAYERS - 1;
const RMS_EPSILON: f32 = 1e-6;
const ROPE_BASE: f32 = 1_000_000.0;

struct TextLayer {
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    q_proj: String,
    k_proj: String,
    v_proj: String,
    o_proj: String,
    gate_proj: String,
    up_proj: String,
    down_proj: String,
}

pub(crate) struct ErnieImageTextEncoder {
    source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    tokenizer: BPETokenizer,
    layers: Vec<TextLayer>,
}

impl ErnieImageTextEncoder {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        validate_component(source.as_ref(), Component::Text)?;
        let tokenizer = BPETokenizer::from_gguf_metadata(|key| {
            source.metadata(key).cloned()
        })
        .map_err(|e| format!("Ministral-3 tokenizer: {e}"))?;
        let mut layers = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            layers.push(load_layer(source.as_ref(), layer)?);
        }
        Ok(Self {
            source,
            pool,
            tokenizer,
            layers,
        })
    }

    /// Encode a prompt into per-token hidden states of width [`HIDDEN`]
    /// (= 3072). The output layout is `[text_tokens, HIDDEN]` (no batch dim).
    pub(crate) fn encode(&self, prompt: &str) -> Result<Vec<f32>, String> {
        let total_start = std::time::Instant::now();
        let t_tok = std::time::Instant::now();
        let ids = self.tokenizer.encode(
            &ernie_image_prompt(prompt),
            EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        );
        if ids.is_empty() {
            return Err("ERNIE-Image prompt produced no tokens".into());
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::token_ids(
            "ernie_image.prompt_ids",
            &ids,
        ));
        let t_tok = t_tok.elapsed();
        let t_fwd = std::time::Instant::now();
        let output = self.forward_to_block(&ids)?;
        let t_fwd = t_fwd.elapsed();
        eprintln!(
            "[ernie-image-text-profile] n_tokens={}  forward={:.1}ms  tokenize={:.1}ms  total={:.1}ms",
            ids.len(),
            t_fwd.as_secs_f64() * 1000.0,
            t_tok.as_secs_f64() * 1000.0,
            total_start.elapsed().as_secs_f64() * 1000.0,
        );
        Ok(output)
    }

    fn forward_to_block(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        if ids.is_empty() {
            return Err("Invalid Ministral-3 token sequence".into());
        }
        let token_count = ids.len();
        let output_len = token_count
            .checked_mul(HIDDEN)
            .ok_or("Ministral-3 layer-35 output size overflow")?;
        let mut output = vec![0.0_f32; output_len];
        let embedding = self
            .source
            .tensor_slice("model.embed_tokens.weight")
            .ok_or("Missing tensor data: model.embed_tokens.weight")?;
        let embd_type = self
            .source
            .tensor_info("model.embed_tokens.weight")
            .ok_or("Missing tensor info: model.embed_tokens.weight")?
            .ggml_type;
        let mut scratch = TextScratch::new(token_count);

        for (position, &id) in ids.iter().enumerate() {
            embedding_lookup(embedding, id, HIDDEN, embd_type, &mut scratch.hidden);
            for layer_index in 0..=STOP_LAYER {
                forward_layer(
                    self.source.as_ref(),
                    &self.layers[layer_index],
                    &mut scratch.hidden,
                    token_count,
                    &mut scratch.q,
                    &mut scratch.k,
                    &mut scratch.v,
                    &mut scratch.attn,
                    &mut scratch.scores,
                    &mut scratch.gate,
                    &mut scratch.up,
                    &mut scratch.q8,
                    self.pool.as_ref(),
                )?;
            }
            // Copy out the post-stop hidden state for this token.
            output[position * HIDDEN..(position + 1) * HIDDEN]
                .copy_from_slice(&scratch.hidden);
        }
        Ok(output)
    }
}

struct TextScratch {
    hidden: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    scores: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    q8: Q8Scratch,
}

impl TextScratch {
    fn new(n_tokens: usize) -> Self {
        Self {
            hidden: vec![0.0; n_tokens * HIDDEN],
            normed: vec![0.0; n_tokens * HIDDEN],
            q: vec![0.0; n_tokens * QUERY_WIDTH],
            k: vec![0.0; n_tokens * KV_WIDTH],
            v: vec![0.0; n_tokens * KV_WIDTH],
            attn: vec![0.0; n_tokens * QUERY_WIDTH],
            scores: vec![0.0; n_tokens],
            gate: vec![0.0; n_tokens * FFN_WIDTH],
            up: vec![0.0; n_tokens * FFN_WIDTH],
            q8: Q8Scratch::new(FFN_WIDTH.max(HIDDEN)),
        }
    }
}

fn load_layer(source: &dyn TensorSource, layer: usize) -> Result<TextLayer, String> {
    let prefix = format!("model.layers.{layer}");
    let vector = |suffix: &str, len: usize| -> Result<Vec<f32>, String> {
        let info = source
            .tensor_info(&format!("{prefix}.{suffix}"))
            .ok_or_else(|| format!("Missing tensor: {prefix}.{suffix}"))?;
        if info.dims != [len as u64] {
            return Err(format!("Invalid {prefix}.{suffix} dimensions"));
        }
        if !matches!(info.ggml_type, GGMLType::F32 | GGMLType::BF16) {
            return Err(format!(
                "Invalid {prefix}.{suffix} type {:?}: expected F32/BF16",
                info.ggml_type
            ));
        }
        let bytes = source
            .tensor_slice(&format!("{prefix}.{suffix}"))
            .ok_or_else(|| format!("Missing tensor data: {prefix}.{suffix}"))?;
        let mut out = vec![0.0_f32; len];
        match info.ggml_type {
            GGMLType::F32 => {
                for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                    *dst = f32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            GGMLType::BF16 => {
                for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    *dst = half::bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap()))
                        .to_f32();
                }
            }
            _ => unreachable!(),
        }
        Ok(out)
    };
    Ok(TextLayer {
        input_norm: vector("input_layernorm.weight", HIDDEN)?,
        post_attention_norm: vector("post_attention_layernorm.weight", HIDDEN)?,
        q_norm: vector("self_attn.q_norm.weight", HEAD_WIDTH)?,
        k_norm: vector("self_attn.k_norm.weight", HEAD_WIDTH)?,
        q_proj: format!("{prefix}.self_attn.q_proj.weight"),
        k_proj: format!("{prefix}.self_attn.k_proj.weight"),
        v_proj: format!("{prefix}.self_attn.v_proj.weight"),
        o_proj: format!("{prefix}.self_attn.o_proj.weight"),
        gate_proj: format!("{prefix}.mlp.gate_proj.weight"),
        up_proj: format!("{prefix}.mlp.up_proj.weight"),
        down_proj: format!("{prefix}.mlp.down_proj.weight"),
    })
}

/// Ministral-3 uses Mistral chat template (see PR #147 / `docs/usage/ministral3.md`).
/// The exact template is `<s>[INST] {prompt} [/INST]` per Mistral 3's documented
/// format; we mirror what `src/app/text/generation.rs` already routes for
/// `mistral3` arch.
fn ernie_image_prompt(prompt: &str) -> String {
    format!("<s>[INST] {prompt} [/INST]")
}

#[allow(clippy::too_many_arguments)]
fn forward_layer(
    source: &dyn TensorSource,
    layer: &TextLayer,
    hidden: &mut [f32],
    n_tokens: usize,
    q_buf: &mut [f32],
    k_buf: &mut [f32],
    v_buf: &mut [f32],
    attn: &mut [f32],
    scores: &mut [f32],
    gate: &mut [f32],
    up: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
) -> Result<(), String> {
    // attention
    let mut normalized = vec![0.0_f32; n_tokens * HIDDEN];
    rms_norm(
        hidden,
        &layer.input_norm,
        &mut normalized,
        RMS_EPSILON,
    );
    for token in 0..n_tokens {
        let row = &normalized[token * HIDDEN..(token + 1) * HIDDEN];
        linear_into(
            source,
            &layer.q_proj,
            HIDDEN,
            QUERY_WIDTH,
            row,
            &mut q_buf[token * QUERY_WIDTH..(token + 1) * QUERY_WIDTH],
            q8,
            pool,
        )?;
        linear_into(
            source,
            &layer.k_proj,
            HIDDEN,
            KV_WIDTH,
            row,
            &mut k_buf[token * KV_WIDTH..(token + 1) * KV_WIDTH],
            q8,
            pool,
        )?;
        linear_into(
            source,
            &layer.v_proj,
            HIDDEN,
            KV_WIDTH,
            row,
            &mut v_buf[token * KV_WIDTH..(token + 1) * KV_WIDTH],
            q8,
            pool,
        )?;
    }

    // Q/K RMS norm + RoPE
    for token in 0..n_tokens {
        let q_row = &mut q_buf[token * QUERY_WIDTH..(token + 1) * QUERY_WIDTH];
        let k_row = &mut k_buf[token * KV_WIDTH..(token + 1) * KV_WIDTH];
        for head in 0..QUERY_HEADS {
            let q_start = head * HEAD_WIDTH;
            let q_chunk = &mut q_row[q_start..q_start + HEAD_WIDTH];
            rms_norm_inplace(q_chunk, &layer.q_norm, RMS_EPSILON);
            rope_neox_inplace(q_chunk, token, HEAD_WIDTH / 2, ROPE_BASE);
        }
        for head in 0..KV_HEADS {
            let k_start = head * HEAD_WIDTH;
            let k_chunk = &mut k_row[k_start..k_start + HEAD_WIDTH];
            rms_norm_inplace(k_chunk, &layer.k_norm, RMS_EPSILON);
            rope_neox_inplace(k_chunk, token, HEAD_WIDTH / 2, ROPE_BASE);
        }
    }

    // attention output: A = softmax(Q K^T / sqrt(d)) V
    let scale = 1.0 / (HEAD_WIDTH as f32).sqrt();
    for head in 0..QUERY_HEADS {
        for query_idx in 0..n_tokens {
            // Q @ K^T, head-wise.
            let q_offset = query_idx * QUERY_WIDTH + head * HEAD_WIDTH;
            let q = &q_buf[q_offset..q_offset + HEAD_WIDTH];
            let mut max_score = f32::NEG_INFINITY;
            for key_idx in 0..n_tokens {
                let k_offset = key_idx * KV_WIDTH + (head / (QUERY_HEADS / KV_HEADS)) * HEAD_WIDTH;
                let k = &k_buf[k_offset..k_offset + HEAD_WIDTH];
                let mut dot = 0.0_f32;
                for d in 0..HEAD_WIDTH {
                    dot += q[d] * k[d];
                }
                let score = dot * scale;
                scores[key_idx] = score;
                if score > max_score {
                    max_score = score;
                }
            }
            let mut sum = 0.0_f32;
            for s in scores.iter_mut().take(n_tokens) {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            let inv = 1.0 / sum;
            for s in scores.iter_mut().take(n_tokens) {
                *s *= inv;
            }
            // Softmax(Q K^T) @ V.
            let out_offset = query_idx * QUERY_WIDTH + head * HEAD_WIDTH;
            for d in 0..HEAD_WIDTH {
                attn[out_offset + d] = 0.0;
            }
            for key_idx in 0..n_tokens {
                let v_offset = key_idx * KV_WIDTH + (head / (QUERY_HEADS / KV_HEADS)) * HEAD_WIDTH;
                let w = scores[key_idx];
                for d in 0..HEAD_WIDTH {
                    attn[out_offset + d] += w * v_buf[v_offset + d];
                }
            }
        }
    }

    // output projection + residual
    for token in 0..n_tokens {
        let (q_proj_in, o_proj_out) = attn.split_at_mut((token + 1) * HIDDEN);
        let q_proj_in = &q_proj_in[token * QUERY_WIDTH..(token + 1) * QUERY_WIDTH];
        let o_proj_out = &mut o_proj_out[..HIDDEN];
        linear_into(
            source,
            &layer.o_proj,
            QUERY_WIDTH,
            HIDDEN,
            q_proj_in,
            o_proj_out,
            q8,
            pool,
        )?;
    }
    for token in 0..n_tokens {
        for d in 0..HIDDEN {
            hidden[token * HIDDEN + d] += attn[token * HIDDEN + d];
        }
    }

    // post-attention norm + MLP
    rms_norm(hidden, &layer.post_attention_norm, &mut normalized, RMS_EPSILON);
    for token in 0..n_tokens {
        let row = &normalized[token * HIDDEN..(token + 1) * HIDDEN];
        linear_into(
            source,
            &layer.gate_proj,
            HIDDEN,
            FFN_WIDTH,
            row,
            &mut gate[token * FFN_WIDTH..(token + 1) * FFN_WIDTH],
            q8,
            pool,
        )?;
        linear_into(
            source,
            &layer.up_proj,
            HIDDEN,
            FFN_WIDTH,
            row,
            &mut up[token * FFN_WIDTH..(token + 1) * FFN_WIDTH],
            q8,
            pool,
        )?;
    }
    silu_mul_inplace(gate, up);
    for token in 0..n_tokens {
        linear_into(
            source,
            &layer.down_proj,
            FFN_WIDTH,
            HIDDEN,
            &gate[token * FFN_WIDTH..(token + 1) * FFN_WIDTH],
            &mut attn[token * HIDDEN..(token + 1) * HIDDEN],
            q8,
            pool,
        )?;
    }
    for token in 0..n_tokens {
        for d in 0..HIDDEN {
            hidden[token * HIDDEN + d] += attn[token * HIDDEN + d];
        }
    }
    Ok(())
}

// Silence unused warning for MetaValue when both #[cfg(feature = "parity-trace")]
// gates remove the call site.
#[allow(dead_code)]
fn _unused_meta_value(_: MetaValue) {}
