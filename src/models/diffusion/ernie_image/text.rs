//! ERNIE-Image text encoder: Ministral-3 hidden_states[-2], after 25 of 26
//! transformer blocks, before the final block and output RMSNorm.
//! Reference: stable-diffusion.cpp 3f8527a, conditioner.hpp (out_layers={25}).

use std::sync::Arc;

use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::ops::{embedding_lookup, rms_norm, rope_neox_inplace, silu_mul_inplace};

use super::{linear_into, validate_component, Component, Q8Scratch};

const HIDDEN: usize = super::dit::TEXT_IN_DIM; // 3072
const QUERY_HEADS: usize = super::dit::TEXT_NUM_HEADS; // 32
const KV_HEADS: usize = super::dit::TEXT_NUM_KV_HEADS; // 8
const HEAD_WIDTH: usize = super::dit::TEXT_HEAD_DIM; // 128
const QUERY_WIDTH: usize = QUERY_HEADS * HEAD_WIDTH;
const KV_WIDTH: usize = KV_HEADS * HEAD_WIDTH;
const FFN_WIDTH: usize = super::dit::TEXT_FFN; // 9216
const LAYERS: usize = super::dit::TEXT_NUM_LAYERS;
const STOP_LAYER: usize = LAYERS - 2;
const RMS_EPSILON: f32 = 1e-5;
const ROPE_BASE: f32 = 1_000_000.0;

struct TextLayer {
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
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
        let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
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
            prompt,
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
            .ok_or("Ministral-3 hidden state size overflow")?;
        let mut output = vec![0.0_f32; output_len];
        let embedding = self
            .source
            .tensor_slice("token_embd.weight")
            .or_else(|| self.source.tensor_slice("model.embed_tokens.weight"))
            .ok_or("Missing tensor data: token_embd.weight (llama.cpp) or model.embed_tokens.weight (HF)")?;
        let embd_type = self
            .source
            .tensor_info("token_embd.weight")
            .or_else(|| self.source.tensor_info("model.embed_tokens.weight"))
            .ok_or("Missing tensor info for token embedding")?
            .ggml_type;
        let mut scratch = TextScratch::new(token_count);

        // Embedding lookup for all tokens at once
        for (position, &id) in ids.iter().enumerate() {
            embedding_lookup(
                embedding,
                id,
                HIDDEN,
                embd_type,
                &mut scratch.hidden[position * HIDDEN..(position + 1) * HIDDEN],
            );
        }

        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "ernie_image.text.embedding",
            None,
            &[token_count, HIDDEN],
            &scratch.hidden,
        ));
        // Full forward through all layers in one pass
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
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "ernie_image.text.block",
                Some(layer_index),
                &[token_count, HIDDEN],
                &scratch.hidden,
            ));
        }
        // The post-stop hidden state is in scratch.hidden, which is exactly
        // `token_count * HIDDEN` elements -- the size of `output`.
        output.copy_from_slice(&scratch.hidden);
        Ok(output)
    }
}

struct TextScratch {
    hidden: Vec<f32>,
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
    // The unsloth Ministral GGUF uses llama.cpp tensor naming (token_embd,
    // blk.X.{attn_q,attn_k,...}, ffn_*, attn_norm, ffn_norm, output_norm) --
    // not the HuggingFace names we used for the Z-Image Qwen3 text encoder.
    let prefix = format!("blk.{layer}");
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
        input_norm: vector("attn_norm.weight", HIDDEN)?,
        post_attention_norm: vector("ffn_norm.weight", HIDDEN)?,
        q_proj: format!("{prefix}.attn_q.weight"),
        k_proj: format!("{prefix}.attn_k.weight"),
        v_proj: format!("{prefix}.attn_v.weight"),
        o_proj: format!("{prefix}.attn_output.weight"),
        gate_proj: format!("{prefix}.ffn_gate.weight"),
        up_proj: format!("{prefix}.ffn_up.weight"),
        down_proj: format!("{prefix}.ffn_down.weight"),
    })
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
    for (input, output) in hidden
        .chunks_exact(HIDDEN)
        .zip(normalized.chunks_exact_mut(HIDDEN))
    {
        rms_norm(input, &layer.input_norm, output, RMS_EPSILON);
    }
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(
        "ernie_image.text.norm",
        None,
        &[n_tokens, HIDDEN],
        &normalized,
    ));
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

    // Q/K RoPE (no Q/K RMS norms in Ministral-3 -- only Qwen3/Qwen3.5 has them)
    for token in 0..n_tokens {
        let q_row = &mut q_buf[token * QUERY_WIDTH..(token + 1) * QUERY_WIDTH];
        let k_row = &mut k_buf[token * KV_WIDTH..(token + 1) * KV_WIDTH];
        for head in 0..QUERY_HEADS {
            let q_start = head * HEAD_WIDTH;
            let q_chunk = &mut q_row[q_start..q_start + HEAD_WIDTH];
            rope_neox_inplace(q_chunk, token, HEAD_WIDTH, ROPE_BASE);
        }
        for head in 0..KV_HEADS {
            let k_start = head * HEAD_WIDTH;
            let k_chunk = &mut k_row[k_start..k_start + HEAD_WIDTH];
            rope_neox_inplace(k_chunk, token, HEAD_WIDTH, ROPE_BASE);
        }
    }

    for d in 0..KV_WIDTH {
        for token in 0..n_tokens {
            gate[d * n_tokens + token] = v_buf[token * KV_WIDTH + d];
        }
    }
    let scale = 1.0 / (HEAD_WIDTH as f32).sqrt();
    for head in 0..QUERY_HEADS {
        let kv_offset = (head / (QUERY_HEADS / KV_HEADS)) * HEAD_WIDTH;
        for query_idx in 0..n_tokens {
            let q_offset = query_idx * QUERY_WIDTH + head * HEAD_WIDTH;
            let q = &q_buf[q_offset..q_offset + HEAD_WIDTH];
            let length = query_idx + 1;
            for key_idx in 0..length {
                let k_offset = key_idx * KV_WIDTH + kv_offset;
                scores[key_idx] =
                    crate::ops::dot_f32(q, &k_buf[k_offset..k_offset + HEAD_WIDTH], HEAD_WIDTH)
                        * scale;
            }
            crate::ops::softmax_inplace(&mut scores[..length]);
            for d in 0..HEAD_WIDTH {
                let column = (kv_offset + d) * n_tokens;
                attn[query_idx * QUERY_WIDTH + head * HEAD_WIDTH + d] =
                    crate::ops::dot_f32(&scores[..length], &gate[column..column + length], length);
            }
        }
    }

    // output projection + residual. We re-use the `up` buffer (FFN up
    // projection output) as scratch for the o_proj output, since the
    // subsequent MLP step overwrites it with the next round's `up_proj`
    // matmul.
    for token in 0..n_tokens {
        let q_proj_in = &attn[token * QUERY_WIDTH..(token + 1) * QUERY_WIDTH];
        let o_proj_out = &mut up[token * HIDDEN..(token + 1) * HIDDEN];
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
            hidden[token * HIDDEN + d] += up[token * HIDDEN + d];
        }
    }

    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(
        "ernie_image.text.attn_residual",
        None,
        &[n_tokens, HIDDEN],
        hidden,
    ));
    // post-attention norm + MLP
    for (input, output) in hidden
        .chunks_exact(HIDDEN)
        .zip(normalized.chunks_exact_mut(HIDDEN))
    {
        rms_norm(input, &layer.post_attention_norm, output, RMS_EPSILON);
    }
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
            &up[token * FFN_WIDTH..(token + 1) * FFN_WIDTH],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the real Ministral GGUF and pinned Oracle fixtures"]
    fn oracle_text_fixture() {
        let model = std::env::var("RMI_MINISTRAL_3_3B_INSTRUCT_Q4_K_MODEL").unwrap();
        let fixtures = std::env::var("RMI_ERNIE_ORACLE_FIXTURES").unwrap();
        let source = crate::format::ggufrs::open_model_source(
            std::path::Path::new(&model),
            crate::format::ggufrs::ComponentRole::Llm,
        )
        .unwrap();
        let encoder =
            ErnieImageTextEncoder::load(Arc::from(source), Arc::new(ComputePool::new(1))).unwrap();
        let output = encoder.encode("a lovely cat").unwrap();
        let expected = std::fs::read(format!("{fixtures}/rmi.ernie.context.f32")).unwrap();
        assert_eq!(output.len() * 4, expected.len());
        let mismatch = output
            .iter()
            .zip(expected.chunks_exact(4))
            .position(|(a, b)| a.to_bits() != u32::from_le_bytes(b.try_into().unwrap()));
        assert_eq!(mismatch, None, "first raw-bit text difference");
    }
}
