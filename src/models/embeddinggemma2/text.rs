//! Text forward path for EmbeddingGemma-2.
//!
//! Bidirectional gemma4-text encoder with:
//! - 24 layers, hidden 512, FFN 2048, 4 Q heads + per-layer KV heads
//! - per-layer head_dim ∈ {256 (sliding), 512 (full)}
//! - per-layer rope.theta ∈ {10K (sliding), 1M (full)}
//! - 5-norm sandwich: input_layernorm / post_attention_layernorm /
//!   pre_feedforward_layernorm / post_feedforward_layernorm / ple_post_norm
//! - per-layer 1-value residual scaling (`layer_scalar`)
//! - per-layer-input PLE block: per_layer_input_gate → rmsnorm → per_layer_projection
//!   → post_per_layer_input_norm
//! - shared per_layer_model_projection over the full vocab (12288-dim)
//! - embedding_projection (`text_embproj.weight`): 512 → 768, mean pool → output
//!
//! Implementation note: first-pass version dequantizes Q8_0 → F32 once per
//! weight and uses dot_f32. Single-threaded. Optimize in a follow-up.

use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{QuantizedTensor, Weight};
use std::sync::Arc;

pub const EMBEDDING_DIM: usize = 768;
pub const EPS: f32 = 1e-6;
pub const SLIDING_WINDOW: usize = 512;

/// Full-attention layer indices (every 6th, matching the E2B/E4B
/// `sliding_window_pattern` convention).
pub const FULL_ATTENTION_LAYERS: &[usize] = &[5, 11, 17, 23];

#[derive(Debug, Clone)]
pub struct EmbeddingGemmaTextConfig {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub head_dims: Vec<usize>,
    pub kv_heads: Vec<usize>,
    pub rope_freq_base_sliding: f32,
    pub rope_freq_base_full: f32,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub per_layer_width: usize,
    pub n_ctx: usize,
}

impl EmbeddingGemmaTextConfig {
    pub fn head_dim(&self, layer: usize) -> usize {
        self.head_dims[layer]
    }
    pub fn kv_heads(&self, layer: usize) -> usize {
        self.kv_heads[layer]
    }
    pub fn is_full_attention(&self, layer: usize) -> bool {
        FULL_ATTENTION_LAYERS.contains(&layer)
    }
    pub fn rope_freq_base(&self, layer: usize) -> f32 {
        if self.is_full_attention(layer) {
            self.rope_freq_base_full
        } else {
            self.rope_freq_base_sliding
        }
    }
    pub fn sliding_window(&self) -> usize {
        SLIDING_WINDOW
    }
    pub fn from_metadata<S: TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        let get = |key: &str| -> Result<usize, String> {
            source
                .metadata(key)
                .and_then(|v| v.to_u64())
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| format!("Missing metadata: {key}"))
        };
        let n_embd = get("gemma4-embedding.embedding_length")?;
        let n_layer = get("gemma4-embedding.block_count")?;
        let n_head = get("gemma4-embedding.attention.head_count")?;
        let n_ff = get("gemma4-embedding.feed_forward_length")?;
        let n_vocab = get("gemma4-embedding.vocab_size")?;
        let n_ctx = get("gemma4-embedding.context_length")?;
        let rope_freq_base_full = source
            .metadata("gemma4-embedding.rope.freq_base_full")
            .and_then(|v| v.to_f64())
            .map(|v| v as f32)
            .unwrap_or(1_000_000.0);
        let rope_freq_base_sliding = source
            .metadata("gemma4-embedding.rope.freq_base_sliding")
            .and_then(|v| v.to_f64())
            .map(|v| v as f32)
            .unwrap_or(10_000.0);

        // Parse the parallel-arrays for per-layer head_dim and kv_heads.
        let parse_int_array = |key: &str| -> Result<Vec<usize>, String> {
            let arr = source
                .metadata(key)
                .and_then(|v| v.to_arr())
                .ok_or_else(|| format!("Missing metadata: {key}"))?;
            Ok(arr
                .iter()
                .filter_map(|v| match v {
                    MetaValue::String(s) => s.parse().ok(),
                    MetaValue::Uint32(n) => Some(*n as usize),
                    MetaValue::Uint64(n) => Some(*n as usize),
                    _ => None,
                })
                .collect())
        };
        let head_dims = parse_int_array("gemma4-embedding.attention.head_dim_per_layer")?;
        let kv_heads = parse_int_array("gemma4-embedding.attention.head_count_kv_per_layer")?;
        if head_dims.len() != n_layer {
            return Err(format!(
                "head_dim_per_layer length {} != block_count {n_layer}",
                head_dims.len()
            ));
        }
        if kv_heads.len() != n_layer {
            return Err(format!(
                "head_count_kv_per_layer length {} != block_count {n_layer}",
                kv_heads.len()
            ));
        }
        let per_layer_all: usize = source
            .tensor_info("per_layer_model_proj.weight")
            .map(|info| {
                let d0: u64 = info.dims[0];
                d0 as usize
            })
            .ok_or("Missing tensor: per_layer_model_proj.weight")?;
        Ok(Self {
            n_embd,
            n_layer,
            n_head,
            head_dims,
            kv_heads,
            rope_freq_base_full,
            rope_freq_base_sliding,
            n_ff,
            n_vocab,
            per_layer_width: per_layer_all,
            n_ctx,
        })
    }
}

fn decode_f32(data: &[u8], expected: usize) -> Vec<f32> {
    let n = data.len() / 4;
    assert_eq!(n, expected, "f32 payload length mismatch: {} != {expected}", n);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u32::from_le_bytes([data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]]);
        out.push(f32::from_bits(bits));
    }
    out
}

fn dequant_to_f32(raw: &[u8], qtype: GGMLType, n_in: usize, n_out: usize) -> Result<Vec<f32>, String> {
    // GGUF stores Linear weights as `[n_in, n_out]` flat row-major.
    // We dequantize to PyTorch convention `[n_out, n_in]` so the
    // downstream Linear::forward can iterate `matrix[out * n_in + in]`.
    let n = n_in * n_out;
    match qtype {
        GGMLType::F32 => {
            // raw is GGUF order: tensor[k * n_out + j] for k in 0..n_in, j in 0..n_out.
            // We want out[o * n_in + i] = raw[i * n_out + o].
            let src = decode_f32(raw, n);
            let mut out = vec![0.0f32; n];
            for i in 0..n_in {
                for o in 0..n_out {
                    out[o * n_in + i] = src[i * n_out + o];
                }
            }
            Ok(out)
        }
        GGMLType::BF16 => {
            // Same transpose.
            let mut src = Vec::with_capacity(n);
            for idx in 0..n {
                let bits = u16::from_le_bytes([raw[idx * 2], raw[idx * 2 + 1]]);
                src.push(f32::from_bits((bits as u32) << 16));
            }
            let mut out = vec![0.0f32; n];
            for i in 0..n_in {
                for o in 0..n_out {
                    out[o * n_in + i] = src[i * n_out + o];
                }
            }
            Ok(out)
        }
        GGMLType::Q8_0 => {
            // Q8_0 blocks of 32 elements, 34 bytes each: [f16 scale, 32 x int8].
            // GGUF storage: tensor shape [n_in, n_out], row-major. Each row contains
            // n_out elements as Q8_0 blocks (n_out/32 blocks per row).
            // Storage layout: row i has n_out/32 blocks of 34 bytes each.
            // row_base = i * (n_out/32) * 34 = i * row_stride.
            let block_bytes = 34;
            if n_out % 32 != 0 {
                return Err(format!("Q8_0 dequant: n_out={n_out} not multiple of 32"));
            }
            let blocks_per_gguf_row = n_out / 32;
            let n_blocks = n_in * blocks_per_gguf_row;
            if raw.len() != n_blocks * block_bytes {
                return Err(format!(
                    "Q8_0 dequant: payload {} != n_in({n_in}) * blocks_per_gguf_row({blocks_per_gguf_row}) * 34",
                    raw.len()
                ));
            }
            // First dequantize GGUF rows into [n_in, n_out] (PyTorch layout would be
            // [n_out, n_in]; we transpose at the end).
            let mut gguf_decoded = vec![0.0f32; n_in * n_out];
            for row in 0..n_in {
                let row_base = row * blocks_per_gguf_row * block_bytes;
                for b in 0..blocks_per_gguf_row {
                    let blk = &raw[row_base + b * block_bytes..row_base + (b + 1) * block_bytes];
                    let scale_bits = u16::from_le_bytes([blk[0], blk[1]]);
                    // bf16 → f32: high 16 bits of f32 = scale_bits as u16
                    let scale = f32::from_bits((scale_bits as u32) << 16);
                    for j in 0..32 {
                        let raw_byte = blk[2 + j];
                        let signed = raw_byte as i8 as f32;
                        gguf_decoded[row * n_out + b * 32 + j] = scale * signed;
                    }
                }
            }
            // Transpose to PyTorch convention [n_out, n_in].
            let mut out = vec![0.0f32; n_out * n_in];
            for i in 0..n_in {
                for o in 0..n_out {
                    out[o * n_in + i] = gguf_decoded[i * n_out + o];
                }
            }
            Ok(out)
        }
        other => Err(format!("dequant_to_f32: unsupported qtype {other:?}")),
    }
}

/// Linear weight stored as a flat F32 matrix (row-major `[n_out, n_in]`,
/// i.e. `out[i, k] = matrix[i * n_in + k]`). We compute
/// `out_vec[i] = sum_k matrix[i, k] * input[k]`.
struct Linear {
    matrix: Vec<f32>,
    n_in: usize,
    n_out: usize,
}

impl Linear {
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.n_in, "Linear::forward input shape");
        assert_eq!(output.len(), self.n_out, "Linear::forward output shape");
        // For each output row i, compute the dot product with input.
        // Single-threaded; the inner loop is dense and amenable to SIMD
        // dot_f32 but for this first pass we just do a scalar loop.
        for i in 0..self.n_out {
            let row = &self.matrix[i * self.n_in..(i + 1) * self.n_in];
            let mut s = 0.0f32;
            for k in 0..self.n_in {
                s += row[k] * input[k];
            }
            output[i] = s;
        }
    }
}

fn linear_from_gguf(source: &dyn TensorSource, name: &str, n_in: usize, n_out: usize) -> Result<Linear, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let raw = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let matrix = dequant_to_f32(raw, info.ggml_type, n_in, n_out)?;
    Ok(Linear {
        matrix,
        n_in,
        n_out,
    })
}

struct LayerWeights {
    attn_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    pre_ffn_norm: Vec<f32>,
    post_ffn_norm: Vec<f32>,
    layer_scalar: f32,
    ple_input_gate: Linear,
    ple_projection: Linear,
    ple_post_norm: Vec<f32>,
    attn_q: Linear,
    attn_k: Linear,
    attn_v: Linear,
    attn_output: Linear,
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
    ffn_gate: Linear,
    ffn_up: Linear,
    ffn_down: Linear,
}

pub struct EmbeddingGemmaTextModel {
    pub config: EmbeddingGemmaTextConfig,
    pub(super) pool: Arc<ComputePool>,
    pub(super) source: Arc<dyn TensorSource>,
    pub(super) token_embd: Vec<f32>,
    pub(super) text_embproj: Linear,
    pub(super) per_layer_model_proj: Linear,
    pub(super) per_layer_proj_norm: Vec<f32>,
    pub(super) output_norm: Vec<f32>,
    pub(super) layers: Vec<LayerWeights>,
}

impl EmbeddingGemmaTextModel {
    pub fn from_source(source: &Arc<dyn TensorSource>) -> Result<Self, String> {
        let pool = Arc::new(ComputePool::new(8));
        let config = EmbeddingGemmaTextConfig::from_metadata(source.as_ref())?;

        let get_f32 = |name: &str, expected: usize| -> Result<Vec<f32>, String> {
            let raw = source
                .tensor_slice(name)
                .ok_or_else(|| format!("Missing tensor: {name}"))?;
            Ok(decode_f32(raw, expected))
        };

        // Token embedding: dequantize token_embd to F32 (vocab × n_embd).
        // The GGUF stores it as `[n_embd, n_vocab]` (reversed); we want
        // `[vocab, n_embd]` for direct lookup.
        let token_info = source
            .tensor_info("token_embd.weight")
            .ok_or("Missing tensor: token_embd.weight")?;
        let token_raw = source
            .tensor_slice("token_embd.weight")
            .ok_or("Missing tensor: token_embd.weight")?;
        if token_info.dims.len() != 2
            || token_info.dims[0] as usize != config.n_embd
            || token_info.dims[1] as usize != config.n_vocab
        {
            return Err(format!(
                "token_embd.weight dims {:?} != [{}, {}]",
                token_info.dims, config.n_embd, config.n_vocab
            ));
        }
        let token_flat = dequant_to_f32(token_raw, token_info.ggml_type, config.n_embd, config.n_vocab)?;
        // Transpose: (n_embd, n_vocab) → (n_vocab, n_embd).
        let mut token_embd = vec![0.0f32; config.n_vocab * config.n_embd];
        for v in 0..config.n_vocab {
            for e in 0..config.n_embd {
                token_embd[v * config.n_embd + e] = token_flat[e * config.n_vocab + v];
            }
        }

        // text_embproj: [n_embd, EMBEDDING_DIM]. Stored as GGUF
        // [EMBEDDING_DIM, n_embd] (reversed). We want [n_embd, EMBEDDING_DIM]
        // for the matvec.
        let te_info = source
            .tensor_info("text_embproj.weight")
            .ok_or("Missing tensor: text_embproj.weight")?;
        if te_info.dims.len() != 2
            || te_info.dims[0] as usize != config.n_embd
            || te_info.dims[1] as usize != EMBEDDING_DIM
        {
            return Err(format!(
                "text_embproj.weight dims {:?} != [{}, {}]",
                te_info.dims, config.n_embd, EMBEDDING_DIM
            ));
        }
        let te_raw = source
            .tensor_slice("text_embproj.weight")
            .ok_or("Missing tensor: text_embproj.weight")?;
        let te_flat = dequant_to_f32(te_raw, te_info.ggml_type, EMBEDDING_DIM, config.n_embd)?;
        let mut text_embproj_matrix = vec![0.0f32; config.n_embd * EMBEDDING_DIM];
        for o in 0..EMBEDDING_DIM {
            for i in 0..config.n_embd {
                text_embproj_matrix[i * EMBEDDING_DIM + o] = te_flat[o * config.n_embd + i];
            }
        }
        let text_embproj = Linear {
            matrix: text_embproj_matrix,
            n_in: config.n_embd,
            n_out: EMBEDDING_DIM,
        };

        // per_layer_model_proj: GGUF stores [per_layer_w, n_embd]; we want
        // [n_embd, per_layer_w] for matvec.
        let ple_info = source
            .tensor_info("per_layer_model_proj.weight")
            .ok_or("Missing tensor: per_layer_model_proj.weight")?;
        if ple_info.dims.len() < 2 || ple_info.dims[0] as usize != config.n_embd {
            return Err(format!(
                "per_layer_model_proj.weight first dim {} != n_embd {}",
                ple_info.dims.first().copied().unwrap_or(0),
                config.n_embd
            ));
        }
        let ple_raw = source
            .tensor_slice("per_layer_model_proj.weight")
            .ok_or("Missing tensor: per_layer_model_proj.weight")?;
        // GGUF shape: [n_embd, per_layer_width * n_layer].
        let ple_n_out = config.per_layer_width * config.n_layer;
        let per_layer_model_proj_matrix = dequant_to_f32(
            ple_raw,
            ple_info.ggml_type,
            config.n_embd,
            ple_n_out,
        )?;
        let per_layer_model_proj = Linear {
            matrix: per_layer_model_proj_matrix,
            n_in: config.n_embd,
            n_out: config.per_layer_width * config.n_layer,
        };
        let per_layer_proj_norm = get_f32(
            "per_layer_proj_norm.weight",
            config.per_layer_width,
        )?;
        let output_norm = get_f32("output_norm.weight", config.n_embd)?;

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let head_dim = config.head_dim(i);
            let kv_heads = config.kv_heads(i);

            let attn_norm = get_f32(&format!("blk.{i}.attn_norm.weight"), config.n_embd)?;
            let post_attention_norm =
                get_f32(&format!("blk.{i}.post_attention_norm.weight"), config.n_embd)?;
            let pre_ffn_norm = get_f32(&format!("blk.{i}.pre_ffn_norm.weight"), config.n_embd)?;
            let post_ffn_norm = get_f32(&format!("blk.{i}.post_ffn_norm.weight"), config.n_embd)?;

            let ls_raw = source
                .tensor_slice(&format!("blk.{i}.layer_scalar"))
                .ok_or_else(|| format!("Missing tensor: blk.{i}.layer_scalar"))?;
            let layer_scalar = if ls_raw.len() == 4 {
                decode_f32(ls_raw, 1)[0]
            } else if ls_raw.len() == 2 {
                let bits = u16::from_le_bytes([ls_raw[0], ls_raw[1]]);
                f32::from_bits((bits as u32) << 16)
            } else {
                return Err(format!(
                    "blk.{i}.layer_scalar payload {} bytes (expected 2 BF16 or 4 F32)",
                    ls_raw.len()
                ));
            };

            // ple_input_gate: per-layer, [n_embd, n_embd] → input gate.
            let ple_input_gate = linear_from_gguf(
                source.as_ref(),
                &format!("blk.{i}.ple_input_gate.weight"),
                config.n_embd,
                config.n_embd,
            )?;
            // ple_projection: per-layer, [n_embd, n_embd] → final projection.
            let ple_projection = linear_from_gguf(
                source.as_ref(),
                &format!("blk.{i}.ple_projection.weight"),
                config.n_embd,
                config.n_embd,
            )?;
            // ple_post_norm: per-layer, [n_embd] (norm on the per-layer-input-gate output).
            let ple_post_norm = get_f32(
                &format!("blk.{i}.ple_post_norm.weight"),
                config.n_embd,
            )?;

            let attn_q = linear_from_gguf(
                source.as_ref(),
                &format!("blk.{i}.attn_q.weight"),
                config.n_embd,
                config.n_head * head_dim,
            )?;
            let attn_k = linear_from_gguf(
                source.as_ref(),
                &format!("blk.{i}.attn_k.weight"),
                config.n_embd,
                kv_heads * head_dim,
            )?;
            let attn_v = linear_from_gguf(
                source.as_ref(),
                &format!("blk.{i}.attn_v.weight"),
                config.n_embd,
                kv_heads * head_dim,
            )?;
            let attn_output = linear_from_gguf(
                source.as_ref(),
                &format!("blk.{i}.attn_output.weight"),
                config.n_head * head_dim,
                config.n_embd,
            )?;
            let attn_q_norm = get_f32(&format!("blk.{i}.attn_q_norm.weight"), head_dim)?;
            let attn_k_norm = get_f32(&format!("blk.{i}.attn_k_norm.weight"), head_dim)?;

            let ffn_gate = linear_from_gguf(
                source.as_ref(),
                &format!("blk.{i}.ffn_gate.weight"),
                config.n_embd,
                config.n_ff,
            )?;
            let ffn_up = linear_from_gguf(
                source.as_ref(),
                &format!("blk.{i}.ffn_up.weight"),
                config.n_embd,
                config.n_ff,
            )?;
            let ffn_down = linear_from_gguf(
                source.as_ref(),
                &format!("blk.{i}.ffn_down.weight"),
                config.n_ff,
                config.n_embd,
            )?;

            layers.push(LayerWeights {
                attn_norm,
                post_attention_norm,
                pre_ffn_norm,
                post_ffn_norm,
                layer_scalar,
                ple_input_gate,
                ple_projection,
                ple_post_norm,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_q_norm,
                attn_k_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        Ok(Self {
            config,
            pool,
            source: Arc::clone(source),
            token_embd,
            text_embproj,
            per_layer_model_proj,
            per_layer_proj_norm,
            output_norm,
            layers,
        })
    }

    /// Forward a sequence of token IDs, return the mean-pooled
    /// 768-dim embedding. Bidirectional: no causal mask.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() {
            return Err("empty input".to_string());
        }
        let n_tokens = token_ids.len();
        let cfg = &self.config;
        if n_tokens > cfg.n_ctx {
            return Err(format!(
                "input length {n_tokens} exceeds context {n_ctx}",
                n_ctx = cfg.n_ctx
            ));
        }
        let n_embd = cfg.n_embd;
        let per_layer_w = cfg.per_layer_width;

        // Step 1: token lookup → scaled by sqrt(n_embd) (per references/llama.cpp/src/models/gemma4.cpp:165).
        let scale = (n_embd as f32).sqrt();
        let mut x = vec![0.0f32; n_tokens * n_embd];
        for (i, &tid) in token_ids.iter().enumerate() {
            if (tid as usize) >= cfg.n_vocab {
                return Err(format!("token id {tid} out of range"));
            }
            let src = &self.token_embd[tid as usize * n_embd..(tid as usize + 1) * n_embd];
            for j in 0..n_embd {
                x[i * n_embd + j] = src[j] * scale;
            }
        }

        // Step 2: precompute per_layer_input[t, il, :] = rms_norm(per_layer_model_proj @ x[t] / sqrt(n_embd)).
        // The 1/sqrt(n_embd) scale is required by EmbeddingGemma2TextPLE.forward (see HF
        // `per_layer_model_projection_scale = config.hidden_size ** -0.5`).
        let n_layer = cfg.n_layer;
        let per_layer_proj_scale = 1.0f32 / (n_embd as f32).sqrt();
        let mut per_layer_input = vec![0.0f32; n_tokens * n_layer * per_layer_w];
        let mut tmp = vec![0.0f32; per_layer_w * n_layer];
        for t in 0..n_tokens {
            let src = &x[t * n_embd..(t + 1) * n_embd];
            self.per_layer_model_proj.forward(src, &mut tmp);
            for j in 0..tmp.len() {
                tmp[j] *= per_layer_proj_scale;
            }
            for il in 0..n_layer {
                let slot = &mut tmp[il * per_layer_w..(il + 1) * per_layer_w];
                crate::ops::norm::rms_norm_inplace(slot, &self.per_layer_proj_norm, EPS);
                per_layer_input[t * n_layer * per_layer_w + il * per_layer_w
                    ..t * n_layer * per_layer_w + (il + 1) * per_layer_w]
                    .copy_from_slice(slot);
            }
        }

        // Debug: print x stats before layers
        let x_mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        let x_max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Debug layer 0 in isolation
        let mut x_dbg = x.clone();
        self.forward_layer(0, &mut x_dbg, &per_layer_input)?;

        // Step 3: 24 bidirectional layers.
        for il in 0..cfg.n_layer {
            self.forward_layer(il, &mut x, &per_layer_input)?;
            let x_mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
            let x_max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        }

        // Step 4: final RMSNorm + text_embproj + mean pool.
        let mut x_normed = x.clone();
        crate::ops::norm::rms_norm_inplace(
            &mut x_normed[..n_tokens * n_embd],
            &self.output_norm,
            EPS,
        );

        let mut pooled = vec![0.0f32; EMBEDDING_DIM];
        let mut out = vec![0.0f32; EMBEDDING_DIM];
        for t in 0..n_tokens {
            let src = &x_normed[t * n_embd..(t + 1) * n_embd];
            self.text_embproj.forward(src, &mut out);
            for j in 0..EMBEDDING_DIM {
                pooled[j] += out[j];
            }
        }
        let inv_n = 1.0 / n_tokens as f32;
        for j in 0..EMBEDDING_DIM {
            pooled[j] *= inv_n;
        }
        Ok(pooled)
    }

    fn forward_layer(
        &self,
        il: usize,
        x: &mut [f32],
        per_layer_input: &[f32],
    ) -> Result<(), String> {
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let n_tokens = x.len() / n_embd;
        let layer = &self.layers[il];
        let head_dim = cfg.head_dim(il);
        let n_head = cfg.n_head;
        let n_kv = cfg.kv_heads(il);
        let full_attn = cfg.is_full_attention(il);
        let sliding_window = cfg.sliding_window();
        let n_ff = cfg.n_ff;

        // Step a: normed = rms_norm(attn_norm, x).
        let mut normed = x.to_vec();
        crate::ops::norm::rms_norm_inplace(
            &mut normed[..n_tokens * n_embd],
            &layer.attn_norm,
            EPS,
        );

        // Step c: Q/K/V projections on `normed`.
        let q_len = n_tokens * n_head * head_dim;
        let kv_len = n_tokens * n_kv * head_dim;
        let mut q = vec![0.0f32; q_len];
        let mut k = vec![0.0f32; kv_len];
        let mut v = vec![0.0f32; kv_len];
        let mut tmp_q = vec![0.0f32; n_head * head_dim];
        let mut tmp_kv = vec![0.0f32; n_kv * head_dim];
        for t in 0..n_tokens {
            let src = &normed[t * n_embd..(t + 1) * n_embd];
            // Debug: check attn_q matrix first row + size
            if t == 0 && il == 0 {
                let nonzero_count = layer.attn_q.matrix.iter().filter(|x| x.abs() > 1e-30).count();
            }
            layer.attn_q.forward(src, &mut tmp_q);
            layer.attn_k.forward(src, &mut tmp_kv);
            k[t * n_kv * head_dim..(t + 1) * n_kv * head_dim].copy_from_slice(&tmp_kv);
            layer.attn_v.forward(src, &mut tmp_kv);
            v[t * n_kv * head_dim..(t + 1) * n_kv * head_dim].copy_from_slice(&tmp_kv);
            q[t * n_head * head_dim..(t + 1) * n_head * head_dim].copy_from_slice(&tmp_q);
        }

        // Step d: per-head Q/K RMSNorm + RoPE.
        // Apply RMSNorm over head_dim for each head (before RoPE).
        let rope_freq = cfg.rope_freq_base(il);
        let mut cos_table = vec![0.0f32; n_tokens * head_dim / 2];
        let mut sin_table = vec![0.0f32; n_tokens * head_dim / 2];
        // Build the sin/cos table: for each position, compute half
        // dimension's worth of (cos, sin) pairs.
        for t in 0..n_tokens {
            for j in 0..head_dim / 2 {
                let freq = 1.0 / rope_freq.powf(2.0 * j as f32 / head_dim as f32);
                let angle = t as f32 * freq;
                cos_table[t * head_dim / 2 + j] = angle.cos();
                sin_table[t * head_dim / 2 + j] = angle.sin();
            }
        }
        // Apply RMSNorm + RoPE.
        for t in 0..n_tokens {
            for h in 0..n_head {
                let off = t * n_head * head_dim + h * head_dim;
                crate::ops::norm::rms_norm_inplace(
                    &mut q[off..off + head_dim],
                    &layer.attn_q_norm,
                    EPS,
                );
                apply_rope_neox(
                    &mut q[off..off + head_dim],
                    head_dim,
                    &cos_table[t * head_dim / 2..(t + 1) * head_dim / 2],
                    &sin_table[t * head_dim / 2..(t + 1) * head_dim / 2],
                );
            }
            for h in 0..n_kv {
                let off = t * n_kv * head_dim + h * head_dim;
                // Reference (gemma4.cpp:251): K uses attn_k_norm, V uses rms_unit (no weight).
                crate::ops::norm::rms_norm_inplace(
                    &mut k[off..off + head_dim],
                    &layer.attn_k_norm,
                    EPS,
                );
                crate::ops::norm::rms_unit_inplace(&mut v[off..off + head_dim], EPS);
                apply_rope_neox(
                    &mut k[off..off + head_dim],
                    head_dim,
                    &cos_table[t * head_dim / 2..(t + 1) * head_dim / 2],
                    &sin_table[t * head_dim / 2..(t + 1) * head_dim / 2],
                );
            }
        }

        // Step e: bidirectional attention (no causal mask, no sliding window — both ignored
        // in HF eager mode for this architecture, see eager_attention_forward in
        // .venv/lib/python3.12/site-packages/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py).
        let attn_dim = n_head * head_dim;
        let mut attn_out = vec![0.0f32; n_tokens * attn_dim];
        let gqa_ratio = n_head / n_kv;
        let scale = 1.0 / (head_dim as f32).sqrt();
        for i in 0..n_tokens {
            for h in 0..n_head {
                let kv_h = h / gqa_ratio;
                let q_off = i * n_head * head_dim + h * head_dim;
                let q_vec = &q[q_off..q_off + head_dim];
                // Scores for ALL key positions (bidirectional).
                let mut scores = vec![0.0f32; n_tokens];
                for j in 0..n_tokens {
                    let k_off = j * n_kv * head_dim + kv_h * head_dim;
                    let k_vec = &k[k_off..k_off + head_dim];
                    let mut s = 0.0f32;
                    for d in 0..head_dim {
                        s += q_vec[d] * k_vec[d];
                    }
                    scores[j] = s * scale;
                }
                // Softmax in fp32 (HF upcasts for stability).
                crate::ops::softmax::softmax_inplace(&mut scores);
                // Weighted V sum.
                let mut out_vec = vec![0.0f32; head_dim];
                for j in 0..n_tokens {
                    let v_off = j * n_kv * head_dim + kv_h * head_dim;
                    let v_vec = &v[v_off..v_off + head_dim];
                    let s = scores[j];
                    for d in 0..head_dim {
                        out_vec[d] += s * v_vec[d];
                    }
                }
                let dst_off = i * attn_dim + h * head_dim;
                attn_out[dst_off..dst_off + head_dim].copy_from_slice(&out_vec);
            }
        }
        let _ = (full_attn, sliding_window);  // suppress unused-variable warnings

        // Step f: output projection.
        let mut proj_out = vec![0.0f32; n_tokens * n_embd];
        let mut tmp_o = vec![0.0f32; n_embd];
        for t in 0..n_tokens {
            let src = &attn_out[t * attn_dim..(t + 1) * attn_dim];
            layer.attn_output.forward(src, &mut tmp_o);
            proj_out[t * n_embd..(t + 1) * n_embd].copy_from_slice(&tmp_o);
        }

        // Step g: post_attention_norm + residual.
        for t in 0..n_tokens {
            let mut tmp = proj_out[t * n_embd..(t + 1) * n_embd].to_vec();
            crate::ops::norm::rms_norm_inplace(&mut tmp, &layer.post_attention_norm, EPS);
            let proj_sum = tmp.iter().sum::<f32>();
            for j in 0..n_embd {
                x[t * n_embd + j] += tmp[j];
            }
        }
        let x_mean = x[..n_tokens * n_embd].iter().sum::<f32>() / x[..n_tokens * n_embd].len() as f32;

        // Step h: pre_ffn_norm.
        let mut ff_x = x.to_vec();
        crate::ops::norm::rms_norm_inplace(
            &mut ff_x[..n_tokens * n_embd],
            &layer.pre_ffn_norm,
            EPS,
        );

        // Step i: GeGLU FFN (per llama.cpp gemma4.cpp line 357: LLM_FFN_GELU, LLM_FFN_PAR).
        // gelu(gate) * up, then ffn_down.
        let mut ffn_out = vec![0.0f32; n_tokens * n_embd];
        let mut tmp_g = vec![0.0f32; n_ff];
        let mut tmp_u = vec![0.0f32; n_ff];
        let mut tmp_d = vec![0.0f32; n_embd];
        for t in 0..n_tokens {
            let src = &ff_x[t * n_embd..(t + 1) * n_embd];
            layer.ffn_gate.forward(src, &mut tmp_g);
            layer.ffn_up.forward(src, &mut tmp_u);
            for j in 0..n_ff {
                tmp_g[j] = crate::ops::activation::gelu(tmp_g[j]) * tmp_u[j];
            }
            layer.ffn_down.forward(&tmp_g, &mut tmp_d);
            ffn_out[t * n_embd..(t + 1) * n_embd].copy_from_slice(&tmp_d);
        }

        // Step j: post_ffn_norm + residual.
        for t in 0..n_tokens {
            let mut tmp = ffn_out[t * n_embd..(t + 1) * n_embd].to_vec();
            crate::ops::norm::rms_norm_inplace(&mut tmp, &layer.post_ffn_norm, EPS);
            for j in 0..n_embd {
                x[t * n_embd + j] += tmp[j];
            }
        }

        // Step k: PLE block (per references/llama.cpp/src/models/gemma4.cpp:367-385).
        //   pe_in = x
        //   cur = per_layer_inp_gate @ x  -> gelu -> * per_layer_input[il] -> per_layer_proj -> rms_norm(per_layer_post_norm)
        //   x = pe_in + cur
        let per_layer_w = cfg.per_layer_width;
        let n_layer = cfg.n_layer;
        let mut pe_in_buf = x.to_vec();
        let mut ple_gate_buf = vec![0.0f32; per_layer_w];
        let mut ple_gated = vec![0.0f32; per_layer_w];
        let mut ple_out = vec![0.0f32; n_embd];
        for t in 0..n_tokens {
            let src = &pe_in_buf[t * n_embd..(t + 1) * n_embd];
            layer.ple_input_gate.forward(src, &mut ple_gate_buf);
            for j in 0..per_layer_w {
                ple_gate_buf[j] = crate::ops::activation::gelu(ple_gate_buf[j]);
            }
            // Use the per-layer slice of per_layer_input corresponding to this layer il.
            let slice_off = t * n_layer * per_layer_w + il * per_layer_w;
            for j in 0..per_layer_w {
                ple_gated[j] = per_layer_input[slice_off + j] * ple_gate_buf[j];
            }
            layer.ple_projection.forward(&ple_gated, &mut ple_out);
            crate::ops::norm::rms_norm_inplace(&mut ple_out, &layer.ple_post_norm, EPS);
            for j in 0..n_embd {
                x[t * n_embd + j] = pe_in_buf[t * n_embd + j] + ple_out[j];
            }
        }

        // Step l: layer_scalar applied to full layer output.
        for v in x.iter_mut() {
            *v *= layer.layer_scalar;
        }

        Ok(())
    }
}

/// Apply NeoX-style RoPE in-place using a precomputed cos/sin table.
/// `x` has shape `[head_dim]` for a single head (per-token per-head).
fn apply_rope_neox(x: &mut [f32], head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    debug_assert_eq!(cos.len(), half);
    debug_assert_eq!(sin.len(), half);
    // NeoX: pair (x[2i], x[2i+1]) with (cos[i], sin[i]).
    // For position p, theta_i = p^(i/p), new x = [x*cos - x_rotated*sin, ...].
    // We split x into two halves: x_first (x[0..half]), x_second (x[half..head_dim]).
    for i in 0..half {
        let x_first = x[i];
        let x_second = x[i + half];
        let c = cos[i];
        let s = sin[i];
        x[i] = x_first * c - x_second * s;
        x[i + half] = x_second * c + x_first * s;
    }
}
