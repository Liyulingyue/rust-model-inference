//! Qwen3-architecture BitNet b1.58 decoder-only trunk.
//!
//! Used by `microsoft/bitnet-embedding-0.6b` (`general.architecture
//! = "qwen3"`, `general.file_type = 40`). Mirrors the layout of
//! the standard qwen3 trunk in [`crate::models::qwen3::trunk`] but
//! with **no** `is_bitnet` branches — every projection is the
//! BitLinear pattern. Architectural differences vs the standard
//! qwen3 trunk:
//!
//! 1. **No Q8_0 matmul**: every `wq/wk/wv/wo/w_gate/w_up/w_down`
//!    is replaced by `bitlinear_projection` (RMSNorm →
//!    `*_norm_in.weight` → absmax int8 quant → I2_S matmul).
//! 2. **No qkv bias**: BitLinear projections don't ship `q_bias/
//!    k_bias/v_bias`.
//! 3. **No MoE path**: BitNet b1.58 is dense-only.
//! 4. **No `is_bitnet` field on `Qwen3Config`**: the BitLinear
//!    forward is the trunk's *only* path; no run-time branch.
//! 5. **2-norm sandwich**: `attn_norm` + `ffn_norm` (Qwen3
//!    pre-norm, same as the standard trunk).
//!
//! # Why this is a separate trunk, not a sub-mode of qwen3
//!
//! Adding `if cfg.is_bitnet { bitlinear } else { matmul }` at
//! every matmul site in the standard qwen3 trunk would pollute the
//! forward loop with 7 branch points × 28 layers × N tokens of
//! branch-prediction overhead for **every** qwen3 forward, even
//! non-BitNet ones. Splitting into a dedicated BitNet qwen3
//! trunk restores the standard qwen3 forward to its pre-BitNet
//! clean state.
//!
//! # Token embedding storage
//!
//! The 0.6B GGUF ships the token embedding table as F16
//! `vocab × n_embd = 151936 × 1024`. To avoid the `unsafe`
//! lifetime-extension trick that the standard qwen3 embed path
//! uses, we expand to F32 in `Vec<f32>(vocab × n_embd)` (≈622 MB
//! on this box). Lookup is a `Vec::copy_from_slice` per token.
//!
//! # Pooling convention
//!
//! `qwen3.pooling_type = 1` with the BitNet flag means last-token
//! pooling (the BitNet Embedding convention, distinct from
//! Qwen3-Embedding's mean pooling). See [`compute_embedding`]
//! below.

use super::embedding::print_embedding_for_arch;
use crate::core::tensor::TensorSource;
use crate::ops::bitnet::{
    bitlinear_forward_packed, quantize_activation_per_token, BitLinearSlotPacked,
    BitLinearWeightsPacked,
};
use crate::ops::float::f16_to_f32;
use crate::ops::rope::rope_neox_inplace_with_factor;

/// Per-projection BitLinear, **packed-weight** variant.
///
/// RMSNorm → absmax int8 quant → ternary matmul → rescale, on
/// the SIMD hot path. Consumes [`BitLinearWeightsPacked`]
/// (pre-dequanted `{-1, 0, +1}` int8 weight matrix). Uses the
/// AVX2 `_mm256_madd_epi16` SIMD kernel via
/// [`bitlinear_forward_packed`], skipping the per-call I2_S dequant
/// walk entirely — see `examples/bitlinear_bench.rs`.
fn bitlinear_projection_packed(
    input: &[f32],
    proj: &BitLinearWeightsPacked,
    output: &mut [f32],
    eps: f32,
) {
    debug_assert_eq!(input.len(), proj.n_in);
    debug_assert_eq!(output.len(), proj.n_out);
    let n_in = proj.n_in;
    let mut normed = vec![0.0f32; n_in];
    crate::ops::norm::rms_norm(input, &proj.norm_in, &mut normed, eps);
    let (x_q, absmax) = quantize_activation_per_token(&normed);
    bitlinear_forward_packed(&proj.weight_i8, &x_q, absmax, n_in, proj.n_out, output);
}

/// Per-head RMSNorm applied to Q (and K) before RoPE.
///
/// Layout: Q/K stored row-major as `[n_tokens, n_heads, head_dim]`.
/// Norm is over each `head_dim`-wide slice independently.
fn apply_qk_norm(
    x: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    gain: &[f32],
    eps: f32,
) {
    assert_eq!(gain.len(), head_dim);
    for tok in 0..n_tokens {
        for head in 0..n_heads {
            let off = tok * n_heads * head_dim + head * head_dim;
            let row = &mut x[off..off + head_dim];
            let mut mean_sq = 0.0f32;
            for v in row.iter() {
                mean_sq += v * v;
            }
            mean_sq /= head_dim as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for (v, g) in row.iter_mut().zip(gain) {
                *v = *v * inv * *g;
            }
        }
    }
}

/// Per-layer weights for the BitNet qwen3 decoder block.
pub struct BitNetQwen3LayerWeights {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    /// Pre-dequanted int8 weights — the layout consumed by the
    /// SIMD hot path (`bitlinear_projection_packed`). Packed once
    /// at model load (no per-call I2_S dequant).
    pub bitlinear: BitLinearSlotPacked,
}

/// Loaded Qwen3-architecture BitNet model.
pub struct BitNetQwen3Model {
    pub config: BitNetQwen3Config,
    pub layers: Vec<BitNetQwen3LayerWeights>,
    pub output_norm: Vec<f32>,
    /// F32 expansion of `token_embd.weight`. Row `token_id`
    /// gives the embedding for that token.
    pub token_embedding_rows: Vec<f32>,
}

/// Configuration extracted from the GGUF metadata. Mirrors the
/// standard qwen3 config without `is_bitnet` — the BitLinear path
/// is the trunk's only mode.
pub struct BitNetQwen3Config {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub n_ff: usize,
    pub vocab: usize,
    pub eps: f32,
    pub freq_base: f32,
}

impl BitNetQwen3Config {
    pub fn n_embd_q(&self) -> usize {
        self.n_head * self.n_embd_head_k
    }
    pub fn n_embd_kv(&self) -> usize {
        self.n_head_kv * self.n_embd_head_k
    }
}

/// Public embedding extraction entry point. Tokenizes + runs the
/// decoder + returns the last-token row.
pub fn compute_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    let _ = n_threads_arg;
    let tokenizer =
        crate::core::tokenizer::BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|e| {
            format!("bitnet::qwen3_arch::compute_embedding: tokenizer init failed: {e}")
        })?;
    let prompt_tokens = encode_embedding_input(&tokenizer, prompt);
    if prompt_tokens.is_empty() {
        return Err("bitnet::qwen3_arch::compute_embedding: empty token sequence".into());
    }
    run_embedding_tokens(source, &prompt_tokens)
}

fn encode_embedding_input(
    tokenizer: &crate::core::tokenizer::BPETokenizer,
    prompt: &str,
) -> Vec<u32> {
    tokenizer
        .encode(
            prompt,
            crate::core::tokenizer::EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        )
        .into_iter()
        .filter(|&id| id != tokenizer.eos_id().unwrap_or(u32::MAX))
        .collect()
}

/// Run the BitNet qwen3 embedding forward given pre-tokenized input.
pub fn run_embedding_tokens(
    source: &dyn TensorSource,
    token_ids: &[u32],
) -> Result<Vec<f32>, String> {
    let model = load_model(source)?;
    text_encode(&model, token_ids)
}

/// Load the full BitNet qwen3 model.
pub fn load_model(source: &dyn TensorSource) -> Result<BitNetQwen3Model, String> {
    let config = build_config(source)?;
    let layers = load_layers(source, &config);
    let output_norm = get_f32_tensor(source, "output_norm.weight", config.n_embd);
    let token_embedding_rows = load_token_embedding_f32(source, config.vocab, config.n_embd)?;
    Ok(BitNetQwen3Model {
        config,
        layers,
        output_norm,
        token_embedding_rows,
    })
}

/// Expand the F16 `token_embd.weight` to an owned F32
/// `Vec<f32>(vocab × n_embd)`. See module docs §"Token embedding
/// storage" for why we don't use the borrowed `Weight<'a>`
/// pattern.
pub fn load_token_embedding_f32(
    source: &dyn TensorSource,
    vocab: usize,
    n_embd: usize,
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info("token_embd.weight")
        .ok_or_else(|| "bitnet::qwen3_arch: token_embd.weight missing".to_string())?;
    let bytes = source
        .tensor_slice("token_embd.weight")
        .ok_or_else(|| "bitnet::qwen3_arch: token_embd.weight slice missing".to_string())?;
    let expected_bytes = vocab * n_embd * 2;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "bitnet::qwen3_arch: token_embd.weight has {} bytes; \
             expected {} for {vocab} x {n_embd} F16",
            bytes.len(),
            expected_bytes
        ));
    }
    let mut out = vec![0.0f32; vocab * n_embd];
    match info.ggml_type {
        crate::core::tensor::GGMLType::F16 => {
            for (value, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                *value = f16_to_f32(bits);
            }
        }
        crate::core::tensor::GGMLType::F32 => {
            for (value, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *value = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        other => {
            return Err(format!(
                "bitnet::qwen3_arch: token_embd.weight ggml_type {other:?}; \
                 expected F16 or F32"
            ));
        }
    }
    Ok(out)
}

/// Load all `n_layer` BitNet qwen3 layer weights. Mirrors the
/// legacy `qwen3::trunk::weights::load_layers` interface but
/// loads BitLinear projections instead of `Weight` matmuls.
pub fn load_layers(
    source: &dyn TensorSource,
    config: &BitNetQwen3Config,
) -> Vec<BitNetQwen3LayerWeights> {
    (0..config.n_layer)
        .map(|l| {
            let bitlinear = load_bitlinear_layer(source, l, config);
            BitNetQwen3LayerWeights {
                attn_norm: get_f32_tensor(
                    source,
                    &format!("blk.{l}.attn_norm.weight"),
                    config.n_embd,
                ),
                ffn_norm: get_f32_tensor(
                    source,
                    &format!("blk.{l}.ffn_norm.weight"),
                    config.n_embd,
                ),
                q_norm: get_f32_tensor(
                    source,
                    &format!("blk.{l}.attn_q_norm.weight"),
                    config.n_embd_head_k,
                ),
                k_norm: get_f32_tensor(
                    source,
                    &format!("blk.{l}.attn_k_norm.weight"),
                    config.n_embd_head_k,
                ),
                bitlinear,
            }
        })
        .collect()
}

/// Load one BitNet layer's seven BitLinear slots, **pre-packed**
/// (the SIMD hot-path format). Used by [`load_layers`] — returns
/// the slots already dequanted to `{-1, 0, +1}` int8 so the
/// forward never walks the I2_S bytes.
fn load_bitlinear_layer(
    source: &dyn TensorSource,
    layer: usize,
    config: &BitNetQwen3Config,
) -> BitLinearSlotPacked {
    let n_embd = config.n_embd;
    let n_embd_q = config.n_embd_q();
    let n_embd_kv = config.n_embd_kv();
    let n_ff = config.n_ff;
    let slots: [(&str, usize, usize); 7] = [
        ("attn_q", n_embd, n_embd_q),
        ("attn_k", n_embd, n_embd_kv),
        ("attn_v", n_embd, n_embd_kv),
        ("attn_output", n_embd_q, n_embd),
        ("ffn_gate", n_embd, n_ff),
        ("ffn_up", n_embd, n_ff),
        ("ffn_down", n_ff, n_embd),
    ];
    let mut out = BitLinearSlotPacked::default();
    for (projection, n_in, n_out) in slots {
        let slot = load_bitlinear_slot(source, layer, projection, n_in, n_out);
        match projection {
            "attn_q" => out.attn_q = slot,
            "attn_k" => out.attn_k = slot,
            "attn_v" => out.attn_v = slot,
            "attn_output" => out.attn_output = slot,
            "ffn_gate" => out.ffn_gate = slot,
            "ffn_up" => out.ffn_up = slot,
            "ffn_down" => out.ffn_down = slot,
            _ => unreachable!(),
        }
    }
    out
}

fn load_bitlinear_slot(
    source: &dyn TensorSource,
    layer: usize,
    projection: &str,
    n_in: usize,
    n_out: usize,
) -> Option<BitLinearWeightsPacked> {
    let norm_name = format!("blk.{layer}.{projection}_norm_in.weight");
    let weight_name = format!("blk.{layer}.{projection}.weight");
    if source.tensor_info(&norm_name).is_none() || source.tensor_info(&weight_name).is_none() {
        return None;
    }
    let norm_in = get_f32_tensor(source, &norm_name, n_in);
    let info = source.tensor_info(&weight_name).unwrap();
    let bytes = source.tensor_slice(&weight_name).unwrap();
    assert_eq!(
        info.ggml_type,
        crate::core::tensor::GGMLType::I2_S,
        "BitLinear {weight_name} must be I2_S, got {:?}",
        info.ggml_type
    );
    let expected_bytes = (n_in * n_out) / 128 * 32;
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "BitLinear {weight_name} has {} bytes; expected {} for {n_in} x {n_out}",
        bytes.len(),
        expected_bytes
    );
    Some(
        crate::ops::bitnet::BitLinearWeights {
            norm_in,
            weight: bytes.to_vec(),
            n_in,
            n_out,
        }
        .prepack(),
    )
}

/// Build a `BitNetQwen3Config` from the GGUF metadata.
pub fn build_config(source: &dyn TensorSource) -> Result<BitNetQwen3Config, String> {
    let pick_u64 = |k: &str| {
        source
            .metadata(k)
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .ok_or_else(|| format!("bitnet::qwen3_arch: missing metadata {k}"))
    };
    let pick_f32 = |k: &str| {
        source
            .metadata(k)
            .and_then(|v| v.to_f64())
            .map(|v| v as f32)
            .ok_or_else(|| format!("bitnet::qwen3_arch: missing metadata {k}"))
    };
    Ok(BitNetQwen3Config {
        n_embd: pick_u64("qwen3.embedding_length")?,
        n_layer: pick_u64("qwen3.block_count")?,
        n_head: pick_u64("qwen3.attention.head_count")?,
        n_head_kv: pick_u64("qwen3.attention.head_count_kv")?,
        n_embd_head_k: pick_u64("qwen3.attention.key_length")?,
        n_embd_head_v: pick_u64("qwen3.attention.value_length")?,
        n_ff: pick_u64("qwen3.feed_forward_length")?,
        vocab: pick_u64("qwen3.vocab_size")?,
        eps: pick_f32("qwen3.attention.layer_norm_rms_epsilon")?,
        freq_base: pick_f32("qwen3.rope.freq_base")?,
    })
}

/// F16 / F32 / BF16 tensor loader. Mirrors
/// `qwen3::trunk::weights::get_f32_tensor` — the F16 arm is
/// required because BitNet ships all RMSNorm weights (`attn_norm`,
/// `ffn_norm`, `attn_q_norm`, `attn_k_norm`, and the seven
/// `*_norm_in`) as F16. Pre-F16 there was a `get_f32_tensor` that
/// silently zero-initialized F16 tensors, which collapsed every
/// BitNet RMSNorm to 0 and cascaded into all-zero BitLinear
/// output.
pub fn get_f32_tensor<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected_len: usize,
) -> Vec<f32> {
    let info = source
        .tensor_info(name)
        .unwrap_or_else(|| panic!("tensor {name} not found"));
    let bytes = source
        .tensor_slice(name)
        .unwrap_or_else(|| panic!("slice {name} not found"));
    let mut output = vec![0.0; expected_len];
    match info.ggml_type {
        crate::core::tensor::GGMLType::F32 => {
            for (value, chunk) in output.iter_mut().zip(bytes.chunks_exact(4)) {
                *value = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        crate::core::tensor::GGMLType::BF16 => {
            for (value, chunk) in output.iter_mut().zip(bytes.chunks_exact(2)) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                *value = crate::ops::float::bf16_to_f32(bits);
            }
        }
        crate::core::tensor::GGMLType::F16 => {
            for (value, chunk) in output.iter_mut().zip(bytes.chunks_exact(2)) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                *value = crate::ops::float::f16_to_f32(bits);
            }
        }
        other => panic!(
            "bitnet::qwen3_arch::get_f32_tensor {name}: unsupported ggml_type {other:?}; \
             expected F32/F16/BF16"
        ),
    }
    output
}

/// The BitNet qwen3 forward loop. Identical layout to the standard
/// qwen3 `text_encode` but with **no `if cfg.is_bitnet` branch** —
/// every projection is BitLinear. Architectural notes in the
/// module-level docs.
pub fn text_encode(model: &BitNetQwen3Model, token_ids: &[u32]) -> Result<Vec<f32>, String> {
    if token_ids.is_empty() {
        return Err("bitnet::qwen3_arch::text_encode: empty token sequence".into());
    }
    let n_tokens = token_ids.len();
    let cfg = &model.config;

    let mut hidden = vec![0.0f32; n_tokens * cfg.n_embd];
    for (row, &tid) in hidden.chunks_exact_mut(cfg.n_embd).zip(token_ids.iter()) {
        if (tid as usize) >= cfg.vocab {
            return Err(format!(
                "bitnet::qwen3_arch::text_encode: token id {tid} >= vocab {}",
                cfg.vocab
            ));
        }
        let src =
            &model.token_embedding_rows[tid as usize * cfg.n_embd..(tid as usize + 1) * cfg.n_embd];
        row.copy_from_slice(src);
    }

    let n_embd_q = cfg.n_embd_q();
    let n_embd_k = cfg.n_embd_kv();
    let n_embd_v = cfg.n_embd_kv();
    let n_attn = cfg.n_head * cfg.n_embd_head_v;
    let group_size = cfg.n_head / cfg.n_head_kv;
    let kq_scale = 1.0 / (cfg.n_embd_head_k as f32).sqrt();

    for layer_idx in 0..cfg.n_layer {
        let layer = &model.layers[layer_idx];

        let mut normed = vec![0.0f32; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            crate::ops::norm::rms_norm(
                &hidden[off..off + cfg.n_embd],
                &layer.attn_norm,
                &mut normed[off..off + cfg.n_embd],
                cfg.eps,
            );
        }

        let mut q_all = vec![0.0f32; n_tokens * n_embd_q];
        let mut k_all = vec![0.0f32; n_tokens * n_embd_k];
        let mut v_all = vec![0.0f32; n_tokens * n_embd_v];
        for tok in 0..n_tokens {
            let norm_row = &normed[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd];
            let q_off = tok * n_embd_q;
            let k_off = tok * n_embd_k;
            let v_off = tok * n_embd_v;
            bitlinear_projection_packed(
                norm_row,
                layer
                    .bitlinear
                    .attn_q
                    .as_ref()
                    .expect("bitnet::qwen3_arch: BitNet layer missing attn_q BitLinear slot"),
                &mut q_all[q_off..q_off + n_embd_q],
                cfg.eps,
            );
            bitlinear_projection_packed(
                norm_row,
                layer
                    .bitlinear
                    .attn_k
                    .as_ref()
                    .expect("bitnet::qwen3_arch: BitNet layer missing attn_k BitLinear slot"),
                &mut k_all[k_off..k_off + n_embd_k],
                cfg.eps,
            );
            bitlinear_projection_packed(
                norm_row,
                layer
                    .bitlinear
                    .attn_v
                    .as_ref()
                    .expect("bitnet::qwen3_arch: BitNet layer missing attn_v BitLinear slot"),
                &mut v_all[v_off..v_off + n_embd_v],
                cfg.eps,
            );
        }

        apply_qk_norm(
            &mut q_all,
            n_tokens,
            cfg.n_head,
            cfg.n_embd_head_k,
            &layer.q_norm,
            cfg.eps,
        );
        apply_qk_norm(
            &mut k_all,
            n_tokens,
            cfg.n_head_kv,
            cfg.n_embd_head_k,
            &layer.k_norm,
            cfg.eps,
        );

        for tok in 0..n_tokens {
            for head in 0..cfg.n_head {
                let off = tok * n_embd_q + head * cfg.n_embd_head_k;
                let q_slice = &mut q_all[off..off + cfg.n_embd_head_k];
                rope_neox_inplace_with_factor(q_slice, tok, cfg.n_embd_head_k, cfg.freq_base, 1.0_f32);
            }
            for head in 0..cfg.n_head_kv {
                let off = tok * n_embd_k + head * cfg.n_embd_head_k;
                let k_slice = &mut k_all[off..off + cfg.n_embd_head_k];
                rope_neox_inplace_with_factor(k_slice, tok, cfg.n_embd_head_k, cfg.freq_base, 1.0_f32);
            }
        }

        let mut attn_out = vec![0.0f32; n_tokens * n_attn];
        for head in 0..cfg.n_head {
            let kv_head = head / group_size;
            let q_off = head * cfg.n_embd_head_k;
            let k_off = kv_head * cfg.n_embd_head_k;
            let v_off = kv_head * cfg.n_embd_head_v;
            let attn_off = head * cfg.n_embd_head_v;

            for i in 0..n_tokens {
                let q_row = &q_all[i * n_embd_q + q_off..i * n_embd_q + q_off + cfg.n_embd_head_k];
                let mut max_val = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; n_tokens];
                for j in 0..=i {
                    let k_row =
                        &k_all[j * n_embd_k + k_off..j * n_embd_k + k_off + cfg.n_embd_head_k];
                    let mut dot = 0.0f32;
                    for d in 0..cfg.n_embd_head_k {
                        dot += q_row[d] * k_row[d];
                    }
                    let s = dot * kq_scale;
                    scores[j] = s;
                    if s > max_val {
                        max_val = s;
                    }
                }
                let mut exp_sum = 0.0f32;
                for j in 0..=i {
                    scores[j] = (scores[j] - max_val).exp();
                    exp_sum += scores[j];
                }
                for j in 0..=i {
                    scores[j] /= exp_sum;
                }
                for dim in 0..cfg.n_embd_head_v {
                    let mut sum = 0.0f32;
                    for j in 0..=i {
                        let v_row =
                            &v_all[j * n_embd_v + v_off..j * n_embd_v + v_off + cfg.n_embd_head_v];
                        sum += scores[j] * v_row[dim];
                    }
                    attn_out[i * n_attn + attn_off + dim] = sum;
                }
            }
        }

        let mut attn_proj_out = vec![0.0f32; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let attn_row = &attn_out[tok * n_attn..tok * n_attn + n_attn];
            bitlinear_projection_packed(
                attn_row,
                layer
                    .bitlinear
                    .attn_output
                    .as_ref()
                    .expect("bitnet::qwen3_arch: BitNet layer missing attn_output BitLinear slot"),
                &mut attn_proj_out[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd],
                cfg.eps,
            );
        }

        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            for j in 0..cfg.n_embd {
                hidden[off + j] += attn_proj_out[off + j];
            }
        }

        let mut ffn_normed = vec![0.0f32; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            crate::ops::norm::rms_norm(
                &hidden[off..off + cfg.n_embd],
                &layer.ffn_norm,
                &mut ffn_normed[off..off + cfg.n_embd],
                cfg.eps,
            );
        }

        let mut gate_buf = vec![0.0f32; n_tokens * cfg.n_ff];
        let mut up_buf = vec![0.0f32; n_tokens * cfg.n_ff];
        for tok in 0..n_tokens {
            let ffn_row = &ffn_normed[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd];
            bitlinear_projection_packed(
                ffn_row,
                layer
                    .bitlinear
                    .ffn_gate
                    .as_ref()
                    .expect("bitnet::qwen3_arch: BitNet layer missing ffn_gate BitLinear slot"),
                &mut gate_buf[tok * cfg.n_ff..tok * cfg.n_ff + cfg.n_ff],
                cfg.eps,
            );
            bitlinear_projection_packed(
                ffn_row,
                layer
                    .bitlinear
                    .ffn_up
                    .as_ref()
                    .expect("bitnet::qwen3_arch: BitNet layer missing ffn_up BitLinear slot"),
                &mut up_buf[tok * cfg.n_ff..tok * cfg.n_ff + cfg.n_ff],
                cfg.eps,
            );
            let off = tok * cfg.n_ff;
            for i in 0..cfg.n_ff {
                gate_buf[off + i] = crate::ops::silu(gate_buf[off + i]) * up_buf[off + i];
            }
        }

        let mut down_buf = vec![0.0f32; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let down_row = &gate_buf[tok * cfg.n_ff..tok * cfg.n_ff + cfg.n_ff];
            bitlinear_projection_packed(
                down_row,
                layer
                    .bitlinear
                    .ffn_down
                    .as_ref()
                    .expect("bitnet::qwen3_arch: BitNet layer missing ffn_down BitLinear slot"),
                &mut down_buf[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd],
                cfg.eps,
            );
        }

        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            for i in 0..cfg.n_embd {
                hidden[off + i] += down_buf[off + i];
            }
        }
    }

    let mut final_hidden = vec![0.0f32; n_tokens * cfg.n_embd];
    for tok in 0..n_tokens {
        let off = tok * cfg.n_embd;
        crate::ops::norm::rms_norm(
            &hidden[off..off + cfg.n_embd],
            &model.output_norm,
            &mut final_hidden[off..off + cfg.n_embd],
            cfg.eps,
        );
    }

    // Last-token pooling (BitNet convention for `pooling_type=1`).
    let last_off = (n_tokens - 1) * cfg.n_embd;
    Ok(final_hidden[last_off..last_off + cfg.n_embd].to_vec())
}

/// CLI entry point. Mirrors `crate::models::qwen3::embedding::run_embedding`.
pub fn run_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
    _kv_format: crate::core::scratchpad::KvFormat,
    output: crate::app::cli::EmbeddingOutput,
) {
    let started = std::time::Instant::now();
    match compute_embedding(source, prompt, n_threads_arg) {
        Ok(pooled) => {
            let elapsed = started.elapsed().as_millis();
            print_embedding_for_arch(&pooled, output, elapsed, "qwen3", cfg_n_layer(source));
        }
        Err(error) => {
            eprintln!("bitnet::qwen3_arch::run_embedding failed: {error}");
        }
    }
}

fn cfg_n_layer(source: &dyn TensorSource) -> usize {
    source
        .metadata("qwen3.block_count")
        .and_then(|v| v.to_u64())
        .map(|v| v as usize)
        .unwrap_or(0)
}
