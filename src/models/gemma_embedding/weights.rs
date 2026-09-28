//! Weight loading for `arch = "gemma-embedding"`.
//!
//! Tensor names follow the llama.cpp `gemma-embedding` model
//! (`references/llama.cpp/src/models/gemma-embedding.cpp:33-66`), which in GGUF
//! land is the standard `blk.{l}.attn_*` / `blk.{l}.ffn_*` vocabulary. Only this
//! module knows the names; `compute.rs` never hard-codes them.

use crate::core::tensor::{GGMLType, TensorSource};
use crate::ops::kernel::{QuantizedTensor, Weight};

pub struct GemmaEmbeddingLayerWeights<'a> {
    pub wq: Weight<'a>,
    pub wk: Weight<'a>,
    pub wv: Weight<'a>,
    /// `blk.{l}.attn_output.weight` — attention output projection.
    pub wo: Weight<'a>,
    /// `blk.{l}.attn_norm.weight` — pre-attention RMSNorm.
    pub attn_norm: Vec<f32>,
    /// `blk.{l}.post_attention_norm.weight` — post-attention RMSNorm.
    pub attn_post_norm: Vec<f32>,
    /// `blk.{l}.attn_q_norm.weight` — per-head Q RMSNorm (over head_dim).
    pub q_norm: Vec<f32>,
    /// `blk.{l}.attn_k_norm.weight` — per-head K RMSNorm (over head_dim).
    pub k_norm: Vec<f32>,
    /// `blk.{l}.ffn_norm.weight` — pre-FFN RMSNorm.
    pub ffn_norm: Vec<f32>,
    /// `blk.{l}.ffn_gate.weight` — geglu gate half.
    pub w_gate: Weight<'a>,
    /// `blk.{l}.ffn_up.weight` — geglu up half (multiplied, not gated).
    pub w_up: Weight<'a>,
    /// `blk.{l}.ffn_down.weight` — FFN down projection.
    pub w_down: Weight<'a>,
    /// `blk.{l}.post_ffw_norm.weight` — post-FFN RMSNorm.
    pub ffn_post_norm: Vec<f32>,
}

pub struct GemmaEmbeddingWeights<'a> {
    pub n_layer: usize,
    pub token_embd: &'a [u8],
    pub token_embd_ggml_type: GGMLType,
    /// `output_norm.weight` — final RMSNorm before pooling.
    pub output_norm: Vec<f32>,
    /// `dense_2.weight` [n_embd, dense_2_feat_out].
    pub dense_2: Weight<'a>,
    /// `dense_3.weight` [dense_3_feat_in, n_embd].
    pub dense_3: Weight<'a>,
    pub layers: Vec<GemmaEmbeddingLayerWeights<'a>>,
}

fn decode_f32_row(ggml_type: GGMLType, bytes: &[u8], expected_len: usize) -> Option<Vec<f32>> {
    match ggml_type {
        GGMLType::F32 => {
            if bytes.len() / 4 < expected_len {
                return None;
            }
            let mut out = vec![0.0f32; expected_len];
            for (index, slot) in out.iter_mut().enumerate() {
                *slot = f32::from_bits(u32::from_le_bytes([
                    bytes[index * 4],
                    bytes[index * 4 + 1],
                    bytes[index * 4 + 2],
                    bytes[index * 4 + 3],
                ]));
            }
            Some(out)
        }
        GGMLType::F16 => {
            if bytes.len() / 2 < expected_len {
                return None;
            }
            let mut out = vec![0.0f32; expected_len];
            for (index, slot) in out.iter_mut().enumerate() {
                *slot = crate::ops::f16_to_f32(u16::from_le_bytes([
                    bytes[index * 2],
                    bytes[index * 2 + 1],
                ]));
            }
            Some(out)
        }
        _ => None,
    }
}

fn get_f32_tensor<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected_len: usize,
) -> Vec<f32> {
    let info = source
        .tensor_info(name)
        .unwrap_or_else(|| panic!("missing tensor {name}"));
    let bytes = source
        .tensor_slice(name)
        .unwrap_or_else(|| panic!("missing tensor data for {name}"));
    decode_f32_row(info.ggml_type, bytes, expected_len)
        .unwrap_or_else(|| panic!("tensor {name} is not a decodable f32 row"))
}

fn load_weight<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Weight<'a> {
    let info = source
        .tensor_info(name)
        .unwrap_or_else(|| panic!("missing tensor {name}"));
    let bytes = source
        .tensor_slice(name)
        .unwrap_or_else(|| panic!("missing tensor data for {name}"));
    Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        n_in,
        n_out,
    ))
}

/// Load a projection whose `[n_in, n_out]` both come from the GGUF dims, so
/// arch metadata and tensor inventory are cross-checked on every load.
fn load_sized_weight<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    expect_in: Option<usize>,
    expect_out: Option<usize>,
) -> Weight<'a> {
    let info = source
        .tensor_info(name)
        .unwrap_or_else(|| panic!("missing tensor {name}"));
    let (n_in, n_out) = (info.dims[0] as usize, info.dims[1] as usize);
    if let Some(expected) = expect_in {
        assert_eq!(n_in, expected, "{name}: n_in {n_in} != expected {expected}");
    }
    if let Some(expected) = expect_out {
        assert_eq!(
            n_out, expected,
            "{name}: n_out {n_out} != expected {expected}"
        );
    }
    load_weight(source, name, n_in, n_out)
}

#[allow(clippy::too_many_arguments)]
pub fn load_weights<S: TensorSource + ?Sized>(
    source: &S,
    n_layer: usize,
    n_embd: usize,
    n_embd_q: usize,
    n_embd_gqa: usize,
    n_embd_head: usize,
    n_ff: usize,
) -> GemmaEmbeddingWeights<'_> {
    let embd_info = source
        .tensor_info("token_embd.weight")
        .expect("missing token_embd.weight");
    let token_embd_ggml_type = embd_info.ggml_type;
    let token_embd = source
        .tensor_slice("token_embd.weight")
        .expect("missing token_embd.weight data");
    debug_assert_eq!(embd_info.dims[0] as usize, n_embd);

    let output_norm = get_f32_tensor(source, "output_norm.weight", n_embd);

    let dense_2 = load_sized_weight(source, "dense_2.weight", Some(n_embd), None);
    let dense_3 = load_sized_weight(source, "dense_3.weight", None, Some(n_embd));

    let layers = (0..n_layer)
        .map(|l| {
            let name_of = |suffix: &str| format!("blk.{l}.{suffix}");
            GemmaEmbeddingLayerWeights {
                wq: load_sized_weight(
                    source,
                    &name_of("attn_q.weight"),
                    Some(n_embd),
                    Some(n_embd_q),
                ),
                wk: load_sized_weight(
                    source,
                    &name_of("attn_k.weight"),
                    Some(n_embd),
                    Some(n_embd_gqa),
                ),
                wv: load_sized_weight(
                    source,
                    &name_of("attn_v.weight"),
                    Some(n_embd),
                    Some(n_embd_gqa),
                ),
                wo: load_sized_weight(
                    source,
                    &name_of("attn_output.weight"),
                    Some(n_embd_q),
                    Some(n_embd),
                ),
                attn_norm: get_f32_tensor(source, &name_of("attn_norm.weight"), n_embd),
                attn_post_norm: get_f32_tensor(
                    source,
                    &name_of("post_attention_norm.weight"),
                    n_embd,
                ),
                q_norm: get_f32_tensor(source, &name_of("attn_q_norm.weight"), n_embd_head),
                k_norm: get_f32_tensor(source, &name_of("attn_k_norm.weight"), n_embd_head),
                ffn_norm: get_f32_tensor(source, &name_of("ffn_norm.weight"), n_embd),
                w_gate: load_sized_weight(
                    source,
                    &name_of("ffn_gate.weight"),
                    Some(n_embd),
                    Some(n_ff),
                ),
                w_up: load_sized_weight(
                    source,
                    &name_of("ffn_up.weight"),
                    Some(n_embd),
                    Some(n_ff),
                ),
                w_down: load_sized_weight(
                    source,
                    &name_of("ffn_down.weight"),
                    Some(n_ff),
                    Some(n_embd),
                ),
                ffn_post_norm: get_f32_tensor(source, &name_of("post_ffw_norm.weight"), n_embd),
            }
        })
        .collect();

    GemmaEmbeddingWeights {
        n_layer,
        token_embd,
        token_embd_ggml_type,
        output_norm,
        dense_2,
        dense_3,
        layers,
    }
}
