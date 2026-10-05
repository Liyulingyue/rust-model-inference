//! Weight loading for the gemma3 trunk (BitNet b1.58 variant).
//!
//! # Tensor naming convention
//!
//! For a BitNet b1.58 GGUF with `file_type=40` and arch=`gemma3`,
//! each of the 18 layers carries 20 tensors:
//!
//! | name                              | ggml_type | shape          | role                          |
//! |-----------------------------------|-----------|----------------|-------------------------------|
//! | `blk.{i}.attn_norm.weight`        | F16       | n_embd         | pre-attention RMSNorm gain    |
//! | `blk.{i}.post_attention_norm.weight` | F16    | n_embd         | post-attention RMSNorm gain   |
//! | `blk.{i}.ffn_norm.weight`         | F16       | n_embd         | pre-FFN RMSNorm gain          |
//! | `blk.{i}.post_ffw_norm.weight`    | F16       | n_embd         | post-FFN RMSNorm gain         |
//! | `blk.{i}.attn_q_norm.weight`      | F16       | n_embd_head_k  | per-head QK-norm gain (Q)     |
//! | `blk.{i}.attn_k_norm.weight`      | F16       | n_embd_head_k  | per-head QK-norm gain (K)     |
//! | `blk.{i}.{proj}_norm_in.weight`   | F16       | n_in           | pre-BitLinear RMSNorm gain    |
//! | `blk.{i}.{proj}.weight`           | I2_S      | n_in × n_out   | ternary BitLinear weights     |
//!
//! Plus per-model `token_embd.weight` (F16, vocab × n_embd) and
//! `output_norm.weight` (F16, n_embd).
//!
//! The 4-norm sandwich and QK-norm make this trunk differ from
//! the qwen3 trunk; the BitLinear slots are identical in wire
//! format and dispatch to the shared [`crate::ops::bitnet`]
//! forward.

use crate::core::loader::GGUFLoader;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::ops::float::{bf16_to_f32, f16_to_f32};
use crate::ops::kernel::{F16Weight, QuantizedTensor};

pub use crate::ops::bitnet::{BitLinearSlotPacked, BitLinearWeights, BitLinearWeightsPacked};
pub use crate::ops::kernel::Weight;

use super::config::{Gemma3Config, Gemma3Rope};

/// Per-layer weights for one gemma3 decoder block. The `'a`
/// lifetime parameter is currently unused — kept for API symmetry
/// with [`crate::models::qwen3::trunk::Qwen3LayerWeights`], which
/// stores borrowed `Weight<'a>` projections.
pub struct Gemma3LayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub post_ffw_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    /// Pre-dequanted int8 weights — the layout consumed by the
    /// SIMD hot path (`bitlinear_projection_packed`). Packed once
    /// at model load (no per-call I2_S dequant).
    pub bitlinear: BitLinearSlotPacked,
    pub _marker: std::marker::PhantomData<&'a ()>,
}

/// Full gemma3 model: layers + embeddings + final norm.
///
/// `token_embedding_rows` is the F16 token-embedding table
/// expanded to F32 in `vocab × n_embd` row-major order. Storing
/// the expansion (instead of a borrowed `Weight<'a>`) avoids the
/// `'static`-lifetime gymnastics of
/// `qwen3::trunk::weights::load_static_weight` — the F32
/// expansion costs `vocab × n_embd × 4 bytes` (270M = 262144 ×
/// 640 × 4 ≈ 671 MB) but lets `Gemma3Model` be a fully owned
/// value with no borrowed `TensorSource` lifetime, which fits the
/// `app::run_embedding` interface where the source is borrowed
/// only for the duration of one call.
pub struct Gemma3Model {
    pub config: Gemma3Config,
    pub layers: Vec<Gemma3LayerWeights<'static>>,
    pub output_norm: Vec<f32>,
    /// F32 expansion of `token_embd.weight`. Row `token_id`
    /// gives the embedding for that token.
    pub token_embedding_rows: Vec<f32>,
}

/// Helper: load an F16 / F32 / BF16 tensor as `Vec<f32>`. Mirrors
/// `qwen3::trunk::weights::get_f32_tensor` — see that function's
/// doc-comment for the BitNet F16-arm history.
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
        GGMLType::F32 => {
            for (value, chunk) in output.iter_mut().zip(bytes.chunks_exact(4)) {
                *value = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        GGMLType::BF16 => {
            for (value, chunk) in output.iter_mut().zip(bytes.chunks_exact(2)) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                *value = bf16_to_f32(bits);
            }
        }
        GGMLType::F16 => {
            for (value, chunk) in output.iter_mut().zip(bytes.chunks_exact(2)) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                *value = f16_to_f32(bits);
            }
        }
        other => panic!(
            "gemma3::get_f32_tensor {name}: unsupported ggml_type {other:?}; \
             expected F32/F16/BF16"
        ),
    }
    output
}

/// Helper: load one BitLinear slot (RMSNorm gain + I2_S payload)
/// for a given layer + projection. Returns `None` when the
/// underlying tensors are missing (non-BitNet model).
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
        GGMLType::I2_S,
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
        BitLinearWeights {
            norm_in,
            weight: bytes.to_vec(),
            n_in,
            n_out,
        }
        .prepack(),
    )
}

/// Load all 18 layers of the 270M BitNet gemma3 GGUF (or any
/// future BitNet gemma3 GGUF). The seven projection shapes per
/// layer are fixed by the metadata and passed in.
pub fn load_layers_static(
    source: &dyn TensorSource,
    config: &Gemma3Config,
) -> Vec<Gemma3LayerWeights<'static>> {
    let n_embd = config.n_embd;
    let n_embd_head_k = config.n_embd_head_k;
    let n_embd_q = config.n_embd_q();
    let n_ff = config.n_ff;
    let projection_shapes: [(&str, usize, usize); 7] = [
        ("attn_q", n_embd, n_embd_q),
        ("attn_k", n_embd, config.n_embd_kv()),
        ("attn_v", n_embd, config.n_embd_kv()),
        ("attn_output", n_embd_q, n_embd),
        ("ffn_gate", n_embd, n_ff),
        ("ffn_up", n_embd, n_ff),
        ("ffn_down", n_ff, n_embd),
    ];
    (0..config.n_layer)
        .map(|i| {
            // gemma3_arch is BitNet-only — the BitLinear slot is
            // always populated. We do not gate on `is_bitnet` because
            // `Gemma3Config` no longer carries that field (the
            // BitNet path is the only path).
            let attn_q = load_bitlinear_slot(source, i, "attn_q", n_embd, n_embd_q);
            let attn_k = load_bitlinear_slot(source, i, "attn_k", n_embd, config.n_embd_kv());
            let attn_v = load_bitlinear_slot(source, i, "attn_v", n_embd, config.n_embd_kv());
            let attn_output = load_bitlinear_slot(source, i, "attn_output", n_embd_q, n_embd);
            let ffn_gate = load_bitlinear_slot(source, i, "ffn_gate", n_embd, n_ff);
            let ffn_up = load_bitlinear_slot(source, i, "ffn_up", n_embd, n_ff);
            let ffn_down = load_bitlinear_slot(source, i, "ffn_down", n_ff, n_embd);
            let bitlinear = BitLinearSlotPacked {
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                ffn_gate,
                ffn_up,
                ffn_down,
            };
            Gemma3LayerWeights {
                attn_norm: get_f32_tensor(source, &format!("blk.{i}.attn_norm.weight"), n_embd),
                post_attention_norm: get_f32_tensor(
                    source,
                    &format!("blk.{i}.post_attention_norm.weight"),
                    n_embd,
                ),
                ffn_norm: get_f32_tensor(source, &format!("blk.{i}.ffn_norm.weight"), n_embd),
                post_ffw_norm: get_f32_tensor(
                    source,
                    &format!("blk.{i}.post_ffw_norm.weight"),
                    n_embd,
                ),
                q_norm: get_f32_tensor(
                    source,
                    &format!("blk.{i}.attn_q_norm.weight"),
                    n_embd_head_k,
                ),
                k_norm: get_f32_tensor(
                    source,
                    &format!("blk.{i}.attn_k_norm.weight"),
                    n_embd_head_k,
                ),
                bitlinear,
                _marker: std::marker::PhantomData,
            }
        })
        .collect()
}

/// Load the F16 token-embedding table and expand it to F32 in
/// row-major `vocab × n_embd` order. The caller gets a single
/// owned `Vec<f32>` slice that doesn't borrow from the
/// `TensorSource`, which lets [`crate::models::bitnet::gemma3_arch::embedding`]
/// keep the model fully owned and survive the end of the
/// `compute_embedding` call's borrow scope.
///
/// Cost: `vocab × n_embd × 4 bytes` of heap. For BitNet-270M
/// (vocab=262144, n_embd=640) this is ~671 MB; for future
/// BitNet gemma3 variants it scales linearly. If memory becomes
/// a constraint, swap this for a borrowed `Weight<'a>` + the
/// `'static` lifetime trick from `qwen3::trunk::weights`.
pub fn static_weight(source: &dyn TensorSource, name: &str) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("gemma3: tensor {name} not found"))?;
    let slice = source
        .tensor_slice(name)
        .ok_or_else(|| format!("gemma3: tensor {name} slice not found"))?;
    let n_in = info.dims[0] as usize;
    let n_out = if info.dims.len() >= 2 {
        info.dims[1] as usize
    } else {
        1
    };
    let expected_bytes = n_in * n_out * 2; // F16 = 2 bytes
    if slice.len() != expected_bytes {
        return Err(format!(
            "gemma3: {name} has {} bytes; expected {} for {n_in} x {n_out} F16",
            slice.len(),
            expected_bytes
        ));
    }
    let mut out = vec![0.0f32; n_in * n_out];
    for (value, chunk) in out.iter_mut().zip(slice.chunks_exact(2)) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        *value = f16_to_f32(bits);
    }
    Ok(out)
}

/// Public alias used by [`crate::models::bitnet::gemma3_arch::embedding`]. Same
/// shape as the legacy qwen3 loader, kept distinct so future
/// gemma3-only tensor-naming tweaks (e.g. dropping `output_norm`)
/// can be added without touching the qwen3 path.
#[allow(dead_code)]
pub fn load_layers<'a>(
    source: &'a dyn TensorSource,
    config: &Gemma3Config,
) -> Vec<Gemma3LayerWeights<'a>> {
    load_layers_static(source, config)
}
