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
//! format and dispatch to the shared [`crate::models::bitnet`]
//! forward.

use crate::core::loader::GGUFLoader;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::ops::float::{bf16_to_f32, f16_to_f32};
use crate::ops::kernel::{F16Weight, QuantizedTensor};

pub use crate::models::bitnet::{BitLinearSlot, BitLinearWeights};
pub use crate::ops::kernel::Weight;

use super::config::{Gemma3Config, Gemma3Rope};

/// Per-layer weights for one gemma3 decoder block. The `'a`
/// lifetime parameter is currently unused — kept for API symmetry
/// with [`crate::models::qwen3::trunk::Qwen3LayerWeights`], which
/// stores borrowed `Weight<'a>` projections.
///
/// Holds **both** the BitNet (`bitlinear`) and standard
/// (`std_proj` byte buffers) projection sets. The loader populates
/// only the set relevant to the model's forward path:
/// - `cfg.is_bitnet == true`  → `bitlinear` populated, `std_proj` empty
/// - `cfg.is_bitnet == false` → `bitlinear` default, `std_proj` populated
///
/// The unused set is small (a few hundred bytes for the empty
/// `Vec`/`Option` fields) so we keep both rather than fork the
/// layer struct per path.
///
/// `StdProjection` stores **owned bytes** (not `Weight<'a>`) so
/// the layer can outlive the borrowed `&dyn TensorSource`: the
/// forward pass rebuilds the `Weight` from `(bytes, ggml_type,
/// n_in, n_out)` per call (cheap; the kernel is just a `&[u8]` view
/// of the byte buffer + dimensions). Storing the Weight directly
/// would force a `'static` byte buffer owned by the model, which
/// the gemma3 trunk's per-call rebuild pattern can't supply
/// without duplicating the entire GGUF payload.
pub struct Gemma3LayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub post_ffw_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    /// Pre-BitLinear RMSNorm gains + I2_S payloads. Always
    /// `BitLinearSlot::default()` (all slots `None`) for non-BitNet
    /// gemma3 GGUFs; for the BitNet 270M GGUF every slot is
    /// `Some(BitLinearWeights)`.
    pub bitlinear: BitLinearSlot,
    /// Standard (Q4_K / Q5_0 / Q6K / Q8_0) matmul weight payloads,
    /// populated for standard gemma3, empty for BitNet 270M.
    /// `(bytes, ggml_type, n_in, n_out)` per projection.
    pub wq: Option<StdProjection>,
    pub wk: Option<StdProjection>,
    pub wv: Option<StdProjection>,
    pub wo: Option<StdProjection>,
    pub w_gate: Option<StdProjection>,
    pub w_up: Option<StdProjection>,
    pub w_down: Option<StdProjection>,
    pub _marker: std::marker::PhantomData<&'a ()>,
}

/// Owned byte buffer + ggml type + shape for one standard
/// matmul projection. The forward pass rebuilds the `Weight`
/// kernel from this each call.
#[derive(Debug, Clone)]
pub struct StdProjection {
    pub bytes: Vec<u8>,
    pub ggml_type: GGMLType,
    pub n_in: usize,
    pub n_out: usize,
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
) -> Option<BitLinearWeights> {
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
    Some(BitLinearWeights {
        norm_in,
        weight: bytes.to_vec(),
        n_in,
        n_out,
    })
}

/// Load all 18 layers of the 270M BitNet gemma3 GGUF (or any
/// future BitNet gemma3 GGUF). The seven projection shapes per
/// layer are fixed by the metadata and passed in.
///
/// For BitNet (`cfg.is_bitnet == true`) the `bitlinear` slot is
/// populated and `w_*` stays empty. For standard gemma3 (`cfg.is_bitnet == false`)
/// the `w_*` fields are populated via
/// [`Weight::from_quantized(QuantizedTensor::from_bytes(...))`] and
/// `bitlinear` stays default.
pub fn load_layers_static(
    source: &dyn TensorSource,
    config: &Gemma3Config,
) -> Vec<Gemma3LayerWeights<'static>> {
    let n_embd = config.n_embd;
    let n_embd_head_k = config.n_embd_head_k;
    let n_embd_q = config.n_embd_q();
    let n_embd_kv = config.n_embd_kv();
    let n_ff = config.n_ff;
    (0..config.n_layer)
        .map(|i| {
            let bitlinear = if config.is_bitnet {
                let attn_q = load_bitlinear_slot(source, i, "attn_q", n_embd, n_embd_q);
                let attn_k = load_bitlinear_slot(source, i, "attn_k", n_embd, n_embd_kv);
                let attn_v = load_bitlinear_slot(source, i, "attn_v", n_embd, n_embd_kv);
                let attn_output = load_bitlinear_slot(source, i, "attn_output", n_embd_q, n_embd);
                let ffn_gate = load_bitlinear_slot(source, i, "ffn_gate", n_embd, n_ff);
                let ffn_up = load_bitlinear_slot(source, i, "ffn_up", n_embd, n_ff);
                let ffn_down = load_bitlinear_slot(source, i, "ffn_down", n_ff, n_embd);
                BitLinearSlot {
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                }
            } else {
                BitLinearSlot::default()
            };
            // For the standard path, load each matmul tensor as a
            // generic quantized Weight. We use Option so missing
            // tensors (e.g. for partial loads / subset of layers)
            // surface as a clear "tensor not found" error instead of
            // a panic.
            let std_proj = |projection: &str, n_in: usize, n_out: usize| -> Option<StdProjection> {
                let name = format!("blk.{i}.{projection}.weight");
                let info = source.tensor_info(&name)?;
                let bytes = source.tensor_slice(&name)?;
                Some(StdProjection {
                    bytes: bytes.to_vec(),
                    ggml_type: info.ggml_type,
                    n_in,
                    n_out,
                })
            };
            let (wq, wk, wv, wo, w_gate, w_up, w_down) = if config.is_bitnet {
                (None, None, None, None, None, None, None)
            } else {
                (
                    std_proj("attn_q", n_embd, n_embd_q),
                    std_proj("attn_k", n_embd, n_embd_kv),
                    std_proj("attn_v", n_embd, n_embd_kv),
                    std_proj("attn_output", n_embd_q, n_embd),
                    std_proj("ffn_gate", n_embd, n_ff),
                    std_proj("ffn_up", n_embd, n_ff),
                    std_proj("ffn_down", n_ff, n_embd),
                )
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
                wq,
                wk,
                wv,
                wo,
                w_gate,
                w_up,
                w_down,
                _marker: std::marker::PhantomData,
            }
        })
        .collect()
}

/// Load the token-embedding table and expand it to F32 in
/// row-major `vocab × n_embd` order. Supports F16 (BitNet 270M)
/// and Q8_0 (standard gemma3-270m-it and friends). The caller gets
/// a single owned `Vec<f32>` slice that doesn't borrow from the
/// `TensorSource`, which lets [`crate::models::gemma3::embedding`]
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
    let n_total = n_in * n_out;
    let mut out = vec![0.0f32; n_total];
    match info.ggml_type {
        GGMLType::F32 => {
            // 1B-it (gemma-3-1b-it-Q4_K_M) packs its norms in
            // native F32 instead of F16; the standard gemma3
            // forward does an F32→F32 byte-reinterpret here.
            let expected_bytes = n_total * 4;
            if slice.len() != expected_bytes {
                return Err(format!(
                    "gemma3: {name} has {} bytes; expected {} for {n_in} x {n_out} F32",
                    slice.len(),
                    expected_bytes
                ));
            }
            for (value, chunk) in out.iter_mut().zip(slice.chunks_exact(4)) {
                *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        GGMLType::F16 => {
            let expected_bytes = n_total * 2;
            if slice.len() != expected_bytes {
                return Err(format!(
                    "gemma3: {name} has {} bytes; expected {} for {n_in} x {n_out} F16",
                    slice.len(),
                    expected_bytes
                ));
            }
            for (value, chunk) in out.iter_mut().zip(slice.chunks_exact(2)) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                *value = f16_to_f32(bits);
            }
        }
        GGMLType::Q8_0 => {
            // Q8_0 block layout: 34 bytes per 32 elements (2-byte F16
            // scale + 32 signed int8 values). For the standard
            // gemma3-270m-it GGUF the token_embd.weight is Q8_0
            // quantized (mixed-quant layout).
            let blocks_per_row = n_in / 32;
            let row_bytes = blocks_per_row * 34;
            if slice.len() < n_out * row_bytes {
                return Err(format!(
                    "gemma3: {name} has {} bytes; expected {} for {n_in} x {n_out} Q8_0",
                    slice.len(),
                    n_out * row_bytes
                ));
            }
            for row in 0..n_out {
                let row_off = row * row_bytes;
                for block in 0..blocks_per_row {
                    let off = row_off + block * 34;
                    let scale = f16_to_f32(u16::from_le_bytes([slice[off], slice[off + 1]]));
                    for lane in 0..32 {
                        let q = slice[off + 2 + lane] as i8 as f32;
                        out[row * n_in + block * 32 + lane] = scale * q;
                    }
                }
            }
        }
        GGMLType::Q4K | GGMLType::Q5K | GGMLType::Q6K => {
            // K-quants use 256-element super-blocks:
            //   Q4_K = 144 bytes/block, Q5_K = 176 bytes/block,
            //   Q6_K = 210 bytes/block.
            // All three delegate to the existing
            // `dequantize_row_q{4,5,6}_k` helpers in `ops::quant`.
            let block_bytes = match info.ggml_type {
                GGMLType::Q4K => 144,
                GGMLType::Q5K => 176,
                GGMLType::Q6K => 210,
                _ => unreachable!(),
            };
            let n_blocks_per_row = n_in / 256;
            let row_bytes = n_blocks_per_row * block_bytes;
            if slice.len() < n_out * row_bytes {
                return Err(format!(
                    "gemma3: {name} has {} bytes; expected {} for {n_in} x {n_out} {:?}",
                    slice.len(),
                    n_out * row_bytes,
                    info.ggml_type
                ));
            }
            let mut row_buf = vec![0.0f32; n_in];
            for row in 0..n_out {
                let row_off = row * row_bytes;
                let row_bytes_slice = &slice[row_off..row_off + row_bytes];
                let out_row = &mut out[row * n_in..(row + 1) * n_in];
                match info.ggml_type {
                    GGMLType::Q4K => {
                        crate::ops::quant::dequantize_row_q4_k(row_bytes_slice, out_row);
                    }
                    GGMLType::Q5K => {
                        crate::ops::quant::dequantize_row_q5_k(row_bytes_slice, out_row);
                    }
                    GGMLType::Q6K => {
                        crate::ops::quant::dequantize_row_q6_k(row_bytes_slice, out_row);
                    }
                    _ => unreachable!(),
                }
                let _ = row_buf; // keep the compiler happy if unused
            }
        }
        other => {
            return Err(format!(
                "gemma3: {name} has unsupported ggml_type {other:?}; expected F16 or Q8_0"
            ));
        }
    }
    Ok(out)
}

/// Public alias used by [`crate::models::gemma3::embedding`]. Same
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
