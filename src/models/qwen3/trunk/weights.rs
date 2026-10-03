//! # Qwen3 Weights — `Qwen3LayerWeights` + load helpers + `Qwen3Model` struct
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2.1:
//! - `Qwen3Model` struct lives here because its fields are weight tables
//!   (`token_embedding`, `output`, `layers`, `output_norm`).
//! - `Qwen3Model::from_source` + stateless accessors live here.
//! - The `text_encode` *method* lives in `forward.rs` (forward-loop concern);
//!   the `text_encode` *free function* lives in `forward.rs`.
//! - `Qwen3Input` / `Qwen3GenerateOptions` / `Qwen3Generation` are in `forward.rs`.
//! - `Qwen3Session` struct lives in `session.rs`.

use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::ops::bf16_to_f32;
use crate::ops::kernel::{QuantizedTensor, Weight};
use std::sync::Arc;

pub use super::config::Qwen3Config;

// =============================================================================
// Qwen3Model struct (weight tables)
// =============================================================================

pub struct Qwen3Model {
    pub(crate) source: Arc<dyn TensorSource>,
    pub(crate) tokenizer: Arc<BPETokenizer>,
    pub(crate) pool: Arc<ComputePool>,
    pub(crate) config: Qwen3Config,
    pub(crate) layers: Vec<Qwen3LayerWeights<'static>>,
    pub(crate) output_norm: Vec<f32>,
    pub(crate) token_embedding: Weight<'static>,
    pub(crate) output: Weight<'static>,
    /// Optional classification / rerank head. Loaded when the GGUF carries a
    /// `cls.output.weight` tensor (as produced by llama.cpp's rerank packer
    /// — e.g. `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF`). When `Some`,
    /// the model is a rerank / cross-encoder; the caller is expected to
    /// build a `[query, document]` prompt and read `score_logits(last_hidden)`
    /// rather than sampling tokens from the lm_head.
    pub(crate) cls_score: Option<Weight<'static>>,
}

// =============================================================================
// Qwen3LayerWeights (per-layer weight stack)
// =============================================================================

pub struct Qwen3LayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    pub moe_router: Option<Vec<f32>>,
    pub moe_gate: Option<Vec<Weight<'a>>>,
    pub moe_up: Option<Vec<Weight<'a>>>,
    pub moe_down: Option<Vec<Weight<'a>>>,
    pub wq: Weight<'a>,
    pub wk: Weight<'a>,
    pub wv: Weight<'a>,
    pub wo: Weight<'a>,
    pub w_gate: Weight<'a>,
    pub w_up: Weight<'a>,
    pub w_down: Weight<'a>,
    /// BitNet b1.58 BitLinear projection slots. All seven slots
    /// remain `None` for non-BitNet models. Loaded from
    /// `blk.{i}.{proj}_norm_in.weight` (F16) + `blk.{i}.{proj}.weight`
    /// (I2_S) when the GGUF carries the BitNet markers (see
    /// `Qwen3Config::is_bitnet`).
    pub bitlinear: BitLinearSlot,
}

/// Microsoft BitNet b1.58 BitLinear projection pair: per-projection
/// RMSNorm (`*_norm_in.weight`, F16) + I2_S ternary weights packed
/// 2-bit per element. The forward in `forward.rs::bitlinear_projection`
/// does: `rms_norm → quantize_per_token_abs8 → ternary matmul → rescale`.
///
/// The `weight` byte slice is the raw GGUF payload (no header, no
/// scale — see `src/ops/kernel/i2_s.rs` for the block layout and the
/// `bitnet.cpp` reference).
pub use crate::ops::bitlinear::{BitLinearSlot, BitLinearWeights};

// =============================================================================
// Load helpers
// =============================================================================

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
                *value = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        // BitNet b1.58 ships all RMSNorm weights as F16
        // (`attn_norm`, `ffn_norm`, `attn_q_norm`, `attn_k_norm`,
        // `attn_v_norm`, `attn_output_norm`, `ffn_gate_norm`,
        // `ffn_up_norm`, `ffn_down_norm`). Pre-BitNet models use
        // BF16; the legacy `get_f32_tensor` only matched F32/BF16
        // and silently left the output all-zero, which made
        // `rms_norm(x, all-zeros) = 0` cascade through every BitLinear
        // and produce all-zero embeddings. Adding the F16 branch is
        // the missing piece for BitNet.
        GGMLType::F16 => {
            for (value, chunk) in output.iter_mut().zip(bytes.chunks_exact(2)) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                *value = crate::ops::float::f16_to_f32(bits);
            }
        }
        other => panic!(
            "get_f32_tensor {name}: unsupported ggml_type {other:?}; \
             expected F32/F16/BF16"
        ),
    }
    output
}

#[allow(clippy::too_many_arguments)]
pub fn load_layers<'a>(
    source: &'a dyn TensorSource,
    n_layer: usize,
    n_embd: usize,
    n_embd_q: usize,
    n_embd_gqa: usize,
    n_ff: usize,
    n_embd_head_k: usize,
    is_bitnet: bool,
) -> Vec<Qwen3LayerWeights<'a>> {
    // Both the Qwen3-Embedding path (mean pooling, has_qk_norm=true)
    // and the BitNet-Embeddings path (last-token pooling, has_qk_norm=true)
    // carry `blk.{i}.attn_{q,k}_norm.weight`. The legacy `load_layers`
    // helper predates the BitNet-specific dispatch and hardcodes
    // qk_norm loading — `load_layers_static` is the modern path that
    // honours both flags from the GGUF metadata.
    let has_qk_norm = true;
    (0..n_layer)
        .map(|l| Qwen3LayerWeights {
            attn_norm: get_f32_tensor(source, &format!("blk.{}.attn_norm.weight", l), n_embd),
            ffn_norm: get_f32_tensor(source, &format!("blk.{}.ffn_norm.weight", l), n_embd),
            q_norm: if has_qk_norm {
                Some(get_f32_tensor(
                    source,
                    &format!("blk.{}.attn_q_norm.weight", l),
                    n_embd_head_k,
                ))
            } else {
                None
            },
            k_norm: if has_qk_norm {
                Some(get_f32_tensor(
                    source,
                    &format!("blk.{}.attn_k_norm.weight", l),
                    n_embd_head_k,
                ))
            } else {
                None
            },
            q_bias: None,
            k_bias: None,
            v_bias: None,
            moe_router: None,
            moe_gate: None,
            moe_up: None,
            moe_down: None,
            wq: Weight::from_quantized(QuantizedTensor::from_bytes(
                source
                    .tensor_slice(&format!("blk.{}.attn_q.weight", l))
                    .unwrap(),
                source
                    .tensor_info(&format!("blk.{}.attn_q.weight", l))
                    .unwrap()
                    .ggml_type,
                n_embd,
                n_embd_q,
            )),
            wk: Weight::from_quantized(QuantizedTensor::from_bytes(
                source
                    .tensor_slice(&format!("blk.{}.attn_k.weight", l))
                    .unwrap(),
                source
                    .tensor_info(&format!("blk.{}.attn_k.weight", l))
                    .unwrap()
                    .ggml_type,
                n_embd,
                n_embd_gqa,
            )),
            wv: Weight::from_quantized(QuantizedTensor::from_bytes(
                source
                    .tensor_slice(&format!("blk.{}.attn_v.weight", l))
                    .unwrap(),
                source
                    .tensor_info(&format!("blk.{}.attn_v.weight", l))
                    .unwrap()
                    .ggml_type,
                n_embd,
                n_embd_gqa,
            )),
            wo: Weight::from_quantized(QuantizedTensor::from_bytes(
                source
                    .tensor_slice(&format!("blk.{}.attn_output.weight", l))
                    .unwrap(),
                source
                    .tensor_info(&format!("blk.{}.attn_output.weight", l))
                    .unwrap()
                    .ggml_type,
                n_embd_q,
                n_embd,
            )),
            w_gate: Weight::from_quantized(QuantizedTensor::from_bytes(
                source
                    .tensor_slice(&format!("blk.{}.ffn_gate.weight", l))
                    .unwrap(),
                source
                    .tensor_info(&format!("blk.{}.ffn_gate.weight", l))
                    .unwrap()
                    .ggml_type,
                n_embd,
                n_ff,
            )),
            w_up: Weight::from_quantized(QuantizedTensor::from_bytes(
                source
                    .tensor_slice(&format!("blk.{}.ffn_up.weight", l))
                    .unwrap(),
                source
                    .tensor_info(&format!("blk.{}.ffn_up.weight", l))
                    .unwrap()
                    .ggml_type,
                n_embd,
                n_ff,
            )),
            w_down: Weight::from_quantized(QuantizedTensor::from_bytes(
                source
                    .tensor_slice(&format!("blk.{}.ffn_down.weight", l))
                    .unwrap(),
                source
                    .tensor_info(&format!("blk.{}.ffn_down.weight", l))
                    .unwrap()
                    .ggml_type,
                n_ff,
                n_embd,
            )),
            bitlinear: if is_bitnet {
                load_bitlinear_layer_borrowed(source, l, n_embd, n_embd_q, n_embd_gqa, n_ff)
                    .expect("BitNet b1.58 BitLinear load failure")
            } else {
                BitLinearSlot::default()
            },
        })
        .collect()
}

pub use crate::core::loader::load_static_weight as static_weight;

#[allow(clippy::too_many_arguments)]
pub fn load_layers_static(
    source: Arc<dyn TensorSource>,
    n_layer: usize,
    n_embd: usize,
    n_embd_q: usize,
    n_embd_gqa: usize,
    n_ff: usize,
    n_embd_head_k: usize,
    has_qk_norm: bool,
    has_qkv_bias: bool,
    is_bitnet: bool,
    moe: Option<crate::core::loader::Qwen3MoeConfig>,
) -> Result<Vec<Qwen3LayerWeights<'static>>, String> {
    let source = source.as_ref();
    let mut layers = Vec::with_capacity(n_layer);
    for l in 0..n_layer {
        let q_bias = load_optional_bias(
            source,
            &format!("blk.{l}.attn_q.bias"),
            n_embd_q,
            has_qkv_bias,
        )?;
        let k_bias = load_optional_bias(
            source,
            &format!("blk.{l}.attn_k.bias"),
            n_embd_gqa,
            has_qkv_bias,
        )?;
        let v_bias = load_optional_bias(
            source,
            &format!("blk.{l}.attn_v.bias"),
            n_embd_gqa,
            has_qkv_bias,
        )?;
        let (w_gate, w_up, w_down, moe_router, moe_gate, moe_up, moe_down) = if let Some(moe) = moe
        {
            if moe.shared_expert_ffn != 0 {
                return Err("Qwen3VL-MoE shared experts are not supported".into());
            }
            let gate_slices = expert_slices(
                source,
                &format!("blk.{l}.ffn_gate_exps.weight"),
                n_embd,
                moe.expert_ffn,
                moe.expert_count,
            )?;
            let up_slices = expert_slices(
                source,
                &format!("blk.{l}.ffn_up_exps.weight"),
                n_embd,
                moe.expert_ffn,
                moe.expert_count,
            )?;
            let down_slices = expert_slices(
                source,
                &format!("blk.{l}.ffn_down_exps.weight"),
                moe.expert_ffn,
                n_embd,
                moe.expert_count,
            )?;
            let gate = gate_slices
                .iter()
                .map(|bytes| {
                    weight_from_bytes(
                        bytes,
                        source
                            .tensor_info(&format!("blk.{l}.ffn_gate_exps.weight"))
                            .unwrap()
                            .ggml_type,
                        n_embd,
                        moe.expert_ffn,
                    )
                })
                .collect::<Vec<_>>();
            let up = up_slices
                .iter()
                .map(|bytes| {
                    weight_from_bytes(
                        bytes,
                        source
                            .tensor_info(&format!("blk.{l}.ffn_up_exps.weight"))
                            .unwrap()
                            .ggml_type,
                        n_embd,
                        moe.expert_ffn,
                    )
                })
                .collect::<Vec<_>>();
            let down = down_slices
                .iter()
                .map(|bytes| {
                    weight_from_bytes(
                        bytes,
                        source
                            .tensor_info(&format!("blk.{l}.ffn_down_exps.weight"))
                            .unwrap()
                            .ggml_type,
                        moe.expert_ffn,
                        n_embd,
                    )
                })
                .collect::<Vec<_>>();
            let router = crate::core::tensor::load_f32_tensor(
                source,
                &format!("blk.{l}.ffn_gate_inp.weight"),
                &[n_embd as u64, moe.expert_count as u64],
            )?;
            (
                weight_from_bytes(
                    gate_slices[0],
                    source
                        .tensor_info(&format!("blk.{l}.ffn_gate_exps.weight"))
                        .unwrap()
                        .ggml_type,
                    n_embd,
                    moe.expert_ffn,
                ),
                weight_from_bytes(
                    up_slices[0],
                    source
                        .tensor_info(&format!("blk.{l}.ffn_up_exps.weight"))
                        .unwrap()
                        .ggml_type,
                    n_embd,
                    moe.expert_ffn,
                ),
                weight_from_bytes(
                    down_slices[0],
                    source
                        .tensor_info(&format!("blk.{l}.ffn_down_exps.weight"))
                        .unwrap()
                        .ggml_type,
                    moe.expert_ffn,
                    n_embd,
                ),
                Some(router),
                Some(gate),
                Some(up),
                Some(down),
            )
        } else {
            (
                static_weight(source, &format!("blk.{}.ffn_gate.weight", l), n_embd, n_ff),
                static_weight(source, &format!("blk.{}.ffn_up.weight", l), n_embd, n_ff),
                static_weight(source, &format!("blk.{}.ffn_down.weight", l), n_ff, n_embd),
                None,
                None,
                None,
                None,
            )
        };
        layers.push(Qwen3LayerWeights {
            attn_norm: get_f32_tensor(source, &format!("blk.{}.attn_norm.weight", l), n_embd),
            ffn_norm: get_f32_tensor(source, &format!("blk.{}.ffn_norm.weight", l), n_embd),
            q_norm: if has_qk_norm {
                Some(get_f32_tensor(
                    source,
                    &format!("blk.{}.attn_q_norm.weight", l),
                    n_embd_head_k,
                ))
            } else {
                None
            },
            q_bias,
            k_bias,
            v_bias,
            k_norm: if has_qk_norm {
                Some(get_f32_tensor(
                    source,
                    &format!("blk.{}.attn_k_norm.weight", l),
                    n_embd_head_k,
                ))
            } else {
                None
            },
            wq: static_weight(
                source,
                &format!("blk.{}.attn_q.weight", l),
                n_embd,
                n_embd_q,
            ),
            wk: static_weight(
                source,
                &format!("blk.{}.attn_k.weight", l),
                n_embd,
                n_embd_gqa,
            ),
            wv: static_weight(
                source,
                &format!("blk.{}.attn_v.weight", l),
                n_embd,
                n_embd_gqa,
            ),
            wo: static_weight(
                source,
                &format!("blk.{}.attn_output.weight", l),
                n_embd_q,
                n_embd,
            ),
            w_gate,
            w_up,
            w_down,
            moe_router,
            moe_gate,
            moe_up,
            moe_down,
            bitlinear: load_bitlinear_layer(source, l, is_bitnet, n_embd, n_embd_q, n_embd_gqa, n_ff)?,
        });
    }
    Ok(layers)
}

/// Load BitLinear projection slots for layer `l` when `is_bitnet` is
/// set; otherwise return a `BitLinearSlot::default()` (all seven
/// slots `None`). Each `Some(BitLinearWeights)` slot carries the
/// pre-projection RMSNorm weight (`*_norm_in.weight`, F16) and the
/// raw I2_S payload (`*.weight`, byte slice — see
/// `src/ops/kernel/i2_s.rs` for the block layout).
///
/// When `is_bitnet` is true but a tensor is missing we return an
/// error rather than silently degrading — the BitLinear forward in
/// `forward.rs::bitlinear_projection` requires both pieces per slot.
fn load_bitlinear_layer(
    source: &dyn TensorSource,
    l: usize,
    is_bitnet: bool,
    n_embd: usize,
    n_embd_q: usize,
    n_embd_gqa: usize,
    n_ff: usize,
) -> Result<BitLinearSlot, String> {
    if !is_bitnet {
        return Ok(BitLinearSlot::default());
    }
    load_bitlinear_layer_inner(
        source,
        l,
        n_embd,
        n_embd_q,
        n_embd_gqa,
        n_ff,
        |name, n_in| {
            let info = source
                .tensor_info(name)
                .ok_or_else(|| format!("BitNet missing tensor info for {name}"))?;
            if info.ggml_type != crate::core::tensor::GGMLType::I2_S {
                return Err(format!(
                    "BitNet {name} must be I2_S, got {:?}",
                    info.ggml_type
                ));
            }
            let bytes = source
                .tensor_slice(name)
                .ok_or_else(|| format!("BitNet missing data for {name}"))?
                .to_vec();
            Ok(bytes)
        },
        |name, n_in| {
            let info = source
                .tensor_info(name)
                .ok_or_else(|| format!("BitNet missing tensor info for {name}"))?;
            let bytes = source
                .tensor_slice(name)
                .ok_or_else(|| format!("BitNet missing data for {name}"))?;
            let mut norm = vec![0.0f32; n_in];
            match info.ggml_type {
                crate::core::tensor::GGMLType::F16 => {
                    for (dst, chunk) in norm.iter_mut().zip(bytes.chunks_exact(2)) {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        *dst = crate::ops::float::f16_to_f32(bits);
                    }
                }
                crate::core::tensor::GGMLType::F32 => {
                    for (dst, chunk) in norm.iter_mut().zip(bytes.chunks_exact(4)) {
                        *dst = f32::from_le_bytes(chunk.try_into().unwrap());
                    }
                }
                crate::core::tensor::GGMLType::BF16 => {
                    for (dst, chunk) in norm.iter_mut().zip(bytes.chunks_exact(2)) {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        *dst = crate::ops::float::bf16_to_f32(bits);
                    }
                }
                other => {
                    return Err(format!(
                        "BitNet {name} must be F16/F32/BF16, got {other:?}"
                    ));
                }
            }
            Ok(norm)
        },
    )
}

/// Borrowed-lifetime variant of `load_bitlinear_layer` for the
/// legacy `load_layers` helper (used by the embed CLI in
/// `qwen3::embedding::run_embedding_tokens`). Mirrors the
/// `'static` version but stores the I2_S payload as `Vec<u8>` (a
/// copy) since the embed CLI does not hold a `'static` source.
fn load_bitlinear_layer_borrowed(
    source: &dyn TensorSource,
    l: usize,
    n_embd: usize,
    n_embd_q: usize,
    n_embd_gqa: usize,
    n_ff: usize,
) -> Result<BitLinearSlot, String> {
    load_bitlinear_layer_inner(
        source,
        l,
        n_embd,
        n_embd_q,
        n_embd_gqa,
        n_ff,
        |name, _n_in| {
            let info = source
                .tensor_info(name)
                .ok_or_else(|| format!("BitNet missing tensor info for {name}"))?;
            if info.ggml_type != crate::core::tensor::GGMLType::I2_S {
                return Err(format!(
                    "BitNet {name} must be I2_S, got {:?}",
                    info.ggml_type
                ));
            }
            Ok(source
                .tensor_slice(name)
                .ok_or_else(|| format!("BitNet missing data for {name}"))?
                .to_vec())
        },
        |name, n_in| {
            let info = source
                .tensor_info(name)
                .ok_or_else(|| format!("BitNet missing tensor info for {name}"))?;
            let bytes = source
                .tensor_slice(name)
                .ok_or_else(|| format!("BitNet missing data for {name}"))?;
            let mut norm = vec![0.0f32; n_in];
            match info.ggml_type {
                crate::core::tensor::GGMLType::F16 => {
                    for (dst, chunk) in norm.iter_mut().zip(bytes.chunks_exact(2)) {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        *dst = crate::ops::float::f16_to_f32(bits);
                    }
                }
                crate::core::tensor::GGMLType::F32 => {
                    for (dst, chunk) in norm.iter_mut().zip(bytes.chunks_exact(4)) {
                        *dst = f32::from_le_bytes(chunk.try_into().unwrap());
                    }
                }
                crate::core::tensor::GGMLType::BF16 => {
                    for (dst, chunk) in norm.iter_mut().zip(bytes.chunks_exact(2)) {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        *dst = crate::ops::float::bf16_to_f32(bits);
                    }
                }
                other => {
                    return Err(format!(
                        "BitNet {name} must be F16/F32/BF16, got {other:?}"
                    ));
                }
            }
            Ok(norm)
        },
    )
}

/// Shared loader body — the only difference between the two
/// callers is the lifetime of the I2_S byte slice (the embed CLI
/// clones the borrowed slice to `Vec<u8>`; the Qwen3Model path
/// transmutes to `'static` and keeps the borrowed reference). Both
/// callers route through this function.
fn load_bitlinear_layer_inner<L, N>(
    source: &dyn TensorSource,
    l: usize,
    n_embd: usize,
    n_embd_q: usize,
    n_embd_gqa: usize,
    n_ff: usize,
    load_i2s: L,
    load_norm: N,
) -> Result<BitLinearSlot, String>
where
    L: Fn(&str, usize) -> Result<Vec<u8>, String>,
    N: Fn(&str, usize) -> Result<Vec<f32>, String>,
{
    let dims: &[(&str, usize, usize)] = &[
        ("attn_q", n_embd, n_embd_q),
        ("attn_k", n_embd, n_embd_gqa),
        ("attn_v", n_embd, n_embd_gqa),
        ("attn_output", n_embd_q, n_embd),
        ("ffn_gate", n_embd, n_ff),
        ("ffn_up", n_embd, n_ff),
        ("ffn_down", n_ff, n_embd),
    ];
    let mut slots = BitLinearSlot::default();
    for (proj, n_in, n_out) in dims.iter() {
        let norm_name = format!("blk.{l}.{proj}_norm_in.weight");
        let weight_name = format!("blk.{l}.{proj}.weight");
        let norm_f32 = load_norm(&norm_name, *n_in)?;
        let weight_bytes = load_i2s(&weight_name, *n_out)?;
        let slot = match *proj {
            "attn_q" => &mut slots.attn_q,
            "attn_k" => &mut slots.attn_k,
            "attn_v" => &mut slots.attn_v,
            "attn_output" => &mut slots.attn_output,
            "ffn_gate" => &mut slots.ffn_gate,
            "ffn_up" => &mut slots.ffn_up,
            "ffn_down" => &mut slots.ffn_down,
            other => panic!("load_bitlinear_layer_inner: unhandled projection {other}"),
        };
        *slot = Some(BitLinearWeights {
            norm_in: norm_f32,
            weight: weight_bytes,
            n_in: *n_in,
            n_out: *n_out,
        });
    }
    Ok(slots)
}

fn weight_from_bytes(
    bytes: &'static [u8],
    ggml_type: GGMLType,
    n_in: usize,
    n_out: usize,
) -> Weight<'static> {
    Weight::from_quantized(QuantizedTensor::from_bytes(bytes, ggml_type, n_in, n_out))
}

fn expert_slices(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
) -> Result<Vec<&'static [u8]>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let expected = [n_in as u64, n_out as u64, expert_count as u64];
    if info.dims != expected {
        return Err(format!(
            "Invalid {name} shape {:?}; expected {expected:?}",
            info.dims
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let total = info
        .checked_nbytes()
        .ok_or_else(|| format!("Invalid tensor byte size: {name}"))? as usize;
    if bytes.len() != total || total % expert_count != 0 {
        return Err(format!("Invalid {name} byte length: {}", bytes.len()));
    }
    let per = total / expert_count;
    let bytes: &'static [u8] = unsafe { std::mem::transmute(bytes) };
    (0..expert_count)
        .map(|expert| {
            bytes
                .get(expert * per..(expert + 1) * per)
                .ok_or_else(|| format!("{name} expert slice overflow"))
        })
        .collect()
}

fn load_optional_bias(
    source: &dyn TensorSource,
    name: &str,
    expected_len: usize,
    required: bool,
) -> Result<Option<Vec<f32>>, String> {
    if source.tensor_info(name).is_none() {
        return if required {
            Err(format!("Missing tensor: {name}"))
        } else {
            Ok(None)
        };
    }
    let dims = [u64::try_from(expected_len).map_err(|_| format!("{name} length overflow"))?];
    crate::core::tensor::load_f32_tensor(source, name, &dims).map(Some)
}

// =============================================================================
// Qwen3Model impl (constructor + stateless accessors)
// =============================================================================

impl Qwen3Model {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        tokenizer: Arc<BPETokenizer>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        use super::util::{checked_product, load_f32_tensor, usize_to_u64};

        let config = Qwen3Config::from_source(source.as_ref())?;
        if config.vocab != tokenizer.vocab_size() {
            return Err(format!(
                "{} vocabulary size {} does not match tokenizer vocab {}",
                config.architecture,
                config.vocab,
                tokenizer.vocab_size()
            ));
        }
        let _n_embd_q = checked_product("query width", config.n_head, config.n_embd_head_k)?;
        let _n_embd_k = checked_product("key width", config.n_head_kv, config.n_embd_head_k)?;
        let _n_embd_v = checked_product("value width", config.n_head_kv, config.n_embd_head_v)?;

        let output_norm = load_f32_tensor(
            source.as_ref(),
            "output_norm.weight",
            &[usize_to_u64(config.n_embd, "embedding width")?],
        )?;
        let token_embedding_info = source
            .tensor_info("token_embd.weight")
            .expect("no token_embd.weight");
        let token_embedding_bytes = source.tensor_slice("token_embd.weight").expect("no embd");
        let token_embedding_bytes_static: &'static [u8] =
            unsafe { std::mem::transmute(token_embedding_bytes) };
        let token_embedding = Weight::from_quantized(QuantizedTensor::from_bytes(
            token_embedding_bytes_static,
            token_embedding_info.ggml_type,
            config.n_embd,
            config.vocab,
        ));

        let output_info = source
            .tensor_info("output.weight")
            .unwrap_or(token_embedding_info);
        let output_bytes = source
            .tensor_slice("output.weight")
            .unwrap_or(token_embedding_bytes);
        let output_bytes_static: &'static [u8] = unsafe { std::mem::transmute(output_bytes) };
        let output = Weight::from_quantized(QuantizedTensor::from_bytes(
            output_bytes_static,
            output_info.ggml_type,
            config.n_embd,
            config.vocab,
        ));

        // Optional cross-encoder / classification head. GGUF tensors for
        // these are typically named `cls.output.weight` (sometimes `.bias`)
        // by llama.cpp's quantizer. We only require the weight tensor;
        // absent means a plain generative model and `cls_score = None`.
        let cls_score = match source.tensor_info("cls.output.weight") {
            Some(info) => {
                let bytes = source
                    .tensor_slice("cls.output.weight")
                    .ok_or_else(|| format!("cls.output.weight data missing"))?;
                if info.dims.len() != 2 || info.dims[0] as usize != config.n_embd {
                    return Err(format!(
                        "cls.output.weight: shape {:?} does not match n_embd {}",
                        info.dims, config.n_embd
                    ));
                }
                let n_cls = info.dims[1] as usize;
                let bytes_static: &'static [u8] = unsafe { std::mem::transmute(bytes) };
                Some(Weight::from_quantized(QuantizedTensor::from_bytes(
                    bytes_static,
                    info.ggml_type,
                    config.n_embd,
                    n_cls,
                )))
            }
            None => None,
        };

        let mut layers: Vec<Qwen3LayerWeights<'static>> = load_layers_static(
            Arc::clone(&source),
            config.n_layer,
            config.n_embd,
            checked_product("query width", config.n_head, config.n_embd_head_k)?,
            checked_product("key width", config.n_head_kv, config.n_embd_head_k)?,
            config.n_ff,
            config.n_embd_head_k,
            config.has_qk_norm,
            config.has_qkv_bias,
            config.is_bitnet,
            config.moe,
        )?;
        if config.architecture == "qwen3" && crate::ops::scalar_mode() {
            for (index, layer) in layers.iter_mut().enumerate() {
                for (name, weight) in [
                    ("attn_q", &mut layer.wq),
                    ("attn_k", &mut layer.wk),
                    ("attn_v", &mut layer.wv),
                    ("attn_output", &mut layer.wo),
                    ("ffn_gate", &mut layer.w_gate),
                    ("ffn_up", &mut layer.w_up),
                    ("ffn_down", &mut layer.w_down),
                ] {
                    if weight.ggml_type == GGMLType::BF16 {
                        let tensor = format!("blk.{index}.{name}.weight");
                        let bytes = source.tensor_slice(&tensor).ok_or(tensor)?;
                        // SAFETY: Qwen3Model owns `source` for the lifetime of these kernels.
                        let bytes: &'static [u8] = unsafe { std::mem::transmute(bytes) };
                        weight.kernel =
                            Box::new(crate::ops::kernel::bf16::BF16Kernel::with_bf16_input(bytes));
                    }
                }
            }
        }

        Ok(Self {
            source,
            tokenizer,
            pool,
            config,
            layers,
            output_norm,
            token_embedding,
            output,
            cls_score,
        })
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    pub fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }

    pub fn pool(&self) -> Arc<ComputePool> {
        Arc::clone(&self.pool)
    }

    pub fn layers(&self) -> &Vec<Qwen3LayerWeights<'_>> {
        &self.layers
    }

    pub fn output_norm(&self) -> &Vec<f32> {
        &self.output_norm
    }

    /// Returns `true` when the GGUF carried a `cls.output.weight` head.
    /// Callers should use [`Qwen3Session::forward_rerank`] instead of
    /// `forward_logits` for such models.
    pub fn is_rerank(&self) -> bool {
        self.cls_score.is_some()
    }

    /// Classify `last_hidden` (length `n_embd`) with the rerank head,
    /// returning one logit per class. Returns an error if the head
    /// wasn't loaded.
    pub fn score_logits(&self, last_hidden: &[f32]) -> Result<Vec<f32>, String> {
        let weight = self
            .cls_score
            .as_ref()
            .ok_or_else(|| "model is not a rerank / cross-encoder".to_string())?;
        if last_hidden.len() != self.config.n_embd {
            return Err(format!(
                "score_logits: hidden size {} does not match n_embd {}",
                last_hidden.len(),
                self.config.n_embd
            ));
        }
        let n_in = self.config.n_embd;
        let n_cls = weight.n_out;
        let mut act_q8 = vec![0u8; n_in];
        let mut act_scales = vec![0.0f32; n_in.div_ceil(32)];
        crate::ops::quantize_q8_0_into(last_hidden, n_in, &mut act_q8, &mut act_scales);
        let mut out = vec![0.0f32; n_cls];
        weight.kernel.forward_prepared(
            last_hidden,
            &act_q8,
            &act_scales,
            None,
            &mut out,
            n_in,
            n_cls,
            0,
            1,
        );
        Ok(out)
    }

    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        use super::util::{check_allocation, checked_product, validate_token_ids};

        validate_token_ids(token_ids, self.config.vocab)?;
        let len = checked_product(
            "token embedding values",
            token_ids.len(),
            self.config.n_embd,
        )?;
        check_allocation("token embeddings", len, std::mem::size_of::<f32>())?;
        let mut embeddings = vec![0.0; len];
        for (row, &token_id) in embeddings
            .chunks_exact_mut(self.config.n_embd)
            .zip(token_ids)
        {
            self.token_embedding.embedding_lookup(token_id, row);
        }
        Ok(embeddings)
    }
}
