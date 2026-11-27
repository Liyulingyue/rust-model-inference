//! # LLaMA Skeleton
//!
//! Tensor loading for LLaMA-family architectures. Tensor names follow the
//! llama.cpp convention: `blk.{i}.attn_norm`, `blk.{i}.attn_q`, etc.
//! LLaMA does NOT use Q/K per-head RMSNorm.

use crate::core::tensor::TensorSource;
use crate::ops::kernel::{QuantizedTensor, Weight};
use std::sync::Arc;

/// Nanbeige shares physical weights across loops but uses separate KV slots.
pub(crate) fn layer_loop_config(
    source: &dyn TensorSource,
    config: &crate::core::traits::ModelConfig,
) -> Result<(usize, bool), String> {
    if source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        != Some("nanbeige")
    {
        return Ok((config.n_layer, false));
    }
    let loops = match source.metadata("nanbeige.num_loops") {
        None => 1,
        Some(value) => value
            .to_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("Invalid nanbeige.num_loops")?,
    };
    let n_layer = config
        .n_layer
        .checked_mul(loops)
        .filter(|&n| loops > 0 && n > 0 && n <= 512)
        .ok_or("Invalid nanbeige logical layer count (must be within 1..=512)")?;
    let skip_norm = match source.metadata("nanbeige.skip_loop_final_norm") {
        None => false,
        Some(crate::core::tensor::MetaValue::Bool(value)) => *value,
        _ => return Err("Invalid nanbeige.skip_loop_final_norm".into()),
    };
    let head_dim = config.n_embd_head;
    if config.n_head_kv == 0
        || config.n_head % config.n_head_kv != 0
        || head_dim % 2 != 0
        || source
            .metadata("nanbeige.attention.value_length")
            .and_then(|v| v.to_u64())
            != Some(head_dim as u64)
        || source
            .metadata("nanbeige.rope.dimension_count")
            .and_then(|v| v.to_u64())
            != Some(head_dim as u64)
        || !config.norm_eps.is_finite()
        || config.norm_eps <= 0.0
        || !config.rope_freq_base.is_finite()
        || config.rope_freq_base <= 0.0
    {
        return Err(
            "Unsupported nanbeige attention/RoPE dimensions or normalization metadata".into(),
        );
    }
    let embd = config.n_embd as u64;
    let q = config
        .n_head
        .checked_mul(head_dim)
        .ok_or("Nanbeige query width overflow")? as u64;
    let kv = config
        .n_head_kv
        .checked_mul(head_dim)
        .ok_or("Nanbeige KV width overflow")? as u64;
    let ff = config.n_ff as u64;
    let check = |name: &str, dims: &[u64]| -> Result<(), String> {
        let info = source
            .tensor_info(name)
            .ok_or_else(|| format!("Missing tensor: {name}"))?;
        if info.dims != dims {
            return Err(format!(
                "Invalid {name} shape: {:?}, expected {dims:?}",
                info.dims
            ));
        }
        Ok(())
    };
    check("token_embd.weight", &[embd, config.vocab_size as u64])?;
    check("output_norm.weight", &[embd])?;
    if source.tensor_info("output.weight").is_some() {
        check("output.weight", &[embd, config.vocab_size as u64])?;
    }
    for layer in 0..config.n_layer {
        for (suffix, dims) in [
            ("attn_norm", vec![embd]),
            ("ffn_norm", vec![embd]),
            ("attn_q", vec![embd, q]),
            ("attn_k", vec![embd, kv]),
            ("attn_v", vec![embd, kv]),
            ("attn_output", vec![q, embd]),
            ("ffn_gate", vec![embd, ff]),
            ("ffn_up", vec![embd, ff]),
            ("ffn_down", vec![ff, embd]),
        ] {
            check(&format!("blk.{layer}.{suffix}.weight"), &dims)?;
        }
    }
    Ok((n_layer, !skip_norm))
}

pub struct LlamaLayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    /// Post-attention RMSNorm (`post_attention_norm.weight`). GLM-4
    /// (`glm4` GGUF) applies RMSNorm on the attention output *before*
    /// the residual add. None for plain llama/nanbeige/etc.
    pub attn_post_norm: Option<Vec<f32>>,
    /// Post-FFN RMSNorm (`post_ffw_norm.weight`). GLM-4 applies RMSNorm
    /// on the FFN output *before* the residual add. None for plain
    /// llama/nanbeige/etc.
    pub ffn_post_norm: Option<Vec<f32>>,
    pub wq: Weight<'a>,
    pub wk: Weight<'a>,
    pub wv: Weight<'a>,
    /// QKV biases (GLM-4 ships separate `attn_q/k/v.bias` tensors; plain
    /// llama has no biases). Loaded as `None` when the GGUF does not
    /// carry the corresponding tensor.
    pub bq: Option<Vec<f32>>,
    pub bk: Option<Vec<f32>>,
    pub bv: Option<Vec<f32>>,
    pub wo: Weight<'a>,
    pub w_gate: Weight<'a>,
    pub w_up: Weight<'a>,
    pub w_down: Weight<'a>,
}

pub fn get_f32_tensor<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected_len: usize,
) -> Vec<f32> {
    crate::core::tensor::load_f32_tensor(source, name, &[expected_len as u64])
        .unwrap_or_else(|e| panic!("{e}"))
}

/// Load an optional f32 tensor (`Some` if present in GGUF, `None` otherwise).
/// Tensors with names like `blk.{l}.{suffix}.bias` are common in
/// GLM-4 (attn_q/k/v.bias) and absent in plain llama.
fn try_load_f32_tensor<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected_len: usize,
) -> Option<Vec<f32>> {
    source.tensor_info(name)?;
    Some(get_f32_tensor(source, name, expected_len))
}

/// Optional f32 RMSNorm load. Returns `Some(weights)` if `blk.{l}.{suffix}.weight`
/// exists in the GGUF, `None` otherwise. GLM-4 (`glm4` arch) carries
/// `post_attention_norm` and `post_ffw_norm` tensors; plain
/// llama/nanbeige do not. Both-or-none within a layer is enforced:
/// if one is present and the other missing we return an error.
fn try_load_post_norms<S: TensorSource + ?Sized>(
    source: &S,
    l: usize,
    n_embd: usize,
) -> Result<(Option<Vec<f32>>, Option<Vec<f32>>), String> {
    let attn_name = format!("blk.{l}.post_attention_norm.weight");
    let ffn_name = format!("blk.{l}.post_ffw_norm.weight");
    let attn_present = source.tensor_info(&attn_name).is_some();
    let ffn_present = source.tensor_info(&ffn_name).is_some();
    match (attn_present, ffn_present) {
        (false, false) => Ok((None, None)),
        (true, false) | (false, true) => Err(format!(
            "blk.{l} post-norms must be both-or-neither (post_attention_norm={attn_present}, post_ffw_norm={ffn_present})"
        )),
        (true, true) => Ok((
            Some(get_f32_tensor(source, &attn_name, n_embd)),
            Some(get_f32_tensor(source, &ffn_name, n_embd)),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn load_layers<'a>(
    source: &'a dyn TensorSource,
    n_layer: usize,
    n_embd: usize,
    n_embd_q: usize,
    n_embd_gqa: usize,
    n_ff: usize,
) -> Vec<LlamaLayerWeights<'a>> {
    (0..n_layer)
        .map(|l| {
            let (attn_post_norm, ffn_post_norm) =
                try_load_post_norms(source, l, n_embd).expect("post-norm load");
            LlamaLayerWeights {
                attn_norm: get_f32_tensor(source, &format!("blk.{}.attn_norm.weight", l), n_embd),
                ffn_norm: get_f32_tensor(source, &format!("blk.{}.ffn_norm.weight", l), n_embd),
                attn_post_norm,
                ffn_post_norm,
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
                bq: try_load_f32_tensor(source, &format!("blk.{}.attn_q.bias", l), n_embd_q),
                bk: try_load_f32_tensor(source, &format!("blk.{}.attn_k.bias", l), n_embd_gqa),
                bv: try_load_f32_tensor(source, &format!("blk.{}.attn_v.bias", l), n_embd_gqa),
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
                w_gate: if source
                    .tensor_info(&format!("blk.{l}.ffn_gate.weight"))
                    .is_some()
                {
                    Weight::from_quantized(QuantizedTensor::from_bytes(
                        source
                            .tensor_slice(&format!("blk.{l}.ffn_gate.weight"))
                            .unwrap(),
                        source
                            .tensor_info(&format!("blk.{l}.ffn_gate.weight"))
                            .unwrap()
                            .ggml_type,
                        n_embd,
                        n_ff,
                    ))
                } else {
                    // GLM-4 ships a single fused `[n_embd, 2*n_ff]`
                    // `ffn_up.weight`; no separate `ffn_gate` exists.
                    // Substitute `w_up`'s Weight data so `w_gate` stays
                    // initialised; the GLM-4 forward path invokes
                    // `w_up.kernel` (n_out=2*n_ff) and never `w_gate.kernel`.
                    Weight::from_quantized(QuantizedTensor::from_bytes(
                        source
                            .tensor_slice(&format!("blk.{l}.ffn_up.weight"))
                            .unwrap(),
                        source
                            .tensor_info(&format!("blk.{l}.ffn_up.weight"))
                            .unwrap()
                            .ggml_type,
                        n_embd,
                        2 * n_ff,
                    ))
                },
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
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn load_layers_static(
    source: Arc<dyn TensorSource>,
    n_layer: usize,
    n_embd: usize,
    n_embd_q: usize,
    n_embd_gqa: usize,
    n_ff: usize,
) -> Vec<LlamaLayerWeights<'static>> {
    let source = source.as_ref();
    (0..n_layer)
        .map(|l| {
            let (attn_post_norm, ffn_post_norm) =
                try_load_post_norms(source, l, n_embd).expect("post-norm load");
            LlamaLayerWeights {
                attn_norm: get_f32_tensor(source, &format!("blk.{}.attn_norm.weight", l), n_embd),
                ffn_norm: get_f32_tensor(source, &format!("blk.{}.ffn_norm.weight", l), n_embd),
                attn_post_norm,
                ffn_post_norm,
                wq: crate::core::loader::load_static_weight(
                    source,
                    &format!("blk.{}.attn_q.weight", l),
                    n_embd,
                    n_embd_q,
                ),
                wk: crate::core::loader::load_static_weight(
                    source,
                    &format!("blk.{}.attn_k.weight", l),
                    n_embd,
                    n_embd_gqa,
                ),
                wv: crate::core::loader::load_static_weight(
                    source,
                    &format!("blk.{}.attn_v.weight", l),
                    n_embd,
                    n_embd_gqa,
                ),
                bq: try_load_f32_tensor(source, &format!("blk.{}.attn_q.bias", l), n_embd_q),
                bk: try_load_f32_tensor(source, &format!("blk.{}.attn_k.bias", l), n_embd_gqa),
                bv: try_load_f32_tensor(source, &format!("blk.{}.attn_v.bias", l), n_embd_gqa),
                wo: crate::core::loader::load_static_weight(
                    source,
                    &format!("blk.{}.attn_output.weight", l),
                    n_embd_q,
                    n_embd,
                ),
                w_gate: crate::core::loader::load_static_weight(
                    source,
                    &format!("blk.{}.ffn_gate.weight", l),
                    n_embd,
                    n_ff,
                ),
                w_up: crate::core::loader::load_static_weight(
                    source,
                    &format!("blk.{}.ffn_up.weight", l),
                    n_embd,
                    n_ff,
                ),
                w_down: crate::core::loader::load_static_weight(
                    source,
                    &format!("blk.{}.ffn_down.weight", l),
                    n_ff,
                    n_embd,
                ),
            }
        })
        .collect()
}
