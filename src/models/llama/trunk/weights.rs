//! # LLaMA Skeleton
//!
//! Tensor loading for LLaMA-family architectures. Tensor names follow the
//! llama.cpp convention: `blk.{i}.attn_norm`, `blk.{i}.attn_q`, etc.
//! LLaMA does NOT use Q/K per-head RMSNorm.

use crate::core::tensor::TensorSource;
use crate::ops::kernel::{QuantizedTensor, Weight};
use std::sync::Arc;

pub struct LlamaLayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub wq: Weight<'a>,
    pub wk: Weight<'a>,
    pub wv: Weight<'a>,
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
        .map(|l| LlamaLayerWeights {
            attn_norm: get_f32_tensor(source, &format!("blk.{}.attn_norm.weight", l), n_embd),
            ffn_norm: get_f32_tensor(source, &format!("blk.{}.ffn_norm.weight", l), n_embd),
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
        .map(|l| LlamaLayerWeights {
            attn_norm: get_f32_tensor(source, &format!("blk.{}.attn_norm.weight", l), n_embd),
            ffn_norm: get_f32_tensor(source, &format!("blk.{}.ffn_norm.weight", l), n_embd),
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
            wo: crate::core::loader::load_static_weight(
                source,
                &format!("blk.{}.attn_output.weight", l),
                n_embd_q,
                n_embd,
            ),
            w_gate: crate::core::loader::load_static_weight(source, &format!("blk.{}.ffn_gate.weight", l), n_embd, n_ff),
            w_up: crate::core::loader::load_static_weight(source, &format!("blk.{}.ffn_up.weight", l), n_embd, n_ff),
            w_down: crate::core::loader::load_static_weight(source, &format!("blk.{}.ffn_down.weight", l), n_ff, n_embd),
        })
        .collect()
}
