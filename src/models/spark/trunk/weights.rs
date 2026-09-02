//! Spark 2.5 weight structures + load helpers.
//!
//! Per-layer weights (8 tensors each, all BF16 except norms which are F32):
//! - `attn_norm` / `ffn_norm` (F32, [n_embd])
//! - `attn_qkv` (BF16, [n_embd, n_embd_qkv]) — FUSED Q+K+V projection
//! - `attn_gate` (BF16, [n_embd, n_head]) — per-head sigmoid gate
//! - `attn_output` (BF16, [n_embd, n_embd])
//! - `ffn_gate` / `ffn_up` (BF16, [n_embd, n_ff]) — GeGLU
//! - `ffn_down` (BF16, [n_ff, n_embd])

use crate::core::tensor::TensorSource;
use crate::ops::kernel::Weight;

pub struct SparkLayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub attn_qkv: Weight<'a>,
    pub attn_gate: Weight<'a>,
    pub attn_output: Weight<'a>,
    pub ffn_gate: Weight<'a>,
    pub ffn_up: Weight<'a>,
    pub ffn_down: Weight<'a>,
}

/// Map an F32-or-BF16 norm tensor.
pub fn get_f32_tensor<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected_len: usize,
) -> Vec<f32> {
    crate::core::tensor::load_f32_tensor(source, name, &[expected_len as u64])
        .unwrap_or_else(|e| panic!("{e}"))
}

fn static_weight(
    source: &dyn TensorSource,
    name: &str,
    rows: usize,
    cols: usize,
) -> Weight<'static> {
    crate::core::loader::load_static_weight(source, name, rows, cols)
}

pub fn load_layers(
    n_layer: usize,
    n_embd: usize,
    n_embd_qkv: usize,
    n_head: usize,
    n_ff: usize,
    source: &dyn TensorSource,
) -> Vec<SparkLayerWeights<'static>> {
    (0..n_layer)
        .map(|l| SparkLayerWeights {
            attn_norm: get_f32_tensor(
                source,
                &format!("blk.{l}.attn_norm.weight"),
                n_embd,
            ),
            ffn_norm: get_f32_tensor(
                source,
                &format!("blk.{l}.ffn_norm.weight"),
                n_embd,
            ),
            attn_qkv: static_weight(
                source,
                &format!("blk.{l}.attn_qkv.weight"),
                n_embd,
                n_embd_qkv,
            ),
            attn_gate: static_weight(
                source,
                &format!("blk.{l}.attn_gate.weight"),
                n_embd,
                n_head,
            ),
            attn_output: static_weight(
                source,
                &format!("blk.{l}.attn_output.weight"),
                n_embd,
                n_embd,
            ),
            ffn_gate: static_weight(
                source,
                &format!("blk.{l}.ffn_gate.weight"),
                n_embd,
                n_ff,
            ),
            ffn_up: static_weight(
                source,
                &format!("blk.{l}.ffn_up.weight"),
                n_embd,
                n_ff,
            ),
            ffn_down: static_weight(
                source,
                &format!("blk.{l}.ffn_down.weight"),
                n_ff,
                n_embd,
            ),
        })
        .collect()
}