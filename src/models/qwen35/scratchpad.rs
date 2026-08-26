//! Buffers reused across forward passes for a single Qwen3.5 inference request.
//!
//! Includes:
//! - per-token activation buffers (`x`, `buf`, `normed_buf`)
//! - dense-attention scratch (`q_buf`, `k_buf`, `v_buf`, `score_buf`, ...)
//! - Mamba SSM cross-call state (`conv_states`, `ssm_states`)
//! - shared matmul scratch (`q8k_buf`, `q8_buf`, `scale_buf`, `matmul_out`)
//!
//! All buffers are sized for `max_tokens` (used for the largest per-call
//! token batch) plus `n_ctx` (used for padded attention buffers).

use crate::core::scratchpad::KvCache;
use crate::models::qwen35::clip_config::Qwen35Config;
use crate::ops::quant;

pub struct Qwen35Scratchpad {
    pub x: Vec<f32>,
    pub buf: Vec<f32>,
    pub normed_buf: Vec<f32>,
    pub q_buf: Vec<f32>,
    pub k_buf: Vec<f32>,
    pub v_buf: Vec<f32>,
    pub k_buf2: Vec<f32>,
    pub v_buf2: Vec<f32>,
    pub qkv_buf: Vec<f32>,
    pub z_buf: Vec<f32>,
    pub beta_buf: Vec<f32>,
    pub alpha_buf: Vec<f32>,
    pub score_buf: Vec<f32>,
    pub attention_value_buf: Vec<f32>,
    pub attn_out_buf: Vec<f32>,
    pub ffn_up_buf: Vec<f32>,
    pub ffn_gate_buf: Vec<f32>,
    pub conv_states: Vec<Vec<f32>>,
    pub ssm_states: Vec<Vec<f32>>,
    pub matmul_out: Vec<f32>,
    pub q8k_buf: Vec<quant::BlockQ8K>,
    pub q8_buf: Vec<u8>,
    pub scale_buf: Vec<f32>,
}

impl Qwen35Scratchpad {
    pub fn new(config: &Qwen35Config, max_tokens: usize) -> Self {
        let n_embd = config.n_embd;
        let n_head = config.n_head;
        let n_head_kv = config.n_head_kv;
        let n_embd_head = config.n_embd_head();
        let n_ff = config.n_ff;
        let n_layer = config.n_layer;
        let d_inner = config.ssm_d_inner;
        let key_dim = config.key_dim();
        let value_dim = config.value_dim();
        let conv_dim = config.conv_dim();
        let d_conv = config.ssm_d_conv;
        let num_v_heads = config.ssm_dt_rank;
        let head_v_dim = d_inner / num_v_heads;
        let q_dim = n_embd_head * n_head * 2;
        let dense_attn_out_dim = n_embd_head * n_head;
        let max_matmul_input = n_embd.max(n_ff).max(value_dim).max(dense_attn_out_dim);

        Self {
            x: vec![0.0; max_tokens * n_embd],
            buf: vec![0.0; max_tokens * n_embd],
            q_buf: vec![0.0; max_tokens * q_dim.max(key_dim)],
            k_buf: vec![0.0; max_tokens * n_embd_head * n_head_kv],
            v_buf: vec![0.0; max_tokens * n_embd_head * n_head_kv],
            k_buf2: vec![0.0; max_tokens * key_dim],
            v_buf2: vec![0.0; max_tokens * value_dim],
            qkv_buf: vec![0.0; max_tokens * conv_dim],
            z_buf: vec![0.0; max_tokens * value_dim],
            beta_buf: vec![0.0; max_tokens * num_v_heads],
            alpha_buf: vec![0.0; max_tokens * num_v_heads],
            score_buf: vec![0.0; config.n_ctx.div_ceil(256) * 256],
            attention_value_buf: vec![0.0; config.n_ctx.div_ceil(256) * 256],
            attn_out_buf: vec![0.0; max_tokens * dense_attn_out_dim.max(value_dim)],
            ffn_up_buf: vec![0.0; max_tokens * n_ff],
            ffn_gate_buf: vec![0.0; max_tokens * n_ff],
            conv_states: (0..n_layer).map(|_| vec![0.0; d_conv * conv_dim]).collect(),
            ssm_states: (0..n_layer).map(|_| vec![0.0; num_v_heads * head_v_dim * head_v_dim]).collect(),
            matmul_out: vec![0.0; (2 * n_ff).max(conv_dim).max(n_embd).max(config.vocab_size)],
            normed_buf: vec![0.0; max_tokens * n_embd],
            q8k_buf: vec![quant::BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }; (max_matmul_input + 255) / 256],
            q8_buf: vec![0u8; max_matmul_input],
            scale_buf: vec![0.0; (max_matmul_input + 31) / 32],
        }
    }
}

// ============================================================================
// KV cache helpers used by the dense attention forward path
// ============================================================================

/// Locate the next free slot in layer `il`'s K cache. Scans for the longest
/// zero prefix of the K rows (the F32 cache is zero-initialized at
/// construction/reset, so zero = unfilled).
pub(crate) fn kv_cache_pos(cache: &KvCache, il: usize, k_dim: usize, n_layer: usize) -> usize {
    if let KvCache::F32(c) = cache {
        let k_len = c.k.len() / n_layer;
        let mut pos = 0;
        for p in 0..k_len / k_dim {
            if c.k[il * k_len + p * k_dim..il * k_len + (p + 1) * k_dim].iter().all(|v| *v == 0.0) { pos = p; break; }
            pos = p + 1;
        }
        pos
    } else { 0 }
}

/// Append `n_tokens` rows of K/V data starting at `pos` for layer `il`.
pub(crate) fn kv_cache_store(
    cache: &mut KvCache, il: usize, n_layer: usize,
    k_data: &[f32], v_data: &[f32], k_dim: usize, v_dim: usize, pos: usize,
) {
    if let KvCache::F32(c) = cache {
        let k_len = c.k.len() / n_layer;
        let v_len = c.v.len() / n_layer;
        let n_tokens = k_data.len() / k_dim;
        for t in 0..n_tokens {
            let k_dst = il * k_len + (pos + t) * k_dim;
            let v_dst = il * v_len + (pos + t) * v_dim;
            c.k[k_dst..k_dst + k_dim].copy_from_slice(&k_data[t * k_dim..(t + 1) * k_dim]);
            c.v[v_dst..v_dst + v_dim].copy_from_slice(&v_data[t * v_dim..(t + 1) * v_dim]);
        }
    }
}