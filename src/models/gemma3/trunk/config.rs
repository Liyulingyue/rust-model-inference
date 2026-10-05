//! Gemma3 decoder configuration (`general.architecture = "gemma3"`).
//!
//! Used by [`crate::models::gemma3::trunk::forward`] and the
//! embedding path.
//!
//! # Forward path dispatch
//!
//! Two forward paths share this config:
//!
//! - **BitNet b1.58** (`cfg.is_bitnet == true`): every projection
//!   goes through [`crate::ops::bitnet::bitlinear_forward_packed`]
//!   (pre-dequant `{-1, 0, +1}` int8 SIMD path; the SIMD hot path
//!   added in afbb172). The `weight` slot in each layer is unused.
//! - **Standard** (`cfg.is_bitnet == false`): every projection
//!   goes through the standard `Weight::kernel.forward_prepared`
//!   path (Q4_K / Q5_0 / Q6K / Q8_0 mixed-quant matmul, the same
//!   kernels the qwen3 trunk uses). The `bitlinear` slot in each
//!   layer is `BitLinearSlot::default()`.
//!
//! # Sliding-window attention
//!
//! Standard Gemma3 (4B/12B/27B and the 270M-it variant) carries a
//! hybrid local/global attention pattern: a layer attends locally
//! to the last `sliding_window` tokens and globally to everything
//! else. The BitNet 270M GGUF does NOT declare this metadata; the
//! standard gemma3 trunks (4B+) do. When `sliding_window == 0` the
//! attention is full causal; otherwise `|i - j| > sliding_window`
//! keys are masked.

#[derive(Debug, Clone)]
pub struct Gemma3Config {
    pub architecture: String,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub n_ff: usize,
    pub vocab: usize,
    pub n_ctx: usize,
    pub eps: f32,
    pub freq_base: f32,
    pub rope: Gemma3Rope,
    pub pooling_type: u32,
    /// Sliding-window size for local attention (0 = full causal,
    /// no masking). Standard gemma3 270M-it declares 512; BitNet
    /// 270M does not declare this metadata at all and is treated
    /// as 0.
    pub sliding_window: usize,
    /// BitNet b1.58 flag — true when `general.file_type == 40` or
    /// when `*_norm_in` tensors are present. Decides whether the
    /// forward goes through [`crate::ops::bitnet::bitlinear_forward_packed`]
    /// at every projection (vs the standard matmul path).
    pub is_bitnet: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma3Rope {
    /// Standard RoPE. The 270M GGUF declares
    /// `gemma3.rope.dimension_count = 256` which equals `head_dim`,
    /// so every Q/K element is rotated (no partial rotation).
    Neox,
}

impl Gemma3Config {
    /// `n_embd_q` for the per-layer attention output before the
    /// `attn_output` projection: `n_head × n_embd_head_k`.
    pub fn n_embd_q(&self) -> usize {
        self.n_head * self.n_embd_head_k
    }
    /// `n_embd_kv` shared across the GQA group:
    /// `n_head_kv × n_embd_head_v`.
    pub fn n_embd_kv(&self) -> usize {
        self.n_head_kv * self.n_embd_head_v
    }
    /// `n_embd_gqa` = q-projection output dim = `n_embd_q`.
    pub fn n_embd_gqa(&self) -> usize {
        self.n_embd_q()
    }
    /// Per-head RoPE dimension. 270M = 256 (== head_dim).
    pub fn n_rot(&self) -> usize {
        self.n_embd_head_k
    }
}
