//! Gemma3 decoder configuration (`general.architecture = "gemma3"`).
//!
//! Used by [`crate::models::gemma3::trunk::forward`] and the
//! embedding path. Mirrors the metadata that
//! `bitnet-embedding-270m` carries; for an un-BitLinear regular
//! Gemma3 (which this engine does not yet support), this struct
//! would also be the right home for the hybrid sliding-window
//! pattern that larger Gemma3 (4B/12B/27B) ships — but 270M does
//! not declare `sliding_window` so it is omitted here.

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
    /// BitNet b1.58 flag — true when `general.file_type == 40` or
    /// when `*_norm_in` tensors are present. Decides whether the
    /// forward goes through [`crate::ops::bitlinear::bitlinear_forward`]
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
