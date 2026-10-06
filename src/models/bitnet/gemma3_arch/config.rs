//! Gemma3 decoder configuration (`general.architecture = "gemma3"`).
//!
//! Used by [`crate::models::bitnet::gemma3_arch::forward`] and the
//! embedding path. Mirrors the metadata that
//! `bitnet-embedding-270m` carries; for an un-BitLinear regular
//! Gemma3 (which this engine does not yet support), this struct
//! would also be the right home for the hybrid sliding-window
//! pattern that larger Gemma3 (4B/12B/27B) ships — but 270M does
//! not declare `sliding_window` so it is omitted here.

use crate::core::tensor::TensorSource;

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

/// Build a `Gemma3Config` from GGUF metadata. Used by
/// [`crate::models::bitnet::gemma3_arch::embedding::load_model`].
pub fn build_config(source: &dyn TensorSource) -> Result<Gemma3Config, String> {
    let pick_u64 = |k: &str| {
        source
            .metadata(k)
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .ok_or_else(|| format!("bitnet::gemma3_arch: missing metadata {k}"))
    };
    let pick_f32 = |k: &str| {
        source
            .metadata(k)
            .and_then(|v| v.to_f64())
            .map(|v| v as f32)
            .ok_or_else(|| format!("bitnet::gemma3_arch: missing metadata {k}"))
    };
    Ok(Gemma3Config {
        architecture: source
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val())
            .unwrap_or("gemma3")
            .to_string(),
        n_embd: pick_u64("gemma3.embedding_length")?,
        n_layer: pick_u64("gemma3.block_count")?,
        n_head: pick_u64("gemma3.attention.head_count")?,
        n_head_kv: pick_u64("gemma3.attention.head_count_kv")?,
        n_embd_head_k: pick_u64("gemma3.attention.key_length")?,
        n_embd_head_v: pick_u64("gemma3.attention.value_length")?,
        n_ff: pick_u64("gemma3.feed_forward_length")?,
        vocab: pick_u64("gemma3.vocab_size")?,
        n_ctx: pick_u64("gemma3.context_length")?,
        eps: pick_f32("gemma3.attention.layer_norm_rms_epsilon")?,
        freq_base: pick_f32("gemma3.rope.freq_base")?,
        rope: Gemma3Rope::Neox,
        pooling_type: pick_u64("gemma3.pooling_type")? as u32,
    })
}
