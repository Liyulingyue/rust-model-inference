//! Data shapes for BitNet b1.58 (W1.58A8) BitLinear projections.
//!
//! These types are **shape-only metadata**, not ops — they describe
//! how one layer's seven BitLinear projections are stored in the
//! GGUF. The actual activation-quant and matmul ops live in
//! [`crate::ops::bitnet::forward`].
//!
//! Used by every trunk under [`crate::models::bitnet`] (currently
//! the Qwen3-arch and Gemma3-arch BitNet trunks for
//! `bitnet-embedding-{0.6b,270m}`). The qwen3 / gemma3 trunks
//! themselves are BitNet-free; this module only exists because the
//! two BitNet architectures share an identical seven-slot
//! projection layout.
//!
//! # Wire format reminder
//!
//! Each [`BitLinearWeights`] holds:
//!
//! - `norm_in: Vec<f32>` of length `n_in` — the pre-projection RMSNorm
//!   gain, absorbing the inverse-of-mean-absweight from the
//!   conversion.
//! - `weight: Vec<u8>` of length `i2_s_row_bytes(n_out × n_in)` — the
//!   raw I2_S block payload (`QK_I2_S=128` elements per block, 32
//!   bytes per block, no in-block scale). Decoded by
//!   [`crate::ops::kernel::i2_s`].
//! - `n_in` / `n_out` — projection shape.
//!
//! See `docs/usage/bitnet_embedding.md` §2 for the spec.

/// One BitLinear projection's payload as stored in the GGUF
/// (`file_type=40`): pre-projection RMSNorm gain + I2_S ternary
/// weight bytes + projection shape.
///
/// `norm_in` is loaded as F32; `weight` is loaded as raw bytes in
/// the I2_S block layout (`QK_I2_S=128`, 32 bytes/block,
/// 4 ternary codes per byte, mapping `0b00→-1`, `0b01→0`,
/// `0b10→+1`, `0b11→reserved→0`).
#[derive(Debug, Clone)]
pub struct BitLinearWeights {
    pub norm_in: Vec<f32>,
    pub weight: Vec<u8>,
    pub n_in: usize,
    pub n_out: usize,
}

/// The seven BitLinear projection slots per decoder layer
/// (attn_q/k/v/output + ffn_gate/up/down).
#[derive(Debug, Clone, Default)]
pub struct BitLinearSlot {
    pub attn_q: Option<BitLinearWeights>,
    pub attn_k: Option<BitLinearWeights>,
    pub attn_v: Option<BitLinearWeights>,
    pub attn_output: Option<BitLinearWeights>,
    pub ffn_gate: Option<BitLinearWeights>,
    pub ffn_up: Option<BitLinearWeights>,
    pub ffn_down: Option<BitLinearWeights>,
}
