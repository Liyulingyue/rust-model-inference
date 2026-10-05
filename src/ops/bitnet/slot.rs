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

impl BitLinearWeights {
    /// Pre-pack the I2_S weight matrix into a flat int8 buffer of
    /// length `n_in × n_out` (row-major: row `j` at offset
    /// `j * n_in`), with each byte holding the corresponding
    /// ternary value in `{-1, 0, +1}`. This is the layout that the
    /// AVX2 SIMD kernel in [`crate::ops::bitnet::forward_avx2`]
    /// consumes directly — calling
    /// [`crate::ops::bitnet::bitlinear_forward_packed`] with the
    /// result skips the per-call dequant-to-int8 walk entirely.
    ///
    /// One-time cost on model load (every projection is pre-packed
    /// exactly once); saves a per-forward dequant pass for every
    /// subsequent forward call. For the 0.6B model with 28 layers
    /// × 7 projections = 196 calls, the pre-pack amortizes over
    /// every forward call in the inference loop.
    ///
    /// See [`BitLinearWeightsPacked`] for the consumer side.
    pub fn prepack(&self) -> BitLinearWeightsPacked {
        let mut weight_i8 = vec![0i8; self.n_in * self.n_out];
        crate::ops::bitnet::forward_avx2::dequant_i2_s_to_i8(
            &self.weight,
            self.n_in,
            self.n_out,
            &mut weight_i8,
        );
        BitLinearWeightsPacked {
            norm_in: self.norm_in.clone(),
            weight_i8,
            n_in: self.n_in,
            n_out: self.n_out,
        }
    }
}

/// Same shape as [`BitLinearWeights`] but with the weight matrix
/// pre-dequanted to `{-1, 0, +1}` int8 — the layout that
/// [`crate::ops::bitnet::bitlinear_forward_packed`] and
/// [`crate::ops::bitnet::forward_avx2::bitlinear_forward_avx2_packed`]
/// consume directly. Produced via [`BitLinearWeights::prepack`]
/// during model load to skip the per-forward dequant walk.
#[derive(Debug, Clone)]
pub struct BitLinearWeightsPacked {
    pub norm_in: Vec<f32>,
    /// Row-major int8 weights of length `n_in × n_out`. Row `j`
    /// occupies indices `[j * n_in, (j + 1) * n_in)`.
    pub weight_i8: Vec<i8>,
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

/// The seven BitLinear projection slots per decoder layer, with weights
/// pre-dequanted to int8 via [`BitLinearWeights::prepack`]. Drop-in
/// replacement for [`BitLinearSlot`] in the forward hot path.
#[derive(Debug, Clone, Default)]
pub struct BitLinearSlotPacked {
    pub attn_q: Option<BitLinearWeightsPacked>,
    pub attn_k: Option<BitLinearWeightsPacked>,
    pub attn_v: Option<BitLinearWeightsPacked>,
    pub attn_output: Option<BitLinearWeightsPacked>,
    pub ffn_gate: Option<BitLinearWeightsPacked>,
    pub ffn_up: Option<BitLinearWeightsPacked>,
    pub ffn_down: Option<BitLinearWeightsPacked>,
}
