//! Shared data types for BitNet b1.58 (W1.58A8) BitLinear projections.
//!
//! Used by every trunk that consumes GGUF `file_type=40` models
//! (currently `qwen3` for `bitnet-embedding-0.6b` and `gemma3` for
//! `bitnet-embedding-270m`). The actual forward computation lives in
//! [`crate::ops::bitlinear`]; this module only owns the per-layer
//! weight storage and the seven-slot grouping.
//!
//! # Why these types live here, not in a per-trunk file
//!
//! The seven BitLinear slots (attn_q/k/v/output + ffn_gate/up/down)
//! and the wire format of each slot
//! (`norm_in: Vec<f32>` + `weight: Vec<u8>` I2_S payload + `n_in`/`n_out`)
//! are identical across all BitNet b1.58 trunks. Putting them in
//! `qwen3::trunk::weights` and forcing gemma3 to reach across would
//! be the wrong layering; `ops::bitlinear::weights` is the natural
//! home. Per-arch loaders (`load_bitlinear_layer_qwen3` /
//! `load_bitlinear_layer_gemma3`) stay next to their trunk because
//! the GGUF tensor names differ per arch.
//!
//! # Wire format reminder
//!
//! Each [`BitLinearWeights`] holds:
//!
//! - `norm_in: Vec<f32>` of length `n_in` — the pre-projection RMSNorm
//!   gain, absorbing the inverse-of-mean-absweight from the
//!   conversion. Used by [`crate::ops::bitlinear::bitlinear_forward`]
//!   before activation quant.
//! - `weight: Vec<u8>` of length `i2_s_row_bytes(n_out × n_in)` — the
//!   raw I2_S block payload (`QK_I2_S=128` elements per block, 32
//!   bytes per block, no in-block scale). Decoded by
//!   [`crate::ops::kernel::i2_s`].
//! - `n_in` / `n_out` — projection shape. For Qwen3-0.6B layer 0:
//!   attn_q has `n_in=1024`, `n_out=1024`; ffn_gate has
//!   `n_in=1024`, `n_out=3072`.
//!
//! See `docs/usage/bitnet_embedding.md` §2.1 for the spec.

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
///
/// `None` for every slot when the model is not BitNet; non-`None`
/// only when the trunk has detected `file_type=40` or `*_norm_in`
/// existence at load time.
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
