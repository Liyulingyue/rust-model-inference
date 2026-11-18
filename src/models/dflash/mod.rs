//! The DFlash / DSpark speculative draft model (`arch == "dflash"`).
//!
//! A DFlash file is a *sidecar*: a small block-diffusion transformer that
//! drafts for a larger target. Its attention is Qwen3-style (plain Q/K/V with
//! per-head norms and M-RoPE) plus a SwiGLU FFN; on top of the trunk sit the
//! DSpark heads — the Markov head that biases each block position's logits by
//! the previously committed token, and the confidence head that estimates
//! acceptance for the `p_min` gate. See
//! [`docs/develop/SPECULATIVE_DECODING.md`](../../../docs/develop/SPECULATIVE_DECODING.md)
//! for the upstream contract and the driving loop.
//!
//! The draft is never a primary model: the target loads it alongside itself
//! and drives it through
//! [`DflashSession`](trunk/struct.DflashSession.html).

pub mod trunk;

pub use trunk::{DflashConfig, DflashSession, DflashWeights, DsparkDraft};
