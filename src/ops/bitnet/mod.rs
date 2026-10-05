//! Microsoft BitNet b1.58 op family.
//!
//! BitNet b1.58 (W1.58A8) replaces every linear projection in a
//! transformer decoder with a 4-step pipeline:
//!
//! ```text
//! x → RMSNorm(x, norm_in.weight)         // pre-projection centering
//!   → per-token absmax quantize(x)         // activations to int8
//!   → matmul(ternary_weights, x_q)         // I2_S × int8
//!   → rescale by (absmax / 127)            // recover F32 magnitude
//! ```
//!
//! This module ships the **scalar reference implementation** of the
//! two pure ops that compose into a [`BitLinear projection`]:
//!
//! - [`quantize_activation_per_token`]: per-row absmax int8 quant
//! - [`bitlinear_forward`]: I2_S × int8 matmul + rescale
//!
//! The data types that carry a layer's seven BitLinear slots
//! (`BitLinearWeights`, `BitLinearSlot`) live in
//! [`crate::ops::bitnet::slot`] — they're not ops, they're
//! shape-only metadata for the BitNet projection pattern.
//!
//! The per-projection helper that stitches RMSNorm + quantize +
//! matmul together is a **model-level** concern (it picks
//! `eps` from the model config and decides how to handle the
//! residual stream); it lives in
//! `crate::models::bitnet::*::bitlinear_projection`. Keeping it
//! out of `ops/` avoids a backward dependency from ops into model
//! configuration.
//!
//! # SIMD / oracle alignment
//!
//! The reference scalar matmul is correctness-only. The production
//! hot path on CPUs with AVX-512/NEON would port
//! `bitnet.cpp::bitnet-lut-kernels.h` (1170 lines of platform-
//! specific SIMD, /tmp/oracle-bitnet-cpp/BitNet/include/) for the
//! 1.4–2.3× speedup the BitNet-Embedding paper reports. This 4-core
//! / 7.5 GiB box has AVX2 + FMA but no AVX-512 (and Rust's
//! `is_x86_feature_detected!("avxvnni")` reports true on this
//! Intel N150 even though the actual `_mm256_dpbssd_epi32`
//! instruction SIGILLs — see the long-form note in
//! `forward_avx2.rs`). The universal AVX2 + `_mm256_madd_epi16`
//! path is the production kernel.

pub mod forward;
#[cfg(target_arch = "x86_64")]
pub mod forward_avx2;
pub mod slot;

pub use forward::{
    bitlinear_forward, bitlinear_forward_from_f32, bitlinear_forward_packed,
    bitlinear_forward_scalar, quantize_activation_per_token,
};
pub use slot::{BitLinearSlot, BitLinearSlotPacked, BitLinearWeights, BitLinearWeightsPacked};