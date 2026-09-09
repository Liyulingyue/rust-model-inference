//! RoPE (Rotary Position Embedding) operations.
//!
//! Module split (Phase 1):
//! - `neox`      — Neox-style rotation (lo/hi halves swapped): single-position API
//!   + AVX2/NEON/scalar apply kernels. Used by llama, qwen3, qwen35, lfm*, gemma4,
//!   qwen3-vision, etc.
//! - `norm`      — GGML "normal" (interleaved-pair) rotation. Used by llama-arch
//!   GGUF (after the GGML converter permutes rotate_half).
//! - `partial`   — Neox rotation on only the first `n_rot` dims of each head.
//!   Used by Spark 2.5's per-layer heterogeneous RoPE.
//! - `mrope`     — Multimodal RoPE (Qwen3-VL): rotates across 4 axes (T/H/W/E) by
//!   section, with both halves-rotation and interleaved-pair layouts.
//! - `sleef_math` — SLEEF-style double-float sin/cos kernels used by dots.tts to
//!   match libsystem_sleef.dylib bit-for-bit on macOS.
//! - `sleef_rope` — Public Neox RoPE entry points built on `sleef_math`.

#[cfg(target_os = "macos")]
extern "C" {
    fn __sincosf(value: f32, sin: *mut f32, cos: *mut f32);
}

mod mrope;
mod neox;
mod norm;
mod partial;
mod sleef_math;
mod sleef_rope;

pub use mrope::{rope_mrope, rope_mrope_interleaved, rope_vision};
pub use neox::{rope_neox, rope_sin_cos};
pub use norm::rope_norm;
pub use partial::rope_neox_partial;

pub(crate) use sleef_math::rope_sin_cos_sleef;
pub(crate) use sleef_rope::{
    rope_neox_sleef, rope_neox_sleef_rows, rope_sin_cos_sleef_table_with_threads,
};

#[cfg(test)]
mod tests;
