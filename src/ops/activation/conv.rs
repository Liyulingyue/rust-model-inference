//! Fused `conv1d + bias + residual + silu` kernel (scalar reference).
//!
//! The fused op is small enough that no SIMD path is provided today; the
//! scalar loop in [`conv1d_silu`] is the hot path used by the TTS audio
//! pipeline.

use super::silu;

#[inline(always)]
pub fn conv1d_silu(
    kernel: &[f32],
    state: &[f32],
    input: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    let out_len = output.len();
    let kernel_len = kernel.len();
    debug_assert!(input.len() >= out_len + kernel_len - 1);
    for o in 0..out_len {
        let mut acc = bias.map_or(0.0, |bias| bias[o]);
        for k in 0..kernel_len {
            acc += input[o + k] * kernel[k];
        }
        acc += state[o] * kernel[0];
        output[o] = silu(acc);
    }
}