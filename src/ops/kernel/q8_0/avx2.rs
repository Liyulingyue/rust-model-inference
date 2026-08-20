//! Q8_0 AVX2 (x86_64) matmul kernels.
//!
//! Phase 2.7-final: split from `ops::matmul`. Selected at runtime by
//! `dispatch::matmul_q8_0_quantized_range` and by `q8_0_dot_row` (scalar).

#![cfg(target_arch = "x86_64")]

use crate::ops::{f16_to_f32, hsum_ps};

/// AVX2+FMA single-row Q8_0 dot product.
///
/// Returns `weight[row, :] ⋅ input` for one row. Caller passes
/// `blocks_per_row` and `row_stride` already precomputed (the scalar
/// `q8_0_dot_row` calculates these once).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn q8_0_dot_row_avx2(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    n_in: usize,
    row: usize,
    blocks_per_row: usize,
    row_stride: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let ones = _mm256_set1_epi16(1);
    let row_off = row * row_stride;
    let mut acc = _mm256_setzero_ps();
    for b in 0..blocks_per_row {
        let w_off = row_off + b * 34;
        let d = f16_to_f32(u16::from_le_bytes([
            *weight.as_ptr().add(w_off),
            *weight.as_ptr().add(w_off + 1),
        ])) * *input_scales.as_ptr().add(b);
        let d_v = _mm256_set1_ps(d);
        let qx = _mm256_loadu_si256(weight.as_ptr().add(w_off + 2) as *const __m256i);
        let qy = _mm256_loadu_si256(input_q8.as_ptr().add(b * 32) as *const __m256i);
        let ax = _mm256_sign_epi8(qx, qx);
        let sy = _mm256_sign_epi8(qy, qx);
        let dot = _mm256_maddubs_epi16(ax, sy);
        let summed = _mm256_madd_epi16(ones, dot);
        acc = _mm256_fmadd_ps(d_v, _mm256_cvtepi32_ps(summed), acc);
    }
    hsum_ps(acc)
}