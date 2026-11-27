//! Strided weighted value reduction for attention output computation.
//!
//! Given attention weights (`scores`) and a strided value matrix, computes:
//!   `output[dim] = sum_{token} scores[token] * values[token][dim]`
//!
//! The value matrix is laid out as `[n_tokens][row_stride]` where each
//! token's value vector starts at `base + token * row_stride + head_offset`
//! and has `head_width` contiguous elements.
//!
//! Internally dispatches to NEON (aarch64) or AVX2 (x86_64) with a scalar
//! fallback. Callers never see `target_arch` or SIMD types.

use super::super::has_avx2_fma;
use super::super::has_neon;

/// Compute `output[dim] = sum_{token} scores[token] * values[token][dim]`
/// for `dim` in `0..head_width`.
///
/// - `values`: flat slice containing the value matrix
/// - `scores`: attention weights, length `n_tokens`
/// - `output`: output buffer, length `head_width` (will be zero-filled)
/// - `base`: byte offset (in elements) to the first token's value vector
/// - `row_stride`: element stride between consecutive tokens
/// - `head_offset`: element offset within each row to the head's value
/// - `n_tokens`: number of tokens (rows) to reduce
/// - `head_width`: number of contiguous value elements per token
#[inline]
pub fn attention_value_reduce(
    values: &[f32],
    scores: &[f32],
    output: &mut [f32],
    base: usize,
    row_stride: usize,
    head_offset: usize,
    n_tokens: usize,
    head_width: usize,
) {
    debug_assert!(output.len() >= head_width);
    debug_assert!(scores.len() >= n_tokens);
    output[..head_width].fill(0.0);

    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            unsafe {
                attention_value_reduce_neon(
                    values,
                    scores,
                    output,
                    base,
                    row_stride,
                    head_offset,
                    n_tokens,
                    head_width,
                );
            }
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe {
                attention_value_reduce_avx2(
                    values,
                    scores,
                    output,
                    base,
                    row_stride,
                    head_offset,
                    n_tokens,
                    head_width,
                );
            }
            return;
        }
    }
    attention_value_reduce_scalar(
        values,
        scores,
        output,
        base,
        row_stride,
        head_offset,
        n_tokens,
        head_width,
    );
}

#[inline(always)]
fn attention_value_reduce_scalar(
    values: &[f32],
    scores: &[f32],
    output: &mut [f32],
    base: usize,
    row_stride: usize,
    head_offset: usize,
    n_tokens: usize,
    head_width: usize,
) {
    for token in 0..n_tokens {
        let weight = scores[token];
        let start = base + token * row_stride + head_offset;
        for (out, &v) in output[..head_width]
            .iter_mut()
            .zip(&values[start..start + head_width])
        {
            *out += weight * v;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn attention_value_reduce_neon(
    values: &[f32],
    scores: &[f32],
    output: &mut [f32],
    base: usize,
    row_stride: usize,
    head_offset: usize,
    n_tokens: usize,
    head_width: usize,
) {
    use std::arch::aarch64::*;
    let mut dim = 0;
    while dim + 4 <= head_width {
        let mut acc = vdupq_n_f32(0.0);
        for token in 0..n_tokens {
            let start = base + token * row_stride + head_offset + dim;
            let weight = vdupq_n_f32(scores[token]);
            let v = vld1q_f32(values.as_ptr().add(start));
            acc = vfmaq_f32(acc, weight, v);
        }
        vst1q_f32(output.as_mut_ptr().add(dim), acc);
        dim += 4;
    }
    while dim < head_width {
        let mut sum = 0.0f32;
        for token in 0..n_tokens {
            let start = base + token * row_stride + head_offset + dim;
            sum += scores[token] * *values.get_unchecked(start);
        }
        *output.get_unchecked_mut(dim) = sum;
        dim += 1;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn attention_value_reduce_avx2(
    values: &[f32],
    scores: &[f32],
    output: &mut [f32],
    base: usize,
    row_stride: usize,
    head_offset: usize,
    n_tokens: usize,
    head_width: usize,
) {
    use std::arch::x86_64::*;
    let mut dim = 0;
    while dim + 8 <= head_width {
        let mut acc = _mm256_setzero_ps();
        for token in 0..n_tokens {
            let start = base + token * row_stride + head_offset + dim;
            let weight = _mm256_set1_ps(scores[token]);
            let v = _mm256_loadu_ps(values.as_ptr().add(start));
            acc = _mm256_fmadd_ps(weight, v, acc);
        }
        _mm256_storeu_ps(output.as_mut_ptr().add(dim), acc);
        dim += 8;
    }
    while dim < head_width {
        let mut sum = 0.0f32;
        for token in 0..n_tokens {
            let start = base + token * row_stride + head_offset + dim;
            sum += scores[token] * *values.get_unchecked(start);
        }
        *output.get_unchecked_mut(dim) = sum;
        dim += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_and_simd_match_single_token() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let scores = [1.0];
        let mut out = vec![0.0f32; 4];
        attention_value_reduce(&values, &scores, &mut out, 0, 4, 0, 1, 4);
        assert_eq!(&out[..4], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn scalar_and_simd_match_multi_token() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let scores = [0.25, 0.75];
        let mut out = vec![0.0f32; 4];
        attention_value_reduce(&values, &scores, &mut out, 0, 4, 0, 2, 4);
        let expected = [
            0.25 * 1.0 + 0.75 * 5.0,
            0.25 * 2.0 + 0.75 * 6.0,
            0.25 * 3.0 + 0.75 * 7.0,
            0.25 * 4.0 + 0.75 * 8.0,
        ];
        for (a, &b) in out[..4].iter().zip(&expected) {
            assert!((a - b).abs() < 1e-5, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn handles_head_offset_and_stride() {
        let values = [0.0, 0.0, 10.0, 20.0, 0.0, 0.0, 30.0, 40.0];
        let scores = [0.5, 0.5];
        let mut out = vec![0.0f32; 2];
        attention_value_reduce(&values, &scores, &mut out, 0, 4, 2, 2, 2);
        assert!((out[0] - 20.0).abs() < 1e-5);
        assert!((out[1] - 30.0).abs() < 1e-5);
    }

    #[test]
    fn handles_non_power_of_2_width() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let scores = [0.5, 0.5];
        let mut out = vec![0.0f32; 3];
        attention_value_reduce(&values, &scores, &mut out, 0, 3, 0, 2, 3);
        assert!((out[0] - 2.5).abs() < 1e-5);
        assert!((out[1] - 3.5).abs() < 1e-5);
        assert!((out[2] - 4.5).abs() < 1e-5);
    }
}
