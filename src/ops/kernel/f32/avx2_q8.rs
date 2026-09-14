//! F32 × Q8_0 AVX2 (x86_64) matmul kernel.
//!
//! Sister to `bf16::avx2_q8` and `f16::avx2_q8` for the F32 weight
//! path that the new `--quant f32` VibeVoice ASR exports route
//! through.  Weights land as native `f32` (not BF16/F16), so the
//! AVX2 inner loop just streams 8 f32 lanes per register via
//! `_mm256_loadu_ps`; the Q8 input lands in f32 through the same
//! `_mm256_cvtepi8_epi32` + `_mm256_cvtepi32_ps` sequence as the
//! other Q8 kernels.
//!
//! Per-block Q8 scales are applied via a blended f32x8 vector so
//! AVX2 lanes that straddle a 32-element Q8 block boundary don't
//! need to be split.

use std::arch::x86_64::*;

use crate::ops::kernel::simd_avx2::hsum256;

/// Q8 block size — every 32 input bytes share one scale.
const Q8_BLOCK: usize = 32;

/// F32 × Q8_0 packed matmul.  Layout mirrors `scalar::forward_q8_rows_scalar`.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn matmul_f32_vs_q8_avx2(
    weight: &[f32],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    row_start: usize,
    row_end: usize,
) {
    let w_ptr = weight.as_ptr();
    let q_ptr = input_q8.as_ptr();
    let s_ptr = input_scales.as_ptr();
    let out_ptr = output.as_mut_ptr();

    assert!(
        input_q8.len() >= n_in,
        "F32×Q8 AVX2: input_q8 len {} < n_in {n_in}",
        input_q8.len()
    );
    assert!(
        input_scales.len() >= n_in.div_ceil(32),
        "F32×Q8 AVX2: input_scales len {} < required {}",
        input_scales.len(),
        n_in.div_ceil(32)
    );

    let mut out_local = 0usize;
    for row in row_start..row_end {
        let row_off = row * n_in;
        let mut acc = _mm256_setzero_ps();

        let mut col = 0usize;
        while col + 8 <= n_in {
            let block_lo = col / Q8_BLOCK;
            let block_hi = (col + 4) / Q8_BLOCK;
            let scale_lo = _mm_set1_ps(*s_ptr.add(block_lo));
            let scale_hi = _mm_set1_ps(*s_ptr.add(block_hi));
            let scale_128 = _mm_blendv_ps(
                scale_lo,
                scale_hi,
                _mm_castsi128_ps(_mm_set_epi32(-1, -1, -1, -1)),
            );
            let scale = _mm256_insertf128_ps(
                _mm256_castps128_ps256(scale_lo),
                scale_128,
                1,
            );

            let w_f32 = _mm256_loadu_ps(w_ptr.add(row_off + col));

            let q_raw = _mm_loadl_epi64(q_ptr.add(col) as *const __m128i);
            let q_i32 = _mm256_cvtepi8_epi32(q_raw);
            let q_f32 = _mm256_cvtepi32_ps(q_i32);

            acc = _mm256_fmadd_ps(w_f32, _mm256_mul_ps(q_f32, scale), acc);
            col += 8;
        }

        while col < n_in {
            let w_val = *w_ptr.add(row_off + col);
            let q_val = *q_ptr.add(col) as i8 as f32;
            let scale = *s_ptr.add(col / Q8_BLOCK);
            acc = _mm256_fmadd_ps(
                _mm256_set1_ps(w_val * q_val * scale),
                _mm256_set1_ps(1.0),
                acc,
            );
            col += 1;
        }

        let total = hsum256(acc);
        let output_index = if output.len() >= n_out { row } else { out_local };
        *out_ptr.add(output_index) = total;
        out_local += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::matmul_f32_vs_q8_avx2;
    use crate::ops::kernel::f32::scalar::forward_q8_rows_scalar;

    fn assert_avx2_eq_scalar(label: &str, weight: &[f32], input: &[f32], n_in: usize, n_out: usize) {
        let mut avx2_out = vec![0.0f32; n_out];
        let mut scalar_out = vec![0.0f32; n_out];
        let mut input_q8 = vec![0u8; n_in];
        let mut input_scales = vec![0.0f32; n_in.div_ceil(32)];
        crate::ops::quantize_q8_0_into(input, n_in, &mut input_q8, &mut input_scales);
        unsafe {
            matmul_f32_vs_q8_avx2(
                weight,
                &input_q8,
                &input_scales,
                &mut avx2_out,
                n_in,
                n_out,
                0,
                n_out,
            );
        }
        forward_q8_rows_scalar(
            weight,
            &input_q8,
            &input_scales,
            &mut scalar_out,
            n_in,
            n_out,
            0,
            n_out,
        );
        for (i, (a, s)) in avx2_out.iter().zip(scalar_out.iter()).enumerate() {
            let denom = s.abs().max(1.0);
            assert!(
                (a - s).abs() / denom < 1e-2,
                "{label} row {i}: avx2={a} scalar={s} diff={}",
                (a - s).abs()
            );
        }
    }

    #[test]
    fn avx2_matches_scalar_small() {
        let n_in = 16;
        let n_out = 4;
        let weight: Vec<f32> = (0..n_in * n_out)
            .map(|i| ((i as f32 * 0.07).sin()))
            .collect();
        let input: Vec<f32> = (0..n_in).map(|i| (i as f32 - 8.0) * 0.05).collect();
        assert_avx2_eq_scalar("small 4x16", &weight, &input, n_in, n_out);
    }

    #[test]
    fn avx2_matches_scalar_large() {
        let n_in = 256;
        let n_out = 8;
        let weight: Vec<f32> = (0..n_in * n_out)
            .map(|i| ((i * 13 + 7) % 47) as f32 * 0.01 - 0.2)
            .collect();
        let input: Vec<f32> = (0..n_in).map(|i| ((i % 23) as f32 - 11.0) * 0.03).collect();
        assert_avx2_eq_scalar("large 8x256", &weight, &input, n_in, n_out);
    }

    #[test]
    fn avx2_matches_scalar_cross_block() {
        let n_in = 96;
        let n_out = 2;
        let weight: Vec<f32> = (0..n_in * n_out)
            .map(|i| ((i as f32 * 0.013).cos()))
            .collect();
        let input: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.13).sin()).collect();
        assert_avx2_eq_scalar("cross-block 2x96", &weight, &input, n_in, n_out);
    }

    #[test]
    fn avx2_matches_scalar_tail() {
        let n_in = 41;
        let n_out = 3;
        let weight: Vec<f32> = (0..n_in * n_out)
            .map(|i| ((i as f32 * 0.01).sin()))
            .collect();
        let input: Vec<f32> = (0..n_in).map(|i| i as f32 * 0.03).collect();
        assert_avx2_eq_scalar("tail 3x41", &weight, &input, n_in, n_out);
    }
}