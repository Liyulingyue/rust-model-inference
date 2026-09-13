//! F16×F32 AVX2 (x86_64) matmul kernel.
//!
//! Strategy mirrors `bf16/avx2.rs` but with the F16→F32 unpack step
//! replaced by `_mm256_cvtph_ps` (F16C).  Each iteration processes 16 F16
//! weight × 16 F32 input across 2 accumulators (8-wide each), then hsum to
//! one f32 per row.
//!
//! Requires AVX2 + FMA + F16C.  Falls back to scalar / NEON if any are
//! missing (see `super::Kernel::forward`).
//!
//! **Precision contract**: matches `super::F16Kernel::forward_scaled_rows`
//! up to f32 accumulation order.  Tests use a 1e-3 relative tolerance.

#![cfg(target_arch = "x86_64")]

#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
pub unsafe fn matmul_f16_vs_f32_avx2(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(n_in % 8, 0);
    let w_ptr = weight.as_ptr();
    let i_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    let mut out_local = 0usize;
    for row in row_start..row_end {
        let row_byte = row * n_in * 2;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let mut i = 0;
        while i + 16 <= n_in {
            // Load 16 f16 weight bytes (32 bytes = two __m128i), each holding
            // 8 f16 lanes.  Convert each __m128i to __m256 f32 via F16C.
            let w_lo_bytes = _mm_loadu_si128(w_ptr.add(row_byte + i * 2) as *const __m128i);
            let w_hi_bytes = _mm_loadu_si128(w_ptr.add(row_byte + i * 2 + 16) as *const __m128i);
            let w_lo = _mm256_cvtph_ps(w_lo_bytes);
            let w_hi = _mm256_cvtph_ps(w_hi_bytes);

            let x_lo = _mm256_loadu_ps(i_ptr.add(i));
            let x_hi = _mm256_loadu_ps(i_ptr.add(i + 8));

            acc0 = _mm256_fmadd_ps(w_lo, x_lo, acc0);
            acc1 = _mm256_fmadd_ps(w_hi, x_hi, acc1);

            i += 16;
        }
        while i + 8 <= n_in {
            let w_bytes = _mm_loadu_si128(w_ptr.add(row_byte + i * 2) as *const __m128i);
            let w_f = _mm256_cvtph_ps(w_bytes);
            let x_f = _mm256_loadu_ps(i_ptr.add(i));
            acc0 = _mm256_fmadd_ps(w_f, x_f, acc0);
            i += 8;
        }
        while i < n_in {
            let bits = u16::from_le_bytes([
                *w_ptr.add(row_byte + i * 2),
                *w_ptr.add(row_byte + i * 2 + 1),
            ]);
            let w_val = crate::ops::f16_to_f32(bits);
            let x_val = *i_ptr.add(i);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(w_val), _mm256_set1_ps(x_val), acc0);
            i += 1;
        }

        let total = hsum256(acc0) + hsum256(acc1);
        *out_ptr.add(out_local) = total;
        out_local += 1;
    }
}

#[inline(always)]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(hi, lo);
    let sh = _mm_movehdup_ps(s);
    let sums = _mm_add_ps(s, sh);
    let shuf = _mm_movehl_ps(sums, sums);
    let final_sum = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(final_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::kernel::f16::scalar::forward_f16_rows;

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| crate::ops::f32_to_f16(v).to_le_bytes())
            .collect()
    }

    fn assert_avx2_eq_scalar(label: &str, weight: &[u8], input: &[f32], n_in: usize, n_out: usize) {
        let mut avx2_out = vec![0.0f32; n_out];
        let mut scalar_out = vec![0.0f32; n_out];
        unsafe {
            matmul_f16_vs_f32_avx2(weight, input, &mut avx2_out, n_in, 0, n_out);
        }
        forward_f16_rows(weight, input, &mut scalar_out, n_in, n_out, 0, 1);
        for (i, (a, s)) in avx2_out.iter().zip(scalar_out.iter()).enumerate() {
            let denom = s.abs().max(1e-3);
            let rel = (a - s).abs() / denom;
            assert!(
                rel < 1e-3 || (a - s).abs() < 1e-4,
                "{} row {}: avx2={} scalar={} diff={}",
                label,
                i,
                a,
                s,
                (a - s).abs(),
            );
        }
    }

    #[test]
    fn avx2_matches_scalar_small() {
        let n_in = 16;
        let n_out = 4;
        let weight: Vec<u8> = (0..n_in * n_out)
            .flat_map(|i| crate::ops::f32_to_f16((i as f32 * 0.1) - 0.3).to_le_bytes())
            .collect();
        let input: Vec<f32> = (0..n_in).map(|i| (i as f32 - 8.0) * 0.05).collect();
        assert_avx2_eq_scalar("small 4x16", &weight, &input, n_in, n_out);
    }

    #[test]
    fn avx2_matches_scalar_large() {
        let n_in = 256;
        let n_out = 16;
        let weight: Vec<u8> = (0..n_in * n_out)
            .flat_map(|i| {
                crate::ops::f32_to_f16((((i * 13 + 7) % 47) as f32 - 23.0) * 0.01).to_le_bytes()
            })
            .collect();
        let input: Vec<f32> = (0..n_in).map(|i| ((i % 23) as f32 - 11.0) * 0.02).collect();
        assert_avx2_eq_scalar("large 16x256", &weight, &input, n_in, n_out);
    }

    #[test]
    fn avx2_matches_scalar_tail() {
        let n_in = 40;
        let n_out = 3;
        let weight: Vec<u8> = (0..n_in * n_out)
            .flat_map(|i| crate::ops::f32_to_f16((i as f32 * 0.013).sin()).to_le_bytes())
            .collect();
        let input: Vec<f32> = (0..n_in).map(|i| (i as f32 - 20.0) * 0.03).collect();
        assert_avx2_eq_scalar("tail 3x40", &weight, &input, n_in, n_out);
    }
}
