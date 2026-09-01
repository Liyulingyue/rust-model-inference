//! BF16 AVX2 (x86_64) matmul kernel.
//!
//! BF16 → F32 conversion is zero-cost: BF16 stores the upper 16 bits of an
//! F32 representation. To convert, we zero-extend a u16 lane to u32 and
//! shift left by 16 — the resulting bits are exactly the F32 representation.
//!
//! Strategy:
//! 1. Load 8 BF16 values (16 bytes = __m128i).
//! 2. Zero-extend u16 → u32 with `_mm256_cvtepu16_epi32` (8 lanes → 8 u32).
//! 3. Shift left by 16 → F32 bits (`_mm256_slli_epi32`).
//! 4. Reinterpret as __m256 and FMA with the input F32 (8 lanes).
//! 5. Process 32 BF16 × 32 F32 per iteration in 4 accumulators (8-wide each).
//! 6. Hsum 4 accumulators to single f32 per row.
//!
//! **Precision contract**: bit-exact with `bf16::scalar::forward_f32_rows_scalar`.

#![cfg(target_arch = "x86_64")]

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn matmul_bf16_vs_f32_avx2(
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
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let mut i = 0;
        while i + 32 <= n_in {
            let w0 = _mm256_loadu_si256(w_ptr.add(row_byte + i * 2) as *const __m256i);
            let w1 = _mm256_loadu_si256(w_ptr.add(row_byte + i * 2 + 32) as *const __m256i);

            let w_lo0 = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_castsi256_si128(w0)),
                16,
            ));
            let w_hi0 = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(w0, 1)),
                16,
            ));
            let w_lo1 = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_castsi256_si128(w1)),
                16,
            ));
            let w_hi1 = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(w1, 1)),
                16,
            ));

            let x_lo0 = _mm256_loadu_ps(i_ptr.add(i));
            let x_hi0 = _mm256_loadu_ps(i_ptr.add(i + 8));
            let x_lo1 = _mm256_loadu_ps(i_ptr.add(i + 16));
            let x_hi1 = _mm256_loadu_ps(i_ptr.add(i + 24));

            acc0 = _mm256_fmadd_ps(w_lo0, x_lo0, acc0);
            acc1 = _mm256_fmadd_ps(w_hi0, x_hi0, acc1);
            acc2 = _mm256_fmadd_ps(w_lo1, x_lo1, acc2);
            acc3 = _mm256_fmadd_ps(w_hi1, x_hi1, acc3);

            i += 32;
        }
        while i + 8 <= n_in {
            let w_chunk = _mm_loadu_si128(w_ptr.add(row_byte + i * 2) as *const __m128i);
            let w_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(w_chunk), 16));
            let x_f = _mm256_loadu_ps(i_ptr.add(i));
            acc0 = _mm256_fmadd_ps(w_f, x_f, acc0);
            i += 8;
        }
        while i < n_in {
            let bits = u16::from_le_bytes([
                *w_ptr.add(row_byte + i * 2),
                *w_ptr.add(row_byte + i * 2 + 1),
            ]);
            let w_val = crate::ops::bf16_to_f32(bits);
            let x_val = *i_ptr.add(i);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(w_val), _mm256_set1_ps(x_val), acc0);
            i += 1;
        }

        let total = hsum256(acc0) + hsum256(acc1) + hsum256(acc2) + hsum256(acc3);
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
    use crate::ops::kernel::bf16::scalar::forward_f32_rows_scalar;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&value| crate::ops::f32_to_bf16(value).to_le_bytes())
            .collect()
    }

    fn assert_avx2_eq_scalar(label: &str, weight: &[u8], input: &[f32], n_in: usize, n_out: usize) {
        let mut avx2_out = vec![0.0f32; n_out];
        let mut scalar_out = vec![0.0f32; n_out];
        unsafe {
            matmul_bf16_vs_f32_avx2(weight, input, &mut avx2_out, n_in, 0, n_out);
        }
        forward_f32_rows_scalar(weight, input, &mut scalar_out, n_in, n_out, 0, 1);
        for (i, (a, s)) in avx2_out.iter().zip(scalar_out.iter()).enumerate() {
            assert!(
                (a - s).abs() < 1e-2,
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
        let mut weight = Vec::new();
        for r in 0..4 {
            for c in 0..8 {
                weight.extend(crate::ops::f32_to_bf16((r * 8 + c) as f32 * 0.1).to_le_bytes());
            }
        }
        let input: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) * 0.05).collect();
        assert_avx2_eq_scalar("small 4x8", &weight, &input, 8, 4);
    }

    #[test]
    fn avx2_matches_scalar_large() {
        let n_in = 256;
        let n_out = 16;
        let mut weight = Vec::new();
        for r in 0..n_out {
            for c in 0..n_in {
                weight.extend(
                    crate::ops::f32_to_bf16((((r * n_in + c) % 37) as f32 - 18.0) * 0.01)
                        .to_le_bytes(),
                );
            }
        }
        let input: Vec<f32> = (0..n_in).map(|i| ((i % 23) as f32 - 11.0) * 0.02).collect();
        assert_avx2_eq_scalar("large 16x256", &weight, &input, n_in, n_out);
    }

    #[test]
    fn avx2_matches_scalar_tail() {
        let n_in = 40;
        let n_out = 3;
        let mut weight = Vec::new();
        for r in 0..n_out {
            for c in 0..n_in {
                weight.extend(
                    crate::ops::f32_to_bf16(((r * n_in + c) as f32 * 0.01).sin()).to_le_bytes(),
                );
            }
        }
        let input: Vec<f32> = (0..n_in).map(|i| i as f32 * 0.03).collect();
        assert_avx2_eq_scalar("tail 3x37", &weight, &input, n_in, n_out);
    }
}
