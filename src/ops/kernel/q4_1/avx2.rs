//! Q4_1 AVX2 (x86_64) matmul kernel.
//!
//! 32-element blocks, 20-byte layout (2-byte F16 scale + 2-byte F16 min +
//! 16-byte nibbles). Strategy mirrors `q4_0::avx2`:
//!
//! 1. Extract low + high nibbles as u8 in [0, 15].
//! 2. Interleave to q4_unpacked = [lo[0..16], hi[0..16]] (32 bytes).
//! 3. Use `_mm256_maddubs_epi16` (u8 × i8 → i16 madd of adjacent pairs).
//! 4. Use `_mm256_madd_epi16` (i16 × i16 → i32) with ones to sum pairs.
//! 5. hsum 8 i32 lanes → nib_total (one i32 per block).
//! 6. hsum input separately → sum_input (one i32 per block).
//! 7. Q4_1 dot = nib_total * d * scale + m * input_sum
//!    where input_sum is `scale * sum_input` when caller did NOT pre-compute
//!    it, or the caller-provided value otherwise.
//! 8. Final accumulate uses explicit mul+add (no FMA) to match scalar rounding.
//!
//! **Precision contract**: bit-exact with `q4_1::scalar::matmul_q4_1_scalar_range`
//! when caller provides the same `input_sums`. Without `input_sums`, the AVX2
//! path computes `scale * sum_input` in f32 (matches scalar's `scale * y_sum`
//! since both flow through the same arithmetic).

#![cfg(target_arch = "x86_64")]

use crate::ops::f16_to_f32;

/// Q4_1 × Q8_0 matmul over a row range. AVX2, no FMA.
#[target_feature(enable = "avx2")]
pub unsafe fn matmul_q4_1_vs_q8_0_avx2(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    input_sums: Option<&[f32]>,
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(n_in % 32, 0);
    let blocks_per_row = n_in / 32;
    let row_stride = blocks_per_row * 20;
    let low_mask = _mm256_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);

    let w_ptr = weight.as_ptr();
    let iq_ptr = input_q8.as_ptr();
    let sc_ptr = input_scales.as_ptr();
    let sums_ptr = input_sums.map(|s| s.as_ptr()).unwrap_or(std::ptr::null());
    let out_ptr = output.as_mut_ptr();

    for (out_idx, row) in (row_start..row_end).enumerate() {
        let row_off = row * row_stride;
        let mut acc: f32 = 0.0;

        let mut b = 0;
        while b + 2 <= blocks_per_row {
            let off0 = row_off + b * 20;
            let off1 = row_off + (b + 1) * 20;

            let d_b0 = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off0) as *const u16));
            let m_b0 = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off0 + 2) as *const u16));
            let d_b1 = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off1) as *const u16));
            let m_b1 = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off1 + 2) as *const u16));

            let si_b0 = *sc_ptr.add(b);
            let si_b1 = *sc_ptr.add(b + 1);

            let (nib_b0, sum_b0) = q4_1_block_pair_dot(
                _mm_loadu_si128(w_ptr.add(off0 + 4) as *const __m128i),
                _mm256_loadu_si256(iq_ptr.add(b * 32) as *const __m256i),
                low_mask,
                ones,
            );
            let (nib_b1, sum_b1) = q4_1_block_pair_dot(
                _mm_loadu_si128(w_ptr.add(off1 + 4) as *const __m128i),
                _mm256_loadu_si256(iq_ptr.add((b + 1) * 32) as *const __m256i),
                low_mask,
                ones,
            );

            let input_sum_b0 = if input_sums.is_some() {
                *sums_ptr.add(b)
            } else {
                si_b0 * sum_b0 as f32
            };
            let input_sum_b1 = if input_sums.is_some() {
                *sums_ptr.add(b + 1)
            } else {
                si_b1 * sum_b1 as f32
            };

            acc += (d_b0 * si_b0) * nib_b0 as f32 + m_b0 * input_sum_b0;
            acc += (d_b1 * si_b1) * nib_b1 as f32 + m_b1 * input_sum_b1;
            b += 2;
        }

        while b < blocks_per_row {
            let off = row_off + b * 20;
            let d_b = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off) as *const u16));
            let m_b = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off + 2) as *const u16));
            let si_b = *sc_ptr.add(b);
            let (nib_b, sum_b) = q4_1_block_pair_dot(
                _mm_loadu_si128(w_ptr.add(off + 4) as *const __m128i),
                _mm256_loadu_si256(iq_ptr.add(b * 32) as *const __m256i),
                low_mask,
                ones,
            );
            let input_sum = if input_sums.is_some() {
                *sums_ptr.add(b)
            } else {
                si_b * sum_b as f32
            };
            acc += (d_b * si_b) * nib_b as f32 + m_b * input_sum;
            b += 1;
        }

        *out_ptr.add(out_idx) = acc;
    }
}

/// Per-block Q4_1 × Q8 SIMD dot (without subtracting 8, unlike Q4_0).
/// Returns (nib_dot, sum_q8) as i32 — caller finishes the F32 math.
#[inline(always)]
unsafe fn q4_1_block_pair_dot(
    q4_bytes: std::arch::x86_64::__m128i,
    q8_input: std::arch::x86_64::__m256i,
    low_mask: std::arch::x86_64::__m256i,
    ones: std::arch::x86_64::__m256i,
) -> (i32, i32) {
    use std::arch::x86_64::*;

    let lo = _mm_and_si128(q4_bytes, _mm256_castsi256_si128(low_mask));
    let hi = _mm_and_si128(
        _mm_srli_epi16(q4_bytes, 4),
        _mm256_castsi256_si128(low_mask),
    );
    let q4_unpacked = _mm256_set_m128i(hi, lo);

    let prod16 = _mm256_maddubs_epi16(q4_unpacked, q8_input);
    let prod32 = _mm256_madd_epi16(ones, prod16);

    let y_lo_input = _mm256_castsi256_si128(q8_input);
    let y_hi_input = _mm256_extracti128_si256(q8_input, 1);
    let y_lo16 = _mm256_cvtepi8_epi16(y_lo_input);
    let y_hi16 = _mm256_cvtepi8_epi16(y_hi_input);
    let y_lo32 = _mm256_madd_epi16(ones, y_lo16);
    let y_hi32 = _mm256_madd_epi16(ones, y_hi16);
    let y_sum32 = _mm256_add_epi32(y_lo32, y_hi32);

    (hsum_epi32(prod32), hsum_epi32(y_sum32))
}

#[inline(always)]
unsafe fn hsum_epi32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extracti128_si256(v, 1);
    let lo128 = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(hi128, lo128);
    let t = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let r = _mm_add_epi32(t, _mm_srli_si128(t, 4));
    _mm_cvtsi128_si32(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::kernel::q4_1::scalar::matmul_q4_1_scalar_range;

    fn build_block(scale: f32, min: f32, low: u8, hi: u8) -> Vec<u8> {
        assert!(low < 16 && hi < 16);
        let mut v = Vec::with_capacity(20);
        v.extend_from_slice(&crate::ops::f32_to_f16(scale).to_le_bytes());
        v.extend_from_slice(&crate::ops::f32_to_f16(min).to_le_bytes());
        for _ in 0..16 {
            v.push((hi << 4) | low);
        }
        v
    }

    fn q8_input_linspace() -> Vec<u8> {
        (0..32).map(|i| (i as i8) as u8).collect()
    }

    fn assert_avx2_eq_scalar(
        label: &str,
        weight: &[u8],
        q8: &[u8],
        scales: &[f32],
        sums: Option<&[f32]>,
    ) {
        let n_in = q8.len();
        let n_out = weight.len() / (n_in / 32 * 20);
        let mut avx2_out = vec![0.0f32; n_out];
        let mut scalar_out = vec![0.0f32; n_out];
        unsafe {
            matmul_q4_1_vs_q8_0_avx2(weight, q8, scales, sums, &mut avx2_out, n_in, 0, n_out);
        }
        matmul_q4_1_scalar_range(weight, q8, scales, sums, &mut scalar_out, n_in, n_out, 0, 1);
        for (i, (a, s)) in avx2_out.iter().zip(scalar_out.iter()).enumerate() {
            assert!(
                (a - s).abs() < 1e-3,
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
    fn avx2_matches_scalar_uniform() {
        let mut weight = Vec::new();
        for _ in 0..16 {
            weight.extend(build_block(0.05, -0.1, 7, 9));
        }
        let q8 = q8_input_linspace();
        let scales = vec![0.01f32; 32];
        assert_avx2_eq_scalar("uniform", &weight, &q8, &scales, None);
    }

    #[test]
    fn avx2_matches_scalar_with_sums() {
        let mut weight = Vec::new();
        for i in 0..32 {
            let lo = (i % 16) as u8;
            let hi = ((i + 5) % 16) as u8;
            weight.extend(build_block(0.07, -0.02, lo, hi));
        }
        let q8: Vec<u8> = (0..32 * 32).map(|i| ((i % 127) as i8) as u8).collect();
        let scales: Vec<f32> = (0..32).map(|i| 0.01 + i as f32 * 0.001).collect();
        let sums: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.005).collect();
        assert_avx2_eq_scalar("with-sums", &weight, &q8, &scales, Some(&sums));
    }

    #[test]
    fn avx2_matches_scalar_zero_q8() {
        let mut weight = Vec::new();
        for _ in 0..32 {
            weight.extend(build_block(0.1, 0.05, 3, 12));
        }
        let q8 = vec![0u8; 32 * 32];
        let scales = vec![1.0f32; 32];
        assert_avx2_eq_scalar("zero-q8", &weight, &q8, &scales, None);
    }

    #[test]
    fn avx2_matches_scalar_extreme_q8() {
        let mut weight = Vec::new();
        for _ in 0..32 {
            weight.extend(build_block(0.1, -0.05, 0, 15));
        }
        let q8 = vec![0x7Fu8; 32 * 32];
        let scales = vec![1.0f32; 32];
        assert_avx2_eq_scalar("all-127-q8", &weight, &q8, &scales, None);
    }
}
