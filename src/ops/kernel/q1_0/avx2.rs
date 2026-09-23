//! Q1_0 AVX2 (x86_64) matmul kernel.
//!
//! 128-element blocks, 18-byte layout (2-byte F16 scale + 16-byte bitfield).
//! Each bit: `1 → +1, 0 → -1` (as i8).
//!
//! Strategy:
//! 1. Load 16 bytes of bitfield → expand to 128 × i8 (+1 or -1) via
//!    bit manipulation: for each byte, use LUT to map bit pattern to
//!    8 × i8 values, then `_mm256_maddubs_epi16` with Q8 input.
//! 2. Per 32-element sub-block (4 per Q1_0 block), compute i32 dot.
//! 3. Accumulate as f32 with `d * scale`.

#![cfg(target_arch = "x86_64")]

use crate::ops::f16_to_f32;

/// Expand 1 byte of Q1_0 bitfield into 8 × i8 values (+1 or -1).
///
/// bit=1 → +1 (0x01), bit=0 → -1 (0xFF as i8).
/// Input byte `b`, output 8 bytes: [bit0, bit1, ..., bit7].
///
/// We use a 256-entry LUT: `lut[b]` = 8 bytes where bit i maps to +1/-1.
/// The LUT is 256 × 8 = 2048 bytes, built at compile time.
static Q1_0_BIT_LUT: [u8; 2048] = {
    let mut lut = [0u8; 2048];
    let mut byte_val = 0;
    while byte_val < 256 {
        let mut bit_idx = 0;
        while bit_idx < 8 {
            let bit = (byte_val >> bit_idx) & 1;
            lut[byte_val * 8 + bit_idx] = if bit != 0 { 1 } else { 0xFF };
            bit_idx += 1;
        }
        byte_val += 1;
    }
    lut
};

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn matmul_q1_0_vs_q8_0_avx2(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(n_in % 128, 0);
    let blocks_per_row = n_in / 128;
    let row_stride = blocks_per_row * 18;

    let w_ptr = weight.as_ptr();
    let iq_ptr = input_q8.as_ptr();
    let sc_ptr = input_scales.as_ptr();
    let out_ptr = output.as_mut_ptr();

    let lut_ptr = Q1_0_BIT_LUT.as_ptr();
    let ones_16 = _mm256_set1_epi16(1);

    for row in row_start..row_end {
        let row_off = row * row_stride;
        let mut sum = 0.0f32;

        for block in 0..blocks_per_row {
            let off = row_off + block * 18;
            let d = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off) as *const u16));
            let bits_ptr = w_ptr.add(off + 2);
            let q8_base = block * 128;

            // Process 4 sub-blocks of 32 elements each
            for sub in 0..4 {
                let scale = *sc_ptr.add(block * 4 + sub);
                let ds = d * scale;

                // Expand 4 bytes (32 bits) into 32 × i8 via LUT
                let byte_base = sub * 4;
                let b0 = *bits_ptr.add(byte_base);
                let b1 = *bits_ptr.add(byte_base + 1);
                let b2 = *bits_ptr.add(byte_base + 2);
                let b3 = *bits_ptr.add(byte_base + 3);

                // Load 8 expanded i8 values per byte from LUT
                let q0 = _mm_loadu_si64(lut_ptr.add(b0 as usize * 8) as *const u8);
                let q1 = _mm_loadu_si64(lut_ptr.add(b1 as usize * 8) as *const u8);
                let q2 = _mm_loadu_si64(lut_ptr.add(b2 as usize * 8) as *const u8);
                let q3 = _mm_loadu_si64(lut_ptr.add(b3 as usize * 8) as *const u8);

                // Combine into 32-byte vector: [q0(8), q1(8), q2(8), q3(8)]
                let q01 = _mm_unpacklo_epi64(q0, q1);
                let q23 = _mm_unpacklo_epi64(q2, q3);
                let q32 = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(q01),
                    q23,
                    1,
                );

                // Load 32 Q8 values as i8
                let y = _mm256_loadu_si256(iq_ptr.add(q8_base + sub * 32) as *const __m256i);

                // maddubs: q (treat as u8, but 0xFF=255 works since i8×i8→i16 madd)
                // Actually _mm256_maddubs_epi16 does u8 × i8 → i16.
                // Our +1 is 0x01 (u8=1), our -1 is 0xFF (u8=255).
                // u8(1) × i8(x) = x, u8(255) × i8(x) = -x (as i16, with saturation).
                // But 255 × 127 = 32255 which overflows i16! Need different approach.
                //
                // Alternative: treat both as i8 and use _mm256_madd_epi16 after
                // widening. But that's more complex.
                //
                // Simpler: use sign bit approach. The Q1 values are ±1 as i8.
                // Q8 values are i8. dot = sum(±1 * q8).
                // We can use _mm256_sign_epi8 to flip Q8 signs based on Q1,
                // then sum the result.
                //
                // _mm256_sign_epi8(y, q32): if q32[i] < 0, negate y[i]; if q32[i] == 0, zero y[i].
                // Our q32 has values +1 (0x01) and -1 (0xFF). +1 → keep y, -1 → negate y.
                // This gives us ±y[i], which is exactly what we want!
                let signed_y = _mm256_sign_epi8(y, q32);

                // Now sum 32 × i8 values. Zero-extend to i16, then madd with ones.
                let lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(signed_y));
                let hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(signed_y, 1));
                let lo_sum = _mm256_madd_epi16(lo, ones_16); // 8 × i32
                let hi_sum = _mm256_madd_epi16(hi, ones_16); // 8 × i32

                // Horizontal sum of 16 × i32 → 1 × i32
                let combined = _mm256_add_epi32(lo_sum, hi_sum);
                let dot = hsum_epi32(combined);

                sum += dot as f32 * ds;
            }
        }

        *out_ptr.add(row - row_start) = sum;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_epi32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extracti128_si256(v, 1);
    let lo128 = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(hi128, lo128);
    let shuf = _mm_shuffle_epi32(sum128, 0b_01_00_11_10);
    let sum64 = _mm_add_epi32(sum128, shuf);
    let shuf2 = _mm_shuffle_epi32(sum64, 0b_01_00_10_11);
    let sum32 = _mm_add_epi32(sum64, shuf2);
    _mm_cvtsi128_si32(sum32)
}
