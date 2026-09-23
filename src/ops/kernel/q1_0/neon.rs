//! Q1_0 NEON (aarch64) matmul kernel.
//!
//! 128-element blocks, 18-byte layout (2-byte F16 scale + 16-byte bitfield).
//! Each bit: `1 → +1, 0 → -1` (as i8).
//!
//! Strategy (mirrors AVX2):
//! 1. Expand 4 bytes of bitfield → 32 × i8 (+1 or -1) via LUT.
//! 2. Multiply in i16 so negating a Q8 value of -128 remains representable.
//! 3. Sum the widened products.

#![cfg(target_arch = "aarch64")]

use crate::ops::f16_to_f32;

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

#[target_feature(enable = "neon")]
pub unsafe fn matmul_q1_0_vs_q8_0_neon(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::aarch64::*;

    debug_assert_eq!(n_in % 128, 0);
    let blocks_per_row = n_in / 128;
    let row_stride = blocks_per_row * 18;

    let w_ptr = weight.as_ptr();
    let iq_ptr = input_q8.as_ptr();
    let sc_ptr = input_scales.as_ptr();
    let out_ptr = output.as_mut_ptr();

    let lut_ptr = Q1_0_BIT_LUT.as_ptr();

    for (out_idx, row) in (row_start..row_end).enumerate() {
        let row_off = row * row_stride;
        let mut sum = 0.0f32;

        for block in 0..blocks_per_row {
            let off = row_off + block * 18;
            let d = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off) as *const u16));
            let bits_ptr = w_ptr.add(off + 2);
            let q8_base = block * 128;

            for sub in 0..4 {
                let scale = *sc_ptr.add(block * 4 + sub);
                let ds = d * scale;

                let byte_base = sub * 4;
                let b0 = *bits_ptr.add(byte_base);
                let b1 = *bits_ptr.add(byte_base + 1);
                let b2 = *bits_ptr.add(byte_base + 2);
                let b3 = *bits_ptr.add(byte_base + 3);

                // Load 8 expanded i8 values per byte from LUT
                let q0 = vld1_s8(lut_ptr.add(b0 as usize * 8) as *const i8);
                let q1 = vld1_s8(lut_ptr.add(b1 as usize * 8) as *const i8);
                let q2 = vld1_s8(lut_ptr.add(b2 as usize * 8) as *const i8);
                let q3 = vld1_s8(lut_ptr.add(b3 as usize * 8) as *const i8);

                // Combine into 16-byte vectors
                let q_lo = vcombine_s8(q0, q1); // 16 × i8
                let q_hi = vcombine_s8(q2, q3); // 16 × i8

                // Load 32 Q8 values
                let y_lo = vld1q_s8(iq_ptr.add(q8_base + sub * 32) as *const i8);
                let y_hi = vld1q_s8(iq_ptr.add(q8_base + sub * 32 + 16) as *const i8);

                let dot = sum_signed_i8x32(y_lo, y_hi, q_lo, q_hi);
                sum += dot as f32 * ds;
            }
        }

        *out_ptr.add(out_idx) = sum;
    }
}

#[inline(always)]
unsafe fn sum_signed_i8x32(
    lo: std::arch::aarch64::int8x16_t,
    hi: std::arch::aarch64::int8x16_t,
    sign_lo: std::arch::aarch64::int8x16_t,
    sign_hi: std::arch::aarch64::int8x16_t,
) -> i32 {
    use std::arch::aarch64::*;
    let a = vmull_s8(vget_low_s8(lo), vget_low_s8(sign_lo));
    let b = vmull_s8(vget_high_s8(lo), vget_high_s8(sign_lo));
    let c = vmull_s8(vget_low_s8(hi), vget_low_s8(sign_hi));
    let d = vmull_s8(vget_high_s8(hi), vget_high_s8(sign_hi));
    vaddvq_s16(vaddq_s16(vaddq_s16(a, b), vaddq_s16(c, d))) as i32
}

#[cfg(test)]
mod tests {
    #[test]
    fn signed_q8_min_value_is_widened_before_negation() {
        use std::arch::aarch64::*;
        let input = [-128i8; 32];
        let signs = [-1i8; 32];
        let dot = unsafe {
            super::sum_signed_i8x32(
                vld1q_s8(input.as_ptr()),
                vld1q_s8(input.as_ptr().add(16)),
                vld1q_s8(signs.as_ptr()),
                vld1q_s8(signs.as_ptr().add(16)),
            )
        };
        assert_eq!(dot, 4096);
    }
}
