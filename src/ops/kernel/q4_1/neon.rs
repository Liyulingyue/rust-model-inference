//! Q4_1 NEON (aarch64) matmul kernel.

#![cfg(target_arch = "aarch64")]

use crate::ops::f16_to_f32;

#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q4_1_vs_q8_0_neon(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    input_sums: Option<&[f32]>,
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::aarch64::*;

    debug_assert_eq!(n_in % 32, 0);
    let blocks_per_row = n_in / 32;
    let row_stride = blocks_per_row * 20;
    let w_ptr = weight.as_ptr();
    let iq_ptr = input_q8.as_ptr();
    let sc_ptr = input_scales.as_ptr();
    let sums_ptr = input_sums.map_or(std::ptr::null(), |s| s.as_ptr());
    let out_ptr = output.as_mut_ptr();
    let low_mask = vdupq_n_u8(0x0f);

    for (out_idx, row) in (row_start..row_end).enumerate() {
        let row_off = row * row_stride;
        let mut sum = 0.0f32;
        for block in 0..blocks_per_row {
            let off = row_off + block * 20;
            let d = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off) as *const u16));
            let m = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off + 2) as *const u16));
            let scale = *sc_ptr.add(block);
            let q4 = vld1q_u8(w_ptr.add(off + 4));
            let q8_low = vld1q_s8(iq_ptr.add(block * 32) as *const i8);
            let q8_high = vld1q_s8(iq_ptr.add(block * 32 + 16) as *const i8);
            let (dot, q8_sum) = block_dot(q4, q8_low, q8_high, low_mask);
            let input_sum = if sums_ptr.is_null() {
                scale * q8_sum as f32
            } else {
                *sums_ptr.add(block)
            };
            sum += (d * scale) * dot as f32 + m * input_sum;
        }
        *out_ptr.add(out_idx) = sum;
    }
}

/// Return the nibble dot and the sum of all 32 signed Q8 values.
#[inline(always)]
unsafe fn block_dot(
    q4_bytes: std::arch::aarch64::uint8x16_t,
    q8_low: std::arch::aarch64::int8x16_t,
    q8_high: std::arch::aarch64::int8x16_t,
    low_mask: std::arch::aarch64::uint8x16_t,
) -> (i32, i32) {
    use std::arch::aarch64::*;
    let lo = vandq_u8(q4_bytes, low_mask);
    let hi = vshrq_n_u8(q4_bytes, 4);
    let nib_acc = vdotq_s32(
        vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(lo), q8_low),
        vreinterpretq_s8_u8(hi),
        q8_high,
    );
    (vaddvq_s32(nib_acc), sum_q8(q8_low, q8_high))
}

#[inline(always)]
unsafe fn sum_q8(
    q8_low: std::arch::aarch64::int8x16_t,
    q8_high: std::arch::aarch64::int8x16_t,
) -> i32 {
    use std::arch::aarch64::*;
    let low = vpaddlq_s16(vmovl_s8(vget_low_s8(q8_low)));
    let high = vpaddlq_s16(vmovl_s8(vget_high_s8(q8_low)));
    let low2 = vpaddlq_s16(vmovl_s8(vget_low_s8(q8_high)));
    let high2 = vpaddlq_s16(vmovl_s8(vget_high_s8(q8_high)));
    vaddvq_s32(vaddq_s32(vaddq_s32(low, high), vaddq_s32(low2, high2)))
}
