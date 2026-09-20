//! AArch64 dot-product kernel for Q4_0 weights and Q8_0 activations.

#![cfg(target_arch = "aarch64")]

#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q4_0_vs_q8_0_neon(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::aarch64::*;

    debug_assert_eq!(n_in % 32, 0);
    let blocks = n_in / 32;
    let row_stride = blocks * 18;
    let mask = vdupq_n_u8(0x0f);
    let bias = vdupq_n_u8(8);

    for row in row_start..row_end {
        let mut sum = 0.0f32;
        for block in 0..blocks {
            let offset = row * row_stride + block * 18;
            let d =
                crate::ops::f16_to_f32(u16::from_le_bytes([weight[offset], weight[offset + 1]]));
            let packed = vld1q_u8(weight.as_ptr().add(offset + 2));
            let low = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(packed, mask), bias));
            let high = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(packed, 4), bias));
            let input = input_q8.as_ptr().add(block * 32).cast::<i8>();
            let dot = vdotq_s32(vdupq_n_s32(0), low, vld1q_s8(input));
            let dot = vdotq_s32(dot, high, vld1q_s8(input.add(16)));
            sum += vaddvq_s32(dot) as f32 * d * input_scales[block];
        }
        output[row] = sum;
    }
}
