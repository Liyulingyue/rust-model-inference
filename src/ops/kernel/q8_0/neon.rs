//! Q8_0 NEON (aarch64) matmul kernels.
//!
//! Phase 2.7-final: split from `ops::matmul`. Selected at runtime by
//! `dispatch::matmul_q8_0_quantized_range` via `has_neon()`, and called
//! directly by `scalar::q8_0_dot_row` for the single-row variant.

#![cfg(target_arch = "aarch64")]

use crate::ops::f16_to_f32;

/// NEON i8×i8 dot product of 32-element vectors (scalar result).
#[target_feature(enable = "neon")]
pub unsafe fn dot_i8x32_neon(a: *const u8, b: *const u8) -> i32 {
    use std::arch::aarch64::*;
    let a0 = vld1q_s8(a as *const i8);
    let b0 = vld1q_s8(b as *const i8);
    let a1 = vld1q_s8(a.add(16) as *const i8);
    let b1 = vld1q_s8(b.add(16) as *const i8);
    let p0 = vaddq_s32(
        vpaddlq_s16(vmull_s8(vget_low_s8(a0), vget_low_s8(b0))),
        vpaddlq_s16(vmull_high_s8(a0, b0)),
    );
    let p1 = vaddq_s32(
        vpaddlq_s16(vmull_s8(vget_low_s8(a1), vget_low_s8(b1))),
        vpaddlq_s16(vmull_high_s8(a1, b1)),
    );
    vaddvq_s32(vaddq_s32(p0, p1))
}

/// NEON i8×i8 dot product returning 4-lane i32x4 (for fused multiply-add).
#[target_feature(enable = "neon")]
pub unsafe fn dot_i8x32_lanes_neon(a: *const u8, b: *const u8) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;
    let a0 = vld1q_s8(a as *const i8);
    let b0 = vld1q_s8(b as *const i8);
    let a1 = vld1q_s8(a.add(16) as *const i8);
    let b1 = vld1q_s8(b.add(16) as *const i8);
    let dot16 = |a: int8x16_t, b: int8x16_t| {
        vpaddq_s32(
            vpaddlq_s16(vmull_s8(vget_low_s8(a), vget_low_s8(b))),
            vpaddlq_s16(vmull_high_s8(a, b)),
        )
    };
    vaddq_s32(dot16(a0, b0), dot16(a1, b1))
}

/// NEON Q8_0 × Q8_0 matmul over a row range.
#[target_feature(enable = "neon")]
pub unsafe fn matmul_q8_0_vs_q8_0_neon(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    let blocks = n_in / 32;
    let stride = blocks * 34;
    for (out_idx, row) in (row_start..row_end).enumerate() {
        let mut sum = 0.0f32;
        for block in 0..blocks {
            let off = row * stride + block * 34;
            let wd = f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
            let dot = dot_i8x32_neon(
                weight.as_ptr().add(off + 2),
                input_q8.as_ptr().add(block * 32),
            );
            sum = (dot as f32).mul_add(wd * input_scales[block], sum);
        }
        output[out_idx] = sum;
    }
}

/// NEON Q8_0 × Q8_0 matmul, NRC=1 (single row, 2-block fused FMA).
#[target_feature(enable = "neon")]
pub unsafe fn matmul_q8_0_vs_q8_0_neon_nrc1(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::aarch64::*;
    let blocks = n_in / 32;
    let stride = blocks * 34;
    for (out_idx, row) in (row_start..row_end).enumerate() {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut block = 0;
        while block + 1 < blocks {
            let off0 = row * stride + block * 34;
            let off1 = off0 + 34;
            let scale0 = f16_to_f32(u16::from_le_bytes([weight[off0], weight[off0 + 1]]))
                * input_scales[block];
            let scale1 = f16_to_f32(u16::from_le_bytes([weight[off1], weight[off1 + 1]]))
                * input_scales[block + 1];
            sum0 = vfmaq_n_f32(
                sum0,
                vcvtq_f32_s32(dot_i8x32_lanes_neon(
                    weight.as_ptr().add(off0 + 2),
                    input_q8.as_ptr().add(block * 32),
                )),
                scale0,
            );
            sum1 = vfmaq_n_f32(
                sum1,
                vcvtq_f32_s32(dot_i8x32_lanes_neon(
                    weight.as_ptr().add(off1 + 2),
                    input_q8.as_ptr().add((block + 1) * 32),
                )),
                scale1,
            );
            block += 2;
        }
        let mut sum = vaddvq_f32(sum0) + vaddvq_f32(sum1);
        if block < blocks {
            let off = row * stride + block * 34;
            let scale = f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]))
                * input_scales[block];
            sum += dot_i8x32_neon(
                weight.as_ptr().add(off + 2),
                input_q8.as_ptr().add(block * 32),
            ) as f32
                * scale;
        }
        output[out_idx] = sum;
    }
}

#[target_feature(enable = "neon,dotprod")]
#[inline]
unsafe fn dot_i8x16_dotprod(
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;
    let mut sum = vdupq_n_s32(0);
    std::arch::asm!(
        "sdot {sum:v}.4s, {a:v}.16b, {b:v}.16b",
        sum = inout(vreg) sum,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nomem, nostack, pure),
    );
    sum
}

/// Dot-product Q8_0 × Q8_0 matmul with four output rows in flight.
///
/// `dotprod` is optional on aarch64, so callers must select this only after
/// `is_aarch64_feature_detected!("dotprod")` succeeds.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q8_0_vs_q8_0_dotprod_nrc4(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::aarch64::*;

    let blocks = n_in / 32;
    let stride = blocks * 34;
    let full4 = (row_end - row_start) / 4;
    for tile in 0..full4 {
        let row = row_start + tile * 4;
        let offsets = [
            row * stride,
            (row + 1) * stride,
            (row + 2) * stride,
            (row + 3) * stride,
        ];
        let mut sum = vdupq_n_f32(0.0);
        for block in 0..blocks {
            let input = input_q8.as_ptr().add(block * 32);
            let input_lo = vld1q_s8(input as *const i8);
            let input_hi = vld1q_s8(input.add(16) as *const i8);
            let mut dots = [0i32; 4];
            let mut scales = [0.0f32; 4];
            for (row_offset, offset) in offsets.iter().enumerate() {
                let weight = weight.as_ptr().add(*offset + block * 34);
                dots[row_offset] = vaddvq_s32(vaddq_s32(
                    dot_i8x16_dotprod(vld1q_s8(weight.add(2) as *const i8), input_lo),
                    dot_i8x16_dotprod(vld1q_s8(weight.add(18) as *const i8), input_hi),
                ));
                scales[row_offset] =
                    f16_to_f32(u16::from_le_bytes([*weight, *weight.add(1)])) * input_scales[block];
            }
            sum = vfmaq_f32(
                sum,
                vcvtq_f32_s32(vld1q_s32(dots.as_ptr())),
                vld1q_f32(scales.as_ptr()),
            );
        }
        let out = tile * 4;
        vst1q_f32(output.as_mut_ptr().add(out), sum);
    }

    let tail_start = row_start + full4 * 4;
    if tail_start < row_end {
        matmul_q8_0_vs_q8_0_neon(
            weight,
            input_q8,
            input_scales,
            &mut output[full4 * 4..],
            n_in,
            tail_start,
            row_end,
        );
    }
}
