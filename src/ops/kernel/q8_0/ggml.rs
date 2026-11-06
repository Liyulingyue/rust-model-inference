//! ggml-exact Q8_0 × Q8_0 matmul.
//!
//! The production Q8_0 kernels (`dispatch.rs` → `dotprod_nrc4`, or the
//! `neon_nrc1` fallback) reduce each block's integer dot to a scalar before
//! the F32 multiply-add. llama.cpp's `ggml_vec_dot_q8_0_q8_0` keeps a
//! 4-lane accumulator per block pair: it converts the *per-lane* integer
//! sums to F32 and applies `vmlaq_n_f32`, then reduces the two vector
//! accumulators at the very end. The per-lane grouping of the integer dot
//! (native `vdotq_s32`: elements `4i..4i+3`) therefore changes the final
//! rounding.
//!
//! This module reproduces that arithmetic so the Nemotron trunk can opt in
//! without touching the default Q8_0 contract used by every other model.
//!
//! Reference: `ggml/src/ggml-cpu/arch/arm/quants.c`
//! `ggml_vec_dot_q8_0_q8_0`, `#elif defined(__ARM_NEON)` branch.

use crate::ops::f16_to_f32;

/// Weight row stride in bytes: `[f16 scale][32 int8]` per block.
const BLOCK_BYTES: usize = 34;

/// ggml-exact Q8_0 × Q8_0 matmul over `[row_start, row_end)`.
///
/// `weight` is row-major Q8_0 (`n_in` must be a multiple of 32) and
/// `input_q8` / `input_scales` are the caller's quantized activation.
pub fn matmul_q8_0_vs_q8_0_ggml(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if crate::ops::has_neon() {
            // SAFETY: NEON is available (runtime probe); every load is
            // bounded by the row range and `n_in`.
            unsafe {
                neon::matmul_q8_0_vs_q8_0_ggml(
                    weight,
                    input_q8,
                    input_scales,
                    output,
                    n_in,
                    row_start,
                    row_end,
                );
            }
            return;
        }
    }
    scalar::matmul_q8_0_vs_q8_0_ggml(
        weight,
        input_q8,
        input_scales,
        output,
        n_in,
        row_start,
        row_end,
    );
}

/// Scalar transcription of the ggml NEON branch. It keeps the same
/// two-block interleaving and the same per-lane (4-element) integer
/// grouping so the F32 rounding order is identical.
pub mod scalar {
    use super::{f16_to_f32, BLOCK_BYTES};

    pub fn matmul_q8_0_vs_q8_0_ggml(
        weight: &[u8],
        input_q8: &[u8],
        input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        row_start: usize,
        row_end: usize,
    ) {
        let blocks = n_in / 32;
        let stride = blocks * BLOCK_BYTES;
        for (out_idx, row) in (row_start..row_end).enumerate() {
            let row_off = row * stride;
            // Four lanes per accumulator, two block accumulators, exactly
            // like `sumv0` / `sumv1` in the C kernel.
            let mut sumv0 = [0.0f32; 4];
            let mut sumv1 = [0.0f32; 4];
            let mut ib = 0;
            while ib + 1 < blocks {
                let off0 = row_off + ib * BLOCK_BYTES;
                let off1 = off0 + BLOCK_BYTES;
                let d0 = f16_to_f32(u16::from_le_bytes([weight[off0], weight[off0 + 1]]))
                    * input_scales[ib];
                let d1 = f16_to_f32(u16::from_le_bytes([weight[off1], weight[off1 + 1]]))
                    * input_scales[ib + 1];
                for lane in 0..4 {
                    let mut dot0 = 0i32;
                    let mut dot1 = 0i32;
                    for k in 0..4 {
                        dot0 += (weight[off0 + 2 + lane * 4 + k] as i8 as i32)
                            * (input_q8[ib * 32 + lane * 4 + k] as i8 as i32);
                        dot1 += (weight[off1 + 2 + lane * 4 + k] as i8 as i32)
                            * (input_q8[(ib + 1) * 32 + lane * 4 + k] as i8 as i32);
                    }
                    sumv0[lane] = (dot0 as f32).mul_add(d0, sumv0[lane]);
                    sumv1[lane] = (dot1 as f32).mul_add(d1, sumv1[lane]);
                }
                ib += 2;
            }
            // `sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1)`; `vaddvq_f32`
            // lowers to `(r0 + r1) + (r2 + r3)`.
            let mut sumf = ((sumv0[0] + sumv0[1]) + (sumv0[2] + sumv0[3]))
                + ((sumv1[0] + sumv1[1]) + (sumv1[2] + sumv1[3]));
            // Odd tail block: ggml's scalar loop over all 32 elements.
            while ib < blocks {
                let off = row_off + ib * BLOCK_BYTES;
                let mut sumi = 0i32;
                for j in 0..32 {
                    sumi += (weight[off + 2 + j] as i8 as i32)
                        * (input_q8[ib * 32 + j] as i8 as i32);
                }
                let d = f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]))
                    * input_scales[ib];
                sumf += sumi as f32 * d;
                ib += 1;
            }
            output[out_idx] = sumf;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use super::{f16_to_f32, BLOCK_BYTES};
    use std::arch::aarch64::*;

    /// `ggml_vdotq_s32` when dotprod is available (native `sdot`).
    #[target_feature(enable = "neon,dotprod")]
    #[inline]
    unsafe fn dot_lanes_dotprod(a: int8x16_t, b: int8x16_t) -> int32x4_t {
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

    /// `ggml_vdotq_s32` without dotprod: same total, different per-lane
    /// grouping. Kept so a non-dotprod build matches a non-dotprod oracle.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn dot_lanes_fallback(a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
        vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))
    }

    #[target_feature(enable = "neon,dotprod")]
    unsafe fn matmul_dotprod(
        weight: &[u8],
        input_q8: &[u8],
        input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        row_start: usize,
        row_end: usize,
    ) {
        let blocks = n_in / 32;
        let stride = blocks * BLOCK_BYTES;
        for (out_idx, row) in (row_start..row_end).enumerate() {
            let row_off = row * stride;
            let mut sumv0 = vdupq_n_f32(0.0);
            let mut sumv1 = vdupq_n_f32(0.0);
            let mut ib = 0;
            while ib + 1 < blocks {
                let off0 = row_off + ib * BLOCK_BYTES;
                let off1 = off0 + BLOCK_BYTES;
                let x0_0 = vld1q_s8(weight.as_ptr().add(off0 + 2) as *const i8);
                let x0_1 = vld1q_s8(weight.as_ptr().add(off0 + 18) as *const i8);
                let x1_0 = vld1q_s8(weight.as_ptr().add(off1 + 2) as *const i8);
                let x1_1 = vld1q_s8(weight.as_ptr().add(off1 + 18) as *const i8);

                let y0_0 = vld1q_s8(input_q8.as_ptr().add(ib * 32) as *const i8);
                let y0_1 = vld1q_s8(input_q8.as_ptr().add(ib * 32 + 16) as *const i8);
                let y1_0 = vld1q_s8(input_q8.as_ptr().add((ib + 1) * 32) as *const i8);
                let y1_1 = vld1q_s8(input_q8.as_ptr().add((ib + 1) * 32 + 16) as *const i8);

                let d0 = f16_to_f32(u16::from_le_bytes([weight[off0], weight[off0 + 1]]))
                    * input_scales[ib];
                let d1 = f16_to_f32(u16::from_le_bytes([weight[off1], weight[off1 + 1]]))
                    * input_scales[ib + 1];

                sumv0 = vmlaq_n_f32(
                    sumv0,
                    vcvtq_f32_s32(vaddq_s32(
                        dot_lanes_dotprod(x0_0, y0_0),
                        dot_lanes_dotprod(x0_1, y0_1),
                    )),
                    d0,
                );
                sumv1 = vmlaq_n_f32(
                    sumv1,
                    vcvtq_f32_s32(vaddq_s32(
                        dot_lanes_dotprod(x1_0, y1_0),
                        dot_lanes_dotprod(x1_1, y1_1),
                    )),
                    d1,
                );
                ib += 2;
            }
            let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
            while ib < blocks {
                let off = row_off + ib * BLOCK_BYTES;
                let mut sumi = 0i32;
                for j in 0..32 {
                    sumi += (weight[off + 2 + j] as i8 as i32)
                        * (input_q8[ib * 32 + j] as i8 as i32);
                }
                let d = f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]))
                    * input_scales[ib];
                sumf += sumi as f32 * d;
                ib += 1;
            }
            output[out_idx] = sumf;
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn matmul_neon(
        weight: &[u8],
        input_q8: &[u8],
        input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        row_start: usize,
        row_end: usize,
    ) {
        let blocks = n_in / 32;
        let stride = blocks * BLOCK_BYTES;
        for (out_idx, row) in (row_start..row_end).enumerate() {
            let row_off = row * stride;
            let mut sumv0 = vdupq_n_f32(0.0);
            let mut sumv1 = vdupq_n_f32(0.0);
            let mut ib = 0;
            while ib + 1 < blocks {
                let off0 = row_off + ib * BLOCK_BYTES;
                let off1 = off0 + BLOCK_BYTES;
                let x0_0 = vld1q_s8(weight.as_ptr().add(off0 + 2) as *const i8);
                let x0_1 = vld1q_s8(weight.as_ptr().add(off0 + 18) as *const i8);
                let x1_0 = vld1q_s8(weight.as_ptr().add(off1 + 2) as *const i8);
                let x1_1 = vld1q_s8(weight.as_ptr().add(off1 + 18) as *const i8);

                let y0_0 = vld1q_s8(input_q8.as_ptr().add(ib * 32) as *const i8);
                let y0_1 = vld1q_s8(input_q8.as_ptr().add(ib * 32 + 16) as *const i8);
                let y1_0 = vld1q_s8(input_q8.as_ptr().add((ib + 1) * 32) as *const i8);
                let y1_1 = vld1q_s8(input_q8.as_ptr().add((ib + 1) * 32 + 16) as *const i8);

                let d0 = f16_to_f32(u16::from_le_bytes([weight[off0], weight[off0 + 1]]))
                    * input_scales[ib];
                let d1 = f16_to_f32(u16::from_le_bytes([weight[off1], weight[off1 + 1]]))
                    * input_scales[ib + 1];

                sumv0 = vmlaq_n_f32(
                    sumv0,
                    vcvtq_f32_s32(vaddq_s32(
                        dot_lanes_fallback(x0_0, y0_0),
                        dot_lanes_fallback(x0_1, y0_1),
                    )),
                    d0,
                );
                sumv1 = vmlaq_n_f32(
                    sumv1,
                    vcvtq_f32_s32(vaddq_s32(
                        dot_lanes_fallback(x1_0, y1_0),
                        dot_lanes_fallback(x1_1, y1_1),
                    )),
                    d1,
                );
                ib += 2;
            }
            let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
            while ib < blocks {
                let off = row_off + ib * BLOCK_BYTES;
                let mut sumi = 0i32;
                for j in 0..32 {
                    sumi += (weight[off + 2 + j] as i8 as i32)
                        * (input_q8[ib * 32 + j] as i8 as i32);
                }
                let d = f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]))
                    * input_scales[ib];
                sumf += sumi as f32 * d;
                ib += 1;
            }
            output[out_idx] = sumf;
        }
    }

    /// SAFETY: caller guarantees NEON is available.
    pub unsafe fn matmul_q8_0_vs_q8_0_ggml(
        weight: &[u8],
        input_q8: &[u8],
        input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        row_start: usize,
        row_end: usize,
    ) {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            matmul_dotprod(
                weight,
                input_q8,
                input_scales,
                output,
                n_in,
                row_start,
                row_end,
            );
        } else {
            matmul_neon(
                weight,
                input_q8,
                input_scales,
                output,
                n_in,
                row_start,
                row_end,
            );
        }
    }
}
