//! Q4_0 NEON (aarch64) matmul kernel.
//!
//! 32-element blocks, 18-byte layout (2-byte F16 scale + 16-byte nibbles).
//! Strategy mirrors the AVX2 kernel in `avx2.rs`:
//!  1. Load the 16 packed nibble bytes as two `uint8x16_t` lanes.
//!  2. Compute `corrected_dot = sum(nibble * input) - 8 * sum(input)` via
//!     `vdotq_u32` (u8 × i8 → u32 madd-of-pairs) on the interleaved
//!     `low | (high << 8)` representation.
//!  3. Per-block: `acc = vfmaq_f32(d_v * scale_v, dc_v, acc)` then hsum.
//!
//! `vdotq_u32` is the ARMv8.4-A "UDOT" instruction. Most aarch64
//! machines we target (M1/M2, Graviton 3+, Apple A14+, Ampere Altra)
//! implement it. For CPUs without UDOT we fall through to the scalar
//! path in `mod.rs::forward_prequantized`, which already uses
//! `matmul_q4_0_scalar_range`.
//!
//! **Precision contract**: identical to AVX2 (≤ 1 ULP drift vs scalar).
//! Same computation: corrected dot product computed via the
//! `nib_total - 8 * sum_input` algebraic identity so we never materialise
//! `(nibble - 8)` per element.

#![cfg(target_arch = "aarch64")]

use crate::ops::f16_to_f32;

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
    let blocks_per_row = n_in / 32;
    let row_stride = blocks_per_row * 18;

    let w_ptr = weight.as_ptr();
    let iq_ptr = input_q8.as_ptr();
    let sc_ptr = input_scales.as_ptr();
    let out_ptr = output.as_mut_ptr();

    let low_mask = vdupq_n_u8(0x0F);
    for (out_idx, row) in (row_start..row_end).enumerate() {
        let row_off = row * row_stride;
        let mut sum = 0.0f32;

        let mut b = 0;
        while b + 2 <= blocks_per_row {
            let off0 = row_off + b * 18;
            let off1 = row_off + (b + 1) * 18;

            let d_b0 = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off0) as *const u16));
            let d_b1 = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off1) as *const u16));
            let si_b0 = *sc_ptr.add(b);
            let si_b1 = *sc_ptr.add(b + 1);

            let q4_b0 = vld1q_u8(w_ptr.add(off0 + 2));
            let q4_b1 = vld1q_u8(w_ptr.add(off1 + 2));
            let q8_b0_lo = vld1q_s8(iq_ptr.add(b * 32) as *const i8);
            let q8_b0_hi = vld1q_s8(iq_ptr.add(b * 32 + 16) as *const i8);
            let q8_b1_lo = vld1q_s8(iq_ptr.add((b + 1) * 32) as *const i8);
            let q8_b1_hi = vld1q_s8(iq_ptr.add((b + 1) * 32 + 16) as *const i8);

            let dc0 = block_dot(q4_b0, q8_b0_lo, q8_b0_hi, low_mask);
            let dc1 = block_dot(q4_b1, q8_b1_lo, q8_b1_hi, low_mask);
            sum += dc0 as f32 * d_b0 * si_b0;
            sum += dc1 as f32 * d_b1 * si_b1;

            b += 2;
        }

        while b < blocks_per_row {
            let off = row_off + b * 18;
            let d_b = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off) as *const u16));
            let si_b = *sc_ptr.add(b);
            let dc = block_dot(
                vld1q_u8(w_ptr.add(off + 2)),
                vld1q_s8(iq_ptr.add(b * 32) as *const i8),
                vld1q_s8(iq_ptr.add(b * 32 + 16) as *const i8),
                low_mask,
            );
            sum += dc as f32 * d_b * si_b;
            b += 1;
        }

        *out_ptr.add(out_idx) = sum;
    }
}

/// Per-block Q4 × Q8 SIMD dot. Returns the corrected dot product as
/// `i32`:
///   sum((nibble - 8) * input)  [exact i32 arithmetic]
/// computed without ever materialising `(nibble - 8)` per element:
///   = sum(nibble × input) - 8 × sum(input)
///
/// Caller multiplies by `d * scale` (f32) and accumulates in the scalar order
/// used by the reference kernel.
///
#[inline(always)]
unsafe fn block_dot(
    q4_bytes: std::arch::aarch64::uint8x16_t,
    q8_low: std::arch::aarch64::int8x16_t,
    q8_high: std::arch::aarch64::int8x16_t,
    low_mask: std::arch::aarch64::uint8x16_t,
) -> i32 {
    use std::arch::aarch64::*;

    // Split the two packed nibbles. The low and high halves pair with
    // separate 16-byte halves of the Q8 input.
    let lo = vandq_u8(q4_bytes, low_mask);
    let hi = vshrq_n_u8(q4_bytes, 4);

    let zero_i32 = vdupq_n_s32(0);
    let nib_acc = vdotq_s32(
        vdotq_s32(zero_i32, vreinterpretq_s8_u8(lo), q8_low),
        vreinterpretq_s8_u8(hi),
        q8_high,
    );
    let nib_total = vaddvq_s32(nib_acc);

    // sum_input = sum(q8) computed in i32 to keep room for the -8×N bias.
    let sum_total = sum_q8(q8_low, q8_high);

    // The corrected dot: (q4 - 8) · q8 = q4 · q8 - 8 · sum(q8).
    (nib_total as i32) - 8 * (sum_total as i32)
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
