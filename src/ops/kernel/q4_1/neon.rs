//! Q4_1 NEON (aarch64) matmul kernel.
//!
//! 32-element blocks, 20-byte layout (2-byte F16 scale + 2-byte F16 min
//! + 16-byte nibbles). Strategy mirrors `q4_1::avx2`:
//!
//! 1. Extract low + high nibbles, interleave to `[lo, hi]` (32 bytes).
//! 2. Use `vdotq_u32` to compute `nib_total = sum(nibble × input)` in i32.
//! 3. Compute `sum_input` via `vpaddlq_s8` → `vpadalq_s16` widening
//!    pair-add chain.
//! 4. Q4_1 dot = `nib_total * d * scale + m * input_sum`
//!    (no FMA — matches scalar rounding exactly).
//!
//! `vdotq_u32` is the ARMv8.4-A "UDOT" instruction. Most aarch64
//! machines we target (M1/M2, Graviton 3+, Apple A14+, Ampere Altra)
//! implement it. Older cores fall through to the scalar baseline in
//! `q4_1::mod::forward_prequantized` / `forward_prepared`.
//!
//! **Precision contract**: bit-exact with `q4_1::scalar` when caller
//! provides the same `input_sums`. Without `input_sums`, both AVX2 and
//! NEON paths compute `scale * sum_input` in f32 — they agree with the
//! scalar code to within 1 ULP.

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
    let sums_ptr = input_sums.map(|s| s.as_ptr()).unwrap_or(std::ptr::null());
    let out_ptr = output.as_mut_ptr();

    let low_mask = vdupq_n_u8(0x0F);

    for (out_idx, row) in (row_start..row_end).enumerate() {
        let row_off = row * row_stride;
        let mut acc = vdupq_n_f32(0.0);

        let mut b = 0;
        while b + 2 <= blocks_per_row {
            let off0 = row_off + b * 20;
            let off1 = row_off + (b + 1) * 20;

            let d_b0 = f16_to_f32(u16::from_le_bytes(
                std::ptr::read_unaligned(w_ptr.add(off0) as *const u16),
            ));
            let m_b0 = f16_to_f32(u16::from_le_bytes(
                std::ptr::read_unaligned(w_ptr.add(off0 + 2) as *const u16),
            ));
            let d_b1 = f16_to_f32(u16::from_le_bytes(
                std::ptr::read_unaligned(w_ptr.add(off1) as *const u16),
            ));
            let m_b1 = f16_to_f32(u16::from_le_bytes(
                std::ptr::read_unaligned(w_ptr.add(off1 + 2) as *const u16),
            ));
            let si_b0 = *sc_ptr.add(b);
            let si_b1 = *sc_ptr.add(b + 1);

            let q4_b0 = vld1q_u8(w_ptr.add(off0 + 4));
            let q4_b1 = vld1q_u8(w_ptr.add(off1 + 4));
            let q8_b0 = vld1q_s8(iq_ptr.add(b * 32));
            let q8_b1 = vld1q_s8(iq_ptr.add((b + 1) * 32));

            let (dc0, dc1) = block_pair_dot(q4_b0, q8_b0, q4_b1, q8_b1, low_mask);
            let ds0 = d_b0 * si_b0;
            let ds1 = d_b1 * si_b1;

            // m * input_sum: prefer caller-provided pre-summed input_sums
            // (saves recomputing the i32 → f32 sum_total inside this loop).
            // Fall back to computing it inline when `input_sums` is None.
            let ms0 = m_b0
                * (if !sums_ptr.is_null() {
                    *sums_ptr.add(b)
                } else {
                    sum_total(q8_b0) as f32 * si_b0
                });
            let ms1 = m_b1
                * (if !sums_ptr.is_null() {
                    *sums_ptr.add(b + 1)
                } else {
                    sum_total(q8_b1) as f32 * si_b1
                });

            // Q4_1 dot = dc * d * scale + m * input_sum. Explicit
            // mul+add (no FMA) to bit-match scalar rounding.
            let contrib0 = dc0 as f32 * ds0 + ms0;
            let contrib1 = dc1 as f32 * ds1 + ms1;
            acc = vaddq_f32(acc, vdupq_n_f32(contrib0));
            acc = vaddq_f32(acc, vdupq_n_f32(contrib1));

            b += 2;
        }

        while b < blocks_per_row {
            let off = row_off + b * 20;
            let d_b = f16_to_f32(u16::from_le_bytes(
                std::ptr::read_unaligned(w_ptr.add(off) as *const u16),
            ));
            let m_b = f16_to_f32(u16::from_le_bytes(
                std::ptr::read_unaligned(w_ptr.add(off + 2) as *const u16),
            ));
            let si_b = *sc_ptr.add(b);
            let q8_b = vld1q_s8(iq_ptr.add(b * 32));
            let dc = block_dot(vld1q_u8(w_ptr.add(off + 4)), q8_b, low_mask);
            let ms = m_b
                * (if !sums_ptr.is_null() {
                    *sums_ptr.add(b)
                } else {
                    sum_total(q8_b) as f32 * si_b
                });
            let contrib = dc as f32 * d_b * si_b + ms;
            acc = vaddq_f32(acc, vdupq_n_f32(contrib));
            b += 1;
        }

        *out_ptr.add(out_idx) = vaddvq_f32(acc);
    }
}

// Tiny helpers for typed pointer loads (NEON `vld1q_*` needs typed
// pointers; raw `*const u8` requires an explicit cast). Centralised so
// the call sites stay readable.
#[inline(always)]
unsafe fn vld1q_s8(p: *const u8) -> std::arch::aarch64::int8x16_t {
    use std::arch::aarch64::*;
    vld1q_s8(p as *const i8)
}

#[inline(always)]
unsafe fn vld1q_u8(p: *const u8) -> std::arch::aarch64::uint8x16_t {
    use std::arch::aarch64::*;
    vld1q_u8(p)
}

/// Per-block Q4 × Q8 SIMD dot. Returns the raw i32 sum
/// `sum(nibble × input)` so the caller can apply the `q4_1` formula
/// `nib_total * d * scale + m * input_sum` separately. This matches
/// `q4_1::avx2` exactly.
#[inline(always)]
unsafe fn block_dot(
    q4_bytes: std::arch::aarch64::uint8x16_t,
    q8_input: std::arch::aarch64::int8x16_t,
    low_mask: std::arch::aarch64::uint8x16_t,
) -> i32 {
    use std::arch::aarch64::*;

    let lo = vandq_u8(q4_bytes, low_mask);
    let hi = vshrq_n_u8(q4_bytes, 4);
    let q4_interleaved = vzipq_u8(lo, hi);
    let nib_acc = vdotq_u32(q4_interleaved, q8_input);
    vaddvq_u32(nib_acc) as i32
}

#[inline(always)]
unsafe fn block_pair_dot(
    q4_b0: std::arch::aarch64::uint8x16_t,
    q8_b0: std::arch::aarch64::int8x16_t,
    q4_b1: std::arch::aarch64::uint8x16_t,
    q8_b1: std::arch::aarch64::int8x16_t,
    low_mask: std::arch::aarch64::uint8x16_t,
) -> (i32, i32) {
    let dc0 = block_dot(q4_b0, q8_b0, low_mask);
    let dc1 = block_dot(q4_b1, q8_b1, low_mask);
    (dc0, dc1)
}

/// Sum of `int8x16_t` widened through i16 → i32 → total scalar.
#[inline(always)]
unsafe fn sum_total(q8_input: std::arch::aarch64::int8x16_t) -> i32 {
    use std::arch::aarch64::*;
    let y_lo16 = vmovl_s8(vget_low_s8(q8_input));
    let y_hi16 = vmovl_s8(vget_high_s8(q8_input));
    let y_pairs32 = vaddq_s32(vpaddlq_s16(y_lo16), vpaddlq_s16(y_hi16));
    vaddvq_s32(y_pairs32)
}
