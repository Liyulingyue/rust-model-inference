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
    let zero = vdupq_n_u8(0);

    for (out_idx, row) in (row_start..row_end).enumerate() {
        let row_off = row * row_stride;
        let mut acc = vdupq_n_f32(0.0);

        let mut b = 0;
        while b + 2 <= blocks_per_row {
            let off0 = row_off + b * 18;
            let off1 = row_off + (b + 1) * 18;

            let d_b0 = f16_to_f32(u16::from_le_bytes(
                std::ptr::read_unaligned(w_ptr.add(off0) as *const u16),
            ));
            let d_b1 = f16_to_f32(u16::from_le_bytes(
                std::ptr::read_unaligned(w_ptr.add(off1) as *const u16),
            ));
            let si_b0 = *sc_ptr.add(b);
            let si_b1 = *sc_ptr.add(b + 1);

            let q4_b0 = vld1q_u8(w_ptr.add(off0 + 2));
            let q4_b1 = vld1q_u8(w_ptr.add(off1 + 2));
            let q8_b0 = vld1q_s8(iq_ptr.add(b * 32));
            let q8_b1 = vld1q_s8(iq_ptr.add((b + 1) * 32));

            let (dc0, dc1) = block_pair_dot(q4_b0, q8_b0, q4_b1, q8_b1, low_mask, zero);
            let ds0 = d_b0 * si_b0;
            let ds1 = d_b1 * si_b1;
            // Convert i32 scalar dot products to f32 and FMA. (The two
            // blocks contribute a single i32 each, so scalar `vcvtq_f32`
            // / `vdupq_n_f32` is fine here; the SIMD win is in `vdotq_u32`.)
            let ds0_v = vdupq_n_f32(ds0);
            let ds1_v = vdupq_n_f32(ds1);
            acc = vfmaq_f32(ds0_v, vdupq_n_f32(dc0 as f32), acc);
            acc = vfmaq_f32(ds1_v, vdupq_n_f32(dc1 as f32), acc);

            b += 2;
        }

        while b < blocks_per_row {
            let off = row_off + b * 18;
            let d_b = f16_to_f32(u16::from_le_bytes(
                std::ptr::read_unaligned(w_ptr.add(off) as *const u16),
            ));
            let si_b = *sc_ptr.add(b);
            let dc = block_dot(
                vld1q_u8(w_ptr.add(off + 2)),
                vld1q_s8(iq_ptr.add(b * 32)),
                low_mask,
                zero,
            );
            let ds_v = vdupq_n_f32(d_b * si_b);
            acc = vfmaq_f32(ds_v, vdupq_n_f32(dc as f32), acc);
            b += 1;
        }

        *out_ptr.add(out_idx) = vaddvq_f32(acc);
    }
}

/// Per-block Q4 × Q8 SIMD dot. Returns the corrected dot product as
/// `i32`:
///   sum((nibble - 8) * input)  [exact i32 arithmetic]
/// computed without ever materialising `(nibble - 8)` per element:
///   = sum(nibble × input) - 8 × sum(input)
///
/// Caller multiplies by `d * scale` (f32) and accumulates via FMA.
///
/// The `zero` parameter exists only so the signature mirrors `q4_0::avx2`;
/// NEON's `vandq_u8` for the high nibble doesn't need a "ones" register
/// the way the AVX2 `_mm256_madd_epi16` does — `vdotq_u32` consumes the
/// 8-bit pairs directly.
#[inline(always)]
unsafe fn block_dot(
    q4_bytes: std::arch::aarch64::uint8x16_t,
    q8_input: std::arch::aarch64::int8x16_t,
    low_mask: std::arch::aarch64::uint8x16_t,
    _zero: std::arch::aarch64::uint8x16_t,
) -> i32 {
    use std::arch::aarch64::*;

    // Low + high nibble expansion into u8 lanes. NEON has no direct
    // "vpand_u8 with immediate" + shift combination as ergonomic as
    // AVX2's `_mm_and_si128` + `_mm_srli_epi16`, but `vshrq_n_u8`
    // + `vandq_u8` is the canonical pair. `low_nibbles` ends up in the
    // even lanes of a 16-byte register, `high_nibbles` in the odd lanes.
    let lo = vandq_u8(q4_bytes, low_mask);
    let hi = vshrq_n_u8(q4_bytes, 4);

    // Interleave: pack as [lo0, hi0, lo1, hi1, ...] so `vdotq_u32` can
    // consume u8 × i8 pairs directly. NEON's `vzipq_u8` does this
    // without a temp buffer.
    let q4_interleaved = vzipq_u8(lo, hi);

    // nib_total = sum(q4 × q8) where q4 ∈ u8 [0, 15] and q8 ∈ i8 [-128, 127].
    // `vdotq_u32` returns four u32 partial sums; we then sum them with
    // pairwise add to land in lane 0.
    let nib_acc = vdotq_u32(q4_interleaved, q8_input);
    let nib_total = vaddvq_u32(nib_acc);

    // sum_input = sum(q8) computed in i32 to keep room for the -8×N bias.
    // `vpaddlq_s8` widens i8 → i16 (saturating). `vpadalq_s16` then
    // adds adjacent pairs of i16 lanes into i32. Two consecutive calls
    // (or `vpaddlqq_s8`) give the full sum.
    let y_lo16 = vmovl_s8(vget_low_s8(q8_input));
    let y_hi16 = vmovl_s8(vget_high_s8(q8_input));
    let y_pairs32 = vaddq_s32(vpaddlq_s16(y_lo16), vpaddlq_s16(y_hi16));
    let sum_total = vaddvq_s32(y_pairs32);

    // The corrected dot: (q4 - 8) · q8 = q4 · q8 - 8 · sum(q8).
    (nib_total as i32) - 8 * (sum_total as i32)
}

#[inline(always)]
unsafe fn block_pair_dot(
    q4_b0: std::arch::aarch64::uint8x16_t,
    q8_b0: std::arch::aarch64::int8x16_t,
    q4_b1: std::arch::aarch64::uint8x16_t,
    q8_b1: std::arch::aarch64::int8x16_t,
    low_mask: std::arch::aarch64::uint8x16_t,
    zero: std::arch::aarch64::uint8x16_t,
) -> (i32, i32) {
    let dc0 = block_dot(q4_b0, q8_b0, low_mask, zero);
    let dc1 = block_dot(q4_b1, q8_b1, low_mask, zero);
    (dc0, dc1)
}
