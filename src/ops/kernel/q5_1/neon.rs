//! Q5_1 NEON (aarch64) matmul kernel.
//!
//! Transcribed from ggml's ARM `ggml_vec_dot_q5_1_q8_1`
//! (`ggml/src/ggml-cpu/arch/arm/quants.c`) so that the result matches
//! upstream **bit for bit**, not just to within a rounding error.
//!
//! Why bit-exactness matters here: a Q5_1 element is `level * d + m`,
//! and ggml's ARM kernel accumulates the two terms in *separate*
//! accumulators -- the scaled integer dots in two 4-lane F32
//! accumulators, the `m * s` terms in two scalar ones -- and only adds
//! them together at the very end. Accumulating both terms in a single
//! scalar instead (the obvious transcription) is algebraically identical
//! but rounds differently, and the resulting 1-ULP difference in the
//! shared expert's up projection is enough to flip an 8-bit Q8_0
//! activation level of the following down projection.
//!
//! The structure below (two blocks per iteration, four-lane integer dots
//! widened to F32, `vmlaq_n_f32` accumulation, `vaddvq_f32` reduction,
//! separate `m * s` accumulators) is ggml's, and so is the per-lane
//! grouping of the integer dot: ggml accumulates each lane separately,
//! so a different grouping rounds differently. ggml's own comment on its
//! `ggml_vdotq_s32` fallback says exactly that ("different per-lane
//! grouping -- do not use when individual lane values matter").
//!
//! This kernel therefore uses the native `vdotq_s32` instruction, which
//! requires the `dotprod` extension. The upstream oracle is built with the
//! same extension enabled (`tools/oracle/nemotron3_nano/build_oracle.sh`)
//! so that both sides run the same instruction.
//!
//! # Expanding the 5-th bit
//!
//! A Q5_1 element is `level * d + m` with `level` in 0..31: four bits
//! live in the nibble table `qs` and the fifth bit lives in the
//! high-bit table `qh`. The fifth bit has to be spread over the 16
//! nibbles before the dot product can run, because NEON's `UDOT`
//! multiplies plain `int8` lanes.
//!
//! ggml solves this with a 256-entry lookup table
//! (`table_b2b_0` in `ggml-cpu/arch/arm/quants.c`): one table entry
//! turns an 8-bit value into 8 bytes, each byte holding that bit
//! shifted into the `0x10` position. The same table is used here, so the
//! layout is identical to the reference implementation.
//!
//! The bit-to-element mapping follows ggml: `qh` bits 0..15 belong to
//! the low nibbles (elements 0..15) and bits 16..31 belong to the high
//! nibbles (elements 16..31).

#![cfg(target_arch = "aarch64")]

use crate::ops::f16_to_f32;

const BLOCK_ELEMENTS: usize = 32;
const BLOCK_BYTES: usize = 24;

/// Expand an 8-bit value into 8 bytes, where byte `i` holds bit `i`
/// shifted into the `0x10` position (`(b >> i & 1) << 4`).
///
/// Byte order matches the little-endian memory layout ggml reads the
/// table with: byte 0 of an entry carries bit 0 of the index.
static B2B_TABLE: [u64; 256] = build_b2b_table();

const fn build_b2b_table() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut index = 0usize;
    while index < 256 {
        let mut value = 0u64;
        let mut bit = 0usize;
        while bit < 8 {
            if (index >> bit) & 1 == 1 {
                value |= 0x10u64 << (8 * bit);
            }
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

/// NEON Q5_1 x Q8_1 matmul over a row range.
///
/// `weight` holds `n_out` rows of `n_in / 32` Q5_1 blocks (24 bytes each),
/// `input_q8` holds the `n_in` Q8_1 levels and `input_scales` the
/// per-block F16-rounded scale of each 32-element activation block.
/// `input_sums` holds the Q8_1 blocks' `s` field, i.e.
/// `f16(d * sum(levels))`; it is `None` only when the caller could not
/// provide the F32 activation, in which case the sum is recomputed from
/// the levels and the (already rounded) block scale.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q5_1_vs_q8_0_neon(
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

    debug_assert_eq!(n_in % BLOCK_ELEMENTS, 0);
    let blocks_per_row = n_in / BLOCK_ELEMENTS;
    let row_stride = blocks_per_row * BLOCK_BYTES;
    let w_ptr = weight.as_ptr();
    let iq_ptr = input_q8.as_ptr();
    let sc_ptr = input_scales.as_ptr();
    let sums_ptr = input_sums.map_or(std::ptr::null(), |s| s.as_ptr());
    let out_ptr = output.as_mut_ptr();
    let low_mask = vdupq_n_u8(0x0f);
    // Scratch for the expanded 5-th bit, reused across blocks.
    let mut expanded = [0u8; 32];

    for (out_idx, row) in (row_start..row_end).enumerate() {
        let row_off = row * row_stride;

        // ggml keeps four accumulators: two 4-lane F32 accumulators for
        // the scaled integer dots (even / odd block) and two scalar ones
        // for the `m * s` terms.
        let mut sumv0 = vdupq_n_f32(0.0f32);
        let mut sumv1 = vdupq_n_f32(0.0f32);
        let mut summs0 = 0.0f32;
        let mut summs1 = 0.0f32;

        let mut block = 0usize;
        while block + 1 < blocks_per_row {
            let off0 = row_off + block * BLOCK_BYTES;
            let off1 = off0 + BLOCK_BYTES;
            let qh0 = u32::from_le_bytes([
                *w_ptr.add(off0 + 4),
                *w_ptr.add(off0 + 5),
                *w_ptr.add(off0 + 6),
                *w_ptr.add(off0 + 7),
            ]);
            let qh1 = u32::from_le_bytes([
                *w_ptr.add(off1 + 4),
                *w_ptr.add(off1 + 5),
                *w_ptr.add(off1 + 6),
                *w_ptr.add(off1 + 7),
            ]);
            let d_x0 = f16_to_f32(u16::from_le_bytes([*w_ptr.add(off0), *w_ptr.add(off0 + 1)]));
            let d_x1 = f16_to_f32(u16::from_le_bytes([*w_ptr.add(off1), *w_ptr.add(off1 + 1)]));
            let m_x0 = f16_to_f32(u16::from_le_bytes([
                *w_ptr.add(off0 + 2),
                *w_ptr.add(off0 + 3),
            ]));
            let m_x1 = f16_to_f32(u16::from_le_bytes([
                *w_ptr.add(off1 + 2),
                *w_ptr.add(off1 + 3),
            ]));
            let d_y0 = *sc_ptr.add(block);
            let d_y1 = *sc_ptr.add(block + 1);

            // The `m * s` terms go into their own scalar accumulators.
            summs0 += m_x0 * q8_1_sum(sums_ptr, input_q8, block, d_y0);
            summs1 += m_x1 * q8_1_sum(sums_ptr, input_q8, block + 1, d_y1);

            // Spread the 5-th bit over the nibbles: bits 0..15 land on
            // the low nibbles, bits 16..31 on the high nibbles.
            expanded[0..8].copy_from_slice(&B2B_TABLE[(qh0 & 0xff) as usize].to_le_bytes());
            expanded[8..16].copy_from_slice(&B2B_TABLE[((qh0 >> 8) & 0xff) as usize].to_le_bytes());
            expanded[16..24]
                .copy_from_slice(&B2B_TABLE[((qh0 >> 16) & 0xff) as usize].to_le_bytes());
            expanded[24..32]
                .copy_from_slice(&B2B_TABLE[((qh0 >> 24) & 0xff) as usize].to_le_bytes());
            let bit_low = vld1q_u8(expanded.as_ptr());
            let bit_high = vld1q_u8(expanded.as_ptr().add(16));

            expanded[0..8].copy_from_slice(&B2B_TABLE[(qh1 & 0xff) as usize].to_le_bytes());
            expanded[8..16].copy_from_slice(&B2B_TABLE[((qh1 >> 8) & 0xff) as usize].to_le_bytes());
            expanded[16..24]
                .copy_from_slice(&B2B_TABLE[((qh1 >> 16) & 0xff) as usize].to_le_bytes());
            expanded[24..32]
                .copy_from_slice(&B2B_TABLE[((qh1 >> 24) & 0xff) as usize].to_le_bytes());
            let bit_low1 = vld1q_u8(expanded.as_ptr());
            let bit_high1 = vld1q_u8(expanded.as_ptr().add(16));

            // 4-bit -> 8-bit, then merge the 5-th bit back in. The level
            // stays in 0..31, which fits into a signed byte, so `UDOT`
            // can multiply it against the Q8 activation directly.
            let qs0 = vld1q_u8(w_ptr.add(off0 + 8));
            let qs1 = vld1q_u8(w_ptr.add(off1 + 8));
            let level0_low = vorrq_u8(vandq_u8(qs0, low_mask), bit_low);
            let level0_high = vorrq_u8(vshrq_n_u8(qs0, 4), bit_high);
            let level1_low = vorrq_u8(vandq_u8(qs1, low_mask), bit_low1);
            let level1_high = vorrq_u8(vshrq_n_u8(qs1, 4), bit_high1);

            let y0_low = vld1q_s8(iq_ptr.add(block * BLOCK_ELEMENTS) as *const i8);
            let y0_high = vld1q_s8(iq_ptr.add(block * BLOCK_ELEMENTS + 16) as *const i8);
            let y1_low = vld1q_s8(iq_ptr.add((block + 1) * BLOCK_ELEMENTS) as *const i8);
            let y1_high = vld1q_s8(iq_ptr.add((block + 1) * BLOCK_ELEMENTS + 16) as *const i8);

            let p0 = vaddq_s32(
                vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(level0_low), y0_low),
                vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(level0_high), y0_high),
            );
            let p1 = vaddq_s32(
                vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(level1_low), y1_low),
                vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(level1_high), y1_high),
            );

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), d_x0 * d_y0);
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), d_x1 * d_y1);

            block += 2;
        }

        // ggml adds the two vector reductions and the two `m` sums in
        // this order: `((vaddvq(sumv0) + vaddvq(sumv1)) + summs0) + summs1`.
        let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs0 + summs1;

        // An odd block count leaves one block over; ggml handles it with
        // its generic scalar path, and so does this.
        if block < blocks_per_row {
            let off = row_off + block * BLOCK_BYTES;
            let d_x = f16_to_f32(u16::from_le_bytes([*w_ptr.add(off), *w_ptr.add(off + 1)]));
            let m_x = f16_to_f32(u16::from_le_bytes([
                *w_ptr.add(off + 2),
                *w_ptr.add(off + 3),
            ]));
            let d_y = *sc_ptr.add(block);
            let qh = u32::from_le_bytes([
                *w_ptr.add(off + 4),
                *w_ptr.add(off + 5),
                *w_ptr.add(off + 6),
                *w_ptr.add(off + 7),
            ]);
            let base_y = block * BLOCK_ELEMENTS;
            let mut sumi0 = 0i32;
            let mut sumi1 = 0i32;
            for j in 0..BLOCK_ELEMENTS / 2 {
                let xh_0 = (((qh >> j) & 1) << 4) as u8;
                let xh_1 = (((qh >> (j + 16)) & 1) << 4) as u8;
                let x0 = ((*w_ptr.add(off + 8 + j) & 0x0F) | xh_0) as i32;
                let x1 = ((*w_ptr.add(off + 8 + j) >> 4) | xh_1) as i32;
                sumi0 += x0 * *iq_ptr.add(base_y + j) as i8 as i32;
                sumi1 += x1 * *iq_ptr.add(base_y + 16 + j) as i8 as i32;
            }
            sumf += (d_x * d_y) * (sumi0 + sumi1) as f32
                + m_x * q8_1_sum(sums_ptr, input_q8, block, d_y);
        }

        *out_ptr.add(out_idx) = sumf;
    }
}

/// The Q8_1 block's `s` field: `f16(d * sum(levels))`.
///
/// When the caller supplied the sums (the F32 activation was available)
/// they are used verbatim. Otherwise they are recomputed from the levels
/// and the block scale; that fallback cannot reproduce ggml's `f16(d *
/// sum)` exactly, because `d` has already been rounded to F16 by then,
/// so it is only an approximation of the reference.
#[inline(always)]
unsafe fn q8_1_sum(sums_ptr: *const f32, input_q8: &[u8], block: usize, scale: f32) -> f32 {
    if !sums_ptr.is_null() {
        return *sums_ptr.add(block);
    }
    let base = block * BLOCK_ELEMENTS;
    let mut sum = 0i32;
    for j in 0..BLOCK_ELEMENTS {
        sum += *input_q8.get_unchecked(base + j) as i8 as i32;
    }
    scale * sum as f32
}
