//! Q5_1 NEON (aarch64) matmul kernel.
//!
//! Mirrors the structure of `q4_1::neon`: the 32-element blocks are
//! processed one at a time, the quantized activation is consumed as
//! Q8_0, and the per-block contribution is accumulated in F32 in the
//! same order as the scalar baseline so the two agree bit for bit.
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
//! shifted into the `0x10` position. The same table is used here, so
//! the layout is identical to the reference implementation.
//!
//! The bit-to-element mapping follows ggml: `qh` bits 0..15 belong to
//! the low nibbles (elements 0..15) and bits 16..31 belong to the high
//! nibbles (elements 16..31).

#![cfg(target_arch = "aarch64")]

use crate::ops::f16_to_f32;

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

    debug_assert_eq!(n_in % 32, 0);
    let blocks_per_row = n_in / 32;
    let row_stride = blocks_per_row * 24;
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
        let mut sum = 0.0f32;
        for block in 0..blocks_per_row {
            let off = row_off + block * 24;
            let d = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off) as *const u16));
            let m = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off + 2) as *const u16));
            let scale = *sc_ptr.add(block);
            let qh = std::ptr::read_unaligned(w_ptr.add(off + 4) as *const u32);
            let qs = vld1q_u8(w_ptr.add(off + 8));
            let q8_low = vld1q_s8(iq_ptr.add(block * 32) as *const i8);
            let q8_high = vld1q_s8(iq_ptr.add(block * 32 + 16) as *const i8);

            // Spread the 5-th bit over the nibbles: bits 0..15 land on
            // the low nibbles, bits 16..31 on the high nibbles.
            expanded[0..8].copy_from_slice(&B2B_TABLE[(qh & 0xff) as usize].to_le_bytes());
            expanded[8..16].copy_from_slice(&B2B_TABLE[((qh >> 8) & 0xff) as usize].to_le_bytes());
            expanded[16..24].copy_from_slice(&B2B_TABLE[((qh >> 16) & 0xff) as usize].to_le_bytes());
            expanded[24..32].copy_from_slice(&B2B_TABLE[((qh >> 24) & 0xff) as usize].to_le_bytes());
            let bit_low = vld1q_u8(expanded.as_ptr());
            let bit_high = vld1q_u8(expanded.as_ptr().add(16));

            // 4-bit -> 8-bit, then merge the 5-th bit back in. The level
            // stays in 0..31, which fits into a signed byte, so `UDOT`
            // can multiply it against the Q8_0 activation directly.
            let level_low = vorrq_u8(vandq_u8(qs, low_mask), bit_low);
            let level_high = vorrq_u8(vshrq_n_u8(qs, 4), bit_high);

            let dot = vaddvq_s32(vaddq_s32(
                vdotq_s32(
                    vdupq_n_s32(0),
                    vreinterpretq_s8_u8(level_low),
                    q8_low,
                ),
                vdotq_s32(
                    vdupq_n_s32(0),
                    vreinterpretq_s8_u8(level_high),
                    q8_high,
                ),
            ));
            let q8_sum = sum_q8(q8_low, q8_high);
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

/// Sum of all 32 signed Q8 values.
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
