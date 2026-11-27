//! Q5_0 NEON (aarch64) matmul kernel.
//!
//! Transcribed from ggml's ARM `ggml_vec_dot_q5_0_q8_0`
//! (`ggml/src/ggml-cpu/arch/arm/quants.c`) so that the result matches
//! upstream **bit for bit**, not just to within a rounding error.
//!
//! Why bit-exactness matters here: the repository's scalar Q5_0 kernel
//! accumulates one integer dot per block and multiplies by the scale
//! afterwards, while ggml's ARM kernel keeps two 4-lane F32 accumulators
//! and folds the scale into an FMA. The two are algebraically identical
//! but round differently, and the resulting 1-ULP difference is enough to
//! flip the 8-bit Q8_K activation levels of the following Q4_K
//! projection -- which turns a 1e-7 relative difference into a 4e-4 one.
//!
//! The structure below (two blocks per iteration, four-lane integer dots
//! widened to F32, `vmlaq_n_f32` accumulation, `vaddvq_f32` reduction) is
//! ggml's, and the per-lane grouping of the integer dot has to match
//! ggml's as well: ggml accumulates each lane separately, so a different
//! grouping rounds differently. ggml's own comment on its `ggml_vdotq_s32`
//! fallback says exactly that ("different per-lane grouping -- do not use
//! when individual lane values matter").
//!
//! This kernel therefore uses the native `vdotq_s32` instruction, which
//! requires the `dotprod` extension. The upstream oracle is built with the
//! same extension enabled (`tools/oracle/nemotron3_nano/build_oracle.sh`)
//! so that both sides run the same instruction.

#![cfg(target_arch = "aarch64")]

use crate::ops::f16_to_f32;

const BLOCK_ELEMENTS: usize = 32;
const BLOCK_BYTES: usize = 22;

/// ggml's `table_b2b_1`: expands 8 bits into 8 bytes, one per bit, where a
/// *set* bit becomes `0x00` and a clear bit `0x10`.
///
/// ggml writes the table as `static const uint64_t table_b2b_1[1 << 8] =
/// { B8(10, 00) };` -- a macro that expands one entry per possible byte.
/// Transcribing 256 magic 64-bit constants by hand is exactly the kind of
/// transcription that silently produces a plausible-looking wrong table, so
/// the table is generated from the rule instead. The rule is ggml's own
/// comment on the table: "(!b) << 4", i.e. the byte is `0x10` when the bit
/// is clear and `0x00` when it is set. The ARM kernel subtracts that from
/// the nibble, which is how the `q - 16` bias is folded in.
#[inline]
fn table_b2b_1_entry(bits: u8) -> u64 {
    let mut value = 0u64;
    let mut bit = 0;
    while bit < 8 {
        let byte = if bits & (1u8 << bit) != 0 { 0x00 } else { 0x10 };
        value |= (byte as u64) << (8 * bit);
        bit += 1;
    }
    value
}

/// NEON Q5_0 x Q8_0 matmul over a row range.
///
/// `weight` holds `n_out` rows of `n_in / 32` Q5_0 blocks (22 bytes each),
/// `input_q8` holds `n_in` Q8_0 levels and `input_scales` the per-block
/// F16-rounded scale of each 32-element activation block.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q5_0_neon_range(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    use std::arch::aarch64::*;

    let n_blocks = n_in / BLOCK_ELEMENTS;
    let row_stride = n_blocks * BLOCK_BYTES;
    let per_thread = (n_out + nth - 1) / nth;
    let my_start = ith * per_thread;
    let my_end = (my_start + per_thread).min(n_out);
    if my_start >= my_end {
        return;
    }

    let m4b = vdupq_n_u8(0x0F);
    for out_idx in my_start..my_end {
        let row_off = out_idx * row_stride;

        // Two 4-lane F32 accumulators, exactly like ggml's ARM kernel.
        let mut sumv0 = vdupq_n_f32(0.0f32);
        let mut sumv1 = vdupq_n_f32(0.0f32);

        let mut ib = 0usize;
        while ib + 1 < n_blocks {
            let x0 = weight.as_ptr().add(row_off + ib * BLOCK_BYTES);
            let x1 = weight.as_ptr().add(row_off + (ib + 1) * BLOCK_BYTES);
            let y0 = input_q8.as_ptr().add(ib * BLOCK_ELEMENTS) as *const i8;
            let y1 = input_q8.as_ptr().add((ib + 1) * BLOCK_ELEMENTS) as *const i8;

            let d_x0 = f16_to_f32(u16::from_le_bytes([*x0, *x0.add(1)]));
            let d_x1 = f16_to_f32(u16::from_le_bytes([*x1, *x1.add(1)]));
            let d_y0 = input_scales[ib];
            let d_y1 = input_scales[ib + 1];

            let qh0 = u32::from_le_bytes([*x0.add(2), *x0.add(3), *x0.add(4), *x0.add(5)]);
            let qh1 = u32::from_le_bytes([*x1.add(2), *x1.add(3), *x1.add(4), *x1.add(5)]);

            // Expand the high-bit table into the per-nibble `0x00` / `0x10`
            // bytes ggml subtracts from the nibbles.
            let t0 = [
                table_b2b_1_entry((qh0 >> 0) as u8),
                table_b2b_1_entry((qh0 >> 8) as u8),
                table_b2b_1_entry((qh0 >> 16) as u8),
                table_b2b_1_entry((qh0 >> 24) as u8),
            ];
            let t1 = [
                table_b2b_1_entry((qh1 >> 0) as u8),
                table_b2b_1_entry((qh1 >> 8) as u8),
                table_b2b_1_entry((qh1 >> 16) as u8),
                table_b2b_1_entry((qh1 >> 24) as u8),
            ];
            let qhl0 = vld1q_s8(t0.as_ptr() as *const i8);
            let qhh0 = vld1q_s8(t0.as_ptr().add(2) as *const i8);
            let qhl1 = vld1q_s8(t1.as_ptr() as *const i8);
            let qhh1 = vld1q_s8(t1.as_ptr().add(2) as *const i8);

            // Split the nibbles and apply the bias.
            let v0 = vld1q_u8(x0.add(6) as *const u8);
            let v1 = vld1q_u8(x1.add(6) as *const u8);
            let lo0 = vreinterpretq_s8_u8(vandq_u8(v0, m4b));
            let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));
            let lo1 = vreinterpretq_s8_u8(vandq_u8(v1, m4b));
            let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));
            let lo0f = vsubq_s8(lo0, qhl0);
            let hi0f = vsubq_s8(hi0, qhh0);
            let lo1f = vsubq_s8(lo1, qhl1);
            let hi1f = vsubq_s8(hi1, qhh1);

            let yl0 = vld1q_s8(y0);
            let yh0 = vld1q_s8(y0.add(16));
            let yl1 = vld1q_s8(y1);
            let yh1 = vld1q_s8(y1.add(16));

            let p0 = vaddq_s32(
                vdotq_s32(vdupq_n_s32(0), lo0f, yl0),
                vdotq_s32(vdupq_n_s32(0), hi0f, yh0),
            );
            let p1 = vaddq_s32(
                vdotq_s32(vdupq_n_s32(0), lo1f, yl1),
                vdotq_s32(vdupq_n_s32(0), hi1f, yh1),
            );

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), d_x0 * d_y0);
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), d_x1 * d_y1);

            ib += 2;
        }

        let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

        // An odd block count leaves one block over; ggml handles it with
        // its generic scalar path, and so does this.
        if ib < n_blocks {
            let off = row_off + ib * BLOCK_BYTES;
            let d = f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
            let qh = u32::from_le_bytes([
                weight[off + 2],
                weight[off + 3],
                weight[off + 4],
                weight[off + 5],
            ]);
            let qs = &weight[off + 6..off + BLOCK_BYTES];
            let base_y = ib * BLOCK_ELEMENTS;
            let scale = input_scales[ib];
            let mut sumi0 = 0i32;
            let mut sumi1 = 0i32;
            for j in 0..BLOCK_ELEMENTS / 2 {
                let xh_0 = (((qh >> j) & 1) << 4) as u8;
                let xh_1 = (((qh >> (j + 16)) & 1) << 4) as u8;
                let x0 = ((qs[j] & 0x0F) | xh_0) as i32 - 16;
                let x1 = ((qs[j] >> 4) | xh_1) as i32 - 16;
                sumi0 += x0 * input_q8[base_y + j] as i8 as i32;
                sumi1 += x1 * input_q8[base_y + 16 + j] as i8 as i32;
            }
            sumf += (d * scale) * (sumi0 + sumi1) as f32;
        }

        output[out_idx] = sumf;
    }
}
