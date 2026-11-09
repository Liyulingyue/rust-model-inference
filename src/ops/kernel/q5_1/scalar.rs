//! Q5_1 scalar matmul kernel.
//!
//! Each Q5_1 block holds 32 elements in 24 bytes:
//! - 2-byte F16 scale (`d`)
//! - 2-byte F16 min (`m`)
//! - 4-byte high-bit table (`qh`, the 5-th bit of each element)
//! - 16-byte nibble table (`qs`, the low four bits of each element)
//!
//! The layout and the arithmetic follow ggml's `dequantize_row_q5_1`
//! (`src/ggml-quants.c`) exactly:
//!
//! ```text
//! xh_0 = ((qh >> (j +  0)) << 4) & 0x10      // 5-th bit of element j
//! xh_1 = ((qh >> (j + 12))     ) & 0x10      // 5-th bit of element j+16
//! x0   = (qs[j] & 0x0F) | xh_0              // 0..31
//! x1   = (qs[j] >>   4) | xh_1              // 0..31
//! y    = x*d + m
//! ```
//!
//! Two details separate Q5_1 from its neighbours and are easy to get wrong:
//!
//! * Q5_1 has **no `- 16` offset**. Q5_0 stores a signed value (`x - 16`),
//!   Q5_1 stores the unsigned 5-bit level and moves the block minimum into
//!   `m`. So the dot product uses the raw level and the `m` term is added
//!   separately, exactly like Q4_1.
//! * The 5-th bit of the **upper** nibble lives in `qh` bit `j + 16`, i.e.
//!   `((qh >> (j + 12)) & 0x10)` and not `((qh >> (j + 16)) << 4) & 0x10`.
//!   Both spellings select the same bit, but only the first one also masks
//!   to the `0x10` position.

use crate::ops::kernel::q5_1::Q5_1Kernel;

pub fn matmul_q5_1_scalar_range(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    input_sums: Option<&[f32]>,
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    let n_blocks = n_in / Q5_1Kernel::BLOCK_ELEMENTS;
    let row_stride = n_blocks * Q5_1Kernel::BLOCK_BYTES;
    let per_thread = (n_out + nth - 1) / nth;
    let my_start = ith * per_thread;
    let my_end = (my_start + per_thread).min(n_out);
    if my_start >= my_end {
        return;
    }
    for out_idx in my_start..my_end {
        let row_off = out_idx * row_stride;
        let mut sum = 0.0f32;
        for block in 0..n_blocks {
            let off = row_off + block * Q5_1Kernel::BLOCK_BYTES;
            let d = crate::ops::f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
            let m = crate::ops::f16_to_f32(u16::from_le_bytes([weight[off + 2], weight[off + 3]]));
            let qh = u32::from_le_bytes([
                weight[off + 4],
                weight[off + 5],
                weight[off + 6],
                weight[off + 7],
            ]);
            let qs = &weight[off + 8..off + Q5_1Kernel::BLOCK_BYTES];
            let base_y = block * Q5_1Kernel::BLOCK_ELEMENTS;
            let scale = input_scales[block];
            // `x` is the unsigned 5-bit level (0..31): the low nibble plus
            // the 5-th bit that `qh` carries for this element.
            let mut dot: i32 = 0;
            let mut y_sum: i32 = 0;
            for l in 0..16 {
                let qb = qs[l];
                let lo_level = (qb & 0x0F) as i32 | (((qh >> l) & 1) as i32) << 4;
                // The 5-th bit of the upper nibble is qh bit l + 16.
                let hi_level = (qb >> 4) as i32 | (((qh >> (l + 16)) & 1) as i32) << 4;
                let y0 = input_q8[base_y + l] as i8 as i32;
                let y1 = input_q8[base_y + 16 + l] as i8 as i32;
                dot += lo_level * y0 + hi_level * y1;
                y_sum += y0 + y1;
            }
            // Every element of the block carries the same `+ m`, so the
            // block contributes `m * sum(y) * scale` next to the scaled
            // integer dot product. ggml's `block_q8_1` stores that sum
            // pre-multiplied by the activation scale (`s = d * sum(qs)`),
            // so the caller may pass it in directly; otherwise it is
            // derived from the quantized activations.
            let input_sum = input_sums.map_or(scale * y_sum as f32, |sums| sums[block]);
            sum += (d * scale) * dot as f32 + m * input_sum;
        }
        output[out_idx] = sum;
    }
}
