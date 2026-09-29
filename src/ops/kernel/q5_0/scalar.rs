//! Q5_0 scalar matmul kernel.
//!
//! Each Q5_0 block holds 32 elements (22 bytes: 2-byte F16 scale +
//! 4-byte high-bit table + 16-byte low-nibble table). Unlike Q5_1,
//! Q5_0 has no per-block `m` field, so the per-element value is just
//! `d * q - 16` where `q` is a 5-bit unsigned value (range 0..31)
//! packed across `qs` (low nibbles) and `qh` (one extra bit per
//! element). The kernel accepts a Q8-prequantized input plus
//! `ith`/`nth` row partitioning so it can be dispatched inside a
//! `pool.compute` closure.

const BLOCK_ELEMENTS: usize = 32;
const BLOCK_BYTES: usize = 22;

/// Q5_0 scalar matmul kernel. Mirrors the Q4_0 baseline contract: take
/// raw GGUF bytes, a Q8-prequantized input + per-block scales, and
/// accumulate F32 outputs in `output[my_start..my_end]`.
pub fn matmul_q5_0_scalar_range(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    let n_blocks = n_in / BLOCK_ELEMENTS;
    let row_stride = n_blocks * BLOCK_BYTES;
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
            let off = row_off + block * BLOCK_BYTES;
            let d = crate::ops::f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
            let qh = u32::from_le_bytes([
                weight[off + 2],
                weight[off + 3],
                weight[off + 4],
                weight[off + 5],
            ]);
            let qs = &weight[off + 6..off + BLOCK_BYTES];
            let base_y = block * BLOCK_ELEMENTS;
            let scale = input_scales[block];
            // Q5_0 stores `q = (qh_bit << 4) | nibble` where `nibble` is
            // a 4-bit unsigned 0..2^4-1 = 0..15. The high bit lifts the
            // representation to 5 bits (0..31). Per llama.cpp's
            // `dequantize_row_q5_0`, the value is `d * q - 16`, with the
            // `-16` constant offset baked in. The `dot` integer running
            // total can therefore use the raw nibble directly without
            // an explicit `q - 16` shift — the constant offset will be
            // subtracted at the end of each block.
            let mut dot: i32 = 0;
            let mut const_offset: i32 = 0;
            for l in 0..16 {
                let qb = qs[l];
                let lo_nibble = (qb & 0x0F) as i32;
                let hi_nibble = (qb >> 4) as i32;
                let hbit_lo = ((qh >> l) & 1) as i32;
                let hbit_hi = ((qh >> (l + 16)) & 1) as i32;
                let q_lo = lo_nibble | (hbit_lo << 4);
                let q_hi = hi_nibble | (hbit_hi << 4);
                let y0 = input_q8[base_y + l] as i8 as i32;
                let y1 = input_q8[base_y + 16 + l] as i8 as i32;
                dot += q_lo * y0 + q_hi * y1;
                // The `-16` per-element offset summed against `y_i`
                // contributes `16 * y_i` per element to the integer
                // dot. (We drop the constant's sign because we will
                // subtract it from the F32 accumulator below.)
                const_offset += y0 + y1;
            }
            // `d * (dot - 16 * sum_i y_i) * scale` rewritten as
            // `d * scale * dot - 16 * d * scale * sum_y`. The sum-of-y
            // term is a small integer cost (at most 32 per block).
            sum += d * scale * (dot - 16 * const_offset) as f32;
        }
        output[out_idx] = sum;
    }
}