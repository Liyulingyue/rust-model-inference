//! Q1_0 scalar matmul kernel.
//!
//! Each Q1_0 block holds 128 elements (18 bytes: 2-byte F16 scale + 16-byte
//! bit-packed values). Each element is 1 bit: `bit ? d : -d`.
//!
//! The dot product with Q8_0 input is:
//!   sum = d * scale * sum_j( (bit_j ? 1 : -1) * q8_j )
//! where bit_j is extracted from the 16-byte bitfield.

pub fn matmul_q1_0_scalar_range(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    const BLOCK_ELEMENTS: usize = 128;
    const BLOCK_BYTES: usize = 18;
    const Q8_BLOCK: usize = 32;
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
            let bits = &weight[off + 2..off + BLOCK_BYTES];
            let base_y = block * BLOCK_ELEMENTS;
            // Q1_0 block = 128 elements = 4 Q8_0 blocks (32 each)
            for sub in 0..4 {
                let q8_base = base_y + sub * Q8_BLOCK;
                let scale = input_scales[block * 4 + sub];
                let mut dot: i32 = 0;
                for j in 0..Q8_BLOCK {
                    let abs_idx = sub * Q8_BLOCK + j;
                    let byte_idx = abs_idx / 8;
                    let bit_idx = abs_idx % 8;
                    let bit = (bits[byte_idx] >> bit_idx) & 1;
                    let x = if bit != 0 { 1i32 } else { -1i32 };
                    let y = input_q8[q8_base + j] as i8 as i32;
                    dot += x * y;
                }
                sum += dot as f32 * d * scale;
            }
        }
        output[out_idx] = sum;
    }
}
