//! Q4_1 scalar matmul kernel.
//!
//! Each Q4_1 block holds 32 elements (20 bytes: F16 scale + F16 min +
//! 16-byte nibbles). Scalar baseline; AVX2/NEON can be added here.
//!
//! Bit-exact contract: AVX2 path must produce identical output to this
//! function. See `q4_1::avx2` for the SIMD kernel.

use crate::ops::kernel::q4_1::Q4_1Kernel;

pub fn matmul_q4_1_scalar_range(
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
    let n_blocks = n_in / Q4_1Kernel::BLOCK_ELEMENTS;
    let row_stride = n_blocks * Q4_1Kernel::BLOCK_BYTES;
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
            let off = row_off + block * Q4_1Kernel::BLOCK_BYTES;
            let d = crate::ops::f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
            let m = crate::ops::f16_to_f32(u16::from_le_bytes([weight[off + 2], weight[off + 3]]));
            let qx = &weight[off + 4..off + Q4_1Kernel::BLOCK_BYTES];
            let base_y = block * Q4_1Kernel::BLOCK_ELEMENTS;
            let scale = input_scales[block];
            let mut dot: i32 = 0;
            let mut y_sum: i32 = 0;
            for l in 0..16 {
                let x0 = (qx[l] & 0x0F) as i32;
                let x1 = (qx[l] >> 4) as i32;
                let y0 = input_q8[base_y + l] as i8 as i32;
                let y1 = input_q8[base_y + 16 + l] as i8 as i32;
                dot += x0 * y0 + x1 * y1;
                y_sum += y0 + y1;
            }
            let input_sum = input_sums.map_or(scale * y_sum as f32, |sums| sums[block]);
            sum += (d * scale) * dot as f32 + m * input_sum;
        }
        output[out_idx] = sum;
    }
}
