//! Q4_0 scalar matmul kernel.
//!
//! Each Q4_0 block holds 32 elements (18 bytes: 2-byte F16 scale + 16-byte
//! nibbles). The hot path is currently scalar — AVX2/NEON variants can
//! be added in a sibling file (e.g. `avx2.rs`) alongside this baseline
//! without touching any other code.

/// Q4_0 scalar matmul kernel. Phase 2.7-final: moved from `ops::matmul`.
/// Each Q4_0 block holds 32 elements (18 bytes: 2-byte F16 scale + 16-byte
/// nibbles). The hot path is currently scalar — AVX2/NEON variants can
/// be added in this file alongside this baseline without touching any
/// other code.
pub fn matmul_q4_0_scalar_range(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 18;
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
            let qx = &weight[off + 2..off + BLOCK_BYTES];
            let base_y = block * BLOCK_ELEMENTS;
            let scale = input_scales[block];
            let mut dot: i32 = 0;
            for l in 0..16 {
                let x0 = (qx[l] & 0x0F) as i32 - 8;
                let x1 = (qx[l] >> 4) as i32 - 8;
                let y0 = input_q8[base_y + l] as i8 as i32;
                let y1 = input_q8[base_y + 16 + l] as i8 as i32;
                dot += x0 * y0 + x1 * y1;
            }
            sum += dot as f32 * d * scale;
        }
        output[out_idx] = sum;
    }
}
