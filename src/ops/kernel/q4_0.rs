//! Q4_0 block matmul kernel implementation.
//!
//! Phase 2.5 + 2.7-final: Q4_0 uses 32-element blocks with 18-byte layout
//! (2-byte F16 scale + 16-byte nibbles). The `forward_prequantized` method
//! accepts a Q8-prequantized input plus `ith`/`nth` row partitioning so the
//! kernel can be dispatched inside a `pool.compute` closure.

use super::Kernel;

/// Q4_0 weight buffer: 32-element blocks, 18-byte layout
/// (2-byte F16 scale + 16-byte nibbles).
///
/// Phase 2.7-final: moved from `ops::matmul` to live alongside `Q4_0Kernel`.
#[derive(Debug, Clone, Copy)]
pub struct Q4_0Weight<'a> {
    pub data: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Q4_0Kernel<'a> {
    pub weight: &'a [u8],
}

impl<'a> Q4_0Kernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 32;
    pub const BLOCK_BYTES: usize = 18;

    pub fn new(weight: &'a [u8]) -> Self {
        Self { weight }
    }
}

impl<'a> Kernel for Q4_0Kernel<'a> {
    fn forward_prequantized(
        &self,
        input_q8: &[u8],
        input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        matmul_q4_0_scalar_range(
            self.weight,
            input_q8,
            input_scales,
            output,
            n_in,
            n_out,
            ith,
            nth,
        );
    }
}

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
    let n_blocks = n_in / Q4_0Kernel::BLOCK_ELEMENTS;
    let row_stride = n_blocks * Q4_0Kernel::BLOCK_BYTES;
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
            let off = row_off + block * Q4_0Kernel::BLOCK_BYTES;
            let d = crate::ops::f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
            let qx = &weight[off + 2..off + Q4_0Kernel::BLOCK_BYTES];
            let base_y = block * Q4_0Kernel::BLOCK_ELEMENTS;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn q4_0_uniform_block(scale: f32, nibble: u8) -> Vec<u8> {
        assert!(nibble < 16);
        let mut block = Vec::with_capacity(18);
        let s_bits = crate::ops::f32_to_f16(scale).to_le_bytes();
        block.extend_from_slice(&s_bits);
        let packed = nibble | (nibble << 4);
        for _ in 0..16 {
            block.push(packed);
        }
        block
    }

    #[test]
    fn q4_0_kernel_uniform_block_yields_zero_for_zero_nibble() {
        let weight = q4_0_uniform_block(1.0, 8);
        let input_q8 = vec![1i8 as u8; 32];
        let input_scales = vec![1.0f32];

        let mut output = [0.0f32; 1];
        let kernel = Q4_0Kernel::new(&weight);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1, 0, 1);

        assert_eq!(output, [0.0]);
    }

    #[test]
    fn q4_0_kernel_nonzero_block() {
        let weight = q4_0_uniform_block(1.0, 9);
        let input_q8 = vec![1i8 as u8; 32];
        let input_scales = vec![1.0f32];

        let mut output = [0.0f32; 1];
        let kernel = Q4_0Kernel::new(&weight);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1, 0, 1);

        assert_eq!(output, [32.0]);
    }
}
