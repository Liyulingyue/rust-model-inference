//! Q4_1 block matmul kernel implementation.
//!
//! Phase 2.5 + 2.7-final: Q4_1 uses 32-element blocks with 20-byte layout
//! (2-byte F16 scale + 2-byte F16 min + 16-byte nibbles).

use super::Kernel;

#[derive(Debug, Clone, Copy)]
pub struct Q4_1Kernel<'a> {
    pub weight: &'a [u8],
}

impl<'a> Q4_1Kernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 32;
    pub const BLOCK_BYTES: usize = 20;

    pub fn new(data: &'a [u8], _n_in: usize, _n_out: usize) -> Self {
        Self { weight: data }
    }
}

impl<'a> Kernel for Q4_1Kernel<'a> {
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
        matmul_q4_1_scalar_range(
            self.weight,
            input_q8,
            input_scales,
            None,
            output,
            n_in,
            n_out,
            ith,
            nth,
        );
    }

    fn forward_prepared(
        &self,
        input_f32: &[f32],
        input_q8: &[u8],
        input_scales: &[f32],
        _q8_k: Option<&[crate::ops::quant::BlockQ8K]>,
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        let input_sums: Vec<f32> = input_f32[..n_in]
            .chunks_exact(32)
            .zip(input_q8[..n_in].chunks_exact(32))
            .map(|(values, quantized)| {
                let amax = values
                    .iter()
                    .fold(0.0f32, |current, value| current.max(value.abs()));
                let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
                let sum = quantized
                    .iter()
                    .map(|&value| i32::from(value as i8))
                    .sum::<i32>();
                crate::ops::f16_to_f32(crate::ops::f32_to_f16(sum as f32 * scale))
            })
            .collect();
        matmul_q4_1_scalar_range(
            self.weight,
            input_q8,
            input_scales,
            Some(&input_sums),
            output,
            n_in,
            n_out,
            ith,
            nth,
        );
    }
}

/// Q4_1 scalar matmul kernel. Phase 2.7-final: moved from `ops::matmul`.
/// Each Q4_1 block holds 32 elements (20 bytes: F16 scale + F16 min +
/// 16-byte nibbles). Scalar baseline; AVX2/NEON can be added here.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn q4_1_uniform_block(scale: f32, min: f32, nibble: u8) -> Vec<u8> {
        assert!(nibble < 16);
        let mut block = Vec::with_capacity(20);
        block.extend_from_slice(&crate::ops::f32_to_f16(scale).to_le_bytes());
        block.extend_from_slice(&crate::ops::f32_to_f16(min).to_le_bytes());
        let packed = nibble | (nibble << 4);
        for _ in 0..16 {
            block.push(packed);
        }
        block
    }

    #[test]
    fn q4_1_kernel_min_contribution_only() {
        let weight = q4_1_uniform_block(0.0, 1.0, 0);
        let input_q8 = vec![1i8 as u8; 32];
        let input_scales = vec![1.0f32];

        let mut output = [0.0f32; 1];
        let kernel = Q4_1Kernel::new(&weight, 32, 1);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1, 0, 1);

        assert_eq!(output, [32.0]);
    }

    #[test]
    fn q4_1_kernel_dot_product_only() {
        let weight = q4_1_uniform_block(1.0, 0.0, 1);
        let input_q8 = vec![1i8 as u8; 32];
        let input_scales = vec![1.0f32];

        let mut output = [0.0f32; 1];
        let kernel = Q4_1Kernel::new(&weight, 32, 1);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1, 0, 1);

        assert_eq!(output, [32.0]);
    }
}
