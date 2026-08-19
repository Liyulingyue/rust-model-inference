//! Q8_0 matmul kernel implementation.
//!
//! Phase 2.3: Second `Kernel` trait impl. The hot path in `app/text.rs` and
//! `app/embedding.rs` uses pre-quantized Q8 input — exposed as the inherent
//! method `forward_prequantized`. The trait's `forward` quantizes internally
//! for general use.

use super::Kernel;

/// Q8_0 matmul kernel: `output = weight × input`, both as Q8_0 blocks.
///
/// `weight` is laid out as Q8_0 blocks: `[2-byte F16 scale][32 bytes data]`,
/// row-major `[n_out rows × n_in cols]` (n_in must be a multiple of 32).
#[derive(Debug, Clone, Copy)]
pub struct Q8Kernel<'a> {
    pub weight: &'a [u8],
}

impl<'a> Q8Kernel<'a> {
    pub fn new(weight: &'a [u8]) -> Self {
        Self { weight }
    }
}

impl<'a> Kernel for Q8Kernel<'a> {
    /// General API: f32 input. Quantizes internally then dispatches to
    /// `forward_prequantized`. Allocates two scratch Vecs per call — for
    /// hot paths, prefer `forward_prequantized` directly.
    fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        let mut input_q8 = vec![0u8; n_in];
        let mut input_scales = vec![0.0f32; n_in / 32];
        crate::ops::quantize_q8_0_into(input, n_in, &mut input_q8, &mut input_scales);
        self.forward_prequantized(&input_q8, &input_scales, output, n_in, n_out);
    }
}

impl<'a> Q8Kernel<'a> {
    /// Hot path: pre-quantized Q8_0 input. No allocation.
    ///
    /// Mirrors `matmul_q8_0_quantized` in `super::super` but as an inherent
    /// method so it can be called without trait dispatch. The scalar fallback
    /// path is used here to avoid the GPU/AVX dispatch chain; the production
    /// path is the existing `ProcessedWeight::matmul` which we'll migrate
    /// in a later phase.
    pub fn forward_prequantized(
        &self,
        input_q8: &[u8],
        input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) {
        let blocks_per_row = n_in / 32;
        let row_stride = blocks_per_row * 34;
        debug_assert_eq!(self.weight.len() / 34 * 32, n_in * n_out / 32 * 32);

        for (out_idx, row) in (0..n_out).enumerate() {
            let row_off = row * row_stride;
            let mut sum = 0.0f32;
            for block in 0..blocks_per_row {
                let off = row_off + block * 34;
                let scale_bytes: [u8; 2] = [self.weight[off], self.weight[off + 1]];
                let wd = crate::ops::f16_to_f32(u16::from_le_bytes(scale_bytes));
                let qx = &self.weight[off + 2..off + 34];
                let qy = &input_q8[block * 32..(block + 1) * 32];
                let mut dot: i32 = 0;
                for lane in 0..32 {
                    dot += (qx[lane] as i8 as i32) * (qy[lane] as i8 as i32);
                }
                sum += wd * input_scales[block] * dot as f32;
            }
            output[out_idx] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a Q8_0 block for a single 32-element row with all values = `value`.
    /// Returns 34 bytes: 2-byte F16 scale + 32-byte data (all same int8 value).
    fn q8_0_uniform_row(value: i8, scale: f32) -> Vec<u8> {
        let mut row = Vec::with_capacity(34);
        let scale_bits = crate::ops::f32_to_f16(scale).to_le_bytes();
        row.extend_from_slice(&scale_bits);
        for _ in 0..32 {
            row.push(value as u8);
        }
        row
    }

    #[test]
    fn q8_kernel_uniform_weights_sum_to_dot_product() {
        // weight row: scale=1.0, all 32 values = 1 → dequantized = [1; 32]
        let mut weight = Vec::new();
        weight.extend(q8_0_uniform_row(1, 1.0));

        // input_q8 with scale=1.0, all values = 1 → dequantized = [1; 32]
        let input_q8 = vec![1u8; 32];
        let input_scales = vec![1.0f32];

        let mut output = [0.0f32; 1];
        let kernel = Q8Kernel::new(&weight);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1);

        // dot product of [1; 32] with [1; 32] = 32.0
        assert_eq!(output, [32.0]);
    }

    #[test]
    fn q8_kernel_multi_row() {
        // 2 rows, each [1; 32] dequantized
        let mut weight = Vec::new();
        weight.extend(q8_0_uniform_row(1, 1.0));
        weight.extend(q8_0_uniform_row(1, 1.0));

        let input_q8 = vec![1u8; 32];
        let input_scales = vec![1.0f32];

        let mut output = [0.0f32; 2];
        let kernel = Q8Kernel::new(&weight);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 2);

        assert_eq!(output, [32.0, 32.0]);
    }

    #[test]
    fn q8_kernel_scale_applied() {
        // weight with scale=0.5: dequantized values are 0.5
        let mut weight = Vec::new();
        weight.extend(q8_0_uniform_row(1, 0.5));

        let input_q8 = vec![1u8; 32];
        let input_scales = vec![1.0f32];

        let mut output = [0.0f32; 1];
        let kernel = Q8Kernel::new(&weight);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1);

        // [0.5; 32] · [1; 32] = 16.0
        assert_eq!(output, [16.0]);
    }

    #[test]
    fn q8_kernel_forward_quantizes_internally() {
        // weight row: scale=1.0, all 1
        let weight = q8_0_uniform_row(1, 1.0);
        // f32 input: all 1.0 → after quantize (scale ≈ 1/127), each lane
        // becomes ~127. dot product of dequantized values ≈ 32.0.
        let input = vec![1.0f32; 32];
        let mut output = [0.0f32; 1];

        let kernel = Q8Kernel::new(&weight);
        kernel.forward(&input, &mut output, 32, 1);

        // Tolerance: quantize + dequantize round-trip isn't perfectly lossless.
        // Actual error ~2^-9 = 0.001953, so 5e-3 gives comfortable margin.
        assert!((output[0] - 32.0).abs() < 5e-3, "got {}", output[0]);
    }
}