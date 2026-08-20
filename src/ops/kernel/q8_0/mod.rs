//! Q8_0 matmul kernel implementation.
//!
//! Phase 2.3 + 2.7-final: Q8_0 is the production hot-path kernel for Qwen3-0.6B.
//! The `forward_prequantized` method accepts a Q8-prequantized input and an
//! `ith`/`nth` row partition so callers can dispatch this inside a
//! `pool.compute` closure for thread-parallel matmul.

use super::Kernel;
pub mod avx2;
pub mod dispatch;
pub mod neon;
pub mod scalar;

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
        // Delegate to the production matmul which handles AVX2/NEON/GPU
        // dispatch. This is the kernel the Qwen3 hot path actually wants.
        crate::ops::matmul_q8_0_quantized_parallel_rows(
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
        let mut weight = Vec::new();
        weight.extend(q8_0_uniform_row(1, 1.0));

        let input_q8 = vec![1u8; 32];
        let input_scales = vec![1.0f32];

        let mut output = [0.0f32; 1];
        let kernel = Q8Kernel::new(&weight);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1, 0, 1);

        assert_eq!(output, [32.0]);
    }

    #[test]
    fn q8_kernel_multi_row() {
        let mut weight = Vec::new();
        weight.extend(q8_0_uniform_row(1, 1.0));
        weight.extend(q8_0_uniform_row(1, 1.0));

        let input_q8 = vec![1u8; 32];
        let input_scales = vec![1.0f32];

        let mut output = [0.0f32; 2];
        let kernel = Q8Kernel::new(&weight);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 2, 0, 1);

        assert_eq!(output, [32.0, 32.0]);
    }

    #[test]
    fn q8_kernel_scale_applied() {
        let mut weight = Vec::new();
        weight.extend(q8_0_uniform_row(1, 0.5));

        let input_q8 = vec![1u8; 32];
        let input_scales = vec![1.0f32];

        let mut output = [0.0f32; 1];
        let kernel = Q8Kernel::new(&weight);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1, 0, 1);

        assert_eq!(output, [16.0]);
    }

    #[test]
    fn q8_kernel_forward_quantizes_internally() {
        let weight = q8_0_uniform_row(1, 1.0);
        let input = vec![1.0f32; 32];
        let mut output = [0.0f32; 1];

        let kernel = Q8Kernel::new(&weight);
        kernel.forward(&input, &mut output, 32, 1);

        assert!((output[0] - 32.0).abs() < 5e-3, "got {}", output[0]);
    }

    #[test]
    fn q8_kernel_thread_partition() {
        // 4 rows; simulate 2 threads each writing to its own partition
        // of the same full output slice.
        let mut weight = Vec::new();
        for _ in 0..4 {
            weight.extend(q8_0_uniform_row(1, 1.0));
        }

        let input_q8 = vec![1u8; 32];
        let input_scales = vec![1.0f32];
        let kernel = Q8Kernel::new(&weight);

        // Full 4-element output; threads 0 and 1 write to disjoint halves.
        let mut out_a = [0.0f32; 4];
        kernel.forward_prequantized(&input_q8, &input_scales, &mut out_a, 32, 4, 0, 2);
        // Reset and have thread 1 only (ith=1, nth=2) write its half.
        let mut out_b = [0.0f32; 4];
        kernel.forward_prequantized(&input_q8, &input_scales, &mut out_b, 32, 4, 1, 2);

        assert_eq!(out_a, [32.0, 32.0, 0.0, 0.0]);
        // Thread 1 writes to indices 2..4.
        assert_eq!(out_b[0..2], [0.0, 0.0]);
        assert_eq!(out_b[2..4], [32.0, 32.0]);
    }
}