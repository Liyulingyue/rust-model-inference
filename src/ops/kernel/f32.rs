//! F32 matmul kernel implementation.
//!
//! Phase 2.2: First `Kernel` trait impl. The semantics follow the standard
//! `y[i] = Σ_w[i][k] * x[k]` definition; input is required (unlike the
//! stub `matmul_f32_scalar_range` in `super::super` which only sums
//! rows). The stub remains untouched for backward compat.

use super::Kernel;

/// F32 matmul kernel: `output = weight × input`.
///
/// `weight` is laid out row-major as `[n_out rows × n_in cols]`.
#[derive(Debug, Clone, Copy)]
pub struct F32Kernel<'a> {
    pub weight: &'a [f32],
}

impl<'a> F32Kernel<'a> {
    pub fn new(weight: &'a [f32]) -> Self {
        Self { weight }
    }
}

impl<'a> Kernel for F32Kernel<'a> {
    #[inline]
    fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        debug_assert!(input.len() >= n_in);
        debug_assert!(output.len() >= n_out);
        debug_assert!(self.weight.len() >= n_out * n_in);

        for (out_idx, row) in (0..n_out).enumerate() {
            let row_off = row * n_in;
            let mut sum = 0.0f32;
            for col in 0..n_in {
                sum += self.weight[row_off + col] * input[col];
            }
            output[out_idx] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_kernel_row_sum() {
        // 2x3 weight: [[1,2,3],[4,5,6]]
        let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = [1.0f32, 1.0, 1.0];
        let mut output = [0.0f32; 2];

        let kernel = F32Kernel::new(&w);
        kernel.forward(&input, &mut output, 3, 2);

        assert_eq!(output, [6.0, 15.0]); // [1+2+3, 4+5+6]
    }

    #[test]
    fn f32_kernel_weighted_input() {
        // 2x3 weight: [[1,2,3],[4,5,6]]
        let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = [10.0f32, 20.0, 30.0];
        let mut output = [0.0f32; 2];

        let kernel = F32Kernel::new(&w);
        kernel.forward(&input, &mut output, 3, 2);

        // [1*10+2*20+3*30, 4*10+5*20+6*30] = [140, 320]
        assert_eq!(output, [140.0, 320.0]);
    }

    #[test]
    fn f32_kernel_batched_default_loop() {
        // 2x2 weight, 3 tokens
        let w = [1.0f32, 2.0, 3.0, 4.0];
        // tokens: [1,1], [2,2], [3,3]
        let input = [1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0];
        let mut output = [0.0f32; 6];

        let kernel = F32Kernel::new(&w);
        kernel.forward_batched(&input, &mut output, 2, 2);

        // row 0: [1*1+2*1, 1*2+2*2, 1*3+2*3] = [3, 6, 9]
        // row 1: [3*1+4*1, 3*2+4*2, 3*3+4*3] = [7, 14, 21]
        assert_eq!(output, [3.0, 7.0, 6.0, 14.0, 9.0, 21.0]);
    }
}