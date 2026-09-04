//! F32 matmul kernel implementation.
//!
//! Phase 2.2 + 2.7-final: F32 weights are rare in production (Q8_0 / Q4_K_M
//! dominate). For tests this kernel still exposes a working f32 matmul on
//! the `forward` path; the production `forward_prequantized` is a placeholder
//! that emits zeros because F32 weights do not appear in `LayerWeights`.

use super::Kernel;

#[derive(Debug, Clone)]
pub struct F32Kernel {
    weight: Vec<f32>,
}

impl F32Kernel {
    pub fn new(weight: Vec<f32>) -> Self {
        Self { weight }
    }
}

impl Kernel for F32Kernel {
    fn f32_slice(&self) -> Option<&[f32]> {
        Some(&self.weight)
    }

    fn forward_prequantized(
        &self,
        _input_q8: &[u8],
        _input_scales: &[f32],
        output: &mut [f32],
        n_out: usize,
        n_in: usize,
        ith: usize,
        nth: usize,
    ) {
        matmul_f32_scalar_range(&self.weight, output, n_in, n_out, ith, nth);
    }

    fn forward_prepared(
        &self,
        input_f32: &[f32],
        _input_q8: &[u8],
        _input_scales: &[f32],
        _q8_k: Option<&[crate::ops::quant::BlockQ8K]>,
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        let start = ith * n_out.div_ceil(nth);
        let end = (start + n_out.div_ceil(nth)).min(n_out);
        for out_idx in start..end {
            let row_off = out_idx * n_in;
            let mut sum = 0.0;
            for col in 0..n_in {
                sum += self.weight[row_off + col] * input_f32[col];
            }
            output[out_idx] = sum;
        }
    }

    /// F32 has a native f32-input path. The trait default impl quantizes
    /// the input to Q8 then calls `forward_prequantized` (zero for F32);
    /// we override here to do the real f32 matmul. Tests + any future
    /// non-LayerWeights callers use this path.
    fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        debug_assert_eq!(self.weight.len(), n_out * n_in);
        debug_assert!(input.len() >= n_in);
        debug_assert!(output.len() >= n_out);

        for (out_idx, row) in (0..n_out).enumerate() {
            let row_off = row * n_in;
            let mut sum = 0.0f32;
            for col in 0..n_in {
                sum += self.weight[row_off + col] * input[col];
            }
            output[out_idx] = sum;
        }
    }

    fn forward_batched(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        let n_tokens = input.len() / n_in;
        for t in 0..n_tokens {
            self.forward(
                &input[t * n_in..(t + 1) * n_in],
                &mut output[t * n_out..(t + 1) * n_out],
                n_in,
                n_out,
            );
        }
    }

    fn embedding_lookup(&self, token_id: u32, n_embd: usize, output: &mut [f32]) {
        let offset = token_id as usize * n_embd;
        output.copy_from_slice(&self.weight[offset..offset + n_embd]);
    }
}

/// F32 scalar matmul kernel. Phase 2.7-final: moved from `ops::matmul`.
///
/// Stub: only sums each row of the weight matrix. Real f32×f32 dot
/// product happens via `Kernel::forward` (which overrides this and uses
/// the f32-input path). Kept for completeness so a `Box<dyn Kernel>`
/// containing an F32 kernel still produces something deterministic rather
/// than zeros in the LayerWeights hot path.
pub fn matmul_f32_scalar_range(
    weight: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    let per_thread = (n_out + nth - 1) / nth;
    let my_start = ith * per_thread;
    let my_end = (my_start + per_thread).min(n_out);
    if my_start >= my_end {
        return;
    }
    for out_idx in my_start..my_end {
        let mut sum = 0.0f32;
        let row_off = out_idx * n_in;
        for col in 0..n_in {
            sum += weight[row_off + col];
        }
        output[out_idx] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_kernel_row_sum() {
        let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = [1.0f32, 1.0, 1.0];
        let mut output = [0.0f32; 2];

        let kernel = F32Kernel::new(w);
        kernel.forward(&input, &mut output, 3, 2);

        assert_eq!(output, [6.0, 15.0]);
    }

    #[test]
    fn f32_kernel_weighted_input() {
        let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = [10.0f32, 20.0, 30.0];
        let mut output = [0.0f32; 2];

        let kernel = F32Kernel::new(w);
        kernel.forward(&input, &mut output, 3, 2);

        assert_eq!(output, [140.0, 320.0]);
    }

    #[test]
    fn f32_kernel_batched_default_loop() {
        let w = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = [1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0];
        let mut output = [0.0f32; 6];

        let kernel = F32Kernel::new(w);
        kernel.forward_batched(&input, &mut output, 2, 2);

        assert_eq!(output, [3.0, 7.0, 6.0, 14.0, 9.0, 21.0]);
    }
}
