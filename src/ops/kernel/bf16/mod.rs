//! BF16 matmul kernel implementation.
//!
//! BF16 weight layout: 2 bytes per element, F32-compatible (upper 16 bits
//! of an F32 representation). Conversion BF16 → F32 is a zero-cost u16→u32
//! shift-and-reinterpret.
//!
//! Module structure:
//! - `scalar.rs` — scalar fallback for both F32-input and Q8-input paths.
//! - `avx2.rs`    — AVX2+FMA SIMD for the F32-input hot path (used by
//!   Qwen3 BF16 inference). Q8-input path remains scalar (rarely hot).

use super::Kernel;
#[cfg(target_arch = "x86_64")]
pub mod avx2;
pub mod scalar;

#[derive(Debug, Clone, Copy)]
pub struct BF16Kernel<'a> {
    pub weight: &'a [u8],
}

impl<'a> BF16Kernel<'a> {
    pub fn new(weight: &'a [u8]) -> Self {
        Self { weight }
    }

    pub fn element_count(&self) -> usize {
        self.weight.len() / 2
    }

    pub(crate) fn row_range(n_out: usize, ith: usize, nth: usize) -> (usize, usize) {
        let nth = nth.max(1);
        let start = n_out.saturating_mul(ith) / nth;
        let end = n_out.saturating_mul(ith.saturating_add(1)) / nth;
        (start, end)
    }

    fn forward_f32_rows(
        &self,
        input: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if n_in % 8 == 0 && crate::ops::has_avx2_fma() {
                let (start, end) = Self::row_range(n_out, ith, nth);
                if end > start {
                    let my_out = &mut output[start..end];
                    unsafe {
                        avx2::matmul_bf16_vs_f32_avx2(
                            self.weight,
                            input,
                            my_out,
                            n_in,
                            start,
                            end,
                        );
                        return;
                    }
                }
            }
        }
        scalar::forward_f32_rows_scalar(
            self.weight,
            input,
            output,
            n_in,
            n_out,
            ith,
            nth,
        );
    }

    fn forward_q8_rows(
        &self,
        input_q8: &[u8],
        input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        scalar::forward_q8_rows_scalar(
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

impl<'a> Kernel for BF16Kernel<'a> {
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
        debug_assert!(input_q8.len() >= n_in);
        debug_assert!(input_scales.len() >= n_in.div_ceil(32));
        debug_assert!(self.weight.len() >= n_in * n_out * 2);
        self.forward_q8_rows(input_q8, input_scales, output, n_in, n_out, ith, nth);
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
        if input_f32.len() >= n_in {
            self.forward_f32_rows(&input_f32[..n_in], output, n_in, n_out, ith, nth);
        } else {
            self.forward_prequantized(
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

    fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        self.forward_f32_rows(&input[..n_in], output, n_in, n_out, 0, 1);
    }

    fn forward_batched(
        &self,
        input: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) {
        let n_tokens = input.len() / n_in;
        debug_assert_eq!(input.len(), n_tokens * n_in);
        debug_assert_eq!(output.len(), n_tokens * n_out);
        for token in 0..n_tokens {
            self.forward(
                &input[token * n_in..(token + 1) * n_in],
                &mut output[token * n_out..(token + 1) * n_out],
                n_in,
                n_out,
            );
        }
    }

    fn embedding_lookup(&self, token_id: u32, n_embd: usize, out: &mut [f32]) {
        crate::ops::embedding::embedding_lookup_bf16(self.weight, token_id, n_embd, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&value| crate::ops::f32_to_bf16(value).to_le_bytes())
            .collect()
    }

    #[test]
    fn bf16_kernel_f32_matmul() {
        let weight = bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let kernel = BF16Kernel::new(&weight);
        let mut output = [0.0f32; 2];
        kernel.forward(&[1.0, 2.0, 3.0], &mut output, 3, 2);
        assert_eq!(output, [14.0, 32.0]);
    }

    #[test]
    fn bf16_kernel_thread_partition_uses_absolute_rows() {
        let weight = bf16_bytes(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]);
        let kernel = BF16Kernel::new(&weight);
        let input = [1.0f32; 3];
        let mut output = [0.0f32; 4];
        kernel.forward_prepared(&input, &[], &[], None, &mut output, 3, 4, 0, 2);
        assert_eq!(output, [3.0, 6.0, 0.0, 0.0]);
        output.fill(0.0);
        kernel.forward_prepared(&input, &[], &[], None, &mut output, 3, 4, 1, 2);
        assert_eq!(output, [0.0, 0.0, 9.0, 12.0]);
    }

    #[test]
    fn bf16_kernel_embedding_lookup() {
        let weight = bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let kernel = BF16Kernel::new(&weight);
        let mut output = [0.0f32; 3];
        kernel.embedding_lookup(1, 3, &mut output);
        assert_eq!(output, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn bf16_kernel_q8_matmul() {
        let weight = bf16_bytes(&[1.0, 2.0, 3.0]);
        let kernel = BF16Kernel::new(&weight);
        let mut input = [0.0f32; 32];
        input[0] = 1.0;
        input[1] = 2.0;
        input[2] = 3.0;
        let mut input_q8 = vec![0u8; 32];
        let mut input_scales = [0.0f32; 1];
        crate::ops::quantize_q8_0_into(&input, 32, &mut input_q8, &mut input_scales);
        let mut output = [0.0f32; 1];
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 3, 1, 0, 1);
        assert!(output[0].is_finite());
        assert!(output[0] > 0.0);
    }
}
