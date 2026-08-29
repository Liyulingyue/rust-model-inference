//! Q4_1 block matmul kernel implementation.
//!
//! Q4_1 uses 32-element blocks with 20-byte layout
//! (2-byte F16 scale + 2-byte F16 min + 16-byte nibbles).
//!
//! Module structure:
//! - `scalar.rs` — scalar fallback (`matmul_q4_1_scalar_range`)
//! - `avx2.rs`    — AVX2 SIMD kernel (mirrors `q4_0::avx2` strategy)

use super::Kernel;
#[cfg(target_arch = "x86_64")]
pub mod avx2;
pub mod scalar;

pub use scalar::matmul_q4_1_scalar_range;

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
        #[cfg(target_arch = "x86_64")]
        {
            if crate::ops::has_avx2_fma() {
                let per_thread = (n_out + nth - 1) / nth;
                let my_start = ith * per_thread;
                let my_end = (my_start + per_thread).min(n_out);
                if my_start >= my_end {
                    return;
                }
                let my_out = &mut output[my_start..my_end];
                unsafe {
                    avx2::matmul_q4_1_vs_q8_0_avx2(
                        self.weight,
                        input_q8,
                        input_scales,
                        None,
                        my_out,
                        n_in,
                        my_start,
                        my_end,
                    );
                    return;
                }
            }
        }
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

        #[cfg(target_arch = "x86_64")]
        {
            if crate::ops::has_avx2_fma() {
                let per_thread = (n_out + nth - 1) / nth;
                let my_start = ith * per_thread;
                let my_end = (my_start + per_thread).min(n_out);
                if my_start >= my_end {
                    return;
                }
                let my_out = &mut output[my_start..my_end];
                unsafe {
                    avx2::matmul_q4_1_vs_q8_0_avx2(
                        self.weight,
                        input_q8,
                        input_scales,
                        Some(&input_sums),
                        my_out,
                        n_in,
                        my_start,
                        my_end,
                    );
                    return;
                }
            }
        }
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
