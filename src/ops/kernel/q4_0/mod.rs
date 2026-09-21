//! Q4_0 block matmul kernel implementation.
//!
//! Phase 2.5 + 2.7-final: Q4_0 uses 32-element blocks with 18-byte layout
//! (2-byte F16 scale + 16-byte nibbles). The `forward_prequantized` method
//! accepts a Q8-prequantized input plus `ith`/`nth` row partitioning so the
//! kernel can be dispatched inside a `pool.compute` closure.
//!
//! Module structure:
//! - `scalar.rs` — scalar fallback (`matmul_q4_0_scalar_range`).
//! - `avx2.rs`   — AVX2 + `_mm256_maddubs_epi16` fast path (x86_64).
//! - `neon.rs`   — ARMv8.4-A dot-product fast path (aarch64).

use super::Kernel;
#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod scalar;

pub use scalar::matmul_q4_0_scalar_range;

#[derive(Debug, Clone, Copy)]
pub struct Q4_0Kernel<'a> {
    pub weight: &'a [u8],
}

impl<'a> Q4_0Kernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 32;
    pub const BLOCK_BYTES: usize = 18;

    pub fn new(data: &'a [u8], _n_in: usize, _n_out: usize) -> Self {
        Self { weight: data }
    }
}

impl<'a> Kernel for Q4_0Kernel<'a> {
    #[cfg(target_arch = "aarch64")]
    fn scalar_q4_0_bytes(&self) -> Option<&[u8]> {
        Some(self.weight)
    }

    fn weight_bytes(&self) -> Option<&[u8]> {
        Some(self.weight)
    }

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
                    avx2::matmul_q4_0_vs_q8_0_avx2(
                        self.weight,
                        input_q8,
                        input_scales,
                        my_out,
                        n_in,
                        my_start,
                        my_end,
                    );
                    return;
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            let per_thread = (n_out + nth - 1) / nth;
            let my_start = ith * per_thread;
            let my_end = (my_start + per_thread).min(n_out);
            if my_start < my_end {
                unsafe {
                    neon::matmul_q4_0_vs_q8_0_neon(
                        self.weight,
                        input_q8,
                        input_scales,
                        &mut output[my_start..my_end],
                        n_in,
                        my_start,
                        my_end,
                    );
                }
                return;
            }
        }
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

    fn embedding_lookup(&self, token_id: u32, n_embd: usize, out: &mut [f32]) {
        crate::ops::embedding::embedding_lookup_q4_0(self.weight, token_id, n_embd, out);
    }
}

#[cfg(target_arch = "aarch64")]
#[path = "tests_neon.rs"]
mod tests_neon;

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
        let kernel = Q4_0Kernel::new(&weight, 32, 1);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1, 0, 1);

        assert_eq!(output, [0.0]);
    }

    #[test]
    fn q4_0_kernel_nonzero_block() {
        let weight = q4_0_uniform_block(1.0, 9);
        let input_q8 = vec![1i8 as u8; 32];
        let input_scales = vec![1.0f32];

        let mut output = [0.0f32; 1];
        let kernel = Q4_0Kernel::new(&weight, 32, 1);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1, 0, 1);

        assert_eq!(output, [32.0]);
    }
}
