//! Q4_0 block matmul kernel implementation.
//!
//! Phase 2.5 + 2.7-final: Q4_0 uses 32-element blocks with 18-byte layout
//! (2-byte F16 scale + 16-byte nibbles). The `forward_prequantized` method
//! accepts a Q8-prequantized input plus `ith`/`nth` row partitioning so the
//! kernel can be dispatched inside a `pool.compute` closure.
//!
//! Module structure:
//! - `scalar.rs` — scalar fallback (`matmul_q4_0_scalar_range`). Hot path today.

use super::Kernel;
pub mod scalar;

pub use scalar::matmul_q4_0_scalar_range;

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