//! Q1_0 block matmul kernel implementation.
//!
//! Q1_0 uses 128-element blocks with 18-byte layout
//! (2-byte F16 scale + 16-byte bitfield, 1 bit per element).
//! Each element dequantizes to `bit ? d : -d`.

use super::Kernel;
pub mod scalar;

pub use scalar::matmul_q1_0_scalar_range;

#[derive(Debug, Clone, Copy)]
pub struct Q1_0Kernel<'a> {
    pub weight: &'a [u8],
}

impl<'a> Q1_0Kernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 128;
    pub const BLOCK_BYTES: usize = 18;

    pub fn new(data: &'a [u8], _n_in: usize, _n_out: usize) -> Self {
        Self { weight: data }
    }
}

impl<'a> Kernel for Q1_0Kernel<'a> {
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
        matmul_q1_0_scalar_range(
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
        crate::ops::embedding::embedding_lookup_q1_0(self.weight, token_id, n_embd, out);
    }
}
