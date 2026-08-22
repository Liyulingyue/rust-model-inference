//! Q5_K super-block matmul kernel implementation.
//!
//! Q5_K uses 256-element super-blocks (176 bytes). Like Q4_K, the production
//! fast path is dequantize-to-f32-then-matmul because llama.cpp's Q5_K
//! kernel family does not have a native Q8-input variant.

use super::Kernel;

/// Q5_K weight buffer: 256-element super-blocks, 176 bytes each.
/// Placeholder for future Q5_K_M production path.
#[derive(Debug, Clone, Copy)]
pub struct Q5_KWeight<'a> {
    pub data: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

pub struct Q5_KKernel<'a> {
    pub weight: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> Q5_KKernel<'a> {
    pub fn new(weight: Q5_KWeight<'a>) -> Self {
        Self {
            weight: weight.data,
            n_in: weight.n_in,
            n_out: weight.n_out,
        }
    }
}

impl<'a> Kernel for Q5_KKernel<'a> {
    fn forward_prequantized(
        &self,
        _input_q8: &[u8],
        _input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        _ith: usize,
        _nth: usize,
    ) {
        // Placeholder: see `q4_k.rs` for the rationale. The production
        // Q5_K path is via `QuantizedLinear::forward_dequant`, not through
        // `LayerWeights`.
        let _ = (n_in, n_out);
        for o in output.iter_mut() {
            *o = 0.0;
        }
    }
}
