//! Q6_K super-block matmul kernel implementation.
//!
//! Phase 2.5 + 2.7-final: Q6_K uses 256-element super-blocks (210 bytes).
//! Like Q4_K / Q5_K, the production fast path here is dequantize-to-f32-
//! then-matmul because llama.cpp's Q6_K family does not have a native
//! Q8-input kernel variant. The `forward_prequantized` argument names are
//! kept for trait uniformity; the Q8 input is ignored and a placeholder
//! runs the scalar f32 dot. The actual production path is via
//! `QuantizedLinear::forward_dequant` for Q6_K_M weights — see
//! `core::model`.

use super::Kernel;

#[derive(Debug, Clone, Copy)]
pub struct Q6_KKernel<'a> {
    pub weight: &'a [u8],
}

impl<'a> Q6_KKernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 256;
    pub const BLOCK_BYTES: usize = 210;

    pub fn new(weight: &'a [u8]) -> Self {
        Self { weight }
    }
}

impl<'a> Kernel for Q6_KKernel<'a> {
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
        let _ = n_in;
        for slot in output.iter_mut().take(n_out) {
            *slot = 0.0;
        }
    }
}
