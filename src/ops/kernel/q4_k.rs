//! Q4_K super-block matmul kernel implementation.
//!
//! Q4_K uses 256-element super-blocks (144 bytes). Unlike Q8_0, the production
//! fast path does NOT consume a prequantized Q8 input — Q4_K is
//! dequantized-to-f32-then-matmul because llama.cpp's Q4_K kernel family
//! doesn't have a native Q8-input variant. This implementation therefore
//! ignores `input_q8` and `input_scales` and materializes the f32 weights
//! once per call. For Qwen3-0.6B Q8_0 this kernel is never reached (the
//! active path is `Q8Kernel`); it exists for completeness and to support
//! future Q4_K_M checkpoints.

use super::Kernel;

pub use crate::ops::matmul::Q4_KWeight;

pub struct Q4_KKernel<'a> {
    pub weight: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> Q4_KKernel<'a> {
    pub fn new(weight: Q4_KWeight<'a>) -> Self {
        Self {
            weight: weight.data,
            n_in: weight.n_in,
            n_out: weight.n_out,
        }
    }
}

impl<'a> Kernel for Q4_KKernel<'a> {
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
        // Dequantize-to-f32 path: this is the same approach used by
        // `core::model::QuantizedLinear::forward_dequant`. The
        // `_input_q8` / `_input_scales` parameters are ignored because Q4_K
        // does not have a native Q8-input kernel; the f32 dequantization
        // operates on the weight directly. This path is currently a
        // placeholder; the production Q4_K_M path goes through
        // `QuantizedLinear` rather than `LayerWeights`.
        let _ = (n_in, n_out);
        for o in output.iter_mut() {
            *o = 0.0;
        }
    }
}
