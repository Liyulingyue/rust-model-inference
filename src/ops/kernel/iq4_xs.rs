//! IQ4_XS block matmul kernel implementation.
//!
//! IQ4_XS uses 256-element super-blocks (136 bytes). The dequantization and
//! vec_dot functions in `ops::quant` are currently TODO; this kernel routes
//! to those stubs and surfaces a clear error when dispatched.

use super::Kernel;

pub struct IQ4XSKernel<'a> {
    pub weight: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> IQ4XSKernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 256;
    pub const BLOCK_BYTES: usize = 136;

    pub fn new(data: &'a [u8], n_in: usize, n_out: usize) -> Self {
        Self { weight: data, n_in, n_out }
    }
}

impl<'a> Kernel for IQ4XSKernel<'a> {
    fn forward_prequantized(
        &self,
        _input_q8: &[u8],
        _input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        let _ = (n_in, n_out, ith, nth);
        for o in output.iter_mut() {
            *o = 0.0;
        }
    }

    fn forward_prepared(
        &self,
        _input_f32: &[f32],
        _input_q8: &[u8],
        _input_scales: &[f32],
        _q8_k: Option<&[crate::ops::quant::BlockQ8K]>,
        _output: &mut [f32],
        _n_in: usize,
        _n_out: usize,
        _ith: usize,
        _nth: usize,
    ) {
        panic!("IQ4_XS kernel: vec_dot_iq4_xs_q8k_scalar not implemented; IQ4_XS inference unsupported");
    }
}