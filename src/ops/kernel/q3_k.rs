//! Q3_K super-block matmul kernel implementation.
//!
//! Q3_K uses 256-element super-blocks (110 bytes). The hot path uses the
//! scalar Q3_K × Q8_K kernel (`vec_dot_q3k_q8k_scalar`); an AVX2 SIMD
//! variant can be added later (mirrors the q5_k / q6_k layout).

use super::Kernel;

pub struct Q3_KKernel<'a> {
    pub weight: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> Q3_KKernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 256;
    pub const BLOCK_BYTES: usize = 110;

    pub fn new(data: &'a [u8], n_in: usize, n_out: usize) -> Self {
        Self { weight: data, n_in, n_out }
    }
}

impl<'a> Kernel for Q3_KKernel<'a> {
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
        let per_thread = n_out.div_ceil(nth);
        let start = ith * per_thread;
        let end = (start + per_thread).min(n_out);
        if start >= end {
            return;
        }

        let row_bytes = n_in / crate::ops::quant::QK_K * crate::ops::quant::BLOCK_Q3K_SIZE;
        let mut row = vec![0.0f32; n_in];
        for out_idx in start..end {
            let offset = out_idx * row_bytes;
            crate::ops::quant::dequantize_row_q3_k(
                &self.weight[offset..offset + row_bytes],
                &mut row,
            );
            output[out_idx] = 0.0;
        }
    }

    fn forward_prepared(
        &self,
        input_f32: &[f32],
        _input_q8: &[u8],
        _input_scales: &[f32],
        q8_k: Option<&[crate::ops::quant::BlockQ8K]>,
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        let per_thread = n_out.div_ceil(nth);
        let start = ith * per_thread;
        let end = (start + per_thread).min(n_out);
        if start >= end {
            return;
        }

        let row_bytes = n_in / crate::ops::quant::QK_K * crate::ops::quant::BLOCK_Q3K_SIZE;
        for out_idx in start..end {
            let offset = out_idx * row_bytes;
            if let Some(q8k_buf) = q8_k {
                output[out_idx] = crate::ops::quant::vec_dot_q3k_q8k(
                    &self.weight[offset..offset + row_bytes],
                    q8k_buf,
                );
            } else {
                let owned_q8k = crate::ops::quant::quantize_row_q8_k(&input_f32[..n_in]);
                output[out_idx] = crate::ops::quant::vec_dot_q3k_q8k(
                    &self.weight[offset..offset + row_bytes],
                    &owned_q8k,
                );
            }
        }
    }
}