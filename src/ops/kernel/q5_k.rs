//! Q5_K super-block matmul kernel implementation.
//!
//! Q5_K uses 256-element super-blocks (176 bytes). The hot path uses the
//! native Q5_K × Q8_K SIMD kernel (`vec_dot_q5k_q8k`), mirroring Q4_K.

use super::Kernel;

pub struct Q5_KKernel<'a> {
    pub weight: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> Q5_KKernel<'a> {
    pub fn new(data: &'a [u8], n_in: usize, n_out: usize) -> Self {
        Self { weight: data, n_in, n_out }
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
        ith: usize,
        nth: usize,
    ) {
        let _ = (n_in, n_out);
        let per_thread = n_out.div_ceil(nth);
        let start = ith * per_thread;
        let end = (start + per_thread).min(n_out);
        if start >= end {
            return;
        }

        let owned_q8k;
        let input_q8_k: &[crate::ops::quant::BlockQ8K] = if _input_q8.is_empty() && _input_scales.is_empty() {
            owned_q8k = crate::ops::quant::quantize_row_q8_k(&[]);
            &owned_q8k
        } else {
            let mut f32 = vec![0.0f32; n_in];
            for (i, &q) in _input_q8.iter().take(n_in).enumerate() {
                f32[i] = q as i8 as f32 * _input_scales[i / 32];
            }
            owned_q8k = crate::ops::quant::quantize_row_q8_k(&f32);
            &owned_q8k
        };

        let row_bytes = n_in / crate::ops::quant::QK_K * crate::ops::quant::BLOCK_Q5K_SIZE;
        for out_idx in start..end {
            let offset = out_idx * row_bytes;
            output[out_idx] = crate::ops::quant::vec_dot_q5k_q8k(
                &self.weight[offset..offset + row_bytes],
                input_q8_k,
            );
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

        let owned_q8k;
        let input_q8_k: &[crate::ops::quant::BlockQ8K] = match q8_k {
            Some(buf) => buf,
            None => {
                owned_q8k = crate::ops::quant::quantize_row_q8_k(&input_f32[..n_in]);
                &owned_q8k
            }
        };

        let row_bytes = n_in / crate::ops::quant::QK_K * crate::ops::quant::BLOCK_Q5K_SIZE;
        for out_idx in start..end {
            let offset = out_idx * row_bytes;
            output[out_idx] = crate::ops::quant::vec_dot_q5k_q8k(
                &self.weight[offset..offset + row_bytes],
                input_q8_k,
            );
        }
    }

    fn embedding_lookup(&self, token_id: u32, n_embd: usize, out: &mut [f32]) {
        let row_bytes = n_embd / crate::ops::quant::QK_K * crate::ops::quant::BLOCK_Q5K_SIZE;
        let offset = token_id as usize * row_bytes;
        crate::ops::quant::dequantize_row_q5_k(
            &self.weight[offset..offset + row_bytes],
            out,
        );
    }
}
