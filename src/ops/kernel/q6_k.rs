//! Q6_K super-block matmul kernel implementation.
//!
//! Phase 2.5 + 2.7-final: Q6_K uses 256-element super-blocks (210 bytes).

use super::Kernel;

#[derive(Debug, Clone, Copy)]
pub struct Q6_KKernel<'a> {
    pub weight: &'a [u8],
}

impl<'a> Q6_KKernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 256;
    pub const BLOCK_BYTES: usize = 210;

    pub fn new(data: &'a [u8], _n_in: usize, _n_out: usize) -> Self {
        Self { weight: data }
    }
}

impl<'a> Kernel for Q6_KKernel<'a> {
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
        matmul_q6_k_scalar_range(
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
        let row_bytes = n_in / crate::ops::quant::QK_K * crate::ops::quant::BLOCK_Q6K_SIZE;
        for out_idx in start..end {
            let offset = out_idx * row_bytes;
            output[out_idx] = crate::ops::quant::vec_dot_q6k_q8k(
                &self.weight[offset..offset + row_bytes],
                input_q8_k,
            );
        }
    }
}

/// Q6_K scalar matmul kernel. Phase 2.7-final: moved from `ops::matmul`.
pub fn matmul_q6_k_scalar_range(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    let row_stride = n_in / Q6_KKernel::BLOCK_ELEMENTS * Q6_KKernel::BLOCK_BYTES;
    let per_thread = n_out.div_ceil(nth);
    let my_start = ith * per_thread;
    let my_end = (my_start + per_thread).min(n_out);
    if my_start >= my_end {
        return;
    }
    // ponytail: scalar row dequantization; add a Q6_K × Q8_K SIMD kernel if profiling needs it.
    let input: Vec<f32> = input_q8
        .iter()
        .take(n_in)
        .enumerate()
        .map(|(i, &q)| q as i8 as f32 * input_scales[i / 32])
        .collect();
    let mut row = vec![0.0; n_in];
    for out_idx in my_start..my_end {
        let row_off = out_idx * row_stride;
        crate::ops::quant::dequantize_row_q6_k(
            &weight[row_off..row_off + row_stride],
            &mut row,
        );
        output[out_idx] = row.iter().zip(&input).map(|(x, y)| x * y).sum();
    }
}

// Phase 2.7-final cleanup: Q6_KWeight was removed in favor of inlining
// the (data: &[u8], n_in, n_out) triple directly into the kernel constructor.
// This `pub use` is kept as a no-op for downstream callers that previously
// imported it; it will be deleted in the next cleanup pass.
