//! Q4_K super-block matmul kernel implementation.
//!
//! Q4_K uses 256-element super-blocks (144 bytes).

use super::Kernel;

pub struct Q4_KKernel<'a> {
    pub weight: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> Q4_KKernel<'a> {
    pub fn new(data: &'a [u8], n_in: usize, n_out: usize) -> Self {
        Self { weight: data, n_in, n_out }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_k_embedding_lookup_dequantizes_a_row() {
        let mut weight = vec![0u8; crate::ops::quant::BLOCK_Q4K_SIZE];
        weight[..2].copy_from_slice(&crate::ops::f32_to_f16(1.0).to_le_bytes());
        for scale in &mut weight[4..16] {
            *scale = 1;
        }
        weight[16..].fill(0x11);

        let kernel = Q4_KKernel::new(&weight, crate::ops::quant::QK_K, 1);
        let mut output = vec![0.0f32; crate::ops::quant::QK_K];
        kernel.embedding_lookup(0, crate::ops::quant::QK_K, &mut output);

        assert!(output.iter().all(|value| *value == 1.0));
    }
}

impl<'a> Kernel for Q4_KKernel<'a> {
    fn embedding_lookup(&self, token_id: u32, n_embd: usize, out: &mut [f32]) {
        crate::ops::embedding::embedding_lookup_q4_k(self.weight, token_id, n_embd, out);
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
        let per_thread = n_out.div_ceil(nth);
        let start = ith * per_thread;
        let end = (start + per_thread).min(n_out);
        if start >= end {
            return;
        }

        // ponytail: scalar row dequantization; add a Q4_K × Q8_K SIMD kernel if profiling needs it.
        let input: Vec<f32> = input_q8
            .iter()
            .take(n_in)
            .enumerate()
            .map(|(i, &q)| q as i8 as f32 * input_scales[i / 32])
            .collect();
        let row_bytes = n_in / crate::ops::quant::QK_K * crate::ops::quant::BLOCK_Q4K_SIZE;
        let mut row = vec![0.0; n_in];
        for out_idx in start..end {
            let offset = out_idx * row_bytes;
            crate::ops::quant::dequantize_row_q4_k(
                &self.weight[offset..offset + row_bytes],
                &mut row,
            );
            output[out_idx] = row.iter().zip(&input).map(|(x, y)| x * y).sum();
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

        // Use caller-prepared Q8_K if provided (shared across threads);
        // otherwise each thread re-quantizes the same input (legacy path).
        let owned_q8k;
        let input_q8_k: &[crate::ops::quant::BlockQ8K] = match q8_k {
            Some(buf) => buf,
            None => {
                owned_q8k = crate::ops::quant::quantize_row_q8_k(&input_f32[..n_in]);
                &owned_q8k
            }
        };
        let row_bytes = n_in / crate::ops::quant::QK_K * crate::ops::quant::BLOCK_Q4K_SIZE;
        for out_idx in start..end {
            let offset = out_idx * row_bytes;
            output[out_idx] = crate::ops::quant::vec_dot_q4k_q8k(
                &self.weight[offset..offset + row_bytes],
                input_q8_k,
            );
        }
    }
}
