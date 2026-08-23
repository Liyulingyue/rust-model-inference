//! Core matmul kernel trait.
//!
//! Every quantized weight type implements this trait via the dispatch
//! pattern `match self { ... }`. The hot-path entry is `forward_prequantized`,
//! which takes a Q8-prequantized input and a row-partition `[ith, nth)` so
//! the caller can dispatch the call inside a `pool.compute` closure for
//! thread-parallel matmul. Pass `ith=0, nth=1` for the scalar single-token
//! path.
//!
//! Default implementations cover the convenience methods (`forward`,
//! `forward_prepared`, `forward_batched`); K-quant kernels only override
//! `forward_prepared` to use a shared Q8K activation, all other kernels
//! retain the Q8_0 path.

pub trait Kernel: Send + Sync {
    /// Hot-path matmul: pre-quantized Q8_0 input, partitioned by row.
    ///
    /// Each call computes `output[i] = sum_k weight[i, k] * dequant(input_q8[k], input_scales[k/32])`
    /// for `i` in `[ith * per_thread, min((ith + 1) * per_thread, n_out))`.
    fn forward_prequantized(
        &self,
        input_q8: &[u8],
        input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    );

    /// Matmul with both the original F32 input and its Q8_0 view available.
    /// K-quant kernels override this to prepare Q8_K activations; all other
    /// kernels retain the existing Q8_0 path.
    ///
    /// `q8_k` lets the caller pass a pre-quantized Q8_K activation (shared
    /// across threads) instead of letting each thread re-quantize internally.
    /// Pass `None` for kernels that don't need it (Q8_0 path) or when the
    /// caller has not pre-quantized.
    fn forward_prepared(
        &self,
        input_f32: &[f32],
        input_q8: &[u8],
        input_scales: &[f32],
        q8_k: Option<&[crate::ops::quant::BlockQ8K]>,
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        let _ = input_f32;
        let _ = q8_k;
        self.forward_prequantized(input_q8, input_scales, output, n_in, n_out, ith, nth);
    }

    /// Convenience: f32 input, single-thread. Default impl quantizes the
    /// input to Q8_0 and delegates to `forward_prequantized`. Kernels that
    /// have a native f32-input path (e.g. F16) override this.
    fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        let mut input_q8 = vec![0u8; n_in];
        let mut input_scales = vec![0.0f32; n_in.div_ceil(32)];
        crate::ops::quantize_q8_0_into(input, n_in, &mut input_q8, &mut input_scales);
        self.forward_prepared(
            input,
            &input_q8,
            &input_scales,
            None,
            output,
            n_in,
            n_out,
            0,
            1,
        );
    }

    /// Batched matmul: `input[n_tokens * n_in] → output[n_tokens * n_out]`.
    /// Default impl quantizes the whole batch up front then loops over tokens.
    fn forward_batched(
        &self,
        input: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) {
        let n_tokens = input.len() / n_in;
        debug_assert_eq!(input.len(), n_tokens * n_in);
        debug_assert_eq!(output.len(), n_tokens * n_out);
        let mut input_q8 = vec![0u8; n_in];
        let mut input_scales = vec![0.0f32; n_in.div_ceil(32)];
        for t in 0..n_tokens {
            crate::ops::quantize_q8_0_into(
                &input[t * n_in..(t + 1) * n_in],
                n_in,
                &mut input_q8,
                &mut input_scales,
            );
            self.forward_prepared(
                &input[t * n_in..(t + 1) * n_in],
                &input_q8,
                &input_scales,
                None,
                &mut output[t * n_out..(t + 1) * n_out],
                n_in,
                n_out,
                0,
                1,
            );
        }
    }

    fn embedding_lookup(&self, token_id: u32, n_embd: usize, out: &mut [f32]) {
        panic!("embedding_lookup not implemented for this kernel type");
    }
}
