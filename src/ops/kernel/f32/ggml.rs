//! ggml-exact F32×F32 dot product.
//!
//! The default `F32Kernel` path (`neon::matmul_f32_vs_f32_neon`) keeps a
//! single scalar accumulator and sums strictly left-to-right. That contract
//! exists because the BF16 depth decoder is sensitive to the reduction
//! order. llama.cpp's `ggml_vec_dot_f32` on aarch64 instead uses four
//! 4-lane F32 accumulators, a pairwise tree reduce and a final
//! `vaddvq_f32`. The two differ by ~1 ULP, which is enough to flip expert
//! routing once the difference is amplified by later quantized matmuls.
//!
//! This module reproduces the ggml arithmetic so callers that need
//! oracle parity (the Nemotron-3-Nano router) can opt in without changing
//! the default F32 contract used elsewhere.
//!
//! Reference: `ggml/src/ggml-cpu/vec.cpp` `ggml_vec_dot_f32` and
//! `ggml/src/ggml-cpu/simd-mappings.h` (`GGML_F32_STEP = 16`,
//! `GGML_F32_EPR = 4`, `GGML_F32_ARR = 4`).

/// ggml's `ggml_vec_dot_f32` over two contiguous F32 slices.
pub fn dot_f32_ggml(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(target_arch = "aarch64")]
    {
        if crate::ops::has_neon() {
            // SAFETY: NEON is available (runtime probe) and the slices are
            // contiguous and equal length.
            return unsafe { neon::dot_f32_ggml(x.as_ptr(), y.as_ptr(), x.len()) };
        }
    }
    scalar::dot_f32_ggml(x, y)
}

/// ggml's `ggml_vec_dot_f32` where `x[i * stride]` is the i-th element.
///
/// Used by the attention `kqv = V × softmax` product, where consecutive
/// keys are `n_attn_v_kv` floats apart in the cache. The arithmetic is the
/// same four-accumulator structure as the contiguous dot; only the loads
/// are strided.
pub fn dot_f32_ggml_stride(x: &[f32], stride: usize, y: &[f32]) -> f32 {
    scalar::dot_f32_ggml_stride(x, stride, y)
}

/// ggml's `ggml_vec_dot_f32` where `x[i * stride]` is the i-th element and
/// the entries beyond `n_valid` are treated as zero.
///
/// llama.cpp's KV cache view is padded to a multiple of 256 cells, so the
/// `V × softmax` product runs over the padded key count. The padded keys
/// hold zeros (the cache cells were never written) and the padded softmax
/// entries are zero as well, so the extra columns contribute exactly
/// `0.0` to the accumulators -- but they do decide how many 16-element
/// groups the dot accumulates over, which is observable in the last
/// mantissa bit.
pub fn dot_f32_ggml_stride_padded(x: &[f32], stride: usize, y: &[f32], n_valid: usize) -> f32 {
    scalar::dot_f32_ggml_stride_padded(x, stride, y, n_valid)
}

/// `output[row] = sum_col weight[row, col] * input[col]` for `row` in
/// `[row_start, row_end)`, using ggml's NEON reduction order.
pub fn forward_f32_rows_ggml(
    weight: &[f32],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if crate::ops::has_neon() {
            // SAFETY: NEON is available (runtime probe) and every load is
            // bounded by `n_in` / `row_end * n_in`.
            unsafe {
                neon::matmul_f32_vs_f32_ggml(
                    weight, input, output, n_in, row_start, row_end,
                );
            }
            return;
        }
    }
    scalar::matmul_f32_vs_f32_ggml(weight, input, output, n_in, row_start, row_end);
}

/// Scalar transcription of the NEON path. `mul_add` is a correctly rounded
/// FMA, so this matches `vfmaq_f32` bit-for-bit; only the four-accumulator
/// structure and the pairwise reduce have to be reproduced.
pub mod scalar {
    /// Contiguous ggml `ggml_vec_dot_f32` transcription.
    pub fn dot_f32_ggml(x: &[f32], y: &[f32]) -> f32 {
        let n = x.len();
        // Four vectors of four lanes, exactly like the NEON path.
        let mut sum = [[0.0f32; 4]; 4];
        let np = n & !(16 - 1);
        let mut i = 0;
        while i < np {
            for (j, acc) in sum.iter_mut().enumerate() {
                for (k, lane) in acc.iter_mut().enumerate() {
                    let at = i + j * 4 + k;
                    *lane = x[at].mul_add(y[at], *lane);
                }
            }
            i += 16;
        }
        reduce_and_tail(&sum, x, y, np, 1, n)
    }

    /// Strided variant: `x[i * stride]` against `y[i]`.
    pub fn dot_f32_ggml_stride(x: &[f32], stride: usize, y: &[f32]) -> f32 {
        let n = y.len();
        let mut sum = [[0.0f32; 4]; 4];
        let np = n & !(16 - 1);
        let mut i = 0;
        while i < np {
            for (j, acc) in sum.iter_mut().enumerate() {
                for (k, lane) in acc.iter_mut().enumerate() {
                    let at = i + j * 4 + k;
                    *lane = x[at * stride].mul_add(y[at], *lane);
                }
            }
            i += 16;
        }
        let mut reduced = [0.0f32; 4];
        for k in 0..4 {
            reduced[k] = (sum[0][k] + sum[2][k]) + (sum[1][k] + sum[3][k]);
        }
        let mut sumf = (reduced[0] + reduced[1]) + (reduced[2] + reduced[3]);
        while i < n {
            sumf += x[i * stride] * y[i];
            i += 1;
        }
        sumf
    }

    /// Strided variant where the entries beyond `n_valid` read as zero.
    pub fn dot_f32_ggml_stride_padded(
        x: &[f32],
        stride: usize,
        y: &[f32],
        n_valid: usize,
    ) -> f32 {
        let n = y.len();
        let mut sum = [[0.0f32; 4]; 4];
        let np = n & !(16 - 1);
        let mut i = 0;
        while i < np {
            for (j, acc) in sum.iter_mut().enumerate() {
                for (k, lane) in acc.iter_mut().enumerate() {
                    let at = i + j * 4 + k;
                    let xv = if at < n_valid { x[at * stride] } else { 0.0 };
                    *lane = xv.mul_add(y[at], *lane);
                }
            }
            i += 16;
        }
        let mut reduced = [0.0f32; 4];
        for k in 0..4 {
            reduced[k] = (sum[0][k] + sum[2][k]) + (sum[1][k] + sum[3][k]);
        }
        let mut sumf = (reduced[0] + reduced[1]) + (reduced[2] + reduced[3]);
        while i < n {
            let xv = if i < n_valid { x[i * stride] } else { 0.0 };
            sumf += xv * y[i];
            i += 1;
        }
        sumf
    }

    /// `GGML_F32_VEC_REDUCE` plus the scalar leftover loop.
    fn reduce_and_tail(
        sum: &[[f32; 4]; 4],
        x: &[f32],
        y: &[f32],
        np: usize,
        stride: usize,
        n: usize,
    ) -> f32 {
        let mut reduced = [0.0f32; 4];
        for k in 0..4 {
            reduced[k] = (sum[0][k] + sum[2][k]) + (sum[1][k] + sum[3][k]);
        }
        let mut sumf = (reduced[0] + reduced[1]) + (reduced[2] + reduced[3]);
        let mut i = np;
        while i < n {
            sumf += x[i * stride] * y[i];
            i += 1;
        }
        sumf
    }

    pub fn matmul_f32_vs_f32_ggml(
        weight: &[f32],
        input: &[f32],
        output: &mut [f32],
        n_in: usize,
        row_start: usize,
        row_end: usize,
    ) {
        for (out_idx, row) in (row_start..row_end).enumerate() {
            let row_off = row * n_in;
            output[out_idx] =
                dot_f32_ggml(&weight[row_off..row_off + n_in], &input[..n_in]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use std::arch::aarch64::*;

    /// Contiguous ggml `ggml_vec_dot_f32`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_f32_ggml(x: *const f32, y: *const f32, n: usize) -> f32 {
        let np = n & !(16 - 1);
        let mut sum = [vdupq_n_f32(0.0); 4];
        let mut i = 0;
        while i < np {
            for (j, acc) in sum.iter_mut().enumerate() {
                let base = i + j * 4;
                *acc = vfmaq_f32(
                    *acc,
                    vld1q_f32(x.add(base)),
                    vld1q_f32(y.add(base)),
                );
            }
            i += 16;
        }
        let s0 = vaddq_f32(sum[0], sum[2]);
        let s1 = vaddq_f32(sum[1], sum[3]);
        let mut sumf = vaddvq_f32(vaddq_f32(s0, s1));
        while i < n {
            sumf += *x.add(i) * *y.add(i);
            i += 1;
        }
        sumf
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn matmul_f32_vs_f32_ggml(
        weight: &[f32],
        input: &[f32],
        output: &mut [f32],
        n_in: usize,
        row_start: usize,
        row_end: usize,
    ) {
        let weight_ptr = weight.as_ptr();
        let input_ptr = input.as_ptr();
        let output_ptr = output.as_mut_ptr();
        let np = n_in & !(16 - 1);

        for (out_idx, row) in (row_start..row_end).enumerate() {
            let row_off = row * n_in;
            // GGML_F32_ARR == GGML_F32_STEP / GGML_F32_EPR == 4.
            let mut sum = [vdupq_n_f32(0.0); 4];
            let mut i = 0;
            while i < np {
                for (j, acc) in sum.iter_mut().enumerate() {
                    let base = i + j * 4;
                    let ax = vld1q_f32(weight_ptr.add(row_off + base));
                    let ay = vld1q_f32(input_ptr.add(base));
                    *acc = vfmaq_f32(*acc, ax, ay);
                }
                i += 16;
            }
            let s0 = vaddq_f32(sum[0], sum[2]);
            let s1 = vaddq_f32(sum[1], sum[3]);
            let mut sumf = vaddvq_f32(vaddq_f32(s0, s1));
            while i < n_in {
                sumf += *weight_ptr.add(row_off + i) * *input_ptr.add(i);
                i += 1;
            }
            *output_ptr.add(out_idx) = sumf;
        }
    }
}
