//! BF16 scalar matmul kernel (F32-input path and Q8-input path).
//!
//! Reference for AVX2 SIMD: see `bf16::avx2` for F32 path. Q8 path is
//! scalar only (no SIMD).

use super::BF16Kernel;

/// llama.cpp's scalar BF16 dot: F32 products accumulated in F64.
pub(crate) fn dot_bf16(weight: &[u8], input: &[u8]) -> f32 {
    debug_assert_eq!(weight.len(), input.len());
    let mut sum = 0.0f64;
    for (w, x) in weight.chunks_exact(2).zip(input.chunks_exact(2)) {
        let w = crate::ops::bf16_to_f32(u16::from_le_bytes([w[0], w[1]]));
        let x = crate::ops::bf16_to_f32(u16::from_le_bytes([x[0], x[1]]));
        sum += f64::from(w * x);
    }
    sum as f32
}

pub fn forward_f32_rows_scalar(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    let (start, end) = BF16Kernel::row_range(n_out, ith, nth);
    for out_idx in start..end {
        let row_start = out_idx * n_in * 2;
        let mut sum = 0.0f32;
        for in_idx in 0..n_in {
            let weight_offset = row_start + in_idx * 2;
            let bits = u16::from_le_bytes([weight[weight_offset], weight[weight_offset + 1]]);
            sum += crate::ops::bf16_to_f32(bits) * input[in_idx];
        }
        let output_index = if output.len() >= n_out {
            out_idx
        } else {
            out_idx - start
        };
        output[output_index] = sum;
    }
}

pub fn forward_q8_rows_scalar(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    let (start, end) = BF16Kernel::row_range(n_out, ith, nth);
    let blocks_per_row = n_in.div_ceil(32);
    for out_idx in start..end {
        let row_start = out_idx * n_in * 2;
        let mut sum = 0.0f32;
        for block in 0..blocks_per_row {
            let input_start = block * 32;
            let input_end = (input_start + 32).min(n_in);
            let input_scale = input_scales[block];
            for in_idx in input_start..input_end {
                let weight_offset = row_start + in_idx * 2;
                let bits = u16::from_le_bytes([weight[weight_offset], weight[weight_offset + 1]]);
                sum +=
                    crate::ops::bf16_to_f32(bits) * (input_q8[in_idx] as i8 as f32) * input_scale;
            }
        }
        let output_index = if output.len() >= n_out {
            out_idx
        } else {
            out_idx - start
        };
        output[output_index] = sum;
    }
}
