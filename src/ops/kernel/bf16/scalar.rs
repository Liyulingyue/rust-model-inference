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

pub(crate) fn forward_bf16_input_rows(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    let (start, end) = BF16Kernel::row_range(n_out, ith, nth);
    let mut sums = vec![0.0f64; rows];
    for out_idx in start..end {
        sums.fill(0.0);
        let row_start = out_idx * n_in * 2;
        for in_idx in 0..n_in {
            let weight_offset = row_start + in_idx * 2;
            let bits = u16::from_le_bytes([weight[weight_offset], weight[weight_offset + 1]]);
            let weight = crate::ops::bf16_to_f32(bits);
            for row in 0..rows {
                sums[row] += f64::from(weight * input[row * n_in + in_idx]);
            }
        }
        for row in 0..rows {
            output[row * n_out + out_idx] = sums[row] as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{dot_bf16, forward_bf16_input_rows};

    #[test]
    fn bf16_input_rows_match_sequential_bits() {
        let rows = 7;
        let n_in = 32;
        let n_out = 5;
        let weight = (0..n_in * n_out)
            .flat_map(|index| {
                crate::ops::f32_to_bf16(((index * 17 % 101) as f32 - 50.0) * 0.013).to_le_bytes()
            })
            .collect::<Vec<_>>();
        let input = (0..rows * n_in)
            .map(|index| ((index * 29 % 73) as f32 - 36.0) * 0.017)
            .collect::<Vec<_>>();
        let rounded = input
            .iter()
            .map(|&value| crate::ops::bf16_to_f32(crate::ops::f32_to_bf16(value)))
            .collect::<Vec<_>>();
        let expected = rounded
            .chunks_exact(n_in)
            .flat_map(|row| {
                let row = row
                    .iter()
                    .flat_map(|&value| crate::ops::f32_to_bf16(value).to_le_bytes())
                    .collect::<Vec<_>>();
                weight
                    .chunks_exact(n_in * 2)
                    .map(move |weight| dot_bf16(weight, &row).to_bits())
            })
            .collect::<Vec<_>>();
        let mut actual = vec![0.0; rows * n_out];

        forward_bf16_input_rows(&weight, &rounded, &mut actual, rows, n_in, n_out, 0, 1);

        assert_eq!(
            actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            expected
        );
    }
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
    assert!(
        input_q8.len() >= n_in,
        "BF16 forward_q8: input_q8 len {} < n_in {n_in}",
        input_q8.len()
    );
    assert!(
        input_scales.len() >= n_in.div_ceil(32),
        "BF16 forward_q8: input_scales len {} < required {}",
        input_scales.len(),
        n_in.div_ceil(32)
    );
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
