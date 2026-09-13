//! F32×F32 NEON (aarch64) matmul kernel.
//!
//! Mirrors `bf16/neon.rs` minus the BF16→F32 unpack step. 4-wide FMA
//! per inner iteration, hsum via `vaddvq_f32` at row end.
//!
//! TODO-005: once a CI target exposes aarch64, validate against the
//! scalar reference in `super::scalar`.

#![cfg(target_arch = "aarch64")]

#[target_feature(enable = "neon")]
pub unsafe fn matmul_f32_vs_f32_neon(
    weight: &[f32],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::aarch64::*;

    let weight_ptr = weight.as_ptr();
    let input_ptr = input.as_ptr();
    let output_ptr = output.as_mut_ptr();

    for (output_index, row) in (row_start..row_end).enumerate() {
        let row_off = row * n_in;
        let mut sum = vdupq_n_f32(0.0);
        let mut index = 0;

        while index + 4 <= n_in {
            sum = vfmaq_f32(sum, vld1q_f32(weight_ptr.add(row_off + index)), vld1q_f32(input_ptr.add(index)));
            index += 4;
        }

        let mut total = vaddvq_f32(sum);
        while index < n_in {
            total += *weight_ptr.add(row_off + index) * *input_ptr.add(index);
            index += 1;
        }
        *output_ptr.add(output_index) = total;
    }
}

#[cfg(test)]
mod tests {
    use super::matmul_f32_vs_f32_neon;
    use crate::ops::kernel::f32::scalar::forward_f32_rows;

    #[test]
    fn neon_matches_scalar_with_tail() {
        let n_in = 13;
        let n_out = 3;
        let weight: Vec<f32> = (0..n_in * n_out)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
            .collect();
        let input: Vec<f32> = (0..n_in).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
        let mut actual = vec![0.0; n_out];
        let mut expected = vec![0.0; n_out];

        unsafe {
            matmul_f32_vs_f32_neon(&weight, &input, &mut actual, n_in, 0, n_out);
        }
        forward_f32_rows(&weight, &input, &mut expected, n_in, n_out, 0, 1);

        for (a, e) in actual.into_iter().zip(expected) {
            assert!((a - e).abs() < 1e-5, "{a} != {e}");
        }
    }
}
