//! BF16 NEON matvec kernel.

#![cfg(target_arch = "aarch64")]

#[target_feature(enable = "neon")]
pub unsafe fn matmul_bf16_vs_f32_neon(
    weight: &[u8],
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
        let row_byte = row * n_in * 2;
        let mut sum = vdupq_n_f32(0.0);
        let mut index = 0;

        while index + 4 <= n_in {
            let bf16 = vld1_u16(weight_ptr.add(row_byte + index * 2).cast());
            let values = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(bf16), 16));
            sum = vfmaq_f32(sum, values, vld1q_f32(input_ptr.add(index)));
            index += 4;
        }

        let mut total = vaddvq_f32(sum);
        while index < n_in {
            let offset = row_byte + index * 2;
            let bits = u16::from_le_bytes([*weight_ptr.add(offset), *weight_ptr.add(offset + 1)]);
            total += crate::ops::bf16_to_f32(bits) * *input_ptr.add(index);
            index += 1;
        }
        *output_ptr.add(output_index) = total;
    }
}

#[cfg(test)]
mod tests {
    use super::matmul_bf16_vs_f32_neon;
    use crate::ops::kernel::bf16::scalar::forward_f32_rows_scalar;

    #[test]
    fn neon_matches_scalar_with_tail() {
        let n_in = 13;
        let n_out = 3;
        let values: Vec<f32> = (0..n_in * n_out)
            .map(|index| ((index % 17) as f32 - 8.0) * 0.125)
            .collect();
        let weight: Vec<u8> = values
            .iter()
            .flat_map(|&value| crate::ops::f32_to_bf16(value).to_le_bytes())
            .collect();
        let input: Vec<f32> = (0..n_in)
            .map(|index| ((index % 7) as f32 - 3.0) * 0.25)
            .collect();
        let mut actual = vec![0.0; n_out];
        let mut expected = vec![0.0; n_out];

        unsafe {
            matmul_bf16_vs_f32_neon(&weight, &input, &mut actual, n_in, 0, n_out);
        }
        forward_f32_rows_scalar(&weight, &input, &mut expected, n_in, n_out, 0, 1);

        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }
}
