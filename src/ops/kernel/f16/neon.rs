//! F16×F32 NEON (aarch64) matmul kernel.
//!
//! Uses aarch64's native f16 NEON support (`float16x8_t` + `vcvt_f32_f16`)
//! to convert 8 f16 weight lanes to 8 f32 lanes per iteration, then FMA
//! against the f32 input. 4-wide per inner loop, hsum via `vaddvq_f32`
//! at row end.
//!
//! TODO-005: validate against the scalar reference in `super::scalar`
//! once a CI target exposes aarch64.

#![cfg(target_arch = "aarch64")]

#[target_feature(enable = "neon")]
pub unsafe fn matmul_f16_vs_f32_neon(
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

        while index + 8 <= n_in {
            // Load 8 f16 weight lanes (16 bytes).
            let f16_bits = vld1q_u16(weight_ptr.add(row_byte + index * 2).cast());
            // Reinterpret as f16x8 and convert to f32x8.
            let f16_lanes: float16x8_t = vreinterpretq_f16_u16(f16_bits);
            let values = vcvt_f32_f16(vget_low_f16(f16_lanes));
            sum = vfmaq_f32(sum, values, vld1q_f32(input_ptr.add(index)));
            index += 8;
        }
        while index + 4 <= n_in {
            let f16_bits = vld1_u16(weight_ptr.add(row_byte + index * 2).cast());
            let f16_lanes: float16x4_t = vreinterpret_f16_u16(f16_bits);
            let values = vcvt_f32_f16(f16_lanes);
            sum = vfmaq_f32(sum, values, vld1q_f32(input_ptr.add(index)));
            index += 4;
        }

        let mut total = vaddvq_f32(sum);
        while index < n_in {
            let offset = row_byte + index * 2;
            let bits = u16::from_le_bytes([*weight_ptr.add(offset), *weight_ptr.add(offset + 1)]);
            total += crate::ops::f16_to_f32(bits) * *input_ptr.add(index);
            index += 1;
        }
        *output_ptr.add(output_index) = total;
    }
}

#[cfg(test)]
mod tests {
    use super::matmul_f16_vs_f32_neon;
    use crate::ops::kernel::f16::scalar::forward_f16_rows;

    #[test]
    fn neon_matches_scalar_with_tail() {
        let n_in = 13;
        let n_out = 3;
        let weight: Vec<u8> = (0..n_in * n_out)
            .flat_map(|i| crate::ops::f32_to_f16((i as f32 * 0.07).sin()).to_le_bytes())
            .collect();
        let input: Vec<f32> = (0..n_in).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
        let mut actual = vec![0.0; n_out];
        let mut expected = vec![0.0; n_out];

        unsafe {
            matmul_f16_vs_f32_neon(&weight, &input, &mut actual, n_in, 0, n_out);
        }
        forward_f16_rows(&weight, &input, &mut expected, n_in, n_out, 0, 1);

        for (a, e) in actual.into_iter().zip(expected) {
            assert!((a - e).abs() < 1e-3, "{a} != {e}");
        }
    }
}
