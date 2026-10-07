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
        // Four independent F32 accumulators instead of a scalar running sum.
        // Every partial sum is still an exact F32 accumulation of the same
        // products the scalar kernel forms; only the summation order changes,
        // so this is a reassociation rather than an approximation. Weight
        // precision (BF16) and input precision (F32) are untouched. Bitwise
        // parity with the Oracle needs RMI_SCALAR=1, which routes to
        // scalar::dot_bf16 through BF16Kernel::with_bf16_input instead.
        let mut acc = vdupq_n_f32(0.0);
        let mut index = 0;

        while index + 4 <= n_in {
            let bf16 = vld1_u16(weight_ptr.add(row_byte + index * 2).cast());
            let values = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(bf16), 16));
            acc = vfmaq_f32(acc, values, vld1q_f32(input_ptr.add(index)));
            index += 4;
        }
        let mut total = vaddvq_f32(acc);
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

    #[test]
    fn neon_matches_scalar_wide() {
        for &n_in in &[32usize, 40, 1152, 2048, 6144] {
            let n_out = 3;
            let values: Vec<f32> = (0..n_in * n_out)
                .map(|i| (((i * 17 + 11) % 101) as f32 - 50.0) * 0.013)
                .collect();
            let weight: Vec<u8> = values
                .iter()
                .flat_map(|&value| crate::ops::f32_to_bf16(value).to_le_bytes())
                .collect();
            let input: Vec<f32> = (0..n_in)
                .map(|i| (((i * 29 + 7) % 73) as f32 - 36.0) * 0.017)
                .collect();
            let mut actual = vec![0.0f32; n_out];
            let mut expected = vec![0.0f32; n_out];
            unsafe {
                matmul_bf16_vs_f32_neon(&weight, &input, &mut actual, n_in, 0, n_out);
            }
            forward_f32_rows_scalar(&weight, &input, &mut expected, n_in, n_out, 0, 1);
            // Four F32 accumulators reassociate the sum, so the SIMD path is
            // not bitwise equal to the scalar one. Bound the drift relative to
            // the operands: it must stay far below BF16 weight precision
            // (2^-8 ~= 3.9e-3), i.e. the kernel reassociates the sum without
            // losing accuracy. Bitwise Oracle parity is covered by
            // `bf16_input_path_is_bitwise_stable` on the RMI_SCALAR=1 route.
            for (a, e) in actual.into_iter().zip(expected) {
                let tolerance = 1e-5 * f64::from(e.abs().max(1.0));
                assert!(
                    f64::from((a - e).abs()) <= tolerance,
                    "n_in={n_in}: {a} != {e} (|delta|={} > {tolerance})",
                    (a - e).abs()
                );
            }
        }
    }

    /// The Oracle parity contract is bitwise, and it lives on the
    /// `RMI_SCALAR=1` route: `with_bf16_input` pins activations to BF16 and
    /// accumulates in F64 via `scalar::dot_bf16`. The F32-input SIMD route is
    /// deliberately allowed to reassociate, so this test guards the path that
    /// must not.
    #[test]
    fn bf16_input_path_is_bitwise_stable() {
        use crate::ops::kernel::bf16::scalar::dot_bf16;
        for &n_in in &[32usize, 1152, 3584] {
            let n_out = 2;
            let values: Vec<f32> = (0..n_in * n_out)
                .map(|i| (((i * 17 + 11) % 101) as f32 - 50.0) * 0.013)
                .collect();
            let weight: Vec<u8> = values
                .iter()
                .flat_map(|&value| crate::ops::f32_to_bf16(value).to_le_bytes())
                .collect();
            let input: Vec<f32> = (0..n_in)
                .map(|i| (((i * 29 + 7) % 73) as f32 - 36.0) * 0.017)
                .collect();
            let rounded: Vec<u8> = input
                .iter()
                .flat_map(|&v| crate::ops::f32_to_bf16(v).to_le_bytes())
                .collect();
            for row in 0..n_out {
                let got = dot_bf16(&weight[row * n_in * 2..(row + 1) * n_in * 2], &rounded);
                let mut want = 0.0f64;
                for (w, x) in weight[row * n_in * 2..(row + 1) * n_in * 2]
                    .chunks_exact(2)
                    .zip(rounded.chunks_exact(2))
                {
                    let w = crate::ops::bf16_to_f32(u16::from_le_bytes([w[0], w[1]]));
                    let x = crate::ops::bf16_to_f32(u16::from_le_bytes([x[0], x[1]]));
                    want += f64::from(w * x);
                }
                assert_eq!(
                    got.to_bits(),
                    (want as f32).to_bits(),
                    "n_in={n_in} row={row}: {got} != {}",
                    want as f32
                );
            }
        }
    }
}
