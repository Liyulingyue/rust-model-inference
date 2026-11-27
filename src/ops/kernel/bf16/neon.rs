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

/// Batched BF16×F32 matmul that reuses each loaded weight vector for four
/// input rows. Each row keeps the same vector accumulation order as the
/// single-row kernel, so batching changes memory traffic without changing
/// the per-row arithmetic contract.
///
/// Only the `[row_start, row_end)` output rows are computed and written:
/// the shared dispatcher (`src/ops/kernel/mod.rs`) hands each pool worker a
/// disjoint row range over a full-size output buffer, and a kernel that
/// writes outside its range races with the worker that owns those rows
/// (docs/develop/PARALLEL_MATMUL_SAFETY.md §4).
#[target_feature(enable = "neon")]
pub unsafe fn matmul_bf16_vs_f32_neon_batched(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    n_tokens: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::aarch64::*;

    debug_assert_eq!(weight.len(), n_in * n_out * 2);
    debug_assert_eq!(input.len(), n_tokens * n_in);
    debug_assert_eq!(output.len(), n_tokens * n_out);
    debug_assert!(row_start <= row_end && row_end <= n_out);

    let weight_ptr = weight.as_ptr();
    let input_ptr = input.as_ptr();
    let output_ptr = output.as_mut_ptr();

    for row in row_start..row_end {
        let row_byte = row * n_in * 2;
        for token_group in (0..n_tokens).step_by(4) {
            let group_len = (n_tokens - token_group).min(4);
            let mut acc = [vdupq_n_f32(0.0); 4];
            let mut index = 0;

            while index + 4 <= n_in {
                let bf16 = vld1_u16(weight_ptr.add(row_byte + index * 2).cast());
                let values = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(bf16), 16));
                for token in 0..group_len {
                    let input_row = input_ptr.add((token_group + token) * n_in + index);
                    acc[token] = vfmaq_f32(acc[token], values, vld1q_f32(input_row));
                }
                index += 4;
            }

            let mut totals = [0.0f32; 4];
            for token in 0..group_len {
                totals[token] = vaddvq_f32(acc[token]);
            }
            while index < n_in {
                let offset = row_byte + index * 2;
                let bits =
                    u16::from_le_bytes([*weight_ptr.add(offset), *weight_ptr.add(offset + 1)]);
                let value = crate::ops::bf16_to_f32(bits);
                for token in 0..group_len {
                    totals[token] += value * *input_ptr.add((token_group + token) * n_in + index);
                }
                index += 1;
            }

            for token in 0..group_len {
                *output_ptr.add((token_group + token) * n_out + row) = totals[token];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{matmul_bf16_vs_f32_neon, matmul_bf16_vs_f32_neon_batched};
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

    #[test]
    fn neon_batched_matches_single_rows_bitwise() {
        // (n_tokens, n_in) matrix: token counts 2..9 cover the tail groups
        // (group_len < 4) of the 4-token tiling; n_in 13 leaves a scalar
        // tail, 8 is an exact vector multiple, 40 spans several blocks.
        for &(n_tokens, n_in) in &[
            (2usize, 13usize),
            (3, 13),
            (5, 13),
            (6, 13),
            (7, 13),
            (9, 13),
            (4, 8),
            (6, 40),
        ] {
            let n_out = 3;
            let values: Vec<f32> = (0..n_in * n_out)
                .map(|index| (((index * 17 + 11) % 101) as f32 - 50.0) * 0.013)
                .collect();
            let weight: Vec<u8> = values
                .iter()
                .flat_map(|&value| crate::f32_to_bf16(value).to_le_bytes())
                .collect();
            let input: Vec<f32> = (0..n_tokens * n_in)
                .map(|index| (((index * 29 + 7) % 73) as f32 - 36.0) * 0.017)
                .collect();
            let mut actual = vec![0.0f32; n_tokens * n_out];
            let mut expected = vec![0.0f32; n_tokens * n_out];

            unsafe {
                matmul_bf16_vs_f32_neon_batched(
                    &weight,
                    &input,
                    &mut actual,
                    n_in,
                    n_out,
                    n_tokens,
                    0,
                    n_out,
                );
                for token in 0..n_tokens {
                    matmul_bf16_vs_f32_neon(
                        &weight,
                        &input[token * n_in..(token + 1) * n_in],
                        &mut expected[token * n_out..(token + 1) * n_out],
                        n_in,
                        0,
                        n_out,
                    );
                }
            }

            assert_eq!(actual, expected, "n_tokens={n_tokens} n_in={n_in}");
        }
    }
}
