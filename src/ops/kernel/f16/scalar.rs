//! F16 scalar matmul helpers. Reference implementation that the AVX2 / NEON
//! SIMD kernels are tested against.

/// Compute `output[row] = sum_{col} f16_to_f32(weight[row*n_in+col]) * input[col]`
/// for each row in `[start, end)`. Weight is laid out as `[n_out rows × n_in
/// cols]` of f16 little-endian bytes.
pub fn forward_f16_rows(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    let (start, end) = row_range(n_out, ith, nth);
    for out_idx in start..end {
        let row_off = out_idx * n_in;
        let mut sum = 0.0f32;
        for col in 0..n_in {
            let bits = u16::from_le_bytes([
                weight[row_off * 2 + col * 2],
                weight[row_off * 2 + col * 2 + 1],
            ]);
            sum += crate::ops::f16_to_f32(bits) * input[col];
        }
        output[out_idx] = sum;
    }
}

/// `n_out`-element `[start, end)` partition used by all SIMD tiers.
pub fn row_range(n_out: usize, ith: usize, nth: usize) -> (usize, usize) {
    let nth = nth.max(1);
    let start = n_out.saturating_mul(ith) / nth;
    let end = n_out.saturating_mul(ith.saturating_add(1)) / nth;
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_range_partitions_evenly() {
        assert_eq!(row_range(64, 0, 4), (0, 16));
        assert_eq!(row_range(64, 1, 4), (16, 32));
        assert_eq!(row_range(64, 2, 4), (32, 48));
        assert_eq!(row_range(64, 3, 4), (48, 64));
    }

    #[test]
    fn row_range_uneven_split() {
        // Floor division: 7 / 3 → 2 / 2 / 3 (largest chunk goes last).
        assert_eq!(row_range(7, 0, 3), (0, 2));
        assert_eq!(row_range(7, 1, 3), (2, 4));
        assert_eq!(row_range(7, 2, 3), (4, 7));
    }

    #[test]
    fn forward_f16_rows_matches_naive() {
        let n_in = 16;
        let n_out = 4;
        let weight: Vec<u8> = (0..n_in * n_out)
            .flat_map(|i| crate::ops::f32_to_f16((i as f32) * 0.01).to_le_bytes())
            .collect();
        let input: Vec<f32> = (0..n_in).map(|i| ((i as f32) - 8.0) * 0.02).collect();
        let mut out = vec![0.0f32; n_out];
        forward_f16_rows(&weight, &input, &mut out, n_in, n_out, 0, 1);
        for row in 0..n_out {
            let mut expected = 0.0f32;
            for col in 0..n_in {
                let bits = u16::from_le_bytes([
                    weight[row * n_in * 2 + col * 2],
                    weight[row * n_in * 2 + col * 2 + 1],
                ]);
                expected += crate::ops::f16_to_f32(bits) * input[col];
            }
            assert!((out[row] - expected).abs() < 1e-3);
        }
    }
}
