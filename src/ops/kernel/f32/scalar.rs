//! F32 scalar matmul helpers. Acts as the reference implementation that the
//! AVX2 / NEON SIMD kernels in `avx2.rs` and `neon.rs` are tested against.
//!
//! Bit-exact match with `crate::ops::kernel::f32::F32Kernel::forward` so the
//! f32×f32 contract is preserved across SIMD tiers.

/// Compute `output[row] = sum_{col} weight[row*n_in+col] * input[col]` for
/// each row in `[start, end)`.
pub fn forward_f32_rows(
    weight: &[f32],
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
            sum += weight[row_off + col] * input[col];
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

/// Per-row sum without input (used by the `forward_prequantized` placeholder).
pub fn row_dot_range(
    weight: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    let per_thread = (n_out + nth - 1) / nth;
    let my_start = ith * per_thread;
    let my_end = (my_start + per_thread).min(n_out);
    if my_start >= my_end {
        return;
    }
    for out_idx in my_start..my_end {
        let mut sum = 0.0f32;
        let row_off = out_idx * n_in;
        for col in 0..n_in {
            sum += weight[row_off + col];
        }
        output[out_idx] = sum;
    }
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
    fn forward_f32_rows_matches_naive() {
        let n_in = 16;
        let n_out = 4;
        let weight: Vec<f32> = (0..n_in * n_out).map(|i| (i as f32) * 0.01).collect();
        let input: Vec<f32> = (0..n_in).map(|i| ((i as f32) - 8.0) * 0.02).collect();
        let mut out = vec![0.0f32; n_out];
        forward_f32_rows(&weight, &input, &mut out, n_in, n_out, 0, 1);
        for row in 0..n_out {
            let mut expected = 0.0f32;
            for col in 0..n_in {
                expected += weight[row * n_in + col] * input[col];
            }
            assert!((out[row] - expected).abs() < 1e-6);
        }
    }
}
