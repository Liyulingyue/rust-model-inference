//! Pure dtype-level weight fusion.
//!
//! Model-agnostic: concatenates two same-dtype weight tensors along the output
//! (row) dimension. Used by SwiGLU FFN (gate + up) and attention QKV fusion
//! to fold two matmuls with the same input into one.
//!
//! Returns `None` when row layouts are inconsistent, leaving the caller to fall
//! back to the per-projection path.

use super::{BLOCK_Q4K_SIZE, BLOCK_Q5K_SIZE, BLOCK_Q6K_SIZE, BLOCK_Q80_SIZE, QK_K};

const Q8_0_BLOCK_ELEMS: usize = 32;

/// Concatenate two Q8_0 weights along the row dimension.
///
/// Q8_0 layout: each block of 32 elements occupies 34 bytes (2-byte f16 scale +
/// 32 bytes int8). All rows have identical byte length = `(n_cols / 32) * 34`.
///
/// The fused byte layout is `concat(a_rows, b_rows)`: the first `a_rows` rows
/// of the fused tensor correspond to `a_data`, the next `b_rows` rows to
/// `b_data`. Downstream code reads output rows `[0, a_rows)` and
/// `[a_rows, a_rows + b_rows)` to recover the original two projections.
pub fn fuse_vstack_q8_0(
    a_data: &[u8],
    b_data: &[u8],
    a_rows: usize,
    b_rows: usize,
) -> Option<(Vec<u8>, usize)> {
    let row_bytes_a = a_data.len() / a_rows.max(1);
    let row_bytes_b = b_data.len() / b_rows.max(1);
    if row_bytes_a != row_bytes_b {
        return None;
    }
    let mut fused = Vec::with_capacity(a_data.len() + b_data.len());
    fused.extend_from_slice(a_data);
    fused.extend_from_slice(b_data);
    Some((fused, a_rows + b_rows))
}

/// Per-row byte length of a Q8_0 weight with the given input dim.
#[inline]
pub fn q8_0_row_bytes(n_cols: usize) -> usize {
    (n_cols / Q8_0_BLOCK_ELEMS) * BLOCK_Q80_SIZE
}

/// Concatenate two Q4_K weights along the row dimension.
pub fn fuse_vstack_q4_k(
    a_data: &[u8],
    b_data: &[u8],
    a_rows: usize,
    b_rows: usize,
) -> Option<(Vec<u8>, usize)> {
    let row_bytes_a = a_data.len() / a_rows.max(1);
    let row_bytes_b = b_data.len() / b_rows.max(1);
    if row_bytes_a != row_bytes_b {
        return None;
    }
    let mut fused = Vec::with_capacity(a_data.len() + b_data.len());
    fused.extend_from_slice(a_data);
    fused.extend_from_slice(b_data);
    Some((fused, a_rows + b_rows))
}

/// Concatenate two Q5_K weights along the row dimension.
pub fn fuse_vstack_q5_k(
    a_data: &[u8],
    b_data: &[u8],
    a_rows: usize,
    b_rows: usize,
) -> Option<(Vec<u8>, usize)> {
    let row_bytes_a = a_data.len() / a_rows.max(1);
    let row_bytes_b = b_data.len() / b_rows.max(1);
    if row_bytes_a != row_bytes_b {
        return None;
    }
    let mut fused = Vec::with_capacity(a_data.len() + b_data.len());
    fused.extend_from_slice(a_data);
    fused.extend_from_slice(b_data);
    Some((fused, a_rows + b_rows))
}

/// Concatenate two Q6_K weights along the row dimension.
pub fn fuse_vstack_q6_k(
    a_data: &[u8],
    b_data: &[u8],
    a_rows: usize,
    b_rows: usize,
) -> Option<(Vec<u8>, usize)> {
    let row_bytes_a = a_data.len() / a_rows.max(1);
    let row_bytes_b = b_data.len() / b_rows.max(1);
    if row_bytes_a != row_bytes_b {
        return None;
    }
    let mut fused = Vec::with_capacity(a_data.len() + b_data.len());
    fused.extend_from_slice(a_data);
    fused.extend_from_slice(b_data);
    Some((fused, a_rows + b_rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{matmul_q8_0_quantized_range, quantize_q8_0_into};

    /// Build a deterministic Q8_0 weight with `n_rows` rows and `n_cols` input
    /// columns. Row `r` and block `b` quantizes the value `r + b` (clamped to
    /// i8 range) so different rows produce different matmul outputs.
    fn make_q8_0_weight(n_rows: usize, n_cols: usize, row_offset: i8) -> Vec<u8> {
        let blocks_per_row = n_cols / Q8_0_BLOCK_ELEMS;
        let row_bytes = blocks_per_row * BLOCK_Q80_SIZE;
        let mut data = Vec::with_capacity(n_rows * row_bytes);
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let scale = half::f16::from_f32(0.5 + 0.01 * (r as f32)).to_bits();
                data.extend_from_slice(&scale.to_le_bytes());
                for j in 0..Q8_0_BLOCK_ELEMS {
                    let v = (row_offset as i32 + r as i32 + b as i32 + j as i32 / 4) as i8;
                    data.push(v as u8);
                }
            }
        }
        assert_eq!(data.len(), n_rows * row_bytes);
        data
    }

    #[test]
    fn q8_0_concat_matches_separate_for_layout() {
        let n_in = 64;
        let n_a = 4;
        let n_b = 6;
        let a = make_q8_0_weight(n_a, n_in, 0);
        let b = make_q8_0_weight(n_b, n_in, 50);

        let (fused, n_rows) = fuse_vstack_q8_0(&a, &b, n_a, n_b).expect("fuse");
        assert_eq!(n_rows, n_a + n_b);
        assert_eq!(fused.len(), (n_a + n_b) * q8_0_row_bytes(n_in));

        // First n_a rows must equal a; next n_b rows must equal b.
        let row_bytes = q8_0_row_bytes(n_in);
        assert_eq!(&fused[..n_a * row_bytes], &a[..]);
        assert_eq!(&fused[n_a * row_bytes..], &b[..]);
    }

    #[test]
    fn q8_0_fused_matmul_matches_separate() {
        // Mirrors the real Qwen3.5 FFN shape: n_in = 1024, n_out per proj = 3584.
        // Use smaller numbers to keep the test fast.
        let n_in = 1024;
        let n_out = 3584;
        let gate = make_q8_0_weight(n_out, n_in, 0);
        let up = make_q8_0_weight(n_out, n_in, 17);

        let input: Vec<f32> = (0..n_in).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let mut input_q8 = vec![0u8; n_in];
        let mut input_scales = vec![0.0f32; n_in / 32];
        quantize_q8_0_into(&input, n_in, &mut input_q8, &mut input_scales);

        // Separate
        let mut out_gate = vec![0.0f32; n_out];
        let mut out_up = vec![0.0f32; n_out];
        matmul_q8_0_quantized_range(
            &gate,
            &input_q8,
            &input_scales,
            &mut out_gate,
            n_in,
            0,
            n_out,
        );
        matmul_q8_0_quantized_range(&up, &input_q8, &input_scales, &mut out_up, n_in, 0, n_out);

        // Fused
        let (fused, fused_rows) = fuse_vstack_q8_0(&gate, &up, n_out, n_out).expect("fuse");
        let mut out_fused = vec![0.0f32; fused_rows];
        matmul_q8_0_quantized_range(
            &fused,
            &input_q8,
            &input_scales,
            &mut out_fused,
            n_in,
            0,
            fused_rows,
        );

        // First n_out rows == gate output; last n_out rows == up output.
        for i in 0..n_out {
            assert_eq!(
                out_fused[i].to_bits(),
                out_gate[i].to_bits(),
                "gate mismatch at row {i}"
            );
            assert_eq!(
                out_fused[n_out + i].to_bits(),
                out_up[i].to_bits(),
                "up mismatch at row {i}"
            );
        }
    }

    #[test]
    fn q8_0_fuse_returns_none_on_inconsistent_layout() {
        // Mismatched row counts → different byte length → None.
        let a = make_q8_0_weight(4, 64, 0); // row_bytes = 2*34 = 68, total 272
        let b = make_q8_0_weight(4, 64, 1); // same shape — should succeed
        assert!(fuse_vstack_q8_0(&a, &b, 4, 4).is_some());

        // Truncate a so its row bytes differ from b's row bytes.
        let mut a_short = a.clone();
        a_short.truncate(a.len() - 4); // remove 4 bytes → first row partial
        assert!(fuse_vstack_q8_0(&a_short, &b, 4, 4).is_none());
    }

    #[test]
    fn q8_0_fuse_rejects_empty_side() {
        // Edge case: one side has 0 rows. `a_data.len() / max(1, 0_rows)` is 0,
        // which cannot match `b`'s row_bytes — so we must return None rather
        // than silently produce garbage. Caller falls back to the per-projection
        // path.
        let a: Vec<u8> = vec![];
        let b = make_q8_0_weight(3, 64, 0);
        assert!(fuse_vstack_q8_0(&a, &b, 0, 3).is_none());
        assert!(fuse_vstack_q8_0(&b, &a, 3, 0).is_none());
    }
}
