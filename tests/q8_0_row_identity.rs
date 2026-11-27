//! Focused microtest for the cross-format swap seen in the full boundary
//! cross-format check (`tests/gliner2_5_base_v1_q8_0_parity.rs`).
//!
//! The full check observes that every adjacent pair of pool-logit slots
//! comes back swapped in Q8_0 vs F32 (slot 24 ↔ 25, slot 15 ↔ 16, ...).
//! A complete repro is too far from the kernel to localise because the
//! pipeline stacks many matmuls on top. Here the Q8_0 kernel is driven
//! directly with a hand-built weight so the row mapping is observable.
//!
//! A `[rows, 128]` weight whose row k is uniformly `row_values[k]` must,
//! on a 1.0 input, produce an output vector whose entries are the
//! per-row sums. Those are k-dependent, so any row permutation in the
//! kernel shows up as a permutation of those values rather than as noise.

use rust_model_inference::ops::kernel::{Kernel, QuantizedTensor};

/// Q8_0-encode a `rows × cols` weight whose row k is uniformly
/// `row_values[k]`, matching `tools/converter/utils/gguf.py:quantize_q8_0`:
/// the block scale is `amax / 127` stored as f16, and the payload is
/// `round(value / scale)` clamped to int8.
fn quantize_rows(cols: usize, row_values: &[f32]) -> Vec<u8> {
    assert!(cols % 32 == 0, "Q8_0 needs a 32-aligned row");
    let mut out = Vec::with_capacity(row_values.len() * (cols / 32) * 34);
    for &value in row_values {
        let amax = value.abs();
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let stored =
            rust_model_inference::ops::f16_to_f32(rust_model_inference::ops::f32_to_f16(scale));
        let safe = if stored == 0.0 { 1.0 } else { stored };
        let q = (value / safe).round().clamp(-127.0, 127.0) as i8 as u8;
        for _ in 0..(cols / 32) {
            out.extend_from_slice(&rust_model_inference::ops::f32_to_f16(scale).to_le_bytes());
            for _ in 0..32 {
                out.push(q);
            }
        }
    }
    out
}

fn q8(rows: usize, cols: usize, bytes: &'static [u8]) -> QuantizedTensor<'static> {
    QuantizedTensor::Q8_0 {
        data: bytes,
        n_cols: cols,
        n_rows: rows,
    }
}

#[test]
fn q8_0_rows_keep_their_identity_through_forward() {
    let cols = 128usize;
    let input = vec![1.0f32; cols];

    // Row values chosen so the per-row sums are far apart relative to the
    // ~1% Q8_0 noise: row k sums to `k * 1.28` on a 1.0 input.
    let row_values = [0.01f32, 0.02, 0.03, 0.04];
    let bytes: &'static [u8] = Box::leak(quantize_rows(cols, &row_values).into_boxed_slice());
    let tensor = q8(row_values.len(), cols, bytes);
    let mut output = vec![0.0f32; row_values.len()];
    tensor.forward(&input, &mut output, cols, row_values.len());

    let expected: Vec<f32> = row_values.iter().map(|v| v * cols as f32).collect();
    for (i, (got, want)) in output.iter().zip(expected.iter()).enumerate() {
        let tolerance = 0.03 * want.abs().max(0.05);
        assert!(
            (got - want).abs() < tolerance,
            "row {i} -> {got}, expected {want} (±{tolerance}); \
             a permutation of these values is the kernel row swap"
        );
    }
    assert!(
        output.windows(2).all(|pair| pair[0] < pair[1]),
        "row-distinct outputs must stay monotone increasing, got {:?}",
        output
    );
}

/// The boundary pipeline's hot shape: 128 rows × 128 cols, the same layout
/// as `start_projection` / `end_projection`. Checks that a full-width
/// batch keeps its row order, not just the 4-row case above.
#[test]
fn q8_0_keeps_row_order_across_a_128_row_batch() {
    let cols = 128usize;
    let rows = 128usize;
    // Row k is uniformly k/1000, so the per-row sum is k/1000 * 128.
    let row_values: Vec<f32> = (0..rows).map(|k| k as f32 / 1000.0).collect();
    let bytes: &'static [u8] = Box::leak(quantize_rows(cols, &row_values).into_boxed_slice());
    let tensor = q8(rows, cols, bytes);
    let input = vec![1.0f32; cols];
    let mut output = vec![0.0f32; rows];
    tensor.forward(&input, &mut output, cols, rows);

    // Row 0 is all zeros, so its Q8_0 scale is 0 and it decodes to 0; the
    // rest must be strictly increasing.
    for k in 1..rows {
        let want = row_values[k] * cols as f32;
        let got = output[k];
        assert!(
            (got - want).abs() <= 0.03 * want.abs(),
            "row {k} -> {got}, expected {want}"
        );
        assert!(
            output[k] > output[k - 1],
            "row {k} ({}) is not above row {} ({}) — row order is permuted",
            output[k],
            k - 1,
            output[k - 1]
        );
    }
}
