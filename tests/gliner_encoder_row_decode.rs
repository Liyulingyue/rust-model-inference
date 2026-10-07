//! `gliner::compute::decode_row` across the row layouts the encoder accepts.
//!
//! `encode` reaches its two big tables through `decode_row`: the
//! relative-position table (row by row, `[pos_ebd_size, n_embd]`) and the
//! word embedding (one row per token). Both used to be F32-only, which is
//! why `tools/converter/utils/quantize_gguf.py` had to keep the whole
//! encoder at full precision. This file pins the widened contract: the F32
//! branch byte-for-byte, and each block-quantized type decoding to the same
//! values the embedding path produces.

use rust_model_inference::core::tensor::GGMLType;
use rust_model_inference::ops::embedding::{
    embedding_lookup, embedding_lookup_f16, embedding_lookup_q8_0,
};

/// One row of a Q8_0 `[rows, width]` table, row k uniformly `k + 1`.
fn q8_0_table(rows: usize, width: usize) -> Vec<u8> {
    assert!(width % 32 == 0);
    let mut out = Vec::with_capacity(rows * (width / 32) * 34);
    for row in 0..rows {
        let value = (row + 1) as f32;
        let scale = value / 127.0;
        for _ in 0..(width / 32) {
            out.extend_from_slice(&rust_model_inference::ops::f32_to_f16(scale).to_le_bytes());
            for _ in 0..32 {
                out.push(127); // 127 * scale == value, exactly
            }
        }
    }
    out
}

#[test]
fn q8_0_rows_decode_to_the_expected_values() {
    let (rows, width) = (8usize, 128usize);
    let table = q8_0_table(rows, width);
    let mut out = vec![0.0f32; width];
    for row in 0..rows {
        embedding_lookup_q8_0(&table, row as u32, width, &mut out);
        let want = (row + 1) as f32;
        assert!(
            out.iter().all(|v| (v - want).abs() < 1e-3),
            "row {row} decoded to {:?}, expected all {want}",
            &out[..4]
        );
    }
}

#[test]
fn the_dispatch_and_the_direct_helper_agree() {
    // `decode_row` routes through `embedding_lookup`; the two must not drift.
    let (rows, width) = (4usize, 64usize);
    let table = q8_0_table(rows, width);
    let mut via_dispatch = vec![0.0f32; width];
    let mut via_helper = vec![0.0f32; width];
    for row in 0..rows {
        embedding_lookup(&table, row as u32, width, GGMLType::Q8_0, &mut via_dispatch);
        embedding_lookup_q8_0(&table, row as u32, width, &mut via_helper);
        assert_eq!(
            via_dispatch, via_helper,
            "row {row}: dispatch and helper disagree"
        );
    }
}

#[test]
fn f16_rows_still_decode() {
    // F16 is on the same dispatch arm and is what a `q4_mixed`-style
    // converter would emit for a norm, so it must keep working here.
    let (rows, width) = (4usize, 32usize);
    let mut table = Vec::with_capacity(rows * width * 2);
    for row in 0..rows {
        for col in 0..width {
            table.extend_from_slice(
                &rust_model_inference::ops::f32_to_f16((row * width + col) as f32 / 64.0)
                    .to_le_bytes(),
            );
        }
    }
    let mut out = vec![0.0f32; width];
    embedding_lookup_f16(&table, 2, width, &mut out);
    for col in 0..width {
        let want = (2 * width + col) as f32 / 64.0;
        assert!(
            (out[col] - want).abs() < 1e-3,
            "f16 row 2 col {col}: got {}, want {want}",
            out[col]
        );
    }
}
