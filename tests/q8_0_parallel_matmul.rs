//! Regression test for the parallel Q8_0 matmul kernel at large `n_out`.
//!
//! Background: `matmul_q8_0_quantized_parallel_rows` is the production hot path
//! called from `qwen35::forward_ffn_parallel` (via `ComputePool::compute`).
//! A latent bug surfaces when `n_out > ~5000` — see the "Bug" comment block at
//! `src/ops/kernel/q8_0/parallel.rs:147-160`.
//!
//! When a future change enables `QWeight::fuse_vstack` for Q8_0 (FFN gate+up
//! fusion), `n_out` for FFN matmul doubles from 3584 to 7168, which is what
//! triggers the bug. Until that kernel is fixed, this test must FAIL at
//! `n_out = 7168` and PASS at `n_out = 3584` (the current production path).
//!
//! To run: `cargo test --release --test q8_0_parallel_matmul`

use std::sync::Arc;

use rust_model_inference::core::thread_pool::ComputePool;
use rust_model_inference::ops::quant::BLOCK_Q80_SIZE;
use rust_model_inference::ops::{
    matmul_q8_0_quantized, matmul_q8_0_quantized_parallel_rows, quantize_q8_0_into,
};

const Q8_0_BLOCK_ELEMS: usize = 32;

/// Deterministic Q8_0 weight: row `r` block `b` stores a non-trivial pattern
/// that varies with both indices, so different rows produce different matmul
/// outputs and partial-output bugs are visible as zero or stale rows.
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

/// Reference single-threaded output: `matmul_q8_0_quantized`.
fn reference_output(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    n_in: usize,
    n_out: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_out];
    matmul_q8_0_quantized(weight, input_q8, input_scales, &mut out, n_in, n_out);
    out
}

/// Production parallel output: `matmul_q8_0_quantized_parallel_rows` via
/// `ComputePool::compute`. This is what `qwen35::QWeight::Q8_0` actually calls.
fn parallel_output(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    n_in: usize,
    n_out: usize,
    pool: &ComputePool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_out];
    let n_out_local = n_out;
    let weight_ptr = weight.as_ptr();
    let weight_len = weight.len();
    let iq_ptr = input_q8.as_ptr();
    let iq_len = input_q8.len();
    let sc_ptr = input_scales.as_ptr();
    let sc_len = input_scales.len();
    let out_ptr = out.as_mut_ptr();
    pool.compute(move |ith, nth| {
        let w = unsafe { std::slice::from_raw_parts(weight_ptr, weight_len) };
        let iq = unsafe { std::slice::from_raw_parts(iq_ptr, iq_len) };
        let sc = unsafe { std::slice::from_raw_parts(sc_ptr, sc_len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, n_out_local) };
        matmul_q8_0_quantized_parallel_rows(w, iq, sc, out_slice, n_in, n_out_local, ith, nth);
    });
    out
}

/// Quantize `input` into Q8_0 form for use with both kernels.
fn quantize_input(input: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let n = input.len();
    let mut q8 = vec![0u8; n];
    let mut scales = vec![0.0f32; n / Q8_0_BLOCK_ELEMS];
    quantize_q8_0_into(input, n, &mut q8, &mut scales);
    (q8, scales)
}

fn assert_outputs_match(label: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len(), "{label}: length mismatch");
    let mut first_mismatch = None;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        if e.to_bits() != a.to_bits() {
            first_mismatch = Some((i, *e, *a));
            break;
        }
    }
    if let Some((i, e, a)) = first_mismatch {
        let mut mismatches = 0;
        let mut max_abs_diff = 0.0f32;
        for (e, a) in expected.iter().zip(actual.iter()) {
            let d = (e - a).abs();
            if d > 0.0 {
                mismatches += 1;
                if d > max_abs_diff {
                    max_abs_diff = d;
                }
            }
        }
        panic!(
            "{label}: first mismatch at row {i}: expected={e} actual={a} \
             (total mismatched rows: {mismatches}/{n}, max abs diff: {d})",
            n = expected.len(),
            d = max_abs_diff,
        );
    }
}

fn run_case(label: &str, n_in: usize, n_out: usize, pool: &ComputePool) {
    let weight = make_q8_0_weight(n_out, n_in, 7);
    let input: Vec<f32> = (0..n_in).map(|i| ((i as f32) * 0.013) - 4.0).collect();
    let (input_q8, input_scales) = quantize_input(&input);

    let reference = reference_output(&weight, &input_q8, &input_scales, n_in, n_out);
    let parallel = parallel_output(&weight, &input_q8, &input_scales, n_in, n_out, pool);

    assert_outputs_match(label, &reference, &parallel);
}

#[test]
fn parallel_matches_reference_at_small_n_out() {
    let pool = Arc::new(ComputePool::new(8));
    // Production path: Qwen3.5 FFN gate/up un-fused, n_out = n_ff = 3584.
    // This MUST pass — current production correctness depends on it.
    run_case("n_out=3584", 1024, 3584, &pool);
}

#[test]
fn parallel_matches_reference_at_double_n_out() {
    let pool = Arc::new(ComputePool::new(8));
    // Fused gate_up path: n_out = 2 * n_ff = 7168. Currently FAILS due to
    // the latent bug in matmul_q8_0_quantized_parallel_rows / ComputePool.
    // When this test passes, Q8_0 fuse_vstack can be safely integrated into
    // qwen35::QWeight::fuse_vstack.
    run_case("n_out=7168", 1024, 7168, &pool);
}

#[test]
fn parallel_matches_reference_with_varied_thread_counts() {
    for n_threads in [1usize, 2, 4, 8] {
        let pool = ComputePool::new(n_threads);
        run_case(
            &format!("n_threads={n_threads} n_out=7168"),
            1024,
            7168,
            &pool,
        );
    }
}
