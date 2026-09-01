//! End-to-end parity test for the fused vs. separate FFN gate+up matmul
//! path. Mirrors what `qwen35::forward_ffn_parallel` does at runtime:
//!
//! - Reference path: two separate calls, each going through `pool.compute`
//!   with row-partitioned per-row dot products.
//! - Fused path: build `Q8_0 { data, n_cols, n_rows = 2 * n_ff }` (or Q4_K /
//!   Q5_K / Q6_K) via `fuse::fuse_vstack_*`, single call.
//!
//! Expected outputs are bit-exact equal in both directions.
//!
//! Background: until `QWeight::fuse_vstack` learned Q8_0, every GGUF model
//! shipped here was Q4_K_M / Q8_0 — neither matched the `Q4K`/`Q5K`/`Q6K`
//! arms in production use. So the fused FFN path was effectively dead code
//! on Qwen3.5. The first end-to-end exercise of the fused path surfaced a
//! wrong-output bug that the per-kernel regression test did not reproduce —
//! this test reproduces the actual production-call shape.
//!
//! To run: `cargo test --release --test fused_ffn_matmul_parity`

use rust_model_inference::core::thread_pool::ComputePool;
use rust_model_inference::ops::quant::fuse::{
    fuse_vstack_q4_k, fuse_vstack_q5_k, fuse_vstack_q6_k, fuse_vstack_q8_0,
};
use rust_model_inference::ops::quant::{
    vec_dot_q4k_q8k_scalar, vec_dot_q5k_q8k_scalar, vec_dot_q6k_q8k_scalar, BlockQ8K,
    BLOCK_Q4K_SIZE, BLOCK_Q5K_SIZE, BLOCK_Q6K_SIZE, BLOCK_Q80_SIZE, QK_K,
};
use rust_model_inference::ops::{
    matmul_q8_0_quantized_parallel_rows, quantize_q8_0_into, quantize_row_q8_k_into,
};

const Q8_0_BLOCK_ELEMS: usize = 32;

// ---------------------------------------------------------------------------
// Q8_0 path — uses the production kernel directly via pool.compute.
// ---------------------------------------------------------------------------

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

fn quantize_input_q8_0(input: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let n = input.len();
    let mut q8 = vec![0u8; n];
    let mut scales = vec![0.0f32; n / Q8_0_BLOCK_ELEMS];
    quantize_q8_0_into(input, n, &mut q8, &mut scales);
    (q8, scales)
}

/// Mirror `quantize_and_matmul_with_scratch` Q8_0 arm exactly: quantize input,
/// then run `matmul_q8_0_quantized_parallel_rows` via pool.compute.
fn quantize_and_parallel_matmul_q8_0(
    weight: &[u8],
    input: &[f32],
    n_in: usize,
    n_out: usize,
    pool: &ComputePool,
) -> Vec<f32> {
    let (input_q8, input_scales) = quantize_input_q8_0(input);
    let mut out = vec![0.0f32; n_out];
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
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, n_out) };
        matmul_q8_0_quantized_parallel_rows(w, iq, sc, out_slice, n_in, n_out, ith, nth);
    });
    out
}

fn run_q8_0_case(label: &str, n_in: usize, n_out_per: usize, pool: &ComputePool) {
    let gate = make_q8_0_weight(n_out_per, n_in, 0);
    let up = make_q8_0_weight(n_out_per, n_in, 17);
    let input: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.013) - 4.0).collect();

    let out_gate = quantize_and_parallel_matmul_q8_0(&gate, &input, n_in, n_out_per, pool);
    let out_up = quantize_and_parallel_matmul_q8_0(&up, &input, n_in, n_out_per, pool);

    let (fused_data, fused_n_rows) =
        fuse_vstack_q8_0(&gate, &up, n_out_per, n_out_per).expect("fuse_vstack_q8_0");
    assert_eq!(fused_n_rows, 2 * n_out_per);
    let out_fused =
        quantize_and_parallel_matmul_q8_0(&fused_data, &input, n_in, fused_n_rows, pool);

    assert_eq_f32(
        &format!("{label} q8_0 gate half"),
        &out_gate,
        &out_fused[..n_out_per],
    );
    assert_eq_f32(
        &format!("{label} q8_0 up half"),
        &out_up,
        &out_fused[n_out_per..],
    );
}

#[test]
fn fused_q8_0_matches_separate_at_production_shape() {
    let pool = ComputePool::new(8);
    run_q8_0_case("production", 1024, 3584, &pool);
}

#[test]
fn fused_q8_0_matches_separate_at_fused_shape() {
    let pool = ComputePool::new(8);
    run_q8_0_case("fused", 1024, 3584, &pool);
}

// ---------------------------------------------------------------------------
// Q4_K / Q5_K / Q6_K path — uses `vec_dot_q*Kk_q8k_scalar` per row, with a
// parallel partitioning that mirrors qwen35's `matmul_with_q8k_into_buf_pooled`.
// Until Q4_K/Q5_K/Q6_K GGUF models are added to the test fixtures, these
// paths are exercised only via synthetic byte weights — the per-row dot
// product function still consumes real bytes; partitioning bugs surface as
// zero or stale rows in `out`.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum KkDtype {
    Q4K,
    Q5K,
    Q6K,
}

impl KkDtype {
    fn row_bytes(self, blocks_per_row: usize) -> usize {
        match self {
            KkDtype::Q4K => blocks_per_row * BLOCK_Q4K_SIZE,
            KkDtype::Q5K => blocks_per_row * BLOCK_Q5K_SIZE,
            KkDtype::Q6K => blocks_per_row * BLOCK_Q6K_SIZE,
        }
    }

    fn row_dot(self, row_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
        match self {
            KkDtype::Q4K => vec_dot_q4k_q8k_scalar(row_data, q8k),
            KkDtype::Q5K => vec_dot_q5k_q8k_scalar(row_data, q8k),
            KkDtype::Q6K => vec_dot_q6k_q8k_scalar(row_data, q8k),
        }
    }
}

fn make_qk_weight(total_bytes: usize, row_bytes: usize, seed: u8) -> Vec<u8> {
    let n_rows = total_bytes / row_bytes;
    let mut data = vec![0u8; n_rows * row_bytes];
    for r in 0..n_rows {
        for j in 0..row_bytes {
            data[r * row_bytes + j] = seed.wrapping_add(r as u8).wrapping_add(j as u8);
        }
    }
    data
}

fn quantize_input_q8k(input: &[f32]) -> Vec<BlockQ8K> {
    let blocks = input.len() / QK_K;
    let mut q8k = vec![
        BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16]
        };
        blocks
    ];
    quantize_row_q8_k_into(input, &mut q8k);
    q8k
}

fn sequential_kk_matmul(
    dtype: KkDtype,
    weight: &[u8],
    q8k: &[BlockQ8K],
    n_out: usize,
    blocks_per_row: usize,
) -> Vec<f32> {
    let row_bytes = dtype.row_bytes(blocks_per_row);
    let mut out = vec![0.0f32; n_out];
    for o in 0..n_out {
        let row_data = &weight[o * row_bytes..(o + 1) * row_bytes];
        out[o] = dtype.row_dot(row_data, q8k);
    }
    out
}

/// Mirror qwen35 `matmul_with_q8k_into_buf_pooled` partitioning: split `n_out`
/// rows into `n_threads` chunks, each thread computes its rows via the
/// scalar per-row dot product and writes into the thread-local slice.
fn parallel_kk_matmul(
    dtype: KkDtype,
    weight: &[u8],
    q8k: &[BlockQ8K],
    n_out: usize,
    blocks_per_row: usize,
    pool: &ComputePool,
) -> Vec<f32> {
    let row_bytes = dtype.row_bytes(blocks_per_row);
    let mut out = vec![0.0f32; n_out];
    let weight_ptr = weight.as_ptr();
    let weight_len = weight.len();
    let q8k_ptr = q8k.as_ptr();
    let q8k_len = q8k.len();
    let out_ptr = out.as_mut_ptr();
    let n_threads = pool.n_threads();
    let chunk_size = (n_out + n_threads - 1) / n_threads;
    pool.compute(move |ith, _nth| {
        let start = ith * chunk_size;
        let end = (start + chunk_size).min(n_out);
        if start >= end {
            return;
        }
        unsafe {
            let w = std::slice::from_raw_parts(weight_ptr, weight_len);
            let q = std::slice::from_raw_parts(q8k_ptr, q8k_len);
            let buf = std::slice::from_raw_parts_mut(out_ptr.add(start), end - start);
            for (local, o) in (start..end).enumerate() {
                let row_data = &w[o * row_bytes..(o + 1) * row_bytes];
                buf[local] = dtype.row_dot(row_data, q);
            }
        }
    });
    out
}

fn run_kk_case(label: &str, dtype: KkDtype, n_in: usize, n_out_per: usize, pool: &ComputePool) {
    let blocks_per_row = n_in / QK_K;
    let row_bytes = dtype.row_bytes(blocks_per_row);

    let gate = make_qk_weight(n_out_per * row_bytes, row_bytes, 0);
    let up = make_qk_weight(n_out_per * row_bytes, row_bytes, 17);

    let input: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.013) - 4.0).collect();
    let q8k = quantize_input_q8k(&input);

    // Reference (sequential) vs parallel: must agree on each row.
    let out_gate_seq = sequential_kk_matmul(dtype, &gate, &q8k, n_out_per, blocks_per_row);
    let out_gate_par = parallel_kk_matmul(dtype, &gate, &q8k, n_out_per, blocks_per_row, pool);
    assert_eq_f32(
        &format!("{label} gate seq==par"),
        &out_gate_seq,
        &out_gate_par,
    );

    let out_up_seq = sequential_kk_matmul(dtype, &up, &q8k, n_out_per, blocks_per_row);
    let out_up_par = parallel_kk_matmul(dtype, &up, &q8k, n_out_per, blocks_per_row, pool);
    assert_eq_f32(&format!("{label} up seq==par"), &out_up_seq, &out_up_par);

    // Fused weight via fuse_vstack_q*Kk, then run sequential + parallel,
    // confirm the first n_out_per rows == gate result, the last == up result.
    let (fused_data, fused_rows) = match dtype {
        KkDtype::Q4K => fuse_vstack_q4_k(&gate, &up, n_out_per, n_out_per).unwrap(),
        KkDtype::Q5K => fuse_vstack_q5_k(&gate, &up, n_out_per, n_out_per).unwrap(),
        KkDtype::Q6K => fuse_vstack_q6_k(&gate, &up, n_out_per, n_out_per).unwrap(),
    };
    assert_eq!(fused_rows, 2 * n_out_per);

    let fused_seq = sequential_kk_matmul(dtype, &fused_data, &q8k, fused_rows, blocks_per_row);
    let fused_par = parallel_kk_matmul(dtype, &fused_data, &q8k, fused_rows, blocks_per_row, pool);

    assert_eq_f32(
        &format!("{label} gate fused seq==separate"),
        &out_gate_seq,
        &fused_seq[..n_out_per],
    );
    assert_eq_f32(
        &format!("{label} up fused seq==separate"),
        &out_up_seq,
        &fused_seq[n_out_per..],
    );
    assert_eq_f32(
        &format!("{label} gate fused par==separate"),
        &out_gate_par,
        &fused_par[..n_out_per],
    );
    assert_eq_f32(
        &format!("{label} up fused par==separate"),
        &out_up_par,
        &fused_par[n_out_per..],
    );
}

#[test]
fn parallel_kk_partitioning_matches_sequential_q6_k() {
    let pool = ComputePool::new(8);
    run_kk_case("ffn", KkDtype::Q6K, 1024, 3584, &pool);
}

#[test]
fn parallel_kk_partitioning_matches_sequential_q4_k() {
    let pool = ComputePool::new(8);
    run_kk_case("ffn", KkDtype::Q4K, 1024, 3584, &pool);
}

#[test]
fn parallel_kk_partitioning_matches_sequential_q5_k() {
    let pool = ComputePool::new(8);
    run_kk_case("ffn", KkDtype::Q5K, 1024, 3584, &pool);
}

#[test]
fn parallel_kk_q6_k_varied_thread_counts() {
    for n_threads in [1usize, 2, 4, 8] {
        let pool = ComputePool::new(n_threads);
        run_kk_case(&format!("t={n_threads}"), KkDtype::Q6K, 1024, 3584, &pool);
    }
}

// ---------------------------------------------------------------------------

fn assert_eq_f32(label: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len(), "{label}: length mismatch");
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        if e.to_bits() != a.to_bits() {
            let mut mismatches = 0usize;
            let mut max_abs = 0.0f32;
            for (e2, a2) in expected.iter().zip(actual.iter()) {
                let d = (e2 - a2).abs();
                if d > 0.0 {
                    mismatches += 1;
                    if d > max_abs {
                        max_abs = d;
                    }
                }
            }
            panic!(
                "{label}: first mismatch at row {i}: expected={e} actual={a} \
                 (mismatched rows: {mismatches}/{n}, max abs diff: {max_abs})",
                n = expected.len(),
            );
        }
    }
}
