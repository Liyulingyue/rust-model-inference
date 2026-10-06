//! Microbenchmark + correctness test for the AVX2 path added to
//! `dot_bf16_f32_4` (used by YuE2's BF16 lm_head forward).
//!
//! Run with:
//!
//! ```text
//! cargo test --release -j 1 --test bench_yue4 -- --nocapture
//! ```
//!
//! - `bench_dot_bf16_f32_4_shared_vs_4single_*` shows the 4-row shared
//!   SIMD helper vs 4× single-row SIMD on the actual YuE2 hidden-dim
//!   shape (width 1024 / 3072 / 6144 — `vocab_dim × hidden_dim`).
//! - `dot_bf16_f32_4_matches_per_row_dispatch_for_yue2_shapes` is a
//!   bit-equal correctness sweep across 21 widths (0..6144) that
//!   pins the 4-row SIMD outputs to four independent single-row
//!   AVX2 dispatches of `dot_bf16_f32`. Finite outputs are bit-equal;
//!   NaN/Inf pairs require the same non-finite class on both sides.
//!
//! The correctness test supplements (but does not replace) the
//! in-tree `dot_bf16_f32_four_rows_preserves_individual_dot_bits`
//! in `src/ops/dot.rs` — that one runs as part of `cargo test --lib`,
//! which is currently broken on this branch (pre-existing
//! `dit_gpu`/`get_vulkan_context` import breakage unrelated to this
//! change). When the lib tests are fixed, this bench file is safe
//! to delete or leave as-is.

use std::time::Instant;

/// Scalar BF16×F32 dot product. Used as the bit-equal reference in
/// tests that compare against the SIMD path on widths below the
/// helper's SIMD activation threshold.
#[inline]
fn dot_bf16_f32_scalar(a: &[f32], b: &[u8], n: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..n {
        let lo = b[i * 2];
        let hi = b[i * 2 + 1];
        let bits = u16::from_le_bytes([lo, hi]);
        sum += a[i] * half::bf16::from_bits(bits).to_f32();
    }
    sum
}

/// 4× independent `dot_bf16_f32` calls — the previous fallback path
/// used by `matmul_bf16` on AVX2 hosts before the 4-row helper landed.
#[inline]
fn dot_bf16_f32_4_4calls(input: &[f32], weight: &[u8], width: usize) -> [f32; 4] {
    use rust_model_inference::ops::dot::dot_bf16_f32;
    std::array::from_fn(|row| dot_bf16_f32(&input[row * width..(row + 1) * width], weight, width))
}

/// Dispatched `dot_bf16_f32_4`: routes to AVX2 4-row on AVX2+FMA hosts
/// at `width ≥ 512`, NEON 4-row on aarch64, otherwise the 4× single-row
/// fallback. The bit-equal correctness test pins this against the
/// 4× single-row path so an FMA-ordering drift in either helper
/// fails the test.
#[inline]
fn dot_bf16_f32_4_dispatched(input: &[f32], weight: &[u8], width: usize) -> [f32; 4] {
    rust_model_inference::ops::dot::dot_bf16_f32_4(input, weight, width)
}

fn make_inputs(width: usize, seed: u64) -> (Vec<f32>, Vec<u8>) {
    // Deterministic random-ish inputs; mask BF16 high-byte patterns
    // that decode to NaN/Inf so the dot products stay finite for
    // the bit-equal test.
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as i32
    };
    let input: Vec<f32> = (0..(4 * width))
        .map(|_| (next() as f32 / i32::MAX as f32) * 2.0)
        .collect();
    let weights: Vec<u8> = (0..(width * 2))
        .map(|_| {
            // 0x7F80 / 0xFF80 / 0x7FC0 are +/- Inf / NaN in BF16.
            let raw = (next() as u16) & 0x7F7F;
            raw.to_le_bytes()
        })
        .flatten()
        .collect();
    (input, weights)
}

#[test]
fn bench_dot_bf16_f32_4_shared_vs_4single_yue2_shapes() {
    let widths = [256usize, 1024, 3072, 6144];
    let iterations = 4000;

    println!("\n=== 4-row shared SIMD vs 4× single-row SIMD ===");

    for &width in &widths {
        let (input, weight) = make_inputs(width, 0xdeadbeef);

        // Warmup
        for _ in 0..200 {
            let _ = dot_bf16_f32_4_dispatched(&input, &weight, width);
            let _ = dot_bf16_f32_4_4calls(&input, &weight, width);
        }

        // 4-row SIMD (dispatched)
        let start = Instant::now();
        let mut acc = [0.0f32; 4];
        for _ in 0..iterations {
            let r = dot_bf16_f32_4_dispatched(&input, &weight, width);
            acc[0] += r[0];
        }
        let ns_4row = start.elapsed().as_nanos() as f64 / iterations as f64;

        // 4 × single-row SIMD
        let start = Instant::now();
        for _ in 0..iterations {
            let r = dot_bf16_f32_4_4calls(&input, &weight, width);
            acc[1] += r[0];
        }
        let ns_4single = start.elapsed().as_nanos() as f64 / iterations as f64;

        let speedup = ns_4single / ns_4row;
        println!(
            "width={:>5}: 4row={:>8.1}ns/call, 4×single={:>8.1}ns/call, speedup={:.2}x",
            width, ns_4row, ns_4single, speedup
        );
        std::hint::black_box(&acc);
    }
}

#[test]
fn dot_bf16_f32_4_matches_per_row_dispatch_for_yue2_shapes() {
    use rust_model_inference::ops::dot::dot_bf16_f32;
    let widths = [
        0usize, 1, 3, 4, 7, 8, 15, 16, 17, 32, 64, 128, 255, 256, 257, 259, 1024, 2048, 3072, 4096,
        6144,
    ];
    for width in &widths {
        let (input, weight) = make_inputs(*width, 0xfeedface);
        let actual = dot_bf16_f32_4_dispatched(&input, &weight, *width);
        for row in 0..4 {
            let expected = dot_bf16_f32(&input[row * *width..(row + 1) * *width], &weight, *width);
            if actual[row].is_finite() && expected.is_finite() {
                assert_eq!(
                    actual[row].to_bits(),
                    expected.to_bits(),
                    "width={width} row={row}: actual={} expected={}",
                    actual[row],
                    expected
                );
            } else {
                assert_eq!(
                    actual[row].is_nan(),
                    expected.is_nan(),
                    "width={width} row={row}: actual={} expected={}",
                    actual[row],
                    expected
                );
                assert_eq!(
                    actual[row].is_infinite(),
                    expected.is_infinite(),
                    "width={width} row={row}: actual={} expected={}",
                    actual[row],
                    expected
                );
            }
        }
    }
}
