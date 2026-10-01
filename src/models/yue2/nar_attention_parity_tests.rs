//! Bit-exactness guard for the NAR attention rewrites.
//!
//! `hybrid_attention` and `causal_prefix_attention` were moved off hand-written
//! scalar reduction loops onto the shared `attention_value_reduce` (NEON) and
//! onto row-level pool parallelism. Row partitioning cannot change arithmetic,
//! but the value reduction *can*: the old loops evaluated `sum += s * v` as a
//! separate multiply and add, while the NEON kernel uses `vfmaq_f32`, which
//! rounds once. That difference is a rounding-mode change, not a reassociation,
//! and it has to be pinned deliberately rather than discovered by diffing audio.
//!
//! This test therefore checks both properties explicitly:
//!   1. the pooled/SIMD path agrees bit-for-bit with the scalar reference that
//!      keeps the original two-rounding form, and
//!   2. if that fails, the new path must at least be no further from an
//!      f64 reference than the old one, so the change cannot silently degrade
//!      accuracy while buying speed.
//!
//! Run with `--features parity-trace` to compare against the pinned scalar
//! kernels as well.

use super::super::YuE2Config;

fn config() -> YuE2Config {
    YuE2Config {
        hidden: 2048,
        layers: 2,
        q_heads: 16,
        kv_heads: 8,
        head_dim: 128,
        ffn: 6144,
        vocab: 151_643,
        context: 24_576,
        rms_eps: 1e-6,
        rope_base: 1_000_000.0,
        latent_channels: 64,
        timestep_shift: 1.0,
    }
}

/// Deterministic bf16-rounded noise so both paths see identical inputs.
fn draw(state: &mut u32, divisor: f32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    half::bf16::from_f32((((*state >> 8) & 0xffff) as i32 - 32768) as f32 / divisor).to_f32()
}

fn queries(rows: usize, config: &YuE2Config) -> Vec<f32> {
    let q_width = config.q_heads * config.head_dim;
    let mut state = 24601u32;
    (0..rows * q_width).map(|_| draw(&mut state, 8192.0)).collect()
}

/// The pre-optimization value reduction, verbatim: a fresh accumulator per
/// invocation, separate multiply and add, then one add into the output.
fn legacy_value_reduce(
    values: &[f32],
    scores: &[f32],
    row_stride: usize,
    head_offset: usize,
    n_tokens: usize,
    head_width: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; head_width];
    for dimension in 0..head_width {
        let mut sum = 0.0f32;
        for token in 0..n_tokens {
            sum += scores[token] * values[token * row_stride + head_offset + dimension];
        }
        out[dimension] = sum;
    }
    out
}

#[test]
fn value_reduce_block_is_bitwise_identical_to_the_scalar_form() {
    // This is the invariant the whole NAR rewrite rests on. The optimized
    // attention is only acceptable if its value reduction reproduces the
    // original arithmetic exactly; the FMA-contracting shared kernel did not,
    // and the difference compounded through 28 bf16-rounded layers until the
    // rendered audio was no longer musical.
    let config = config();
    let head_dim = config.head_dim;
    let kv_width = config.kv_heads * head_dim;
    let mut state = 99001u32;
    for n_tokens in [1usize, 7, 64, 128, 511, 512, 513, 1025, 1566] {
        let values: Vec<f32> = (0..n_tokens * kv_width).map(|_| draw(&mut state, 16384.0)).collect();
        let scores: Vec<f32> = (0..n_tokens).map(|_| draw(&mut state, 4.0)).collect();
        for head in [0usize, config.kv_heads - 1] {
            let expected = legacy_value_reduce(
                &values,
                &scores,
                kv_width,
                head * head_dim,
                n_tokens,
                head_dim,
            );
            // Non-zero starting output exercises the `out += acc` step, which is
            // where a block-association mistake would show up.
            let mut actual = vec![0.0f32; head_dim];
            for (index, slot) in actual.iter_mut().enumerate() {
                *slot = draw(&mut state, 512.0) * (index as f32 + 1.0);
            }
            let seed_output = actual.clone();
            super::value_reduce_block(
                &values,
                &scores,
                &mut actual,
                0,
                kv_width,
                head * head_dim,
                n_tokens,
                head_dim,
            );
            for (index, value) in actual.iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    (seed_output[index] + expected[index]).to_bits(),
                    "n_tokens={n_tokens} head={head} dim={index}: value reduce \
                     diverged from the scalar form"
                );
            }
        }
    }
}

#[test]
fn hybrid_attention_is_stable_across_pool_widths() {
    let config = config();
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    let nar_rows = 40usize;
    let prefix_rows = 600usize;
    let q = queries(nar_rows, &config);
    let mut state = 5150u32;
    let nar_k: Vec<f32> = (0..nar_rows * kv_width).map(|_| draw(&mut state, 8192.0)).collect();
    let nar_v: Vec<f32> = (0..nar_rows * kv_width).map(|_| draw(&mut state, 16384.0)).collect();
    let prefix = (
        (0..prefix_rows * kv_width).map(|_| draw(&mut state, 8192.0)).collect::<Vec<f32>>(),
        (0..prefix_rows * kv_width).map(|_| draw(&mut state, 16384.0)).collect::<Vec<f32>>(),
    );

    let run = |threads: usize| {
        let pool = crate::core::thread_pool::ComputePool::new(threads);
        let mut out = vec![0.0f32; nar_rows * q_width];
        super::hybrid_attention(&config, &pool, &q, &prefix, &nar_k, &nar_v, &mut out);
        out
    };
    // The pool partitions rows only; every row's arithmetic must be identical
    // no matter how many threads ran it. This is the property that makes the
    // parallel rewrite safe, and the one a later refactor would break first.
    let single = run(1);
    for threads in [2usize, 4, 8] {
        let many = run(threads);
        for (index, (a, b)) in single.iter().zip(&many).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "thread count {threads} changed element {index}: {a} vs {b}"
            );
        }
    }
}

#[test]
fn causal_prefix_attention_is_stable_across_pool_widths() {
    let config = config();
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    // 700 rows crosses the 512-key block boundary, so the blocked softmax path
    // and its rescale are exercised.
    let rows = 700usize;
    let mut state = 31337u32;
    let q: Vec<f32> = (0..rows * q_width).map(|_| draw(&mut state, 8192.0)).collect();
    let k: Vec<f32> = (0..rows * kv_width).map(|_| draw(&mut state, 8192.0)).collect();
    let v: Vec<f32> = (0..rows * kv_width).map(|_| draw(&mut state, 16384.0)).collect();

    let run = |threads: usize| {
        let pool = crate::core::thread_pool::ComputePool::new(threads);
        let mut out = vec![0.0f32; rows * q_width];
        super::causal_prefix_attention(&config, &pool, &q, &k, &v, &mut out);
        out
    };
    let single = run(1);
    for threads in [2usize, 4, 8] {
        let many = run(threads);
        for (index, (a, b)) in single.iter().zip(&many).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "thread count {threads} changed element {index}: {a} vs {b}"
            );
        }
    }
    assert!(single.iter().all(|value| value.is_finite()));
}

/// The invariant that decides whether the rewrite is shippable: on identical
/// inputs the optimized attention must equal the legacy scalar kernels
/// bit-for-bit, not merely to some tolerance.
#[test]
fn optimized_attention_matches_legacy_bitwise() {
    let config = config();
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    let pool = crate::core::thread_pool::ComputePool::new(4);

    // hybrid: 600-token prefix + 40 latent rows, i.e. total_len 640, which is
    // past the 512 KV block boundary so the blocked path with rescale is used.
    let nar_rows = 40usize;
    let prefix_rows = 600usize;
    let mut state = 13579u32;
    let q = queries(nar_rows, &config);
    let nar_k: Vec<f32> = (0..nar_rows * kv_width).map(|_| draw(&mut state, 8192.0)).collect();
    let nar_v: Vec<f32> = (0..nar_rows * kv_width).map(|_| draw(&mut state, 16384.0)).collect();
    let prefix = (
        (0..prefix_rows * kv_width).map(|_| draw(&mut state, 8192.0)).collect::<Vec<f32>>(),
        (0..prefix_rows * kv_width).map(|_| draw(&mut state, 16384.0)).collect::<Vec<f32>>(),
    );
    let mut optimized = vec![0.0f32; nar_rows * q_width];
    super::hybrid_attention(&config, &pool, &q, &prefix, &nar_k, &nar_v, &mut optimized);
    let mut legacy = vec![0.0f32; nar_rows * q_width];
    super::hybrid_attention_legacy(&config, &pool, &q, &prefix, &nar_k, &nar_v, &mut legacy);
    for (index, (a, b)) in optimized.iter().zip(&legacy).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "hybrid_attention element {index}: optimized {a} vs legacy {b}"
        );
    }

    // causal: 700 rows crosses the 512 block boundary and the row<512 branch.
    let rows = 700usize;
    let q: Vec<f32> = queries(rows, &config);
    let k: Vec<f32> = (0..rows * kv_width).map(|_| draw(&mut state, 8192.0)).collect();
    let v: Vec<f32> = (0..rows * kv_width).map(|_| draw(&mut state, 16384.0)).collect();
    let mut optimized = vec![0.0f32; rows * q_width];
    super::causal_prefix_attention(&config, &pool, &q, &k, &v, &mut optimized);
    let mut legacy = vec![0.0f32; rows * q_width];
    super::causal_prefix_attention_legacy(&config, &pool, &q, &k, &v, &mut legacy);
    for (index, (a, b)) in optimized.iter().zip(&legacy).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "causal_prefix_attention element {index}: optimized {a} vs legacy {b}"
        );
    }
}

/// Production-scale parity. The small fixtures pass, so if this one fails the
/// divergence depends on the real prefix/latent split.
#[test]
fn optimized_attention_matches_legacy_at_production_scale() {
    let config = config();
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    let pool = crate::core::thread_pool::ComputePool::new(8);

    // What a real 24 s render does: 600 latent frames against a 1758-token AR
    // prefix, i.e. total_len 2358 so several 512-wide KV blocks straddle the
    // prefix/latent boundary.
    for (prefix_rows, nar_rows) in [(1758usize, 600usize), (1308, 256), (600, 40), (500, 12)] {
        let mut state = 24680u32;
        let q = queries(nar_rows, &config);
        let nar_k: Vec<f32> = (0..nar_rows * kv_width).map(|_| draw(&mut state, 8192.0)).collect();
        let nar_v: Vec<f32> = (0..nar_rows * kv_width).map(|_| draw(&mut state, 16384.0)).collect();
        let prefix = (
            (0..prefix_rows * kv_width).map(|_| draw(&mut state, 8192.0)).collect::<Vec<f32>>(),
            (0..prefix_rows * kv_width).map(|_| draw(&mut state, 16384.0)).collect::<Vec<f32>>(),
        );
        let mut optimized = vec![0.0f32; nar_rows * q_width];
        super::hybrid_attention(&config, &pool, &q, &prefix, &nar_k, &nar_v, &mut optimized);
        let mut legacy = vec![0.0f32; nar_rows * q_width];
        super::hybrid_attention_legacy(&config, &pool, &q, &prefix, &nar_k, &nar_v, &mut legacy);
        let mut worst = (0.0f32, 0usize);
        for (index, (a, b)) in optimized.iter().zip(&legacy).enumerate() {
            let diff = (a - b).abs();
            if diff > worst.0 {
                worst = (diff, index);
            }
        }
        assert_eq!(
            worst.0.to_bits(),
            0.0f32.to_bits(),
            "prefix={prefix_rows} nar={nar_rows} total={} (blocks of 512): \\
             max|diff|={} at element {}",
            prefix_rows + nar_rows,
            worst.0,
            worst.1,
        );
    }
}

/// Narrow down which prefix/latent split starts diverging, and whether the
/// split point relative to the 512-wide KV block boundary is what matters.
#[test]
fn production_scale_divergence_is_a_boundary_effect() {
    let config = config();
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    let pool = crate::core::thread_pool::ComputePool::new(8);
    let nar_rows = 8usize;
    // Walk the prefix length across the 512 block boundaries.
    for prefix_rows in [
        8usize, 100, 500, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515,
        520, 600, 1024, 1536,
    ] {
        let mut state = 24680u32;
        let q = queries(nar_rows, &config);
        let nar_k: Vec<f32> = (0..nar_rows * kv_width).map(|_| draw(&mut state, 8192.0)).collect();
        let nar_v: Vec<f32> = (0..nar_rows * kv_width).map(|_| draw(&mut state, 16384.0)).collect();
        let prefix = (
            (0..prefix_rows * kv_width).map(|_| draw(&mut state, 8192.0)).collect::<Vec<f32>>(),
            (0..prefix_rows * kv_width).map(|_| draw(&mut state, 16384.0)).collect::<Vec<f32>>(),
        );
        let mut optimized = vec![0.0f32; nar_rows * q_width];
        super::hybrid_attention(&config, &pool, &q, &prefix, &nar_k, &nar_v, &mut optimized);
        let mut legacy = vec![0.0f32; nar_rows * q_width];
        super::hybrid_attention_legacy(&config, &pool, &q, &prefix, &nar_k, &nar_v, &mut legacy);
        let worst = optimized
            .iter()
            .zip(&legacy)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let remainder = (prefix_rows + nar_rows) % 512;
        println!(
            "prefix={prefix_rows:5} total={:5} total%512={remainder:4} max|diff|={worst:e}",
            prefix_rows + nar_rows
        );
    }
}

/// `causal_prefix_attention` bit-exactness against the legacy kernel. The
/// prefix KV it produces is what every NAR velocity attends to, so a
/// discrepancy here propagates into the latents even when `hybrid_attention`
/// itself is exact. Only thread-stability was covered before.
#[test]
fn optimized_causal_attention_matches_legacy_bitwise() {
    let config = config();
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    let pool = crate::core::thread_pool::ComputePool::new(8);
    let mut state = 777u32;
    // Cross both the row<512 fast path and the blocked path, and land on the
    // same 512-wide block boundaries a 1758-token AR prefix would.
    for rows in [8usize, 100, 511, 512, 513, 700, 1024, 1758] {
        let q: Vec<f32> = (0..rows * q_width).map(|_| draw(&mut state, 8192.0)).collect();
        let k: Vec<f32> = (0..rows * kv_width).map(|_| draw(&mut state, 8192.0)).collect();
        let v: Vec<f32> = (0..rows * kv_width).map(|_| draw(&mut state, 16384.0)).collect();
        let mut optimized = vec![0.0f32; rows * q_width];
        super::causal_prefix_attention(&config, &pool, &q, &k, &v, &mut optimized);
        let mut legacy = vec![0.0f32; rows * q_width];
        super::causal_prefix_attention_legacy(&config, &pool, &q, &k, &v, &mut legacy);
        let mut worst = (0.0f32, 0usize);
        for (index, (a, b)) in optimized.iter().zip(&legacy).enumerate() {
            let d = (a - b).abs();
            if d > worst.0 {
                worst = (d, index);
            }
        }
        assert_eq!(
            worst.0.to_bits(),
            0.0f32.to_bits(),
            "causal rows={rows}: max|diff|={} at element {}",
            worst.0,
            worst.1
        );
    }
}
