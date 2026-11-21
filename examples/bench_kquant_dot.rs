//! Kernel-level microbenchmark for the Q2_K / Q3_K × Q8_K dot products.
//!
//! Measures the scalar reference and the dispatched (AVX2 on x86_64)
//! implementations directly on synthetic super-block data shaped like a real
//! matmul row (n_in = 4096 → 16 super-blocks per row, 512 rows), avoiding
//! end-to-end noise (frequency scaling, sampling, KV work).
//!
//! Usage: cargo run --profile release-fast --example bench_kquant_dot

use rust_model_inference::ops::quant::{
    vec_dot_q2k_q8k, vec_dot_q2k_q8k_scalar, vec_dot_q3k_q8k, vec_dot_q3k_q8k_scalar, BlockQ8K,
};

const N_ROWS: usize = 512;
const N_IN: usize = 4096; // 16 super-blocks of 256 per row
const ITERS: usize = 200;

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn prng_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| (xorshift(&mut state).wrapping_mul(0x2545_f491_4f6c_dd1d) >> 24) as u8)
        .collect()
}

fn quantize_row_q8_k(x: &[f32]) -> Vec<BlockQ8K> {
    let nb = x.len() / 256;
    let mut out = Vec::with_capacity(nb);
    for b in 0..nb {
        let chunk = &x[b * 256..(b + 1) * 256];
        let amax = chunk.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = 127.0 / amax;
        let mut qs = [0i8; 256];
        let mut bsums = [0i16; 16];
        for (i, &v) in chunk.iter().enumerate() {
            let q = ((v * id).round() as i32).clamp(-127, 127) as i8;
            qs[i] = q;
            bsums[i / 16] += q as i16;
        }
        out.push(BlockQ8K { d, qs, bsums });
    }
    out
}

fn bench<F: FnMut() -> f32>(name: &str, mut f: F) -> f64 {
    // warmup
    let mut sink = 0.0f32;
    for _ in 0..8 {
        sink += f();
    }
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        sink += f();
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let ns_per_row = elapsed / (ITERS as f64 * N_ROWS as f64) * 1e9;
    println!(
        "{name:<38} {:>8.1} ms total  {:>7.0} ns/row  (sink {:.3}, iters {ITERS})",
        elapsed * 1e3,
        ns_per_row,
        sink
    );
    ns_per_row
}

fn main() {
    println!("N_ROWS={N_ROWS} N_IN={N_IN} ITERS={ITERS}");

    // Q2_K weights: 84 bytes per 256 elements.
    let mut state = 0x2026_1121_9e37_79b9u64;
    let mut wq2 = prng_bytes(0x22_0001, N_ROWS * N_IN / 256 * 84);
    for (b, bytes) in wq2.chunks_exact_mut(84).enumerate() {
        let bits = rust_model_inference::ops::f32_to_f16(
            0.01 + 0.001 * ((b % 7) as f32) + ((xorshift(&mut state) >> 60) as f32) * 1e-5,
        );
        bytes[80] = (bits & 0xff) as u8;
        bytes[81] = (bits >> 8) as u8;
        bytes[82] = bytes[80];
        bytes[83] = bytes[81];
    }

    // Q3_K weights: 110 bytes per 256 elements.
    let mut wq3 = prng_bytes(0x33_0001, N_ROWS * N_IN / 256 * 110);
    for (b, bytes) in wq3.chunks_exact_mut(110).enumerate() {
        let bits = rust_model_inference::ops::f32_to_f16(
            0.01 + 0.001 * ((b % 5) as f32) + ((xorshift(&mut state) >> 60) as f32) * 1e-5,
        );
        bytes[108] = (bits & 0xff) as u8;
        bytes[109] = (bits >> 8) as u8;
    }

    let input: Vec<f32> = (0..N_IN)
        .map(|i| (((i * 37 + 11) % 251) as f32 - 125.0) * 0.7)
        .collect();
    let q8k = quantize_row_q8_k(&input);

    println!("--- Q2_K × Q8_K ---");
    let s_ns = bench("scalar", || {
        wq2.chunks_exact(N_IN / 256 * 84)
            .map(|row| vec_dot_q2k_q8k_scalar(row, &q8k))
            .sum::<f32>()
    });
    let a_ns = bench("dispatched (AVX2)", || {
        wq2.chunks_exact(N_IN / 256 * 84)
            .map(|row| vec_dot_q2k_q8k(row, &q8k))
            .sum::<f32>()
    });
    println!("speedup: {:.2}x", s_ns / a_ns);

    println!("--- Q3_K × Q8_K ---");
    let s_ns = bench("scalar", || {
        wq3.chunks_exact(N_IN / 256 * 110)
            .map(|row| vec_dot_q3k_q8k_scalar(row, &q8k))
            .sum::<f32>()
    });
    let a_ns = bench("dispatched (AVX2)", || {
        wq3.chunks_exact(N_IN / 256 * 110)
            .map(|row| vec_dot_q3k_q8k(row, &q8k))
            .sum::<f32>()
    });
    println!("speedup: {:.2}x", s_ns / a_ns);
}
