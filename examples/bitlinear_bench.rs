//! Microbenchmark: AVX2 vs scalar BitLinear forward on a realistic
//! BitNet-Embedding 0.6B projection shape (1024×1024 attn_q).
//!
//! Run with:
//!   cargo run --profile release-fast --example bitlinear_bench

use rust_model_inference::ops::bitnet::forward_avx2::dequant_i2_s_to_i8;
use rust_model_inference::ops::bitnet::{
    bitlinear_forward, bitlinear_forward_from_f32, bitlinear_forward_packed,
    bitlinear_forward_scalar, quantize_activation_per_token,
};
use std::hint::black_box;
use std::time::Instant;

fn time_scalar(
    weights: &[u8],
    x_q: &[i8],
    absmax: f32,
    n_in: usize,
    n_out: usize,
    iters: usize,
) -> u128 {
    let mut y = vec![0.0f32; n_out];
    let start = Instant::now();
    for _ in 0..iters {
        black_box(bitlinear_forward_scalar(
            weights, x_q, absmax, n_in, n_out, &mut y,
        ));
    }
    start.elapsed().as_nanos()
}

fn time_dispatch(
    weights: &[u8],
    x_q: &[i8],
    absmax: f32,
    n_in: usize,
    n_out: usize,
    iters: usize,
) -> u128 {
    let mut y = vec![0.0f32; n_out];
    let start = Instant::now();
    for _ in 0..iters {
        black_box(bitlinear_forward(weights, x_q, absmax, n_in, n_out, &mut y));
    }
    start.elapsed().as_nanos()
}

fn time_packed(
    weights_packed: &[i8],
    x_q: &[i8],
    absmax: f32,
    n_in: usize,
    n_out: usize,
    iters: usize,
) -> u128 {
    let mut y = vec![0.0f32; n_out];
    let start = Instant::now();
    for _ in 0..iters {
        black_box(bitlinear_forward_packed(
            weights_packed,
            x_q,
            absmax,
            n_in,
            n_out,
            &mut y,
        ));
    }
    start.elapsed().as_nanos()
}

fn main() {
    let n_in = 1024;
    let n_out = 1024;
    let iters = 200;

    let mut weights = vec![0u8; n_in / 128 * 32 * n_out];
    for (idx, byte) in weights.iter_mut().enumerate() {
        *byte = ((idx as u8).wrapping_mul(31) ^ 0x5a) & 0b11;
        if *byte == 0b11 {
            *byte = 0b01;
        }
    }

    // Pre-pack once (mirrors what model-load would do).
    let mut weights_packed: Vec<i8> = vec![0i8; n_in * n_out];
    dequant_i2_s_to_i8(&weights, n_in, n_out, &mut weights_packed);

    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32) * 0.013).sin() * 0.7 - 0.4)
        .collect();
    let (x_q, absmax) = quantize_activation_per_token(&x);

    // Warmup
    let mut y = vec![0.0f32; n_out];
    for _ in 0..5 {
        bitlinear_forward_scalar(&weights, &x_q, absmax, n_in, n_out, &mut y);
        bitlinear_forward(&weights, &x_q, absmax, n_in, n_out, &mut y);
        bitlinear_forward_packed(&weights_packed, &x_q, absmax, n_in, n_out, &mut y);
    }

    let scalar_ns = time_scalar(&weights, &x_q, absmax, n_in, n_out, iters);
    let dispatch_ns = time_dispatch(&weights, &x_q, absmax, n_in, n_out, iters);
    let packed_ns = time_packed(&weights_packed, &x_q, absmax, n_in, n_out, iters);

    let scalar_us = scalar_ns as f64 / iters as f64 / 1000.0;
    let dispatch_us = dispatch_ns as f64 / iters as f64 / 1000.0;
    let packed_us = packed_ns as f64 / iters as f64 / 1000.0;
    let speedup_dispatch = scalar_us / dispatch_us;
    let speedup_packed = scalar_us / packed_us;

    println!("BitLinear microbenchmark: {n_in}×{n_out}, {iters} iters");
    println!("  scalar reference:       {scalar_us:.2} us/iter");
    println!(
        "  dispatcher (unpacked):  {dispatch_us:.2} us/iter  ({speedup_dispatch:.2}x vs scalar)"
    );
    println!("  dispatcher (packed):    {packed_us:.2} us/iter  ({speedup_packed:.2}x vs scalar)");
    println!(
        "  AVX2 active:            {}",
        rust_model_inference::ops::has_avx2_fma()
    );

    // Bit-exactness check.
    let mut y_scalar = vec![0.0f32; n_out];
    let mut y_dispatch = vec![0.0f32; n_out];
    let mut y_packed = vec![0.0f32; n_out];
    bitlinear_forward_scalar(&weights, &x_q, absmax, n_in, n_out, &mut y_scalar);
    bitlinear_forward(&weights, &x_q, absmax, n_in, n_out, &mut y_dispatch);
    bitlinear_forward_packed(&weights_packed, &x_q, absmax, n_in, n_out, &mut y_packed);
    let max_diff: f32 = y_scalar
        .iter()
        .zip(y_packed.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  packed vs scalar diff:    {max_diff:.3e} (must be 0.0)");

    // Verify against the convenience wrapper too.
    let mut y_wrap = vec![0.0f32; n_out];
    bitlinear_forward_from_f32(&weights, &x, n_in, n_out, &mut y_wrap);
    let wrap_diff: f32 = y_scalar
        .iter()
        .zip(y_wrap.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  from_f32 vs scalar diff: {wrap_diff:.3e} (must be 0.0)");
}
