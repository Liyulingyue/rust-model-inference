use rust_model_inference::ops::bitnet::{bitlinear_forward, quantize_activation_per_token};
use rust_model_inference::ops::kernel::i2_s::{dequant_i2_s_row, BLOCK_I2_S_SIZE, QK_I2_S};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let n_in = 1024;
    let n_out = 1024;
    let iters = 200;
    let mut weights = vec![0u8; n_in / QK_I2_S * 32 * n_out];
    for (idx, byte) in weights.iter_mut().enumerate() {
        *byte = ((idx as u8).wrapping_mul(31) ^ 0x5a) & 0b11;
        if *byte == 0b11 {
            *byte = 0b01;
        }
    }
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i as f32) * 0.013).sin() * 0.7 - 0.4)
        .collect();
    let (x_q, absmax) = quantize_activation_per_token(&x);

    let mut y = vec![0.0f32; n_out];
    for _ in 0..5 {
        bitlinear_forward(&weights, &x_q, absmax, n_in, n_out, &mut y);
    }

    let t = Instant::now();
    for _ in 0..iters {
        black_box(bitlinear_forward(
            &weights, &x_q, absmax, n_in, n_out, &mut y,
        ));
    }
    let full_ns = t.elapsed().as_nanos() / iters;

    let mut dequant = vec![0.0f32; n_in];
    let row_bytes = n_in / QK_I2_S * BLOCK_I2_S_SIZE;
    let t = Instant::now();
    for _ in 0..iters {
        for j in 0..n_out {
            let row = black_box(&weights[j * row_bytes..(j + 1) * row_bytes]);
            black_box(dequant_i2_s_row(row, n_in, &mut dequant));
        }
        black_box(dequant.as_mut_ptr());
    }
    let dequant_ns = t.elapsed().as_nanos() / iters;

    let mut all_dequant = vec![0.0f32; n_in * n_out];
    for j in 0..n_out {
        dequant_i2_s_row(
            &weights[j * row_bytes..(j + 1) * row_bytes],
            n_in,
            &mut all_dequant[j * n_in..(j + 1) * n_in],
        );
    }
    let t = Instant::now();
    for _ in 0..iters {
        for j in 0..n_out {
            let mut acc = 0.0f32;
            for i in 0..n_in {
                acc += all_dequant[j * n_in + i] * x[i];
            }
            y[j] = acc * (absmax / 127.0);
        }
        black_box(y.as_mut_ptr());
    }
    let dot_ns = t.elapsed().as_nanos() / iters;

    println!("Decompose {n_in}x{n_out} BitLinear forward ({iters} iters):");
    println!("  full (scalar):      {} us/iter", full_ns as f64 / 1000.0);
    println!(
        "  dequant-only:       {} us/iter ({:.1}%)",
        dequant_ns as f64 / 1000.0,
        dequant_ns as f64 / full_ns as f64 * 100.0
    );
    println!(
        "  dot-only:           {} us/iter ({:.1}%)",
        dot_ns as f64 / 1000.0,
        dot_ns as f64 / full_ns as f64 * 100.0
    );
}
