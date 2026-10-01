#![cfg(feature = "vulkan")]
//! GPU-vs-CPU feasibility probe for porting the Z-Image DiT to Vulkan.
//!
//! The DiT is memory-bandwidth bound, so "the GPU is faster at matmul" is not
//! the question -- the question is whether it is faster *than the 20-core CPU
//! on this GB10 box*, which already streams 6.7 GB of Q8_0 weights per step.
//! This prints the speedup for the exact tensor shapes one DiT step performs,
//! so we can decide whether a day of DiT work buys anything.
use rust_model_inference::ops::float::enable_gpu;
use rust_model_inference::ops::get_vulkan_context;
use rust_model_inference::ops::kernel::q8_0::parallel::matmul_q8_0_quantized_parallel;
use std::time::Instant;

/// Build a Q8_0 block row of `n_out` rows, matching the layout the GPU shader
/// and `matmul_q8_0_quantized_parallel` both expect: 34 bytes per 32-value
/// block, i.e. 16 f16 weights + 16 int8 weights + one f16 scale.
fn synth_q8_0(n_in: usize, n_out: usize) -> Vec<u8> {
    let blocks = n_in / 32;
    let mut w = vec![0u8; n_out * blocks * 34];
    for (i, b) in w.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    for row in 0..n_out {
        for b in 0..blocks {
            let off = (row * blocks + b) * 34;
            w[off] = 0x00;
            w[off + 1] = 0x18;
        }
    }
    w
}

fn bench_one(
    ctx: &rust_model_inference::vulkan::VulkanContext,
    label: &str,
    n_in: usize,
    n_out: usize,
) {
    let weight = synth_q8_0(n_in, n_out);
    let input_q8: Vec<u8> = (0..n_in).map(|i| (i % 63) as u8).collect();
    let input_scales: Vec<f32> = vec![0.001; n_in / 32];
    let mut gpu_out = vec![0f32; n_out];
    let mut cpu_out = vec![0f32; n_out];

    // Warm the pipeline so shader JIT is not charged to either side.
    unsafe { ctx.matmul_q8_0(&weight, &input_q8, &input_scales, &mut gpu_out, n_in, n_out) }
        .expect("gpu matmul warmup");
    matmul_q8_0_quantized_parallel(&weight, &input_q8, &input_scales, &mut cpu_out, n_in, n_out);

    // Size the iteration count from the work itself so a single call cannot be
    // dominated by submit overhead on the small shapes.
    let macs = (n_in as f64) * (n_out as f64);
    let iters = (if macs < 2e6 { 20_000 } else { 30 } as u64).max(10);

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe { ctx.matmul_q8_0(&weight, &input_q8, &input_scales, &mut gpu_out, n_in, n_out) }
            .expect("gpu matmul");
    }
    let gpu_us = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;

    let t0 = Instant::now();
    for _ in 0..iters {
        matmul_q8_0_quantized_parallel(&weight, &input_q8, &input_scales, &mut cpu_out, n_in, n_out);
    }
    let cpu_us = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;

    // Report the largest disagreement so a fast-but-wrong GPU cannot look good.
    let max_abs = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "{label:>22} ({n_in:5},{n_out:5}): gpu {gpu_us:9.1} µs  cpu {cpu_us:9.1} µs  \
         speedup {:5.2}x  |Δ|max {max_abs:.3e}",
        cpu_us / gpu_us,
    );
}

fn main() {
    enable_gpu();
    let ctx = get_vulkan_context().expect("no vulkan");
    println!("device: {}", ctx.device_name());

    // Z-Image Turbo at 512x512, 8 steps. The latent is 32 channels on a 64x64
    // grid, patched 2x2 with a 2x text prefix stream, which is the 4096-token
    // sequence the attention runs over.
    println!("\n-- one DiT step, 512x512 (seq=4096) --");
    let seq = 4096usize;
    let dim = 3072usize; // Z-Image Turbo hidden size
    for (label, n_out) in [
        ("attn qkv proj", 3 * dim),
        ("attn out proj", dim),
        ("mlp gate+up", 2 * 2 * dim),
        ("mlp down", dim),
    ] {
        bench_one(&ctx, label, seq, n_out);
    }
}
