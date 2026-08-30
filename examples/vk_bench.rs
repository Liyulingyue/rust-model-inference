#![cfg(feature = "vulkan")]
use rust_model_inference::ops::float::enable_gpu;
use rust_model_inference::ops::get_vulkan_context;
use std::time::Instant;

fn main() {
    enable_gpu();
    let ctx = get_vulkan_context().expect("no vulkan");
    println!("device: {}", ctx.device_name());

    for (n_in, n_out) in [(1024usize, 1usize), (1024, 3072), (1024, 3072), (3072, 1024), (1024, 151936)] {
        let blocks = n_in / 32;
        let mut weight = vec![0u8; n_out * blocks * 34];
        for (i, b) in weight.iter_mut().enumerate() { *b = (i % 251) as u8; }
        // make scales valid f16-ish: set high byte of each 2-byte scale to 0x18 (small exponent)
        for row in 0..n_out { for b in 0..blocks {
            let off = (row * blocks + b) * 34;
            weight[off] = 0x00; weight[off + 1] = 0x18;
        }}
        let input_q8: Vec<u8> = (0..n_in).map(|i| (i % 63) as u8).collect();
        let input_scales: Vec<f32> = vec![0.001; blocks];
        let mut out = vec![0f32; n_out];
        unsafe { ctx.matmul_q8_0(&weight, &input_q8, &input_scales, &mut out, n_in, n_out).unwrap(); }

        let iters = if n_out <= 4096 { 20000 } else { 50 };
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe { ctx.matmul_q8_0(&weight, &input_q8, &input_scales, &mut out, n_in, n_out).unwrap(); }
        }
        let per = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;
        let macs = (n_in as f64) * (n_out as f64) / 1e6;
        println!("({n_in:6},{n_out:6}): {per:8.1} µs/call  ({macs:6.1} MMAC → {:.1} GMAC/s)", macs / (per / 1e6) / 1000.0);
    }
}
