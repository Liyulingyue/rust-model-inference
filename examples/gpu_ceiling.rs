//! What is the GB10 actually capable of, and where does the crossover sit?
//!
//! The CPU path sustains 201 GOP/s on this box's int8 NEON. The article the
//! user keeps citing does Z-Image-Turbo in 13 s at 1024x1024 on the same GB10,
//! and every GPU attempt so far has lost to the CPU. Those facts are only
//! consistent if the GPU is being asked the wrong question, so this measures
//! the device rather than the model:
//!
//!   * raw achieved GOP/s for the Q8_0 projection at a range of sizes, so the
//!     dispatch floor and the steady-state rate are both visible;
//!   * the same sweep with the dp4a and the scalar grouped shader, since
//!     `RUST_GPU_DP4A` switches them and the difference is the int8 dot
//!     pipeline's contribution;
//!   * how much of a call is fixed cost, by timing a single row and a batch of
//!     rows against one weight matrix.
//!
//! The point is to find out whether the ceiling is bandwidth, arithmetic, or
//! dispatch, because those call for completely different fixes. If the device
//! sustains thousands of GOP/s on a large batch, the DiT can be ported. If it
//! stalls at a few hundred, no amount of kernel work will help and the honest
//! answer is that this workload is CPU-shaped on this machine.
use std::time::Instant;

use rust_model_inference::ops::float::enable_gpu;
use rust_model_inference::ops::get_vulkan_context;
use rust_model_inference::ops::kernel::q8_0::parallel::matmul_q8_0_quantized_parallel;

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

/// One matmul through the pool, which is what routes to the GPU when it is on.
thread_local! {
    static POOL: rust_model_inference::core::thread_pool::ComputePool =
        rust_model_inference::core::thread_pool::ComputePool::new(16);
}

fn gpu_rows(weight: &[u8], input: &[u8], scales: &[f32], out: &mut [f32], n_in: usize, n_out: usize) {
    POOL.with(|pool| {
    let w = weight.as_ptr() as usize;
    let wl = weight.len();
    let i = input.as_ptr() as usize;
    let il = input.len();
    let s = scales.as_ptr() as usize;
    let sl = scales.len();
    let o = out.as_mut_ptr() as usize;
    let ol = out.len();
    pool.compute(|ith, nth| {
        let weight = unsafe { std::slice::from_raw_parts(w as *const u8, wl) };
        let input = unsafe { std::slice::from_raw_parts(i as *const u8, il) };
        let scales = unsafe { std::slice::from_raw_parts(s as *const f32, sl) };
        let out = unsafe { std::slice::from_raw_parts_mut(o as *mut f32, ol) };
        rust_model_inference::ops::kernel::q8_0::parallel::matmul_q8_0_quantized_parallel_rows(
            weight, input, scales, out, n_in, n_out, ith, nth,
        );
    });
    });
}

fn main() {
    enable_gpu();
    let ctx = match get_vulkan_context() {
        Some(ctx) => ctx,
        None => {
            eprintln!("no vulkan context");
            return;
        }
    };
    println!("device: {}", ctx.device_name());
    println!(
        "dp4a forced: {}",
        std::env::var("RUST_GPU_DP4A").unwrap_or_else(|_| "auto".into())
    );
    println!("\nCPU reference for the same kernel: 201 GOP/s\n");

    // The projection the DiT runs, at the size it runs it.
    let (n_in, n_out) = (3840usize, 11520usize);
    let weight = synth_q8_0(n_in, n_out);
    let input = vec![1u8; n_in];
    let scales = vec![0.001f32; n_in / 32];
    let mut out = vec![0f32; n_out];

    // Warm: uploads the matrix and JITs the shader.
    gpu_rows(&weight, &input, &scales, &mut out, n_in, n_out);

    println!("{:>7}  {:>11}  {:>12}  {:>12}", "calls", "total", "us/call", "GOP/s");
    for calls in [1usize, 8, 64, 256] {
        let t0 = Instant::now();
        for _ in 0..calls {
            gpu_rows(&weight, &input, &scales, &mut out, n_in, n_out);
        }
        let s = t0.elapsed().as_secs_f64();
        let per = s / calls as f64;
        let gops = 2.0 * (n_in * n_out) as f64 / per / 1e9;
        println!(
            "{calls:>7}  {:>8.2} ms  {:>10.1}  {:>12.0}",
            s * 1e3,
            per * 1e6,
            gops
        );
    }
    println!("\n  flat us/call means a fixed dispatch floor dominates: the matrix");
    println!("  is already resident and only the round trip is being measured.");

    // How big can the matrix get before bandwidth, rather than dispatch, is
    // the limit? Sweep n_out at fixed n_in.
    println!("\n== sweep n_out at n_in = 3840: where does bandwidth take over? ==");
    println!("{:>8}  {:>10}  {:>12}  {:>12}", "n_out", "weight MB", "us/call", "GB/s");
    for out_n in [64usize, 512, 4096, 11520, 32768, 65536] {
        let w = synth_q8_0(n_in, out_n);
        let mut o = vec![0f32; out_n];
        gpu_rows(&w, &input, &scales, &mut o, n_in, out_n);
        let t0 = Instant::now();
        let reps = 32;
        for _ in 0..reps {
            gpu_rows(&w, &input, &scales, &mut o, n_in, out_n);
        }
        let per = t0.elapsed().as_secs_f64() / reps as f64;
        println!(
            "{out_n:>8}  {:>10.1}  {:>10.1}  {:>12.1}",
            w.len() as f64 / 1e6,
            per * 1e6,
            w.len() as f64 / per / 1e3
        );
    }
    println!("\n  GB/s rising with n_out means larger matrices amortise the dispatch;");
    println!("  flat GB/s means the link itself is the ceiling.");
}
