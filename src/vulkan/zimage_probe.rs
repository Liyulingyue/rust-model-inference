//! Minimal go/no-go for batching the Z-Image DiT onto the GPU.
//!
//! `run_block` in `src/models/diffusion/z_image/dit.rs` calls `linear_into`
//! once per sequence row, so a 512x512 step issues 1536 x 30 x 4 = 184,320
//! dispatches at ~926 us each -- 150.3 s/step, slower than the 140.5 s CPU
//! path. `Qwen3Ops::record_weight_matmul_rows` already accepts an input
//! *matrix* (`token_rows` x `input_stride`) and quantizes it on the GPU, which
//! is exactly what batching would need, but it is `pub(crate)` and wired only
//! for Qwen3's prefill, so an example cannot reach it.
//!
//! This drives it directly with the DiT's real QKV projection shape and
//! compares against the row-at-a-time path the DiT uses today. One question:
//! does a single batched dispatch over 1536 rows cost less than 1536
//! single-row dispatches, by enough to pay for DiT-specific bindings? If not,
//! the port is not worth starting.
//!
//! Opt in with RUST_ZIMAGE_VK_PROBE=1; it needs a Vulkan device and a
//! `--features vulkan` build.
use std::time::Instant;

use crate::core::thread_pool::ComputePool;
use crate::ops::float::enable_gpu;
use crate::ops::get_vulkan_context;
use crate::ops::kernel::q8_0::parallel::matmul_q8_0_quantized_parallel_rows;
use crate::vulkan::ops::{GpuWeightFormat, Qwen3Ops, TokenCommands};

const HIDDEN: usize = 3840;
const QKV: usize = HIDDEN * 3;
const ROWS: usize = 1536; // 512x512: 1024 image tokens + 512 text
const HEADS: usize = 30;
const HEAD_WIDTH: usize = HIDDEN / HEADS;

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

#[test]
fn batched_rows_beat_row_at_a_time_on_the_dit_qkv_shape() {
    if std::env::var("RUST_ZIMAGE_VK_PROBE").is_err() {
        eprintln!("skipped: set RUST_ZIMAGE_VK_PROBE=1 to run");
        return;
    }
    run_probe().expect("batched path unavailable");
}

/// Body split out so the `?` operator is available.
fn run_probe() -> Result<(), String> {
    enable_gpu();
    let context = match get_vulkan_context() {
        Some(context) => context,
        None => {
            eprintln!("skipped: no Vulkan context");
            return Ok(());
        }
    };
    eprintln!("device: {}", context.device_name());

    let weight = synth_q8_0(HIDDEN, QKV);
    let input: Vec<f32> = (0..ROWS * HIDDEN)
        .map(|i| ((i % 251) as f32 / 251.0) - 0.5)
        .collect();

    // --- reference: what run_block does today, one row per dispatch ----------
    let input_q8: Vec<u8> = (0..HIDDEN).map(|i| (i % 63) as u8).collect();
    let input_scales: Vec<f32> = vec![0.001; HIDDEN / 32];
    let pool = ComputePool::new(8);
    let mut row_out = vec![0f32; QKV];
    let row_ptr = row_out.as_mut_ptr();
    let row_at_a_time = |rows: usize| -> f64 {
        let t0 = Instant::now();
        for _ in 0..rows {
            pool.compute(|ith, nth| {
                let out = unsafe { std::slice::from_raw_parts_mut(row_ptr, QKV) };
                matmul_q8_0_quantized_parallel_rows(
                    &weight, &input_q8, &input_scales, out, HIDDEN, QKV, ith, nth,
                );
            });
        }
        t0.elapsed().as_secs_f64()
    };
    // The difference between 2 and 64 is the marginal dispatch price, with the
    // first-call weight upload and shader JIT amortised away.
    let two = row_at_a_time(2);
    let sixty_four = row_at_a_time(64);
    let per_row = (sixty_four - two) / 62.0;
    let per_row_us = per_row * 1e6;
    eprintln!("row-at-a-time: {sixty_four:.4} s / 64 rows -> {per_row_us:.1} us/row");
    eprintln!("  {ROWS} rows would take {:.1} s", per_row * ROWS as f64);

    // --- candidate: one batched dispatch over every row ----------------------
    // `ArenaLayout::for_dims` is sized for a Qwen3 decode step, where the x
    // region only has to hold one token. The DiT needs all ROWS of activations
    // resident, so lay the arena out by hand for this shape.
    //
    // Sizes come from `quantize_rows_push` / `matmul_rows_push`:
    //   x        ROWS * HIDDEN f32
    //   q8       ROWS * HIDDEN i8        scales  ROWS * (HIDDEN/32) f32
    //   q4_1_sum ROWS * (HIDDEN/32) f32  q8k     unused for Q8_0, but sized anyway
    let blocks = HIDDEN / 32;
    let mut cursor = 0usize;
    let mut region = |elements: usize, align: usize| {
        cursor = cursor.div_ceil(align) * align;
        let start = cursor;
        cursor += elements;
        crate::vulkan::ops::ArenaRegion {
            offset: start,
            size: elements,
        }
    };
    let x = region(ROWS * HIDDEN * 4, 4);
    let q8 = region(ROWS * HIDDEN, 4);
    let q8_scales = region(ROWS * blocks * 4, 4);
    let q4_1_input_sums = region(ROWS * blocks * 4, 4);
    let q8k = region(ROWS * HIDDEN, 4);
    let q8k_scales = region(ROWS * blocks * 4, 4);
    let out = region(ROWS * QKV * 4, 4);
    let arena_bytes = cursor.next_power_of_two();
    eprintln!("arena: {:.1} MiB", arena_bytes as f64 / 1048576.0);

    let mut ops = Qwen3Ops::new_with_size(&context, arena_bytes, 8).map_err(|e| e.to_string())?;
    let weight_buffer = unsafe { context.upload_static(&weight) }.map_err(|e| e.to_string())?;
    let bindings = ops
        .bind_weight_buffers(&[weight_buffer], &[GpuWeightFormat::Q8_0])
        .map_err(|e| e.to_string())?;

    // The first call pays shader JIT plus a {:.0} MiB host->device arena
    // upload. The DiT issues 96 of these per step and 8 steps per render, but
    // only the first ever pays that, so the steady state is what decides
    // whether batching wins.
    let mut head_value = 0.0f32;
    let mut first_s = 0.0f64;
    let mut steady_s = 0.0f64;
    let passes = 8;
    for pass in 0..passes {
        let t0 = Instant::now();
        ops.write_f32(x, &input).map_err(|e| e.to_string())?;
        let mut commands = TokenCommands::begin(&context).map_err(|e| e.to_string())?;
        ops.record_weight_matmul_rows(
            &commands,
            bindings,
            x,
            q8,
            q8_scales,
            q4_1_input_sums,
            q8k,
            q8k_scales,
            &[(out, QKV, QKV * 4)],
            HIDDEN,
            ROWS,
            HIDDEN,
        ).map_err(|e| e.to_string())?;
        commands.submit_and_wait().map_err(|e| e.to_string())?;
        head_value = ops.read_f32(out, 1).map_err(|e| e.to_string())?[0];
        let elapsed = t0.elapsed().as_secs_f64();
        if pass == 0 {
            first_s = elapsed;
        } else {
            steady_s += elapsed;
        }
    }
    let batched_s = steady_s / (passes - 1) as f64;
    eprintln!(
        "batched first call (JIT + {:.0} MiB upload): {first_s:.4} s",
        arena_bytes as f64 / 1048576.0
    );
    let outcome: Result<f32, crate::vulkan::VulkanError> = Ok(head_value);

    match outcome {
        Ok(value) => {
            eprintln!("batched: {batched_s:.4} s for {ROWS} rows (q[0] = {value:.4e})");
            let batched_per_row = batched_s / ROWS as f64;
            eprintln!(
                "  per row {batched_per_row:.6} s vs {per_row_us:.1} us -> {:.0}x cheaper",
                per_row / batched_per_row
            );
            assert!(
                batched_per_row < per_row,
                "batched dispatch ({batched_per_row:.9} s/row) did not beat row-at-a-time ({per_row:.9} s/row)"
            );
        }
        Err(error) => return Err(error.to_string()),
    }
    Ok(())
}
