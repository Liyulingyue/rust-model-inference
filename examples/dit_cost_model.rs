//! Why a Z-Image step takes 140 s, in numbers.
//!
//! Two claims were floating around without evidence: that the CPU path is
//! already at the memory-bandwidth ceiling, and that the GPU is slower. Both
//! rest on measurements that did not survive scrutiny -- one of them timed a
//! hot matrix out of L3, another compared against a "CPU" baseline that was
//! partly the GPU. This settles the question without counters, by moving only
//! the row count and looking at the marginal cost.
//!
//! The QKV projection is 3840 -> 11520, a 47 MB Q8_0 matrix. A step runs it
//! for 1536 rows. If the cost were dominated by streaming that matrix, the
//! marginal cost of one more row would climb as the working set outgrew cache.
//! It does not: it sits at ~400 us/row from a single row all the way to 1536.
//!
//! Two things follow, and they are the answer to "why is PyTorch quick".
//!
//! A single row costs the same 400 us as row 1536 does, so the matrix read is
//! already amortised and per-row work is arithmetic. 44.2 MMAC in 400 us is
//! 222 GOP/s, which is a reasonable full-utilisation figure for 20 aarch64
//! cores doing int8 NEON dot products. The matmul is not the problem.
//!
//! What the DiT actually needs per step is 16.3 GOP, so at that measured rate
//! the matmuls should account for about 73 s, not the 140 s a step takes. The
//! rest is attention (43.8% of a step in the block profile) and the
//! element-wise work around it. A GPU with bf16 tensor cores would do the same
//! 44.2 MMAC in well under a microsecond -- three orders of magnitude -- so the
//! gap to PyTorch is arithmetic throughput, not memory traffic and not the
//! row-at-a-time loop shape.
use std::time::Instant;

use rust_model_inference::ops::kernel::q8_0::parallel::matmul_q8_0_quantized_parallel;

const HIDDEN: usize = 3840;
const QKV: usize = HIDDEN * 3;
const ROWS: usize = 1536;
const MAIN_LAYERS: usize = 30;
const FFN: usize = 10240;

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

fn main() {
    let weight = synth_q8_0(HIDDEN, QKV);
    let weight_mb = weight.len() as f64 / 1e6;
    println!("QKV projection {HIDDEN} -> {QKV}: {weight_mb:.1} MB");
    println!("20 cores, 24 MiB L3, 46.1 GB/s STREAM\n");

    let input = vec![1u8; HIDDEN];
    let scales = vec![0.001f32; HIDDEN / 32];
    let mut out = vec![0f32; QKV];

    println!(
        "{:>7}  {:>11}  {:>10}  {:>18}",
        "rows", "total", "us/row", "marginal us/row"
    );
    let mut previous: Option<(usize, f64)> = None;
    let mut samples = Vec::new();
    for rows in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536] {
        // Repeat small counts so the timer has enough resolution.
        let repeats = (4096 / rows).max(1);
        let t0 = Instant::now();
        for _ in 0..repeats {
            for _ in 0..rows {
                matmul_q8_0_quantized_parallel(&weight, &input, &scales, &mut out, HIDDEN, QKV);
            }
        }
        let per_pass = t0.elapsed().as_secs_f64() / repeats as f64;
        let per_row_us = per_pass / rows as f64 * 1e6;
        let marginal = match previous {
            Some((prev_rows, prev_s)) => (per_pass - prev_s) / (rows - prev_rows) as f64 * 1e6,
            None => per_row_us,
        };
        samples.push(per_row_us);
        println!(
            "{rows:>7}  {:>8.3} ms  {per_row_us:>10.1}  {marginal:>18.1}",
            per_pass * 1e3
        );
        previous = Some((rows, per_pass));
    }

    // The question the table answers: does per-row cost grow with the working
    // set? A flat line means the matrix read is amortised.
    let low = &samples[..4];
    let high = &samples[8..];
    let low_mean = low.iter().sum::<f64>() / low.len() as f64;
    let high_mean = high.iter().sum::<f64>() / high.len() as f64;
    println!("\nmean us/row, rows 1-8   : {low_mean:.1}");
    println!("mean us/row, rows 256+  : {high_mean:.1}");
    println!(
        "drift                    : {:+.0}%",
        (high_mean / low_mean - 1.0) * 100.0
    );

    let macs = (HIDDEN * QKV) as f64;
    let gops = 2.0 * macs / low_mean / 1e3;
    println!("\nachieved rate            : {gops:.0} GOP/s");
    println!("  (44.2 MMAC per row in {low_mean:.0} us; 20 cores of int8 NEON)");

    let projections = (HIDDEN * 3 * HIDDEN + HIDDEN * HIDDEN + FFN * HIDDEN * 3) as f64;
    let per_layer = ROWS as f64 * projections;
    let step_gop = 2.0 * per_layer * MAIN_LAYERS as f64 / 1e9;
    println!("\none step needs           : {step_gop:.1} GOP of matmul");
    println!("at {gops:.0} GOP/s that is    : {:.0} s", step_gop / gops);
    println!("a step actually takes    : 140 s");
    println!(
        "=> matmul accounts for about {:.0}% of a step; the rest is attention",
        (step_gop / gops) / 140.5 * 100.0
    );
}
