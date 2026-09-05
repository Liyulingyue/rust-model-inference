#![cfg(feature = "vulkan")]
use rust_model_inference::ops::float::enable_gpu;
use rust_model_inference::ops::get_vulkan_context;
use rust_model_inference::vulkan::{VulkanContext, VulkanError};
use std::process::ExitCode;

const CASES: &[(usize, usize)] = &[
    (1024, 1024),
    (1024, 3072),
    (3072, 1024),
    (1024, 151_936),
    (16_384, 32),
];

fn main() -> ExitCode {
    enable_gpu();
    let ctx = match get_vulkan_context() {
        Some(ctx) => ctx,
        None => {
            eprintln!("no vulkan context");
            return ExitCode::FAILURE;
        }
    };

    let mut failed = false;
    for &(n_in, n_out) in CASES {
        let (max_abs, max_rel, first_bad_row) = match run_case(ctx, n_in, n_out) {
            Ok(result) => result,
            Err(error) => {
                eprintln!(
                    "device={} shape=({n_in},{n_out}) error={error}",
                    ctx.device_name()
                );
                return ExitCode::FAILURE;
            }
        };
        println!(
            "device={} shape=({n_in},{n_out}) max_abs={max_abs:.6} max_rel={max_rel:.2e} first_bad_row={first_bad_row:?}",
            ctx.device_name()
        );
        failed |= first_bad_row.is_some();
    }

    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn run_case(
    ctx: &VulkanContext,
    n_in: usize,
    n_out: usize,
) -> Result<(f32, f32, Option<usize>), VulkanError> {
    let blocks = n_in / 32;
    let mut seed = 12345u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as i64
    };
    let mut weight = vec![0u8; n_out * blocks * 34];
    for row in 0..n_out {
        for block in 0..blocks {
            let offset = (row * blocks + block) * 34;
            let scale = 0.0005 + next().rem_euclid(2000) as f32 / 1e6;
            weight[offset..offset + 2]
                .copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            for lane in 0..32 {
                weight[offset + 2 + lane] = next() as i8 as u8;
            }
        }
    }
    let input_q8: Vec<u8> = (0..n_in).map(|_| next() as i8 as u8).collect();
    let input_scales: Vec<f32> = (0..blocks)
        .map(|_| 0.001 + next().rem_euclid(1000) as f32 / 1e6)
        .collect();

    let mut cpu = vec![0.0f32; n_out];
    for (row, output) in cpu.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for block in 0..blocks {
            let offset = (row * blocks + block) * 34;
            let scale =
                half::f16::from_bits(u16::from_le_bytes([weight[offset], weight[offset + 1]]))
                    .to_f32();
            let mut dot = 0i32;
            for lane in 0..32 {
                dot += (weight[offset + 2 + lane] as i8 as i32)
                    * (input_q8[block * 32 + lane] as i8 as i32);
            }
            sum += scale * input_scales[block] * dot as f32;
        }
        *output = sum;
    }

    let mut gpu = vec![0.0f32; n_out];
    unsafe {
        ctx.matmul_q8_0(&weight, &input_q8, &input_scales, &mut gpu, n_in, n_out)?;
    }

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut first_bad_row = None;
    for (row, (&gpu_value, &cpu_value)) in gpu.iter().zip(&cpu).enumerate() {
        let absolute = if gpu_value.is_finite() {
            (gpu_value - cpu_value).abs()
        } else {
            f32::INFINITY
        };
        let relative = absolute / cpu_value.abs().max(1e-9);
        max_abs = max_abs.max(absolute);
        max_rel = max_rel.max(relative);
        if first_bad_row.is_none() && !within_tolerance(gpu_value, cpu_value) {
            first_bad_row = Some(row);
        }
    }
    Ok((max_abs, max_rel, first_bad_row))
}

fn within_tolerance(gpu: f32, cpu: f32) -> bool {
    gpu.is_finite() && (gpu - cpu).abs() <= 1e-4 + 1e-4 * cpu.abs()
}

#[cfg(test)]
mod tests {
    use super::within_tolerance;

    #[test]
    fn combined_absolute_and_relative_tolerance_is_accepted() {
        assert!(within_tolerance(10.001, 10.0));
    }

    #[test]
    fn non_finite_gpu_output_is_rejected() {
        assert!(!within_tolerance(f32::NAN, 1.0));
        assert!(!within_tolerance(f32::INFINITY, 1.0));
    }

    #[test]
    fn output_beyond_tolerance_is_rejected() {
        assert!(!within_tolerance(10.01, 10.0));
    }
}
