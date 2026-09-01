#![cfg(feature = "vulkan")]
use rust_model_inference::ops::float::enable_gpu;
use rust_model_inference::ops::get_vulkan_context;

fn main() {
    enable_gpu();
    let ctx = match get_vulkan_context() {
        Some(ctx) => ctx,
        None => {
            eprintln!("no vulkan context");
            return;
        }
    };
    let cases: Vec<(usize, usize)> = vec![
        (1024, 1024), // qwen3 wq-like: the exact first model dispatch
    ];
    for (n_in, n_out) in cases {
        let blocks = n_in / 32;
        // deterministic pseudo-random q8 weight + input
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as i64
        };
        let mut weight = vec![0u8; n_out * blocks * 34];
        for row in 0..n_out {
            for b in 0..blocks {
                let off = (row * blocks + b) * 34;
                // valid f16: generate f32 in range, convert via half crate
                let scale_f32 = 0.0005 + (next().rem_euclid(2000)) as f32 / 1e6;
                let f16_bits = half::f16::from_f32(scale_f32).to_bits();
                weight[off..off + 2].copy_from_slice(&f16_bits.to_le_bytes());
                for i in 0..32 {
                    weight[off + 2 + i] = (next() as i8) as u8;
                }
            }
        }
        let input_q8: Vec<u8> = (0..n_in).map(|_| (next() as i8) as u8).collect();
        let input_scales: Vec<f32> = (0..blocks)
            .map(|_| 0.001 + (next() % 1000) as f32 / 1e6)
            .collect();

        // CPU reference: exact Q8_0 semantics
        let mut cpu = vec![0f32; n_out];
        for row in 0..n_out {
            let mut sum = 0f32;
            for b in 0..blocks {
                let off = (row * blocks + b) * 34;
                let scale = f16_bits_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
                let mut dot = 0i32;
                for i in 0..32 {
                    dot += (weight[off + 2 + i] as i8 as i32) * (input_q8[b * 32 + i] as i8 as i32);
                }
                sum += scale * input_scales[b] * dot as f32;
            }
            cpu[row] = sum;
        }

        let mut gpu = vec![0f32; n_out];
        unsafe {
            if let Err(e) =
                ctx.matmul_q8_0(&weight, &input_q8, &input_scales, &mut gpu, n_in, n_out)
            {
                println!("({n_in},{n_out}): GPU ERROR {e}");
                continue;
            }
        }
        let max_diff = gpu
            .iter()
            .zip(&cpu)
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        let first_bad = gpu
            .iter()
            .zip(&cpu)
            .position(|(g, c)| (g - c).abs() > 0.01 * c.abs().max(1.0));
        let rel = max_diff / cpu.iter().copied().fold(0.0f32, f32::max).max(1e-9);
        println!("({n_in:6},{n_out:6}): max_diff={max_diff:.6} rel={rel:.2e} first_bad_row={first_bad:?}  gpu[0]={:.4} cpu[0]={:.4}", gpu[0], cpu[0]);
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as f32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let man = (bits & 0x3FF) as f32;
    let v = if exp == 0 {
        man / 1024.0 * 2f32.powi(-14)
    } else {
        (1.0 + man / 1024.0) * 2f32.powi(exp - 15)
    };
    if sign > 0.5 {
        -v
    } else {
        v
    }
}
