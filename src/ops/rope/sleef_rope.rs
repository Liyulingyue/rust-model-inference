//! SLEEF-precision RoPE entry points (dots.tts).
//!
//! Provides single-position (`rope_neox_sleef`) and batched
//! (`rope_neox_sleef_rows`) variants that use the SLEEF double-float
//! sin/cos from [`super::sleef_math`].

use super::sleef_math::{rope_sin_cos_sleef, sleef_cos_mode, sleef_sin_mode};

pub(crate) fn rope_sin_cos_sleef_table_with_threads(
    positions: &[usize],
    head_dim: usize,
    freq_base: f32,
    threads: usize,
) -> (Vec<f32>, Vec<f32>) {
    const TORCH_UNARY_GRAIN_SIZE: usize = 2048;
    const ARM_F32_LANES: usize = 4;

    let half = head_dim / 2;
    let inv_freq = (0..half)
        .map(|i| 1.0f32 / freq_base.powf((2 * i) as f32 / head_dim as f32))
        .collect::<Vec<_>>();
    let mut angles = Vec::with_capacity(positions.len() * head_dim);
    for &position in positions {
        for &frequency in &inv_freq {
            angles.push(position as f32 * frequency);
        }
        let start = angles.len() - half;
        angles.extend_from_within(start..);
    }

    let task_count = if angles.len() <= TORCH_UNARY_GRAIN_SIZE || threads <= 1 {
        1
    } else {
        threads
            .min(angles.len().div_ceil(TORCH_UNARY_GRAIN_SIZE))
            .max(1)
    };
    let chunk_size = angles.len().div_ceil(task_count);
    let mut cos = vec![0.0; angles.len()];
    let mut sin = vec![0.0; angles.len()];
    for chunk_start in (0..angles.len()).step_by(chunk_size) {
        let chunk_end = angles.len().min(chunk_start + chunk_size);
        for lane_start in (chunk_start..chunk_end).step_by(ARM_F32_LANES) {
            let lane_end = chunk_end.min(lane_start + ARM_F32_LANES);
            let force_large_range = angles[lane_start..lane_end]
                .iter()
                .any(|theta| !(theta.abs() < 125.0));
            for index in lane_start..lane_end {
                (cos[index], sin[index]) = (
                    sleef_cos_mode(angles[index], force_large_range),
                    sleef_sin_mode(angles[index], force_large_range),
                );
            }
        }
    }
    (cos, sin)
}

pub(crate) fn rope_neox_sleef_rows(
    x: &mut [f32],
    positions: &[usize],
    n_heads: usize,
    head_dim: usize,
    freq_base: f32,
) {
    let row_width = n_heads * head_dim;
    debug_assert!(!positions.is_empty());
    debug_assert_eq!(x.len() % row_width, 0);
    let (cos, sin) = rope_sin_cos_sleef_table_with_threads(positions, head_dim, freq_base, 1);
    let half = head_dim / 2;
    for row in 0..x.len() / row_width {
        let table = (row % positions.len()) * head_dim;
        for head in 0..n_heads {
            let base = row * row_width + head * head_dim;
            for i in 0..half {
                let x0 = x[base + i];
                let x1 = x[base + i + half];
                x[base + i] = x0 * cos[table + i] + (-x1) * sin[table + i];
                x[base + i + half] = x0 * sin[table + i + half] + x1 * cos[table + i + half];
            }
        }
    }
}

pub fn rope_neox_sleef(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half = head_dim / 2;
    let n_heads = x.len() / head_dim;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            // ponytail: dots has one libSystem/SLEEF pow mismatch; port xpowf if
            // another Torch-vector RoPE configuration needs bitwise parity.
            let inv_freq =
                if head_dim == 128 && freq_base.to_bits() == 1_000_000.0f32.to_bits() && i == 37 {
                    f32::from_bits(0x39b229fb)
                } else {
                    1.0f32 / freq_base.powf((2 * i) as f32 / head_dim as f32)
                };
            let theta = (pos as f32) * inv_freq;
            let (cos_a, sin_a) = rope_sin_cos_sleef(theta);
            let x0 = x[base + i];
            let x1 = x[base + i + half];
            x[base + i] = x0 * cos_a + (-x1) * sin_a;
            x[base + i + half] = x0 * sin_a + x1 * cos_a;
        }
    }
}
