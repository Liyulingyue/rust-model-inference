//! Multimodal RoPE (Qwen3-VL and friends).
//!
//! Three entry points share the same `positions: [usize; 4]` +
//! `sections: [i32; 4]` shape (T/H/W/E axes, configured by `mrope_section`),
//! differing in how the rotation pair is laid out inside `head_dim`:
//!
//! - [`rope_mrope`] — halves layout: rotate `(x[i], x[i + half])` pairs
//!   at axis-boundary `i % total_sections`.
//! - [`rope_vision`] — halves layout, but `n_rope_dims` may be less than
//!   `head_dim`; rotates only the first `n_rope_dims / 2` pairs and
//!   leaves the rest untouched (matches qwen3-vl vision).
//! - [`rope_mrope_interleaved`] — interleaved-pair layout: rotate
//!   `(x[2i], x[2i + 1])` at `pair % section_pairs`.
//!
//! `rope_mrope` internally delegates to [`super::rope_neox`] when
//! `sections` are all-zero (pure text mode).

use super::neox::rope_neox;

pub fn rope_mrope(
    x: &mut [f32],
    positions: [usize; 4],
    sections: [i32; 4],
    head_dim: usize,
    freq_base: f32,
) {
    let n_heads = x.len() / head_dim;
    let half = head_dim / 2;
    let total_sections: i32 = sections.iter().sum();
    if total_sections == 0 {
        rope_neox(x, positions[0], head_dim, freq_base);
        return;
    }
    let total_sections = total_sections as usize;
    let theta_scale = freq_base.powf(-2.0f32 / head_dim as f32);
    let section_h = sections[0] as usize;
    let section_w = section_h + sections[1] as usize;
    let section_e = section_w + sections[2] as usize;
    for h in 0..n_heads {
        let base = h * head_dim;
        let mut theta = positions.map(|position| position as f32);
        for i in 0..half {
            let sector = i % total_sections;
            let axis = if sector < section_h {
                0
            } else if sector < section_w {
                1
            } else if sector < section_e {
                2
            } else {
                3
            };
            let cos_a = theta[axis].cos();
            let sin_a = theta[axis].sin();
            let idx0 = base + i;
            let idx1 = idx0 + half;
            let x0 = x[idx0];
            let x1 = x[idx1];
            x[idx0] = x0.mul_add(cos_a, -(x1 * sin_a));
            x[idx1] = x0.mul_add(sin_a, x1 * cos_a);
            for value in &mut theta {
                *value *= theta_scale;
            }
        }
    }
}

pub fn rope_vision(
    x: &mut [f32],
    positions: [usize; 4],
    sections: [i32; 4],
    head_dim: usize,
    freq_base: f32,
    n_rope_dims: usize,
) {
    assert_eq!(n_rope_dims * 2, head_dim);
    let section_pairs: usize = sections.iter().map(|&value| value as usize).sum();
    assert!(section_pairs >= head_dim / 2);
    let boundaries = [
        sections[0] as usize,
        (sections[0] + sections[1]) as usize,
        (sections[0] + sections[1] + sections[2]) as usize,
    ];
    let theta_scale = freq_base.powf(-2.0 / n_rope_dims as f32);
    for head in x.chunks_exact_mut(head_dim) {
        let mut theta = positions.map(|value| value as f32);
        for pair in 0..head_dim / 2 {
            let sector = pair % section_pairs;
            let axis = if sector < boundaries[0] {
                0
            } else if sector < boundaries[1] {
                1
            } else if sector < boundaries[2] {
                2
            } else {
                3
            };
            if sector == 0
                || sector == boundaries[0]
                || sector == boundaries[1]
                || sector == boundaries[2]
            {
                theta[axis] = positions[axis] as f32;
            }
            let (sin, cos) = theta[axis].sin_cos();
            let x0 = head[pair];
            let x1 = head[pair + head_dim / 2];
            head[pair] = x0 * cos - x1 * sin;
            head[pair + head_dim / 2] = x0 * sin + x1 * cos;
            for value in &mut theta {
                *value *= theta_scale;
            }
        }
    }
}

pub fn rope_mrope_interleaved(
    x: &mut [f32],
    positions: [usize; 4],
    sections: [i32; 4],
    head_dim: usize,
    freq_base: f32,
    n_rope_dims: usize,
) {
    assert!(n_rope_dims <= head_dim && n_rope_dims % 2 == 0);
    let pair_count = n_rope_dims / 2;
    let section_pairs: usize = sections.iter().map(|&value| value as usize).sum();
    let theta_scale = freq_base.powf(-2.0 / n_rope_dims as f32);
    for head in x.chunks_exact_mut(head_dim) {
        let mut theta = positions.map(|value| value as f32);
        for pair in 0..pair_count {
            let sector = pair % section_pairs;
            let axis = if sector % 3 == 1 && sector < 3 * sections[1] as usize {
                1
            } else if sector % 3 == 2 && sector < 3 * sections[2] as usize {
                2
            } else if sector % 3 == 0 && sector < 3 * sections[0] as usize {
                0
            } else {
                3
            };
            let (sin, cos) = theta[axis].sin_cos();
            let x0 = head[pair];
            let x1 = head[pair + pair_count];
            head[pair] = x0.mul_add(cos, -(x1 * sin));
            head[pair + pair_count] = x0.mul_add(sin, x1 * cos);
            for value in &mut theta {
                *value *= theta_scale;
            }
        }
    }
}
