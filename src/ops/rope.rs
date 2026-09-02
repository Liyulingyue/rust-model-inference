//! RoPE (Rotary Position Embedding) operations.

#[cfg(target_os = "macos")]
extern "C" {
    fn __sincosf(value: f32, sin: *mut f32, cos: *mut f32);
}

#[inline]
pub(crate) fn rope_sin_cos(theta: f32) -> (f32, f32) {
    #[cfg(target_os = "macos")]
    {
        let mut sin = 0.0f32;
        let mut cos = 0.0f32;
        unsafe { __sincosf(theta, &mut sin, &mut cos) };
        return (cos, sin);
    }
    #[cfg(not(target_os = "macos"))]
    (theta.cos(), theta.sin())
}

pub fn rope_neox(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half = head_dim / 2;
    let n_heads = x.len() / head_dim;
    let theta_scale = freq_base.powf(-2.0f32 / head_dim as f32);
    for h in 0..n_heads {
        let base = h * head_dim;
        let mut theta = pos as f32;
        for i in 0..half {
            let (cos_a, sin_a) = rope_sin_cos(theta);
            let x0 = x[base + i];
            let x1 = x[base + i + half];
            x[base + i] = x0.mul_add(cos_a, x1 * -sin_a);
            x[base + i + half] = x0.mul_add(sin_a, x1 * cos_a);
            theta *= theta_scale;
        }
    }
}

/// Partial RoPE: applies Neox RoPE to only the first `n_rot` dims of
/// each head of size `head_dim`. Used by architectures where `n_rot <
/// head_dim` (e.g. Spark 2.5's per-layer heterogeneous RoPE: full-attn
/// layers use `n_rot=64`, SWA layers use `n_rot=256`, both with
/// `head_dim=256`).
///
/// `x` must be laid out as `[n_heads * head_dim]`. Elements beyond
/// `n_rot` per head are left untouched, matching `ggml_rope_ext(ctx, x,
/// pos, nullptr, n_rot, ...)` in llama.cpp.
pub fn rope_neox_partial(x: &mut [f32], pos: usize, head_dim: usize, n_rot: usize, freq_base: f32) {
    assert!(n_rot <= head_dim, "n_rot ({n_rot}) must be <= head_dim ({head_dim})");
    assert!(n_rot % 2 == 0, "n_rot ({n_rot}) must be even");
    let half = n_rot / 2;
    let n_heads = x.len() / head_dim;
    let theta_scale = freq_base.powf(-2.0f32 / n_rot as f32);
    for h in 0..n_heads {
        let base = h * head_dim;
        let mut theta = pos as f32;
        for i in 0..half {
            let (cos_a, sin_a) = rope_sin_cos(theta);
            let x0 = x[base + i];
            let x1 = x[base + i + half];
            x[base + i] = x0.mul_add(cos_a, x1 * -sin_a);
            x[base + i + half] = x0.mul_add(sin_a, x1 * cos_a);
            theta *= theta_scale;
        }
    }
}

/// RoPE in GGML "normal" (interleaved-pair) style, as used by the classic
/// `llama` GGUF architecture: rotates adjacent `(x[2i], x[2i+1])` pairs.
/// HF `rotate_half`-style weights are permuted to this layout by the
/// llama.cpp llama-arch converter, so a `llama`-arch GGUF must use this
/// variant, not [`rope_neox`].
pub fn rope_norm(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half = head_dim / 2;
    let n_heads = x.len() / head_dim;
    let theta_scale = freq_base.powf(-2.0f32 / head_dim as f32);
    for h in 0..n_heads {
        let base = h * head_dim;
        let mut theta = pos as f32;
        for i in 0..half {
            let (cos_a, sin_a) = rope_sin_cos(theta);
            let x0 = x[base + 2 * i];
            let x1 = x[base + 2 * i + 1];
            x[base + 2 * i] = x0.mul_add(cos_a, x1 * -sin_a);
            x[base + 2 * i + 1] = x0.mul_add(sin_a, x1 * cos_a);
            theta *= theta_scale;
        }
    }
}

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

#[cfg(test)]
mod tests {
    #[test]
    fn vision_rope_rotates_both_halves_with_independent_axes() {
        let mut values = [0.0f32; 64];
        values[0] = 1.0;
        values[32] = 2.0;
        values[31] = 3.0;
        values[63] = 4.0;

        super::rope_vision(&mut values, [1, 2, 1, 2], [16, 16, 16, 16], 64, 1.0, 32);

        let (sin_h, cos_h) = 1.0f32.sin_cos();
        let (sin_w, cos_w) = 2.0f32.sin_cos();
        assert!((values[0] - (cos_h - 2.0 * sin_h)).abs() < 1e-6);
        assert!((values[32] - (sin_h + 2.0 * cos_h)).abs() < 1e-6);
        assert!((values[31] - (3.0 * cos_w - 4.0 * sin_w)).abs() < 1e-6);
        assert!((values[63] - (3.0 * sin_w + 4.0 * cos_w)).abs() < 1e-6);
    }
}
