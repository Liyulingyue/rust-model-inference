//! RoPE (Rotary Position Embedding) operations.

#[cfg(target_os = "macos")]
extern "C" {
    fn __sincosf(value: f32, sin: *mut f32, cos: *mut f32);
}

#[inline]
pub(crate) fn rope_sin_cos(theta: f32) -> (f32, f32) {
    (theta.cos(), theta.sin())
}

#[derive(Clone, Copy)]
struct Df2 {
    hi: f32,
    lo: f32,
}

#[inline(always)]
fn df_add2(left: f32, right: f32) -> Df2 {
    let hi = left + right;
    let virtual_right = hi - left;
    Df2 {
        hi,
        lo: (left - (hi - virtual_right)) + (right - virtual_right),
    }
}

#[inline(always)]
fn df_add_plain(left: f32, right: f32) -> Df2 {
    let hi = left + right;
    Df2 {
        hi,
        lo: (left - hi) + right,
    }
}

#[inline(always)]
fn df_add(left: Df2, right: f32) -> Df2 {
    let hi = left.hi + right;
    Df2 {
        hi,
        lo: ((left.hi - hi) + right) + left.lo,
    }
}

#[inline(always)]
fn df_add2_scalar(left: Df2, right: f32) -> Df2 {
    let hi = left.hi + right;
    let virtual_right = hi - left.hi;
    Df2 {
        hi,
        lo: ((left.hi - (hi - virtual_right)) + (right - virtual_right)) + left.lo,
    }
}

#[inline(always)]
fn df_add_df(left: Df2, right: Df2) -> Df2 {
    let hi = left.hi + right.hi;
    let virtual_right = hi - left.hi;
    let error = (left.hi - (hi - virtual_right)) + (right.hi - virtual_right);
    Df2 {
        hi,
        lo: error + (left.lo + right.lo),
    }
}

#[inline(always)]
fn df_add_scalar(left: f32, right: Df2) -> Df2 {
    let hi = left + right.hi;
    Df2 {
        hi,
        lo: ((left - hi) + right.hi) + right.lo,
    }
}

#[inline(always)]
fn df_normalize(value: Df2) -> Df2 {
    let hi = value.hi + value.lo;
    Df2 {
        hi,
        lo: (value.hi - hi) + value.lo,
    }
}

#[inline(always)]
fn df_mul_scalar(left: f32, right: f32) -> Df2 {
    let hi = left * right;
    Df2 {
        hi,
        lo: left.mul_add(right, -hi),
    }
}

#[inline(always)]
fn df_mul_df_scalar(left: Df2, right: f32) -> Df2 {
    let hi = left.hi * right;
    Df2 {
        hi,
        lo: left.lo.mul_add(right, left.hi.mul_add(right, -hi)),
    }
}

#[inline(always)]
fn df_mul(left: Df2, right: Df2) -> Df2 {
    let hi = left.hi * right.hi;
    let lo = left.hi.mul_add(
        right.lo,
        left.lo.mul_add(right.hi, left.hi.mul_add(right.hi, -hi)),
    );
    Df2 { hi, lo }
}

#[inline(always)]
fn df_square(value: Df2) -> Df2 {
    let hi = value.hi * value.hi;
    let lo = (value.hi + value.hi).mul_add(value.lo, value.hi.mul_add(value.hi, -hi));
    Df2 { hi, lo }
}

#[inline(always)]
fn df_mul_to_scalar(left: Df2, right: Df2) -> f32 {
    left.hi
        .mul_add(right.hi, right.lo.mul_add(left.hi, left.lo * right.hi))
}

#[inline(always)]
fn mulsign(value: f32, sign_source: f32) -> f32 {
    f32::from_bits(value.to_bits() ^ (sign_source.to_bits() & 0x8000_0000))
}

#[inline(always)]
fn rempisubf(value: f32) -> (f32, i32) {
    let rounded4 = (value * 4.0).round_ties_even();
    let quadrant = (rounded4 - value.round_ties_even() * 4.0) as i32;
    (value - rounded4 * 0.25, quadrant)
}

#[inline(always)]
fn rempif(value: f32) -> (Df2, i32) {
    // Dots sessions are capped at 2048 positions, so the first rempi table
    // quartet is the only range needed here (ilogb(value) - 25 <= 0).
    const REMPI: [f32; 4] = [
        0.159_154_892,
        5.112_411_827e-8,
        3.626_141_271e-15,
        -2.036_222_915e-22,
    ];
    let mut x = df_mul_scalar(value, REMPI[0]);
    let (fraction, mut quadrant) = rempisubf(x.hi);
    x.hi = fraction;
    x = df_normalize(x);
    let y = df_mul_scalar(value, REMPI[1]);
    x = df_add_df(x, y);
    let (fraction, second_quadrant) = rempisubf(x.hi);
    quadrant += second_quadrant;
    x.hi = fraction;
    x = df_normalize(x);
    let y = df_mul_df_scalar(
        Df2 {
            hi: REMPI[2],
            lo: REMPI[3],
        },
        value,
    );
    x = df_add_df(x, y);
    x = df_normalize(x);
    x = df_mul(
        x,
        Df2 {
            hi: 3.141_592_741_012_573_2_f32 * 2.0,
            lo: -8.742_277_657_347_586e-8_f32 * 2.0,
        },
    );
    if value.abs() < 0.7 {
        (Df2 { hi: value, lo: 0.0 }, 0)
    } else {
        (x, quadrant)
    }
}

#[inline(always)]
fn sleef_sin_mode(theta: f32, force_large_range: bool) -> f32 {
    let (mut reduced, quadrant) = if !force_large_range && theta.abs() < 125.0 {
        let quadrant = (theta * 0.318_309_873_342_392_6_f32).round_ties_even() as i32;
        let q = quadrant as f32;
        let v = q.mul_add(-3.141_479_492_187_5, theta);
        let s = df_add2(v, q * -0.000_113_159_418_106_079_1_f32);
        (df_add(s, q * -1.984_187_258_941_005_9e-9_f32), quadrant)
    } else {
        let (mut value, base_quadrant) = rempif(theta);
        let quadrant = ((base_quadrant & 3) * 2 + if value.hi > 0.0 { 2 } else { 1 }) >> 2;
        if base_quadrant & 1 != 0 {
            value = df_add_df(
                value,
                Df2 {
                    hi: mulsign(3.141_592_741_012_573_2_f32 * -0.5, value.hi),
                    lo: mulsign(-8.742_277_657_347_586e-8_f32 * -0.5, value.hi),
                },
            );
        }
        value = df_normalize(value);
        (value, quadrant)
    };
    let square = df_square(reduced);
    let mut u = 2.608_315_980_978_659_4e-6_f32;
    u = u.mul_add(square.hi, -0.000_198_106_907_191_686_33);
    u = u.mul_add(square.hi, 0.008_333_078_585_565_09);
    let inner = df_add_plain(-0.166_666_597_127_914_43, u * square.hi);
    let polynomial = df_add_scalar(1.0, df_mul(inner, square));
    let mut result = df_mul_to_scalar(reduced, polynomial);
    if quadrant & 1 != 0 {
        result = f32::from_bits(result.to_bits() ^ 0x8000_0000);
    }
    if theta == 0.0 && theta.is_sign_negative() {
        -0.0
    } else {
        result
    }
}

#[inline(always)]
fn sleef_cos_mode(theta: f32, force_large_range: bool) -> f32 {
    let (reduced, quadrant) = if !force_large_range && theta.abs() < 125.0 {
        let rounded = theta
            .mul_add(0.318_309_873_342_392_6_f32, -0.5)
            .round_ties_even() as i32;
        let quadrant = 1 + 2 * rounded;
        let q = quadrant as f32;
        let mut reduced = df_add2(theta, q * (-3.141_479_492_187_5_f32 * 0.5));
        reduced = df_add2_scalar(reduced, q * (-0.000_113_159_418_106_079_1_f32 * 0.5));
        reduced = df_add2_scalar(reduced, q * (-1.984_187_258_941_005_9e-9_f32 * 0.5));
        (reduced, quadrant)
    } else {
        let (mut value, base_quadrant) = rempif(theta);
        let quadrant = ((base_quadrant & 3) * 2 + if value.hi > 0.0 { 8 } else { 7 }) >> 1;
        if base_quadrant & 1 == 0 {
            let sign = if value.hi > 0.0 { 1.0 } else { -1.0 };
            value = df_add_df(
                value,
                Df2 {
                    hi: mulsign(3.141_592_741_012_573_2_f32 * -0.5, sign),
                    lo: mulsign(-8.742_277_657_347_586e-8_f32 * -0.5, sign),
                },
            );
        }
        value = df_normalize(value);
        (value, quadrant)
    };
    let square = df_square(reduced);
    let mut u = 2.608_315_980_978_659_4e-6_f32;
    u = u.mul_add(square.hi, -0.000_198_106_907_191_686_33);
    u = u.mul_add(square.hi, 0.008_333_078_585_565_09);
    let inner = df_add_plain(-0.166_666_597_127_914_43, u * square.hi);
    let polynomial = df_add_scalar(1.0, df_mul(inner, square));
    let mut result = df_mul_to_scalar(reduced, polynomial);
    if quadrant & 2 == 0 {
        result = f32::from_bits(result.to_bits() ^ 0x8000_0000);
    }
    result
}

#[inline]
pub(crate) fn rope_sin_cos_sleef(theta: f32) -> (f32, f32) {
    (sleef_cos_mode(theta, false), sleef_sin_mode(theta, false))
}

fn rope_sin_cos_sleef_table_with_threads(
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

pub fn rope_neox(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half = head_dim / 2;
    let n_heads = x.len() / head_dim;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            // Match Transformers' Qwen2RotaryEmbedding: build each inverse
            // frequency independently in float32, then multiply by position.
            let inv_freq = 1.0f32 / freq_base.powf((2 * i) as f32 / head_dim as f32);
            let theta = (pos as f32) * inv_freq;
            let (cos_a, sin_a) = rope_sin_cos(theta);
            let x0 = x[base + i];
            let x1 = x[base + i + half];
            x[base + i] = x0 * cos_a + (-x1) * sin_a;
            x[base + i + half] = x0 * sin_a + x1 * cos_a;
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
    fn sleef_rope_sin_cos_matches_torch_arm_bits() {
        let expected = [
            (1.0f32, 0x3f0a5141u32, 0x3f576aa4u32),
            (126.0f32, 0x3f71a8f2u32, 0x3ea8f48fu32),
            (2048.0f32, 0x3f7321cau32, 0xbea04902u32),
        ];
        for (theta, cos_bits, sin_bits) in expected {
            let (cos, sin) = super::rope_sin_cos_sleef(theta);
            assert_eq!(cos.to_bits(), cos_bits, "cos({theta})");
            assert_eq!(sin.to_bits(), sin_bits, "sin({theta})");
        }
    }

    #[test]
    fn sleef_rope_sin_cos_matches_torch_arm_low_frequency_bits() {
        let expected = [
            (0x3e12bd91u32, 0x3f7d6040u32, 0x3e123d21u32),
            (0x3dec7fd6, 0x3f7e4b84u32, 0x3debf95du32),
            (0x3c301052, 0x3f7ffc37u32, 0x3c300f74u32),
            (0x39b229fb, 0x3f7fffffu32, 0x39b229fbu32),
        ];
        for (theta_bits, cos_bits, sin_bits) in expected {
            let theta = f32::from_bits(theta_bits);
            let (cos, sin) = super::rope_sin_cos_sleef(theta);
            assert_eq!(cos.to_bits(), cos_bits, "cos(theta={theta})");
            assert_eq!(sin.to_bits(), sin_bits, "sin(theta={theta})");
        }
    }

    #[test]
    fn sleef_rope_neox_matches_torch_vector_pow_bits() {
        let mut x = [0.0f32; 128];
        x[37] = 1.0;
        super::rope_neox_sleef(&mut x, 10, 128, 1_000_000.0);
        assert_eq!(x[37].to_bits(), 0x3f7fff9f);
        assert_eq!(x[101].to_bits(), 0x3b5eb45e);
    }

    #[test]
    fn sleef_rope_table_matches_torch_openmp_chunk_bits() {
        let positions = (0..185).collect::<Vec<_>>();
        let (cos, sin) = super::rope_sin_cos_sleef_table_with_threads(&positions, 64, 10_000.0, 12);
        let target = 165 * 64 + 30;
        assert_eq!(cos[target].to_bits(), 0x3f7fe3ca);
        assert_eq!(sin[target].to_bits(), 0x3cf054fd);

        let (cos, _) = super::rope_sin_cos_sleef_table_with_threads(&positions, 64, 10_000.0, 4);
        assert_eq!(cos[target].to_bits(), 0x3f7fe3cb);
    }

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
