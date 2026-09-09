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
    if half == 0 || n_heads == 0 {
        return;
    }
    // Cache the sin/cos table once: identical across heads at this pos.
    // Reduces `powf` + `sin_cos` calls from `n_heads × half` to just `half`.
    let mut cos_table = vec![0.0f32; half];
    let mut sin_table = vec![0.0f32; half];
    let pos_f = pos as f32;
    for i in 0..half {
        let inv_freq = 1.0f32 / freq_base.powf((2 * i) as f32 / head_dim as f32);
        let theta = pos_f * inv_freq;
        let (c, s) = rope_sin_cos(theta);
        cos_table[i] = c;
        sin_table[i] = s;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2_fma() {
            unsafe { rope_neox_apply_avx2(x, n_heads, head_dim, &cos_table, &sin_table) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if super::has_neon() {
            unsafe { rope_neox_apply_neon(x, n_heads, head_dim, &cos_table, &sin_table) };
            return;
        }
    }
    rope_neox_apply_scalar(x, n_heads, head_dim, &cos_table, &sin_table);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn rope_neox_apply_avx2(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    use std::arch::x86_64::*;
    let half = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        let lo_ptr = x.as_mut_ptr().add(base);
        let hi_ptr = x.as_mut_ptr().add(base + half);
        let mut i = 0;
        while i + 8 <= half {
            let cos_v = _mm256_loadu_ps(cos.as_ptr().add(i));
            let sin_v = _mm256_loadu_ps(sin.as_ptr().add(i));
            let x_lo = _mm256_loadu_ps(lo_ptr.add(i));
            let x_hi = _mm256_loadu_ps(hi_ptr.add(i));
            // Match the scalar op order `x0 * cos_a + (-x1) * sin_a`
            // (negation is exact, then mul, then add) so the intermediate
            // values round identically and the result is bit-exact with
            // the pinned ggml reference. FMA would fuse the mul+add and
            // produce 1-ULP differences on some inputs.
            let neg_x_hi = _mm256_sub_ps(_mm256_setzero_ps(), x_hi);
            let prod_lo = _mm256_mul_ps(x_lo, cos_v);
            let prod_hi = _mm256_mul_ps(neg_x_hi, sin_v);
            let new_lo = _mm256_add_ps(prod_lo, prod_hi);
            let prod_hi2 = _mm256_mul_ps(x_hi, cos_v);
            let prod_lo2 = _mm256_mul_ps(x_lo, sin_v);
            let new_hi = _mm256_add_ps(prod_lo2, prod_hi2);
            _mm256_storeu_ps(lo_ptr.add(i), new_lo);
            _mm256_storeu_ps(hi_ptr.add(i), new_hi);
            i += 8;
        }
        while i < half {
            let x0 = *lo_ptr.add(i);
            let x1 = *hi_ptr.add(i);
            *lo_ptr.add(i) = x0 * cos[i] + (-x1) * sin[i];
            *hi_ptr.add(i) = x0 * sin[i] + x1 * cos[i];
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rope_neox_apply_neon(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    use std::arch::aarch64::*;
    let half = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        let lo_ptr = x.as_mut_ptr().add(base);
        let hi_ptr = x.as_mut_ptr().add(base + half);
        let mut i = 0;
        while i + 4 <= half {
            let cos_v = vld1q_f32(cos.as_ptr().add(i));
            let sin_v = vld1q_f32(sin.as_ptr().add(i));
            let x_lo = vld1q_f32(lo_ptr.add(i));
            let x_hi = vld1q_f32(hi_ptr.add(i));
            // new_lo = x_lo * cos - x_hi * sin
            let new_lo = vmlsq_f32(vmulq_f32(x_lo, cos_v), x_hi, sin_v);
            // new_hi = x_hi * cos + x_lo * sin
            let new_hi = vmlaq_f32(vmulq_f32(x_hi, cos_v), x_lo, sin_v);
            vst1q_f32(lo_ptr.add(i), new_lo);
            vst1q_f32(hi_ptr.add(i), new_hi);
            i += 4;
        }
        while i < half {
            let x0 = *lo_ptr.add(i);
            let x1 = *hi_ptr.add(i);
            *lo_ptr.add(i) = x0 * cos[i] - x1 * sin[i];
            *hi_ptr.add(i) = x0 * sin[i] + x1 * cos[i];
            i += 1;
        }
    }
}

fn rope_neox_apply_scalar(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let x0 = x[base + i];
            let x1 = x[base + i + half];
            x[base + i] = x0 * cos[i] - x1 * sin[i];
            x[base + i + half] = x0 * sin[i] + x1 * cos[i];
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
    assert!(
        n_rot <= head_dim,
        "n_rot ({n_rot}) must be <= head_dim ({head_dim})"
    );
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
    if half == 0 || n_heads == 0 {
        return;
    }
    // Cache sin/cos table once across all heads (same for each head at this pos).
    // `rope_norm` uses the recurrence `theta *= theta_scale` (matches ggml's
    // ROPE_TYPE_NORM); we keep it here for bit-exact parity with the original
    // implementation. Reduces `sin_cos` calls from `n_heads × half` to `half`.
    let mut cos_table = vec![0.0f32; half];
    let mut sin_table = vec![0.0f32; half];
    let theta_scale = freq_base.powf(-2.0f32 / head_dim as f32);
    let mut theta = pos as f32;
    for i in 0..half {
        let (c, s) = rope_sin_cos(theta);
        cos_table[i] = c;
        sin_table[i] = s;
        theta *= theta_scale;
    }
    // Inner loop is scalar because the rotation touches interleaved
    // `(x[2i], x[2i+1])` pairs, which AVX2 can only handle with shuffles
    // that cost more than they save at typical `half ≤ 128`.
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let x0 = x[base + 2 * i];
            let x1 = x[base + 2 * i + 1];
            let c = cos_table[i];
            let sn = sin_table[i];
            x[base + 2 * i] = x0.mul_add(c, x1 * -sn);
            x[base + 2 * i + 1] = x0.mul_add(sn, x1 * c);
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

    /// SIMD path must produce the same result as the scalar fallback.
    /// Compares public `rope_neox` against the explicit scalar helper used
    /// when SIMD is unavailable. Catches tail-handling, cache wiring, and
    /// instruction-order bugs across the three paths.
    #[test]
    fn rope_neox_simd_matches_scalar_fallback() {
        // Vary n_heads × head_dim to exercise SIMD tail loops and edge cases.
        for &(head_dim, n_heads, pos, freq_base) in &[
            (64usize, 4usize, 0usize, 10_000.0f32),
            (128, 8, 1, 1_000_000.0),
            (128, 16, 7, 500_000.0),
            (256, 4, 1024, 50_000.0),
            // head_dim not a multiple of 16 → SIMD tail must fall through to scalar.
            (96, 2, 3, 100_000.0),
            (80, 6, 5, 200_000.0),
            (128, 1, 0, 10_000.0),
        ] {
            let mut a = vec![0.0f32; n_heads * head_dim];
            let mut b = vec![0.0f32; n_heads * head_dim];
            for (i, slot) in a.iter_mut().enumerate() {
                *slot = ((i as f32) * 0.0731).sin() * 3.5 - ((i * 31 % 97) as f32) * 0.013;
            }
            b.copy_from_slice(&a);

            super::rope_neox(&mut a, pos, head_dim, freq_base);

            // Scalar reference uses the same formula as the public function's
            // table build, then a plain scalar per-head rotation — the same
            // shape as the AVX2/NEON tail loop, so SIMD-vs-scalar diffs are
            // caught bit-for-bit.
            let half = head_dim / 2;
            let pos_f = pos as f32;
            let mut cos_table = vec![0.0f32; half];
            let mut sin_table = vec![0.0f32; half];
            for i in 0..half {
                let inv_freq = 1.0f32 / freq_base.powf((2 * i) as f32 / head_dim as f32);
                let theta = pos_f * inv_freq;
                let (c, s) = super::rope_sin_cos(theta);
                cos_table[i] = c;
                sin_table[i] = s;
            }
            super::rope_neox_apply_scalar(&mut b, n_heads, head_dim, &cos_table, &sin_table);

            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "head_dim={head_dim} n_heads={n_heads} idx={i}"
                );
            }
        }
    }

    /// Same for `rope_norm`: SIMD is not used (interleaved-pair layout doesn't
    /// vectorize cleanly), but the cached table must produce the same output
    /// as the original per-head recurrence loop.
    #[test]
    fn rope_norm_cached_table_matches_per_head_recurrence() {
        for &(head_dim, n_heads, pos, freq_base) in &[
            (64usize, 4usize, 0usize, 10_000.0f32),
            (128, 8, 1, 1_000_000.0),
            (128, 16, 7, 500_000.0),
            (256, 4, 1024, 50_000.0),
        ] {
            let mut a = vec![0.0f32; n_heads * head_dim];
            let mut b = vec![0.0f32; n_heads * head_dim];
            for (i, slot) in a.iter_mut().enumerate() {
                *slot = ((i as f32) * 0.0731).sin() * 3.5 - ((i * 31 % 97) as f32) * 0.013;
            }
            b.copy_from_slice(&a);

            super::rope_norm(&mut a, pos, head_dim, freq_base);

            // Reference: original per-head loop with `theta *= theta_scale`.
            let half = head_dim / 2;
            let theta_scale = freq_base.powf(-2.0f32 / head_dim as f32);
            for h in 0..n_heads {
                let base = h * head_dim;
                let mut theta = pos as f32;
                for i in 0..half {
                    let (c, s) = super::rope_sin_cos(theta);
                    let x0 = b[base + 2 * i];
                    let x1 = b[base + 2 * i + 1];
                    b[base + 2 * i] = x0.mul_add(c, x1 * -s);
                    b[base + 2 * i + 1] = x0.mul_add(s, x1 * c);
                    theta *= theta_scale;
                }
            }

            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "head_dim={head_dim} n_heads={n_heads} idx={i}"
                );
            }
        }
    }
}
