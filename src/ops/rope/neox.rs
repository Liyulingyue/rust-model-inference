//! Neox-style RoPE: rotate lo/hi halves independently.
//!
//! Public API:
//! - [`rope_neox_inplace`] — high-level entry point that caches the sin/cos table once
//!   per token and dispatches to the AVX2 / NEON / scalar apply kernel.
//!
//! Private kernels:
//! - [`rope_neox_inplace_avx2`] / [`rope_neox_inplace_neon`] /
//!   [`rope_neox_inplace_scalar`] — the actual rotation loop, taking a
//!   precomputed `(cos, sin)` table. Naming follows the
//!   `[name]-[inplace]-[arch]` convention from `math/exp.rs`.

#[inline]
pub fn rope_sin_cos(theta: f32) -> (f32, f32) {
    (theta.cos(), theta.sin())
}

pub fn rope_neox_inplace(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
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
        if super::super::has_avx2_fma() {
            unsafe { rope_neox_inplace_avx2(x, n_heads, head_dim, &cos_table, &sin_table) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if super::super::has_neon() {
            unsafe { rope_neox_inplace_neon(x, n_heads, head_dim, &cos_table, &sin_table) };
            return;
        }
    }
    rope_neox_inplace_scalar(x, n_heads, head_dim, &cos_table, &sin_table);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn rope_neox_inplace_avx2(
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
unsafe fn rope_neox_inplace_neon(
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

pub(crate) fn rope_neox_inplace_scalar(
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

/// Rope with a caller-supplied cos/sin table and bf16-quantised
/// intermediates.
///
/// Unlike [`rope_neox_inplace`] this entry point does not compute the
/// sin/cos table internally — the caller is expected to pre-quantise
/// cos/sin to BF16 (matching the upstream `bf(angle.cos())` /
/// `bf(angle.sin())` round-trips) and to handle any rope variant
/// (`linear_factor`, llama3 wavelength smoothing, etc.).
///
/// The mul/add output is round-tripped through BF16 to mirror the
/// upstream `bf(bf(a*c) + bf(-b*s))` / `bf(bf(b*c) + bf(a*s))`
/// rotation contract.  This is what the Breeze transformer needs to
/// keep its per-element bf-round trip semantics bit-exact while
/// running the per-head rotation through SIMD.
///
/// TODO-007: this is the rope wrapper that the Breeze dispatcher
/// gates behind `cfg(any())`.  When this function lands, Breeze
/// replaces its hand-written scalar rope with a single
/// `rope_neox_inplace_with_table(...)` call.
#[allow(dead_code)]
pub fn rope_neox_inplace_with_table(
    x: &mut [f32],
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    debug_assert_eq!(cos.len(), sin.len());
    debug_assert!(head_dim % 2 == 0);
    let half = head_dim / 2;
    let n_heads = x.len() / head_dim;
    if half == 0 || n_heads == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if super::super::has_avx2_fma() {
            unsafe { rope_neox_inplace_with_table_avx2(x, n_heads, head_dim, cos, sin) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if super::super::has_neon() {
            // TODO-005: aarch64 NEON variant; falls back to scalar for now.
            rope_neox_inplace_with_table_scalar(x, n_heads, head_dim, cos, sin);
            return;
        }
    }
    rope_neox_inplace_with_table_scalar(x, n_heads, head_dim, cos, sin);
}

fn rope_neox_inplace_with_table_scalar(
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
            x[base + i] = bf16_round(bf16_round(x0 * cos[i]) + bf16_round(-x1 * sin[i]));
            x[base + i + half] = bf16_round(bf16_round(x1 * cos[i]) + bf16_round(x0 * sin[i]));
        }
    }
}

#[inline(always)]
fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    let rounding = 0x7fff_u32 + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(rounding) >> 16;
    f32::from_bits(rounded << 16)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn rope_neox_inplace_with_table_avx2(
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
            // Mirror the scalar op order exactly so the bf-round
            // contract lines up: bf_round(a * c), bf_round(-b * s),
            // bf_round(sum).  The intermediate bf-round steps match
            // the upstream BF16-quantised rope simulation; skipping
            // them collapses three rounding decisions into one and
            // causes 1-bf16-ULP drift.
            let neg_x_hi = _mm256_sub_ps(_mm256_setzero_ps(), x_hi);
            let ac = bf16_round_ps(_mm256_mul_ps(x_lo, cos_v));
            let neg_bs = bf16_round_ps(_mm256_mul_ps(neg_x_hi, sin_v));
            let new_lo = bf16_round_ps(_mm256_add_ps(ac, neg_bs));
            let bc = bf16_round_ps(_mm256_mul_ps(x_hi, cos_v));
            let as_ = bf16_round_ps(_mm256_mul_ps(x_lo, sin_v));
            let new_hi = bf16_round_ps(_mm256_add_ps(bc, as_));
            _mm256_storeu_ps(lo_ptr.add(i), new_lo);
            _mm256_storeu_ps(hi_ptr.add(i), new_hi);
            i += 8;
        }
        while i < half {
            let x0 = *lo_ptr.add(i);
            let x1 = *hi_ptr.add(i);
            *lo_ptr.add(i) = bf16_round(bf16_round(x0 * cos[i]) + bf16_round(-x1 * sin[i]));
            *hi_ptr.add(i) = bf16_round(bf16_round(x1 * cos[i]) + bf16_round(x0 * sin[i]));
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn bf16_round_ps(a: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), a);
    for lane in &mut buf {
        let bits = lane.to_bits();
        let rounding = 0x7fff_u32 + ((bits >> 16) & 1);
        let rounded = bits.wrapping_add(rounding) >> 16;
        *lane = f32::from_bits(rounded << 16);
    }
    _mm256_loadu_ps(buf.as_ptr())
}
