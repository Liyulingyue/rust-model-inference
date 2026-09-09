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
