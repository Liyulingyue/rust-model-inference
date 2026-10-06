//! Neox-style RoPE: rotate lo/hi halves independently.
//!
//! Public API (vanilla first, scaling last):
//! - [`rope_neox_inplace`] — vanilla RoPE-Neox. The default for
//!   every llama-family arch without linear position scaling
//!   (llama / mistral / qwen / qwen3 / phi-2 / phi-3 / phi-4 /
//!    gemma2 / gemma3 270M / gemma3 1B / bitnet b1.58 / etc.).
//!   Builds the sin/cos table once per token and dispatches to
//!   the AVX2 / NEON / scalar apply kernel. No `pos * factor`
//!   multiply — callers that don't need linear scaling should
//!   use this entry point instead of passing `factor = 1.0`
//!   to [`rope_neox_inplace_with_factor`].
//! - [`rope_neox_inplace_with_factor`] — same but with linear
//!   position scaling (`theta_i = pos * factor * freq_base^(-2i/d)`).
//!   Only gemma3 4B+ / 12B / 27B needs this — they declare
//!   `gemma3.rope.scaling.factor = 8.0` to extend context 32k
//!   → 256k. Every other arch passes `1.0` to this entry
//!   point, but those callers should migrate to
//!   [`rope_neox_inplace`] to avoid the wasted multiply.
//!
//! Shared private:
//! - [`build_cos_sin_table_and_apply`] — builds the cos/sin
//!   table starting from a caller-supplied `theta_0`, then
//!   dispatches to the AVX2 / NEON / scalar kernel. The two
//!   public entry points funnel through this helper so the
//!   per-arch kernels (`rope_neox_inplace_avx2` / `_neon` /
//!   `_scalar`) live in exactly one place.
//! - [`rope_neox_inplace_avx2`] / [`rope_neox_inplace_neon`] /
//!   [`rope_neox_inplace_scalar`] — the actual rotation loop.
//!   Naming follows the `[name]-[inplace]-[arch]` convention
//!   from `math/exp.rs`.

#[inline]
pub fn rope_sin_cos(theta: f32) -> (f32, f32) {
    let (sin, cos) = super::mrope::sin_cos(theta);
    (cos, sin)
}

/// Vanilla RoPE-Neox with no position scaling.
///
/// `theta_i = pos * freq_base^(-2i/d)`. Use this for plain
/// llama / mistral / qwen / qwen3 / phi-2 / phi-3 / phi-4 /
/// gemma2 / gemma3 270M / gemma3 1B / bitnet b1.58 — every arch
/// that doesn't declare `rope.scaling.factor > 1.0`.
///
/// Equivalent to [`rope_neox_inplace_with_factor`] with
/// `factor = 1.0` but skips the `pos * factor` multiply (saves
/// one fmul per call; the rotation loop is the same).
pub fn rope_neox_inplace(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    // `theta_0 = pos as f32` — no factor multiply.
    build_cos_sin_table_and_apply(x, pos as f32, head_dim, freq_base);
}

/// RoPE-Neox with linear position scaling.
///
/// `theta_i = pos * factor * freq_base^(-2i/d)`. Currently the
/// only arch that uses `factor > 1.0` is gemma3 4B+/12B/27B
/// (`gemma3.rope.scaling.factor = 8.0` to extend context 32k
/// → 256k); the gemma3 trunk reads `cfg.rope_factor` from
/// metadata and passes it here. Same form as GPT-NeoX /
/// PaLM-style RoPE extension. Callers that pass `factor = 1.0`
/// should migrate to [`rope_neox_inplace`] to avoid the
/// extra multiply.
pub fn rope_neox_inplace_with_factor(
    x: &mut [f32],
    pos: usize,
    head_dim: usize,
    freq_base: f32,
    factor: f32,
) {
    // `theta_0 = pos as f32 * factor` — this is the only place
    // the public entry points diverge.
    build_cos_sin_table_and_apply(x, pos as f32 * factor, head_dim, freq_base);
}

/// Build the cos/sin table once per token and dispatch to the
/// SIMD / scalar rotation kernel. The two public entry points
/// differ only in their `theta_0`; everything below is shared.
///
/// Table-build cost: `O(head_dim / 2)` `powf` + `sin_cos` calls
/// (identical across heads at the same `pos`, so we cache once
/// and broadcast). Apply cost: `O(x.len())` f32 muls + adds
/// under the per-arch kernel below.
#[inline]
fn build_cos_sin_table_and_apply(
    x: &mut [f32],
    theta_0: f32,
    head_dim: usize,
    freq_base: f32,
) {
    let half = head_dim / 2;
    let n_heads = x.len() / head_dim;
    if half == 0 || n_heads == 0 {
        return;
    }
    let mut cos_table = vec![0.0f32; half];
    let mut sin_table = vec![0.0f32; half];
    let theta_scale = freq_base.powf(-2.0 / head_dim as f32);
    let mut theta = theta_0;
    for i in 0..half {
        let (c, s) = rope_sin_cos(theta);
        cos_table[i] = c;
        sin_table[i] = s;
        theta *= theta_scale;
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
            let neg_hi_sin = vmulq_f32(vnegq_f32(x_hi), sin_v);
            let new_lo = vfmaq_f32(neg_hi_sin, x_lo, cos_v);
            let hi_cos = vmulq_f32(x_hi, cos_v);
            let new_hi = vfmaq_f32(hi_cos, x_lo, sin_v);
            vst1q_f32(lo_ptr.add(i), new_lo);
            vst1q_f32(hi_ptr.add(i), new_hi);
            i += 4;
        }
        while i < half {
            let x0 = *lo_ptr.add(i);
            let x1 = *hi_ptr.add(i);
            *lo_ptr.add(i) = x0.mul_add(cos[i], -(x1 * sin[i]));
            *hi_ptr.add(i) = x0.mul_add(sin[i], x1 * cos[i]);
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
/// Unlike [`rope_neox_inplace_with_factor`] this entry point does not compute the
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
pub fn rope_neox_inplace_with_table(x: &mut [f32], head_dim: usize, cos: &[f32], sin: &[f32]) {
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

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    #[test]
    fn rope_neox_matches_llama_arm_raw_bits() {
        let theta_scale = 10_000.0f32.powf(-2.0 / 256.0);
        let theta = (0..10).fold(1.0f32, |theta, _| theta * theta_scale);
        let (cos, sin) = rope_sin_cos(theta);
        assert_eq!(cos.to_bits(), 0x3f62_3dd5);
        assert_eq!(sin.to_bits(), 0x3eef_96e2);

        let mut values = vec![0.0f32; 256];
        values[2] = f32::from_bits(0x3ec4_d666);
        values[10] = f32::from_bits(0x3e82_5ca5);
        values[130] = f32::from_bits(0xbccb_b52e);
        values[138] = f32::from_bits(0xbd7e_afee);

        rope_neox_inplace(&mut values, 1, 256, 10_000.0 );

        assert_eq!(values[2].to_bits(), 0x3e89_3aee);
        assert_eq!(values[10].to_bits(), 0x3e82_1b0c);
        assert_eq!(values[130].to_bits(), 0x3e8d_afa8);
        assert_eq!(values[138].to_bits(), 0x3d83_783d);
    }
}
