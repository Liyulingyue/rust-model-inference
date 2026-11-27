//! Exact and approximate SiLU activations.

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

/// ggml's SiLU: `x / (1 + ggml_v_expf(-x))`.
///
/// ggml does **not** use libm's `expf` in its vectorized SiLU. It uses the
/// polynomial approximation in `ggml_v_expf`
/// (`ggml/src/ggml-cpu/vec.h`), which differs from libm by up to one ULP.
///
/// That one ULP is not noise here. In the Nemotron-3-Nano 30B SSM branch
/// the SiLU output feeds a Q4_K projection whose activations are
/// quantized to 8 bits; a one-ULP difference in the SiLU output is enough
/// to push an activation across a rounding boundary, which changes the
/// projected value by a whole quantization step -- a 4e-4 relative
/// difference instead of a 1e-7 one. Matching ggml's polynomial is
/// therefore what makes the branch bit-exact.
///
/// The polynomial below is transcribed from ggml's **ARM** implementation
/// (`ggml_v_expf` in `ggml/src/ggml-cpu/vec.h`). ggml's x86 build uses a
/// different polynomial for the same function, so this is the aarch64
/// behaviour; on x86 the repository keeps using libm's `expf`.
#[inline]
pub fn silu_ggml(x: f32) -> f32 {
    x / (1.0f32 + exp_ggml_scalar(-x))
}

/// ggml's `ggml_v_expf`, transcribed lane-wise from the ARM NEON
/// implementation.
///
/// The fast path (`k * j + k`) and the overflow path (`k * k + j`) are
/// *not* algebraically identical, and ggml picks between them per group
/// of four lanes: if any lane of the group overflows, the whole group
/// takes the overflow path. This scalar version therefore only matches
/// ggml for inputs that stay inside the fast path's range
/// (`|x| < 87.3`, i.e. `|n| <= 126`). The NEON version below reproduces
/// the group behaviour exactly and is what the batched entry point uses.
/// ggml's polynomial `exp` constants.
///
/// Rust has no hexadecimal float literals, so each constant is written as
/// its IEEE-754 bit pattern with the C literal ggml uses in a comment.
/// The bit patterns are the f32 rounding of those C literals: the C
/// compiler rounds `0x1.715476p+0f` to the nearest f32, which is *not*
/// the same as the literal's digits read as a bit pattern.
mod ggml_exp_consts {
    /// `0x1.8p23f` -- the magic number the exponent is built around.
    pub const R: f32 = f32::from_bits(0x4B40_0000);
    /// `0x1.715476p+0f` -- `log2(e)`, 1.4426950216293335 as an f32.
    pub const LOG2E: f32 = f32::from_bits(0x3FB8_AA3B);
    /// `0x1.62e4p-1f` -- 0.693145751953125 as an f32.
    pub const C1: f32 = f32::from_bits(0x3F31_7200);
    /// `0x1.7f7d1cp-20f` -- 1.428606765330187e-06 as an f32.
    pub const C2: f32 = f32::from_bits(0x35BF_BE8E);
    /// `0x1.ffffecp-1f` -- 0.9999994039535522 as an f32.
    pub const P0: f32 = f32::from_bits(0x3F7F_FFF6);
    /// `0x1.fffdb6p-2f` -- 0.49999126791954041 as an f32.
    pub const P1: f32 = f32::from_bits(0x3EFF_FEDB);
    /// `0x1.555e66p-3f` -- 0.16668395698070526 as an f32.
    pub const P2: f32 = f32::from_bits(0x3E2A_AF33);
    /// `0x1.573e2ep-5f` -- 0.04189976677298546 as an f32.
    pub const P3: f32 = f32::from_bits(0x3D2B_9F17);
    /// `0x1.0e4020p-7f` -- 0.008247390389442444 as an f32.
    pub const P4: f32 = f32::from_bits(0x3C07_2010);
}

#[inline]
fn exp_ggml_scalar(x: f32) -> f32 {
    use ggml_exp_consts::*;
    let z = R.mul_add(x, LOG2E);
    let n = z - R;
    let b = (-n).mul_add(C2, (-n).mul_add(C1, x));
    let e = z.to_bits() << 23;
    let k = f32::from_bits(e.wrapping_add(0x3f80_0000));
    let u = b * b;
    let inner1 = P1.mul_add(P2, b);
    let inner2 = P3.mul_add(b, P4);
    let d = inner1.mul_add(inner2, u);
    let j = (P0 * b).mul_add(d, u);
    if n.abs() > 126.0 {
        let mask = if n <= 0.0 { 0x8200_0000u32 } else { 0 };
        let s1 = f32::from_bits(mask.wrapping_add(0x7f00_0000));
        let s2 = f32::from_bits(e.wrapping_sub(mask));
        if n.abs() > 192.0 {
            s1 * s1
        } else {
            s2.mul_add(s2, j) * s1
        }
    } else {
        k.mul_add(j, k)
    }
}

/// ggml's SiLU applied in place, matching ggml's vectorized path.
#[inline(always)]
pub fn silu_ggml_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if crate::ops::has_neon() {
        unsafe { silu_ggml_inplace_neon(values) };
        return;
    }
    for value in values {
        *value = silu_ggml(*value);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn silu_ggml_inplace_neon(values: &mut [f32]) {
    use std::arch::aarch64::*;

    let n4 = values.len() / 4 * 4;
    let mut i = 0;
    while i < n4 {
        let x = vld1q_f32(values.as_ptr().add(i));
        let neg_x = vsubq_f32(vdupq_n_f32(0.0), x);
        let exp_neg_x = ggml_v_expf_neon(neg_x);
        let out = vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.0), exp_neg_x));
        vst1q_f32(values.as_mut_ptr().add(i), out);
        i += 4;
    }
    while i < values.len() {
        *values.get_unchecked_mut(i) = silu_ggml(*values.get_unchecked(i));
        i += 1;
    }
}

/// `a + b * c`, the ACLE / C convention.
///
/// A trap worth spelling out: the ACLE's `vfmaq_f32(a, b, c)` is
/// **a + b * c** -- the accumulator form of the fused multiply-add, like
/// the `FMLA` instruction -- and *not* the `a * b + c` of the `FMA`
/// instruction that the name suggests. Rust's intrinsic follows the
/// ACLE, so the raw intrinsic is already correct here; the wrapper only
/// exists to make the transcription read like the C source and to keep
/// the convention in one place.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fma(
    a: std::arch::aarch64::float32x4_t,
    b: std::arch::aarch64::float32x4_t,
    c: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    std::arch::aarch64::vfmaq_f32(a, b, c)
}

/// `a - b * c`, the ACLE / C convention.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fms(
    a: std::arch::aarch64::float32x4_t,
    b: std::arch::aarch64::float32x4_t,
    c: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    std::arch::aarch64::vfmsq_f32(a, b, c)
}

/// ggml's ARM `ggml_v_expf`: a polynomial `exp` approximation.
///
/// Transcribed from `ggml/src/ggml-cpu/vec.h` line by line. The overflow
/// handling is kept because ggml's fast path and its overflow path are
/// not algebraically identical for the same lane, and ggml chooses
/// between them per group of four lanes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ggml_v_expf_neon(
    x: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    use self::ggml_exp_consts::*;
    let r = vdupq_n_f32(R);
    let z = fma(r, x, vdupq_n_f32(LOG2E));
    let n = vsubq_f32(z, r);
    let b = fms(fms(x, n, vdupq_n_f32(C1)), n, vdupq_n_f32(C2));
    let e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    let k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1.0))));
    let c = vcagtq_f32(n, vdupq_n_f32(126.0));
    let u = vmulq_f32(b, b);
    let j = fma(
        vmulq_f32(vdupq_n_f32(P0), b),
        fma(
            fma(vdupq_n_f32(P1), vdupq_n_f32(P2), b),
            fma(vdupq_n_f32(P3), b, vdupq_n_f32(P4)),
            u,
        ),
        u,
    );
    if vaddvq_u32(c) == 0 {
        return fma(k, j, k);
    }
    let d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x8200_0000));
    // ggml writes `vdupq_n_u32(0x7f000000)` -- a *u32* literal, not a float
    // one. That matters: `0x7f000000` reinterpreted as a float is a NaN
    // pattern, so `vdupq_n_f32(0x7f000000)` would add `0x4EFE0000` (the
    // rounding of 2139095040.0f) instead and turn the overflow path's
    // `s1 * s1` into a huge finite number instead of a flush to zero.
    // With the correct u32 literal `s1` is `0x01000000` (a denormal), so
    // `exp(-inf)` comes out as exactly zero -- which is what the softmax
    // needs for its masked (`-inf`) columns.
    let s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f00_0000)));
    let s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    let big = vcagtq_f32(n, vdupq_n_f32(192.0));
    let inner = vbslq_f32(c, vmulq_f32(fma(s2, s2, j), s1), fma(k, k, j));
    vbslq_f32(big, vmulq_f32(s1, s1), inner)
}

#[inline(always)]
pub fn silu_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        unsafe { silu_inplace_avx2(values) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if crate::ops::has_neon() {
        unsafe { silu_inplace_neon(values) };
        return;
    }
    for value in values {
        *value = silu(*value);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn silu_inplace_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;

    let n8 = values.len() / 8 * 8;
    let mut i = 0;
    while i < n8 {
        let x = _mm256_loadu_ps(values.as_ptr().add(i));
        let zero = _mm256_setzero_ps();
        let neg_x = _mm256_sub_ps(zero, x);
        // Exact SiLU: per-lane libm exp (no AVX2 exp fast path; exact is
        // intentionally not bit-exact with approximate for the breeze
        // path that already uses crate::ops::silu(x)).
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), neg_x);
        for lane in &mut buf {
            *lane = (*lane).exp();
        }
        let exp_neg_x = _mm256_loadu_ps(buf.as_ptr());
        let one = _mm256_set1_ps(1.0);
        let one_plus_exp = _mm256_add_ps(one, exp_neg_x);
        let result = _mm256_div_ps(x, one_plus_exp);
        _mm256_storeu_ps(values.as_mut_ptr().add(i), result);
        i += 8;
    }
    while i < values.len() {
        values[i] = silu(values[i]);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn silu_inplace_neon(values: &mut [f32]) {
    use std::arch::aarch64::*;

    let mut i = 0;
    while i + 4 <= values.len() {
        let x = vld1q_f32(values.as_ptr().add(i));
        let neg_x = vsubq_f32(vdupq_n_f32(0.0), x);
        let mut buf = [0.0f32; 4];
        vst1q_f32(buf.as_mut_ptr(), neg_x);
        for lane in &mut buf {
            *lane = (*lane).exp();
        }
        let exp_neg_x = vld1q_f32(buf.as_ptr());
        let one_plus_exp_neg_x = vaddq_f32(vdupq_n_f32(1.0), exp_neg_x);
        vst1q_f32(values.as_mut_ptr().add(i), vdivq_f32(x, one_plus_exp_neg_x));
        i += 4;
    }
    while i < values.len() {
        values[i] = silu(values[i]);
        i += 1;
    }
}

#[inline(always)]
pub fn silu_approx_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        unsafe { silu_approx_inplace_avx2(values) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if crate::ops::has_neon() {
        unsafe { silu_approx_inplace_neon(values) };
        return;
    }
    silu_inplace(values);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn silu_approx_inplace_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;

    let n8 = values.len() / 8 * 8;
    let mut i = 0;
    while i < n8 {
        let x = _mm256_loadu_ps(values.as_ptr().add(i));
        let zero = _mm256_setzero_ps();
        let neg_x = _mm256_sub_ps(zero, x);
        let exp_neg_x = super::super::math::exp::exp_approx_avx2(neg_x);
        let one = _mm256_set1_ps(1.0);
        let one_plus_exp = _mm256_add_ps(one, exp_neg_x);
        let result = _mm256_div_ps(x, one_plus_exp);
        _mm256_storeu_ps(values.as_mut_ptr().add(i), result);
        i += 8;
    }
    while i < values.len() {
        values[i] = silu(values[i]);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn silu_approx_inplace_neon(values: &mut [f32]) {
    use std::arch::aarch64::*;

    let mut i = 0;
    while i + 4 <= values.len() {
        let x = vld1q_f32(values.as_ptr().add(i));
        let neg_x = vsubq_f32(vdupq_n_f32(0.0), x);
        let exp_neg_x = super::super::math::exp::exp_approx_neon(neg_x);
        let one_plus_exp_neg_x = vaddq_f32(vdupq_n_f32(1.0), exp_neg_x);
        vst1q_f32(values.as_mut_ptr().add(i), vdivq_f32(x, one_plus_exp_neg_x));
        i += 4;
    }
    while i < values.len() {
        values[i] = silu(values[i]);
        i += 1;
    }
}

#[inline(always)]
pub fn silu_mul_inplace(gate: &[f32], up: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    for i in 0..gate.len() {
        up[i] *= silu(gate[i]);
    }
}

#[inline(always)]
pub fn silu_mul_approx_inplace(gate: &[f32], up: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        unsafe { silu_mul_approx_inplace_avx2(gate, up) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if crate::ops::has_neon() {
        unsafe { silu_mul_approx_inplace_neon(gate, up) };
        return;
    }
    silu_mul_inplace(gate, up);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn silu_mul_approx_inplace_avx2(gate: &[f32], up: &mut [f32]) {
    use std::arch::x86_64::*;

    let n8 = gate.len() / 8 * 8;
    let mut i = 0;
    while i < n8 {
        let x = _mm256_loadu_ps(gate.as_ptr().add(i));
        let multiplier = _mm256_loadu_ps(up.as_ptr().add(i));
        let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        let exp_neg_x = super::super::math::exp::exp_approx_avx2(neg_x);
        let one = _mm256_set1_ps(1.0);
        let silu = _mm256_div_ps(x, _mm256_add_ps(one, exp_neg_x));
        _mm256_storeu_ps(up.as_mut_ptr().add(i), _mm256_mul_ps(silu, multiplier));
        i += 8;
    }
    while i < gate.len() {
        up[i] *= silu(gate[i]);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn silu_mul_approx_inplace_neon(gate: &[f32], up: &mut [f32]) {
    use std::arch::aarch64::*;

    let mut i = 0;
    while i + 4 <= gate.len() {
        let x = vld1q_f32(gate.as_ptr().add(i));
        let multiplier = vld1q_f32(up.as_ptr().add(i));
        let neg_x = vsubq_f32(vdupq_n_f32(0.0), x);
        let exp_neg_x = super::super::math::exp::exp_approx_neon(neg_x);
        let silu = vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.0), exp_neg_x));
        vst1q_f32(up.as_mut_ptr().add(i), vmulq_f32(silu, multiplier));
        i += 4;
    }
    while i < gate.len() {
        up[i] *= silu(gate[i]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::silu;
    use super::silu_inplace;

    fn assert_inplace_matches_scalar(values: &[f32]) {
        let mut simd = values.to_vec();
        let mut scalar = values.to_vec();
        silu_inplace(&mut simd);
        for (i, v) in scalar.iter_mut().enumerate() {
            *v = silu(*v);
        }
        for (i, (a, b)) in simd.iter().zip(scalar.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "lane {i}: simd={a} scalar={b}");
        }
    }

    #[test]
    fn silu_inplace_matches_scalar_small() {
        assert_inplace_matches_scalar(&[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn silu_inplace_matches_scalar_large() {
        let values: Vec<f32> = (0..512)
            .map(|i| ((i * 13 + 7) % 47) as f32 * 0.13 - 6.0)
            .collect();
        assert_inplace_matches_scalar(&values);
    }

    #[test]
    fn silu_inplace_matches_scalar_with_tail() {
        let values: Vec<f32> = (0..37).map(|i| (i as f32 * 0.21).sin() * 4.0).collect();
        assert_inplace_matches_scalar(&values);
    }

    /// Regression: a gate of 131.219 made `exp(-131.219)` wrap to `-inf`, so
    /// `silu` returned `-0` and the whole FFN channel was silently dropped.
    /// This is the exact pair that broke PhysBrain1.5-2B generation.
    #[test]
    fn silu_mul_approx_keeps_large_positive_gates() {
        let gates = vec![131.219f32; 16];
        let mut ups = vec![-151.609f32; 16];
        super::silu_mul_approx_inplace(&gates, &mut ups);
        for (i, got) in ups.iter().enumerate() {
            assert!(
                got.is_finite() && *got < -1e4,
                "lane {i}: activation collapsed to {got}"
            );
        }
    }
}
