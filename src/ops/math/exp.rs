//! Exact and approximate exponential helpers.

#[inline(always)]
pub fn exp_inplace(values: &mut [f32]) {
    for value in values {
        *value = value.exp();
    }
}

#[inline(always)]
pub fn exp_approx_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        unsafe { exp_approx_inplace_avx2(values) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if crate::ops::has_neon() {
        unsafe { exp_approx_inplace_neon(values) };
        return;
    }
    exp_inplace(values);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn exp_approx_inplace_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;

    let mut i = 0;
    while i + 8 <= values.len() {
        let x = _mm256_loadu_ps(values.as_ptr().add(i));
        _mm256_storeu_ps(values.as_mut_ptr().add(i), exp_approx_avx2(x));
        i += 8;
    }
    while i < values.len() {
        values[i] = values[i].exp();
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn exp_approx_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    // The exponent below is built by shifting float bits, so it can only
    // represent 2**n while n stays within the f32 exponent field. Clamping at
    // 88.376_26 lets n reach -128, and that wrap turns exp(-88.376) into -inf
    // instead of an underflowed zero -- which made SiLU of any gate above
    // ~88.4 collapse to zero and silently drop that FFN channel. 88.0 keeps
    // n at +-127, the largest magnitude still representable.
    let x = _mm256_max_ps(
        _mm256_min_ps(x, _mm256_set1_ps(88.0)),
        _mm256_set1_ps(-88.0),
    );
    let magic = _mm256_set1_ps(f32::from_bits(0x4b40_0000));
    let z = _mm256_add_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.442_695_1)), magic);
    let n = _mm256_sub_ps(z, magic);
    let r = _mm256_sub_ps(
        _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(0.693_359_4))),
        _mm256_mul_ps(n, _mm256_set1_ps(-2.121_944_4e-4)),
    );
    let b2 = _mm256_mul_ps(r, r);
    let low = _mm256_add_ps(
        _mm256_set1_ps(f32::from_bits(0x3eff_fedb)),
        _mm256_mul_ps(_mm256_set1_ps(f32::from_bits(0x3e2a_af33)), r),
    );
    let high = _mm256_add_ps(
        _mm256_set1_ps(f32::from_bits(0x3d2b_9f17)),
        _mm256_mul_ps(_mm256_set1_ps(f32::from_bits(0x3c07_2010)), r),
    );
    let j = _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(f32::from_bits(0x3f7f_fff6)), r),
        _mm256_mul_ps(_mm256_add_ps(low, _mm256_mul_ps(high, b2)), b2),
    );
    let exponent = _mm256_add_epi32(
        _mm256_slli_epi32(_mm256_castps_si256(z), 23),
        _mm256_set1_epi32(0x3f80_0000),
    );
    let scale = _mm256_castsi256_ps(exponent);
    // shortcut: flush subnormal scales (n=-128 wraps); use split scaling for subnormal accuracy.
    _mm256_andnot_ps(
        _mm256_cmp_ps::<_CMP_LT_OQ>(n, _mm256_set1_ps(-126.0)),
        _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(1.0), j), scale),
    )
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    #[test]
    fn avx2_underflow_remains_nonnegative_and_finite() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        let input = [
            f32::NEG_INFINITY,
            -1000.0,
            -100.0,
            -89.0,
            -88.0,
            -87.0,
            -1.0,
            0.0,
        ];
        let mut actual = input;
        super::exp_approx_inplace(&mut actual);
        for (x, actual) in input.into_iter().zip(actual) {
            let expected = x.exp();
            assert!(actual.is_finite() && actual >= 0.0, "exp({x}) = {actual}");
            assert!((actual - expected).abs() <= (expected * 1e-6).max(f32::MIN_POSITIVE));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn exp_approx_inplace_neon(values: &mut [f32]) {
    use std::arch::aarch64::*;

    let mut i = 0;
    while i + 4 <= values.len() {
        let x = vld1q_f32(values.as_ptr().add(i));
        vst1q_f32(values.as_mut_ptr().add(i), exp_approx_neon(x));
        i += 4;
    }
    while i < values.len() {
        values[i] = values[i].exp();
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn exp_approx_neon(
    x: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    let r = vdupq_n_f32(f32::from_bits(0x4b40_0000));
    let z = vfmaq_f32(r, x, vdupq_n_f32(f32::from_bits(0x3fb8_aa3b)));
    let n = vsubq_f32(z, r);
    let b = vfmsq_f32(
        vfmsq_f32(x, n, vdupq_n_f32(f32::from_bits(0x3f31_7200))),
        n,
        vdupq_n_f32(f32::from_bits(0x35bf_be8e)),
    );
    let e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    let k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1.0))));
    let c = vcagtq_f32(n, vdupq_n_f32(126.0));
    let u = vmulq_f32(b, b);
    let j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(f32::from_bits(0x3f7f_fff6)), b),
        vfmaq_f32(
            vfmaq_f32(
                vdupq_n_f32(f32::from_bits(0x3eff_fedb)),
                vdupq_n_f32(f32::from_bits(0x3e2a_af33)),
                b,
            ),
            vfmaq_f32(
                vdupq_n_f32(f32::from_bits(0x3d2b_9f17)),
                vdupq_n_f32(f32::from_bits(0x3c07_2010)),
                b,
            ),
            u,
        ),
        u,
    );
    if vaddvq_u32(c) == 0 {
        return vfmaq_f32(k, j, k);
    }
    let d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x8200_0000));
    let s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f00_0000)));
    let s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    vbslq_f32(
        vcagtq_f32(n, vdupq_n_f32(192.0)),
        vmulq_f32(s1, s1),
        vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)),
    )
}

#[cfg(test)]
mod tests {
    use super::exp_approx_inplace;

    /// The fast path builds `2**n` by shifting float bits, so the clamp bound
    /// must keep `n` inside the f32 exponent field. A bound that allowed
    /// `n == -128` wrapped to `-inf` instead of underflowing to zero, which
    /// made `silu` of any gate above ~88.4 collapse to zero and silently drop
    /// that FFN channel.
    #[test]
    fn exp_approx_never_wraps_negative() {
        let inputs: Vec<f32> = (0..512).map(|i| i as f32 * 0.8 - 200.0).collect();
        let mut values = inputs.clone();
        exp_approx_inplace(&mut values);
        for (input, got) in inputs.iter().zip(&values) {
            assert!(*got >= 0.0, "exp_approx({input}) wrapped to {got}");
        }
    }

    /// The exact value that exposed the bug: gate 131.219 in the PhysBrain
    /// FFN made `exp(-131.219)` wrap to `-inf`. Padded past the vector width so
    /// the SIMD path runs rather than the scalar tail.
    #[test]
    fn exp_approx_underflows_to_zero_at_the_clamp_floor() {
        let probes = [-88.0f32, -88.376_26, -90.0, -100.0, -131.219];
        let mut values = vec![0.0f32; 16];
        for (slot, probe) in values.iter_mut().zip(probes.iter()) {
            *slot = *probe;
        }
        exp_approx_inplace(&mut values);
        for (input, got) in probes.iter().zip(&values) {
            assert!(*got >= 0.0, "exp_approx({input}) = {got}");
            assert!(
                got.abs() < 1e-30,
                "exp_approx({input}) should underflow to ~0, got {got}"
            );
        }
    }
}
