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
pub(crate) unsafe fn exp_approx_avx2(
    x: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let x = _mm256_max_ps(
        _mm256_min_ps(x, _mm256_set1_ps(88.376_26)),
        _mm256_set1_ps(-88.376_26),
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
    _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(1.0), j), scale)
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
