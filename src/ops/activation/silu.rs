//! Exact and approximate SiLU activations.

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

#[inline(always)]
pub fn silu_inplace(values: &mut [f32]) {
    for value in values {
        *value = silu(*value);
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
