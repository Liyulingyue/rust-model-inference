//! Exact and approximate SiLU activations.

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
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
        let values: Vec<f32> = (0..37)
            .map(|i| (i as f32 * 0.21).sin() * 4.0)
            .collect();
        assert_inplace_matches_scalar(&values);
    }
}
