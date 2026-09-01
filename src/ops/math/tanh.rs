//! Exact and approximate hyperbolic tangent helpers.

#[inline(always)]
pub fn tanh_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        unsafe { tanh_inplace_avx2(values) };
        return;
    }
    for value in values {
        *value = value.tanh();
    }
}

#[inline(always)]
pub fn tanh_approx_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        unsafe { tanh_approx_inplace_avx2(values) };
        return;
    }
    tanh_inplace(values);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn tanh_inplace_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;

    let mut i = 0;
    while i + 8 <= values.len() {
        let x = _mm256_loadu_ps(values.as_ptr().add(i));
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), x);
        for value in &mut lanes {
            *value = value.tanh();
        }
        let y = _mm256_loadu_ps(lanes.as_ptr());
        _mm256_storeu_ps(values.as_mut_ptr().add(i), y);
        i += 8;
    }
    while i < values.len() {
        values[i] = values[i].tanh();
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn tanh_approx_inplace_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;

    let mut i = 0;
    while i + 8 <= values.len() {
        let x = _mm256_loadu_ps(values.as_ptr().add(i));
        _mm256_storeu_ps(values.as_mut_ptr().add(i), tanh_approx_avx2(x));
        i += 8;
    }
    while i < values.len() {
        values[i] = values[i].tanh();
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn tanh_approx_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let two = _mm256_set1_ps(2.0);
    let min_x = _mm256_max_ps(x, _mm256_set1_ps(-10.0));
    let max_x = _mm256_min_ps(min_x, _mm256_set1_ps(10.0));
    let exp_2x = super::exp::exp_approx_avx2(_mm256_mul_ps(two, max_x));
    let numerator = _mm256_sub_ps(exp_2x, one);
    let denominator = _mm256_add_ps(exp_2x, one);
    let result = _mm256_div_ps(numerator, denominator);
    _mm256_blendv_ps(result, zero, _mm256_cmp_ps(x, x, _CMP_UNORD_Q))
}

// 待验证，速度/精度问题
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn tanh_approx_neon(
    x: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let two = vdupq_n_f32(2.0);
    let min_x = vmaxq_f32(x, vdupq_n_f32(-10.0));
    let max_x = vminq_f32(min_x, vdupq_n_f32(10.0));
    let exp_2x = super::exp::exp_approx_neon(vmulq_f32(two, max_x));
    let numerator = vsubq_f32(exp_2x, one);
    let denominator = vaddq_f32(exp_2x, one);
    vdivq_f32(numerator, denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tanh_inplace_matches_scalar() {
        let input: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01 - 5.0).collect();
        let mut output = input.clone();
        tanh_inplace(&mut output);
        for (actual, expected) in output.iter().zip(input.iter().map(|value| value.tanh())) {
            assert_eq!(*actual, expected);
        }
    }

    #[test]
    fn tanh_approx_inplace_is_close() {
        let input: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01 - 5.0).collect();
        let mut output = input.clone();
        tanh_approx_inplace(&mut output);
        for (actual, expected) in output.iter().zip(input.iter().map(|value| value.tanh())) {
            assert!(
                (*actual - expected).abs() < 2e-3,
                "actual={actual}, expected={expected}"
            );
        }
    }
}
