//! Exact and approximate GELU activation functions.

#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

#[inline(always)]
pub fn gelu_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        unsafe { gelu_inplace_avx2(values) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if crate::ops::has_neon() {
        unsafe { gelu_inplace_neon(values) };
        return;
    }
    for value in values {
        *value = gelu(*value);
    }
}

#[inline(always)]
pub fn gelu_approx_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        unsafe { gelu_approx_inplace_avx2(values) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if crate::ops::has_neon() {
        unsafe { gelu_approx_inplace_neon(values) };
        return;
    }
    gelu_inplace(values);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gelu_inplace_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;

    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let c = _mm256_set1_ps(0.044715);
    let sq2opi = _mm256_set1_ps(sqrt_2_over_pi);
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let n8 = values.len() / 8 * 8;
    let mut i = 0;
    while i < n8 {
        let x = _mm256_loadu_ps(values.as_ptr().add(i));
        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x2, x);
        let lin = _mm256_add_ps(x, _mm256_mul_ps(c, x3));
        let inner = _mm256_mul_ps(sq2opi, lin);
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), inner);
        for value in &mut lanes {
            *value = value.tanh();
        }
        let y = _mm256_loadu_ps(lanes.as_ptr());
        let one_plus_y = _mm256_add_ps(one, y);
        let result = _mm256_mul_ps(half, _mm256_mul_ps(x, one_plus_y));
        _mm256_storeu_ps(values.as_mut_ptr().add(i), result);
        i += 8;
    }
    while i < values.len() {
        values[i] = gelu(values[i]);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gelu_approx_inplace_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;

    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let c = _mm256_set1_ps(0.044715);
    let sq2opi = _mm256_set1_ps(sqrt_2_over_pi);
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let n8 = values.len() / 8 * 8;
    let mut i = 0;
    while i < n8 {
        let x = _mm256_loadu_ps(values.as_ptr().add(i));
        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x2, x);
        let lin = _mm256_add_ps(x, _mm256_mul_ps(c, x3));
        let inner = _mm256_mul_ps(sq2opi, lin);
        let y = super::super::math::tanh::tanh_approx_avx2(inner);
        let one_plus_y = _mm256_add_ps(one, y);
        let result = _mm256_mul_ps(half, _mm256_mul_ps(x, one_plus_y));
        _mm256_storeu_ps(values.as_mut_ptr().add(i), result);
        i += 8;
    }
    while i < values.len() {
        values[i] = gelu(values[i]);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gelu_inplace_neon(values: &mut [f32]) {
    use std::arch::aarch64::*;

    let mut i = 0;
    while i + 4 <= values.len() {
        let x = vld1q_f32(values.as_ptr().add(i));
        let x2 = vmulq_f32(x, x);
        let x3 = vmulq_f32(x2, x);
        let c = vdupq_n_f32(0.044715);
        let sq2opi = vdupq_n_f32((2.0f32 / std::f32::consts::PI).sqrt());
        let half = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);
        let lin = vaddq_f32(x, vmulq_f32(c, x3));
        let inner = vmulq_f32(sq2opi, lin);
        let mut buf = [0.0f32; 4];
        vst1q_f32(buf.as_mut_ptr(), inner);
        for value in &mut buf {
            *value = value.tanh();
        }
        let y = vld1q_f32(buf.as_ptr());
        let one_plus_y = vaddq_f32(one, y);
        let result = vmulq_f32(half, vmulq_f32(x, one_plus_y));
        vst1q_f32(values.as_mut_ptr().add(i), result);
        i += 4;
    }
    while i < values.len() {
        values[i] = gelu(values[i]);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gelu_approx_inplace_neon(values: &mut [f32]) {
    use std::arch::aarch64::*;

    let mut i = 0;
    while i + 4 <= values.len() {
        let x = vld1q_f32(values.as_ptr().add(i));
        let x2 = vmulq_f32(x, x);
        let x3 = vmulq_f32(x2, x);
        let c = vdupq_n_f32(0.044715);
        let sq2opi = vdupq_n_f32((2.0f32 / std::f32::consts::PI).sqrt());
        let half = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);
        let lin = vaddq_f32(x, vmulq_f32(c, x3));
        let inner = vmulq_f32(sq2opi, lin);
        let mut buf = [0.0f32; 4];
        vst1q_f32(buf.as_mut_ptr(), inner);
        for value in &mut buf {
            *value = value.tanh();
        }
        let y = vld1q_f32(buf.as_ptr());
        // 待验证是否可以替换
        // let y = super::super::math::tanh::tanh_approx_neon(inner);
        let one_plus_y = vaddq_f32(one, y);
        let result = vmulq_f32(half, vmulq_f32(x, one_plus_y));
        vst1q_f32(values.as_mut_ptr().add(i), result);
        i += 4;
    }
    while i < values.len() {
        values[i] = gelu(values[i]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) {
        let rel_diff = (a - b).abs() / a.abs().max(b.abs()).max(1e-6);
        assert!(
            rel_diff < 1e-5 || (a - b).abs() < 1e-5,
            "a={a}, b={b}, rel={rel_diff}"
        );
    }

    fn gelu_scalar(values: &[f32]) -> Vec<f32> {
        values.iter().map(|&x| gelu(x)).collect()
    }

    #[test]
    fn gelu_inplace_matches_scalar() {
        let input: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01 - 5.0).collect();
        let mut output = input.clone();
        gelu_inplace(&mut output);
        for (actual, expected) in output.iter().zip(gelu_scalar(&input).iter()) {
            close(*actual, *expected);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn gelu_inplace_avx2_matches_scalar_value() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        let input: Vec<f32> = (0..1031)
            .map(|i| ((i as f32 * 0.037).sin() * 8.0) - 2.0)
            .collect();
        let mut output = input.clone();
        gelu_inplace(&mut output);
        for (index, (actual, expected)) in output.iter().zip(gelu_scalar(&input).iter()).enumerate()
        {
            let error = (actual - expected).abs();
            let tolerance = 2e-6f32.max(expected.abs() * 2e-5);
            assert!(
                error <= tolerance,
                "index={index}, actual={actual}, expected={expected}, error={error}"
            );
        }
    }

    #[test]
    fn gelu_approx_inplace_matches_scalar() {
        let input: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01 - 5.0).collect();
        let mut output = input.clone();
        gelu_approx_inplace(&mut output);
        for (actual, expected) in output.iter().zip(gelu_scalar(&input).iter()) {
            assert!(
                (actual - expected).abs() < 2e-3,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn gelu_approx_inplace_tail_elements() {
        for &len in &[6usize, 7, 8, 13, 14, 15] {
            let input: Vec<f32> = (0..len).map(|i| i as f32 * 0.1 - 2.0).collect();
            let mut output = input.clone();
            gelu_approx_inplace(&mut output);
            for (actual, expected) in output.iter().zip(gelu_scalar(&input).iter()) {
                assert!(
                    (actual - expected).abs() < 2e-3,
                    "actual={actual}, expected={expected}"
                );
            }
        }
    }
}
