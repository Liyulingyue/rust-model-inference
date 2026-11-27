//! Exact ReLU activation: `out[i] = max(0, in[i])`.

#[inline(always)]
pub fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

#[inline]
pub fn relu_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        unsafe { relu_inplace_avx2(values) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if crate::ops::has_neon() {
        unsafe { relu_inplace_neon(values) };
        return;
    }
    for value in values {
        *value = relu(*value);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn relu_inplace_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;

    let n8 = values.len() / 8 * 8;
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i < n8 {
        let v = _mm256_loadu_ps(values.as_ptr().add(i));
        _mm256_storeu_ps(values.as_mut_ptr().add(i), _mm256_max_ps(v, zero));
        i += 8;
    }
    while i < values.len() {
        values[i] = relu(values[i]);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn relu_inplace_neon(values: &mut [f32]) {
    use std::arch::aarch64::*;

    let zero = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= values.len() {
        let v = vld1q_f32(values.as_ptr().add(i));
        vst1q_f32(values.as_mut_ptr().add(i), vmaxq_f32(v, zero));
        i += 4;
    }
    while i < values.len() {
        values[i] = relu(values[i]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::relu;
    use super::relu_inplace;

    fn assert_inplace_matches_scalar(values: &[f32]) {
        let mut simd = values.to_vec();
        let mut scalar = values.to_vec();
        relu_inplace(&mut simd);
        for (i, v) in scalar.iter_mut().enumerate() {
            *v = relu(*v);
        }
        for (i, (a, b)) in simd.iter().zip(scalar.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "lane {i}: simd={a} scalar={b}");
        }
    }

    #[test]
    fn relu_inplace_matches_scalar_small() {
        assert_inplace_matches_scalar(&[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn relu_inplace_matches_scalar_large() {
        let values: Vec<f32> = (0..512)
            .map(|i| ((i * 13 + 7) % 47) as f32 * 0.13 - 6.0)
            .collect();
        assert_inplace_matches_scalar(&values);
    }

    #[test]
    fn relu_inplace_matches_scalar_with_tail() {
        let values: Vec<f32> = (0..37).map(|i| (i as f32 * 0.21).sin() * 4.0).collect();
        assert_inplace_matches_scalar(&values);
    }
}
