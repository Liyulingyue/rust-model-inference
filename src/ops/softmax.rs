//! Softmax activation and related utilities.

#[inline(always)]
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for value in x.iter_mut() {
        *value = (*value - max).exp();
        sum += f64::from(*value);
    }
    let scale = (1.0 / sum) as f32;
    for value in x {
        *value *= scale;
    }
}

#[inline(always)]
pub fn softmax_approx_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        unsafe {
            softmax_approx_inplace_avx2(x);
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if super::has_neon() {
        unsafe {
            softmax_approx_inplace_neon(x);
        }
        return;
    }
    softmax_inplace(x);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn softmax_approx_inplace_avx2(x: &mut [f32]) {
    use std::arch::x86_64::*;

    let n8 = x.len() / 8 * 8;
    let mut i = 0;

    // 第一遍：找 max (AVX2 批量)
    let mut max_val = f32::NEG_INFINITY;
    while i < n8 {
        let vals = _mm256_loadu_ps(x.as_ptr().add(i));
        let max_v = _mm256_max_ps(vals, _mm256_set1_ps(max_val));
        // 水平提取 max
        let lo = _mm256_castps256_ps128(max_v);
        let hi = _mm_castsi128_ps(_mm256_extractf128_si256(_mm256_castps_si256(max_v), 1));
        let combined = _mm_max_ps(lo, hi);
        let high = _mm_movehl_ps(combined, combined);
        let final_max = _mm_max_ss(combined, high);
        max_val = max_val.max(_mm_cvtss_f32(final_max));
        i += 8;
    }
    while i < x.len() {
        max_val = max_val.max(x[i]);
        i += 1;
    }

    // 第二遍：计算 exp(x - max) 并累加 sum
    let max_v = _mm256_set1_ps(max_val);
    let mut sum: f64 = 0.0;
    i = 0;
    while i < n8 {
        let vals = _mm256_loadu_ps(x.as_ptr().add(i));
        let sub_vals = _mm256_sub_ps(vals, max_v);
        let exp_vals = super::math::exp::exp_approx_avx2(sub_vals);
        _mm256_storeu_ps(x.as_mut_ptr().add(i), exp_vals);
        // 水平累加 exp_vals: 先高低128位相加，再用hadd得到最终sum
        let lo = _mm256_castps256_ps128(exp_vals);
        let hi = _mm_castsi128_ps(_mm256_extractf128_si256(_mm256_castps_si256(exp_vals), 1));
        let sum128 = _mm_add_ps(lo, hi);
        let hadded = _mm_hadd_ps(sum128, sum128);
        let final_sum = _mm_cvtss_f32(_mm_hadd_ps(hadded, hadded));
        sum += f64::from(final_sum);
        i += 8;
    }
    while i < x.len() {
        x[i] = (x[i] - max_val).exp();
        sum += f64::from(x[i]);
        i += 1;
    }

    // 第三遍：scale
    let scale = (1.0 / sum) as f32;
    let scale_v = _mm256_set1_ps(scale);
    i = 0;
    while i < n8 {
        let vals = _mm256_loadu_ps(x.as_ptr().add(i));
        let scaled = _mm256_mul_ps(vals, scale_v);
        _mm256_storeu_ps(x.as_mut_ptr().add(i), scaled);
        i += 8;
    }
    while i < x.len() {
        x[i] *= scale;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn softmax_approx_inplace_neon(x: &mut [f32]) {
    use std::arch::aarch64::*;

    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    let mut i = 0;
    while i + 4 <= x.len() {
        let values = super::math::exp::exp_approx_neon(vsubq_f32(
            vld1q_f32(x.as_ptr().add(i)),
            vdupq_n_f32(max),
        ));
        vst1q_f32(x.as_mut_ptr().add(i), values);
        sum += f64::from(vaddvq_f32(values));
        i += 4;
    }
    while i < x.len() {
        x[i] = (x[i] - max).exp();
        sum += f64::from(x[i]);
        i += 1;
    }
    vec_scale_f32_neon(x, (1.0 / sum) as f32);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vec_scale_f32_neon(x: &mut [f32], scale: f32) {
    use std::arch::aarch64::*;
    let vscale = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= x.len() {
        vst1q_f32(
            x.as_mut_ptr().add(i),
            vmulq_f32(vld1q_f32(x.as_ptr().add(i)), vscale),
        );
        i += 4;
    }
    while i < x.len() {
        x[i] *= scale;
        i += 1;
    }
}

pub(crate) fn softmax_exp_sum(x: &mut [f32], max: f32) -> f64 {
    #[cfg(target_arch = "aarch64")]
    if super::has_neon() {
        return unsafe { softmax_exp_sum_approx_inplace_neon(x, max) };
    }
    let mut sum = 0.0f64;
    for value in x {
        *value = (*value - max).exp();
        sum += f64::from(*value);
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn softmax_exp_sum_approx_inplace_neon(x: &mut [f32], max: f32) -> f64 {
    use std::arch::aarch64::*;

    let mut sum = 0.0f64;
    let mut i = 0;
    while i + 4 <= x.len() {
        let values = super::math::exp::exp_approx_neon(vsubq_f32(
            vld1q_f32(x.as_ptr().add(i)),
            vdupq_n_f32(max),
        ));
        vst1q_f32(x.as_mut_ptr().add(i), values);
        sum += f64::from(vaddvq_f32(values));
        i += 4;
    }
    while i < x.len() {
        x[i] = (x[i] - max).exp();
        sum += f64::from(x[i]);
        i += 1;
    }
    sum
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use std::time::Instant;

    fn softmax_scalar(values: &mut [f32]) {
        if values.is_empty() {
            return;
        }
        let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in values.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in values.iter_mut() {
                *v /= sum;
            }
        }
    }

    #[test]
    fn softmax_approx_avx2_matches_scalar() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        let inputs: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![-10.0, -5.0, 0.0, 5.0, 10.0, 15.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![f32::NEG_INFINITY, 0.0, 1.0, 2.0],
            (0..32).map(|i| (i as f32) * 0.1 - 2.0).collect(),
        ];

        for input in inputs {
            let mut scalar_out = input.clone();
            softmax_scalar(&mut scalar_out);

            let mut approx_out = input.clone();
            unsafe { softmax_approx_inplace_avx2(&mut approx_out) };

            for (s, a) in scalar_out.iter().zip(approx_out.iter()) {
                let rel_diff = (s - a).abs() / s.abs().max(a.abs()).max(1e-6);
                assert!(
                    rel_diff < 1e-6,
                    "scalar={}, approx={}, rel={}",
                    s,
                    a,
                    rel_diff
                );
            }
        }
    }

    #[test]
    fn softmax_approx_avx2_tail_elements() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        for len in [7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let input: Vec<f32> = (0..len).map(|i| i as f32 * 0.1 - 2.0).collect();
            let mut scalar_out = input.clone();
            softmax_scalar(&mut scalar_out);

            let mut approx_out = input.clone();
            unsafe { softmax_approx_inplace_avx2(&mut approx_out) };

            for (s, a) in scalar_out.iter().zip(approx_out.iter()) {
                let rel_diff = (s - a).abs() / s.abs().max(a.abs()).max(1e-6);
                assert!(
                    rel_diff < 1e-6,
                    "len={}, scalar={}, approx={}, rel={}",
                    len,
                    s,
                    a,
                    rel_diff
                );
            }
        }
    }
    #[test]
    fn softmax_speed_comparison() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        let sizes = [1024, 8192, 65536];
        let iterations = 100;

        for size in sizes {
            let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 50.0).collect();

            for _ in 0..10 {
                let mut out = input.clone();
                unsafe { softmax_approx_inplace_avx2(&mut out) };
            }

            let start = Instant::now();
            for _ in 0..iterations {
                let mut out = input.clone();
                unsafe { softmax_approx_inplace_avx2(&mut out) };
            }
            let avx2_time = start.elapsed();

            let start = Instant::now();
            for _ in 0..iterations {
                let mut out = input.clone();
                softmax_scalar(&mut out);
            }
            let scalar_time = start.elapsed();

            eprintln!(
                "size={}, AVX2={:.2}ms, Scalar={:.2}ms, Speedup={:.2}x",
                size,
                avx2_time.as_secs_f64() * 1000.0 / iterations as f64,
                scalar_time.as_secs_f64() * 1000.0 / iterations as f64,
                scalar_time.as_nanos() as f64 / avx2_time.as_nanos() as f64
            );
        }
    }
}
