//! RMS normalization + scale helpers.

pub fn rms_norm(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let n = input.len().min(weight.len()).min(output.len());
    let sum_sq: f64 = input[..n].iter().map(|&x| f64::from(x * x)).sum();
    let mean_sq = (sum_sq / n as f64) as f32;
    let scale = 1.0f32 / (mean_sq + eps).sqrt();
    for i in 0..n {
        output[i] = input[i] * scale * weight[i];
    }
}

pub fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len().min(weight.len());
    let sum_sq: f64 = x[..n].iter().map(|&value| f64::from(value * value)).sum();
    let mean_sq = (sum_sq / n as f64) as f32;
    let scale = 1.0f32 / (mean_sq + eps).sqrt();
    scale_mul_inplace(scale, &weight[..n], &mut x[..n]);
}

fn scale_mul_inplace(scale: f32, weight: &[f32], x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2_fma() {
            unsafe { scale_mul_avx2(scale, weight, x) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if super::has_neon() {
            unsafe {
                scale_mul_neon(scale, weight, x);
            }
            return;
        }
    }
    for i in 0..weight.len() {
        x[i] = x[i] * scale * weight[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_mul_neon(scale: f32, weight: &[f32], x: &mut [f32]) {
    use std::arch::aarch64::*;
    let scale_v = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= x.len() {
        let value = vmulq_f32(
            vmulq_f32(vld1q_f32(x.as_ptr().add(i)), scale_v),
            vld1q_f32(weight.as_ptr().add(i)),
        );
        vst1q_f32(x.as_mut_ptr().add(i), value);
        i += 4;
    }
    while i < x.len() {
        x[i] = x[i] * scale * weight[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn scale_mul_avx2(scale: f32, weight: &[f32], x: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = weight.len();
    let n8 = n / 8 * 8;
    let vscale = _mm256_set1_ps(scale);
    let mut i = 0;
    while i < n8 {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        let vw = _mm256_loadu_ps(weight.as_ptr().add(i));
        _mm256_storeu_ps(
            x.as_mut_ptr().add(i),
            _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw),
        );
        i += 8;
    }
    while i < n {
        x[i] = x[i] * scale * weight[i];
        i += 1;
    }
}