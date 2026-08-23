//! Attention helpers: softmax + value accumulation.

pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    #[cfg(feature = "parity-trace")]
    {
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
        return;
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "parity-trace")))]
    if super::has_neon() {
        unsafe {
            softmax_neon_ggml(x);
        }
        return;
    }
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

pub fn attention_value_f32(
    values: &[f32],
    weights: &[f32],
    n_cached: usize,
    n_padded: usize,
) -> f32 {
    debug_assert!(n_cached <= n_padded);
    super::dot_f32(values, weights, n_padded)
}

pub(crate) fn softmax_exp_sum(x: &mut [f32], max: f32) -> f64 {
    #[cfg(target_arch = "aarch64")]
    if super::has_neon() {
        return unsafe { softmax_exp_sum_neon(x, max) };
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
unsafe fn softmax_exp_sum_neon(x: &mut [f32], max: f32) -> f64 {
    use std::arch::aarch64::*;

    let mut sum = 0.0f64;
    let mut i = 0;
    while i + 4 <= x.len() {
        let values = super::math::exp::exp_approx_neon(vsubq_f32(vld1q_f32(x.as_ptr().add(i)), vdupq_n_f32(max)));
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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn softmax_neon_ggml(x: &mut [f32]) {
    use std::arch::aarch64::*;

    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    let mut i = 0;
    while i + 4 <= x.len() {
        let values = super::math::exp::exp_approx_neon(vsubq_f32(vld1q_f32(x.as_ptr().add(i)), vdupq_n_f32(max)));
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
