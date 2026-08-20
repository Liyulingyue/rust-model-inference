//! Activation functions: silu + element-wise vector helpers + conv1d.

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

#[inline(always)]
pub fn silu_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if super::has_neon() {
        unsafe {
            silu_inplace_neon(values);
        }
        return;
    }
    for value in values {
        *value = *value / (1.0 + (-*value).exp());
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
        let exp_neg_x = super::attention::ggml_expf_neon(neg_x);
        let one_plus_exp_neg_x = vaddq_f32(vdupq_n_f32(1.0), exp_neg_x);
        vst1q_f32(values.as_mut_ptr().add(i), vdivq_f32(x, one_plus_exp_neg_x));
        i += 4;
    }
    while i < values.len() {
        let value = values[i];
        values[i] = value / (1.0 + (-value).exp());
        i += 1;
    }
}

#[inline(always)]
pub fn silu_mul_inplace(gate: &[f32], up: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    #[cfg(target_arch = "aarch64")]
    if super::has_neon() {
        unsafe {
            silu_mul_inplace_neon(gate, up);
        }
        return;
    }
    let n = gate.len();
    for i in 0..n {
        let g = gate[i];
        up[i] *= g / (1.0 + (-g).exp());
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn silu_mul_inplace_neon(gate: &[f32], up: &mut [f32]) {
    use std::arch::aarch64::*;

    let mut i = 0;
    while i + 4 <= gate.len() {
        let x = vld1q_f32(gate.as_ptr().add(i));
        let multiplier = vld1q_f32(up.as_ptr().add(i));
        let one = vdupq_n_f32(1.0);
        let zero = vdupq_n_f32(0.0);
        let neg_x = vsubq_f32(zero, x);
        let exp_neg_x = super::attention::ggml_expf_neon(neg_x);
        let one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
        let silu = vdivq_f32(x, one_plus_exp_neg_x);
        vst1q_f32(up.as_mut_ptr().add(i), vmulq_f32(silu, multiplier));
        i += 4;
    }
    while i < gate.len() {
        let g = gate[i];
        up[i] *= g / (1.0 + (-g).exp());
        i += 1;
    }
}

#[inline(always)]
pub fn vec_mul_inplace(a: &[f32], b: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2_fma() {
            unsafe { vec_mul_avx2(a, b) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if super::has_neon() {
            unsafe {
                vec_mul_neon(a, b);
            }
            return;
        }
    }
    for i in 0..a.len() {
        b[i] *= a[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vec_mul_neon(a: &[f32], b: &mut [f32]) {
    use std::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= b.len() {
        let value = vmulq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
        vst1q_f32(b.as_mut_ptr().add(i), value);
        i += 4;
    }
    while i < b.len() {
        b[i] *= a[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vec_mul_avx2(a: &[f32], b: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let n8 = n / 8 * 8;
    let mut i = 0;
    while i < n8 {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        _mm256_storeu_ps(b.as_mut_ptr().add(i), _mm256_mul_ps(va, vb));
        i += 8;
    }
    while i < n {
        b[i] *= a[i];
        i += 1;
    }
}

#[inline(always)]
pub fn vec_add_into(a: &[f32], b: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2_fma() {
            unsafe { vec_add_avx2(a, b) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if super::has_neon() {
            unsafe {
                vec_add_neon(a, b);
            }
            return;
        }
    }
    for i in 0..a.len() {
        b[i] += a[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vec_add_neon(a: &[f32], b: &mut [f32]) {
    use std::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= b.len() {
        let value = vaddq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
        vst1q_f32(b.as_mut_ptr().add(i), value);
        i += 4;
    }
    while i < b.len() {
        b[i] += a[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vec_add_avx2(a: &[f32], b: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let n8 = n / 8 * 8;
    let mut i = 0;
    while i < n8 {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        _mm256_storeu_ps(b.as_mut_ptr().add(i), _mm256_add_ps(va, vb));
        i += 8;
    }
    while i < n {
        b[i] += a[i];
        i += 1;
    }
}

#[inline(always)]
pub fn conv1d_silu(
    kernel: &[f32],
    state: &[f32],
    input: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    let out_len = output.len();
    let kernel_len = kernel.len();
    debug_assert!(input.len() >= out_len + kernel_len - 1);
    for o in 0..out_len {
        let mut acc = bias.map_or(0.0, |bias| bias[o]);
        for k in 0..kernel_len {
            acc += input[o + k] * kernel[k];
        }
        acc += state[o] * kernel[0];
        output[o] = silu(acc);
    }
}