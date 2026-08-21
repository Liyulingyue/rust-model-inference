//! Activation functions: silu + gelu + element-wise vector helpers + conv1d.

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

#[inline(always)]
pub fn gelu_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if super::has_avx2_fma() {
        unsafe { gelu_avx2(values) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if super::has_neon() {
        unsafe { gelu_inplace_neon(values) };
        return;
    }
    for value in values {
        *value = gelu(*value);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gelu_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let c = _mm256_set1_ps(0.044715f32);
    let sq2opi = _mm256_set1_ps(sqrt_2_over_pi);
    let half = _mm256_set1_ps(0.5f32);
    let one = _mm256_set1_ps(1.0f32);
    let n = values.len();
    let n8 = n / 8 * 8;
    let mut i = 0;
    while i < n8 {
        let x = _mm256_loadu_ps(values.as_ptr().add(i));
        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x2, x);
        let lin = _mm256_add_ps(x, _mm256_mul_ps(c, x3));
        let inner = _mm256_mul_ps(sq2opi, lin);
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), inner);
        for j in 0..8 {
            buf[j] = buf[j].tanh();
        }
        let y = _mm256_loadu_ps(buf.as_ptr());
        let one_plus_y = _mm256_add_ps(one, y);
        let result = _mm256_mul_ps(half, _mm256_mul_ps(x, one_plus_y));
        _mm256_storeu_ps(values.as_mut_ptr().add(i), result);
        i += 8;
    }
    while i < n {
        values[i] = gelu(values[i]);
        i += 1;
    }
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
        let mut avx2_out = input.clone();
        gelu_inplace(&mut avx2_out);
        let scalar_out = gelu_scalar(&input);
        for (a, b) in avx2_out.iter().zip(scalar_out.iter()) {
            close(*a, *b);
        }
    }

    #[test]
    fn gelu_inplace_real_network_values() {
        let input: Vec<f32> = (0..7077888)
            .map(|i| {
                let x = (i as f32 * 0.0001).sin() * 10.0;
                x
            })
            .collect();
        let mut avx2_out = input.clone();
        gelu_inplace(&mut avx2_out);
        let scalar_out = gelu_scalar(&input);
        let max_rel_diff = avx2_out
            .iter()
            .zip(scalar_out.iter())
            .map(|(a, b)| {
                let rel = if a.abs().max(b.abs()) > 1e-5 {
                    (a - b).abs() / a.abs().max(b.abs())
                } else {
                    (a - b).abs()
                };
                rel
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_rel_diff < 2e-2,
            "max_rel_diff={max_rel_diff} too large"
        );
    }

    #[test]
    fn gelu_inplace_tail_elements() {
        let n = 7;
        for &len in &[n - 1, n, n + 1, n * 2 - 1, n * 2, n * 2 + 1] {
            let input: Vec<f32> = (0..len).map(|i| i as f32 * 0.1 - 2.0).collect();
            let mut avx2_out = input.clone();
            gelu_inplace(&mut avx2_out);
            let scalar_out = gelu_scalar(&input);
            for (a, b) in avx2_out.iter().zip(scalar_out.iter()) {
                close(*a, *b);
            }
        }
    }
}
