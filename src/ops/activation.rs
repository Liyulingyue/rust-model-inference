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
pub fn vec_add(a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2_fma() {
            unsafe { vec_add3_avx2(a, b, c) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if super::has_neon() {
            unsafe { vec_add3_neon(a, b, c) };
            return;
        }
    }
    for i in 0..a.len() {
        c[i] = a[i] + b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vec_add3_neon(a: &[f32], b: &[f32], c: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = a.len();
    let n4 = n / 4 * 4;
    let mut i = 0;
    while i < n4 {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        vst1q_f32(c.as_mut_ptr().add(i), vaddq_f32(va, vb));
        i += 4;
    }
    while i < n {
        c[i] = a[i] + b[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vec_add3_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let n8 = n / 8 * 8;
    let mut i = 0;
    while i < n8 {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        _mm256_storeu_ps(c.as_mut_ptr().add(i), _mm256_add_ps(va, vb));
        i += 8;
    }
    while i < n {
        c[i] = a[i] + b[i];
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

    #[test]
    fn vec_add_basic() {
        let a: Vec<f32> = vec![1.0, -2.0, 3.5, -4.25, 0.0, 100.0, -200.0, 1e-6, 1e6];
        let b: Vec<f32> = vec![0.5, 0.5, -1.0, 1.0, 7.0, -50.0, 250.0, 1e-7, -1e6];
        let mut c = vec![0.0f32; a.len()];
        vec_add(&a, &b, &mut c);
        for i in 0..a.len() {
            assert_eq!(c[i], a[i] + b[i], "index {i}");
        }
    }

    #[test]
    fn vec_add_matches_scalar() {
        for &len in &[1usize, 7, 8, 15, 16, 17, 63, 64, 65, 1000, 3072, 2304 * 3072] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.013).sin() * 50.0).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.029).cos() * 10.0).collect();
            let mut c_avx2 = vec![0.0f32; len];
            vec_add(&a, &b, &mut c_avx2);
            for i in 0..len {
                assert_eq!(c_avx2[i], a[i] + b[i], "len={len} index {i}");
            }
        }
    }

    #[test]
    fn vec_add_into_matches_scalar() {
        for &len in &[1usize, 7, 8, 15, 16, 17, 63, 64, 65, 1000, 3072] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.013).sin() * 50.0).collect();
            let mut b_avx2: Vec<f32> = (0..len).map(|i| (i as f32 * 0.029).cos() * 10.0).collect();
            vec_add_into(&a, &mut b_avx2);
            for i in 0..len {
                let expected = ((i as f32 * 0.029).cos() * 10.0) + ((i as f32 * 0.013).sin() * 50.0);
                assert_eq!(b_avx2[i], expected, "len={len} index {i}");
            }
        }
    }

    #[test]
    fn vec_add_real_network_values() {
        let n = 7077888usize;
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0001).sin() * 10.0).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0003).cos() * 5.0).collect();
        let mut c = vec![0.0f32; n];
        vec_add(&a, &b, &mut c);
        let mut max_abs = 0.0f32;
        for i in 0..n {
            let expected = a[i] + b[i];
            let diff = (c[i] - expected).abs();
            if diff > max_abs { max_abs = diff; }
        }
        assert_eq!(max_abs, 0.0, "max_abs={max_abs}");
    }

    #[test]
    fn vec_add_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut c: Vec<f32> = vec![];
        vec_add(&a, &b, &mut c);
        assert_eq!(c.len(), 0);
    }
}
