//! Activation functions: silu / gelu / fused silu·mul.
//!
//! SIMD dispatch: AVX2+FMA on `x86_64`, NEON on `aarch64`, scalar fallback.
//! Element-wise vector helpers (`vec_mul` / `vec_add*`) live in [`vector`],
//! and the fused `conv1d + silu` kernel lives in [`conv`].

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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gelu_inplace_neon(values: &mut [f32]) {
    let mut i = 0;
    while i + 4 <= values.len() {
        use std::arch::aarch64::*;
        let x = vld1q_f32(values.as_ptr().add(i));
        let x2 = vmulq_f32(x, x);
        let x3 = vmulq_f32(x2, x);
        let c = vdupq_n_f32(0.044715f32);
        let sq2opi = vdupq_n_f32((2.0f32 / std::f32::consts::PI).sqrt());
        let half = vdupq_n_f32(0.5f32);
        let one = vdupq_n_f32(1.0f32);
        let lin = vaddq_f32(x, vmulq_f32(c, x3));
        let inner = vmulq_f32(sq2opi, lin);
        let mut buf = [0.0f32; 4];
        vst1q_f32(buf.as_mut_ptr(), inner);
        for j in 0..4 {
            buf[j] = buf[j].tanh();
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

pub mod conv;
pub mod vector;

pub use conv::*;
pub use vector::*;

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