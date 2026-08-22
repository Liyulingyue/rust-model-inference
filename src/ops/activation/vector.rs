//! Element-wise vector helpers: `b[i] *= a[i]`, `b[i] += a[i]`, `c[i] = a[i] + b[i]`.
//!
//! SIMD dispatch: AVX2+FMA on `x86_64`, NEON on `aarch64`, scalar fallback.

#[inline(always)]
pub fn vec_mul_inplace(a: &[f32], b: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if super::super::has_avx2_fma() {
            unsafe { vec_mul_avx2(a, b) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if super::super::has_neon() {
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
        if super::super::has_avx2_fma() {
            unsafe { vec_add_avx2(a, b) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if super::super::has_neon() {
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
        if super::super::has_avx2_fma() {
            unsafe { vec_add3_avx2(a, b, c) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if super::super::has_neon() {
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

#[cfg(test)]
mod tests {
    use super::*;

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
            let mut b_avx2: Vec<f32> =
                (0..len).map(|i| (i as f32 * 0.029).cos() * 10.0).collect();
            vec_add_into(&a, &mut b_avx2);
            for i in 0..len {
                let expected =
                    ((i as f32 * 0.029).cos() * 10.0) + ((i as f32 * 0.013).sin() * 50.0);
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
            if diff > max_abs {
                max_abs = diff;
            }
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