use super::super::f16_to_f32;
use super::super::has_avx2_fma;
use super::super::has_f16c;
use super::super::has_neon;

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32], n: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    let mut sum = hsum_ps(acc);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[inline(always)]
fn dot_f32_scalar(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut s = 0.0f32;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32], n: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 16 <= n {
        acc0 = vfmaq_f32(
            acc0,
            vld1q_f32(a.as_ptr().add(i)),
            vld1q_f32(b.as_ptr().add(i)),
        );
        acc1 = vfmaq_f32(
            acc1,
            vld1q_f32(a.as_ptr().add(i + 4)),
            vld1q_f32(b.as_ptr().add(i + 4)),
        );
        acc2 = vfmaq_f32(
            acc2,
            vld1q_f32(a.as_ptr().add(i + 8)),
            vld1q_f32(b.as_ptr().add(i + 8)),
        );
        acc3 = vfmaq_f32(
            acc3,
            vld1q_f32(a.as_ptr().add(i + 12)),
            vld1q_f32(b.as_ptr().add(i + 12)),
        );
        i += 16;
    }
    acc0 = vaddq_f32(acc0, acc2);
    acc1 = vaddq_f32(acc1, acc3);
    let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

pub fn dot_f16_f32(a: &[f32], b_f16: &[u16], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() && has_f16c() {
            return unsafe { dot_f16_f32_avx2(a, b_f16, n) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            return unsafe { dot_f16_f32_neon(a, b_f16, n) };
        }
    }
    let mut s = 0.0f32;
    for i in 0..n {
        s += a[i] * f16_to_f32(b_f16[i]);
    }
    s
}

pub fn dot_f16(a: &[u16], b: &[u16], n: usize) -> f32 {
    debug_assert!(a.len() >= n && b.len() >= n);
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    let (mut sum, tail_start) = {
        let prefix = n & !31;
        if prefix > 0 && std::arch::is_aarch64_feature_detected!("fp16") {
            (
                f64::from(unsafe { dot_f16_fp16_neon(a.as_ptr(), b.as_ptr(), prefix) }),
                prefix,
            )
        } else {
            (0.0, 0)
        }
    };
    #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
    let (mut sum, tail_start) = (0.0f64, 0usize);
    for index in tail_start..n {
        sum += f64::from(f16_to_f32(a[index]) * f16_to_f32(b[index]));
    }
    sum as f32
}

pub fn dot_f16_f16_bytes(a: &[u16], b: &[u8], n: usize) -> f32 {
    debug_assert!(a.len() >= n && b.len() >= n * 2);
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    let (mut sum, tail_start) = {
        let prefix = n & !31;
        if prefix > 0 && std::arch::is_aarch64_feature_detected!("fp16") {
            (
                f64::from(unsafe {
                    dot_f16_fp16_neon(a.as_ptr(), b.as_ptr().cast::<u16>(), prefix)
                }),
                prefix,
            )
        } else {
            (0.0, 0)
        }
    };
    #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
    let (mut sum, tail_start) = (0.0f64, 0usize);
    for index in tail_start..n {
        let weight = u16::from_le_bytes(b[index * 2..index * 2 + 2].try_into().unwrap());
        sum += f64::from(f16_to_f32(a[index]) * f16_to_f32(weight));
    }
    sum as f32
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
unsafe fn dot_f16_fp16_neon(x: *const u16, y: *const u16, n: usize) -> f32 {
    debug_assert_eq!(n % 32, 0);
    let bits: u32;
    asm!(
        "movi v0.8h, #0", "movi v1.8h, #0",
        "movi v2.8h, #0", "movi v3.8h, #0",
        "cbz {n}, 2f",
        "1:",
        "ld1 {{v4.8h-v7.8h}}, [{x}], #64",
        "ld1 {{v16.8h-v19.8h}}, [{y}], #64",
        "fmla v0.8h, v4.8h, v16.8h",
        "fmla v1.8h, v5.8h, v17.8h",
        "fmla v2.8h, v6.8h, v18.8h",
        "fmla v3.8h, v7.8h, v19.8h",
        "subs {n}, {n}, #32",
        "b.ne 1b",
        "2:",
        "fadd v0.8h, v0.8h, v2.8h",
        "fadd v1.8h, v1.8h, v3.8h",
        "fadd v0.8h, v0.8h, v1.8h",
        "fcvtl v4.4s, v0.4h",
        "fcvtl2 v5.4s, v0.8h",
        "fadd v4.4s, v4.4s, v5.4s",
        "faddp v4.4s, v4.4s, v4.4s",
        "faddp s4, v4.2s",
        "fmov {bits:w}, s4",
        x = inout(reg) x => _,
        y = inout(reg) y => _,
        n = inout(reg) n => _,
        bits = lateout(reg) bits,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v16") _, out("v17") _, out("v18") _, out("v19") _,
        options(nostack, readonly),
    );
    f32::from_bits(bits)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f16_f32_neon(a: &[f32], b: &[u16], n: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let halves = vreinterpret_f16_u16(vld1_u16(b.as_ptr().add(i)));
        acc = vfmaq_f32(acc, vld1q_f32(a.as_ptr().add(i)), vcvt_f32_f16(halves));
        i += 4;
    }
    let mut sum = vaddvq_f32(acc);
    while i < n {
        sum += a[i] * f16_to_f32(b[i]);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
unsafe fn dot_f16_f32_avx2(a: &[f32], b_f16: &[u16], n: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let hb = _mm_loadu_si128(b_f16.as_ptr().add(i) as *const __m128i);
        let vb = _mm256_cvtph_ps(hb);
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    let mut sum = hsum_ps(acc);
    while i < n {
        sum += a[i] * f16_to_f32(b_f16[i]);
        i += 1;
    }
    sum
}

pub fn vec_mad_f16_f32(y: &mut [f32], x_f16: &[u16], v: f32) {
    debug_assert_eq!(y.len(), x_f16.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() && has_f16c() {
            unsafe {
                vec_mad_f16_f32_avx2(y, x_f16, v);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            unsafe {
                vec_mad_f16_f32_neon(y, x_f16, v);
            }
            return;
        }
    }
    for i in 0..y.len() {
        y[i] += v * f16_to_f32(x_f16[i]);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vec_mad_f16_f32_neon(y: &mut [f32], x: &[u16], scale: f32) {
    use std::arch::aarch64::*;
    let scale_v = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= y.len() {
        let halves = vreinterpret_f16_u16(vld1_u16(x.as_ptr().add(i)));
        let result = vfmaq_f32(vld1q_f32(y.as_ptr().add(i)), vcvt_f32_f16(halves), scale_v);
        vst1q_f32(y.as_mut_ptr().add(i), result);
        i += 4;
    }
    while i < y.len() {
        y[i] += scale * f16_to_f32(x[i]);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn vec_mad_f16_f32_avx2(y: &mut [f32], x_f16: &[u16], v: f32) {
    use std::arch::x86_64::*;
    let vv = _mm256_set1_ps(v);
    let n = y.len();
    let mut i = 0;
    while i + 8 <= n {
        let yi = _mm256_loadu_ps(y.as_ptr().add(i));
        let hx = _mm_loadu_si128(x_f16.as_ptr().add(i) as *const __m128i);
        let xf = _mm256_cvtph_ps(hx);
        _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_fmadd_ps(vv, xf, yi));
        i += 8;
    }
    if i + 4 <= n {
        let vv128 = _mm256_castps256_ps128(vv);
        let yi = _mm_loadu_ps(y.as_ptr().add(i));
        let hx = _mm_loadl_epi64(x_f16.as_ptr().add(i) as *const __m128i);
        let xf = _mm_cvtph_ps(hx);
        _mm_storeu_ps(y.as_mut_ptr().add(i), _mm_fmadd_ps(vv128, xf, yi));
        i += 4;
    }
    while i < n {
        y[i] += v * f16_to_f32(x_f16[i]);
        i += 1;
    }
}

#[inline(always)]
pub fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    #[cfg(feature = "parity-trace")]
    {
        let mut sum = 0.0f64;
        for i in 0..n {
            sum += f64::from(a[i] * b[i]);
        }
        return sum as f32;
    }
    #[cfg(not(feature = "parity-trace"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2_fma() {
                return unsafe { dot_f32_avx2(a, b, n) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if has_neon() {
                return unsafe { dot_f32_neon(a, b, n) };
            }
        }
        dot_f32_scalar(a, b, n)
    }
}

#[inline(always)]
pub fn vec_scale_f32(y: &mut [f32], v: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe {
                vec_scale_f32_avx2(y, v);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            unsafe {
                vec_scale_f32_neon(y, v);
            }
            return;
        }
    }
    for y_i in y.iter_mut() {
        *y_i *= v;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vec_scale_f32_neon(y: &mut [f32], scale: f32) {
    use std::arch::aarch64::*;
    let scale_v = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= y.len() {
        vst1q_f32(
            y.as_mut_ptr().add(i),
            vmulq_f32(vld1q_f32(y.as_ptr().add(i)), scale_v),
        );
        i += 4;
    }
    while i < y.len() {
        y[i] *= scale;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn vec_scale_f32_avx2(y: &mut [f32], v: f32) {
    use std::arch::x86_64::*;
    let vv = _mm256_set1_ps(v);
    let n = y.len();
    let mut i = 0;
    while i + 8 <= n {
        let yi = _mm256_loadu_ps(y.as_ptr().add(i));
        _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_mul_ps(yi, vv));
        i += 8;
    }
    if i + 4 <= n {
        let vv128 = _mm256_castps256_ps128(vv);
        let yi = _mm_loadu_ps(y.as_ptr().add(i));
        _mm_storeu_ps(y.as_mut_ptr().add(i), _mm_mul_ps(yi, vv128));
        i += 4;
    }
    while i < n {
        y[i] *= v;
        i += 1;
    }
}

#[inline(always)]

#[inline]
#[cfg(target_arch = "x86_64")]
pub unsafe fn hsum_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(s128);
    let s2 = _mm_add_ps(s128, shuf);
    let s3 = _mm_movehl_ps(shuf, s2);
    _mm_cvtss_f32(_mm_add_ss(s2, s3))
}
pub fn vec_mad_f32(y: &mut [f32], x: &[f32], v: f32) {
    debug_assert_eq!(y.len(), x.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe {
                vec_mad_f32_avx2(y, x, v);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            unsafe {
                vec_mad_f32_neon(y, x, v);
            }
            return;
        }
    }
    let vv = v;
    for i in 0..y.len() {
        y[i] += vv * x[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vec_mad_f32_neon(y: &mut [f32], x: &[f32], scale: f32) {
    use std::arch::aarch64::*;
    let scale_v = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= y.len() {
        let result = vfmaq_f32(
            vld1q_f32(y.as_ptr().add(i)),
            vld1q_f32(x.as_ptr().add(i)),
            scale_v,
        );
        vst1q_f32(y.as_mut_ptr().add(i), result);
        i += 4;
    }
    while i < y.len() {
        y[i] += x[i] * scale;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn vec_mad_f32_avx2(y: &mut [f32], x: &[f32], v: f32) {
    use std::arch::x86_64::*;
    let vv = _mm256_set1_ps(v);
    let n = y.len();
    let mut i = 0;
    while i + 8 <= n {
        let yi = _mm256_loadu_ps(y.as_ptr().add(i));
        let xi = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_fmadd_ps(vv, xi, yi));
        i += 8;
    }
    if i + 4 <= n {
        let vv128 = _mm256_castps256_ps128(vv);
        let yi = _mm_loadu_ps(y.as_ptr().add(i));
        let xi = _mm_loadu_ps(x.as_ptr().add(i));
        _mm_storeu_ps(y.as_mut_ptr().add(i), _mm_fmadd_ps(vv128, xi, yi));
        i += 4;
    }
    while i < n {
        y[i] += v * x[i];
        i += 1;
    }
}
