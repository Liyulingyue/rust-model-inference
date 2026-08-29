use super::super::f16_to_f32;
use super::super::has_avx2_fma;
use super::super::has_f16c;
use super::super::has_neon;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::asm;

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
pub(crate) unsafe fn dot_f32_neon(a: &[f32], b: &[f32], n: usize) -> f32 {
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
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() && has_f16c() && n >= 16 {
            return unsafe { dot_f16_f16_bytes_avx2(a, b, n) };
        }
    }
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

/// AVX2 + F16C + FMA implementation of `dot_f16_f16_bytes`.
///
/// Both inputs are stored as `u16` (a) and packed 2-byte little-endian (b),
/// but `b` is read into 128-bit lanes via `_mm_loadu_si128` (which never
/// faults — it just reads raw bytes) and then converted to FP32 via
/// `_mm256_cvtph_ps` (F16C). Each iteration processes 8 elements.
#[cfg(target_arch = "x86_64")]
unsafe fn dot_f16_f16_bytes_avx2(a: &[u16], b: &[u8], n: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_cvtph_ps(_mm_loadu_si128(a.as_ptr().add(i) as *const __m128i));
        let vb = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i * 2) as *const __m128i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    let mut sum = hsum_ps(acc);
    // 4-element tail with `_mm_cvtph_ps` (SSE conversion of 4 F16).
    if i + 4 <= n {
        let va = _mm_cvtph_ps(_mm_loadl_epi64(a.as_ptr().add(i) as *const __m128i));
        let vb = _mm_cvtph_ps(_mm_loadl_epi64(b.as_ptr().add(i * 2) as *const __m128i));
        let v = _mm_fmadd_ps(va, vb, _mm_setzero_ps());
        let tail = _mm_hsum_ps_4(v);
        // extract the single F32 from the lowest lane of the 128-bit register
        let t = std::mem::transmute::<__m128, [f32; 4]>(tail);
        sum += t[0];
        i += 4;
    }
    while i < n {
        let weight = u16::from_le_bytes(b[i * 2..i * 2 + 2].try_into().unwrap());
        sum += f16_to_f32(a[i]) * f16_to_f32(weight);
        i += 1;
    }
    sum
}

// Helper: horizontal sum of 4 lanes in a 128-bit __m128 (requires SSE3).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn _mm_hsum_ps_4(v: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;
    let shuf = _mm_movehdup_ps(v);
    let sums = _mm_add_ps(v, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_add_ps(sums, shuf2)
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

// Horizontal sum of an __m256 (8x f32 lanes).
//
// TODO(minicpm): the current reduction order is `_mm_add_ps(v_hi, v_lo) ->
// _mm_add_ps(_, movehl) -> _mm_add_ss(_, movehdup)`. For MiniCPM-Small (Q8_0
// matmul kernels) the same total is reproduced bit-exactly by:
//   res = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
//   res = _mm_add_ps(res, _mm_movehl_ps(res, res));
//   res = _mm_add_ss(res, _mm_movehdup_ps(res));
//   _mm_cvtss_f32(res)
// which matches llama.cpp's `hsum_float_8` (the same pattern, no extra
// rounds).  If a future model needs a different rounding mode, e.g. Kahan
// pairwise summation, add it here behind a config flag and keep the current
// fast path as the default.
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

/// y[i] = y[i] + y[i] * x[i]  (fused self-mul-add).
///
/// 算子语义：`y += y * x`。两个操作数都是向量（y 既是累加器又是被乘数）。
/// AVX2/FMA 走 `_mm256_fmadd_ps(y, x, y)`：累加器先读后写，FMA 硬件语义保证正确。
/// NEON 走 `vfmaq_f32(y, y, x)`。
/// 不走 `vec_mad_f32(y, x, v)`，因为 `v` 只能是标量 broadcast，无法表达向量被乘数。
pub fn vec_mad_self_f32(y: &mut [f32], x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe {
                vec_mad_self_f32_avx2(y, x);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            unsafe {
                vec_mad_self_f32_neon(y, x);
            }
            return;
        }
    }
    for i in 0..y.len() {
        y[i] += y[i] * x[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vec_mad_self_f32_avx2(y: &mut [f32], x: &[f32]) {
    use std::arch::x86_64::*;
    let n = y.len();
    let mut i = 0;
    while i + 8 <= n {
        let yi = _mm256_loadu_ps(y.as_ptr().add(i));
        let xi = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_fmadd_ps(yi, xi, yi));
        i += 8;
    }
    if i + 4 <= n {
        let yi = _mm_loadu_ps(y.as_ptr().add(i));
        let xi = _mm_loadu_ps(x.as_ptr().add(i));
        _mm_storeu_ps(y.as_mut_ptr().add(i), _mm_fmadd_ps(yi, xi, yi));
        i += 4;
    }
    while i < n {
        y[i] += y[i] * x[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vec_mad_self_f32_neon(y: &mut [f32], x: &[f32]) {
    use std::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= y.len() {
        let yi = vld1q_f32(y.as_ptr().add(i));
        let xi = vld1q_f32(x.as_ptr().add(i));
        vst1q_f32(y.as_mut_ptr().add(i), vfmaq_f32(yi, yi, xi));
        i += 4;
    }
    while i < y.len() {
        y[i] += y[i] * x[i];
        i += 1;
    }
}

/// `sum_f32(values) = Σ values[i]`，返回 f64 保证累加精度。
/// 通用 reduce op：与 `sum_sq_f32` 配对，`qwen35/vision.rs` 用它构造
/// `Σ(x - mean)² = sum_sq - 2·mean·sum + n·mean²`。
/// 实测：AVX2 上 ~4-5× 标量 f64 reduce。
pub fn sum_f32(values: &[f32]) -> f64 {
    let n = values.len();
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            return unsafe { sum_f32_avx2(values) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            return unsafe { sum_f32_neon(values) };
        }
    }
    let mut acc = 0.0f64;
    for &value in values {
        acc += f64::from(value);
    }
    acc
}

/// `sum_sq_centered_f32(x, mean) = Σ (xᵢ - mean)²`，返回 f64。
///
/// **为什么有这个函数**：直接做 `(xᵢ - mean)²` per-element（每元素 bounded 误差）
/// 然后求和，比代数恒等式 `Σx² - n·mean²` **无条件数值稳定**——后者在
/// `mean² >> Var` 时会有灾难性 cancellation（位损失随 `mean²/Var` 放大）。
///
/// 实测对比（`mean=1000, Var=1`, n=576）：
/// - 代数恒等式：f64 位损失 ~20 bit（worst case 可放大 1e6 倍）
/// - per-element：f64 位损失 ≤3 bit，与 `mean/Var` 大小无关
///
/// AVX2 实测：~4-5× 标量 f64 reduce（单 pass：broadcast mean → sub → square → 累加到 f64）。
pub fn sum_sq_centered_f32(values: &[f32], mean: f32) -> f64 {
    let n = values.len();
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            return unsafe { sum_sq_centered_f32_avx2(values, mean) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            return unsafe { sum_sq_centered_f32_neon(values, mean) };
        }
    }
    let mut acc = 0.0f64;
    for &value in values {
        let d = value - mean;
        acc += f64::from(d * d);
    }
    acc
}

/// `sum_sq_f32(values) = Σ values[i]²`，返回 f64 保证 reduce 精度。
/// 通用 reduce op：被 `rms_norm` / `rms_norm_inplace` 用于 mean_sq = sum_sq / n，
/// 也可被 layer norm、variance computation、L2 norm 等复用。
/// 实测：AVX2 上 ~4-5× 标量 f64 reduce（@1024 denoise 节省 ~20s）。
/// 精度策略：每 8 f32 squares 立即 promote 到 4 f64 doubles 再 hsum 到 1 f64，
/// 不在 f32 lane 内 partial sum（hsum_ps 会引入 ~3 ULP 误差 → PNG 不一致）。
pub fn sum_sq_f32(values: &[f32]) -> f64 {
    let n = values.len();
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            return unsafe { sum_sq_f32_avx2(values) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            return unsafe { sum_sq_f32_neon(values) };
        }
    }
    let mut acc = 0.0f64;
    for &value in values {
        acc += f64::from(value * value);
    }
    acc
}

/// `__m256d` (4 f64) 横向归约到 1 个 f64。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_pd_256(v: std::arch::x86_64::__m256d) -> f64 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_pd(v, 1);
    let lo = _mm256_castpd256_pd128(v);
    let sum2 = _mm_add_pd(lo, hi);
    let shuf = _mm_shuffle_pd(sum2, sum2, 0x1);
    let sum1 = _mm_add_sd(sum2, shuf);
    _mm_cvtsd_f64(sum1)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn widen_f32x4_to_f64_pairs(
    values: std::arch::aarch64::float32x4_t,
) -> (
    std::arch::aarch64::float64x2_t,
    std::arch::aarch64::float64x2_t,
) {
    use std::arch::aarch64::*;
    (
        vcvt_f64_f32(vget_low_f32(values)),
        vcvt_f64_f32(vget_high_f32(values)),
    )
}

/// AVX2 reduce：`sum_sq = Σ values[i]²` as f64（bit-exact 与标量参考一致）。
/// 流程：每 8 元素做 `x*x`（f32 lane 内），立即 `_mm256_cvtps_pd` promote
/// 到 4 f64 doubles（low + high half），再 hsum 到 1 f64 加到 accumulator。
/// @1024 denoise 比标量 ~4-5× 加速。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sum_sq_f32_avx2(values: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let n = values.len();
    let n8 = n / 8 * 8;
    let mut acc = 0.0f64;
    let mut i = 0;
    while i + 8 <= n8 {
        let v = _mm256_loadu_ps(values.as_ptr().add(i));
        let sq = _mm256_mul_ps(v, v);
        let lo = _mm256_cvtps_pd(_mm256_castps256_ps128(sq));
        let hi = _mm256_cvtps_pd(_mm256_extractf128_ps(sq, 1));
        acc += hsum_pd_256(lo) + hsum_pd_256(hi);
        i += 8;
    }
    let mut tail = 0.0f64;
    while i < n {
        let v = values[i];
        tail += f64::from(v * v);
        i += 1;
    }
    acc + tail
}

/// NEON reduce：`sum_sq = Σ values[i]²` as f64（bit-exact 与标量参考一致）。
/// 流程：每 4 元素做 `x*x`（f32），立即 `vcvt_f32_f64` promote 到 f64（逐 lane），
/// 再 `vfmaq_f64` f64 FMA 累加到 `__m128d` accumulator，末尾 `vaddvq_f64` 归约。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_sq_f32_neon(values: &[f32]) -> f64 {
    use std::arch::aarch64::*;
    let n = values.len();
    let n4 = n / 4 * 4;
    let mut acc_low = vdupq_n_f64(0.0);
    let mut acc_high = vdupq_n_f64(0.0);
    let mut i = 0;
    while i + 16 <= n4 {
        let v0 = vld1q_f32(values.as_ptr().add(i));
        let v1 = vld1q_f32(values.as_ptr().add(i + 4));
        let v2 = vld1q_f32(values.as_ptr().add(i + 8));
        let v3 = vld1q_f32(values.as_ptr().add(i + 12));
        let (s0_low, s0_high) = widen_f32x4_to_f64_pairs(vmulq_f32(v0, v0));
        let (s1_low, s1_high) = widen_f32x4_to_f64_pairs(vmulq_f32(v1, v1));
        let (s2_low, s2_high) = widen_f32x4_to_f64_pairs(vmulq_f32(v2, v2));
        let (s3_low, s3_high) = widen_f32x4_to_f64_pairs(vmulq_f32(v3, v3));
        acc_low = vaddq_f64(acc_low, s0_low);
        acc_high = vaddq_f64(acc_high, s0_high);
        acc_low = vaddq_f64(acc_low, s1_low);
        acc_high = vaddq_f64(acc_high, s1_high);
        acc_low = vaddq_f64(acc_low, s2_low);
        acc_high = vaddq_f64(acc_high, s2_high);
        acc_low = vaddq_f64(acc_low, s3_low);
        acc_high = vaddq_f64(acc_high, s3_high);
        i += 16;
    }
    let mut scalar_acc = vaddvq_f64(vaddq_f64(acc_low, acc_high));
    while i + 4 <= n4 {
        let v = vld1q_f32(values.as_ptr().add(i));
        let (low, high) = widen_f32x4_to_f64_pairs(vmulq_f32(v, v));
        scalar_acc += vaddvq_f64(low) + vaddvq_f64(high);
        i += 4;
    }
    let mut tail = 0.0f64;
    while i < n {
        let v = values[i];
        tail += f64::from(v * v);
        i += 1;
    }
    scalar_acc + tail
}

/// AVX2 reduce：`sum = Σ values[i]` as f64。
/// 流程：每 8 元素 promote 到 4 f64 doubles（low + high half），
/// 再 hsum 到 1 f64 加到 accumulator。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sum_f32_avx2(values: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let n = values.len();
    let n8 = n / 8 * 8;
    let mut acc = 0.0f64;
    let mut i = 0;
    while i + 8 <= n8 {
        let v = _mm256_loadu_ps(values.as_ptr().add(i));
        let lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));
        let hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
        acc += hsum_pd_256(lo) + hsum_pd_256(hi);
        i += 8;
    }
    let mut tail = 0.0f64;
    while i < n {
        acc += f64::from(values[i]);
        i += 1;
    }
    acc
}

/// NEON reduce：`sum = Σ values[i]` as f64。
/// 流程：每 4 元素 promote 到 f64（2 个 f64 per 4 f32），对子对求和，
/// 再 `vaddq_f64` 累加到 `__m128d` accumulator，末尾 `vaddvq_f64` 归约。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_f32_neon(values: &[f32]) -> f64 {
    use std::arch::aarch64::*;
    let n = values.len();
    let n4 = n / 4 * 4;
    let mut acc_low = vdupq_n_f64(0.0);
    let mut acc_high = vdupq_n_f64(0.0);
    let mut i = 0;
    while i + 16 <= n4 {
        let v0 = vld1q_f32(values.as_ptr().add(i));
        let v1 = vld1q_f32(values.as_ptr().add(i + 4));
        let v2 = vld1q_f32(values.as_ptr().add(i + 8));
        let v3 = vld1q_f32(values.as_ptr().add(i + 12));
        let (s0_low, s0_high) = widen_f32x4_to_f64_pairs(v0);
        let (s1_low, s1_high) = widen_f32x4_to_f64_pairs(v1);
        let (s2_low, s2_high) = widen_f32x4_to_f64_pairs(v2);
        let (s3_low, s3_high) = widen_f32x4_to_f64_pairs(v3);
        acc_low = vaddq_f64(acc_low, vaddq_f64(s0_low, s1_low));
        acc_high = vaddq_f64(acc_high, vaddq_f64(s0_high, s1_high));
        acc_low = vaddq_f64(acc_low, vaddq_f64(s2_low, s3_low));
        acc_high = vaddq_f64(acc_high, vaddq_f64(s2_high, s3_high));
        i += 16;
    }
    let mut scalar_acc = vaddvq_f64(vaddq_f64(acc_low, acc_high));
    while i + 4 <= n4 {
        let v = vld1q_f32(values.as_ptr().add(i));
        let (low, high) = widen_f32x4_to_f64_pairs(v);
        scalar_acc += vaddvq_f64(low) + vaddvq_f64(high);
        i += 4;
    }
    let mut tail = 0.0f64;
    while i < n {
        scalar_acc += f64::from(values[i]);
        i += 1;
    }
    scalar_acc
}

/// AVX2 reduce：`sum_sq_centered = Σ (xᵢ - mean)²` as f64（bit-exact）。
/// 流程：每 8 元素 broadcast mean，`_mm256_sub_ps` 减，`_mm256_mul_ps` 平方，
/// 立即 `_mm256_cvtps_pd` promote 到 4 f64 doubles（low + high half），
/// hsum 到 1 f64 加到 accumulator。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sum_sq_centered_f32_avx2(values: &[f32], mean: f32) -> f64 {
    use std::arch::x86_64::*;
    let n = values.len();
    let n8 = n / 8 * 8;
    let vmean = _mm256_set1_ps(mean);
    let mut acc = 0.0f64;
    let mut i = 0;
    while i + 8 <= n8 {
        let v = _mm256_loadu_ps(values.as_ptr().add(i));
        let d = _mm256_sub_ps(v, vmean);
        let sq = _mm256_mul_ps(d, d);
        let lo = _mm256_cvtps_pd(_mm256_castps256_ps128(sq));
        let hi = _mm256_cvtps_pd(_mm256_extractf128_ps(sq, 1));
        acc += hsum_pd_256(lo) + hsum_pd_256(hi);
        i += 8;
    }
    let mut tail = 0.0f64;
    while i < n {
        let d = values[i] - mean;
        tail += f64::from(d * d);
        i += 1;
    }
    acc + tail
}

/// NEON reduce：`sum_sq_centered = Σ (xᵢ - mean)²` as f64（bit-exact）。
/// 流程：每 4 元素 broadcast mean，`vsubq_f32` 减，`vmulq_f32` 平方，
/// `vcvt_f32_f64` promote 到 f64 lane，`vfmaq_f64` f64 FMA 累加到 `__m128d` accumulator。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_sq_centered_f32_neon(values: &[f32], mean: f32) -> f64 {
    use std::arch::aarch64::*;
    let n = values.len();
    let n4 = n / 4 * 4;
    let vmean = vdupq_n_f32(mean);
    let mut acc_low = vdupq_n_f64(0.0);
    let mut acc_high = vdupq_n_f64(0.0);
    let mut i = 0;
    while i + 16 <= n4 {
        let v0 = vld1q_f32(values.as_ptr().add(i));
        let v1 = vld1q_f32(values.as_ptr().add(i + 4));
        let v2 = vld1q_f32(values.as_ptr().add(i + 8));
        let v3 = vld1q_f32(values.as_ptr().add(i + 12));
        let d0 = vsubq_f32(v0, vmean);
        let d1 = vsubq_f32(v1, vmean);
        let d2 = vsubq_f32(v2, vmean);
        let d3 = vsubq_f32(v3, vmean);
        let (d0_low, d0_high) = widen_f32x4_to_f64_pairs(vmulq_f32(d0, d0));
        let (d1_low, d1_high) = widen_f32x4_to_f64_pairs(vmulq_f32(d1, d1));
        let (d2_low, d2_high) = widen_f32x4_to_f64_pairs(vmulq_f32(d2, d2));
        let (d3_low, d3_high) = widen_f32x4_to_f64_pairs(vmulq_f32(d3, d3));
        acc_low = vaddq_f64(acc_low, d0_low);
        acc_high = vaddq_f64(acc_high, d0_high);
        acc_low = vaddq_f64(acc_low, d1_low);
        acc_high = vaddq_f64(acc_high, d1_high);
        acc_low = vaddq_f64(acc_low, d2_low);
        acc_high = vaddq_f64(acc_high, d2_high);
        acc_low = vaddq_f64(acc_low, d3_low);
        acc_high = vaddq_f64(acc_high, d3_high);
        i += 16;
    }
    let mut scalar_acc = vaddvq_f64(vaddq_f64(acc_low, acc_high));
    while i + 4 <= n4 {
        let v = vld1q_f32(values.as_ptr().add(i));
        let delta = vsubq_f32(v, vmean);
        let (low, high) = widen_f32x4_to_f64_pairs(vmulq_f32(delta, delta));
        scalar_acc += vaddvq_f64(low) + vaddvq_f64(high);
        i += 4;
    }
    let mut tail = 0.0f64;
    while i < n {
        let d = values[i] - mean;
        tail += f64::from(d * d);
        i += 1;
    }
    scalar_acc + tail
}

#[cfg(test)]
mod tests {
    use super::{sum_f32, sum_sq_centered_f32, sum_sq_f32};

    #[test]
    fn f64_reductions_cover_vector_and_tail_lengths() {
        let values = [
            -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0,
            3.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0,
        ];
        let expected_sum: f64 = values.iter().map(|&value| f64::from(value)).sum();
        let expected_sq: f64 = values
            .iter()
            .map(|&value| f64::from(value * value))
            .sum();
        let expected_centered: f64 = values
            .iter()
            .map(|&value| {
                let delta = value - 1.0;
                f64::from(delta * delta)
            })
            .sum();

        assert_eq!(sum_f32(&values).to_bits(), expected_sum.to_bits());
        assert_eq!(sum_sq_f32(&values).to_bits(), expected_sq.to_bits());
        assert_eq!(
            sum_sq_centered_f32(&values, 1.0).to_bits(),
            expected_centered.to_bits()
        );
    }
}
