//! Q8_0 quantization kernels.

pub fn quantize_q8_0_into(input: &[f32], n: usize, q8: &mut [u8], scales: &mut [f32]) {
    let blocks = n / 32;
    #[cfg(target_arch = "x86_64")]
    {
        if super::super::has_avx2_fma() {
            unsafe {
                quantize_q8_0_into_avx2(input, n, q8, scales);
                return;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if super::super::has_neon() {
            unsafe {
                quantize_q8_0_into_neon_range(input, n, q8, scales, 0, blocks);
                return;
            }
        }
    }
    quantize_q8_0_into_scalar_range(input, n, q8, scales, 0, blocks);
}

pub fn quantize_q8_0_into_parallel(
    input: &[f32],
    n: usize,
    q8: &mut [u8],
    scales: &mut [f32],
    ith: usize,
    nth: usize,
) {
    let blocks = n / 32;
    let block_start = ith * blocks / nth;
    let block_end = (ith + 1) * blocks / nth;
    #[cfg(target_arch = "x86_64")]
    {
        if super::super::has_avx2_fma() {
            unsafe {
                quantize_q8_0_into_avx2_range(input, q8, scales, block_start, block_end);
                return;
            }
        }
    }
    quantize_q8_0_into_scalar_range(input, n, q8, scales, block_start, block_end);
}

pub(crate) fn quantize_q8_0_into_scalar_range(
    input: &[f32],
    n: usize,
    q8: &mut [u8],
    scales: &mut [f32],
    block_start: usize,
    block_end: usize,
) {
    for block in block_start..block_end {
        let values = &input[block * 32..(block + 1) * 32];
        let amax = values
            .iter()
            .fold(0.0f32, |current, value| current.max(value.abs()));
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        scales[block] = super::super::f16_to_f32(super::super::f32_to_f16(scale));
        for lane in 0..32 {
            q8[block * 32 + lane] =
                (values[lane] * inverse).round().clamp(-128.0, 127.0) as i8 as u8;
        }
    }
    let _ = n;
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quantize_q8_0_into_neon_range(
    input: &[f32],
    n: usize,
    q8: &mut [u8],
    scales: &mut [f32],
    block_start: usize,
    block_end: usize,
) {
    use std::arch::aarch64::*;
    let _ = n;
    for b in block_start..block_end {
        let base = b * 32;
        let chunk = vld1q_f32(input.as_ptr().add(base));
        let chunk2 = vld1q_f32(input.as_ptr().add(base + 4));
        let chunk3 = vld1q_f32(input.as_ptr().add(base + 8));
        let chunk4 = vld1q_f32(input.as_ptr().add(base + 12));
        let chunk5 = vld1q_f32(input.as_ptr().add(base + 16));
        let chunk6 = vld1q_f32(input.as_ptr().add(base + 20));
        let chunk7 = vld1q_f32(input.as_ptr().add(base + 24));
        let chunk8 = vld1q_f32(input.as_ptr().add(base + 28));
        let max_abs = vmaxq_f32(
            vmaxq_f32(vmaxq_f32(vabsq_f32(chunk), vabsq_f32(chunk2)), vmaxq_f32(vabsq_f32(chunk3), vabsq_f32(chunk4))),
            vmaxq_f32(vmaxq_f32(vabsq_f32(chunk5), vabsq_f32(chunk6)), vmaxq_f32(vabsq_f32(chunk7), vabsq_f32(chunk8))),
        );
        let max_scalar = vmaxvq_f32(max_abs);
        let scale = max_scalar / 127.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        scales[b] = super::super::f16_to_f32(super::super::f32_to_f16(scale));
        let v_inv = vdupq_n_f32(inv_scale);
        let q1 = vminq_f32(vmaxq_f32(vmulq_f32(chunk, v_inv), vdupq_n_f32(-127.0)), vdupq_n_f32(127.0));
        let q2 = vminq_f32(vmaxq_f32(vmulq_f32(chunk2, v_inv), vdupq_n_f32(-127.0)), vdupq_n_f32(127.0));
        let q3 = vminq_f32(vmaxq_f32(vmulq_f32(chunk3, v_inv), vdupq_n_f32(-127.0)), vdupq_n_f32(127.0));
        let q4 = vminq_f32(vmaxq_f32(vmulq_f32(chunk4, v_inv), vdupq_n_f32(-127.0)), vdupq_n_f32(127.0));
        let q5 = vminq_f32(vmaxq_f32(vmulq_f32(chunk5, v_inv), vdupq_n_f32(-127.0)), vdupq_n_f32(127.0));
        let q6 = vminq_f32(vmaxq_f32(vmulq_f32(chunk6, v_inv), vdupq_n_f32(-127.0)), vdupq_n_f32(127.0));
        let q7 = vminq_f32(vmaxq_f32(vmulq_f32(chunk7, v_inv), vdupq_n_f32(-127.0)), vdupq_n_f32(127.0));
        let q8 = vminq_f32(vmaxq_f32(vmulq_f32(chunk8, v_inv), vdupq_n_f32(-127.0)), vdupq_n_f32(127.0));
        let i1 = vcvtq_s32_f32(q1);
        let i2 = vcvtq_s32_f32(q2);
        let i3 = vcvtq_s32_f32(q3);
        let i4 = vcvtq_s32_f32(q4);
        let i5 = vcvtq_s32_f32(q5);
        let i6 = vcvtq_s32_f32(q6);
        let i7 = vcvtq_s32_f32(q7);
        let i8 = vcvtq_s32_f32(q8);
        let n1 = vminq_s32(vmaxq_s32(i1, vdupq_n_s32(-127)), vdupq_n_s32(127));
        let n2 = vminq_s32(vmaxq_s32(i2, vdupq_n_s32(-127)), vdupq_n_s32(127));
        let n3 = vminq_s32(vmaxq_s32(i3, vdupq_n_s32(-127)), vdupq_n_s32(127));
        let n4 = vminq_s32(vmaxq_s32(i4, vdupq_n_s32(-127)), vdupq_n_s32(127));
        let n5 = vminq_s32(vmaxq_s32(i5, vdupq_n_s32(-127)), vdupq_n_s32(127));
        let n6 = vminq_s32(vmaxq_s32(i6, vdupq_n_s32(-127)), vdupq_n_s32(127));
        let n7 = vminq_s32(vmaxq_s32(i7, vdupq_n_s32(-127)), vdupq_n_s32(127));
        let n8 = vminq_s32(vmaxq_s32(i8, vdupq_n_s32(-127)), vdupq_n_s32(127));
        let b1 = vreinterpretq_u8_s8(n1);
        let b2 = vreinterpretq_u8_s8(n2);
        let b3 = vreinterpretq_u8_s8(n3);
        let b4 = vreinterpretq_u8_s8(n4);
        let b5 = vreinterpretq_u8_s8(n5);
        let b6 = vreinterpretq_u8_s8(n6);
        let b7 = vreinterpretq_u8_s8(n7);
        let b8 = vreinterpretq_u8_s8(n8);
        vst1q_u8(q8.as_mut_ptr().add(base), b1);
        vst1q_u8(q8.as_mut_ptr().add(base + 4), b2);
        vst1q_u8(q8.as_mut_ptr().add(base + 8), b3);
        vst1q_u8(q8.as_mut_ptr().add(base + 12), b4);
        vst1q_u8(q8.as_mut_ptr().add(base + 16), b5);
        vst1q_u8(q8.as_mut_ptr().add(base + 20), b6);
        vst1q_u8(q8.as_mut_ptr().add(base + 24), b7);
        vst1q_u8(q8.as_mut_ptr().add(base + 28), b8);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn quantize_q8_0_into_avx2(
    input: &[f32],
    n: usize,
    q8: &mut [u8],
    scales: &mut [f32],
) {
    let blocks = n / 32;
    quantize_q8_0_into_avx2_range(input, q8, scales, 0, blocks);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn quantize_q8_0_into_avx2_range(
    input: &[f32],
    q8: &mut [u8],
    scales: &mut [f32],
    b_start: usize,
    b_end: usize,
) {
    use std::arch::x86_64::*;
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let max_i8 = _mm256_set1_ps(127.0);
    let min_i8 = _mm256_set1_ps(-128.0);
    for b in b_start..b_end {
        let ptr = input.as_ptr().add(b * 32);
        let v0 = _mm256_loadu_ps(ptr);
        let v1 = _mm256_loadu_ps(ptr.add(8));
        let v2 = _mm256_loadu_ps(ptr.add(16));
        let v3 = _mm256_loadu_ps(ptr.add(24));
        let a0 = _mm256_andnot_ps(sign_mask, v0);
        let a1 = _mm256_andnot_ps(sign_mask, v1);
        let a2 = _mm256_andnot_ps(sign_mask, v2);
        let a3 = _mm256_andnot_ps(sign_mask, v3);
        let m01 = _mm256_max_ps(a0, a1);
        let m23 = _mm256_max_ps(a2, a3);
        let m0123 = _mm256_max_ps(m01, m23);
        let hi = _mm256_extractf128_ps(m0123, 1);
        let lo = _mm256_castps256_ps128(m0123);
        let m128 = _mm_max_ps(hi, lo);
        let shuf = _mm_movehdup_ps(m128);
        let m2 = _mm_max_ps(m128, shuf);
        let m3 = _mm_movehl_ps(shuf, m2);
        let amax = _mm_cvtss_f32(_mm_max_ss(m2, m3));
        let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let id = if amax == 0.0 { 0.0 } else { 127.0 / amax };
        scales[b] = super::super::f16_to_f32(super::super::f32_to_f16(d));
        let id_v = _mm256_set1_ps(id);
        let r0 = _mm256_round_ps(_mm256_mul_ps(v0, id_v), _MM_FROUND_TO_NEAREST_INT);
        let r1 = _mm256_round_ps(_mm256_mul_ps(v1, id_v), _MM_FROUND_TO_NEAREST_INT);
        let r2 = _mm256_round_ps(_mm256_mul_ps(v2, id_v), _MM_FROUND_TO_NEAREST_INT);
        let r3 = _mm256_round_ps(_mm256_mul_ps(v3, id_v), _MM_FROUND_TO_NEAREST_INT);
        let c0 = _mm256_min_ps(_mm256_max_ps(r0, min_i8), max_i8);
        let c1 = _mm256_min_ps(_mm256_max_ps(r1, min_i8), max_i8);
        let c2 = _mm256_min_ps(_mm256_max_ps(r2, min_i8), max_i8);
        let c3 = _mm256_min_ps(_mm256_max_ps(r3, min_i8), max_i8);
        let i0 = _mm256_cvtps_epi32(c0);
        let i1 = _mm256_cvtps_epi32(c1);
        let i2 = _mm256_cvtps_epi32(c2);
        let i3 = _mm256_cvtps_epi32(c3);
        let p01 = _mm256_packs_epi32(i0, i1);
        let p23 = _mm256_packs_epi32(i2, i3);
        let packed = _mm256_packs_epi16(p01, p23);
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let fixed = _mm256_permutevar8x32_epi32(packed, perm);
        _mm256_storeu_si256(q8.as_mut_ptr().add(b * 32) as *mut __m256i, fixed);
    }
}

pub fn quantize_q8_0(input: &[f32], n: usize) -> (Vec<u8>, Vec<f32>) {
    let blocks = n / 32;
    let mut q8 = vec![0u8; n];
    let mut scales = vec![0.0f32; blocks];
    quantize_q8_0_into(input, n, &mut q8, &mut scales);
    (q8, scales)
}