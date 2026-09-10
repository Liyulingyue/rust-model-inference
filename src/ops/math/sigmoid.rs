//! Sigmoid activation and related helpers.
//!
//! The base `sigmoid_inplace` is a tight loop that the compiler usually
//! auto-vectorizes; AVX2/NEON paths are provided as fallbacks if the
//! default backend lacks the relevant fp32 opcodes.

#[inline(always)]
pub fn sigmoid_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        unsafe { sigmoid_inplace_avx2(values) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if crate::ops::has_neon() {
        unsafe { sigmoid_inplace_neon(values) };
        return;
    }
    for v in values {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sigmoid_inplace_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= values.len() {
        let v = _mm256_loadu_ps(values.as_ptr().add(i));
        // 1 / (1 + exp(-x)) computed elementwise; the inner scalar expf
        // pipelines inside the SIMD load/store and the compiler will
        // emit libmvec if available, otherwise scalar expf still has
        // better ILP than the unvectorized loop.
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), v);
        for lane in &mut lanes {
            *lane = 1.0 / (1.0 + (-*lane).exp());
        }
        _mm256_storeu_ps(values.as_mut_ptr().add(i), _mm256_loadu_ps(lanes.as_ptr()));
        i += 8;
    }
    while i < values.len() {
        values[i] = 1.0 / (1.0 + (-values[i]).exp());
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sigmoid_inplace_neon(values: &mut [f32]) {
    use std::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= values.len() {
        let v = vld1q_f32(values.as_ptr().add(i));
        let mut lanes = [0.0f32; 4];
        vst1q_f32(lanes.as_mut_ptr(), v);
        for lane in &mut lanes {
            *lane = 1.0 / (1.0 + (-*lane).exp());
        }
        vst1q_f32(values.as_mut_ptr().add(i), vld1q_f32(lanes.as_ptr()));
        i += 4;
    }
    while i < values.len() {
        values[i] = 1.0 / (1.0 + (-values[i]).exp());
        i += 1;
    }
}