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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn softmax_neon_ggml(x: &mut [f32]) {
    use std::arch::aarch64::*;

    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    let mut i = 0;
    while i + 4 <= x.len() {
        let values = ggml_expf_neon(vsubq_f32(vld1q_f32(x.as_ptr().add(i)), vdupq_n_f32(max)));
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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn ggml_expf_neon(
    x: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    let r = vdupq_n_f32(f32::from_bits(0x4b40_0000));
    let z = vfmaq_f32(r, x, vdupq_n_f32(f32::from_bits(0x3fb8_aa3b)));
    let n = vsubq_f32(z, r);
    let b = vfmsq_f32(
        vfmsq_f32(x, n, vdupq_n_f32(f32::from_bits(0x3f31_7200))),
        n,
        vdupq_n_f32(f32::from_bits(0x35bf_be8e)),
    );
    let e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    let k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1.0))));
    let c = vcagtq_f32(n, vdupq_n_f32(126.0));
    let u = vmulq_f32(b, b);
    let j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(f32::from_bits(0x3f7f_fff6)), b),
        vfmaq_f32(
            vfmaq_f32(
                vdupq_n_f32(f32::from_bits(0x3eff_fedb)),
                vdupq_n_f32(f32::from_bits(0x3e2a_af33)),
                b,
            ),
            vfmaq_f32(
                vdupq_n_f32(f32::from_bits(0x3d2b_9f17)),
                vdupq_n_f32(f32::from_bits(0x3c07_2010)),
                b,
            ),
            u,
        ),
        u,
    );
    if vaddvq_u32(c) == 0 {
        return vfmaq_f32(k, j, k);
    }
    let d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x8200_0000));
    let s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f00_0000)));
    let s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    vbslq_f32(
        vcagtq_f32(n, vdupq_n_f32(192.0)),
        vmulq_f32(s1, s1),
        vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)),
    )
}
