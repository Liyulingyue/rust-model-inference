//! ARM NEON SIMD kernels for I-quant × Q8_K matmul.
//!
//! Phase 2.8: NEON equivalents of `avx2_k.rs`. 128-bit SIMD (vs AVX2 256-bit)
//! means each instruction processes half the data per iteration; the algorithmic
//! structure is identical: nibble split → LUT shuffle → i8→i16 widen → hadd/madd.
//!
//! Precision contract: ≤ 1 ULP drift vs scalar oracle (same root cause as AVX2:
//! FMA single rounding vs scalar sequential accumulation).

#![cfg(target_arch = "aarch64")]

use crate::ops::quant::{f16_to_f32, BlockQ8K, KVALUES_IQ4NL};

#[target_feature(enable = "neon")]
pub(crate) unsafe fn vec_dot_iq4_nl_q8k_neon(iq4nl_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    use std::arch::aarch64::*;

    let nb = q8k.len();
    let lut = vld1q_u8(KVALUES_IQ4NL.as_ptr() as *const u8);
    let mask = vdupq_n_u8(0x0f);
    let mut acc_sum = 0.0f32;

    for i in 0..nb {
        let super_off = i * 8 * 18;
        if super_off + 8 * 18 > iq4nl_data.len() {
            break;
        }

        for sb in 0..8usize {
            let boff = super_off + sb * 18;
            let d_raw = u16::from_le_bytes([iq4nl_data[boff], iq4nl_data[boff + 1]]);
            let d = f16_to_f32(d_raw) * q8k[i].d;

            let qs_ptr = iq4nl_data.as_ptr().add(boff + 2);
            let q8_ptr = q8k[i].qs[sb * 32..].as_ptr();

            let qb = vld1q_u8(qs_ptr);
            let q8_lo = vld1q_u8(q8_ptr);
            let q8_hi = vld1q_u8(q8_ptr.add(16));

            let lo_nib = vandq_u8(qb, mask);
            let hi_nib = vshrq_n_u8(
                vandq_u8(vreinterpretq_u8_u16(vshrq_n_u16(
                    vreinterpretq_u16_u8(qb),
                    4,
                ))),
                1,
            );

            let lo_lut_u8 = vqtbl1q_u8(lut, lo_nib);
            let hi_lut_u8 = vqtbl1q_u8(lut, hi_nib);
            let lo_lut_i16: int16x8_t = vmovl_s16(vget_low_s16(vreinterpretq_s16_u8(lo_lut_u8)));
            let hi_lut_i16: int16x8_t = vmovl_s16(vget_low_s16(vreinterpretq_s16_u8(hi_lut_u8)));
            let q8_lo_i16: int16x8_t = vmovl_u8(vget_low_u8(q8_lo));
            let q8_hi_i16: int16x8_t = vmovl_u8(vget_low_u8(q8_hi));

            let p_lo: int32x4_t = vqdmull_s16(lo_lut_i16, q8_lo_i16);
            let p_hi: int32x4_t = vqdmull_s16(hi_lut_i16, q8_hi_i16);
            let dot_i32 = vpaddq_s32(p_lo, p_hi);
            let dot = vaddvq_s32(dot_i32);

            acc_sum += d * dot as f32;
        }
    }

    acc_sum
}
