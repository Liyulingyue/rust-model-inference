//! ARM NEON dot-product kernel for IQ4_NL × Q8_K.

#![cfg(target_arch = "aarch64")]

use crate::ops::f16_to_f32;
use crate::ops::quant::{BlockQ8K, KVALUES_IQ4NL};

#[target_feature(enable = "neon,dotprod")]
pub(crate) unsafe fn vec_dot_iq4_nl_q8k_neon(iq4nl_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    use std::arch::aarch64::*;

    let lut = vld1q_u8(KVALUES_IQ4NL.as_ptr() as *const u8);
    let mask = vdupq_n_u8(0x0f);
    let mut total = 0.0f32;
    for (block_index, q8_block) in q8k.iter().enumerate() {
        let super_off = block_index * 8 * 18;
        if super_off + 8 * 18 > iq4nl_data.len() {
            break;
        }
        let mut super_sum = 0.0f32;
        for subblock in 0..8 {
            let off = super_off + subblock * 18;
            let d =
                f16_to_f32(u16::from_le_bytes([iq4nl_data[off], iq4nl_data[off + 1]])) * q8_block.d;
            let q4 = vld1q_u8(iq4nl_data.as_ptr().add(off + 2));
            let q8 = q8_block.qs.as_ptr().add(subblock * 32);
            let q8_low = vld1q_s8(q8 as *const i8);
            let q8_high = vld1q_s8(q8.add(16) as *const i8);
            let lo = vqtbl1q_u8(lut, vandq_u8(q4, mask));
            let hi = vqtbl1q_u8(lut, vshrq_n_u8(q4, 4));
            let dots = vdotq_s32(
                vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(lo), q8_low),
                vreinterpretq_s8_u8(hi),
                q8_high,
            );
            super_sum += d * vaddvq_s32(dots) as f32;
        }
        total += super_sum;
    }
    total
}
