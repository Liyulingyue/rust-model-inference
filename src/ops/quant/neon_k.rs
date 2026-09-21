//! AArch64 dot-product kernel for IQ4_NL weights and Q8_K activations.

#![cfg(target_arch = "aarch64")]

use crate::ops::quant::{BlockQ8K, KVALUES_IQ4NL};

#[target_feature(enable = "neon,dotprod")]
pub(crate) unsafe fn vec_dot_iq4_nl_q8k_neon(iq4nl_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    use std::arch::aarch64::*;

    let table = vld1q_s8(KVALUES_IQ4NL.as_ptr());
    let mask = vdupq_n_u8(0x0f);
    let mut sum = 0.0f32;
    for (block_index, activation) in q8k.iter().enumerate() {
        let super_offset = block_index * 8 * 18;
        if super_offset + 8 * 18 > iq4nl_data.len() {
            break;
        }
        for subblock in 0..8 {
            let offset = super_offset + subblock * 18;
            let d = crate::ops::f16_to_f32(u16::from_le_bytes([
                iq4nl_data[offset],
                iq4nl_data[offset + 1],
            ])) * activation.d;
            let packed = vld1q_u8(iq4nl_data.as_ptr().add(offset + 2));
            let low = vqtbl1q_s8(table, vandq_u8(packed, mask));
            let high = vqtbl1q_s8(table, vshrq_n_u8(packed, 4));
            let input = activation.qs.as_ptr().add(subblock * 32);
            let dot = vdotq_s32(vdupq_n_s32(0), low, vld1q_s8(input));
            let dot = vdotq_s32(dot, high, vld1q_s8(input.add(16)));
            sum += d * vaddvq_s32(dot) as f32;
        }
    }
    sum
}
