//! ARM NEON dot-product kernel for IQ4_NL × Q8_K and Q4_K × Q8_K.

#![cfg(target_arch = "aarch64")]

use crate::ops::f16_to_f32;
use crate::ops::quant::{BlockQ8K, BLOCK_Q4K_SIZE, KVALUES_IQ4NL};

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

/// NEON dotprod kernel for Q4_K × Q8_K vec dot.
///
/// Q4_K super-block layout (144 bytes):
///   [0:2]   d (f16)
///   [2:4]   dmin (f16)
///   [4:16]  12-byte packed scales/mins (6-bit each, 8+8)
///   [16:144] 128 bytes of 4-bit nibbles (256 elements)
///
/// The nibbles are split into 4 sub-blocks of 32 bytes each.
/// Each sub-block contains 32 byte-pairs; low nibbles form the first
/// 64 values, high nibbles the next 64. The 8 per-group scales
/// apply to 8 groups of 32 values each.
#[target_feature(enable = "neon,dotprod")]
pub(crate) unsafe fn vec_dot_q4k_q8k_neon(q4k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    use std::arch::aarch64::*;

    let nb = q8k.len();
    let kmask1: u32 = 0x3f3f3f3f;
    let kmask2: u32 = 0x0f0f0f0f;
    let kmask3: u32 = 0x03030303;

    let lo_mask = vdupq_n_u8(0x0F);

    let mut total = 0.0f32;

    for i in 0..nb {
        let boff = i * BLOCK_Q4K_SIZE;
        if boff + BLOCK_Q4K_SIZE > q4k_data.len() {
            break;
        }

        let d_raw = u16::from_le_bytes([q4k_data[boff], q4k_data[boff + 1]]);
        let dmin_raw = u16::from_le_bytes([q4k_data[boff + 2], q4k_data[boff + 3]]);
        let d = f16_to_f32(d_raw) * q8k[i].d;
        let dmin = f16_to_f32(dmin_raw) * q8k[i].d;

        // --- Unpack scales and mins via NEON bit manipulation ---
        let sc_base = boff + 4;
        let raw = vld1q_u8(q4k_data.as_ptr().add(sc_base));
        // We only use 12 bytes; the 4th u32 is padding. Extract three u32 lanes.
        let raw_u32 = vreinterpretq_u32_u8(raw);
        let u0 = vgetq_lane_u32(raw_u32, 0);
        let u1 = vgetq_lane_u32(raw_u32, 1);
        let u2 = vgetq_lane_u32(raw_u32, 2);

        let utmp3 = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4);
        let uaux = u1 & kmask1;
        let utmp1 = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4);
        let utmp2 = uaux;
        let utmp0 = u0 & kmask1;

        // Pack into 16 bytes: [scales(8), mins(8)]
        let packed_lo = vcreate_u64(utmp0 as u64 | ((utmp1 as u64) << 32));
        let packed_hi = vcreate_u64(utmp2 as u64 | ((utmp3 as u64) << 32));
        let packed_u8 = vcombine_u8(
            vreinterpret_u8_u64(packed_lo),
            vreinterpret_u8_u64(packed_hi),
        );
        let scales_v = vget_low_u8(packed_u8); // 8 bytes = scales
        let mins_v = vget_high_u8(packed_u8); // 8 bytes = mins

        // --- Min correction via NEON: sum(mins[j/2] * bsums[j]) ---
        // bsums is [i16; 16], mins is [u8; 8] -> each min applies to 2 bsums.
        // Duplicate each min to pairs: [m0,m0, m1,m1, ... m7,m7]
        let mins_dup = vzip1_u8(mins_v, mins_v); // [m0,m0,m1,m1,...,m7,m7] (8 bytes)
        let mins_dup16 = vcombine_u8(mins_dup, mins_dup); // 16 bytes
        let mins_i16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins_dup16))); // 8 x i16
        let mins_i16_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins_dup16))); // 8 x i16
        let bsums_v = vld1q_s16(q8k[i].bsums.as_ptr());
        let min_prod = vmull_s16(vget_low_s16(mins_i16), vget_low_s16(bsums_v));
        let min_prod2 = vmull_s16(vget_low_s16(mins_i16_hi), vget_high_s16(bsums_v));
        let min_sum_lo = vaddvq_s32(min_prod);
        let min_sum_hi = vaddvq_s32(min_prod2);
        let min_correction = -dmin * (min_sum_lo + min_sum_hi) as f32;

        // --- Main dot product: 8 groups of 32 values ---
        let q4_ptr = q4k_data.as_ptr().add(boff + 16);
        let q8_ptr = q8k[i].qs.as_ptr();

        let mut group_sums = [0i32; 8];

        for j in 0..4 {
            // Load 32 bytes of Q4 nibbles (two 16-byte loads)
            let q4bits_lo = vld1q_u8(q4_ptr.add(j * 32));
            let q4bits_hi = vld1q_u8(q4_ptr.add(j * 32 + 16));

            let q4_l0 = vandq_u8(q4bits_lo, lo_mask);
            let q4_h0 = vshrq_n_u8(q4bits_lo, 4);
            let q4_l1 = vandq_u8(q4bits_hi, lo_mask);
            let q4_h1 = vshrq_n_u8(q4bits_hi, 4);

            let q8_g0 = vld1q_s8(q8_ptr.add(j * 64) as *const i8);
            let q8_g0b = vld1q_s8(q8_ptr.add(j * 64 + 16) as *const i8);
            let q8_g1 = vld1q_s8(q8_ptr.add(j * 64 + 32) as *const i8);
            let q8_g1b = vld1q_s8(q8_ptr.add(j * 64 + 48) as *const i8);

            let dot_g0 = vdotq_s32(
                vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(q4_l0), q8_g0),
                vreinterpretq_s8_u8(q4_l1),
                q8_g0b,
            );
            group_sums[2 * j] = vaddvq_s32(dot_g0);

            let dot_g1 = vdotq_s32(
                vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(q4_h0), q8_g1),
                vreinterpretq_s8_u8(q4_h1),
                q8_g1b,
            );
            group_sums[2 * j + 1] = vaddvq_s32(dot_g1);
        }

        // Apply scales via NEON: sum(scales[g] * group_sums[g]) for g=0..8
        // scales_v is u8x8, group_sums is [i32; 8]
        let scales_wide = vmovl_u8(scales_v); // u16x8
        let scales_lo = vget_low_u16(scales_wide); // u16x4
        let scales_hi = vget_high_u16(scales_wide); // u16x4
        let scales_i32_lo = vreinterpretq_s32_u32(vmovl_u16(scales_lo)); // i32x4
        let scales_i32_hi = vreinterpretq_s32_u32(vmovl_u16(scales_hi)); // i32x4
        let group_lo = vld1q_s32(group_sums.as_ptr()); // i32x4
        let group_hi = vld1q_s32(group_sums.as_ptr().add(4)); // i32x4
        let prod_lo = vmulq_s32(scales_i32_lo, group_lo);
        let prod_hi = vmulq_s32(scales_i32_hi, group_hi);
        let main_sum = d * (vaddvq_s32(prod_lo) + vaddvq_s32(prod_hi)) as f32;

        total += main_sum + min_correction;
    }

    total
}
