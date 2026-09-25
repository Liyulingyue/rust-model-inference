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

        // --- Unpack scales and mins (same bit twiddling as scalar/AVX2) ---
        let sc_base = boff + 4;
        let mut utmp = [0u32; 4];
        utmp[0] = u32::from_le_bytes([
            q4k_data[sc_base],
            q4k_data[sc_base + 1],
            q4k_data[sc_base + 2],
            q4k_data[sc_base + 3],
        ]);
        utmp[1] = u32::from_le_bytes([
            q4k_data[sc_base + 4],
            q4k_data[sc_base + 5],
            q4k_data[sc_base + 6],
            q4k_data[sc_base + 7],
        ]);
        utmp[2] = u32::from_le_bytes([
            q4k_data[sc_base + 8],
            q4k_data[sc_base + 9],
            q4k_data[sc_base + 10],
            q4k_data[sc_base + 11],
        ]);

        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        let uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        // utmp[0..3] bytes: first 8 bytes = scales, next 8 bytes = mins
        // (each utmp contributes 4 bytes; scales come from utmp[0..2] low halves,
        //  mins from utmp[2..3] — but the layout is interleaved per the scalar code)
        // Actually: all_bytes[0..8] = scales, all_bytes[8..16] = mins
        let mut all_bytes = [0u8; 16];
        for k in 0..4 {
            let bytes = utmp[k].to_le_bytes();
            all_bytes[k * 4..k * 4 + 4].copy_from_slice(&bytes);
        }
        let scales = &all_bytes[0..8];
        let mins = &all_bytes[8..16];

        // --- Min correction: sum(mins[j/2] * bsums[j]) for j=0..16 ---
        // bsums is [i16; 16], mins is [u8; 8] -> each min applies to 2 bsums
        let mut sumi_min = 0i32;
        for j in 0..16 {
            sumi_min += q8k[i].bsums[j] as i32 * mins[j / 2] as i32;
        }
        let min_correction = -dmin * sumi_min as f32;

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

        // Apply scales and accumulate
        let mut main_sum = 0.0f32;
        for g in 0..8 {
            main_sum += scales[g] as f32 * group_sums[g] as f32;
        }
        main_sum *= d;

        total += main_sum + min_correction;
    }

    total
}
