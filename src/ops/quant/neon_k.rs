//! ARM NEON dot-product kernels for IQ4_NL, Q4_K, Q5_K and Q6_K × Q8_K.

#![cfg(target_arch = "aarch64")]

use crate::ops::f16_to_f32;
use crate::ops::quant::{BlockQ8K, BLOCK_Q4K_SIZE, BLOCK_Q5K_SIZE, BLOCK_Q6K_SIZE, KVALUES_IQ4NL};

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
        // Build [m0,m0, m1,m1, ..., m7,m7] from [m0..m7]: vzip1 duplicates
        // the low 4 elements, vzip2 the high 4.
        let mins_dup_lo = vzip1_u8(mins_v, mins_v); // [m0,m0,m1,m1,m2,m2,m3,m3]
        let mins_dup_hi = vzip2_u8(mins_v, mins_v); // [m4,m4,m5,m5,m6,m6,m7,m7]
        let mins_dup16 = vcombine_u8(mins_dup_lo, mins_dup_hi); // 16 bytes
        let mins_i16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins_dup16))); // 8 x i16
        let mins_i16_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins_dup16))); // 8 x i16
        let bsums_v = vld1q_s16(q8k[i].bsums.as_ptr());
        let bsums_v_hi = vld1q_s16(q8k[i].bsums.as_ptr().add(8));
        let min_prod = vmull_s16(vget_low_s16(mins_i16), vget_low_s16(bsums_v));
        let min_prod2 = vmull_s16(vget_high_s16(mins_i16), vget_high_s16(bsums_v));
        let min_prod3 = vmull_s16(vget_low_s16(mins_i16_hi), vget_low_s16(bsums_v_hi));
        let min_prod4 = vmull_s16(vget_high_s16(mins_i16_hi), vget_high_s16(bsums_v_hi));
        let min_sum_lo = vaddvq_s32(min_prod);
        let min_sum_hi = vaddvq_s32(min_prod2);
        let min_sum_lo2 = vaddvq_s32(min_prod3);
        let min_sum_hi2 = vaddvq_s32(min_prod4);
        let min_sum = min_sum_lo + min_sum_hi + min_sum_lo2 + min_sum_hi2;
        // ggml accumulates the two terms into the running total as two
        // separate operations, in this order:
        //
        //     sumf -= dmin * vaddvq_s32(prod);
        //     ...
        //     sumf += d * (sumi1 + sumi2);
        //
        // Folding them into one `total += main + min` is algebraically
        // the same but rounds differently, and a one-ULP difference in
        // the SSM branch's out projection is enough to change the
        // quantized output.
        total -= dmin * min_sum as f32;

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
        let main_sum = vaddvq_s32(prod_lo) + vaddvq_s32(prod_hi);
        total += d * main_sum as f32;
    }

    total
}

/// The dot product of one 32-element Q5_K group.
///
/// The group index is a const generic so the high-bit extraction compiles to
/// an immediate shift, and so the even/odd nibble choice folds away.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q5_group_dot<const G: i32>(
    qh: *const u8,
    chunk: *const u8,
    act: *const i8,
    nibble_mask: std::arch::aarch64::uint8x16_t,
    zero32: std::arch::aarch64::int32x4_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;

    let nib0 = vld1q_u8(chunk);
    let nib1 = vld1q_u8(chunk.add(16));
    let (nib0, nib1) = if G & 1 == 0 {
        (vandq_u8(nib0, nibble_mask), vandq_u8(nib1, nibble_mask))
    } else {
        (vshrq_n_u8::<4>(nib0), vshrq_n_u8::<4>(nib1))
    };
    // The high bit of the group lives in bit `G` of qh. Comparing against a
    // broadcast mask keeps the extraction to a constant number of
    // instructions: the immediate-shift form cannot express a shift of zero.
    let bit = vdupq_n_u8(1u8 << G);
    let high0 = vceqq_u8(vandq_u8(vld1q_u8(qh), bit), bit);
    let high1 = vceqq_u8(vandq_u8(vld1q_u8(qh.add(16)), bit), bit);
    let weight0 = vorrq_u8(nib0, vandq_u8(high0, vdupq_n_u8(0x10)));
    let weight1 = vorrq_u8(nib1, vandq_u8(high1, vdupq_n_u8(0x10)));
    let act0 = vld1q_s8(act.add(G as usize * 32));
    let act1 = vld1q_s8(act.add(G as usize * 32 + 16));
    let dot0 = vdotq_s32(zero32, vreinterpretq_s8_u8(weight0), act0);
    let dot1 = vdotq_s32(zero32, vreinterpretq_s8_u8(weight1), act1);
    vaddq_s32(dot0, dot1)
}

/// ARM NEON dot-product kernel for Q5_K × Q8_K.
///
/// The scalar reference expands a block into 256 bytes and then walks them
/// one at a time, so every element costs a branch, a nibble split and a
/// multiply. This kernel keeps the block packed and dequantizes one
/// 32-element group at a time: the packed nibbles of groups 2c and 2c+1
/// share one 32-byte chunk, and the high bits are one 32-byte read.
#[target_feature(enable = "neon,dotprod")]
pub(crate) unsafe fn vec_dot_q5k_q8k_neon(q5k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    use std::arch::aarch64::*;

    let nb = q8k.len();
    let nibble_mask = vdupq_n_u8(0x0F);
    let zero32 = vdupq_n_s32(0);

    // Four accumulators, one per position class inside a 32-element group.
    // Two integer accumulators keep the eight groups off one dependency
    // chain.
    let mut sums = vdupq_n_f32(0.0);
    let mut correction = 0.0f32;

    for i in 0..nb {
        let boff = i * BLOCK_Q5K_SIZE;
        if boff + BLOCK_Q5K_SIZE > q5k_data.len() {
            break;
        }
        let d = f16_to_f32(u16::from_le_bytes([q5k_data[boff], q5k_data[boff + 1]])) * q8k[i].d;
        let dmin =
            -f16_to_f32(u16::from_le_bytes([q5k_data[boff + 2], q5k_data[boff + 3]])) * q8k[i].d;

        // Eight scales and eight minimums, six bits each, packed in 12 bytes.
        // Each minimum covers the 32 activations of its group.
        let packed = &q5k_data[boff + 4..boff + 16];
        let mut scales = [0i32; 8];
        let mut mins = [0u8; 8];
        for j in 0..8 {
            if j < 4 {
                scales[j] = i32::from(packed[j] & 63);
                mins[j] = packed[j + 4] & 63;
            } else {
                scales[j] = i32::from((packed[j + 4] & 0x0F) | ((packed[j - 4] >> 6) << 4));
                mins[j] = (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
            }
        }

        let qh = q5k_data.as_ptr().add(boff + 16);
        let qs = q5k_data.as_ptr().add(boff + 48);
        let act = q8k[i].qs.as_ptr();
        let mut acc_even = zero32;
        let mut acc_odd = zero32;
        // Group `g` reads the packed nibbles of chunk `g / 2`, the low nibble
        // when `g` is even, and the high bit of qh at bit `g`.
        let chunk0 = qs;
        let chunk1 = qs.add(32);
        let chunk2 = qs.add(64);
        let chunk3 = qs.add(96);
        acc_even = vmlaq_n_s32(
            acc_even,
            q5_group_dot::<0>(qh, chunk0, act, nibble_mask, zero32),
            scales[0],
        );
        acc_odd = vmlaq_n_s32(
            acc_odd,
            q5_group_dot::<1>(qh, chunk0, act, nibble_mask, zero32),
            scales[1],
        );
        acc_even = vmlaq_n_s32(
            acc_even,
            q5_group_dot::<2>(qh, chunk1, act, nibble_mask, zero32),
            scales[2],
        );
        acc_odd = vmlaq_n_s32(
            acc_odd,
            q5_group_dot::<3>(qh, chunk1, act, nibble_mask, zero32),
            scales[3],
        );
        acc_even = vmlaq_n_s32(
            acc_even,
            q5_group_dot::<4>(qh, chunk2, act, nibble_mask, zero32),
            scales[4],
        );
        acc_odd = vmlaq_n_s32(
            acc_odd,
            q5_group_dot::<5>(qh, chunk2, act, nibble_mask, zero32),
            scales[5],
        );
        acc_even = vmlaq_n_s32(
            acc_even,
            q5_group_dot::<6>(qh, chunk3, act, nibble_mask, zero32),
            scales[6],
        );
        acc_odd = vmlaq_n_s32(
            acc_odd,
            q5_group_dot::<7>(qh, chunk3, act, nibble_mask, zero32),
            scales[7],
        );
        sums = vfmaq_n_f32(sums, vcvtq_f32_s32(vaddq_s32(acc_even, acc_odd)), d);

        // The minimum correction: the Q8_K block carries the activation sums
        // of its 16 half-groups, and each minimum covers two of them.
        let mins_wide = vmovl_u8(vld1_u8(mins.as_ptr()));
        let mins_lo = vreinterpretq_s16_u16(vzip1q_u16(mins_wide, mins_wide));
        let mins_hi = vreinterpretq_s16_u16(vzip2q_u16(mins_wide, mins_wide));
        let bsums_lo = vld1q_s16(q8k[i].bsums.as_ptr());
        let bsums_hi = vld1q_s16(q8k[i].bsums.as_ptr().add(8));
        let prod0 = vmlal_s16(zero32, vget_low_s16(mins_lo), vget_low_s16(bsums_lo));
        let prod1 = vmlal_s16(zero32, vget_high_s16(mins_lo), vget_high_s16(bsums_lo));
        let prod2 = vmlal_s16(zero32, vget_low_s16(mins_hi), vget_low_s16(bsums_hi));
        let prod3 = vmlal_s16(zero32, vget_high_s16(mins_hi), vget_high_s16(bsums_hi));
        let sumi = vaddvq_s32(vaddq_s32(prod0, prod1)) + vaddvq_s32(vaddq_s32(prod2, prod3));
        correction += dmin * sumi as f32;
    }

    correction + vaddvq_f32(sums)
}

/// The weights of one 32-element Q6_K group, as two signed byte vectors.
///
/// Group `G` takes the low or the high nibble of one 64-byte `ql` chunk and
/// two bits of the 32-byte `qh` chunk. The two bits are the field at bit
/// `2 * G` of every `qh` byte, and the value is the six-bit code minus 32.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn q6_group_weights<const G: usize>(
    ql: *const u8,
    qh: *const u8,
    nibble_mask: std::arch::aarch64::uint8x16_t,
    high_mask: std::arch::aarch64::uint8x16_t,
    shift: std::arch::aarch64::uint8x16_t,
) -> (std::arch::aarch64::int8x16_t, std::arch::aarch64::int8x16_t) {
    use std::arch::aarch64::*;

    let ql_offset = if G == 1 || G == 3 { 32 } else { 0 };
    let high_nibble = G == 2 || G == 3;
    let ql0 = vld1q_u8(ql.add(ql_offset));
    let ql1 = vld1q_u8(ql.add(ql_offset + 16));
    let qh0 = vld1q_u8(qh);
    let qh1 = vld1q_u8(qh.add(16));
    let (nib0, nib1) = if high_nibble {
        (vshrq_n_u8::<4>(ql0), vshrq_n_u8::<4>(ql1))
    } else {
        (vandq_u8(ql0, nibble_mask), vandq_u8(ql1, nibble_mask))
    };
    // A negative shift amount turns the variable left shift into a right
    // shift, and the immediate-shift form cannot express a shift of zero.
    let back = vnegq_s8(vreinterpretq_s8_u8(shift));
    let high0 = vshlq_n_u8(vandq_u8(vshlq_u8(qh0, back), high_mask), 4);
    let high1 = vshlq_n_u8(vandq_u8(vshlq_u8(qh1, back), high_mask), 4);
    // The unsigned subtraction wraps into the signed six-bit range.
    (
        vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(nib0, high0), vdupq_n_u8(32))),
        vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(nib1, high1), vdupq_n_u8(32))),
    )
}

/// Adds one scaled 16-element half-group into the eight integer lanes.
///
/// Lane `l` takes the products of elements `l` and `l + 8`. That is the
/// pairing the scalar reference uses, and Q6_K is compared bit for bit, so
/// the integer grouping has to match even though any regrouping would give
/// the same real sum.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn q6_half_group(
    weight: std::arch::aarch64::int8x16_t,
    act: std::arch::aarch64::int8x16_t,
    scale: i32,
    acc_lo: std::arch::aarch64::int32x4_t,
    acc_hi: std::arch::aarch64::int32x4_t,
) -> (std::arch::aarch64::int32x4_t, std::arch::aarch64::int32x4_t) {
    use std::arch::aarch64::*;

    let mut prod = vmlal_s8(vdupq_n_s16(0), vget_low_s8(weight), vget_low_s8(act));
    prod = vmlal_high_s8(prod, weight, act);
    (
        vmlal_n_s16(acc_lo, vget_low_s16(prod), scale as i16),
        vmlal_high_n_s16(acc_hi, prod, scale as i16),
    )
}

/// Bit-for-bit NEON kernel for Q6_K × Q8_K.
///
/// Reproduces the scalar reference reduction exactly: the `{l, l + 8}` lane
/// pairing, the per-block `d * aux` rounding, and the sequential eight-lane
/// final sum. Slower than the dotprod path, but the only version that the
/// Vulkan self-check can compare against with zero tolerance. Used in tests
/// and as a verification reference; production calls
/// [`vec_dot_q6k_q8k_neon`], which is mathematically equivalent but
/// reorganizes the integer grouping for dotprod throughput.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vec_dot_q6k_q8k_neon_bits(q6k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    use std::arch::aarch64::*;

    let nb = q8k.len();
    let nibble_mask = vdupq_n_u8(0x0F);
    let high_mask = vdupq_n_u8(0x03);

    // Eight integer lanes split across two vectors; the last four live in
    // `sums_hi` so the final sum can be walked in lane order.
    let mut sums_lo = vdupq_n_f32(0.0);
    let mut sums_hi = vdupq_n_f32(0.0);

    for i in 0..nb {
        let boff = i * BLOCK_Q6K_SIZE;
        if boff + BLOCK_Q6K_SIZE > q6k_data.len() {
            break;
        }
        let d = f16_to_f32(u16::from_le_bytes([
            q6k_data[boff + 208],
            q6k_data[boff + 209],
        ])) * q8k[i].d;
        // One signed scale per 16 elements.
        let scales = &q6k_data[boff + 192..boff + 208];
        let ql = q6k_data.as_ptr().add(boff);
        let qh = q6k_data.as_ptr().add(boff + 128);
        let act = q8k[i].qs.as_ptr();

        let mut acc_lo = vdupq_n_s32(0);
        let mut acc_hi = vdupq_n_s32(0);
        for chunk in 0..2usize {
            let ql_chunk = ql.add(chunk * 64);
            let qh_chunk = qh.add(chunk * 32);
            let act_chunk = act.add(chunk * 128);
            let base = chunk * 8;
            macro_rules! group {
                ($g:literal, $shift:literal) => {{
                    let (low, high) = q6_group_weights::<$g>(
                        ql_chunk,
                        qh_chunk,
                        nibble_mask,
                        high_mask,
                        vdupq_n_u8($shift),
                    );
                    let group_act = act_chunk.add($g * 32);
                    (acc_lo, acc_hi) = q6_half_group(
                        low,
                        vld1q_s8(group_act),
                        i32::from(scales[base + $g * 2] as i8),
                        acc_lo,
                        acc_hi,
                    );
                    (acc_lo, acc_hi) = q6_half_group(
                        high,
                        vld1q_s8(group_act.add(16)),
                        i32::from(scales[base + $g * 2 + 1] as i8),
                        acc_lo,
                        acc_hi,
                    );
                }};
            }
            group!(0, 0);
            group!(1, 2);
            group!(2, 4);
            group!(3, 6);
        }
        // The scalar reference multiplies by `d` and adds per lane; keep the
        // two roundings separate instead of fusing them into an FMA.
        sums_lo = vaddq_f32(sums_lo, vmulq_n_f32(vcvtq_f32_s32(acc_lo), d));
        sums_hi = vaddq_f32(sums_hi, vmulq_n_f32(vcvtq_f32_s32(acc_hi), d));
    }

    // Sequential, lane 0 through lane 7, like `for l in 0..8 { sumf += sums[l] }`.
    let lanes = [
        vgetq_lane_f32::<0>(sums_lo),
        vgetq_lane_f32::<1>(sums_lo),
        vgetq_lane_f32::<2>(sums_lo),
        vgetq_lane_f32::<3>(sums_lo),
        vgetq_lane_f32::<0>(sums_hi),
        vgetq_lane_f32::<1>(sums_hi),
        vgetq_lane_f32::<2>(sums_hi),
        vgetq_lane_f32::<3>(sums_hi),
    ];
    let mut sumf = 0.0f32;
    for lane in lanes {
        sumf += lane;
    }
    sumf
}

/// Fast NEON dotprod kernel for Q6_K × Q8_K.
///
/// Mathematically equivalent to [`vec_dot_q6k_q8k_neon_bits`] but uses
/// `vdotq_s32` with four consecutive elements per lane instead of the
/// scalar reference's `{l, l + 8}` pairing. The real sum is identical; only
/// the intermediate integer grouping differs. This is the production kernel
/// because it runs ~40 % faster (32 vs 23 GINTOP/s on GB10). The Vulkan
/// self-check tolerates the regrouping because Q6_K's GPU kernel uses the
/// same `vdotq_s32` instruction, and both sit well inside the 3e-3 tolerance
/// that k-quant formats allow.
#[target_feature(enable = "neon,dotprod")]
pub(crate) unsafe fn vec_dot_q6k_q8k_neon(q6k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    use std::arch::aarch64::*;

    let nb = q8k.len();
    let nibble_mask = vdupq_n_u8(0x0F);
    let high_mask = vdupq_n_u8(0x03);
    let zero32 = vdupq_n_s32(0);

    let mut sums = vdupq_n_f32(0.0);

    for i in 0..nb {
        let boff = i * BLOCK_Q6K_SIZE;
        if boff + BLOCK_Q6K_SIZE > q6k_data.len() {
            break;
        }
        let d = f16_to_f32(u16::from_le_bytes([
            q6k_data[boff + 208],
            q6k_data[boff + 209],
        ])) * q8k[i].d;
        let scales = &q6k_data[boff + 192..boff + 208];
        let ql = q6k_data.as_ptr().add(boff);
        let qh = q6k_data.as_ptr().add(boff + 128);
        let act = q8k[i].qs.as_ptr();

        let mut acc_even = zero32;
        let mut acc_odd = zero32;
        for chunk in 0..2usize {
            let ql_chunk = ql.add(chunk * 64);
            let qh_chunk = qh.add(chunk * 32);
            let act_chunk = act.add(chunk * 128);
            let base = chunk * 8;

            macro_rules! group {
                ($g:literal, $shift:literal) => {{
                    let ql_offset = if $g == 1 || $g == 3 { 32 } else { 0 };
                    let high_nibble = $g == 2 || $g == 3;
                    let ql0 = vld1q_u8(ql_chunk.add(ql_offset));
                    let ql1 = vld1q_u8(ql_chunk.add(ql_offset + 16));
                    let qh0 = vld1q_u8(qh_chunk);
                    let qh1 = vld1q_u8(qh_chunk.add(16));
                    let (nib0, nib1) = if high_nibble {
                        (vshrq_n_u8::<4>(ql0), vshrq_n_u8::<4>(ql1))
                    } else {
                        (vandq_u8(ql0, nibble_mask), vandq_u8(ql1, nibble_mask))
                    };
                    let back = vnegq_s8(vreinterpretq_s8_u8(vdupq_n_u8($shift)));
                    let high0 = vshlq_n_u8(vandq_u8(vshlq_u8(qh0, back), high_mask), 4);
                    let high1 = vshlq_n_u8(vandq_u8(vshlq_u8(qh1, back), high_mask), 4);
                    let weight0 =
                        vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(nib0, high0), vdupq_n_u8(32)));
                    let weight1 =
                        vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(nib1, high1), vdupq_n_u8(32)));
                    let dot0 = vdotq_s32(zero32, weight0, vld1q_s8(act_chunk.add($g * 32)));
                    let dot1 = vdotq_s32(zero32, weight1, vld1q_s8(act_chunk.add($g * 32 + 16)));
                    if $g % 2 == 0 {
                        acc_even = vaddq_s32(
                            acc_even,
                            vaddq_s32(
                                vmulq_n_s32(dot0, i32::from(scales[base + $g * 2] as i8)),
                                vmulq_n_s32(dot1, i32::from(scales[base + $g * 2 + 1] as i8)),
                            ),
                        );
                    } else {
                        acc_odd = vaddq_s32(
                            acc_odd,
                            vaddq_s32(
                                vmulq_n_s32(dot0, i32::from(scales[base + $g * 2] as i8)),
                                vmulq_n_s32(dot1, i32::from(scales[base + $g * 2 + 1] as i8)),
                            ),
                        );
                    }
                }};
            }
            group!(0, 0);
            group!(1, 2);
            group!(2, 4);
            group!(3, 6);
        }
        sums = vfmaq_n_f32(sums, vcvtq_f32_s32(vaddq_s32(acc_even, acc_odd)), d);
    }

    vaddvq_f32(sums)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::ops::quant::{
        vec_dot_q4k_q8k_scalar, vec_dot_q5k_q8k_scalar, vec_dot_q6k_q8k, vec_dot_q6k_q8k_scalar,
        BlockQ8K, BLOCK_Q4K_SIZE, BLOCK_Q5K_SIZE, BLOCK_Q6K_SIZE,
    };

    #[test]
    fn q6k_neon_matches_scalar_signed_scales_and_block_boundaries() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let mut bytes = Vec::new();
        let mut activations = Vec::new();
        for seed in 0..9usize {
            let mut block = vec![0u8; BLOCK_Q6K_SIZE];
            for (index, value) in block[..208].iter_mut().enumerate() {
                *value = (index * 131 + seed * 53) as u8;
            }
            block[208..]
                .copy_from_slice(&crate::ops::f32_to_f16(0.125 + seed as f32 * 0.25).to_le_bytes());
            bytes.extend(block);
            let qs = std::array::from_fn(|index| (index * 193 + seed * 29) as u8 as i8);
            let bsums = std::array::from_fn(|group| {
                qs[group * 16..group * 16 + 16]
                    .iter()
                    .map(|&v| v as i16)
                    .sum()
            });
            activations.push(BlockQ8K {
                d: 0.03125,
                qs,
                bsums,
            });
        }
        for blocks in [0, 1, 2, 9] {
            let input = &activations[..blocks];
            for length in [blocks * BLOCK_Q6K_SIZE, blocks * BLOCK_Q6K_SIZE + 1] {
                let length = length.min(bytes.len());
                let weights = &bytes[..length];
                let expected = vec_dot_q6k_q8k_scalar(weights, input);
                let neon = unsafe { vec_dot_q6k_q8k_neon(weights, input) };
                let dispatched = vec_dot_q6k_q8k(weights, input);
                let tolerance = 2e-5 * expected.abs().max(1.0);
                assert!(
                    (neon - expected).abs() <= tolerance,
                    "blocks={blocks}: {neon} != {expected}"
                );
                assert!((dispatched - expected).abs() <= tolerance);
            }
        }
        let weights = &bytes[..bytes.len() - 1];
        let expected = vec_dot_q6k_q8k_scalar(weights, &activations);
        let actual = unsafe { vec_dot_q6k_q8k_neon(weights, &activations) };
        assert!((actual - expected).abs() <= 2e-5 * expected.abs().max(1.0));
    }

    /// Deterministic Q4_K block fixture: random-ish but reproducible
    /// weights with known scales/mins, so a NEON vs scalar mismatch
    /// shows up as a non-zero relative diff.
    fn make_block(seed: u8) -> Vec<u8> {
        let mut v = vec![0u8; BLOCK_Q4K_SIZE];
        // d=f16(0.7), dmin=f16(0.05)
        v[0..2].copy_from_slice(&crate::ops::f32_to_f16(0.7).to_le_bytes());
        v[2..4].copy_from_slice(&crate::ops::f32_to_f16(0.05).to_le_bytes());
        // 12 bytes packed scales/mins: use a deterministic pattern that
        // exercises the 6-bit unpacking.
        for i in 0..12 {
            v[4 + i] = (seed.wrapping_mul(7).wrapping_add(i as u8 * 3)) & 0x7f;
        }
        // 128 bytes of nibbles
        for i in 0..128 {
            v[16 + i] = (seed.wrapping_mul(13).wrapping_add(i as u8 * 5)) & 0xff;
        }
        v
    }

    fn make_q8k(seed: u8) -> BlockQ8K {
        let d = 0.6 + seed as f32 * 0.01;
        let mut qs = [0i8; 256];
        for (i, q) in qs.iter_mut().enumerate() {
            let raw = (seed as usize * 31 + i * 17) % 255;
            *q = (raw as i16 - 127) as i8;
        }
        let mut bsums = [0i16; 16];
        for (i, b) in bsums.iter_mut().enumerate() {
            let raw = (seed as usize * 13 + i * 5) % 100;
            *b = (raw as i16 - 50) as i16;
        }
        BlockQ8K { d, qs, bsums }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q4k_neon_matches_scalar_single_block() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        for seed in 0..8u8 {
            let block = make_block(seed);
            let q8k = [make_q8k(seed)];
            let neon = unsafe { vec_dot_q4k_q8k_neon(&block, &q8k) };
            let scalar = vec_dot_q4k_q8k_scalar(&block, &q8k);
            let diff = (neon - scalar).abs();
            let rel = if scalar.abs() > 1e-3 {
                diff / scalar.abs()
            } else {
                diff
            };
            assert!(
                rel < 1e-3,
                "seed={seed}: neon={neon} scalar={scalar} rel={rel}"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q4k_neon_matches_scalar_multi_block() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let mut block = Vec::new();
        let mut q8k = Vec::new();
        for seed in 0..4u8 {
            block.extend(make_block(seed));
            q8k.push(make_q8k(seed));
        }
        let neon = unsafe { vec_dot_q4k_q8k_neon(&block, &q8k) };
        let scalar = vec_dot_q4k_q8k_scalar(&block, &q8k);
        let diff = (neon - scalar).abs();
        let rel = if scalar.abs() > 1e-3 {
            diff / scalar.abs()
        } else {
            diff
        };
        assert!(
            rel < 1e-3,
            "multi-block: neon={neon} scalar={scalar} rel={rel}"
        );
    }

    /// Deterministic Q5_K block fixture. The packed nibbles, the high bits
    /// and the six-bit scales and minimums all have to be exercised, so the
    /// fixture uses a pattern that puts a different bit in every group.
    fn make_q5k_block(seed: u8) -> Vec<u8> {
        let mut v = vec![0u8; BLOCK_Q5K_SIZE];
        v[0..2].copy_from_slice(&crate::ops::f32_to_f16(0.7).to_le_bytes());
        v[2..4].copy_from_slice(&crate::ops::f32_to_f16(0.05).to_le_bytes());
        // 12 bytes packed scales/mins: the six-bit unpacking has to be
        // exercised on both the low and the high halves.
        for i in 0..12 {
            v[4 + i] = (seed.wrapping_mul(7).wrapping_add(i as u8 * 3)) & 0xff;
        }
        // 32 bytes of high bits: one bit per group.
        for i in 0..32 {
            v[16 + i] = (seed.wrapping_mul(11).wrapping_add(i as u8 * 5)) & 0xff;
        }
        // 128 bytes of packed nibbles.
        for i in 0..128 {
            v[48 + i] = (seed.wrapping_mul(13).wrapping_add(i as u8 * 7)) & 0xff;
        }
        v
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q5k_neon_matches_scalar_single_block() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        for seed in 0..8u8 {
            let block = make_q5k_block(seed);
            let q8k = [make_q8k(seed)];
            let neon = unsafe { vec_dot_q5k_q8k_neon(&block, &q8k) };
            let scalar = vec_dot_q5k_q8k_scalar(&block, &q8k);
            let diff = (neon - scalar).abs();
            let rel = if scalar.abs() > 1e-3 {
                diff / scalar.abs()
            } else {
                diff
            };
            assert!(
                rel < 1e-3,
                "seed={seed}: neon={neon} scalar={scalar} rel={rel}"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q5k_neon_matches_scalar_multi_block() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let mut block = Vec::new();
        let mut q8k = Vec::new();
        for seed in 0..4u8 {
            block.extend(make_q5k_block(seed));
            q8k.push(make_q8k(seed));
        }
        let neon = unsafe { vec_dot_q5k_q8k_neon(&block, &q8k) };
        let scalar = vec_dot_q5k_q8k_scalar(&block, &q8k);
        let diff = (neon - scalar).abs();
        let rel = if scalar.abs() > 1e-3 {
            diff / scalar.abs()
        } else {
            diff
        };
        assert!(
            rel < 1e-3,
            "multi-block: neon={neon} scalar={scalar} rel={rel}"
        );
    }

    /// Deterministic Q6_K block fixture: 128 bytes of packed nibbles, 64
    /// bytes of two-bit high parts, 16 signed scales and the f16 delta.
    fn make_q6k_block(seed: u8) -> Vec<u8> {
        let mut v = vec![0u8; BLOCK_Q6K_SIZE];
        for i in 0..128 {
            v[i] = (seed.wrapping_mul(7).wrapping_add(i as u8 * 3)) & 0xff;
        }
        for i in 0..64 {
            v[128 + i] = (seed.wrapping_mul(11).wrapping_add(i as u8 * 5)) & 0xff;
        }
        for i in 0..16 {
            v[192 + i] = (seed.wrapping_mul(13).wrapping_add(i as u8 * 7)) as i8 as u8;
        }
        v[208..210].copy_from_slice(&crate::ops::f32_to_f16(0.9).to_le_bytes());
        v
    }

    /// The production dotprod kernel is not bit-identical to scalar, but the
    /// difference is well inside k-quant noise. These tests pin the fast path
    /// that `vec_dot_q6k_q8k()` actually dispatches to.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q6k_neon_dotprod_close_single_block() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        for seed in 0..8u8 {
            let block = make_q6k_block(seed);
            let q8k = [make_q8k(seed)];
            let neon = unsafe { vec_dot_q6k_q8k_neon(&block, &q8k) };
            let scalar = vec_dot_q6k_q8k_scalar(&block, &q8k);
            let diff = (neon - scalar).abs();
            let rel = if scalar.abs() > 1e-3 {
                diff / scalar.abs()
            } else {
                diff
            };
            assert!(
                rel < 3e-3,
                "seed={seed}: dotprod={neon} scalar={scalar} rel={rel}"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q6k_neon_dotprod_close_multi_block() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let mut block = Vec::new();
        let mut q8k = Vec::new();
        for seed in 0..4u8 {
            block.extend(make_q6k_block(seed));
            q8k.push(make_q8k(seed));
        }
        let neon = unsafe { vec_dot_q6k_q8k_neon(&block, &q8k) };
        let scalar = vec_dot_q6k_q8k_scalar(&block, &q8k);
        let diff = (neon - scalar).abs();
        let rel = if scalar.abs() > 1e-3 {
            diff / scalar.abs()
        } else {
            diff
        };
        assert!(
            rel < 3e-3,
            "multi-block: dotprod={neon} scalar={scalar} rel={rel}"
        );
    }

    /// The bit-exact vmull kernel reproduces the scalar reference reduction
    /// lane-for-lane. This is the verification reference, not the production
    /// kernel; `vec_dot_q6k_q8k_neon` (dotprod) is faster but groups integers
    /// differently, so this test pins the path that is truly bit-identical.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q6k_neon_bits_match_scalar() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let mut block = Vec::new();
        let mut q8k = Vec::new();
        for seed in 0..8u8 {
            block.extend(make_q6k_block(seed));
            q8k.push(make_q8k(seed));
        }
        let neon = unsafe { vec_dot_q6k_q8k_neon_bits(&block, &q8k) };
        let scalar = vec_dot_q6k_q8k_scalar(&block, &q8k);
        assert_eq!(
            neon.to_bits(),
            scalar.to_bits(),
            "neon_bits={neon} scalar={scalar}"
        );
    }

    /// The production dotprod kernel reorganizes the integer grouping for
    /// throughput. It is mathematically equivalent to the scalar reference but
    /// not bit-identical; this test pins the tolerance the Vulkan self-check
    /// uses (3e-3, matching Q4_K/Q5_K).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q6k_neon_dotprod_close_to_scalar() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let mut block = Vec::new();
        let mut q8k = Vec::new();
        for seed in 0..8u8 {
            block.extend(make_q6k_block(seed));
            q8k.push(make_q8k(seed));
        }
        let neon = unsafe { vec_dot_q6k_q8k_neon(&block, &q8k) };
        let scalar = vec_dot_q6k_q8k_scalar(&block, &q8k);
        let diff = (neon - scalar).abs();
        let rel = if scalar.abs() > 1e-3 {
            diff / scalar.abs()
        } else {
            diff
        };
        assert!(rel < 3e-3, "dotprod={neon} scalar={scalar} rel={rel}");
    }
}
