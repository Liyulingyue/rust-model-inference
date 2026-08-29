#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let shuf = _mm_movehl_ps(s, s);
    let s2 = _mm_add_ps(s, shuf);
    let s2_arr: [f32; 4] = std::mem::transmute(s2);
    s2_arr[0] + s2_arr[1]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_dot_iq4_xs_q8k_avx2(
    q4xs_data: &[u8],
    q8k: &[super::BlockQ8K],
) -> f32 {
    use std::arch::x86_64::*;

    let nb = q8k.len();
    let m0f = _mm256_set1_epi8(0x0f);
    let lut128 = _mm_loadu_si128(super::KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let mut acc_sum = 0.0f32;

    for i in 0..nb {
        let boff = i * super::BLOCK_IQ4_XS_SIZE;
        if boff + super::BLOCK_IQ4_XS_SIZE > q4xs_data.len() {
            break;
        }
        let d_raw = u16::from_le_bytes([q4xs_data[boff], q4xs_data[boff + 1]]);
        let d = super::f16_to_f32(d_raw) * q8k[i].d;
        let scales_h = u16::from_le_bytes([q4xs_data[boff + 2], q4xs_data[boff + 3]]);
        let scales_l = &q4xs_data[boff + 4..boff + 8];
        let qs = &q4xs_data[boff + 8..boff + 8 + 128];
        let q8 = &q8k[i].qs;
        let mut h = scales_h;
        for pair in 0..4usize {
            let scale_byte = scales_l[pair] as u16;
            let ls1 = ((scale_byte & 0x0f) | ((h << 4) & 0x30)) as i32;
            let ls2 = (((scale_byte >> 4) as u16) | ((h << 2) & 0x30)) as i32;
            h >>= 4;
            let qs_base = pair * 32;
            let q8_base = pair * 64;
            let qb0 = _mm_loadu_si128(qs.as_ptr().add(qs_base) as *const __m128i);
            let qb1 = _mm_loadu_si128(qs.as_ptr().add(qs_base + 16) as *const __m128i);
            let q8_a = _mm_loadu_si128(q8.as_ptr().add(q8_base) as *const __m128i);
            let q8_b = _mm_loadu_si128(q8.as_ptr().add(q8_base + 16) as *const __m128i);
            let q8_c = _mm_loadu_si128(q8.as_ptr().add(q8_base + 32) as *const __m128i);
            let q8_d = _mm_loadu_si128(q8.as_ptr().add(q8_base + 48) as *const __m128i);
            let m0f128 = _mm256_castsi256_si128(m0f);

            let lo_nib0 = _mm_and_si128(qb0, m0f128);
            let hi_nib0 = _mm_and_si128(_mm_srli_epi16(qb0, 4), m0f128);
            let lo_lut0 = _mm_shuffle_epi8(lut128, lo_nib0);
            let hi_lut0 = _mm_shuffle_epi8(lut128, hi_nib0);
            let lo_i16_0 = _mm256_cvtepi8_epi16(lo_lut0);
            let hi_i16_0 = _mm256_cvtepi8_epi16(hi_lut0);
            let q8_a_i16 = _mm256_cvtepi8_epi16(q8_a);
            let q8_b_i16 = _mm256_cvtepi8_epi16(q8_b);
            let p_lo1 = _mm256_madd_epi16(lo_i16_0, q8_a_i16);
            let p_hi1 = _mm256_madd_epi16(hi_i16_0, q8_b_i16);
            let p1 = _mm256_add_epi32(p_lo1, p_hi1);
            let dot1 = hsum_i32(p1);
            let dl1 = d * (ls1 as f32 - 32.0f32);
            acc_sum += dl1 * (dot1 as f32);

            let lo_nib1 = _mm_and_si128(qb1, m0f128);
            let hi_nib1 = _mm_and_si128(_mm_srli_epi16(qb1, 4), m0f128);
            let lo_lut1 = _mm_shuffle_epi8(lut128, lo_nib1);
            let hi_lut1 = _mm_shuffle_epi8(lut128, hi_nib1);
            let lo_i16_1 = _mm256_cvtepi8_epi16(lo_lut1);
            let hi_i16_1 = _mm256_cvtepi8_epi16(hi_lut1);
            let q8_c_i16 = _mm256_cvtepi8_epi16(q8_c);
            let q8_d_i16 = _mm256_cvtepi8_epi16(q8_d);
            let p_lo2 = _mm256_madd_epi16(lo_i16_1, q8_c_i16);
            let p_hi2 = _mm256_madd_epi16(hi_i16_1, q8_d_i16);
            let p2 = _mm256_add_epi32(p_lo2, p_hi2);
            let dot2 = hsum_i32(p2);
            let dl2 = d * (ls2 as f32 - 32.0f32);
            acc_sum += dl2 * (dot2 as f32);
        }
    }

    acc_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_dot_q2k_q8k_avx2(
    q2k_data: &[u8],
    q8k: &[super::BlockQ8K],
) -> f32 {
    use std::arch::x86_64::*;

    let nb = q8k.len();
    let m4 = _mm_set1_epi8(0x0f);
    let ones_i16 = _mm256_set1_epi16(1);
    let mut acc_sum = 0.0f32;
    let mut qs_buf = [0i16; 16];

    for i in 0..nb {
        let boff = i * super::BLOCK_Q2K_SIZE;
        if boff + super::BLOCK_Q2K_SIZE > q2k_data.len() {
            break;
        }
        let d_raw = u16::from_le_bytes([q2k_data[boff + 80], q2k_data[boff + 81]]);
        let dmin_raw = u16::from_le_bytes([q2k_data[boff + 82], q2k_data[boff + 83]]);
        let d_total = super::f16_to_f32(d_raw) * q8k[i].d;
        let dmin_total = -super::f16_to_f32(dmin_raw) * q8k[i].d;

        let mins_and_scales = _mm_loadu_si128(q2k_data.as_ptr().add(boff) as *const __m128i);
        let scales8 = _mm_and_si128(mins_and_scales, m4);
        let mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        let mins = _mm256_cvtepi8_epi16(mins8);
        let bsums = _mm256_loadu_si256(q8k[i].bsums.as_ptr() as *const __m256i);
        let min_prod = _mm256_madd_epi16(mins, bsums);
        let min_total = hsum256_ps(_mm256_cvtepi32_ps(min_prod));
        acc_sum += dmin_total * min_total;

        let scales_ptr = q2k_data.as_ptr().add(boff) as *const u8;
        let q2_ptr = q2k_data.as_ptr().add(boff + 16);
        let q8_base_ptr = q8k[i].qs.as_ptr();

        for outer in 0..2usize {
            let qs_base_outer = outer * 32;
            for j in 0..4usize {
                let shift = (j as i32) * 2;
                let qs_base = qs_base_outer;
                let q8_base = outer * 128 + j * 32;
                let scale_byte_a = *scales_ptr.add(outer * 8 + j * 2);
                let scale_byte_b = *scales_ptr.add(outer * 8 + j * 2 + 1);
                let scale_a = (scale_byte_a & 0x0f) as i32;
                let scale_b = (scale_byte_b & 0x0f) as i32;
                for sub in 0..2usize {
                    let qs_off = qs_base + sub * 16;
                    let q8_off = q8_base + sub * 16;
                    let qs_bytes = std::slice::from_raw_parts(q2_ptr.add(qs_off), 16);
                    for l in 0..16 {
                        qs_buf[l] = ((qs_bytes[l] as i16) >> shift) & 3;
                    }
                    let qs_v = _mm256_loadu_si256(qs_buf.as_ptr() as *const __m256i);
                    let q8_128 = _mm_loadu_si128(q8_base_ptr.add(q8_off) as *const __m128i);
                    let q8_v = _mm256_cvtepi8_epi16(q8_128);
                    let prods = _mm256_mullo_epi16(qs_v, q8_v);
                    let dot_v = _mm256_madd_epi16(ones_i16, prods);
                    let dot = hsum_i32(dot_v);
                    let scale = if sub == 0 { scale_a } else { scale_b };
                    acc_sum += (dot as f32 * scale as f32) * d_total;
                }
            }
        }
    }

    acc_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_i32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let s = _mm_add_epi32(lo, hi);
    let shuf = _mm_shuffle_epi32::<0b10_11_00_01>(s);
    let s_ab = _mm_add_epi32(s, shuf);
    let shuf2 = _mm_shuffle_epi32::<0b11_10_11_10>(s_ab);
    let s_sum = _mm_add_epi32(s_ab, shuf2);
    _mm_cvtsi128_si32(s_sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_dot_q3k_q8k_avx2(
    q3k_data: &[u8],
    q8k: &[super::BlockQ8K],
) -> f32 {
    use std::arch::x86_64::*;

    let nb = q8k.len();
    let mut acc_sum = 0.0f32;

    for i in 0..nb {
        let boff = i * super::BLOCK_Q3K_SIZE;
        if boff + super::BLOCK_Q3K_SIZE > q3k_data.len() {
            break;
        }
        let d_raw = u16::from_le_bytes([q3k_data[boff + 108], q3k_data[boff + 109]]);
        let d = super::f16_to_f32(d_raw) * q8k[i].d;

        // Decode scales (same as scalar).
        let aux0 = u32::from_le_bytes([
            q3k_data[boff + 96], q3k_data[boff + 97], q3k_data[boff + 98], q3k_data[boff + 99],
        ]);
        let aux1 = u32::from_le_bytes([
            q3k_data[boff + 100], q3k_data[boff + 101], q3k_data[boff + 102], q3k_data[boff + 103],
        ]);
        let aux2 = u32::from_le_bytes([
            q3k_data[boff + 104], q3k_data[boff + 105], q3k_data[boff + 106], q3k_data[boff + 107],
        ]);
        let kmask1 = 0x03030303u32;
        let kmask2 = 0x0f0f0f0fu32;
        let tmp = aux2;
        let scale0u = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
        let scale1u = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
        let scale2u = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        let scale3u = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        let scales_unsigned: [u8; 16] = bytemuck::cast([scale0u, scale1u, scale2u, scale3u]);

        let hbits = _mm256_loadu_si256(q3k_data.as_ptr().add(boff) as *const __m256i);
        let m3 = _mm256_set1_epi8(3);
        let m4 = _mm256_set1_epi8(4);
        let mut acc = _mm256_setzero_si256();

        let mut qs_ptr = q3k_data.as_ptr().add(boff + 32);
        let mut q8_ptr = q8k[i].qs.as_ptr();

        // 2 outer iterations; per outer: 4 shifts × 32 aux8 elements.
        // Per shift: 2 sub-iterations of 16 aux8 elements each (one scale per sub).
        for outer in 0..2usize {
            let q3bits = _mm256_loadu_si256(qs_ptr as *const __m256i);
            qs_ptr = qs_ptr.add(32);

            for shift_idx in 0..4usize {
                // q2 (32 elements, 2-bit field at shift_idx * 2).
                let q2 = match shift_idx {
                    0 => _mm256_and_si256(q3bits, m3),
                    1 => _mm256_and_si256(_mm256_srli_epi16::<2>(q3bits), m3),
                    2 => _mm256_and_si256(_mm256_srli_epi16::<4>(q3bits), m3),
                    _ => _mm256_and_si256(_mm256_srli_epi16::<6>(q3bits), m3),
                };

                // high_v per byte: 4 if hbits bit not set, 0 if set.
                let bit_pos = outer * 4 + shift_idx;
                let mask_v = _mm256_set1_epi8(1i8 << bit_pos);
                let has_high = _mm256_and_si256(hbits, mask_v);
                let cmp_zero = _mm256_cmpeq_epi8(has_high, _mm256_setzero_si256());
                let high_v = _mm256_and_si256(cmp_zero, m4);

                // aux8_signed = q2 - high_v (i8, range -4..3).
                let aux8_signed = _mm256_sub_epi8(q2, high_v);

                for sub in 0..2usize {
                    // Extract 16 aux8 elements (low or high half of __m256i).
                    let aux8_16: __m128i = if sub == 0 {
                        _mm256_castsi256_si128(aux8_signed)
                    } else {
                        _mm256_extracti128_si256(aux8_signed, 1)
                    };
                    let q8_16 = _mm_loadu_si128(q8_ptr as *const __m128i);
                    q8_ptr = q8_ptr.add(16);

                    // Sign-extend to i16 (full 16 lanes per __m256i).
                    let aux16 = _mm256_cvtepi8_epi16(aux8_16);
                    let q816 = _mm256_cvtepi8_epi16(q8_16);

                    // Element-wise i16 products: lane k = aux16[k] * q816[k].
                    let prods = _mm256_mullo_epi16(aux16, q816);

                    // Sum prods[i] + prods[i+8] for i=0..7 (scalar pairs distance-8 elements).
                    // Lower half of prods has prods[0..8], upper half has prods[8..16].
                    // Extract upper half and add to lower half (which has 16-bit lanes).
                    let prods_hi = _mm256_extracti128_si256(prods, 1);
                    let prods_lo_128 = _mm256_castsi256_si128(prods);
                    let pairs_128 = _mm_add_epi16(prods_lo_128, prods_hi);
                    // Widen pairs (i16) to i32 (in low 256 bits).
                    let pairs_i32 = _mm256_cvtepi16_epi32(pairs_128);

                    let scale_idx = outer * 8 + shift_idx * 2 + sub;
                    let scale_broadcast = _mm256_set1_epi32(scales_unsigned[scale_idx] as i32 - 32);
                    let prod_scaled = _mm256_mullo_epi32(pairs_i32, scale_broadcast);

                    acc = _mm256_add_epi32(acc, prod_scaled);
                }
            }
        }

        // Horizontal sum of the low 8 i32 lanes of acc (high half is zero).
        let lo = _mm256_castsi256_si128(acc);
        let hi = _mm256_extracti128_si256(acc, 1);
        let s = _mm_add_epi32(lo, hi);
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
        let dot = _mm_cvtsi128_si32(s) as f32;

        acc_sum += d * dot;
    }

    acc_sum
}