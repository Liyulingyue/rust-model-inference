#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let shuf = _mm_movehl_ps(s, s);
    let s2 = _mm_add_ps(s, shuf);
    let shuf2 = _mm_shuffle_ps::<0b11_10_11_10>(s2, s2);
    let s3 = _mm_add_ps(s2, shuf2);
    _mm_cvtss_f32(s3)
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
        let mut qs_off = 0usize;
        let mut q8_off = 0usize;
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
    let m3 = _mm_set1_epi8(3);
    let m4 = _mm_set1_epi8(0x0f);
    let ones_i16 = _mm256_set1_epi16(1);
    let mut acc_sum = 0.0f32;

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
            for j in 0..4usize {
                let qs_off = outer * 32 + j * 8;
                let q8_off = outer * 128 + j * 32;
                let q2_raw = _mm_loadl_epi64(q2_ptr.add(qs_off) as *const __m128i);
                let q2_shifted = match j {
                    0 => q2_raw,
                    1 => _mm_srli_epi16::<2>(q2_raw),
                    2 => _mm_srli_epi16::<4>(q2_raw),
                    _ => _mm_srli_epi16::<6>(q2_raw),
                };
                let q2_u16 = _mm_and_si128(q2_shifted, m3);
                let q2_vec = _mm256_zextsi128_si256(q2_u16);

                let q8_a = _mm_loadu_si128(q8_base_ptr.add(q8_off) as *const __m128i);
                let q8_b = _mm_loadu_si128(q8_base_ptr.add(q8_off + 16) as *const __m128i);
                let q8_a_vec = _mm256_zextsi128_si256(q8_a);
                let q8_b_vec = _mm256_zextsi128_si256(q8_b);

                let p_a = _mm256_maddubs_epi16(q2_vec, q8_a_vec);
                let p_b = _mm256_maddubs_epi16(q2_vec, q8_b_vec);

                let dot_a_vec = _mm256_madd_epi16(ones_i16, p_a);
                let dot_b_vec = _mm256_madd_epi16(ones_i16, p_b);
                let dot_a = hsum_i32(dot_a_vec);
                let dot_b = hsum_i32(dot_b_vec);

                let scale_byte = *scales_ptr.add(outer * 8 + j * 2);
                let scale_lo = (scale_byte & 0x0f) as i32;
                let scale_hi = ((scale_byte >> 4) & 0x0f) as i32;
                let contrib_i32 = dot_a * scale_lo + dot_b * scale_hi;
                acc_sum += (contrib_i32 as f32) * d_total;
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
    let shuf = _mm_shuffle_epi32::<0b01_00_01_00>(s);
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
    let m3 = _mm256_set1_epi8(3);
    let mut acc_sum = 0.0f32;

    for i in 0..nb {
        let boff = i * super::BLOCK_Q3K_SIZE;
        if boff + super::BLOCK_Q3K_SIZE > q3k_data.len() {
            break;
        }
        let d_raw = u16::from_le_bytes([q3k_data[boff + 108], q3k_data[boff + 109]]);
        let d = super::f16_to_f32(d_raw) * q8k[i].d;

        let aux0 = u32::from_le_bytes([
            q3k_data[boff + 96],
            q3k_data[boff + 97],
            q3k_data[boff + 98],
            q3k_data[boff + 99],
        ]);
        let aux1 = u32::from_le_bytes([
            q3k_data[boff + 100],
            q3k_data[boff + 101],
            q3k_data[boff + 102],
            q3k_data[boff + 103],
        ]);
        let aux2 = u32::from_le_bytes([
            q3k_data[boff + 104],
            q3k_data[boff + 105],
            q3k_data[boff + 106],
            q3k_data[boff + 107],
        ]);
        let kmask1 = 0x03030303u32;
        let kmask2 = 0x0f0f0f0fu32;
        let tmp = aux2;
        let scale0 = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
        let scale1 = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
        let scale2 = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        let scale3 = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        let scales_unsigned: [u8; 16] = bytemuck::cast([scale0, scale1, scale2, scale3]);
        let scales128 = _mm_loadu_si128(scales_unsigned.as_ptr() as *const __m128i);
        let m32 = _mm_set1_epi8(32);
        let scales_signed128 = _mm_sub_epi8(scales128, m32);
        let all_scales = _mm256_cvtepi8_epi16(scales_signed128);
        let low_scales = _mm256_extracti128_si256(all_scales, 0);
        let high_scales = _mm256_extracti128_si256(all_scales, 1);
        let scales = [
            _mm256_set_m128i(low_scales, low_scales),
            _mm256_set_m128i(high_scales, high_scales),
        ];

        let scale_shuffles = [
            _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                              2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3),
            _mm256_setr_epi8(4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
                              6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7),
            _mm256_setr_epi8(8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
                              10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11),
            _mm256_setr_epi8(12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
                              14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15),
        ];

        let hbits = _mm256_loadu_si256(q3k_data.as_ptr().add(boff) as *const __m256i);
        let mut q3_ptr = q3k_data.as_ptr().add(boff + 32);
        let mut q8_ptr = q8k[i].qs.as_ptr();
        for outer in 0..2usize {
            let q3bits = _mm256_loadu_si256(q3_ptr as *const __m256i);
            q3_ptr = q3_ptr.add(32);
            let q8_0 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);
            let q8_1 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);
            let q8_2 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);
            let q8_3 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);
            let q8_v = [q8_0, q8_1, q8_2, q8_3];
            let base_shift: u32 = if outer == 0 { 0 } else { 4 };

            for j in 0..4usize {
                let total_shift = base_shift + (j as u32) * 2;
                let q2_vec = match total_shift {
                    0 => _mm256_and_si256(q3bits, m3),
                    2 => _mm256_and_si256(_mm256_srli_epi16::<2>(q3bits), m3),
                    4 => _mm256_and_si256(_mm256_srli_epi16::<4>(q3bits), m3),
                    6 => _mm256_and_si256(_mm256_srli_epi16::<6>(q3bits), m3),
                    8 => _mm256_and_si256(_mm256_srli_epi16::<8>(q3bits), m3),
                    _ => _mm256_and_si256(_mm256_srli_epi16::<10>(q3bits), m3),
                };
                let mask_byte: i8 = 1 << (total_shift as i8);
                let mask_v = _mm256_set1_epi8(mask_byte);
                let high_v = _mm256_and_si256(_mm256_cmpeq_epi8(hbits, mask_v), _mm256_set1_epi8(4));
                let q8_vec = q8_v[j];
                let p = _mm256_maddubs_epi16(q2_vec, q8_vec);
                let high_p = _mm256_maddubs_epi16(high_v, q8_vec);
                let net = _mm256_sub_epi16(p, high_p);
                let scaled = _mm256_madd_epi16(
                    _mm256_shuffle_epi8(scales[outer], scale_shuffles[j]),
                    net,
                );
                let scaled_ps = _mm256_cvtepi32_ps(scaled);
                let prod = _mm256_mul_ps(scaled_ps, _mm256_set1_ps(d));
                acc_sum += hsum256_ps(prod);
            }
        }
    }

    acc_sum
}