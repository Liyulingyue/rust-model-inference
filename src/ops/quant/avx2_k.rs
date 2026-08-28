#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::__m256i;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q2k_q3k_scale_shuffle(index: usize) -> __m256i {
    use std::arch::x86_64::*;

    let tables: [[u8; 32]; 4] = [
        [
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        ],
        [
            4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
            6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        ],
        [
            8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
            10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
            10, 11, 10, 11,
        ],
        [
            12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
            14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        ],
    ];
    _mm256_loadu_si256(tables[index].as_ptr() as *const __m256i)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_dot_q2k_q8k_avx2(
    q2k_data: &[u8],
    q8k: &[super::BlockQ8K],
) -> f32 {
    use std::arch::x86_64::*;

    let nb = q8k.len();
    let m3 = _mm256_set1_epi8(3);
    let m4 = _mm_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_ps();
    let scale_shuffles = [
        q2k_q3k_scale_shuffle(0),
        q2k_q3k_scale_shuffle(1),
        q2k_q3k_scale_shuffle(2),
        q2k_q3k_scale_shuffle(3),
    ];

    for i in 0..nb {
        let boff = i * super::BLOCK_Q2K_SIZE;
        if boff + super::BLOCK_Q2K_SIZE > q2k_data.len() {
            break;
        }
        let d_raw = u16::from_le_bytes([q2k_data[boff + 80], q2k_data[boff + 81]]);
        let dmin_raw = u16::from_le_bytes([q2k_data[boff + 82], q2k_data[boff + 83]]);
        let d = super::f16_to_f32(d_raw) * q8k[i].d;
        let dmin = -super::f16_to_f32(dmin_raw) * q8k[i].d;

        let mins_and_scales = _mm_loadu_si128(q2k_data.as_ptr().add(boff) as *const __m128i);
        let scales8 = _mm_and_si128(mins_and_scales, m4);
        let mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        let mins = _mm256_cvtepi8_epi16(mins8);
        let bsums = _mm256_loadu_si256(q8k[i].bsums.as_ptr() as *const __m256i);
        let min_sums = _mm256_cvtepi32_ps(_mm256_madd_epi16(mins, bsums));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(dmin), min_sums));

        let all_scales = _mm256_cvtepi8_epi16(scales8);
        let low_scales = _mm256_extracti128_si256(all_scales, 0);
        let high_scales = _mm256_extracti128_si256(all_scales, 1);
        let scales = [
            _mm256_set_m128i(low_scales, low_scales),
            _mm256_set_m128i(high_scales, high_scales),
        ];

        let mut sumi = _mm256_setzero_si256();
        let mut q2_ptr = q2k_data.as_ptr().add(boff + 16);
        let mut q8_ptr = q8k[i].qs.as_ptr();
        for outer in 0..2usize {
            let q2bits = _mm256_loadu_si256(q2_ptr as *const __m256i);
            q2_ptr = q2_ptr.add(32);
            let q8_0 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);
            let q8_1 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);
            let q8_2 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);
            let q8_3 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);

            let q2_0 = _mm256_and_si256(q2bits, m3);
            let q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            let q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            let q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

            let p0 = _mm256_madd_epi16(
                _mm256_shuffle_epi8(scales[outer], scale_shuffles[0]),
                _mm256_maddubs_epi16(q2_0, q8_0),
            );
            let p1 = _mm256_madd_epi16(
                _mm256_shuffle_epi8(scales[outer], scale_shuffles[1]),
                _mm256_maddubs_epi16(q2_1, q8_1),
            );
            let p2 = _mm256_madd_epi16(
                _mm256_shuffle_epi8(scales[outer], scale_shuffles[2]),
                _mm256_maddubs_epi16(q2_2, q8_2),
            );
            let p3 = _mm256_madd_epi16(
                _mm256_shuffle_epi8(scales[outer], scale_shuffles[3]),
                _mm256_maddubs_epi16(q2_3, q8_3),
            );
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)));
        }
        acc = _mm256_add_ps(
            acc,
            _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)),
        );
    }
    crate::ops::hsum_ps(acc)
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
    let mone = _mm256_set1_epi8(1);
    let m32 = _mm_set1_epi8(32);
    let mut acc = _mm256_setzero_ps();
    let scale_shuffles = [
        q2k_q3k_scale_shuffle(0),
        q2k_q3k_scale_shuffle(1),
        q2k_q3k_scale_shuffle(2),
        q2k_q3k_scale_shuffle(3),
    ];

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
        let all_scales = _mm256_cvtepi8_epi16(_mm_sub_epi8(scales128, m32));
        let low_scales = _mm256_extracti128_si256(all_scales, 0);
        let high_scales = _mm256_extracti128_si256(all_scales, 1);
        let scales = [
            _mm256_set_m128i(low_scales, low_scales),
            _mm256_set_m128i(high_scales, high_scales),
        ];
        let hbits = _mm256_loadu_si256(q3k_data.as_ptr().add(boff) as *const __m256i);
        let mut sumi = _mm256_setzero_si256();
        let mut q3_ptr = q3k_data.as_ptr().add(boff + 32);
        let mut q8_ptr = q8k[i].qs.as_ptr();

        for outer in 0..2usize {
            let q3bits = _mm256_loadu_si256(q3_ptr as *const __m256i);
            q3_ptr = q3_ptr.add(32);
            let (mask0, mask1, mask2, mask3) = if outer == 0 {
                (
                    _mm256_set1_epi8(1),
                    _mm256_set1_epi8(2),
                    _mm256_set1_epi8(4),
                    _mm256_set1_epi8(8),
                )
            } else {
                (
                    _mm256_set1_epi8(16),
                    _mm256_set1_epi8(32),
                    _mm256_set1_epi8(64),
                    _mm256_set1_epi8(-128),
                )
            };
            let high0 = _mm256_and_si256(_mm256_cmpeq_epi8(hbits, mask0), _mm256_set1_epi8(4));
            let high1 = _mm256_and_si256(_mm256_cmpeq_epi8(hbits, mask1), _mm256_set1_epi8(4));
            let high2 = _mm256_and_si256(_mm256_cmpeq_epi8(hbits, mask2), _mm256_set1_epi8(4));
            let high3 = _mm256_and_si256(_mm256_cmpeq_epi8(hbits, mask3), _mm256_set1_epi8(4));

            let q3l_0 = _mm256_and_si256(q3bits, m3);
            let q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            let q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            let q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);

            let q8_0 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);
            let q8_1 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);
            let q8_2 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);
            let q8_3 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);

            let p0 = _mm256_madd_epi16(
                _mm256_shuffle_epi8(scales[outer], scale_shuffles[0]),
                _mm256_sub_epi16(
                    _mm256_maddubs_epi16(q3l_0, q8_0),
                    _mm256_maddubs_epi16(high0, q8_0),
                ),
            );
            let p1 = _mm256_madd_epi16(
                _mm256_shuffle_epi8(scales[outer], scale_shuffles[1]),
                _mm256_sub_epi16(
                    _mm256_maddubs_epi16(q3l_1, q8_1),
                    _mm256_maddubs_epi16(high1, q8_1),
                ),
            );
            let p2 = _mm256_madd_epi16(
                _mm256_shuffle_epi8(scales[outer], scale_shuffles[2]),
                _mm256_sub_epi16(
                    _mm256_maddubs_epi16(q3l_2, q8_2),
                    _mm256_maddubs_epi16(high2, q8_2),
                ),
            );
            let p3 = _mm256_madd_epi16(
                _mm256_shuffle_epi8(scales[outer], scale_shuffles[3]),
                _mm256_sub_epi16(
                    _mm256_maddubs_epi16(q3l_3, q8_3),
                    _mm256_maddubs_epi16(high3, q8_3),
                ),
            );
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)));
        }
        acc = _mm256_add_ps(
            acc,
            _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)),
        );
    }
    crate::ops::hsum_ps(acc)
}
