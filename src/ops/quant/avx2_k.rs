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
pub(crate) unsafe fn vec_dot_iq4_xs_q8k_avx2(q4xs_data: &[u8], q8k: &[super::BlockQ8K]) -> f32 {
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

/// Scale shuffle tables for Q2_K / Q3_K dot products: half `i` of table `k`
/// broadcasts the (2k+i)-th 16-bit scale lane of `scales256` across all 8 i16
/// lanes of that half, so that a following `_mm256_madd_epi16(scale_v, p16)`
/// multiplies every pair-sum lane with its sub-block scale. Same tables as
/// llama.cpp's `get_scale_shuffle_q3k` (pinned reference).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn scale_shuffle_kq() -> [std::arch::x86_64::__m256i; 4] {
    use std::arch::x86_64::*;
    [
        _mm256_setr_epi8(
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, //
            2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        ),
        _mm256_setr_epi8(
            4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, //
            6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        ),
        _mm256_setr_epi8(
            8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, //
            10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
        ),
        _mm256_setr_epi8(
            12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, //
            14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        ),
    ]
}

/// AVX2 SIMD kernel for Q2_K × Q8_K dot product.
///
/// Port of the pinned llama.cpp `ggml_vec_dot_q2_K_q8_K` AVX2 branch
/// (`ggml/src/ggml-cpu/arch/x86/quants.c`), aligned with
/// `vec_dot_q2k_q8k_scalar` in `super`. Per 256-element super-block:
///
/// 1. `mins` (high nibbles) × `bsums` folded via `_mm256_madd_epi16`,
///    accumulated as `acc += (-dmin·d8) · Σ min_k·bsums_k`.
/// 2. The 32-byte `qs` chunk is unpacked into four 2-bit lanes with
///    `_mm256_srli_epi16` + mask (no scalar loop), paired with four 32-byte
///    Q8 loads through `_mm256_maddubs_epi16` (u8 × i8 → i16 pair sums).
/// 3. Sub-block scales are folded in with `_mm256_madd_epi16` over a
///    `SCALE_SHUFFLE_K2` broadcast, keeping everything in one i32 accumulator
///    (`sumi`) per super-block; a single float FMA joins the block scale.
///
/// Integer overflow bounds (i16 lanes): |q2·q8| ≤ 3·128, pair ≤ 768; scale
/// folds stay ≤ 15·1536 per i32 lane, far below 2³¹ even across both chunks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_dot_q2k_q8k_avx2(q2k_data: &[u8], q8k: &[super::BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    let nb = q8k.len();
    let m3 = _mm256_set1_epi8(3);
    let m4 = _mm_set1_epi8(0x0f);
    let scale_shuffles = scale_shuffle_kq();
    let mut acc = _mm256_setzero_ps();

    for i in 0..nb {
        let boff = i * super::BLOCK_Q2K_SIZE;
        if boff + super::BLOCK_Q2K_SIZE > q2k_data.len() {
            break;
        }
        let d_raw = u16::from_le_bytes([q2k_data[boff + 80], q2k_data[boff + 81]]);
        let dmin_raw = u16::from_le_bytes([q2k_data[boff + 82], q2k_data[boff + 83]]);
        let d = super::f16_to_f32(d_raw) * q8k[i].d;
        let dmin = -super::f16_to_f32(dmin_raw) * q8k[i].d;

        // scales[16]: low nibble = sub-block scale, high nibble = sub-block min.
        let mins_and_scales = _mm_loadu_si128(q2k_data.as_ptr().add(boff) as *const __m128i);
        let scales8 = _mm_and_si128(mins_and_scales, m4);
        let mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        let mins = _mm256_cvtepi8_epi16(mins8);
        let bsums = _mm256_loadu_si256(q8k[i].bsums.as_ptr() as *const __m256i);
        let prod = _mm256_madd_epi16(mins, bsums);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(prod), acc);

        // Scales 0..7 drive the first 128-element chunk, 8..15 the second.
        let all_scales = _mm256_cvtepi8_epi16(scales8);
        let l_scales = _mm256_castsi256_si128(all_scales);
        let h_scales = _mm256_extracti128_si256(all_scales, 1);
        let scales = [
            _mm256_set_m128i(l_scales, l_scales),
            _mm256_set_m128i(h_scales, h_scales),
        ];

        let mut sumi = _mm256_setzero_si256();
        let mut q2_off = boff + 16;
        let mut q8_off = 0usize;

        for j in 0..2usize {
            let q2bits = _mm256_loadu_si256(q2k_data.as_ptr().add(q2_off) as *const __m256i);
            q2_off += 32;

            let q8_0 = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off) as *const __m256i);
            let q8_1 = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off + 32) as *const __m256i);
            let q8_2 = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off + 64) as *const __m256i);
            let q8_3 = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off + 96) as *const __m256i);
            q8_off += 128;

            let q2_0 = _mm256_and_si256(q2bits, m3);
            let q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            let q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            let q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

            let mut p0 = _mm256_maddubs_epi16(q2_0, q8_0);
            let mut p1 = _mm256_maddubs_epi16(q2_1, q8_1);
            let mut p2 = _mm256_maddubs_epi16(q2_2, q8_2);
            let mut p3 = _mm256_maddubs_epi16(q2_3, q8_3);

            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], scale_shuffles[0]), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], scale_shuffles[1]), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], scale_shuffles[2]), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], scale_shuffles[3]), p3);

            let p01 = _mm256_add_epi32(p0, p1);
            let p23 = _mm256_add_epi32(p2, p3);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p01, p23));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    hsum256_ps(acc)
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

/// AVX2 SIMD kernel for Q3_K × Q8_K dot product.
///
/// Port of the pinned llama.cpp `ggml_vec_dot_q3_K_q8_K` AVX2 branch
/// (`ggml/src/ggml-cpu/arch/x86/quants.c`), aligned with
/// `vec_dot_q3k_q8k_scalar` in `super`. Per 256-element super-block the
/// signed 3-bit values `q3 = q2 − 4·(high_bit == 0)` are split into an
/// unsigned 2-bit part and a 0/4 high-bit part so the dot products can run
/// on `_mm256_maddubs_epi16` (u8 × i8 → i16 pair sums):
///
/// ```text
/// p16 = maddubs(q3l, q8) − maddubs(q3h, q8)   // q3h ∈ {0, 4}
/// ```
///
/// Sub-block scales (−32..31 after the byte-level `_mm_sub_epi8(32)`) are
/// folded in with `_mm256_madd_epi16` over the `scale_shuffle_kq` broadcast,
/// accumulating into one i32 vector per super-block; a single float FMA joins
/// the block scale. The high-bit test uses `and + cmpeq` instead of llama.cpp's
/// shift-based form so the bit index can stay a runtime value.
///
/// Integer overflow bounds (i16 lanes): |p16| ≤ 3·128·2 = 768 and |q8s| ≤
/// 4·128·2 = 1024, so the subtraction stays within ±1792; scale folds keep
/// every i32 lane below 2³¹ even after both 128-element chunks accumulate.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_dot_q3k_q8k_avx2(q3k_data: &[u8], q8k: &[super::BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    let nb = q8k.len();
    let m3 = _mm256_set1_epi8(3);
    let m4 = _mm256_set1_epi8(4);
    let m32 = _mm_set1_epi8(32);
    let scale_shuffles = scale_shuffle_kq();
    let mut acc = _mm256_setzero_ps();

    for i in 0..nb {
        let boff = i * super::BLOCK_Q3K_SIZE;
        if boff + super::BLOCK_Q3K_SIZE > q3k_data.len() {
            break;
        }
        let d_raw = u16::from_le_bytes([q3k_data[boff + 108], q3k_data[boff + 109]]);
        let d = super::f16_to_f32(d_raw) * q8k[i].d;

        // Decode the 16 packed 6-bit scales (same bit surgery as scalar).
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
        let scales128 = _mm_set_epi32(
            (((aux1 >> 4) & kmask2) | (((aux2 >> 6) & kmask1) << 4)) as i32,
            (((aux0 >> 4) & kmask2) | (((aux2 >> 4) & kmask1) << 4)) as i32,
            ((aux1 & kmask2) | (((aux2 >> 2) & kmask1) << 4)) as i32,
            ((aux0 & kmask2) | ((aux2 & kmask1) << 4)) as i32,
        );
        // Byte-level −32 turns the 0..63 nibble pairs into signed −32..31 scales.
        let scales128 = _mm_sub_epi8(scales128, m32);
        let all_scales = _mm256_cvtepi8_epi16(scales128);
        let l_scales = _mm256_castsi256_si128(all_scales);
        let h_scales = _mm256_extracti128_si256(all_scales, 1);
        let scales = [
            _mm256_set_m128i(l_scales, l_scales),
            _mm256_set_m128i(h_scales, h_scales),
        ];

        let hbits = _mm256_loadu_si256(q3k_data.as_ptr().add(boff) as *const __m256i);
        let mut sumi = _mm256_setzero_si256();
        let mut q3_off = boff + 32;
        let mut q8_off = 0usize;

        for j in 0..2usize {
            let q3bits = _mm256_loadu_si256(q3k_data.as_ptr().add(q3_off) as *const __m256i);
            q3_off += 32;

            let q8_0 = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off) as *const __m256i);
            let q8_1 = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off + 32) as *const __m256i);
            let q8_2 = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off + 64) as *const __m256i);
            let q8_3 = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off + 96) as *const __m256i);
            q8_off += 128;

            let q3l_0 = _mm256_and_si256(q3bits, m3);
            let q3l_1 = _mm256_and_si256(_mm256_srli_epi16::<2>(q3bits), m3);
            let q3l_2 = _mm256_and_si256(_mm256_srli_epi16::<4>(q3bits), m3);
            let q3l_3 = _mm256_and_si256(_mm256_srli_epi16::<6>(q3bits), m3);

            // Per byte: 4 if the hbits bit is clear, 0 if set (scalar: `4 - has_bit`).
            let q3h_0 = high_bit_part(hbits, j * 4, m4);
            let q3h_1 = high_bit_part(hbits, j * 4 + 1, m4);
            let q3h_2 = high_bit_part(hbits, j * 4 + 2, m4);
            let q3h_3 = high_bit_part(hbits, j * 4 + 3, m4);

            let q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
            let q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
            let q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
            let q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

            let mut p0 = _mm256_maddubs_epi16(q3l_0, q8_0);
            let mut p1 = _mm256_maddubs_epi16(q3l_1, q8_1);
            let mut p2 = _mm256_maddubs_epi16(q3l_2, q8_2);
            let mut p3 = _mm256_maddubs_epi16(q3l_3, q8_3);

            p0 = _mm256_sub_epi16(p0, q8s_0);
            p1 = _mm256_sub_epi16(p1, q8s_1);
            p2 = _mm256_sub_epi16(p2, q8s_2);
            p3 = _mm256_sub_epi16(p3, q8s_3);

            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], scale_shuffles[0]), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], scale_shuffles[1]), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], scale_shuffles[2]), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], scale_shuffles[3]), p3);

            let p01 = _mm256_add_epi32(p0, p1);
            let p23 = _mm256_add_epi32(p2, p3);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p01, p23));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    hsum256_ps(acc)
}

/// Per-byte `4` where `hbits`'s `bit`-th lane bit is clear, `0` where set —
/// the additive high-bit part of the signed Q3_K value.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn high_bit_part(
    hbits: std::arch::x86_64::__m256i,
    bit: usize,
    m4: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let mask = _mm256_set1_epi8(1i8 << bit);
    let has_high = _mm256_and_si256(hbits, mask);
    let bit_clear = _mm256_cmpeq_epi8(has_high, _mm256_setzero_si256());
    _mm256_and_si256(bit_clear, m4)
}

/// AVX2 SIMD kernel for IQ4_NL × Q8_K dot product.
///
/// Each IQ4_NL super-block groups 8 blocks of 32 elements (256 total), sharing
/// one Q8_K activation block. Per block: 2 bytes f16 d + 16 bytes qs (4-bit
/// nibbles). Lookup table `KVALUES_IQ4NL` (16-entry non-linear) replaces the
/// linear `(nibble - 8) * d` mapping of Q4_0.
///
/// Precision contract: ≤ 1 ULP drift vs `vec_dot_iq4_nl_q8k_scalar` (FMA single
/// rounding vs scalar sequential accumulation, see OPTIMIZATION.md §1.2). End
/// to end on Qwen3-0.6B-IQ4_NL.gguf produces "Paris" — accepted production drift.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_dot_iq4_nl_q8k_avx2(iq4nl_data: &[u8], q8k: &[super::BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    let nb = q8k.len();
    let m0f128 = _mm_set1_epi8(0x0f);
    let lut128 = _mm_loadu_si128(super::KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let mut acc_sum = 0.0f32;

    for i in 0..nb {
        let super_off = i * 8 * 18;
        if super_off + 8 * 18 > iq4nl_data.len() {
            break;
        }

        for sb in 0..8usize {
            let boff = super_off + sb * 18;
            let d_raw = u16::from_le_bytes([iq4nl_data[boff], iq4nl_data[boff + 1]]);
            let d = super::f16_to_f32(d_raw) * q8k[i].d;

            let qs = &iq4nl_data[boff + 2..boff + 18];
            let q8_block = &q8k[i].qs[sb * 32..(sb + 1) * 32];

            let qb = _mm_loadu_si128(qs.as_ptr() as *const __m128i);
            let q8_lo = _mm_loadu_si128(q8_block.as_ptr() as *const __m128i);
            let q8_hi = _mm_loadu_si128(q8_block.as_ptr().add(16) as *const __m128i);

            let lo_nib = _mm_and_si128(qb, m0f128);
            let hi_nib = _mm_and_si128(_mm_srli_epi16(qb, 4), m0f128);
            let lo_lut = _mm_shuffle_epi8(lut128, lo_nib);
            let hi_lut = _mm_shuffle_epi8(lut128, hi_nib);
            let lo_i16 = _mm256_cvtepi8_epi16(lo_lut);
            let hi_i16 = _mm256_cvtepi8_epi16(hi_lut);
            let q8_lo_i16 = _mm256_cvtepi8_epi16(q8_lo);
            let q8_hi_i16 = _mm256_cvtepi8_epi16(q8_hi);

            let p_lo = _mm256_madd_epi16(lo_i16, q8_lo_i16);
            let p_hi = _mm256_madd_epi16(hi_i16, q8_hi_i16);
            let p = _mm256_add_epi32(p_lo, p_hi);
            let dot = hsum_i32(p);

            acc_sum += d * dot as f32;
        }
    }

    acc_sum
}
