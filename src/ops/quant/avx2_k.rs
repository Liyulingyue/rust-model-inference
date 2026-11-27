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

// ============================================================================
// IQ2/IQ3/IQ1 × Q8_K AVX2 kernels — ports of the AVX2 branches in the pinned
// llama.cpp (`b96806d96061049a5b574269b049bf6241d63d46`,
// `ggml/src/ggml-cpu/arch/x86/quants.c`). Scalar references live in
// `super::vec_dot_iq*_q8k_scalar`; the dispatched entry points in `super`
// select AVX2 when `has_avx2_fma()` and fall back to scalar otherwise.
// ============================================================================

/// Sign table for the 7-bit IQ2/IQ3 sign codes, expanded to 8 bytes of ±1 per
/// entry (llama.cpp `keven_signs_q2xs`, an even-parity expansion of
/// `ksigns_iq2xs`). Derived from the scalar-side `KSIGNS_IQ2XS` table so the
/// two can never drift apart.
#[cfg(target_arch = "x86_64")]
const KEVEN_SIGNS_Q2XS: [u64; 128] = keven_signs_q2xs();

#[cfg(target_arch = "x86_64")]
const fn keven_signs_q2xs() -> [u64; 128] {
    use super::iq_tables::KSIGNS_IQ2XS;
    let mut out = [0u64; 128];
    let mut i = 0;
    while i < 128 {
        let mut v = 0u64;
        let mut j = 0;
        while j < 8 {
            let byte: u64 = if (KSIGNS_IQ2XS[i] >> j) & 1 == 1 {
                0xFF
            } else {
                0x01
            };
            v |= byte << (8 * j);
            j += 1;
        }
        out[i] = v;
        i += 1;
    }
    out
}

/// llama.cpp `hsum_float_8`: the add order pins the last-ulp rounding of the
/// f32 block accumulator, so the oracle comparison stays bit-exact.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_float_8_llama(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let mut res = _mm256_extractf128_ps(v, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(v));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

/// llama.cpp `mul_add_epi8`: `|x| · sign(x)·y` folded through one `maddubs`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mul_add_epi8(
    x: std::arch::x86_64::__m256i,
    y: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    _mm256_maddubs_epi16(ax, sy)
}

/// llama.cpp `get_scale_shuffle(i)`: row `i` of `k_shuffle` — 8 selector
/// bytes of `2i` followed by 8 of `2i+1`, so the following `madd` applies the
/// lo nibble scale to the first two grid rows and the hi nibble to the last
/// two (NOT a single-byte broadcast; the row pairs two scale nibbles).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn get_scale_shuffle(i: usize) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    let mut v = [0u8; 16];
    for (j, b) in v.iter_mut().enumerate() {
        *b = (2 * i + j / 8) as u8;
    }
    _mm_loadu_si128(v.as_ptr() as *const __m128i)
}

/// llama.cpp `get_scale_shuffle_k4(i)`: 32 selector bytes alternating
/// `(2i, 2i+1)`, pairing each row pair of `dot` with its nibble scale.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn get_scale_shuffle_k4(i: usize) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let mut v = [0u8; 32];
    for (j, b) in v.iter_mut().enumerate() {
        *b = (2 * i + (j & 1)) as u8;
    }
    _mm256_loadu_si256(v.as_ptr() as *const __m256i)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_dot_iq2_xxs_q8k_avx2(iq2xxs_data: &[u8], q8k: &[super::BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    let grid = &super::iq_tables::IQ2XXS_GRID;
    let signs64 = &KEVEN_SIGNS_Q2XS;
    let nb = q8k.len();
    let mut accumf = _mm256_setzero_ps();
    for i in 0..nb {
        let boff = i * super::BLOCK_IQ2_XXS_SIZE;
        if boff + super::BLOCK_IQ2_XXS_SIZE > iq2xxs_data.len() {
            break;
        }
        let d = super::f16_to_f32(u16::from_le_bytes([
            iq2xxs_data[boff],
            iq2xxs_data[boff + 1],
        ])) * q8k[i].d;
        let q2 = &iq2xxs_data[boff + 2..boff + 2 + 64];
        let q8 = &q8k[i].qs;
        let mut sumi1 = _mm256_setzero_si256();
        let mut sumi2 = _mm256_setzero_si256();
        let mut q2_off = 0usize;
        let mut q8_off = 0usize;
        for _ in 0..4usize {
            let q8_1 = _mm256_loadu_si256(q8.as_ptr().add(q8_off) as *const __m256i);
            let q8_2 = _mm256_loadu_si256(q8.as_ptr().add(q8_off + 32) as *const __m256i);
            q8_off += 64;
            let a0 =
                u32::from_le_bytes([q2[q2_off], q2[q2_off + 1], q2[q2_off + 2], q2[q2_off + 3]]);
            let a1 = u32::from_le_bytes([
                q2[q2_off + 4],
                q2[q2_off + 5],
                q2[q2_off + 6],
                q2[q2_off + 7],
            ]);
            let a2 = u32::from_le_bytes([
                q2[q2_off + 8],
                q2[q2_off + 9],
                q2[q2_off + 10],
                q2[q2_off + 11],
            ]);
            let a3 = u32::from_le_bytes([
                q2[q2_off + 12],
                q2[q2_off + 13],
                q2[q2_off + 14],
                q2[q2_off + 15],
            ]);
            q2_off += 16;
            let ab0 = a0.to_le_bytes();
            let ab2 = a2.to_le_bytes();
            let q2_1 = _mm256_set_epi64x(
                grid[ab0[3] as usize] as i64,
                grid[ab0[2] as usize] as i64,
                grid[ab0[1] as usize] as i64,
                grid[ab0[0] as usize] as i64,
            );
            let q2_2 = _mm256_set_epi64x(
                grid[ab2[3] as usize] as i64,
                grid[ab2[2] as usize] as i64,
                grid[ab2[1] as usize] as i64,
                grid[ab2[0] as usize] as i64,
            );
            let s2_1 = _mm256_set_epi64x(
                signs64[((a1 >> 21) & 127) as usize] as i64,
                signs64[((a1 >> 14) & 127) as usize] as i64,
                signs64[((a1 >> 7) & 127) as usize] as i64,
                signs64[(a1 & 127) as usize] as i64,
            );
            let s2_2 = _mm256_set_epi64x(
                signs64[((a3 >> 21) & 127) as usize] as i64,
                signs64[((a3 >> 14) & 127) as usize] as i64,
                signs64[((a3 >> 7) & 127) as usize] as i64,
                signs64[(a3 & 127) as usize] as i64,
            );
            let q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            let q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            let dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            let dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            let ls1 = (a1 >> 28) as u16;
            let ls2 = (a3 >> 28) as u16;
            let p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16((2 * ls1 + 1) as i16));
            let p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16((2 * ls2 + 1) as i16));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        accumf = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
            accumf,
        );
    }
    0.125f32 * hsum_float_8_llama(accumf)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_dot_iq2_xs_q8k_avx2(iq2xs_data: &[u8], q8k: &[super::BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    let grid = &super::iq_tables::IQ2XS_GRID;
    let nb = q8k.len();

    let mone = _mm256_set1_epi8(1);
    let block_sign_shuffle_mask_1: [i8; 32] = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x02, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
        0x06, 0x06,
    ];
    let block_sign_shuffle_mask_2: [i8; 32] = [
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0a, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
        0x0e, 0x0e,
    ];
    let bit_selector_mask_bytes: [u8; 32] = [
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
        0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
        0x40, 0x80,
    ];
    let bit_selector_mask = _mm256_loadu_si256(bit_selector_mask_bytes.as_ptr() as *const __m256i);
    let block_sign_shuffle_1 =
        _mm256_loadu_si256(block_sign_shuffle_mask_1.as_ptr() as *const __m256i);
    let block_sign_shuffle_2 =
        _mm256_loadu_si256(block_sign_shuffle_mask_2.as_ptr() as *const __m256i);

    let k_bit_helper: [u8; 32] = [
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80,
        0x00, 0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80,
        0x80, 0x00,
    ];
    let bit_helper = _mm256_loadu_si256(k_bit_helper.as_ptr() as *const __m256i);
    let m511 = _mm256_set1_epi16(511);
    let m4 = _mm_set1_epi8(0xf);
    let m1 = _mm_set1_epi8(1);

    let mut accumf = _mm256_setzero_ps();
    for i in 0..nb {
        let boff = i * super::BLOCK_IQ2_XS_SIZE;
        if boff + super::BLOCK_IQ2_XS_SIZE > iq2xs_data.len() {
            break;
        }
        let d = super::f16_to_f32(u16::from_le_bytes([iq2xs_data[boff], iq2xs_data[boff + 1]]))
            * q8k[i].d;
        let q2 = &iq2xs_data[boff + 2..boff + 2 + 64];
        let scales_bytes = &iq2xs_data[boff + 66..boff + 74];
        let q8 = &q8k[i].qs;

        let aux64 = u64::from_le_bytes(scales_bytes.try_into().unwrap());
        let stmp = _mm_set1_epi64x(aux64 as i64);
        let stmp = _mm_unpacklo_epi8(
            _mm_and_si128(stmp, m4),
            _mm_and_si128(_mm_srli_epi16(stmp, 4), m4),
        );
        let scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        let mut sumi1 = _mm256_setzero_si256();
        let mut sumi2 = _mm256_setzero_si256();
        let mut q2_off = 0usize;
        let mut q8_off = 0usize;
        for ib32 in [0usize, 4usize] {
            let q2_data = _mm256_loadu_si256(q2.as_ptr().add(q2_off) as *const __m256i);
            q2_off += 32;
            let aux_gindex = _mm256_and_si256(q2_data, m511);
            let partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
            let partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
            let partial_sign_bits_for_counting =
                _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);
            let odd_bits = _mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
            let full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);

            let q8_1 = _mm256_loadu_si256(q8.as_ptr().add(q8_off) as *const __m256i);
            let q8_2 = _mm256_loadu_si256(q8.as_ptr().add(q8_off + 32) as *const __m256i);
            let q8_3 = _mm256_loadu_si256(q8.as_ptr().add(q8_off + 64) as *const __m256i);
            let q8_4 = _mm256_loadu_si256(q8.as_ptr().add(q8_off + 96) as *const __m256i);
            q8_off += 128;

            let gindex: [u16; 16] = std::mem::transmute(aux_gindex);
            let q2_1 = _mm256_set_epi64x(
                grid[gindex[3] as usize] as i64,
                grid[gindex[2] as usize] as i64,
                grid[gindex[1] as usize] as i64,
                grid[gindex[0] as usize] as i64,
            );
            let q2_2 = _mm256_set_epi64x(
                grid[gindex[7] as usize] as i64,
                grid[gindex[6] as usize] as i64,
                grid[gindex[5] as usize] as i64,
                grid[gindex[4] as usize] as i64,
            );
            let q2_3 = _mm256_set_epi64x(
                grid[gindex[11] as usize] as i64,
                grid[gindex[10] as usize] as i64,
                grid[gindex[9] as usize] as i64,
                grid[gindex[8] as usize] as i64,
            );
            let q2_4 = _mm256_set_epi64x(
                grid[gindex[15] as usize] as i64,
                grid[gindex[14] as usize] as i64,
                grid[gindex[13] as usize] as i64,
                grid[gindex[12] as usize] as i64,
            );

            let full_signs_l = _mm256_castsi256_si128(full_sign_bits);
            let full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
            let full_signs_1 = _mm256_set_m128i(full_signs_l, full_signs_l);
            let full_signs_2 = _mm256_set_m128i(full_signs_h, full_signs_h);

            let mut signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(
                _mm256_and_si256(signs, bit_selector_mask),
                bit_selector_mask,
            );
            let q8s_1 = _mm256_sign_epi8(q8_1, _mm256_or_si256(signs, mone));

            let mut signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(
                _mm256_and_si256(signs, bit_selector_mask),
                bit_selector_mask,
            );
            let q8s_2 = _mm256_sign_epi8(q8_2, _mm256_or_si256(signs, mone));

            let mut signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(
                _mm256_and_si256(signs, bit_selector_mask),
                bit_selector_mask,
            );
            let q8s_3 = _mm256_sign_epi8(q8_3, _mm256_or_si256(signs, mone));

            let mut signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(
                _mm256_and_si256(signs, bit_selector_mask),
                bit_selector_mask,
            );
            let q8s_4 = _mm256_sign_epi8(q8_4, _mm256_or_si256(signs, mone));

            let dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            let dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            let dot3 = _mm256_maddubs_epi16(q2_3, q8s_3);
            let dot4 = _mm256_maddubs_epi16(q2_4, q8s_4);

            let sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32)));
            let sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 1)));
            let sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 2)));
            let sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 3)));

            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, sc1));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, sc2));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot3, sc3));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot4, sc4));
        }
        accumf = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
            accumf,
        );
    }
    0.125f32 * hsum_float_8_llama(accumf)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_dot_iq2_s_q8k_avx2(iq2s_data: &[u8], q8k: &[super::BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    let grid = &super::iq_tables::IQ2S_GRID;
    let nb = q8k.len();

    let k_mask1: [u8; 32] = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
        0x03, 0x03,
    ];
    let k_mask2: [u8; 32] = [
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
        0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
        0x40, 0x80,
    ];
    let m4 = _mm_set1_epi8(0xf);
    let m1 = _mm_set1_epi8(1);
    let mask1 = _mm256_loadu_si256(k_mask1.as_ptr() as *const __m256i);
    let mask2 = _mm256_loadu_si256(k_mask2.as_ptr() as *const __m256i);

    let mut accumf = _mm256_setzero_ps();
    for i in 0..nb {
        let boff = i * super::BLOCK_IQ2_S_SIZE;
        if boff + super::BLOCK_IQ2_S_SIZE > iq2s_data.len() {
            break;
        }
        let d = super::f16_to_f32(u16::from_le_bytes([iq2s_data[boff], iq2s_data[boff + 1]]))
            * q8k[i].d;
        let qs_all = &iq2s_data[boff + 2..boff + 2 + 64]; // qs[64] incl. signs
        let qh = &iq2s_data[boff + 66..boff + 74];
        let scales_bytes = &iq2s_data[boff + 74..boff + 82];
        let q8 = &q8k[i].qs;

        let aux64 = u64::from_le_bytes(scales_bytes.try_into().unwrap());
        let scales8 = _mm_add_epi8(
            _mm_slli_epi16(
                _mm_and_si128(_mm_set_epi64x((aux64 >> 4) as i64, aux64 as i64), m4),
                1,
            ),
            m1,
        );
        let scales16 = _mm256_cvtepi8_epi16(scales8);

        let mut sumi1 = _mm256_setzero_si256();
        let mut sumi2 = _mm256_setzero_si256();
        let mut qs_off = 0usize;
        let mut signs_off = 32usize; // signs live in the last 32 bytes of qs
        let mut q8_off = 0usize;
        let mut ib32 = 0usize;
        while ib32 < 8usize {
            let q8_1 = _mm256_loadu_si256(q8.as_ptr().add(q8_off) as *const __m256i);
            let q8_2 = _mm256_loadu_si256(q8.as_ptr().add(q8_off + 32) as *const __m256i);
            q8_off += 64;
            let qh0 = qh[ib32];
            let qh1 = qh[ib32 + 1];
            let q2_1 = _mm256_set_epi64x(
                grid[(qs_all[qs_off + 3] as usize | (((qh0 as usize) << 2) & 0x300)) as usize]
                    as i64,
                grid[(qs_all[qs_off + 2] as usize | (((qh0 as usize) << 4) & 0x300)) as usize]
                    as i64,
                grid[(qs_all[qs_off + 1] as usize | (((qh0 as usize) << 6) & 0x300)) as usize]
                    as i64,
                grid[(qs_all[qs_off] as usize | (((qh0 as usize) << 8) & 0x300)) as usize] as i64,
            );
            let q2_2 = _mm256_set_epi64x(
                grid[(qs_all[qs_off + 7] as usize | (((qh1 as usize) << 2) & 0x300)) as usize]
                    as i64,
                grid[(qs_all[qs_off + 6] as usize | (((qh1 as usize) << 4) & 0x300)) as usize]
                    as i64,
                grid[(qs_all[qs_off + 5] as usize | (((qh1 as usize) << 6) & 0x300)) as usize]
                    as i64,
                grid[(qs_all[qs_off + 4] as usize | (((qh1 as usize) << 8) & 0x300)) as usize]
                    as i64,
            );
            qs_off += 8;

            let s0 = u16::from_le_bytes([qs_all[signs_off], qs_all[signs_off + 1]]);
            let s1 = u16::from_le_bytes([qs_all[signs_off + 2], qs_all[signs_off + 3]]);
            let mut aux256 = _mm256_set1_epi32((s0 as u32 | ((s1 as u32) << 16)) as i32);
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            let s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            let q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            let s2 = u16::from_le_bytes([qs_all[signs_off + 4], qs_all[signs_off + 5]]);
            let s3 = u16::from_le_bytes([qs_all[signs_off + 6], qs_all[signs_off + 7]]);
            let mut aux256 = _mm256_set1_epi32((s2 as u32 | ((s3 as u32) << 16)) as i32);
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            let s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            let q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs_off += 8;

            let dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            let dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);

            let p1 = _mm256_madd_epi16(
                dot1,
                _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32)),
            );
            let p2 = _mm256_madd_epi16(
                dot2,
                _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32 + 1)),
            );
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
            ib32 += 2;
        }
        accumf = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
            accumf,
        );
    }
    0.125f32 * hsum_float_8_llama(accumf)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_dot_iq3_xxs_q8k_avx2(iq3xxs_data: &[u8], q8k: &[super::BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    let grid = &super::iq_tables::IQ3XXS_GRID;
    let signs64 = &KEVEN_SIGNS_Q2XS;
    let nb = q8k.len();

    let mut accumf = _mm256_setzero_ps();
    for i in 0..nb {
        let boff = i * super::BLOCK_IQ3_XXS_SIZE;
        if boff + super::BLOCK_IQ3_XXS_SIZE > iq3xxs_data.len() {
            break;
        }
        let d = super::f16_to_f32(u16::from_le_bytes([
            iq3xxs_data[boff],
            iq3xxs_data[boff + 1],
        ])) * q8k[i].d;
        let q3 = &iq3xxs_data[boff + 2..boff + 2 + 96]; // qs[64] + gas[32]
        let q8 = &q8k[i].qs;
        let mut sumi1 = _mm256_setzero_si256();
        let mut sumi2 = _mm256_setzero_si256();
        let mut q3_off = 0usize;
        let mut gas_off = 64usize;
        let mut q8_off = 0usize;
        for _ in 0..4usize {
            let q8_1 = _mm256_loadu_si256(q8.as_ptr().add(q8_off) as *const __m256i);
            let q8_2 = _mm256_loadu_si256(q8.as_ptr().add(q8_off + 32) as *const __m256i);
            q8_off += 64;
            let q2_1 = _mm256_set_epi32(
                grid[q3[q3_off + 7] as usize] as i32,
                grid[q3[q3_off + 6] as usize] as i32,
                grid[q3[q3_off + 5] as usize] as i32,
                grid[q3[q3_off + 4] as usize] as i32,
                grid[q3[q3_off + 3] as usize] as i32,
                grid[q3[q3_off + 2] as usize] as i32,
                grid[q3[q3_off + 1] as usize] as i32,
                grid[q3[q3_off] as usize] as i32,
            );
            q3_off += 8;
            let q2_2 = _mm256_set_epi32(
                grid[q3[q3_off + 7] as usize] as i32,
                grid[q3[q3_off + 6] as usize] as i32,
                grid[q3[q3_off + 5] as usize] as i32,
                grid[q3[q3_off + 4] as usize] as i32,
                grid[q3[q3_off + 3] as usize] as i32,
                grid[q3[q3_off + 2] as usize] as i32,
                grid[q3[q3_off + 1] as usize] as i32,
                grid[q3[q3_off] as usize] as i32,
            );
            q3_off += 8;
            let a0 = u32::from_le_bytes([
                q3[gas_off],
                q3[gas_off + 1],
                q3[gas_off + 2],
                q3[gas_off + 3],
            ]);
            let a1 = u32::from_le_bytes([
                q3[gas_off + 4],
                q3[gas_off + 5],
                q3[gas_off + 6],
                q3[gas_off + 7],
            ]);
            gas_off += 8;
            let s2_1 = _mm256_set_epi64x(
                signs64[((a0 >> 21) & 127) as usize] as i64,
                signs64[((a0 >> 14) & 127) as usize] as i64,
                signs64[((a0 >> 7) & 127) as usize] as i64,
                signs64[(a0 & 127) as usize] as i64,
            );
            let s2_2 = _mm256_set_epi64x(
                signs64[((a1 >> 21) & 127) as usize] as i64,
                signs64[((a1 >> 14) & 127) as usize] as i64,
                signs64[((a1 >> 7) & 127) as usize] as i64,
                signs64[(a1 & 127) as usize] as i64,
            );
            let q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            let q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            let dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            let dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            let ls1 = (a0 >> 28) as u16;
            let ls2 = (a1 >> 28) as u16;
            let p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16((2 * ls1 + 1) as i16));
            let p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16((2 * ls2 + 1) as i16));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        accumf = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
            accumf,
        );
    }
    0.25f32 * hsum_float_8_llama(accumf)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_dot_iq3_s_q8k_avx2(iq3s_data: &[u8], q8k: &[super::BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    let grid = &super::iq_tables::IQ3S_GRID;
    let nb = q8k.len();

    let k_mask1: [u8; 32] = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
        0x03, 0x03,
    ];
    let k_mask2: [u8; 32] = [
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
        0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
        0x40, 0x80,
    ];
    let mask1 = _mm256_loadu_si256(k_mask1.as_ptr() as *const __m256i);
    let mask2 = _mm256_loadu_si256(k_mask2.as_ptr() as *const __m256i);

    let idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    let idx_mask = _mm256_set1_epi32(256);

    let mut accumf = _mm256_setzero_ps();
    for i in 0..nb {
        let boff = i * super::BLOCK_IQ3_S_SIZE;
        if boff + super::BLOCK_IQ3_S_SIZE > iq3s_data.len() {
            break;
        }
        let d = super::f16_to_f32(u16::from_le_bytes([iq3s_data[boff], iq3s_data[boff + 1]]))
            * q8k[i].d;
        // block_iq3_s layout: d(2) + qs[QK_K/4](64) + qh[QK_K/32](8)
        // + signs[QK_K/8](32) + scales[IQ3S_N_SCALE](4)
        let qs_all = &iq3s_data[boff + 2..boff + 2 + 64];
        let qh = &iq3s_data[boff + 66..boff + 74];
        let signs = &iq3s_data[boff + 74..boff + 106];
        let scales_bytes = &iq3s_data[boff + 106..boff + 110];
        let q8 = &q8k[i].qs;

        let mut sumi1 = _mm256_setzero_si256();
        let mut sumi2 = _mm256_setzero_si256();
        let mut qs_off = 0usize;
        let mut signs_off = 0usize;
        let mut q8_off = 0usize;
        let mut ib32 = 0usize;
        while ib32 < 8usize {
            let q8_1 = _mm256_loadu_si256(q8.as_ptr().add(q8_off) as *const __m256i);
            let q8_2 = _mm256_loadu_si256(q8.as_ptr().add(q8_off + 32) as *const __m256i);
            q8_off += 64;
            let idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                qs_all.as_ptr().add(qs_off) as *const __m128i
            ));
            qs_off += 16;
            let lo = _mm256_set1_epi32(qh[ib32] as i32);
            let hi = _mm256_set1_epi32(qh[ib32 + 1] as i32);
            let lo = _mm256_and_si256(_mm256_sllv_epi32(lo, idx_shift), idx_mask);
            let hi = _mm256_and_si256(_mm256_sllv_epi32(hi, idx_shift), idx_mask);
            let lo = _mm256_or_si256(lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            let hi = _mm256_or_si256(
                hi,
                _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)),
            );
            let idx_lo: [u32; 8] = std::mem::transmute(lo);
            let idx_hi: [u32; 8] = std::mem::transmute(hi);
            // C's `index_t` union lays these out as index[0..8] / index[8..16].
            let q2_1 = _mm256_set_epi32(
                grid[idx_lo[7] as usize] as i32,
                grid[idx_lo[6] as usize] as i32,
                grid[idx_lo[5] as usize] as i32,
                grid[idx_lo[4] as usize] as i32,
                grid[idx_lo[3] as usize] as i32,
                grid[idx_lo[2] as usize] as i32,
                grid[idx_lo[1] as usize] as i32,
                grid[idx_lo[0] as usize] as i32,
            );
            let q2_2 = _mm256_set_epi32(
                grid[idx_hi[7] as usize] as i32,
                grid[idx_hi[6] as usize] as i32,
                grid[idx_hi[5] as usize] as i32,
                grid[idx_hi[4] as usize] as i32,
                grid[idx_hi[3] as usize] as i32,
                grid[idx_hi[2] as usize] as i32,
                grid[idx_hi[1] as usize] as i32,
                grid[idx_hi[0] as usize] as i32,
            );

            let s0 = u16::from_le_bytes([signs[signs_off], signs[signs_off + 1]]);
            let s1 = u16::from_le_bytes([signs[signs_off + 2], signs[signs_off + 3]]);
            let mut aux256 = _mm256_set1_epi32((s0 as u32 | ((s1 as u32) << 16)) as i32);
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            let s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            let q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            let s2 = u16::from_le_bytes([signs[signs_off + 4], signs[signs_off + 5]]);
            let s3 = u16::from_le_bytes([signs[signs_off + 6], signs[signs_off + 7]]);
            let mut aux256 = _mm256_set1_epi32((s2 as u32 | ((s3 as u32) << 16)) as i32);
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            let s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            let q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs_off += 8;

            let dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            let dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            let ls1 = (scales_bytes[ib32 / 2] & 0xf) as u16;
            let ls2 = (scales_bytes[ib32 / 2] >> 4) as u16;
            let p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16((2 * ls1 + 1) as i16));
            let p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16((2 * ls2 + 1) as i16));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
            ib32 += 2;
        }
        accumf = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
            accumf,
        );
    }
    hsum_float_8_llama(accumf)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_dot_iq1_s_q8k_avx2(iq1s_data: &[u8], q8k: &[super::BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    let grid = &super::iq_tables::IQ1S_GRID;
    let nb = q8k.len();

    let mut accum = _mm256_setzero_ps();
    let mut accum1 = 0f32;
    for i in 0..nb {
        let boff = i * super::BLOCK_IQ1_S_SIZE;
        if boff + super::BLOCK_IQ1_S_SIZE > iq1s_data.len() {
            break;
        }
        let qs = &iq1s_data[boff + 2..boff + 2 + 32];
        let qh_bytes = &iq1s_data[boff + 34..boff + 34 + 16];
        let q8 = &q8k[i].qs;
        let bsums = &q8k[i].bsums;

        let mut sumi = _mm256_setzero_si256();
        let mut sumi1 = 0i32;
        let mut qs_off = 0usize;
        let mut q8_off = 0usize;
        let mut ib = 0usize;
        while ib < 8usize {
            let qh0 = u16::from_le_bytes([qh_bytes[ib * 2], qh_bytes[ib * 2 + 1]]);
            let qh1 = u16::from_le_bytes([qh_bytes[ib * 2 + 2], qh_bytes[ib * 2 + 3]]);
            let q1b_1 = _mm256_set_epi64x(
                grid[(qs[qs_off + 3] as u16 | ((qh0 >> 1) & 0x700)) as usize] as i64,
                grid[(qs[qs_off + 2] as u16 | ((qh0 << 2) & 0x700)) as usize] as i64,
                grid[(qs[qs_off + 1] as u16 | ((qh0 << 5) & 0x700)) as usize] as i64,
                grid[(qs[qs_off] as u16 | ((qh0 << 8) & 0x700)) as usize] as i64,
            );
            let q1b_2 = _mm256_set_epi64x(
                grid[(qs[qs_off + 7] as u16 | ((qh1 >> 1) & 0x700)) as usize] as i64,
                grid[(qs[qs_off + 6] as u16 | ((qh1 << 2) & 0x700)) as usize] as i64,
                grid[(qs[qs_off + 5] as u16 | ((qh1 << 5) & 0x700)) as usize] as i64,
                grid[(qs[qs_off + 4] as u16 | ((qh1 << 8) & 0x700)) as usize] as i64,
            );
            qs_off += 8;
            let q8b_1 = _mm256_loadu_si256(q8.as_ptr().add(q8_off) as *const __m256i);
            let q8b_2 = _mm256_loadu_si256(q8.as_ptr().add(q8_off + 32) as *const __m256i);
            q8_off += 64;

            let dot1 = mul_add_epi8(q1b_1, q8b_1);
            let dot2 = mul_add_epi8(q1b_2, q8b_2);
            let ls1 = (2 * ((qh0 >> 12) & 7) + 1) as i16;
            let ls2 = (2 * ((qh1 >> 12) & 7) + 1) as i16;
            let p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1));
            let p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(ls2));

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p1, p2));
            sumi1 += (bsums[2 * ib] as i32 + bsums[2 * ib + 1] as i32)
                * (if qh0 & 0x8000 != 0 { -1 } else { 1 })
                * ls1 as i32
                + (bsums[2 * ib + 2] as i32 + bsums[2 * ib + 3] as i32)
                    * (if qh1 & 0x8000 != 0 { -1 } else { 1 })
                    * ls2 as i32;
            ib += 2;
        }

        let d = q8k[i].d
            * super::f16_to_f32(u16::from_le_bytes([iq1s_data[boff], iq1s_data[boff + 1]]));
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        accum1 += d * sumi1 as f32;
    }

    hsum_float_8_llama(accum) + 0.125f32 * accum1
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_dot_iq1_m_q8k_avx2(iq1m_data: &[u8], q8k: &[super::BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    let grid = &super::iq_tables::IQ1S_GRID;
    let nb = q8k.len();

    let mone8 = _mm256_set1_epi8(1);
    let mtwo8 = _mm256_set1_epi8(2);
    let mut accum1 = _mm256_setzero_ps();
    let mut accum2 = _mm256_setzero_ps();
    for i in 0..nb {
        let boff = i * super::BLOCK_IQ1_M_SIZE;
        if boff + super::BLOCK_IQ1_M_SIZE > iq1m_data.len() {
            break;
        }
        let qs = &iq1m_data[boff..boff + 32];
        let qh = &iq1m_data[boff + 32..boff + 48];
        let sc_bytes = &iq1m_data[boff + 48..boff + 56];
        let sc_u16 = [
            u16::from_le_bytes([sc_bytes[0], sc_bytes[1]]),
            u16::from_le_bytes([sc_bytes[2], sc_bytes[3]]),
            u16::from_le_bytes([sc_bytes[4], sc_bytes[5]]),
            u16::from_le_bytes([sc_bytes[6], sc_bytes[7]]),
        ];
        let scale_u16 = (sc_u16[0] >> 12)
            | ((sc_u16[1] >> 8) & 0x00f0)
            | ((sc_u16[2] >> 4) & 0x0f00)
            | (sc_u16[3] & 0xf000);
        let q8 = &q8k[i].qs;

        // 16 3-bit sub-scales, lane order identical to llama.cpp's
        // `_mm256_srlv_epi64` construction (srlv_epi64 is AVX-512; the values
        // are built scalar-side instead).
        let mut scales16 = [0i16; 16];
        let shifts = [0u32, 6, 3, 9];
        for lane in 0..16usize {
            let idx = lane % 4;
            let sh = shifts[lane / 4];
            scales16[lane] = (2 * ((sc_u16[idx] >> sh) & 7) + 1) as i16;
        }
        let scales = _mm256_loadu_si256(scales16.as_ptr() as *const __m256i);

        let mut scales_idx1 = _mm256_set1_epi16(0x0100);
        let mut scales_idx2 = _mm256_add_epi8(scales_idx1, _mm256_set1_epi8(8));

        let mut sumi1 = _mm256_setzero_si256();
        let mut sumi2 = _mm256_setzero_si256();
        let mut qs_off = 0usize;
        let mut qh_off = 0usize;
        let mut q8_off = 0usize;
        let mut ib = 0usize;
        while ib < 8usize {
            let qh0 = qh[qh_off] as u16;
            let qh1 = qh[qh_off + 1] as u16;
            let qh2 = qh[qh_off + 2] as u16;
            let qh3 = qh[qh_off + 3] as u16;
            let q1b_1 = _mm256_set_epi64x(
                grid[(qs[qs_off + 3] as u16 | ((qh1 << 4) & 0x700)) as usize] as i64,
                grid[(qs[qs_off + 2] as u16 | ((qh1 << 8) & 0x700)) as usize] as i64,
                grid[(qs[qs_off + 1] as u16 | ((qh0 << 4) & 0x700)) as usize] as i64,
                grid[(qs[qs_off] as u16 | ((qh0 << 8) & 0x700)) as usize] as i64,
            );
            let q1b_2 = _mm256_set_epi64x(
                grid[(qs[qs_off + 7] as u16 | ((qh3 << 4) & 0x700)) as usize] as i64,
                grid[(qs[qs_off + 6] as u16 | ((qh3 << 8) & 0x700)) as usize] as i64,
                grid[(qs[qs_off + 5] as u16 | ((qh2 << 4) & 0x700)) as usize] as i64,
                grid[(qs[qs_off + 4] as u16 | ((qh2 << 8) & 0x700)) as usize] as i64,
            );

            let delta1 = _mm256_set_epi64x(
                if qh1 & 0x80 != 0 {
                    -1i64
                } else {
                    0x0101_0101_0101_0101u64 as i64
                },
                if qh1 & 0x08 != 0 {
                    -1i64
                } else {
                    0x0101_0101_0101_0101u64 as i64
                },
                if qh0 & 0x80 != 0 {
                    -1i64
                } else {
                    0x0101_0101_0101_0101u64 as i64
                },
                if qh0 & 0x08 != 0 {
                    -1i64
                } else {
                    0x0101_0101_0101_0101u64 as i64
                },
            );
            let delta2 = _mm256_set_epi64x(
                if qh3 & 0x80 != 0 {
                    -1i64
                } else {
                    0x0101_0101_0101_0101u64 as i64
                },
                if qh3 & 0x08 != 0 {
                    -1i64
                } else {
                    0x0101_0101_0101_0101u64 as i64
                },
                if qh2 & 0x80 != 0 {
                    -1i64
                } else {
                    0x0101_0101_0101_0101u64 as i64
                },
                if qh2 & 0x08 != 0 {
                    -1i64
                } else {
                    0x0101_0101_0101_0101u64 as i64
                },
            );

            let q8b_1 = _mm256_loadu_si256(q8.as_ptr().add(q8_off) as *const __m256i);
            let q8b_2 = _mm256_loadu_si256(q8.as_ptr().add(q8_off + 32) as *const __m256i);
            q8_off += 64;

            let dot1 = mul_add_epi8(q1b_1, q8b_1);
            let dot2 = mul_add_epi8(q1b_2, q8b_2);
            let dot3 = _mm256_maddubs_epi16(mone8, _mm256_sign_epi8(q8b_1, delta1));
            let dot4 = _mm256_maddubs_epi16(mone8, _mm256_sign_epi8(q8b_2, delta2));

            let scale1 = _mm256_shuffle_epi8(scales, scales_idx1);
            let scale2 = _mm256_shuffle_epi8(scales, scales_idx2);

            scales_idx1 = _mm256_add_epi8(scales_idx1, mtwo8);
            scales_idx2 = _mm256_add_epi8(scales_idx2, mtwo8);

            let p1 = _mm256_madd_epi16(dot1, scale1);
            let p2 = _mm256_madd_epi16(dot2, scale2);
            let p3 = _mm256_madd_epi16(dot3, scale1);
            let p4 = _mm256_madd_epi16(dot4, scale2);

            sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(p1, p2));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_add_epi32(p3, p4));

            qs_off += 8;
            qh_off += 4;
            ib += 2;
        }

        let d = _mm256_set1_ps(q8k[i].d * super::f16_to_f32(scale_u16));

        accum1 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi1), accum1);
        accum2 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi2), accum2);
    }

    hsum_float_8_llama(accum1) + 0.125f32 * hsum_float_8_llama(accum2)
}

#[cfg(all(target_arch = "x86_64", test))]
mod iq_avx2_tests {
    // populated below in this change set
}
