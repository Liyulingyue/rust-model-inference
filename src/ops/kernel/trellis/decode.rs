//! TRL1..TRL6 encode a circular, MSB-first bitstream per 16 × 16 tile.
//! Eight consecutive output rows at one column have windows exactly `bits` words apart.
use std::sync::OnceLock;

#[derive(Clone, Copy, Default)]
struct Window {
    hi: [i32; 8],
    lo: [i32; 8],
    offset: u32,
}

fn windows(bits: usize) -> &'static [Window; 32] {
    static TABLE: OnceLock<[[Window; 32]; 6]> = OnceLock::new();
    &TABLE.get_or_init(|| {
        std::array::from_fn(|b| {
            let bits = b + 1;
            std::array::from_fn(|i| {
                let col = i / 2;
                let half = i % 2;
                let p = 8 * ((col % 8) / 2) + 4 * half + 2 * (col / 8) + col % 2;
                let start = ((p + 1) * bits + 256 * bits - 16) % (256 * bits);
                let hi = std::array::from_fn(|row| ((start / 32 + row * bits) % (8 * bits)) as i32);
                Window {
                    hi,
                    lo: hi.map(|w| (w + 1) % (8 * bits) as i32),
                    offset: (start % 32) as u32,
                }
            })
        })
    })[bits - 1]
}

#[inline]
fn value(tile: &[u8], window: &Window, row: usize) -> f32 {
    let word = |i: i32| u32::from_le_bytes(tile[4 * i as usize..][..4].try_into().unwrap());
    let pair = (u64::from(word(window.hi[row])) << 32) | u64::from(word(window.lo[row]));
    let state = ((pair >> (48 - window.offset)) & 0xffff) as u32;
    let x = state.wrapping_mul(0x83dc_d12d);
    let sum = ((x >> 2) & 63) + ((x >> 10) & 63) + ((x >> 18) & 63) + ((x >> 26) & 63);
    (sum as i32 - 126) as f32
}

pub(super) fn embedding_row(
    codes: &[u8],
    bits: usize,
    row: usize,
    scales: &[f32],
    out: &mut [f32],
) {
    let windows = windows(bits);
    for ((tile, block), &scale) in codes
        .chunks_exact(32 * bits)
        .zip(out.chunks_exact_mut(16))
        .zip(scales)
    {
        for (col, x) in block.iter_mut().enumerate() {
            *x = value(tile, &windows[2 * col + row / 8], row % 8) * scale;
        }
    }
}

pub(super) fn dot16(codes: &[u8], bits: usize, input: &[f32], out: &mut [f32]) {
    let windows = windows(bits);
    #[cfg(target_arch = "x86_64")]
    if !crate::ops::scalar_mode() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: validated tiles have 8 * bits words, and every gather index
        // is reduced modulo that count. Exactly sixteen results are written.
        unsafe { dot16_avx2(codes, bits, input, windows, out) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if !crate::ops::scalar_mode() && std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: each tile contains 8 * bits little-endian words. All four
        // indices in a gather are reduced modulo this count in windows().
        unsafe { dot16_neon(codes, bits, input, windows, out) };
        return;
    }
    let mut sums = [0.0; 16];
    for (tile, x) in codes.chunks_exact(32 * bits).zip(input.chunks_exact(16)) {
        for (col, &a) in x.iter().enumerate() {
            for (r, sum) in sums.iter_mut().enumerate() {
                *sum += a * value(tile, &windows[2 * col + r / 8], r % 8);
            }
        }
    }
    out.copy_from_slice(&sums);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot16_neon(
    codes: &[u8],
    bits: usize,
    input: &[f32],
    windows: &[Window; 32],
    out: &mut [f32],
) {
    use std::arch::aarch64::*;
    let mut sums = [vdupq_n_f32(0.0); 4];
    for (tile, x) in codes.chunks_exact(32 * bits).zip(input.chunks_exact(16)) {
        let words = tile.as_ptr().cast::<u32>();
        for (col, &a) in x.iter().enumerate() {
            for (quarter, sum) in sums.iter_mut().enumerate() {
                let w = &windows[2 * col + quarter / 2];
                let gather = |indices: &[i32; 8]| {
                    let lanes: [u32; 4] = std::array::from_fn(|lane| {
                        u32::from_le(
                            words
                                .add(indices[4 * (quarter % 2) + lane] as usize)
                                .read_unaligned(),
                        )
                    });
                    vld1q_u32(lanes.as_ptr())
                };
                let shift = w.offset as i32 - 16;
                let mut state = vshlq_u32(gather(&w.hi), vdupq_n_s32(shift));
                if shift > 0 {
                    state = vorrq_u32(state, vshlq_u32(gather(&w.lo), vdupq_n_s32(shift - 32)));
                }
                let hash = vmulq_n_u32(vandq_u32(state, vdupq_n_u32(65535)), 0x83dc_d12d);
                let digits = vandq_u32(vshrq_n_u32::<2>(hash), vdupq_n_u32(0x3f3f3f3f));
                let total = vpaddlq_u16(vpaddlq_u8(vreinterpretq_u8_u32(digits)));
                let grid = vsubq_s32(vreinterpretq_s32_u32(total), vdupq_n_s32(126));
                *sum = vfmaq_n_f32(*sum, vcvtq_f32_s32(grid), a);
            }
        }
    }
    for (dst, sum) in out.chunks_exact_mut(4).zip(sums) {
        vst1q_f32(dst.as_mut_ptr(), sum);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot16_avx2(
    codes: &[u8],
    bits: usize,
    input: &[f32],
    windows: &[Window; 32],
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    let mut sums = [_mm256_setzero_ps(); 2];
    for (tile, x) in codes.chunks_exact(32 * bits).zip(input.chunks_exact(16)) {
        let words = tile.as_ptr().cast::<i32>();
        for (c, &a) in x.iter().enumerate() {
            for (half, sum) in sums.iter_mut().enumerate() {
                let w = &windows[2 * c + half];
                let high =
                    _mm256_i32gather_epi32::<4>(words, _mm256_loadu_si256(w.hi.as_ptr().cast()));
                let state = if w.offset <= 16 {
                    _mm256_srl_epi32(high, _mm_cvtsi32_si128((16 - w.offset) as i32))
                } else {
                    let low = _mm256_i32gather_epi32::<4>(
                        words,
                        _mm256_loadu_si256(w.lo.as_ptr().cast()),
                    );
                    _mm256_or_si256(
                        _mm256_sll_epi32(high, _mm_cvtsi32_si128((w.offset - 16) as i32)),
                        _mm256_srl_epi32(low, _mm_cvtsi32_si128((48 - w.offset) as i32)),
                    )
                };
                let hash = _mm256_mullo_epi32(
                    _mm256_and_si256(state, _mm256_set1_epi32(65535)),
                    _mm256_set1_epi32(0x83dc_d12du32 as i32),
                );
                let digits =
                    _mm256_and_si256(_mm256_srli_epi32::<2>(hash), _mm256_set1_epi32(0x3f3f3f3f));
                let pairs = _mm256_maddubs_epi16(digits, _mm256_set1_epi8(1));
                let grid = _mm256_sub_epi32(
                    _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)),
                    _mm256_set1_epi32(126),
                );
                *sum = _mm256_add_ps(
                    *sum,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(grid), _mm256_set1_ps(a)),
                );
            }
        }
    }
    for (half, sum) in sums.into_iter().enumerate() {
        _mm256_storeu_ps(out.as_mut_ptr().add(8 * half), sum);
    }
}
