//! BF16 × Q8_0 AVX2 (x86_64) matmul kernel.
//!
//! Hot-path for production LLM inference: every matmul in the dense
//! transformer layers routes through `forward_prequantized`, which
//! dequantizes the activations from Q8_0 blocks and consumes the BF16
//! weight matrix.  The scalar fallback in `scalar::forward_q8_rows_scalar`
//! was the only implementation until this kernel landed.
//!
//! Strategy: unpack 8 BF16 lanes per AVX2 register via
//! `_mm256_cvtepu16_epi32` + `_mm256_slli_epi32` (zero-extend and shift
//! into the F32 high half), unpack the matching 8 i8 input lanes via
//! `_mm_loadl_epi64` + `_mm256_cvtepi8_epi32` + `_mm256_cvtepi32_ps`,
//! look up the per-block Q8 scale (`input_scales[col / 32]`) and apply
//! it once at the end of each 32-lane block.  Cross-block boundaries
//! inside an 8-lane AVX2 iteration are handled by reading two scales
//! (one for the low lanes, one for the high lanes) and blending with
//! `_mm256_blendv_ps`.

use std::arch::x86_64::*;

use crate::ops::kernel::simd_avx2::hsum256;

/// Q8 block size — every 32 input bytes share one scale.
const Q8_BLOCK: usize = 32;

/// BF16 × Q8_0 packed matmul.  Layout mirrors `scalar::forward_q8_rows_scalar`:
/// `weight` is `[n_out rows × n_in cols]` of BF16 (2 bytes/element, little-endian),
/// `input_q8` is `n_in` int8, `input_scales[k]` scales `input_q8[k*32..(k+1)*32]`.
/// `output` is either `[n_out]` (full rows) or `[row_end - row_start]`
/// (a single thread's slice); the dispatch in `BF16Kernel::forward_q8_rows`
/// picks the slice variant when `ith > 0`.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn matmul_bf16_vs_q8_avx2(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    row_start: usize,
    row_end: usize,
) {
    let w_ptr = weight.as_ptr();
    let q_ptr = input_q8.as_ptr();
    let s_ptr = input_scales.as_ptr();
    let out_ptr = output.as_mut_ptr();

    assert!(
        input_q8.len() >= n_in,
        "BF16×Q8 AVX2: input_q8 len {} < n_in {n_in}",
        input_q8.len()
    );
    assert!(
        input_scales.len() >= n_in.div_ceil(32),
        "BF16×Q8 AVX2: input_scales len {} < required {}",
        input_scales.len(),
        n_in.div_ceil(32)
    );

    let mut out_local = 0usize;
    for row in row_start..row_end {
        let row_byte = row * n_in * 2;
        let mut acc = _mm256_setzero_ps();

        let mut col = 0usize;
        while col + 8 <= n_in {
            // Per-block Q8 scale: the 8 lanes may straddle a Q8 block
            // boundary (Q8_BLOCK = 32).  Look up the block index for the
            // low and high half (each 4 lanes) and blend the scale vector.
            let block_lo = col / Q8_BLOCK;
            let block_hi = (col + 4) / Q8_BLOCK;
            let scale_lo = _mm_set1_ps(*s_ptr.add(block_lo));
            let scale_hi = _mm_set1_ps(*s_ptr.add(block_hi));
            let scale_128 = _mm_blendv_ps(
                scale_lo,
                scale_hi,
                _mm_castsi128_ps(_mm_set_epi32(-1, -1, -1, -1)),
            );
            let scale = _mm256_insertf128_ps(_mm256_castps128_ps256(scale_lo), scale_128, 1);

            // unpack 8 BF16 weight lanes
            let chunk = _mm_loadu_si128(w_ptr.add(row_byte + col * 2) as *const __m128i);
            let w_i32 = _mm256_cvtepu16_epi32(chunk);
            let w_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(w_i32, 16));

            // unpack 8 i8 input lanes
            let q_raw = _mm_loadl_epi64(q_ptr.add(col) as *const __m128i);
            let q_i32 = _mm256_cvtepi8_epi32(q_raw);
            let q_f32 = _mm256_cvtepi32_ps(q_i32);

            // accumulate (w * q * scale) per lane
            acc = _mm256_fmadd_ps(w_f32, _mm256_mul_ps(q_f32, scale), acc);
            col += 8;
        }

        // Scalar tail for the leftover columns.
        while col < n_in {
            let bits = u16::from_le_bytes([
                *w_ptr.add(row_byte + col * 2),
                *w_ptr.add(row_byte + col * 2 + 1),
            ]);
            let w_val = crate::ops::bf16_to_f32(bits);
            let q_val = *q_ptr.add(col) as i8 as f32;
            let scale = *s_ptr.add(col / Q8_BLOCK);
            acc = _mm256_fmadd_ps(
                _mm256_set1_ps(w_val * q_val * scale),
                _mm256_set1_ps(1.0),
                acc,
            );
            col += 1;
        }

        let total = hsum256(acc);
        let output_index = if output.len() >= n_out {
            row
        } else {
            out_local
        };
        *out_ptr.add(output_index) = total;
        out_local += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::matmul_bf16_vs_q8_avx2;
    use crate::ops::kernel::bf16::scalar::forward_q8_rows_scalar;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for &v in values {
            out.extend(crate::ops::f32_to_bf16(v).to_le_bytes());
        }
        out
    }

    fn assert_avx2_eq_scalar(label: &str, weight: &[u8], input: &[f32], n_in: usize, n_out: usize) {
        let mut avx2_out = vec![0.0f32; n_out];
        let mut scalar_out = vec![0.0f32; n_out];
        // Quantize the F32 input to Q8 (32-element blocks).
        let mut input_q8 = vec![0u8; n_in];
        let mut input_scales = vec![0.0f32; n_in.div_ceil(32)];
        crate::ops::quantize_q8_0_into(input, n_in, &mut input_q8, &mut input_scales);
        unsafe {
            matmul_bf16_vs_q8_avx2(
                weight,
                &input_q8,
                &input_scales,
                &mut avx2_out,
                n_in,
                n_out,
                0,
                n_out,
            );
        }
        forward_q8_rows_scalar(
            weight,
            &input_q8,
            &input_scales,
            &mut scalar_out,
            n_in,
            n_out,
            0,
            1,
        );
        for (i, (a, s)) in avx2_out.iter().zip(scalar_out.iter()).enumerate() {
            let denom = s.abs().max(1.0);
            assert!(
                (a - s).abs() / denom < 1e-2,
                "{label} row {i}: avx2={a} scalar={s} diff={}",
                (a - s).abs()
            );
        }
    }

    #[test]
    fn avx2_matches_scalar_small() {
        let n_in = 16;
        let n_out = 4;
        let weight = bf16_bytes(&{
            let mut v = Vec::new();
            for r in 0..n_out {
                for c in 0..n_in {
                    v.push(((r * n_in + c) as f32 * 0.07).sin());
                }
            }
            v
        });
        let input: Vec<f32> = (0..n_in).map(|i| (i as f32 - 8.0) * 0.05).collect();
        assert_avx2_eq_scalar("small 4x16", &weight, &input, n_in, n_out);
    }

    #[test]
    fn avx2_matches_scalar_large() {
        let n_in = 256;
        let n_out = 8;
        let weight = bf16_bytes(&{
            let mut v = Vec::new();
            for r in 0..n_out {
                for c in 0..n_in {
                    v.push(((r * n_in + c) % 37) as f32 * 0.01 - 0.2);
                }
            }
            v
        });
        let input: Vec<f32> = (0..n_in).map(|i| ((i % 23) as f32 - 11.0) * 0.03).collect();
        assert_avx2_eq_scalar("large 8x256", &weight, &input, n_in, n_out);
    }

    #[test]
    fn avx2_matches_scalar_cross_block() {
        // Force 8-lane groups that straddle a Q8 block boundary (32-wide).
        let n_in = 96; // 3 blocks
        let n_out = 2;
        let weight = bf16_bytes(&{
            let mut v = Vec::new();
            for r in 0..n_out {
                for c in 0..n_in {
                    v.push((c as f32 * 0.04 + r as f32 * 0.1).cos());
                }
            }
            v
        });
        let input: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.13).sin()).collect();
        assert_avx2_eq_scalar("cross-block 2x96", &weight, &input, n_in, n_out);
    }

    #[test]
    fn avx2_matches_scalar_tail() {
        // n_in not a multiple of 8 → exercises the scalar tail.
        let n_in = 41;
        let n_out = 3;
        let weight = bf16_bytes(&{
            let mut v = Vec::new();
            for r in 0..n_out {
                for c in 0..n_in {
                    v.push(((r * n_in + c) as f32 * 0.01).sin());
                }
            }
            v
        });
        let input: Vec<f32> = (0..n_in).map(|i| i as f32 * 0.03).collect();
        assert_avx2_eq_scalar("tail 3x41", &weight, &input, n_in, n_out);
    }
}
