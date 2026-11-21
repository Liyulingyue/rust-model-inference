//! Q4_0 AVX2 (x86_64) matmul kernel.
//!
//! 32-element blocks, 18-byte layout (2-byte F16 scale + 16-byte nibbles).
//! Strategy (mirrors `q8_0/avx2.rs`):
//!  1. Tile 4 weight rows per pass; the shared Q8 input block is loaded once
//!     per block and reused by all 4 rows.
//!  2. Extract low + high nibbles as u8 in [0, 15] and interleave to
//!     `[lo[0..16], hi[0..16]]` so nibble k lines up with input element k.
//!  3. `_mm256_maddubs_epi16` (u8 × i8) + `_mm256_madd_epi16` give the
//!     per-lane partial dot: each of the 8 i32 lanes sums exactly 4 products.
//!  4. Correction without materializing `(nibble - 8)`: the per-lane partial
//!     of `sum(input)` is computed once per block and subtracted scaled by 8 —
//!     both sides are exact i32, so `dot_lane = nib_lane - 8 * y_lane` is the
//!     exact per-lane dot of that block.
//!  5. Per-lane f32 accumulation via `_mm256_fmadd_ps(ds, dot_lane, acc)`; a
//!     single horizontal sum at the END of each row (the old kernel did a
//!     `hsum_epi32` per block and fed one scalar into lane 7 of the
//!     accumulator — 1/8 of the FMA throughput wasted).
//!
//! **Precision contract**: near-bit-exact vs scalar — the per-block integer
//! dot is exact on both sides; only the f32 accumulation order differs
//! (scalar: whole-block dot × d·scale, sequential; here: 8 independent
//! per-lane accumulators). Same class as the Q4_K / Q8_0 kernels, which use
//! rel < 1e-3 parity bounds.

#![cfg(target_arch = "x86_64")]

use crate::ops::f16_to_f32;

#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_q4_0_vs_q8_0_avx2(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(n_in % 32, 0);
    let blocks_per_row = n_in / 32;
    let row_stride = blocks_per_row * 18;
    let m0f128 = _mm_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);

    let w_ptr = weight.as_ptr();
    let iq_ptr = input_q8.as_ptr();
    let sc_ptr = input_scales.as_ptr();
    let out_ptr = output.as_mut_ptr();

    let n_rows = row_end - row_start;
    let full4 = n_rows / 4;
    for tile in 0..full4 {
        let r0 = row_start + tile * 4;
        let off0 = r0 * row_stride;
        let off1 = (r0 + 1) * row_stride;
        let off2 = (r0 + 2) * row_stride;
        let off3 = (r0 + 3) * row_stride;
        let mut cv0 = _mm256_setzero_ps();
        let mut cv1 = _mm256_setzero_ps();
        let mut cv2 = _mm256_setzero_ps();
        let mut cv3 = _mm256_setzero_ps();
        for b in 0..blocks_per_row {
            // The Q8 input block and its per-lane partial sums (exact i32;
            // lane k sums input elements 4k..4k+3) are shared by all 4 rows.
            let qy = _mm256_loadu_si256(iq_ptr.add(b * 32) as *const __m256i);
            let y_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(qy));
            let y_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(qy, 1));
            let y_lane = _mm256_add_epi32(
                _mm256_madd_epi16(ones, y_lo16),
                _mm256_madd_epi16(ones, y_hi16),
            );
            let y8 = _mm256_slli_epi32(y_lane, 3);

            let bd = *sc_ptr.add(b);
            let p0 = w_ptr.add(off0 + b * 18);
            let p1 = w_ptr.add(off1 + b * 18);
            let p2 = w_ptr.add(off2 + b * 18);
            let p3 = w_ptr.add(off3 + b * 18);

            let ds0 = f16_to_f32(std::ptr::read_unaligned(p0 as *const u16)) * bd;
            let ds1 = f16_to_f32(std::ptr::read_unaligned(p1 as *const u16)) * bd;
            let ds2 = f16_to_f32(std::ptr::read_unaligned(p2 as *const u16)) * bd;
            let ds3 = f16_to_f32(std::ptr::read_unaligned(p3 as *const u16)) * bd;

            // Per-lane exact block dot for each row: nib_lane - 8 * y_lane.
            let dot0 = q4_0_row_block_dot(
                _mm_loadu_si128(p0.add(2) as *const __m128i),
                qy,
                y8,
                m0f128,
                ones,
            );
            let dot1 = q4_0_row_block_dot(
                _mm_loadu_si128(p1.add(2) as *const __m128i),
                qy,
                y8,
                m0f128,
                ones,
            );
            let dot2 = q4_0_row_block_dot(
                _mm_loadu_si128(p2.add(2) as *const __m128i),
                qy,
                y8,
                m0f128,
                ones,
            );
            let dot3 = q4_0_row_block_dot(
                _mm_loadu_si128(p3.add(2) as *const __m128i),
                qy,
                y8,
                m0f128,
                ones,
            );

            cv0 = _mm256_fmadd_ps(_mm256_set1_ps(ds0), _mm256_cvtepi32_ps(dot0), cv0);
            cv1 = _mm256_fmadd_ps(_mm256_set1_ps(ds1), _mm256_cvtepi32_ps(dot1), cv1);
            cv2 = _mm256_fmadd_ps(_mm256_set1_ps(ds2), _mm256_cvtepi32_ps(dot2), cv2);
            cv3 = _mm256_fmadd_ps(_mm256_set1_ps(ds3), _mm256_cvtepi32_ps(dot3), cv3);
        }
        let base_out = tile * 4;
        *out_ptr.add(base_out) = crate::ops::dot::hsum_ps(cv0);
        *out_ptr.add(base_out + 1) = crate::ops::dot::hsum_ps(cv1);
        *out_ptr.add(base_out + 2) = crate::ops::dot::hsum_ps(cv2);
        *out_ptr.add(base_out + 3) = crate::ops::dot::hsum_ps(cv3);
    }

    // Trailing rows: same per-lane structure, one row at a time.
    for (out_idx, row) in (row_start + full4 * 4..row_end).enumerate() {
        let row_off = row * row_stride;
        let mut acc = _mm256_setzero_ps();
        for b in 0..blocks_per_row {
            let qy = _mm256_loadu_si256(iq_ptr.add(b * 32) as *const __m256i);
            let y_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(qy));
            let y_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(qy, 1));
            let y_lane = _mm256_add_epi32(
                _mm256_madd_epi16(ones, y_lo16),
                _mm256_madd_epi16(ones, y_hi16),
            );
            let off = row_off + b * 18;
            let d_b = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off) as *const u16));
            let dot = q4_0_row_block_dot(
                _mm_loadu_si128(w_ptr.add(off + 2) as *const __m128i),
                qy,
                _mm256_slli_epi32(y_lane, 3),
                m0f128,
                ones,
            );
            acc = _mm256_fmadd_ps(
                _mm256_set1_ps(d_b * *sc_ptr.add(b)),
                _mm256_cvtepi32_ps(dot),
                acc,
            );
        }
        *out_ptr.add(full4 * 4 + out_idx) = crate::ops::dot::hsum_ps(acc);
    }
}

/// Per-lane exact Q4_0 × Q8 block dot: `nib_lane - 8 * y_lane`, where each of
/// the 8 i32 lanes sums exactly 4 products. The caller multiplies lane-wise by
/// `d * scale` (f32) and accumulates; no horizontal reduction per block.
#[inline(always)]
unsafe fn q4_0_row_block_dot(
    q4_bytes: std::arch::x86_64::__m128i,
    qy: std::arch::x86_64::__m256i,
    y8: std::arch::x86_64::__m256i,
    m0f128: std::arch::x86_64::__m128i,
    ones: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let lo = _mm_and_si128(q4_bytes, m0f128);
    let hi = _mm_and_si128(_mm_srli_epi16(q4_bytes, 4), m0f128);
    let q4_unpacked = _mm256_set_m128i(hi, lo);
    let nib_lane = _mm256_madd_epi16(ones, _mm256_maddubs_epi16(q4_unpacked, qy));
    _mm256_sub_epi32(nib_lane, y8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::kernel::q4_0::matmul_q4_0_scalar_range;

    fn build_block(scale: f32, low: u8, hi: u8) -> Vec<u8> {
        assert!(low < 16 && hi < 16);
        let mut v = Vec::with_capacity(18);
        let s_bits = crate::ops::f32_to_f16(scale).to_le_bytes();
        v.extend_from_slice(&s_bits);
        for _ in 0..16 {
            v.push((hi << 4) | low);
        }
        v
    }

    fn q8_input_zero() -> Vec<u8> {
        vec![0u8; 32]
    }
    fn q8_input_max() -> Vec<u8> {
        // All +127 (i8). Byte value 0x7F = 127.
        vec![0x7Fu8; 32]
    }
    fn q8_input_min() -> Vec<u8> {
        // All -128 (i8). Byte value 0x80 = 128 as u8, but i8 interpretation = -128.
        vec![0x80u8; 32]
    }
    fn q8_input_linspace() -> Vec<u8> {
        // 0, 1, 2, ..., 127, -128, -127, ..., -1 (32 values, wrapping i8)
        (0..32).map(|i| (i as i8) as u8).collect()
    }
    fn q8_input_alt() -> Vec<u8> {
        // alternating +max, -max
        (0..32)
            .map(|i| if i % 2 == 0 { 0x7F } else { 0x80 })
            .collect()
    }

    /// Parity bound follows the Q4_K / Q8_0 kernel convention (rel < 1e-3):
    /// the per-block integer dot is exact on both sides, only the f32
    /// accumulation order differs (scalar sequential vs 8 per-lane
    /// accumulators + final horizontal sum). ULP drift is printed for
    /// visibility.
    fn assert_avx2_eq_scalar(label: &str, weight: &[u8], q8: &[u8], scales: &[f32]) {
        let n_in = q8.len();
        let n_out = weight.len() / (n_in / 32 * 18);
        let mut avx2_out = vec![0.0f32; n_out];
        let mut scalar_out = vec![0.0f32; n_out];
        unsafe {
            matmul_q4_0_vs_q8_0_avx2(weight, q8, scales, &mut avx2_out, n_in, 0, n_out);
        }
        matmul_q4_0_scalar_range(weight, q8, scales, &mut scalar_out, n_in, n_out, 0, 1);
        let max_diff = avx2_out
            .iter()
            .zip(scalar_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_scalar = scalar_out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let rel = if max_scalar > 1e-3 {
            max_diff / max_scalar
        } else {
            max_diff
        };
        for (i, (a, b)) in avx2_out.iter().zip(scalar_out.iter()).enumerate() {
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            let diff = (a_bits as i32).wrapping_sub(b_bits as i32).unsigned_abs();
            eprintln!(
                "{} row {}: avx2={} (bits {:x}) scalar={} (bits {:x}) diff={} ULP",
                label, i, a, a_bits, b, b_bits, diff
            );
        }
        assert!(
            rel < 1e-3,
            "{} parity: max_diff={} max_scalar={} rel={}",
            label,
            max_diff,
            max_scalar,
            rel
        );
    }

    #[test]
    fn parity_block_uniform_zero_nibble_zero_input() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        // weight = -8 (nibble 0), input = 0 → dot = 0
        let weight = build_block(1.0, 0, 0);
        let q8 = q8_input_zero();
        let scales = vec![1.0f32];
        assert_avx2_eq_scalar("zero-nibble/zero-input", &weight, &q8, &scales);
    }

    #[test]
    fn parity_block_uniform_max_nibble_zero_input() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        // weight = 7 (nibble 15), input = 0 → dot = 0
        let weight = build_block(1.0, 15, 15);
        let q8 = q8_input_zero();
        let scales = vec![1.0f32];
        assert_avx2_eq_scalar("max-nibble/zero-input", &weight, &q8, &scales);
    }

    #[test]
    fn parity_block_zero_nibble_max_input() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        // weight = -8, input = +127 → dot = -8 * sum(127 over 32)
        // = -8 * (16*127 + 16*127) = -8 * 4064 = -32512
        let weight = build_block(1.0, 0, 0);
        let q8 = q8_input_max();
        let scales = vec![1.0f32];
        assert_avx2_eq_scalar("zero-nibble/max-input", &weight, &q8, &scales);
    }

    #[test]
    fn parity_block_zero_nibble_min_input() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        // weight = -8, input = -128 → dot = -8 * sum(-128 over 32) = -8 * -4096 = 32768
        let weight = build_block(1.0, 0, 0);
        let q8 = q8_input_min();
        let scales = vec![1.0f32];
        assert_avx2_eq_scalar("zero-nibble/min-input", &weight, &q8, &scales);
    }

    #[test]
    fn parity_block_max_nibble_max_input() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        // weight = 7, input = +127
        let weight = build_block(1.0, 15, 15);
        let q8 = q8_input_max();
        let scales = vec![1.0f32];
        assert_avx2_eq_scalar("max-nibble/max-input", &weight, &q8, &scales);
    }

    #[test]
    fn parity_block_mixed_nibbles_linspace() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        // low=0, hi=15 (so weight[0]=-8, weight[1]=7, alternating)
        let weight = build_block(0.7, 0, 15);
        let q8 = q8_input_linspace();
        let scales = vec![1.3f32];
        assert_avx2_eq_scalar("mixed-nibble/linspace", &weight, &q8, &scales);
    }

    #[test]
    fn parity_block_alt_input_signs() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        let weight = build_block(2.5, 8, 3);
        let q8 = q8_input_alt();
        let scales = vec![0.42f32];
        assert_avx2_eq_scalar("alt-input-signs", &weight, &q8, &scales);
    }

    #[test]
    fn parity_many_blocks_random() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        // 16 blocks per row, 4 rows.
        let mut weight = Vec::new();
        let mut state: u64 = 0xdead_beef_1234_5678;
        for _ in 0..4 {
            for _ in 0..16 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let d = 0.01 + (state >> 33) as f32 / u32::MAX as f32;
                let s_bits = crate::ops::f32_to_f16(d).to_le_bytes();
                weight.extend_from_slice(&s_bits);
                for _ in 0..16 {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    weight.push((state >> 33) as u8);
                }
            }
        }
        let q8: Vec<u8> = (0..512)
            .map(|i| ((i as i32 % 31) - 15) as i8 as u8)
            .collect();
        let scales: Vec<f32> = (0..16).map(|b| 0.01 + (b as f32) * 0.001).collect();
        assert_avx2_eq_scalar("random-4x512", &weight, &q8, &scales);
    }

    #[test]
    fn parity_real_model_q4_0_weights() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        let model_path = std::env::var("Q4_MODEL")
            .unwrap_or_else(|_| "../models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf".to_string());
        let loader = match crate::core::loader::GGUFLoader::from_file(&model_path) {
            Ok(l) => l,
            Err(_) => return,
        };
        let tensor = loader
            .tensors()
            .iter()
            .find(|t| {
                t.name == "blk.0.attn_q.weight"
                    && t.ggml_type == crate::core::tensor::GGMLType::Q4_0
            })
            .expect("blk.0.attn_q.weight Q4_0 not found");
        let weight = loader.tensor_slice(&tensor.name).unwrap();
        let n_in = tensor.dims[0] as usize;
        let n_out = tensor.dims[1] as usize;
        let blocks = n_in / 32;
        let q8: Vec<u8> = (0..blocks * 32)
            .map(|i| ((i as i32 % 31) - 15) as i8 as u8)
            .collect();
        let scales: Vec<f32> = (0..blocks).map(|b| 0.01 + (b as f32) * 0.001).collect();
        assert_avx2_eq_scalar("model-blk0-attnq", &weight, &q8, &scales);
    }
}
