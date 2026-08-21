//! Q4_0 AVX2 (x86_64) matmul kernel.
//!
//! Phase 2.7-final + 2026-08: 32-element blocks, 18-byte layout
//! (2-byte F16 scale + 16-byte nibbles). Strategy:
//!
//! 1. Extract low + high nibbles as u8 in [0, 15].
//! 2. Interleave to q4_unpacked = [lo[0..16], hi[0..16]] (32 bytes).
//! 3. Use `_mm256_maddubs_epi16` (u8 × i8 → i16 madd of adjacent pairs).
//! 4. Use `_mm256_madd_epi16` (i16 × i16 → i32) with ones to sum pairs.
//! 5. hsum 8 i32 lanes → nib_total (one i32 per block).
//! 6. hsum input separately to get sum_input (also i32).
//! 7. Corrected dot = nib_total - 8 * sum_input (exact i32 arithmetic).
//! 8. Multiply by d * scale in f32 with explicit `_mm256_mul_ps` + `_mm256_add_ps`
//!    (no `_mm256_fmadd_ps`) to match scalar's mul+add rounding exactly.
//!
//! **Precision contract**: bit-exact with the scalar implementation, including
//! for edge cases (all-zero Q8, all-127 Q8, all-0 nibble, all-15 nibble).

#![cfg(target_arch = "x86_64")]

use crate::ops::f16_to_f32;

/// Q4_0 × Q8_0 matmul over a row range. AVX2, no FMA.
#[target_feature(enable = "avx2")]
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
    let low_mask = _mm256_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);

    let w_ptr = weight.as_ptr();
    let iq_ptr = input_q8.as_ptr();
    let sc_ptr = input_scales.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for (out_idx, row) in (row_start..row_end).enumerate() {
        let row_off = row * row_stride;
        let mut acc: f32 = 0.0;

        let mut b = 0;
        // Process 2 blocks per iteration (process 2 blocks of weights
        // sequentially). The 2-block batch shares no state — we just
        // read 2 weights blocks and 2 input blocks in parallel and add
        // to acc — except the JIT may issue them as a single fused loop.
        while b + 2 <= blocks_per_row {
            let off0 = row_off + b * 18;
            let off1 = row_off + (b + 1) * 18;

            let d_b0 = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off0) as *const u16));
            let d_b1 = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off1) as *const u16));
            let si_b0 = *sc_ptr.add(b);
            let si_b1 = *sc_ptr.add(b + 1);

            let (dc0, dc1) = q4_0_block_pair_dot(
                _mm_loadu_si128(w_ptr.add(off0 + 2) as *const __m128i),
                _mm256_loadu_si256(iq_ptr.add(b * 32) as *const __m256i),
                _mm_loadu_si128(w_ptr.add(off1 + 2) as *const __m128i),
                _mm256_loadu_si256(iq_ptr.add((b + 1) * 32) as *const __m256i),
                low_mask,
                ones,
            );

            // CRITICAL: accumulate block-by-block to match scalar's
            // `sum += prod` order. Doing `acc += prod0 + prod1` here
            // would be `acc += (prod0 + prod1)` — a different order
            // than scalar's `sum = (sum + prod0) + prod1`, and f32
            // addition is not associative, so they can differ by 1 ULP.
            acc += dc0 * d_b0 * si_b0;
            acc += dc1 * d_b1 * si_b1;
            b += 2;
        }

        // Trailing single block.
        while b < blocks_per_row {
            let off = row_off + b * 18;
            let d_b = f16_to_f32(std::ptr::read_unaligned(w_ptr.add(off) as *const u16));
            let si_b = *sc_ptr.add(b);
            let dc = q4_0_block_dot(
                _mm_loadu_si128(w_ptr.add(off + 2) as *const __m128i),
                _mm256_loadu_si256(iq_ptr.add(b * 32) as *const __m256i),
                low_mask,
                ones,
            );
            let prod = dc * d_b * si_b;
            acc += prod;
            b += 1;
        }

        *out_ptr.add(out_idx) = acc;
    }
}
/// Per-block Q4×Q8 SIMD dot. Returns corrected dot product as f32:
///   `sum((nibble - 8) * input) * d * scale`
/// but computed without ever materializing `(nibble - 8)` per element:
///   = (sum(nibble × input) - 8 × sum(input)) × d × scale
///
/// We return the **f32 product** (already multiplied by d and scale), so
/// the caller just adds to its row accumulator without further scale math.
#[inline(always)]
unsafe fn q4_0_block_dot(
    q4_bytes: std::arch::x86_64::__m128i,
    q8_input: std::arch::x86_64::__m256i,
    low_mask: std::arch::x86_64::__m256i,
    ones: std::arch::x86_64::__m256i,
) -> f32 {
    use std::arch::x86_64::*;

    // Split 16 nibble bytes into 16 lo-nibbles and 16 hi-nibbles.
    let lo = _mm_and_si128(q4_bytes, _mm256_castsi256_si128(low_mask));
    let hi = _mm_and_si128(_mm_srli_epi16(q4_bytes, 4), _mm256_castsi256_si128(low_mask));

    // Interleave lo | hi into a single 32-byte vector matching q8_input layout.
    // q8_input layout (scalar convention): [q8[0..16], q8[16..32]]
    // So q4 must be: [lo[0..16], hi[0..16]] so that
    //   q4_unpacked[i] pairs with q8_input[i] in scalar.
    let q4_unpacked = _mm256_set_m128i(hi, lo);

    // u8 × i8 → i16 madd of adjacent pairs (16 i16 results).
    let prod16 = _mm256_maddubs_epi16(q4_unpacked, q8_input);

    // i16 × i16 → i32 pairs-summed (8 i32 results).
    let prod32 = _mm256_madd_epi16(ones, prod16);

    // Sum input separately: sign-extend i8 → i16, then madd with ones.
    let y_lo_input = _mm256_castsi256_si128(q8_input);
    let y_hi_input = _mm256_extracti128_si256(q8_input, 1);
    let y_lo16 = _mm256_cvtepi8_epi16(y_lo_input);
    let y_hi16 = _mm256_cvtepi8_epi16(y_hi_input);
    let y_lo32 = _mm256_madd_epi16(ones, y_lo16);
    let y_hi32 = _mm256_madd_epi16(ones, y_hi16);
    let y_sum32 = _mm256_add_epi32(y_lo32, y_hi32);

    let nib_total = hsum_epi32(prod32);
    let sum_total = hsum_epi32(y_sum32);

    // Corrected dot = sum((nibble - 8) * input).
    // NOTE: scalar computes (x - 8) * y per element; we compute
    //   sum(nibble × y) - 8 × sum(y).
    // Both are exact i32 arithmetic (i32 cannot overflow at our magnitudes).
    let corrected_i32 = nib_total - 8 * sum_total;

    corrected_i32 as f32
}

/// Process 2 blocks: returns (f32_dot_block0, f32_dot_block1).
/// Caller still needs to multiply by per-block d and scale, then accumulate.

/// Process 2 blocks: returns (f32_dot_block0, f32_dot_block1).
/// Caller still needs to multiply by per-block d and scale, then accumulate.
#[inline(always)]
unsafe fn q4_0_block_pair_dot(
    q4_b0: std::arch::x86_64::__m128i,
    q8_b0: std::arch::x86_64::__m256i,
    q4_b1: std::arch::x86_64::__m128i,
    q8_b1: std::arch::x86_64::__m256i,
    low_mask: std::arch::x86_64::__m256i,
    ones: std::arch::x86_64::__m256i,
) -> (f32, f32) {
    let dc0 = q4_0_block_dot(q4_b0, q8_b0, low_mask, ones);
    let dc1 = q4_0_block_dot(q4_b1, q8_b1, low_mask, ones);
    (dc0, dc1)
}

/// Horizontal sum of 8 i32 lanes in a __m256i → single i32.
#[inline(always)]
unsafe fn hsum_epi32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extracti128_si256(v, 1);
    let lo128 = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(hi128, lo128);
    let t = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let r = _mm_add_epi32(t, _mm_srli_si128(t, 4));
    _mm_cvtsi128_si32(r)
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
        (0..32)
            .map(|i| (i as i8) as u8)
            .collect()
    }
    fn q8_input_alt() -> Vec<u8> {
        // alternating +max, -max
        (0..32).map(|i| if i % 2 == 0 { 0x7F } else { 0x80 }).collect()
    }

    fn assert_avx2_eq_scalar(label: &str, weight: &[u8], q8: &[u8], scales: &[f32]) {
        let n_in = q8.len();
        let n_out = weight.len() / (n_in / 32 * 18);
        let mut avx2_out = vec![0.0f32; n_out];
        let mut scalar_out = vec![0.0f32; n_out];
        unsafe {
            matmul_q4_0_vs_q8_0_avx2(weight, q8, scales, &mut avx2_out, n_in, 0, n_out);
        }
        matmul_q4_0_scalar_range(weight, q8, scales, &mut scalar_out, n_in, n_out, 0, 1);
        for (i, (a, b)) in avx2_out.iter().zip(scalar_out.iter()).enumerate() {
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            let diff = (a_bits as i32).wrapping_sub(b_bits as i32).unsigned_abs();
            assert!(
                a_bits == b_bits,
                "{} row {}: avx2={} (bits {:x}) scalar={} (bits {:x}) diff={}",
                label, i, a, a_bits, b, b_bits, diff
            );
        }
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
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let d = 0.01 + (state >> 33) as f32 / u32::MAX as f32;
                let s_bits = crate::ops::f32_to_f16(d).to_le_bytes();
                weight.extend_from_slice(&s_bits);
                for _ in 0..16 {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
            .find(|t| t.name == "blk.0.attn_q.weight" && t.ggml_type == crate::core::tensor::GGMLType::Q4_0)
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