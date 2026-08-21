//! Q8_0 AVX2 (x86_64) matmul kernels.
//!
//! Phase 2.7-final: split from `ops::matmul`. Selected at runtime by
//! `dispatch::matmul_q8_0_quantized_range` and by `q8_0_dot_row` (scalar).

#![cfg(target_arch = "x86_64")]

use crate::ops::{f16_to_f32, hsum_ps};

/// AVX2+FMA single-row Q8_0 dot product.
///
/// Returns `weight[row, :] ⋅ input` for one row. Caller passes
/// `blocks_per_row` and `row_stride` already precomputed (the scalar
/// `q8_0_dot_row` calculates these once).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn q8_0_dot_row_avx2(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    n_in: usize,
    row: usize,
    blocks_per_row: usize,
    row_stride: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let ones = _mm256_set1_epi16(1);
    let row_off = row * row_stride;
    let mut acc = _mm256_setzero_ps();
    for b in 0..blocks_per_row {
        let w_off = row_off + b * 34;
        let d = f16_to_f32(u16::from_le_bytes([
            *weight.as_ptr().add(w_off),
            *weight.as_ptr().add(w_off + 1),
        ])) * *input_scales.as_ptr().add(b);
        let d_v = _mm256_set1_ps(d);
        let qx = _mm256_loadu_si256(weight.as_ptr().add(w_off + 2) as *const __m256i);
        let qy = _mm256_loadu_si256(input_q8.as_ptr().add(b * 32) as *const __m256i);
        let ax = _mm256_sign_epi8(qx, qx);
        let sy = _mm256_sign_epi8(qy, qx);
        let dot = _mm256_maddubs_epi16(ax, sy);
        let summed = _mm256_madd_epi16(ones, dot);
        acc = _mm256_fmadd_ps(d_v, _mm256_cvtepi32_ps(summed), acc);
    }
    hsum_ps(acc)
}

/// AVX2 Q8_0 × Q8_0 matmul over a row range (prequantized input).
///
/// Tiles rows in groups of 4 for ILP; trailing rows handled by a single-row
/// fallback. Selected by `dispatch::matmul_q8_0_quantized_range` on
/// x86_64 hosts with AVX2.
#[inline(never)]
pub unsafe fn matmul_q8_0_vs_q8_0_avx2(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;
    let blocks_per_row = n_in / 32;
    let row_stride = blocks_per_row * 34;
    let ones = _mm256_set1_epi16(1);
    let n_rows = row_end - row_start;
    let w_ptr = weight.as_ptr();
    let sc_ptr = input_scales.as_ptr();
    let out_ptr = output.as_mut_ptr();

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
            let qy = _mm256_loadu_si256(input_q8.as_ptr().add(b * 32) as *const __m256i);
            let bd = *sc_ptr.add(b);

            let p0 = w_ptr.add(off0 + b * 34);
            let p1 = w_ptr.add(off1 + b * 34);
            let p2 = w_ptr.add(off2 + b * 34);
            let p3 = w_ptr.add(off3 + b * 34);

            let a0_d = std::ptr::read_unaligned(p0 as *const u16);
            let a1_d = std::ptr::read_unaligned(p1 as *const u16);
            let a2_d = std::ptr::read_unaligned(p2 as *const u16);
            let a3_d = std::ptr::read_unaligned(p3 as *const u16);

            let da = _mm_mul_ps(
                _mm_cvtph_ps(_mm_set_epi16(
                    0,
                    0,
                    0,
                    0,
                    a3_d as i16,
                    a2_d as i16,
                    a1_d as i16,
                    a0_d as i16,
                )),
                _mm_set1_ps(bd),
            );
            let s0 = _mm256_broadcastss_ps(da);
            let s1 = _mm256_broadcastss_ps(_mm_shuffle_ps(da, da, 0x55));
            let s2 = _mm256_broadcastss_ps(_mm_shuffle_ps(da, da, 0xAA));
            let s3 = _mm256_broadcastss_ps(_mm_shuffle_ps(da, da, 0xFF));

            let av0 = _mm256_loadu_si256(p0.add(2) as *const __m256i);
            let av1 = _mm256_loadu_si256(p1.add(2) as *const __m256i);
            let av2 = _mm256_loadu_si256(p2.add(2) as *const __m256i);
            let av3 = _mm256_loadu_si256(p3.add(2) as *const __m256i);

            let ax0 = _mm256_sign_epi8(av0, av0);
            let ax1 = _mm256_sign_epi8(av1, av1);
            let ax2 = _mm256_sign_epi8(av2, av2);
            let ax3 = _mm256_sign_epi8(av3, av3);
            let sy0 = _mm256_sign_epi8(qy, av0);
            let sy1 = _mm256_sign_epi8(qy, av1);
            let sy2 = _mm256_sign_epi8(qy, av2);
            let sy3 = _mm256_sign_epi8(qy, av3);

            cv0 = _mm256_fmadd_ps(
                s0,
                _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, _mm256_maddubs_epi16(ax0, sy0))),
                cv0,
            );
            cv1 = _mm256_fmadd_ps(
                s1,
                _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, _mm256_maddubs_epi16(ax1, sy1))),
                cv1,
            );
            cv2 = _mm256_fmadd_ps(
                s2,
                _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, _mm256_maddubs_epi16(ax2, sy2))),
                cv2,
            );
            cv3 = _mm256_fmadd_ps(
                s3,
                _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, _mm256_maddubs_epi16(ax3, sy3))),
                cv3,
            );
        }
        let base_out = tile * 4;
        *out_ptr.add(base_out) = hsum_ps(cv0);
        *out_ptr.add(base_out + 1) = hsum_ps(cv1);
        *out_ptr.add(base_out + 2) = hsum_ps(cv2);
        *out_ptr.add(base_out + 3) = hsum_ps(cv3);
    }

    for (out_idx, j) in (row_start + full4 * 4..row_end).enumerate() {
        let row_off = j * row_stride;
        let mut acc = _mm256_setzero_ps();
        for b in 0..blocks_per_row {
            let w_off = row_off + b * 34;
            let wd = std::ptr::read_unaligned(w_ptr.add(w_off) as *const u16);
            let d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(wd as i16))) * *sc_ptr.add(b);
            let d_v = _mm256_set1_ps(d);
            let qx = _mm256_loadu_si256(w_ptr.add(w_off + 2) as *const __m256i);
            let qy = _mm256_loadu_si256(input_q8.as_ptr().add(b * 32) as *const __m256i);
            let ax = _mm256_sign_epi8(qx, qx);
            let sy = _mm256_sign_epi8(qy, qx);
            let dot = _mm256_maddubs_epi16(ax, sy);
            let summed = _mm256_madd_epi16(ones, dot);
            acc = _mm256_fmadd_ps(d_v, _mm256_cvtepi32_ps(summed), acc);
        }
        *out_ptr.add(full4 * 4 + out_idx) = hsum_ps(acc);
    }
}

/// AVX2 Q8_0 × f32 matmul over a row range (raw f32 input).
///
/// Used by `matmul_q8_0_via_q8_parallel` (legacy API) and the older
/// `matmul_q8_0_parallel` f32-input path.
pub unsafe fn matmul_q8_0_avx2_range(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;
    let blocks_per_row = n_in / 32;
    let row_stride = blocks_per_row * 34;
    for (out_idx, j) in (row_start..row_end).enumerate() {
        let row_off = j * row_stride;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        for b in 0..blocks_per_row {
            let off = row_off + b * 34;
            let d = f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
            let d_v = _mm256_set1_ps(d);
            let qs = weight.as_ptr().add(off + 2);
            let inp = input.as_ptr().add(b * 32);
            let q0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(qs as *const __m128i));
            let q1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(8) as *const __m128i));
            let q2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(16) as *const __m128i));
            let q3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(24) as *const __m128i));
            let i0 = _mm256_loadu_ps(inp);
            let i1 = _mm256_loadu_ps(inp.add(8));
            let i2 = _mm256_loadu_ps(inp.add(16));
            let i3 = _mm256_loadu_ps(inp.add(24));
            acc0 = _mm256_fmadd_ps(_mm256_mul_ps(d_v, _mm256_cvtepi32_ps(q0)), i0, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_mul_ps(d_v, _mm256_cvtepi32_ps(q1)), i1, acc1);
            acc0 = _mm256_fmadd_ps(_mm256_mul_ps(d_v, _mm256_cvtepi32_ps(q2)), i2, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_mul_ps(d_v, _mm256_cvtepi32_ps(q3)), i3, acc1);
        }
        let s = _mm256_add_ps(acc0, acc1);
        output[out_idx] = hsum_ps(s);
    }
}#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::kernel::q8_0::scalar::matmul_q8_0_quantized_scalar_range;

    fn q8_uniform_block(weight: i8, scale: f32) -> Vec<u8> {
        let mut v = Vec::with_capacity(34);
        let s = crate::ops::f32_to_f16(scale).to_le_bytes();
        v.extend_from_slice(&s);
        for _ in 0..32 { v.push(weight as u8); }
        v
    }

    fn assert_avx2_matches_scalar(label: &str, weight: &[u8], q8: &[u8], scales: &[f32]) {
        let n_in = q8.len();
        let n_out = weight.len() / (n_in / 32 * 34);
        let mut avx2_out = vec![0.0f32; n_out];
        let mut scalar_out = vec![0.0f32; n_out];
        unsafe {
            matmul_q8_0_vs_q8_0_avx2(weight, q8, scales, &mut avx2_out, n_in, 0, n_out);
        }
        matmul_q8_0_quantized_scalar_range(weight, q8, scales, &mut scalar_out, n_in, n_out, 0);
        let max_diff = avx2_out
            .iter()
            .zip(scalar_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_scalar = scalar_out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let rel = if max_scalar > 1e-3 { max_diff / max_scalar } else { max_diff };
        eprintln!(
            "[{}] {}x{} max_diff={} rel={}",
            label, n_out, n_in, max_diff, rel
        );
        assert!(
            rel < 1e-3,
            "{} AVX2 diverged: max_diff={} rel={}",
            label, max_diff, rel
        );
    }

    #[test]
    fn q8_0_avx2_matches_scalar_uniform() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        let mut weight = Vec::new();
        for _ in 0..128 {
            weight.extend(q8_uniform_block(1, 0.5));
        }
        let q8: Vec<u8> = (0..(128 * 32))
            .map(|i| (i as i8) as u8)
            .collect();
        let scales: Vec<f32> = (0..128).map(|i| 0.01 + (i as f32) * 0.001).collect();
        assert_avx2_matches_scalar("q8_0-uniform-128", &weight, &q8, &scales);
    }

    #[test]
    fn q8_0_avx2_matches_scalar_real_model() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        let loader = match crate::core::loader::GGUFLoader::from_file(
            "../models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
        ) {
            Ok(l) => l,
            Err(_) => return,
        };
        let tensor = loader
            .tensors()
            .iter()
            .find(|t| t.name == "blk.0.attn_q.weight" && t.ggml_type == crate::core::tensor::GGMLType::Q8_0)
            .expect("blk.0.attn_q.weight Q8_0 not found");
        let weight = loader.tensor_slice(&tensor.name).unwrap();
        let n_in = tensor.dims[0] as usize;
        let n_out = tensor.dims[1] as usize;
        let blocks = n_in / 32;
        let q8: Vec<u8> = (0..blocks * 32)
            .map(|i| (i as i8) as u8)
            .collect();
        let scales: Vec<f32> = (0..blocks).map(|b| 0.01 + (b as f32) * 0.001).collect();
        assert_avx2_matches_scalar("q8_0-model-blk0-attnq", &weight, &q8, &scales);
    }
}
