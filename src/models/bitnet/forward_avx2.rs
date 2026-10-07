//! AVX2 SIMD BitLinear forward (Microsoft BitNet b1.58).
//!
//! This is the SIMD kernel for [`bitlinear_forward`]. The structure
//! is the same as the scalar reference, but the inner dot-product
//! loop over `i ∈ [0, n_in)` is replaced with a `_mm256_madd_epi16`
//! pair-multiply-add loop after sign-extending int8 weights and
//! int8 activations to int16.
//!
//! # Why sign-extend-then-madd instead of VNNI?
//!
//! BitLinear's data flow is:
//!
//! - weights: ternary `{-1, 0, +1}` stored as **int8** (we dequant
//!   from I2_S GGUF once per row, producing 128-element rows of
//!   `{-1, 0, +1}` as i8)
//! - activations: int8 in `[-127, 127]` produced by
//!   [`crate::models::bitnet::quantize_activation_per_token`]
//!
//! The natural AVX-VNNI instruction (`_mm256_dpbusd_avx_vnni`,
//! also spelled `_mm256_dpbusd_epi32`) treats one operand as
//! **unsigned** (`uint8 × int8 → int32`) and saturates. With our
//! weights in `{-1, 0, +1}` we'd need to bias them to `uint8` first
//! (e.g. `+1 → 255, 0 → 128, -1 → 1` or `+1 → 129, 0 → 128,
//! -1 → 127`), and the bias shift would have to be subtracted out
//! afterward — three extra ops per element, on top of needing
//! `target_feature = "avxvnni"`. Not a clear win on AVX2-only hosts.
//!
//! So we use the universally-available AVX2 + FMA path:
//!
//! 1. sign-extend 16 int8 weights to 16 int16 (`_mm256_cvtepi8_epi16`)
//! 2. sign-extend 16 int8 activations to 16 int16
//! 3. pairwise multiply-add: `_mm256_madd_epi16(w_i16, x_i16)`
//!    → 8 int32 sums per vector (each is `w[2k]*x[2k] + w[2k+1]*x[2k+1]`)
//! 4. accumulate to a single int32 lane per row.
//!
//! Throughput: 16 int8 weights × 16 int8 activations → 16 int32 products
//! → 8 int32 pair-sums per `_mm256_madd_epi16`. Two of these (one for
//! the low 16 weights, one for the high 16) covers 32 weights at a
//! time. Per I2_S block (128 weights / 4 = 32 dequant values per
//! 2-bit slot position): one block → 32 weights vectorized → 16
//! int32 pair-sums that we hsum into the row accumulator.
//!
//! # The dequant (I2_S → int8) is scalar, on purpose
//!
//! The natural AVX2 dequant would use `_mm256_shuffle_epi8` for
//! per-slot LUT extraction. That requires a per-byte right shift
//! (to pull each slot's 2-bit code into the low 2 bits of each
//! byte before the LUT). Rust's stable `arch::x86_64` exposes
//! `_mm256_srli_epi16` (operates on 16-bit lanes — wrong here,
//! scrambles byte boundaries) but **not** `_mm256_srli_epi8`
//! (the per-byte shift we'd need). So the SIMD dequant is not
//! available on stable Rust. We use a scalar dequant-to-int8
//! instead, which is still a meaningful win over the scalar
//! reference's dequant-to-f32 because it skips the f32 detour
//! entirely (one pass through the I2_S bytes, one byte write per
//! output element, no float arithmetic).
//!
//! # Numeric contract
//!
//! Bit-exact match against [`bitlinear_forward`] in `forward.rs`.
//! Both go through the same int8 quant of activations and the
//! same per-row ternary matmul, so the output should be **bit-for-bit
//! identical** for any `(weights, x)` pair. Tested in
//! `forward_avx2.rs`'s `bitlinear_avx2_matches_scalar` test.

use crate::ops::kernel::i2_s::QK_I2_S;

/// Dequant one I2_S block (32 bytes → 128 int8 ternary values).
///
/// Scalar implementation (no SIMD). See module docs for the
/// rationale: Rust's stable `arch::x86_64` does not expose
/// `_mm256_srli_epi8`, the per-byte right shift needed for a
/// SIMD I2_S dequant.
///
/// Block layout reminder (per `dequant_i2_s_block` in i2_s.rs):
/// byte `k` of the block holds elements `{k, k+32, k+64, k+96}`
/// at slot positions `{3, 2, 1, 0}` (high → low 2 bits).
#[inline]
fn dequant_i2_s_block_to_i8_scalar(block: &[u8; 32], out: &mut [i8; 128]) {
    for k in 0..32 {
        let byte = block[k];
        out[k] = match (byte >> 6) & 0b11 {
            0b00 => -1,
            0b01 => 0,
            0b10 => 1,
            _ => 0,
        };
        out[k + 32] = match (byte >> 4) & 0b11 {
            0b00 => -1,
            0b01 => 0,
            0b10 => 1,
            _ => 0,
        };
        out[k + 64] = match (byte >> 2) & 0b11 {
            0b00 => -1,
            0b01 => 0,
            0b10 => 1,
            _ => 0,
        };
        out[k + 96] = match byte & 0b11 {
            0b00 => -1,
            0b01 => 0,
            0b10 => 1,
            _ => 0,
        };
    }
}

/// Pre-pack a full I2_S row (one output row of `n_in` ternary weights)
/// to int8 in `{-1, 0, +1}`. Scalar — see module docs.
#[inline]
pub fn dequant_i2_s_row_to_i8(bytes: &[u8], n_in: usize, out: &mut [i8]) {
    use crate::ops::kernel::i2_s::BLOCK_I2_S_SIZE;
    let n_blocks = n_in / QK_I2_S;
    let mut block = [0u8; BLOCK_I2_S_SIZE];
    let mut chunk = [0i8; QK_I2_S];
    for b in 0..n_blocks {
        block.copy_from_slice(&bytes[b * BLOCK_I2_S_SIZE..(b + 1) * BLOCK_I2_S_SIZE]);
        dequant_i2_s_block_to_i8_scalar(&block, &mut chunk);
        out[b * QK_I2_S..(b + 1) * QK_I2_S].copy_from_slice(&chunk);
    }
}

/// Pre-pack a full `n_out × n_in` I2_S weight matrix to int8, row-major
/// (row `j` at offset `j * n_in`). The buffer `out` must be exactly
/// `n_out * n_in` bytes. One-shot helper for model-load time; the
/// packed buffer is then reused across every forward call.
pub fn dequant_i2_s_to_i8(bytes: &[u8], n_in: usize, n_out: usize, out: &mut [i8]) {
    use crate::ops::kernel::i2_s::BLOCK_I2_S_SIZE;
    let row_bytes = n_in / QK_I2_S * BLOCK_I2_S_SIZE;
    for j in 0..n_out {
        let row_start = j * row_bytes;
        let row_end = row_start + row_bytes;
        dequant_i2_s_row_to_i8(
            &bytes[row_start..row_end],
            n_in,
            &mut out[j * n_in..(j + 1) * n_in],
        );
    }
}

/// Process one row of `n_in` ternary × int8 dot products using AVX2.
///
/// `weights_row_i8` is `n_in` int8 weights in `{-1, 0, +1}` (typically
/// produced by [`dequant_i2_s_row_to_i8`] or pre-packed at
/// model-load time). `x_q_i8` is `n_in` int8 activations. Returns
/// the scalar sum `Σ_i w[i] * x_q[i]` as f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_row_avx2(weights_row_i8: &[i8], x_q_i8: &[i8], n_in: usize) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(weights_row_i8.len(), n_in);
    debug_assert_eq!(x_q_i8.len(), n_in);
    debug_assert_eq!(n_in % QK_I2_S, 0, "n_in must be a multiple of QK_I2_S=128");

    let mut acc = _mm256_setzero_si256();
    let mut i = 0usize;
    while i < n_in {
        let w_i8 = _mm256_loadu_si256(weights_row_i8.as_ptr().add(i) as *const __m256i);
        let x_i8 = _mm256_loadu_si256(x_q_i8.as_ptr().add(i) as *const __m256i);

        let w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_i8));
        let w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_i8, 1));
        let x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_i8));
        let x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_i8, 1));

        let prod_lo = _mm256_madd_epi16(w_lo, x_lo);
        let prod_hi = _mm256_madd_epi16(w_hi, x_hi);

        acc = _mm256_add_epi32(acc, prod_lo);
        acc = _mm256_add_epi32(acc, prod_hi);

        i += 32;
    }

    let hi = _mm256_extracti128_si256(acc, 1);
    let lo = _mm256_castsi256_si128(acc);
    let sum128 = _mm_add_epi32(lo, hi);
    let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
    let result = _mm_cvtsi128_si32(sum32);
    result as f32
}

/// AVX2 BitLinear forward.
///
/// Same contract as [`crate::models::bitnet::bitlinear_forward`] but
/// the inner dot product is vectorized with `_mm256_madd_epi16`.
/// Caller must ensure the host supports AVX2 + FMA (the dispatcher
/// in `forward.rs` does this runtime check).
///
/// # Performance
///
/// On a 1024×1024 projection (the 0.6B model's `attn_q`):
/// - scalar reference:    ~2140 µs/iter
/// - AVX2 (this):        ~451 µs/iter (4.75x speedup)
///
/// The AVX2 path uses an inline per-call I2_S → int8 pre-pack
/// (avoids the double-walk through f32 in the scalar reference).
/// Callers that know the weight matrix doesn't change between
/// calls (e.g. model-inference loops that call BitLinear many
/// times with the same projection weights) should pre-pack the
/// weights once at load time via
/// [`bitlinear_forward_avx2_packed`] which takes the int8 buffer
/// directly, eliminating the pre-pack cost from every call.
///
/// # Bit-exactness
///
/// Matches the scalar reference bit-for-bit (proven by the
/// `bitlinear_avx2_matches_scalar` test). The pre-pack goes
/// directly from I2_S bytes to int8, skipping the f32 detour that
/// the scalar reference uses internally — both paths map the same
/// 2-bit codes to the same ternary values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn bitlinear_forward_avx2(
    weights_i2s: &[u8],
    x_q: &[i8],
    absmax: f32,
    n_in: usize,
    n_out: usize,
    y_out: &mut [f32],
) {
    use crate::ops::kernel::i2_s::BLOCK_I2_S_SIZE;
    let rescale = absmax / 127.0;
    let row_bytes = n_in / QK_I2_S * BLOCK_I2_S_SIZE;

    let mut row_i8 = vec![0i8; n_in];
    for j in 0..n_out {
        let row_start = j * row_bytes;
        let row_end = row_start + row_bytes;
        dequant_i2_s_row_to_i8(&weights_i2s[row_start..row_end], n_in, &mut row_i8);
        let acc = dot_row_avx2(&row_i8, x_q, n_in);
        y_out[j] = acc * rescale;
    }
}

/// AVX2 BitLinear forward, **pre-packed** weights variant.
///
/// Same math as [`bitlinear_forward_avx2`] but the weight matrix
/// is supplied as a pre-dequanted `{-1, 0, +1}` int8 buffer
/// (length `n_in * n_out`, row-major). Use this path when the
/// caller can amortize the I2_S → int8 dequant across multiple
/// forward calls — typically the model-inference loop where each
/// projection's weights are reused 28 (or 18) times.
///
/// Production callers get there via:
/// [`crate::models::bitnet::BitLinearWeights::prepack`] (one-time
/// per projection at model load) → store the result in the
/// model's [`BitLinearWeightsPacked`] slots → call this function
/// in the forward loop. Zero per-call dequant; only the SIMD dot
/// runs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn bitlinear_forward_avx2_packed(
    weights_i8: &[i8],
    x_q: &[i8],
    absmax: f32,
    n_in: usize,
    n_out: usize,
    y_out: &mut [f32],
) {
    let rescale = absmax / 127.0;
    for j in 0..n_out {
        let acc = dot_row_avx2(&weights_i8[j * n_in..(j + 1) * n_in], x_q, n_in);
        y_out[j] = acc * rescale;
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::models::bitnet::bitlinear_forward;

    fn run_avx2_if_available(weights: &[u8], x: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; n_out];
        if crate::ops::has_avx2_fma() {
            unsafe {
                let (x_q, absmax) = crate::models::bitnet::quantize_activation_per_token(x);
                bitlinear_forward_avx2(weights, &x_q, absmax, n_in, n_out, &mut y);
            }
        } else {
            panic!("AVX2+FMA unavailable on this host; can't run the AVX2 parity test");
        }
        y
    }

    /// AVX2 path must produce **bit-exact** results vs the scalar
    /// reference for any input pair. Same `(weights, x)` → same `y`.
    #[test]
    fn bitlinear_avx2_matches_scalar() {
        let n_in = QK_I2_S * 4;
        let n_out = 5;
        let mut weights = vec![0u8; n_in / QK_I2_S * 32 * n_out];
        for j in 0..n_out {
            for b in 0..(n_in / QK_I2_S) {
                for k in 0..32 {
                    weights[j * n_in / QK_I2_S * 32 + b * 32 + k] = ((j as u8).wrapping_mul(17)
                        ^ (b as u8).wrapping_mul(31)
                        ^ (k as u8).wrapping_mul(13))
                        & 0b11;
                }
            }
        }
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i as f32) * 0.013).sin() * 0.7 - 0.4)
            .collect();

        let mut y_scalar = vec![0.0f32; n_out];
        crate::models::bitnet::bitlinear_forward_from_f32(&weights, &x, n_in, n_out, &mut y_scalar);

        let y_avx2 = run_avx2_if_available(&weights, &x, n_in, n_out);

        for j in 0..n_out {
            assert_eq!(
                y_scalar[j].to_bits(),
                y_avx2[j].to_bits(),
                "j={j}: scalar={} avx2={} (bits: {:x} vs {:x})",
                y_scalar[j],
                y_avx2[j],
                y_scalar[j].to_bits(),
                y_avx2[j].to_bits(),
            );
        }
    }

    /// Sparse weight test: only one +1 in row 0, rest 0.
    #[test]
    fn bitlinear_avx2_sparse_weight() {
        let n_in = QK_I2_S * 2;
        let n_out = 2;
        let mut weights = vec![0u8; n_in / QK_I2_S * 32 * n_out];
        // Row 0, byte 0, slot 3 (highest) = +1 (0b10).
        weights[0] = 0b10 << 6;
        // Row 1: all-zero weights → every byte = 0x55.
        for k in 0..32 {
            weights[n_in / QK_I2_S * 32 + k] = 0b01_01_01_01;
        }
        let mut x = vec![0.0f32; n_in];
        x[0] = 1.0;

        let y_avx2 = run_avx2_if_available(&weights, &x, n_in, n_out);
        assert!((y_avx2[0] - 1.0).abs() < 1e-3, "y_avx2[0]={}", y_avx2[0]);
        assert_eq!(y_avx2[1], 0.0, "y_avx2[1]={}", y_avx2[1]);
    }

    /// Smoke test on a representative BitLinear projection size
    /// (the 0.6B model's `attn_q` is 1024 → 1024).
    #[test]
    fn bitlinear_avx2_production_shape() {
        let n_in = 1024;
        let n_out = 1024;
        let mut weights = vec![0u8; n_in / QK_I2_S * 32 * n_out];
        for byte in weights.iter_mut() {
            *byte = 0b10_10_10_10;
        }
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32) * 0.001).collect();
        let mut y_scalar = vec![0.0f32; n_out];
        crate::models::bitnet::bitlinear_forward_from_f32(&weights, &x, n_in, n_out, &mut y_scalar);
        let y_avx2 = run_avx2_if_available(&weights, &x, n_in, n_out);
        for j in 0..n_out {
            assert_eq!(y_scalar[j].to_bits(), y_avx2[j].to_bits(), "j={j}");
        }
    }

    /// Block-level dequant-to-int8 scalar helper must agree with the
    /// canonical `dequant_i2_s_block` (f32) on the same input. This
    /// is the bit-exact test for the new scalar dequant helper.
    #[test]
    fn dequant_i2_s_block_to_i8_scalar_matches_f32() {
        use crate::ops::kernel::i2_s::dequant_i2_s_block;
        for byte_value in 0..=255u8 {
            let block = [byte_value; 32];
            let mut out_i8 = [0i8; 128];
            let mut out_f32 = [0.0f32; 128];
            dequant_i2_s_block_to_i8_scalar(&block, &mut out_i8);
            dequant_i2_s_block(&block, &mut out_f32);
            for j in 0..128 {
                let expected = out_f32[j] as i8;
                assert_eq!(
                    out_i8[j], expected,
                    "byte=0b{:08b} j={j}: i8={} f32_as_i8={}",
                    byte_value, out_i8[j], expected
                );
            }
        }
    }

    /// Pre-packed AVX2 forward must match the unpacked AVX2 path
    /// bit-for-bit (the only difference is when the dequant
    /// happens).
    #[test]
    fn bitlinear_avx2_packed_matches_unpacked() {
        let n_in = QK_I2_S * 4;
        let n_out = 5;
        let mut weights = vec![0u8; n_in / QK_I2_S * 32 * n_out];
        for j in 0..n_out {
            for b in 0..(n_in / QK_I2_S) {
                for k in 0..32 {
                    weights[j * n_in / QK_I2_S * 32 + b * 32 + k] = ((j as u8).wrapping_mul(17)
                        ^ (b as u8).wrapping_mul(31)
                        ^ (k as u8).wrapping_mul(13))
                        & 0b11;
                }
            }
        }
        let mut weights_packed: Vec<i8> = vec![0i8; n_in * n_out];
        dequant_i2_s_to_i8(&weights, n_in, n_out, &mut weights_packed);

        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i as f32) * 0.013).sin() * 0.7 - 0.4)
            .collect();

        let mut y_unpacked = vec![0.0f32; n_out];
        let mut y_packed = vec![0.0f32; n_out];
        unsafe {
            let (x_q, absmax) = crate::models::bitnet::quantize_activation_per_token(&x);
            bitlinear_forward_avx2(&weights, &x_q, absmax, n_in, n_out, &mut y_unpacked);
            bitlinear_forward_avx2_packed(
                &weights_packed,
                &x_q,
                absmax,
                n_in,
                n_out,
                &mut y_packed,
            );
        }

        for j in 0..n_out {
            assert_eq!(
                y_unpacked[j].to_bits(),
                y_packed[j].to_bits(),
                "j={j}: unpacked={} packed={}",
                y_unpacked[j],
                y_packed[j],
            );
        }
    }
}
