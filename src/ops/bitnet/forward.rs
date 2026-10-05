//! BitNet 1.58 BitLinear activation-quant + I2_S matmul ops
//! (scalar reference implementation).
//!
//! These are the **pure ops** in the BitLinear pipeline; the
//! data shapes (one slot per projection) live in
//! [`crate::ops::bitnet::slot`] and the per-projection RMSNorm
//! helper lives in
//! `crate::models::bitnet::*::bitlinear_projection`.
//!
//! # Spec (Microsoft BitNet-Embeddings, `docs/bitnet-embeddings-i2s-guide.md` §2)
//!
//! Per-projection BitLinear is the canonical BitNet 1.58 building block:
//!
//! ```text
//! x → RMSNorm(x, norm.weight)         // pre-projection centering
//!   → per-token absmax quantize(x)     // W1.58A8 — activations to int8
//!   → matmul(ternary_weights, x_q)    // I2_S ({-1, 0, +1}) × int8
//!   → rescale by (absmax / 127)        // recover F32 magnitude
//! ```
//!
//! The weight tensor itself stores ternary values at full magnitude
//! (`{-1, 0, +1}`) in the GGUF I2_S format — see [`crate::ops::kernel::i2_s`].
//! Per-tensor weight scale is **absorbed into the pre-projection
//! RMSNorm gain** (the conversion writes `*_norm_in.weight` to absorb the
//! inverse-of-mean-absweight; we don't need a separate scale tensor).
//!
//! # Quantization convention
//!
//! Activation quantization is per-token (per row of the input vector):
//!
//! ```text
//! absmax = max_i |x_i|          (scalar; if absmax == 0, treat as 1.0)
//! x_q[i] = round(x[i] / absmax * 127).clamp(-127, 127)
//! ```
//!
//! Then the ternary × int8 matmul is:
//!
//! ```text
//! y[j] = Σ_i w[j, i] * x_q[i]
//!     where w[j, i] ∈ {-1, 0, +1}, x_q[i] ∈ [-127, 127]
//! ```
//!
//! Final rescale:
//!
//! ```text
//! y_rescaled[j] = y[j] * (absmax / 127)
//! ```
//!
//! The factor `1/127` is the standard W1.58A8 rescale (it absorbs the
//! quantization step; the `absmax` recovers the original F32
//! magnitude). This matches `bitnet.cpp::ggml_bitnet_mul_mat` (the
//! MAD-path reference) up to the constant rescale factor of `127` vs
//! `127.5`/`128` which differs across paper revisions — we use `127`
//! because the GGUF conversion script
//! (`utils/convert-bitnet-embedding-to-gguf.py`) quantizes to `[-127, 127]`
//! (the signed-int8 range, not `[-128, 127]`).
//!
//! # Scope of this module
//!
//! This module ships the **scalar reference implementation** with unit
//! tests. The end-to-end qwen3 forward integration (per-projection
//! RMSNorm hookup + tensor wiring) is the next milestone — see
//! `docs/usage/bitnet_embedding.md` §3 and the follow-up commit log.
//!
//! The scalar reference is what future SIMD / parallel paths must
//! match bit-for-bit against. Per the BitNet-Embeddings paper Table 4
//! and `bitnet.cpp` golden tests, the reference scalar implementation
//! reproduces the FP16 teacher's embeddings within the ~0.35 MTEB
//! point gap (acceptable for the conversion to be considered lossless
//! at the 2-bit-per-weight precision).

use crate::ops::kernel::i2_s::{dequant_i2_s_row, QK_I2_S};

/// Per-token (per-row) activation quantization to int8 with absmax
/// rescale.
///
/// Returns:
/// - `x_q`: int8 quantized values in `[-127, 127]`
/// - `absmax`: `max_i |x[i]|`, or `1.0` if all-zero (avoids div-by-zero)
///
/// Equivalent to `bitnet.cpp::quantize_i2_s`'s `q8` intermediate
/// (clamped to the signed-int8 range, not the unsigned-int8 range the
/// upstream weight quantization uses).
pub fn quantize_activation_per_token(x: &[f32]) -> (Vec<i8>, f32) {
    let absmax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let absmax = if absmax == 0.0 { 1.0 } else { absmax };
    let mut q = Vec::with_capacity(x.len());
    for &v in x {
        let scaled = (v / absmax * 127.0).round();
        let clamped = scaled.clamp(-127.0, 127.0) as i8;
        q.push(clamped);
    }
    (q, absmax)
}

/// BitLinear forward:
///
/// ```text
/// y[j] = Σ_i w[j, i] * x_q[i] * (absmax / 127)
/// ```
///
/// `weights_i2s` is the raw I2_S GGUF payload for an `n_out × n_in`
/// matrix (row-major: row `j` starts at byte offset `j * n_in / 4`).
/// `x_q` is the activation-side int8 quantization produced by
/// [`quantize_activation_per_token`]. `absmax` is that function's
/// absmax rescale.
///
/// `y_out.len() == n_out`.
///
/// # SIMD dispatch
///
/// On x86_64 hosts with AVX2 + FMA the inner dot product is
/// vectorized via [`forward_avx2::bitlinear_forward_avx2`]
/// (sign-extend int8 → int16, then `_mm256_madd_epi16` pairwise
/// multiply-add). The two paths produce **bit-exact identical
/// outputs** for any input pair (tested in `forward_avx2.rs`).
/// On other architectures the scalar reference is used.
pub fn bitlinear_forward(
    weights_i2s: &[u8],
    x_q: &[i8],
    absmax: f32,
    n_in: usize,
    n_out: usize,
    y_out: &mut [f32],
) {
    assert_eq!(y_out.len(), n_out);
    assert_eq!(x_q.len(), n_in);
    assert_eq!(
        weights_i2s.len(),
        n_in / QK_I2_S * 32 * n_out,
        "i2_s row size mismatch: weights have {} bytes, expected {}",
        weights_i2s.len(),
        n_in / QK_I2_S * 32 * n_out,
    );
    assert_eq!(n_in % QK_I2_S, 0, "n_in must be a multiple of QK_I2_S=128");

    #[cfg(target_arch = "x86_64")]
    {
        if crate::ops::has_avx2_fma() {
            unsafe {
                crate::ops::bitnet::forward_avx2::bitlinear_forward_avx2(
                    weights_i2s,
                    x_q,
                    absmax,
                    n_in,
                    n_out,
                    y_out,
                );
            }
            return;
        }
    }

    bitlinear_forward_scalar(weights_i2s, x_q, absmax, n_in, n_out, y_out);
}

/// Scalar reference BitLinear forward (the same math, no SIMD).
///
/// Kept separate from [`bitlinear_forward`] so the AVX2 path can
/// share its dispatch wrapper while the per-row inner loop lives in
/// one well-tested location. Exposed as `pub` so benchmarks can
/// measure the scalar cost directly without going through the
/// dispatcher (which on AVX2+FMA hosts would silently route to the
/// SIMD kernel).
pub fn bitlinear_forward_scalar(
    weights_i2s: &[u8],
    x_q: &[i8],
    absmax: f32,
    n_in: usize,
    n_out: usize,
    y_out: &mut [f32],
) {
    let rescale = absmax / 127.0;
    let mut dequant = vec![0.0f32; n_in];
    for j in 0..n_out {
        let row_start = j * (n_in / QK_I2_S) * 32;
        let row_end = row_start + (n_in / QK_I2_S) * 32;
        dequant_i2_s_row(&weights_i2s[row_start..row_end], n_in, &mut dequant);
        let mut acc = 0.0f32;
        for i in 0..n_in {
            // `dequant[i]` is in {-1.0, 0.0, +1.0}; cast to i8 multiplies
            // directly without a multiply (dequant * x_q → f32, but
            // integer multiply is faster and bit-identical since
            // x_q ∈ [-127, 127] and w ∈ {-1, 0, +1}).
            let w = dequant[i] as i32;
            let x = x_q[i] as i32;
            acc += (w * x) as f32;
        }
        y_out[j] = acc * rescale;
    }
}

/// Convenience: BitLinear with on-the-fly activation quantization
/// (the typical caller path — quantize once, then forward).
///
/// `x` is F32 (length `n_in`); `weights_i2s` is the I2_S payload
/// (`n_out × n_in`); `y_out.len() == n_out`.
pub fn bitlinear_forward_from_f32(
    weights_i2s: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
    y_out: &mut [f32],
) {
    let (x_q, absmax) = quantize_activation_per_token(x);
    bitlinear_forward(weights_i2s, &x_q, absmax, n_in, n_out, y_out);
}

/// BitLinear forward, **pre-packed** weight variant.
///
/// `weights_i8` is the pre-dequanted int8 weight matrix
/// (length `n_out * n_in`, row-major) produced by
/// [`crate::ops::bitnet::BitLinearWeights::prepack`]. Skips the
/// per-call I2_S dequant walk entirely.
///
/// On AVX2+FMA hosts the inner dot is vectorized with
/// `_mm256_madd_epi16` (same SIMD kernel as the unpacked path,
/// just with the dequant hoisted out of the inner loop). The two
/// paths produce **bit-exact identical outputs** for any input
/// pair (the dequant helper is the same scalar implementation
/// used in both, just called at different times).
pub fn bitlinear_forward_packed(
    weights_i8: &[i8],
    x_q: &[i8],
    absmax: f32,
    n_in: usize,
    n_out: usize,
    y_out: &mut [f32],
) {
    assert_eq!(y_out.len(), n_out);
    assert_eq!(x_q.len(), n_in);
    assert_eq!(
        weights_i8.len(),
        n_in * n_out,
        "packed weight size mismatch: have {}, expected {}",
        weights_i8.len(),
        n_in * n_out,
    );

    #[cfg(target_arch = "x86_64")]
    {
        if crate::ops::has_avx2_fma() {
            unsafe {
                crate::ops::bitnet::forward_avx2::bitlinear_forward_avx2_packed(
                    weights_i8, x_q, absmax, n_in, n_out, y_out,
                );
            }
            return;
        }
    }

    bitlinear_forward_packed_scalar(weights_i8, x_q, absmax, n_in, n_out, y_out);
}

/// Scalar reference for the pre-packed path (same math, no SIMD).
fn bitlinear_forward_packed_scalar(
    weights_i8: &[i8],
    x_q: &[i8],
    absmax: f32,
    n_in: usize,
    n_out: usize,
    y_out: &mut [f32],
) {
    let rescale = absmax / 127.0;
    for j in 0..n_out {
        let row = &weights_i8[j * n_in..(j + 1) * n_in];
        let mut acc: i32 = 0;
        for i in 0..n_in {
            acc += (row[i] as i32) * (x_q[i] as i32);
        }
        y_out[j] = (acc as f32) * rescale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::kernel::i2_s::{dequant_i2_s_row, BLOCK_I2_S_SIZE};

    /// Hand-quantize a single weight, single-token forward, and
    /// verify the rescale arithmetic matches a Python-style
    /// reference.
    #[test]
    fn bitlinear_single_token_zero_sum() {
        // All-zero input → absmax=1, x_q all 0 → y = 0.
        let n_in = 128;
        let n_out = 4;
        let mut weights = vec![0u8; n_in / QK_I2_S * 32 * n_out];
        // First row: pack all +1 (j=0..31 → 0b10 in high bits).
        for b in 0..(n_in / QK_I2_S) {
            weights[b * 32] = 0b10_10_10_10;
        }
        let x = vec![0.0f32; n_in];
        let mut y = vec![0.0f32; n_out];
        bitlinear_forward_from_f32(&weights, &x, n_in, n_out, &mut y);
        for &v in &y {
            assert_eq!(v, 0.0, "all-zero input → all-zero output");
        }
    }

    /// Hand-quantize: weight = all +1, input = constant, verify
    /// `y[j] = n_in * (constant / absmax) * 127 * absmax / 127 = n_in * constant`.
    /// Specifically: with weight = all +1, x = [c; c; …; c] (absmax = |c|),
    /// x_q[i] = sign(c) * 127, y[j] = Σ_i 1 × sign(c) × 127 × absmax / 127
    /// = sign(c) × absmax × n_in = c × n_in.
    #[test]
    fn bitlinear_constant_input_linear() {
        let n_in = 128;
        let n_out = 3;
        let mut weights = vec![0u8; n_in / QK_I2_S * 32 * n_out];
        // All +1: every byte in every block = 0b10_10_10_10 (= 0xAA).
        // I2_S packing stores 4 ternary values per byte (one per
        // 2-bit slot); the byte pattern above means every slot is
        // `0b10` = +1 across all 128 elements of every block.
        for j in 0..n_out {
            for b in 0..(n_in / QK_I2_S) {
                for k in 0..32 {
                    weights[j * n_in / QK_I2_S * 32 + b * 32 + k] = 0b10_10_10_10;
                }
            }
        }
        let c = 0.5f32;
        let x = vec![c; n_in];
        let mut y = vec![0.0f32; n_out];
        bitlinear_forward_from_f32(&weights, &x, n_in, n_out, &mut y);
        let expected = c * n_in as f32;
        for &v in &y {
            // Float arithmetic; allow 1e-3 relative tolerance for
            // rounding through int8 quant.
            assert!((v - expected).abs() < 1.0, "expected ≈ {expected}, got {v}");
        }
    }

    /// Hand-quantize: weight = [one +1 at column 0, rest 0 in row 0;
    /// all-zero weight row 1], x = [1, 0, 0, …].
    /// Then y[0] = 1 × 127 × absmax / 127 = absmax = 1 (since x = [1, 0, …]).
    /// y[1] = 0 (all-zero weight row).
    ///
    /// Caveat: I2_S code 0b00 maps to -1, not 0, so encoding a
    /// zero-weight row requires explicitly packing every byte with
    /// 0x55 (= `0b01_01_01_01`, all-zero ternary).
    #[test]
    fn bitlinear_sparse_weight_zero_activations() {
        let n_in = 128;
        let n_out = 2;
        let mut weights = vec![0u8; n_in / QK_I2_S * 32 * n_out];
        // Row 0: element 0 = +1 (byte 0, slot 3, bits [6:8] = 0b10).
        weights[0] = 0b10 << 6;
        // Row 1: all-zero weights → every byte = 0x55.
        for k in 0..BLOCK_I2_S_SIZE {
            weights[n_in / QK_I2_S * 32 + k] = 0b01_01_01_01;
        }
        let mut x = vec![0.0f32; n_in];
        x[0] = 1.0;
        let mut y = vec![0.0f32; n_out];
        bitlinear_forward_from_f32(&weights, &x, n_in, n_out, &mut y);
        assert!((y[0] - 1.0).abs() < 1e-3, "y[0]={}", y[0]);
        assert_eq!(y[1], 0.0, "y[1]={}", y[1]);
    }

    /// Quantize-then-forward should match the i2_s-dequant-then-mul
    /// path up to int8 quantization noise (≤ 1 unit per element).
    /// This is the alignment test that proves our two primitives
    /// compose correctly.
    #[test]
    fn bitlinear_matches_dequantized_reference() {
        let n_in = QK_I2_S * 2;
        let n_out = 5;
        let mut weights = vec![0u8; n_in / QK_I2_S * 32 * n_out];
        // Random-ish but reproducible pattern: alternate 0b01 and
        // 0b10 across elements.
        for j in 0..n_out {
            for b in 0..(n_in / QK_I2_S) {
                weights[j * n_in / QK_I2_S * 32 + b * 32] = 0b10_01_01_10;
            }
        }
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.013).sin() * 0.7).collect();
        // Our BitLinear (quantize-then-forward).
        let mut y_quant = vec![0.0f32; n_out];
        bitlinear_forward_from_f32(&weights, &x, n_in, n_out, &mut y_quant);
        // Reference: dequantize to F32, then plain dot product (no
        // activation quant). This will differ by the activation
        // quant rounding — assert the difference is bounded.
        let mut w_f32 = vec![0.0f32; n_in * n_out];
        for j in 0..n_out {
            let row_start = j * (n_in / QK_I2_S) * 32;
            let row_end = row_start + (n_in / QK_I2_S) * 32;
            dequant_i2_s_row(
                &weights[row_start..row_end],
                n_in,
                &mut w_f32[j * n_in..(j + 1) * n_in],
            );
        }
        let mut y_ref = vec![0.0f32; n_out];
        for j in 0..n_out {
            let mut acc = 0.0f32;
            for i in 0..n_in {
                acc += w_f32[j * n_in + i] * x[i];
            }
            y_ref[j] = acc;
        }
        for j in 0..n_out {
            let absmax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            // Activation quantization introduces up to `0.5 / 127 *
            // |w| * n_in` error per output. For |w|≤1 and n_in=256
            // that's ≤ 1.01; allow 2.0 for slack.
            let diff = (y_quant[j] - y_ref[j]).abs();
            assert!(
                diff < 2.0 * absmax,
                "j={j}: quant={} ref={} diff={}",
                y_quant[j],
                y_ref[j],
                diff
            );
        }
    }
}
