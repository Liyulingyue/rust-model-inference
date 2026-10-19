//! Small math/format helpers used by the Qwen3.5 forward path.
//!
//! - `f16_at` / `f16_bits_to_f32`: F16 → F32 conversion without depending on
//!   the `half` crate (matches llama.cpp scalar order; bit-exact pinned in
//!   `qwen35_l2_norm_matches_pinned_llama_cpp_bits`).
//! - `l2_norm`, `softplus_f32`: tiny scalar helpers used by the Mamba SSM
//!   (recurrent) layer. Sigmoid moved to `ops::sigmoid_inplace` (SIMD).

/// Read the f32 value at `idx` of a little-endian f16 buffer.
/// Returns 0.0 if `idx` is out of range.
pub fn f16_at(data: &[u8], idx: usize) -> f32 {
    if idx * 2 + 2 > data.len() {
        return 0.0;
    }
    let bits = u16::from_le_bytes([data[idx * 2], data[idx * 2 + 1]]);
    f16_bits_to_f32(bits)
}

/// Bit-exact F16 → F32 decode matching llama.cpp's scalar reference. Pinned
/// by the test `qwen35_l2_norm_matches_pinned_llama_cpp_bits` in `tests.rs`.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as f32 / 1024.0;
    if exp == 0 {
        sign * frac * 2.0f32.powi(-14)
    } else if exp == 31 {
        if frac == 0.0 {
            sign * f32::INFINITY
        } else {
            sign * f32::NAN
        }
    } else {
        sign * (1.0 + frac) * 2.0f32.powi(exp - 15)
    }
}

/// In-place L2 normalization of `x`, matching ggml's `GGML_OP_L2_NORM`
/// scalar kernel (`ggml-cpu/ops.cpp`, `ggml_compute_forward_l2_norm_f32`).
///
/// That kernel is:
///
/// ```text
/// ggml_float sum = 0.0;                     // ggml_float is `double`
/// for (...) sum += (ggml_float)(xi * xi);
/// const float scale = 1.0f/fmaxf(sqrtf(sum), eps);
/// ```
///
/// so `eps` is a floor on the *denominator*, applied outside the square root.
/// Writing it as `1/sqrt(sum + eps)` is algebraically different and was the
/// bug: on a head vector whose norm sits near `eps`, one form clamps the
/// denominator at `eps` and the other does not, and the two disagree on a value
/// that then feeds the router. Measured on MiniCPM-V 4.7 35B-A3B, layer 1:
/// the input to this function was bit-exact on both sides and the output
/// differed by up to 3.6e-6, which then propagated to 4e-3 by the last layer.
///
/// The comment that used to sit here claimed eps went inside the square root.
/// It does not. See `tools/oracle/qwen35/qwen35moe-ffn-dump.patch.README.md`.
pub(crate) fn l2_norm(x: &mut [f32], eps: f32) {
    let mut sum = 0.0f64;
    for &v in x.iter() {
        sum += f64::from(v * v);
    }
    // `sqrtf` takes a float, so the double accumulator is narrowed first --
    // `(sum as f32).sqrt()` -- rather than summed in f32.
    let scale = 1.0f32 / (sum as f32).sqrt().max(eps);
    for v in x.iter_mut() {
        *v *= scale;
    }
}

pub(crate) fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}
