//! Small math/format helpers used by the Qwen3.5 forward path.
//!
//! - `f16_at` / `f16_bits_to_f32`: F16 → F32 conversion without depending on
//!   the `half` crate.
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

/// F16 → F32 decode matching llama.cpp's scalar reference.
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

/// GDN normalization, matching llama.cpp's `ggml_l2_norm`
/// (`ggml_compute_forward_l2_norm_f32`).
///
/// The sum of squares accumulates in `ggml_float` (a double), and the scale
/// is `1 / max(sqrt(sum), eps)`: the epsilon **floors** the norm, it is not
/// added inside the square root.
///
/// The other formula -- `x / sqrt(sum(x²) + eps)`, written as an RMSNorm
/// followed by a scale -- comes from llama.cpp's older `build_gdn_l2_norm`
/// (`11fe0215`, `models/models.h`). Upstream has since moved to
/// `ggml_l2_norm`, and the two disagree whenever the norm is small: the old
/// one adds `eps` to the sum of squares, the new one compares `sqrt(sum)`
/// against `eps`. With `eps = 1e-5` and a GDN q/k norm around 0.14 that is
/// `eps / (2 * sum) = 2.4e-6` relative -- which is exactly where the Qwen3.8
/// parity run against the pinned llama.cpp diverged.
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

/// The GDN normalization of llama.cpp before `b96806d`:
/// `x / sqrt(sum(x²) + eps)`, built as `build_gdn_l2_norm` does it --
/// `ggml_scale(ggml_rms_norm(x, eps/n), 1/sqrt(n))`.
///
/// Kept for the models whose *runtime* Oracle is an older llama.cpp: the
/// Oracle's own numbers only reproduce with the formula it was built from.
/// See [`legacy_gdn_norm`].
pub(crate) fn l2_norm_legacy(x: &mut [f32], eps: f32) {
    let mut sum = 0.0f64;
    for &v in x.iter() {
        sum += f64::from(v * v);
    }
    let n = x.len() as f32;
    let mean = (sum / f64::from(n)) as f32;
    let rms_scale = 1.0f32 / (mean + eps / n).sqrt();
    let head_scale = 1.0f32 / n.sqrt();
    for v in x.iter_mut() {
        *v = (*v * rms_scale) * head_scale;
    }
}

/// Whether the GDN normalization follows llama.cpp before `b96806d`.
///
/// Upstream moved the GDN q/k normalization from `build_gdn_l2_norm` (the
/// epsilon inside the square root) to `ggml_l2_norm` (the epsilon floors the
/// denominator) between `11fe0215` and `b96806d`. The repository follows
/// upstream, which is also the commit every model's *conversion* is pinned
/// to.
///
/// `RMI_QWEN35_GDN_LEGACY_NORM=1` restores the older formula. It exists for
/// the models whose runtime Oracle is still an older llama.cpp -- OvisOCR3
/// is pinned to `11fe0215` -- because that Oracle's numbers only reproduce
/// with the formula it was built from. Drop the switch once that Oracle is
/// re-pinned to `b96806d`.
fn legacy_gdn_norm() -> bool {
    static LEGACY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LEGACY.get_or_init(|| std::env::var_os("RMI_QWEN35_GDN_LEGACY_NORM").is_some())
}

/// The GDN normalization the model runs with: upstream's by default, the
/// older llama.cpp formula when [`legacy_gdn_norm`] selects it.
pub(crate) fn gdn_l2_norm(x: &mut [f32], eps: f32) {
    if legacy_gdn_norm() {
        l2_norm_legacy(x, eps);
    } else {
        l2_norm(x, eps);
    }
}

pub(crate) fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}
