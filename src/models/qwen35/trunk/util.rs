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

/// GDN normalization: x / sqrt(sum(x²) + eps), as RMSNorm then scale in
/// llama.cpp's `build_gdn_l2_norm` (11fe0215, models/models.h).
pub(crate) fn l2_norm(x: &mut [f32], eps: f32) {
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

pub(crate) fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}
