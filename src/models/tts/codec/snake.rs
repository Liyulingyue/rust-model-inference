//! Snake1d / Snake1d-style activations used by the Qwen3-TTS codec decoder.

/// Snake1d activation: `y = x + (1 / beta) * sin(alpha * x).powi(2)`.
///
/// `alpha` and `beta` are per-channel learnable parameters with the same
/// length as `channels`. `x` and `out` are stored as `[channels, length]`
/// contiguous row-major slices — i.e. the inner dimension is `length`.
pub fn snake1d_inplace(
    x: &mut [f32],
    length: usize,
    alpha: &[f32],
    beta: &[f32],
) -> Result<(), String> {
    let channels = alpha.len();
    if beta.len() != channels {
        return Err(format!(
            "snake1d: alpha/beta length mismatch ({} vs {})",
            alpha.len(),
            beta.len()
        ));
    }
    if x.len() != channels * length {
        return Err(format!(
            "snake1d: x length {} != channels*length {}",
            x.len(),
            channels * length,
        ));
    }
    for c in 0..channels {
        let a = alpha[c];
        let b = beta[c];
        let row = &mut x[c * length..(c + 1) * length];
        for sample in row.iter_mut() {
            let s = (a * *sample).sin();
            *sample += s * s / b;
        }
    }
    Ok(())
}