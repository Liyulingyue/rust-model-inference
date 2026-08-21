//! Snake1d / Snake1d-style activations used by the Qwen3-TTS codec decoder.

/// Snake1d activation with GGUF-folded inverse beta:
/// `y = x + beta * sin(alpha * x).powi(2)`.
///
/// `alpha` and `beta` are per-channel learnable parameters with the same
/// length as `channels`. `x` is stored as `[length, channels]`.
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
    for row in x.chunks_exact_mut(channels) {
        for (c, sample) in row.iter_mut().enumerate() {
            let a = alpha[c];
            let s = (a * *sample).sin();
            *sample += s * s * beta[c];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snake_uses_t_first_channels_and_folded_inverse_beta() {
        let mut values = [
            std::f32::consts::FRAC_PI_2,
            2.0,
            3.0 * std::f32::consts::FRAC_PI_2,
            4.0,
        ];
        snake1d_inplace(&mut values, 2, &[1.0, 0.0], &[2.0, 3.0]).unwrap();
        assert!((values[0] - (std::f32::consts::FRAC_PI_2 + 2.0)).abs() < 1e-6);
        assert_eq!(values[1], 2.0);
        assert!((values[2] - (3.0 * std::f32::consts::FRAC_PI_2 + 2.0)).abs() < 1e-6);
        assert_eq!(values[3], 4.0);
    }
}
