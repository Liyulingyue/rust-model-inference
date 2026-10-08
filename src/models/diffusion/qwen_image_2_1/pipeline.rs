use super::{Condition, QwenImage21Dit, ReferenceLatent};

/// Pinned sd.cpp FluxScheduler, including Qwen 2.1's terminal shift.
pub fn flow_sigmas(steps: usize, image_tokens: usize) -> Result<Vec<f32>, String> {
    if steps == 0 || image_tokens == 0 || steps > 1000 {
        return Err("Qwen-Image-2.1 requires 1..=1000 steps and a nonempty image".into());
    }
    let slope = (0.9f32 - 0.5) / (8192.0 - 256.0);
    let mu = image_tokens as f32 * slope + (0.5 - slope * 256.0);
    let shift = mu.exp();
    let mut sigmas: Vec<_> = (0..steps)
        .map(|i| {
            let t = 1.0 - i as f32 / steps as f32;
            shift / (shift + (1.0 / t - 1.0))
        })
        .collect();
    if steps > 1 {
        let scale = (1.0 - sigmas[steps - 1]) / (1.0 - 0.02);
        for sigma in &mut sigmas {
            *sigma = 1.0 - (1.0 - *sigma) / scale;
        }
    }
    sigmas.push(0.0);
    Ok(sigmas)
}

pub(crate) fn trace(name: &str, shape: &[usize], values: &[f32]) -> Result<(), String> {
    let _ = (name, shape, values);
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::checkpoint(name, None, shape, values).map_err(|e| e.to_string())?;
    Ok(())
}

pub(crate) fn sample(
    dit: &mut QwenImage21Dit,
    mut latent: Vec<f32>,
    width: usize,
    height: usize,
    positive: &Condition,
    negative: &Condition,
    references: &[ReferenceLatent],
    steps: usize,
    cfg: f32,
) -> Result<Vec<f32>, String> {
    if !cfg.is_finite() || cfg < 1.0 {
        return Err("Qwen-Image-2.1 CFG must be finite and at least 1".into());
    }
    let sigmas = flow_sigmas(
        steps,
        width.checked_mul(height).ok_or("Image size overflow")?,
    )?;
    trace("qwen.sample.sigmas", &[sigmas.len()], &sigmas)?;
    trace("qwen.sample.noise", &[width, height, 64], &latent)?;
    for step in 0..steps {
        let sigma = sigmas[step];
        let timestep = sigma * 1000.0;
        let mut velocity =
            dit.forward_conditioned(&latent, width, height, positive, references, timestep)?;
        if cfg != 1.0 {
            let uncond =
                dit.forward_conditioned(&latent, width, height, negative, references, timestep)?;
            for (v, u) in velocity.iter_mut().zip(uncond) {
                *v = u + cfg * (*v - u);
            }
        }
        trace("qwen.sample.velocity", &[width, height, 64], &velocity)?;
        let dt = sigmas[step + 1] - sigma;
        for (x, v) in latent.iter_mut().zip(velocity) {
            // Keep the Oracle's denoised -> derivative -> Euler evaluation order.
            let denoised = v * -sigma + *x;
            let derivative = (*x - denoised) / sigma;
            *x += derivative * dt;
        }
        if latent.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "Non-finite Qwen-Image-2.1 latent at step {}",
                step + 1
            ));
        }
        trace("qwen.sample.latent", &[width, height, 64], &latent)?;
        eprintln!("Qwen-Image-2.1 step {}/{steps}", step + 1);
    }
    Ok(latent)
}

/// Planar [-1, 1] RGBA to interleaved bytes; sd.cpp truncates after scaling.
pub fn rgba_bytes(values: &[f32], width: usize, height: usize) -> Result<Vec<u8>, String> {
    let pixels = width
        .checked_mul(height)
        .filter(|&n| n > 0)
        .ok_or("RGBA size overflow")?;
    let len = pixels.checked_mul(4).ok_or("RGBA size overflow")?;
    if values.len() != len || values.iter().any(|v| !v.is_finite()) {
        return Err("Invalid Qwen-Image-2.1 RGBA tensor".into());
    }
    let mut bytes = Vec::with_capacity(len);
    for pixel in 0..pixels {
        for channel in 0..4 {
            bytes.push(
                ((values[channel * pixels + pixel] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8,
            );
        }
    }
    Ok(bytes)
}
