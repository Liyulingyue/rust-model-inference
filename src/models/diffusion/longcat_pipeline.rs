//! LongCat latent layout and FlowMatch Euler sampling for Edit and Edit Turbo.

use super::longcat::{LongCatKind, LongCatTransformer, IMAGE_WIDTH, TEXT_WIDTH};

const LATENT_CHANNELS: usize = 16;

/// Channel-first `[16, height, width]` to LongCat's `[height/2 * width/2, 64]`.
pub fn pack_latents(latents: &[f32], height: usize, width: usize) -> Result<Vec<f32>, String> {
    let spatial = height
        .checked_mul(width)
        .ok_or("LongCat latent size overflow")?;
    if height == 0
        || width == 0
        || height % 2 != 0
        || width % 2 != 0
        || latents.len() != spatial * LATENT_CHANNELS
    {
        return Err("LongCat requires even, nonempty 16-channel latents".into());
    }
    let mut packed = vec![0.0; latents.len()];
    for y in 0..height / 2 {
        for x in 0..width / 2 {
            let token = (y * (width / 2) + x) * IMAGE_WIDTH;
            for channel in 0..LATENT_CHANNELS {
                for dy in 0..2 {
                    for dx in 0..2 {
                        packed[token + channel * 4 + dy * 2 + dx] =
                            latents[channel * spatial + (y * 2 + dy) * width + x * 2 + dx];
                    }
                }
            }
        }
    }
    Ok(packed)
}

pub fn unpack_latents(packed: &[f32], height: usize, width: usize) -> Result<Vec<f32>, String> {
    let spatial = height
        .checked_mul(width)
        .ok_or("LongCat latent size overflow")?;
    if height == 0
        || width == 0
        || height % 2 != 0
        || width % 2 != 0
        || packed.len() != spatial * LATENT_CHANNELS
    {
        return Err("LongCat requires even, nonempty 16-channel latents".into());
    }
    let mut latents = vec![0.0; packed.len()];
    for y in 0..height / 2 {
        for x in 0..width / 2 {
            let token = (y * (width / 2) + x) * IMAGE_WIDTH;
            for channel in 0..LATENT_CHANNELS {
                for dy in 0..2 {
                    for dx in 0..2 {
                        latents[channel * spatial + (y * 2 + dy) * width + x * 2 + dx] =
                            packed[token + channel * 4 + dy * 2 + dx];
                    }
                }
            }
        }
    }
    Ok(latents)
}

fn positions(text_tokens: usize, height: usize, width: usize) -> Vec<[f32; 3]> {
    let mut ids = Vec::with_capacity(text_tokens + 2 * height * width);
    for row in 0..text_tokens {
        ids.push([0.0, row as f32, row as f32]);
    }
    for modality in [1.0, 2.0] {
        for row in 0..height {
            for col in 0..width {
                ids.push([
                    modality,
                    (text_tokens + row) as f32,
                    (text_tokens + col) as f32,
                ]);
            }
        }
    }
    ids
}

/// Includes the terminal zero. Matches the two published scheduler configurations.
pub fn sigmas(kind: LongCatKind, image_tokens: usize, steps: usize) -> Result<Vec<f32>, String> {
    if steps == 0 || image_tokens == 0 {
        return Err("LongCat needs positive image tokens and steps".into());
    }
    let (base, max) = match kind {
        LongCatKind::Edit => (0.5, 1.15),
        LongCatKind::EditTurbo => (1.15, 1.15),
    };
    let slope = (max - base) / (4096.0 - 256.0);
    let mu = image_tokens as f64 * slope + (base - slope * 256.0);
    let exp_mu = mu.exp() as f32;
    let mut schedule = Vec::with_capacity(steps + 1);
    for index in 0..steps {
        let raw = if steps == 1 {
            1.0
        } else {
            1.0 + index as f64 * (1.0 / steps as f64 - 1.0) / (steps - 1) as f64
        } as f32;
        schedule.push(exp_mu / (exp_mu + (1.0f32 / raw - 1.0)));
    }
    schedule.push(0.0);
    Ok(schedule)
}

fn euler_flow_step(target: &mut [f32], velocity: &[f32], sigma: f32, next_sigma: f32) {
    let dt = next_sigma - sigma;
    for (value, &prediction) in target.iter_mut().zip(velocity) {
        // sd.cpp converts velocity to denoised x0, then back to a derivative.
        let denoised = prediction * -sigma + *value;
        let derivative = (*value - denoised) / sigma;
        *value += derivative * dt;
    }
}

/// Denoise pre-encoded target noise against a fixed reference latent.
/// The caller owns Qwen2.5-VL encoding, VAE encoding/decoding and noise creation.
pub fn denoise(
    transformer: &LongCatTransformer<'_>,
    mut target: Vec<f32>,
    reference: &[f32],
    positive: &[f32],
    negative: Option<&[f32]>,
    height: usize,
    width: usize,
    steps: usize,
    guidance: f32,
) -> Result<Vec<f32>, String> {
    let tokens = height
        .checked_mul(width)
        .ok_or("LongCat token count overflow")?;
    let expected = tokens
        .checked_mul(IMAGE_WIDTH)
        .ok_or("LongCat latent size overflow")?;
    if tokens == 0
        || target.len() != expected
        || reference.len() != expected
        || positive.is_empty()
        || positive.len() % TEXT_WIDTH != 0
        || negative.is_some_and(|n| n.len() != positive.len())
        || !guidance.is_finite()
        || target
            .iter()
            .chain(reference)
            .chain(positive)
            .chain(negative.into_iter().flatten())
            .any(|v| !v.is_finite())
    {
        return Err("Invalid LongCat denoising input".into());
    }
    let positions = positions(positive.len() / TEXT_WIDTH, height, width);
    let schedule = sigmas(transformer.kind, tokens, steps)?;
    let mut input = Vec::with_capacity(target.len() + reference.len());
    for step in 0..steps {
        input.clear();
        input.extend_from_slice(&target);
        input.extend_from_slice(reference);
        // The official pipeline materializes `sigma * 1000` as F32, then divides it.
        let timestep = (schedule[step] * 1000.0f32) / 1000.0f32;
        let conditional = transformer.forward(&input, positive, &positions, timestep)?;
        let prediction = if let Some(negative) = negative {
            let unconditional = transformer.forward(&input, negative, &positions, timestep)?;
            unconditional[..expected]
                .iter()
                .zip(&conditional[..expected])
                .map(|(uncond, cond)| uncond + guidance * (cond - uncond))
                .collect::<Vec<_>>()
        } else {
            conditional[..expected].to_vec()
        };
        euler_flow_step(&mut target, &prediction, schedule[step], schedule[step + 1]);
    }
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_round_trip_and_positions() {
        let source: Vec<f32> = (0..16 * 4 * 6).map(|value| value as f32).collect();
        let packed = pack_latents(&source, 4, 6).unwrap();
        assert_eq!(&packed[..4], &[0.0, 1.0, 6.0, 7.0]);
        assert_eq!(unpack_latents(&packed, 4, 6).unwrap(), source);
        let ids = positions(2, 2, 3);
        assert_eq!(ids[0], [0.0, 0.0, 0.0]);
        assert_eq!(ids[2], [1.0, 2.0, 2.0]);
        assert_eq!(ids[8], [2.0, 2.0, 2.0]);
    }

    #[test]
    fn edit_and_turbo_reference_schedule_bits() {
        let edit = sigmas(LongCatKind::Edit, 256, 50).unwrap();
        let turbo = sigmas(LongCatKind::EditTurbo, 256, 8).unwrap();
        assert_eq!(edit[0].to_bits(), 0x3f800000);
        assert_eq!(edit[1].to_bits(), 0x3f7cdeb4);
        assert_eq!(edit[49].to_bits(), 0x3d055555);
        assert_eq!(turbo[1].to_bits(), 0x3f74ebd8);
        assert_eq!(turbo[7].to_bits(), 0x3e9f2e6d);
        assert_eq!(edit[50], 0.0);
    }

    #[test]
    fn euler_step_keeps_oracle_rounding_order() {
        let mut target = [1.2199279069900513f32];
        euler_flow_step(
            &mut target,
            &[-4.37460470199585f32],
            f32::from_bits(0x3f426f4f),
            0.0,
        );
        assert_eq!(target[0].to_bits(), 0x40915c11);
    }
}
