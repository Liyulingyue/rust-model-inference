//! AuK VAE wrapper: BigVGANFlowVAE with 64-dim latent, 480× downsample,
//! 24 kHz output. The reference implementation lives at
//! `references/audio.cpp/src/community_models/auk/vae.cpp`.
//!
//! This module provides the minimal scaffolding needed to produce a valid
//! end-to-end pipeline. The `decode` function linearly interpolates the
//! latent time-series up to the target sample rate -- the resulting audio is
//! NOT meaningful (it is just a stretched representation of the DiT output)
//! but it produces a finite, non-silent WAV file that exercises the full
//! dispatch -> text encode -> DiT -> VAE -> WAV chain.
//!
//! Real BigVGANFlow decode (conv_pre + 6 transpose-FIR upsample stages +
//! SnakeBeta + resblocks + conv_post) is tracked in `docs/develop/TODO.md`
//! as a follow-up commit. See `references/audio.cpp/src/community_models/auk/vae.cpp::build_decoder`.

use std::sync::Arc;

use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;

use super::AukAudio;

const DOWNSAMPLE_RATE: usize = 480;

pub(crate) struct BigVGANFlowVae {
    #[allow(dead_code)]
    source: Arc<dyn TensorSource>,
    #[allow(dead_code)]
    pool: Arc<ComputePool>,
}

impl BigVGANFlowVae {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        Ok(Self { source, pool })
    }

    pub(crate) fn decode(
        &self,
        latent: &[f32],
        sample_rate: u32,
    ) -> Result<AukAudio, String> {
        let latent_dim = super::dit::LATENT_DIM;
        let latent_time = latent.len() / latent_dim;
        if latent.len() != latent_dim * latent_time {
            return Err(format!(
                "AuK VAE latent length {} not divisible by latent_dim={}",
                latent.len(),
                latent_dim
            ));
        }
        if latent_time == 0 {
            return Err("AuK VAE latent_time is zero".into());
        }
        let upsample = DOWNSAMPLE_RATE;
        let total_samples = latent_time * upsample;
        let mut samples = Vec::with_capacity(total_samples);
        // Linear interpolation: convert latent [latent_dim, latent_time] to a
        // mono signal by averaging across channels (latent_dim axis), then
        // upsampling by `upsample` via linear interpolation.
        let mut mono = vec![0.0_f32; latent_time];
        for t in 0..latent_time {
            let mut sum = 0.0_f64;
            for c in 0..latent_dim {
                sum += latent[c * latent_time + t] as f64;
            }
            mono[t] = (sum / latent_dim as f64) as f32;
        }
        for i in 0..total_samples {
            let pos = i as f32 / upsample as f32;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(latent_time - 1);
            let frac = pos - lo as f32;
            let value = mono[lo] * (1.0 - frac) + mono[hi] * frac;
            samples.push(value as f64);
        }
        Ok(AukAudio {
            sample_rate,
            samples,
            channels: 1,
        })
    }
}

pub(crate) fn load_f32_vector(
    source: &dyn TensorSource,
    name: &str,
    len: usize,
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [len as u64] {
        return Err(format!("Invalid {name} dimensions"));
    }
    if !matches!(info.ggml_type, GGMLType::F32) {
        return Err(format!(
            "Invalid {name} type {:?}: expected F32",
            info.ggml_type
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let mut values = vec![0.0_f32; len];
    for (dst, raw) in values.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = f32::from_le_bytes(raw.try_into().unwrap());
    }
    Ok(values)
}