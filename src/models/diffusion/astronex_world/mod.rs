//! Astronex-World (Wan2.2-TI2V-5B) video pipeline.
//!
//! This module intentionally has its own tensor namespace and model graph.
//! DreamX is a different audio/video model and must not be used as an
//! architecture fallback.  The graph keeps activations in F32 so the strict
//! `RMI_SCALAR=1` route can be used for the parity checkpoints; the released
//! BF16 weights are converted only at the weight boundary.

use std::sync::Arc;

use image::RgbImage;

use super::dreamx::kernels::checked_len;
use super::dreamx::video_vae::Wan22Vae;
use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;

const DIM: usize = 3072;
const FFN: usize = 14336;
const HEADS: usize = 24;
const HEAD_DIM: usize = DIM / HEADS;
const LAYERS: usize = 30;
const TEXT_DIM: usize = 4096;
const TEXT_LEN: usize = 512;
const FREQ_DIM: usize = 256;
const LATENT_CHANNELS: usize = 48;
const PATCH: [usize; 3] = [1, 2, 2];
const SOURCE_REVISION: &str = "dec4ec502f7113a1ab07b78ae40a7ac1b88b2eeb";
const ORACLE_COMMIT: &str = "27584bf1a89da01ef35a03b48d83bc818d2adf2b";

mod denoiser;
mod scheduler;

use denoiser::Denoiser;
use scheduler::FlowUniPc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AstronexMode {
    TextToVideo,
    ImageToVideo,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AstronexOptions {
    pub mode: AstronexMode,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps: usize,
    pub steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
}

impl Default for AstronexOptions {
    fn default() -> Self {
        Self {
            mode: AstronexMode::TextToVideo,
            width: 832,
            height: 480,
            frames: 9,
            fps: 24,
            steps: 8,
            guidance_scale: 3.0,
            seed: 42,
        }
    }
}

impl AstronexOptions {
    pub fn validate(&self) -> Result<(), String> {
        if self.width == 0
            || self.height == 0
            || !self.width.is_multiple_of(16)
            || !self.height.is_multiple_of(16)
        {
            return Err("Astronex width and height must be positive multiples of 16".into());
        }
        if self.frames == 0 || (self.frames - 1) % 4 != 0 {
            return Err("Astronex frame count must be 1 plus a multiple of 4".into());
        }
        if self.fps == 0 || self.steps == 0 {
            return Err("Astronex fps and steps must be positive".into());
        }
        if self.steps != 8 {
            return Err(
                "Astronex currently supports the validated 8-step UniPC budget only".into(),
            );
        }
        if !self.guidance_scale.is_finite() || self.guidance_scale <= 0.0 {
            return Err("Astronex guidance scale must be finite and positive".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AstronexConfig {
    pub pair_id: String,
    pub source_revision: String,
    pub oracle_commit: String,
    pub layers: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub heads: usize,
    pub latent_channels: usize,
    pub text_len: usize,
}

impl AstronexConfig {
    pub fn from_sources(
        main: &dyn TensorSource,
        components: &dyn TensorSource,
    ) -> Result<Self, String> {
        let architecture = string_meta(main, "general.architecture")?;
        if architecture != "astronex_world" {
            return Err(format!(
                "Astronex main architecture mismatch: {architecture:?}"
            ));
        }
        let role = string_meta(main, "astronex.file_role")?;
        if role != "main" {
            return Err(format!("Astronex main file role mismatch: {role:?}"));
        }
        if string_meta(components, "astronex.file_role")? != "components" {
            return Err("Astronex component file role mismatch".into());
        }
        let pair_id = string_meta(main, "astronex.pair_id")?;
        if string_meta(components, "astronex.pair_id")? != pair_id {
            return Err("Astronex main/components pair_id mismatch".into());
        }
        let source_revision = string_meta(main, "astronex.source_revision")?;
        let oracle_commit = string_meta(main, "astronex.oracle_commit")?;
        if source_revision != SOURCE_REVISION {
            return Err(format!(
                "Astronex source revision mismatch: {source_revision:?}"
            ));
        }
        if oracle_commit != ORACLE_COMMIT {
            return Err(format!(
                "Astronex Oracle commit mismatch: {oracle_commit:?}"
            ));
        }
        let config = Self {
            pair_id,
            source_revision,
            oracle_commit,
            layers: LAYERS,
            hidden: DIM,
            ffn: FFN,
            heads: HEADS,
            latent_channels: LATENT_CHANNELS,
            text_len: TEXT_LEN,
        };
        validate_required_tensors(main, components)?;
        Ok(config)
    }
}

fn string_meta(source: &dyn TensorSource, key: &str) -> Result<String, String> {
    source
        .metadata(key)
        .and_then(MetaValue::to_string_val)
        .map(str::to_owned)
        .ok_or_else(|| format!("Astronex metadata {key:?} must be a string"))
}

fn validate_required_tensors(
    main: &dyn TensorSource,
    components: &dyn TensorSource,
) -> Result<(), String> {
    let required = [
        ("astronex.patch_embedding.weight", &[2, 2, 1, 48, 3072][..]),
        ("astronex.patch_embedding.bias", &[3072][..]),
        ("astronex.text_embedding.0.weight", &[4096, 3072][..]),
        ("astronex.text_embedding.2.weight", &[3072, 3072][..]),
        ("astronex.time_embedding.0.weight", &[256, 3072][..]),
        ("astronex.time_embedding.2.weight", &[3072, 3072][..]),
        ("astronex.time_projection.1.weight", &[3072, 18432][..]),
        ("astronex.head.head.weight", &[3072, 192][..]),
    ];
    for (name, dims) in required {
        let info = main
            .tensor_info(name)
            .ok_or_else(|| format!("Missing Astronex tensor {name}"))?;
        if info.dims != dims {
            return Err(format!(
                "Astronex tensor {name} has GGUF shape {:?}; expected {dims:?}",
                info.dims
            ));
        }
    }
    let token_embedding = components
        .tensor_info("astronex.text.token_embedding.weight")
        .ok_or("Missing Astronex UMT5 token embedding")?;
    if token_embedding.dims != [TEXT_DIM as u64, 256384] {
        return Err(format!(
            "Astronex token embedding shape {:?} is not [4096, 256384]",
            token_embedding.dims
        ));
    }
    if components
        .tensor_info("astronex.vae.conv1.weight")
        .is_none()
    {
        return Err("Missing Astronex Wan VAE tensors".into());
    }
    Ok(())
}

pub struct AstronexPipeline {
    pub config: AstronexConfig,
    text: super::dreamx::text::DreamXTextEncoder,
    denoiser: Denoiser,
    vae: Wan22Vae,
}

impl AstronexPipeline {
    pub fn load(
        main: Arc<dyn TensorSource>,
        components: Arc<dyn TensorSource>,
        n_threads: usize,
    ) -> Result<Self, String> {
        if n_threads == 0 {
            return Err("Astronex thread count must be non-zero".into());
        }
        let config = AstronexConfig::from_sources(main.as_ref(), components.as_ref())?;
        let pool = Arc::new(ComputePool::new(n_threads));
        let text = super::dreamx::text::DreamXTextEncoder::load_astronex(
            components.clone(),
            pool.clone(),
        )?;
        let denoiser = Denoiser::load(main.clone(), pool.clone())?;
        let vae = Wan22Vae::load_astronex(components.clone(), pool.clone())?;
        Ok(Self {
            config,
            text,
            denoiser,
            vae,
        })
    }

    pub fn generate(
        &self,
        prompt: &str,
        image: Option<&RgbImage>,
        options: &AstronexOptions,
    ) -> Result<Vec<RgbImage>, String> {
        options.validate()?;
        if options.mode == AstronexMode::ImageToVideo && image.is_none() {
            return Err("Astronex image-to-video requires an input image".into());
        }
        let latent_height = options.height / 16;
        let latent_width = options.width / 16;
        let latent_shape = [
            LATENT_CHANNELS,
            options.frames.div_ceil(4),
            latent_height,
            latent_width,
        ];
        let mut latent = gaussian(options.seed, checked_len("Astronex latent", &latent_shape)?)?;
        let mut reference = None;
        if let Some(frame) = image {
            let encoded = self.vae.encode_first_frame(frame)?;
            if encoded.shape() != [LATENT_CHANNELS, 1, latent_height, latent_width] {
                return Err("Astronex reference image latent shape mismatch".into());
            }
            for c in 0..LATENT_CHANNELS {
                for h in 0..latent_height {
                    for w in 0..latent_width {
                        latent[(c * latent_shape[1]) * latent_height * latent_width
                            + h * latent_width
                            + w] = encoded.as_slice()[(c * latent_height + h) * latent_width + w];
                    }
                }
            }
            reference = Some(encoded);
        }
        let positive = self.text.encode(prompt, "")?;
        let negative = self.text.encode("", "")?;
        let positive_len = positive.positive_len;
        let negative_len = negative.negative_len;
        let positive = positive.positive;
        let negative = negative.negative;
        let mut scheduler = FlowUniPc::new(options.steps, 5.0)?;
        for step in 0..options.steps {
            let timestep = scheduler.timesteps[step];
            let t = timestep as f32;
            let model_input = if let Some(ref frame) = reference {
                let mut input = latent.clone();
                let plane = latent_height * latent_width;
                for c in 0..LATENT_CHANNELS {
                    input[c * latent_shape[1] * plane..c * latent_shape[1] * plane + plane]
                        .copy_from_slice(&frame.as_slice()[c * plane..(c + 1) * plane]);
                }
                input
            } else {
                latent.clone()
            };
            let cond =
                self.denoiser
                    .forward(&model_input, latent_shape, t, &positive, positive_len)?;
            let uncond =
                self.denoiser
                    .forward(&model_input, latent_shape, t, &negative, negative_len)?;
            let mut flow = vec![0.0; latent.len()];
            for ((value, &unconditional), &conditional) in flow.iter_mut().zip(&uncond).zip(&cond) {
                *value = unconditional + options.guidance_scale * (conditional - unconditional);
            }
            latent = scheduler.step(&flow, timestep, &latent)?;
            if let Some(ref frame) = reference {
                let plane = latent_height * latent_width;
                for c in 0..LATENT_CHANNELS {
                    latent[c * latent_shape[1] * plane..c * latent_shape[1] * plane + plane]
                        .copy_from_slice(&frame.as_slice()[c * plane..(c + 1) * plane]);
                }
            }
        }
        let latent = super::dreamx::video_vae::VideoLatent::new(latent, latent_shape)?;
        self.vae.decode_frames(&latent)
    }
}

fn gaussian(seed: u64, len: usize) -> Result<Vec<f32>, String> {
    let mut state = seed.max(1);
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u1 = ((state >> 11) as f64 / ((1u64 << 53) as f64)).max(1e-12);
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u2 = (state >> 11) as f64 / ((1u64 << 53) as f64);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        out.push((radius * angle.cos()) as f32);
        if out.len() < len {
            out.push((radius * angle.sin()) as f32);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_reject_non_contract_frame_boundaries() {
        let mut options = AstronexOptions::default();
        options.frames = 10;
        assert!(options.validate().is_err());
    }

    #[test]
    fn rope_axes_cover_head_dimension() {
        assert_eq!((22 + 21 + 21) * 2, HEAD_DIM);
    }

    #[test]
    fn unipc_uses_official_eight_step_timesteps() {
        let scheduler = FlowUniPc::new(8, 5.0).unwrap();
        assert_eq!(
            scheduler.timesteps,
            [999, 972, 937, 892, 833, 749, 624, 416]
        );
        assert_eq!(scheduler.sigmas.last().copied(), Some(0.0));
    }
}
