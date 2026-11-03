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

mod cache;
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
            return Err(
                "Astronex width and height must be positive multiples of 16; latent patches are padded to 32 and cropped after decoding".into(),
            );
        }
        if self.frames == 0 {
            return Err("Astronex frame count must be positive".into());
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
        generation_geometry(self)?;
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
        self,
        prompt: &str,
        image: Option<&RgbImage>,
        options: &AstronexOptions,
    ) -> Result<Vec<RgbImage>, String> {
        options.validate()?;
        if options.mode == AstronexMode::ImageToVideo && image.is_none() {
            return Err("Astronex image-to-video requires an input image".into());
        }
        let Self {
            text,
            denoiser,
            vae,
            ..
        } = self;
        let (width, height, latent_frames) = generation_geometry(options)?;
        let latent_shape = [LATENT_CHANNELS, latent_frames, height / 16, width / 16];
        let plane = latent_shape[2] * latent_shape[3];
        let noise = gaussian(options.seed, checked_len("Astronex latent", &latent_shape)?)?;
        let reference = image
            .map(|image| {
                let mut padded = RgbImage::new(width as u32, height as u32);
                for (x, y, pixel) in padded.enumerate_pixels_mut() {
                    *pixel = *image.get_pixel(x.min(image.width() - 1), y.min(image.height() - 1));
                }
                vae.encode_first_frame(&padded)
            })
            .transpose()?;
        let encoded = text.encode(prompt, NEGATIVE_PROMPT)?;
        drop(text);
        let frame_tokens = (height / 32) * (width / 32);
        let mut positive = denoiser.prepare_branch(&encoded.positive, frame_tokens)?;
        let mut negative = denoiser.prepare_branch(&encoded.negative, frame_tokens)?;
        drop(encoded);
        let mut latent = vec![0.0; noise.len()];
        let block_shape = [LATENT_CHANNELS, 8, height / 16, width / 16];
        eprintln!("[Astronex] causal generation {width}x{height}, {latent_frames} latent frames, block=8 window=20 sink=4");
        for start in (0..latent_frames).step_by(8) {
            let started = std::time::Instant::now();
            let mut block = vec![0.0; LATENT_CHANNELS * 8 * plane];
            for c in 0..LATENT_CHANNELS {
                block[c * 8 * plane..(c + 1) * 8 * plane].copy_from_slice(
                    &noise[(c * latent_frames + start) * plane
                        ..(c * latent_frames + start + 8) * plane],
                );
            }
            let restore_reference = |block: &mut [f32]| {
                if start == 0 {
                    if let Some(reference) = &reference {
                        for c in 0..LATENT_CHANNELS {
                            block[c * 8 * plane..c * 8 * plane + plane]
                                .copy_from_slice(&reference.as_slice()[c * plane..(c + 1) * plane]);
                        }
                    }
                }
            };
            restore_reference(&mut block);
            let mut scheduler = FlowUniPc::new(options.steps, 10.0)?;
            for step in 0..options.steps {
                let timestep = scheduler.timesteps[step];
                let cond = denoiser.forward(
                    &block,
                    block_shape,
                    timestep as f32,
                    &mut positive,
                    start,
                    start == 0 && reference.is_some(),
                )?;
                let mut flow = denoiser.forward(
                    &block,
                    block_shape,
                    timestep as f32,
                    &mut negative,
                    start,
                    start == 0 && reference.is_some(),
                )?;
                for (value, &conditional) in flow.iter_mut().zip(&cond) {
                    *value += options.guidance_scale * (conditional - *value);
                }
                block = scheduler.step(&flow, timestep, &block)?;
                restore_reference(&mut block);
                eprintln!(
                    "[Astronex] block {}/{} step {}/{} elapsed={:.3}s",
                    start / 8 + 1,
                    latent_frames / 8,
                    step + 1,
                    options.steps,
                    started.elapsed().as_secs_f64()
                );
            }
            for c in 0..LATENT_CHANNELS {
                latent
                    [(c * latent_frames + start) * plane..(c * latent_frames + start + 8) * plane]
                    .copy_from_slice(&block[c * 8 * plane..(c + 1) * 8 * plane]);
            }
            if start + 8 < latent_frames {
                // Only clean context is visible to the next block.
                denoiser.forward(
                    &block,
                    block_shape,
                    0.0,
                    &mut positive,
                    start,
                    start == 0 && reference.is_some(),
                )?;
                denoiser.forward(
                    &block,
                    block_shape,
                    0.0,
                    &mut negative,
                    start,
                    start == 0 && reference.is_some(),
                )?;
            }
        }
        drop(noise);
        drop(positive);
        drop(negative);
        drop(denoiser);
        if let Some(path) = std::env::var_os("RMI_ASTRONEX_LATENT_OUT") {
            let file =
                std::fs::File::create(path).map_err(|e| format!("Create Astronex latent: {e}"))?;
            let mut file = std::io::BufWriter::new(file);
            use std::io::Write;
            for dimension in latent_shape {
                file.write_all(&(dimension as u64).to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
            for value in &latent {
                file.write_all(&value.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
            file.flush().map_err(|e| e.to_string())?;
        }
        let latent = super::dreamx::video_vae::VideoLatent::new(latent, latent_shape)?;
        eprintln!("[Astronex] sampling complete; decoding VAE");
        let mut frames = vae.decode_frames(&latent)?;
        frames.truncate(options.frames);
        if width != options.width || height != options.height {
            frames = frames
                .iter()
                .map(|frame| {
                    image::imageops::crop_imm(
                        frame,
                        0,
                        0,
                        options.width as u32,
                        options.height as u32,
                    )
                    .to_image()
                })
                .collect();
        }
        Ok(frames)
    }
}

const NEGATIVE_PROMPT: &str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

fn generation_geometry(options: &AstronexOptions) -> Result<(usize, usize, usize), String> {
    let align = |n: usize, unit: usize| {
        n.checked_add(unit - 1)
            .and_then(|n| (n / unit).checked_mul(unit))
            .ok_or("Astronex generation geometry overflow".to_string())
    };
    let width = align(options.width, 32)?;
    let height = align(options.height, 32)?;
    if width > u32::MAX as usize || height > u32::MAX as usize {
        return Err("Astronex dimensions exceed the image format".into());
    }
    let frames = options
        .frames
        .checked_add(3)
        .ok_or("Astronex frame count overflow")?
        .div_ceil(4);
    Ok((width, height, align(frames, 8)?))
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
    fn conventional_720p_15_seconds_aligns_then_crops_without_scaling() {
        let options = AstronexOptions {
            width: 1280,
            height: 720,
            frames: 360,
            ..AstronexOptions::default()
        };
        assert!(options.validate().is_ok());
        assert_eq!(generation_geometry(&options).unwrap(), (1280, 736, 96));
        for options in [
            AstronexOptions {
                frames: usize::MAX,
                ..options.clone()
            },
            AstronexOptions {
                width: usize::MAX - 15,
                ..options
            },
        ] {
            assert!(options.validate().is_err());
        }
    }

    #[test]
    fn options_accept_trimmed_frame_boundaries() {
        let mut options = AstronexOptions::default();
        options.frames = 10;
        assert!(options.validate().is_ok());
        options.frames = 0;
        assert!(options.validate().is_err());
    }

    #[test]
    fn options_require_complete_spatial_patches() {
        for (width, height) in [(0, 32), (32, 0), (15, 32), (32, 15)] {
            let options = AstronexOptions {
                width,
                height,
                ..AstronexOptions::default()
            };
            assert!(options.validate().is_err(), "{width}x{height}");
        }
        for (width, height) in [
            (16, 32),
            (32, 16),
            (32, 32),
            (832, 480),
            (1280, 720),
            (1280, 736),
        ] {
            let options = AstronexOptions {
                width,
                height,
                ..AstronexOptions::default()
            };
            assert!(options.validate().is_ok(), "{width}x{height}");
        }
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
        assert_eq!(
            FlowUniPc::new(8, 10.0).unwrap().timesteps,
            [999, 985, 967, 943, 908, 856, 768, 587]
        );
    }
}
