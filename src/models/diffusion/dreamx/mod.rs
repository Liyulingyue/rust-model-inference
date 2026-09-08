pub mod audio_vae;
pub mod config;
pub mod creator;
pub mod kernels;
pub mod lightvae;
pub mod media;
pub mod refiner;
pub mod text;
pub mod upsampler;
pub mod video_vae;

pub use config::DreamXConfig;
pub use refiner::DreamXRefiner;

use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use audio_vae::{CreatorDacVae, HOP_LENGTH, SAMPLE_RATE};
use creator::{CreatorModel, CreatorOutput};
use image::{imageops::FilterType, RgbImage};
use media::{mux_audio_atomic, write_wav_atomic, FfmpegVideoWriter};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use text::DreamXTextEncoder;
use video_vae::Wan22Vae;

const SPATIAL_DIVISOR: usize = 32;
const TEXT_WEIGHT_BYTES: u64 = 11_361_920_418;
const VIDEO_VAE_WEIGHT_BYTES: u64 = 2_818_839_170;
const AUDIO_VAE_WEIGHT_BYTES: u64 = 743_102_794;
const CREATOR_BF16_WEIGHT_BYTES: u64 = 28_232_544_288;
const REFINER_BF16_WEIGHT_BYTES: u64 = 9_999_839_408;
const FLASH_WEIGHT_BYTES: u64 = 20_014_099;
const CAUSAL2D_WEIGHT_BYTES: u64 = 463_683_163;
const LIGHTVAE_WEIGHT_BYTES: u64 = 590_511_261;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentUpsampleKind {
    Bilinear,
    Flash,
    Causal2d,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefinerDecoderKind {
    Wan,
    LightVae,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DreamXRefinerOptions {
    pub kv_len: usize,
    pub latent_upsample: LatentUpsampleKind,
    pub decoder: RefinerDecoderKind,
}

impl Default for DreamXRefinerOptions {
    fn default() -> Self {
        Self {
            kv_len: 9,
            latent_upsample: LatentUpsampleKind::Bilinear,
            decoder: RefinerDecoderKind::Wan,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DreamXOptions {
    pub duration_seconds: f32,
    pub fps: usize,
    pub steps: usize,
    pub seed: i64,
    pub target_spatial_tokens: usize,
    pub refine: bool,
    pub refiner: DreamXRefinerOptions,
}

impl Default for DreamXOptions {
    fn default() -> Self {
        Self {
            duration_seconds: 5.0,
            fps: 24,
            steps: 50,
            seed: 0,
            target_spatial_tokens: 880,
            refine: true,
            refiner: DreamXRefinerOptions::default(),
        }
    }
}

pub struct DreamXRequest {
    pub image: RgbImage,
    pub prompt: String,
    pub negative_prompt: String,
    pub output: PathBuf,
    pub options: DreamXOptions,
    pub overwrite: bool,
    pub allow_memory_overcommit: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DreamXArtifacts {
    pub base_video: PathBuf,
    pub audio: PathBuf,
    pub base_muxed: PathBuf,
    pub refined_video: Option<PathBuf>,
    pub refined_muxed: Option<PathBuf>,
}

impl DreamXArtifacts {
    pub fn for_output(output: &Path, refine: bool) -> Result<Self, String> {
        if output
            .extension()
            .and_then(|extension| extension.to_str())
            .is_none_or(|extension| !extension.eq_ignore_ascii_case("mp4"))
        {
            return Err("DreamX --out must use the .mp4 extension".into());
        }
        Ok(Self {
            base_video: sibling_path(output, ".base.video", "mp4")?,
            audio: sibling_path(output, ".audio", "wav")?,
            base_muxed: output.to_owned(),
            refined_video: refine
                .then(|| sibling_path(output, ".refined.video", "mp4"))
                .transpose()?,
            refined_muxed: refine
                .then(|| sibling_path(output, ".refined", "mp4"))
                .transpose()?,
        })
    }

    fn paths(&self) -> impl Iterator<Item = &Path> {
        [
            Some(self.base_video.as_path()),
            Some(self.audio.as_path()),
            Some(self.base_muxed.as_path()),
            self.refined_video.as_deref(),
            self.refined_muxed.as_deref(),
        ]
        .into_iter()
        .flatten()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DreamXEstimate {
    pub width: usize,
    pub height: usize,
    pub spatial_tokens: usize,
    pub output_frames: usize,
    pub latent_frames: usize,
    pub audio_frames: usize,
    pub audio_sample_rate: u32,
    pub text_weight_bytes: u64,
    pub text_scratch_bytes: u64,
    pub creator_weight_bytes: u64,
    pub creator_scratch_bytes: u64,
    pub video_vae_weight_bytes: u64,
    pub audio_vae_weight_bytes: u64,
    pub refiner_weight_bytes: u64,
    pub refiner_scratch_bytes: u64,
    pub refiner_kv_bytes: u64,
    pub latent_bytes: u64,
    pub frame_bytes: u64,
    pub waveform_bytes: u64,
    pub peak_bytes: u64,
    pub physical_memory_bytes: Option<u64>,
}

pub struct DreamXPipeline {
    main: Arc<dyn TensorSource>,
    mmproj: Arc<dyn TensorSource>,
    config: DreamXConfig,
    pool: Arc<ComputePool>,
}

impl DreamXPipeline {
    pub fn load(
        main: Arc<dyn TensorSource>,
        mmproj: Arc<dyn TensorSource>,
        n_threads: usize,
    ) -> Result<Self, String> {
        if n_threads == 0 {
            return Err("DreamX thread count must be non-zero".into());
        }
        let config = DreamXConfig::from_sources(main.as_ref(), mmproj.as_ref())?;
        Ok(Self {
            main,
            mmproj,
            config,
            pool: Arc::new(ComputePool::new(n_threads)),
        })
    }

    pub fn estimate(&self, request: &DreamXRequest) -> Result<DreamXEstimate, String> {
        validate_request(request)?;
        let width = request.image.width() as usize;
        let height = request.image.height() as usize;
        let spatial_tokens = checked_mul(height / SPATIAL_DIVISOR, width / SPATIAL_DIVISOR)?;
        let requested_frames =
            ((request.options.duration_seconds as f64) * request.options.fps as f64) as usize;
        let output_frames = ((requested_frames.max(1) - 1) / 4) * 4 + 1;
        let latent_frames = (output_frames - 1) / 4 + 1;
        let output_duration = output_frames as f64 / request.options.fps as f64;
        let audio_frames = f64_to_usize_ceil(output_duration * 50.0, "audio frame count")?.max(1);

        let cfg = &self.config;
        let text_tokens = cfg.text_context_length;
        let text_conditioning_elements =
            checked_mul(2, checked_mul(text_tokens, cfg.text_embedding_length)?)?;
        let text_conditioning = bytes_f32(text_conditioning_elements)?;
        let text_scratch_bytes = bytes_f32(checked_add(
            checked_mul(
                text_tokens,
                checked_add(
                    checked_mul(6, cfg.text_embedding_length)?,
                    checked_mul(3, cfg.text_feed_forward_length)?,
                )?,
            )?,
            text_conditioning_elements,
        )?)?;

        let video_latent_elements = checked_mul(
            48,
            checked_mul(latent_frames, checked_mul(height / 16, width / 16)?)?,
        )?;
        let audio_latent_elements = checked_mul(128, audio_frames)?;
        let latent_bytes = bytes_f32(checked_add(video_latent_elements, audio_latent_elements)?)?;
        let video_tokens = checked_mul(latent_frames, spatial_tokens)?;
        let creator_scratch_bytes = bytes_f32(checked_add(
            checked_add(
                checked_mul(
                    video_tokens,
                    checked_add(
                        checked_mul(6, cfg.video_embedding_length)?,
                        checked_mul(3, cfg.video_feed_forward_length)?,
                    )?,
                )?,
                checked_mul(
                    audio_frames,
                    checked_add(
                        checked_mul(6, cfg.audio_embedding_length)?,
                        checked_mul(3, cfg.audio_feed_forward_length)?,
                    )?,
                )?,
            )?,
            checked_mul(
                3,
                checked_add(video_latent_elements, audio_latent_elements)?,
            )?,
        )?)?;

        let refiner_spatial_tokens = checked_mul(4, spatial_tokens)?;
        let cached_frames = request.options.refiner.kv_len.min(latent_frames);
        let refiner_kv_bytes = bytes_f32(checked_mul(
            cfg.video_block_count,
            checked_mul(
                cached_frames,
                checked_mul(
                    refiner_spatial_tokens,
                    checked_mul(2, cfg.video_embedding_length)?,
                )?,
            )?,
        )?)?;
        let refiner_rows = checked_mul(latent_frames.min(3), refiner_spatial_tokens)?;
        let refiner_scratch_bytes = bytes_f32(checked_add(
            checked_mul(
                refiner_rows,
                checked_add(
                    checked_mul(6, cfg.video_embedding_length)?,
                    checked_mul(3, cfg.video_feed_forward_length)?,
                )?,
            )?,
            checked_mul(8, video_latent_elements)?,
        )?)?;

        let frame_bytes = bytes_rgb(output_frames, height, width)?;
        let refined_frame_bytes = if request.options.refine {
            bytes_rgb(
                output_frames,
                height.checked_mul(2).ok_or("DreamX height overflow")?,
                width.checked_mul(2).ok_or("DreamX width overflow")?,
            )?
        } else {
            0
        };
        let waveform_bytes = bytes_f32(checked_mul(audio_frames, HOP_LENGTH)?)?;

        let creator_weight_bytes = main_weight_bytes(CREATOR_BF16_WEIGHT_BYTES, cfg);
        let mut refiner_weight_bytes = main_weight_bytes(REFINER_BF16_WEIGHT_BYTES, cfg)
            .checked_add(VIDEO_VAE_WEIGHT_BYTES)
            .ok_or("DreamX refiner weight estimate overflow")?;
        refiner_weight_bytes = refiner_weight_bytes
            .checked_add(match request.options.refiner.latent_upsample {
                LatentUpsampleKind::Bilinear => 0,
                LatentUpsampleKind::Flash => FLASH_WEIGHT_BYTES,
                LatentUpsampleKind::Causal2d => CAUSAL2D_WEIGHT_BYTES,
            })
            .and_then(|bytes| {
                bytes.checked_add(match request.options.refiner.decoder {
                    RefinerDecoderKind::Wan => 0,
                    RefinerDecoderKind::LightVae => LIGHTVAE_WEIGHT_BYTES,
                })
            })
            .ok_or("DreamX refiner weight estimate overflow")?;

        let text_peak = sum_bytes(&[TEXT_WEIGHT_BYTES, text_scratch_bytes])?;
        let vae_peak = sum_bytes(&[
            VIDEO_VAE_WEIGHT_BYTES,
            frame_bytes,
            latent_bytes,
            text_conditioning,
        ])?;
        let creator_peak = sum_bytes(&[
            creator_weight_bytes,
            creator_scratch_bytes,
            latent_bytes,
            text_conditioning,
        ])?;
        let audio_peak = sum_bytes(&[
            AUDIO_VAE_WEIGHT_BYTES,
            latent_bytes,
            waveform_bytes,
            frame_bytes,
        ])?;
        let refiner_peak = if request.options.refine {
            sum_bytes(&[
                refiner_weight_bytes,
                refiner_scratch_bytes,
                refiner_kv_bytes,
                frame_bytes,
                refined_frame_bytes,
                waveform_bytes,
                text_conditioning,
            ])?
        } else {
            0
        };

        Ok(DreamXEstimate {
            width,
            height,
            spatial_tokens,
            output_frames,
            latent_frames,
            audio_frames,
            audio_sample_rate: SAMPLE_RATE as u32,
            text_weight_bytes: TEXT_WEIGHT_BYTES,
            text_scratch_bytes,
            creator_weight_bytes,
            creator_scratch_bytes,
            video_vae_weight_bytes: VIDEO_VAE_WEIGHT_BYTES,
            audio_vae_weight_bytes: AUDIO_VAE_WEIGHT_BYTES,
            refiner_weight_bytes,
            refiner_scratch_bytes,
            refiner_kv_bytes,
            latent_bytes,
            frame_bytes,
            waveform_bytes,
            peak_bytes: [text_peak, vae_peak, creator_peak, audio_peak, refiner_peak]
                .into_iter()
                .max()
                .unwrap_or(0),
            physical_memory_bytes: physical_memory_bytes(),
        })
    }

    pub fn generate(&self, request: &DreamXRequest) -> Result<DreamXArtifacts, String> {
        let estimate = self.estimate(request)?;
        ensure_memory(&estimate, request.allow_memory_overcommit)?;
        let artifacts = DreamXArtifacts::for_output(&request.output, request.options.refine)?;
        reject_existing_outputs(&artifacts, request.overwrite)?;

        let conditioning = {
            let text = DreamXTextEncoder::load(self.mmproj.clone(), self.pool.clone())?;
            text.encode(&request.prompt, &request.negative_prompt)?
        };
        let first_frame = {
            let vae = Wan22Vae::load(self.mmproj.clone(), self.pool.clone())?;
            vae.encode_first_frame(&request.image)?
        };
        let CreatorOutput {
            video,
            audio,
            audio_frames,
        } = {
            let creator = CreatorModel::load(self.main.clone(), self.pool.clone())?;
            creator.denoise(first_frame, &conditioning, &request.options)?
        };
        let base_frames = {
            let vae = Wan22Vae::load(self.mmproj.clone(), self.pool.clone())?;
            vae.decode_frames(&video)?
        };
        let waveform = {
            let decoder = CreatorDacVae::load(self.mmproj.clone(), self.pool.clone())?;
            decoder.decode(&audio, audio_frames)?
        };

        write_video(
            &artifacts.base_video,
            &base_frames,
            request.options.fps,
            request.overwrite,
        )?;
        write_wav_atomic(
            &artifacts.audio,
            &waveform,
            SAMPLE_RATE as u32,
            request.overwrite,
        )?;
        mux_audio_atomic(
            &artifacts.base_video,
            &artifacts.audio,
            &artifacts.base_muxed,
            request.overwrite,
        )?;

        if let (Some(refined_video), Some(refined_muxed)) =
            (&artifacts.refined_video, &artifacts.refined_muxed)
        {
            let refined_frames = {
                let mut refiner = DreamXRefiner::load(
                    self.main.clone(),
                    self.mmproj.clone(),
                    self.pool.clone(),
                    request.options.refiner,
                )?;
                refiner.refine(&base_frames, &conditioning, &request.options)?
            };
            write_video(
                refined_video,
                &refined_frames,
                request.options.fps,
                request.overwrite,
            )?;
            mux_audio_atomic(
                refined_video,
                &artifacts.audio,
                refined_muxed,
                request.overwrite,
            )?;
        }
        Ok(artifacts)
    }
}

pub fn resolve_spatial_size(
    source_height: usize,
    source_width: usize,
    target_spatial_tokens: usize,
) -> Result<(usize, usize, usize), String> {
    if source_height == 0 || source_width == 0 || target_spatial_tokens == 0 {
        return Err("DreamX source dimensions and spatial-token budget must be positive".into());
    }
    let minimum_tokens = target_spatial_tokens
        .checked_mul(95)
        .and_then(|value| value.checked_add(99))
        .ok_or("DreamX spatial-token budget overflow")?
        / 100;
    let source_ratio = source_height as f64 / source_width as f64;
    let mut best: Option<((usize, f64, usize, usize, usize), usize)> = None;
    for token_height in 1..=target_spatial_tokens {
        let max_token_width = target_spatial_tokens / token_height;
        if max_token_width == 0 {
            continue;
        }
        let ideal_width = source_width as f64 * token_height as f64 / source_height as f64;
        for token_width in [
            1,
            max_token_width,
            ideal_width.floor() as usize,
            ideal_width.ceil() as usize,
        ] {
            if token_width == 0 || token_width > max_token_width {
                continue;
            }
            let tokens = checked_mul(token_height, token_width)?;
            let height = checked_mul(token_height, SPATIAL_DIVISOR)?;
            let width = checked_mul(token_width, SPATIAL_DIVISOR)?;
            let aspect_error = ((height as f64 / width as f64) / source_ratio).ln().abs();
            let score = (
                minimum_tokens.saturating_sub(tokens),
                aspect_error,
                target_spatial_tokens - tokens,
                height,
                width,
            );
            if best.as_ref().is_none_or(|(current, _)| score < *current) {
                best = Some((score, tokens));
            }
        }
    }
    let (score, tokens) = best.ok_or("Unable to resolve DreamX spatial-token budget")?;
    Ok((score.3, score.4, tokens))
}

pub fn resize_to_token_budget(
    image: &RgbImage,
    target_spatial_tokens: usize,
) -> Result<RgbImage, String> {
    let (height, width, _) = resolve_spatial_size(
        image.height() as usize,
        image.width() as usize,
        target_spatial_tokens,
    )?;
    let width = u32::try_from(width).map_err(|_| "DreamX resized width exceeds u32")?;
    let height = u32::try_from(height).map_err(|_| "DreamX resized height exceeds u32")?;
    Ok(image::imageops::resize(
        image,
        width,
        height,
        FilterType::CatmullRom,
    ))
}

pub fn ensure_memory(estimate: &DreamXEstimate, allow_overcommit: bool) -> Result<(), String> {
    if !allow_overcommit
        && estimate
            .physical_memory_bytes
            .is_some_and(|physical| estimate.peak_bytes > physical)
    {
        return Err(format!(
            "DreamX peak memory estimate {:.2} GiB exceeds physical memory {:.2} GiB; pass --allow-memory-overcommit to continue",
            gib(estimate.peak_bytes),
            gib(estimate.physical_memory_bytes.unwrap_or(0)),
        ));
    }
    Ok(())
}

fn validate_request(request: &DreamXRequest) -> Result<(), String> {
    let width = request.image.width() as usize;
    let height = request.image.height() as usize;
    if width == 0
        || height == 0
        || !width.is_multiple_of(SPATIAL_DIVISOR)
        || !height.is_multiple_of(SPATIAL_DIVISOR)
    {
        return Err("DreamX image dimensions must be positive multiples of 32".into());
    }
    if request.prompt.trim().is_empty()
        || request.options.steps == 0
        || request.options.fps == 0
        || request.options.target_spatial_tokens == 0
        || !request.options.duration_seconds.is_finite()
        || request.options.duration_seconds <= 0.0
        || request.options.refiner.kv_len == 0
    {
        return Err("Invalid DreamX request".into());
    }
    let actual_tokens = checked_mul(height / SPATIAL_DIVISOR, width / SPATIAL_DIVISOR)?;
    if actual_tokens > request.options.target_spatial_tokens {
        return Err("DreamX image exceeds the spatial-token budget".into());
    }
    DreamXArtifacts::for_output(&request.output, request.options.refine).map(|_| ())
}

fn sibling_path(output: &Path, suffix: &str, extension: &str) -> Result<PathBuf, String> {
    let stem = output
        .file_stem()
        .filter(|stem| !stem.is_empty())
        .ok_or("DreamX output path requires a file name")?;
    let mut name = OsString::from(stem);
    name.push(suffix);
    name.push(".");
    name.push(extension);
    Ok(output.with_file_name(name))
}

fn reject_existing_outputs(artifacts: &DreamXArtifacts, overwrite: bool) -> Result<(), String> {
    if !overwrite {
        if let Some(path) = artifacts.paths().find(|path| path.exists()) {
            return Err(format!(
                "DreamX output {} already exists; pass --overwrite to replace it",
                path.display()
            ));
        }
    }
    Ok(())
}

fn write_video(
    path: &Path,
    frames: &[RgbImage],
    fps: usize,
    overwrite: bool,
) -> Result<(), String> {
    let first = frames.first().ok_or("DreamX video has no frames")?;
    let fps = u32::try_from(fps).map_err(|_| "DreamX FPS exceeds u32")?;
    let mut writer =
        FfmpegVideoWriter::create(path, first.width(), first.height(), fps, overwrite)?;
    for frame in frames {
        writer.write_frame(frame)?;
    }
    writer.finish()
}

fn physical_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let output = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        if output.status.success() {
            return std::str::from_utf8(&output.stdout)
                .ok()?
                .trim()
                .parse()
                .ok();
        }
    }
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kib = meminfo
            .lines()
            .find_map(|line| line.strip_prefix("MemTotal:"))?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()?;
        return kib.checked_mul(1024);
    }
    None
}

fn main_weight_bytes(bf16_bytes: u64, config: &DreamXConfig) -> u64 {
    if config.main_outtype == "q8_0" {
        // Q8_0 stores 34 bytes per 32 values; this margin covers unquantized vectors.
        bf16_bytes.saturating_mul(9) / 16
    } else {
        bf16_bytes
    }
}

fn checked_mul(left: usize, right: usize) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| "DreamX size overflow".into())
}

fn checked_add(left: usize, right: usize) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| "DreamX size overflow".into())
}

fn bytes_f32(elements: usize) -> Result<u64, String> {
    u64::try_from(elements)
        .ok()
        .and_then(|value| value.checked_mul(size_of::<f32>() as u64))
        .ok_or_else(|| "DreamX byte-size overflow".into())
}

fn bytes_rgb(frames: usize, height: usize, width: usize) -> Result<u64, String> {
    u64::try_from(checked_mul(
        checked_mul(checked_mul(frames, height)?, width)?,
        3,
    )?)
    .map_err(|_| "DreamX frame byte-size overflow".into())
}

fn sum_bytes(values: &[u64]) -> Result<u64, String> {
    values.iter().try_fold(0u64, |total, value| {
        total
            .checked_add(*value)
            .ok_or_else(|| "DreamX memory estimate overflow".into())
    })
}

fn f64_to_usize_ceil(value: f64, name: &str) -> Result<usize, String> {
    if !value.is_finite() || value < 0.0 || value.ceil() > usize::MAX as f64 {
        return Err(format!("DreamX {name} overflow"));
    }
    Ok(value.ceil() as usize)
}

pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}
