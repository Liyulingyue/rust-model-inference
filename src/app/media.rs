//! Shared media utilities: types, validation, decoding.
//!
//! Consumed by `text/multimodal.rs`, `omni.rs`, and `qwen_drive.rs`.
//! Eliminates the circular dependency between omni.rs and text/.

use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
    Audio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectorFamily {
    Qwen3VlMerger,
    Qwen25Omni,
}

pub fn validate_mmproj_capabilities(
    llm_arch: &str,
    mmproj: &dyn TensorSource,
    media: MediaKind,
) -> Result<ProjectorFamily, String> {
    let family = match llm_arch {
        "qwen3vl" | "qwen3vlmoe" => ProjectorFamily::Qwen3VlMerger,
        "qwen2vl" | "qwen35" => ProjectorFamily::Qwen25Omni,
        other => return Err(format!("Unsupported multimodal architecture: {other}")),
    };
    if llm_arch == "qwen3vl" && media == MediaKind::Audio {
        return Err("Qwen3-VL does not support audio input".into());
    }
    if let Some(projector) = mmproj
        .metadata("clip.projector_type")
        .and_then(|v| v.to_string_val())
    {
        let allowed: &[&str] = match family {
            ProjectorFamily::Qwen3VlMerger => &["qwen3vl_merger"],
            ProjectorFamily::Qwen25Omni => &["qwen2.5o", "qwen2.5vl_merger"],
        };
        if !allowed.iter().any(|p| *p == projector) {
            return Err(format!(
                "{llm_arch} requires one of {:?}, got {projector}",
                allowed
            ));
        }
    }
    let has_encoder = match media {
        MediaKind::Audio => "clip.has_audio_encoder",
        MediaKind::Image | MediaKind::Video => "clip.has_vision_encoder",
    };
    if let Some(MetaValue::Bool(false)) = mmproj.metadata(has_encoder) {
        return Err(format!("mmproj does not provide {has_encoder}"));
    }
    Ok(family)
}

pub fn marker_names(kind: MediaKind) -> (&'static str, &'static str, &'static str) {
    match kind {
        MediaKind::Image => ("vision_start", "image_pad", "vision_end"),
        MediaKind::Video => ("vision_start", "video_pad", "vision_end"),
        MediaKind::Audio => ("audio_start", "audio_pad", "audio_end"),
    }
}

pub fn frame_pairs(count: usize) -> Vec<(usize, usize)> {
    (0..count)
        .step_by(2)
        .map(|index| (index, (index + 1).min(count.saturating_sub(1))))
        .collect()
}

pub fn vision_markers(start: u32, pad: u32, end: u32, rows: usize) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(rows.saturating_add(2));
    tokens.push(start);
    tokens.extend(std::iter::repeat_n(pad, rows));
    tokens.push(end);
    tokens
}

pub fn append_media_markers(
    tokens: &mut Vec<u32>,
    tokenizer: &BPETokenizer,
    kind: MediaKind,
    start: u32,
    pad: u32,
    end: u32,
    block_rows: &[usize],
) {
    for (index, &rows) in block_rows.iter().enumerate() {
        if kind == MediaKind::Video {
            tokens.extend(tokenizer.encode(
                &format!("<{:.1} seconds>", index as f32 + 0.25),
                EncodeOptions {
                    add_special: false,
                    parse_special: false,
                },
            ));
        }
        tokens.extend(vision_markers(start, pad, end, rows));
    }
}

pub fn decode_video(path: &Path) -> Result<Vec<image::DynamicImage>, String> {
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
        ])
        .arg(path)
        .output()
        .map_err(|error| format!("Failed to run ffprobe; install FFmpeg: {error}"))?;
    if !probe.status.success() {
        return Err(format!(
            "ffprobe failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&probe.stderr).trim()
        ));
    }
    let dimensions = String::from_utf8(probe.stdout)
        .map_err(|error| format!("ffprobe returned invalid UTF-8: {error}"))?;
    let (width, height) = dimensions
        .trim()
        .split_once('x')
        .ok_or_else(|| format!("ffprobe returned invalid dimensions: {dimensions:?}"))?;
    let width = width
        .parse::<usize>()
        .map_err(|error| format!("Invalid video width: {error}"))?;
    let height = height
        .parse::<usize>()
        .map_err(|error| format!("Invalid video height: {error}"))?;
    let frame_bytes = width
        .checked_mul(height)
        .and_then(|value| value.checked_mul(3))
        .ok_or("Video frame size overflow")?;
    if frame_bytes == 0 {
        return Err("Video dimensions must be nonzero".into());
    }

    let decoded = Command::new("ffmpeg")
        .args(["-v", "error", "-noautorotate", "-i"])
        .arg(path)
        .args([
            "-vf",
            "fps=2",
            "-frames:v",
            "32",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .output()
        .map_err(|error| format!("Failed to run ffmpeg; install FFmpeg: {error}"))?;
    if !decoded.status.success() {
        return Err(format!(
            "ffmpeg failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&decoded.stderr).trim()
        ));
    }
    if decoded.stdout.len() % frame_bytes != 0 {
        return Err("ffmpeg returned a partial RGB frame".into());
    }
    let width_u32 = u32::try_from(width).map_err(|_| "Video width exceeds u32")?;
    let height_u32 = u32::try_from(height).map_err(|_| "Video height exceeds u32")?;
    let mut frames = decoded
        .stdout
        .chunks_exact(frame_bytes)
        .map(|bytes| {
            image::RgbImage::from_raw(width_u32, height_u32, bytes.to_vec())
                .map(image::DynamicImage::ImageRgb8)
                .ok_or_else(|| "Failed to construct decoded RGB frame".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if frames.is_empty() {
        return Err(format!("Video {} produced no frames", path.display()));
    }
    while frames.len() < 4 {
        frames.push(frames.last().expect("nonempty frames").clone());
    }
    Ok(frames)
}

pub fn decode_audio(path: &Path) -> Result<Vec<f32>, String> {
    if let Ok(decoded) = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ])
        .output()
    {
        if decoded.status.success() {
            if decoded.stdout.is_empty() || decoded.stdout.len() % 4 != 0 {
                return Err("ffmpeg returned invalid F32 audio".into());
            }
            let samples = decoded
                .stdout
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
                .collect::<Vec<_>>();
            if samples.iter().any(|sample| !sample.is_finite()) {
                return Err("ffmpeg returned non-finite audio".into());
            }
            return Ok(samples);
        }
        return Err(format!(
            "ffmpeg failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&decoded.stderr).trim()
        ));
    }

    let bytes = std::fs::read(path)
        .map_err(|error| format!("Failed to read audio {}: {error}", path.display()))?;
    let decoded = crate::models::qwen3::asr::audio_processor::decode_pcm16_wav_any(&bytes)
        .map_err(|error| {
            format!(
                "Pure-Rust audio decode failed for {}: {:?}",
                path.display(),
                error
            )
        })?;
    if decoded.channels != 1 {
        return Err(format!(
            "Audio {} has {} channels; Omni requires mono. Install ffmpeg to mix-down.",
            path.display(),
            decoded.channels
        ));
    }
    if decoded.sample_rate != 16_000 {
        return Err(format!(
            "Audio {} has {} Hz; Omni requires 16000 Hz. Install ffmpeg to resample.",
            path.display(),
            decoded.sample_rate
        ));
    }
    Ok(decoded.samples)
}

pub fn decode_image(path: &Path) -> Result<image::DynamicImage, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("Failed to read image {}: {error}", path.display()))?;
    image::load_from_memory(&bytes)
        .map_err(|error| format!("Failed to decode image {}: {error}", path.display()))
}

pub fn normalize_resized_image(
    image: &image::DynamicImage,
    target_w: usize,
    target_h: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Result<Vec<f32>, String> {
    if std.iter().any(|value| *value == 0.0) {
        return Err("Vision normalization std must be nonzero".into());
    }
    let source = image.to_rgb8();
    let resized = crate::models::gemma4::vision::resize_bicubic_pillow(
        source.as_raw(),
        source.width() as usize,
        source.height() as usize,
        target_w,
        target_h,
    )?;
    let output_len = target_w
        .checked_mul(target_h)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or("Normalized image length overflow")?;
    let mut output = vec![0.0f32; output_len];
    for y in 0..target_h {
        for x in 0..target_w {
            let offset = (y * target_w + x) * 3;
            for channel in 0..3 {
                output[offset + channel] =
                    (f32::from(resized[offset + channel]) / 255.0 - mean[channel]) / std[channel];
            }
        }
    }
    Ok(output)
}
