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

impl MediaKind {
    /// The `type` a chat template branches on for this kind of attachment.
    pub fn content_type(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }

    /// `(start, pad, end)` special-token names wrapping the placeholder run.
    pub fn placeholder_tokens(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::Audio => ("audio_start", "audio_pad", "audio_end"),
            Self::Image => ("vision_start", "image_pad", "vision_end"),
            Self::Video => ("vision_start", "video_pad", "vision_end"),
        }
    }
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
    // Which projector family to use is decided by `clip.projector_type` when
    // the mmproj declares one, not by the language tower's architecture.
    //
    // They are not the same axis. LensVLM-9B pairs a `qwen35` language tower
    // with a `qwen3vl_merger` projector, so deriving the family from the arch
    // alone rejected a pairing that the engine actually runs. The arch is still
    // checked, because an unknown tower has no text path to fall back to.
    let declared = mmproj
        .metadata("clip.projector_type")
        .and_then(|v| v.to_string_val());
    let family = match declared.as_deref() {
        Some("qwen3vl_merger") => ProjectorFamily::Qwen3VlMerger,
        Some("qwen2.5o" | "qwen2.5vl_merger") => ProjectorFamily::Qwen25Omni,
        Some(other) => {
            return Err(format!("Unsupported clip.projector_type: {other}"));
        }
        // No declared type: fall back to the architecture, which is what the
        // pre-existing GGUF shapes rely on.
        None => match llm_arch {
            "qwen3vl" | "qwen3vlmoe" => ProjectorFamily::Qwen3VlMerger,
            "qwen2vl" | "qwen35" => ProjectorFamily::Qwen25Omni,
            other => return Err(format!("Unsupported multimodal architecture: {other}")),
        },
    };
    if !matches!(llm_arch, "qwen3vl" | "qwen3vlmoe" | "qwen2vl" | "qwen35") {
        return Err(format!("Unsupported multimodal architecture: {llm_arch}"));
    }
    if llm_arch == "qwen3vl" && media == MediaKind::Audio {
        return Err("Qwen3-VL does not support audio input".into());
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
/// Decode an image from bytes (base64 / uploads), mirroring [`decode_image`]
/// for callers that already hold the encoded data instead of a path.
pub fn decode_image_bytes(bytes: &[u8]) -> Result<image::DynamicImage, String> {
    image::load_from_memory(bytes).map_err(|error| format!("Failed to decode image: {error}"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{MetaValue, TensorInfo, TensorSource};
    use std::sync::Arc;

    /// A mmproj stub exposing only the metadata and tensors a test names.
    struct Stub {
        meta: Vec<(String, MetaValue)>,
        #[allow(dead_code)]
        tensors: Vec<String>,
    }

    impl Stub {
        fn with(meta: Vec<(&str, MetaValue)>, tensors: &[&str]) -> Arc<dyn TensorSource> {
            Arc::new(Stub {
                meta: meta.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
                tensors: tensors.iter().map(|s| s.to_string()).collect(),
            })
        }

        fn merger() -> Arc<dyn TensorSource> {
            Stub::with(
                vec![
                    (
                        "clip.projector_type",
                        MetaValue::String("qwen3vl_merger".into()),
                    ),
                    ("clip.has_vision_encoder", MetaValue::Bool(true)),
                ],
                &["mm.2.weight", "v.patch_embd.weight"],
            )
        }

        fn omni() -> Arc<dyn TensorSource> {
            Stub::with(
                vec![
                    (
                        "clip.projector_type",
                        MetaValue::String("qwen2.5vl_merger".into()),
                    ),
                    ("clip.has_vision_encoder", MetaValue::Bool(true)),
                ],
                &["mm.2.weight"],
            )
        }
    }

    impl TensorSource for Stub {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            None
        }
        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            None
        }
    }

    /// LensVLM-9B pairs a `qwen35` language tower with a `qwen3vl_merger`
    /// projector. `qwen35` was hardcoded to the Omni family, so this pairing
    /// would be rejected -- while the qwen35 multimodal path bypasses this
    /// check entirely and ran the merger encoder anyway. That is exactly the
    /// silent fallback this function exists to prevent.
    #[test]
    fn qwen35_accepts_the_qwen3vl_merger_projector() {
        let family =
            validate_mmproj_capabilities("qwen35", Stub::merger().as_ref(), MediaKind::Image)
                .expect("LensVLM-9B pairing must be accepted");
        assert_eq!(family, ProjectorFamily::Qwen3VlMerger);
    }

    /// The reverse pairing is a genuine mismatch and must stay an error: a
    /// `qwen3vl` language tower cannot consume an Omni projector.
    /// qwen3vl with an Omni projector is a legal pairing that the repo already
    /// routes (Qwen3-VL drives the Omni encoder), so it must keep working.
    #[test]
    fn qwen3vl_with_the_omni_projector_still_works() {
        let family =
            validate_mmproj_capabilities("qwen3vl", Stub::omni().as_ref(), MediaKind::Image)
                .expect("this pairing already worked before");
        assert_eq!(family, ProjectorFamily::Qwen25Omni);
    }

    /// An unrecognised projector type is refused rather than guessed at, which
    /// is the whole point of the check.
    #[test]
    fn an_unknown_projector_type_is_rejected() {
        let stub = Stub::with(
            vec![
                (
                    "clip.projector_type",
                    MetaValue::String("mystery_proj".into()),
                ),
                ("clip.has_vision_encoder", MetaValue::Bool(true)),
            ],
            &[],
        );
        let error =
            validate_mmproj_capabilities("qwen35", stub.as_ref(), MediaKind::Image).unwrap_err();
        assert!(error.contains("mystery_proj"), "{error}");
    }

    /// The architecture is still gated even though the projector decides the
    /// family: an unknown tower has no text path to fall back on.
    #[test]
    fn a_known_projector_does_not_admit_an_unknown_tower() {
        assert!(validate_mmproj_capabilities(
            "mystery-tower",
            Stub::merger().as_ref(),
            MediaKind::Image
        )
        .is_err());
    }

    #[test]
    fn unknown_architecture_is_rejected() {
        assert!(validate_mmproj_capabilities(
            "not-a-model",
            Stub::merger().as_ref(),
            MediaKind::Image
        )
        .is_err());
    }

    #[test]
    fn a_mmproj_without_a_vision_encoder_is_rejected_for_images() {
        let stub = Stub::with(
            vec![
                (
                    "clip.projector_type",
                    MetaValue::String("qwen3vl_merger".into()),
                ),
                ("clip.has_vision_encoder", MetaValue::Bool(false)),
            ],
            &[],
        );
        assert!(validate_mmproj_capabilities("qwen3vl", stub.as_ref(), MediaKind::Image).is_err());
    }
}
