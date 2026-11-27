//! Decode selected source frames; sampling and timestamps follow the pinned D3 runtime.
use base64::Engine;
use image::RgbImage;
use serde_json::Value;
use std::{
    io::{Read, Write},
    path::PathBuf,
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};

const MAX_BYTES: usize = 32_000_000;
static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

pub(super) struct Video {
    path: PathBuf,
    temporary: bool,
    pub fps: f64,
    pub total: usize,
    width: u32,
    height: u32,
}

impl Drop for Video {
    fn drop(&mut self) {
        if self.temporary {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

pub(crate) fn validate_inline(value: &Value) -> Result<(), String> {
    let text = value
        .as_str()
        .ok_or("HTTP videos require inline video data URLs")?;
    let (header, data) = text.split_once(',').ok_or("invalid video data URL")?;
    let kind = header
        .strip_prefix("data:video/")
        .and_then(|v| v.split(';').next());
    if !matches!(kind, Some("mp4" | "webm" | "quicktime" | "x-matroska"))
        || !header.ends_with(";base64")
    {
        return Err("HTTP videos require base64 MP4, WebM, QuickTime or Matroska data URLs; local files are CLI-only".into());
    }
    if data.len() > MAX_BYTES.div_ceil(3) * 4
        || base64::engine::general_purpose::STANDARD
            .decode(data)
            .map_err(|e| e.to_string())?
            .len()
            > MAX_BYTES
    {
        return Err("video exceeds 32,000,000 bytes".into());
    }
    Ok(())
}

fn sampled(total: usize, fps: f64) -> Vec<usize> {
    let count = ((total as f64 / fps * 2.) as usize).clamp(4, 32).min(total);
    let mut indices = (0..count)
        .map(|i| {
            if count == 1 {
                0
            } else {
                (i as f64 * (total - 1) as f64 / (count - 1) as f64).round_ties_even() as usize
            }
        })
        .collect::<Vec<_>>();
    if indices.len() % 2 == 1 {
        indices.push(*indices.last().unwrap());
    }
    indices
}

pub(super) fn windows(total: usize, fps: f64, rate: usize) -> Vec<Vec<usize>> {
    let step = fps / (2. * rate as f64);
    let count = ((total as f64 / step) as usize).max(1);
    let indices = (0..count)
        .map(|i| ((i as f64 * step).round_ties_even() as usize).min(total - 1))
        .collect::<Vec<_>>();
    indices
        .chunks(32)
        .map(|chunk| {
            let mut window = chunk.to_vec();
            if window.len() % 2 == 1 {
                window.push(*window.last().unwrap());
            }
            while window.len() < 4 {
                window.push(*window.last().unwrap());
            }
            window
        })
        .collect()
}

pub(super) fn times(indices: &[usize], fps: f64) -> Vec<f64> {
    indices
        .chunks_exact(2)
        .map(|p| (p[0] as f64 / fps + p[1] as f64 / fps) / 2.)
        .collect()
}

pub(super) fn prompt(indices: &[usize], fps: f64) -> String {
    let pairs = times(indices, fps)
        .into_iter()
        .map(|time| format!("<{time:.1} seconds><|vision_start|><|video_pad|><|vision_end|>"))
        .collect::<String>();
    // The processor expands video_pad inside the chat template's outer vision pair.
    format!("<|vision_start|>{pairs}<|vision_end|>")
}

impl Video {
    pub fn validate_spatial(&self) -> Result<(), String> {
        if self.total as f64 > 60. * self.fps {
            return Err("video box and track support clips of at most 60 seconds".into());
        }
        Ok(())
    }

    pub fn open(value: &Value) -> Result<Self, String> {
        let text = value
            .as_str()
            .ok_or("a video must be a file, URL or video data URL")?;
        let mut video = Self {
            path: PathBuf::from(text),
            temporary: false,
            fps: 24.,
            total: 0,
            width: 0,
            height: 0,
        };
        let data = if text.starts_with("data:") {
            validate_inline(value)?;
            Some(
                base64::engine::general_purpose::STANDARD
                    .decode(text.split_once(',').unwrap().1)
                    .map_err(|e| e.to_string())?,
            )
        } else if text.starts_with("http://") || text.starts_with("https://") {
            let mut response = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .map_err(|e| e.to_string())?
                .get(text)
                .send()
                .and_then(reqwest::blocking::Response::error_for_status)
                .map_err(|e| e.to_string())?;
            let mut bytes = Vec::new();
            response
                .by_ref()
                .take(MAX_BYTES as u64 + 1)
                .read_to_end(&mut bytes)
                .map_err(|e| e.to_string())?;
            if bytes.len() > MAX_BYTES {
                return Err("video exceeds 32,000,000 bytes".into());
            }
            Some(bytes)
        } else {
            if !video.path.is_file() {
                return Err("video file does not exist".into());
            }
            None
        };
        if let Some(data) = data {
            video.path = std::env::temp_dir().join(format!(
                "rmi-d3-video-{}-{}",
                std::process::id(),
                NEXT_FILE.fetch_add(1, Ordering::Relaxed)
            ));
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options.open(&video.path).map_err(|e| e.to_string())?;
            video.temporary = true;
            file.write_all(&data).map_err(|e| e.to_string())?;
        }
        let output = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-format_whitelist",
                "mov,matroska,webm,avi",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames",
                "-of",
                "json",
            ])
            .arg(&video.path)
            .output()
            .map_err(|e| format!("d3 video needs ffprobe on PATH: {e}"))?;
        if !output.status.success() {
            return Err("ffprobe could not read the video".into());
        }
        let probe: Value = serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())?;
        let stream = &probe["streams"][0];
        video.width = stream["width"]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(0);
        video.height = stream["height"]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(0);
        if video.width == 0
            || video.height == 0
            || u64::from(video.width) * u64::from(video.height) > 8_294_400
        {
            return Err("video frame size must be positive and at most 8,294,400 pixels".into());
        }
        if let Some((a, b)) = stream["avg_frame_rate"]
            .as_str()
            .and_then(|s| s.split_once('/'))
        {
            let fps = a.parse::<f64>().unwrap_or(0.) / b.parse::<f64>().unwrap_or(0.);
            if fps.is_finite() && (0.1..=1000.).contains(&fps) {
                video.fps = fps;
            }
        }
        video.total = stream["nb_frames"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .filter(|n| *n > 0)
            .or_else(|| {
                stream["nb_read_frames"]
                    .as_str()
                    .and_then(|s| s.parse().ok())
            })
            .unwrap_or(0);
        if video.total == 0 || video.total as f64 / video.fps > 300. {
            return Err("video must contain frames and last at most 300 seconds".into());
        }
        Ok(video)
    }

    pub fn samples(&self) -> Vec<usize> {
        sampled(self.total, self.fps)
    }

    pub fn frames(&self, indices: &[usize]) -> Result<Vec<RgbImage>, String> {
        let mut unique = indices.to_vec();
        unique.sort_unstable();
        unique.dedup();
        if unique.is_empty() || unique.last().is_some_and(|i| *i >= self.total) {
            return Err("invalid video frame indices".into());
        }
        let filter = unique
            .iter()
            .map(|i| format!("eq(n\\,{i})"))
            .collect::<Vec<_>>()
            .join("+");
        let expected = self.width as usize * self.height as usize * 3 * unique.len();
        let mut child = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-threads",
                "1",
                "-protocol_whitelist",
                "file,pipe",
                "-format_whitelist",
                "mov,matroska,webm,avi",
                "-i",
            ])
            .arg(&self.path)
            .args([
                "-map",
                "0:v:0",
                "-vf",
                &format!("select={filter}"),
                "-fps_mode",
                "passthrough",
                "-frames:v",
                &unique.len().to_string(),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("d3 video needs ffmpeg on PATH: {e}"))?;
        let mut bytes = Vec::new();
        let read = child
            .stdout
            .take()
            .unwrap()
            .take(expected as u64 + 1)
            .read_to_end(&mut bytes);
        if read.is_err() || bytes.len() != expected {
            let _ = child.kill();
            let _ = child.wait();
            return Err("video decoder returned an incomplete frame selection".into());
        }
        if !child.wait().map_err(|e| e.to_string())?.success() {
            return Err("video decoding failed".into());
        }
        let stride = self.width as usize * self.height as usize * 3;
        indices
            .iter()
            .map(|index| {
                let at = unique.binary_search(index).unwrap() * stride;
                RgbImage::from_raw(self.width, self.height, bytes[at..at + stride].to_vec())
                    .ok_or_else(|| "invalid decoded frame".into())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sampling_and_track_windows_follow_upstream_rounding() {
        assert_eq!(sampled(10, 24.), vec![0, 3, 6, 9]);
        assert_eq!(sampled(1, 24.), vec![0, 0]);
        assert_eq!(windows(5, 2., 1), vec![vec![0, 1, 2, 3, 4, 4]]);
        assert_eq!(windows(1, 24., 2), vec![vec![0, 0, 0, 0]]);
        assert_eq!(times(&[0, 3, 6, 9], 24.), vec![0.0625, 0.3125]);
        assert_eq!(prompt(&[0, 1, 2, 3], 2.), "<|vision_start|><0.2 seconds><|vision_start|><|video_pad|><|vision_end|><1.2 seconds><|vision_start|><|video_pad|><|vision_end|><|vision_end|>");
        assert!(validate_inline(&serde_json::json!("/etc/passwd")).is_err());
        assert!(validate_inline(&serde_json::json!("data:video/mp4;base64,AA==")).is_ok());
        let mut video = Video {
            path: PathBuf::new(),
            temporary: false,
            fps: 24.,
            total: 60 * 24,
            width: 64,
            height: 64,
        };
        assert!(video.validate_spatial().is_ok());
        video.total += 1;
        assert!(video.validate_spatial().is_err());
    }

    #[test]
    #[ignore = "requires ffmpeg, ffprobe and RMI_D3_VIDEO_FIXTURE on the model host"]
    fn decodes_sampled_frames_with_current_ffmpeg() {
        let video = Video::open(&serde_json::json!(
            std::env::var("RMI_D3_VIDEO_FIXTURE").unwrap()
        ))
        .unwrap();
        let indices = video.samples();
        let frames = video.frames(&indices).unwrap();
        assert_eq!(frames.len(), indices.len());
        assert!(frames
            .iter()
            .all(|frame| frame.dimensions() == (video.width, video.height)));
        let inline = format!(
            "data:video/x-matroska;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(std::fs::read(&video.path).unwrap())
        );
        let inline = Video::open(&serde_json::json!(inline)).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&inline.path)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        let path = inline.path.clone();
        drop(inline);
        assert!(!path.exists());
    }
}
