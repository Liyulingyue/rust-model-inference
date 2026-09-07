use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

use image::RgbImage;

use crate::models::qwen3::tts::codec::encode_wav_pcm16;

pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    encode_wav_pcm16(samples, sample_rate).map_err(|error| error.to_string())
}

pub fn write_wav_atomic(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    overwrite: bool,
) -> Result<(), String> {
    reject_existing(path, overwrite)?;
    let bytes = encode_wav(samples, sample_rate)?;
    let (temp_path, mut file) = reserve_sibling_temp(path)?;
    let result = (|| {
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("Write DreamX WAV {}: {error}", temp_path.display()))?;
        drop(file);
        publish_temp(&temp_path, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temp_path);
    }
    result
}

pub fn ffmpeg_video_args(
    width: u32,
    height: u32,
    fps: u32,
    output: &Path,
) -> Result<Vec<OsString>, String> {
    if width == 0 || height == 0 || fps == 0 {
        return Err("DreamX video width, height, and fps must be nonzero".into());
    }
    Ok([
        "-y".into(),
        "-f".into(),
        "rawvideo".into(),
        "-pix_fmt".into(),
        "rgb24".into(),
        "-video_size".into(),
        format!("{width}x{height}").into(),
        "-r".into(),
        fps.to_string().into(),
        "-i".into(),
        "-".into(),
        "-an".into(),
        "-c:v".into(),
        "libx264".into(),
        "-pix_fmt".into(),
        "yuv420p".into(),
        "-movflags".into(),
        "+faststart".into(),
        "-f".into(),
        "mp4".into(),
        output.as_os_str().to_owned(),
    ]
    .into())
}

pub struct FfmpegVideoWriter {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    temp_path: PathBuf,
    output_path: PathBuf,
    width: u32,
    height: u32,
    frame_bytes: usize,
}

impl FfmpegVideoWriter {
    pub fn create(
        output: &Path,
        width: u32,
        height: u32,
        fps: u32,
        overwrite: bool,
    ) -> Result<Self, String> {
        reject_existing(output, overwrite)?;
        let frame_bytes = usize::try_from(width)
            .ok()
            .and_then(|width| {
                usize::try_from(height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or("DreamX RGB24 frame size overflow")?;
        let (temp_path, reservation) = reserve_sibling_temp(output)?;
        drop(reservation);
        std::fs::remove_file(&temp_path).map_err(|error| {
            format!(
                "Prepare DreamX video temporary output {}: {error}",
                temp_path.display()
            )
        })?;
        let mut child = Command::new("ffmpeg")
            .args(ffmpeg_video_args(width, height, fps, &temp_path)?)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .map_err(|error| format!("Start ffmpeg for DreamX video: {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or("ffmpeg did not expose DreamX video stdin")?;
        Ok(Self {
            child: Some(child),
            stdin: Some(stdin),
            temp_path,
            output_path: output.to_owned(),
            width,
            height,
            frame_bytes,
        })
    }

    pub fn write_frame(&mut self, frame: &RgbImage) -> Result<(), String> {
        if frame.width() != self.width
            || frame.height() != self.height
            || frame.as_raw().len() != self.frame_bytes
        {
            return Err(format!(
                "Invalid DreamX RGB24 frame: expected {}x{} and {} bytes, got {}x{} and {} bytes",
                self.width,
                self.height,
                self.frame_bytes,
                frame.width(),
                frame.height(),
                frame.as_raw().len()
            ));
        }
        self.stdin
            .as_mut()
            .ok_or("DreamX video writer is already finished")?
            .write_all(frame.as_raw())
            .map_err(|error| format!("Write DreamX RGB24 frame to ffmpeg: {error}"))
    }

    pub fn finish(mut self) -> Result<(), String> {
        drop(self.stdin.take());
        let status = self
            .child
            .take()
            .ok_or("DreamX video writer is already finished")?
            .wait()
            .map_err(|error| format!("Wait for DreamX video ffmpeg: {error}"))?;
        if !status.success() {
            return Err(format!("DreamX video ffmpeg exited with {status}"));
        }
        File::open(&self.temp_path)
            .and_then(|file| file.sync_all())
            .map_err(|error| format!("Sync DreamX video output: {error}"))?;
        publish_temp(&self.temp_path, &self.output_path)
    }
}

impl Drop for FfmpegVideoWriter {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        let _ = std::fs::remove_file(&self.temp_path);
    }
}

pub fn mux_audio_atomic(
    video: &Path,
    audio: &Path,
    output: &Path,
    overwrite: bool,
) -> Result<(), String> {
    reject_existing(output, overwrite)?;
    let (temp_path, reservation) = reserve_sibling_temp(output)?;
    drop(reservation);
    std::fs::remove_file(&temp_path).map_err(|error| {
        format!(
            "Prepare DreamX mux temporary output {}: {error}",
            temp_path.display()
        )
    })?;
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(video)
        .arg("-i")
        .arg(audio)
        .args(["-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-shortest"])
        .arg("-f")
        .arg("mp4")
        .arg(&temp_path)
        .status()
        .map_err(|error| format!("Start ffmpeg for DreamX mux: {error}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&temp_path);
        return Err(format!("DreamX mux ffmpeg exited with {status}"));
    }
    let result = File::open(&temp_path)
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("Sync DreamX mux output: {error}"))
        .and_then(|()| publish_temp(&temp_path, output));
    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

fn reject_existing(path: &Path, overwrite: bool) -> Result<(), String> {
    if path.exists() && !overwrite {
        return Err(format!(
            "DreamX output {} already exists; pass --overwrite to replace it",
            path.display()
        ));
    }
    Ok(())
}

fn reserve_sibling_temp(path: &Path) -> Result<(PathBuf, File), String> {
    let file_name = path
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or("DreamX output path requires a file name")?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    for counter in 0..256u16 {
        let mut temp_name = OsString::from(".");
        temp_name.push(file_name);
        temp_name.push(format!(".tmp-{}-{counter}", std::process::id()));
        let temp_path = parent.join(temp_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => return Ok((temp_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!(
                    "Create DreamX temporary output {}: {error}",
                    temp_path.display()
                ));
            }
        }
    }
    Err("Could not create a unique DreamX temporary output".into())
}

fn publish_temp(temp_path: &Path, output: &Path) -> Result<(), String> {
    std::fs::rename(temp_path, output).map_err(|error| {
        format!(
            "Publish DreamX output {} from {}: {error}",
            output.display(),
            temp_path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn wav_header_declares_mono_48khz_pcm16() {
        let bytes = encode_wav(&[0.0, 1.0, -1.0], 48_000).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 1);
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            48_000
        );
        assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 16);
    }

    #[test]
    fn video_writer_uses_raw_rgb24_and_requested_fps() {
        let args = ffmpeg_video_args(96, 64, 24, Path::new("out.mp4")).unwrap();
        assert!(args
            .windows(2)
            .any(|values| values == ["-pix_fmt", "rgb24"]));
        assert!(args.windows(2).any(|values| values == ["-r", "24"]));
    }
}
