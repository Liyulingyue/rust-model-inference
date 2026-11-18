//! Video transport and CLI orchestration; inference is owned by models/.
use crate::{
    app::CliOptions,
    core::tensor::TensorSource,
    models::diffusion::{
        dreamx::media::FfmpegVideoWriter,
        fastframes_l::{Engine, Options, Video},
    },
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub fn options(cli: &CliOptions, server: bool) -> Result<Options, String> {
    for flag in &cli.supplied_flags {
        let supported = matches!(
            flag.as_str(),
            "--model"
                | "--width"
                | "--height"
                | "--video-frames"
                | "--fps"
                | "--seed"
                | "--threads"
                | "--compute"
                | "--gpu"
        ) || (!server
            && matches!(
                flag.as_str(),
                "--prompt" | "--out" | "--noise" | "--overwrite"
            ))
            || (server && flag == "--serve");
        if !supported {
            return Err(format!("FastFrames-L does not support {flag}"));
        }
    }
    if server && (cli.prompt.is_some() || cli.out.is_some() || cli.noise.is_some() || cli.overwrite)
    {
        return Err("HTTP video generation accepts prompts per request; --prompt/--out/--noise/--overwrite are CLI-only".into());
    }
    let d = Options::default();
    let options = Options {
        width: cli.width.unwrap_or(d.width),
        height: cli.height.unwrap_or(d.height),
        frames: cli.astronex_frames.unwrap_or(d.frames),
        fps: cli.fps.unwrap_or(d.fps),
        seed: u64::try_from(cli.seed.unwrap_or(0))
            .map_err(|_| "FastFrames-L --seed must be non-negative")?,
    };
    options.validate()?;
    Ok(options)
}

pub fn run_cli(
    source: Arc<dyn TensorSource>,
    cli: &CliOptions,
    threads: usize,
) -> Result<(), String> {
    let options = options(cli, false)?;
    let prompt = cli
        .prompt
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .ok_or("FastFrames-L requires --prompt")?;
    let out = cli
        .out
        .as_deref()
        .ok_or("FastFrames-L requires --out video.mp4")?;
    if out.extension().and_then(|s| s.to_str()) != Some("mp4") {
        return Err("FastFrames-L --out must end in .mp4".into());
    }
    if !cli.overwrite && out.exists() {
        return Err(format!("output already exists: {}", out.display()));
    }
    if prompt.len() > 65536 {
        return Err("FastFrames-L prompt exceeds 64 KiB".into());
    }
    if let Some(path) = cli.noise.as_deref() {
        let expected =
            4 * options.latent_frames() * 16 * (options.height / 8) * (options.width / 8) * 4;
        if std::fs::metadata(path).map_err(|e| e.to_string())?.len() != expected as u64 {
            return Err(format!("shared noise must contain {expected} bytes"));
        }
    }
    let noise = cli
        .noise
        .as_deref()
        .map(crate::app::read_f32_file)
        .transpose()?;
    let video = Engine::load(source, threads)?.generate(
        prompt,
        &options,
        cli.compute_policy(),
        noise.as_deref(),
    )?;
    write_mp4(&video, out, cli.overwrite)?;
    println!(
        "{}",
        serde_json::json!({"path":out,"width":video.width,"height":video.height,"frames":video.frames,"fps":video.fps})
    );
    Ok(())
}

pub(super) fn write_mp4(video: &Video, path: &Path, overwrite: bool) -> Result<(), String> {
    if !overwrite && path.exists() {
        return Err(format!("output already exists: {}", path.display()));
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let directory = private_directory(parent)?;
    let temporary = directory.0.join("video.mp4");
    let mut writer = FfmpegVideoWriter::create(
        &temporary,
        video.width as u32,
        video.height as u32,
        video.fps as u32,
        false,
    )?;
    for frame in video.rgb.chunks_exact(video.width * video.height * 3) {
        let image =
            image::RgbImage::from_raw(video.width as u32, video.height as u32, frame.to_vec())
                .ok_or("invalid video RGB frame")?;
        writer.write_frame(&image)?;
    }
    writer.finish()?;
    if overwrite {
        std::fs::rename(&temporary, path)
    } else {
        std::fs::hard_link(&temporary, path)
    }
    .map_err(|e| format!("publish video {}: {e}", path.display()))
}

struct PrivateDirectory(PathBuf);
impl Drop for PrivateDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn private_directory(parent: &Path) -> Result<PrivateDirectory, String> {
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    let path = parent.join(format!(
        "rmi-video-{}-{:016x}",
        std::process::id(),
        rand::random::<u64>()
    ));
    builder
        .create(&path)
        .map_err(|e| format!("create private video directory: {e}"))?;
    Ok(PrivateDirectory(path))
}

pub(super) fn encode_mp4(video: &Video) -> Result<Vec<u8>, String> {
    let dir = private_directory(&std::env::temp_dir())?;
    let path = dir.0.join("video.mp4");
    write_mp4(video, &path, false)?;
    if std::fs::metadata(&path).map_err(|e| e.to_string())?.len() > 128 * 1024 * 1024 {
        return Err("encoded video exceeds 128 MiB".into());
    }
    std::fs::read(path).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_video_only_flags_and_server_cli_boundaries() {
        let parse = |extra: &[&str]| {
            let mut args = vec![
                "rust-model-inference".to_owned(),
                "--model".to_owned(),
                "ff.gguf".to_owned(),
            ];
            args.extend(extra.iter().map(|s| s.to_string()));
            crate::app::parse_cli_options(&args).unwrap()
        };
        assert!(options(
            &parse(&["--video-frames", "5", "--width", "16", "--height", "16"]),
            false
        )
        .is_ok());
        for flags in [
            &["--steps", "1"][..],
            &["--kv-cache", "f16"][..],
            &["--embedding"][..],
            &["--seed", "-1"][..],
        ] {
            assert!(options(&parse(flags), false).is_err());
        }
        assert!(options(&parse(&["--prompt", "hello"]), true).is_err());
    }
}
