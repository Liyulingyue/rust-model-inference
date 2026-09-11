use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

pub use crate::core::scratchpad::KvFormat;
pub use crate::models::diffusion::dreamx::{
    DreamXOptions, DreamXRefinerOptions, LatentUpsampleKind, RefinerDecoderKind,
};
use crate::models::dots::XVectorMode;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EmbeddingOutput {
    #[default]
    Summary,
    Raw,
}

#[derive(Debug, Default)]
pub struct CliOptions {
    pub dreamx: bool,
    pub planner: Option<PathBuf>,
    pub perception: Option<PathBuf>,
    pub scenes: Option<PathBuf>,
    pub image_root: Option<PathBuf>,
    pub frames: Option<PathBuf>,
    pub planning_mode: Option<String>,
    pub num_samples: Option<usize>,
    pub num_steps: Option<usize>,
    pub output: Option<PathBuf>,
    pub model: PathBuf,
    pub mmproj: Option<PathBuf>,
    pub audio: Option<PathBuf>,
    pub ref_audio: Option<PathBuf>,
    pub ref_text: Option<String>,
    pub image: Option<PathBuf>,
    pub video: Option<PathBuf>,
    pub vae: Option<PathBuf>,
    pub text_encoder: Option<PathBuf>,
    pub prompt: Option<String>,
    pub negative_prompt: Option<String>,
    pub language: Option<String>,
    pub max_tokens: Option<usize>,
    pub steps: Option<usize>,
    pub resolution: Option<usize>,
    pub seed: Option<i64>,
    pub duration_seconds: Option<f32>,
    pub fps: Option<usize>,
    pub target_spatial_tokens: Option<usize>,
    pub refine: Option<bool>,
    pub refiner_kv_len: Option<usize>,
    pub latent_upsample: Option<LatentUpsampleKind>,
    pub refiner_decoder: Option<RefinerDecoderKind>,
    pub dry_run: bool,
    pub overwrite: bool,
    pub allow_memory_overcommit: bool,
    pub temperature: Option<f32>,
    pub threads: usize,
    pub thinking: bool,
    pub embedding: bool,
    pub embedding_output: EmbeddingOutput,
    pub dump_logits: bool,
    pub bench: bool,
    pub profile: bool,
    pub kv_format: KvFormat,
    pub gpu: bool,
    pub tts: bool,
    pub edit: bool,
    pub source_audio: Option<PathBuf>,
    pub source_text: Option<String>,
    pub target_text: Option<String>,
    pub instruction: Option<String>,
    pub use_xvector: XVectorMode,
    pub use_xvector_supplied: bool,
    pub out: Option<PathBuf>,
}

#[derive(Debug)]
pub struct ZImageCliOptions {
    pub steps: usize,
    pub resolution: usize,
    pub seed: i64,
    pub out: PathBuf,
}

#[derive(Debug, PartialEq)]
pub struct DreamXCliOptions {
    pub model: PathBuf,
    pub mmproj: PathBuf,
    pub image: PathBuf,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub out: PathBuf,
    pub options: DreamXOptions,
    pub dry_run: bool,
    pub overwrite: bool,
    pub allow_memory_overcommit: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum QwenDriveHead {
    Planner(PathBuf),
    Perception(PathBuf),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanningMode {
    Direct,
    Reasoning,
}

#[derive(Debug, PartialEq, Eq)]
pub struct QwenDriveCliOptions {
    pub model: PathBuf,
    pub mmproj: PathBuf,
    pub head: QwenDriveHead,
    pub mode: PlanningMode,
    pub scenes: Option<PathBuf>,
    pub image_root: Option<PathBuf>,
    pub frames: Option<PathBuf>,
    pub output: PathBuf,
    pub samples: usize,
    pub steps: usize,
    pub seed: i64,
}

pub fn parse_embedding_output(value: Option<&str>) -> Result<EmbeddingOutput, String> {
    match value {
        Some("summary") => Ok(EmbeddingOutput::Summary),
        Some("raw") => Ok(EmbeddingOutput::Raw),
        Some(value) => Err(format!(
            "Invalid --embedding-output {value:?}; expected summary or raw"
        )),
        None => Err("Missing value for --embedding-output".into()),
    }
}

pub fn normalize_tts_language(language: Option<&str>) -> Result<&'static str, String> {
    match language.unwrap_or("en").to_ascii_lowercase().as_str() {
        "cn" | "zh" | "chinese" => Ok("chinese"),
        "en" | "english" => Ok("english"),
        "ge" | "de" | "german" => Ok("german"),
        "it" | "italian" => Ok("italian"),
        "po" | "pt" | "portuguese" => Ok("portuguese"),
        "sp" | "es" | "spanish" => Ok("spanish"),
        "ja" | "japanese" => Ok("japanese"),
        "ko" | "korean" => Ok("korean"),
        "fr" | "french" => Ok("french"),
        "ru" | "russian" => Ok("russian"),
        value => Err(format!("Unsupported TTS language {value:?}")),
    }
}

pub fn validate_qwen3vl_decoder_mode(
    arch: &str,
    dump_logits: bool,
    bench: bool,
    profile: bool,
    kv_format: KvFormat,
    interactive: bool,
) -> Result<(), String> {
    if arch != "qwen3vl" {
        return Ok(());
    }
    let unsupported = if dump_logits {
        Some("--dump-logits")
    } else if bench {
        Some("--bench")
    } else if profile {
        Some("--profile")
    } else if kv_format == KvFormat::F32 {
        Some("--kv-cache f32")
    } else if interactive {
        Some("interactive mode")
    } else {
        None
    };
    match unsupported {
        Some(option) => Err(format!(
            "{option} is not supported for qwen3vl; use default F16 generation"
        )),
        None => Ok(()),
    }
}

pub const DEFAULT_THREAD_CAP: usize = 8;

pub fn resolve_thread_count(requested: usize, available: usize) -> usize {
    if requested > 0 {
        requested
    } else {
        available.clamp(1, DEFAULT_THREAD_CAP)
    }
}

/// Initialize rayon's global thread pool to match the resolved thread
/// count. Idempotent: subsequent calls (or env-var-only setups) silently
/// succeed because `build_global` errors after the first call.
///
/// See the TODO at the top of `src/core/thread_pool.rs` for the rationale
/// of the two-pool model and the preferred direction for unification.
pub fn init_rayon_global_pool(thread_count: usize) {
    let n = thread_count.max(1);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global();
}

pub fn inference_step_budget(prompt_tokens: usize, max_tokens: usize, bench: bool) -> usize {
    prompt_tokens
        + if bench {
            max_tokens
        } else {
            max_tokens.saturating_sub(1)
        }
}

pub fn per_second(count: usize, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds > 0.0 {
        count as f64 / seconds
    } else {
        0.0
    }
}

pub fn parse_cli_options(args: &[String]) -> Result<CliOptions, String> {
    let mut options = CliOptions::default();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dreamx" => options.dreamx = true,
            "--planner" => {
                options.planner = Some(required_path_value(args, &mut i, "--planner")?);
            }
            "--perception" => {
                options.perception = Some(required_path_value(args, &mut i, "--perception")?);
            }
            "--scenes" => {
                options.scenes = Some(required_path_value(args, &mut i, "--scenes")?);
            }
            "--image-root" => {
                options.image_root = Some(required_path_value(args, &mut i, "--image-root")?);
            }
            "--frames" => {
                options.frames = Some(required_path_value(args, &mut i, "--frames")?);
            }
            "--mode" => {
                options.planning_mode = Some(required_string_value(args, &mut i, "--mode")?);
            }
            "--num-samples" => {
                options.num_samples = Some(required_usize_value(args, &mut i, "--num-samples")?);
            }
            "--num-steps" => {
                options.num_steps = Some(required_usize_value(args, &mut i, "--num-steps")?);
            }
            "--output" => {
                options.output = Some(required_path_value(args, &mut i, "--output")?);
            }
            "--model" => {
                if i + 1 < args.len() {
                    options.model = args[i + 1].as_str().into();
                    i += 1;
                }
            }
            "--prompt" => {
                if i + 1 < args.len() {
                    options.prompt = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--negative-prompt" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.starts_with("--"))
                    .ok_or("Missing value for --negative-prompt")?;
                options.negative_prompt = Some(value.clone());
                i += 1;
            }
            "--max-tokens" | "--n-gen" => {
                if i + 1 < args.len() {
                    options.max_tokens = Some(args[i + 1].parse().unwrap_or(128));
                    i += 1;
                }
            }
            "--steps" => {
                let value = args.get(i + 1).ok_or("Missing value for --steps")?;
                options.steps = Some(
                    value
                        .parse::<usize>()
                        .map_err(|error| format!("Invalid --steps value: {error}"))?,
                );
                i += 1;
            }
            "--resolution" | "--size" => {
                let flag = args[i].clone();
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| format!("Missing value for {flag}"))?;
                options.resolution = Some(
                    value
                        .parse::<usize>()
                        .map_err(|error| format!("Invalid {flag} value: {error}"))?,
                );
                i += 1;
            }
            "--seed" => {
                let value = args.get(i + 1).ok_or("Missing value for --seed")?;
                options.seed = Some(
                    value
                        .parse::<i64>()
                        .map_err(|error| format!("Invalid --seed value: {error}"))?,
                );
                i += 1;
            }
            "--duration" => {
                let value = args.get(i + 1).ok_or("Missing value for --duration")?;
                options.duration_seconds = Some(
                    value
                        .parse::<f32>()
                        .map_err(|error| format!("Invalid --duration value: {error}"))?,
                );
                i += 1;
            }
            "--fps" => {
                let value = args.get(i + 1).ok_or("Missing value for --fps")?;
                options.fps = Some(
                    value
                        .parse::<usize>()
                        .map_err(|error| format!("Invalid --fps value: {error}"))?,
                );
                i += 1;
            }
            "--target-spatial-tokens" => {
                let value = args
                    .get(i + 1)
                    .ok_or("Missing value for --target-spatial-tokens")?;
                options.target_spatial_tokens =
                    Some(value.parse::<usize>().map_err(|error| {
                        format!("Invalid --target-spatial-tokens value: {error}")
                    })?);
                i += 1;
            }
            "--refine" => options.refine = Some(true),
            "--no-refine" => options.refine = Some(false),
            "--refiner-kv-len" => {
                let value = args
                    .get(i + 1)
                    .ok_or("Missing value for --refiner-kv-len")?;
                options.refiner_kv_len = Some(
                    value
                        .parse::<usize>()
                        .map_err(|error| format!("Invalid --refiner-kv-len value: {error}"))?,
                );
                i += 1;
            }
            "--latent-upsample" => {
                let value = args
                    .get(i + 1)
                    .ok_or("Missing value for --latent-upsample")?;
                options.latent_upsample = Some(match value.as_str() {
                    "bilinear" => LatentUpsampleKind::Bilinear,
                    "flash" => LatentUpsampleKind::Flash,
                    "causal2d" => LatentUpsampleKind::Causal2d,
                    _ => {
                        return Err(format!(
                            "Invalid --latent-upsample {value:?}; expected bilinear, flash, or causal2d"
                        ));
                    }
                });
                i += 1;
            }
            "--refiner-decoder" => {
                let value = args
                    .get(i + 1)
                    .ok_or("Missing value for --refiner-decoder")?;
                options.refiner_decoder = Some(match value.as_str() {
                    "wan" => RefinerDecoderKind::Wan,
                    "lightvae" => RefinerDecoderKind::LightVae,
                    _ => {
                        return Err(format!(
                            "Invalid --refiner-decoder {value:?}; expected wan or lightvae"
                        ));
                    }
                });
                i += 1;
            }
            "--dry-run" => options.dry_run = true,
            "--overwrite" => options.overwrite = true,
            "--allow-memory-overcommit" => options.allow_memory_overcommit = true,
            "--temp" => {
                if i + 1 < args.len() {
                    options.temperature = Some(args[i + 1].parse().unwrap_or(0.6));
                    i += 1;
                }
            }
            "--threads" => {
                if i + 1 < args.len() {
                    options.threads = args[i + 1].parse().unwrap_or(0);
                    i += 1;
                }
            }
            "--dump-logits" => options.dump_logits = true,
            "--embedding" => options.embedding = true,
            "--embedding-output" => {
                options.embedding_output =
                    parse_embedding_output(args.get(i + 1).map(String::as_str))?;
                i += 1;
            }
            "--bench" => options.bench = true,
            "--thinking" => options.thinking = true,
            "--profile" => options.profile = true,
            "--gpu" => options.gpu = true,
            "--kv-cache" => {
                if i + 1 < args.len() {
                    options.kv_format = match args[i + 1].as_str() {
                        "f32" => KvFormat::F32,
                        _ => KvFormat::F16,
                    };
                    i += 1;
                }
            }
            "--mmproj" => {
                if i + 1 < args.len() {
                    options.mmproj = Some(args[i + 1].as_str().into());
                    i += 1;
                }
            }
            "--image" => {
                if i + 1 < args.len() {
                    options.image = Some(args[i + 1].as_str().into());
                    i += 1;
                }
            }
            "--video" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.is_empty() && !value.starts_with("--"))
                    .ok_or("Missing value for --video")?;
                options.video = Some(value.as_str().into());
                i += 1;
            }
            "--vae" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.is_empty() && !value.starts_with("--"))
                    .ok_or("Missing value for --vae")?;
                options.vae = Some(value.as_str().into());
                i += 1;
            }
            "--text-encoder" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.is_empty() && !value.starts_with("--"))
                    .ok_or("Missing value for --text-encoder")?;
                options.text_encoder = Some(value.as_str().into());
                i += 1;
            }
            "--audio" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.is_empty() && !value.starts_with("--"))
                    .ok_or("Missing value for --audio")?;
                options.audio = Some(value.as_str().into());
                i += 1;
            }
            "--ref-audio" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.is_empty() && !value.starts_with("--"))
                    .ok_or("Missing value for --ref-audio")?;
                options.ref_audio = Some(value.as_str().into());
                i += 1;
            }
            "--ref-text" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.is_empty() && !value.starts_with("--"))
                    .ok_or("Missing value for --ref-text")?;
                options.ref_text = Some(value.clone());
                i += 1;
            }
            "--source-audio" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.is_empty() && !value.starts_with("--"))
                    .ok_or("Missing value for --source-audio")?;
                options.source_audio = Some(value.as_str().into());
                i += 1;
            }
            "--source-text" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.is_empty() && !value.starts_with("--"))
                    .ok_or("Missing value for --source-text")?;
                options.source_text = Some(value.clone());
                i += 1;
            }
            "--target-text" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.is_empty() && !value.starts_with("--"))
                    .ok_or("Missing value for --target-text")?;
                options.target_text = Some(value.clone());
                i += 1;
            }
            "--instruction" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.is_empty() && !value.starts_with("--"))
                    .ok_or("Missing value for --instruction")?;
                options.instruction = Some(value.clone());
                i += 1;
            }
            "--use-xvector" => {
                let value = args.get(i + 1).ok_or("Missing value for --use-xvector")?;
                options.use_xvector = XVectorMode::from_str(value)?;
                options.use_xvector_supplied = true;
                i += 1;
            }
            "--language" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.starts_with("--"))
                    .ok_or("Missing value for --language")?;
                options.language = Some(value.clone());
                i += 1;
            }
            "--tts" => options.tts = true,
            "--edit" => options.edit = true,
            "--out" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.starts_with("--"))
                    .ok_or("Missing value for --out")?;
                options.out = Some(value.as_str().into());
                i += 1;
            }
            _ => {
                if options.audio.is_none() && !args[i].starts_with("--") && !args[i].is_empty() {
                    options.audio = Some(args[i].as_str().into());
                }
            }
        }
        i += 1;
    }
    Ok(options)
}

fn required_string_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    let value = args
        .get(*index + 1)
        .filter(|value| !value.is_empty() && !value.starts_with("--"))
        .ok_or_else(|| format!("Missing value for {flag}"))?;
    *index += 1;
    Ok(value.clone())
}

fn required_path_value(args: &[String], index: &mut usize, flag: &str) -> Result<PathBuf, String> {
    required_string_value(args, index, flag).map(PathBuf::from)
}

fn required_usize_value(args: &[String], index: &mut usize, flag: &str) -> Result<usize, String> {
    required_string_value(args, index, flag)?
        .parse()
        .map_err(|error| format!("Invalid {flag} value: {error}"))
}

pub fn qwen_drive_cli_options(
    options: &CliOptions,
) -> Result<Option<QwenDriveCliOptions>, String> {
    let requested = options.planner.is_some()
        || options.perception.is_some()
        || options.scenes.is_some()
        || options.image_root.is_some()
        || options.frames.is_some()
        || options.planning_mode.is_some()
        || options.num_samples.is_some()
        || options.num_steps.is_some()
        || options.output.is_some();
    if !requested {
        return Ok(None);
    }
    let head = match (&options.planner, &options.perception) {
        (Some(_), Some(_)) => return Err("--planner and --perception are mutually exclusive".into()),
        (Some(path), None) => QwenDriveHead::Planner(path.clone()),
        (None, Some(path)) => QwenDriveHead::Perception(path.clone()),
        (None, None) => return Err("Qwen-Drive requires --planner or --perception".into()),
    };
    if options.model.as_os_str().is_empty() {
        return Err("Qwen-Drive requires --model".into());
    }
    let mmproj = options
        .mmproj
        .clone()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("Qwen-Drive requires --mmproj")?;
    let output = options
        .output
        .clone()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("Qwen-Drive requires --output")?;
    let conflict = if options.dreamx {
        Some("--dreamx")
    } else if options.tts {
        Some("--tts")
    } else if options.edit {
        Some("--edit")
    } else if options.audio.is_some() {
        Some("--audio")
    } else if options.image.is_some() {
        Some("--image")
    } else if options.video.is_some() {
        Some("--video")
    } else if options.vae.is_some() || options.text_encoder.is_some() {
        Some("Z-Image flags")
    } else if options.embedding {
        Some("--embedding")
    } else if options.out.is_some() {
        Some("--out")
    } else {
        None
    };
    if let Some(conflict) = conflict {
        return Err(format!("Qwen-Drive cannot be used with {conflict}"));
    }

    let (mode, scenes, image_root, frames, samples, steps, seed) = match head {
        QwenDriveHead::Planner(_) => {
            let mode = match options.planning_mode.as_deref() {
                Some("direct_planning") => PlanningMode::Direct,
                Some("reasoning_planning") => PlanningMode::Reasoning,
                Some(value) => {
                    return Err(format!(
                        "Invalid --mode {value:?}; expected direct_planning or reasoning_planning"
                    ));
                }
                None => return Err("--planner requires --mode".into()),
            };
            let scenes = options
                .scenes
                .clone()
                .filter(|path| !path.as_os_str().is_empty())
                .ok_or("--planner requires --scenes")?;
            let image_root = options
                .image_root
                .clone()
                .filter(|path| !path.as_os_str().is_empty())
                .ok_or("--planner requires --image-root")?;
            if options.frames.is_some() {
                return Err("--frames requires --perception".into());
            }
            let samples = options.num_samples.ok_or("--planner requires --num-samples")?;
            let steps = options.num_steps.ok_or("--planner requires --num-steps")?;
            let seed = options.seed.ok_or("--planner requires --seed")?;
            if samples == 0 || steps == 0 {
                return Err("--num-samples and --num-steps must be greater than zero".into());
            }
            (mode, Some(scenes), Some(image_root), None, samples, steps, seed)
        }
        QwenDriveHead::Perception(_) => {
            if options.scenes.is_some()
                || options.image_root.is_some()
                || options.planning_mode.is_some()
                || options.num_samples.is_some()
                || options.num_steps.is_some()
                || options.seed.is_some()
            {
                return Err("Planning flags require --planner".into());
            }
            let frames = options
                .frames
                .clone()
                .filter(|path| !path.as_os_str().is_empty())
                .ok_or("--perception requires --frames")?;
            (PlanningMode::Direct, None, None, Some(frames), 1, 10, 42)
        }
    };

    Ok(Some(QwenDriveCliOptions {
        model: options.model.clone(),
        mmproj,
        head,
        mode,
        scenes,
        image_root,
        frames,
        output,
        samples,
        steps,
        seed,
    }))
}

pub fn dreamx_cli_options(options: &CliOptions) -> Result<Option<DreamXCliOptions>, String> {
    let unique_option = if options.negative_prompt.is_some() {
        Some("--negative-prompt")
    } else if options.duration_seconds.is_some() {
        Some("--duration")
    } else if options.fps.is_some() {
        Some("--fps")
    } else if options.target_spatial_tokens.is_some() {
        Some("--target-spatial-tokens")
    } else if options.refine.is_some() {
        Some("--refine/--no-refine")
    } else if options.refiner_kv_len.is_some() {
        Some("--refiner-kv-len")
    } else if options.latent_upsample.is_some() {
        Some("--latent-upsample")
    } else if options.refiner_decoder.is_some() {
        Some("--refiner-decoder")
    } else if options.dry_run {
        Some("--dry-run")
    } else if options.overwrite {
        Some("--overwrite")
    } else if options.allow_memory_overcommit {
        Some("--allow-memory-overcommit")
    } else {
        None
    };
    if !options.dreamx {
        return match unique_option {
            Some(option) => Err(format!("{option} requires --dreamx")),
            None => Ok(None),
        };
    }

    let conflict = if options.tts {
        Some("--tts")
    } else if options.edit {
        Some("--edit")
    } else if options.source_audio.is_some() {
        Some("--source-audio")
    } else if options.source_text.is_some() {
        Some("--source-text")
    } else if options.target_text.is_some() {
        Some("--target-text")
    } else if options.instruction.is_some() {
        Some("--instruction")
    } else if options.use_xvector_supplied {
        Some("--use-xvector")
    } else if options.audio.is_some() {
        Some("--audio")
    } else if options.video.is_some() {
        Some("--video")
    } else if options.vae.is_some() {
        Some("--vae")
    } else if options.text_encoder.is_some() {
        Some("--text-encoder")
    } else if options.ref_audio.is_some() {
        Some("--ref-audio")
    } else if options.ref_text.is_some() {
        Some("--ref-text")
    } else if options.embedding {
        Some("--embedding")
    } else if options.dump_logits {
        Some("--dump-logits")
    } else if options.bench {
        Some("--bench")
    } else if options.profile {
        Some("--profile")
    } else if options.gpu {
        Some("--gpu")
    } else if options.thinking {
        Some("--thinking")
    } else if options.language.is_some() {
        Some("--language")
    } else if options.max_tokens.is_some() {
        Some("--max-tokens")
    } else if options.temperature.is_some() {
        Some("--temp")
    } else if options.resolution.is_some() {
        Some("--resolution")
    } else {
        None
    };
    if let Some(conflict) = conflict {
        return Err(format!("--dreamx cannot be used with {conflict}"));
    }

    if options.model.as_os_str().is_empty() {
        return Err("--dreamx requires --model".into());
    }
    let model = options.model.clone();
    let mmproj = options
        .mmproj
        .clone()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("--dreamx requires --mmproj")?;
    let image = options
        .image
        .clone()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("--dreamx requires --image")?;
    let prompt = options
        .prompt
        .clone()
        .filter(|prompt| !prompt.trim().is_empty())
        .ok_or("--dreamx requires a non-empty --prompt")?;
    let out = options
        .out
        .clone()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("--dreamx requires --out")?;
    if options
        .negative_prompt
        .as_deref()
        .is_some_and(|prompt| prompt.trim().is_empty())
    {
        return Err("--negative-prompt must not be empty".into());
    }

    let defaults = DreamXOptions::default();
    let duration_seconds = options
        .duration_seconds
        .unwrap_or(defaults.duration_seconds);
    let fps = options.fps.unwrap_or(defaults.fps);
    let steps = options.steps.unwrap_or(defaults.steps);
    let target_spatial_tokens = options
        .target_spatial_tokens
        .unwrap_or(defaults.target_spatial_tokens);
    let kv_len = options.refiner_kv_len.unwrap_or(defaults.refiner.kv_len);
    if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        return Err("--dreamx requires a finite positive --duration".into());
    }
    if fps == 0 || steps == 0 || target_spatial_tokens == 0 || kv_len == 0 {
        return Err(
            "--dreamx requires positive --fps, --steps, --target-spatial-tokens, and --refiner-kv-len"
                .into(),
        );
    }

    Ok(Some(DreamXCliOptions {
        model,
        mmproj,
        image,
        prompt,
        negative_prompt: options.negative_prompt.clone(),
        out,
        options: DreamXOptions {
            duration_seconds,
            fps,
            steps,
            seed: options.seed.unwrap_or(defaults.seed),
            target_spatial_tokens,
            refine: options.refine.unwrap_or(defaults.refine),
            refiner: DreamXRefinerOptions {
                kv_len,
                latent_upsample: options
                    .latent_upsample
                    .unwrap_or(defaults.refiner.latent_upsample),
                decoder: options.refiner_decoder.unwrap_or(defaults.refiner.decoder),
            },
        },
        dry_run: options.dry_run,
        overwrite: options.overwrite,
        allow_memory_overcommit: options.allow_memory_overcommit,
    }))
}

pub fn z_image_cli_options(options: &CliOptions) -> Result<Option<ZImageCliOptions>, String> {
    if options.text_encoder.is_none() && options.vae.is_none() {
        return if options.seed.is_some()
            && options.planner.is_none()
            && options.perception.is_none()
            && !options.tts
            && !options.dreamx
        {
            Err("--seed requires Z-Image components or --tts".into())
        } else {
            Ok(None)
        };
    }
    let conflict = if options.tts {
        Some("--tts")
    } else if options.audio.is_some() {
        Some("--audio")
    } else if options.ref_audio.is_some() {
        Some("--ref-audio")
    } else if options.image.is_some() {
        Some("--image")
    } else if options.video.is_some() {
        Some("--video")
    } else if options.mmproj.is_some() {
        Some("--mmproj")
    } else if options.embedding {
        Some("--embedding")
    } else if options.dump_logits {
        Some("--dump-logits")
    } else if options.bench {
        Some("--bench")
    } else if options.profile {
        Some("--profile")
    } else if options.gpu {
        Some("--gpu")
    } else if options.thinking {
        Some("--thinking")
    } else if options.language.is_some() {
        Some("--language")
    } else if options.max_tokens.is_some() {
        Some("--max-tokens")
    } else if options.temperature.is_some() {
        Some("--temp")
    } else if options.embedding_output != EmbeddingOutput::Summary {
        Some("--embedding-output")
    } else {
        None
    };
    if let Some(conflict) = conflict {
        return Err(format!("Z-Image cannot be used with {conflict}"));
    }
    if options.model.as_os_str().is_empty() {
        return Err("Z-Image requires --model for the diffusion component".into());
    }
    options
        .text_encoder
        .as_ref()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("Z-Image requires --text-encoder")?;
    options
        .vae
        .as_ref()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("Z-Image requires --vae")?;
    let out = options
        .out
        .clone()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("Z-Image requires --out")?;
    if options
        .prompt
        .as_deref()
        .is_none_or(|prompt| prompt.trim().is_empty())
    {
        return Err("Z-Image requires a non-empty --prompt".into());
    }
    let steps = options.steps.unwrap_or(8);
    let resolution = options.resolution.unwrap_or(512);
    if steps == 0 || resolution == 0 || resolution % 16 != 0 {
        return Err("Z-Image requires positive --steps and --resolution divisible by 16".into());
    }
    Ok(Some(ZImageCliOptions {
        steps,
        resolution,
        seed: options.seed.unwrap_or(0),
        out,
    }))
}

pub fn validate_cli_options(options: &CliOptions) -> Result<(), String> {
    if qwen_drive_cli_options(options)?.is_some() {
        return Ok(());
    }
    if dreamx_cli_options(options)?.is_some() {
        return Ok(());
    }
    if !options.edit
        && (options.source_audio.is_some()
            || options.source_text.is_some()
            || options.target_text.is_some()
            || options.instruction.is_some()
            || options.use_xvector_supplied)
    {
        return Err("--source-audio, --source-text, --target-text, --instruction, and --use-xvector require --tts --edit".into());
    }
    if options.edit && !options.tts {
        return Err("--edit requires --tts".into());
    }
    z_image_cli_options(options)?;
    if options.tts {
        if options.model.as_os_str().is_empty() {
            return Err("--tts requires --model".into());
        }
        if options
            .mmproj
            .as_deref()
            .is_none_or(|path| path.as_os_str().is_empty())
        {
            return Err("--tts requires --mmproj".into());
        }
        if options
            .out
            .as_deref()
            .is_none_or(|path| path.as_os_str().is_empty())
        {
            return Err("--tts requires --out".into());
        }
        if options.max_tokens == Some(0) {
            return Err("--tts requires --max-tokens greater than 0".into());
        }
        if options.steps == Some(0) {
            return Err("--tts requires --steps greater than 0".into());
        }
        let conflict = if options.audio.is_some() {
            Some("--audio")
        } else if options.image.is_some() {
            Some("--image")
        } else if options.video.is_some() {
            Some("--video")
        } else if options.embedding {
            Some("--embedding")
        } else if options.dump_logits {
            Some("--dump-logits")
        } else if options.bench {
            Some("--bench")
        } else if options.profile {
            Some("--profile")
        } else {
            None
        };
        if let Some(conflict) = conflict {
            return Err(format!("--tts cannot be used with {conflict}"));
        }
        if options.edit {
            if options.source_audio.is_none() {
                return Err("--tts --edit requires --source-audio".into());
            }
            if options
                .instruction
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
            {
                return Err("--tts --edit requires a non-empty --instruction".into());
            }
            let conflict = if options.prompt.is_some() {
                Some("--prompt")
            } else if options.ref_audio.is_some() {
                Some("--ref-audio")
            } else if options.ref_text.is_some() {
                Some("--ref-text")
            } else if options.language.is_some() {
                Some("--language")
            } else {
                None
            };
            if let Some(conflict) = conflict {
                return Err(format!("--tts --edit cannot be used with {conflict}"));
            }
        } else {
            if options
                .prompt
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
            {
                return Err("--tts requires a non-empty --prompt".into());
            }
            normalize_tts_language(options.language.as_deref())?;
            if options.ref_text.is_some() && options.ref_audio.is_none() {
                return Err("--ref-text requires --ref-audio".into());
            }
        }
        return Ok(());
    }
    if options.ref_audio.is_some() || options.ref_text.is_some() {
        return Err("--ref-audio/--ref-text require --tts".into());
    }
    let media_count = usize::from(options.image.is_some())
        + usize::from(options.video.is_some())
        + usize::from(options.audio.is_some());
    if options.embedding {
        if media_count > 1 {
            return Err("--embedding accepts only one of --image, --video, or --audio".into());
        }
        if media_count == 1 {
            if options
                .mmproj
                .as_deref()
                .is_none_or(|path| path.as_os_str().is_empty())
            {
                return Err("media embedding requires --mmproj".into());
            }
            return Ok(());
        }
    }
    if options.video.is_some() {
        if options
            .mmproj
            .as_deref()
            .is_none_or(|path| path.as_os_str().is_empty())
        {
            return Err("--video requires --mmproj".into());
        }
        return Ok(());
    }
    if options.audio.is_none() {
        return if options.language.is_some() {
            Err("--language requires --audio".into())
        } else {
            Ok(())
        };
    }
    if options
        .mmproj
        .as_deref()
        .is_none_or(|path| path.as_os_str().is_empty())
    {
        return Err("--audio requires --mmproj".into());
    }
    let conflict = if options.embedding {
        Some("--embedding")
    } else if options.dump_logits {
        Some("--dump-logits")
    } else if options.bench {
        Some("--bench")
    } else if options.profile {
        Some("--profile")
    } else {
        None
    };
    if let Some(conflict) = conflict {
        return Err(format!("--audio cannot be used with {conflict}"));
    }
    if options.max_tokens == Some(0) {
        return Err("--audio requires --max-tokens greater than 0".into());
    }
    Ok(())
}

pub fn resolve_cli_generation_options(options: &CliOptions) -> (usize, f32) {
    (
        options
            .max_tokens
            .unwrap_or(if options.audio.is_some() { 256 } else { 128 }),
        options
            .temperature
            .unwrap_or(if options.audio.is_some() { 0.0 } else { 0.6 }),
    )
}

pub fn transcription_options(
    options: &CliOptions,
) -> crate::models::qwen3::asr::model::TranscriptionOptions {
    let language = options
        .language
        .as_ref()
        .filter(|language| !language.eq_ignore_ascii_case("auto"))
        .cloned();
    crate::models::qwen3::asr::model::TranscriptionOptions {
        language,
        prompt: options.prompt.clone(),
        max_new_tokens: resolve_cli_generation_options(options).0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
    use crate::core::tokenizer::BPETokenizer;
    use crate::models::qwen3::asr::model::{normalize_language, TranscriptionOptions};
    use std::collections::HashMap;
    use std::path::Path;

    struct TestTensorSource {
        info: TensorInfo,
        bytes: Vec<u8>,
    }

    impl TensorSource for TestTensorSource {
        fn metadata(&self, _key: &str) -> Option<&MetaValue> {
            None
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            (name == self.info.name).then_some(&self.info)
        }

        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            (name == self.info.name).then_some(&self.bytes)
        }
    }

    #[test]
    fn embedding_output_accepts_only_summary_or_raw() {
        assert_eq!(
            parse_embedding_output(Some("summary")).unwrap(),
            EmbeddingOutput::Summary,
        );
        assert_eq!(
            parse_embedding_output(Some("raw")).unwrap(),
            EmbeddingOutput::Raw,
        );
        assert!(parse_embedding_output(Some("json")).is_err());
        assert!(parse_embedding_output(None).is_err());
    }

    #[test]
    fn default_threads_are_capped_but_explicit_value_wins() {
        assert_eq!(resolve_thread_count(0, 16), 8);
        assert_eq!(resolve_thread_count(0, 4), 4);
        assert_eq!(resolve_thread_count(0, 0), 1);
        assert_eq!(resolve_thread_count(12, 16), 12);
    }

    #[test]
    fn normal_generation_does_not_run_the_final_unused_forward() {
        assert_eq!(inference_step_budget(5, 32, false), 36);
        assert_eq!(inference_step_budget(5, 0, false), 5);
    }

    #[test]
    fn bench_budget_has_exact_decode_eval_count() {
        assert_eq!(inference_step_budget(5, 32, true), 37);
        assert_eq!(per_second(32, Duration::from_millis(250)), 128.0);
    }

    #[test]
    fn qwen3vl_rejects_legacy_decoder_modes() {
        for (result, expected_mode) in [
            (
                validate_qwen3vl_decoder_mode("qwen3vl", true, false, false, KvFormat::F16, false),
                "--dump-logits",
            ),
            (
                validate_qwen3vl_decoder_mode("qwen3vl", false, true, false, KvFormat::F16, false),
                "--bench",
            ),
            (
                validate_qwen3vl_decoder_mode("qwen3vl", false, false, true, KvFormat::F16, false),
                "--profile",
            ),
            (
                validate_qwen3vl_decoder_mode("qwen3vl", false, false, false, KvFormat::F32, false),
                "--kv-cache f32",
            ),
            (
                validate_qwen3vl_decoder_mode("qwen3vl", false, false, false, KvFormat::F16, true),
                "interactive mode",
            ),
        ] {
            assert!(result.unwrap_err().contains(expected_mode));
        }

        assert!(validate_qwen3vl_decoder_mode(
            "qwen3vl",
            false,
            false,
            false,
            KvFormat::F16,
            false
        )
        .is_ok());
        assert!(
            validate_qwen3vl_decoder_mode("qwen3", true, true, true, KvFormat::F32, true).is_ok()
        );
    }

    fn asr_cli_options() -> CliOptions {
        CliOptions {
            model: "missing.gguf".into(),
            mmproj: Some("missing-mmproj.gguf".into()),
            audio: Some("missing.wav".into()),
            ..CliOptions::default()
        }
    }

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn planner_cli_requires_complete_component_set() {
        let options = parse_cli_options(&args(&[
            "rmi",
            "--model",
            "vlm.gguf",
            "--mmproj",
            "mmproj.gguf",
            "--planner",
            "planner.gguf",
            "--scenes",
            "scenes.jsonl",
            "--image-root",
            "frames",
            "--mode",
            "direct_planning",
            "--num-samples",
            "1",
            "--num-steps",
            "10",
            "--seed",
            "42",
            "--output",
            "predictions.jsonl",
        ]))
        .unwrap();
        let planner = qwen_drive_cli_options(&options).unwrap().unwrap();
        assert_eq!(planner.model, PathBuf::from("vlm.gguf"));
        assert_eq!(planner.mmproj, PathBuf::from("mmproj.gguf"));
        assert_eq!(planner.output, PathBuf::from("predictions.jsonl"));
        assert_eq!(planner.samples, 1);
        assert_eq!(planner.steps, 10);
        assert_eq!(planner.seed, 42);
        assert!(z_image_cli_options(&options).unwrap().is_none());

        let incomplete = parse_cli_options(&args(&["rmi", "--planner", "p.gguf"])).unwrap();
        assert!(qwen_drive_cli_options(&incomplete).is_err());
    }

    #[test]
    fn video_input_is_parsed_and_media_inputs_are_exclusive() {
        let options = parse_cli_options(&args(&[
            "rmi",
            "--model",
            "model.gguf",
            "--video",
            "clip.mp4",
        ]))
        .unwrap();
        assert_eq!(options.video.as_deref(), Some(Path::new("clip.mp4")));
        assert!(options.audio.is_none());

        let mixed = parse_cli_options(&args(&[
            "rmi",
            "--model",
            "model.gguf",
            "--embedding",
            "--image",
            "still.png",
            "--video",
            "clip.mp4",
        ]))
        .unwrap();
        assert!(validate_cli_options(&mixed)
            .unwrap_err()
            .contains("only one of --image, --video, or --audio"));
    }

    #[test]
    fn z_image_cli_requires_all_components_prompt_and_out() {
        let complete = parse_cli_options(&args(&[
            "rmi",
            "--model",
            "dit.gguf",
            "--text-encoder",
            "text.gguf",
            "--vae",
            "vae.gguf",
            "--prompt",
            "fox",
            "--out",
            "fox.png",
            "--seed",
            "42",
        ]))
        .unwrap();
        assert_eq!(z_image_cli_options(&complete).unwrap().unwrap().seed, 42);
        for argv in [
            ["rmi", "--model", "dit.gguf", "--text-encoder", "text.gguf"].as_slice(),
            [
                "rmi", "--model", "dit.gguf", "--vae", "vae.gguf", "--prompt", "fox",
            ]
            .as_slice(),
        ] {
            assert!(
                z_image_cli_options(&parse_cli_options(&args(argv)).unwrap()).is_err(),
                "{argv:?}"
            );
        }
    }

    #[test]
    fn dreamx_cli_requires_model_mmproj_image_prompt_and_out() {
        let complete = parse_cli_options(&args(&[
            "rmi",
            "--dreamx",
            "--model",
            "dreamx.gguf",
            "--mmproj",
            "mmproj.gguf",
            "--image",
            "first.png",
            "--prompt",
            "scene",
            "--out",
            "scene.mp4",
        ]))
        .unwrap();
        let dreamx = dreamx_cli_options(&complete).unwrap().unwrap();
        assert_eq!(dreamx.model, PathBuf::from("dreamx.gguf"));
        assert_eq!(dreamx.mmproj, PathBuf::from("mmproj.gguf"));
        assert_eq!(dreamx.image, PathBuf::from("first.png"));
        assert_eq!(dreamx.prompt, "scene");
        assert_eq!(dreamx.out, PathBuf::from("scene.mp4"));
        assert_eq!(dreamx.options, DreamXOptions::default());

        for argv in [
            vec![
                "rmi",
                "--dreamx",
                "--mmproj",
                "mmproj.gguf",
                "--image",
                "first.png",
                "--prompt",
                "scene",
                "--out",
                "scene.mp4",
            ],
            vec![
                "rmi",
                "--dreamx",
                "--model",
                "dreamx.gguf",
                "--image",
                "first.png",
                "--prompt",
                "scene",
                "--out",
                "scene.mp4",
            ],
            vec![
                "rmi",
                "--dreamx",
                "--model",
                "dreamx.gguf",
                "--mmproj",
                "mmproj.gguf",
                "--prompt",
                "scene",
                "--out",
                "scene.mp4",
            ],
            vec![
                "rmi",
                "--dreamx",
                "--model",
                "dreamx.gguf",
                "--mmproj",
                "mmproj.gguf",
                "--image",
                "first.png",
                "--out",
                "scene.mp4",
            ],
            vec![
                "rmi",
                "--dreamx",
                "--model",
                "dreamx.gguf",
                "--mmproj",
                "mmproj.gguf",
                "--image",
                "first.png",
                "--prompt",
                "scene",
            ],
        ] {
            let options = parse_cli_options(&args(&argv)).unwrap();
            assert!(dreamx_cli_options(&options).is_err(), "{argv:?}");
        }
    }

    #[test]
    fn dreamx_cli_parses_pipeline_controls() {
        let options = parse_cli_options(&args(&[
            "rmi",
            "--dreamx",
            "--model",
            "dreamx.gguf",
            "--mmproj",
            "mmproj.gguf",
            "--image",
            "first.png",
            "--prompt",
            "scene",
            "--negative-prompt",
            "blur",
            "--out",
            "scene.mp4",
            "--duration",
            "0.2",
            "--fps",
            "5",
            "--steps",
            "1",
            "--seed",
            "-7",
            "--target-spatial-tokens",
            "4",
            "--no-refine",
            "--refine",
            "--refiner-kv-len",
            "3",
            "--latent-upsample",
            "causal2d",
            "--refiner-decoder",
            "lightvae",
            "--dry-run",
            "--overwrite",
            "--allow-memory-overcommit",
        ]))
        .unwrap();
        let dreamx = dreamx_cli_options(&options).unwrap().unwrap();
        assert_eq!(dreamx.negative_prompt.as_deref(), Some("blur"));
        assert_eq!(dreamx.options.duration_seconds, 0.2);
        assert_eq!(dreamx.options.fps, 5);
        assert_eq!(dreamx.options.steps, 1);
        assert_eq!(dreamx.options.seed, -7);
        assert_eq!(dreamx.options.target_spatial_tokens, 4);
        assert!(dreamx.options.refine);
        assert_eq!(dreamx.options.refiner.kv_len, 3);
        assert_eq!(
            dreamx.options.refiner.latent_upsample,
            LatentUpsampleKind::Causal2d
        );
        assert_eq!(dreamx.options.refiner.decoder, RefinerDecoderKind::LightVae);
        assert!(dreamx.dry_run);
        assert!(dreamx.overwrite);
        assert!(dreamx.allow_memory_overcommit);
    }

    #[test]
    fn dreamx_seed_is_not_claimed_by_z_image() {
        let options = parse_cli_options(&args(&["rmi", "--dreamx", "--seed", "-7"])).unwrap();
        assert!(z_image_cli_options(&options).unwrap().is_none());
    }

    #[test]
    fn dreamx_cli_rejects_malformed_controls() {
        for argv in [
            vec!["rmi", "--dreamx", "--duration", "nan"],
            vec!["rmi", "--dreamx", "--fps", "0"],
            vec!["rmi", "--dreamx", "--target-spatial-tokens", "0"],
            vec!["rmi", "--dreamx", "--refiner-kv-len", "0"],
            vec!["rmi", "--dreamx", "--latent-upsample", "nearest"],
            vec!["rmi", "--dreamx", "--refiner-decoder", "fast"],
            vec!["rmi", "--dreamx", "--negative-prompt"],
        ] {
            assert!(
                parse_cli_options(&args(&argv))
                    .and_then(|options| dreamx_cli_options(&options).map(|_| options))
                    .is_err(),
                "{argv:?}"
            );
        }
    }

    #[test]
    fn dreamx_cli_rejects_edit_only_options() {
        let base = [
            "rmi",
            "--dreamx",
            "--model",
            "dreamx.gguf",
            "--mmproj",
            "mmproj.gguf",
            "--image",
            "first.png",
            "--prompt",
            "scene",
            "--out",
            "scene.mp4",
        ];
        for (extra, expected) in [
            (vec!["--source-audio", "source.wav"], "--source-audio"),
            (vec!["--source-text", "source"], "--source-text"),
            (vec!["--target-text", "target"], "--target-text"),
            (vec!["--instruction", "replace"], "--instruction"),
            (vec!["--use-xvector", "auto"], "--use-xvector"),
        ] {
            let mut argv = base.to_vec();
            argv.extend(extra);
            let error =
                validate_cli_options(&parse_cli_options(&args(&argv)).unwrap()).unwrap_err();
            assert!(error.contains(expected), "{argv:?}: {error}");
        }
    }

    #[test]
    fn seed_requires_a_signed_i64_value() {
        assert!(parse_cli_options(&args(&["rmi", "--seed"])).is_err());
        assert!(parse_cli_options(&args(&["rmi", "--seed", "nan"])).is_err());
    }

    #[test]
    fn dots_edit_cli_parses_and_validates_before_model_loading() {
        let options = parse_cli_options(&args(&[
            "rmi",
            "--tts",
            "--edit",
            "--model",
            "edit.gguf",
            "--mmproj",
            "edit-mmproj.gguf",
            "--source-audio",
            "source.wav",
            "--instruction",
            "<del>旧</del><ins>新</ins>",
            "--use-xvector",
            "auto",
            "--max-tokens",
            "8",
            "--steps",
            "2",
            "--seed",
            "42",
            "--out",
            "edited.wav",
        ]))
        .unwrap();
        assert!(options.edit);
        assert_eq!(
            options.source_audio.as_deref(),
            Some(Path::new("source.wav"))
        );
        assert_eq!(options.use_xvector, XVectorMode::Auto);
        assert!(validate_cli_options(&options).is_ok());
    }

    #[test]
    fn dots_edit_cli_rejects_incomplete_or_cross_mode_inputs() {
        let parse = |values: &[&str]| parse_cli_options(&args(values)).unwrap();
        for invalid in [
            vec!["rmi", "--source-audio", "source.wav"],
            vec!["rmi", "--use-xvector", "auto"],
            vec![
                "rmi",
                "--tts",
                "--edit",
                "--mmproj",
                "m",
                "--out",
                "o",
                "--instruction",
                "x",
            ],
            vec![
                "rmi",
                "--tts",
                "--edit",
                "--mmproj",
                "m",
                "--out",
                "o",
                "--source-audio",
                "s.wav",
            ],
            vec![
                "rmi",
                "--tts",
                "--edit",
                "--mmproj",
                "m",
                "--out",
                "o",
                "--source-audio",
                "s.wav",
                "--instruction",
                "x",
                "--ref-audio",
                "r.wav",
            ],
        ] {
            assert!(
                validate_cli_options(&parse(&invalid)).is_err(),
                "{invalid:?}"
            );
        }
        assert!(parse_cli_options(&args(&["rmi", "--use-xvector", "maybe"])).is_err());
    }

    #[test]
    fn seed_is_valid_for_tts_but_still_rejected_for_unscoped_model_mode() {
        let tts = parse_cli_options(&args(&[
            "rmi", "--tts", "--model", "m", "--mmproj", "p", "--prompt", "hello", "--seed", "7",
            "--out", "o.wav",
        ]))
        .unwrap();
        assert!(validate_cli_options(&tts).is_ok());
        let plain = parse_cli_options(&args(&["rmi", "--seed", "7"])).unwrap();
        assert!(validate_cli_options(&plain)
            .unwrap_err()
            .contains("Z-Image"));
    }

    #[test]
    fn validate_cli_options_enforces_z_image_contract() {
        let parse = |values: &[&str]| parse_cli_options(&args(values)).unwrap();
        assert!(validate_cli_options(&parse(&["rmi", "--seed", "42"])).is_err());
        assert!(validate_cli_options(&parse(&["rmi", "--text-encoder", "text.gguf",])).is_err());
        assert!(validate_cli_options(&parse(&[
            "rmi",
            "--text-encoder",
            "text.gguf",
            "--vae",
            "vae.gguf",
            "--prompt",
            "fox",
            "--out",
            "fox.png",
        ]))
        .is_err());
        assert!(validate_cli_options(&parse(&[
            "rmi",
            "--model",
            "dit.gguf",
            "--text-encoder",
            "text.gguf",
            "--vae",
            "vae.gguf",
            "--prompt",
            "fox",
            "--out",
            "fox.png",
        ]))
        .is_ok());
    }

    #[test]
    fn z_image_rejects_other_modes_before_model_loading() {
        let base = [
            "rmi",
            "--model",
            "dit.gguf",
            "--text-encoder",
            "text.gguf",
            "--vae",
            "vae.gguf",
            "--prompt",
            "fox",
            "--out",
            "fox.png",
        ];
        for (extra, expected) in [
            (vec!["--tts"], "--tts"),
            (vec!["--audio", "speech.wav"], "--audio"),
            (vec!["--ref-audio", "voice.wav"], "--ref-audio"),
            (vec!["--image", "input.png"], "--image"),
            (vec!["--mmproj", "mmproj.gguf"], "--mmproj"),
            (vec!["--embedding"], "--embedding"),
            (vec!["--dump-logits"], "--dump-logits"),
            (vec!["--bench"], "--bench"),
            (vec!["--profile"], "--profile"),
            (vec!["--gpu"], "--gpu"),
            (vec!["--thinking"], "--thinking"),
            (vec!["--language", "en"], "--language"),
            (vec!["--max-tokens", "1"], "--max-tokens"),
            (vec!["--temp", "0"], "--temp"),
            (vec!["--embedding-output", "raw"], "--embedding-output"),
        ] {
            let mut argv = base.to_vec();
            argv.extend(extra);
            let error =
                validate_cli_options(&parse_cli_options(&args(&argv)).unwrap()).unwrap_err();
            assert!(error.contains(expected), "{argv:?}: {error}");
        }
    }

    #[test]
    fn z_image_cli_rejects_malformed_steps_and_resolution() {
        for flag in ["--steps", "--resolution"] {
            assert!(
                parse_cli_options(&args(&[
                    "rmi",
                    "--text-encoder",
                    "text.gguf",
                    "--vae",
                    "vae.gguf",
                    "--prompt",
                    "fox",
                    "--out",
                    "fox.png",
                    flag,
                    "nope",
                ]))
                .is_err(),
                "{flag}"
            );
            assert!(parse_cli_options(&args(&["rmi", flag])).is_err(), "{flag}");
        }
    }

    #[test]
    fn z_image_component_flags_require_values() {
        for flag in ["--text-encoder", "--vae"] {
            assert!(parse_cli_options(&args(&["rmi", flag])).is_err(), "{flag}");
            assert!(
                parse_cli_options(&args(&["rmi", flag, ""])).is_err(),
                "{flag}"
            );
        }
    }

    #[test]
    fn tts_cli_requires_complete_waveform_inputs_before_model_load() {
        let parse = |args: &[&str]| {
            let args: Vec<String> = args.iter().map(ToString::to_string).collect();
            parse_cli_options(&args).unwrap()
        };

        let valid = parse(&[
            "rmi",
            "--tts",
            "--model",
            "missing.gguf",
            "--mmproj",
            "missing-mmproj.gguf",
            "--prompt",
            "hello",
            "--language",
            "cn",
            "--ref-audio",
            "speaker.wav",
            "--out",
            "output.wav",
        ]);
        assert_eq!(valid.ref_audio.as_deref(), Some(Path::new("speaker.wav")));
        assert!(validate_cli_options(&valid).is_ok());

        for args in [
            vec!["rmi", "--ref-audio", "speaker.wav"],
            vec!["rmi", "--tts", "--prompt", "hello", "--out", "output.wav"],
            vec!["rmi", "--tts", "--prompt", "hello", "--mmproj", "mm.gguf"],
            vec![
                "rmi", "--tts", "--prompt", "", "--mmproj", "mm.gguf", "--out", "o.wav",
            ],
        ] {
            let options = parse(&args);
            assert!(validate_cli_options(&options).is_err(), "{args:?}");
        }
    }

    #[test]
    fn tts_languages_match_cli_and_oracle_aliases() {
        for (input, expected) in [
            (None, "english"),
            (Some("cn"), "chinese"),
            (Some("zh"), "chinese"),
            (Some("chinese"), "chinese"),
            (Some("en"), "english"),
            (Some("english"), "english"),
            (Some("ge"), "german"),
            (Some("de"), "german"),
            (Some("german"), "german"),
            (Some("it"), "italian"),
            (Some("italian"), "italian"),
            (Some("po"), "portuguese"),
            (Some("pt"), "portuguese"),
            (Some("portuguese"), "portuguese"),
            (Some("sp"), "spanish"),
            (Some("es"), "spanish"),
            (Some("spanish"), "spanish"),
            (Some("ja"), "japanese"),
            (Some("japanese"), "japanese"),
            (Some("ko"), "korean"),
            (Some("korean"), "korean"),
            (Some("fr"), "french"),
            (Some("french"), "french"),
            (Some("ru"), "russian"),
            (Some("russian"), "russian"),
        ] {
            assert_eq!(normalize_tts_language(input).unwrap(), expected);
        }
        assert!(normalize_tts_language(Some("auto"))
            .unwrap_err()
            .contains("TTS language"));
    }

    #[test]
    fn asr_cli_rejects_conflicting_modes_before_model_load() {
        let mut options = asr_cli_options();
        options.dump_logits = true;
        assert!(validate_cli_options(&options)
            .unwrap_err()
            .contains("--dump-logits"));

        let mut options = asr_cli_options();
        options.bench = true;
        assert!(validate_cli_options(&options)
            .unwrap_err()
            .contains("--bench"));

        let mut options = asr_cli_options();
        options.profile = true;
        assert!(validate_cli_options(&options)
            .unwrap_err()
            .contains("--profile"));

        let mut options = asr_cli_options();
        options.temperature = Some(0.1);
        assert!(validate_cli_options(&options).is_ok());

        let mut options = asr_cli_options();
        options.max_tokens = Some(0);
        assert!(validate_cli_options(&options)
            .unwrap_err()
            .contains("--max-tokens"));

        let mut options = asr_cli_options();
        options.audio = None;
        options.language = Some("English".into());
        assert!(validate_cli_options(&options)
            .unwrap_err()
            .contains("--language"));

        let mut options = asr_cli_options();
        options.prompt = Some("domain context".into());
        assert!(validate_cli_options(&options).is_ok());

        let args = ["rmi".to_string(), "--audio".to_string()];
        assert!(parse_cli_options(&args).unwrap_err().contains("--audio"));
    }

    #[test]
    fn gemma4_media_requires_mmproj() {
        let parse = |values: &[&str]| parse_cli_options(&args(values)).unwrap();
        assert!(validate_cli_options(&parse(&[
            "rmi",
            "--model",
            "gemma.gguf",
            "--audio",
            "a.wav",
            "--prompt",
            "x",
        ]))
        .is_err());
        assert!(validate_cli_options(&parse(&[
            "rmi",
            "--model",
            "gemma.gguf",
            "--mmproj",
            "mm.gguf",
            "--image",
            "a.png",
            "--audio",
            "a.wav",
            "--prompt",
            "x",
        ]))
        .is_ok());
    }

    #[test]
    fn asr_cli_rejects_empty_and_flag_shaped_values() {
        for args in [
            vec!["rmi", "--audio", ""],
            vec!["rmi", "--audio", "--image", "missing.png"],
            vec!["rmi", "--audio", "--language", "English"],
        ] {
            let args: Vec<String> = args.into_iter().map(str::to_string).collect();
            assert!(parse_cli_options(&args).unwrap_err().contains("--audio"));
        }

        let args: Vec<String> = ["rmi", "--audio", "missing.wav", "--language", "--prompt"]
            .into_iter()
            .map(str::to_string)
            .collect();
        assert!(parse_cli_options(&args).unwrap_err().contains("--language"));

        let args: Vec<String> = ["rmi", "-recording.wav", "--language", "English"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let options = parse_cli_options(&args).unwrap();
        assert_eq!(options.audio.as_deref(), Some(Path::new("-recording.wav")));
        assert_eq!(options.language.as_deref(), Some("English"));

        let args: Vec<String> = [
            "rmi",
            "missing.wav",
            "--mmproj",
            "missing-mmproj.gguf",
            "--language",
            "",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        let options = parse_cli_options(&args).unwrap();
        assert!(validate_cli_options(&options).is_ok());
        assert!(
            normalize_language(transcription_options(&options).language.as_deref())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn asr_cli_defaults_are_greedy_and_256_tokens() {
        let mut options = asr_cli_options();
        options.language = Some("auto".into());
        options.prompt = Some("domain context".into());

        let (max_tokens, temperature) = resolve_cli_generation_options(&options);
        assert_eq!(max_tokens, 256);
        assert_eq!(temperature, 0.0);
        let transcription = transcription_options(&options);
        assert_eq!(transcription.language, None);
        assert_eq!(transcription.prompt.as_deref(), Some("domain context"));
        assert_eq!(transcription.max_new_tokens, 256);
        assert!(normalize_language(Some("auto")).is_err());

        let args = [
            "rmi".to_string(),
            "--audio".to_string(),
            "missing.wav".to_string(),
            "--n-gen".to_string(),
            "7".to_string(),
        ];
        assert_eq!(parse_cli_options(&args).unwrap().max_tokens, Some(7));

        let args = [
            "rmi".to_string(),
            "--unknown".to_string(),
            "--prompt".to_string(),
            "hello".to_string(),
        ];
        let text = parse_cli_options(&args).unwrap();
        assert_eq!(text.prompt.as_deref(), Some("hello"));
        assert_eq!(resolve_cli_generation_options(&text), (128, 0.6));
    }

    #[test]
    fn omni_embedding_accepts_exactly_one_media_kind() {
        let image = parse_cli_options(&args(&[
            "rmi",
            "--model",
            "text.gguf",
            "--embedding",
            "--mmproj",
            "vision.gguf",
            "--image",
            "image.png",
            "--prompt",
            "Document: caption",
        ]))
        .unwrap();
        assert!(validate_cli_options(&image).is_ok());

        let video = parse_cli_options(&args(&[
            "rmi",
            "--model",
            "text.gguf",
            "--embedding",
            "--mmproj",
            "vision.gguf",
            "--video",
            "video.mp4",
            "--prompt",
            "Document: clip",
        ]))
        .unwrap();
        assert_eq!(video.video.as_deref(), Some(Path::new("video.mp4")));
        assert!(validate_cli_options(&video).is_ok());

        let audio = parse_cli_options(&args(&[
            "rmi",
            "--model",
            "text.gguf",
            "--embedding",
            "--mmproj",
            "audio.gguf",
            "--audio",
            "audio.flac",
            "--prompt",
            "Document: sound",
        ]))
        .unwrap();
        assert!(validate_cli_options(&audio).is_ok());

        for argv in [
            [
                "rmi",
                "--embedding",
                "--mmproj",
                "m.gguf",
                "--image",
                "i.png",
                "--video",
                "v.mp4",
            ]
            .as_slice(),
            [
                "rmi",
                "--embedding",
                "--mmproj",
                "m.gguf",
                "--video",
                "v.mp4",
                "--audio",
                "a.wav",
            ]
            .as_slice(),
            [
                "rmi",
                "--embedding",
                "--mmproj",
                "m.gguf",
                "--image",
                "i.png",
                "--audio",
                "a.wav",
            ]
            .as_slice(),
        ] {
            let error = validate_cli_options(&parse_cli_options(&args(argv)).unwrap()).unwrap_err();
            assert!(
                error.contains("one of --image, --video, or --audio"),
                "{argv:?}: {error}"
            );
        }
    }

    #[test]
    fn generative_video_requires_mmproj_and_media_is_exclusive() {
        let video = parse_cli_options(&args(&[
            "rmi",
            "--model",
            "text.gguf",
            "--mmproj",
            "vision.gguf",
            "--video",
            "video.mp4",
        ]))
        .unwrap();
        assert!(validate_cli_options(&video).is_ok());

        let missing_mmproj = parse_cli_options(&args(&[
            "rmi",
            "--model",
            "text.gguf",
            "--video",
            "video.mp4",
        ]))
        .unwrap();
        assert!(validate_cli_options(&missing_mmproj)
            .unwrap_err()
            .contains("--mmproj"));
    }

    #[test]
    fn text_embedding_does_not_bypass_language_validation() {
        let options = parse_cli_options(&args(&[
            "rmi",
            "--embedding",
            "--prompt",
            "query",
            "--language",
            "English",
        ]))
        .unwrap();

        let error = validate_cli_options(&options).unwrap_err();
        assert_eq!(error, "--language requires --audio");
    }

    #[test]
    fn legacy_cli_parser_and_dispatch_semantics_are_preserved() {
        type Check = fn(&CliOptions) -> bool;
        let parse = |args: &[&str]| {
            let args: Vec<String> = args.iter().map(ToString::to_string).collect();
            parse_cli_options(&args).unwrap()
        };
        let cases: &[(&[&str], &str, Check)] = &[
            (&["rmi", "--embedding", "--prompt", "x"], "embedding", |o| {
                o.embedding && o.prompt.as_deref() == Some("x")
            }),
            (&["rmi", "--image", "image.png"], "image", |o| {
                o.image.as_deref() == Some(Path::new("image.png"))
            }),
            (&["rmi", "--mmproj", "projector.gguf"], "mmproj", |o| {
                o.mmproj.as_deref() == Some(Path::new("projector.gguf"))
            }),
            (&["rmi", "--model", "model.gguf"], "interactive", |o| {
                o.prompt.is_none() && o.image.is_none() && o.mmproj.is_none()
            }),
            (
                &["rmi", "positional", "--prompt", "x"],
                "unknown/positional",
                |o| o.prompt.as_deref() == Some("x"),
            ),
            (&["rmi"], "text defaults", |o| {
                resolve_cli_generation_options(o) == (128, 0.6)
            }),
            (&["rmi", "--max-tokens", "bad"], "malformed max", |o| {
                o.max_tokens == Some(128)
            }),
            (&["rmi", "--n-gen", "bad"], "malformed n-gen", |o| {
                o.max_tokens == Some(128)
            }),
            (&["rmi", "--temp", "bad"], "malformed temp", |o| {
                o.temperature == Some(0.6)
            }),
            (&["rmi", "--threads", "bad"], "malformed threads", |o| {
                o.threads == 0
            }),
            (&["rmi", "--kv-cache", "f32"], "F32 KV", |o| {
                o.kv_format == KvFormat::F32
            }),
            (&["rmi", "--kv-cache", "bad"], "fallback F16 KV", |o| {
                o.kv_format == KvFormat::F16
            }),
            (
                &[
                    "rmi",
                    "--model",
                    "",
                    "--prompt",
                    "",
                    "--max-tokens",
                    "",
                    "--temp",
                    "",
                    "--threads",
                    "",
                    "--kv-cache",
                    "",
                    "--mmproj",
                    "",
                    "--image",
                    "",
                ],
                "empty legacy values",
                |o| {
                    o.model.as_os_str().is_empty()
                        && o.prompt.as_deref() == Some("")
                        && o.max_tokens == Some(128)
                        && o.temperature == Some(0.6)
                        && o.threads == 0
                        && o.kv_format == KvFormat::F16
                        && o.mmproj.as_deref() == Some(Path::new(""))
                        && o.image.as_deref() == Some(Path::new(""))
                },
            ),
        ];
        for (args, name, check) in cases {
            assert!(check(&parse(args)), "{name}");
        }

        let absent: &[(&str, Check)] = &[
            ("--model", |o| o.model.as_os_str().is_empty()),
            ("--prompt", |o| o.prompt.is_none()),
            ("--max-tokens", |o| o.max_tokens.is_none()),
            ("--n-gen", |o| o.max_tokens.is_none()),
            ("--temp", |o| o.temperature.is_none()),
            ("--threads", |o| o.threads == 0),
            ("--kv-cache", |o| o.kv_format == KvFormat::F32),
            ("--mmproj", |o| o.mmproj.is_none()),
            ("--image", |o| o.image.is_none()),
        ];
        for (flag, check) in absent {
            assert!(check(&parse(&["rmi", flag])), "absent {flag}");
        }
        assert!(parse_cli_options(&["rmi".into(), "--embedding-output".into()]).is_err());

        for (value, expected) in [
            ("0", 0.0),
            ("-1", -1.0),
            ("NaN", f32::NAN),
            ("inf", f32::INFINITY),
            ("-inf", f32::NEG_INFINITY),
        ] {
            let options = parse(&["rmi", "--temp", value]);
            let actual = options.temperature.unwrap();
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual, expected);
            }
            assert!(validate_cli_options(&options).is_ok());
        }
    }
}
