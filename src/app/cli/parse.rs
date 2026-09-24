use std::path::PathBuf;
use std::str::FromStr;

use super::types::{
    CliOptions, JevBlockInput, JevQuestion, KvFormat, LatentUpsampleKind, RefinerDecoderKind,
    XVectorMode, parse_embedding_output,
};

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
            "--max-context" => {
                if i + 1 < args.len() {
                    options.max_context = Some(args[i + 1].parse().unwrap_or(8192));
                    i += 1;
                }
            }
            "--repetition-penalty" => {
                if i + 1 < args.len() {
                    let v: f32 = args[i + 1].parse().unwrap_or(1.0);
                    if v <= 0.0 {
                        return Err(format!(
                            "--repetition-penalty must be > 0 (1.0 = disabled, > 1.0 = suppress repeats)"
                        ));
                    }
                    options.repetition_penalty = Some(v);
                    i += 1;
                }
            }
            "--prefill-batch-size" => {
                let value = args
                    .get(i + 1)
                    .ok_or("Missing value for --prefill-batch-size")?;
                options.prefill_batch_size = Some(
                    value
                        .parse::<usize>()
                        .map_err(|error| format!("Invalid --prefill-batch-size value: {error}"))?,
                );
                i += 1;
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
            "--temperature" => {
                let value = args.get(i + 1).ok_or("Missing value for --temperature")?;
                options.temperature = Some(
                    value
                        .parse::<f32>()
                        .map_err(|error| format!("Invalid --temperature value: {error}"))?,
                );
                i += 1;
            }
            "--cfg-scale" => {
                let value = args.get(i + 1).ok_or("Missing value for --cfg-scale")?;
                options.cfg_scale = Some(
                    value
                        .parse::<f32>()
                        .map_err(|error| format!("Invalid --cfg-scale value: {error}"))?,
                );
                i += 1;
            }
            "--top-k" => {
                let value = args.get(i + 1).ok_or("Missing value for --top-k")?;
                options.top_k = Some(
                    value
                        .parse::<usize>()
                        .map_err(|error| format!("Invalid --top-k value: {error}"))?,
                );
                i += 1;
            }
            "--top-p" => {
                let value = args.get(i + 1).ok_or("Missing value for --top-p")?;
                options.top_p = Some(
                    value
                        .parse::<f32>()
                        .map_err(|error| format!("Invalid --top-p value: {error}"))?,
                );
                i += 1;
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
            "--no-thinking" => options.thinking = false,
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
            "--tts-model" => {
                if i + 1 < args.len() {
                    options.tts_model = Some(args[i + 1].as_str().into());
                    i += 1;
                }
            }
            "--tts-mmproj" => {
                if i + 1 < args.len() {
                    options.tts_mmproj = Some(args[i + 1].as_str().into());
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
            "--chunk" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.is_empty() && !value.starts_with("--"))
                    .ok_or("Missing value for --chunk")?;
                options.chunk_seconds = Some(value.parse().map_err(|_| "Invalid --chunk value")?);
                i += 1;
            }
            "--srt" => options.srt = true,
            "--vad" => {
                options.vad = Some(required_path_value(args, &mut i, "--vad")?);
            }
            "--vad-maxseg" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.is_empty() && !value.starts_with("--"))
                    .ok_or("Missing value for --vad-maxseg")?;
                options.vad_maxseg = value.parse().map_err(|_| "Invalid --vad-maxseg value")?;
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
            "--jev" => options.jev = true,
            "--jev-multi" => options.jev_multi = true,
            "--jev-block" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.starts_with("--"))
                    .ok_or("Missing value for --jev-block")?;
                if options.jev_questions.is_empty() {
                    return Err("--jev-block must follow a --jev-question".into());
                }
                options.jev_blocks.push(JevBlockInput {
                    label: value.clone(),
                    options: Vec::new(),
                });
                i += 1;
            }
            "--jev-context" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.starts_with("--"))
                    .ok_or("Missing value for --jev-context")?;
                options.jev_context = Some(value.clone());
                i += 1;
            }
            "--jev-question" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.starts_with("--"))
                    .ok_or("Missing value for --jev-question")?;
                options.jev_questions.push(JevQuestion {
                    text: value.clone(),
                    options: Vec::new(),
                });
                i += 1;
            }
            "--jev-option" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.starts_with("--"))
                    .ok_or("Missing value for --jev-option")?;
                if !options.jev_blocks.is_empty() {
                    options
                        .jev_blocks
                        .last_mut()
                        .unwrap()
                        .options
                        .push(value.clone());
                } else if options.jev_questions.is_empty() {
                    return Err(
                        "--jev-option must follow a --jev-question (or be the first argument after --jev)"
                            .into(),
                    );
                } else {
                    options
                        .jev_questions
                        .last_mut()
                        .unwrap()
                        .options
                        .push(value.clone());
                }
                i += 1;
            }
            "--jev-positive" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.starts_with("--"))
                    .ok_or("Missing value for --jev-positive")?;
                options.jev_positive = Some(value.clone());
                i += 1;
            }
            "--jev-output" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.starts_with("--"))
                    .ok_or("Missing value for --jev-output")?;
                match value.as_str() {
                    "text" => options.jev_output_json = false,
                    "json" => options.jev_output_json = true,
                    other => {
                        return Err(format!(
                            "--jev-output must be 'text' or 'json', got {other:?}"
                        ));
                    }
                }
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
