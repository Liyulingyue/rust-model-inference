use std::path::PathBuf;
use std::time::Duration;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum KvFormat {
    #[default]
    F16,
    F32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EmbeddingOutput {
    #[default]
    Summary,
    Raw,
}

#[derive(Debug, Default)]
pub struct CliOptions {
    pub model: PathBuf,
    pub mmproj: Option<PathBuf>,
    pub audio: Option<PathBuf>,
    pub ref_audio: Option<PathBuf>,
    pub image: Option<PathBuf>,
    pub vae: Option<PathBuf>,
    pub text_encoder: Option<PathBuf>,
    pub prompt: Option<String>,
    pub language: Option<String>,
    pub max_tokens: Option<usize>,
    pub steps: Option<usize>,
    pub resolution: Option<usize>,
    pub seed: Option<i64>,
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
    pub out: Option<PathBuf>,
}

#[derive(Debug)]
pub struct ZImageCliOptions {
    pub steps: usize,
    pub resolution: usize,
    pub seed: i64,
    pub out: PathBuf,
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
            "--language" => {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.starts_with("--"))
                    .ok_or("Missing value for --language")?;
                options.language = Some(value.clone());
                i += 1;
            }
            "--tts" => options.tts = true,
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

pub fn z_image_cli_options(options: &CliOptions) -> Result<Option<ZImageCliOptions>, String> {
    if options.text_encoder.is_none() && options.vae.is_none() {
        return if options.seed.is_some() {
            Err("--seed requires Z-Image components".into())
        } else {
            Ok(None)
        };
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
    z_image_cli_options(options)?;
    if options.tts {
        if options.prompt.as_deref().is_none_or(|value| value.trim().is_empty()) {
            return Err("--tts requires a non-empty --prompt".into());
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
        let conflict = if options.audio.is_some() {
            Some("--audio")
        } else if options.image.is_some() {
            Some("--image")
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
        normalize_tts_language(options.language.as_deref())?;
        return Ok(());
    }
    if options.ref_audio.is_some() {
        return Err("--ref-audio requires --tts".into());
    }
    if options.audio.is_none() {
        return if options.language.is_some() {
            Err("--language requires --audio".into())
        } else {
            Ok(())
        };
    }
    let conflict = if options.image.is_some() {
        Some("--image")
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
        return Err(format!("--audio cannot be used with {conflict}"));
    }
    if options.temperature.is_some_and(|temperature| temperature != 0.0) {
        return Err("--audio requires greedy decoding; --temp must be 0".into());
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

pub fn transcription_options(options: &CliOptions) -> crate::models::asr::TranscriptionOptions {
    let language = options
        .language
        .as_ref()
        .filter(|language| !language.eq_ignore_ascii_case("auto"))
        .cloned();
    crate::models::asr::TranscriptionOptions {
        language,
        prompt: options.prompt.clone(),
        max_new_tokens: resolve_cli_generation_options(options).0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::asr::{normalize_language, TranscriptionOptions};
    use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
    use crate::core::tokenizer::BPETokenizer;
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
            validate_qwen3vl_decoder_mode(
                "qwen3",
                true,
                true,
                true,
                KvFormat::F32,
                true
            )
            .is_ok()
        );
    }

    fn asr_cli_options() -> CliOptions {
        CliOptions {
            model: "missing.gguf".into(),
            audio: Some("missing.wav".into()),
            ..CliOptions::default()
        }
    }

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(ToString::to_string).collect()
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
            ["rmi", "--model", "dit.gguf", "--vae", "vae.gguf", "--prompt", "fox"].as_slice(),
        ] {
            assert!(
                z_image_cli_options(&parse_cli_options(&args(argv)).unwrap()).is_err(),
                "{argv:?}"
            );
        }
    }

    #[test]
    fn seed_requires_a_signed_i64_value() {
        assert!(parse_cli_options(&args(&["rmi", "--seed"])).is_err());
        assert!(parse_cli_options(&args(&["rmi", "--seed", "nan"])).is_err());
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
        .is_ok());
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
                "rmi",
                "--tts",
                "--prompt",
                "",
                "--mmproj",
                "mm.gguf",
                "--out",
                "o.wav",
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
        options.image = Some("missing.png".into());
        assert!(validate_cli_options(&options).unwrap_err().contains("--image"));

        let mut options = asr_cli_options();
        options.embedding = true;
        assert!(validate_cli_options(&options).unwrap_err().contains("--embedding"));

        let mut options = asr_cli_options();
        options.dump_logits = true;
        assert!(validate_cli_options(&options).unwrap_err().contains("--dump-logits"));

        let mut options = asr_cli_options();
        options.bench = true;
        assert!(validate_cli_options(&options).unwrap_err().contains("--bench"));

        let mut options = asr_cli_options();
        options.profile = true;
        assert!(validate_cli_options(&options).unwrap_err().contains("--profile"));

        let mut options = asr_cli_options();
        options.temperature = Some(0.1);
        assert!(validate_cli_options(&options).unwrap_err().contains("--temp"));

        let mut options = asr_cli_options();
        options.max_tokens = Some(0);
        assert!(validate_cli_options(&options)
            .unwrap_err()
            .contains("--max-tokens"));

        let mut options = asr_cli_options();
        options.audio = None;
        options.language = Some("English".into());
        assert!(validate_cli_options(&options).unwrap_err().contains("--language"));

        let mut options = asr_cli_options();
        options.prompt = Some("domain context".into());
        assert!(validate_cli_options(&options).is_ok());

        let args = ["rmi".to_string(), "--audio".to_string()];
        assert!(parse_cli_options(&args).unwrap_err().contains("--audio"));
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
        assert!(parse_cli_options(&args)
            .unwrap_err()
            .contains("--language"));

        let args: Vec<String> = ["rmi", "-recording.wav", "--language", "English"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let options = parse_cli_options(&args).unwrap();
        assert_eq!(options.audio.as_deref(), Some(Path::new("-recording.wav")));
        assert_eq!(options.language.as_deref(), Some("English"));

        let args: Vec<String> = ["rmi", "missing.wav", "--language", ""]
            .into_iter()
            .map(str::to_string)
            .collect();
        let options = parse_cli_options(&args).unwrap();
        assert!(validate_cli_options(&options).is_ok());
        assert!(normalize_language(
            transcription_options(&options).language.as_deref()
        )
        .unwrap()
        .is_none());
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
    fn legacy_cli_parser_and_dispatch_semantics_are_preserved() {
        type Check = fn(&CliOptions) -> bool;
        let parse = |args: &[&str]| {
            let args: Vec<String> = args.iter().map(ToString::to_string).collect();
            parse_cli_options(&args).unwrap()
        };
        let cases: &[(&[&str], &str, Check)] = &[
            (&["rmi", "--embedding", "--prompt", "x"], "embedding", |o| o.embedding && o.prompt.as_deref() == Some("x")),
            (&["rmi", "--image", "image.png"], "image", |o| o.image.as_deref() == Some(Path::new("image.png"))),
            (&["rmi", "--mmproj", "projector.gguf"], "mmproj", |o| o.mmproj.as_deref() == Some(Path::new("projector.gguf"))),
            (&["rmi", "--model", "model.gguf"], "interactive", |o| o.prompt.is_none() && o.image.is_none() && o.mmproj.is_none()),
            (&["rmi", "positional", "--prompt", "x"], "unknown/positional", |o| o.prompt.as_deref() == Some("x")),
            (&["rmi"], "text defaults", |o| resolve_cli_generation_options(o) == (128, 0.6)),
            (&["rmi", "--max-tokens", "bad"], "malformed max", |o| o.max_tokens == Some(128)),
            (&["rmi", "--n-gen", "bad"], "malformed n-gen", |o| o.max_tokens == Some(128)),
            (&["rmi", "--temp", "bad"], "malformed temp", |o| o.temperature == Some(0.6)),
            (&["rmi", "--threads", "bad"], "malformed threads", |o| o.threads == 0),
            (&["rmi", "--kv-cache", "f32"], "F32 KV", |o| o.kv_format == KvFormat::F32),
            (&["rmi", "--kv-cache", "bad"], "fallback F16 KV", |o| o.kv_format == KvFormat::F16),
            (&["rmi", "--model", "", "--prompt", "", "--max-tokens", "", "--temp", "", "--threads", "", "--kv-cache", "", "--mmproj", "", "--image", ""], "empty legacy values", |o| o.model.as_os_str().is_empty() && o.prompt.as_deref() == Some("") && o.max_tokens == Some(128) && o.temperature == Some(0.6) && o.threads == 0 && o.kv_format == KvFormat::F16 && o.mmproj.as_deref() == Some(Path::new("")) && o.image.as_deref() == Some(Path::new(""))),
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
            ("--kv-cache", |o| o.kv_format == KvFormat::F16),
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
