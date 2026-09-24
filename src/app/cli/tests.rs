use super::*;
use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
use crate::core::tokenizer::BPETokenizer;
use crate::models::qwen3::asr::model::{normalize_language, TranscriptionOptions};
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

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
fn cli_parses_prefill_batch_size_strictly() {
    assert_eq!(
        CliOptions::default()
            .effective_prefill_batch_size()
            .unwrap(),
        crate::core::prefill::DEFAULT_PREFILL_BATCH_SIZE
    );
    let parsed = parse_cli_options(&args(&["rmi", "--prefill-batch-size", "32"])).unwrap();
    assert_eq!(parsed.effective_prefill_batch_size().unwrap(), 32);
    assert!(parse_cli_options(&args(&["rmi", "--prefill-batch-size", "x"])).is_err());
    let zero = parse_cli_options(&args(&["rmi", "--prefill-batch-size", "0"])).unwrap();
    assert!(zero.effective_prefill_batch_size().is_err());
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
fn perception_cli_requires_frame_manifest_and_perception_head() {
    let argv = args(&[
        "rmi",
        "--model",
        "vlm.gguf",
        "--mmproj",
        "mmproj.gguf",
        "--perception",
        "perception.gguf",
        "--frames",
        "frame-dir",
        "--output",
        "result.json",
    ]);
    let options = parse_cli_options(&argv).unwrap();
    let drive = qwen_drive_cli_options(&options).unwrap().unwrap();
    assert!(matches!(drive.head, QwenDriveHead::Perception(_)));
    assert_eq!(drive.frames.as_deref(), Some(Path::new("frame-dir")));

    let missing = parse_cli_options(&args(&[
        "rmi",
        "--model",
        "vlm.gguf",
        "--mmproj",
        "mmproj.gguf",
        "--perception",
        "perception.gguf",
        "--output",
        "result.json",
    ]))
    .unwrap();
    assert!(qwen_drive_cli_options(&missing)
        .unwrap_err()
        .contains("--frames"));
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
fn breeze_instruction_is_valid_without_edit_and_temperature_alias_parses() {
    let options = parse_cli_options(&args(&[
        "rmi",
        "--tts",
        "--model",
        "breeze.gguf",
        "--mmproj",
        "codec.gguf",
        "--prompt",
        "你好",
        "--out",
        "output.wav",
        "--instruction",
        "温柔的女声",
        "--temperature",
        "0",
    ]))
    .unwrap();
    assert!(validate_cli_options(&options).is_ok());
    assert_eq!(options.temperature, Some(0.0));
    assert!(options.audio.is_none());
}

#[test]
fn breeze_cfg_scale_parses_and_rejects_invalid_values_or_other_modes() {
    let base = [
        "rmi",
        "--tts",
        "--model",
        "breeze.gguf",
        "--mmproj",
        "codec.gguf",
        "--prompt",
        "你好",
        "--out",
        "output.wav",
    ];
    for value in ["0", "-1", "NaN", "inf"] {
        let mut values = base.to_vec();
        values.extend(["--cfg-scale", value]);
        let options = parse_cli_options(&args(&values)).unwrap();
        assert!(validate_cli_options(&options)
            .unwrap_err()
            .contains("--cfg-scale"));
    }
    let mut values = base.to_vec();
    values.extend(["--cfg-scale", "3"]);
    let options = parse_cli_options(&args(&values)).unwrap();
    assert_eq!(options.cfg_scale, Some(3.0));
    assert!(validate_cli_options(&options).is_ok());
    let options = parse_cli_options(&args(&["rmi", "--cfg-scale", "3"])).unwrap();
    assert!(validate_cli_options(&options).is_err());
    assert!(parse_cli_options(&args(&["rmi", "--cfg-scale"])).is_err());
    let mut values = base.to_vec();
    values.extend(["--top-k", "50", "--top-p", "0.9"]);
    let options = parse_cli_options(&args(&values)).unwrap();
    assert_eq!(options.top_k, Some(50));
    assert_eq!(options.top_p, Some(0.9));
    assert!(validate_cli_options(&options).is_ok());
    for (flag, value) in [
        ("--top-k", "-1"),
        ("--top-p", "bad"),
        ("--temperature", "bad"),
    ] {
        assert!(parse_cli_options(&args(&["rmi", flag, value])).is_err());
    }
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
