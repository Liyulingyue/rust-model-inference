use std::process::Command;

#[cfg(feature = "parity-trace")]
use parity_support::*;

fn oracle_runner() -> Command {
    let mut command = Command::new("python3");
    command.arg(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tools/dots/run_dots_tts_oracle.py"
    ));
    command
}

#[test]
fn oracle_base_schedule_includes_encoded_prompt_and_target_budget() {
    let script = concat!(
        "import importlib.util; ",
        "spec=importlib.util.spec_from_file_location('runner', r'",
        env!("CARGO_MANIFEST_DIR"),
        "/tools/dots/run_dots_tts_oracle.py'); ",
        "runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner); ",
        "print(runner.base_max_generate_length(19_201, 7_680))"
    );
    let output = Command::new("python3")
        .args(["-c", script])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap().trim(), "5");
}

#[test]
fn oracle_runner_rejects_mode_specific_arguments_before_importing_runtime() {
    let directory =
        std::env::temp_dir().join(format!("rmi-dots-tts-oracle-runner-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    for name in ["ref.wav", "latent.f32", "dit.f32"] {
        std::fs::write(directory.join(name), [0u8; 4]).unwrap();
    }
    let output = oracle_runner()
        .args([
            "--checkout",
            directory.to_str().unwrap(),
            "--model-dir",
            directory.to_str().unwrap(),
            "--mode",
            "base",
            "--text",
            "hello",
            "--ref-audio",
            directory.join("ref.wav").to_str().unwrap(),
            "--ref-text",
            "reference",
            "--instruction",
            "must be rejected",
            "--latent-noise",
            directory.join("latent.f32").to_str().unwrap(),
            "--dit-noise",
            directory.join("dit.f32").to_str().unwrap(),
            "--trace",
            directory.join("trace.jsonl").to_str().unwrap(),
            "--out",
            directory.join("out.wav").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("base mode rejects edit-only"));
    assert!(!directory.join("trace.jsonl").exists());
    assert!(!directory.join("out.wav").exists());
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn oracle_runner_rejects_trace_and_wav_inside_canonical_checkout() {
    let root = std::env::temp_dir().join(format!(
        "rmi-dots-tts-oracle-descendant-{}",
        std::process::id()
    ));
    let checkout = root.join("checkout");
    let inputs = root.join("inputs");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(checkout.join("src")).unwrap();
    std::fs::create_dir_all(&inputs).unwrap();
    assert!(Command::new("git")
        .args(["init", "-q"])
        .arg(&checkout)
        .status()
        .unwrap()
        .success());
    assert!(Command::new("git")
        .args([
            "-C",
            checkout.to_str().unwrap(),
            "config",
            "user.email",
            "test@example.com"
        ])
        .status()
        .unwrap()
        .success());
    assert!(Command::new("git")
        .args([
            "-C",
            checkout.to_str().unwrap(),
            "config",
            "user.name",
            "Test"
        ])
        .status()
        .unwrap()
        .success());
    std::fs::write(checkout.join("src/.keep"), b"").unwrap();
    assert!(Command::new("git")
        .args(["-C", checkout.to_str().unwrap(), "add", "."])
        .status()
        .unwrap()
        .success());
    assert!(Command::new("git")
        .args(["-C", checkout.to_str().unwrap(), "commit", "-qm", "test"])
        .status()
        .unwrap()
        .success());
    for name in ["ref.wav", "latent.f32", "dit.f32"] {
        std::fs::write(inputs.join(name), [0u8; 4]).unwrap();
    }
    std::fs::create_dir(inputs.join("model")).unwrap();
    let sha = String::from_utf8(
        Command::new("git")
            .args(["-C", checkout.to_str().unwrap(), "rev-parse", "HEAD"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap()
    .trim()
    .to_owned();
    let script = format!(
        "import importlib.util,sys; spec=importlib.util.spec_from_file_location('runner', r'{}'); runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner); runner.PINNED_SHA={:?}; sys.argv=['runner','--checkout',r'{}','--model-dir',r'{}','--mode','base','--text','hello','--ref-audio',r'{}','--ref-text','reference','--latent-noise',r'{}','--dit-noise',r'{}','--trace',r'{}','--out',r'{}']; runner.main()",
        env!("CARGO_MANIFEST_DIR").to_owned() + "/tools/dots/run_dots_tts_oracle.py",
        sha,
        checkout.display(),
        inputs.join("model").display(),
        inputs.join("ref.wav").display(),
        inputs.join("latent.f32").display(),
        inputs.join("dit.f32").display(),
        checkout.join("trace.jsonl").display(),
        checkout.join("out.wav").display(),
    );
    let output = Command::new("python3")
        .args(["-c", &script])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("output must not be inside oracle checkout"),
        "{stderr}"
    );
    assert!(!stderr.contains("No module named 'dots_tts'"), "{stderr}");
    assert!(!checkout.join("trace.jsonl").exists());
    assert!(!checkout.join("out.wav").exists());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn oracle_builder_treats_input_checkout_as_read_only() {
    let directory =
        std::env::temp_dir().join(format!("rmi-dots-tts-oracle-input-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    assert!(Command::new("git")
        .args(["init", "-q"])
        .arg(&directory)
        .status()
        .unwrap()
        .success());
    assert!(Command::new("git")
        .args(["-C", directory.to_str().unwrap(), "remote", "add", "origin"])
        .arg(directory.join("missing-origin"))
        .status()
        .unwrap()
        .success());
    std::fs::write(directory.join("dirty"), b"untracked").unwrap();

    let before = Command::new("git")
        .args([
            "-C",
            directory.to_str().unwrap(),
            "status",
            "--porcelain=v1",
        ])
        .output()
        .unwrap();

    let output = Command::new("bash")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tools/dots/build_dots_tts_oracle.sh"
        ))
        .arg(&directory)
        .output()
        .unwrap();
    let after = Command::new("git")
        .args([
            "-C",
            directory.to_str().unwrap(),
            "status",
            "--porcelain=v1",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert_eq!(before.stdout, after.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("No such file or directory"), "{stderr}");
    assert!(!stderr.contains("clean checkout"));
    let _ = std::fs::remove_dir_all(&directory);
}

#[cfg(feature = "parity-trace")]
mod parity_support {
    use rand::SeedableRng;
    use rust_model_inference::models::dots::generate::{
        read_dots_wav_for_parity, synthesize_request_with_noise, GenerateOptions, GenerationRequest,
    };
    use rust_model_inference::models::dots::schedule::{
        build_edit_generation_schedule, build_generation_schedule,
    };
    use rust_model_inference::models::dots::DotsTtsModel;
    use rust_model_inference::models::qwen3::tts::codec::write_wav_f32;
    use rust_model_inference::{
        open_model_source, BPETokenizer, ComponentRole, ComputePool, TensorSource,
    };
    use serde_json::Value;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    pub const REFERENCE_TEXT: &str =
        "The CrossLand acquisition gave Washington Mutual a toe hold entry into Oregon via Portland.";
    pub const TRACE_FILTER: &str = "dots.schedule.ids,dots.audio.input48k,dots.speaker.input16k,dots.speaker.fbank,dots.speaker.xvector,dots.condition.g_cond,dots.prompt.distribution,dots.prompt.latents,dots.patch.embedding,dots.llm.hidden,dots.fm.sequence,dots.dit.time_embedding,dots.dit.time_linear0,dots.dit.time_silu,dots.dit.time,dots.dit.mods_input,dots.dit.input,dots.dit.projected_input,dots.dit.block_mods,dots.dit.block0.norm1,dots.dit.block0.attn_in,dots.dit.block0.qkv,dots.dit.block0.q_norm,dots.dit.block0.k_norm,dots.dit.block0.attn_out,dots.dit.block0.attn_proj,dots.dit.block0.norm2,dots.dit.block0.ffn_in,dots.dit.block0.fc1,dots.dit.block0.gelu,dots.dit.block0.fc2,dots.dit.block,dots.dit.final.norm,dots.dit.final.input,dots.dit.final.output,dots.dit.raw_velocity,dots.dit.velocity,dots.dit.z,dots.latent.consumed,dots.latent.payload,dots.vocoder.post_proj,dots.vocoder.dec_mi.linear0,dots.vocoder.dec_mi.lstm,dots.vocoder.dec_mi.residual,dots.vocoder.dec_mi.linear2,dots.vocoder.dec_mi,dots.vocoder.decoder.conv_pre,dots.vocoder.decoder.up,dots.vocoder.decoder.resblock0.activation0.upsample,dots.vocoder.decoder.resblock0.activation0.snakebeta,dots.vocoder.decoder.resblock0.act1,dots.vocoder.decoder.resblock0.conv1,dots.vocoder.decoder.resblock0.act2,dots.vocoder.decoder.resblock0.conv2,dots.vocoder.decoder.resblock0.residual,dots.vocoder.decoder.resblock,dots.vocoder.decoder.stage,dots.vocoder.decoder.activation_post.upsample,dots.vocoder.decoder.activation_post.snakebeta,dots.vocoder.decoder.activation_post,dots.vocoder.waveform";
    const CHECKPOINT_NAMES: &[&str] = &[
        "dots.audio.input48k",
        "dots.speaker.input16k",
        "dots.speaker.fbank",
        "dots.speaker.xvector",
        "dots.condition.g_cond",
        "dots.prompt.distribution",
        "dots.prompt.latents",
        "dots.patch.embedding",
        "dots.llm.hidden",
        "dots.fm.sequence",
        "dots.dit.time_embedding",
        "dots.dit.time_linear0",
        "dots.dit.time_silu",
        "dots.dit.time",
        "dots.dit.mods_input",
        "dots.dit.input",
        "dots.dit.projected_input",
        "dots.dit.block_mods",
        "dots.dit.block0.norm1",
        "dots.dit.block0.attn_in",
        "dots.dit.block0.qkv",
        "dots.dit.block0.q_norm",
        "dots.dit.block0.k_norm",
        "dots.dit.block0.attn_out",
        "dots.dit.block0.attn_proj",
        "dots.dit.block0.norm2",
        "dots.dit.block0.ffn_in",
        "dots.dit.block0.fc1",
        "dots.dit.block0.gelu",
        "dots.dit.block0.fc2",
        "dots.dit.block",
        "dots.dit.final.norm",
        "dots.dit.final.input",
        "dots.dit.final.output",
        "dots.dit.raw_velocity",
        "dots.dit.velocity",
        "dots.dit.z",
        "dots.latent.consumed",
        "dots.latent.payload",
        "dots.vocoder.post_proj",
        "dots.vocoder.dec_mi.linear0",
        "dots.vocoder.dec_mi.lstm",
        "dots.vocoder.dec_mi.residual",
        "dots.vocoder.dec_mi.linear2",
        "dots.vocoder.dec_mi",
        "dots.vocoder.decoder.conv_pre",
        "dots.vocoder.decoder.up",
        "dots.vocoder.decoder.resblock0.activation0.upsample",
        "dots.vocoder.decoder.resblock0.activation0.snakebeta",
        "dots.vocoder.decoder.resblock0.act1",
        "dots.vocoder.decoder.resblock0.conv1",
        "dots.vocoder.decoder.resblock0.act2",
        "dots.vocoder.decoder.resblock0.conv2",
        "dots.vocoder.decoder.resblock0.residual",
        "dots.vocoder.decoder.resblock",
        "dots.vocoder.decoder.stage",
        "dots.vocoder.decoder.activation_post.upsample",
        "dots.vocoder.decoder.activation_post.snakebeta",
        "dots.vocoder.decoder.activation_post",
        "dots.vocoder.waveform",
    ];

    pub struct LoadedDots {
        pub model: DotsTtsModel,
        pub tokenizer: BPETokenizer,
    }

    #[derive(Clone, Copy)]
    pub enum Fixture<'a> {
        Base {
            text: &'a str,
        },
        Edit {
            instruction: &'a str,
            target_text: &'a str,
        },
    }

    pub struct FixtureOutput {
        pub payload: Vec<f32>,
        pub pcm: Vec<i16>,
    }

    pub fn required_path(name: &str) -> PathBuf {
        PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("missing {name}")))
    }

    pub fn temp_root() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "rmi-dots-tts-parity-{}-{nonce}",
            std::process::id()
        ))
    }

    pub fn build_oracle(input: &Path) -> PathBuf {
        let output = Command::new("bash")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tools/dots/build_dots_tts_oracle.sh"
            ))
            .arg(input)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "oracle builder stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let path = PathBuf::from(String::from_utf8(output.stdout).unwrap().trim());
        assert!(path.is_dir(), "{}", path.display());
        path
    }

    pub fn load_dots(directory: &Path, variant: &str) -> LoadedDots {
        let llm_path = directory.join(format!("dots-tts-{variant}.gguf"));
        let mmproj_path = directory.join(format!("dots-tts-{variant}-mmproj.gguf"));
        let llm_source: Arc<dyn TensorSource> =
            Arc::from(open_model_source(&llm_path, ComponentRole::Llm).unwrap());
        let tokenizer =
            BPETokenizer::from_gguf_metadata(|key| llm_source.metadata(key).cloned()).unwrap();
        let mmproj_source: Arc<dyn TensorSource> =
            Arc::from(open_model_source(&mmproj_path, ComponentRole::Mmproj).unwrap());
        let model =
            DotsTtsModel::from_sources(llm_source, mmproj_source, Arc::new(ComputePool::new(1)))
                .unwrap();
        LoadedDots { model, tokenizer }
    }

    fn oracle_python() -> std::ffi::OsString {
        std::env::var_os("DOTS_TTS_PYTHON").unwrap_or_else(|| "python3".into())
    }

    fn noise(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| ((index.wrapping_mul(73) % 257) as f32 - 128.0) / 97.0)
            .collect()
    }

    fn write_f32(path: &Path, values: &[f32]) {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path, bytes).unwrap();
    }

    fn records(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn shape(record: &Value) -> Vec<usize> {
        record["shape"]
            .as_array()
            .expect("checkpoint must contain shape")
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect()
    }

    fn sidecar(trace: &Path, record: &Value) -> PathBuf {
        let value = record
            .get("binary_path")
            .or_else(|| record.get("path"))
            .and_then(Value::as_str)
            .expect("float checkpoint must contain its own sidecar path");
        let path = PathBuf::from(value);
        if path.is_absolute() {
            path
        } else {
            trace.parent().unwrap().join(path)
        }
    }

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len() % 4, 0, "{}", path.display());
        bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }

    pub fn assert_f32_bits(name: &str, rust: &[f32], oracle: &[f32]) {
        assert_eq!(rust.len(), oracle.len(), "{name} length");
        for (index, (&left, &right)) in rust.iter().zip(oracle).enumerate() {
            assert_eq!(
                left.to_bits(),
                right.to_bits(),
                "{name}[{index}] rust={:08x} oracle={:08x}",
                left.to_bits(),
                right.to_bits()
            );
        }
    }

    fn compare_traces(rust_trace: &Path, oracle_trace: &Path) -> Vec<f32> {
        let rust = records(rust_trace);
        let oracle = records(oracle_trace);
        let mut payload = Vec::new();
        for index in 0..rust.len().max(oracle.len()) {
            let (left, right) = match (rust.get(index), oracle.get(index)) {
                (Some(left), Some(right)) => (left, right),
                (Some(left), None) => panic!(
                    "record {index} Rust {} present, Oracle record missing",
                    left["name"]
                ),
                (None, Some(right)) => panic!(
                    "record {index} Oracle {} present, Rust record missing",
                    right["name"]
                ),
                (None, None) => unreachable!(),
            };
            if index == 0 {
                assert_eq!(left["name"], "dots.schedule.ids");
                assert_eq!(right["name"], "dots.schedule.ids");
                assert_eq!(left, right, "schedule record");
                continue;
            }
            for field in ["name", "occurrence", "step", "shape"] {
                assert_eq!(left[field], right[field], "record {index} field {field}");
            }
            let left_values = read_f32(&sidecar(rust_trace, left));
            let right_values = read_f32(&sidecar(oracle_trace, right));
            let expected = shape(left).iter().product::<usize>();
            assert_eq!(left_values.len(), expected, "record {index} Rust shape");
            assert_eq!(right_values.len(), expected, "record {index} oracle shape");
            let name = left["name"].as_str().unwrap();
            assert_f32_bits(
                &format!("{name} occurrence {}", left["occurrence"]),
                &left_values,
                &right_values,
            );
            if name == "dots.latent.payload" {
                payload.extend(left_values);
            }
        }
        for required in CHECKPOINT_NAMES {
            assert!(
                rust.iter().any(|record| record["name"] == *required),
                "missing {required}"
            );
            assert!(
                oracle.iter().any(|record| record["name"] == *required),
                "missing {required}"
            );
        }
        payload
    }

    fn pcm16(path: &Path) -> Vec<i16> {
        let bytes = std::fs::read(path).unwrap();
        assert!(bytes.len() >= 44, "{}", path.display());
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes(bytes[20..22].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 1);
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            48_000
        );
        assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 16);
        assert_eq!(&bytes[36..40], b"data");
        let data_len = u32::from_le_bytes(bytes[40..44].try_into().unwrap()) as usize;
        assert!(data_len > 0 && data_len % 2 == 0);
        assert_eq!(bytes.len(), 44 + data_len);
        bytes[44..]
            .chunks_exact(2)
            .map(|word| i16::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }

    fn run(command: &mut Command, label: &str) {
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{label} stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    pub fn run_fixture(
        index: usize,
        root: &Path,
        oracle_checkout: &Path,
        oracle_model: &Path,
        reference: &Path,
        loaded: &LoadedDots,
        fixture: Fixture<'_>,
    ) -> FixtureOutput {
        let directory = root.join(index.to_string());
        std::fs::create_dir(&directory).unwrap();
        let edit = matches!(fixture, Fixture::Edit { .. });
        let samples_per_patch = loaded.model.config.samples_per_patch();
        let wav = read_dots_wav_for_parity(reference, edit, samples_per_patch).unwrap();
        let padded_samples = wav.len().div_ceil(samples_per_patch) * samples_per_patch;
        let padded_patches = padded_samples / samples_per_patch;
        let latent_noise =
            noise(padded_samples / loaded.model.config.hop_size * loaded.model.config.latent_dim);
        let decoded_patches = if edit { 2 } else { 3 };
        let dit_noise = noise(
            decoded_patches * loaded.model.config.patch_size * loaded.model.config.latent_dim,
        );
        let latent_path = directory.join("latent.f32");
        let dit_path = directory.join("dit.f32");
        let rust_trace = directory.join("rust.jsonl");
        let oracle_trace = directory.join("oracle.jsonl");
        let rust_wav = directory.join("rust.wav");
        let oracle_wav = directory.join("oracle.wav");
        write_f32(&latent_path, &latent_noise);
        write_f32(&dit_path, &dit_noise);

        let mut oracle = Command::new(oracle_python());
        oracle
            .env("PYTHONNOUSERSITE", "1")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tools/dots/run_dots_tts_oracle.py"
            ))
            .args(["--checkout", oracle_checkout.to_str().unwrap()])
            .args(["--model-dir", oracle_model.to_str().unwrap()])
            .args(["--ref-audio", reference.to_str().unwrap()])
            .args(["--latent-noise", latent_path.to_str().unwrap()])
            .args(["--dit-noise", dit_path.to_str().unwrap()])
            .args(["--trace", oracle_trace.to_str().unwrap()])
            .args(["--out", oracle_wav.to_str().unwrap()]);
        match fixture {
            Fixture::Base { text } => {
                oracle.args([
                    "--mode",
                    "base",
                    "--text",
                    text,
                    "--ref-text",
                    REFERENCE_TEXT,
                ]);
            }
            Fixture::Edit {
                instruction,
                target_text,
            } => {
                oracle.args([
                    "--mode",
                    "edit",
                    "--instruction",
                    instruction,
                    "--source-text",
                    REFERENCE_TEXT,
                    "--target-text",
                    target_text,
                ]);
            }
        }
        run(&mut oracle, "Python oracle");

        let oracle_records = records(&oracle_trace);
        assert_eq!(oracle_records[0]["name"], "dots.schedule.ids");
        let audio_record = &oracle_records[1];
        assert_eq!(audio_record["name"], "dots.audio.input48k");
        let oracle_wav_input = read_f32(&sidecar(&oracle_trace, audio_record));
        assert_f32_bits("dots.audio.input48k", &wav, &oracle_wav_input);

        std::env::set_var("RMI_PARITY_TRACE", &rust_trace);
        std::env::set_var("RMI_PARITY_FILTER", "dots.schedule.ids");
        let schedule = match fixture {
            Fixture::Base { text } => build_generation_schedule(
                &loaded.tokenizer,
                &format!("{REFERENCE_TEXT}\n{text}"),
                padded_patches - 1,
                decoded_patches,
            ),
            Fixture::Edit {
                instruction,
                target_text,
            } => build_edit_generation_schedule(
                &loaded.tokenizer,
                REFERENCE_TEXT,
                instruction,
                target_text,
                padded_patches,
                decoded_patches,
            ),
        }
        .unwrap();
        rust_model_inference::parity_trace::report(rust_model_inference::parity_trace::token_ids(
            "dots.schedule.ids",
            &schedule.ids,
        ));
        std::env::set_var(
            "RMI_PARITY_FILTER",
            TRACE_FILTER
                .strip_prefix("dots.schedule.ids,")
                .expect("schedule must be the first trace filter"),
        );
        let prompt = loaded
            .model
            .prepare_prompt_conditioning_with_noise(
                &wav,
                1.5,
                true,
                usize::from(!edit),
                &latent_noise,
            )
            .unwrap();
        assert_eq!(
            schedule.fill_span_positions.len(),
            prompt.patches.len()
                / (loaded.model.config.patch_size * loaded.model.config.latent_dim),
        );
        assert_eq!(schedule.decode_span_positions.len(), decoded_patches);
        let mut options = GenerateOptions::for_model(&loaded.model.config);
        options.max_patches = 2;
        options.nfe = 10;
        options.guidance = 1.2;
        options.speaker_scale = 1.5;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let waveform = match fixture {
            Fixture::Base { text } => {
                let schedule_text = format!("{REFERENCE_TEXT}\n{text}");
                synthesize_request_with_noise(
                    &loaded.model,
                    &loaded.tokenizer,
                    GenerationRequest::Base {
                        text: &schedule_text,
                        prompt: Some(&prompt),
                    },
                    &options,
                    &mut rng,
                    &dit_noise,
                )
            }
            Fixture::Edit {
                instruction,
                target_text,
            } => synthesize_request_with_noise(
                &loaded.model,
                &loaded.tokenizer,
                GenerationRequest::Edit {
                    source_text: REFERENCE_TEXT,
                    instruction,
                    target_text,
                    source: &prompt,
                },
                &options,
                &mut rng,
                &dit_noise,
            ),
        }
        .unwrap();
        std::env::remove_var("RMI_PARITY_TRACE");
        std::env::remove_var("RMI_PARITY_FILTER");
        write_wav_f32(&rust_wav, &waveform, 48_000).unwrap();

        let payload = compare_traces(&rust_trace, &oracle_trace);
        let pcm = pcm16(&rust_wav);
        assert_eq!(pcm, pcm16(&oracle_wav));
        FixtureOutput { payload, pcm }
    }
}

#[cfg(feature = "parity-trace")]
#[test]
#[ignore = "requires fresh dots.tts GGUF pairs, source checkpoints, spoken WAV, and pinned Python oracle"]
fn dots_base_and_edit_match_pinned_oracle_bitwise() {
    let base_dir = required_path("DOTS_TTS_BASE_DIR");
    let edit_dir = required_path("DOTS_TTS_EDIT_DIR");
    let oracle_input = required_path("DOTS_TTS_ORACLE_CHECKOUT");
    let reference = required_path("DOTS_TTS_REF_WAV");
    let root = temp_root();
    std::fs::create_dir(&root).unwrap();
    let oracle = build_oracle(&oracle_input);
    let model_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models");

    let (first, second) = {
        let loaded = load_dots(&base_dir, "base");
        (
            run_fixture(
                0,
                &root,
                &oracle,
                &model_root.join("dots.tts-base"),
                &reference,
                &loaded,
                Fixture::Base {
                    text: "In many cases, such as France, no distinct regional substructures have been employed.",
                },
            ),
            run_fixture(
                1,
                &root,
                &oracle,
                &model_root.join("dots.tts-base"),
                &reference,
                &loaded,
                Fixture::Base {
                    text: "The weather changed before the evening train arrived.",
                },
            ),
        )
    };
    assert_ne!(
        first.payload, second.payload,
        "base payload prompt sensitivity"
    );
    assert_ne!(first.pcm, second.pcm, "base PCM prompt sensitivity");

    let loaded = load_dots(&edit_dir, "edit");
    run_fixture(
        2,
        &root,
        &oracle,
        &model_root.join("dots.tts.edit"),
        &reference,
        &loaded,
        Fixture::Edit {
            instruction: "<sub targ=\"a local bank\">Washington Mutual</sub>",
            target_text: "The CrossLand acquisition gave a local bank a toe hold entry into Oregon via Portland.",
        },
    );

    std::fs::remove_dir_all(&root).unwrap();
    std::fs::remove_dir_all(oracle.parent().unwrap()).unwrap();
}
