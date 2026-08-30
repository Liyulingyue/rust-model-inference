use rust_model_inference::app::{run_gemma4, Gemma4Request};
use rust_model_inference::core::scratchpad::KvFormat;
use rust_model_inference::models::gemma4::asr::Gemma4AudioModel;
use rust_model_inference::models::gemma4::vision::Gemma4VisionModel;
use rust_model_inference::{GGMLType, GGUFLoader, MetaValue};
use serde_json::Value;
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

const GEMMA4_MODEL_NAME: &str = "gemma-4-E2B-it-Q8_0.gguf";
const GEMMA4_MMPROJ_NAME: &str = "mmproj-F16.gguf";
const GEMMA4_THREADS: usize = 4;
const GEMMA4_PROMPT: &str = "describe";
const GEMMA4_LAYERS: usize = 35;
const GEMMA4_CHAT_TEMPLATE: &str = r#"{{ '<|turn>user\n' + messages[0].content + '<turn|>\n' }}{% if add_generation_prompt %}{{ '<|turn>model\n' }}{% endif %}"#;

static GEMMA4_TRACE_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Debug, PartialEq)]
enum TraceValues {
    F32(Vec<u32>),
    Tokens(Vec<u32>),
}

#[derive(Clone, Debug, PartialEq)]
struct TraceRecord {
    checkpoint: String,
    occurrence: usize,
    shape: Vec<usize>,
    len: usize,
    values: TraceValues,
}

fn record(checkpoint: &str, occurrence: usize, shape: &[usize], values: &[f32]) -> TraceRecord {
    TraceRecord {
        checkpoint: checkpoint.to_owned(),
        occurrence,
        shape: shape.to_vec(),
        len: values.len(),
        values: TraceValues::F32(values.iter().map(|value| value.to_bits()).collect()),
    }
}

fn token_record(checkpoint: &str, occurrence: usize, values: Vec<u32>) -> TraceRecord {
    TraceRecord {
        checkpoint: checkpoint.to_owned(),
        occurrence,
        shape: vec![values.len()],
        len: values.len(),
        values: TraceValues::Tokens(values),
    }
}

fn assert_trace_equal(
    case: &str,
    rust: &[TraceRecord],
    oracle: &[TraceRecord],
) -> Result<(), String> {
    let record_count = rust.len().max(oracle.len());
    for record_index in 0..record_count {
        let (rust_record, oracle_record) = match (rust.get(record_index), oracle.get(record_index))
        {
            (Some(rust_record), Some(oracle_record)) => (rust_record, oracle_record),
            (Some(rust_record), None) => {
                return Err(format!(
                    "case {case} checkpoint {} occurrence {} index {record_index}: Rust record present, Oracle record missing",
                    rust_record.checkpoint, rust_record.occurrence,
                ));
            }
            (None, Some(oracle_record)) => {
                return Err(format!(
                    "case {case} checkpoint {} occurrence {} index {record_index}: Rust record missing, Oracle record present",
                    oracle_record.checkpoint, oracle_record.occurrence,
                ));
            }
            (None, None) => unreachable!(),
        };

        if rust_record.checkpoint != oracle_record.checkpoint {
            return Err(format!(
                "case {case} checkpoint Rust={} Oracle={} occurrence Rust={} Oracle={} index {record_index}: record order mismatch",
                rust_record.checkpoint,
                oracle_record.checkpoint,
                rust_record.occurrence,
                oracle_record.occurrence,
            ));
        }
        if rust_record.occurrence != oracle_record.occurrence {
            return Err(format!(
                "case {case} checkpoint {} occurrence Rust={} Oracle={} index {record_index}: occurrence mismatch",
                rust_record.checkpoint, rust_record.occurrence, oracle_record.occurrence,
            ));
        }
        if rust_record.shape != oracle_record.shape {
            let index = rust_record
                .shape
                .iter()
                .zip(&oracle_record.shape)
                .position(|(rust, oracle)| rust != oracle)
                .unwrap_or_else(|| rust_record.shape.len().min(oracle_record.shape.len()));
            return Err(format!(
                "case {case} checkpoint {} occurrence {} index {index}: shape Rust={:?} Oracle={:?}",
                rust_record.checkpoint,
                rust_record.occurrence,
                rust_record.shape,
                oracle_record.shape,
            ));
        }
        if rust_record.len != oracle_record.len {
            return Err(format!(
                "case {case} checkpoint {} occurrence {} index {}: length Rust={} Oracle={}",
                rust_record.checkpoint,
                rust_record.occurrence,
                rust_record.len.min(oracle_record.len),
                rust_record.len,
                oracle_record.len,
            ));
        }

        match (&rust_record.values, &oracle_record.values) {
            (TraceValues::F32(rust_words), TraceValues::F32(oracle_words)) => {
                for (index, (&rust_bits, &oracle_bits)) in
                    rust_words.iter().zip(oracle_words).enumerate()
                {
                    if rust_bits != oracle_bits {
                        return Err(format!(
                            "case {case} checkpoint {} occurrence {} index {index}: Rust=0x{rust_bits:08x} Oracle=0x{oracle_bits:08x}",
                            rust_record.checkpoint, rust_record.occurrence,
                        ));
                    }
                }
            }
            (TraceValues::Tokens(rust_tokens), TraceValues::Tokens(oracle_tokens)) => {
                for (index, (&rust_id, &oracle_id)) in
                    rust_tokens.iter().zip(oracle_tokens).enumerate()
                {
                    if rust_id != oracle_id {
                        return Err(format!(
                            "case {case} checkpoint {} occurrence {} index {index}: Rust=0x{rust_id:08x} Oracle=0x{oracle_id:08x}",
                            rust_record.checkpoint, rust_record.occurrence,
                        ));
                    }
                }
            }
            (rust_values, oracle_values) => {
                return Err(format!(
                    "case {case} checkpoint {} occurrence {} index 0: value kind Rust={} Oracle={}",
                    rust_record.checkpoint,
                    rust_record.occurrence,
                    trace_kind(rust_values),
                    trace_kind(oracle_values),
                ));
            }
        }
    }
    Ok(())
}

fn trace_kind(values: &TraceValues) -> &'static str {
    match values {
        TraceValues::F32(_) => "f32",
        TraceValues::Tokens(_) => "token_ids",
    }
}

fn trace_records(path: &Path) -> Result<Vec<TraceRecord>, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let mut occurrences = HashMap::<String, usize>::new();
    let mut records = Vec::new();
    for (record_index, line) in contents.lines().enumerate() {
        let value: Value = serde_json::from_str(line).map_err(|error| {
            format!(
                "invalid trace JSON {} record {record_index}: {error}",
                path.display()
            )
        })?;
        let checkpoint = value["name"]
            .as_str()
            .ok_or_else(|| format!("{} record {record_index} has no name", path.display()))?
            .to_owned();
        let expected_occurrence = *occurrences.get(&checkpoint).unwrap_or(&0);
        let occurrence = match value.get("occurrence") {
            Some(value) => usize::try_from(value.as_u64().ok_or_else(|| {
                format!(
                    "{} record {record_index} has invalid occurrence",
                    path.display()
                )
            })?)
            .map_err(|_| format!("{} occurrence does not fit usize", path.display()))?,
            None => expected_occurrence,
        };
        if occurrence != expected_occurrence {
            return Err(format!(
                "{} checkpoint {checkpoint} occurrence {occurrence} is not the next observed occurrence {expected_occurrence}",
                path.display()
            ));
        }
        occurrences.insert(checkpoint.clone(), occurrence + 1);

        let shape = value["shape"]
            .as_array()
            .ok_or_else(|| {
                format!(
                    "{} checkpoint {checkpoint} occurrence {occurrence} has no shape",
                    path.display()
                )
            })?
            .iter()
            .map(|dimension| {
                dimension
                    .as_u64()
                    .ok_or_else(|| {
                        format!(
                            "{} checkpoint {checkpoint} occurrence {occurrence} has an invalid shape",
                            path.display()
                        )
                    })
                    .and_then(|dimension| {
                        usize::try_from(dimension).map_err(|_| {
                            format!(
                                "{} checkpoint {checkpoint} occurrence {occurrence} shape does not fit usize",
                                path.display()
                            )
                        })
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let len = usize::try_from(value["len"].as_u64().ok_or_else(|| {
            format!(
                "{} checkpoint {checkpoint} occurrence {occurrence} has no length",
                path.display()
            )
        })?)
        .map_err(|_| format!("{} length does not fit usize", path.display()))?;
        let shape_len = shape.iter().try_fold(1usize, |length, dimension| {
            length.checked_mul(*dimension).ok_or_else(|| {
                format!(
                    "{} checkpoint {checkpoint} occurrence {occurrence} shape overflows",
                    path.display()
                )
            })
        })?;
        if shape_len != len {
            return Err(format!(
                "{} checkpoint {checkpoint} occurrence {occurrence} shape has {shape_len} words but length is {len}",
                path.display()
            ));
        }

        let values = if let Some(token_ids) = value.get("token_ids") {
            let token_ids = token_ids.as_array().ok_or_else(|| {
                format!(
                    "{} checkpoint {checkpoint} occurrence {occurrence} has invalid token_ids",
                    path.display()
                )
            })?;
            let tokens = token_ids
                .iter()
                .map(|token| {
                    u32::try_from(token.as_u64().ok_or_else(|| {
                        format!(
                            "{} checkpoint {checkpoint} occurrence {occurrence} has an invalid token ID",
                            path.display()
                        )
                    })?)
                    .map_err(|_| {
                        format!(
                            "{} checkpoint {checkpoint} occurrence {occurrence} token ID does not fit u32",
                            path.display()
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if tokens.len() != len {
                return Err(format!(
                    "{} checkpoint {checkpoint} occurrence {occurrence} has {} token IDs but length is {len}",
                    path.display(),
                    tokens.len(),
                ));
            }
            TraceValues::Tokens(tokens)
        } else {
            let binary_path = value["binary_path"].as_str().ok_or_else(|| {
                format!(
                    "{} checkpoint {checkpoint} occurrence {occurrence} has no binary_path",
                    path.display()
                )
            })?;
            let bytes = std::fs::read(binary_path)
                .map_err(|error| format!("failed to read {binary_path}: {error}"))?;
            if bytes.len() % 4 != 0 {
                return Err(format!("{binary_path} has a trailing partial F32 word"));
            }
            let words = bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>();
            if words.len() != len {
                return Err(format!(
                    "{} checkpoint {checkpoint} occurrence {occurrence} sidecar has {} F32 words but length is {len}",
                    path.display(),
                    words.len(),
                ));
            }
            TraceValues::F32(words)
        };
        records.push(TraceRecord {
            checkpoint,
            occurrence,
            shape,
            len,
            values,
        });
    }
    Ok(records)
}

fn generated_id(records: &[TraceRecord]) -> Result<u32, String> {
    let logits = records
        .iter()
        .rev()
        .find(|record| record.checkpoint == "gemma4.logits")
        .ok_or_else(|| "Rust trace has no gemma4.logits checkpoint".to_owned())?;
    let TraceValues::F32(words) = &logits.values else {
        return Err("Rust gemma4.logits checkpoint is not F32".to_owned());
    };
    if words.is_empty() {
        return Err("Rust gemma4.logits checkpoint is empty".to_owned());
    }
    let mut best = 0usize;
    for id in 0..words.len() {
        let value = f32::from_bits(words[id]);
        if !value.is_finite() {
            return Err(format!("Rust logits contain {value:?} at index {id}"));
        }
        if value > f32::from_bits(words[best]) {
            best = id;
        }
    }
    u32::try_from(best).map_err(|_| format!("generated token ID {best} does not fit u32"))
}

#[derive(Clone, Copy)]
struct ParityCase {
    name: &'static str,
    image: bool,
    audio: bool,
}

const PARITY_CASES: [ParityCase; 4] = [
    ParityCase {
        name: "text",
        image: false,
        audio: false,
    },
    ParityCase {
        name: "image",
        image: true,
        audio: false,
    },
    ParityCase {
        name: "audio",
        image: false,
        audio: true,
    },
    ParityCase {
        name: "image+audio",
        image: true,
        audio: true,
    },
];

fn required_trace_names(case: ParityCase) -> Vec<String> {
    let mut names = Vec::with_capacity(4 + GEMMA4_LAYERS * 3);
    if case.image {
        names.extend([
            "gemma4.vision.preprocessed".to_owned(),
            "gemma4.vision.projected".to_owned(),
        ]);
    }
    if case.audio {
        names.extend([
            "gemma4.audio.mel".to_owned(),
            "gemma4.audio.projected".to_owned(),
        ]);
    }
    names.push("gemma4.tokens".to_owned());
    for layer in 0..GEMMA4_LAYERS {
        names.extend([
            format!("gemma4.layer.{layer}.attn_out"),
            format!("gemma4.layer.{layer}.ffn_out"),
            format!("gemma4.layer.{layer}.per_layer_out"),
        ]);
    }
    names.extend([
        "gemma4.final.norm".to_owned(),
        "gemma4.logits".to_owned(),
        "gemma4.generated_ids".to_owned(),
    ]);
    names
}

fn require_trace_names(
    case: ParityCase,
    side: &str,
    records: &[TraceRecord],
) -> Result<(), String> {
    let required = required_trace_names(case);
    for name in &required {
        if !records.iter().any(|record| &record.checkpoint == name) {
            return Err(format!(
                "case {} {side} trace is missing required checkpoint {name}",
                case.name
            ));
        }
    }
    if let Some(record) = records
        .iter()
        .find(|record| !required.contains(&record.checkpoint))
    {
        return Err(format!(
            "case {} {side} trace contains unexpected checkpoint {} occurrence {}",
            case.name, record.checkpoint, record.occurrence
        ));
    }
    Ok(())
}

fn write_image_fixture(path: &Path) -> Result<(), String> {
    image::RgbImage::from_fn(64, 48, |x, y| {
        image::Rgb([
            (x.wrapping_mul(3).wrapping_add(y)) as u8,
            (y.wrapping_mul(5).wrapping_add(x)) as u8,
            (x ^ y) as u8,
        ])
    })
    .save(path)
    .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn write_audio_fixture(path: &Path) -> Result<(), String> {
    let samples = (0..320)
        .map(|index| ((index as f32 * 0.03125).sin() * 16_384.0) as i16)
        .collect::<Vec<_>>();
    let data_len = u32::try_from(samples.len() * 2).unwrap();
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&16_000u32.to_le_bytes());
    wav.extend_from_slice(&32_000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    std::fs::write(path, wav)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn restore_env(name: &str, value: Option<OsString>) {
    match value {
        Some(value) => std::env::set_var(name, value),
        None => std::env::remove_var(name),
    }
}

fn run_rust_case(
    case: ParityCase,
    model: &Path,
    mmproj: &Path,
    image: &Path,
    audio: &Path,
    trace: &Path,
) -> Result<(), String> {
    let old_trace = std::env::var_os("RMI_PARITY_TRACE");
    let old_filter = std::env::var_os("RMI_PARITY_FILTER");
    let filter = required_trace_names(case)
        .into_iter()
        .filter(|name| name != "gemma4.generated_ids")
        .collect::<Vec<_>>()
        .join(",");
    std::env::set_var("RMI_PARITY_TRACE", trace);
    std::env::set_var("RMI_PARITY_FILTER", filter);
    let result = run_gemma4(Gemma4Request {
        model,
        mmproj: (case.image || case.audio).then_some(mmproj),
        image: case.image.then_some(image),
        audio: case.audio.then_some(audio),
        prompt: GEMMA4_PROMPT,
        max_tokens: 1,
        threads: GEMMA4_THREADS,
        kv_format: KvFormat::F32,
    });
    restore_env("RMI_PARITY_TRACE", old_trace);
    restore_env("RMI_PARITY_FILTER", old_filter);
    result.map_err(|error| format!("case {} Rust inference failed: {error}", case.name))
}

fn run_command(command: &mut Command, label: &str) -> Result<(), String> {
    let rendered = format!("{command:?}");
    let output = command
        .output()
        .map_err(|error| format!("failed to run {rendered}: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "{label} failed ({rendered})\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    ))
}

fn run_oracle_case(
    case: ParityCase,
    oracle: &Path,
    model: &Path,
    mmproj: &Path,
    image: &Path,
    audio: &Path,
    trace: &Path,
) -> Result<(), String> {
    let threads = GEMMA4_THREADS.to_string();
    let mut command = Command::new(oracle);
    command
        .env("RMI_PARITY_TRACE", trace)
        .arg("-m")
        .arg(model)
        .arg("--mmproj")
        .arg(mmproj)
        .args([
            "-p",
            GEMMA4_PROMPT,
            "-n",
            "1",
            "-t",
            &threads,
            "-tb",
            &threads,
            "-b",
            "1",
            "-ub",
            "1",
            "-ngl",
            "0",
            "-ctk",
            "f32",
            "-ctv",
            "f32",
            "--temp",
            "0",
            "--top-k",
            "1",
            "--top-p",
            "1.0",
            "--repeat-penalty",
            "1.0",
            "--jinja",
            "--chat-template",
            GEMMA4_CHAT_TEMPLATE,
            "--flash-attn",
            "off",
            "--no-warmup",
            "--no-mmproj-offload",
        ]);
    if case.image {
        command.arg("--image").arg(image);
    }
    if case.audio {
        command.arg("--audio").arg(audio);
    }
    run_command(
        &mut command,
        &format!("case {} Oracle inference", case.name),
    )
}

fn run_parity_case(
    case: ParityCase,
    root: &Path,
    oracle: &Path,
    model: &Path,
    mmproj: &Path,
    image: &Path,
    audio: &Path,
) -> Result<(), String> {
    let directory = root.join(case.name.replace('+', "-"));
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("failed to create {}: {error}", directory.display()))?;
    let rust_trace = directory.join("rust.jsonl");
    let oracle_trace = directory.join("oracle.jsonl");

    run_rust_case(case, model, mmproj, image, audio, &rust_trace)?;
    let mut rust_records = trace_records(&rust_trace)?;
    let generated_id = generated_id(&rust_records)?;
    rust_records.push(token_record("gemma4.generated_ids", 0, vec![generated_id]));
    require_trace_names(case, "Rust", &rust_records)?;

    run_oracle_case(case, oracle, model, mmproj, image, audio, &oracle_trace)?;
    let oracle_records = trace_records(&oracle_trace)?;
    require_trace_names(case, "Oracle", &oracle_records)?;
    assert_trace_equal(case.name, &rust_records, &oracle_records)
        .map_err(|error| format!("{error}\ntraces retained in {}", directory.display()))?;
    std::fs::remove_dir_all(&directory)
        .map_err(|error| format!("failed to remove {}: {error}", directory.display()))?;
    eprintln!("Gemma4 parity case {}: PASS", case.name);
    Ok(())
}

fn gemma4_model_path() -> PathBuf {
    std::env::var_os("RMI_GEMMA4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-e2b").join(GEMMA4_MODEL_NAME))
}

fn gemma4_mmproj_path() -> PathBuf {
    std::env::var_os("RMI_GEMMA4_MMPROJ")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-e2b").join(GEMMA4_MMPROJ_NAME))
}

fn require_gemma4_gguf(
    path: &Path,
    expected_name: &str,
    architecture: &str,
    tensor: &str,
    ggml_type: GGMLType,
) -> Result<(), String> {
    let actual_name = path.file_name().and_then(|name| name.to_str());
    if actual_name != Some(expected_name) {
        return Err(format!(
            "invalid Gemma4 GGUF basename: expected {expected_name:?}, got {actual_name:?}"
        ));
    }
    let loader = GGUFLoader::from_file(path)?;
    match loader.metadata("general.architecture") {
        Some(MetaValue::String(value)) if value == architecture => {}
        actual => {
            return Err(format!(
                "invalid general.architecture in {}: expected {architecture:?}, got {actual:?}",
                path.display()
            ));
        }
    }
    let actual_type = loader.tensor_info(tensor).map(|info| info.ggml_type);
    if actual_type != Some(ggml_type) {
        return Err(format!(
            "invalid {tensor} in {}: expected {ggml_type:?}, got {actual_type:?}",
            path.display()
        ));
    }
    Ok(())
}

fn preflight_gemma4_ggufs(model: &Path, mmproj: &Path) -> Result<(), String> {
    require_gemma4_gguf(
        model,
        GEMMA4_MODEL_NAME,
        "gemma4",
        "token_embd.weight",
        GGMLType::Q8_0,
    )?;
    require_gemma4_gguf(
        mmproj,
        GEMMA4_MMPROJ_NAME,
        "clip",
        "v.patch_embd.weight",
        GGMLType::F16,
    )
}

#[test]
fn gemma4_preflight_rejects_arbitrary_files() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "rmi-gemma4-invalid-preflight-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();

    let mut empty_gguf = Vec::new();
    empty_gguf.extend_from_slice(b"GGUF");
    empty_gguf.extend_from_slice(&3u32.to_le_bytes());
    empty_gguf.extend_from_slice(&0u64.to_le_bytes());
    empty_gguf.extend_from_slice(&0u64.to_le_bytes());

    let model = root.join(GEMMA4_MODEL_NAME);
    let mmproj = root.join(GEMMA4_MMPROJ_NAME);
    std::fs::write(&model, &empty_gguf).unwrap();
    std::fs::write(&mmproj, &empty_gguf).unwrap();

    for (path, expected_name, architecture, tensor, ggml_type) in [
        (
            model.as_path(),
            GEMMA4_MODEL_NAME,
            "gemma4",
            "token_embd.weight",
            GGMLType::Q8_0,
        ),
        (
            mmproj.as_path(),
            GEMMA4_MMPROJ_NAME,
            "clip",
            "v.patch_embd.weight",
            GGMLType::F16,
        ),
    ] {
        let error =
            require_gemma4_gguf(path, expected_name, architecture, tensor, ggml_type).unwrap_err();
        assert!(error.contains("general.architecture"), "got: {error}");
    }

    let mut oracle_started = false;
    let error = preflight_gemma4_ggufs(&model, &mmproj)
        .and_then(|()| {
            oracle_started = true;
            Ok(())
        })
        .unwrap_err();
    assert!(error.contains("general.architecture"), "got: {error}");
    assert!(
        !oracle_started,
        "Oracle must not run after failed preflight"
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn readme_lists_gemma4_model_projector_and_media_flags() {
    let readme = std::fs::read_to_string("README.md").unwrap();
    for value in [
        "gemma-4-E2B-it-Q8_0.gguf",
        "mmproj-F16.gguf",
        "--image",
        "--audio",
    ] {
        assert!(readme.contains(value), "README missing {value}");
    }
}

fn ensure_gemma4_oracle() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("LLAMA_GEMMA4_TRACE_BIN").map(PathBuf::from) {
        if path.is_file() {
            return Ok(path);
        }
    }

    let llama_dir = std::env::var_os("LLAMA_CPP_DIR").ok_or_else(|| {
        "LLAMA_CPP_DIR is required when LLAMA_GEMMA4_TRACE_BIN is not a file".to_owned()
    })?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tools/gemma4/build_oracle.sh");
    let output = Command::new("bash")
        .arg(&script)
        .arg(llama_dir)
        .output()
        .map_err(|error| format!("failed to run {}: {error}", script.display()))?;
    if !output.status.success() {
        return Err(format!(
            "Gemma4 oracle build failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<_> = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    if lines.len() != 1 {
        return Err(format!(
            "Gemma4 oracle builder must print exactly one path, got {} non-empty lines",
            lines.len()
        ));
    }
    let path = PathBuf::from(lines[0]);
    if !path.is_file() {
        return Err(format!(
            "Gemma4 oracle builder printed a non-file: {}",
            path.display()
        ));
    }
    Ok(path)
}

#[test]
#[ignore = "requires Gemma4 GGUFs and pinned llama.cpp trace binary"]
fn gemma4_matches_pinned_cpu_oracle() {
    let _guard = GEMMA4_TRACE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let model = gemma4_model_path();
    let mmproj = gemma4_mmproj_path();
    let oracle = preflight_gemma4_ggufs(&model, &mmproj)
        .and_then(|()| ensure_gemma4_oracle())
        .unwrap();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("rmi-gemma4-parity-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let image = root.join("fixture.png");
    let audio = root.join("fixture.wav");
    write_image_fixture(&image).unwrap();
    write_audio_fixture(&audio).unwrap();

    let mut failures = Vec::new();
    for case in PARITY_CASES {
        if let Err(error) = run_parity_case(case, &root, &oracle, &model, &mmproj, &image, &audio) {
            failures.push(error);
        }
    }
    if !failures.is_empty() {
        panic!(
            "Gemma4 strict pinned-Oracle parity failed\n{}\nartifacts retained in {}",
            failures.join("\n\n"),
            root.display()
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "requires the Gemma4 model"]
fn gemma4_text_smoke() {
    let _guard = GEMMA4_TRACE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let model = gemma4_model_path();
    require_gemma4_gguf(
        &model,
        GEMMA4_MODEL_NAME,
        "gemma4",
        "token_embd.weight",
        GGMLType::Q8_0,
    )
    .unwrap();
    run_gemma4(Gemma4Request {
        model: &model,
        mmproj: None,
        image: None,
        audio: None,
        prompt: "hello",
        max_tokens: 1,
        threads: 4,
        kv_format: KvFormat::F32,
    })
    .unwrap();
}

#[test]
#[ignore = "requires the Gemma4 mmproj"]
fn gemma4_image_smoke() {
    let _guard = GEMMA4_TRACE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mmproj = gemma4_mmproj_path();
    require_gemma4_gguf(
        &mmproj,
        GEMMA4_MMPROJ_NAME,
        "clip",
        "v.patch_embd.weight",
        GGMLType::F16,
    )
    .unwrap();
    let loader = GGUFLoader::from_file(&mmproj).unwrap();
    let model = Gemma4VisionModel::from_source(&loader, 4).unwrap();
    let fixture =
        std::env::temp_dir().join(format!("rmi-gemma4-image-smoke-{}.png", std::process::id()));
    write_image_fixture(&fixture).unwrap();
    let projected = model.encode_path(&fixture).unwrap();
    let _ = std::fs::remove_file(&fixture);
    assert!(!projected.is_empty());
    assert_eq!(projected.len() % 1536, 0);
    assert!(projected.iter().all(|value| value.is_finite()));
    #[cfg(feature = "parity-trace")]
    if let Some(trace) = std::env::var_os("RMI_PARITY_TRACE") {
        let records: Vec<serde_json::Value> = std::fs::read_to_string(trace)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let shape = |name: &str| {
            records
                .iter()
                .find(|record| record["name"] == name)
                .unwrap_or_else(|| panic!("missing {name} trace checkpoint"))["shape"]
                .clone()
        };
        assert_eq!(
            shape("gemma4.vision.preprocessed"),
            serde_json::json!([384, 288, 3, 1])
        );
        assert_eq!(
            shape("gemma4.vision.projected"),
            serde_json::json!([1536, projected.len() / 1536, 1, 1])
        );
    }
}

#[test]
#[ignore = "requires the Gemma4 mmproj"]
fn gemma4_audio_smoke() {
    let _guard = GEMMA4_TRACE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mmproj = gemma4_mmproj_path();
    require_gemma4_gguf(
        &mmproj,
        GEMMA4_MMPROJ_NAME,
        "clip",
        "a.pre_encode.out.weight",
        GGMLType::F16,
    )
    .unwrap();
    let loader = GGUFLoader::from_file(&mmproj).unwrap();
    let model = Gemma4AudioModel::from_source(&loader, 4).unwrap();
    let fixture =
        std::env::temp_dir().join(format!("rmi-gemma4-audio-smoke-{}.wav", std::process::id()));
    write_audio_fixture(&fixture).unwrap();
    let projected = model.encode_wav_path(&fixture).unwrap();
    let _ = std::fs::remove_file(&fixture);
    assert!(!projected.is_empty());
    assert_eq!(projected.len() % 1536, 0);
    assert!(projected.iter().all(|value| value.is_finite()));
    #[cfg(feature = "parity-trace")]
    if let Some(trace) = std::env::var_os("RMI_PARITY_TRACE") {
        let records: Vec<serde_json::Value> = std::fs::read_to_string(trace)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let shape = |name: &str| {
            records
                .iter()
                .find(|record| record["name"] == name)
                .unwrap_or_else(|| panic!("missing {name} trace checkpoint"))["shape"]
                .clone()
        };
        assert_eq!(shape("gemma4.audio.mel"), serde_json::json!([128, 1]));
        assert_eq!(
            shape("gemma4.audio.projected"),
            serde_json::json!([1536, projected.len() / 1536, 1, 1])
        );
    }
}

#[test]
#[ignore = "requires the Gemma4 model and mmproj"]
fn gemma4_image_audio_smoke() {
    let _guard = GEMMA4_TRACE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let model = gemma4_model_path();
    let mmproj = gemma4_mmproj_path();
    require_gemma4_gguf(
        &model,
        GEMMA4_MODEL_NAME,
        "gemma4",
        "token_embd.weight",
        GGMLType::Q8_0,
    )
    .unwrap();
    require_gemma4_gguf(
        &mmproj,
        GEMMA4_MMPROJ_NAME,
        "clip",
        "v.patch_embd.weight",
        GGMLType::F16,
    )
    .unwrap();

    let suffix = std::process::id();
    let image_path = std::env::temp_dir().join(format!("rmi-gemma4-turn-{suffix}.png"));
    write_image_fixture(&image_path).unwrap();

    let audio_path = std::env::temp_dir().join(format!("rmi-gemma4-turn-{suffix}.wav"));
    write_audio_fixture(&audio_path).unwrap();

    let result = run_gemma4(Gemma4Request {
        model: &model,
        mmproj: Some(&mmproj),
        image: Some(&image_path),
        audio: Some(&audio_path),
        prompt: "describe",
        max_tokens: 1,
        threads: 4,
        kv_format: KvFormat::F32,
    });
    let _ = std::fs::remove_file(image_path);
    let _ = std::fs::remove_file(audio_path);
    result.unwrap();
}

#[test]
fn trace_comparator_reports_first_f32_word_mismatch() {
    let rust = record("gemma4.logits", 0, &[1, 2], &[1.0, 2.0]);
    let oracle = record(
        "gemma4.logits",
        0,
        &[1, 2],
        &[1.0, f32::from_bits(0x4000_0001)],
    );
    let error = assert_trace_equal("text", &[rust], &[oracle]).unwrap_err();
    assert!(error.contains("gemma4.logits"));
    assert!(error.contains("occurrence 0"));
    assert!(error.contains("index 1"));
    assert!(error.contains("0x40000000"));
    assert!(error.contains("0x40000001"));
}

#[test]
fn trace_comparator_reports_first_token_mismatch() {
    let rust = token_record("gemma4.generated_ids", 0, vec![7, 20, 30]);
    let oracle = token_record("gemma4.generated_ids", 0, vec![7, 21, 99]);
    let error = assert_trace_equal("audio", &[rust], &[oracle]).unwrap_err();
    assert!(error.contains("case audio"));
    assert!(error.contains("gemma4.generated_ids"));
    assert!(error.contains("occurrence 0"));
    assert!(error.contains("index 1"));
    assert!(error.contains("0x00000014"));
    assert!(error.contains("0x00000015"));
    assert!(!error.contains("0x0000001e"));
}

#[test]
fn trace_comparator_checks_shape_before_f32_words() {
    let rust = record("gemma4.final.norm", 3, &[1, 2], &[1.0, 2.0]);
    let oracle = record("gemma4.final.norm", 3, &[2, 1], &[1.0, 2.0]);
    let error = assert_trace_equal("image", &[rust], &[oracle]).unwrap_err();
    assert!(error.contains("shape"));
    assert!(error.contains("occurrence 3"));
    assert!(error.contains("index 0"));
}

#[test]
fn trace_comparator_checks_observed_record_order() {
    let first = record("gemma4.layer.0.attn_out", 0, &[1, 1], &[1.0]);
    let second = record("gemma4.layer.0.ffn_out", 0, &[1, 1], &[2.0]);
    let error =
        assert_trace_equal("text", &[first.clone(), second.clone()], &[second, first]).unwrap_err();
    assert!(error.contains("record order mismatch"));
    assert!(error.contains("index 0"));
}
