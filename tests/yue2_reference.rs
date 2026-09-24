use std::collections::HashMap;
use std::io::{BufRead, BufReader, Seek};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rust_model_inference::models::yue2::{
    song_chunks, SamplingConfig, YuE2Model, YuE2NarSession, YuE2Protocol, YuE2Request, YuE2Vae,
};

#[test]
#[ignore = "requires fixed YuE2 VAE GGUF"]
fn yue2_vae_real_decoder_loads_all_f32_tensors() {
    let loader = GGUFLoader::from_file(required_path("YUE2_VAE_GGUF")).unwrap();
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let _vae = YuE2Vae::from_source(source).unwrap();
}

#[test]
#[cfg(feature = "parity-trace")]
#[ignore = "requires fixed YuE2 VAE GGUF and pinned wheel"]
fn yue2_vae_full_and_tiled_decode_match_oracle_bits() {
    let _lock = YUE2_TRACE_LOCK.lock().unwrap();
    let gguf = required_path("YUE2_VAE_GGUF");
    let wheel = required_path("YUE2_ORACLE_WHEEL");
    let vae_dir = required_path("YUE2_ORACLE_VAE_DIR");
    let root = std::env::var_os("YUE2_TRACE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("rmi-yue2-vae-{}", std::process::id()))
        });
    std::fs::create_dir_all(&root).unwrap();
    let frames = std::env::var("YUE2_VAE_FRAMES")
        .ok()
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(33);
    let oracle_path = root.join(format!("oracle-{frames}.jsonl"));
    let rust_path = root.join(format!("rust-{frames}.jsonl"));
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/yue2_reference/trace_yue2.py");
    let result = Command::new("uv")
        .args(["run", "--no-project", "--with"])
        .arg(&wheel)
        .arg("python")
        .arg(script)
        .args([
            "--wheel",
            wheel.to_str().unwrap(),
            "--model-dir",
            vae_dir.to_str().unwrap(),
            "--vae-dir",
            vae_dir.to_str().unwrap(),
            "--phase",
            "vae",
            "--out",
            oracle_path.to_str().unwrap(),
            "--latent-frames",
            &frames.to_string(),
            "--core",
            "16",
            "--halo",
            "16",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );

    std::env::set_var("RMI_PARITY_TRACE", &rust_path);
    let loader = GGUFLoader::from_file(gguf).unwrap();
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let vae = YuE2Vae::from_source(source).unwrap();
    let latents = (0..frames * 64)
        .map(|index| (index % 11) as f32 * 0.01)
        .collect::<Vec<_>>();
    let full = vae.decode(&latents, frames).unwrap();
    rust_model_inference::parity_trace::report(rust_model_inference::parity_trace::checkpoint_at(
        "yue2.vae.full",
        None,
        None,
        &[1, 2, full.len() / 2],
        &full,
    ));
    let tiled = vae.decode_tiled(&latents, frames, 16, 16).unwrap();
    rust_model_inference::parity_trace::report(rust_model_inference::parity_trace::checkpoint_at(
        "yue2.vae.tiled",
        None,
        None,
        &[1, 2, tiled.len() / 2],
        &tiled,
    ));
    std::env::remove_var("RMI_PARITY_TRACE");
    assert_eq!(full.len(), 2 * (1920 * frames - 64));
    assert_trace_equal(
        &trace_records(&rust_path).unwrap(),
        &trace_records(&oracle_path).unwrap(),
    )
    .unwrap();
}

#[test]
#[cfg(feature = "parity-trace")]
#[ignore = "requires fixed YuE2 VAE GGUF and pinned real E2E Oracle"]
fn yue2_vae_saved_200_frame_decode_matches_oracle_bits() {
    let _lock = YUE2_TRACE_LOCK.lock().unwrap();
    let oracle_path = required_path("YUE2_E2E_ORACLE_TRACE");
    let root = std::env::var_os("YUE2_TRACE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("rmi-yue2-vae-saved-200"));
    std::fs::create_dir_all(&root).unwrap();
    let rust_path = root.join("rust.jsonl");
    let bytes =
        std::fs::read(oracle_path.parent().unwrap().join("yue2.nar.latents.0.f32")).unwrap();
    let latents = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(latents.len(), 200 * 64);
    let loader = GGUFLoader::from_file(required_path("YUE2_VAE_GGUF")).unwrap();
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let vae = YuE2Vae::from_source(source).unwrap();
    std::env::set_var("RMI_PARITY_TRACE", &rust_path);
    let waveform = vae.decode(&latents, 200).unwrap();
    std::env::remove_var("RMI_PARITY_TRACE");
    assert_eq!(waveform.len(), 2 * (200 * 1920 - 64));

    let mut oracle = Vec::new();
    let mut occurrences = HashMap::new();
    for line in BufReader::new(std::fs::File::open(&oracle_path).unwrap()).lines() {
        let line = line.unwrap();
        if line.starts_with("{\"name\":\"yue2.vae.") {
            oracle.push(trace_record(&oracle_path, &line, oracle.len(), &mut occurrences).unwrap());
        }
    }
    assert!(!oracle.is_empty());
    assert_trace_equal(&trace_records(&rust_path).unwrap(), &oracle).unwrap();
}
use rust_model_inference::{BPETokenizer, ComputePool, EncodeOptions, GGUFLoader, TensorSource};
use serde_json::Value;

static YUE2_TRACE_LOCK: Mutex<()> = Mutex::new(());
static YUE2_CASE: AtomicUsize = AtomicUsize::new(0);

struct RealCliResult {
    bytes: Vec<u8>,
    channels: u16,
    sample_rate: u32,
    frames: usize,
    samples: Vec<f32>,
    latent_frames: usize,
    trace: PathBuf,
    oracle_trace: PathBuf,
}

#[derive(Debug, PartialEq)]
enum TraceValues {
    F32(Vec<u32>),
    Tokens(Vec<u32>),
}

#[derive(Debug, PartialEq)]
struct TraceRecord {
    name: String,
    layer: Option<usize>,
    step: Option<usize>,
    occurrence: usize,
    shape: Vec<usize>,
    values: TraceValues,
}

fn required_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("missing {name}"))
}

fn decoded_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn run_tokenizer_oracle(wheel: &Path, model_dir: &Path, text: &str, out: &Path) {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/yue2_reference/trace_yue2.py");
    let output = Command::new("uv")
        .args(["run", "--no-project", "--with", "tiktoken", "python"])
        .arg(script)
        .args(["--wheel", wheel.to_str().unwrap()])
        .args(["--model-dir", model_dir.to_str().unwrap()])
        .args(["--phase", "tokenizer"])
        .args(["--out", out.to_str().unwrap()])
        .args(["--style", text])
        .args(["--lyrics", ""])
        .args(["--seed", "831001"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn run_ar_oracle(
    wheel: &Path,
    model_dir: &Path,
    style: &str,
    lyrics: &str,
    seed: u64,
    abc_tokens: usize,
    semantic_tokens: usize,
    out: &Path,
) {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/yue2_reference/trace_yue2.py");
    let output = Command::new("uv")
        .args(["run", "--no-project", "--with"])
        .arg(wheel)
        .arg("python")
        .arg(script)
        .args(["--wheel", wheel.to_str().unwrap()])
        .args(["--model-dir", model_dir.to_str().unwrap()])
        .args(["--phase", "ar"])
        .args(["--out", out.to_str().unwrap()])
        .args(["--style", style])
        .args(["--lyrics", lyrics])
        .args(["--seed", &seed.to_string()])
        .args(["--max-abc-tokens", &abc_tokens.to_string()])
        .args(["--max-music-tokens", &semantic_tokens.to_string()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn run_nar_oracle(
    wheel: &Path,
    model_dir: &Path,
    seed: u64,
    frames: usize,
    steps: usize,
    context: usize,
    out: &Path,
    prefill_only: bool,
) {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/yue2_reference/trace_yue2.py");
    let mut command = Command::new("uv");
    command
        .args(["run", "--no-project", "--with"])
        .arg(wheel)
        .arg("python")
        .arg(script)
        .args(["--wheel", wheel.to_str().unwrap()])
        .args(["--model-dir", model_dir.to_str().unwrap()])
        .args(["--phase", "nar"])
        .args(["--out", out.to_str().unwrap()])
        .args(["--seed", &seed.to_string()])
        .args(["--latent-frames", &frames.to_string()])
        .args(["--steps", &steps.to_string()])
        .args(["--context", &context.to_string()]);
    if prefill_only {
        command.arg("--prefill-only");
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn trace_records(path: &Path) -> Result<Vec<TraceRecord>, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let mut occurrences = HashMap::<String, usize>::new();
    contents
        .lines()
        .enumerate()
        .map(|(record_index, line)| trace_record(path, line, record_index, &mut occurrences))
        .collect()
}

fn trace_record(
    path: &Path,
    line: &str,
    record_index: usize,
    occurrences: &mut HashMap<String, usize>,
) -> Result<TraceRecord, String> {
    let value: Value = serde_json::from_str(line).map_err(|error| {
        format!(
            "invalid trace JSON {} record {record_index}: {error}",
            path.display()
        )
    })?;
    let name = value["name"]
        .as_str()
        .ok_or_else(|| format!("{} record {record_index} has no name", path.display()))?
        .to_owned();
    let expected_occurrence = *occurrences.get(&name).unwrap_or(&0);
    let occurrence = match value.get("occurrence") {
        Some(value) => value
            .as_u64()
            .map(|value| value as usize)
            .ok_or_else(|| format!("invalid occurrence for {name}"))?,
        None => expected_occurrence,
    };
    if occurrence != expected_occurrence {
        return Err(format!(
            "{} checkpoint {name} occurrence {occurrence}, expected {expected_occurrence}",
            path.display()
        ));
    }
    occurrences.insert(name.clone(), occurrence + 1);
    let optional_usize = |field: &str| -> Result<Option<usize>, String> {
        value
            .get(field)
            .filter(|value| !value.is_null())
            .map(|value| {
                value
                    .as_u64()
                    .map(|value| value as usize)
                    .ok_or_else(|| format!("invalid {field} for {name}"))
            })
            .transpose()
    };
    let shape = value["shape"]
        .as_array()
        .ok_or_else(|| format!("missing shape for {name}"))?
        .iter()
        .map(|dimension| {
            dimension
                .as_u64()
                .map(|dimension| dimension as usize)
                .ok_or_else(|| format!("invalid shape for {name}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let len = shape.iter().try_fold(1usize, |length, &dimension| {
        length
            .checked_mul(dimension)
            .ok_or_else(|| format!("shape overflow for {name}"))
    })?;
    if let Some(declared) = value.get("len") {
        if declared.as_u64() != Some(len as u64) {
            return Err(format!("declared count mismatch for {name}"));
        }
    }
    let values = if let Some(tokens) = value.get("token_ids") {
        let tokens = tokens
            .as_array()
            .ok_or_else(|| format!("invalid token IDs for {name}"))?
            .iter()
            .map(|token| {
                token
                    .as_u64()
                    .and_then(|token| u32::try_from(token).ok())
                    .ok_or_else(|| format!("invalid token ID for {name}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if tokens.len() != len {
            return Err(format!("token length mismatch for {name}"));
        }
        TraceValues::Tokens(tokens)
    } else {
        let sidecar = value
            .get("binary_path")
            .or_else(|| value.get("path"))
            .and_then(Value::as_str)
            .ok_or_else(|| format!("missing sidecar for {name}"))?;
        let sidecar = PathBuf::from(sidecar);
        let sidecar = if sidecar.is_absolute() {
            sidecar
        } else {
            path.parent().unwrap().join(sidecar)
        };
        let bytes = std::fs::read(&sidecar)
            .map_err(|error| format!("failed to read {}: {error}", sidecar.display()))?;
        if bytes.len() != len * 4 {
            return Err(format!(
                "{} has {} bytes, expected {}",
                sidecar.display(),
                bytes.len(),
                len * 4
            ));
        }
        TraceValues::F32(
            bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect(),
        )
    };
    let layer = optional_usize("layer")?;
    let step = optional_usize("step")?;
    Ok(TraceRecord {
        name,
        layer,
        step,
        occurrence,
        shape,
        values,
    })
}

#[test]
fn trace_record_rejects_declared_count_mismatch() {
    let line = r#"{"name":"tokens","shape":[1],"len":2,"token_ids":[7]}"#;
    let error = trace_record(Path::new("unused.jsonl"), line, 0, &mut HashMap::new()).unwrap_err();
    assert!(error.contains("declared count mismatch"), "{error}");
}

fn assert_trace_equal(rust: &[TraceRecord], oracle: &[TraceRecord]) -> Result<(), String> {
    for index in 0..rust.len().max(oracle.len()) {
        let rust_record = rust.get(index).ok_or_else(|| {
            format!(
                "Rust trace ended before Oracle record {index}: {:?}",
                oracle[index]
            )
        })?;
        let oracle_record = oracle.get(index).ok_or_else(|| {
            format!("Oracle trace ended before Rust record {index}: {rust_record:?}")
        })?;
        assert_trace_record_equal(index, rust_record, oracle_record)?;
    }
    Ok(())
}

fn assert_trace_record_equal(
    index: usize,
    rust_record: &TraceRecord,
    oracle_record: &TraceRecord,
) -> Result<(), String> {
    if (
        rust_record.name.as_str(),
        rust_record.layer,
        rust_record.step,
        rust_record.occurrence,
        &rust_record.shape,
    ) != (
        oracle_record.name.as_str(),
        oracle_record.layer,
        oracle_record.step,
        oracle_record.occurrence,
        &oracle_record.shape,
    ) {
        return Err(format!(
            "record {index} metadata mismatch:\nRust: {rust_record:?}\nOracle: {oracle_record:?}"
        ));
    }
    match (&rust_record.values, &oracle_record.values) {
        (TraceValues::F32(rust_words), TraceValues::F32(oracle_words)) => {
            for (element, (&rust_bits, &oracle_bits)) in
                rust_words.iter().zip(oracle_words).enumerate()
            {
                if !f32::from_bits(rust_bits).is_finite() {
                    return Err(format!(
                        "record {index} checkpoint {} contains non-finite Rust value at element {element}",
                        rust_record.name
                    ));
                }
                if rust_bits != oracle_bits {
                    return Err(format!(
                            "record {index} checkpoint {} layer {:?} step {:?} occurrence {} element {element}: Rust=0x{rust_bits:08x} Oracle=0x{oracle_bits:08x}",
                            rust_record.name,
                            rust_record.layer,
                            rust_record.step,
                            rust_record.occurrence
                    ));
                }
            }
        }
        (TraceValues::Tokens(rust_tokens), TraceValues::Tokens(oracle_tokens)) => {
            if rust_tokens != oracle_tokens {
                return Err(format!(
                    "record {index} checkpoint {} tokens differ: Rust={rust_tokens:?} Oracle={oracle_tokens:?}",
                    rust_record.name
                ));
            }
        }
        _ => return Err(format!("record {index} value kind mismatch")),
    }
    Ok(())
}

fn compare_full_trace_bits(rust_trace: &Path, oracle_trace: &Path) {
    let mut rust_lines = BufReader::new(std::fs::File::open(rust_trace).unwrap()).lines();
    let mut oracle_lines = BufReader::new(std::fs::File::open(oracle_trace).unwrap()).lines();
    let mut rust_occurrences = HashMap::new();
    let mut oracle_occurrences = HashMap::new();
    for index in 0.. {
        let rust_line = rust_lines.next().transpose().unwrap();
        let oracle_line = oracle_lines.next().transpose().unwrap();
        match (rust_line, oracle_line) {
            (None, None) => break,
            (None, Some(_)) => panic!("Rust trace ended before Oracle record {index}"),
            (Some(_), None) => panic!("Oracle trace ended before Rust record {index}"),
            (Some(rust_line), Some(oracle_line)) => {
                let rust =
                    trace_record(rust_trace, &rust_line, index, &mut rust_occurrences).unwrap();
                let oracle =
                    trace_record(oracle_trace, &oracle_line, index, &mut oracle_occurrences)
                        .unwrap();
                assert_trace_record_equal(index, &rust, &oracle).unwrap();
            }
        }
    }
}

// Compare committed JSONL records as they arrive so the new run never retains
// a second full set of F32 sidecars alongside the pinned Oracle.
fn compare_live_trace_bits(
    rust_trace: &Path,
    oracle_trace: &Path,
    mut finished: impl FnMut() -> std::io::Result<bool>,
) -> Result<usize, String> {
    let mut oracle_lines =
        BufReader::new(std::fs::File::open(oracle_trace).map_err(|error| error.to_string())?)
            .lines();
    while !rust_trace.exists() {
        if finished().map_err(|error| error.to_string())? {
            return Err("Rust process ended without a trace".into());
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let mut rust_lines =
        BufReader::new(std::fs::File::open(rust_trace).map_err(|error| error.to_string())?);
    let mut rust_occurrences = HashMap::new();
    let mut oracle_occurrences = HashMap::new();
    let mut index = 0;
    loop {
        let offset = rust_lines
            .stream_position()
            .map_err(|error| error.to_string())?;
        let mut bytes = Vec::new();
        rust_lines
            .read_until(b'\n', &mut bytes)
            .map_err(|error| error.to_string())?;
        if !bytes.ends_with(b"\n") {
            rust_lines
                .seek(std::io::SeekFrom::Start(offset))
                .map_err(|error| error.to_string())?;
            if finished().map_err(|error| error.to_string())? {
                if !bytes.is_empty() {
                    return Err(format!("Rust trace ended with incomplete record {index}"));
                }
                if oracle_lines
                    .next()
                    .transpose()
                    .map_err(|error| error.to_string())?
                    .is_some()
                {
                    return Err(format!("Rust trace ended before Oracle record {index}"));
                }
                return Ok(index);
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }
        let rust_line = std::str::from_utf8(&bytes).map_err(|error| error.to_string())?;
        let oracle_line = oracle_lines
            .next()
            .transpose()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| format!("Oracle trace ended before Rust record {index}"))?;
        let rust = trace_record(rust_trace, rust_line, index, &mut rust_occurrences)?;
        let oracle = trace_record(oracle_trace, &oracle_line, index, &mut oracle_occurrences)?;
        assert_trace_record_equal(index, &rust, &oracle)?;
        let value: Value = serde_json::from_str(rust_line).map_err(|error| error.to_string())?;
        if let Some(sidecar) = value.get("binary_path").and_then(Value::as_str) {
            let sidecar = Path::new(sidecar);
            let prefix = format!("{}.", rust_trace.file_name().unwrap().to_string_lossy());
            let name = sidecar.file_name().unwrap().to_string_lossy();
            if sidecar.parent() != rust_trace.parent()
                || !name.starts_with(&prefix)
                || !name.ends_with(".f32")
            {
                return Err(format!(
                    "Rust sidecar outside trace scope: {}",
                    sidecar.display()
                ));
            }
            std::fs::remove_file(sidecar).map_err(|error| error.to_string())?;
        }
        index += 1;
    }
}

#[test]
#[cfg(feature = "parity-trace")]
fn live_trace_comparison_waits_for_full_lines_and_reclaims_rust_sidecars() {
    use std::io::Write;
    use std::sync::atomic::AtomicBool;

    let root = std::env::temp_dir().join(format!(
        "rmi-yue2-live-trace-{}-{}",
        std::process::id(),
        YUE2_CASE.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root).unwrap();
    let rust_trace = root.join("rust.jsonl");
    let oracle_trace = root.join("oracle.jsonl");
    let rust_sidecar = root.join("rust.jsonl.example.f32");
    let oracle_sidecar = root.join("oracle.jsonl.example.f32");
    std::fs::write(&rust_sidecar, 1.5_f32.to_le_bytes()).unwrap();
    std::fs::write(&oracle_sidecar, 1.5_f32.to_le_bytes()).unwrap();
    let rust_line = format!(
        "{{\"name\":\"example\",\"shape\":[1],\"binary_path\":\"{}\"}}\n",
        rust_sidecar.display()
    );
    let oracle_line = format!(
        "{{\"name\":\"example\",\"shape\":[1],\"binary_path\":\"{}\"}}\n",
        oracle_sidecar.display()
    );
    std::fs::write(&oracle_trace, oracle_line).unwrap();
    std::fs::write(&rust_trace, &rust_line[..rust_line.len() / 2]).unwrap();
    let finished = Arc::new(AtomicBool::new(false));
    let writer_finished = Arc::clone(&finished);
    let writer_trace = rust_trace.clone();
    let remaining = rust_line[rust_line.len() / 2..].to_owned();
    let writer = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::OpenOptions::new()
            .append(true)
            .open(writer_trace)
            .unwrap()
            .write_all(remaining.as_bytes())
            .unwrap();
        writer_finished.store(true, Ordering::Release);
    });
    assert_eq!(
        compare_live_trace_bits(&rust_trace, &oracle_trace, || {
            Ok(finished.load(Ordering::Acquire))
        })
        .unwrap(),
        1
    );
    writer.join().unwrap();
    assert!(!rust_sidecar.exists());
    assert!(oracle_sidecar.exists());
    std::fs::remove_dir_all(root).unwrap();
}

fn run_real_cli(output_name: &str, seed: u64, max_tokens: usize, steps: usize) -> RealCliResult {
    run_real_cli_impl(output_name, seed, max_tokens, steps, true)
}

fn run_repeat_cli(output_name: &str, seed: u64, max_tokens: usize, steps: usize) -> RealCliResult {
    run_real_cli_impl(output_name, seed, max_tokens, steps, false)
}

fn run_real_cli_impl(
    output_name: &str,
    seed: u64,
    max_tokens: usize,
    steps: usize,
    capture_trace: bool,
) -> RealCliResult {
    let model = required_path("YUE2_MODEL_GGUF");
    let vae = required_path("YUE2_VAE_GGUF");
    let wheel = required_path("YUE2_ORACLE_WHEEL");
    let model_dir = required_path("YUE2_ORACLE_MODEL_DIR");
    let vae_dir = required_path("YUE2_ORACLE_VAE_DIR");
    let root = std::env::var_os("YUE2_TRACE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(format!(
            "{}-{}-{}",
            Path::new(output_name)
                .file_stem()
                .unwrap()
                .to_string_lossy(),
            std::process::id(),
            YUE2_CASE.fetch_add(1, Ordering::Relaxed)
        ));
    std::fs::create_dir_all(&root).unwrap();
    let output_path = root.join(output_name);
    let trace = root.join("rust.jsonl");
    let reused_oracle = std::env::var_os("YUE2_E2E_ORACLE_TRACE").map(PathBuf::from);
    let oracle_trace = reused_oracle
        .clone()
        .unwrap_or_else(|| root.join("oracle.jsonl"));
    let style = "warm jazz, upright bass, brushed drums";
    let lyrics = "[Verse]\nTonight the city hums.";
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/yue2_reference/trace_yue2.py");

    if capture_trace && reused_oracle.is_none() {
        let oracle = Command::new("uv")
            .args(["run", "--no-project", "--with"])
            .arg(&wheel)
            .arg("python")
            .arg(script)
            .args(["--wheel", wheel.to_str().unwrap()])
            .args(["--model-dir", model_dir.to_str().unwrap()])
            .args(["--vae-dir", vae_dir.to_str().unwrap()])
            .args(["--phase", "e2e"])
            .args(["--out", oracle_trace.to_str().unwrap()])
            .args(["--style", style])
            .args(["--lyrics", lyrics])
            .args(["--seed", &seed.to_string()])
            .args(["--max-music-tokens", &max_tokens.to_string()])
            .args(["--steps", &steps.to_string()])
            .args(["--core", "1024", "--halo", "16"])
            .env("VECLIB_MAXIMUM_THREADS", "1")
            .output()
            .unwrap();
        assert!(
            oracle.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&oracle.stdout),
            String::from_utf8_lossy(&oracle.stderr)
        );
    }

    let threads = std::env::var("YUE2_E2E_THREADS").unwrap_or_else(|_| "1".into());
    let mut command = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"));
    command
        .args(["--yue2", "--model"])
        .arg(&model)
        .arg("--vae")
        .arg(&vae)
        .args(["--prompt", style, "--lyrics", lyrics, "--out"])
        .arg(&output_path)
        .args([
            "--seed",
            &seed.to_string(),
            "--max-tokens",
            &max_tokens.to_string(),
            "--steps",
            &steps.to_string(),
            "--threads",
            &threads,
        ]);
    if capture_trace {
        command.env("RMI_PARITY_TRACE", &trace);
    }
    let rust = if capture_trace {
        let stdout_path = root.join("rust.stdout");
        let stderr_path = root.join("rust.stderr");
        command.stdout(Stdio::from(std::fs::File::create(&stdout_path).unwrap()));
        command.stderr(Stdio::from(std::fs::File::create(&stderr_path).unwrap()));
        let mut child = command.spawn().unwrap();
        let compared = compare_live_trace_bits(&trace, &oracle_trace, || {
            child.try_wait().map(|status| status.is_some())
        });
        if compared.is_err() {
            let _ = child.kill();
        }
        let status = child.wait().unwrap();
        let stdout = std::fs::read(stdout_path).unwrap();
        let stderr = std::fs::read(stderr_path).unwrap();
        assert!(
            compared.is_ok(),
            "{}\n{}\n{}",
            compared.unwrap_err(),
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        );
        Output {
            status,
            stdout,
            stderr,
        }
    } else {
        command.output().unwrap()
    };
    assert!(
        rust.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&rust.stdout),
        String::from_utf8_lossy(&rust.stderr)
    );
    let stdout = String::from_utf8(rust.stdout).unwrap();
    let latent_frames = stdout
        .lines()
        .find_map(|line| {
            line.strip_prefix("YuE2 nar stage: ")?
                .strip_suffix(" latent frames")?
                .parse::<usize>()
                .ok()
        })
        .expect("YuE2 CLI did not report latent frames");

    let bytes = std::fs::read(&output_path).unwrap();
    assert_eq!(&bytes[..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert_eq!(&bytes[12..16], b"fmt ");
    assert_eq!(&bytes[36..40], b"data");
    let channels = u16::from_le_bytes(bytes[22..24].try_into().unwrap());
    let sample_rate = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
    let data_bytes = u32::from_le_bytes(bytes[40..44].try_into().unwrap()) as usize;
    assert_eq!(bytes.len(), 44 + data_bytes);
    let samples = bytes[44..]
        .chunks_exact(2)
        .map(|word| i16::from_le_bytes(word.try_into().unwrap()) as f32)
        .collect::<Vec<_>>();
    let frames = samples.len() / usize::from(channels);
    RealCliResult {
        bytes,
        channels,
        sample_rate,
        frames,
        samples,
        latent_frames,
        trace,
        oracle_trace,
    }
}

fn compare_ar_fixture(
    style: &str,
    lyrics: &str,
    seed: u64,
    abc_tokens: usize,
    semantic_tokens: usize,
) {
    let _lock = YUE2_TRACE_LOCK.lock().unwrap();
    let gguf = required_path("YUE2_MODEL_GGUF");
    let wheel = required_path("YUE2_ORACLE_WHEEL");
    let model_dir = required_path("YUE2_ORACLE_MODEL_DIR");
    let root = std::env::var_os("YUE2_TRACE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(format!(
            "rmi-yue2-ar-{}-{}",
            std::process::id(),
            YUE2_CASE.fetch_add(1, Ordering::Relaxed)
        ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let oracle_path = root.join("oracle.jsonl");
    let rust_path = root.join("rust.jsonl");
    run_ar_oracle(
        &wheel,
        &model_dir,
        style,
        lyrics,
        seed,
        abc_tokens,
        semantic_tokens,
        &oracle_path,
    );

    let loader = GGUFLoader::from_file(gguf).unwrap();
    let tokenizer =
        Arc::new(BPETokenizer::from_gguf_metadata(|key| loader.metadata(key).cloned()).unwrap());
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let protocol = YuE2Protocol::from_source(source.as_ref()).unwrap();
    let model = YuE2Model::from_source(
        Arc::clone(&source),
        Arc::clone(&tokenizer),
        Arc::new(ComputePool::new(1)),
    )
    .unwrap();
    let request = YuE2Request::new(style, lyrics, seed).unwrap();
    std::env::set_var("RMI_PARITY_TRACE", &rust_path);
    let abc_prefix = protocol.abc_prefix(&tokenizer, &request).unwrap();
    rust_model_inference::parity_trace::token_ids("yue2.abc.prefix_ids", &abc_prefix).unwrap();
    let abc_ids = model
        .generate_abc(
            &abc_prefix,
            SamplingConfig {
                temperature: 0.0,
                min_tokens: abc_tokens,
                max_tokens: abc_tokens,
                ..SamplingConfig::abc()
            },
            seed,
        )
        .unwrap();
    let semantic_prefix = protocol
        .semantic_prefix(&tokenizer, &request, &abc_ids)
        .unwrap();
    rust_model_inference::parity_trace::token_ids("yue2.semantic.prefix_ids", &semantic_prefix)
        .unwrap();
    model
        .generate_semantic(
            &semantic_prefix,
            SamplingConfig {
                temperature: 0.0,
                min_tokens: semantic_tokens,
                max_tokens: semantic_tokens,
                ..SamplingConfig::semantic()
            },
            seed,
        )
        .unwrap();
    std::env::remove_var("RMI_PARITY_TRACE");

    let rust = trace_records(&rust_path).unwrap();
    let oracle = trace_records(&oracle_path).unwrap();
    assert_trace_equal(&rust, &oracle).unwrap();
}

fn compare_nar_fixture(seed: u64, frames: usize, steps: usize, context: usize) -> usize {
    const TRACE_NAMES: &str = "yue2.nar.noise,yue2.nar.chunk_ranges,yue2.nar.prefix_k,yue2.nar.prefix_v,yue2.nar.time_embedding,yue2.nar.position_embedding,yue2.nar.input,yue2.nar.attn_norm,yue2.nar.q,yue2.nar.k,yue2.nar.v,yue2.nar.attn,yue2.nar.attn_output,yue2.nar.attn_residual,yue2.nar.ffn_norm,yue2.nar.ffn_gate,yue2.nar.ffn_up,yue2.nar.ffn_down,yue2.nar.ffn_residual,yue2.nar.final_norm,yue2.nar.velocity_first,yue2.nar.velocity_midpoint,yue2.nar.latents";
    let _lock = YUE2_TRACE_LOCK.lock().unwrap();
    let gguf = required_path("YUE2_MODEL_GGUF");
    let wheel = required_path("YUE2_ORACLE_WHEEL");
    let model_dir = required_path("YUE2_ORACLE_MODEL_DIR");
    let root = std::env::var_os("YUE2_TRACE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(format!(
            "rmi-yue2-nar-{}-{}",
            std::process::id(),
            YUE2_CASE.fetch_add(1, Ordering::Relaxed)
        ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let oracle_path = root.join("oracle.jsonl");
    let rust_path = root.join("rust.jsonl");
    run_nar_oracle(
        &wheel,
        &model_dir,
        seed,
        frames,
        steps,
        context,
        &oracle_path,
        false,
    );

    let loader = GGUFLoader::from_file(gguf).unwrap();
    let tokenizer =
        Arc::new(BPETokenizer::from_gguf_metadata(|key| loader.metadata(key).cloned()).unwrap());
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let model = YuE2Model::from_source(
        Arc::clone(&source),
        tokenizer,
        Arc::new(ComputePool::new(1)),
    )
    .unwrap();
    let prefix = [151_643, 151_851];
    let codec = (0..frames as u32).collect::<Vec<_>>();
    std::env::set_var("RMI_PARITY_TRACE", &rust_path);
    std::env::set_var("RMI_PARITY_FILTER", TRACE_NAMES);
    let chunks = song_chunks(&prefix, &codec, seed, context).unwrap();
    let chunk_count = chunks.len();
    for chunk in chunks {
        YuE2NarSession::new(&model, chunk)
            .unwrap()
            .solve(steps)
            .unwrap();
    }
    std::env::remove_var("RMI_PARITY_FILTER");
    std::env::remove_var("RMI_PARITY_TRACE");

    let rust = trace_records(&rust_path).unwrap();
    let oracle = trace_records(&oracle_path).unwrap();
    assert_trace_equal(&rust, &oracle).unwrap();
    chunk_count
}

#[test]
#[ignore = "requires fixed YuE2 GGUF and pinned Python wheel"]
fn yue2_tokenizer_matches_pinned_oracle() {
    let gguf = required_path("YUE2_MODEL_GGUF");
    let wheel = required_path("YUE2_ORACLE_WHEEL");
    let model_dir = required_path("YUE2_ORACLE_MODEL_DIR");
    let loader = GGUFLoader::from_file(gguf).unwrap();
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| loader.metadata(key).cloned()).unwrap();
    let root = std::env::temp_dir().join(format!("rmi-yue2-tokenizer-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    for (index, text) in [
        "ASCII",
        "中文",
        "e\u{301}",
        "  line one\nline two  ",
        "X:1\nK:C\nC D E F|G A B c|",
        "",
    ]
    .into_iter()
    .enumerate()
    {
        let out = root.join(format!("tokenizer-{index}.jsonl"));
        run_tokenizer_oracle(&wheel, &model_dir, text, &out);
        let line = std::fs::read_to_string(&out).unwrap();
        let record: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        let expected_ids: Vec<u32> = record["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect();
        let actual_ids = tokenizer.encode(text, EncodeOptions::default());
        assert_eq!(actual_ids, expected_ids, "{text:?}");
        assert_eq!(
            decoded_hex(&tokenizer.decode_bytes(&actual_ids, false)),
            record["decoded_hex"].as_str().unwrap(),
            "{text:?}"
        );
    }

    let _ = std::fs::remove_dir_all(root);
}

#[test]
#[ignore = "requires fixed YuE2 GGUF and pinned Python wheel"]
fn yue2_ar_prefill_kv_logits_and_greedy_tokens_match_oracle_bits() {
    for (style, lyrics) in [
        ("jazz, warm", "[Verse]\nhello"),
        ("古风，笛子", "[主歌]\n月落乌啼"),
    ] {
        compare_ar_fixture(style, lyrics, 831001, 4, 6);
    }
}

#[test]
#[ignore = "requires fixed YuE2 GGUF and pinned Python wheel"]
fn yue2_nar_noise_chunks_midpoint_and_latents_match_oracle_bits() {
    assert_eq!(compare_nar_fixture(831001, 1, 32, 24_576), 1);
    assert_eq!(compare_nar_fixture(831001, 2, 1, 7), 2);
}

#[test]
#[ignore = "requires fixed YuE2 GGUF and pinned Python wheel"]
fn yue2_nar_nine_token_prefix_matches_oracle_bits() {
    assert_eq!(compare_nar_fixture(831001, 6, 1, 24_576), 1);
}

#[test]
#[ignore = "requires fixed YuE2 GGUF and pinned Python wheel"]
fn yue2_nar_257_token_prefix_matches_oracle_bits() {
    let _lock = YUE2_TRACE_LOCK.lock().unwrap();
    let root = std::env::var_os("YUE2_TRACE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("rmi-yue2-nar-257"));
    std::fs::create_dir_all(&root).unwrap();
    let oracle_path = root.join("oracle.jsonl");
    let rust_path = root.join("rust.jsonl");
    run_nar_oracle(
        &required_path("YUE2_ORACLE_WHEEL"),
        &required_path("YUE2_ORACLE_MODEL_DIR"),
        831001,
        254,
        1,
        24_576,
        &oracle_path,
        true,
    );

    let loader = GGUFLoader::from_file(required_path("YUE2_MODEL_GGUF")).unwrap();
    let tokenizer =
        Arc::new(BPETokenizer::from_gguf_metadata(|key| loader.metadata(key).cloned()).unwrap());
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let model = YuE2Model::from_source(source, tokenizer, Arc::new(ComputePool::new(12))).unwrap();
    std::env::set_var("RMI_PARITY_TRACE", &rust_path);
    let chunk = song_chunks(
        &[151_643, 151_851],
        &(0..254).collect::<Vec<_>>(),
        831001,
        24_576,
    )
    .unwrap()
    .remove(0);
    assert_eq!(chunk.ar_tokens.len(), 257);
    YuE2NarSession::new(&model, chunk).unwrap();
    std::env::remove_var("RMI_PARITY_TRACE");
    assert_trace_equal(
        &trace_records(&rust_path).unwrap(),
        &trace_records(&oracle_path).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "requires the pinned real E2E Oracle and fixed YuE2 GGUF"]
fn yue2_nar_saved_1832_token_prefix_matches_oracle_bits() {
    let _lock = YUE2_TRACE_LOCK.lock().unwrap();
    let oracle_path = required_path("YUE2_E2E_ORACLE_TRACE");
    let root = std::env::var_os("YUE2_TRACE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("rmi-yue2-nar-1832"));
    std::fs::create_dir_all(&root).unwrap();
    let rust_path = root.join("rust.jsonl");

    let mut abc_ids = None;
    let mut semantic_ids = None;
    for line in BufReader::new(std::fs::File::open(&oracle_path).unwrap()).lines() {
        let line = line.unwrap();
        if !line.contains("generated_ids") {
            continue;
        }
        let record: Value = serde_json::from_str(&line).unwrap();
        match record["name"].as_str() {
            Some("yue2.abc.generated_ids") => {
                abc_ids = Some(
                    record["token_ids"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u32)
                        .collect::<Vec<_>>(),
                )
            }
            Some("yue2.semantic.generated_ids") => {
                semantic_ids = Some(
                    record["token_ids"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u32)
                        .collect::<Vec<_>>(),
                )
            }
            _ => {}
        }
    }
    let abc_ids = abc_ids.expect("pinned Oracle has no ABC IDs");
    let semantic_ids = semantic_ids.expect("pinned Oracle has no semantic IDs");
    let loader = GGUFLoader::from_file(required_path("YUE2_MODEL_GGUF")).unwrap();
    let tokenizer =
        Arc::new(BPETokenizer::from_gguf_metadata(|key| loader.metadata(key).cloned()).unwrap());
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let protocol = YuE2Protocol::from_source(source.as_ref()).unwrap();
    let request = YuE2Request::new(
        "warm jazz, upright bass, brushed drums",
        "[Verse]\nTonight the city hums.",
        831001,
    )
    .unwrap();
    let prefix = protocol
        .semantic_prefix(&tokenizer, &request, &abc_ids)
        .unwrap();
    let codec = semantic_ids
        .iter()
        .map(|&token| token - 151_853)
        .collect::<Vec<_>>();
    let chunk = song_chunks(&prefix, &codec, 831001, 24_576)
        .unwrap()
        .remove(0);
    assert_eq!(chunk.ar_tokens.len(), 1832);
    let model = YuE2Model::from_source(source, tokenizer, Arc::new(ComputePool::new(12))).unwrap();
    std::env::set_var("RMI_PARITY_TRACE", &rust_path);
    std::env::set_var("RMI_PARITY_FILTER", "yue2.nar.prefix_k,yue2.nar.prefix_v");
    YuE2NarSession::new(&model, chunk).unwrap();
    std::env::remove_var("RMI_PARITY_FILTER");
    std::env::remove_var("RMI_PARITY_TRACE");

    let rust = trace_records(&rust_path).unwrap();
    let mut oracle = Vec::new();
    let mut occurrences = HashMap::new();
    for line in BufReader::new(std::fs::File::open(&oracle_path).unwrap()).lines() {
        let line = line.unwrap();
        if line.starts_with("{\"name\":\"yue2.nar.prefix_k\"")
            || line.starts_with("{\"name\":\"yue2.nar.prefix_v\"")
        {
            oracle.push(trace_record(&oracle_path, &line, oracle.len(), &mut occurrences).unwrap());
            if oracle.len() == 56 {
                break;
            }
        }
    }
    assert_trace_equal(&rust, &oracle).unwrap();
}

#[test]
#[cfg(feature = "parity-trace")]
#[ignore = "requires fixed YuE2 GGUF pair and pinned Python wheel"]
fn yue2_real_cli_matches_oracle_and_is_repeatable() {
    let _lock = YUE2_TRACE_LOCK.lock().unwrap();
    let first = run_real_cli("song-a.wav", 831001, 200, 2);
    assert_eq!(first.channels, 2);
    assert_eq!(first.sample_rate, 48_000);
    assert_eq!(first.frames, 1920 * first.latent_frames - 64);
    assert!(first.samples.iter().any(|sample| *sample != 0.0));
    let second = run_repeat_cli("song-b.wav", 831001, 200, 2);
    assert_eq!(first.bytes, second.bytes);
}
