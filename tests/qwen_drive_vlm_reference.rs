use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const LLAMA_PIN: &str = "b96806d96061049a5b574269b049bf6241d63d46";
const VISION_FILTER: &str = "omni.vision.patch_conv0,omni.vision.patch_conv1,omni.vision.patch_sum,omni.vision.patch_bias,omni.vision.inp_pos_emb,omni.vision.layer_out,omni.vision.projected";
const TEXT_FILTER: &str = "qwen35.prompt_ids,qwen35.mrope_positions,qwen35.layer_is_recurrent,qwen35.embedding,conv_output_raw-0,q_conv_predelta-0,k_conv_predelta-0,state_predelta-0,final_output-0,layer_output-0,attn_norm-3,Qcur_normed-3,Kcur_normed-3,Qcur-3,Kcur-3,layer_output-3,attn_norm-31,Qcur_normed-31,Kcur_normed-31,Qcur-31,Kcur-31,layer_output-31,result_norm,result_output,qwen35.greedy_token_ids";

#[derive(Deserialize)]
struct TraceContract {
    name: String,
    layer: Option<usize>,
    shape: Vec<usize>,
    occurrence: usize,
    oracle_sha256: String,
}

fn compare_words(name: &str, got: &[u32], expected: &[u32]) -> Result<(), String> {
    if got.len() != expected.len() {
        return Err(format!(
            "{name}: word count Rust={} Oracle={}",
            got.len(),
            expected.len()
        ));
    }
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        if got != expected {
            return Err(format!(
                "{name}: index {index} Rust=0x{got:08x} Oracle=0x{expected:08x}"
            ));
        }
    }
    Ok(())
}

fn required_path(name: &str) -> PathBuf {
    let path =
        PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("{name} is required")));
    assert!(path.exists(), "{} does not exist", path.display());
    path
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    path
}

fn command_output(command: &mut Command, label: &str) -> Output {
    let rendered = format!("{command:?}");
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to run {label} ({rendered}): {error}"));
    assert!(
        output.status.success(),
        "{label} failed ({rendered})\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn records(path: &Path) -> Result<Vec<Value>, String> {
    BufReader::new(File::open(path).map_err(|error| format!("{}: {error}", path.display()))?)
        .lines()
        .map(|line| {
            serde_json::from_str(&line.map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())
        })
        .collect()
}

fn sidecar(record: &Value) -> Result<PathBuf, String> {
    record["binary_path"]
        .as_str()
        .map(PathBuf::from)
        .ok_or_else(|| format!("record has no binary_path: {record}"))
}

fn shape(record: &Value) -> Result<Vec<usize>, String> {
    record["shape"]
        .as_array()
        .ok_or_else(|| format!("record has no shape: {record}"))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("invalid shape in {record}"))
        })
        .collect()
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut input = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = input
            .read(&mut buffer)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let mut digest = String::with_capacity(64);
    for byte in hasher.finalize() {
        write!(digest, "{byte:02x}").unwrap();
    }
    Ok(digest)
}

fn next_word(reader: &mut BufReader<File>, path: &Path) -> Result<Option<u32>, String> {
    let mut word = [0u8; 4];
    let first = reader
        .read(&mut word[..1])
        .map_err(|error| format!("{}: {error}", path.display()))?;
    if first == 0 {
        return Ok(None);
    }
    reader
        .read_exact(&mut word[1..])
        .map_err(|error| format!("{} has a partial F32 word: {error}", path.display()))?;
    Ok(Some(u32::from_le_bytes(word)))
}

fn compare_sidecars(name: &str, rust: &Path, oracle: &Path) -> Result<(), String> {
    let mut rust_reader =
        BufReader::new(File::open(rust).map_err(|error| format!("{}: {error}", rust.display()))?);
    let mut oracle_reader = BufReader::new(
        File::open(oracle).map_err(|error| format!("{}: {error}", oracle.display()))?,
    );
    let mut index = 0usize;
    loop {
        match (
            next_word(&mut rust_reader, rust)?,
            next_word(&mut oracle_reader, oracle)?,
        ) {
            (None, None) => return Ok(()),
            (Some(got), Some(expected)) if got == expected => index += 1,
            (Some(got), Some(expected)) => {
                return Err(format!(
                    "{name}: index {index} Rust=0x{got:08x} Oracle=0x{expected:08x}"
                ));
            }
            (got, expected) => {
                return Err(format!(
                    "{name}: word count differs at index {index}: Rust={} Oracle={}",
                    got.is_some(),
                    expected.is_some()
                ));
            }
        }
    }
}

fn compare_record_contract(
    index: usize,
    record: &Value,
    contract: &TraceContract,
) -> Result<(), String> {
    let actual_layer = record["layer"]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok());
    let actual_occurrence = record["occurrence"]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok());
    if record["name"].as_str() != Some(&contract.name)
        || actual_layer != contract.layer
        || shape(record)? != contract.shape
        || actual_occurrence != Some(contract.occurrence)
    {
        return Err(format!(
            "record {index} contract mismatch: actual={record}, expected name={} layer={:?} shape={:?} occurrence={}",
            contract.name, contract.layer, contract.shape, contract.occurrence
        ));
    }
    Ok(())
}

fn compare_vision_traces(rust: &Path, oracle: &Path) -> Result<(), String> {
    let rust = records(rust)?;
    let oracle = records(oracle)?;
    let contracts: Vec<TraceContract> =
        serde_json::from_str(include_str!("fixtures/qwen_drive/vlm-checkpoints.json"))
            .map_err(|error| error.to_string())?;
    if rust.len() != contracts.len() || oracle.len() != contracts.len() {
        return Err(format!(
            "vision record count Rust={} Oracle={} contract={}",
            rust.len(),
            oracle.len(),
            contracts.len()
        ));
    }
    for (index, ((got, expected), contract)) in rust.iter().zip(&oracle).zip(&contracts).enumerate()
    {
        compare_record_contract(index, got, contract)?;
        compare_record_contract(index, expected, contract)?;
        let oracle_sidecar = sidecar(expected)?;
        let digest = sha256_file(&oracle_sidecar)?;
        if digest != contract.oracle_sha256 {
            return Err(format!(
                "{} Oracle SHA256 {digest}, expected {}",
                contract.name, contract.oracle_sha256
            ));
        }
        compare_sidecars(&contract.name, &sidecar(got)?, &oracle_sidecar)?;
    }
    Ok(())
}

fn inline_values(record: &Value) -> Option<&Vec<Value>> {
    ["token_ids", "usize_values", "bool_values"]
        .into_iter()
        .find_map(|field| record[field].as_array())
}

fn compare_text_traces(rust: &Path, oracle: &Path) -> Result<(), String> {
    let rust = records(rust)?;
    let oracle = records(oracle)?;
    if rust.len() != oracle.len() {
        return Err(format!(
            "text record count Rust={} Oracle={}",
            rust.len(),
            oracle.len()
        ));
    }
    for (index, (got, expected)) in rust.iter().zip(&oracle).enumerate() {
        for field in ["name", "layer", "step", "occurrence", "shape", "len"] {
            if got.get(field) != expected.get(field) {
                return Err(format!(
                    "text record {index} field {field}: Rust={:?} Oracle={:?}",
                    got.get(field),
                    expected.get(field)
                ));
            }
        }
        if let (Some(got), Some(expected)) = (inline_values(got), inline_values(expected)) {
            if got != expected {
                return Err(format!("text record {index} inline values differ"));
            }
        } else {
            compare_sidecars(
                got["name"].as_str().unwrap_or("<unnamed>"),
                &sidecar(got)?,
                &sidecar(expected)?,
            )?;
        }
    }
    Ok(())
}

#[test]
fn strict_vlm_comparator_rejects_one_ulp() {
    let got = [1.0f32.to_bits()];
    let expected = [1.0f32.to_bits() + 1];
    let error = compare_words("qwen35.result_output", &got, &expected).unwrap_err();
    assert!(error.contains("index 0"));
    assert!(error.contains("0x3f800000"));
    assert!(error.contains("0x3f800001"));
}

fn git_head(path: &Path) -> String {
    let output = command_output(
        Command::new("git")
            .args(["-C"])
            .arg(path)
            .args(["rev-parse", "HEAD"]),
        "git rev-parse",
    );
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn build_oracles(llama: &Path, artifacts: &Path) -> (PathBuf, PathBuf) {
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/parity/build_qwen_drive_vlm_oracle.sh");
    let output = command_output(
        Command::new("sh").arg(script).arg(llama).arg(artifacts),
        "Qwen-Drive Oracle build",
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value = |prefix: &str| {
        PathBuf::from(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(prefix))
                .unwrap_or_else(|| panic!("Oracle builder did not print {prefix}")),
        )
    };
    let text = value("text=");
    let vision = value("vision=");
    assert!(text.is_file(), "{} is not an Oracle binary", text.display());
    assert!(
        vision.is_file(),
        "{} is not an Oracle binary",
        vision.display()
    );
    (text, vision)
}

fn run_rust_vision(mmproj: &Path, artifacts: &Path) -> PathBuf {
    use rust_model_inference::core::tensor::TensorSource;
    use rust_model_inference::models::qwen35::vision::{
        qwen_smart_resize, VisionEncoder, VisionScratchpad,
    };

    let trace = artifacts.join("rust-vision.jsonl");
    std::env::set_var("RMI_PARITY_TRACE", &trace);
    std::env::set_var("RMI_PARITY_FILTER", VISION_FILTER);
    let result = (|| {
        let source = rust_model_inference::GGUFLoader::from_file(mmproj)?;
        assert_eq!(
            source
                .metadata("general.architecture")
                .and_then(rust_model_inference::MetaValue::to_string_val),
            Some("clip")
        );
        let mut encoder = VisionEncoder::from_source(&source)?;
        encoder.precompute();
        let grid = qwen_smart_resize(256, 256, &encoder.config)?;
        assert_eq!((grid.image_width(), grid.image_height()), (256, 256));
        let pixels = vec![1.0f32; 256 * 256 * 3];
        let mut scratch = VisionScratchpad::new(&encoder.config);
        encoder.encode_image(&pixels, 256, 256, &mut scratch)?;
        rust_model_inference::parity_trace::report(rust_model_inference::parity_trace::checkpoint(
            "omni.vision.projected",
            None,
            &[grid.token_count(), encoder.config.projection_dim],
            &scratch.projected,
        ));
        Ok::<(), String>(())
    })();
    std::env::remove_var("RMI_PARITY_FILTER");
    std::env::remove_var("RMI_PARITY_TRACE");
    result.unwrap();
    trace
}

fn run_oracle_vision(oracle: &Path, model: &Path, mmproj: &Path, artifacts: &Path) -> PathBuf {
    let trace = artifacts.join("oracle-vision.jsonl");
    command_output(
        Command::new(oracle)
            .args(["-m"])
            .arg(model)
            .args(["--mmproj"])
            .arg(mmproj)
            .args([
                "-p",
                "encode",
                "-n",
                "256",
                "--image",
                "white",
                "--no-mmproj-offload",
                "-t",
                "1",
                "-fa",
                "off",
            ])
            .env("RMI_PARITY_TRACE", &trace),
        "llama.cpp Qwen-Drive vision",
    );
    trace
}

#[test]
#[ignore = "diagnostic Rust vision trace"]
fn qwen_drive_vlm_rust_vision_trace() {
    let artifacts = unique_temp_dir("rmi-qwen-drive-rust-vision");
    let trace = run_rust_vision(&required_path("RMI_QWEN_DRIVE_MMPROJ"), &artifacts);
    eprintln!("trace={}", trace.display());
}

fn check_tokenizer(model: &Path, hf: &Path) {
    use rust_model_inference::core::tensor::TensorSource;

    let source = rust_model_inference::GGUFLoader::from_file(model).unwrap();
    assert_eq!(
        source
            .metadata("general.architecture")
            .and_then(rust_model_inference::MetaValue::to_string_val),
        Some("qwen35")
    );
    let rust =
        rust_model_inference::BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
            .unwrap();
    let oracle = tokenizers::Tokenizer::from_file(hf.join("tokenizer.json")).unwrap();
    for text in [
        "",
        "你好",
        "e\u{301}",
        "  mixed \t whitespace\n\n",
        "<|im_start|>user\nPlan the trajectory.<|im_end|>\n<|im_start|>assistant\n",
        "<|vision_start|><|image_pad|><|vision_end|>",
    ] {
        let got = rust.encode(
            text,
            rust_model_inference::EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        );
        let expected = oracle.encode(text, false).unwrap();
        assert_eq!(got, expected.get_ids(), "{text:?}");
    }
}

fn run_rust_text(model: &Path, artifacts: &Path) -> PathBuf {
    let trace = artifacts.join("rust-text.jsonl");
    command_output(
        Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
            .args(["--model"])
            .arg(model)
            .args([
                "--prompt",
                "你好",
                "--max-tokens",
                "4",
                "--temp",
                "0",
                "--threads",
                "1",
                "--kv-cache",
                "f32",
            ])
            .env("RMI_PARITY_TRACE", &trace)
            .env("RMI_PARITY_FILTER", TEXT_FILTER),
        "Rust Qwen-Drive text",
    );
    trace
}

fn run_oracle_text(oracle: &Path, model: &Path, artifacts: &Path) -> PathBuf {
    let trace = artifacts.join("oracle-text.jsonl");
    let prompt = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    command_output(
        Command::new(oracle)
            .args(["-m"])
            .arg(model)
            .args([
                "-p",
                prompt,
                "-n",
                "4",
                "-c",
                "128",
                "-b",
                "128",
                "-ub",
                "128",
                "-t",
                "1",
                "-tb",
                "1",
                "-ngl",
                "0",
                "-ctk",
                "f32",
                "-ctv",
                "f32",
                "-fa",
                "off",
                "--temp",
                "0",
                "--top-k",
                "1",
                "--top-p",
                "1.0",
                "--repeat-penalty",
                "1.0",
            ])
            .env("RMI_PARITY_TRACE", &trace),
        "llama.cpp Qwen-Drive text",
    );
    trace
}

#[test]
#[ignore = "requires Qwen-Drive BF16 GGUF pair and pinned llama.cpp"]
fn qwen_drive_vlm_matches_llama_cpp_bitwise() {
    let model = required_path("RMI_QWEN_DRIVE_VLM");
    let mmproj = required_path("RMI_QWEN_DRIVE_MMPROJ");
    let hf = required_path("RMI_QWEN_DRIVE_HF");
    let llama = required_path("RMI_LLAMA_CPP");
    assert_eq!(git_head(&llama), LLAMA_PIN);
    check_tokenizer(&model, &hf);

    let artifacts = unique_temp_dir("rmi-qwen-drive-vlm");
    let (text_oracle, vision_oracle) = build_oracles(&llama, &artifacts);
    let rust_vision = run_rust_vision(&mmproj, &artifacts);
    let oracle_vision = run_oracle_vision(&vision_oracle, &model, &mmproj, &artifacts);
    if let Err(error) = compare_vision_traces(&rust_vision, &oracle_vision) {
        panic!("{error}\nartifacts retained in {}", artifacts.display());
    }
    let rust_text = run_rust_text(&model, &artifacts);
    let oracle_text = run_oracle_text(&text_oracle, &model, &artifacts);
    if let Err(error) = compare_text_traces(&rust_text, &oracle_text) {
        panic!("{error}\nartifacts retained in {}", artifacts.display());
    }
    std::fs::remove_dir_all(artifacts).unwrap();
}
