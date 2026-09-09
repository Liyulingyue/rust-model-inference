use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const LLAMA_PIN: &str = "b96806d96061049a5b574269b049bf6241d63d46";
const TRACE_FILTER: &str = "qwen35.prompt_ids,qwen35.mrope_positions,qwen35.layer_is_recurrent,qwen35.embedding,conv_output_raw-0,q_conv_predelta-0,k_conv_predelta-0,state_predelta-0,final_output-0,layer_output-0,attn_norm-3,Qcur_normed-3,Kcur_normed-3,Qcur-3,Kcur-3,layer_output-3,attn_norm-63,Qcur_normed-63,Kcur_normed-63,Qcur-63,Kcur-63,layer_output-63,result_norm,result_output,qwen35.greedy_token_ids";
const BITWISE: &[&str] = &["qwen35.embedding", "state_predelta-0"];
const LOSSY_BOUNDS: &[(&str, f32, f32)] = &[
    ("conv_output_raw-0", 5e-6, 1e-6),
    ("q_conv_predelta-0", 5e-7, 1e-6),
    ("k_conv_predelta-0", 5e-7, 1e-6),
    ("final_output-0", 1e-6, 1e-6),
    ("layer_output-0", 1e-3, 1e-4),
    ("attn_norm-3", 2.5e-2, 1e-4),
    ("Qcur_normed-3", 1.5e-1, 1e-4),
    ("Kcur_normed-3", 1.5e-1, 1e-4),
    ("Qcur-3", 1.5e-1, 1e-4),
    ("Kcur-3", 1.5e-1, 1e-4),
    ("layer_output-3", 7.5e-2, 1e-4),
    ("attn_norm-63", 1.25, 1e-4),
    ("Qcur_normed-63", 5e-1, 1e-4),
    ("Kcur_normed-63", 5e-1, 1e-4),
    ("Qcur-63", 5e-1, 1e-4),
    ("Kcur-63", 5e-1, 1e-4),
    ("layer_output-63", 2.5, 1e-4),
    ("result_norm", 5e-1, 1e-4),
    ("result_output", 5e-1, 1e-4),
];

fn required_path(name: &str) -> PathBuf {
    let path =
        PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("{name} is required")));
    assert!(path.exists(), "{} does not exist", path.display());
    path
}

fn git_head(path: &Path) -> String {
    let output = Command::new("git")
        .args(["-C"])
        .arg(path)
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git rev-parse failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
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

fn build_oracle(llama: &Path, artifacts: &Path) -> PathBuf {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/parity/build_qwen35_oracle.sh");
    assert!(script.exists(), "{} does not exist", script.display());
    let output = command_output(
        Command::new("sh").arg(script).arg(llama).arg(artifacts),
        "llama.cpp Oracle build",
    );
    let oracle = PathBuf::from(
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .last()
            .expect("Oracle builder produced no path"),
    );
    assert!(
        oracle.is_file(),
        "{} is not an Oracle binary",
        oracle.display()
    );
    oracle
}

fn run_rust(model: &Path, prompt: &str, max_tokens: usize, artifacts: &Path) -> PathBuf {
    let trace = artifacts.join("rust.jsonl");
    let source = rust_model_inference::GGUFLoader::from_file(model).unwrap();
    let config = rust_model_inference::models::qwen35::Qwen35Config::from_source(&source).unwrap();
    let filter = TRACE_FILTER.replace("63", &(config.n_layer_impl() - 1).to_string());
    command_output(
        Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
            .args(["--model"])
            .arg(model)
            .args([
                "--prompt",
                prompt,
                "--max-tokens",
                &max_tokens.to_string(),
                "--temp",
                "0",
                "--threads",
                "1",
                "--kv-cache",
                "f32",
            ])
            .env("RMI_PARITY_TRACE", &trace)
            .env("RMI_PARITY_FILTER", filter),
        "Rust Qwen3.5 inference",
    );
    trace
}

fn run_oracle(
    oracle: &Path,
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    artifacts: &Path,
) -> PathBuf {
    let trace = artifacts.join("llama.jsonl");
    let chat_prompt = format!(
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    command_output(
        Command::new(oracle)
            .args(["-m"])
            .arg(model)
            .args([
                "-p",
                &chat_prompt,
                "-n",
                &max_tokens.to_string(),
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
        "llama.cpp Qwen3.5 inference",
    );
    trace
}

fn records(path: &Path) -> Result<Vec<Value>, String> {
    std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?
        .lines()
        .map(|line| serde_json::from_str(line).map_err(|error| error.to_string()))
        .collect()
}

fn values<'a>(record: &'a Value) -> Option<&'a Vec<Value>> {
    ["token_ids", "usize_values", "bool_values"]
        .into_iter()
        .find_map(|field| record.get(field).and_then(Value::as_array))
}

fn shape(record: &Value) -> Result<Vec<usize>, String> {
    if let Some(shape) = record.get("shape").and_then(Value::as_array) {
        return shape
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| format!("invalid shape in {record}"))
            })
            .collect();
    }
    Ok(vec![values(record).map_or(0, Vec::len)])
}

fn len(record: &Value) -> Result<usize, String> {
    record
        .get("len")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .or_else(|| values(record).map(Vec::len))
        .ok_or_else(|| format!("record has no length: {record}"))
}

fn f32_words(record: &Value) -> Result<Vec<u32>, String> {
    let path = record
        .get("binary_path")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("record has no binary_path: {record}"))?;
    let bytes = std::fs::read(path).map_err(|error| format!("failed to read {path}: {error}"))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{path} has a partial F32 word"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect())
}

fn close(got: f32, expected: f32, abs_tol: f32, rel_tol: f32) -> bool {
    got.is_finite()
        && expected.is_finite()
        && (got - expected).abs() <= abs_tol + rel_tol * expected.abs()
}

fn compare_qwen35_traces(rust: &Path, llama: &Path, bitwise: bool) -> Result<(), String> {
    let rust = records(rust)?;
    let llama = records(llama)?;
    if rust.len() != llama.len() {
        return Err(format!(
            "record count Rust={} llama.cpp={}",
            rust.len(),
            llama.len()
        ));
    }
    for (record_index, (got, expected)) in rust.iter().zip(&llama).enumerate() {
        for field in ["name", "layer", "step", "occurrence"] {
            if got.get(field) != expected.get(field) {
                return Err(format!(
                    "record {record_index} field {field}: Rust={:?} llama.cpp={:?}",
                    got.get(field),
                    expected.get(field)
                ));
            }
        }
        if shape(got)? != shape(expected)? || len(got)? != len(expected)? {
            return Err(format!(
                "record {record_index} {} shape/length: Rust={:?}/{} llama.cpp={:?}/{}",
                got["name"],
                shape(got)?,
                len(got)?,
                shape(expected)?,
                len(expected)?
            ));
        }
        if let (Some(got), Some(expected)) = (values(got), values(expected)) {
            if got != expected {
                return Err(format!(
                    "record {record_index} exact values differ: Rust={got:?} llama.cpp={expected:?}"
                ));
            }
            continue;
        }

        let name = got["name"].as_str().unwrap();
        let got_words = f32_words(got)?;
        let expected_words = f32_words(expected)?;
        if got_words.len() != expected_words.len() {
            return Err(format!(
                "checkpoint {name} word count Rust={} llama.cpp={}",
                got_words.len(),
                expected_words.len()
            ));
        }
        let bounds = LOSSY_BOUNDS
            .iter()
            .find(|(checkpoint, _, _)| *checkpoint == name)
            .map(|(_, abs_tol, rel_tol)| (*abs_tol, *rel_tol));
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut first_mismatch = None;
        for (index, (&got_bits, &expected_bits)) in
            got_words.iter().zip(&expected_words).enumerate()
        {
            if bitwise || BITWISE.contains(&name) {
                if got_bits != expected_bits {
                    return Err(format!(
                        "checkpoint {name} index {index}: Rust=0x{got_bits:08x} llama.cpp=0x{expected_bits:08x}"
                    ));
                }
            } else {
                let (abs_tol, rel_tol) =
                    bounds.ok_or_else(|| format!("checkpoint {name} has no precision contract"))?;
                let got = f32::from_bits(got_bits);
                let expected = f32::from_bits(expected_bits);
                let abs = (got - expected).abs();
                let rel = if expected == 0.0 {
                    if abs == 0.0 {
                        0.0
                    } else {
                        f32::INFINITY
                    }
                } else {
                    abs / expected.abs()
                };
                max_abs = max_abs.max(abs);
                max_rel = max_rel.max(rel);
                if first_mismatch.is_none() && !close(got, expected, abs_tol, rel_tol) {
                    first_mismatch = Some((index, got, expected, abs_tol, rel_tol));
                }
            }
        }
        if let Some((index, got, expected, abs_tol, rel_tol)) = first_mismatch {
            return Err(format!(
                "checkpoint {name} index {index}: Rust={got} llama.cpp={expected} exceeds abs={abs_tol} rel={rel_tol}; record max_abs={max_abs} max_rel={max_rel}"
            ));
        }
    }
    Ok(())
}

#[test]
fn bitwise_comparison_rejects_one_ulp() {
    let artifacts = unique_temp_dir("rmi-bitwise-check");
    let mut traces = Vec::new();
    for (side, word) in [("rust", 1.0f32.to_bits()), ("llama", 1.0f32.to_bits() + 1)] {
        let binary = artifacts.join(format!("{side}.f32"));
        std::fs::write(&binary, word.to_le_bytes()).unwrap();
        let trace = artifacts.join(format!("{side}.jsonl"));
        std::fs::write(
            &trace,
            serde_json::json!({
                "name": "result_output", "shape": [1], "len": 1,
                "occurrence": 0, "binary_path": binary,
            })
            .to_string(),
        )
        .unwrap();
        traces.push(trace);
    }
    assert!(compare_qwen35_traces(&traces[0], &traces[0], true).is_ok());
    assert!(compare_qwen35_traces(&traces[0], &traces[1], true)
        .unwrap_err()
        .contains("0x3f800000 llama.cpp=0x3f800001"));
    std::fs::remove_dir_all(artifacts).unwrap();
}

#[test]
#[ignore = "requires Qwen3.8 model and pinned llama.cpp"]
fn qwen38_matches_pinned_llama_cpp_at_lossless_checkpoints() {
    let model = required_path("RMI_QWEN35_MODEL");
    let llama = required_path("RMI_LLAMA_CPP");
    assert_eq!(git_head(&llama), LLAMA_PIN);
    let artifacts = unique_temp_dir("rmi-qwen38-parity");
    let oracle = build_oracle(&llama, &artifacts);
    let rust_trace = run_rust(&model, "你好", 1, &artifacts);
    let llama_trace = run_oracle(&oracle, &model, "你好", 1, &artifacts);
    if let Err(error) = compare_qwen35_traces(&rust_trace, &llama_trace, false) {
        panic!("{error}\nartifacts retained in {}", artifacts.display());
    }
    std::fs::remove_dir_all(artifacts).unwrap();
}

#[test]
#[ignore = "requires RMI_NEOHORSE_MODEL and RMI_NEOHORSE_HF"]
fn neohorse_tokenizer_matches_published_tokenizer() {
    let model = required_path("RMI_NEOHORSE_MODEL");
    let hf = required_path("RMI_NEOHORSE_HF");
    let reference = tokenizers::Tokenizer::from_file(hf.join("tokenizer.json")).unwrap();
    let source = rust_model_inference::GGUFLoader::from_file(model).unwrap();
    let tokenizer =
        rust_model_inference::BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
            .unwrap();
    assert_eq!(tokenizer.vocab_size(), 248320);
    assert_eq!(tokenizer.bos_id(), None);
    assert_eq!(tokenizer.eos_id(), Some(248046));
    let malformed = rust_model_inference::BPETokenizer::from_gguf_metadata(|key| {
        if key == "tokenizer.ggml.normalizer.nfc" {
            Some(rust_model_inference::MetaValue::String("true".into()))
        } else {
            source.metadata(key).cloned()
        }
    });
    assert!(malformed
        .unwrap_err()
        .contains("tokenizer.ggml.normalizer.nfc"));
    for text in [
        "",
        "你好",
        " hello   world\n\n\t1234567890",
        "e\u{301} résumé 日本語 🐎",
        "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ] {
        let actual = tokenizer.encode(
            text,
            rust_model_inference::EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        );
        assert_eq!(
            actual,
            reference.encode(text, false).unwrap().get_ids(),
            "{text:?}"
        );
        assert_eq!(
            tokenizer.decode(&actual, true),
            reference.decode(&actual, false).unwrap()
        );
    }
}

#[test]
#[ignore = "requires NeoHorse BF16 GGUF, pinned llama.cpp, and --features parity-trace"]
fn neohorse_matches_pinned_llama_cpp_bitwise() {
    let model = required_path("RMI_NEOHORSE_MODEL");
    let llama = required_path("RMI_LLAMA_CPP");
    assert_eq!(git_head(&llama), LLAMA_PIN);
    let source = rust_model_inference::GGUFLoader::from_file(&model).unwrap();
    let config = rust_model_inference::models::qwen35::Qwen35Config::from_source(&source).unwrap();
    assert_eq!((config.n_layer, config.n_nextn), (32, 0));
    assert_eq!((config.n_embd, config.vocab_size), (4096, 248320));
    assert_eq!((config.key_dim(), config.value_dim()), (2048, 4096));
    assert_eq!(config.is_recurrent.iter().filter(|&&r| !r).count(), 8);
    let artifacts = unique_temp_dir("rmi-neohorse-parity");
    let oracle = std::env::var_os("RMI_QWEN35_ORACLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| build_oracle(&llama, &artifacts));
    let rust_trace = run_rust(&model, "你好", 4, &artifacts);
    let llama_trace = run_oracle(&oracle, &model, "你好", 4, &artifacts);
    if let Err(error) = compare_qwen35_traces(&rust_trace, &llama_trace, true) {
        panic!("{error}\nartifacts retained in {}", artifacts.display());
    }
    std::fs::remove_dir_all(artifacts).unwrap();
}

#[test]
#[ignore = "requires Qwen3.8 model and mmproj"]
fn qwen38_mmproj_image_smoke() {
    let model = required_path("RMI_QWEN35_MODEL");
    let mmproj = required_path("RMI_QWEN35_MMPROJ");
    let directory = unique_temp_dir("rmi-qwen38-image");
    let image_path = directory.join("fixture.png");
    let image = image::RgbImage::from_fn(32, 32, |x, y| {
        image::Rgb([(x * 7) as u8, (y * 5) as u8, ((x + y) * 3) as u8])
    });
    image.save(&image_path).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .args(["--model"])
        .arg(model)
        .args(["--mmproj"])
        .arg(mmproj)
        .args(["--image"])
        .arg(&image_path)
        .args([
            "--prompt",
            "描述图片",
            "--max-tokens",
            "1",
            "--temp",
            "0",
            "--threads",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_dir_all(directory).unwrap();
}
