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
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/oracle/qwen35/build_qwen35_oracle.sh");
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
    use sha2::{Digest, Sha256};
    eprintln!(
        "Qwen3.5 Oracle pin={LLAMA_PIN} sha256={:x}",
        Sha256::digest(std::fs::read(&oracle).unwrap())
    );
    oracle
}

fn run_rust(
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    artifacts: &Path,
    batch: usize,
    jinja: bool,
) -> PathBuf {
    let trace = artifacts.join(format!("rust-batch-{batch}.jsonl"));
    let source = rust_model_inference::GGUFLoader::from_file(model).unwrap();
    let config = rust_model_inference::models::qwen35::Qwen35Config::from_source(&source).unwrap();
    let filter = TRACE_FILTER
        .replace("63", &(config.n_layer_impl() - 1).to_string())
        .replace("layer_output-", "l_out-");
    command_output(
        Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
            .args(jinja.then_some("--jinja"))
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
                "--prefill-batch-size",
                &batch.to_string(),
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
        .map(|line| {
            let mut record: Value =
                serde_json::from_str(line).map_err(|error| error.to_string())?;
            // Rust uses the current GGML name; the pinned Oracle exports the older label.
            if let Some(layer) = record["name"]
                .as_str()
                .and_then(|name| name.strip_prefix("l_out-"))
            {
                record["name"] = format!("layer_output-{layer}").into();
            }
            Ok(record)
        })
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

fn precision_bounds(name: &str, last_layer: Option<usize>) -> Option<(f32, f32)> {
    let contract_name = last_layer
        .and_then(|layer| name.strip_suffix(&format!("-{layer}")))
        .map(|prefix| format!("{prefix}-63"));
    LOSSY_BOUNDS
        .iter()
        .find(|(checkpoint, _, _)| *checkpoint == name)
        .or_else(|| {
            LOSSY_BOUNDS
                .iter()
                .find(|(checkpoint, _, _)| Some(*checkpoint) == contract_name.as_deref())
        })
        .map(|(_, abs_tol, rel_tol)| (*abs_tol, *rel_tol))
}

#[test]
fn precision_contract_follows_only_the_actual_final_layer() {
    assert_eq!(
        precision_bounds("attn_norm-23", Some(23)),
        Some((1.25, 1e-4))
    );
    assert_eq!(precision_bounds("attn_norm-22", Some(23)), None);
    assert_eq!(
        precision_bounds("attn_norm-3", Some(23)),
        Some((2.5e-2, 1e-4))
    );
    assert_eq!(
        precision_bounds("attn_norm-63", Some(63)),
        Some((1.25, 1e-4))
    );
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
    let last_layer = rust
        .iter()
        .find(|record| record["name"] == "qwen35.layer_is_recurrent")
        .and_then(|record| record["bool_values"].as_array())
        .and_then(|layers| layers.len().checked_sub(1));
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
        let bounds = precision_bounds(name, last_layer);
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
fn bitwise_comparison_normalizes_layer_alias_and_rejects_one_ulp() {
    let artifacts = unique_temp_dir("rmi-bitwise-check");
    let mut traces = Vec::new();
    for (side, word) in [("rust", 1.0f32.to_bits()), ("llama", 1.0f32.to_bits() + 1)] {
        let binary = artifacts.join(format!("{side}.f32"));
        std::fs::write(&binary, word.to_le_bytes()).unwrap();
        let trace = artifacts.join(format!("{side}.jsonl"));
        std::fs::write(
            &trace,
            serde_json::json!({
                "name": if side == "rust" { "l_out-3" } else { "layer_output-3" },
                "shape": [1], "len": 1,
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
    let scalar_trace = run_rust(&model, "你好", 4, &artifacts, 1, false);
    let rust_trace = run_rust(&model, "你好", 4, &artifacts, 64, false);
    compare_qwen35_traces(&scalar_trace, &rust_trace, true).unwrap_or_else(|error| {
        panic!(
            "Rust batch=1/64: {error}; artifacts {}",
            artifacts.display()
        )
    });
    let llama_trace = run_oracle(&oracle, &model, "你好", 4, &artifacts);
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
#[ignore = "requires NeoHorse 4B/9B GGUF, pinned llama.cpp, and --features parity-trace"]
fn neohorse_matches_pinned_llama_cpp_bitwise() {
    let model = required_path("RMI_NEOHORSE_MODEL");
    let llama = required_path("RMI_LLAMA_CPP");
    assert_eq!(git_head(&llama), LLAMA_PIN);
    let source = rust_model_inference::GGUFLoader::from_file(&model).unwrap();
    let config = rust_model_inference::models::qwen35::Qwen35Config::from_source(&source).unwrap();
    assert_eq!((config.n_layer, config.n_nextn), (32, 0));
    assert!(matches!(config.n_embd, 2560 | 4096));
    assert_eq!(config.vocab_size, 248320);
    assert_eq!((config.key_dim(), config.value_dim()), (2048, 4096));
    assert_eq!(config.is_recurrent.iter().filter(|&&r| !r).count(), 8);
    let artifacts = unique_temp_dir("rmi-neohorse-parity");
    let oracle = std::env::var_os("RMI_QWEN35_ORACLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| build_oracle(&llama, &artifacts));
    let rust_trace = run_rust(&model, "你好", 4, &artifacts, 64, false);
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

#[test]
#[ignore = "requires RMI_HOLO4_MODEL and RMI_HOLO4_ORACLE (pinned llama-tokenize beside it)"]
fn holo4_contract_and_tokenizer_match_llama_cpp() {
    use rust_model_inference::{BPETokenizer, EncodeOptions, GGMLType, GGUFLoader};
    let model = required_path("RMI_HOLO4_MODEL");
    let oracle = required_path("RMI_HOLO4_ORACLE");
    let source = GGUFLoader::from_file(&model).unwrap();
    assert_eq!(
        source
            .metadata("general.architecture")
            .unwrap()
            .to_string_val(),
        Some("qwen35")
    );
    let config = rust_model_inference::models::qwen35::Qwen35Config::from_source(&source).unwrap();
    assert_eq!(
        (config.n_layer, config.n_layer_impl(), config.n_nextn),
        (64, 64, 0)
    );
    assert_eq!(
        (config.n_embd, config.n_ff, config.vocab_size),
        (5120, 17408, 248320)
    );
    assert_eq!(
        (config.n_head, config.n_head_kv, config.key_length),
        (24, 4, 256)
    );
    assert_eq!(
        (
            config.ssm_d_conv,
            config.ssm_d_state,
            config.ssm_n_group,
            config.ssm_dt_rank,
            config.ssm_d_inner
        ),
        (4, 128, 16, 48, 6144)
    );
    assert_eq!(config.n_ctx, 262144);
    assert_eq!(config.rope_dimension_sections, [11, 11, 10, 0]);
    assert_eq!(
        config.is_recurrent,
        (0..64).map(|layer| layer % 4 != 3).collect::<Vec<_>>()
    );
    assert!(config.moe.is_none());
    assert_eq!(source.tensors().len(), 851);
    for (name, dims, kind) in [
        ("token_embd.weight", vec![5120, 248320], GGMLType::Q6K),
        ("output.weight", vec![5120, 248320], GGMLType::Q6K),
        ("blk.0.attn_qkv.weight", vec![5120, 10240], GGMLType::Q6K),
        ("blk.0.ffn_gate.weight", vec![5120, 17408], GGMLType::Q4K),
        ("blk.0.ssm_conv1d.weight", vec![4, 10240], GGMLType::F32),
        ("blk.63.attn_q.weight", vec![5120, 12288], GGMLType::Q4K),
    ] {
        let tensor = source.tensors().iter().find(|t| t.name == name).unwrap();
        assert_eq!(tensor.dims, dims, "{name}");
        assert_eq!(tensor.ggml_type, kind, "{name}");
    }
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap();
    assert_eq!(
        (tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.add_bos()),
        (Some(248044), Some(248046), false)
    );
    let template = rust_model_inference::prompt::jinja::template_from_source(&|key| {
        source.metadata(key).cloned()
    })
    .unwrap()
    .unwrap();
    let special = rust_model_inference::prompt::jinja::SpecialTokens::default();
    let artifacts = unique_temp_dir("rmi-holo4-tokenizer");
    for text in [
        "",
        "你好",
        " hello   world\n\n\t1234567890",
        "e\u{301} résumé 日本語 🐎",
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    ] {
        let input = artifacts.join("input.txt");
        std::fs::write(&input, text).unwrap();
        let output = command_output(
            Command::new(oracle.with_file_name("llama-tokenize"))
                .arg("-m")
                .arg(&model)
                .arg("-f")
                .arg(input)
                .args(["--ids", "--no-bos"]),
            "Holo4 tokenizer Oracle",
        );
        let expected: Vec<u32> = serde_json::from_slice(&output.stdout).unwrap();
        let actual = tokenizer.encode(
            text,
            EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        );
        assert_eq!(actual, expected, "{text:?}");
        eprintln!("Holo4 tokenizer {text:?}: {actual:?}");
        let rendered = template
            .render(
                &[rust_model_inference::prompt::jinja::ChatMessage::text(
                    "user", text,
                )],
                true,
                false,
                &special,
            )
            .unwrap();
        assert_eq!(
            rendered,
            format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                text.trim()
            )
        );
    }
    std::fs::remove_dir_all(artifacts).unwrap();
}

#[test]
#[ignore = "requires Holo4 Q4_K_M, both pinned Oracles, and --features parity-trace"]
fn holo4_matches_llama_cpp_with_precision_bounds() {
    assert!(cfg!(feature = "parity-trace"), "enable parity-trace");
    assert_eq!(git_head(&required_path("RMI_LLAMA_CPP")), LLAMA_PIN);
    let model = required_path("RMI_HOLO4_MODEL");
    let oracle = required_path("RMI_HOLO4_ORACLE");
    let scalar_oracle = required_path("RMI_HOLO4_SCALAR_ORACLE");
    let artifacts = std::env::var_os("RMI_HOLO4_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| unique_temp_dir("rmi-holo4-parity"));
    std::fs::create_dir_all(&artifacts).unwrap();
    let rust = run_rust(&model, "你好", 4, &artifacts, 64, true);
    let llama = run_oracle(&oracle, &model, "你好", 4, &artifacts);
    let scalar_dir = artifacts.join("scalar");
    std::fs::create_dir(&scalar_dir).unwrap();
    let scalar = run_oracle(&scalar_oracle, &model, "你好", 4, &scalar_dir);
    compare_holo4_traces(&rust, &llama, &scalar)
        .unwrap_or_else(|error| panic!("{error}; artifacts {}", artifacts.display()));
    let generated = records(&rust).unwrap();
    assert_eq!(generated.len(), 325);
    assert_eq!(
        generated.last().unwrap()["token_ids"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    eprintln!(
        "Holo4 greedy IDs: {}",
        generated.last().unwrap()["token_ids"]
    );
    eprintln!("Holo4 parity passed; artifacts {}", artifacts.display());
}

fn holo4_precision_bound(spread: f64) -> f64 {
    // Q8 activation thresholds amplify rounding; calibrate against the fixed Oracle's two kernels.
    assert!(spread.is_finite() && (0.0..=0.35).contains(&spread));
    (1.5 * spread).clamp(1e-5, 0.35)
}

#[cfg(feature = "vulkan")]
fn check_holo4_vulkan(batch: usize) {
    use rust_model_inference::compute::ComputePolicy;
    use rust_model_inference::models::qwen35::Qwen35Session;
    use rust_model_inference::{ComputePool, GGUFLoader, Qwen35Model};
    use std::sync::Arc;

    assert!(std::env::var_os("RMI_PARITY_TRACE").is_none());
    let native = records(&required_path("RMI_HOLO4_NATIVE_TRACE")).unwrap();
    let scalar = records(&required_path("RMI_HOLO4_SCALAR_TRACE")).unwrap();
    let outputs = |trace: &[Value]| {
        trace
            .iter()
            .filter(|record| record["name"] == "result_output")
            .map(|record| {
                assert_eq!(shape(record).unwrap(), [248320]);
                f32_words(record).unwrap()
            })
            .collect::<Vec<_>>()
    };
    let (native_logits, scalar_logits) = (outputs(&native), outputs(&scalar));
    assert_eq!(native_logits.len(), 16);
    assert_eq!(scalar_logits.len(), 16);
    let spread = native_logits
        .iter()
        .zip(&scalar_logits)
        .map(|(native, scalar)| relative_l2(scalar, native))
        .fold(0.0f64, f64::max);
    let bound = holo4_precision_bound(spread);
    let prompt = [
        248045, 846, 198, 109266, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ];
    let expected = [109266u32, 6115, 98691, 95815];
    for trace in [&native, &scalar] {
        let ids = trace
            .iter()
            .find(|record| record["name"] == "qwen35.prompt_ids")
            .unwrap();
        assert_eq!(ids["token_ids"], serde_json::json!(prompt));
        assert_eq!(
            trace.last().unwrap()["token_ids"],
            serde_json::json!(expected)
        );
    }
    let artifacts = std::env::var_os("RMI_HOLO4_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| unique_temp_dir("rmi-holo4-vulkan"))
        .join(format!("batch-{batch}"));
    std::fs::create_dir_all(&artifacts).unwrap();
    let source = GGUFLoader::from_file(required_path("RMI_HOLO4_MODEL")).unwrap();
    let mut model = Qwen35Model::from_source(&source).unwrap();
    rust_model_inference::ops::enable_gpu();
    let context = rust_model_inference::ops::get_vulkan_context().expect("Vulkan required");
    let mut session = Qwen35Session::new_with_compute(
        &mut model,
        128,
        batch,
        Arc::new(ComputePool::new(8)),
        ComputePolicy::Vulkan,
    )
    .unwrap();
    let before = context.submission_count();
    let mut logits = session
        .step_with_tokens(
            &prompt,
            &rust_model_inference::qwen_text_positions(prompt.len()),
        )
        .unwrap();
    let mut tokens = Vec::new();
    let mut metrics = Vec::new();
    for step in 0..expected.len() {
        let words: Vec<_> = logits.iter().map(|value| value.to_bits()).collect();
        std::fs::write(
            artifacts.join(format!("logits-{step}.f32")),
            bytemuck::cast_slice(&logits),
        )
        .unwrap();
        let error = relative_l2(&words, &native_logits[prompt.len() - 1 + step]);
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0 as u32;
        tokens.push(token);
        metrics.push(error);
        eprintln!("Holo4 Vulkan batch={batch} step={step} token={token} relative-L2={error:.9} bound={bound:.9}");
        if step + 1 < expected.len() {
            let position = session.next_position();
            logits = session
                .step_with_tokens(&[token], &[[position, position, position, 0]])
                .unwrap();
        }
    }
    let submissions = context.submission_count() - before;
    let proof = serde_json::json!({
        "oracle_pin": LLAMA_PIN, "device": context.device_name(), "batch": batch,
        "prompt_ids": prompt, "greedy_ids": tokens, "submissions": submissions,
        "relative_l2": metrics, "oracle_spread": spread, "bound": bound,
    });
    std::fs::write(
        artifacts.join("proof.json"),
        serde_json::to_vec_pretty(&proof).unwrap(),
    )
    .unwrap();
    eprintln!(
        "Holo4 Vulkan proof: {}; artifacts {}",
        proof,
        artifacts.display()
    );
    assert!(submissions > 0, "no GPU submissions");
    assert_eq!(tokens, expected);
    assert!(metrics.iter().all(|&error| error <= bound));
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires Holo4, Vulkan and native/scalar pinned Oracle traces"]
fn holo4_vulkan_single_row_matches_precision_bounds() {
    check_holo4_vulkan(1);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires Holo4, Vulkan and native/scalar pinned Oracle traces"]
fn holo4_vulkan_batched_matches_precision_bounds() {
    check_holo4_vulkan(64);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires Holo4 decoder/mmproj, Vulkan and native/scalar image references"]
fn holo4_cpu_vision_vulkan_decoder_matches_precision_bounds() {
    use rust_model_inference::compute::ComputePolicy;
    use rust_model_inference::models::qwen35::{build_qwen35_positions, vision::VisionGrid};
    assert!(std::env::var_os("RMI_PARITY_TRACE").is_none());
    let source =
        rust_model_inference::GGUFLoader::from_file(required_path("RMI_HOLO4_MODEL")).unwrap();
    let mmproj = required_path("RMI_HOLO4_MMPROJ");
    let reference = required_path("RMI_HOLO4_IMAGE_REFERENCE");
    let mut references = Vec::new();
    let mut contracts = Vec::new();
    for name in ["native", "scalar"] {
        let contract: Value =
            serde_json::from_slice(&std::fs::read(reference.join(format!("{name}.json"))).unwrap())
                .unwrap();
        assert_eq!(contract["oracle_pin"], LLAMA_PIN);
        let path = reference.join(format!("{name}.f32"));
        let words = f32_words(&serde_json::json!({"binary_path":path})).unwrap();
        assert_eq!(words.len(), 248320);
        references.push(words);
        contracts.push(contract);
    }
    assert_eq!(contracts[0], contracts[1]);
    let chunks = contracts[0]["chunks"].as_array().unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[1]["count"], 9);
    assert_eq!(chunks[1]["n_pos"], 3);
    let tokenizer =
        rust_model_inference::BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .unwrap();
    let prompt = format!("<|im_start|>user\n<|vision_start|>{}<|vision_end|>What color is the image? Answer with one word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|image_pad|>".repeat(9));
    let ids = tokenizer.encode(
        &prompt,
        rust_model_inference::EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    let mut oracle_ids: Vec<u32> = serde_json::from_value(chunks[0]["text"].clone()).unwrap();
    oracle_ids.extend(std::iter::repeat(tokenizer.special_token_id("image_pad").unwrap()).take(9));
    oracle_ids.extend(serde_json::from_value::<Vec<u32>>(chunks[2]["text"].clone()).unwrap());
    assert_eq!(ids, oracle_ids);
    let grid = VisionGrid {
        grid_t: 1,
        grid_h: 3,
        grid_w: 3,
        patch_size: 16,
        merge_size: 2,
    };
    let (positions, end) =
        build_qwen35_positions(&ids, tokenizer.special_token_id("image_pad"), &[grid]).unwrap();
    let mut oracle_positions: Vec<[usize; 4]> =
        serde_json::from_value(chunks[1]["image_positions"].clone()).unwrap();
    // mtmd exposes x/y; its decoder packs time, row, column.
    for position in &mut oracle_positions {
        position.swap(1, 2);
    }
    assert_eq!(&positions[4..13], oracle_positions);
    assert_eq!(serde_json::json!(end), contracts[0]["end_pos"]);
    let spread = relative_l2(&references[1], &references[0]);
    let bound = holo4_precision_bound(spread);
    let artifacts = unique_temp_dir("rmi-holo4-image-gpu");
    let image = artifacts.join("red.png");
    image::RgbImage::from_pixel(64, 64, image::Rgb([255, 0, 0]))
        .save(&image)
        .unwrap();
    rust_model_inference::ops::enable_gpu();
    let context = rust_model_inference::ops::get_vulkan_context().expect("Vulkan required");
    let mut metrics = Vec::new();
    for policy in [ComputePolicy::Cpu, ComputePolicy::Vulkan] {
        let _scope = policy.enter_legacy_scope();
        let before = context.submission_count();
        let (logits, _) = rust_model_inference::app::run_qwen35_family_multimodal_logits(
            &source,
            &mmproj,
            Some(&image),
            None,
            None,
            "What color is the image? Answer with one word.",
            8,
            64,
            128,
            None,
            None,
        )
        .unwrap();
        let submissions = context.submission_count() - before;
        assert_eq!(submissions > 0, policy == ComputePolicy::Vulkan);
        assert_eq!(logits.len(), 248320);
        assert!(logits.iter().all(|x| x.is_finite()));
        std::fs::write(
            artifacts.join(format!("{policy:?}.f32")),
            bytemuck::cast_slice(&logits),
        )
        .unwrap();
        let words: Vec<_> = logits.iter().map(|x| x.to_bits()).collect();
        let error = relative_l2(&words, &references[0]);
        let top1 = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        assert_eq!(serde_json::json!(top1), contracts[0]["top1"]);
        metrics.push(serde_json::json!({"decoder":format!("{policy:?}"), "submissions":submissions,"top1":top1,"relative_l2":error}));
        eprintln!("Holo4 image decoder={policy:?} submissions={submissions} relative-L2={error:.9} bound={bound:.9}");
    }
    let proof = serde_json::json!({"oracle_pin":LLAMA_PIN,"device":context.device_name(),"oracle_spread":spread,"bound":bound,"metrics":metrics});
    std::fs::write(
        artifacts.join("proof.json"),
        serde_json::to_vec_pretty(&proof).unwrap(),
    )
    .unwrap();
    eprintln!(
        "Holo4 image proof: {proof}; artifacts {}",
        artifacts.display()
    );
    assert!(metrics
        .iter()
        .all(|m| m["relative_l2"].as_f64().unwrap() <= bound));
}

fn compare_holo4_traces(rust: &Path, native: &Path, scalar: &Path) -> Result<(), String> {
    let (rust, native, scalar) = (records(rust)?, records(native)?, records(scalar)?);
    if rust.len() != native.len() || rust.len() != scalar.len() || rust.is_empty() {
        return Err("Holo4 record counts differ or are empty".into());
    }
    let mut metrics = std::collections::BTreeMap::<String, (f64, f64)>::new();
    for ((a, b), c) in rust.iter().zip(&native).zip(&scalar) {
        for other in [b, c] {
            for field in ["name", "layer", "step", "occurrence"] {
                if a.get(field) != other.get(field) {
                    return Err(format!("Holo4 {} field {field} differs", a["name"]));
                }
            }
            if shape(a)? != shape(other)? || len(a)? != len(other)? {
                return Err(format!("Holo4 {} shape/length differs", a["name"]));
            }
        }
        if let Some(exact) = values(a) {
            if Some(exact) != values(b) || Some(exact) != values(c) {
                return Err(format!("Holo4 {} exact values differ", a["name"]));
            }
            continue;
        }
        let name = a["name"].as_str().ok_or("missing checkpoint name")?;
        let initial_state = name == "state_predelta-0" && a["occurrence"].as_u64() == Some(0);
        let expected_len = len(a)?;
        let (a, b, c) = (f32_words(a)?, f32_words(b)?, f32_words(c)?);
        if a.len() != expected_len || a.len() != b.len() || a.len() != c.len() || a.is_empty() {
            return Err(format!("Holo4 {name} binary lengths differ or are empty"));
        }
        if name == "qwen35.embedding" || initial_state {
            if a != b || a != c {
                return Err(format!("Holo4 {name} lossless words differ"));
            }
        } else {
            let metric = metrics.entry(name.to_owned()).or_default();
            metric.0 = metric.0.max(relative_l2(&a, &b));
            metric.1 = metric.1.max(relative_l2(&c, &b));
        }
    }
    for (name, (error, spread)) in metrics {
        let bound = holo4_precision_bound(spread);
        eprintln!(
            "Holo4 {name}: relative-L2={error:.6e}, Oracle spread={spread:.6e}, bound={bound:.6e}"
        );
        if error > bound {
            return Err(format!("Holo4 {name} relative-L2 {error} exceeds {bound}"));
        }
    }
    Ok(())
}

fn relative_l2(actual: &[u32], expected: &[u32]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    let (mut error, mut scale) = (0.0f64, 0.0f64);
    for (&actual, &expected) in actual.iter().zip(expected) {
        let (a, b) = (
            f32::from_bits(actual) as f64,
            f32::from_bits(expected) as f64,
        );
        assert!(a.is_finite() && b.is_finite());
        error += (a - b).powi(2);
        scale += b * b;
    }
    assert!(scale > 0.0);
    (error / scale).sqrt()
}

#[test]
fn precision_metric_rejects_nonfinite_and_detects_drift() {
    let words = |values: &[f32]| values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(relative_l2(&words(&[3.0, 4.0]), &words(&[3.0, 4.0])), 0.0);
    assert!((relative_l2(&words(&[3.0, 5.0]), &words(&[3.0, 4.0])) - 0.2).abs() < 1e-12);
    assert!(std::panic::catch_unwind(|| relative_l2(&words(&[f32::NAN]), &words(&[1.0]))).is_err());
    assert_eq!(holo4_precision_bound(0.0), 1e-5);
    assert!((holo4_precision_bound(0.2) - 0.3).abs() < 1e-12);
    assert!(relative_l2(&words(&[4.0, 5.0]), &words(&[3.0, 4.0])) > holo4_precision_bound(0.1));
    assert!(std::panic::catch_unwind(|| holo4_precision_bound(0.36)).is_err());
}

#[test]
#[ignore = "requires Holo4 F16 mmproj, vision Oracle, and --features parity-trace"]
fn holo4_vision_matches_llama_cpp_with_precision_bounds() {
    use rust_model_inference::models::qwen35::vision::{VisionEncoder, VisionScratchpad};
    assert!(cfg!(feature = "parity-trace"), "enable parity-trace");
    assert_eq!(git_head(&required_path("RMI_LLAMA_CPP")), LLAMA_PIN);
    let mmproj = required_path("RMI_HOLO4_MMPROJ");
    let oracle = required_path("RMI_HOLO4_VISION_ORACLE");
    let artifacts = unique_temp_dir("rmi-holo4-vision");
    let source = rust_model_inference::GGUFLoader::from_file(&mmproj).unwrap();
    let pool = std::sync::Arc::new(rust_model_inference::ComputePool::new(1));
    let mut encoder = VisionEncoder::from_source(&source, pool.clone()).unwrap();
    assert_eq!(
        (
            encoder.config.n_layer,
            encoder.config.n_embd,
            encoder.config.patch_size,
            encoder.config.projection_dim
        ),
        (27, 1152, 16, 5120)
    );
    let pixels: Vec<f32> = (0..64 * 64 * 3)
        .map(|i| ((i * 17 % 251) as f32 - 125.0) / 128.0)
        .collect();
    let input = artifacts.join("pixels.f32");
    std::fs::write(
        &input,
        pixels
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let output = artifacts.join("oracle");
    std::fs::create_dir(&output).unwrap();
    command_output(
        Command::new(oracle)
            .arg(&mmproj)
            .arg(input)
            .args(["64", "64"])
            .arg(&output),
        "Holo4 vision Oracle",
    );
    let expected = records(&output.join("vision.jsonl")).unwrap();
    let trace = artifacts.join("rust.jsonl");
    encoder.precompute();
    std::env::set_var("RMI_PARITY_TRACE", &trace);
    std::env::set_var(
        "RMI_PARITY_FILTER",
        "omni.vision.inp_pos_emb,omni.vision.layer_out",
    );
    let mut scratch = VisionScratchpad::new(&encoder.config);
    let grid = encoder
        .encode_image(&pixels, 64, 64, &mut scratch, &pool)
        .unwrap();
    std::env::remove_var("RMI_PARITY_TRACE");
    std::env::remove_var("RMI_PARITY_FILTER");
    assert_eq!(grid.token_count(), 4);
    let actual = records(&trace).unwrap();
    assert_eq!((actual.len(), expected.len()), (28, 29));
    for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
        let name = match index {
            0 => {
                assert_eq!(a["name"], "omni.vision.inp_pos_emb");
                "qwen.vision.embedding".to_string()
            }
            layer => {
                assert_eq!(a["name"], "omni.vision.layer_out");
                assert_eq!(a["layer"], layer - 1);
                format!("qwen.vision.block.{}", layer - 1)
            }
        };
        assert_eq!(b["name"], name);
        // Both buffers are token-major; the trace APIs label their axes oppositely.
        assert_eq!(
            shape(a).unwrap().into_iter().rev().collect::<Vec<_>>(),
            shape(b).unwrap()
        );
        eprintln!("{name}");
        assert!(relative_l2(&f32_words(a).unwrap(), &f32_words(b).unwrap()) < 0.05);
    }
    eprintln!("qwen.vision.output");
    let projected: Vec<u32> = scratch.projected.iter().map(|x| x.to_bits()).collect();
    assert_eq!(shape(expected.last().unwrap()).unwrap(), [5120, 4]);
    assert!(relative_l2(&projected, &f32_words(expected.last().unwrap()).unwrap()) < 0.05);
    eprintln!("Holo4 vision artifacts {}", artifacts.display());
}
