use serde_json::Value;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

static PARITY_LOCK: Mutex<()> = Mutex::new(());

const LAYER_RECORDS: &[&str] = &[
    "attn_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_norm",
    "k_norm",
    "q_rope",
    "k_rope",
    "attn_scores",
    "attn_probs",
    "attn_values",
    "attn_proj",
    "post_attn_residual",
    "ffn_norm",
    "ffn_gate",
    "ffn_up",
    "ffn_silu_gate",
    "ffn_down",
    "post_ffn_residual",
];

fn remove_trace(trace: &Path, records: &[Value]) {
    for record in records {
        if let Some(path) = record["binary_path"].as_str() {
            let _ = std::fs::remove_file(path);
        }
    }
    let _ = std::fs::remove_file(trace);
}

fn records(trace: &Path) -> Result<Vec<Value>, String> {
    std::fs::read_to_string(trace)
        .map_err(|error| format!("failed to read {}: {error}", trace.display()))?
        .lines()
        .map(|line| serde_json::from_str(line).map_err(|error| error.to_string()))
        .collect()
}

fn run(command: &mut Command, label: &str) -> Result<(), String> {
    let output = command.output().map_err(|error| error.to_string())?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "{label} failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

fn compare_traces(model: &str, prompt: &str, rust: &Path, oracle: &Path) -> Result<(), String> {
    let rust_records = records(rust)?;
    let oracle_records = records(oracle)?;
    if rust_records.len() != oracle_records.len() {
        return Err(format!(
            "model={model} prompt={prompt:?}: record count rust={} oracle={}",
            rust_records.len(),
            oracle_records.len()
        ));
    }

    for (record_index, (got, expected)) in rust_records.iter().zip(&oracle_records).enumerate() {
        for field in ["name", "layer", "step", "shape", "len"] {
            if got[field] != expected[field] {
                return Err(format!(
                    "model={model} prompt={prompt:?} record={record_index}: {field} rust={} oracle={}",
                    got[field], expected[field]
                ));
            }
        }
        if got.get("token_ids").is_some() || expected.get("token_ids").is_some() {
            if got["token_ids"] != expected["token_ids"] {
                return Err(format!(
                    "model={model} prompt={prompt:?} checkpoint={}: token IDs rust={} oracle={}",
                    got["name"], got["token_ids"], expected["token_ids"]
                ));
            }
            continue;
        }

        let got_path = got["binary_path"]
            .as_str()
            .ok_or_else(|| format!("record {record_index} has no Rust binary_path"))?;
        let expected_path = expected["binary_path"]
            .as_str()
            .ok_or_else(|| format!("record {record_index} has no oracle binary_path"))?;
        let got_bytes = std::fs::read(got_path).map_err(|error| error.to_string())?;
        let expected_bytes = std::fs::read(expected_path).map_err(|error| error.to_string())?;
        if got_bytes.len() != expected_bytes.len() || got_bytes.len() % 4 != 0 {
            return Err(format!(
                "model={model} prompt={prompt:?} checkpoint={}: byte lengths rust={} oracle={}",
                got["name"],
                got_bytes.len(),
                expected_bytes.len()
            ));
        }
        for (index, (got_word, expected_word)) in got_bytes
            .chunks_exact(4)
            .zip(expected_bytes.chunks_exact(4))
            .enumerate()
        {
            let got_bits = u32::from_le_bytes(got_word.try_into().unwrap());
            let expected_bits = u32::from_le_bytes(expected_word.try_into().unwrap());
            if got_bits != expected_bits {
                return Err(format!(
                    "model={model} prompt={prompt:?} step={} layer={} checkpoint={} index={index}: \
                     rust=0x{got_bits:08x} ({}) oracle=0x{expected_bits:08x} ({})",
                    got["step"],
                    got["layer"],
                    got["name"],
                    f32::from_bits(got_bits),
                    f32::from_bits(expected_bits)
                ));
            }
        }
    }
    Ok(())
}

fn run_fixture(model_name: &str, model: &Path, prompt: &str, max_tokens: usize) {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "rmi-parity-{}-{nonce}",
        model_name.to_ascii_lowercase()
    ));
    std::fs::create_dir(&directory).unwrap();
    let rust_trace = directory.join("rust.jsonl");
    let oracle_trace = directory.join("oracle.jsonl");
    let max_tokens = max_tokens.to_string();

    let mut rust = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"));
    rust.args(["--model"])
        .arg(model)
        .args([
            "--prompt",
            prompt,
            "--max-tokens",
            &max_tokens,
            "--threads",
            "1",
            "--dump-logits",
            "--kv-cache",
            "f32",
        ])
        .env("RMI_PARITY_TRACE", &rust_trace);
    run(&mut rust, "Rust inference").unwrap();

    let oracle = std::env::var_os("RMI_LLAMA_ORACLE").expect("RMI_LLAMA_ORACLE is required");
    let mut reference = Command::new(oracle);
    reference
        .args(["-m"])
        .arg(model)
        .args(["-p", prompt, "-n", &max_tokens])
        .env("RMI_PARITY_TRACE", &oracle_trace);
    run(&mut reference, "llama.cpp oracle").unwrap();

    if let Err(error) = compare_traces(model_name, prompt, &rust_trace, &oracle_trace) {
        panic!("{error}\ntraces retained in {}", directory.display());
    }

    let rust_records = records(&rust_trace).unwrap();
    let oracle_records = records(&oracle_trace).unwrap();
    remove_trace(&rust_trace, &rust_records);
    remove_trace(&oracle_trace, &oracle_records);
    std::fs::remove_dir(directory).unwrap();
}

fn run_model(model_name: &str, variable: &str) {
    let _guard = PARITY_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let model = std::env::var_os(variable).unwrap_or_else(|| panic!("{variable} is required"));
    let model = Path::new(&model);
    for (prompt, max_tokens) in [("a", 0), ("17 + 25 =", 0), ("你好", 0), ("a", 2)] {
        run_fixture(model_name, model, prompt, max_tokens);
    }
}

#[test]
#[ignore = "requires RMI_Q4_0_MODEL"]
fn dump_logits_emits_complete_step_and_layer_trace() {
    let model = std::env::var_os("RMI_Q4_0_MODEL").expect("RMI_Q4_0_MODEL is required");
    let trace = std::env::temp_dir().join(format!(
        "rmi-inference-trace-{}-{}.jsonl",
        std::process::id(),
        line!()
    ));
    let _ = std::fs::remove_file(&trace);

    let output = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .args([
            "--model",
            model.to_str().unwrap(),
            "--prompt",
            "a",
            "--max-tokens",
            "0",
            "--threads",
            "1",
            "--dump-logits",
            "--kv-cache",
            "f32",
        ])
        .env("RMI_PARITY_TRACE", &trace)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let records: Vec<Value> = std::fs::read_to_string(&trace)
        .unwrap_or_default()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let names: Vec<&str> = records
        .iter()
        .map(|record| record["name"].as_str().unwrap())
        .collect();

    let steps = records
        .first()
        .and_then(|record| record["token_ids"].as_array())
        .expect("first record must contain prompt_ids")
        .len();
    let mut expected = vec!["prompt_ids"];
    for _ in 0..steps {
        expected.extend(["input_token", "embedding"]);
        for _ in 0..28 {
            expected.extend_from_slice(LAYER_RECORDS);
        }
        expected.extend(["result_norm", "result_output"]);
    }
    expected.push("generated_ids");

    assert_eq!(names, expected);
    assert_eq!(records.last().unwrap()["token_ids"], serde_json::json!([]));
    for record in records
        .iter()
        .filter(|record| record["name"] == "result_output")
    {
        assert_eq!(record["shape"], serde_json::json!([151936]));
        assert_eq!(record["len"], 151936);
    }

    remove_trace(&trace, &records);
}

#[test]
#[ignore = "requires RMI_Q4_0_MODEL and RMI_LLAMA_ORACLE"]
fn q4_0_matches_pinned_scalar_oracle_bit_for_bit() {
    run_model("Q4_0", "RMI_Q4_0_MODEL");
}

#[test]
#[ignore = "requires RMI_Q4_K_M_MODEL and RMI_LLAMA_ORACLE"]
fn q4_k_m_matches_pinned_scalar_oracle_bit_for_bit() {
    run_model("Q4_K_M", "RMI_Q4_K_M_MODEL");
}
