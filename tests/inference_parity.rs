use serde_json::Value;
use std::path::Path;
use std::process::Command;

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
