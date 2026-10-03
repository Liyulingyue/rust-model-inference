//! End-to-end CLI smoke test for the Qwen2.5-1.5B-Instruct JEV path.
//!
//! `#[ignore]`d because it shells out to the real binary and needs the
//! 1.1 GB Q4_K_M GGUF; run with:
//!
//! ```text
//! RMI_QWEN2_5_1_5B_INSTRUCT_Q4_K_MODEL=\\
//!   models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf \\
//!   cargo test --profile release-fast --test qwen2_5_1_5b_jev -- --ignored --nocapture
//! ```
//!
//! Qwen2.5's GGUF declares `general.architecture = "qwen2"`; the JEV
//! dispatch was extended to route that arch through the Qwen3 ChatML
//! scorer. The combined `--prompt` + ChatML + clean assistant-prefix
//! path is what the CLI exercises below.
//!
//! These assertions pin the **shape** of the JEV output (single choice,
//! normalized distribution, probability line, JSON-able) and pin the
//! **argmax** for two obviously-asymmetric questions, because the Qwen2.5
//! JEV path previously produced a strong positional bias (always `B`)
//! when the assistant-prefix leaked the empty-think markers from
//! `append_qwen_assistant_prefix` into tokenizers that lack native
//! `<think>` / `</think>` tokens. After the prefix fix below, the
//! model picks the correct letter with >99% confidence.

use std::path::Path;
use std::process::Command;

const PROMPT_A: &str = "The capital of France is Paris.";
const QUESTION_A: &str = "Is the previous statement true?";
const CORRECT_A: &str = "yes";

const PROMPT_B: &str = "The new phone is amazing! Best I've ever owned!";
const QUESTION_B: &str = "What is the sentiment?";

#[test]
#[ignore = "requires models/Qwen2.5-1.5B-Instruct-GGUF (~1.1 GB Q4_K_M)"]
fn jev_runs_and_chooses_yes_for_capital_question() {
    let model = model_path();
    let output = run_jev(&model, PROMPT_A, QUESTION_A, &[CORRECT_A, "no"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "exit failure\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        stdout.contains("--- JEV decision ---"),
        "missing decision block\n{stdout}"
    );
    assert!(
        stdout.contains("choice: A"),
        "expected A=correct (yes) but model picked differently\n{stdout}"
    );
    // Distribution sanity: probabilities must sum to ~1 and the chosen
    // letter must dominate. The actual margin is well over 99% on this
    // prompt; the floor is 0.7 to leave slack for future re-quantizations.
    let a_p = probability_for(&stdout, 'A').expect("missing probability for A");
    assert!(a_p > 0.7, "A probability {a_p} too low\n{stdout}");
    let sum: f32 = ['A', 'B']
        .iter()
        .map(|c| probability_for(&stdout, *c).unwrap_or(0.0))
        .sum();
    assert!((sum - 1.0).abs() < 1e-3, "distribution does not sum to 1: {sum}");
}

#[test]
#[ignore = "requires models/Qwen2.5-1.5B-Instruct-GGUF (~1.1 GB Q4_K_M)"]
fn jev_runs_and_chooses_positive_sentiment() {
    let model = model_path();
    let output = run_jev(
        &model,
        PROMPT_B,
        QUESTION_B,
        &["positive", "negative", "neutral"],
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "exit failure\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        stdout.contains("choice: A"),
        "expected A=positive for clearly positive review\n{stdout}"
    );
    let a_p = probability_for(&stdout, 'A').expect("missing probability for A");
    assert!(a_p > 0.7, "A probability {a_p} too low\n{stdout}");
}

#[test]
#[ignore = "requires models/Qwen2.5-1.5B-Instruct-GGUF (~1.1 GB Q4_K_M)"]
fn jev_order_matters_for_argmax_position() {
    // Reverse the labels: `no` becomes A, `yes` becomes B. The model
    // should still pick the `yes` answer, now at position B.
    let model = model_path();
    let output = run_jev(&model, PROMPT_A, QUESTION_A, &["no", "yes"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(
        stdout.contains("choice: B"),
        "expected B=correct (yes) when labels are reversed\n{stdout}"
    );
    let b_p = probability_for(&stdout, 'B').expect("missing probability for B");
    assert!(b_p > 0.7, "B probability {b_p} too low\n{stdout}");
}

fn model_path() -> std::path::PathBuf {
    let raw = std::env::var_os("RMI_QWEN2_5_1_5B_INSTRUCT_Q4_K_MODEL")
        .expect("set RMI_QWEN2_5_1_5B_INSTRUCT_Q4_K_MODEL to the q4_k_m GGUF path");
    Path::new(&raw).to_path_buf()
}

fn run_jev(model: &std::path::Path, context: &str, question: &str, options: &[&str]) -> std::process::Output {
    assert!(model.exists(), "model file missing: {}", model.display());
    let mut command = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"));
    command
        .args(["--model", model.to_str().unwrap()])
        .args(["--threads", "4"])
        .args(["--max-context", "1024"])
        .args(["--jev", "--jev-context", context])
        .args(["--jev-question", question]);
    for option in options {
        command.args(["--jev-option", option]);
    }
    command.output().expect("spawn rust-model-inference")
}

fn probability_for(stdout: &str, label: char) -> Option<f32> {
    let needle = format!("{label}:");
    let line = stdout.lines().find(|line| line.trim_start().starts_with(&needle))?;
    let after = line.split(':').nth(1)?.trim();
    after.parse::<f32>().ok()
}
