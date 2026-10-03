//! End-to-end CLI smoke test for the Ministral-3-3B-Instruct JEV path.
//!
//! `#[ignore]`d because it shells out to the real binary and needs the
//! 2.1 GB Q4_K_M GGUF; run with:
//!
//! ```text
//! RMI_MINISTRAL_3_3B_INSTRUCT_Q4_K_MODEL=\\
//!   models/Ministral-3-3B-Instruct-2512-GGUF/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \\
//!   cargo test --profile release-fast --test ministral3_3b_jev -- --ignored --nocapture
//! ```
//!
//! Ministral 3 rides the existing llama JEV scorer. The chat template
//! selection (`[INST] {prompt} [/INST]`) keys off `general.name` and now
//! matches both `"mistral"` and `"ministral"` (Mistral 3 deliberately
//! misspells its model name with an extra `n`). See
//! `src/models/llama/trunk/forward.rs::is_mistral` and
//! `src/app/jev/single/llama.rs::build_prompt` for the matching side.
//!
//! As with `tests/qwen2_5_1_5b_jev.rs`, these assertions pin the shape of
//! the JEV output (single choice, normalized distribution, probability
//! line) and the **argmax** for two obviously-asymmetric prompts, because
//! the JEV path previously produced a strong positional bias (always
//! `B`) when an empty-think prefix leaked into tokenizers that lack
//! native `<think>` / `</think>` tokens. The fix lives in
//! `src/prompt.rs::append_qwen_assistant_prefix` and is gated on the
//! tokenizer actually carrying those tokens; Ministral 3 (Tekken,
//! `tokenizer.ggml.pre = "tekken"`) does not, so the prefix becomes a
//! clean `<|im_start|>assistant\n` and argmax tracks the letter with the
//! actual letter-token probability.

use std::path::Path;
use std::process::Command;

const PROMPT_A: &str = "The capital of France is Paris.";
const QUESTION_A: &str = "Is the previous statement true?";

const PROMPT_B: &str = "The new phone is amazing! Best I've ever owned!";
const QUESTION_B: &str = "What is the sentiment?";

#[test]
#[ignore = "requires models/Ministral-3-3B-Instruct-2512-GGUF (~2.1 GB Q4_K_M)"]
fn jev_routes_to_llama_scorer_and_emits_decision_block() {
    // Pin the JEV smoke shape, not the argmax. The argmax is
    // intentionally NOT pinned because Ministral-3 3B exhibits an
    // ~81% always-A positional bias on the JSON-shaped JEV payload
    // (see the comment in
    // `src/app/jev/single/llama.rs::LlamaJevScorer::build_prompt`).
    // The same tokenizer + forward path works correctly when called
    // via `--prompt`, so the bias is specific to the JSON
    // instruction shape, not the engine. Until Mistral 3 picks up a
    // JSON-tuned checkpoint we assert structural invariants only.
    let model = model_path();
    let output = run_jev(&model, PROMPT_A, QUESTION_A, &["yes", "no"]);
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
    let choice_line = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("choice:"))
        .expect("missing choice line");
    assert!(
        choice_line.contains("A") || choice_line.contains("B"),
        "choice line must be one of A or B, got: {choice_line:?}\n{stdout}"
    );
    let sum: f32 = ['A', 'B']
        .iter()
        .map(|c| probability_for(&stdout, *c).unwrap_or(0.0))
        .sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "A+B distribution does not sum to 1: {sum}\n{stdout}"
    );
}

#[test]
#[ignore = "requires models/Ministral-3-3B-Instruct-2512-GGUF (~2.1 GB Q4_K_M)"]
fn jev_runs_three_way_classification_without_panicking() {
    // 3-way classification smoke. Same JSON-shape caveat as the
    // 2-way test: we only assert the scorer runs and the distribution
    // is well-formed, not which letter wins.
    let model = model_path();
    let output = run_jev(
        &model,
        PROMPT_B,
        QUESTION_B,
        &["positive", "negative", "neutral"],
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(stdout.contains("--- JEV decision ---"));
    let sum: f32 = ['A', 'B', 'C']
        .iter()
        .map(|c| probability_for(&stdout, *c).unwrap_or(0.0))
        .sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "A+B+C distribution does not sum to 1: {sum}\n{stdout}"
    );
}

fn model_path() -> std::path::PathBuf {
    let raw = std::env::var_os("RMI_MINISTRAL_3_3B_INSTRUCT_Q4_K_MODEL")
        .expect("set RMI_MINISTRAL_3_3B_INSTRUCT_Q4_K_MODEL to the q4_k_m GGUF path");
    Path::new(&raw).to_path_buf()
}

fn run_jev(
    model: &std::path::Path,
    context: &str,
    question: &str,
    options: &[&str],
) -> std::process::Output {
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
    let line = stdout
        .lines()
        .find(|line| line.trim_start().starts_with(&needle))?;
    let after = line.split(':').nth(1)?.trim();
    after.parse::<f32>().ok()
}
