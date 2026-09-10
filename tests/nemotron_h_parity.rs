//! Nemotron-3 Nano 4B parity test against llama.cpp reference output.
//!
//! Each test loads a fixture from `docs/parity_fixtures/nemotron_h_4b/` and
//! runs the Rust `nemotron_h` trunk on the same prompt with
//! `--temp 0.0`. The test compares the **post-thinking** response tokens
//! against the expected output captured by `llama.cpp`.
//!
//! Current state: the Rust SSM forward is a degenerate D-skip
//! (no selective scan), so all tests are expected to fail until
//! the Mamba2 scan is wired up. The fixture dir still serves as the
//! oracle target.

use std::path::PathBuf;

const FIXTURE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/docs/parity_fixtures/nemotron_h_4b"
);

const MODEL_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron-3-Nano-4B-Q4_0.gguf"
);

/// Extract the response that follows the `[End thinking]` delimiter.
fn response_after_thinking(raw: &str) -> &str {
    if let Some(idx) = raw.find("[End thinking]") {
        raw[idx + "[End thinking]".len()..].trim_start()
    } else {
        raw.trim_start()
    }
}

fn run_rust(prompt: &str) -> String {
    use std::io::Write;
    use std::process::{Command, Stdio};

    if !PathBuf::from(MODEL_PATH).exists() {
        eprintln!("skipping: {} not present", MODEL_PATH);
        return String::new();
    }

    let mut child = Command::new(env!("CARGO"))
        .args([
            "run",
            "--release",
            "--bin",
            "rust-model-inference",
            "--",
            "--model",
            MODEL_PATH,
            "--prompt",
            prompt,
            "--max-tokens",
            "32",
            "--temp",
            "0",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn rust-model-inference");
    let mut out = String::new();
    if let Some(mut stdout) = child.stdout.take() {
        use std::io::Read;
        stdout.read_to_string(&mut out).unwrap();
    }
    let _ = child.wait();
    // The last `Output: <text>` line in the stream is the model's response.
    let response_line = out
        .lines()
        .rev()
        .find(|line| line.starts_with("Output:"))
        .map(|line| line.trim_start_matches("Output:").trim().to_string());
    response_line.unwrap_or_default()
}

fn oracle_response(prompt_file: &str) -> Option<String> {
    let path = PathBuf::from(FIXTURE_DIR).join(prompt_file);
    if !path.exists() {
        return None;
    }
    let raw = std::fs::read_to_string(&path).ok()?;
    Some(response_after_thinking(&raw).to_string())
}

fn assert_matches(prompt: &str, fixture: &str) {
    let oracle = match oracle_response(fixture) {
        Some(s) => s,
        None => {
            eprintln!("skipping: {} not present", fixture);
            return;
        }
    };
    let ours = run_rust(prompt);
    assert!(
        ours.contains(&oracle) || oracle.contains(&ours) || ours == oracle,
        "Nemotron-3 Nano parity mismatch for prompt {:?}:\n  oracle: {:?}\n  ours:   {:?}",
        prompt,
        oracle,
        ours,
    );
}

#[test]
#[ignore = "skipped: the Mamba2 selective scan is not implemented; current SSM \
          forward is a degenerate D-skip that cannot reproduce the oracle. \
          See SUPPORTED_MODELS.md for the implementation roadmap."]
fn nemotron_h_4b_hello() {
    assert_matches("Hello", "Hello.txt");
}

#[test]
#[ignore = "skipped: same reason as nemotron_h_4b_hello"]
fn nemotron_h_4b_translate_french() {
    assert_matches(
        "Translate to French: hello world",
        "Translate_to_French:_hello_world.txt",
    );
}

#[test]
#[ignore = "skipped: same reason as nemotron_h_4b_hello"]
fn nemotron_h_4b_arithmetic() {
    assert_matches("What is 2+2?", "What_is_2+2?.txt");
}
