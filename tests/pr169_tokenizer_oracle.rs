//! Explicit real-model checks; see tools/oracle/pr169/README.md.
use rust_model_inference::{
    core::tokenizer::{BPETokenizer, EncodeOptions},
    GGUFLoader,
};
use std::{
    io::Write,
    process::{Command, Stdio},
};

fn compare(model_env: &str) {
    let path = std::env::var_os(model_env).unwrap_or_else(|| panic!("{model_env} is required"));
    let binary = std::env::var_os("RMI_LLAMA_TOKENIZE").expect("RMI_LLAMA_TOKENIZE is required");
    let source = GGUFLoader::from_file(&path).expect("load model");
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap();
    for prompt in [
        "",
        "Hello world",
        " 你好，Rust! 👋\n\n",
        "<|im_start|>user\nHi<|im_end|>\n",
        "<tool_call>hello</tool_call> <think>想一想</think>",
    ] {
        let mut process = Command::new(&binary)
            .arg("-m")
            .arg(&path)
            .args(["--stdin", "--ids", "--no-escape"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        process
            .stdin
            .take()
            .unwrap()
            .write_all(prompt.as_bytes())
            .unwrap();
        let output = process.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<u32> = serde_json::from_slice(&output.stdout).expect("Oracle token IDs");
        assert_eq!(
            tokenizer.encode(
                prompt,
                EncodeOptions {
                    add_special: true,
                    parse_special: true
                }
            ),
            expected,
            "tokenizer mismatch for {prompt:?}"
        );
    }
}

#[test]
#[ignore = "requires real model and pinned llama-tokenize"]
fn intern_decision_tokenizer() {
    compare("RMI_INTERN_DECISION_0_8B_TEXT");
}
#[test]
#[ignore = "requires real model and pinned llama-tokenize"]
fn physbrain_tokenizer() {
    compare("RMI_PHYSBRAIN_1_5_2B_TEXT");
}
#[test]
#[ignore = "requires real model and pinned llama-tokenize"]
fn zdt_tokenizer() {
    compare("RMI_ZDT_TEXT");
}
#[test]
#[ignore = "requires real model and pinned llama-tokenize"]
fn occamy_tokenizer() {
    compare("RMI_OCCAMY_TEXT");
}
