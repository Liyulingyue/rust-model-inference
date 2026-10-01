//! Smoke test for Mistral-7B-Instruct chat template + token-shape parity.
//!
//! Verifies that the `[INST] {prompt} [/INST]` chat template branch is
//! correctly detected via `general.name` containing "mistral", and that
//! `build_prompt_tokens` produces the right SentencePiece special-token
//! shape (`[INST]` = id 3, `[/INST]` = id 4, BOS = id 1).
//!
//! Regression for the PR-#118 multi-turn refactor that dropped the
//! Mistral/Zephyr branches from the new `build_prompt_tokens_from_turns`
//! / `llama_turn_text` path. Without this fix, Mistral models render
//! through the default Qwen2-style `user\n…\nassistant\n🤔\n` fallback
//! and the model echoes the template back instead of answering.
//!
//! Model: models/Mistral-7B-Instruct-v0.3-GGUF/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
use std::path::PathBuf;
use std::sync::Arc;

use rust_model_inference::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::llama::trunk::forward::build_prompt_tokens;

fn model_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("models");
    p.push("Mistral-7B-Instruct-v0.3-GGUF");
    p.push("Mistral-7B-Instruct-v0.3.Q4_K_M.gguf");
    p
}

#[test]
fn mistral_instruct_v0_3_chat_template_smoke() {
    if !model_path().exists() {
        eprintln!("skipping: {} not present", model_path().display());
        return;
    }
    let source: Arc<dyn TensorSource> = Arc::new(
        GGUFLoader::from_file(model_path()).expect("failed to load Mistral-7B-Instruct-v0.3 GGUF"),
    );

    let prompt = "What is the capital of France?";
    let ids = build_prompt_tokens(source.as_ref(), prompt, false)
        .expect("build_prompt_tokens should succeed for llama-arch mistral");

    eprintln!(
        "[MISTRAL_TEST] prompt token ids (len={}): first 12 = {:?}",
        ids.len(),
        &ids[..ids.len().min(12)],
    );

    // Test 1: BOS at start (id=1).
    assert_eq!(ids.first().copied(), Some(1), "BOS at start");

    // Test 2: position 1 must be `[INST]` (id=3) for the Mistral chat
    // template. Pre-fix, this would be 2956 (the "▁What" piece) because
    // the default `user\n{prompt}\nassistant\n🤔\n` template renders
    // through without recognising `[INST]` as a special token.
    assert_eq!(
        ids.get(1).copied(),
        Some(3),
        "Mistral chat template should emit [INST]=3 right after BOS; got ids={:?}",
        ids,
    );

    // Test 3: `[/INST]` (id=4) somewhere after the user content, closing
    // the user turn so generation starts there.
    let inst_close_pos = ids.iter().position(|&i| i == 4).expect("[/INST] missing");
    assert!(
        inst_close_pos > 2,
        "[/INST] should follow at least one content token after [INST]; got position={}",
        inst_close_pos,
    );
}
