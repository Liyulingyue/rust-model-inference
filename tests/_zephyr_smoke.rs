//! Smoke test for Zephyr-7B-alpha chat template + token-shape parity.
//!
//! Verifies that the `<|user|>\n{prompt}</s>\n<|assistant|>\n` chat template
//! branch is correctly detected via `general.name` containing "zephyr" OR
//! `tokenizer.chat_template` containing both `<|user|>` and `<|assistant|>`,
//! and that `build_prompt_tokens` produces the right SentencePiece token
//! shape.
//!
//! Regression for the PR-#118 multi-turn refactor that dropped the
//! Mistral/Zephyr branches from the new `build_prompt_tokens_from_turns`
//! / `llama_turn_text` path. Without this fix, Zephyr models would render
//! through the default Qwen2-style `user\n…\nassistant\n🤔\n` fallback and
//! the model echoes the template back instead of answering.
//!
//! The detection covers publishers that rename `general.name`: mradermacher
//! writes `general.name="."` and MaziyarPanahi writes `"hub"`, so the
//! detection also matches on `tokenizer.chat_template` (which is the unique
//! signature of Zephyr's H4-style template).
//!
//! Zephyr's vocab (mradermacher/zephyr-7b-alpha-GGUF Q4_K_M) does NOT contain
//! `<|user|>` / `<|assistant|>` as single SentencePiece tokens — they are
//! literal text in the chat template that the SPM tokenizer splits into
//! multiple subword tokens (e.g. `<`, `|`, `user`, `|`, `>`).
//!
//! Model: models/zephyr-7b-alpha-GGUF/zephyr-7b-alpha.Q4_K_M.gguf
//! (downloaded from HF mirror mradermacher/zephyr-7b-alpha-GGUF since the
//! ModelScope publishers ship GGUF files with stripped tokenizer metadata.)
use std::path::PathBuf;
use std::sync::Arc;

use rust_model_inference::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::llama::trunk::forward::build_prompt_tokens;

fn model_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("models");
    p.push("zephyr-7b-alpha-GGUF");
    p.push("zephyr-7b-alpha.Q4_K_M.gguf");
    p
}

#[test]
fn zephyr_7b_alpha_chat_template_smoke() {
    if !model_path().exists() {
        eprintln!("skipping: {} not present", model_path().display());
        return;
    }
    let source: Arc<dyn TensorSource> = Arc::new(
        GGUFLoader::from_file(model_path()).expect("failed to load Zephyr-7B-alpha GGUF"),
    );

    let prompt = "What is the capital of France?";
    let ids = build_prompt_tokens(source.as_ref(), prompt, false)
        .expect("build_prompt_tokens should succeed for llama-arch zephyr");

    eprintln!(
        "[ZEPHYR_TEST] prompt token ids (len={}): first 24 = {:?}",
        ids.len(),
        &ids[..ids.len().min(24)],
    );

    // Test 1: BOS at start (id=1). Zephyr uses Mistral's tokenizer (Zephyr
    // is finetuned from Mistral-7B), so BOS=1.
    assert_eq!(ids.first().copied(), Some(1), "BOS at start");

    // Test 2: `</s>` (id=2) appears as the user-turn terminator. The
    // template wraps the user content with `<|user|>\n{content}</s>\n`,
    // so `</s>` should be in the prompt token stream somewhere after the
    // user content. Per the GGUF metadata `tokenizer.ggml.eos_token_id=2`.
    let eos_positions: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter_map(|(i, &id)| if id == 2 { Some(i) } else { None })
        .collect();
    assert!(
        !eos_positions.is_empty(),
        "</s> missing from prompt; ids={:?}",
        ids,
    );
    // The closing `</s>` should NOT be the very first non-BOS token (it
    // comes after the user content), so eos must be in the second half
    // of the prompt.
    let last_eos_pos = *eos_positions.last().expect("at least one </s>");
    assert!(
        last_eos_pos > ids.len() / 2,
        "</s> should appear in the second half of the prompt; got pos={} of len={}, ids={:?}",
        last_eos_pos,
        ids.len(),
        ids,
    );
}
