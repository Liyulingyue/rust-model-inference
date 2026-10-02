//! Multi-turn chat-prompt shape regression for the llama-arch trunk.
//!
//! Covers the bug where `build_prompt_tokens_from_turns` only appended the
//! assistant generation prompt when `turns.len() == 1`. A multi-turn
//! prompt therefore ended at the last user turn, so the model was asked to
//! continue mid-user-message: GLM-4 sampled an immediate stop token and
//! returned an empty completion.
//!
//! Tokenize-only checks (no forward pass), so they run in milliseconds
//! once the GGUF is mmap'd.

use std::path::PathBuf;
use std::sync::Arc;

use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::core::tokenizer::BPETokenizer;
use rust_model_inference::models::llama::trunk::forward::build_prompt_tokens_from_turns;
use rust_model_inference::GGUFLoader;

fn gguf(parts: &[&str]) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("models");
    for part in parts {
        p.push(part);
    }
    p
}

type Loaded = (Arc<dyn TensorSource>, BPETokenizer);

fn load(path: &PathBuf) -> Option<Loaded> {
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return None;
    }
    let source: Arc<dyn TensorSource> =
        Arc::new(GGUFLoader::from_file(path.clone()).expect("failed to open GGUF"));
    let tok = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned()).expect("tokenizer");
    Some((source, tok))
}

const DIALOGUE: [(&str, &str); 3] = [
    ("user", "My name is Ada."),
    ("assistant", "Nice to meet you, Ada."),
    ("user", "What is my name?"),
];

#[test]
fn glm4_multi_turn_prompt_ends_with_assistant_marker() {
    let Some((source, tok)) = load(&gguf(&["GLM-4-9B-0414-GGUF", "GLM-4-9B-0414-Q4_K_M.gguf"]))
    else {
        return;
    };

    let ids = build_prompt_tokens_from_turns(source.as_ref(), &tok, &DIALOGUE, false)
        .expect("build_prompt_tokens_from_turns");
    let rendered = tok.decode(&ids, true);

    // The last thing in the prompt must be the generation marker, otherwise
    // the model is asked to continue the user's message.
    assert!(
        rendered.trim_end().ends_with("<|assistant|>"),
        "multi-turn prompt must end with <|assistant|> so generation starts there; got {rendered:?}",
    );
    // [gMASK]<sop> must appear exactly once, at the very front (after the
    // tokenizer's BOS, which GLM-4 needs even though add_bos_token=false).
    let body = rendered.strip_prefix("<|endoftext|>").unwrap_or(&rendered);
    assert!(
        body.starts_with("[gMASK]"),
        "prompt should start with [gMASK] after BOS; got {rendered:?}",
    );
    assert_eq!(
        rendered.matches("[gMASK]").count(),
        1,
        "[gMASK] must be emitted exactly once; got {rendered:?}",
    );
    // Both user turns must survive the round-trip.
    assert!(
        rendered.contains("My name is Ada."),
        "first user turn was dropped; got {rendered:?}",
    );
    assert!(
        rendered.contains("What is my name?"),
        "last user turn was dropped; got {rendered:?}",
    );
    // Generation prompt must appear exactly once at the very end.
    assert_eq!(
        rendered.matches("<|assistant|>").count(),
        2,
        "expected one <|assistant|> marker per assistant turn (middle + generation); got {rendered:?}",
    );
}

#[test]
fn llama32_multi_turn_prompt_is_unchanged() {
    // No-regression guard. Llama-3.2's template is the `user\n{content}\n
    // assistant\n<think>\n` fallback, which already folds the generation
    // marker into the USER turn. Appending another generation prompt on top
    // of that used to render a doubled
    // `…What is my name?\nassistant\n<think>\nuser\n\nassistant\n<think>\n`.
    // Pinned byte-for-byte against the pre-fix rendering.
    let Some((source, tok)) = load(&gguf(&[
        "Llama-3.2-1B-Instruct-GGUF",
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
    ])) else {
        return;
    };

    let ids = build_prompt_tokens_from_turns(source.as_ref(), &tok, &DIALOGUE, false)
        .expect("build_prompt_tokens_from_turns");
    let rendered = tok.decode(&ids, true);

    assert_eq!(
        rendered,
        concat!(
            "<|begin_of_text|>",
            "user\nMy name is Ada.\nassistant\n<|think|>\n",
            "user\nNice to meet you, Ada.\nassistant\n<|think|>\n",
            "user\nWhat is my name?\nassistant\n<|think|>\n",
        ),
        "Llama-3.2 multi-turn prompt must stay byte-identical to the baseline",
    );
}
