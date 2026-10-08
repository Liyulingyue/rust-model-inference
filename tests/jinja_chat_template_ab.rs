//! A/B: Jinja2 chat template vs the hand-written builders.
//!
//! The whole point of rendering `tokenizer.chat_template` instead of
//! hardcoding formats is that the two must agree — otherwise switching a
//! model onto the Jinja path silently changes every token it sees. These
//! tests render both ways from a real GGUF and diff the token ids.
//!
//! Models are supplied through env vars so the default `cargo test` does
//! not need 9 GB on disk:
//!
//! ```sh
//! JINJA_AB_QWEN3=path/Qwen3-0.6B-Q8_0.gguf \
//! JINJA_AB_LFM25=path/LFM2.5-8B-A1B-Q8_0.gguf \
//!   cargo test --profile release-fast --test jinja_chat_template_ab -- --ignored --nocapture
//! ```
//!
//! Nothing here is allowed to pass on a tolerance. A model whose two paths
//! disagree is reported as a divergence with the first differing position,
//! because "close enough" is exactly the failure mode that produced the
//! hardcoded-builder drift in the first place.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};
use rust_model_inference::prompt::jinja::{render_tokens, template_from_source, ChatMessage};
use rust_model_inference::prompt::{build_qwen_chat_prompt, QwenMessage};

const PROMPT: &str = "Capital of France?";

fn env_path(name: &str) -> Option<PathBuf> {
    let raw = std::env::var(name).ok()?;
    if raw.is_empty() {
        return None;
    }
    let path = PathBuf::from(raw);
    path.is_file().then_some(path)
}

struct Loaded {
    tokenizer: BPETokenizer,
    source: Arc<dyn TensorSource>,
}

fn load(path: &Path) -> Option<Loaded> {
    if !path.is_file() {
        eprintln!("skipped: {} not present", path.display());
        return None;
    }
    let source: Arc<dyn TensorSource> =
        Arc::from(open_model_source(path, ComponentRole::Llm).ok()?);
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|e| format!("tokenizer: {e}"))
        .ok()?;
    Some(Loaded { tokenizer, source })
}

fn decode(t: &BPETokenizer, ids: &[u32]) -> String {
    t.decode(ids, true)
}

/// Report where and how two token sequences diverge.
fn describe_divergence(label_a: &str, label_b: &str, a: &[u32], b: &[u32]) -> String {
    let n = a.len().min(b.len());
    let at = (0..n).find(|&i| a[i] != b[i]).unwrap_or(n);
    format!(
        "{label_a} vs {label_b}: len {} vs {}, first difference at index {at} of {n}\n\
         {label_a}: {:?}\n{label_b}: {:?}",
        a.len(),
        b.len(),
        &a[at.min(a.len())..(at + 8).min(a.len())],
        &b[at.min(b.len())..(at + 8).min(b.len())],
    )
}

#[test]
#[ignore = "needs real GGUFs via JINJA_AB_QWEN3 / JINJA_AB_LFM25"]
fn qwen3_jinja_matches_hardcoded_builder() {
    let Some(path) = env_path("JINJA_AB_QWEN3") else {
        eprintln!("skipped: JINJA_AB_QWEN3 not set");
        return;
    };
    let Some(loaded) = load(&path) else { return };
    let t = &loaded.tokenizer;

    let jinja_src = template_from_source(&|k| loaded.source.metadata(k).cloned());
    let Some(Ok(jinja)) = jinja_src else {
        panic!("Qwen3 GGUF must ship tokenizer.chat_template");
    };

    for thinking in [true, false] {
        let jinja_ids = render_tokens(
            t,
            &jinja,
            &[ChatMessage::text("user", PROMPT)],
            true,
            thinking,
        )
        .expect("jinja render");
        let hard_ids = build_qwen_chat_prompt(
            t,
            &[QwenMessage {
                role: "user",
                content: PROMPT,
            }],
            thinking,
        )
        .expect("hardcoded render");

        eprintln!(
            "--- Qwen3 thinking={thinking} ---\n  {}\n  decoded: {}",
            describe_divergence("jinja", "hardcoded", &jinja_ids, &hard_ids),
            decode(t, &jinja_ids).replace('\n', "\\n"),
        );

        assert_eq!(
            jinja_ids,
            hard_ids,
            "Qwen3 thinking={thinking}: the Jinja2 path and the hand-written builder \
             disagree, so flipping the model onto Jinja2 would change every token:\n{}",
            describe_divergence("jinja", "hardcoded", &jinja_ids, &hard_ids)
        );
    }
}

#[test]
#[ignore = "needs real GGUFs via JINJA_AB_LFM25"]
fn lfm25_jinja_renders_and_tokenizes() {
    let Some(path) = env_path("JINJA_AB_LFM25") else {
        eprintln!("skipped: JINJA_AB_LFM25 not set");
        return;
    };
    let Some(loaded) = load(&path) else { return };
    let t = &loaded.tokenizer;

    let Some(Ok(jinja)) = template_from_source(&|k| loaded.source.metadata(k).cloned()) else {
        panic!("LFM2.5 GGUF must ship tokenizer.chat_template");
    };

    let ids = render_tokens(t, &jinja, &[ChatMessage::text("user", PROMPT)], true, true)
        .expect("jinja render");
    let text = decode(t, &ids);

    eprintln!(
        "--- LFM2.5 ---\n  ids:  {ids:?}\n  text: {}",
        text.replace('\n', "\\n")
    );

    // The shipped template emits ChatML control tokens, which must survive
    // as single ids rather than being split into punctuation.
    for literal in ["<|im_start|>", "<|im_end|>"] {
        let id = t
            .token_id(literal)
            .unwrap_or_else(|| panic!("tokenizer must know {literal}"));
        assert!(
            ids.contains(&id),
            "{literal} (id {id}) was shredded instead of kept whole; \
             the prompt would reach the model as punctuation. ids={ids:?}"
        );
    }
    assert_eq!(
        t.encode(
            &text,
            EncodeOptions {
                add_special: false,
                parse_special: true
            }
        ),
        ids,
        "jinja render must round-trip through the tokenizer"
    );
}

/// Known divergence between the shipped LFM2.5 template and the
/// hand-written `build_lfm25_chat_prompt_with_thinking`.
///
/// Recorded, not fixed. `prompt.rs:213` emits
/// `<|im_start|>user\n{content}\n<|im_start|>assistant\n` — it never emits
/// `<|im_end|>`, and it puts a newline where the template does not. The
/// shipped template emits
/// `{bos}<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n`.
///
/// The hardcoded form is very likely wrong, but `LFM2.5-1.2B-Instruct` is
/// recorded `Verified` against llama.cpp with an 8/8 greedy token match, so
/// "fixing" this could invalidate that evidence. It needs a deliberate
/// decision plus a fresh parity run, not a drive-by patch. This test exists
/// so the gap is a failing signal someone has to look at, never a silent
/// drift.
#[test]
#[ignore = "needs a real GGUF via JINJA_AB_LFM25"]
fn lfm25_jinja_known_divergence_from_hardcoded_builder() {
    let Some(path) = env_path("JINJA_AB_LFM25") else {
        eprintln!("skipped: JINJA_AB_LFM25 not set");
        return;
    };
    let Some(loaded) = load(&path) else { return };
    let t = &loaded.tokenizer;
    let Some(Ok(jinja)) = template_from_source(&|k| loaded.source.metadata(k).cloned()) else {
        panic!("LFM2.5 GGUF must ship tokenizer.chat_template");
    };

    let jinja_ids = render_tokens(t, &jinja, &[ChatMessage::text("user", PROMPT)], true, true)
        .expect("jinja render");
    let hard_ids = rust_model_inference::prompt::build_lfm25_chat_prompt_with_thinking(
        t,
        &[rust_model_inference::prompt::Lfm2Message {
            role: "user",
            content: PROMPT,
        }],
        true,
    )
    .expect("hardcoded render");

    eprintln!(
        "LFM2.5 known divergence:\n  jinja:     {}\n  hardcoded: {}\n  {}",
        decode(t, &jinja_ids).replace('\n', "\\n"),
        decode(t, &hard_ids).replace('\n', "\\n"),
        describe_divergence("jinja", "hardcoded", &jinja_ids, &hard_ids),
    );

    // The hardcoded builder omits <|im_end|>. If this ever starts passing,
    // the two paths converged and this test should be deleted along with
    // the note above.
    let im_end = t
        .token_id("<|im_end|>")
        .unwrap_or_else(|| panic!("tokenizer must know <|im_end|>"));
    assert!(
        jinja_ids.contains(&im_end),
        "jinja render should contain <|im_end|>"
    );
    assert!(
        !hard_ids.contains(&im_end),
        "KNOWN DIVERGENCE GONE: the hardcoded LFM2.5 builder now emits \
         <|im_end|>, so it may finally match the shipped template. Re-run the \
         llama.cpp parity for LFM2.5-1.2B-Instruct and update SUPPORTED_MODELS.md."
    );
}
