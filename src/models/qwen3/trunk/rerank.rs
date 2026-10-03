//! Qwen3 cross-encoder rerank scoring.
//!
//! Cross-encoder rerank contract: build the standard ChatML prompt
//! (`system` + `<Instruct>/<Query>/<Document>`), prefill as a single
//! causal pass, take the last token's post-RMSNorm hidden state, project
//! it through the model's 2-class classification head, return the
//! softmaxed `yes` probability per document.
//!
//! Mirror of `src/app/server/rerank.rs::score_one_doc_qwen3`, which
//! runs the same per-doc loop inside a `/v1/rerank` HTTP handler. The
//! `--rerank` main-binary CLI path (`src/main.rs`) calls into this
//! module instead, so the standalone CLI and the HTTP endpoint share
//! the same scoring contract and there is one source of truth for
//! "what does rerank mean for qwen3".

use std::sync::Arc;

use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::qwen3::trunk::{qwen_text_positions, Qwen3Input, Qwen3Model, Qwen3Session};

const DEFAULT_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

fn render_prompt(instruction: &str, query: &str, document: &str) -> String {
    format!(
        "system\nJudge whether the Document meets the requirements based on \
         the Query and the Instruct provided. Note that the answer can only be \
         \"yes\" or \"no\".\nuser\n<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}\n"
    )
}

fn softmax_pair(a: f32, b: f32) -> (f32, f32) {
    let m = a.max(b);
    let ea = (a - m).exp();
    let eb = (b - m).exp();
    let s = ea + eb;
    (ea / s, eb / s)
}

/// Score a single `(query, document)` pair. Returns the `yes` logit
/// and the softmaxed `yes_prob` (in `[0, 1]`); the caller picks which
/// to surface. Mirrors `src/app/server/rerank.rs::score_one_doc_qwen3`.
fn score_one_doc(
    model: &Qwen3Model,
    tokenizer: &BPETokenizer,
    context_length: usize,
    query: &str,
    document: &str,
    instruction: &str,
) -> Result<(f32, f32), String> {
    let prompt = render_prompt(instruction, query, document);
    let token_ids = tokenizer.encode(
        &prompt,
        EncodeOptions {
            add_special: false,
            parse_special: false,
        },
    );
    let n = token_ids.len();
    if n == 0 {
        return Err("empty rerank tokenization".into());
    }
    if n > context_length {
        return Err(format!(
            "rerank prompt length {n} exceeds model context {context_length}"
        ));
    }
    let positions = qwen_text_positions(n);
    let mut session =
        Qwen3Session::new(model, n + 4).map_err(|e| format!("rerank session: {e}"))?;
    let hidden = session
        .forward_rerank(
            Qwen3Input {
                token_ids: &token_ids,
                positions: &positions,
                embeddings: None,
                deepstack_embeddings: None,
            },
            n,
        )
        .map_err(|e| format!("rerank forward: {e}"))?;
    let logits = model
        .score_logits(&hidden)
        .map_err(|e| format!("rerank score: {e}"))?;
    if logits.len() < 2 {
        return Err(format!(
            "rerank head returned {} logits; expected >=2",
            logits.len()
        ));
    }
    let yes = logits[0];
    let no = logits[1];
    let (yes_prob, _no_prob) = softmax_pair(yes, no);
    Ok((yes, yes_prob))
}

/// Batch rerank: score `documents` against `query`, returning one
/// `yes_prob` per document in input order.
///
/// Loads the model fresh on each call. The HTTP path builds its own
/// session from a `'static` model instead; see `src/app/server/rerank.rs`.
pub fn score_qwen3_rerank(
    source: Arc<dyn TensorSource>,
    query: &str,
    documents: &[String],
    n_threads_arg: usize,
    instruction: Option<&str>,
) -> Result<Vec<f32>, String> {
    if documents.is_empty() {
        return Ok(Vec::new());
    }
    if query.trim().is_empty() {
        return Err("--rerank-query must be non-empty".into());
    }
    let tokenizer = Arc::new(
        BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|e| format!("init tokenizer: {e}"))?,
    );
    let pool = Arc::new(ComputePool::new(n_threads_arg.max(1)));
    let model = Qwen3Model::from_source(source, tokenizer.clone(), pool)
        .map_err(|e| format!("load model: {e}"))?;
    if model.config().architecture != "qwen3" {
        return Err(format!(
            "expected qwen3 architecture for rerank, got {}",
            model.config().architecture
        ));
    }
    if !model.is_rerank() {
        return Err(
            "GGUF is qwen3 but has no cls.output.weight — not a rerank model \
             (need ggml-org/Qwen3-Reranker-*-Q8_0-GGUF or similar)"
                .into(),
        );
    }
    let context_length = model.config().n_ctx;
    let instruction = instruction.unwrap_or(DEFAULT_INSTRUCTION);

    let mut scored = Vec::with_capacity(documents.len());
    for (idx, doc) in documents.iter().enumerate() {
        let (_yes_logit, yes_prob) =
            score_one_doc(&model, &tokenizer, context_length, query, doc, instruction)
                .map_err(|e| format!("doc {idx}: {e}"))?;
        scored.push(yes_prob);
    }
    Ok(scored)
}
