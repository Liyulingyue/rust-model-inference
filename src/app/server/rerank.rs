//! `/v1/rerank` HTTP endpoint for Qwen3-style cross-encoder reranking.
//!
//! Request schema (Cohere-compatible, simplified):
//! ```json
//! {
//!   "model": "qwen3-reranker-0.6b",   // informational, ignored
//!   "query": "What is the capital of France?",
//!   "documents": ["doc1", "doc2", ...],
//!   "top_n": 3,                       // optional, default: all
//!   "return_documents": false,        // optional, default false
//!   "max_tokens_per_doc": 512         // optional, soft word-based cap
//! }
//! ```
//!
//! Response:
//! ```json
//! {
//!   "id": "rerank-...",
//!   "model": "qwen3-reranker-0.6b",
//!   "results": [
//!     {"index": 0, "relevance_score": 0.938},
//!     ...
//!   ]
//! }
//! ```
//!
//! Each (query, document) pair is encoded as the standard
//! Qwen3-Reranker chat prompt and run as a single causal prefill
//! (positions 0..=T). The final-token post-RMSNorm hidden state is
//! scored through the model's 2-class classification head; the
//! "yes" logit drives `relevance_score` (softmaxed to [0, 1]).
//!
//! Performance: per-request = fresh `Qwen3Session` (cheap, allocations
//! only). A 76-token prefill + 1 head matmul ≈ 200ms on 4 threads.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde::{Deserialize, Serialize};

use super::{AppState, Backend};
use crate::models::qwen3::trunk::{
    qwen_text_positions, Qwen3Input, Qwen3Model, Qwen3Session,
};

#[derive(Deserialize)]
pub struct RerankRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub query: String,
    pub documents: Vec<String>,
    #[serde(default)]
    pub top_n: Option<usize>,
    #[serde(default)]
    pub return_documents: Option<bool>,
    #[serde(default)]
    pub max_tokens_per_doc: Option<usize>,
}

#[derive(Serialize)]
struct RerankItem {
    index: usize,
    relevance_score: f32,
}

#[derive(Serialize)]
struct RerankResponse {
    id: String,
    model: String,
    results: Vec<RerankItem>,
}

#[derive(Serialize)]
struct RerankError {
    error: String,
}

const DEFAULT_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

fn render_prompt(instruction: &str, query: &str, document: &str) -> String {
    format!(
        "system\nJudge whether the Document meets the requirements based on \
         the Query and the Instruct provided. Note that the answer can only be \
         \"yes\" or \"no\".\nuser\n<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}\n"
    )
}

/// Soft word-based truncation. The model context window is much
/// larger than typical RAG passages, but pre-truncating to ~512 tokens
/// keeps the per-doc prefill cost bounded.
fn truncate_words(text: &str, max_tokens: usize) -> String {
    if max_tokens == 0 {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len().min(text.len()));
    let mut count = 0usize;
    for word in text.split_whitespace() {
        if count > 0 {
            out.push(' ');
        }
        out.push_str(word);
        count += 1;
        if count >= max_tokens {
            break;
        }
    }
    out
}

/// `clamp(x, lo, hi)` helper kept local to avoid pulling extra deps.
fn clamp_f32(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}

fn softmax_pair(a: f32, b: f32) -> (f32, f32) {
    let m = a.max(b);
    let ea = (a - m).exp();
    let eb = (b - m).exp();
    let s = ea + eb;
    (ea / s, eb / s)
}

fn score_one_doc(
    raw: &'static Qwen3Model,
    tokenizer: &crate::core::tokenizer::BPETokenizer,
    context_length: usize,
    query: &str,
    document: &str,
    instruction: &str,
) -> Result<f32, String> {
    let prompt = render_prompt(instruction, query, document);
    let token_ids = tokenizer.encode(
        &prompt,
        crate::core::tokenizer::EncodeOptions {
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
    // Per-request session. KV cache is per-session and discarded
    // after the request; prefill_batch_size pushes the whole prompt
    // in one chunk (single-pass forward).
    let mut session = Qwen3Session::new(raw, n + 4)
        .map_err(|e| format!("rerank session: {e}"))?;
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
    let logits = raw
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
    let (yes_prob, _) = softmax_pair(yes, no);
    Ok(clamp_f32(yes_prob, 0.0, 1.0))
}

pub async fn rerank(
    State(state): State<AppState>,
    Json(req): Json<RerankRequest>,
) -> impl axum::response::IntoResponse {
    let Backend::Rerank(backend) = state.model.as_ref() else {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(RerankError {
                error: "server is not configured for rerank".into(),
            }),
        )
            .into_response();
    };
    if req.documents.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(RerankError {
                error: "documents must be non-empty".into(),
            }),
        )
            .into_response();
    }
    if req.query.trim().is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(RerankError {
                error: "query must be non-empty".into(),
            }),
        )
            .into_response();
    }

    // Pre-truncate documents to keep prefill bounded. `0` → unlimited.
    let max_words = req.max_tokens_per_doc.unwrap_or(512);
    let truncated: Vec<String> = req
        .documents
        .iter()
        .map(|d| truncate_words(d, max_words))
        .collect();

    let instruction = DEFAULT_INSTRUCTION;

    // Run scoring sequentially. Per-doc sessions are independent; a
    // future batched implementation could prefill all (query+doc_i)
    // prompts in one graph for ~Nx throughput.
    let raw: &'static Qwen3Model = *backend.model;
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(truncated.len());
    for (idx, doc) in truncated.iter().enumerate() {
        match score_one_doc(
            raw,
            &backend.tokenizer,
            backend.context_length,
            &req.query,
            doc,
            instruction,
        ) {
            Ok(score) => scored.push((idx, score)),
            Err(e) => {
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Json(RerankError { error: e }),
                )
                    .into_response();
            }
        }
    }

    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let top_n = req.top_n.unwrap_or(scored.len()).min(scored.len());
    let results: Vec<RerankItem> = scored
        .into_iter()
        .take(top_n)
        .map(|(index, relevance_score)| RerankItem {
            index,
            relevance_score,
        })
        .collect();

    let resp = RerankResponse {
        id: format!("rerank-{}", short_id()),
        model: req.model.unwrap_or_else(|| state.model_name.clone()),
        results,
    };
    Json(resp).into_response()
}

fn short_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{:x}-{:x}", secs & 0xffff, nanos & 0xffff_ffff)
}

// Re-export the Backend enum from the parent module so we can pattern
// match on its variants without exposing it elsewhere.