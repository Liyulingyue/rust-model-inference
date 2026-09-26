//! Standalone Qwen3-Reranker CLI.
//!
//! Usage:
//!     qwen3_rerank --model <rerank.gguf> \
//!                 --query "<question>" \
//!                 --documents <path>          # newline-separated
//!     qwen3_rerank --model <rerank.gguf> \
//!                 --query "<question>" \
//!                 --doc "<d1>" --doc "<d2>"    # repeated
//!
//! Prints each document with its cross-encoder relevance logit, sorted
//! by score (descending). The score for a (query, doc) pair is the
//! "yes" logit from the model's 2-class classification head applied to
//! the last token's hidden state after the standard prompt:
//!
//!     system: "Judge whether the Document meets the requirements
//!              based on the Query and the Instruct provided. Note that
//!              the answer can only be "yes" or "no.""
//!     user:   "<Instruct>: <instruction>
//!             <Query>: <query>
//!             <Document>: <document>"
//!
//! Run with --threads N to control ComputePool parallelism.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use rust_model_inference::core::thread_pool::ComputePool;
use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
use rust_model_inference::models::qwen3::trunk::{
    Qwen3Config, Qwen3Input, Qwen3Model, Qwen3Session,
};
use rust_model_inference::{GGUFLoader, TensorSource};

const USAGE: &str = "\
Usage:
  qwen3_rerank --model <rerank.gguf> --query <text> --documents <path>
  qwen3_rerank --model <rerank.gguf> --query <text> --doc <text> [--doc <text> ...]

Options:
  --model <PATH>         GGUF model (must be qwen3 with cls.output.weight)
  --query <TEXT>         The retrieval query
  --documents <PATH>     Read candidate documents from file, newline-separated
  --doc <TEXT>           Inline document (repeatable)
  --threads N            ComputePool parallelism (default 4)
  --max-tokens N         Truncate documents to N whitespace-separated tokens
                         (0 = unlimited; default 512)
  --instruction <TEXT>   Override the default system instruction
  --verbose              Print per-document scores to stderr";

const DEFAULT_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

fn parse_args() -> Result<Opts, String> {
    let mut opts = Opts::default();
    let mut docs: Vec<String> = Vec::new();
    let mut args: VecDeque<String> = std::env::args().skip(1).collect();
    while let Some(a) = args.pop_front() {
        let v = |args: &mut VecDeque<String>, name: &str| -> Result<String, String> {
            args.pop_front()
                .ok_or_else(|| format!("missing value for {name}"))
        };
        match a.as_str() {
            "--model" => opts.model = Some(PathBuf::from(v(&mut args, "--model")?)),
            "--query" => opts.query = Some(v(&mut args, "--query")?),
            "--documents" => {
                let path = PathBuf::from(v(&mut args, "--documents")?);
                let text = fs::read_to_string(&path).map_err(|e| {
                    format!("read documents file {}: {e}", path.display())
                })?;
                for chunk in text.split('\n') {
                    if !chunk.is_empty() {
                        opts.docs.push(chunk.to_string());
                    }
                }
            }
            "--doc" => opts.docs.push(v(&mut args, "--doc")?),
            "--threads" => opts.threads = v(&mut args, "--threads")?.parse().map_err(|e| {
                format!("invalid --threads value: {e}")
            })?,
            "--max-tokens" => opts.max_tokens = v(&mut args, "--max-tokens")?
                .parse()
                .map_err(|e| format!("invalid --max-tokens: {e}"))?,
            "--instruction" => opts.instruction = Some(v(&mut args, "--instruction")?),
            "--verbose" => opts.verbose = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}\n\n{USAGE}")),
        }
        // Keep `args` alias consistent (above v closure used &mut args).
        let _ = docs;
    }
    if opts.model.is_none() {
        return Err(format!("--model is required\n\n{USAGE}"));
    }
    if opts.query.is_none() {
        return Err(format!("--query is required\n\n{USAGE}"));
    }
    if opts.docs.is_empty() {
        return Err(format!(
            "at least one document required (use --documents <file> or repeated --doc <text>)\n\n{USAGE}"
        ));
    }
    Ok(opts)
}

#[derive(Default)]
struct Opts {
    model: Option<PathBuf>,
    query: Option<String>,
    docs: Vec<String>,
    threads: usize,
    max_tokens: usize,
    instruction: Option<String>,
    verbose: bool,
}

/// Truncate `text` to roughly `max_tokens` whitespace-separated tokens
/// (0 = unlimited). Used to keep long documents from blowing up the
/// context window; this is a heuristic, not a tokenizer-accurate limit.
fn truncate_tokens(text: &str, max_tokens: usize) -> String {
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

/// Render the standard Qwen3-Reranker prompt template.
fn render_prompt(instruction: &str, query: &str, document: &str) -> String {
    format!(
        "system\nJudge whether the Document meets the requirements based on \
         the Query and the Instruct provided. Note that the answer can only be \
         \"yes\" or \"no\".\nuser\n<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}\n"
    )
}

fn main() {
    if let Err(e) = run() {
        eprintln!("qwen3_rerank: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let opts = parse_args()?;
    let model_path = opts.model.clone().unwrap();
    let query = opts.query.clone().unwrap();
    let instruction = opts
        .instruction
        .clone()
        .unwrap_or_else(|| DEFAULT_INSTRUCTION.to_string());

    eprintln!("qwen3_rerank: loading {}", model_path.display());
    let loader = GGUFLoader::from_file(model_path.to_str().ok_or("non-utf8 model path")?)
        .map_err(|e| format!("open GGUF: {e}"))?;
    let tokenizer = Arc::new(
        BPETokenizer::from_gguf_metadata(|k| loader.metadata(k).cloned())
            .map_err(|e| format!("init tokenizer: {e}"))?,
    );
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let threads = opts.threads.max(1);
    let pool = Arc::new(ComputePool::new(threads));
    let tokenizer_for_model = Arc::clone(&tokenizer);
    let model = Qwen3Model::from_source(source, tokenizer_for_model, pool)
        .map_err(|e| format!("load model: {e}"))?;
    let config = model.config().clone();
    if config.architecture != "qwen3" {
        return Err(format!(
            "expected qwen3 architecture, got {}",
            config.architecture
        ));
    }
    if !model.is_rerank() {
        return Err(
            "model has no cls.output.weight — not a rerank model (need ggml-org/Qwen3-Reranker-*-Q8_0-GGUF or similar)".to_string(),
        );
    }
    let n_embd = config.n_embd;

    // Cap documents at model context length minus the prompt overhead.
    let max_doc_tokens = if opts.max_tokens > 0 {
        opts.max_tokens
    } else {
        512
    };
    let docs: Vec<String> = opts
        .docs
        .into_iter()
        .map(|d| truncate_tokens(&d, max_doc_tokens))
        .collect();

    // One session reused across all docs (reset KV between calls). The
    // per-doc prompt has variable length, but `new_with_kv_state`
    // creates a fresh state to avoid carry-over between docs.
    eprintln!(
        "qwen3_rerank: model loaded ({} threads); scoring {} documents",
        threads,
        docs.len()
    );

    let mut scored: Vec<(usize, f32, f32)> = Vec::with_capacity(docs.len());
    let mut stderr = std::io::stderr().lock();
    for (idx, doc) in docs.iter().enumerate() {
        let prompt = render_prompt(&instruction, &query, doc);
        let token_ids = tokenizer.encode(
            &prompt,
            EncodeOptions {
                add_special: false,
                parse_special: false,
            },
        );
        let positions = rust_model_inference::models::qwen3::trunk::qwen_text_positions(token_ids.len());
        let session_capacity = token_ids.len() + 4;
        let mut session = Qwen3Session::new(&model, session_capacity)
            .map_err(|e| format!("session {idx}: {e}"))?;
        let last_hidden = session
            .forward_rerank(
                Qwen3Input {
                    token_ids: &token_ids,
                    positions: &positions,
                    embeddings: None,
                    deepstack_embeddings: None,
                },
                token_ids.len(),
            )
            .map_err(|e| format!("forward {idx}: {e}"))?;
        if last_hidden.len() != n_embd {
            return Err(format!(
                "doc {idx}: hidden size {} (expected {})",
                last_hidden.len(),
                n_embd
            ));
        }
        let cls_logits = model
            .score_logits(&last_hidden)
            .map_err(|e| format!("score_logits {idx}: {e}"))?;
        // cls_logits = [logit_yes, logit_no]; relevance = logit_yes.
        let relevance = cls_logits.first().copied().unwrap_or(f32::NEG_INFINITY);
        let mut yes_prob = 0.0f32;
        if cls_logits.len() >= 2 {
            let max = cls_logits[0].max(cls_logits[1]);
            let e0 = (cls_logits[0] - max).exp();
            let e1 = (cls_logits[1] - max).exp();
            yes_prob = e0 / (e0 + e1);
        }
        if opts.verbose {
            let _ = writeln!(
                stderr,
                "[{}] yes_logit={:.4} no_logit={:.4} yes_prob={:.4}",
                idx, relevance,
                cls_logits.get(1).copied().unwrap_or(0.0),
                yes_prob
            );
        }
        scored.push((idx, relevance, yes_prob));
    }

    // Sort by relevance desc; preserve input order for ties.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "rank\tidx\trelevance_logit\tyes_prob");
    for (rank, (idx, relevance, yes_prob)) in scored.iter().enumerate() {
        let _ = writeln!(stdout, "{rank}\t{idx}\t{relevance:.6}\t{yes_prob:.6}");
    }
    let _ = writeln!(stdout, "\n--- documents ---");
    for (rank, (idx, relevance, yes_prob)) in scored.iter().enumerate() {
        let snippet: String = docs[*idx]
            .chars()
            .take(120)
            .collect::<String>()
            .replace('\n', " ");
        let _ = writeln!(
            stdout,
            "[rank={rank} idx={idx} score={relevance:.3} yes={yes_prob:.3}] {snippet}{}",
            if docs[*idx].chars().count() > 120 { "..." } else { "" }
        );
    }
    Ok(())
}