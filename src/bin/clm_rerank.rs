//! CLM rerank CLI: score candidates against a state with CLM projection
//! heads on top of a Qwen3 encoder.
//!
//! CLM is a scorer, not a generator: each side is embedded by the shared
//! encoder (Qwen3-8B, last-token pooled), pushed through its own MLP head
//! and L2-normalised, and the pair score is
//! `logit_scale * cos(state_head(state), action_head(candidate))`.
//!
//! Usage:
//!     clm_rerank --model <qwen3-8b.gguf> --clm-head <clm-heads.gguf> \
//!                --state "Customer: my invoice was charged twice" \
//!                --cand "billing" --cand "bug"
//!
//! Prints each candidate with its score, sorted descending.

use std::path::Path;
use std::sync::Arc;

use rust_model_inference::core::thread_pool::ComputePool;
use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};
use rust_model_inference::{GGUFLoader, TensorSource};
use rust_model_inference::models::clm::ClmHeads;
use rust_model_inference::models::qwen3::trunk::{
    qwen_text_positions, Qwen3Input, Qwen3Model, Qwen3Session,
};

const USAGE: &str = "\
Usage:
  clm_rerank --model <encoder.gguf> --clm-head <heads.gguf> --state <text> --cand <text> [--cand <text> ...]

Options:
  --model <PATH>     Encoder GGUF (Qwen3; CLM-v0.1-8B expects Qwen3-8B)
  --clm-head <PATH>  CLM head GGUF from tools/converter/clm/convert_clm.py
  --state <TEXT>     The state description
  --cand <TEXT>      A candidate (repeatable)
  --candidates <F>   Read candidates from a file, newline-separated
  --threads N        ComputePool parallelism (default 4)
";

#[derive(Default)]
struct Opts {
    model: Option<String>,
    clm_head: Option<String>,
    state: Option<String>,
    cands: Vec<String>,
    candidates: Option<String>,
    threads: usize,
}

fn parse_args() -> Result<Opts, String> {
    let mut o = Opts { threads: 4, ..Default::default() };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => o.model = it.next(),
            "--clm-head" => o.clm_head = it.next(),
            "--state" => o.state = it.next(),
            "--cand" => o.cands.push(it.next().ok_or("--cand needs a value")?),
            "--candidates" => o.candidates = it.next(),
            "--threads" => {
                o.threads = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--threads needs a number")?
            }
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument {other}\n\n{USAGE}")),
        }
    }
    Ok(o)
}

fn embed(
    tokenizer: &BPETokenizer,
    model: &Qwen3Model,
    text: &str,
) -> Result<Vec<f32>, String> {
    // No chat template: CLM's encoder was trained on raw text, and the
    // reference embeds the state/candidate as-is.
    let token_ids = tokenizer.encode(
        text,
        EncodeOptions { add_special: false, parse_special: false },
    );
    if token_ids.is_empty() {
        return Err("empty input".into());
    }
    let positions = qwen_text_positions(token_ids.len());
    let capacity = token_ids.len() + 4;
    let mut session = Qwen3Session::new(model, capacity).map_err(|e| format!("session: {e}"))?;
    session
        .forward_last_hidden(
            Qwen3Input {
                token_ids: &token_ids,
                positions: &positions,
                embeddings: None,
                deepstack_embeddings: None,
            },
            token_ids.len(),
        )
        .map_err(|e| format!("forward: {e}"))
}

fn main() -> Result<(), String> {
    let opts = parse_args()?;
    let (model_path, head_path, state) = match (opts.model, opts.clm_head, opts.state) {
        (Some(m), Some(h), Some(s)) => (m, h, s),
        _ => return Err(format!("--model, --clm-head and --state are required\n\n{USAGE}")),
    };

    let mut cands = opts.cands;
    if let Some(path) = &opts.candidates {
        let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
        cands.extend(
            text.lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .map(str::to_string),
        );
    }
    if cands.is_empty() {
        return Err(format!("at least one --cand or --candidates is required\n\n{USAGE}"));
    }

    if !Path::new(&head_path).exists() {
        return Err(format!("CLM head GGUF not found: {head_path}\n\n{USAGE}"));
    }
    let head_source: Box<dyn TensorSource> =
        open_model_source(Path::new(&head_path), ComponentRole::Llm)
            .map_err(|e| format!("open clm heads: {e}"))?;
    let heads = ClmHeads::from_source(head_source.as_ref())?;

    let loader = GGUFLoader::from_file(Path::new(&model_path)).map_err(|e| format!("open model: {e}"))?;
    let tokenizer = Arc::new(
        BPETokenizer::from_gguf_metadata(|k| loader.metadata(k).cloned())
            .map_err(|e| format!("init tokenizer: {e}"))?,
    );
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let pool = Arc::new(ComputePool::new(opts.threads.max(1)));
    let model = Qwen3Model::from_source(source, Arc::clone(&tokenizer), pool)
        .map_err(|e| format!("load model: {e}"))?;
    let config = model.config().clone();
    if config.architecture != "qwen3" {
        return Err(format!(
            "expected a qwen3 encoder, got {}",
            config.architecture
        ));
    }
    if config.n_embd != heads.encoder_dim() {
        return Err(format!(
            "encoder hidden size {} does not match the CLM heads (expect {})",
            config.n_embd,
            heads.encoder_dim()
        ));
    }

    eprintln!(
        "clm_rerank: encoder {}x{} + CLM heads ({} candidates)",
        config.n_layer, config.n_embd, cands.len()
    );

    let mut scratch: Vec<f32> = Vec::new();
    let z_state = heads.project_state(&embed(&tokenizer, &model, &state)?, &mut scratch)?;
    let mut scored: Vec<(usize, f32, String)> = Vec::with_capacity(cands.len());
    for (i, cand) in cands.iter().enumerate() {
        let e = embed(&tokenizer, &model, cand)?;
        let z = heads.project_candidate(&e, &mut scratch)?;
        scored.push((i, heads.score(&z_state, &z), cand.clone()));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (rank, (_, score, cand)) in scored.iter().enumerate() {
        println!("{}. {score:.4}  {cand}", rank + 1);
    }
    Ok(())
}
