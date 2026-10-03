//! Standalone jina-bert-v2 rerank CLI.
//!
//! Usage:
//!     jina_rerank --model <rerank.gguf> \
//!                 --query "<question>" \
//!                 --documents <path>          # newline-separated
//!     jina_rerank --model <rerank.gguf> \
//!                 --query "<question>" \
//!                 --doc "<d1>" --doc "<d2>"    # repeated
//!
//! Prints each document with its cross-encoder relevance logit (and the
//! matching sigmoid score in [0, 1]), sorted by logit descending. The
//! score for a (query, doc) pair is the single-logit projection of the
//! CLS-token row through `cls.weight` + `cls.bias` after a bidirectional
//! forward over `[BOS] query [EOS] [SEP] doc [EOS]` — matching
//! `references/llama.cpp/tools/server/server-common.cpp:1817-1830`.
//!
//! Run with --threads N to control ComputePool parallelism.

use std::collections::VecDeque;
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;

use rust_model_inference::models::bert_family::compute_rerank_score;
use rust_model_inference::GGUFLoader;

const USAGE: &str = "\
Usage:
  jina_rerank --model <rerank.gguf> --query <text> --documents <path>
  jina_rerank --model <rerank.gguf> --query <text> --doc <text> [--doc <text> ...]

Options:
  --model <PATH>         GGUF model (must be jina-bert-v2 with cls.weight + cls.bias)
  --query <TEXT>         The retrieval query
  --documents <PATH>     Read candidate documents from file, newline-separated
  --doc <TEXT>           Inline document (repeatable)
  --threads N            ComputePool parallelism (default 4)
  --max-tokens N         Truncate documents to N whitespace-separated tokens
                         (0 = unlimited; default 512)
  --verbose              Print per-document scores to stderr";

fn parse_args() -> Result<Opts, String> {
    let mut opts = Opts::default();
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
                let text = fs::read_to_string(&path)
                    .map_err(|e| format!("read documents file {}: {e}", path.display()))?;
                for chunk in text.split('\n') {
                    if !chunk.is_empty() {
                        opts.docs.push(chunk.to_string());
                    }
                }
            }
            "--doc" => opts.docs.push(v(&mut args, "--doc")?),
            "--threads" => {
                opts.threads = v(&mut args, "--threads")?
                    .parse()
                    .map_err(|e| format!("invalid --threads value: {e}"))?
            }
            "--max-tokens" => {
                opts.max_tokens = v(&mut args, "--max-tokens")?
                    .parse()
                    .map_err(|e| format!("invalid --max-tokens: {e}"))?
            }
            "--verbose" => opts.verbose = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}\n\n{USAGE}")),
        }
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
    verbose: bool,
}

/// Soft word-based truncation; `0` = unlimited.
fn truncate_tokens(text: &str, max_tokens: usize) -> String {
    if max_tokens == 0 {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
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

fn main() {
    if let Err(e) = run() {
        eprintln!("jina_rerank: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let opts = parse_args()?;
    let model_path = opts.model.clone().unwrap();
    let query = opts.query.clone().unwrap();

    eprintln!("jina_rerank: loading {}", model_path.display());
    let loader = GGUFLoader::from_file(model_path.to_str().ok_or("non-utf8 model path")?)
        .map_err(|e| format!("open GGUF: {e}"))?;
    let arch = loader
        .metadata("general.architecture")
        .and_then(rust_model_inference::MetaValue::to_string_val)
        .unwrap_or_default();
    if arch != "jina-bert-v2" {
        return Err(format!("expected jina-bert-v2 architecture, got {arch}"));
    }
    if loader.tensor_info("cls.weight").is_none() || loader.tensor_info("cls.bias").is_none() {
        return Err(
            "model has no cls.weight + cls.bias — not a jina rerank GGUF (need ggml-org/jina-reranker-v1-turbo-en-GGUF or similar)".to_string(),
        );
    }

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

    eprintln!(
        "jina_rerank: model loaded ({} threads); scoring {} documents",
        opts.threads,
        docs.len()
    );

    let scores = compute_rerank_score(&loader, &query, &docs, opts.threads.max(1))?;
    assert_eq!(scores.len(), docs.len());

    let mut scored: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut stderr = std::io::stderr().lock();
    if opts.verbose {
        for (idx, s) in scores.iter().enumerate() {
            let sigmoid = 1.0f32 / (1.0f32 + (-*s).exp());
            let _ = writeln!(
                stderr,
                "[doc {idx}] cls_logit={:.6} sigmoid={:.4}",
                s, sigmoid
            );
        }
    }

    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "rank\tidx\tcls_logit\tsigmoid");
    for (rank, (idx, logit)) in scored.iter().enumerate() {
        let sigmoid = 1.0f32 / (1.0f32 + (-*logit).exp());
        let _ = writeln!(stdout, "{rank}\t{idx}\t{logit:.6}\t{sigmoid:.4}");
    }
    let _ = writeln!(stdout, "\n--- documents ---");
    for (rank, (idx, logit)) in scored.iter().enumerate() {
        let sigmoid = 1.0f32 / (1.0f32 + (-*logit).exp());
        let snippet: String = docs[*idx]
            .chars()
            .take(120)
            .collect::<String>()
            .replace('\n', " ");
        let _ = writeln!(
            stdout,
            "[rank={rank} idx={idx} logit={logit:.3} sigmoid={sigmoid:.3}] {snippet}{}",
            if docs[*idx].chars().count() > 120 {
                "..."
            } else {
                ""
            }
        );
    }
    Ok(())
}
