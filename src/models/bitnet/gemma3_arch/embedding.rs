//! Gemma3-architecture BitNet embedding extraction.
//!
//! Mirrors `crate::models::qwen3::embedding` but routes through the
//! gemma3 BitNet trunk instead. Used by
//! `crate::models::bitnet::compute_embedding` for
//! `microsoft/bitnet-embedding-270m` (the only currently-supported
//! gemma3-arch BitNet model).

use std::time::Instant;

use crate::app::cli::EmbeddingOutput;
use crate::core::loader::GGUFLoader;
use crate::core::scratchpad::KvFormat;
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::SPMTokenizer;

use super::forward::text_encode;
use super::weights::{load_layers_static, static_weight, Gemma3LayerWeights, Gemma3Model};

/// `general` file path (not directly recoverable from the
/// `TensorSource` trait object — the loader is given in `load_model`).
fn source_path_unused() {}

/// Build the gemma3-arch BitNet model from a `TensorSource`. The
/// returned model is fully owned (no borrowed `TensorSource`
/// lifetime) and can outlive the CLI's temporary source borrow.
pub fn load_model(source: &dyn TensorSource) -> Result<Gemma3Model, String> {
    let config = super::config::build_config(source)?;
    let layers: Vec<Gemma3LayerWeights<'static>> = load_layers_static(source, &config);
    let output_norm = super::weights::get_f32_tensor(
        source,
        "output_norm.weight",
        config.n_embd,
    );
    let token_embedding_rows = static_weight(source, "token_embd.weight")
        .map_err(|e| format!("bitnet::gemma3_arch: token_embd.weight load failed: {e}"))?;
    Ok(Gemma3Model {
        config,
        layers,
        output_norm,
        token_embedding_rows,
    })
}

/// Tokenize + run embedding forward for `arch=gemma3` BitNet.
pub fn compute_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    let _ = n_threads_arg;
    let tokenizer = SPMTokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|e| format!("bitnet::gemma3_arch::compute_embedding: tokenizer init failed: {e}"))?;
    let prompt_tokens = encode_embedding_input(&tokenizer, prompt);
    if prompt_tokens.is_empty() {
        return Err("bitnet::gemma3_arch::compute_embedding: empty token sequence".into());
    }
    run_embedding_tokens(source, &prompt_tokens)
}

fn encode_embedding_input(tokenizer: &SPMTokenizer, prompt: &str) -> Vec<u32> {
    tokenizer
        .encode(
            prompt,
            crate::core::tokenizer::EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        )
        .into_iter()
        .filter(|&id| id != tokenizer.eos_id().unwrap_or(u32::MAX))
        .collect()
}

/// Run the embedding extraction given pre-tokenized input.
pub fn run_embedding_tokens(
    source: &dyn TensorSource,
    token_ids: &[u32],
) -> Result<Vec<f32>, String> {
    if token_ids.is_empty() {
        return Err("bitnet::gemma3_arch::run_embedding_tokens: empty token sequence".into());
    }
    let model = load_model(source)?;
    text_encode(&model, token_ids)
}

/// CLI entry point used by `app::run_embedding` for
/// `arch=gemma3`. Mirrors `crate::models::qwen3::embedding::run_embedding`.
pub fn run_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
    _kv_format: KvFormat,
    output: EmbeddingOutput,
) {
    let started = Instant::now();
    match compute_embedding(source, prompt, n_threads_arg) {
        Ok(pooled) => {
            let elapsed = started.elapsed().as_millis();
            let n_layers = source
                .metadata("gemma3.block_count")
                .and_then(|v| v.to_u64())
                .map(|v| v as usize)
                .unwrap_or(0);
            super::super::embedding::print_embedding_for_arch(
                &pooled,
                output,
                elapsed,
                "gemma3",
                n_layers,
            );
        }
        Err(error) => {
            eprintln!("bitnet::gemma3_arch::run_embedding failed: {error}");
        }
    }
}