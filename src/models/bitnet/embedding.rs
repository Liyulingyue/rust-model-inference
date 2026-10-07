//! BitNet-family embedding entry points.
//!
//! The two BitNet trunks ([`qwen3_arch`] and [`gemma3_arch`])
//! share the same embedding CLI / HTTP interface. Both routes
//! produce raw (unnormalized) dense embeddings; downstream cosine
//! users normalize themselves.
//!
//! # Public surface
//!
//! - [`compute_embedding`] — entry point that auto-dispatches on
//!   `general.architecture` (qwen3 → qwen3_arch; gemma3 →
//!   gemma3_arch). Used by `app::run_embedding`'s bitnet arms.
//! - [`run_embedding`] — CLI-facing wrapper around
//!   [`compute_embedding`] that prints results.
//! - [`run_embedding_tokens`] — pre-tokenized variant.
//! - [`print_embedding_for_arch`] — shared printer that
//!   emits the same `Embedding (N dims, M layers, arch=X ms)`
//!   summary line regardless of the underlying trunk.

use std::time::Instant;

use crate::app::cli::EmbeddingOutput;
use crate::core::scratchpad::KvFormat;
use crate::core::tensor::TensorSource;

/// Architecture-aware compute_embedding. Dispatches to the right
/// BitNet trunk based on `general.architecture`.
pub fn compute_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or("");
    match arch {
        "qwen3" => super::qwen3_arch::compute_embedding(source, prompt, n_threads_arg),
        "gemma3" => super::gemma3_arch::compute_embedding(source, prompt, n_threads_arg),
        other => Err(format!(
            "bitnet::compute_embedding: unsupported arch {other:?}; \
             expected qwen3 or gemma3"
        )),
    }
}

/// Architecture-aware run_embedding_tokens.
pub fn run_embedding_tokens(
    source: &dyn TensorSource,
    token_ids: &[u32],
) -> Result<Vec<f32>, String> {
    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or("");
    match arch {
        "qwen3" => super::qwen3_arch::run_embedding_tokens(source, token_ids),
        "gemma3" => super::gemma3_arch::run_embedding_tokens(source, token_ids),
        other => Err(format!(
            "bitnet::run_embedding_tokens: unsupported arch {other:?}; \
             expected qwen3 or gemma3"
        )),
    }
}

/// CLI entry point used by `app::run_embedding` for both
/// `arch=qwen3`+BitNet and `arch=gemma3`.
pub fn run_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
    kv_format: KvFormat,
    output: EmbeddingOutput,
) {
    let started = Instant::now();
    match compute_embedding(source, prompt, n_threads_arg) {
        Ok(pooled) => {
            let elapsed = started.elapsed().as_millis();
            let arch = source
                .metadata("general.architecture")
                .and_then(|v| v.to_string_val())
                .unwrap_or("bitnet")
                .to_string();
            let n_layers = source
                .metadata(&format!("{arch}.block_count"))
                .and_then(|v| v.to_u64())
                .map(|v| v as usize)
                .unwrap_or(0);
            print_embedding_for_arch(&pooled, output, elapsed, &arch, n_layers);
        }
        Err(error) => {
            eprintln!("bitnet::run_embedding failed: {error}");
        }
    }
    let _ = kv_format;
}

/// Backwards-compatible alias for the old `qwen3::embedding::print_embedding`
/// shape (which took only `(pooled, output, elapsed_ms)` and read
/// n_layers / arch from elsewhere). New callers should prefer
/// [`print_embedding_for_arch`].
pub fn print_embedding(pooled: &[f32], output: EmbeddingOutput, elapsed_ms: u128) {
    print_embedding_for_arch(pooled, output, elapsed_ms, "bitnet", pooled.len());
}

/// Shared embedding printer used by both `qwen3_arch::run_embedding`
/// and `gemma3_arch::run_embedding`. Emits the canonical
/// `Embedding (N dims, M layers, arch=X ms): ...` summary or the
/// `embedding_raw: ...` raw output.
pub fn print_embedding_for_arch(
    pooled: &[f32],
    output: EmbeddingOutput,
    elapsed_ms: u128,
    arch: &str,
    n_layers: usize,
) {
    match output {
        EmbeddingOutput::Summary => {
            println!(
                "Embedding ({} dims, {} layers, arch={} {}ms):",
                pooled.len(),
                n_layers,
                arch,
                elapsed_ms
            );
            let preview = pooled.len().min(8);
            let mut first_part = pooled[..preview].to_vec();
            let mut last_part: Vec<f32> = if pooled.len() > preview {
                pooled[pooled.len() - 4..].to_vec()
            } else {
                vec![]
            };
            for v in &mut first_part {
                if v.abs() < 1e-9 {
                    *v = 0.0;
                }
            }
            for v in &mut last_part {
                if v.abs() < 1e-9 {
                    *v = 0.0;
                }
            }
            let first_str: Vec<String> = first_part.iter().map(|v| format!("{v:.9}")).collect();
            let last_str: Vec<String> = last_part.iter().map(|v| format!("{v:.9}")).collect();
            println!("{} ... {}", first_str.join(" "), last_str.join(" "));
        }
        EmbeddingOutput::Raw => {
            let s: Vec<String> = pooled.iter().map(|v| format!("{v:.9}")).collect();
            println!("embedding_raw: {}", s.join(" "));
        }
    }
}
