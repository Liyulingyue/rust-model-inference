//! Last-token embedding for BGE decoder checkpoints.
//!
//! BGE's Mistral, Qwen2 and Gemma2 variants use the ordinary decoder hidden
//! state, followed by the model output RMSNorm and L2 normalization.  The
//! llama session already owns the causal attention, KV cache and architecture
//! specific FFN details, so this module only supplies the embedding contract.

use crate::app::cli::{EmbeddingOutput, KvFormat};
use crate::core::loader::model_config_from_source;
use crate::core::prefill::ChunkedPrefill;
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::{load_tokenizer, EncodeOptions};
use crate::models::llama::trunk::LlamaSession;
use crate::models::qwen3::embedding::l2_normalize_embedding as normalize;
use crate::ops::rms_norm_grouped;
use std::time::Instant;

pub fn compute_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    compute_embedding_with_kv(source, prompt, n_threads_arg, KvFormat::F16)
}

fn compute_embedding_with_kv(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
    kv_format: KvFormat,
) -> Result<Vec<f32>, String> {
    let arch = source
        .metadata("general.architecture")
        .and_then(|value| value.to_string_val())
        .unwrap_or_default();
    if !matches!(arch, "llama" | "qwen2" | "gemma2") {
        return Err(format!(
            "decoder embedding does not support architecture {arch:?}"
        ));
    }
    let tokenizer = load_tokenizer(|key| source.metadata(key).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
    let tokens = tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    if tokens.is_empty() {
        return Err("Embedding input produced no tokens".into());
    }
    #[cfg(feature = "parity-trace")]
    if crate::parity_trace::enabled("embedding.tokens") {
        crate::parity_trace::token_ids("embedding.tokens", &tokens).map_err(|e| e.to_string())?;
    }
    let config = model_config_from_source(source)?;
    if tokens.len() > config.n_ctx {
        return Err(format!(
            "Embedding input has {} tokens, exceeding context length {}",
            tokens.len(),
            config.n_ctx
        ));
    }

    // Limit the cache to this request.  A full 32K/40K decoder cache is
    // needlessly large for one embedding and can exceed a laptop's memory.
    let mut session = LlamaSession::from_source_with_max_rows(
        source,
        n_threads_arg,
        kv_format,
        tokens.len().max(1),
        1,
    )
    .map_err(|error| format!("Decoder embedding session init failed: {error}"))?;
    session
        .prefill(&tokens, 1)
        .map_err(|error| format!("Decoder embedding prefill failed: {error}"))?;

    let n_embd = config.n_embd;
    let mut pooled = vec![0.0f32; n_embd];
    rms_norm_grouped(
        &session.scratch.x[..n_embd],
        &session.weights.output_norm,
        &mut pooled,
        session.norm_groups,
        config.norm_eps,
    );
    #[cfg(feature = "parity-trace")]
    if crate::parity_trace::enabled("embedding.pooled") {
        crate::parity_trace::checkpoint("embedding.pooled", None, &[n_embd], &pooled)
            .map_err(|e| e.to_string())?;
    }
    normalize(&mut pooled)?;
    #[cfg(feature = "parity-trace")]
    if crate::parity_trace::enabled("embedding.final") {
        crate::parity_trace::checkpoint("embedding.final", None, &[n_embd], &pooled)
            .map_err(|e| e.to_string())?;
    }
    Ok(pooled)
}

pub fn run_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
    kv_format: KvFormat,
    output: EmbeddingOutput,
) {
    let arch = source
        .metadata("general.architecture")
        .and_then(|value| value.to_string_val())
        .unwrap_or_default();
    let started = Instant::now();
    let embedding = match compute_embedding_with_kv(source, prompt, n_threads_arg, kv_format) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    match output {
        EmbeddingOutput::Summary => {
            print!(
                "Embedding ({} dims, arch={} {}ms):",
                embedding.len(),
                arch,
                started.elapsed().as_millis()
            );
            for value in embedding.iter().take(8) {
                print!(" {value:.9}");
            }
            if embedding.len() > 8 {
                print!(" ...");
            }
            println!();
        }
        EmbeddingOutput::Raw => {
            print!("embedding_raw:");
            for value in embedding {
                print!(" {value:.9}");
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::normalize;

    #[test]
    fn normalize_is_unit_length() {
        let mut values = [3.0f32, 4.0];
        normalize(&mut values).unwrap();
        assert!((f64::from(values[0]).powi(2) + f64::from(values[1]).powi(2) - 1.0).abs() < 1e-6);
    }
}
