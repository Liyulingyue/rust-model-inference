//! Gemma3 embedding extraction (decoder-only + last-token pooling).
//!
//! Mirrors `crate::models::qwen3::embedding` but routes through
//! the gemma3 trunk instead. Used by the CLI `--embedding` path
//! for `microsoft/bitnet-embedding-270m` and any future
//! `arch=gemma3` GGUF.

use std::time::Instant;

use crate::app::cli::EmbeddingOutput;
use crate::core::loader::GGUFLoader;
use crate::core::scratchpad::KvFormat;
use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::tokenizer::SPMTokenizer;
use crate::models::gemma3::trunk::{
    load_layers_static, text_encode, Gemma3Config, Gemma3Model, Gemma3Rope,
};

/// Detect whether a `TensorSource` carries a BitNet b1.58 gemma3
/// GGUF. Mirrors the heuristic in
/// `crate::models::qwen3::embedding::embedding_config`:
///
/// - `general.file_type == 40` (Microsoft's BitNet I2_S marker), OR
/// - presence of `blk.0.attn_q_norm_in.weight` (per-projection
///   BitLinear RMSNorm gain).
///
/// The second condition catches non-Microsoft re-exports that
/// forget to set `file_type` but keep the BitLinear tensor
/// inventory. The first is the canonical signal from the official
/// `microsoft/bitnet` conversion script.
fn detect_is_bitnet(source: &dyn TensorSource) -> bool {
    let file_type_matches = source
        .metadata("general.file_type")
        .and_then(|v| v.to_u64())
        .map(|v| v == 40)
        .unwrap_or(false);
    let has_norm_in = source.tensor_info("blk.0.attn_q_norm_in.weight").is_some();
    file_type_matches || has_norm_in
}

/// Public embedding entry point used by the CLI / HTTP server.
/// Same shape as `qwen3::embedding::compute_embedding`.
pub fn compute_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    let _ = n_threads_arg;
    // Gemma3 BitNet 270M uses an SPM-style tokenizer
    // (`tokenizer.ggml.model = "llama"`, `tokenizer.ggml.pre =
    // "default"`, `tokenizer.ggml.scores` populated, **no
    // `tokenizer.ggml.merges`**). The SPM tokenizer in
    // `src/core/tokenizer/mod.rs::SPMTokenizer` accepts exactly
    // this combination; the BPE tokenizer rejects it (BPE needs
    // merge rules, which the 270M GGUF doesn't ship).
    let tokenizer =
        SPMTokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|e| format!("gemma3::compute_embedding: tokenizer init failed: {e}"))?;
    let prompt_tokens = encode_embedding_input(&tokenizer, prompt);
    if prompt_tokens.is_empty() {
        return Err("gemma3::compute_embedding: empty token sequence".into());
    }
    run_embedding_tokens(source, &prompt_tokens)
}

/// Tokenize a prompt using the SPM tokenizer with BOS prepended
/// and EOS stripped from the middle. The Gemma3 tokenizer declares
/// `add_bos_token=true` and `add_eos_token=true`; we want BOS
/// prepended (standard for embedding models) but no EOS in the
/// middle of the sequence.
pub fn encode_embedding_input(tokenizer: &SPMTokenizer, prompt: &str) -> Vec<u32> {
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

/// Build the `Gemma3Config` from a `TensorSource`. Pulls
/// `gemma3.*` metadata, derives `is_bitnet`, and selects the
/// Neox RoPE variant (270M declares `rope.dimension_count=256`
/// which equals `head_dim`, so partial rotation is not used).
pub fn build_config(source: &dyn TensorSource) -> Result<Gemma3Config, String> {
    let pick_u64 = |k: &str| {
        source
            .metadata(k)
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .ok_or_else(|| format!("gemma3: missing metadata {k}"))
    };
    let pick_f32 = |k: &str| {
        source
            .metadata(k)
            .and_then(|v| v.to_f64())
            .map(|v| v as f32)
            .ok_or_else(|| format!("gemma3: missing metadata {k}"))
    };
    Ok(Gemma3Config {
        architecture: source
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val())
            .unwrap_or("gemma3")
            .to_string(),
        n_embd: pick_u64("gemma3.embedding_length")?,
        n_layer: pick_u64("gemma3.block_count")?,
        n_head: pick_u64("gemma3.attention.head_count")?,
        n_head_kv: pick_u64("gemma3.attention.head_count_kv")?,
        n_embd_head_k: pick_u64("gemma3.attention.key_length")?,
        n_embd_head_v: pick_u64("gemma3.attention.value_length")?,
        n_ff: pick_u64("gemma3.feed_forward_length")?,
        // vocab is encoded as `tokenizer.ggml.tokens` array length
        // in standard gemma3 GGUFs (gemma3-270m-it and friends).
        // BitNet 270M declares `gemma3.vocab_size` directly; fall
        // back to that if the tokens array is missing.
        vocab: source
            .metadata("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                MetaValue::Array(_, items) => Some(items.len()),
                _ => None,
            })
            .or_else(|| pick_u64("gemma3.vocab_size").ok())
            .ok_or_else(|| "gemma3: cannot determine vocab (no tokenizer.ggml.tokens array and no gemma3.vocab_size)".to_string())?,
        n_ctx: pick_u64("gemma3.context_length").unwrap_or(0),
        eps: pick_f32("gemma3.attention.layer_norm_rms_epsilon")?,
        freq_base: pick_f32("gemma3.rope.freq_base")?,
        rope: Gemma3Rope::Neox,
        // RoPE linear scaling: standard Gemma 3 4B+/12B/27B
        // declares `gemma3.rope.scaling.type = "linear"` + a
        // `factor` (typically 8.0). Smaller variants (270M) omit
        // the metadata entirely and we default to factor=1.0.
        rope_factor: source
            .metadata("gemma3.rope.scaling.factor")
            .and_then(|v| v.to_f64())
            .map(|v| v as f32)
            .unwrap_or(1.0),
        rope_scaling_type: source
            .metadata("gemma3.rope.scaling.type")
            .and_then(|v| v.to_string_val())
            .unwrap_or("")
            .to_string(),
        pooling_type: pick_u64("gemma3.pooling_type").unwrap_or(1) as u32,
        sliding_window: source
            .metadata("gemma3.attention.sliding_window")
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .unwrap_or(0),
        is_bitnet: detect_is_bitnet(source),
    })
}

/// Load the full gemma3 model from a `TensorSource`. The
/// returned model is fully owned (no borrowed `TensorSource`
/// lifetime), so it can outlive the CLI's temporary
/// `TensorSource` borrow.
pub fn load_model(source: &dyn TensorSource) -> Result<Gemma3Model, String> {
    let config = build_config(source)?;
    let layers = load_layers_static(source, &config);
    let output_norm = crate::models::gemma3::trunk::get_f32_tensor(
        source,
        "output_norm.weight",
        config.n_embd,
    );
    let token_embedding_rows = crate::models::gemma3::trunk::static_weight(
        source,
        "token_embd.weight",
    )
    .map_err(|e| format!("gemma3: token_embd.weight load failed: {e}"))?;
    Ok(Gemma3Model {
        config,
        layers,
        output_norm,
        token_embedding_rows,
    })
}

/// Run the embedding extraction given pre-tokenized input.
pub fn run_embedding_tokens(
    source: &dyn TensorSource,
    token_ids: &[u32],
) -> Result<Vec<f32>, String> {
    if token_ids.is_empty() {
        return Err("gemma3::run_embedding_tokens: empty token sequence".into());
    }
    let model = load_model(source)?;
    text_encode(&model, token_ids)
}

/// CLI-facing print helper. Mirrors
/// `qwen3::embedding::print_embedding`.
pub fn print_embedding(pooled: &[f32], output: EmbeddingOutput, elapsed_ms: u128) {
    match output {
        EmbeddingOutput::Summary => {
            println!(
                "Embedding ({} dims, {} layers, arch=gemma3 {}ms):",
                pooled.len(),
                "(see config)",
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

/// CLI entry point used by `app::run_embedding` for
/// `arch=gemma3`. Loads + tokenizes + extracts + prints.
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
            print_embedding(&pooled, output, elapsed);
        }
        Err(error) => {
            eprintln!("gemma3::run_embedding failed: {error}");
        }
    }
}
