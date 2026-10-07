use crate::app::cli::{CliOptions, KvFormat};
use crate::models::chat_template_jinja as jinja;
use crate::core::tensor::TensorSource;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;

pub(super) fn validate_gemma4_temperature(arch: &str, temperature: f32) -> Result<(), String> {
    if arch == "gemma4" && temperature != 0.0 {
        return Err("Gemma4 requires greedy decoding; --temp must be 0".into());
    }
    Ok(())
}

pub(crate) fn uses_llama_trunk(arch: &str) -> bool {
    matches!(
        arch,
        // `mistral3` rides the llama trunk; YaRN RoPE is detected from
        // `rope.scaling.{type,factor,original_context_length,yarn_beta_*,
        // yarn_log_multiplier}` inside the trunk, so this dispatch is
        // unconditional — Mistral 3 3B / Shieldstral / Ministral-3
        // (`harshatheg/Ministral-3-3B-Instruct-2512-GGUF`) all route here.
        // `gemma2` rides the llama trunk too — it shares the standard
        // llama.cpp tensor layout (no BitLinear packing) and adds
        // GeGLU + sliding-window attention + logit softcapping,
        // all detected from `gemma2.*` GGUF metadata inside the trunk.
        "llama"
            | "exaone"
            | "k2-horizon"
            | "granite"
            | "nanbeige"
            | "phi3"
            | "glm4"
            | "mistral3"
            | "gemma2"
    )
}

pub fn run_inference(
    source: Arc<dyn TensorSource>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    thinking: bool,
    bench: bool,
    profile: bool,
    kv_format: KvFormat,
    prefill_batch_size: usize,
    max_context: usize,
    repetition_penalty: f32,
    chat_template: Option<&str>,
    jinja: &crate::models::chat_template_jinja::Options,
) -> Result<(), String> {
    // Read arch into an owned String so the borrow of `source.metadata`
    // is released before the phi3 wrap below moves `source`.
    let arch: String = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .map(|s| s.to_string())
        .unwrap_or_default();

    // Phi-3 / Phi-4 stores attention and FFN as fused QKV / fused
    // gate-up tensors. Wrap the source so the llama trunk sees the
    // standard per-projection layout (attn_q/attn_k/attn_v, ffn_gate/ffn_up).
    let source: Arc<dyn TensorSource> = if arch == "phi3" {
        // Pull dimensions off the source first; once those borrow
        // lifetimes drop, we can hand a clone to Phi3Source::new.
        let n_embd: usize = source
            .metadata("phi3.embedding_length")
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .unwrap_or(0);
        let n_head: usize = source
            .metadata("phi3.attention.head_count")
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .unwrap_or(1);
        let n_head_kv: usize = source
            .metadata("phi3.attention.head_count_kv")
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .unwrap_or(n_head);
        let n_ff: usize = source
            .metadata("phi3.feed_forward_length")
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .unwrap_or(0);
        let head_dim = if n_head > 0 { n_embd / n_head } else { 0 };
        let n_embd_q = n_head * head_dim;
        let n_embd_gqa = n_head_kv * head_dim;
        Arc::new(crate::models::phi3::Phi3Source::new(
            source.clone(),
            n_embd_q,
            n_embd_gqa,
            n_ff,
        ))
    } else {
        source
    };

    if arch == "hunyuan-dense" {
        crate::app::text::run_hunyuan_inference(
            source.clone(),
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            profile,
            kv_format,
            prefill_batch_size,
            max_context,
            repetition_penalty,
            jinja,
        )
    } else if arch == "lfm2" {
        let is_lfm25 = source
            .metadata("general.basename")
            .and_then(|v| v.to_string_val())
            .map(|v| v.contains("2.5"))
            .unwrap_or(false);

        if is_lfm25 {
            crate::models::lfm25::run_inference(
                source.as_ref(),
                prompt,
                max_tokens,
                temperature,
                n_threads_arg,
                profile,
                kv_format,
                max_context,
                thinking,
                jinja,
            )
        } else {
            crate::models::lfm2::run_inference(
                source.as_ref(),
                prompt,
                max_tokens,
                temperature,
                n_threads_arg,
                profile,
                kv_format,
                max_context,
                repetition_penalty,
                thinking,
                jinja,
            )
        }
    } else if arch == "lfm2moe" {
        crate::models::lfm2moe::run_inference_with_batch(
            source.as_ref(),
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            profile,
            kv_format,
            max_context,
            repetition_penalty,
            thinking,
            jinja,
            prefill_batch_size,
        )
    } else if uses_llama_trunk(&arch) {
        crate::models::llama::run_inference(
            source.as_ref(),
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            bench,
            profile,
            kv_format,
            max_context,
            repetition_penalty,
            thinking,
            jinja,
        )
    } else if arch == "xing4_0" {
        crate::models::xing4_0::trunk::run::run_inference(
            source.as_ref(),
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            bench,
            profile,
            kv_format,
            max_context,
            repetition_penalty,
            thinking,
            jinja,
        )
    } else if arch == "spark2_5" {
        crate::models::spark::run_inference(
            source.as_ref(),
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            thinking,
            bench,
            profile,
            kv_format,
            jinja,
        )
    } else if arch == "nemotron_h" {
        crate::models::nemotron_h::trunk::run_inference(
            source.clone(),
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            kv_format,
            repetition_penalty,
            chat_template,
            thinking,
            jinja,
        )
    } else if arch == "falcon-h1" {
        crate::models::falcon_h1::trunk::run_inference(
            source.clone(),
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            kv_format,
            repetition_penalty,
            chat_template,
            thinking,
            jinja,
        )
    } else {
        crate::app::text::run_qwen3_inference(
            source.clone(),
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            thinking,
            bench,
            profile,
            kv_format,
            prefill_batch_size,
            max_context,
            repetition_penalty,
            jinja,
        )
    }
}

/// One question inside a JEV request. Options are stored in their original
/// user-supplied form (e.g. "晴天:5" for score mode, "晴天" otherwise); the
/// decision logic parses the `:value` suffix and decides the per-question
/// mode from the option shapes.

// JEV types + decision scoring live in `src/app/jev/mod.rs`.

// JEV decision scoring has moved to `src/app/jev/mod.rs`.

pub fn run_interactive(
    source: Arc<dyn TensorSource>,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    prefill_batch_size: usize,
    repetition_penalty: f32,
) -> Result<(), String> {
    println!("=== RustModelInference Interactive Mode ===");
    println!("Type your prompt and press Enter. Ctrl+C to exit.\n");

    loop {
        print!("> ");
        io::stdout()
            .flush()
            .map_err(|error| format!("Failed to flush prompt: {error}"))?;
        let mut line = String::new();
        if io::stdin()
            .read_line(&mut line)
            .map_err(|error| format!("Failed to read prompt: {error}"))?
            == 0
        {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        run_inference(
            source.clone(),
            line,
            max_tokens,
            temperature,
            n_threads_arg,
            false,
            false,
            false,
            KvFormat::F16,
            prefill_batch_size,
            CliOptions::DEFAULT_MAX_CONTEXT,
            repetition_penalty,
            None,
            &jinja::Options::default(),
        )?;
        println!();
    }
    Ok(())
}

/// Interactive REPL for qwen35 architecture (uses multimodal path even for text-only).
pub fn run_interactive_qwen35(
    source: Arc<dyn TensorSource>,
    model_path: &Path,
    max_tokens: usize,
    temperature: f32,
    n_threads: usize,
    prefill_batch_size: usize,
    max_context: usize,
    repetition_penalty: f32,
) -> Result<(), String> {
    println!("=== RustModelInference Interactive Mode (qwen35) ===");
    println!("Type your prompt and press Enter. Ctrl+C to exit.\n");
    loop {
        print!("> ");
        io::stdout()
            .flush()
            .map_err(|error| format!("Failed to flush prompt: {error}"))?;
        let mut line = String::new();
        if io::stdin()
            .read_line(&mut line)
            .map_err(|error| format!("Failed to read prompt: {error}"))?
            == 0
        {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        super::multimodal::run_multimodal_with_video(
            source.clone(),
            model_path,
            None,
            None,
            None,
            None,
            line,
            max_tokens,
            temperature,
            n_threads,
            prefill_batch_size,
            max_context,
            repetition_penalty,
            &jinja::Options::default(),
        )?;
        println!();
    }
    Ok(())
}

pub fn run_shared_inference(
    source: Arc<dyn TensorSource>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    thinking: bool,
    prefill_batch_size: usize,
) -> Result<(), String> {
    crate::models::qwen3::run_shared_inference(
        source,
        prompt,
        max_tokens,
        temperature,
        n_threads_arg,
        thinking,
        prefill_batch_size,
    )
}

/// Delegates to the canonical sampler in `ops::sampling`.
///
/// NOTE (behaviour change): this used to hardcode the draw threshold to
/// `0.5` — i.e. always pick the token at the median of the distribution.
/// That was a stub, not a design: it made `--temp > 0` deterministic and
/// arbitrary. Greedy (`--temp 0`) is unaffected; temperature sampling is now
/// a real random draw like every other front-end.
pub fn sample_token(logits: &[f32], temperature: f32) -> i32 {
    crate::ops::sampling::sample_greedy_or_temperature(logits, temperature)
        .map(|id| id as i32)
        .unwrap_or(0)
}
