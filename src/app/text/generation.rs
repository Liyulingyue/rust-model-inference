use crate::app::cli::{CliOptions, KvFormat};
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

pub(super) fn uses_llama_trunk(arch: &str) -> bool {
    matches!(arch, "llama" | "k2-horizon" | "granite" | "nanbeige")
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
    system: Option<&str>,
    chat_mode: bool,
) -> Result<(), String> {
    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();

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
            system,
            chat_mode,
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
            false,
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

pub fn sample_token(logits: &[f32], temperature: f32) -> i32 {
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as i32)
            .unwrap_or(0);
    }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut probs = vec![0.0f32; logits.len()];
    for (i, l) in logits.iter().enumerate() {
        probs[i] = ((l - max_logit) / temperature).exp();
        sum += probs[i];
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    let r = 0.5f32;
    let mut cumsum = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i as i32;
        }
    }
    (logits.len() - 1) as i32
}
