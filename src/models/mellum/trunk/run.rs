//! CLI inference entry for Mellum 2.x (`general.architecture = "mellum"`).
//!
//! The checkpoint's chat template is ChatML: `<|im_start|>role\n …
//! <|im_end|>\n`, generation prompt `<|im_start|>assistant\n`, and — for the
//! Thinking variants — the caller decides whether to open `<think>`. The GGUF
//! sets `add_bos_token = false`, so the builder below is the whole prompt
//! contract: nothing is prepended to the rendered text.

use super::config::ARCH;
use super::forward::MellumRunner;
use super::weights::MellumModel;
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::EncodeOptions;
use std::io::Write;
use std::sync::Arc;

/// Render one user turn (plus the generation prompt) in ChatML.
///
/// `thinking == false` closes an empty `<think></think>` block, matching the
/// `enable_thinking == false` branch of the GGUF template.
pub fn build_mellum_prompt(system: Option<&str>, user: &str, thinking: bool) -> String {
    let mut out = String::new();
    if let Some(text) = system.filter(|text| !text.is_empty()) {
        out.push_str("<|im_start|>system\n");
        out.push_str(text);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>user\n");
    out.push_str(user);
    out.push_str("<|im_end|>\n<|im_start|>assistant\n");
    if !thinking {
        out.push_str("<think>\n\n</think>\n\n");
    }
    out
}

/// Load the model, run one greedy/temperature-sampled continuation and print
/// it. `jinja` (when it yields text) replaces the built-in ChatML builder.
#[allow(clippy::too_many_arguments)]
pub fn run_inference(
    source: Arc<dyn TensorSource>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    profile: bool,
    kv_format: KvFormat,
    max_context: usize,
    repetition_penalty: f32,
    thinking: bool,
    jinja: &crate::prompt::jinja::Options,
) -> Result<(), String> {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available);
    let tokenizer = Arc::new(
        crate::core::tokenizer::BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?,
    );
    let pool = Arc::new(ComputePool::new(n_threads));
    let model = MellumModel::from_source(Arc::clone(&source), Arc::clone(&tokenizer), pool)?;
    let config = model.config().clone();
    let capacity = max_context.min(config.n_ctx).max(1);

    let special = crate::prompt::jinja::special_token_literals(&*tokenizer);
    let jinja_text = crate::prompt::jinja::single_turn_text(
        jinja,
        &|k| source.metadata(k).cloned(),
        &special.bos,
        &special.eos,
        prompt,
        thinking,
    )?;
    let prompt_text = jinja_text.unwrap_or_else(|| build_mellum_prompt(None, prompt, thinking));
    let ids = tokenizer.encode(
        &prompt_text,
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    if ids.is_empty() {
        return Err("Mellum prompt produced no tokens".into());
    }
    if ids.len() >= capacity {
        return Err(format!(
            "prompt is {} tokens but the context holds {capacity}",
            ids.len()
        ));
    }

    let mut runner = MellumRunner::new(&model, capacity, kv_format)?;
    // Sampling defaults come from the GGUF's `general.sampling.*`, falling back
    // to llama.cpp's usual 40 / 0.95 when the checkpoint does not pin them.
    let (top_k, top_p) = (
        source
            .metadata("general.sampling.top_k")
            .and_then(|value| value.to_u64())
            .map(|value| value as usize)
            .unwrap_or(40),
        source
            .metadata("general.sampling.top_p")
            .and_then(|value| value.to_f64())
            .unwrap_or(0.95) as f32,
    );
    let mut sampler = crate::ops::sampling::LlamaSampler::new(top_k, top_p);
    sampler.prime(&ids);
    let mut decoder = tokenizer.streaming_decoder(false);
    let eos_id = tokenizer.eos_id();

    println!(
        "Model: {ARCH} | n_embd={} n_layer={} n_head={} n_head_kv={} head_dim={} experts={}/{} | threads={n_threads}",
        config.n_embd,
        config.n_layer,
        config.n_head,
        config.n_head_kv,
        config.n_embd_head_k,
        config.expert_count,
        config.expert_used_count,
    );
    println!(
        "Prompt: {} ({} tokens)",
        prompt_text.replace('\n', "\\n"),
        ids.len()
    );
    print!("Output: ");
    std::io::stdout().flush().map_err(|e| e.to_string())?;

    let started = std::time::Instant::now();
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    for step in 0..ids.len() {
        runner.forward_token(ids[step], step)?;
    }
    while generated.len() < max_tokens {
        let mut logits = runner.logits().to_vec();
        let chosen = sampler.sample(&mut logits, temperature, repetition_penalty);
        if Some(chosen) == eos_id {
            break;
        }
        generated.push(chosen);
        let text = decoder.push(chosen);
        print!("{text}");
        std::io::stdout().flush().map_err(|e| e.to_string())?;
        let position = ids.len() + generated.len() - 1;
        if position + 1 >= capacity {
            break;
        }
        runner.forward_token(chosen, position)?;
    }
    let tail = decoder.finish();
    if !tail.is_empty() {
        print!("{tail}");
    }
    println!();
    let elapsed = started.elapsed().as_millis();
    let tok_s = if elapsed > 0 {
        generated.len() as f64 / elapsed as f64 * 1000.0
    } else {
        0.0
    };
    let _ = profile;
    println!(
        "[end-to-end: {} output tokens in {}ms | {:.1} tok/s]",
        generated.len(),
        elapsed,
        tok_s
    );
    Ok(())
}

/// Prefill `prompt_tokens` and return the final-position logits.
///
/// Used by the JEV decision scorer, which only needs the next-token
/// distribution and never autoregressively decodes. `batch_size` is accepted
/// for trait compatibility with the other `run_forward_logits_*` entry points
/// but ignored: `MellumRunner::forward_token` walks the MoE routing one token
/// at a time by construction, so there is no `rows > 1` batched path to lift
/// yet.
pub fn run_forward_logits_mellum_with_batch(
    source: Arc<dyn TensorSource>,
    prompt_tokens: &[u32],
    n_threads_arg: usize,
    kv_format: KvFormat,
    max_context: usize,
    batch_size: usize,
) -> Result<(Vec<f32>, std::time::Duration), String> {
    let _ = batch_size;
    let started = std::time::Instant::now();
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available);
    let tokenizer = Arc::new(
        crate::core::tokenizer::BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?,
    );
    let pool = Arc::new(ComputePool::new(n_threads));
    let model = MellumModel::from_source(Arc::clone(&source), tokenizer, pool)?;
    let config = model.config().clone();
    let capacity = max_context.min(config.n_ctx).max(1);
    if prompt_tokens.is_empty() {
        return Err("Mellum JEV prompt produced no tokens".into());
    }
    if prompt_tokens.len() >= capacity {
        return Err(format!(
            "prompt is {} tokens but the context holds {capacity}",
            prompt_tokens.len()
        ));
    }
    let mut runner = MellumRunner::new(&model, capacity, kv_format)?;
    for (position, &token) in prompt_tokens.iter().enumerate() {
        runner.forward_token(token, position)?;
    }
    Ok((runner.logits().to_vec(), started.elapsed()))
}

#[cfg(test)]
mod tests {
    use super::build_mellum_prompt;

    #[test]
    fn thinking_prompt_ends_with_the_generation_marker() {
        let prompt = build_mellum_prompt(None, "hi", true);
        assert_eq!(
            prompt,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn non_thinking_prompt_closes_an_empty_reasoning_block() {
        let prompt = build_mellum_prompt(Some("sys"), "hi", false);
        assert_eq!(
            prompt,
            "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
