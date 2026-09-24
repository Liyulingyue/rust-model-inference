//! Qwen3 / Hunyuan text inference entry points.
//!
//! These are model-specific CLI wrappers that build the chat prompt
//! (Qwen3 ChatML or Hunyuan format) and delegate to `Qwen3Session`.

use crate::app::cli::{per_second, resolve_thread_count, KvFormat};
use crate::core::loader::model_config_from_source;
use crate::core::scratchpad::KvLifecycle;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::qwen3::{Qwen3GenerateOptions, Qwen3Input, Qwen3Model, Qwen3Session};
use crate::prompt::{
    build_hunyuan_chat_prompt, build_qwen_chat_prompt, HunyuanMessage, QwenMessage,
};
use std::io::{self, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub fn run_qwen3_inference(
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
) -> Result<(), String> {
    let input_tokens = {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        if bench {
            tokenizer.encode(
                prompt,
                EncodeOptions {
                    add_special: true,
                    parse_special: true,
                },
            )
        } else {
            build_qwen_chat_prompt(
                &tokenizer,
                &[QwenMessage {
                    role: "user",
                    content: prompt,
                }],
                thinking,
            )?
        }
    };
    run_qwen3_inference_tokens(
        source,
        input_tokens,
        max_tokens,
        temperature,
        n_threads_arg,
        bench,
        profile,
        kv_format,
        prefill_batch_size,
        max_context,
        repetition_penalty,
    )
}

pub fn run_hunyuan_inference(
    source: Arc<dyn TensorSource>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    profile: bool,
    kv_format: KvFormat,
    prefill_batch_size: usize,
    max_context: usize,
    repetition_penalty: f32,
) -> Result<(), String> {
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
    let input_tokens = build_hunyuan_chat_prompt(
        &tokenizer,
        &[HunyuanMessage {
            role: "user",
            content: prompt,
        }],
        true,
    )?;
    run_qwen3_inference_tokens(
        source,
        input_tokens,
        max_tokens,
        temperature,
        n_threads_arg,
        false,
        profile,
        kv_format,
        prefill_batch_size,
        max_context,
        repetition_penalty,
    )
}

fn run_qwen3_inference_tokens(
    source: Arc<dyn TensorSource>,
    input_tokens: Vec<u32>,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    bench: bool,
    profile: bool,
    kv_format: KvFormat,
    prefill_batch_size: usize,
    max_context: usize,
    repetition_penalty: f32,
) -> Result<(), String> {
    let _ = (bench, profile);
    let t0 = Instant::now();
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
    let model_config = model_config_from_source(source.as_ref())?;
    let max_ctx = max_context.min(model_config.n_ctx).max(1);
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let pool = Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());
    let model = Qwen3Model::from_source(source, Arc::new(tokenizer), pool)?;
    let load_ms = t0.elapsed().as_millis();
    println!(
        "Model: {} | n_embd={} n_layer={} n_head={} n_head_kv={} n_ff={} | loaded in {}ms",
        model.config.architecture,
        model.config.n_embd,
        model.config.n_layer,
        model.config.n_head,
        model.config.n_head_kv,
        model.config.n_ff,
        load_ms
    );
    println!("Prompt: {} tokens", input_tokens.len());
    let mut session =
        Qwen3Session::new_with_kv_state(&model, max_ctx, kv_format, KvLifecycle::Ephemeral)?;
    let positions: Vec<[usize; 4]> = (0..input_tokens.len()).map(|i| [i, 0, 0, 0]).collect();
    print!("Output: ");
    io::stdout().flush().unwrap();
    let t_infer = Instant::now();
    let mut prefill_time = Duration::ZERO;
    let generation = session.generate_streaming(
        Qwen3Input {
            token_ids: &input_tokens,
            positions: &positions,
            embeddings: None,
            deepstack_embeddings: None,
        },
        Qwen3GenerateOptions {
            max_new_tokens: max_tokens,
            temperature,
            prefill_batch_size,
        },
        repetition_penalty,
        |text| {
            if prefill_time.is_zero() {
                prefill_time = t_infer.elapsed();
            }
            print!("{}", text);
            io::stdout().flush().unwrap();
        },
    )?;
    let decode_time = t_infer.elapsed().saturating_sub(prefill_time);
    let prompt_len = input_tokens.len();
    let decode_count = generation.token_ids.len();
    println!();
    let infer_ms = t_infer.elapsed().as_millis();
    let tok_s = if infer_ms > 0 {
        generation.token_ids.len() as f64 / infer_ms as f64 * 1000.0
    } else {
        0.0
    };
    eprintln!(
        "Prompt: {:.1} t/s | Generation: {:.1} t/s | end-to-end: {:.1} tok/s",
        per_second(prompt_len, prefill_time),
        per_second(decode_count, decode_time),
        tok_s
    );
    println!(
        "[{} output tokens in {}ms]",
        generation.token_ids.len(),
        infer_ms
    );
    Ok(())
}
