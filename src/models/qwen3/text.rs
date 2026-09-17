//! # Qwen3 纯文本推理（CLI 入口）
//!
//! 此模块是 CLI 文本推理的 thin wrapper：直接复用
//! `models::qwen3::base::Qwen3Model` + `Qwen3Session`，
//! 调用其 `generate_streaming` 接口完成 token 生成。
//!
//! 底层推理逻辑已统一到 `base`（`Qwen3Session::generate_inner`）。
//! 本模块负责：
//! 1. 把 `&dyn TensorSource` 升级成 `Arc<dyn TensorSource>` 以满足 `Qwen3Model::from_source` 的接口
//! 2. CLI 特有的 prompt 模板（`build_qwen_chat_prompt`）
//! 3. CLI 输出格式（streaming 文本 + bench 计时）

use crate::app::cli::{per_second, resolve_thread_count, DSparkOptions, KvFormat};
use crate::core::loader::model_config_from_source;
use crate::core::scratchpad::KvLifecycle;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::models::dspark::{
    prefill as dspark_prefill, run_greedy as run_dspark_greedy, DSparkModel, DSparkSession,
    RunOptions as DSparkRunOptions, SharedHead, TargetShape,
};
use crate::models::qwen3::{Qwen3GenerateOptions, Qwen3Input, Qwen3Model, Qwen3Session};
use crate::prompt::{build_qwen_chat_prompt, QwenMessage};

use std::io::{self, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[macro_export]
macro_rules! slice_from_mut {
    ($ptr:expr, $len:expr) => {
        unsafe { std::slice::from_raw_parts_mut($ptr, $len) }
    };
}

#[macro_export]
macro_rules! slice_from_ref {
    ($ptr:expr, $len:expr) => {
        unsafe { std::slice::from_raw_parts($ptr, $len) }
    };
}

#[macro_export]
macro_rules! raw_parts {
    ($ptr:expr, $len:expr) => {
        unsafe { std::slice::from_raw_parts($ptr, $len) }
    };
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
    dspark: Option<DSparkOptions>,
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

    run_inference_tokens(
        source,
        input_tokens,
        max_tokens,
        temperature,
        n_threads_arg,
        bench,
        profile,
        kv_format,
        prefill_batch_size,
        dspark,
    )
}

pub fn run_inference_tokens(
    source: Arc<dyn TensorSource>,
    input_tokens: Vec<u32>,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    bench: bool,
    profile: bool,
    kv_format: KvFormat,
    prefill_batch_size: usize,
    dspark: Option<DSparkOptions>,
) -> Result<(), String> {
    let _ = (bench, profile); // bench/profile 暂由 wall-clock 估算
    let t0 = Instant::now();

    let tokenizer = Arc::new(
        BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?,
    );

    let max_ctx = 512usize.min(model_config_from_source(source.as_ref())?.n_ctx);
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let pool = Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());

    // 1) 加载共享模型（与 ASR/TTS/Image 共用同一份 Qwen3Model）
    let model = Qwen3Model::from_source(
        Arc::clone(&source),
        Arc::clone(&tokenizer),
        Arc::clone(&pool),
    )?;
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

    // 2) Streaming 调用 + bench 计时
    print!("Output: ");
    io::stdout().flush().unwrap();

    let t_infer = Instant::now();
    let (decode_count, prefill_time, decode_time) = if let Some(options) = dspark {
        if model.config.architecture != "qwen3" {
            return Err(format!(
                "DSpark Qwen adapter requires qwen3, got {}",
                model.config.architecture
            ));
        }
        let required = input_tokens
            .len()
            .checked_add(max_tokens)
            .ok_or("DSpark session capacity overflow")?;
        if required > max_ctx {
            return Err(format!(
                "Generation requires capacity {required}; session has {max_ctx}"
            ));
        }
        let draft_source: Arc<dyn TensorSource> = Arc::from(
            open_model_source(&options.draft_model, ComponentRole::Llm)
                .map_err(|error| format!("Failed to open DSpark sidecar: {error}"))?,
        );
        let target = TargetShape {
            hidden: model.config.n_embd,
            vocab: model.config.vocab,
            layers: model.config.n_layer,
        };
        let shared_head =
            SharedHead::new(Arc::clone(&source), Arc::clone(&tokenizer), target, max_ctx)?;
        let draft_model = DSparkModel::from_source(draft_source, shared_head, Arc::clone(&pool))?;
        let draft_n_max = options.draft_n_max.unwrap_or(draft_model.config.block_size);
        let mut target_session =
            Qwen3Session::new_with_kv_state(&model, max_ctx, kv_format, KvLifecycle::Ephemeral)?;
        let mut draft_session = DSparkSession::new(&draft_model, max_ctx)?;
        dspark_prefill(
            &mut target_session,
            &mut draft_session,
            &input_tokens,
            prefill_batch_size,
        )?;
        let prefill_time = t_infer.elapsed();
        let mut decoder = tokenizer.streaming_decoder(false);
        let mut generated = Vec::with_capacity(max_tokens);
        let mut stop_tokens = tokenizer.eos_id().into_iter().collect::<Vec<_>>();
        if let Some(im_end) = tokenizer.special_token_id("im_end") {
            if !stop_tokens.contains(&im_end) {
                stop_tokens.push(im_end);
            }
        }
        let stats = run_dspark_greedy(
            &mut target_session,
            &mut draft_session,
            DSparkRunOptions {
                max_tokens,
                draft_n_max,
                confidence_min: options.confidence_min,
                stop_tokens,
            },
            |token| {
                generated.push(token);
                let text = decoder.push(token);
                if !text.is_empty() {
                    print!("{text}");
                    io::stdout().flush().unwrap();
                }
            },
        )?;
        let tail = decoder.finish();
        if !tail.is_empty() {
            print!("{tail}");
            io::stdout().flush().unwrap();
        }
        eprintln!(
            "DSpark: drafted={} accepted={} target_evaluations={}",
            stats.drafted, stats.accepted, stats.target_evaluations
        );
        (
            generated.len(),
            prefill_time,
            t_infer.elapsed().saturating_sub(prefill_time),
        )
    } else {
        let mut session =
            Qwen3Session::new_with_kv_state(&model, max_ctx, kv_format, KvLifecycle::Ephemeral)?;
        let positions: Vec<[usize; 4]> = (0..input_tokens.len()).map(|i| [i, 0, 0, 0]).collect();
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
            |text| {
                if prefill_time.is_zero() {
                    prefill_time = t_infer.elapsed();
                }
                print!("{text}");
                io::stdout().flush().unwrap();
            },
        )?;
        (
            generation.token_ids.len(),
            prefill_time,
            t_infer.elapsed().saturating_sub(prefill_time),
        )
    };

    let prompt_len = input_tokens.len();

    println!();
    let infer_ms = t_infer.elapsed().as_millis();
    let tok_s = if infer_ms > 0 {
        decode_count as f64 / infer_ms as f64 * 1000.0
    } else {
        0.0
    };

    eprintln!(
        "Prompt: {:.1} t/s | Generation: {:.1} t/s | end-to-end: {:.1} tok/s",
        per_second(prompt_len, prefill_time),
        per_second(decode_count, decode_time),
        tok_s
    );
    println!("[{} output tokens in {}ms]", decode_count, infer_ms);
    Ok(())
}
