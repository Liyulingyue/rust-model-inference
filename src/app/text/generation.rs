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
    jinja: &crate::prompt::jinja::Options,
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

    if arch == "hy_v3" {
        // `hy_v3` backs two products: Hy-SimulMT (bundles the `hyv3_speech`
        // audio tower, needs `--audio`) and Hy-MT3-30B-A3B (plain text). The
        // tower's `clip.projector_type` is the only reliable discriminator —
        // the architecture string is identical — so a speech checkpoint keeps
        // the old guidance instead of silently generating text without the
        // audio conditioning it was trained on.
        if crate::models::hy_simulmt::has_speech_tower(source.as_ref()) {
            return Err("Hy-SimulMT requires --audio with a mono 16 kHz PCM16 WAV; use --language for the target language".into());
        }
        return run_hy_v3_text_inference(
            source,
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            bench,
            kv_format,
            prefill_batch_size,
            max_context,
            repetition_penalty,
            jinja,
        );
    }
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
        // LFM2.5 dense shares this path unconditionally. It used to branch on
        // whether `general.basename` contained "2.5", but that key is written
        // only by Liquid's QAD quantisation pipeline, so the same checkpoint
        // took a different code path depending on which quantizer produced the
        // file. llama.cpp has no equivalent split: every dense generation goes
        // through one `src/models/lfm2.cpp`. See `crate::models::lfm25`.
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
    } else if arch == "lfm3" {
        crate::models::lfm3::run_inference(
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
    } else if arch == "qwen35moe" {
        crate::models::occamy::run::run_inference(
            source.clone(),
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            bench,
            kv_format,
            max_context,
            repetition_penalty,
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
    } else if arch == "nemotron_h" || arch == "nemotron_h_moe" {
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
    } else if arch == "mellum" {
        crate::models::mellum::trunk::run::run_inference(
            source.clone(),
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
    } else if arch == "dory" {
        crate::models::dory::run_inference(
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
    // Threaded through so `--jinja` / `--chat-template-file` are honoured in
    // the REPL. Hardcoding `Options::default()` here accepted both flags and
    // silently ignored them.
    jinja: &crate::prompt::jinja::Options,
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
            jinja,
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
    // See `run_interactive`: the REPL must honour the chat-template flags.
    jinja: &crate::prompt::jinja::Options,
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
            jinja,
        )?;
        println!();
    }
    Ok(())
}

/// Text-only `hy_v3` CLI entry (Hy-MT3-30B-A3B).
///
/// Hy-SimulMT lives on the same GGUF architecture; callers must reject that
/// checkpoint before reaching here (see the `has_speech_tower` gate in
/// [`run_inference`]).
#[allow(clippy::too_many_arguments)]
fn run_hy_v3_text_inference(
    source: Arc<dyn TensorSource>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    bench: bool,
    kv_format: KvFormat,
    prefill_batch_size: usize,
    max_context: usize,
    repetition_penalty: f32,
    jinja: &crate::prompt::jinja::Options,
) -> Result<(), String> {
    use crate::ops::generation_runtime::{GenerationRequest, StdoutSink, TextRuntime};
    let tokenizer = std::sync::Arc::new(
        crate::core::tokenizer::BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?,
    );
    // Same prompt policy as the `hunyuan-dense` siblings: the GGUF template
    // wins when `--jinja` is given, otherwise the Hy-MT family builder picks
    // the control-token (v1) or raw-passthrough (v2) form from the vocab.
    let input_tokens = if bench {
        tokenizer.encode(
            prompt,
            crate::core::tokenizer::EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        )
    } else {
        match crate::prompt::jinja::single_turn_tokens(
            tokenizer.as_ref(),
            jinja,
            &|k| source.metadata(k).cloned(),
            prompt,
            Some(false),
        )? {
            Some(ids) => ids,
            None => crate::prompt::build_hunyuan_chat_prompt(
                tokenizer.as_ref(),
                &[crate::prompt::HunyuanMessage {
                    role: "user",
                    content: prompt,
                }],
                true,
            )?,
        }
    };
    // The Hy-MT3 GGUF template opens with `<｜hy_begin▁of▁sentence｜>`, and the
    // 30B-A3B needs it: a control token at position 0 with no BOS in front puts
    // the model out of distribution, and *both* llama.cpp and this repository
    // degenerate into repeated punctuation there (see
    // `tools/oracle/hy_mt3/validation.md` §3). `--jinja` renders the template
    // and already emits it, so only the hand-written builder needs the prefix.
    let mut input_tokens = input_tokens;
    if let Some(bos) = tokenizer.special_token_id("hy_begin") {
        if input_tokens.first() != Some(&bos) {
            input_tokens.insert(0, bos);
        }
    }
    let available = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4);
    let pool = std::sync::Arc::new(crate::core::thread_pool::ComputePool::new(
        crate::app::cli::resolve_thread_count(n_threads_arg, available),
    ));
    // Build through the same registry the server uses instead of constructing
    // the adapter here: one construction path means the CLI cannot drift from
    // HTTP, and `build_text_runtime` stays the single place a new arch is
    // wired into both front-ends.
    let mut runtime = super::build_text_runtime(
        "hy_v3",
        super::RuntimeOptions {
            compute: crate::compute::ComputePolicy::legacy(),
            threads: 0,
            kv_format,
            max_context,
            prefill_batch_size,
            source,
            pool,
            tokenizer,
            mmproj: None,
            mtp_draft: 0,
            // Hy-MT3's GGUFs ship no `nextn.*` tensors, so there is nothing to
            // draft from: keep the plain one-token-at-a-time decode.
            dspark_model: None,
            ngram_draft: 0,
            ngram_lookup: crate::core::ngram::NGRAM_SIMPLE_SIZE_N,
        },
    )?;
    let mut request = GenerationRequest::new(input_tokens, max_tokens);
    request.sampling.temperature = temperature;
    request.sampling.repetition_penalty = repetition_penalty;
    runtime.generate(&request, &mut StdoutSink)?;
    println!();
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
