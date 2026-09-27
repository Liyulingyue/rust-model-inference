//! Per-arch text-generation adapters.
//!
//! Phase 3 of the CLI/HTTP unification
//! (see `docs/develop/TEXT_RUNTIME_UNIFICATION.md`). Each adapter wraps one
//! architecture family behind the shared [`TextRuntime`] shape:
//!
//! * [`LlamaTextRuntime`] — llama / nanbeige / exaone / k2-horizon / granite /
//!   MiniCPM5, driving `LlamaSession::forward_logits_per_token`.
//! * [`Qwen3TextRuntime`] — qwen3 / qwen3vl text-only, driving the existing
//!   `generate_streaming_until` (already sink-shaped).
//! * [`Qwen35TextRuntime`] — qwen35, building a per-request session from a
//!   held `Mutex<Qwen35Model>` (the model needs `&mut self` on step).
//! * [`Lfm2MoeTextRuntime`] — lfm2moe, driving `forward_token`.
//!
//! The HTTP layer calls [`build_text_runtime`] once at startup and then only
//! talks to `Box<dyn TextRuntime>`; no per-arch match remains in the server.
//!
//! CLI behavior is unchanged — the CLI still calls the arch `run_inference`
//! functions. These adapters are what replaces the server's four hand-written
//! decode loops.
use crate::app::cli::KvFormat;
use crate::core::tensor::TensorSource;
use crate::models::llama::trunk::{build_prompt_tokens, LlamaSession};
use crate::ops::generation_runtime::{Flow, GeneratedText, GenerationRequest, Finish, TextRuntime, TokenSink};
use crate::ops::sampling::sample_temperature_greedy_or_random;
use std::sync::{Arc, Mutex};

/// What the server needs to know to construct a runtime.
pub struct RuntimeOptions {
    pub threads: usize,
    pub kv_format: KvFormat,
    pub max_context: usize,
    pub prefill_batch_size: usize,
    /// Reuse the server's already-loaded source handle (the same GGUF mmap).
    pub source: Arc<dyn TensorSource>,
    /// The compute pool the server built for this model.
    pub pool: Arc<crate::core::thread_pool::ComputePool>,
    /// The tokenizer the server built for this model.
    pub tokenizer: Arc<crate::core::tokenizer::BPETokenizer>,
}

/// Build the per-arch runtime. Returns `None` for archs this pass does not
/// cover (the server keeps its `Fallback` 501 for those).
pub fn build_text_runtime(
    arch: &str,
    options: RuntimeOptions,
) -> Result<Box<dyn TextRuntime>, String> {
    if crate::app::text::uses_llama_trunk(arch) {
        Ok(Box::new(LlamaTextRuntime::new(options)?))
    } else if matches!(arch, "qwen3" | "qwen3vl") {
        Ok(Box::new(Qwen3TextRuntime::new(options)?))
    } else if arch == "qwen35" {
        Ok(Box::new(Qwen35TextRuntime::new(options)?))
    } else if arch == "lfm2moe" {
        Ok(Box::new(Lfm2MoeTextRuntime::new(options)?))
    } else {
        Err(format!("No TextRuntime adapter for architecture {arch}"))
    }
}

/// Read `general.architecture` as an owned string, defaulting to empty.
fn arch_of(source: &Arc<dyn TensorSource>) -> String {
    source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .map(|value| value.to_string())
        .unwrap_or_default()
}

/// Shared decode helpers used by the adapters below.
///
/// Samples `logits` with the server's single-token sampler and returns the
/// id, or `None` when the eos / `im_end` stop ids are hit.
fn sample_step(
    logits: &[f32],
    temperature: f32,
    eos_id: Option<u32>,
    im_end_id: Option<u32>,
) -> Option<u32> {
    let id = u32::try_from(sample_temperature_greedy_or_random(logits, temperature)).ok()?;
    if Some(id) == eos_id || Some(id) == im_end_id {
        return None;
    }
    Some(id)
}

// ---------------------------------------------------------------------------
// llama family
// ---------------------------------------------------------------------------

pub struct LlamaTextRuntime {
    session: Mutex<LlamaSession<'static>>,
    _source: Arc<dyn TensorSource>,
    tokenizer: Arc<crate::core::tokenizer::BPETokenizer>,
    arch: String,
}

impl LlamaTextRuntime {
    pub fn new(options: RuntimeOptions) -> Result<Self, String> {
        let source = options.source.clone();
        let session = LlamaSession::from_source(
            source.as_ref(),
            options.threads,
            options.kv_format,
            options.max_context,
        )?;
        // SAFETY: `source` is held by the runtime for its whole lifetime, so
        // borrowing session data from it for `'static` stays valid.
        let session: LlamaSession<'static> = unsafe { std::mem::transmute(session) };
        Ok(Self {
            session: Mutex::new(session),
            _source: source,
            tokenizer: options.tokenizer,
            arch: arch_of(&options.source),
        })
    }
}

impl TextRuntime for LlamaTextRuntime {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn context_length(&self) -> usize {
        self.session
            .lock()
            .map(|session| session.config.max_ctx)
            .unwrap_or(0)
    }

    fn generate(
        &mut self,
        request: &GenerationRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<GeneratedText, String> {
        let mut session = self.session.lock().map_err(|e| e.to_string())?;
        session.reset();
        let eos_id = self.tokenizer.eos_id();
        let im_end_id = self.tokenizer.special_token_id("im_end");
        let mut decoder = self.tokenizer.streaming_decoder(false);
        let mut text = String::new();
        let mut token_ids = Vec::new();
        let mut finish = Finish::Limit;
        // Prefill (the ids are the prompt the caller tokenized via
        // `build_prompt_tokens`), then one token at a time.
        let mut logits = session.forward_logits_per_token(&request.token_ids)?;
        for _ in 0..request.max_new_tokens {
            let Some(id) = sample_step(
                &logits,
                request.sampling.temperature,
                eos_id,
                im_end_id,
            ) else {
                finish = Finish::Eos;
                break;
            };
            token_ids.push(id);
            let chunk = decoder.push(id);
            if !chunk.is_empty() {
                text.push_str(&chunk);
                if sink.push_text(&chunk) == Flow::Stop {
                    finish = Finish::Cancelled;
                    break;
                }
            }
            logits = session.forward_logits_per_token(&[id])?;
        }
        let tail = decoder.finish();
        if !tail.is_empty() && finish != Finish::Cancelled {
            text.push_str(&tail);
            let _ = sink.push_text(&tail);
        }
        Ok(GeneratedText {
            text,
            token_ids,
            finish,
        })
    }
}

// ---------------------------------------------------------------------------
// qwen3 / qwen3vl (text-only)
// ---------------------------------------------------------------------------

pub struct Qwen3TextRuntime {
    model: Arc<crate::models::qwen3::Qwen3Model>,
    tokenizer: Arc<crate::core::tokenizer::BPETokenizer>,
    pool: Arc<crate::core::thread_pool::ComputePool>,
    prefill_batch_size: usize,
    kv_format: KvFormat,
    arch: String,
}

impl Qwen3TextRuntime {
    pub fn new(options: RuntimeOptions) -> Result<Self, String> {
        let model = crate::models::qwen3::Qwen3Model::from_source(
            options.source.clone(),
            options.tokenizer.clone(),
            options.pool.clone(),
        )?;
        Ok(Self {
            kv_format: KvFormat::F16,
            model: Arc::new(model),
            tokenizer: options.tokenizer,
            pool: options.pool,
            prefill_batch_size: options.prefill_batch_size,
            arch: arch_of(&options.source),
        })
    }
}

impl TextRuntime for Qwen3TextRuntime {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn context_length(&self) -> usize {
        self.model.config().n_ctx
    }

    fn generate(
        &mut self,
        request: &GenerationRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<GeneratedText, String> {
        let mut session = crate::models::qwen3::Qwen3Session::new_with_kv_state(
            &self.model,
            request.token_ids.len() + request.max_new_tokens,
            self.kv_format,
            crate::KvLifecycle::Ephemeral,
        )?;
        let positions: Vec<_> = (0..request.token_ids.len()).map(|i| [i, 0, 0, 0]).collect();
        let input = crate::models::qwen3::Qwen3Input {
            token_ids: &request.token_ids,
            positions: &positions,
            embeddings: None,
            deepstack_embeddings: None,
        };
        let options_qwen3 = crate::models::qwen3::Qwen3GenerateOptions {
            max_new_tokens: request.max_new_tokens,
            temperature: request.sampling.temperature,
            prefill_batch_size: self.prefill_batch_size,
        };
        let mut text = String::new();
        let mut finish = Finish::Limit;
        // `generate_streaming_until` already owns the decode loop + sampler;
        // we only translate its boolean callback into `TokenSink`.
        let sink = &mut *sink;
        let generation = session.generate_streaming_until(
            input,
            options_qwen3,
            request.sampling.repetition_penalty,
            |chunk: &str| {
                if chunk.is_empty() {
                    return true;
                }
                text.push_str(chunk);
                sink.push_text(chunk) == Flow::Continue
            },
        )?;
        // The closure cannot report `Cancelled` back, so infer it from the
        // shortfall: fewer ids than requested plus a stopped sink implies a
        // client-side stop. `Eos` vs `Limit` is not distinguishable here and
        // stays `Limit` (the wire encoder only maps `Stop`/`Length` anyway).
        if generation.token_ids.len() < request.max_new_tokens {
            finish = Finish::Eos;
        }
        Ok(GeneratedText {
            text,
            token_ids: generation.token_ids.clone(),
            finish,
        })
    }
}

// ---------------------------------------------------------------------------
// qwen35
// ---------------------------------------------------------------------------

pub struct Qwen35TextRuntime {
    // Qwen35Model needs `&mut self` on step (Vulkan state), so it lives in a
    // Mutex exactly like the old `TextInner::Qwen35`; a fresh session is
    // borrowed from it per request because `Qwen35Session` holds `&mut Model`.
    model: Mutex<crate::models::qwen35::Qwen35Model<'static>>,
    _source: Arc<dyn TensorSource>,
    tokenizer: Arc<crate::core::tokenizer::BPETokenizer>,
    pool: Arc<crate::core::thread_pool::ComputePool>,
    prefill_batch_size: usize,
    arch: String,
}

impl Qwen35TextRuntime {
    pub fn new(options: RuntimeOptions) -> Result<Self, String> {
        let source = options.source.clone();
        let model = crate::models::qwen35::Qwen35Model::from_source(source.as_ref())?;
        // SAFETY: source outlives the runtime (see LlamaTextRuntime::new).
        let model: crate::models::qwen35::Qwen35Model<'static> =
            unsafe { std::mem::transmute(model) };
        Ok(Self {
            model: Mutex::new(model),
            _source: source,
            tokenizer: options.tokenizer,
            pool: options.pool,
            prefill_batch_size: options.prefill_batch_size,
            arch: "qwen35".to_string(),
        })
    }
}

impl TextRuntime for Qwen35TextRuntime {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn context_length(&self) -> usize {
        self.model
            .lock()
            .map(|model| model.config.n_ctx)
            .unwrap_or(0)
    }

    fn generate(
        &mut self,
        request: &GenerationRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<GeneratedText, String> {
        let mut model = self.model.lock().map_err(|e| e.to_string())?;
        let (positions, _) = crate::models::qwen35::build_qwen35_positions(
            &request.token_ids,
            None,
            &[],
        )?;
        let mut session = crate::models::qwen35::Qwen35Session::new_with_prefill_batch_size(
            &mut model,
            request.token_ids.len() + request.max_new_tokens,
            self.prefill_batch_size,
            self.pool.clone(),
        )?;
        let eos_id = self.tokenizer.eos_id();
        let im_end_id = self.tokenizer.special_token_id("im_end");
        let mut decoder = self.tokenizer.streaming_decoder(false);
        let mut text = String::new();
        let mut token_ids = Vec::new();
        let mut finish = Finish::Limit;
        for step in 0..request.max_new_tokens {
            let pos = session.next_position();
            let decode_positions = [[pos, pos, pos, 0]];
            let (tokens, pos_slice) = if step == 0 {
                (request.token_ids.as_slice(), &positions[..])
            } else {
                (&token_ids[token_ids.len() - 1..], &decode_positions[..])
            };
            let logits = session.step_with_tokens(tokens, pos_slice)?;
            let Some(id) = sample_step(
                &logits,
                request.sampling.temperature,
                eos_id,
                im_end_id,
            ) else {
                finish = Finish::Eos;
                break;
            };
            token_ids.push(id);
            let chunk = decoder.push(id);
            if !chunk.is_empty() {
                text.push_str(&chunk);
                if sink.push_text(&chunk) == Flow::Stop {
                    finish = Finish::Cancelled;
                    break;
                }
            }
        }
        let tail = decoder.finish();
        if !tail.is_empty() && finish != Finish::Cancelled {
            text.push_str(&tail);
            let _ = sink.push_text(&tail);
        }
        Ok(GeneratedText {
            text,
            token_ids,
            finish,
        })
    }
}

// ---------------------------------------------------------------------------
// lfm2moe
// ---------------------------------------------------------------------------

pub struct Lfm2MoeTextRuntime {
    session: Mutex<crate::models::lfm2moe::Lfm2MoeSession<'static>>,
    _source: Arc<dyn TensorSource>,
    tokenizer: Arc<crate::core::tokenizer::BPETokenizer>,
    arch: String,
}

impl Lfm2MoeTextRuntime {
    pub fn new(options: RuntimeOptions) -> Result<Self, String> {
        let source = options.source.clone();
        let session = crate::models::lfm2moe::Lfm2MoeSession::from_source(
            source.as_ref(),
            options.threads,
            KvFormat::F16,
            options.max_context,
        )?;
        // SAFETY: source outlives the runtime (see LlamaTextRuntime::new).
        let session: crate::models::lfm2moe::Lfm2MoeSession<'static> =
            unsafe { std::mem::transmute(session) };
        Ok(Self {
            session: Mutex::new(session),
            _source: source,
            tokenizer: options.tokenizer,
            arch: "lfm2moe".to_string(),
        })
    }
}

impl TextRuntime for Lfm2MoeTextRuntime {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn context_length(&self) -> usize {
        self.session
            .lock()
            .map(|session| session.config.n_ctx)
            .unwrap_or(0)
    }

    fn generate(
        &mut self,
        request: &GenerationRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<GeneratedText, String> {
        let mut session = self.session.lock().map_err(|e| e.to_string())?;
        session.reset();
        let eos_id = self.tokenizer.eos_id();
        let im_end_id = self.tokenizer.special_token_id("im_end");
        let mut decoder = self.tokenizer.streaming_decoder(false);
        let mut text = String::new();
        let mut token_ids = Vec::new();
        let mut finish = Finish::Limit;
        // Prefill token by token (the session has no batched prefill entry).
        let mut logits = Vec::new();
        for &token_id in &request.token_ids {
            logits = session.forward_token(token_id)?;
        }
        for _ in 0..request.max_new_tokens {
            let Some(id) = sample_step(
                &logits,
                request.sampling.temperature,
                eos_id,
                im_end_id,
            ) else {
                finish = Finish::Eos;
                break;
            };
            token_ids.push(id);
            let chunk = decoder.push(id);
            if !chunk.is_empty() {
                text.push_str(&chunk);
                if sink.push_text(&chunk) == Flow::Stop {
                    finish = Finish::Cancelled;
                    break;
                }
            }
            logits = session.forward_token(id)?;
        }
        let tail = decoder.finish();
        if !tail.is_empty() && finish != Finish::Cancelled {
            text.push_str(&tail);
            let _ = sink.push_text(&tail);
        }
        Ok(GeneratedText {
            text,
            token_ids,
            finish,
        })
    }
}

/// The llama-family prompt builder, re-exported so the server's
/// `tools::build_prompt` does not need to reach into the model tree.
pub fn llama_prompt_tokens(
    source: &dyn TensorSource,
    prompt: &str,
    thinking: bool,
) -> Result<Vec<u32>, String> {
    build_prompt_tokens(source, prompt, thinking)
}