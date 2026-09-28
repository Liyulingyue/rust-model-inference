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
use crate::ops::generation_runtime::StepAction;
use crate::ops::generation_runtime::{
    Finish, Flow, GeneratedText, GenerationRequest, TextRuntime, TokenSink,
};
use crate::ops::sampling::sample_greedy_or_temperature;
use crate::ops::sampling::{Lfm2MoeSampler, LlamaSampler};
use std::sync::{Arc, Mutex};

/// Runtime-level defaults, kept in ONE place.
///
/// Every value here must stay in lockstep with the CLI's own resolution
/// (`app::cli::validate::resolve_cli_generation_options`,
/// `CliOptions::effective_prefill_batch_size` / `effective_max_context`);
/// `runtime_options_match_cli_defaults` in the test module fails if they
/// drift. Adapters must never re-derive these.
pub mod defaults {
    use crate::app::cli::KvFormat;

    /// Same as `core::prefill::DEFAULT_PREFILL_BATCH_SIZE` (64).
    pub const PREFILL_BATCH_SIZE: usize = 64;
    /// Same as `CliOptions::DEFAULT_MAX_CONTEXT` (8192).
    pub const MAX_CONTEXT: usize = 8192;
    /// Same as `CliOptions::default()` — F32. Previously frozen at F16 to
    /// preserve legacy HTTP numerics, but the CLI/HTTP sentinel proved that
    /// choice was itself the divergence: the CLI runs F32 by default, so HTTP
    /// answered differently for the same prompt. Matching the CLI default is
    /// the whole point of these constants.
    pub const KV_FORMAT: KvFormat = KvFormat::F32;
    /// 0 means "let the pool decide", matching `resolve_thread_count(0, _)`.
    pub const THREADS: usize = 0;
}

/// Upper bound on vision placeholder tokens per request.
///
/// A 2400x2400 photo expands to ~4096 vision tokens, and at that size the
/// prefill dominates (on a small model it effectively stalls) the request.
/// 2048 keeps normal images (the docs' apple.png is 117) working while making
/// the failure explicit instead of a multi-minute hang.
const MAX_VISION_TOKENS: usize = 2048;
/// What a caller needs to construct a runtime.
///
/// Build it with [`RuntimeOptions::from_model`] and override only what the
/// caller actually wants to change; the remaining fields carry the defaults
/// above, so the server, tests, and any future front-end cannot disagree.
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
    /// Optional multimodal projector (the server's `--mmproj`). Required for
    /// image input; without it an image-bearing request fails with a clear
    /// error instead of silently dropping the images.
    pub mmproj: Option<Arc<dyn TensorSource>>,
}

impl RuntimeOptions {
    /// Canonical constructor: every field starts at `defaults::*`.
    pub fn from_model(
        source: Arc<dyn TensorSource>,
        pool: Arc<crate::core::thread_pool::ComputePool>,
        tokenizer: Arc<crate::core::tokenizer::BPETokenizer>,
    ) -> Self {
        Self {
            threads: defaults::THREADS,
            kv_format: defaults::KV_FORMAT,
            max_context: defaults::MAX_CONTEXT,
            prefill_batch_size: defaults::PREFILL_BATCH_SIZE,
            source,
            pool,
            tokenizer,
            mmproj: None,
        }
    }

    pub fn with_mmproj(mut self, mmproj: Option<Arc<dyn TensorSource>>) -> Self {
        self.mmproj = mmproj;
        self
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    pub fn with_max_context(mut self, max_context: usize) -> Self {
        self.max_context = max_context;
        self
    }

    pub fn with_kv_format(mut self, kv_format: KvFormat) -> Self {
        self.kv_format = kv_format;
        self
    }

    pub fn with_prefill_batch_size(mut self, prefill_batch_size: usize) -> Self {
        self.prefill_batch_size = prefill_batch_size;
        self
    }

    /// Session capacity for a model whose GGUF declares `model_n_ctx`.
    ///
    /// `min(model_n_ctx, max_context).max(1)` — the CLI's rule. Adapters must
    /// use this instead of sizing to `prompt + max_new_tokens`: capacity lays
    /// out the KV cache and the chunked-prefill grouping, and a different
    /// value changes the numerics (the qwen3 `Hi!` vs `Hello!` split).
    pub fn capacity_for(&self, model_n_ctx: usize) -> usize {
        model_n_ctx.min(self.max_context).max(1)
    }
}

/// Build the per-arch runtime. Returns `None` for archs this pass does not
/// cover (the server keeps its `Fallback` 501 for those).
pub fn build_text_runtime(
    arch: &str,
    options: RuntimeOptions,
) -> Result<Box<dyn TextRuntime>, String> {
    if crate::app::text::uses_llama_trunk(arch) {
        Ok(Box::new(LlamaTextRuntime::new(options)?))
    } else if matches!(arch, "qwen3" | "qwen3vl" | "qwen2vl" | "qwen3vlmoe") {
        // qwen2vl / qwen3vlmoe share the Qwen3 model + session + Qwen3Input
        // surface (the CLI's `run_qwen3_family_multimodal` covers them too);
        // only the projector family differs, which the image path handles.
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
) -> Result<Option<u32>, String> {
    let id = sample_greedy_or_temperature(logits, temperature)?;
    if Some(id) == eos_id || Some(id) == im_end_id {
        return Ok(None);
    }
    Ok(Some(id))
}

// ---------------------------------------------------------------------------
// llama family
// ---------------------------------------------------------------------------

pub struct LlamaTextRuntime {
    session: Mutex<LlamaSession<'static>>,
    _source: Arc<dyn TensorSource>,
    tokenizer: Arc<crate::core::tokenizer::BPETokenizer>,
    arch: String,
    /// Same sampler the CLI llama trunk uses (rep-penalty + llama.cpp chain +
    /// history-seeded RNG), so the two front-ends agree token for token.
    sampler: LlamaSampler,
}

impl LlamaTextRuntime {
    pub fn new(options: RuntimeOptions) -> Result<Self, String> {
        let source = options.source.clone();
        // Use batched prefill (`max_rows = prefill_batch_size`) when the
        // caller opts in (>= 2). Phi-4's per-head RoPE bug was fixed by
        // iterating `apply_rope` over each head's `head_dim`-sized slice,
        // so the session path now matches the CLI path; `max_rows == 1`
        // stays as a safe fallback when the operator keeps the legacy
        // single-row scratchpad.
        let prefill_rows = options.prefill_batch_size.max(1);
        let session = LlamaSession::from_source_with_max_rows(
            source.as_ref(),
            options.threads,
            options.kv_format,
            options.max_context,
            prefill_rows,
        )?;
        // SAFETY: `source` is held by the runtime for its whole lifetime, so
        // borrowing session data from it for `'static` stays valid.
        let session: LlamaSession<'static> = unsafe { std::mem::transmute(session) };
        // top_k / top_p from GGUF metadata, matching
        // `llama::trunk::sample_defaults`.
        let (top_k, top_p) = crate::models::llama::trunk::sample_defaults(source.as_ref());
        Ok(Self {
            session: Mutex::new(session),
            _source: source,
            tokenizer: options.tokenizer,
            arch: arch_of(&options.source),
            sampler: LlamaSampler::new(top_k, top_p),
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
        // `build_prompt_tokens`), then one token at a time. The sampler and
        // the stop rule are the CLI's, so the same prompt produces the same
        // tokens from both front-ends.
        // Seed the sampler's history RNG with the prompt so temperature
        // sampling matches the CLI byte for byte.
        self.sampler.prime(&request.token_ids);
        let mut logits = session.forward_logits_per_token(&request.token_ids)?;
        for _ in 0..request.max_new_tokens {
            let mut logits_owned = logits;
            let id = self
                .sampler
                .sample(&mut logits_owned, request.sampling.temperature, 1.0);
            logits = logits_owned;
            match crate::ops::generation_runtime::stop_after_sample(
                id,
                token_ids.len(),
                request.max_new_tokens,
                false,
                eos_id,
                im_end_id,
            ) {
                StepAction::StopEos => {
                    finish = Finish::Eos;
                    break;
                }
                StepAction::StopLimit => {
                    finish = Finish::Limit;
                    break;
                }
                StepAction::Continue => {}
            }
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
        // A reasoning opener that never closed cannot be stripped after the
        // fact (the sink already streamed it), so this only matters for
        // callers that read `GeneratedText::text`.
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
    /// Vision projector for image input (`--mmproj`); None means text-only.
    mmproj: Option<Arc<dyn TensorSource>>,
    threads: usize,
    /// Held so capacity / batch come from the one place that owns them.
    options: RuntimeOptions,
    arch: String,
}

impl Qwen3TextRuntime {
    pub fn new(options: RuntimeOptions) -> Result<Self, String> {
        let arch = arch_of(&options.source);
        let model = crate::models::qwen3::Qwen3Model::from_source(
            options.source.clone(),
            options.tokenizer.clone(),
            options.pool.clone(),
        )?;
        Ok(Self {
            model: Arc::new(model),
            tokenizer: options.tokenizer.clone(),
            pool: options.pool.clone(),
            mmproj: options.mmproj.clone(),
            threads: options.threads,
            options,
            arch,
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
        let capacity = self.options.capacity_for(self.model.config().n_ctx);
        let mut session = crate::models::qwen3::Qwen3Session::new_with_kv_state(
            &self.model,
            capacity,
            self.options.kv_format,
            crate::KvLifecycle::Ephemeral,
        )?;
        // Image path for qwen3vl: encode through the Qwen3-VL projector, splice
        // the placeholder run into the user turn, then inject the projected
        // embeddings AND the per-layer deepstack features. Mirrors
        // `app::text::multimodal::run_qwen3_family_multimodal`.
        let (token_ids, embeddings, deepstack, positions) = if request.images.is_empty() {
            let positions: Vec<_> = (0..request.token_ids.len()).map(|i| [i, 0, 0, 0]).collect();
            (request.token_ids.clone(), None, None, positions)
        } else {
            self.prepare_qwen3vl_image_prefill(&request.token_ids, &request.images)?
        };
        let input = crate::models::qwen3::Qwen3Input {
            token_ids: &token_ids,
            positions: &positions,
            embeddings: embeddings.as_deref(),
            deepstack_embeddings: deepstack.as_deref(),
        };
        let options_qwen3 = crate::models::qwen3::Qwen3GenerateOptions {
            max_new_tokens: request.max_new_tokens,
            temperature: request.sampling.temperature,
            prefill_batch_size: self.options.prefill_batch_size,
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

impl Qwen3TextRuntime {
    /// Encode images for the Qwen3-VL projector and prepare the prefill.
    ///
    /// Returns `(token_ids, embeddings, deepstack, positions)`. The placeholder
    /// run goes inside the user turn (right after `user\n`, before the text) —
    /// the position the CLI's multimodal path uses. Getting it wrong returns
    /// 200 but blinds the model, so the insert point is computed, not guessed.
    fn prepare_qwen3vl_image_prefill(
        &self,
        prompt_ids: &[u32],
        images: &[Vec<u8>],
    ) -> Result<
        (
            Vec<u32>,
            Option<Vec<f32>>,
            Option<Vec<f32>>,
            Vec<[usize; 4]>,
        ),
        String,
    > {
        let mmproj = self.mmproj.as_ref().ok_or_else(|| {
            "image input requires a vision projector; start the server with --mmproj <path>"
                .to_string()
        })?;
        let vision_start = self
            .tokenizer
            .special_token_id("vision_start")
            .ok_or("Required token missing: <|vision_start|>")?;
        let image_pad = self
            .tokenizer
            .special_token_id("image_pad")
            .ok_or("Required token missing: <|image_pad|>")?;
        let vision_end = self
            .tokenizer
            .special_token_id("vision_end")
            .ok_or("Required token missing: <|vision_end|>")?;
        let im_start = self
            .tokenizer
            .special_token_id("im_start")
            .ok_or("Required token missing: ")?;

        let n_embd = self.model.config().n_embd;
        // Projector family decides the encoder AND whether a system turn is
        // required. Qwen3-VL (merger) has per-layer deepstack; Qwen2.5-Omni
        // (qwen2vl) uses the qwen35-style projector and needs the virtual-human
        // system prompt.
        let qwen3vl_family = matches!(self.arch(), "qwen3vl" | "qwen3vlmoe");
        let mut media = Vec::new();
        let mut deepstack_layers: Vec<Vec<f32>> = Vec::new();
        let mut grid_shapes: Vec<(usize, usize)> = Vec::new();
        let mut rows_total = 0usize;
        for bytes in images {
            let image = crate::app::media::decode_image_bytes(bytes)?;
            let (encoded_media, encoded_deepstack, encoded_grids) = if qwen3vl_family {
                let encoded = crate::app::text::vision::encode_qwen3vl_image_dynamic(
                    mmproj.as_ref(),
                    &image,
                    self.threads,
                )?;
                (encoded.media, encoded.deepstack_layers, encoded.grid_shapes)
            } else {
                // Qwen2.5-Omni projector: same encoder type as qwen35, no
                // deepstack. One grid shape per image.
                let (grid, embedding) = crate::app::text::vision::encode_qwen35_image_dynamic(
                    mmproj.as_ref(),
                    &image,
                    self.threads,
                )?;
                (embedding, Vec::new(), vec![(grid.grid_h, grid.grid_w)])
            };
            let rows = if n_embd == 0 {
                0
            } else {
                encoded_media.len() / n_embd
            };
            if rows * n_embd != encoded_media.len() {
                return Err("Vision projector width does not match model width".into());
            }
            rows_total += rows;
            media.extend_from_slice(&encoded_media);
            for (layer, output) in encoded_deepstack.into_iter().enumerate() {
                if deepstack_layers.len() <= layer {
                    deepstack_layers.resize_with(layer + 1, Vec::new);
                }
                deepstack_layers[layer].extend_from_slice(&output);
            }
            grid_shapes.extend(encoded_grids);
        }
        if rows_total == 0 {
            return Err("Vision encoder produced zero image tokens".into());
            if rows_total > MAX_VISION_TOKENS {
                return Err(format!(
                    "image(s) expand to {rows_total} vision tokens, over the {MAX_VISION_TOKENS} \
            token budget; downscale the image before sending"
                ));
            }
        }

        // Qwen2.5-Omni requires the modality-aware system turn before the user
        // turn (the CLI injects it for the same family); without it the
        // assistant role drifts. Skip when one is already present.
        let mut base_ids: Vec<u32> = Vec::new();
        if !qwen3vl_family
            && !prompt_ids
                .iter()
                .any(|&id| Some(id) == self.tokenizer.special_token_id("system"))
        {
            let system = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.";
            let system_tokens = self.tokenizer.encode(
                system,
                crate::core::tokenizer::EncodeOptions {
                    add_special: false,
                    parse_special: false,
                },
            );
            crate::prompt::append_qwen_message_tokens(
                &mut base_ids,
                &self.tokenizer,
                "system",
                &system_tokens,
            )?;
        }
        base_ids.extend_from_slice(prompt_ids);
        let prompt_ids = &base_ids[..];

        // Splice `[vision_start][image_pad x rows][vision_end]` right after the
        // user turn's role tokens.
        let role_len = self
            .tokenizer
            .encode(
                "user\n",
                crate::core::tokenizer::EncodeOptions {
                    add_special: false,
                    parse_special: false,
                },
            )
            .len();
        let insert_at = prompt_ids
            .iter()
            .position(|&id| id == im_start)
            .map(|first| first + 1 + role_len)
            .unwrap_or(prompt_ids.len());
        let mut token_ids = Vec::with_capacity(prompt_ids.len() + rows_total + 2);
        token_ids.extend_from_slice(&prompt_ids[..insert_at]);
        token_ids.push(vision_start);
        token_ids.extend(std::iter::repeat(image_pad).take(rows_total));
        token_ids.push(vision_end);
        token_ids.extend_from_slice(&prompt_ids[insert_at..]);

        // Embedding lookup + media injection + media positions, in that order.
        let mut embeddings = self.model.embed_tokens(&token_ids)?;
        let deepstack = crate::app::text::vision::inject_qwen_media_embeddings(
            &token_ids,
            image_pad,
            &mut embeddings,
            &media,
            &deepstack_layers.concat(),
            n_embd,
        )?;
        let positions = crate::app::text::vision::build_qwen3_media_positions(
            &token_ids,
            image_pad,
            &grid_shapes,
        )?;
        Ok((
            token_ids,
            Some(embeddings),
            (!deepstack.is_empty()).then_some(deepstack),
            positions,
        ))
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
    /// Vision projector for image input (`--mmproj`); None means text-only.
    mmproj: Option<Arc<dyn TensorSource>>,
    threads: usize,
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
            mmproj: options.mmproj.clone(),
            threads: options.threads,
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
        // Image path: expand the prompt with vision placeholders, inject the
        // projected embeddings, and prefill through `session.step` (which takes
        // embeddings rather than token ids). Mirrors the CLI multimodal path.
        //
        // NOTE: the embedding injection needs `&Qwen35Model`, but
        // `Qwen35Session` needs `&mut Qwen35Model`, and the mutex is not
        // reentrant — so the image preparation runs BEFORE the session borrow.
        let (prefill_ids, prefill_embeddings, prefill_positions) = if request.images.is_empty() {
            (
                request.token_ids.clone(),
                None,
                crate::models::qwen35::build_qwen35_positions(&request.token_ids, None, &[])?.0,
            )
        } else {
            self.prepare_image_prefill(&request.token_ids, &request.images)?
        };
        let mut model = self.model.lock().map_err(|e| e.to_string())?;
        let mut session = crate::models::qwen35::Qwen35Session::new_with_prefill_batch_size(
            &mut model,
            prefill_ids.len() + request.max_new_tokens,
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
            let logits = if step == 0 {
                match &prefill_embeddings {
                    Some(embeddings) => {
                        session.step(embeddings, prefill_ids.len(), &prefill_positions)?
                    }
                    None => session.step_with_tokens(&prefill_ids, &prefill_positions)?,
                }
            } else {
                session.step_with_tokens(&token_ids[token_ids.len() - 1..], &decode_positions)?
            };
            let Some(id) = sample_step(&logits, request.sampling.temperature, eos_id, im_end_id)?
            else {
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
        // A reasoning opener that never closed cannot be stripped after the
        // fact (the sink already streamed it), so this only matters for
        // callers that read `GeneratedText::text`.
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
    /// Same sampler the CLI lfm2moe trunk uses, so both front-ends agree.
    sampler: Lfm2MoeSampler,
}

impl Lfm2MoeTextRuntime {
    pub fn new(options: RuntimeOptions) -> Result<Self, String> {
        let source = options.source.clone();
        let session = crate::models::lfm2moe::Lfm2MoeSession::from_source(
            source.as_ref(),
            options.threads,
            // Honor the caller's KV format, like every other adapter. Hard-
            // coding F16 here diverged from the CLI (which defaults to F32)
            // and the CLI/HTTP sentinel caught the resulting logit drift.
            options.kv_format,
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
            sampler: Lfm2MoeSampler::new(),
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
        // `im_end` is intentionally not a stop id here: lfm2moe's template has
        // no ChatML assistant turn, so the model never emits it and treating
        // it as a stop would truncate answers. Matches the CLI trunk.
        let mut decoder = self.tokenizer.streaming_decoder(false);
        let mut text = String::new();
        let mut token_ids = Vec::new();
        let mut finish = Finish::Limit;
        // Prefill token by token (the session has no batched prefill entry).
        // `prompt_len` tells the session which steps are prefill, matching the
        // CLI's fused loop; without it decode steps take the prefill branch.
        session.prompt_len = request.token_ids.len();
        let mut logits = Vec::new();
        for &token_id in &request.token_ids {
            logits = session.forward_token(token_id)?;
        }
        self.sampler.prime(&request.token_ids);
        for _ in 0..request.max_new_tokens {
            let mut logits_owned = logits;
            let id = self.sampler.sample(
                &mut logits_owned,
                request.sampling.temperature,
                request.sampling.repetition_penalty,
            );
            logits = logits_owned;
            match crate::ops::generation_runtime::stop_after_sample(
                id,
                token_ids.len(),
                request.max_new_tokens,
                false,
                eos_id,
                None,
            ) {
                StepAction::StopEos => {
                    finish = Finish::Eos;
                    break;
                }
                StepAction::StopLimit => {
                    finish = Finish::Limit;
                    break;
                }
                StepAction::Continue => {}
            }
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
        // A reasoning opener that never closed cannot be stripped after the
        // fact (the sink already streamed it), so this only matters for
        // callers that read `GeneratedText::text`.
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
#[cfg(test)]
mod tests {
    use super::defaults;
    use super::RuntimeOptions;
    use crate::app::cli::{validate_cli_options, CliOptions};
    use std::sync::Arc;

    fn cli_defaults() -> CliOptions {
        // `--model x` is the minimum the validator needs to reach the
        // generation-option resolution.
        CliOptions {
            model: std::path::PathBuf::from("x.gguf").into(),
            ..CliOptions::default()
        }
    }

    /// The whole point of `RuntimeOptions::defaults`: they must equal what the
    /// CLI resolves for the same (absent) flags. If the CLI changes a default,
    /// this test fails and forces both to move together.
    #[test]
    fn runtime_options_match_cli_defaults() {
        let options = cli_defaults();
        validate_cli_options(&options).expect("minimal CliOptions must validate");
        assert_eq!(
            defaults::PREFILL_BATCH_SIZE,
            options.effective_prefill_batch_size().unwrap(),
            "prefill batch size drifted from the CLI"
        );
        assert_eq!(
            defaults::MAX_CONTEXT,
            options.effective_max_context(),
            "max context drifted from the CLI"
        );
        assert_eq!(
            defaults::PREFILL_BATCH_SIZE,
            crate::core::prefill::DEFAULT_PREFILL_BATCH_SIZE,
            "prefill default must mirror core::prefill"
        );
    }

    /// `capacity_for` is the single capacity rule adapters must use.
    #[test]
    fn capacity_for_is_the_cli_rule() {
        let source: Arc<dyn crate::core::tensor::TensorSource> = Arc::new(StubSource);
        let pool = Arc::new(crate::core::thread_pool::ComputePool::new(1));
        let tokenizer = Arc::new(stub_tokenizer());
        let options = RuntimeOptions::from_model(source, pool, tokenizer);

        // Default max_context (8192) caps models with a bigger n_ctx …
        assert_eq!(options.capacity_for(1_000_000), defaults::MAX_CONTEXT);
        // … and a smaller n_ctx wins when the model is smaller.
        assert_eq!(options.capacity_for(512), 512);
        // Never zero, even for absurd metadata.
        assert_eq!(options.capacity_for(0), 1);

        // An override must take effect (server passes --max-context through).
        let capped = RuntimeOptions::from_model(
            Arc::new(StubSource),
            Arc::new(crate::core::thread_pool::ComputePool::new(1)),
            Arc::new(stub_tokenizer()),
        )
        .with_max_context(256);
        assert_eq!(capped.capacity_for(4096), 256);
        assert_eq!(capped.capacity_for(128), 128);
    }

    /// Every field starts at `defaults::*` — a caller that constructs via
    /// `from_model` and changes nothing must get the documented behaviour.
    #[test]
    fn from_model_fills_every_default() {
        let options = RuntimeOptions::from_model(
            Arc::new(StubSource),
            Arc::new(crate::core::thread_pool::ComputePool::new(1)),
            Arc::new(stub_tokenizer()),
        );
        assert_eq!(options.threads, defaults::THREADS);
        assert_eq!(options.max_context, defaults::MAX_CONTEXT);
        assert_eq!(options.prefill_batch_size, defaults::PREFILL_BATCH_SIZE);
        // F32, matching `CliOptions::default()` so HTTP and CLI agree.
        assert!(matches!(options.kv_format, crate::app::cli::KvFormat::F32));
        // The default must equal the CLI's own resolution, not merely the
        // module constant — otherwise the two front-ends drift again.
        let cli = crate::app::cli::CliOptions::default();
        assert_eq!(options.kv_format, cli.kv_format);
        assert_eq!(options.max_context, cli.effective_max_context());
        assert_eq!(
            options.prefill_batch_size,
            cli.effective_prefill_batch_size().unwrap()
        );
    }

    struct StubSource;
    impl crate::core::tensor::TensorSource for StubSource {
        fn metadata(&self, _key: &str) -> Option<&crate::core::tensor::MetaValue> {
            None
        }
        fn tensor_info(&self, _name: &str) -> Option<&crate::core::tensor::TensorInfo> {
            None
        }
        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    fn stub_tokenizer() -> crate::core::tokenizer::BPETokenizer {
        use crate::core::tokenizer::BPETokenizer;
        use std::collections::HashMap;
        let metadata: HashMap<String, crate::core::tensor::MetaValue> = HashMap::from([
            (
                "tokenizer.ggml.model".to_string(),
                crate::core::tensor::MetaValue::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.pre".to_string(),
                crate::core::tensor::MetaValue::String("qwen2".into()),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                crate::core::tensor::MetaValue::Array(
                    crate::core::tensor::MetaValueType::String,
                    ["u", "s", "e", "r", "\u{010a}", "H", "i"]
                        .map(|v| crate::core::tensor::MetaValue::String(v.into()))
                        .to_vec(),
                ),
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                crate::core::tensor::MetaValue::Array(
                    crate::core::tensor::MetaValueType::Uint32,
                    [1u32, 1, 1, 1, 1, 1, 1]
                        .map(crate::core::tensor::MetaValue::Uint32)
                        .to_vec(),
                ),
            ),
            (
                "tokenizer.ggml.merges".to_string(),
                crate::core::tensor::MetaValue::Array(
                    crate::core::tensor::MetaValueType::String,
                    vec![],
                ),
            ),
        ]);
        BPETokenizer::from_gguf_metadata(|k| metadata.get(k).cloned()).unwrap()
    }
}

impl Qwen35TextRuntime {
    /// Encode the request's images and expand the prompt for vision prefill.
    ///
    /// Returns `(ids, embeddings, positions)`: `ids` is the prompt with
    /// `[vision_start][image_pad x n_vis][vision_end]` spliced in at the start
    /// of the last user turn (before the assistant prefix), `embeddings` is the
    /// full prompt embedding row with the vision projections written over the
    /// placeholders, and `positions` is the m-rope position list for the
    /// expanded ids.
    ///
    /// Mirrors the CLI's qwen3.5 multimodal path (`app/text/multimodal.rs`),
    /// including where the placeholder run goes and the
    /// `inject_vision_embeddings` call.
    fn prepare_image_prefill(
        &self,
        prompt_ids: &[u32],
        images: &[Vec<u8>],
    ) -> Result<(Vec<u32>, Option<Vec<f32>>, Vec<[usize; 4]>), String> {
        // The caller must NOT hold the model lock: we take it here for the
        // embedding injection and the mutex is not reentrant.
        let mmproj = self.mmproj.as_ref().ok_or_else(|| {
            "image input requires a vision projector; start the server with --mmproj <path>"
                .to_string()
        })?;
        let n_embd = self
            .model
            .lock()
            .map(|model| model.config.n_embd)
            .unwrap_or(0);
        if n_embd == 0 {
            return Err("Qwen3.5 config has n_embd = 0".into());
        }
        let image_pad = self
            .tokenizer
            .special_token_id("image_pad")
            .ok_or("Required token missing: <|image_pad|>")?;
        let vision_start = self
            .tokenizer
            .special_token_id("vision_start")
            .ok_or("Required token missing: <|vision_start|>")?;
        let vision_end = self
            .tokenizer
            .special_token_id("vision_end")
            .ok_or("Required token missing: <|vision_end|>")?;
        let im_start = self
            .tokenizer
            .special_token_id("im_start")
            .ok_or("Required token missing: <|im_start|>")?;

        // 1. Encode every image, collecting grids and projected embeddings.
        let mut grids = Vec::with_capacity(images.len());
        let mut projected: Vec<f32> = Vec::new();
        let mut n_vis_total = 0usize;
        for bytes in images {
            let image = crate::app::media::decode_image_bytes(bytes)?;
            let (grid, embedding) = crate::app::text::vision::encode_qwen35_image_dynamic(
                mmproj.as_ref(),
                &image,
                self.threads,
            )?;
            n_vis_total += grid.token_count();
            grids.push(grid);
            projected.extend_from_slice(&embedding);
        }
        if n_vis_total == 0 {
            return Err("Vision encoder produced zero image tokens".into());
        }
        if n_vis_total > MAX_VISION_TOKENS {
            return Err(format!(
                "image(s) expand to {n_vis_total} vision tokens, over the {MAX_VISION_TOKENS} \
                 token budget; downscale the image before sending"
            ));
        }

        // 2. Splice the placeholder run into the user turn, right after the
        //    `user\n` role tokens and BEFORE the user's text — exactly where
        //    the CLI's multimodal path puts them
        //    (`append_qwen_message_tokens(..., "user", [vision][pad…][vision][text])`).
        //    Getting this position wrong silently blinds the model: it answered
        //    "black" for a solid-blue image when the run sat outside the turn.
        //    `tools.rs` builds the turn as `` `user\n` <text> ``,
        //    so the insert point is (first ``) + 1 + len(encode("user\n")).
        let role_len = self
            .tokenizer
            .encode(
                "user\n",
                crate::core::tokenizer::EncodeOptions {
                    add_special: false,
                    parse_special: false,
                },
            )
            .len();
        let insert_at = prompt_ids
            .iter()
            .position(|&id| id == im_start)
            .map(|first| first + 1 + role_len)
            .unwrap_or(prompt_ids.len());
        let mut ids = Vec::with_capacity(prompt_ids.len() + n_vis_total + 2);
        ids.extend_from_slice(&prompt_ids[..insert_at]);
        ids.push(vision_start);
        ids.extend(std::iter::repeat(image_pad).take(n_vis_total));
        ids.push(vision_end);
        ids.extend_from_slice(&prompt_ids[insert_at..]);

        // 3. m-rope positions over the expanded ids.
        let (positions, _next) =
            crate::models::qwen35::build_qwen35_positions(&ids, Some(image_pad), &grids)?;

        // 4. Inject the vision projections over the placeholders.
        let prompt_i32: Vec<i32> = ids
            .iter()
            .copied()
            .map(|id| i32::try_from(id).map_err(|_| format!("Token ID {id} exceeds i32")))
            .collect::<Result<_, _>>()?;
        let model = self.model.lock().map_err(|e| e.to_string())?;
        let embeddings = crate::app::text::vision::inject_vision_embeddings(
            &model,
            &prompt_i32,
            Some(i32::try_from(image_pad).map_err(|_| "image_pad exceeds i32")?),
            &projected,
            n_vis_total,
            n_embd,
        )?;
        drop(model);
        Ok((ids, Some(embeddings), positions))
    }
}
