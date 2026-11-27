//! HTTP `TextRuntime` adapter for Falcon-H1 (`arch = "falcon-h1"`).
//!
//! Falcon-H1 has no `Qwen3Session`-style stateful session — the trunk
//! exposes `FalconH1Model::prefill(&[u32], &mut FalconH1Scratch)`, which
//! advances the KV cache **and** the Mamba2 scan state / conv history in
//! one call. That is enough to serve the `TextRuntime` contract directly:
//! a fresh scratch per request, prefill the prompt, then loop
//! `prefill(&[sampled_id])`.
//!
//! The CLI (`falcon_h1::trunk::forward::run_inference`) prints to stdout
//! and owns its own RNG, so this adapter is a separate path rather than a
//! refactor — but it uses the *same* `sample_argmax` helper so the two
//! front-ends agree on greedy / temperature sampling.

use std::sync::{Arc, Mutex};

use crate::app::text::runtime::RuntimeOptions;
use crate::core::tensor::TensorSource;
use crate::models::falcon_h1::trunk::forward::{sample_falcon_h1, FalconH1Model, FalconH1Scratch};
use crate::ops::apply_repetition_penalty;
use crate::ops::generation_runtime::{
    Finish, Flow, GeneratedText, GenerationRequest, TextRuntime, TokenSink,
};

/// `TextRuntime` impl for `arch = "falcon-h1"`.
pub struct FalconH1TextRuntime {
    compute: crate::compute::ComputePolicy,
    model: Mutex<FalconH1Model>,
    /// Per-request scratch, behind the same lock as the model: `prefill`
    /// takes `&self` on the model but `&mut scratch`, and they must be
    /// from the same generation, so one lock covers both.
    scratch: Mutex<Option<FalconH1Scratch>>,
    tokenizer: Arc<crate::core::tokenizer::BPETokenizer>,
    arch: String,
    context_length: usize,
}

impl FalconH1TextRuntime {
    pub fn new(options: RuntimeOptions) -> Result<Self, String> {
        options.compute.check_build()?;
        options.compute.validate_text_arch("falcon-h1")?;
        let _scope = options.compute.enter_legacy_scope();
        let source = options.source.clone();
        let context_length = options.max_context;
        let model = FalconH1Model::from_source(source, 0)
            .map_err(|error| format!("Failed to load Falcon-H1 model: {error}"))?;
        let context_length = context_length.min(model.config.n_ctx).max(1);
        Ok(Self {
            compute: options.compute,
            model: Mutex::new(model),
            scratch: Mutex::new(None),
            tokenizer: options.tokenizer,
            arch: "falcon-h1".to_string(),
            context_length,
        })
    }
}

impl TextRuntime for FalconH1TextRuntime {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn context_length(&self) -> usize {
        self.context_length
    }

    fn generate(
        &mut self,
        request: &GenerationRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<GeneratedText, String> {
        let _scope = self.compute.enter_legacy_scope();
        if request.token_ids.is_empty() {
            return Err("Falcon-H1 generate: empty prompt".into());
        }
        let model = self.model.lock().map_err(|e| e.to_string())?;
        let mut scratch_guard = self.scratch.lock().map_err(|e| e.to_string())?;
        // Fresh scratch per request: the trunk's `prefill` mutates KV cache
        // + SSM scan state + conv history in place, so reusing a previous
        // generation's scratch would leak state into this one. Rebuild only
        // when the capacity is insufficient (the common case reuses).
        let capacity = request.token_ids.len() + request.max_new_tokens;
        if scratch_guard.as_ref().is_none_or(|s| s.capacity < capacity) {
            *scratch_guard = Some(FalconH1Scratch::new(&model.config, capacity));
        }
        let scratch = scratch_guard.as_mut().expect("scratch just built");
        scratch.reset();

        let eos_id = self.tokenizer.eos_id();
        let mut decoder = self.tokenizer.streaming_decoder(false);
        let mut text = String::new();
        let mut token_ids = Vec::with_capacity(request.max_new_tokens);
        let mut finish = Finish::Limit;

        let mut logits = model
            .prefill(&request.token_ids, scratch)
            .map_err(|e| format!("Falcon-H1 prefill failed: {e}"))?;

        // Repetition penalty tracking, llama.cpp semantics (count each
        // occurrence so penalty^count divides the logit).
        let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for &id in &request.token_ids {
            *counts.entry(id).or_insert(0) += 1;
        }

        for _ in 0..request.max_new_tokens {
            apply_repetition_penalty(&mut logits, &counts, request.sampling.repetition_penalty);
            let id = sample_falcon_h1(&logits, request.sampling.temperature);
            if Some(id) == eos_id {
                finish = Finish::Eos;
                break;
            }
            token_ids.push(id);
            *counts.entry(id).or_insert(0) += 1;
            let chunk = decoder.push(id);
            if !chunk.is_empty() {
                text.push_str(&chunk);
                if sink.push_text(&chunk) == Flow::Stop {
                    finish = Finish::Cancelled;
                    break;
                }
            }
            if token_ids.len() >= request.max_new_tokens {
                break;
            }
            logits = model
                .prefill(&[id], scratch)
                .map_err(|e| format!("Falcon-H1 decode failed: {e}"))?;
        }

        // Flush any incomplete UTF-8 the streaming decoder held back.
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
