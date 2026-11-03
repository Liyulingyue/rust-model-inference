//! HTTP `TextRuntime` adapter for Mellum 2.x (`arch = "mellum"`).
//!
//! Mellum's forward is a single-token KV append ([`MellumRunner::forward_token`]),
//! so the adapter allocates a fresh [`MellumRunner`] per request and walks the
//! prompt one token at a time before decoding. Per-request allocation is the
//! price of a stateless API; the KV cache is the largest buffer and the model
//! stays shared behind the mutex.

use std::sync::{Arc, Mutex};

use crate::app::text::runtime::RuntimeOptions;
use crate::models::mellum::trunk::forward::MellumRunner;
use crate::models::mellum::trunk::weights::MellumModel;
use crate::ops::apply_repetition_penalty;
use crate::ops::generation_runtime::{
    Finish, Flow, GeneratedText, GenerationRequest, TextRuntime, TokenSink,
};
use crate::ops::sampling::sample_greedy_or_temperature;

pub struct MellumTextRuntime {
    model: MellumModel,
    tokenizer: Arc<crate::core::tokenizer::BPETokenizer>,
    arch: String,
    context_length: usize,
    kv_format: crate::app::cli::KvFormat,
}

impl MellumTextRuntime {
    pub fn new(options: RuntimeOptions) -> Result<Self, String> {
        let source = options.source.clone();
        let model = MellumModel::from_source(
            Arc::clone(&source),
            Arc::clone(&options.tokenizer),
            options.pool.clone(),
        )
        .map_err(|error| format!("Failed to load Mellum model: {error}"))?;
        let context_length = options.max_context.min(model.config().n_ctx).max(1);
        Ok(Self {
            model,
            tokenizer: options.tokenizer,
            arch: "mellum".to_string(),
            context_length,
            kv_format: options.kv_format,
        })
    }
}

impl TextRuntime for MellumTextRuntime {
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
        if request.token_ids.is_empty() {
            return Err("Mellum generate: empty prompt".into());
        }
        let required = request.token_ids.len() + request.max_new_tokens;
        if required > self.context_length {
            return Err(format!(
                "Mellum generate: prompt {} + max_new_tokens {} exceeds the {} context",
                request.token_ids.len(),
                request.max_new_tokens,
                self.context_length
            ));
        }
        let mut runner = MellumRunner::new(&self.model, self.context_length, self.kv_format)?;
        for (position, &token) in request.token_ids.iter().enumerate() {
            runner.forward_token(token, position)?;
        }

        let eos_id = self.tokenizer.eos_id();
        let mut decoder = self.tokenizer.streaming_decoder(false);
        let mut text = String::new();
        let mut token_ids = Vec::with_capacity(request.max_new_tokens);
        let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for &id in &request.token_ids {
            *counts.entry(id).or_insert(0) += 1;
        }
        let mut finish = Finish::Limit;
        let mut position = request.token_ids.len();

        for _ in 0..request.max_new_tokens {
            let mut logits = runner.logits().to_vec();
            apply_repetition_penalty(&mut logits, &counts, request.sampling.repetition_penalty);
            let id = sample_greedy_or_temperature(&logits, request.sampling.temperature)?;
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
            if token_ids.len() >= request.max_new_tokens || position >= self.context_length {
                break;
            }
            runner.forward_token(id, position)?;
            position += 1;
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

/// The model is only borrowed through `&self`, so a single lock around it is
/// enough; the server keeps one runtime per model and serialises requests.
pub type SharedMellumRuntime = Mutex<MellumTextRuntime>;
