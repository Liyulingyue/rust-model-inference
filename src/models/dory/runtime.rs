//! HTTP `TextRuntime` adapter for NVIDIA Nemotron-Labs Dory 1.0
//! (`arch = "dory"`).
//!
//! Dory's forward API is stateless per call: [`DoryModel::prefill`] takes
//! the token ids plus a session and returns the logits for the last
//! position. The adapter therefore allocates a fresh [`DorySession`] per
//! request, prefills the whole prompt in one call and then walks the
//! decode one token at a time.
//!
//! The weights stay shared behind the server's mutex; only the KV cache is
//! per-request, so the memory cost of a request scales with the tokens it
//! actually uses rather than with the model's 262144-token context.

use std::collections::HashMap;
use std::sync::Arc;

use crate::app::cli::KvFormat;
use crate::app::text::runtime::RuntimeOptions;
use crate::core::tokenizer::BPETokenizer;
use crate::models::dory::DoryModel;
use crate::ops::apply_repetition_penalty;
use crate::ops::generation_runtime::{
    Finish, Flow, GeneratedText, GenerationRequest, TextRuntime, TokenSink,
};
use crate::ops::sampling::sample_greedy_or_temperature;

pub struct DoryTextRuntime {
    model: DoryModel,
    tokenizer: Arc<BPETokenizer>,
    arch: String,
    context_length: usize,
    kv_format: KvFormat,
}

impl DoryTextRuntime {
    pub fn new(options: RuntimeOptions) -> Result<Self, String> {
        let mut model = DoryModel::from_source(options.source.clone(), options.threads)
            .map_err(|error| format!("Failed to load Dory model: {error}"))?;
        // The server already built a pool sized for `--threads`; reuse it so
        // the row-partitioned quantized kernels see the same parallelism as
        // the rest of the server. The per-row reduction order does not depend
        // on the pool size, so the numerics are unchanged.
        model.pool = options.pool.clone();
        let context_length = options.max_context.min(model.config.n_ctx).max(1);
        Ok(Self {
            model,
            tokenizer: options.tokenizer,
            arch: "dory".to_string(),
            context_length,
            kv_format: options.kv_format,
        })
    }
}

impl TextRuntime for DoryTextRuntime {
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
            return Err("Dory generate: empty prompt".into());
        }
        let required = request.token_ids.len() + request.max_new_tokens;
        if required > self.context_length {
            return Err(format!(
                "Dory generate: prompt {} + max_new_tokens {} exceeds the {} context",
                request.token_ids.len(),
                request.max_new_tokens,
                self.context_length
            ));
        }
        // Size the session to the request, not to the model's full context:
        // the Vulkan path pre-allocates the KV region for the whole capacity
        // and the CPU path grows its cache per token.
        let mut session = self.model.new_session_with_compute(
            required.min(self.context_length),
            self.kv_format,
            crate::compute::ComputePolicy::legacy(),
        )?;
        let mut logits = self.model.prefill(&request.token_ids, &mut session)?;

        let eos_id = self.tokenizer.eos_id();
        let im_end_id = self.tokenizer.special_token_id("im_end");
        let mut decoder = self.tokenizer.streaming_decoder(false);
        let mut text = String::new();
        let mut token_ids = Vec::with_capacity(request.max_new_tokens);
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for &id in &request.token_ids {
            *counts.entry(id).or_insert(0) += 1;
        }
        let mut finish = Finish::Limit;

        for _ in 0..request.max_new_tokens {
            apply_repetition_penalty(&mut logits, &counts, request.sampling.repetition_penalty);
            let id = sample_greedy_or_temperature(&logits, request.sampling.temperature)?;
            if Some(id) == eos_id || Some(id) == im_end_id {
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
            logits = self.model.prefill(&[id], &mut session)?;
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
pub type SharedDoryRuntime = std::sync::Mutex<DoryTextRuntime>;
