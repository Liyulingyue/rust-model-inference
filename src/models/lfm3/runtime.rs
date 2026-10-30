//! Text-only HTTP adapter; shares the CLI's session and sampler.
use std::sync::Arc;

use super::{Lfm3Model, Lfm3Session};
use crate::app::text::runtime::RuntimeOptions;
use crate::core::tensor::TensorSource;
use crate::ops::generation_runtime::{
    Finish, Flow, GeneratedText, GenerationRequest, TextRuntime, TokenSink,
};
use crate::ops::sampling::Lfm2MoeSampler;

pub struct Lfm3TextRuntime {
    // Drop borrowed weights before their backing mmap.
    model: Lfm3Model<'static>,
    options: RuntimeOptions,
    _source: Arc<dyn TensorSource>,
}

impl Lfm3TextRuntime {
    pub fn new(options: RuntimeOptions) -> Result<Self, String> {
        options.compute.check_build()?;
        options.compute.validate_text_arch("lfm3")?;
        crate::core::prefill::checked_prefill_batch_size(Some(options.prefill_batch_size))?;
        if options.max_context == 0 || options.mmproj.is_some() {
            return Err("LFM3 requires positive context capacity and text input only".into());
        }
        let source = options.source.clone();
        let model = Lfm3Model::from_source_with_pool(source.as_ref(), options.pool.clone())?;
        // SAFETY: the immutable source is retained until after model is dropped.
        let model: Lfm3Model<'static> = unsafe { std::mem::transmute(model) };
        Ok(Self {
            model,
            options,
            _source: source,
        })
    }
}

impl TextRuntime for Lfm3TextRuntime {
    fn arch(&self) -> &str {
        "lfm3"
    }

    fn context_length(&self) -> usize {
        self.model.config.n_ctx.min(self.options.max_context)
    }

    fn generate(
        &mut self,
        request: &GenerationRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<GeneratedText, String> {
        if !request.images.is_empty() {
            return Err("LFM3 supports text input only".into());
        }
        let capacity = self.context_length();
        if request.token_ids.is_empty()
            || request
                .token_ids
                .len()
                .saturating_add(request.max_new_tokens.saturating_sub(1))
                > capacity
        {
            return Err(format!(
                "LFM3 prompt and generation exceed context capacity {capacity} or prompt is empty"
            ));
        }
        let mut session = Lfm3Session::new_with_compute(
            &self.model,
            self.options.kv_format,
            capacity,
            self.options.compute,
        )?;
        let mut logits =
            session.forward_logits_chunked(&request.token_ids, self.options.prefill_batch_size)?;
        let mut sampler = Lfm2MoeSampler::new();
        sampler.prime(&request.token_ids);
        let mut decoder = self.options.tokenizer.streaming_decoder(false);
        let mut text = String::new();
        let mut token_ids = Vec::new();
        let mut finish = Finish::Limit;
        for step in 0..request.max_new_tokens {
            let token = sampler.sample(
                &mut logits,
                request.sampling.temperature,
                request.sampling.repetition_penalty,
            );
            if Some(token) == self.options.tokenizer.eos_id() {
                finish = Finish::Eos;
                break;
            }
            token_ids.push(token);
            let chunk = decoder.push(token);
            text.push_str(&chunk);
            if !chunk.is_empty() && sink.push_text(&chunk) == Flow::Stop {
                finish = Finish::Cancelled;
                break;
            }
            if step + 1 < request.max_new_tokens {
                logits = session.forward_logits_chunked(&[token], 1)?;
            }
        }
        let tail = decoder.finish();
        if finish != Finish::Cancelled && !tail.is_empty() {
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
