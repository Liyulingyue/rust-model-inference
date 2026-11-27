use super::{AionModel, AionSession};
use crate::app::text::RuntimeOptions;
use crate::compute::ComputePolicy;
use crate::core::tokenizer::BPETokenizer;
use crate::ops::generation_runtime::{
    Finish, Flow, GeneratedText, GenerationRequest, TextRuntime, TokenSink,
};
use crate::ops::sampling::LlamaSampler;
use std::time::Instant;

pub(crate) struct AionTextRuntime {
    // The options retain the immutable mmap until after the model is dropped.
    model: AionModel<'static>,
    options: RuntimeOptions,
}

impl AionTextRuntime {
    pub(crate) fn new(options: RuntimeOptions) -> Result<Self, String> {
        options.compute.check_build()?;
        if options.mmproj.is_some() {
            return Err("Aion supports text only; --mmproj is unsupported".into());
        }
        if options.max_context == 0 || options.prefill_batch_size == 0 {
            return Err("Aion requires positive context capacity and prefill batch size".into());
        }
        #[cfg(feature = "vulkan")]
        if options.compute == ComputePolicy::Vulkan {
            options.compute.context().map_err(|e| e.to_string())?;
        }
        let model = AionModel::load(options.source.as_ref(), options.pool.clone())?;
        if options.tokenizer.vocab_size() != model.vocab_size() {
            return Err("Aion tokenizer vocabulary does not match the checkpoint".into());
        }
        if !options
            .tokenizer
            .eos_id()
            .is_some_and(|id| model.stop_ids.contains(&id))
        {
            return Err("Aion tokenizer EOS is missing from the generation stop tokens".into());
        }
        // SAFETY: TensorSource remains immutable and is retained in options,
        // which is dropped after model and outlives all sessions.
        let model = unsafe { std::mem::transmute::<AionModel<'_>, AionModel<'static>>(model) };
        Ok(Self { model, options })
    }

    pub(crate) fn generate_text(
        &self,
        request: &GenerationRequest,
        sink: &mut dyn TokenSink,
        bench: bool,
        profile: bool,
    ) -> Result<GeneratedText, String> {
        self.validate_request(&request.token_ids, &request.images, request.max_new_tokens)?;
        let capacity = request.token_ids.len() + request.max_new_tokens.saturating_sub(1);
        let mut session = AionSession::new(
            &self.model,
            self.options.compute,
            self.options.kv_format,
            capacity,
            self.options.prefill_batch_size.min(request.token_ids.len()),
        )?;
        let start = Instant::now();
        let mut logits = session.forward(&request.token_ids)?;
        let prefill = start.elapsed();
        let mut sampler = LlamaSampler::new(50, 1.0);
        sampler.prime(&request.token_ids);
        let mut decoder = self.options.tokenizer.streaming_decoder(false);
        let mut token_ids = Vec::new();
        let mut text = String::new();
        let mut finish = Finish::Limit;
        let decode_start = Instant::now();
        for step in 0..request.max_new_tokens {
            let token = sampler.sample(
                logits,
                request.sampling.temperature,
                request.sampling.repetition_penalty,
            );
            if !bench && self.model.stop_ids.contains(&token) {
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
                logits = session.forward(&[token])?;
            }
        }
        if finish != Finish::Cancelled {
            let tail = decoder.finish();
            text.push_str(&tail);
            if !tail.is_empty() && sink.push_text(&tail) == Flow::Stop {
                finish = Finish::Cancelled;
            }
        }
        if profile || bench {
            let decode = decode_start.elapsed();
            eprintln!(
                "Aion prefill: {} tokens in {prefill:?}; generation: {} tokens in {decode:?}",
                request.token_ids.len(),
                token_ids.len()
            );
        }
        Ok(GeneratedText {
            text,
            token_ids,
            finish,
        })
    }

    pub(crate) fn forward_logits(
        &self,
        tokens: &[u32],
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        self.validate_request(tokens, &[], 0)?;
        let mut session = AionSession::new(
            &self.model,
            self.options.compute,
            self.options.kv_format,
            tokens.len(),
            self.options.prefill_batch_size.min(tokens.len()),
        )?;
        let start = Instant::now();
        let logits = session.forward(tokens)?.to_vec();
        Ok((logits, start.elapsed()))
    }

    pub(crate) fn tokenizer(&self) -> &BPETokenizer {
        &self.options.tokenizer
    }
}

impl TextRuntime for AionTextRuntime {
    fn arch(&self) -> &str {
        "aion_instruct"
    }

    fn context_length(&self) -> usize {
        self.options.max_context.min(self.model.context_length())
    }

    fn validate_request(
        &self,
        ids: &[u32],
        images: &[Vec<u8>],
        max_tokens: usize,
    ) -> Result<(), String> {
        if !images.is_empty() {
            return Err("Aion supports text input only".into());
        }
        self.model.validate_tokens(ids)?;
        let capacity = self.context_length();
        if ids.len().saturating_add(max_tokens.saturating_sub(1)) > capacity {
            return Err(format!(
                "Aion prompt and generation exceed context capacity {capacity}"
            ));
        }
        Ok(())
    }

    fn generate(
        &mut self,
        request: &GenerationRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<GeneratedText, String> {
        self.generate_text(request, sink, false, false)
    }
}
