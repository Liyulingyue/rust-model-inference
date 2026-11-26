//! Text-only entry for the `hy_v3` decoder (Hy-MT3-30B-A3B).
//!
//! `hy_v3` is shared by two different products:
//!
//! * Hy-SimulMT — a speech-translation checkpoint that bundles the
//!   `hyv3_speech` audio tower in the same GGUF (`clip.projector_type`), driven
//!   by [`super::HySimulMt`] and reachable only through `--audio`;
//! * Hy-MT3-30B-A3B — a plain text translation model with the same decoder
//!   contract and no audio tower at all.
//!
//! The decoder in `decoder.rs` / `weights.rs` / `moe.rs` is already the generic
//! `hy_v3` trunk (dense leading layer, sigmoid router + `exp_probs_b` selection
//! bias, weight normalisation, ungated shared expert), so the text product only
//! needs an entry point: this adapter implements [`TextRuntime`] on top of
//! [`HySession`] and is registered for `arch == "hy_v3"` in
//! `app::text::build_text_runtime`. Both CLI and HTTP then share it.
use super::{HyDecoder, HySession};
use crate::app::text::RuntimeOptions;
use crate::compute::ComputePolicy;
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::BPETokenizer;
use crate::models::llama::trunk::sample_defaults;
use crate::ops::generation_runtime::{
    Finish, Flow, GeneratedText, GenerationRequest, TextRuntime, TokenSink,
};
use crate::ops::sampling::LlamaSampler;
use std::sync::Arc;

/// True when the GGUF carries the bundled Hy-SimulMT speech tower.
///
/// `general.architecture` alone cannot tell the two `hy_v3` products apart, and
/// routing a text-only Hy-MT3 checkpoint into the audio backend (or vice versa)
/// would either fail late or, worse, silently generate without the audio
/// conditioning the model was trained for. The tower advertises itself with
/// `clip.projector_type = hyv3_speech` (see `audio.rs`), so that key is the
/// discriminator everywhere the two paths are selected.
pub fn has_speech_tower(source: &dyn TensorSource) -> bool {
    source
        .metadata("clip.projector_type")
        .and_then(|value| value.to_string_val())
        == Some("hyv3_speech")
}

/// Text-generation adapter for a text-only `hy_v3` checkpoint.
pub struct HyV3TextRuntime {
    // `session` borrows the `HyDecoder` allocation, so it must be dropped
    // before `decoder` hands the allocation back. Rust drops struct fields in
    // declaration order, hence this order.
    session: std::sync::Mutex<HySession<'static>>,
    decoder: Arc<HyDecoder>,
    tokenizer: Arc<BPETokenizer>,
    source: Arc<dyn TensorSource>,
    compute: ComputePolicy,
    capacity: usize,
    prefill_batch_size: usize,
}

impl HyV3TextRuntime {
    pub fn new(options: RuntimeOptions) -> Result<Self, String> {
        options.compute.check_build()?;
        let arch = options
            .source
            .metadata("general.architecture")
            .and_then(|value| value.to_string_val())
            .unwrap_or_default();
        if arch != "hy_v3" {
            return Err(format!(
                "Hy-MT3 text runtime requires general.architecture=hy_v3, got {arch:?}"
            ));
        }
        if has_speech_tower(options.source.as_ref()) {
            return Err(
                "this hy_v3 checkpoint bundles the hyv3_speech audio tower (Hy-SimulMT); \
                 use --audio/--language instead of the text path"
                    .into(),
            );
        }
        if options.mmproj.is_some() {
            return Err(
                "hy_v3 text generation has no vision path; --mmproj is not supported".into(),
            );
        }
        let _scope = options.compute.enter_legacy_scope();
        let decoder = Arc::new(HyDecoder::from_source(
            options.source.clone(),
            options.tokenizer.clone(),
            options.pool.clone(),
        )?);
        let n_ctx = decoder.config().n_ctx;
        let capacity = options.max_context.min(n_ctx).max(1);
        // SAFETY: `HySession` holds a reference to the `HyDecoder` *allocation*,
        // and `decoder` is an `Arc` owned by this struct. Moving the `Arc` into
        // the struct copies only the pointer, so the allocation address the
        // session borrows never moves, and the `Arc` keeps it alive for at
        // least as long as the session (see the field-order note above). The
        // tensor bytes themselves are owned by `HyDecoder::_source`, which the
        // decoder holds for its whole life. Same pattern as
        // `models::lfm3::runtime` and `LlamaTextRuntime`.
        let session = HySession::new(&decoder, capacity, options.kv_format, options.compute)?;
        let session: HySession<'static> = unsafe { std::mem::transmute(session) };
        Ok(Self {
            session: std::sync::Mutex::new(session),
            decoder,
            tokenizer: options.tokenizer,
            source: options.source,
            compute: options.compute,
            capacity,
            prefill_batch_size: options.prefill_batch_size.max(1),
        })
    }

    /// Token positions a request consumes: the prompt plus every generated
    /// token except the last, whose hidden state is never projected (see the
    /// loop in [`Self::generate`]).
    fn positions_needed(prompt_len: usize, max_new_tokens: usize) -> usize {
        prompt_len.saturating_add(max_new_tokens.saturating_sub(1))
    }
}

impl TextRuntime for HyV3TextRuntime {
    fn arch(&self) -> &str {
        "hy_v3"
    }

    fn context_length(&self) -> usize {
        self.capacity
    }

    fn validate_request(
        &self,
        token_ids: &[u32],
        images: &[Vec<u8>],
        max_new_tokens: usize,
    ) -> Result<(), String> {
        if !images.is_empty() {
            return Err("hy_v3 has no vision path; --mmproj/--image are not supported".into());
        }
        if token_ids.is_empty() {
            return Err("hy_v3 requires a non-empty prompt".into());
        }
        let needed = Self::positions_needed(token_ids.len(), max_new_tokens);
        if needed > self.capacity {
            return Err(format!(
                "hy_v3 request needs {needed} token positions but the KV cache holds {}; \
                 raise --max-context or shorten the prompt",
                self.capacity
            ));
        }
        Ok(())
    }

    fn generate(
        &mut self,
        request: &GenerationRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<GeneratedText, String> {
        let _scope = self.compute.enter_legacy_scope();
        if !request.images.is_empty() {
            return Err("hy_v3 has no vision path; --mmproj/--image are not supported".into());
        }
        if request.token_ids.is_empty() {
            return Err("hy_v3 requires a non-empty prompt".into());
        }
        let needed = Self::positions_needed(request.token_ids.len(), request.max_new_tokens);
        if needed > self.capacity {
            return Err(format!(
                "hy_v3 request needs {needed} token positions but the KV cache holds {}",
                self.capacity
            ));
        }
        let mut session = self.session.lock().map_err(|e| e.to_string())?;
        // The session is cached across requests so the prepared weight layout
        // is reused; the sequence state is not, so every request starts at
        // position 0 with an empty attention window.
        session.reset();
        let eos_id = self.tokenizer.eos_id();
        let mut logits = session.forward(&request.token_ids, None, self.prefill_batch_size)?;
        let (top_k, top_p) = sample_defaults(self.source.as_ref());
        let mut sampler = LlamaSampler::new(top_k, top_p);
        sampler.prime(&request.token_ids);
        let mut decoder = self.tokenizer.streaming_decoder(false);
        let mut text = String::new();
        let mut token_ids = Vec::with_capacity(request.max_new_tokens);
        let mut finish = Finish::Limit;
        for step in 0..request.max_new_tokens {
            let id = sampler.sample(
                &mut logits,
                request.sampling.temperature,
                request.sampling.repetition_penalty,
            );
            if Some(id) == eos_id {
                finish = Finish::Eos;
                break;
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
            // The last step's hidden state is never projected, so skip its
            // forward pass: on the 30B-A3B this is the most expensive step.
            if step + 1 < request.max_new_tokens {
                logits = session.forward(&[id], None, 1)?;
            }
        }
        let tail = decoder.finish();
        if !tail.is_empty() {
            text.push_str(&tail);
            if finish != Finish::Cancelled {
                sink.push_text(&tail);
            }
        }
        Ok(GeneratedText {
            text,
            token_ids,
            finish,
        })
    }
}
