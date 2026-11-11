//! HTTP `TextRuntime` adapter for the Nemotron-3 Nano family
//! (`arch = "nemotron_h"` for the 4B, `arch = "nemotron_h_moe"` for the
//! 30B-A3B mixture-of-experts variant).
//!
//! The trunk's forward is a plain prefill API: [`NemotronModel::prefill`]
//! takes the token ids plus a scratch pad and returns the logits of the
//! last position. The adapter therefore prefills the whole prompt in one
//! call and then walks the decode one token at a time, exactly like the
//! CLI's `run_inference`.
//!
//! The weights are shared behind the server's mutex; only the scratch pad
//! (the hidden state, the KV cache and the Mamba-2 recurrence) is per
//! request, so a request costs the memory of the tokens it actually uses
//! rather than of the model's whole context.

use std::collections::HashMap;
use std::sync::Arc;

use crate::app::cli::KvFormat;
use crate::app::text::runtime::RuntimeOptions;
use crate::core::tokenizer::BPETokenizer;
use crate::models::nemotron_h::trunk::forward::{NemotronModel, NemotronScratch};
use crate::ops::apply_repetition_penalty;
use crate::ops::generation_runtime::{
    Finish, Flow, GeneratedText, GenerationRequest, TextRuntime, TokenSink,
};
use crate::ops::sampling::sample_greedy_or_temperature;

pub struct NemotronTextRuntime {
    model: NemotronModel,
    tokenizer: Arc<BPETokenizer>,
    arch: String,
    context_length: usize,
    kv_format: KvFormat,
    /// The device arena's row limit, `None` when no device is attached.
    ///
    /// A request whose prompt has more rows than this cannot use the
    /// device: the arena's row buffers would overflow. Such a request
    /// runs on the CPU instead.
    #[cfg(feature = "vulkan")]
    device_rows: Option<usize>,
    /// Whether an over-long request was already reported to the user.
    #[cfg(feature = "vulkan")]
    warned_long_request: bool,
}

/// The longest prompt the device path takes, and the number of token
/// positions its KV cache holds.
///
/// The device session is built once -- building it uploads the weights
/// to the device -- so its arena has to be sized for the requests the
/// server will see rather than for one of them. The row limit bounds
/// the arena's row buffers; the position limit bounds the KV cache.
/// A request beyond either limit runs on the CPU.
#[cfg(feature = "vulkan")]
const DEVICE_ROWS: usize = 512;
#[cfg(feature = "vulkan")]
const DEVICE_POSITIONS: usize = 2048;

impl NemotronTextRuntime {
    pub fn new(options: RuntimeOptions) -> Result<Self, String> {
        let mut model = NemotronModel::from_source(options.source.clone(), options.threads)
            .map_err(|error| format!("Failed to load Nemotron model: {error}"))?;
        // The server already built a pool sized for `--threads`; reuse it so
        // the row-partitioned quantized kernels see the same parallelism as
        // the rest of the server. The per-row reduction order does not
        // depend on the pool size, so the numerics are unchanged.
        model.pool = options.pool.clone();
        let arch = model.config.architecture.clone();
        let context_length = options.max_context.min(model.config.n_ctx).max(1);
        // The device session is built once, for the whole server lifetime.
        // `Auto` falls back to the CPU when the device cannot take the
        // model; `Vulkan` refuses the server instead, so a strict device
        // run never silently answers from the CPU.
        #[cfg(feature = "vulkan")]
        let device_rows = {
            let positions = context_length.min(DEVICE_POSITIONS);
            let rows = positions.min(DEVICE_ROWS);
            model.attach_gpu(positions, rows, options.compute)?;
            model.gpu_max_rows()
        };
        Ok(Self {
            model,
            tokenizer: options.tokenizer,
            arch,
            context_length,
            kv_format: options.kv_format,
            #[cfg(feature = "vulkan")]
            device_rows,
            #[cfg(feature = "vulkan")]
            warned_long_request: false,
        })
    }
}

impl TextRuntime for NemotronTextRuntime {
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
            return Err("Nemotron generate: empty prompt".into());
        }
        let required = request.token_ids.len() + request.max_new_tokens;
        if required > self.context_length {
            return Err(format!(
                "Nemotron generate: prompt {} + max_new_tokens {} exceeds the {} context",
                request.token_ids.len(),
                request.max_new_tokens,
                self.context_length
            ));
        }
        // The scratch pad carries the hidden state, the KV cache and the
        // Mamba-2 recurrence, so it is sized for the request rather than
        // for the model's whole context (the 30B's context is a million
        // tokens; the 4B's is 8K).
        let mut scratch = NemotronScratch::new(&self.model.config, required);
        // A prompt with more rows than the device arena holds cannot run
        // there: the row buffers would overflow. That request goes to the
        // CPU, with the device session kept for the next one.
        #[cfg(feature = "vulkan")]
        {
            let fits = self
                .device_rows
                .is_none_or(|rows| request.token_ids.len() <= rows);
            self.model.set_gpu_active(fits);
            if !fits && !self.warned_long_request {
                self.warned_long_request = true;
                eprintln!(
                    "[GPU] Nemotron prompt of {} tokens exceeds the device arena's {} rows; \
                     this request runs on the CPU",
                    request.token_ids.len(),
                    self.device_rows.unwrap_or(0)
                );
            }
        }
        let mut logits = self.model.prefill(&request.token_ids, &mut scratch)?;

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
            logits = self.model.prefill(&[id], &mut scratch)?;
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

/// The model is only borrowed through `&mut self`, so a single lock around
/// it is enough; the server keeps one runtime per model and serialises
/// requests.
pub type SharedNemotronRuntime = std::sync::Mutex<NemotronTextRuntime>;
