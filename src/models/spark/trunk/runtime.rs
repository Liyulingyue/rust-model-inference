//! HTTP `TextRuntime` adapter for Spark 2.5 (`arch = "spark2_5"`).
//!
//! Spark's forward is a single-token KV append
//! ([`SparkSession::forward_step_logits`]), so the adapter allocates a
//! fresh [`SparkSession`] per request and walks the prompt one token at a
//! time before decoding. Per-request allocation is the price of a
//! stateless API; the model weights stay shared behind the mutex.
//!
//! The decode loop mirrors the CLI's `run_inference`: during prefill each
//! prompt token is forwarded via [`SparkSession::decode_step`], which
//! advances the KV cache and produces the next token. The last prefill
//! step's output is the first generated token; subsequent steps are
//! pure decode.
//!
//! Repetition penalty is not applied — the server's sampling is
//! greedy/temperature only, matching the CLI's `run_inference` path.

use std::sync::{Arc, Mutex};

use crate::app::cli::KvFormat;
use crate::app::text::runtime::RuntimeOptions;
use crate::models::spark::trunk::config::SparkConfig;
use crate::models::spark::trunk::forward::SparkSession;
use crate::ops::generation_runtime::{
    Finish, Flow, GeneratedText, GenerationRequest, TextRuntime, TokenSink,
};

pub struct SparkTextRuntime {
    session: Mutex<SparkSession>,
    tokenizer: Arc<crate::core::tokenizer::BPETokenizer>,
    arch: String,
    context_length: usize,
    /// Stored for parity with other adapters; Spark's KV cache is F16.
    #[allow(dead_code)]
    kv_format: KvFormat,
}

impl SparkTextRuntime {
    pub fn new(options: RuntimeOptions) -> Result<Self, String> {
        let source = options.source.clone();
        let pool = options.pool.clone();

        // SparkSession requires a capacity hint; use the model's n_ctx
        // clamped to the runtime's max_context.
        let model_n_ctx = SparkConfig::from_source(&*source)
            .map(|cfg| cfg.n_ctx)
            .unwrap_or(options.max_context);
        let capacity = model_n_ctx.min(options.max_context).max(1);

        let session = SparkSession::new(&*source, pool, capacity)
            .map_err(|error| format!("Failed to create Spark session: {error}"))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: options.tokenizer,
            arch: "spark2_5".to_string(),
            context_length: capacity,
            kv_format: options.kv_format,
        })
    }
}

impl TextRuntime for SparkTextRuntime {
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
            return Err("Spark generate: empty prompt".into());
        }
        let required = request.token_ids.len() + request.max_new_tokens;
        if required > self.context_length {
            return Err(format!(
                "Spark generate: prompt {} + max_new_tokens {} exceeds the {} context",
                request.token_ids.len(),
                request.max_new_tokens,
                self.context_length
            ));
        }

        let mut session = self.session.lock().map_err(|e| e.to_string())?;
        let temperature = request.sampling.temperature;
        let eos_id = self.tokenizer.eos_id();
        let mut decoder = self.tokenizer.streaming_decoder(false);
        let mut text = String::new();
        let mut token_ids = Vec::with_capacity(request.max_new_tokens);
        let mut finish = Finish::Limit;

        // Prefill: forward each prompt token via decode_step. The last
        // call produces the first generated token.
        let mut next_token = 0u32;
        for (pos, &tok) in request.token_ids.iter().enumerate() {
            next_token = session
                .decode_step(tok, pos, temperature)
                .map_err(|e| format!("Spark prefill failed at position {pos}: {e}"))?;
        }

        // Decode loop: `next_token` is the token to output; `decode_step`
        // forwards it and produces the successor.
        let mut position = request.token_ids.len();
        for _ in 0..request.max_new_tokens {
            if Some(next_token) == eos_id {
                finish = Finish::Eos;
                break;
            }
            token_ids.push(next_token);
            let chunk = decoder.push(next_token);
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
            next_token = session
                .decode_step(next_token, position, temperature)
                .map_err(|e| format!("Spark decode failed at position {position}: {e}"))?;
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

/// The session is the largest buffer and the model stays shared behind the
/// mutex; the server keeps one runtime per model and serialises requests.
pub type SharedSparkRuntime = Mutex<SparkTextRuntime>;
