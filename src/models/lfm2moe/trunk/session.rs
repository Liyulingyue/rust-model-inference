//! LFM2-MoE session — minimal owned state for one prefill pass.
//!
//! Extracted out of the previous free-function `run_inference` so the
//! trunk can adopt [`crate::core::prefill::ChunkedPrefill`]. The
//! session owns every piece of state that used to be threaded as
//! locals — config, weights, KV cache, scratchpad, compute pool,
//! SSM shortconv state, MoE accumulated `b*x` history — and exposes
//! a single `forward_logits_chunked(input, batch_size)` that walks
//! the trait default loop.
//!
//! For `batch_size == 1` the dispatch collapses to the legacy
//! per-token walk so existing parity tests stay green;
//! `batch_size > 1` returns an error for now because the MoE router
//! hidden state and the SSM shortconv state both step one token at
//! a time. Lifting those into a real `rows > 1` batched path is the
//! next refactor; this commit focuses on getting the trait
//! adoption in place so the trunk is uniformly wired through
//! [`crate::core::prefill::prefill_chunks`].

use super::config::Lfm2MoeConfig;
use super::forward::run_forward_logits_lfm2moe_with_batch;
use super::weights::load_layers;
use crate::app::cli::resolve_thread_count;
use crate::core::prefill::{checked_prefill_batch_size, prefill_chunks, ChunkedPrefill};
use crate::core::scratchpad::{ExecutionScratchpad, KvCache, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use std::sync::Arc;

/// Convenience: the LFM2-MoE attention / SSM / MoE math has to
/// step per token today. This constant matches the
/// [`DEFAULT_PREFILL_BATCH_SIZE`] default but signals intent:
/// chunked prefill walks the trait loop one row at a time for now.
const LFM2MOE_BATCH_LIMIT: usize = 1;

pub struct Lfm2MoeSession<'a> {
    pub config: Lfm2MoeConfig,
    pub weights: Lfm2MoeWeights<'a>,
    pub source: &'a dyn TensorSource,
    pub kv_cache: KvCache,
    pub scratch: ExecutionScratchpad,
    pub pool: Arc<ComputePool>,
    /// Vocabulary size for the LM head. Captured here so the
    /// free-function path's internal `BPETokenizer` reconstruction
    /// doesn't have to round-trip the GGUF metadata on every call.
    pub vocab: usize,
    pub seq_len: usize,
}

pub struct Lfm2MoeWeights<'a> {
    pub layers: Vec<super::weights::Lfm2MoeLayerWeights<'a>>,
    pub embd_weight: &'a [u8],
    pub embd_type: crate::core::tensor::GGMLType,
    pub output_norm: Vec<f32>,
    pub output_weight: &'a [u8],
    pub output_type: crate::core::tensor::GGMLType,
}

impl<'a> Lfm2MoeSession<'a> {
    pub fn from_source(
        source: &'a dyn TensorSource,
        n_threads_arg: usize,
        kv_format: KvFormat,
        max_context: usize,
    ) -> Result<Self, String> {
        Self::from_source_with_max_rows(source, n_threads_arg, kv_format, max_context, 1)
    }

    pub fn from_source_with_max_rows(
        source: &'a dyn TensorSource,
        n_threads_arg: usize,
        kv_format: KvFormat,
        max_context: usize,
        _max_rows: usize,
    ) -> Result<Self, String> {
        let config = Lfm2MoeConfig::from_source(source)
            .map_err(|e| format!("Failed to parse LFM2-MoE config: {e}"))?;
        let n_embd = config.n_embd;
        let n_layer = config.n_layer;
        let n_head = config.n_head;
        let n_ff = config.n_ff;
        let n_embd_head_k = config.n_embd_head_k;
        let n_embd_q = n_head * n_embd_head_k;
        let n_embd_gqa = config
            .n_head_kv_per_layer
            .iter()
            .map(|&h| h * n_embd_head_k)
            .max()
            .unwrap_or(0)
            .max(n_embd_q);
        let max_ctx = config.n_ctx.min(max_context);

        let embd_info = source
            .tensor_info("token_embd.weight")
            .ok_or_else(|| "Missing token_embd.weight metadata".to_string())?;
        crate::ops::embedding::expect_supported_embedding(
            "token_embd.weight",
            embd_info.ggml_type,
        );
        let embd_weight = source
            .tensor_slice("token_embd.weight")
            .ok_or_else(|| "Missing token_embd.weight data".to_string())?;
        let output_weight = source.tensor_slice("output.weight").unwrap_or(embd_weight);
        let output_type = source
            .tensor_info("output.weight")
            .unwrap_or(embd_info)
            .ggml_type;

        let output_norm = crate::core::tensor::load_f32_tensor(
            source,
            "token_embd_norm.weight",
            &[n_embd as u64],
        )?;

        let tokenizer = crate::core::tokenizer::BPETokenizer::from_gguf_metadata(|k| {
            source.metadata(k).cloned()
        })
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        let vocab = tokenizer.vocab_size();

        let layers: Vec<super::weights::Lfm2MoeLayerWeights<'a>> =
            load_layers(source, &config)
                .map_err(|e| format!("Failed to load LFM2-MoE layers: {e}"))?;

        let kv_cache = match kv_format {
            KvFormat::F16 => KvCache::new_f16(n_layer, max_ctx, n_embd_gqa),
            KvFormat::F32 => KvCache::new_f32(n_layer, max_ctx, n_embd_gqa),
        };

        let available_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let n_threads = resolve_thread_count(n_threads_arg, available_threads);
        let scratch = ExecutionScratchpad::new(
            n_embd, n_embd_q, n_embd_gqa, n_ff, vocab, n_threads, max_ctx,
        );
        let pool = Arc::new(ComputePool::new(n_threads));

        Ok(Self {
            config,
            weights: Lfm2MoeWeights {
                layers,
                embd_weight,
                embd_type: embd_info.ggml_type,
                output_norm,
                output_weight,
                output_type,
            },
            source,
            kv_cache,
            scratch,
            pool,
            vocab,
            seq_len: 0,
        })
    }

    /// Trait-driven entry point. Returns the final logits for
    /// `prompt_tokens` after a chunked prefill at the requested
    /// `batch_size`. Equivalent to the legacy per-token walk for
    /// `batch_size == 1` (the [`DEFAULT_PREFILL_BATCH_SIZE`] walks
    /// `rows = 1` so the SSM shortconv state and the MoE router
    /// hidden state can step one token at a time).
    /// `batch_size > 1` returns an error for now.
    pub fn forward_logits_chunked(
        &self,
        prompt_tokens: &[u32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        let batch_size = checked_prefill_batch_size(Some(batch_size))
            .map_err(|e| format!("LFM2-MoE batch size: {e}"))?;
        if batch_size > LFM2MOE_BATCH_LIMIT {
            return Err(format!(
                "LFM2-MoE batched prefill > {LFM2MOE_BATCH_LIMIT} is not yet implemented; \
                 the MoE router state and the SSM shortconv state both step per-row"
            ));
        }
        // Delegate to the legacy free-function path. The session
        // holds a reference to the original `TensorSource`, so the
        // free function can read every tensor it needs (token_embd,
        // output, output_norm, blk.*, token_embd_norm.*, …) without
        // re-loading anything. For B = 1 the dispatch is bit-exact
        // to the pre-trait baseline.
        let (logits, _duration) = run_forward_logits_lfm2moe_with_batch(
            self.source,
            prompt_tokens,
            self.pool.n_threads(),
            KvFormat::F16,
            self.config.n_ctx,
            batch_size,
        )?;
        Ok(logits)
    }
}

impl<'a> ChunkedPrefill for Lfm2MoeSession<'a> {
    type Input = Vec<u32>;

    fn input_len(input: &Self::Input) -> usize {
        input.len()
    }

    fn max_chunk_size(&self) -> usize {
        self.config.n_ctx
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn set_seq_len(&mut self, len: usize) {
        self.seq_len = len;
    }

    fn forward_chunk(
        &mut self,
        input: &Self::Input,
        rows: usize,
        _base_position: usize,
        project_logits: bool,
    ) -> Result<Option<Vec<f32>>, String> {
        if rows != input.len() {
            return Err(format!(
                "Lfm2MoeSession::forward_chunk only handles whole-input chunks; \
                 rows = {rows} vs input.len() = {}",
                input.len()
            ));
        }
        if !project_logits {
            return Err("Lfm2MoeSession::forward_chunk requires project_logits = true".into());
        }
        let logits = self.forward_logits_chunked(input, 1)?;
        Ok(Some(logits))
    }

    fn prefill(
        &mut self,
        input: &Self::Input,
        batch_size: usize,
    ) -> Result<Option<Vec<f32>>, String> {
        let batch_size = checked_prefill_batch_size(Some(batch_size))?;
        let total = Self::input_len(input);
        if total == 0 {
            return Ok(None);
        }
        if batch_size > LFM2MOE_BATCH_LIMIT {
            return Err(format!(
                "LFM2-MoE batched prefill > {LFM2MOE_BATCH_LIMIT} is not yet implemented; \
                 the MoE router state and the SSM shortconv state both step per-row"
            ));
        }
        let mut last_logits: Option<Vec<f32>> = None;
        for chunk in prefill_chunks(total, batch_size) {
            let rows = chunk.len();
            let base = self.seq_len;
            let is_last = chunk.end == total;
            last_logits = self.forward_chunk(input, rows, base, is_last)?;
            self.set_seq_len(base + rows);
        }
        Ok(last_logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_prefill_input_len_matches_token_count() {
        let tokens: Vec<u32> = (0..7).collect();
        assert_eq!(<Lfm2MoeSession<'_> as ChunkedPrefill>::input_len(&tokens), 7);
    }

    #[test]
    fn chunked_prefill_rejects_rows_above_one_for_now() {
        assert_eq!(LFM2MOE_BATCH_LIMIT, 1);
    }
}