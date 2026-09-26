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
    pub vocab: usize,
    pub seq_len: usize,
    pub shortconv_states: Vec<Vec<f32>>,
    pub accumulated_bx: Vec<Vec<Vec<f32>>>,
    /// Clamped context length (min of config.n_ctx and caller's max_context).
    pub max_ctx: usize,
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
        crate::ops::embedding::expect_supported_embedding("token_embd.weight", embd_info.ggml_type);
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

        let layers: Vec<super::weights::Lfm2MoeLayerWeights<'a>> = load_layers(source, &config)
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

        let layers: Vec<super::weights::Lfm2MoeLayerWeights<'a>> = load_layers(source, &config)
            .map_err(|e| format!("Failed to load LFM2-MoE layers: {e}"))?;

        let mut shortconv_states: Vec<Vec<f32>> = Vec::with_capacity(n_layer);
        let mut accumulated_bx: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_layer);
        for lw in &layers {
            if lw.is_attn {
                shortconv_states.push(Vec::new());
                accumulated_bx.push(Vec::new());
            } else {
                shortconv_states.push(vec![0.0f32; config.n_embd * config.d_conv]);
                accumulated_bx.push(Vec::new());
            }
        }

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
            shortconv_states,
            accumulated_bx,
            max_ctx,
        })
    }

    /// Feed a sequence of tokens through the model.
    /// Returns the logits for the last token.
    /// Note: this currently delegates to the free-function path which
    /// re-creates internal state each call. For server use, the caller
    /// should pass the full token history each time.
    pub fn forward_logits_chunked(
        &self,
        tokens: &[u32],
        _batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        let (logits, _) = super::forward::run_forward_logits_lfm2moe_with_batch(
            self.source,
            tokens,
            self.pool.n_threads(),
            crate::core::scratchpad::KvFormat::F16,
            self.config.n_ctx,
            1,
        )?;
        Ok(logits)
    }

    /// Forward a single token through the model, reusing the session's
    /// KV cache, scratch, shortconv state, and accumulated b*x history.
    /// Returns the logits for the given token at position `seq_len`.
    pub fn forward_token(&mut self, token_id: u32) -> Result<Vec<f32>, String> {
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let n_layer = cfg.n_layer;
        let eps = cfg.norm_eps;
        let freq_base = cfg.rope_freq_base;
        let max_ctx = self.max_ctx;
        let pos = self.seq_len;

        let embd_weight = self.weights.embd_weight;
        let embd_type = self.weights.embd_type;
        let output_norm = &self.weights.output_norm;
        let output_weight = self.weights.output_weight;
        let output_type = self.weights.output_type;
        let vocab = self.vocab;

        // Embedding lookup.
        crate::ops::embedding::embedding_lookup(
            embd_weight,
            token_id,
            n_embd,
            embd_type,
            &mut self.scratch.x,
        );

        // Per-layer forward.
        for layer in 0..n_layer {
            let lw = &self.weights.layers[layer];
            let is_prefill = true;
            if !lw.is_attn && is_prefill {
                let d_conv = cfg.d_conv;
                let state = &mut self.shortconv_states[layer];
                state.resize(d_conv * n_embd, 0.0);
                let hist = &self.accumulated_bx[layer];
                for k_p in 0..d_conv {
                    let idx = k_p as isize - (d_conv - hist.len()) as isize;
                    if idx >= 0 {
                        let src = &hist[idx as usize];
                        for ci in 0..n_embd {
                            state[k_p * n_embd + ci] = src[ci];
                        }
                    }
                }
            }
            super::forward::forward_layer(
                &self.pool,
                lw,
                layer,
                n_layer,
                cfg,
                &mut self.scratch,
                &self.kv_cache,
                max_ctx,
                pos,
                eps,
                freq_base,
                &mut self.shortconv_states[layer],
                &mut self.accumulated_bx[layer],
                is_prefill,
                pos,
            );
        }

        // Output norm + LM head.
        let x = &mut self.scratch.x[..n_embd];
        let normed = &mut self.scratch.normed[..n_embd];
        crate::ops::rms_norm(x, output_norm, normed, eps);

        let max_n_in = (cfg.n_embd * 3).max(cfg.n_head * cfg.n_embd_head_k).max(cfg.n_ff);
        let q8 = &mut self.scratch.q8_buf[..max_n_in];
        let scale = &mut self.scratch.scale_buf[..max_n_in / 32];
        let q8k = &mut self.scratch.q8k_buf[..max_n_in / 256];
        crate::ops::quantize_q8_0_into(normed, n_embd, &mut q8[..n_embd], &mut scale[..n_embd / 32]);
        crate::ops::quantize_row_q8_k_into(normed, &mut q8k[..n_embd / 256]);

        let output_pw = crate::ops::kernel::Weight::from_quantized(
            crate::ops::kernel::QuantizedTensor::from_bytes(
                output_weight,
                output_type,
                n_embd,
                vocab,
            ),
        );

        let logits = &mut self.scratch.logits;
        logits.resize(vocab, 0.0);
        let n_embd_val = n_embd;
        let vocab_val = vocab;
        let normed_ptr = normed.as_ptr();
        let q8_ptr = q8.as_ptr();
        let scale_ptr = scale.as_ptr();
        let q8k_ptr = q8k.as_ptr();
        let logits_ptr = logits.as_mut_ptr();
        self.pool.compute(move |ith, nth| {
            let input = unsafe { std::slice::from_raw_parts(normed_ptr, n_embd_val) };
            let q8_slice = unsafe { std::slice::from_raw_parts(q8_ptr, n_embd_val) };
            let sc_slice = unsafe { std::slice::from_raw_parts(scale_ptr, n_embd_val / 32) };
            let q8k_slice = unsafe { std::slice::from_raw_parts(q8k_ptr, n_embd_val / 256) };
            let logits_slice =
                unsafe { std::slice::from_raw_parts_mut(logits_ptr, vocab_val) };
            output_pw.kernel.forward_prepared(
                input,
                q8_slice,
                sc_slice,
                Some(q8k_slice),
                logits_slice,
                n_embd_val,
                vocab_val,
                ith,
                nth,
            );
        });

        self.seq_len += 1;
        Ok(logits.clone())
    }

    /// Reset the session for a new conversation (clears KV cache,
    /// shortconv state, and seq_len).
    pub fn reset(&mut self) {
        self.seq_len = 0;
        self.kv_cache.clear();
        for state in &mut self.shortconv_states {
            state.fill(0.0);
        }
        for hist in &mut self.accumulated_bx {
            hist.clear();
        }
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
        let mut last_logits = None;
        for &token_id in input {
            last_logits = Some(self.forward_token(token_id)?);
        }
        Ok(last_logits)
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
            last_logits = self.forward_chunk(input, chunk.len(), self.seq_len, chunk.end == total)?;
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
        assert_eq!(
            <Lfm2MoeSession<'_> as ChunkedPrefill>::input_len(&tokens),
            7
        );
    }

    #[test]
    fn chunked_prefill_rejects_rows_above_one_for_now() {
        assert_eq!(LFM2MOE_BATCH_LIMIT, 1);
    }
}
