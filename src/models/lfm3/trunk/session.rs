//! Stateful CPU prefill and decode for one LFM3 sequence.

use super::config::Lfm3Config;
use super::forward::forward_layer;
use super::trace;
use super::weights::{load_layers, quant_weight, Lfm3LayerWeights};
use crate::app::cli::resolve_thread_count;
use crate::core::prefill::checked_prefill_batch_size;
use crate::core::scratchpad::{ExecutionScratchpad, KvCache, KvFormat};
use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::Weight;
use crate::ops::rms_norm;

pub struct Lfm3Session<'a> {
    pub config: Lfm3Config,
    pub vocab: usize,
    pub seq_len: usize,
    pub(crate) max_context: usize,
    layers: Vec<Lfm3LayerWeights<'a>>,
    embedding: Weight<'a>,
    output: Weight<'a>,
    output_norm: Vec<f32>,
    kv: Vec<KvCache>,
    conv: Vec<Vec<f32>>,
    scratch: ExecutionScratchpad,
    pool: ComputePool,
}

impl<'a> Lfm3Session<'a> {
    pub fn from_source(
        source: &'a dyn TensorSource,
        n_threads: usize,
        kv_format: KvFormat,
        max_context: usize,
    ) -> Result<Self, String> {
        let config = Lfm3Config::from_source(source)?;
        let max_context = config.n_ctx.min(max_context);
        if max_context == 0 {
            return Err("LFM3 max context must be positive".into());
        }
        let vocab = config.vocab_size;
        let embedding = quant_weight(source, "token_embd.weight", config.n_embd, vocab)?;
        if !crate::ops::embedding::is_supported_embedding(embedding.ggml_type) {
            return Err(format!(
                "unsupported LFM3 embedding type {:?}",
                embedding.ggml_type
            ));
        }
        let output_name = if source.tensor_info("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let output = quant_weight(source, output_name, config.n_embd, vocab)?;
        let output_norm = load_f32_tensor(source, "output_norm.weight", &[config.n_embd as u64])?;
        let layers = load_layers(source, &config)?;
        let mut kv = Vec::with_capacity(config.n_layer);
        let mut conv = Vec::with_capacity(config.n_layer);
        for &heads in &config.n_head_kv_per_layer {
            let width = heads * config.n_embd_head_k;
            kv.push(match kv_format {
                KvFormat::F32 => KvCache::new_f32(1, max_context, width),
                KvFormat::F16 => KvCache::new_f16(1, max_context, width),
            });
            conv.push(vec![
                0.0;
                if heads == 0 {
                    config.n_embd * config.d_conv
                } else {
                    0
                }
            ]);
        }
        let available = std::thread::available_parallelism().map_or(1, usize::from);
        let n_threads = resolve_thread_count(n_threads, available);
        let q_width = config.n_head * config.n_embd_head_k;
        let scratch = ExecutionScratchpad::new(
            config.n_embd,
            q_width,
            q_width,
            config.n_ff,
            vocab,
            n_threads,
            max_context,
        );
        Ok(Self {
            config,
            vocab,
            seq_len: 0,
            max_context,
            layers,
            embedding,
            output,
            output_norm,
            kv,
            conv,
            scratch,
            pool: ComputePool::new(n_threads),
        })
    }

    /// Append input to this sequence. Chunks retain the same causal, per-token state.
    pub fn forward_logits_chunked(
        &mut self,
        input: &[u32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        let batch_size = checked_prefill_batch_size(Some(batch_size))?;
        if input.is_empty() {
            return Err("LFM3 input is empty".into());
        }
        if input.len() > self.max_context.saturating_sub(self.seq_len) {
            return Err(format!(
                "LFM3 input exceeds context capacity {}",
                self.max_context
            ));
        }
        if input.iter().any(|&id| id as usize >= self.vocab) {
            return Err("LFM3 token id exceeds vocabulary".into());
        }
        let end = self.seq_len + input.len();
        // shortcut: chunks step per token; batch projections when optimizing prefill throughput.
        for chunk in input.chunks(batch_size) {
            for &token in chunk {
                self.forward_token(token, self.seq_len + 1 == end);
            }
        }
        Ok(self.scratch.logits.clone())
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
        for cache in &mut self.kv {
            cache.clear();
        }
        for state in &mut self.conv {
            state.fill(0.0);
        }
    }

    fn forward_token(&mut self, token: u32, project_logits: bool) {
        let pos = self.seq_len;
        self.embedding.embedding_lookup(token, &mut self.scratch.x);
        trace::global("embedding", pos, &self.scratch.x);
        for (layer, weights) in self.layers.iter().enumerate() {
            forward_layer(
                &self.pool,
                weights,
                layer,
                &self.config,
                &mut self.scratch,
                &mut self.kv[layer],
                pos,
                &mut self.conv[layer],
            );
        }
        if project_logits {
            let s = &mut self.scratch;
            rms_norm(&s.x, &self.output_norm, &mut s.normed, self.config.norm_eps);
            trace::global("result_norm", pos, &s.normed);
            self.output.quantize_and_matmul_with_scratch(
                &s.normed,
                &mut s.q8k_buf,
                &mut s.q8_buf,
                &mut s.scale_buf,
                &mut s.logits,
                &self.pool,
            );
            trace::global("result_output", pos, &s.logits);
        }
        self.seq_len += 1;
    }
}
