//! Shared model weights and stateful prefill/decode for one LFM3 sequence.

use super::config::Lfm3Config;
use super::forward::forward_layer;
use super::trace;
use super::weights::{load_layers, quant_weight, Lfm3LayerWeights};
use crate::app::cli::resolve_thread_count;
use crate::compute::linear::{LinearBinding, LinearExecutor, LinearMode};
use crate::compute::ComputePolicy;
use crate::core::prefill::checked_prefill_batch_size;
use crate::core::scratchpad::{ExecutionScratchpad, KvCache, KvFormat};
use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::Weight;
use crate::ops::rms_norm;
use std::sync::Arc;

pub struct Lfm3Model<'a> {
    pub config: Lfm3Config,
    layers: Vec<Lfm3LayerWeights<'a>>,
    embedding: Weight<'a>,
    output: Weight<'a>,
    output_norm: Vec<f32>,
    pool: Arc<ComputePool>,
}

pub struct Lfm3Session<'model, 'weights> {
    pub config: &'model Lfm3Config,
    pub vocab: usize,
    pub seq_len: usize,
    pub(crate) max_context: usize,
    model: &'model Lfm3Model<'weights>,
    kv: Vec<KvCache>,
    conv: Vec<Vec<f32>>,
    scratch: ExecutionScratchpad,
    linear: LinearExecutor<'model, 'weights>,
}

impl<'a> Lfm3Model<'a> {
    pub fn from_source(source: &'a dyn TensorSource, n_threads: usize) -> Result<Self, String> {
        let available = std::thread::available_parallelism().map_or(1, usize::from);
        let n_threads = resolve_thread_count(n_threads, available);
        Self::from_source_with_pool(source, Arc::new(ComputePool::new(n_threads)))
    }

    pub(crate) fn from_source_with_pool(
        source: &'a dyn TensorSource,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let config = Lfm3Config::from_source(source)?;
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
        Ok(Self {
            config,
            layers,
            embedding,
            output,
            output_norm,
            pool,
        })
    }
}

impl<'model, 'weights> Lfm3Session<'model, 'weights> {
    pub fn new(
        model: &'model Lfm3Model<'weights>,
        kv_format: KvFormat,
        max_context: usize,
    ) -> Result<Self, String> {
        Self::new_with_compute(model, kv_format, max_context, ComputePolicy::legacy())
    }

    pub fn new_with_compute(
        model: &'model Lfm3Model<'weights>,
        kv_format: KvFormat,
        max_context: usize,
        policy: ComputePolicy,
    ) -> Result<Self, String> {
        policy.check_build()?;
        policy.validate_text_arch("lfm3")?;
        let config = &model.config;
        let max_context = config.n_ctx.min(max_context);
        if max_context == 0 {
            return Err("LFM3 max context must be positive".into());
        }
        let vocab = config.vocab_size;
        let mut bindings = Vec::new();
        for layer in &model.layers {
            bindings.extend(
                [
                    Some(&layer.w_gate),
                    Some(&layer.w_up),
                    Some(&layer.w_down),
                    layer.wq.as_ref(),
                    layer.wk.as_ref(),
                    layer.wv.as_ref(),
                    layer.wo.as_ref(),
                    layer.attn_gate.as_ref(),
                    layer.shortconv_in.as_ref(),
                    layer.shortconv_out.as_ref(),
                ]
                .into_iter()
                .flatten()
                .map(|weight| LinearBinding {
                    weight,
                    mode: LinearMode::Prepared,
                }),
            );
        }
        bindings.push(LinearBinding {
            weight: &model.output,
            mode: LinearMode::Prepared,
        });
        let linear = LinearExecutor::new(policy, bindings, 1, model.pool.clone())
            .map_err(|e| e.to_string())?;
        if policy != ComputePolicy::Cpu {
            eprintln!(
                "LFM3 projections: requested={policy:?} backend={}; attention/shortconv/KV=Cpu",
                if linear.uses_vulkan() {
                    "Vulkan"
                } else {
                    "Cpu"
                }
            );
        }
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
        let n_threads = model.pool.n_threads();
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
            model,
            kv,
            conv,
            scratch,
            linear,
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
                if let Err(error) = self.forward_token(token, self.seq_len + 1 == end) {
                    self.reset();
                    return Err(error);
                }
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

    fn forward_token(&mut self, token: u32, project_logits: bool) -> Result<(), String> {
        let pos = self.seq_len;
        self.model
            .embedding
            .embedding_lookup(token, &mut self.scratch.x);
        trace::global("embedding", pos, &self.scratch.x);
        for (layer, weights) in self.model.layers.iter().enumerate() {
            forward_layer(
                &mut self.linear,
                weights,
                layer,
                &self.config,
                &mut self.scratch,
                &mut self.kv[layer],
                pos,
                &mut self.conv[layer],
            )
            .map_err(|e| e.to_string())?;
        }
        if project_logits {
            let s = &mut self.scratch;
            rms_norm(
                &s.x,
                &self.model.output_norm,
                &mut s.normed,
                self.config.norm_eps,
            );
            trace::global("result_norm", pos, &s.normed);
            self.linear
                .run(
                    self.linear
                        .id_for(&self.model.output)
                        .map_err(|e| e.to_string())?,
                    &s.normed,
                    1,
                    &mut s.logits,
                )
                .map_err(|e| e.to_string())?;
            trace::global("result_output", pos, &s.logits);
        }
        self.seq_len += 1;
        Ok(())
    }
}
