//! dotstts LLM half: a 28×1536 Qwen2 decoder (arch `qwen2`) driven step by
//! step so the flow-matching pipeline can interleave LLM forwards with FM
//! decodes. Mirrors `models::qwen3::tts::talker::TtsSession` but for the
//! plain Qwen2 layout (no per-head Q/K RMSNorm, single scalar Neox rope) and
//! exposes the raw hidden state of every step for `hidden_proj`/`eos_proj`.

use std::sync::Arc;

use crate::core::scratchpad::KvCache;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::Weight;
use crate::ops::{
    dot_f16, f16_to_f32, f32_slice_to_f16, quantize_q8_0_into, rms_norm, rope_neox, silu,
    vec_mad_f32, vec_scale_f32,
};

#[derive(Debug, Clone)]
pub struct DotsLlmConfig {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head: usize,
    pub n_ff: usize,
    pub vocab_size: usize,
    pub n_ctx: usize,
    pub eps: f32,
    pub freq_base: f32,
}

impl DotsLlmConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let cfg = crate::models::qwen3::Qwen3Config::from_source(source)?;
        let vocab_size = source
            .metadata("tokenizer.ggml.tokens")
            .and_then(|v| v.to_arr())
            .map(Vec::len)
            .unwrap_or(0);
        Ok(Self {
            n_embd: cfg.n_embd,
            n_layer: cfg.n_layer,
            n_head: cfg.n_head,
            n_head_kv: cfg.n_head_kv,
            n_embd_head: cfg.n_embd_head_k,
            n_ff: cfg.n_ff,
            vocab_size,
            n_ctx: cfg.n_ctx,
            eps: cfg.eps,
            freq_base: cfg.freq_base,
        })
    }
}

pub(crate) struct DotsLayerWeights {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) wq: Weight<'static>,
    pub(crate) wk: Weight<'static>,
    pub(crate) wv: Weight<'static>,
    pub(crate) wo: Weight<'static>,
    pub(crate) w_gate: Weight<'static>,
    pub(crate) w_up: Weight<'static>,
    pub(crate) w_down: Weight<'static>,
}

/// Loaded Qwen2 LLM for dots.tts.
pub struct DotsLlm {
    /// Keep the source alive: all weights are 'static views into its mmap.
    pub source: Arc<dyn TensorSource>,
    pub pool: Arc<ComputePool>,
    pub config: DotsLlmConfig,
    pub output_norm: Vec<f32>,
    pub(crate) layers: Vec<DotsLayerWeights>,
    pub token_embedding: Weight<'static>,
}

impl DotsLlm {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let config = DotsLlmConfig::from_source(source.as_ref())?;
        let n_embd_q = config.n_head * config.n_embd_head;
        let n_embd_k = config.n_head_kv * config.n_embd_head;
        let output_norm = crate::core::tensor::load_f32_tensor(
            source.as_ref(),
            "output_norm.weight",
            &[config.n_embd as u64],
        )?;
        let token_embedding = crate::models::qwen3::static_weight(
            source.as_ref(),
            "token_embd.weight",
            config.n_embd,
            config.vocab_size,
        );
        let mut layers = Vec::with_capacity(config.n_layer);
        for layer in 0..config.n_layer {
            let name = |suffix: &str| format!("blk.{layer}.{suffix}");
            let n_embd = [config.n_embd as u64];
            layers.push(DotsLayerWeights {
                attn_norm: crate::core::tensor::load_f32_tensor(
                    source.as_ref(),
                    &name("attn_norm.weight"),
                    &n_embd,
                )?,
                ffn_norm: crate::core::tensor::load_f32_tensor(
                    source.as_ref(),
                    &name("ffn_norm.weight"),
                    &n_embd,
                )?,
                wq: crate::models::qwen3::static_weight(
                    source.as_ref(),
                    &name("attn_q.weight"),
                    config.n_embd,
                    n_embd_q,
                ),
                wk: crate::models::qwen3::static_weight(
                    source.as_ref(),
                    &name("attn_k.weight"),
                    config.n_embd,
                    n_embd_k,
                ),
                wv: crate::models::qwen3::static_weight(
                    source.as_ref(),
                    &name("attn_v.weight"),
                    config.n_embd,
                    n_embd_k,
                ),
                wo: crate::models::qwen3::static_weight(
                    source.as_ref(),
                    &name("attn_output.weight"),
                    n_embd_q,
                    config.n_embd,
                ),
                w_gate: crate::models::qwen3::static_weight(
                    source.as_ref(),
                    &name("ffn_gate.weight"),
                    config.n_embd,
                    config.n_ff,
                ),
                w_up: crate::models::qwen3::static_weight(
                    source.as_ref(),
                    &name("ffn_up.weight"),
                    config.n_embd,
                    config.n_ff,
                ),
                w_down: crate::models::qwen3::static_weight(
                    source.as_ref(),
                    &name("ffn_down.weight"),
                    config.n_ff,
                    config.n_embd,
                ),
            });
        }
        Ok(Self {
            source,
            pool,
            config,
            output_norm,
            layers,
            token_embedding,
        })
    }

    pub fn new_session(&self) -> Result<DotsLlmSession<'_>, String> {
        DotsLlmSession::new(self)
    }
}

/// Input row for prefill / decode: a token id (embedded by the table) or an
/// already-computed projection (patch-encoder embeddings or codec feedback).
pub enum LlmInputRow<'a> {
    Token(u32),
    Embedding(&'a [f32]),
}

/// One LLM step with hidden-state capture. Safe single-threaded matmuls
/// (correctness-first; parallelizing is a later optimization).
pub struct DotsLlmSession<'model> {
    model: &'model DotsLlm,
    kv: KvCache,
    /// Reusable scratch buffers (allocated once per session).
    x: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    acc: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    q8: Vec<u8>,
    scales: Vec<f32>,
    q16: Vec<u16>,
    step: usize,
    capacity: usize,
}

fn matmul(
    weight: &Weight<'_>,
    input: &[f32],
    q8: &mut [u8],
    scales: &mut [f32],
    output: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    quantize_q8_0_into(input, in_dim, &mut q8[..in_dim], &mut scales[..in_dim / 32]);
    weight.kernel.forward_prepared(
        &[],
        &q8[..in_dim],
        &scales[..in_dim / 32],
        None,
        output,
        in_dim,
        out_dim,
        0,
        1,
    );
}

impl<'model> DotsLlmSession<'model> {
    pub fn new(model: &'model DotsLlm) -> Result<Self, String> {
        let cfg = &model.config;
        let n_embd_q = cfg.n_head * cfg.n_embd_head;
        let n_embd_kv = cfg.n_head_kv * cfg.n_embd_head;
        let max_buf = cfg.n_ff.max(n_embd_q.max(cfg.n_embd));
        // the reference runtime caps the static LLM cache at
        // DEFAULT_MAX_SEQUENCE_LENGTH = 2048; keep the same bound so a
        // 131k-context gguf does not force multi-GB caches
        let capacity = cfg.n_ctx.min(2048);
        Ok(Self {
            model,
            kv: KvCache::new_f16(cfg.n_layer, capacity, n_embd_kv),
            x: vec![0.0; cfg.n_embd],
            normed: vec![0.0; cfg.n_embd],
            q: vec![0.0; n_embd_q],
            k: vec![0.0; n_embd_kv],
            v: vec![0.0; n_embd_kv],
            attn_out: vec![0.0; n_embd_q],
            acc: vec![0.0; cfg.n_embd_head],
            gate: vec![0.0; cfg.n_ff],
            up: vec![0.0; cfg.n_ff],
            down: vec![0.0; cfg.n_embd],
            q8: vec![0; max_buf],
            scales: vec![0.0; max_buf / 32],
            q16: vec![0; cfg.n_embd_head],
            step: 0,
            capacity,
        })
    }

    /// Length of the currently cached prefix.
    pub fn position(&self) -> usize {
        self.step
    }

    /// Embed + run one forward; returns the raw (pre output-norm) hidden row.
    pub fn step_row(&mut self, row: LlmInputRow<'_>) -> Result<Vec<f32>, String> {
        match row {
            LlmInputRow::Token(id) => {
                self.model.token_embedding.embedding_lookup(id, &mut self.x);
            }
            LlmInputRow::Embedding(embedding) => {
                if embedding.len() != self.model.config.n_embd {
                    return Err(format!(
                        "dotstts LLM embedding length {} != {}",
                        embedding.len(),
                        self.model.config.n_embd
                    ));
                }
                self.x.copy_from_slice(embedding);
            }
        }
        self.forward()?;
        self.step += 1;
        Ok(self.x.clone())
    }

    /// Run the Qwen2 decoder for the current `self.x` at position `self.step`.
    fn forward(&mut self) -> Result<(), String> {
        let cfg = &self.model.config;
        let n_embd_q = cfg.n_head * cfg.n_embd_head;
        let n_embd_kv = cfg.n_head_kv * cfg.n_embd_head;
        let group_size = cfg.n_head / cfg.n_head_kv;
        let kq_scale = 1.0 / (cfg.n_embd_head as f32).sqrt();
        if self.step >= self.capacity {
            return Err(format!(
                "dotstts LLM session exceeds context {}",
                self.capacity
            ));
        }
        let kv_stride = n_embd_kv;
        let (k_cache, v_cache) = match &mut self.kv {
            KvCache::F16(cache) => (&mut cache.k, &mut cache.v),
            KvCache::F32(_) => return Err("dotstts LLM requires an F16 KV cache".into()),
        };

        for layer in 0..cfg.n_layer {
            let weights = &self.model.layers[layer];
            // 1. attention norm + QKV
            rms_norm(&self.x, &weights.attn_norm, &mut self.normed, cfg.eps);
            matmul(
                &weights.wq,
                &self.normed,
                &mut self.q8,
                &mut self.scales,
                &mut self.q,
                cfg.n_embd,
                n_embd_q,
            );
            matmul(
                &weights.wk,
                &self.normed,
                &mut self.q8,
                &mut self.scales,
                &mut self.k,
                cfg.n_embd,
                n_embd_kv,
            );
            matmul(
                &weights.wv,
                &self.normed,
                &mut self.q8,
                &mut self.scales,
                &mut self.v,
                cfg.n_embd,
                n_embd_kv,
            );
            // 2. rope + KV store (F16)
            for head in self.q.chunks_exact_mut(cfg.n_embd_head) {
                rope_neox(head, self.step, cfg.n_embd_head, cfg.freq_base);
            }
            for head in self.k.chunks_exact_mut(cfg.n_embd_head) {
                rope_neox(head, self.step, cfg.n_embd_head, cfg.freq_base);
            }
            let layer_base = layer * self.capacity * kv_stride;
            let row_base = layer_base + self.step * kv_stride;
            for kv_head in 0..cfg.n_head_kv {
                let offset = kv_head * cfg.n_embd_head;
                let dst = row_base + offset;
                f32_slice_to_f16(
                    &self.k[offset..offset + cfg.n_embd_head],
                    &mut k_cache[dst..dst + cfg.n_embd_head],
                );
                f32_slice_to_f16(
                    &self.v[offset..offset + cfg.n_embd_head],
                    &mut v_cache[dst..dst + cfg.n_embd_head],
                );
            }
            // 3. attention: online softmax over the F16 cache (safe slices)
            self.attn_out.fill(0.0);
            for head in 0..cfg.n_head {
                let kv_head = head / group_size;
                let q_offset = head * cfg.n_embd_head;
                let out_offset = head * cfg.n_embd_head;
                let (query, out) = (
                    &self.q[q_offset..q_offset + cfg.n_embd_head],
                    &mut self.attn_out[out_offset..out_offset + cfg.n_embd_head],
                );
                self.acc.fill(0.0);
                f32_slice_to_f16(query, &mut self.q16);
                let mut sum = 0.0f32;
                let mut max = f32::NEG_INFINITY;
                for token in 0..=self.step {
                    let row = layer_base + token * kv_stride + kv_head * cfg.n_embd_head;
                    let score = dot_f16(
                        &self.q16,
                        &k_cache[row..row + cfg.n_embd_head],
                        cfg.n_embd_head,
                    ) * kq_scale;
                    let weight = if score > max {
                        let rescale = (max - score).exp();
                        max = score;
                        vec_scale_f32(&mut self.acc, rescale);
                        sum = sum.mul_add(rescale, 1.0);
                        1.0
                    } else {
                        let weight = (score - max).exp();
                        sum += weight;
                        weight
                    };
                    vec_mad_f32(
                        &mut self.acc,
                        &f16_to_f32_buf(&v_cache[row..row + cfg.n_embd_head]),
                        weight,
                    );
                }
                out.copy_from_slice(&self.acc);
                vec_scale_f32(out, if sum == 0.0 { 0.0 } else { sum.recip() });
            }
            // 4. output projection + residual
            matmul(
                &weights.wo,
                &self.attn_out,
                &mut self.q8,
                &mut self.scales,
                &mut self.down,
                n_embd_q,
                cfg.n_embd,
            );
            vec_mad_f32(&mut self.x, &self.down, 1.0);
            // 5. FFN: gate·up with SiLU, then down
            rms_norm(&self.x, &weights.ffn_norm, &mut self.normed, cfg.eps);
            matmul(
                &weights.w_gate,
                &self.normed,
                &mut self.q8,
                &mut self.scales,
                &mut self.gate,
                cfg.n_embd,
                cfg.n_ff,
            );
            matmul(
                &weights.w_up,
                &self.normed,
                &mut self.q8,
                &mut self.scales,
                &mut self.up,
                cfg.n_embd,
                cfg.n_ff,
            );
            for i in 0..cfg.n_ff {
                self.gate[i] = silu(self.gate[i]) * self.up[i];
            }
            matmul(
                &weights.w_down,
                &self.gate,
                &mut self.q8,
                &mut self.scales,
                &mut self.down,
                cfg.n_ff,
                cfg.n_embd,
            );
            vec_mad_f32(&mut self.x, &self.down, 1.0);
        }
        Ok(())
    }

    /// Raw hidden state (pre output-norm) of the most recent step.
    pub fn last_hidden(&self) -> &[f32] {
        &self.x
    }

    /// Normalized hidden state (after `output_norm.weight`).
    pub fn normalized_hidden(&self) -> Result<Vec<f32>, String> {
        if self.model.output_norm.len() != self.model.config.n_embd {
            return Err("dotstts output norm shape mismatch".into());
        }
        let mut out = vec![0.0; self.model.config.n_embd];
        rms_norm(
            &self.x,
            &self.model.output_norm,
            &mut out,
            self.model.config.eps,
        );
        Ok(out)
    }
}

fn f16_to_f32_buf(half: &[u16]) -> Vec<f32> {
    half.iter().map(|&bits| f16_to_f32(bits)).collect()
}