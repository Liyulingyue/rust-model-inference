//! VibeVoice ASR LLM half: the Qwen2.5-7B decoder (arch `qwen2`) driven one
//! row at a time so the streaming loop can interleave audio-feature prefills
//! with greedy text decoding over a shared KV cache.
//!
//! Weights stay as mmap-backed `Weight` kernels (Q8_0 in the exported gguf);
//! activations are quantized per step for the q8×q8 dot-product kernels, and
//! attention runs the F16 online-softmax path parallelized over heads.

use std::sync::Arc;

use crate::core::scratchpad::KvCache;
use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::qwen3::Qwen3Config;
use crate::ops::kernel::Weight;
use crate::ops::{dot_f32, quantize_q8_0_into, rms_norm};

pub(crate) struct AsrLayerWeights {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_bias: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub wq: Weight<'static>,
    pub wk: Weight<'static>,
    pub wv: Weight<'static>,
    pub wo: Weight<'static>,
    pub w_gate: Weight<'static>,
    pub w_up: Weight<'static>,
    pub w_down: Weight<'static>,
}

/// Loaded Qwen2 decoder for VibeVoice ASR.
pub struct VibeVoiceAsrLlm {
    /// Keeps the mmap alive: all weights borrow from it.
    pub source: Arc<dyn TensorSource>,
    pub pool: Arc<ComputePool>,
    pub config: Qwen3Config,
    /// Vocabulary size from the tokenizer metadata (includes padding).
    pub vocab_size: usize,
    pub output_norm: Vec<f32>,
    pub token_embedding: Weight<'static>,
    /// Untied output head (`lm_head.weight` in the checkpoint).
    pub lm_head: Weight<'static>,
    pub(crate) layers: Vec<AsrLayerWeights>,
}

impl VibeVoiceAsrLlm {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let config = Qwen3Config::from_source(source.as_ref())?;
        if config.architecture != "qwen2" {
            return Err(format!(
                "VibeVoice ASR requires an arch-qwen2 LLM gguf, found {:?}",
                config.architecture
            ));
        }
        let vocab_size = source
            .metadata("tokenizer.ggml.tokens")
            .and_then(|value| value.to_arr())
            .map(Vec::len)
            .unwrap_or(0);
        if vocab_size == 0 {
            return Err("LLM gguf is missing tokenizer.ggml.tokens metadata".into());
        }
        let n_embd_q = config.n_head * config.n_embd_head_k;
        let n_embd_k = config.n_head_kv * config.n_embd_head_k;
        let n_embd_v = config.n_head_kv * config.n_embd_head_v;
        let n_attn = config.n_head * config.n_embd_head_v;
        let dims = [config.n_embd as u64];

        let output_norm = load_f32_tensor(source.as_ref(), "output_norm.weight", &dims)?;
        // static_weight's first size argument becomes the kernel's n_in —
        // the embedding lookup needs the row width (n_embd) there
        let token_embedding = crate::models::qwen3::static_weight(
            source.as_ref(),
            "token_embd.weight",
            config.n_embd,
            config.vocab,
        );
        let lm_head = crate::models::qwen3::static_weight(
            source.as_ref(),
            "output.weight",
            config.n_embd,
            config.vocab,
        );

        let mut layers = Vec::with_capacity(config.n_layer);
        for layer in 0..config.n_layer {
            let name = |suffix: &str| format!("blk.{layer}.{suffix}");
            let load_bias = |key: &str, len: usize| -> Result<Vec<f32>, String> {
                load_f32_tensor(source.as_ref(), &name(key), &[len as u64])
            };
            layers.push(AsrLayerWeights {
                attn_norm: load_f32_tensor(source.as_ref(), &name("attn_norm.weight"), &dims)?,
                ffn_norm: load_f32_tensor(source.as_ref(), &name("ffn_norm.weight"), &dims)?,
                q_bias: load_bias("attn_q.bias", n_embd_q)?,
                k_bias: load_bias("attn_k.bias", n_embd_k)?,
                v_bias: load_bias("attn_v.bias", n_embd_v)?,
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
                    n_embd_v,
                ),
                wo: crate::models::qwen3::static_weight(
                    source.as_ref(),
                    &name("attn_output.weight"),
                    n_attn,
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
            vocab_size,
            output_norm,
            token_embedding,
            lm_head,
            layers,
        })
    }
}

/// Input row for the session: a token id (embedded) or a precomputed
/// projection row (audio features).
pub enum AsrInputRow<'a> {
    Token(u32),
    Embedding(&'a [f32]),
}

/// One-row-at-a-time decode session with a shared F16 KV cache.
pub struct AsrLlmSession<'model> {
    model: &'model VibeVoiceAsrLlm,
    kv: KvCache,
    x: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    attn_proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    q8_buf: Vec<u8>,
    scale_buf: Vec<f32>,
    logits: Vec<f32>,
    #[cfg(feature = "parity-trace")]
    layer_hidden_trace: Vec<f32>,
    next_step: usize,
    capacity: usize,
}

impl<'model> AsrLlmSession<'model> {
    pub fn new(model: &'model VibeVoiceAsrLlm, capacity: usize) -> Result<Self, String> {
        let config = &model.config;
        if capacity == 0 || capacity > config.n_ctx {
            return Err(format!(
                "ASR session capacity {capacity} outside 1..={}",
                config.n_ctx
            ));
        }
        let n_embd_q = config.n_head * config.n_embd_head_k;
        let n_embd_k = config.n_head_kv * config.n_embd_head_k;
        let n_embd_v = config.n_head_kv * config.n_embd_head_v;
        let n_attn = config.n_head * config.n_embd_head_v;
        let max_n_in = n_embd_q.max(n_attn).max(config.n_ff);
        Ok(Self {
            model,
            kv: KvCache::new_f32(config.n_layer, capacity, n_embd_k.max(n_embd_v)),
            x: vec![0.0; config.n_embd],
            normed: vec![0.0; config.n_embd],
            q: vec![0.0; n_embd_q],
            k: vec![0.0; n_embd_k],
            v: vec![0.0; n_embd_v],
            attn_out: vec![0.0; n_attn],
            attn_proj: vec![0.0; config.n_embd],
            gate: vec![0.0; config.n_ff],
            up: vec![0.0; config.n_ff],
            down: vec![0.0; config.n_embd],
            q8_buf: vec![0; max_n_in],
            scale_buf: vec![0.0; max_n_in / 32],
            logits: Vec::new(),
            #[cfg(feature = "parity-trace")]
            layer_hidden_trace: Vec::with_capacity(config.n_layer * config.n_embd),
            next_step: 0,
            capacity,
        })
    }

    /// Number of KV rows already written.
    pub fn position(&self) -> usize {
        self.next_step
    }

    /// Embed or copy one input row and run the full decoder pass, leaving the
    /// post-`output_norm` hidden state in `self.normed` for `logits()`.
    pub fn forward_step(&mut self, row: AsrInputRow<'_>) -> Result<(), String> {
        let config = &self.model.config;
        let step = self.next_step;
        if step >= self.capacity {
            return Err(format!(
                "ASR session exceeded context capacity {}; the audio is too long for one session",
                self.capacity
            ));
        }
        match row {
            AsrInputRow::Token(id) => {
                if id as usize >= self.model.vocab_size {
                    return Err(format!(
                        "token id {id} outside vocabulary {}",
                        self.model.vocab_size
                    ));
                }
                self.model.token_embedding.embedding_lookup(id, &mut self.x);
            }
            AsrInputRow::Embedding(embedding) => {
                if embedding.len() != config.n_embd {
                    return Err(format!(
                        "ASR embedding row length {} != {}",
                        embedding.len(),
                        config.n_embd
                    ));
                }
                self.x.copy_from_slice(embedding);
            }
        }
        #[cfg(feature = "parity-trace")]
        self.layer_hidden_trace.clear();

        let n_embd_q = config.n_head * config.n_embd_head_k;
        let n_embd_k = config.n_head_kv * config.n_embd_head_k;
        let n_embd_v = config.n_head_kv * config.n_embd_head_v;
        let n_attn = config.n_head * config.n_embd_head_v;
        let group_size = config.n_head / config.n_head_kv;
        let kq_scale = 1.0 / (config.n_embd_head_k as f32).sqrt();
        let kv_stride = n_embd_k.max(n_embd_v);
        let kv_cache_values = config.n_layer * self.capacity * kv_stride;
        let (k_cache_ptr, v_cache_ptr) = match &mut self.kv {
            KvCache::F32(cache) => (cache.k.as_mut_ptr(), cache.v.as_mut_ptr()),
            KvCache::F16(_) => return Err("ASR session requires an F32 KV cache".into()),
        };

        for layer in 0..config.n_layer {
            let weights = &self.model.layers[layer];
            // attention norm + QKV
            rms_norm(&self.x, &weights.attn_norm, &mut self.normed, config.eps);
            quantize_q8_0_into(
                &self.normed,
                config.n_embd,
                &mut self.q8_buf[..config.n_embd],
                &mut self.scale_buf[..config.n_embd / 32],
            );
            {
                let q8 = self.q8_buf[..config.n_embd].as_ptr();
                let scales = self.scale_buf[..config.n_embd / 32].as_ptr();
                let pool = Arc::clone(&self.model.pool);
                let wq = &weights.wq;
                let wk = &weights.wk;
                let wv = &weights.wv;
                let q_ptr = self.q.as_mut_ptr();
                let k_ptr = self.k.as_mut_ptr();
                let v_ptr = self.v.as_mut_ptr();
                pool.compute(move |thread, threads| {
                    let q8 = unsafe { std::slice::from_raw_parts(q8, config.n_embd) };
                    let scales = unsafe { std::slice::from_raw_parts(scales, config.n_embd / 32) };
                    let q = unsafe { std::slice::from_raw_parts_mut(q_ptr, n_embd_q) };
                    let k = unsafe { std::slice::from_raw_parts_mut(k_ptr, n_embd_k) };
                    let v = unsafe { std::slice::from_raw_parts_mut(v_ptr, n_embd_v) };
                    wq.kernel.forward_prepared(
                        &[],
                        q8,
                        scales,
                        None,
                        q,
                        config.n_embd,
                        n_embd_q,
                        thread,
                        threads,
                    );
                    wk.kernel.forward_prepared(
                        &[],
                        q8,
                        scales,
                        None,
                        k,
                        config.n_embd,
                        n_embd_k,
                        thread,
                        threads,
                    );
                    wv.kernel.forward_prepared(
                        &[],
                        q8,
                        scales,
                        None,
                        v,
                        config.n_embd,
                        n_embd_v,
                        thread,
                        threads,
                    );
                });
            }
            for (value, &bias) in self.q.iter_mut().zip(&weights.q_bias) {
                *value += bias;
            }
            for (value, &bias) in self.k.iter_mut().zip(&weights.k_bias) {
                *value += bias;
            }
            for (value, &bias) in self.v.iter_mut().zip(&weights.v_bias) {
                *value += bias;
            }

            // rope + KV store
            {
                for head in self.q[..n_embd_q].chunks_exact_mut(config.n_embd_head_k) {
                    crate::ops::rope::rope_neox_sleef(
                        head,
                        step,
                        config.n_embd_head_k,
                        config.freq_base,
                    );
                }
                for head in self.k[..n_embd_k].chunks_exact_mut(config.n_embd_head_k) {
                    crate::ops::rope::rope_neox_sleef(
                        head,
                        step,
                        config.n_embd_head_k,
                        config.freq_base,
                    );
                }
                let layer_base = layer * self.capacity * kv_stride;
                let row_base = layer_base + step * kv_stride;
                let k_cache =
                    unsafe { std::slice::from_raw_parts_mut(k_cache_ptr, kv_cache_values) };
                let v_cache =
                    unsafe { std::slice::from_raw_parts_mut(v_cache_ptr, kv_cache_values) };
                for head in 0..config.n_head_kv {
                    let offset = head * config.n_embd_head_k;
                    k_cache[row_base + offset..row_base + offset + config.n_embd_head_k]
                        .copy_from_slice(&self.k[offset..offset + config.n_embd_head_k]);
                    v_cache[row_base + offset..row_base + offset + config.n_embd_head_v]
                        .copy_from_slice(&self.v[offset..offset + config.n_embd_head_v]);
                }
            }

            // online attention over the F32 cache, parallelized across heads
            // (F32 like the dots LLM: 7B Qwen2.5 activations overflow F16
            // accumulators around the middle layers)
            {
                let pool = Arc::clone(&self.model.pool);
                let q = self.q.as_ptr();
                let attn_out_ptr = self.attn_out.as_mut_ptr();
                let layer_capacity = self.capacity;
                pool.compute(move |thread, threads| {
                    let q = unsafe { std::slice::from_raw_parts(q, n_embd_q) };
                    let attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_attn) };
                    let k_cache =
                        unsafe { std::slice::from_raw_parts(k_cache_ptr, kv_cache_values) };
                    let v_cache =
                        unsafe { std::slice::from_raw_parts(v_cache_ptr, kv_cache_values) };
                    let head_start = thread * config.n_head / threads;
                    let head_end = (thread + 1) * config.n_head / threads;
                    let layer_base = layer * layer_capacity * kv_stride;
                    let mut accumulator = vec![0.0f32; config.n_embd_head_v];
                    for head in head_start..head_end {
                        let kv_head = head / group_size;
                        let q_offset = head * config.n_embd_head_k;
                        let out_offset = head * config.n_embd_head_v;
                        let output = &mut attn_out[out_offset..out_offset + config.n_embd_head_v];
                        let query = &q[q_offset..q_offset + config.n_embd_head_k];
                        accumulator.fill(0.0);
                        let mut sum = 0.0f32;
                        let mut max = f32::NEG_INFINITY;
                        for token in 0..=step {
                            let row = layer_base + token * kv_stride;
                            let key_offset = row + kv_head * config.n_embd_head_k;
                            let score = dot_f32(
                                query,
                                &k_cache[key_offset..key_offset + config.n_embd_head_k],
                                config.n_embd_head_k,
                            ) * kq_scale;
                            let mut rescale = 1.0f32;
                            let mut weight = 1.0f32;
                            if score > max {
                                rescale = (max - score).exp();
                                max = score;
                                for value in accumulator.iter_mut() {
                                    *value *= rescale;
                                }
                            } else {
                                weight = (score - max).exp();
                            }
                            let value_offset = row + kv_head * config.n_embd_head_v;
                            let value = &v_cache[value_offset..value_offset + config.n_embd_head_v];
                            for (acc, &v) in accumulator.iter_mut().zip(value) {
                                *acc += weight * v;
                            }
                            sum = sum.mul_add(rescale, weight);
                        }
                        let reciprocal = if sum == 0.0 { 0.0 } else { sum.recip() };
                        for (out_value, &acc) in output.iter_mut().zip(accumulator.iter()) {
                            *out_value = acc * reciprocal;
                        }
                    }
                });
            }

            // output projection + residual
            quantize_q8_0_into(
                &self.attn_out,
                n_attn,
                &mut self.q8_buf[..n_attn],
                &mut self.scale_buf[..n_attn / 32],
            );
            {
                let q8 = self.q8_buf[..n_attn].as_ptr();
                let scales = self.scale_buf[..n_attn / 32].as_ptr();
                let pool = Arc::clone(&self.model.pool);
                let wo = &weights.wo;
                let attn_proj_ptr = self.attn_proj.as_mut_ptr();
                pool.compute(move |thread, threads| {
                    let q8 = unsafe { std::slice::from_raw_parts(q8, n_attn) };
                    let scales = unsafe { std::slice::from_raw_parts(scales, n_attn / 32) };
                    let output =
                        unsafe { std::slice::from_raw_parts_mut(attn_proj_ptr, config.n_embd) };
                    wo.kernel.forward_prepared(
                        &[],
                        q8,
                        scales,
                        None,
                        output,
                        n_attn,
                        config.n_embd,
                        thread,
                        threads,
                    );
                });
            }
            for (hidden, &projection) in self.x.iter_mut().zip(self.attn_proj.iter()) {
                *hidden += projection;
            }
            let weights_attn_fin = [self.x.iter().all(|value| value.is_finite())];

            // FFN: silu(gate)·up → down
            rms_norm(&self.x, &weights.ffn_norm, &mut self.normed, config.eps);
            quantize_q8_0_into(
                &self.normed,
                config.n_embd,
                &mut self.q8_buf[..config.n_embd],
                &mut self.scale_buf[..config.n_embd / 32],
            );
            {
                let q8 = self.q8_buf[..config.n_embd].as_ptr();
                let scales = self.scale_buf[..config.n_embd / 32].as_ptr();
                let pool = Arc::clone(&self.model.pool);
                let w_gate = &weights.w_gate;
                let w_up = &weights.w_up;
                let gate_ptr = self.gate.as_mut_ptr();
                let up_ptr = self.up.as_mut_ptr();
                pool.compute(move |thread, threads| {
                    let q8 = unsafe { std::slice::from_raw_parts(q8, config.n_embd) };
                    let scales = unsafe { std::slice::from_raw_parts(scales, config.n_embd / 32) };
                    let gate = unsafe { std::slice::from_raw_parts_mut(gate_ptr, config.n_ff) };
                    let up = unsafe { std::slice::from_raw_parts_mut(up_ptr, config.n_ff) };
                    w_gate.kernel.forward_prepared(
                        &[],
                        q8,
                        scales,
                        None,
                        gate,
                        config.n_embd,
                        config.n_ff,
                        thread,
                        threads,
                    );
                    w_up.kernel.forward_prepared(
                        &[],
                        q8,
                        scales,
                        None,
                        up,
                        config.n_embd,
                        config.n_ff,
                        thread,
                        threads,
                    );
                    let start = thread * config.n_ff / threads;
                    let end = (thread + 1) * config.n_ff / threads;
                    for index in start..end {
                        let value = gate[index];
                        gate[index] = value / (1.0 + (-value).exp()) * up[index];
                    }
                });
            }
            quantize_q8_0_into(
                &self.gate,
                config.n_ff,
                &mut self.q8_buf[..config.n_ff],
                &mut self.scale_buf[..config.n_ff / 32],
            );
            {
                let q8 = self.q8_buf[..config.n_ff].as_ptr();
                let scales = self.scale_buf[..config.n_ff / 32].as_ptr();
                let pool = Arc::clone(&self.model.pool);
                let w_down = &weights.w_down;
                let down_ptr = self.down.as_mut_ptr();
                pool.compute(move |thread, threads| {
                    let q8 = unsafe { std::slice::from_raw_parts(q8, config.n_ff) };
                    let scales = unsafe { std::slice::from_raw_parts(scales, config.n_ff / 32) };
                    let down = unsafe { std::slice::from_raw_parts_mut(down_ptr, config.n_embd) };
                    w_down.kernel.forward_prepared(
                        &[],
                        q8,
                        scales,
                        None,
                        down,
                        config.n_ff,
                        config.n_embd,
                        thread,
                        threads,
                    );
                });
            }
            for (hidden, &projection) in self.x.iter_mut().zip(self.down.iter()) {
                *hidden += projection;
            }
            let weights_ffn_fin = [self.x.iter().all(|value| value.is_finite())];
            if std::env::var_os("VIBEVOICE_ASR_DEBUG").is_some()
                && self.x.iter().any(|value| !value.is_finite())
            {
                let bad_attn = weights_attn_fin[0];
                let bad_ffn = weights_ffn_fin[0];
                panic!(
                    "NaN after layer {layer} (attn_out finite: {bad_attn}, ffn finite: {bad_ffn})"
                );
            }
            if std::env::var_os("VIBEVOICE_ASR_LAYER_TRACE").is_some() {
                let norm: f32 = self.x.iter().map(|v| v * v).sum::<f32>().sqrt();
                let head: Vec<String> = self.x.iter().take(4).map(|v| format!("{v:.4}")).collect();
                eprintln!("layer {layer}: hidden_norm {norm:.4} head {head:?}");
            }
            #[cfg(feature = "parity-trace")]
            self.layer_hidden_trace.extend_from_slice(&self.x);
        }

        rms_norm(
            &self.x,
            &self.model.output_norm,
            &mut self.normed,
            config.eps,
        );
        self.next_step += 1;
        Ok(())
    }

    #[cfg(feature = "parity-trace")]
    pub fn layer_hidden_trace(&self) -> &[f32] {
        &self.layer_hidden_trace
    }

    #[cfg(feature = "parity-trace")]
    pub fn normalized_hidden(&self) -> &[f32] {
        &self.normed
    }

    /// Run the untied LM head on the last hidden state; returns the logits
    /// slice (valid until the next `forward_step`/`logits` call).
    pub fn logits(&mut self) -> Result<&[f32], String> {
        let config = &self.model.config;
        if self.logits.len() != self.model.vocab_size {
            self.logits.resize(self.model.vocab_size, 0.0);
        }
        quantize_q8_0_into(
            &self.normed,
            config.n_embd,
            &mut self.q8_buf[..config.n_embd],
            &mut self.scale_buf[..config.n_embd / 32],
        );
        let q8 = self.q8_buf[..config.n_embd].as_ptr();
        let scales = self.scale_buf[..config.n_embd / 32].as_ptr();
        let pool = Arc::clone(&self.model.pool);
        let head = &self.model.lm_head;
        let vocab = self.model.vocab_size;
        let out_ptr = self.logits.as_mut_ptr();
        pool.compute(move |thread, threads| {
            let q8 = unsafe { std::slice::from_raw_parts(q8, config.n_embd) };
            let scales = unsafe { std::slice::from_raw_parts(scales, config.n_embd / 32) };
            let output = unsafe { std::slice::from_raw_parts_mut(out_ptr, vocab) };
            head.kernel.forward_prepared(
                &[],
                q8,
                scales,
                None,
                output,
                config.n_embd,
                vocab,
                thread,
                threads,
            );
        });
        Ok(&self.logits)
    }

    /// Debug: summarize the current hidden state and logits.
    pub fn debug_summary(&self, label: &str) {
        if std::env::var_os("VIBEVOICE_ASR_DEBUG").is_none() {
            return;
        }
        let norm: f32 = self.normed.iter().map(|v| v * v).sum::<f32>().sqrt();
        let head: Vec<String> = self
            .normed
            .iter()
            .take(6)
            .map(|v| format!("{v:.4}"))
            .collect();
        eprintln!(
            "[debug] {label}: pos={} hidden_norm={norm:.4} head={head:?}",
            self.next_step
        );
    }

    /// Greedy argmax over the LM-head logits.
    pub fn sample_argmax(&mut self) -> Result<u32, String> {
        let logits = self.logits()?;
        if std::env::var_os("VIBEVOICE_ASR_DEBUG").is_some() {
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
            let top: Vec<String> = order
                .iter()
                .take(5)
                .map(|&i| format!("{}:{:.3}", i, logits[i]))
                .collect();
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
            eprintln!("[debug] logits: max={max:.4} min={min:.4} top={top:?}");
        }
        let mut best = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, &value) in logits.iter().enumerate() {
            if value > best_value {
                best_value = value;
                best = index;
            }
        }
        Ok(best as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo};
    use std::collections::HashMap;

    #[derive(Default)]
    struct Source {
        metadata: HashMap<String, MetaValue>,
        infos: HashMap<String, TensorInfo>,
        bytes: HashMap<String, Vec<u8>>,
    }

    impl TensorSource for Source {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.metadata.get(key)
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.infos.get(name)
        }

        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.bytes.get(name).map(Vec::as_slice)
        }
    }

    fn add_tensor(source: &mut Source, name: &str, dims: &[u64], ggml_type: GGMLType) {
        let elements = dims.iter().product::<u64>() as usize;
        let bytes = match ggml_type {
            GGMLType::F32 => vec![0; elements * 4],
            GGMLType::Q8_0 => vec![0; elements / 32 * 34],
            other => panic!("unsupported test tensor type {other:?}"),
        };
        source.infos.insert(
            name.into(),
            TensorInfo {
                name: name.into(),
                dims: dims.to_vec(),
                ggml_type,
                offset: 0,
            },
        );
        source.bytes.insert(name.into(), bytes);
    }

    fn set_f32(source: &mut Source, name: &str, values: &[f32]) {
        source.bytes.insert(
            name.into(),
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        );
    }

    fn q8_identity(size: usize) -> Vec<u8> {
        assert_eq!(size % 32, 0);
        let mut bytes = Vec::with_capacity(size * size / 32 * 34);
        for row in 0..size {
            for block in 0..size / 32 {
                bytes.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
                for column in block * 32..(block + 1) * 32 {
                    bytes.push(u8::from(column == row));
                }
            }
        }
        bytes
    }

    fn tiny_source() -> Source {
        let mut source = Source::default();
        source.metadata.extend([
            (
                "general.architecture".into(),
                MetaValue::String("qwen2".into()),
            ),
            ("qwen2.embedding_length".into(), MetaValue::Uint64(32)),
            ("qwen2.block_count".into(), MetaValue::Uint64(1)),
            ("qwen2.attention.head_count".into(), MetaValue::Uint64(4)),
            ("qwen2.attention.head_count_kv".into(), MetaValue::Uint64(1)),
            ("qwen2.feed_forward_length".into(), MetaValue::Uint64(64)),
            ("qwen2.context_length".into(), MetaValue::Uint64(128)),
            ("qwen2.vocab_size".into(), MetaValue::Uint64(64)),
            (
                "qwen2.rope.freq_base".into(),
                MetaValue::Float64(1_000_000.0),
            ),
            (
                "qwen2.attention.layer_norm_rms_epsilon".into(),
                MetaValue::Float64(1e-6),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                MetaValue::Array(
                    MetaValueType::String,
                    (0..64)
                        .map(|index| MetaValue::String(format!("token-{index}")))
                        .collect(),
                ),
            ),
        ]);

        add_tensor(&mut source, "token_embd.weight", &[32, 64], GGMLType::Q8_0);
        add_tensor(&mut source, "output.weight", &[32, 64], GGMLType::Q8_0);
        add_tensor(&mut source, "output_norm.weight", &[32], GGMLType::F32);
        add_tensor(&mut source, "blk.0.attn_norm.weight", &[32], GGMLType::F32);
        add_tensor(&mut source, "blk.0.ffn_norm.weight", &[32], GGMLType::F32);
        add_tensor(&mut source, "blk.0.attn_q.bias", &[32], GGMLType::F32);
        add_tensor(&mut source, "blk.0.attn_k.bias", &[8], GGMLType::F32);
        add_tensor(&mut source, "blk.0.attn_v.bias", &[8], GGMLType::F32);
        add_tensor(
            &mut source,
            "blk.0.attn_q.weight",
            &[32, 32],
            GGMLType::Q8_0,
        );
        add_tensor(&mut source, "blk.0.attn_k.weight", &[32, 8], GGMLType::Q8_0);
        add_tensor(&mut source, "blk.0.attn_v.weight", &[32, 8], GGMLType::Q8_0);
        add_tensor(
            &mut source,
            "blk.0.attn_output.weight",
            &[32, 32],
            GGMLType::Q8_0,
        );
        add_tensor(
            &mut source,
            "blk.0.ffn_gate.weight",
            &[32, 64],
            GGMLType::Q8_0,
        );
        add_tensor(
            &mut source,
            "blk.0.ffn_up.weight",
            &[32, 64],
            GGMLType::Q8_0,
        );
        add_tensor(
            &mut source,
            "blk.0.ffn_down.weight",
            &[64, 32],
            GGMLType::Q8_0,
        );
        set_f32(
            &mut source,
            "blk.0.attn_v.bias",
            &(1..=8).map(|value| value as f32).collect::<Vec<_>>(),
        );
        source
            .bytes
            .insert("blk.0.attn_output.weight".into(), q8_identity(32));
        source
    }

    #[test]
    fn loads_projection_weights_with_input_dimension_first() {
        let model =
            VibeVoiceAsrLlm::from_source(Arc::new(tiny_source()), Arc::new(ComputePool::new(1)))
                .unwrap();
        let layer = &model.layers[0];
        assert_eq!((layer.wk.n_in, layer.wk.n_out), (32, 8));
        assert_eq!((layer.wv.n_in, layer.wv.n_out), (32, 8));
        assert_eq!((layer.w_gate.n_in, layer.w_gate.n_out), (32, 64));
        assert_eq!((layer.w_up.n_in, layer.w_up.n_out), (32, 64));
        assert_eq!((layer.w_down.n_in, layer.w_down.n_out), (64, 32));
    }

    #[test]
    fn qkv_bias_is_independent_of_thread_count() {
        fn hidden(threads: usize) -> Vec<f32> {
            let model = VibeVoiceAsrLlm::from_source(
                Arc::new(tiny_source()),
                Arc::new(ComputePool::new(threads)),
            )
            .unwrap();
            let mut session = AsrLlmSession::new(&model, 1).unwrap();
            session.forward_step(AsrInputRow::Token(0)).unwrap();
            session.x
        }

        assert_eq!(hidden(1), hidden(4));
    }

    #[test]
    fn rejected_input_does_not_advance_position() {
        let model =
            VibeVoiceAsrLlm::from_source(Arc::new(tiny_source()), Arc::new(ComputePool::new(1)))
                .unwrap();
        let mut session = AsrLlmSession::new(&model, 1).unwrap();

        assert!(session.forward_step(AsrInputRow::Token(64)).is_err());
        assert_eq!(session.position(), 0);
        assert!(session
            .forward_step(AsrInputRow::Embedding(&[0.0; 31]))
            .is_err());
        assert_eq!(session.position(), 0);
        session.forward_step(AsrInputRow::Token(0)).unwrap();
        assert_eq!(session.position(), 1);
        assert!(session.forward_step(AsrInputRow::Token(0)).is_err());
        assert_eq!(session.position(), 1);
    }
}
