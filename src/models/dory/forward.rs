//! Dory forward pass.
//!
//! Dory is deliberately kept separate from the llama trunk.  Its weights use
//! the same GGUF block names, but its layer stack is looped and its residual
//! update is nGPT's unit-norm interpolation rather than RMSNorm + add.

use half::f16;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::app::cli::KvFormat;
use crate::core::tensor::{load_f32_tensor, BlockQ8K, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::ops::kernel::Weight;
use crate::ops::{dot_f32_f64_acc, rope_neox_inplace_precise, sample_llama_cpp, softmax_inplace};

use super::config::DoryConfig;
use super::weights::{load_layers, load_weight, DoryLayerWeights};

const NORM_EPS: f32 = 1e-6;

pub(crate) fn trace(name: &str, layer: Option<usize>, step: usize, values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    if crate::parity_trace::enabled(name)
        && step
            >= std::env::var("RMI_DORY_TRACE_FROM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
    {
        crate::parity_trace::report(crate::parity_trace::checkpoint_at(
            name,
            layer,
            Some(step),
            &[values.len()],
            values,
        ));
    }
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, layer, step, values);
}

/// A cache is kept in F32 storage so the session stays simple; `KvFormat::F16`
/// still applies the requested half precision round-trip at insertion time.
/// The cache is an intentionally non-bitwise path, because the requested
/// model is quantized and the F16 KV option itself discards mantissa bits.
struct DoryCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    width: usize,
    kv_format: KvFormat,
}

impl DoryCache {
    fn new(width: usize, kv_format: KvFormat) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            width,
            kv_format,
        }
    }

    fn push(&mut self, key: &[f32], value: &[f32], max_len: usize) {
        debug_assert_eq!(key.len(), self.width);
        debug_assert_eq!(value.len(), self.width);
        match self.kv_format {
            KvFormat::F32 => {
                self.keys.extend_from_slice(key);
                self.values.extend_from_slice(value);
            }
            KvFormat::F16 => {
                self.keys
                    .extend(key.iter().map(|&v| f16::from_f32(v).to_f32()));
                self.values
                    .extend(value.iter().map(|&v| f16::from_f32(v).to_f32()));
            }
        }
        let len = self.keys.len() / self.width;
        if len > max_len {
            let drop = (len - max_len) * self.width;
            self.keys.drain(..drop);
            self.values.drain(..drop);
        }
    }

    fn len(&self) -> usize {
        self.keys.len() / self.width
    }
}

pub struct DoryScratch {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub gate: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub projection: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub output: Vec<f32>,
    pub logits: Vec<f32>,
    pub q8: Vec<u8>,
    pub scales: Vec<f32>,
    pub q8k: Vec<BlockQ8K>,
}

impl DoryScratch {
    fn new(cfg: &DoryConfig) -> Self {
        let q = cfg.n_head * cfg.head_dim;
        let kv = cfg.n_head_kv * cfg.head_dim;
        let max_in = cfg.n_ff.max(q).max(cfg.n_embd);
        Self {
            q: vec![0.0; q],
            k: vec![0.0; kv],
            v: vec![0.0; kv],
            gate: vec![0.0; q],
            attn_out: vec![0.0; q],
            projection: vec![0.0; cfg.n_embd],
            ffn_gate: vec![0.0; cfg.n_ff],
            ffn_up: vec![0.0; cfg.n_ff],
            output: vec![0.0; cfg.n_embd],
            logits: vec![0.0; cfg.vocab_size],
            q8: vec![0; max_in],
            scales: vec![0.0; max_in.div_ceil(32)],
            q8k: vec![
                BlockQ8K {
                    d: 0.0,
                    qs: [0; 256],
                    bsums: [0; 16],
                };
                max_in.div_ceil(256)
            ],
        }
    }
}

pub struct DoryModel {
    _source: Arc<dyn TensorSource>,
    pub config: DoryConfig,
    pub layers: Vec<DoryLayerWeights<'static>>,
    pub token_embedding: Weight<'static>,
    pub output: Weight<'static>,
    pub output_scale: Vec<f32>,
    pub pool: Arc<ComputePool>,
}

impl DoryModel {
    pub fn from_source(source: Arc<dyn TensorSource>, n_threads: usize) -> Result<Self, String> {
        let config = DoryConfig::from_source(source.as_ref())?;
        if config.recurrent_kv_cache_mode != "per_loop" {
            return Err(
                "Dory GGUF uses recurrent_kv_cache_mode=last_loop, which is not supported by this session yet".into(),
            );
        }
        let layers = load_layers(source.as_ref(), &config)?;
        let token_embedding = load_weight(
            source.as_ref(),
            "token_embd.weight",
            config.n_embd,
            config.vocab_size,
        )?;
        let output = load_weight(
            source.as_ref(),
            "output.weight",
            config.n_embd,
            config.vocab_size,
        )?;
        let output_scale = load_f32_tensor(
            source.as_ref(),
            "output_scale.weight",
            &[config.vocab_size as u64],
        )?;
        // The weights borrow the mmap owned by `source`, which is retained in
        // the model for the whole session lifetime.
        let layers = unsafe {
            std::mem::transmute::<Vec<DoryLayerWeights<'_>>, Vec<DoryLayerWeights<'static>>>(layers)
        };
        let token_embedding = unsafe { std::mem::transmute(token_embedding) };
        let output = unsafe { std::mem::transmute(output) };
        Ok(Self {
            _source: source,
            config,
            layers,
            token_embedding,
            output,
            output_scale,
            pool: Arc::new(ComputePool::new(n_threads.max(1))),
        })
    }

    pub fn new_session(&self, capacity: usize, kv_format: KvFormat) -> DorySession {
        let mut cache_slots = Vec::new();
        let mut layer_slots = vec![None; self.config.n_layer];
        for layer in 0..self.config.n_layer {
            if !self.config.is_attention(layer) {
                continue;
            }
            let loops = if self.config.in_recurrent_block(layer) {
                self.config.n_loops
            } else {
                1
            };
            layer_slots[layer] = Some(cache_slots.len());
            for _ in 0..loops {
                cache_slots.push(DoryCache::new(
                    self.config.n_head_kv * self.config.head_dim,
                    kv_format,
                ));
            }
        }
        DorySession {
            model_capacity: capacity.min(self.config.n_ctx).max(1),
            position: 0,
            model_source: self._source.clone(),
            caches: cache_slots,
            layer_slots,
            scratch: DoryScratch::new(&self.config),
            kv_format,
            policy: crate::compute::ComputePolicy::Cpu,
            #[cfg(feature = "vulkan")]
            gpu: None,
        }
    }

    pub fn new_session_with_compute(
        &self,
        capacity: usize,
        kv_format: KvFormat,
        policy: crate::compute::ComputePolicy,
    ) -> Result<DorySession, String> {
        policy.check_build()?;
        let mut session = self.new_session(capacity, kv_format);
        session.policy = policy;
        #[cfg(feature = "vulkan")]
        if let Some(context) = policy.context().map_err(|e| e.to_string())? {
            match crate::vulkan::dory::DoryVulkanSession::new(self, context, session.model_capacity)
            {
                Ok(gpu) => session.gpu = Some(gpu),
                Err(error) if policy == crate::compute::ComputePolicy::Auto => {
                    eprintln!("Dory Vulkan unavailable: {error}")
                }
                Err(error) => return Err(error.to_string()),
            }
        }
        Ok(session)
    }

    fn matmul(
        &self,
        weight: &Weight<'_>,
        input: &[f32],
        output: &mut [f32],
        q8: &mut [u8],
        scales: &mut [f32],
        q8k: &mut [BlockQ8K],
    ) {
        let n = input.len();
        let blocks = n.div_ceil(32);
        let q8k_blocks = n.div_ceil(256);
        weight.quantize_and_matmul_with_scratch(
            input,
            &mut q8k[..q8k_blocks],
            &mut q8[..n],
            &mut scales[..blocks],
            output,
            &self.pool,
        );
    }

    fn unit_norm(values: &mut [f32]) {
        // Oracle builds L2 normalization as RMSNorm(eps / n) * 1/sqrt(n).
        // Keep GGML's double accumulator and F32 intermediate order.
        if values.is_empty() {
            return;
        }
        let n = values.len() as f32;
        let mut sum = 0.0f64;
        for &value in values.iter() {
            sum += f64::from(value * value);
        }
        let mean = (sum / values.len() as f64) as f32;
        let rms_scale = 1.0f32 / (mean + NORM_EPS / n).sqrt();
        let unit_scale = 1.0f32 / n.sqrt();
        for value in values.iter_mut() {
            // The Oracle rounds after RMSNorm, before the second scale node.
            *value *= rms_scale;
            *value *= unit_scale;
        }
    }

    fn slerp_residual(hidden: &mut [f32], output: &[f32], alpha: &[f32]) {
        let mut a = hidden.to_vec();
        let mut b = output.to_vec();
        Self::unit_norm(&mut a);
        Self::unit_norm(&mut b);
        for i in 0..hidden.len() {
            hidden[i] = a[i] + alpha[i].abs() * (b[i] - a[i]);
        }
        Self::unit_norm(hidden);
    }

    fn forward_attention(
        &self,
        layer: usize,
        weights: &DoryLayerWeights<'static>,
        hidden: &mut [f32],
        position: usize,
        cache: &mut DoryCache,
        cache_len: usize,
        scratch: &mut DoryScratch,
    ) {
        let DoryLayerWeights::Attention {
            residual_alpha,
            gate_scale,
            qk_scale,
            w_gate,
            wk,
            wo,
            wq,
            wv,
        } = weights
        else {
            unreachable!()
        };
        let cfg = &self.config;
        self.matmul(
            wq,
            hidden,
            &mut scratch.q,
            &mut scratch.q8,
            &mut scratch.scales,
            &mut scratch.q8k,
        );
        self.matmul(
            wk,
            hidden,
            &mut scratch.k,
            &mut scratch.q8,
            &mut scratch.scales,
            &mut scratch.q8k,
        );
        self.matmul(
            wv,
            hidden,
            &mut scratch.v,
            &mut scratch.q8,
            &mut scratch.scales,
            &mut scratch.q8k,
        );
        self.matmul(
            w_gate,
            hidden,
            &mut scratch.gate,
            &mut scratch.q8,
            &mut scratch.scales,
            &mut scratch.q8k,
        );

        let theta = if cfg.attention_is_sliding(layer) {
            cfg.rope_freq_base_2
        } else {
            cfg.rope_freq_base
        };
        // Recurrent Q8 quantization amplifies fused-rotation rounding.
        rope_neox_inplace_precise(&mut scratch.q, position, cfg.head_dim, theta);
        rope_neox_inplace_precise(&mut scratch.k, position, cfg.head_dim, theta);
        let q_factor = (cfg.head_dim as f32).sqrt() / cfg.initializer_range;
        for head in 0..cfg.n_head {
            let start = head * cfg.head_dim;
            Self::unit_norm(&mut scratch.q[start..start + cfg.head_dim]);
            for d in 0..cfg.head_dim {
                let scale = qk_scale[start + d] * q_factor;
                scratch.q[start + d] *= scale * scale;
            }
        }
        for head in 0..cfg.n_head_kv {
            let start = head * cfg.head_dim;
            Self::unit_norm(&mut scratch.k[start..start + cfg.head_dim]);
        }
        if matches!(cache.kv_format, KvFormat::F16) {
            // GGML's F16 cache matmul also rounds its query input to F16.
            for value in &mut scratch.q {
                *value = f16::from_f32(*value).to_f32();
            }
        }
        cache.push(&scratch.k, &scratch.v, cache_len);
        let seq = cache.len();
        scratch.attn_out.fill(0.0);
        let group = cfg.n_head / cfg.n_head_kv;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        for head in 0..cfg.n_head {
            let kv_head = head / group;
            let q = &scratch.q[head * cfg.head_dim..(head + 1) * cfg.head_dim];
            let mut scores = vec![0.0; seq];
            for (j, score) in scores.iter_mut().enumerate() {
                let off = j * cfg.n_head_kv * cfg.head_dim + kv_head * cfg.head_dim;
                *score =
                    dot_f32_f64_acc(q, &cache.keys[off..off + cfg.head_dim], cfg.head_dim) * scale;
            }
            softmax_inplace(&mut scores);
            if matches!(cache.kv_format, KvFormat::F16) {
                for score in &mut scores {
                    *score = f16::from_f32(*score).to_f32();
                }
            }
            let out = &mut scratch.attn_out[head * cfg.head_dim..(head + 1) * cfg.head_dim];
            for (d, value) in out.iter_mut().enumerate() {
                // GGML's F32 value projection accumulates F32 products in f64.
                *value = scores
                    .iter()
                    .enumerate()
                    .map(|(j, &score)| {
                        let off = j * cfg.n_head_kv * cfg.head_dim + kv_head * cfg.head_dim;
                        f64::from(score * cache.values[off + d])
                    })
                    .sum::<f64>() as f32;
            }
        }
        let gate_factor = cfg.gate_init.sqrt() / cfg.initializer_range;
        for i in 0..scratch.attn_out.len() {
            scratch.attn_out[i] *=
                1.0 / (1.0 + (-scratch.gate[i] * (gate_scale[i] * gate_factor).powi(2)).exp());
        }
        Self::unit_norm(&mut scratch.attn_out);
        self.matmul(
            wo,
            &scratch.attn_out,
            &mut scratch.projection,
            &mut scratch.q8,
            &mut scratch.scales,
            &mut scratch.q8k,
        );
        let alpha_scale = self.config.alpha_init / self.config.initializer_range;
        let alpha: Vec<f32> = residual_alpha.iter().map(|v| *v * alpha_scale).collect();
        Self::slerp_residual(hidden, &scratch.projection, &alpha);
    }

    fn forward_mlp(
        &self,
        weights: &DoryLayerWeights<'static>,
        hidden: &mut [f32],
        scratch: &mut DoryScratch,
    ) {
        let DoryLayerWeights::Mlp {
            residual_alpha,
            suv_gate,
            suv_up,
            w_down,
            w_gate,
            w_up,
        } = weights
        else {
            unreachable!()
        };
        self.matmul(
            w_gate,
            hidden,
            &mut scratch.ffn_gate,
            &mut scratch.q8,
            &mut scratch.scales,
            &mut scratch.q8k,
        );
        self.matmul(
            w_up,
            hidden,
            &mut scratch.ffn_up,
            &mut scratch.q8,
            &mut scratch.scales,
            &mut scratch.q8k,
        );
        let scale = (self.config.n_embd as f32).sqrt();
        for i in 0..self.config.n_ff {
            scratch.ffn_gate[i] *= suv_gate[i] * scale;
            scratch.ffn_up[i] *= suv_up[i] * scale;
        }
        crate::ops::silu_mul_inplace(&scratch.ffn_gate, &mut scratch.ffn_up);
        Self::unit_norm(&mut scratch.ffn_up);
        self.matmul(
            w_down,
            &scratch.ffn_up,
            &mut scratch.projection,
            &mut scratch.q8,
            &mut scratch.scales,
            &mut scratch.q8k,
        );
        let alpha_scale = self.config.alpha_init / self.config.initializer_range;
        let alpha: Vec<f32> = residual_alpha.iter().map(|v| *v * alpha_scale).collect();
        Self::slerp_residual(hidden, &scratch.projection, &alpha);
    }

    fn forward_layer(
        &self,
        layer: usize,
        hidden: &mut [f32],
        position: usize,
        session: &mut DorySession,
        loop_idx: usize,
    ) {
        let weights = &self.layers[layer];
        if self.config.is_attention(layer) {
            let base = session.layer_slots[layer].expect("attention cache slot");
            let slot = if self.config.in_recurrent_block(layer) {
                base + loop_idx
            } else {
                base
            };
            let max_len = if self.config.attention_is_sliding(layer) {
                self.config.sliding_window.min(session.model_capacity)
            } else {
                session.model_capacity
            };
            let cache = &mut session.caches[slot];
            self.forward_attention(
                layer,
                weights,
                hidden,
                position,
                cache,
                max_len,
                &mut session.scratch,
            );
        } else {
            self.forward_mlp(weights, hidden, &mut session.scratch);
        }
    }

    /// Evaluate a sequence and return logits for its final token.
    pub fn prefill(
        &self,
        token_ids: &[u32],
        session: &mut DorySession,
    ) -> Result<Vec<f32>, String> {
        if !Arc::ptr_eq(&self._source, &session.model_source) {
            return Err("Dory session belongs to a different model".into());
        }
        if token_ids.is_empty() {
            return Err("Dory prefill: empty token sequence".into());
        }
        if session.position.saturating_add(token_ids.len()) > session.model_capacity {
            return Err(format!(
                "Dory prefill exceeds context capacity {}",
                session.model_capacity
            ));
        }
        if token_ids
            .iter()
            .any(|&t| t as usize >= self.config.vocab_size)
        {
            return Err("Dory token is outside vocabulary".into());
        }
        #[cfg(feature = "vulkan")]
        if let Some(gpu) = &mut session.gpu {
            let logits = gpu
                .prefill(self, token_ids, session.position, session.kv_format)
                .map_err(|e| e.to_string())?;
            session.position += token_ids.len();
            session
                .policy
                .trace("dory", crate::compute::UsedBackend::Vulkan, token_ids.len());
            return Ok(logits);
        }
        session
            .policy
            .trace("dory", crate::compute::UsedBackend::Cpu, token_ids.len());
        for &token in token_ids {
            self.token_embedding
                .embedding_lookup(token, &mut session.scratch.output);
            let mut hidden = std::mem::take(&mut session.scratch.output);
            trace("embd", None, session.position, &hidden);
            for layer in 0..self.config.n_input_layers {
                self.forward_layer(layer, &mut hidden, session.position, session, 0);
                trace("l_out", Some(layer), session.position, &hidden);
            }
            let input_h = hidden.clone();
            let mut state = input_h.clone();
            for loop_idx in 0..self.config.n_loops {
                hidden.copy_from_slice(&state);
                for i in 0..self.config.n_embd {
                    hidden[i] += input_h[i];
                }
                Self::unit_norm(&mut hidden);
                for layer in self.config.n_input_layers
                    ..self.config.n_input_layers + self.config.n_recurrent_layers
                {
                    self.forward_layer(layer, &mut hidden, session.position, session, loop_idx);
                    if layer + 1 < self.config.n_input_layers + self.config.n_recurrent_layers {
                        trace(
                            "l_out",
                            Some(layer + loop_idx * self.config.n_recurrent_layers),
                            session.position,
                            &hidden,
                        );
                    }
                }
                state = hidden.clone();
                Self::unit_norm(&mut state);
                trace(
                    "l_out",
                    Some(
                        self.config.n_input_layers
                            + (loop_idx + 1) * self.config.n_recurrent_layers
                            - 1,
                    ),
                    session.position,
                    &state,
                );
            }
            hidden.copy_from_slice(&state);
            for layer in
                self.config.n_input_layers + self.config.n_recurrent_layers..self.config.n_layer
            {
                self.forward_layer(layer, &mut hidden, session.position, session, 0);
                trace(
                    "l_out",
                    Some(layer + (self.config.n_loops - 1) * self.config.n_recurrent_layers),
                    session.position,
                    &hidden,
                );
            }
            session.scratch.output = hidden;
            session.position += 1;
        }
        trace(
            "result_norm",
            None,
            session.position - 1,
            &session.scratch.output,
        );
        self.matmul(
            &self.output,
            &session.scratch.output,
            &mut session.scratch.logits,
            &mut session.scratch.q8,
            &mut session.scratch.scales,
            &mut session.scratch.q8k,
        );
        for (logit, scale) in session.scratch.logits.iter_mut().zip(&self.output_scale) {
            *logit *= *scale * (1.0 / self.config.initializer_range);
        }
        trace(
            "result_output",
            None,
            session.position - 1,
            &session.scratch.logits,
        );
        Ok(session.scratch.logits.clone())
    }
}

pub struct DorySession {
    model_capacity: usize,
    position: usize,
    model_source: Arc<dyn TensorSource>,
    caches: Vec<DoryCache>,
    layer_slots: Vec<Option<usize>>,
    scratch: DoryScratch,
    kv_format: KvFormat,
    policy: crate::compute::ComputePolicy,
    #[cfg(feature = "vulkan")]
    gpu: Option<crate::vulkan::dory::DoryVulkanSession>,
}

impl DorySession {
    pub fn reset(&mut self, model: &DoryModel, kv_format: KvFormat) {
        #[cfg(feature = "vulkan")]
        if let Some(gpu) = &mut self.gpu {
            gpu.failed = false;
            self.position = 0;
            self.kv_format = kv_format;
            return;
        }
        let policy = self.policy;
        *self = model.new_session(self.model_capacity, kv_format);
        self.policy = policy;
    }
}

fn sample(logits: &[f32], temperature: f32) -> u32 {
    if temperature <= 0.0 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    } else {
        let mut values = logits.to_vec();
        sample_llama_cpp(&mut values, 0, 1.0, temperature, rand::random()) as u32
    }
}

pub fn run_inference(
    source: Arc<dyn TensorSource>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    kv_format: KvFormat,
    repetition_penalty: f32,
    chat_template: Option<&str>,
    thinking: bool,
    jinja: &crate::prompt::jinja::Options,
) -> Result<(), String> {
    let model = DoryModel::from_source(source.clone(), n_threads_arg)?;
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
        .map_err(|e| format!("Failed to initialize Dory tokenizer: {e}"))?;
    let template = jinja.resolve(&|key| source.metadata(key).cloned())?;
    let prompt_ids = if let Some(template) = template {
        crate::prompt::jinja::render_tokens(
            &tokenizer,
            &template,
            &[crate::prompt::jinja::ChatMessage::text("user", prompt)],
            true,
            thinking,
        )?
    } else {
        let formatted_prompt = crate::prompt::legacy::format_chat("dory", chat_template, prompt)
            .unwrap_or_else(|| prompt.to_string());
        tokenizer.encode(
            &formatted_prompt,
            EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        )
    };
    if prompt_ids.is_empty() {
        return Err("Dory prompt produced no tokens".into());
    }
    println!(
        "Model: dory | n_embd={} n_layer={} loops={} vocab={}",
        model.config.n_embd, model.config.n_layer, model.config.n_loops, model.config.vocab_size
    );
    let capacity = model.config.n_ctx.min(
        prompt_ids
            .len()
            .saturating_add(max_tokens)
            .max(prompt_ids.len()),
    );
    let mut session = model.new_session_with_compute(
        capacity,
        kv_format,
        crate::compute::ComputePolicy::legacy(),
    )?;
    let started = Instant::now();
    let mut logits = model.prefill(&prompt_ids, &mut session)?;
    let prefill = started.elapsed();
    let mut generated = Vec::with_capacity(max_tokens);
    let mut counts = HashMap::<u32, u32>::new();
    let decode_start = Instant::now();
    for step in 0..max_tokens {
        crate::ops::apply_repetition_penalty(&mut logits, &counts, repetition_penalty);
        let next = sample(&logits, temperature);
        if Some(next) == tokenizer.eos_id() {
            break;
        }
        generated.push(next);
        *counts.entry(next).or_insert(0) += 1;
        if step + 1 < max_tokens {
            logits = model.prefill(&[next], &mut session)?;
        }
    }
    println!("Output: {}", tokenizer.decode(&generated, true));
    eprintln!(
        "Prompt: {:.1} t/s | Generation: {:.1} t/s | end-to-end: {:.1} tok/s",
        crate::app::cli::per_second(prompt_ids.len(), prefill),
        crate::app::cli::per_second(generated.len(), decode_start.elapsed()),
        crate::app::cli::per_second(prompt_ids.len() + generated.len(), started.elapsed())
    );
    Ok(())
}

pub fn run_forward_logits_dory(
    source: Arc<dyn TensorSource>,
    prompt_tokens: &[u32],
    n_threads: usize,
) -> Result<(Vec<f32>, Duration), String> {
    let model = DoryModel::from_source(source, n_threads)?;
    let mut session = model.new_session_with_compute(
        prompt_tokens.len().max(1),
        KvFormat::F32,
        crate::compute::ComputePolicy::legacy(),
    )?;
    let started = Instant::now();
    let logits = model.prefill(prompt_tokens, &mut session)?;
    Ok((logits, started.elapsed()))
}

#[cfg(test)]
mod tests {
    use super::DoryModel;

    #[test]
    fn unit_norm_is_l2_not_rms() {
        let mut values = [3.0f32, 4.0];
        DoryModel::unit_norm(&mut values);
        assert!((values[0] - 0.6).abs() < 1e-6);
        assert!((values[1] - 0.8).abs() < 1e-6);
    }
}
