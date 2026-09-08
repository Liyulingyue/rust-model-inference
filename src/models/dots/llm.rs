//! dotstts LLM half: a 28×1536 Qwen2 decoder (arch `qwen2`) driven step by
//! step so the flow-matching pipeline can interleave LLM forwards with FM
//! decodes. Mirrors `models::qwen3::tts::talker::TtsSession` but for the
//! plain Qwen2 layout (no per-head Q/K RMSNorm, single scalar Neox rope) and
//! exposes the raw hidden state of every step for `hidden_proj`/`eos_proj`.

use std::sync::Arc;

use crate::core::scratchpad::KvCache;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::models::dots::patch_encoder::torch_rms_norm_with_eps;
use crate::models::dots::speaker::exp::torch28_exp;
use crate::models::dots::weights::load_weight;
use crate::ops::kernel::Weight;
use crate::ops::{dot_f32, vec_mad_f32};

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
        if cfg.architecture != "qwen2" {
            return Err(format!(
                "dots LLM requires qwen2, found {}",
                cfg.architecture
            ));
        }
        let vocab_size = source
            .metadata("tokenizer.ggml.tokens")
            .and_then(|v| v.to_arr())
            .map(Vec::len)
            .unwrap_or(0);
        if vocab_size == 0 || cfg.n_layer == 0 || cfg.n_ff == 0 || cfg.n_ctx == 0 {
            return Err("dots LLM requires nonzero vocabulary, layers, FFN and context".into());
        }
        if cfg.n_head_kv == 0
            || cfg.n_head % cfg.n_head_kv != 0
            || cfg.n_embd_head_k != cfg.n_embd_head_v
            || cfg.n_embd_head_k % 2 != 0
            || cfg.n_head.checked_mul(cfg.n_embd_head_k).is_none()
        {
            return Err("dots LLM has invalid grouped attention or rotary head dimensions".into());
        }
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

struct DotsLayerWeights {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q_bias: Vec<f32>,
    k_bias: Vec<f32>,
    v_bias: Vec<f32>,
    wq: Weight<'static>,
    wk: Weight<'static>,
    wv: Weight<'static>,
    wo: Weight<'static>,
    w_gate: Weight<'static>,
    w_up: Weight<'static>,
    w_down: Weight<'static>,
}

/// Loaded Qwen2 LLM for dots.tts.
pub struct DotsLlm {
    pub pool: Arc<ComputePool>,
    pub config: DotsLlmConfig,
    pub output_norm: Vec<f32>,
    layers: Vec<DotsLayerWeights>,
    token_embedding: Weight<'static>,
    /// Dropped after the private weight views, which cannot escape this owner.
    _source: Arc<dyn TensorSource>,
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
        let linear = |name: &str, n_in: usize, n_out: usize| -> Result<Weight<'static>, String> {
            let weight = load_weight(source.as_ref(), name, &[n_in as u64, n_out as u64])?;
            // SAFETY: all borrowed views stay private and are dropped before _source.
            Ok(unsafe { std::mem::transmute::<Weight<'_>, Weight<'static>>(weight) })
        };
        let token_embedding = linear("token_embd.weight", config.n_embd, config.vocab_size)?;
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
                q_bias: crate::core::tensor::load_f32_tensor(
                    source.as_ref(),
                    &name("attn_q.bias"),
                    &[n_embd_q as u64],
                )?,
                k_bias: crate::core::tensor::load_f32_tensor(
                    source.as_ref(),
                    &name("attn_k.bias"),
                    &[n_embd_k as u64],
                )?,
                v_bias: crate::core::tensor::load_f32_tensor(
                    source.as_ref(),
                    &name("attn_v.bias"),
                    &[n_embd_k as u64],
                )?,
                wq: linear(&name("attn_q.weight"), config.n_embd, n_embd_q)?,
                wk: linear(&name("attn_k.weight"), config.n_embd, n_embd_k)?,
                wv: linear(&name("attn_v.weight"), config.n_embd, n_embd_k)?,
                wo: linear(&name("attn_output.weight"), n_embd_q, config.n_embd)?,
                w_gate: linear(&name("ffn_gate.weight"), config.n_embd, config.n_ff)?,
                w_up: linear(&name("ffn_up.weight"), config.n_embd, config.n_ff)?,
                w_down: linear(&name("ffn_down.weight"), config.n_ff, config.n_embd)?,
            });
        }
        Ok(Self {
            _source: source,
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

/// One LLM step with hidden-state capture and pooled native weight kernels.
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
    attention_scratch: Vec<f32>,
    input_q8: Vec<u8>,
    input_scales: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    step: usize,
    capacity: usize,
}

fn matmul(
    weight: &Weight<'_>,
    input: &[f32],
    output: &mut [f32],
    bias: Option<&[f32]>,
    input_q8: &mut [u8],
    input_scales: &mut [f32],
    pool: &ComputePool,
) {
    debug_assert_eq!(input.len() % weight.n_in, 0);
    debug_assert_eq!(output.len(), input.len() / weight.n_in * weight.n_out);
    if weight.ggml_type == crate::core::tensor::GGMLType::F32 {
        super::weights::linear_forward(weight, bias, input, weight.n_in, weight.n_out, output);
        return;
    }
    for (input, output) in input
        .chunks_exact(weight.n_in)
        .zip(output.chunks_exact_mut(weight.n_out))
    {
        weight.quantize_and_matmul_with_scratch(
            input,
            &mut [],
            input_q8,
            input_scales,
            output,
            pool,
        );
        if let Some(bias) = bias {
            vec_mad_f32(output, bias, 1.0);
        }
    }
}

/// Causal SIMD attention with Torch's operand scaling and four-lane softmax.
/// Head slices retain the full [token, head, dim] row stride.
#[allow(clippy::too_many_arguments)]
fn attention_head(
    q: &[f32],
    rows: usize,
    q_stride: usize,
    k_cache: &[f32],
    v_cache: &[f32],
    keys: usize,
    first_query: usize,
    head_offset: usize,
    head_dim: usize,
    cache_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    scratch: &mut [f32],
) -> Result<(), String> {
    if rows == 0 {
        return Ok(());
    }
    let query_end = first_query
        .checked_add(rows)
        .ok_or_else(|| "dots LLM attention query length overflow".to_string())?;
    if keys < query_end {
        return Err("dots LLM attention cache does not cover causal queries".into());
    }
    if head_dim == 0 || q_stride < head_dim || output_stride < head_dim {
        return Err("dots LLM attention head stride is narrower than its head".into());
    }
    let q_span = rows
        .checked_sub(1)
        .and_then(|last| last.checked_mul(q_stride))
        .and_then(|start| start.checked_add(head_dim))
        .ok_or_else(|| "dots LLM attention query span overflow".to_string())?;
    let output_span = rows
        .checked_sub(1)
        .and_then(|last| last.checked_mul(output_stride))
        .and_then(|start| start.checked_add(head_dim))
        .ok_or_else(|| "dots LLM attention output span overflow".to_string())?;
    let head_end = head_offset
        .checked_add(head_dim)
        .ok_or_else(|| "dots LLM attention head span overflow".to_string())?;
    let scratch_len = head_dim
        .checked_mul(2)
        .and_then(|width| width.checked_add(keys))
        .ok_or_else(|| "dots LLM attention scratch length overflow".to_string())?;
    let cache_span = keys
        .checked_sub(1)
        .and_then(|last| last.checked_mul(cache_stride))
        .and_then(|start| start.checked_add(head_end))
        .ok_or_else(|| "dots LLM attention cache span overflow".to_string())?;
    if q.len() < q_span
        || output.len() < output_span
        || k_cache.len() < cache_span
        || v_cache.len() < cache_span
    {
        return Err("dots LLM attention buffer is shorter than its declared shape".into());
    }

    if cache_stride < head_end || scratch.len() < scratch_len {
        return Err("dots LLM attention cache stride or scratch is too small".into());
    }
    let (query, scratch) = scratch.split_at_mut(head_dim);
    let (key_scaled, scores) = scratch.split_at_mut(head_dim);
    let scale_sqrt = (1.0 / (head_dim as f32).sqrt()).sqrt();
    for row in 0..rows {
        let valid = first_query + row + 1;
        for (scaled, value) in query
            .iter_mut()
            .zip(&q[row * q_stride..row * q_stride + head_dim])
        {
            *scaled = *value * scale_sqrt;
        }
        let mut max = f32::NEG_INFINITY;
        for (key, score) in scores[..valid].iter_mut().enumerate() {
            let offset = key * cache_stride + head_offset;
            for (scaled, value) in key_scaled
                .iter_mut()
                .zip(&k_cache[offset..offset + head_dim])
            {
                *scaled = *value * scale_sqrt;
            }
            *score = dot_f32(query, key_scaled, head_dim);
            max = max.max(*score);
        }
        let mut sum4 = [0.0f32; 4];
        for (key, score) in scores[..valid].iter_mut().enumerate() {
            *score = torch28_exp(*score - max);
            sum4[key % 4] += *score;
        }
        let reciprocal = ((sum4[0] + sum4[2]) + (sum4[1] + sum4[3])).recip();
        let out = &mut output[row * output_stride..row * output_stride + head_dim];
        out.fill(0.0);
        for (key, score) in scores[..valid].iter().enumerate() {
            let offset = key * cache_stride + head_offset;
            vec_mad_f32(
                out,
                &v_cache[offset..offset + head_dim],
                *score * reciprocal,
            );
        }
    }
    Ok(())
}

#[cfg(feature = "parity-trace")]
fn debug_bits(label: &str, values: &[f32]) {
    if std::env::var_os("DOTS_LLM_DEBUG").is_some() {
        let bits = values
            .iter()
            .take(if std::env::var_os("DOTS_LLM_DEBUG_ALL").is_some() {
                values.len()
            } else {
                8
            })
            .map(|v| format!("{:08x}", v.to_bits()))
            .collect::<Vec<_>>()
            .join(",");
        eprintln!("dots.llm.debug {label} {bits}");
    }
}

// Optional parity-trace sidecar dumps; the helper is a no-op in normal builds.
#[allow(unused_variables)]
fn dump_stage(name: &str, position: usize, layer: usize, values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    if let Some(dir) = std::env::var_os("DOTS_LLM_STAGE_OUT") {
        let dir = std::path::Path::new(&dir);
        if let Err(error) = std::fs::create_dir_all(dir) {
            panic!("create dots LLM stage directory {}: {error}", dir.display());
        }
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let path = dir.join(format!("{name}.p{position}.l{layer}.f32"));
        if let Err(error) = std::fs::write(&path, bytes) {
            panic!("write dots LLM stage {}: {error}", path.display());
        }
    }
}

impl<'model> DotsLlmSession<'model> {
    pub fn new(model: &'model DotsLlm) -> Result<Self, String> {
        let cfg = &model.config;
        let n_embd_q = cfg.n_head * cfg.n_embd_head;
        let n_embd_kv = cfg.n_head_kv * cfg.n_embd_head;
        // the reference runtime caps the static LLM cache at
        // DEFAULT_MAX_SEQUENCE_LENGTH = 2048; keep the same bound so a
        // 131k-context gguf does not force multi-GB caches
        let capacity = cfg.n_ctx.min(2048);
        Ok(Self {
            model,
            kv: KvCache::new_f32(cfg.n_layer, capacity, n_embd_kv),
            x: vec![0.0; cfg.n_embd],
            normed: vec![0.0; cfg.n_embd],
            q: vec![0.0; n_embd_q],
            k: vec![0.0; n_embd_kv],
            v: vec![0.0; n_embd_kv],
            attn_out: vec![0.0; n_embd_q],
            attention_scratch: vec![0.0; capacity + 2 * cfg.n_embd_head],
            input_q8: vec![0; cfg.n_embd.max(n_embd_q).max(cfg.n_ff)],
            input_scales: vec![0.0; cfg.n_embd.max(n_embd_q).max(cfg.n_ff).div_ceil(32)],
            gate: vec![0.0; cfg.n_ff],
            up: vec![0.0; cfg.n_ff],
            down: vec![0.0; cfg.n_embd],
            step: 0,
            capacity,
        })
    }

    /// Length of the currently cached prefix.
    pub fn position(&self) -> usize {
        self.step
    }

    /// Embed + run one forward; returns the final normalized hidden row.
    pub fn step_row(&mut self, row: LlmInputRow<'_>) -> Result<Vec<f32>, String> {
        if self.step >= self.capacity {
            return Err(format!(
                "dotstts LLM session exceeds context {}",
                self.capacity
            ));
        }
        match row {
            LlmInputRow::Token(id) => {
                if id as usize >= self.model.config.vocab_size {
                    return Err(format!("dotstts LLM token {id} is outside the vocabulary"));
                }
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
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.llm.hidden",
            None,
            &[1, self.x.len()],
            &self.x,
        ));
        self.step += 1;
        Ok(self.x.clone())
    }

    /// Run the initial prefill as one batched causal forward.
    pub fn prefill_rows(&mut self, rows: &[LlmInputRow<'_>]) -> Result<Vec<f32>, String> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        if self.step != 0 {
            return Err("dotstts LLM batched prefill requires a fresh session".into());
        }
        let hidden = self.forward_batch(rows)?;
        #[cfg(feature = "parity-trace")]
        for row in hidden.chunks_exact(self.model.config.n_embd) {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.llm.hidden",
                None,
                &[1, self.model.config.n_embd],
                row,
            ));
        }
        self.step = rows.len();
        self.x.copy_from_slice(
            &hidden[(rows.len() - 1) * self.model.config.n_embd
                ..rows.len() * self.model.config.n_embd],
        );
        Ok(hidden)
    }

    /// Run the Qwen2 decoder for the current `self.x` at position `self.step`.
    fn forward(&mut self) -> Result<(), String> {
        let cfg = &self.model.config;
        let n_embd_q = cfg.n_head * cfg.n_embd_head;
        let n_embd_kv = cfg.n_head_kv * cfg.n_embd_head;
        let group_size = cfg.n_head / cfg.n_head_kv;
        let mut matmul =
            |weight: &Weight<'_>, input: &[f32], output: &mut [f32], bias: Option<&[f32]>| {
                matmul(
                    weight,
                    input,
                    output,
                    bias,
                    &mut self.input_q8,
                    &mut self.input_scales,
                    &self.model.pool,
                );
            };
        let kv_stride = n_embd_kv;
        let (k_cache, v_cache) = match &mut self.kv {
            KvCache::F32(cache) => (&mut cache.k, &mut cache.v),
            KvCache::F16(_) => return Err("dotstts LLM requires an F32 KV cache".into()),
        };

        for layer in 0..cfg.n_layer {
            let weights = &self.model.layers[layer];
            // 1. attention norm + QKV
            if layer == 0 {
                dump_stage("input", self.step, layer, &self.x);
            }
            torch_rms_norm_with_eps(&self.x, &weights.attn_norm, &mut self.normed, cfg.eps);
            if layer == 0 {
                dump_stage("rms_attn", self.step, layer, &self.normed);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("norm", &self.normed);
            }
            matmul(
                &weights.wq,
                &self.normed,
                &mut self.q,
                Some(&weights.q_bias),
            );
            if layer == 0 {
                dump_stage("q", self.step, layer, &self.q);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("q", &self.q);
            }
            matmul(
                &weights.wk,
                &self.normed,
                &mut self.k,
                Some(&weights.k_bias),
            );
            if layer == 0 {
                dump_stage("k", self.step, layer, &self.k);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("k", &self.k);
            }
            matmul(
                &weights.wv,
                &self.normed,
                &mut self.v,
                Some(&weights.v_bias),
            );
            if layer == 0 {
                dump_stage("v", self.step, layer, &self.v);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("v", &self.v);
            }
            // 2. rope + KV store (F32, matching the float32 Torch oracle)
            for head in self.q.chunks_exact_mut(cfg.n_embd_head) {
                crate::ops::rope::rope_neox_sleef(head, self.step, cfg.n_embd_head, cfg.freq_base);
            }
            for head in self.k.chunks_exact_mut(cfg.n_embd_head) {
                crate::ops::rope::rope_neox_sleef(head, self.step, cfg.n_embd_head, cfg.freq_base);
            }
            if layer == 0 {
                dump_stage("rope_q", self.step, layer, &self.q);
                dump_stage("rope_k", self.step, layer, &self.k);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("q_rope", &self.q);
                debug_bits("k_rope", &self.k);
            }
            let layer_base = layer * self.capacity * kv_stride;
            let row_base = layer_base + self.step * kv_stride;
            for kv_head in 0..cfg.n_head_kv {
                let offset = kv_head * cfg.n_embd_head;
                let dst = row_base + offset;
                k_cache[dst..dst + cfg.n_embd_head]
                    .copy_from_slice(&self.k[offset..offset + cfg.n_embd_head]);
                v_cache[dst..dst + cfg.n_embd_head]
                    .copy_from_slice(&self.v[offset..offset + cfg.n_embd_head]);
            }
            if layer == 0 {
                dump_stage(
                    "cache_k",
                    self.step,
                    layer,
                    &k_cache[layer_base..layer_base + (self.step + 1) * kv_stride],
                );
                dump_stage(
                    "cache_v",
                    self.step,
                    layer,
                    &v_cache[layer_base..layer_base + (self.step + 1) * kv_stride],
                );
            }
            // 3. attention over the F32 cache (safe slices)
            self.attn_out.fill(0.0);
            for head in 0..cfg.n_head {
                let kv_head = head / group_size;
                let q_offset = head * cfg.n_embd_head;
                let out_offset = head * cfg.n_embd_head;
                attention_head(
                    &self.q[q_offset..],
                    1,
                    n_embd_q,
                    &k_cache[layer_base..layer_base + self.capacity * kv_stride],
                    &v_cache[layer_base..layer_base + self.capacity * kv_stride],
                    self.step + 1,
                    self.step,
                    kv_head * cfg.n_embd_head,
                    cfg.n_embd_head,
                    kv_stride,
                    &mut self.attn_out[out_offset..],
                    n_embd_q,
                    &mut self.attention_scratch,
                )?;
            }
            if layer == 0 {
                dump_stage("attention", self.step, layer, &self.attn_out);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("attn_out", &self.attn_out);
            }
            // 4. output projection + residual
            matmul(&weights.wo, &self.attn_out, &mut self.down, None);
            if layer == 0 {
                dump_stage("o", self.step, layer, &self.down);
            }
            vec_mad_f32(&mut self.x, &self.down, 1.0);
            if layer == 0 {
                dump_stage("o_residual", self.step, layer, &self.x);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("attn_resid", &self.x);
            }
            // 5. FFN: gate·up with SiLU, then down
            torch_rms_norm_with_eps(&self.x, &weights.ffn_norm, &mut self.normed, cfg.eps);
            if layer == 0 {
                dump_stage("rms_ffn", self.step, layer, &self.normed);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("ffn_norm", &self.normed);
            }
            matmul(&weights.w_gate, &self.normed, &mut self.gate, None);
            matmul(&weights.w_up, &self.normed, &mut self.up, None);
            if layer == 0 {
                dump_stage("gate", self.step, layer, &self.gate);
                dump_stage("up", self.step, layer, &self.up);
            }
            for i in 0..cfg.n_ff {
                let gate = self.gate[i];
                self.gate[i] = gate / (1.0 + torch28_exp(-gate)) * self.up[i];
            }
            if layer == 0 {
                dump_stage("ffn_act", self.step, layer, &self.gate);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("ffn_gate_up", &self.gate);
            }
            matmul(&weights.w_down, &self.gate, &mut self.down, None);
            if layer == 0 {
                dump_stage("ffn_down", self.step, layer, &self.down);
            }
            vec_mad_f32(&mut self.x, &self.down, 1.0);
            if layer == 0 {
                dump_stage("layer0_hidden", self.step, layer, &self.x);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("ffn_resid", &self.x);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 {
                debug_bits(&format!("layer{layer}.ffn_resid"), &self.x);
            }
        }
        // Qwen2Model applies its final RMSNorm before exposing hidden states.
        torch_rms_norm_with_eps(&self.x, &self.model.output_norm, &mut self.normed, cfg.eps);
        self.x.copy_from_slice(&self.normed);
        dump_stage("hidden", self.step, 0, &self.x);
        #[cfg(feature = "parity-trace")]
        if self.step == 0 {
            debug_bits("final", &self.x);
        }
        Ok(())
    }

    fn forward_batch(&mut self, rows: &[LlmInputRow<'_>]) -> Result<Vec<f32>, String> {
        let cfg = &self.model.config;
        let rows_len = rows.len();
        let n_embd_q = cfg.n_head * cfg.n_embd_head;
        let n_embd_kv = cfg.n_head_kv * cfg.n_embd_head;
        let group_size = cfg.n_head / cfg.n_head_kv;
        if rows_len > self.capacity {
            return Err(format!(
                "dotstts LLM prefill exceeds context {}",
                self.capacity
            ));
        }

        let mut matmul =
            |weight: &Weight<'_>, input: &[f32], output: &mut [f32], bias: Option<&[f32]>| {
                matmul(
                    weight,
                    input,
                    output,
                    bias,
                    &mut self.input_q8,
                    &mut self.input_scales,
                    &self.model.pool,
                );
            };
        let mut x = vec![0.0f32; rows_len * cfg.n_embd];
        for (row_index, row) in rows.iter().enumerate() {
            let dst = &mut x[row_index * cfg.n_embd..(row_index + 1) * cfg.n_embd];
            match row {
                LlmInputRow::Token(id) => {
                    if *id as usize >= cfg.vocab_size {
                        return Err(format!("dotstts LLM token {id} is outside the vocabulary"));
                    }
                    self.model.token_embedding.embedding_lookup(*id, dst);
                }
                LlmInputRow::Embedding(embedding) => {
                    if embedding.len() != cfg.n_embd {
                        return Err(format!(
                            "dotstts LLM embedding length {} != {}",
                            embedding.len(),
                            cfg.n_embd
                        ));
                    }
                    dst.copy_from_slice(embedding);
                }
            }
            #[cfg(feature = "parity-trace")]
            dump_stage("batch_input", row_index, 0, dst);
        }
        let mut normed = vec![0.0f32; rows_len * cfg.n_embd];
        let mut q = vec![0.0f32; rows_len * n_embd_q];
        let mut k = vec![0.0f32; rows_len * n_embd_kv];
        let mut v = vec![0.0f32; rows_len * n_embd_kv];
        let mut attn_out = vec![0.0f32; rows_len * n_embd_q];
        let mut down = vec![0.0f32; rows_len * cfg.n_embd];
        let mut gate = vec![0.0f32; rows_len * cfg.n_ff];
        let mut up = vec![0.0f32; rows_len * cfg.n_ff];

        for layer in 0..cfg.n_layer {
            let weights = &self.model.layers[layer];
            for row in 0..rows_len {
                torch_rms_norm_with_eps(
                    &x[row * cfg.n_embd..(row + 1) * cfg.n_embd],
                    &weights.attn_norm,
                    &mut normed[row * cfg.n_embd..(row + 1) * cfg.n_embd],
                    cfg.eps,
                );
            }
            matmul(&weights.wq, &normed, &mut q, Some(&weights.q_bias));
            matmul(&weights.wk, &normed, &mut k, Some(&weights.k_bias));
            matmul(&weights.wv, &normed, &mut v, Some(&weights.v_bias));
            dump_stage("batch_q", 0, layer, &q);
            dump_stage("batch_k", 0, layer, &k);
            dump_stage("batch_v", 0, layer, &v);
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.norm.row1"),
                    &normed[cfg.n_embd..2 * cfg.n_embd],
                );
                debug_bits(
                    &format!("batch.l{layer}.q.row1"),
                    &q[n_embd_q..2 * n_embd_q],
                );
                debug_bits(
                    &format!("batch.l{layer}.k.row1"),
                    &k[n_embd_kv..2 * n_embd_kv],
                );
                debug_bits(
                    &format!("batch.l{layer}.v.row1"),
                    &v[n_embd_kv..2 * n_embd_kv],
                );
                if std::env::var_os("DOTS_LLM_DEBUG_BATCH_ROW2").is_some() && layer == 0 {
                    debug_bits(
                        "batch.l0.q.row2.head0",
                        &q[2 * n_embd_q..2 * n_embd_q + cfg.n_embd_head],
                    );
                    for row in 0..3 {
                        let start = row * n_embd_kv;
                        debug_bits(
                            &format!("batch.l0.k.row{row}.head0"),
                            &k[start..start + cfg.n_embd_head],
                        );
                        debug_bits(
                            &format!("batch.l0.v.row{row}.head0"),
                            &v[start..start + cfg.n_embd_head],
                        );
                    }
                }
            }
            for row in 0..rows_len {
                for head in
                    q[row * n_embd_q..(row + 1) * n_embd_q].chunks_exact_mut(cfg.n_embd_head)
                {
                    crate::ops::rope::rope_neox_sleef(head, row, cfg.n_embd_head, cfg.freq_base);
                }
                for head in
                    k[row * n_embd_kv..(row + 1) * n_embd_kv].chunks_exact_mut(cfg.n_embd_head)
                {
                    crate::ops::rope::rope_neox_sleef(head, row, cfg.n_embd_head, cfg.freq_base);
                }
            }
            dump_stage("batch_rope_q", 0, layer, &q);
            dump_stage("batch_rope_k", 0, layer, &k);
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.rope_q.row1"),
                    &q[n_embd_q..2 * n_embd_q],
                );
                if std::env::var_os("DOTS_LLM_DEBUG_BATCH_ROW2").is_some() && layer == 0 {
                    for head in 0..cfg.n_head {
                        let start = 2 * n_embd_q + head * cfg.n_embd_head;
                        debug_bits(
                            &format!("batch.l0.rope_q.row2.head{head}"),
                            &q[start..start + cfg.n_embd_head],
                        );
                    }
                    for head in 0..cfg.n_head_kv {
                        let start = 2 * n_embd_kv + head * cfg.n_embd_head;
                        debug_bits(
                            &format!("batch.l0.rope_k.row2.head{head}"),
                            &k[start..start + cfg.n_embd_head],
                        );
                    }
                }
                debug_bits(&format!("batch.l{layer}.rope_k.row0"), &k[..n_embd_kv]);
                debug_bits(
                    &format!("batch.l{layer}.rope_k.row1"),
                    &k[n_embd_kv..2 * n_embd_kv],
                );
            }
            let layer_base = layer * self.capacity * n_embd_kv;
            let (k_cache, v_cache) = match &mut self.kv {
                KvCache::F32(cache) => (&mut cache.k, &mut cache.v),
                KvCache::F16(_) => return Err("dotstts LLM requires an F32 KV cache".into()),
            };
            for row in 0..rows_len {
                let row_base = layer_base + row * n_embd_kv;
                k_cache[row_base..row_base + n_embd_kv]
                    .copy_from_slice(&k[row * n_embd_kv..(row + 1) * n_embd_kv]);
                v_cache[row_base..row_base + n_embd_kv]
                    .copy_from_slice(&v[row * n_embd_kv..(row + 1) * n_embd_kv]);
            }
            for head in 0..cfg.n_head {
                let kv_head = head / group_size;
                let offset = head * cfg.n_embd_head;
                attention_head(
                    &q[offset..],
                    rows_len,
                    n_embd_q,
                    &k_cache[layer_base..layer_base + self.capacity * n_embd_kv],
                    &v_cache[layer_base..layer_base + self.capacity * n_embd_kv],
                    rows_len,
                    0,
                    kv_head * cfg.n_embd_head,
                    cfg.n_embd_head,
                    n_embd_kv,
                    &mut attn_out[offset..],
                    n_embd_q,
                    &mut self.attention_scratch,
                )?;
            }
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.attn_out.row1"),
                    &attn_out[n_embd_q..2 * n_embd_q],
                );
                if std::env::var_os("DOTS_LLM_DEBUG_BATCH_ROW2").is_some() && layer == 0 {
                    for head in 0..cfg.n_head {
                        let start = 2 * n_embd_q + head * cfg.n_embd_head;
                        debug_bits(
                            &format!("batch.l0.attn_out.row2.head{head}"),
                            &attn_out[start..start + cfg.n_embd_head],
                        );
                    }
                    debug_bits(
                        "batch.l0.attn_out.row2.head0",
                        &attn_out[2 * n_embd_q..2 * n_embd_q + cfg.n_embd_head],
                    );
                }
            }
            matmul(&weights.wo, &attn_out, &mut down, None);
            dump_stage("batch_attn", 0, layer, &attn_out);
            dump_stage("batch_o", 0, layer, &down);
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.o.row1"),
                    &down[n_embd_q..n_embd_q + cfg.n_embd],
                );
                if std::env::var_os("DOTS_LLM_DEBUG_BATCH_ROW2").is_some() && layer == 0 {
                    debug_bits("batch.l0.o.row2", &down[2 * cfg.n_embd..3 * cfg.n_embd]);
                }
            }
            for (value, residual) in down.iter_mut().zip(x.iter()) {
                *value += *residual;
            }
            x.copy_from_slice(&down);
            dump_stage("batch_layer_hidden", 0, layer, &x);
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.o_residual.row1"),
                    &x[cfg.n_embd..2 * cfg.n_embd],
                );
                if std::env::var_os("DOTS_LLM_DEBUG_BATCH_ROW2").is_some() && layer == 0 {
                    debug_bits(
                        "batch.l0.o_residual.row2",
                        &x[2 * cfg.n_embd..3 * cfg.n_embd],
                    );
                }
            }
            for row in 0..rows_len {
                torch_rms_norm_with_eps(
                    &x[row * cfg.n_embd..(row + 1) * cfg.n_embd],
                    &weights.ffn_norm,
                    &mut normed[row * cfg.n_embd..(row + 1) * cfg.n_embd],
                    cfg.eps,
                );
            }
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.rms_ffn.row1"),
                    &normed[cfg.n_embd..2 * cfg.n_embd],
                );
            }
            matmul(&weights.w_gate, &normed, &mut gate, None);
            matmul(&weights.w_up, &normed, &mut up, None);
            dump_stage("batch_gate", 0, layer, &gate);
            dump_stage("batch_up", 0, layer, &up);
            for (gate_value, up_value) in gate.iter_mut().zip(up.iter()) {
                let value = *gate_value;
                *gate_value = value / (1.0 + torch28_exp(-value)) * *up_value;
            }
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.gate.row1"),
                    &gate[cfg.n_ff..2 * cfg.n_ff],
                );
                debug_bits(
                    &format!("batch.l{layer}.up.row1"),
                    &up[cfg.n_ff..2 * cfg.n_ff],
                );
                debug_bits(
                    &format!("batch.l{layer}.ffn_act.row1"),
                    &gate[cfg.n_ff..2 * cfg.n_ff],
                );
            }
            matmul(&weights.w_down, &gate, &mut down, None);
            dump_stage("batch_ffn_down", 0, layer, &down);
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.ffn_down.row1"),
                    &down[cfg.n_embd..2 * cfg.n_embd],
                );
            }
            for (value, residual) in down.iter_mut().zip(x.iter()) {
                *value += *residual;
            }
            x.copy_from_slice(&down);
            dump_stage("batch_full_hidden", 0, layer, &x);
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() {
                debug_bits(
                    &format!("batch.layer{layer}.hidden.row1"),
                    &x[cfg.n_embd..2 * cfg.n_embd],
                );
            }
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.hidden.row1"),
                    &x[cfg.n_embd..2 * cfg.n_embd],
                );
            }
        }
        let mut hidden = vec![0.0f32; rows_len * cfg.n_embd];
        for row in 0..rows_len {
            torch_rms_norm_with_eps(
                &x[row * cfg.n_embd..(row + 1) * cfg.n_embd],
                &self.model.output_norm,
                &mut hidden[row * cfg.n_embd..(row + 1) * cfg.n_embd],
                cfg.eps,
            );
        }
        dump_stage("batch_hidden", 0, 0, &hidden);
        #[cfg(feature = "parity-trace")]
        if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() {
            debug_bits("batch.final.row1", &hidden[cfg.n_embd..2 * cfg.n_embd]);
        }
        Ok(hidden)
    }

    /// Final normalized hidden state of the most recent step.
    pub fn last_hidden(&self) -> &[f32] {
        &self.x
    }

    /// Normalized hidden state (after `output_norm.weight`).
    pub fn normalized_hidden(&self) -> Result<Vec<f32>, String> {
        if self.model.output_norm.len() != self.model.config.n_embd {
            return Err("dotstts output norm shape mismatch".into());
        }
        let mut out = vec![0.0; self.model.config.n_embd];
        torch_rms_norm_with_eps(
            &self.x,
            &self.model.output_norm,
            &mut out,
            self.model.config.eps,
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo};
    use std::collections::BTreeMap;

    struct Source {
        metadata: BTreeMap<String, MetaValue>,
        tensors: BTreeMap<String, (TensorInfo, Vec<u8>)>,
    }

    impl TensorSource for Source {
        fn metadata(&self, name: &str) -> Option<&MetaValue> {
            self.metadata.get(name)
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.tensors.get(name).map(|tensor| &tensor.0)
        }
        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.tensors.get(name).map(|tensor| tensor.1.as_slice())
        }
    }

    impl Source {
        fn insert(&mut self, name: &str, dims: &[u64], dtype: GGMLType, values: &[f32]) {
            let bytes = match dtype {
                GGMLType::BF16 => values
                    .iter()
                    .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
                    .collect(),
                GGMLType::F16 => values
                    .iter()
                    .flat_map(|value| half::f16::from_f32(*value).to_le_bytes())
                    .collect(),
                GGMLType::F32 => values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                GGMLType::Q8_0 => {
                    let mut bytes = Vec::new();
                    for block in values.chunks_exact(32) {
                        // Fixture values are exact multiples of this binary scale.
                        bytes.extend_from_slice(&half::f16::from_f32(1.0 / 128.0).to_le_bytes());
                        bytes.extend(block.iter().map(|value| (value * 128.0) as i8 as u8));
                    }
                    bytes
                }
                _ => unreachable!(),
            };
            self.tensors.insert(
                name.into(),
                (
                    TensorInfo {
                        name: name.into(),
                        dims: dims.into(),
                        ggml_type: dtype,
                        offset: 0,
                    },
                    bytes,
                ),
            );
        }
    }

    fn tiny_source(dtype: GGMLType) -> Source {
        let mut source = Source {
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        source.metadata.insert(
            "general.architecture".into(),
            MetaValue::String("qwen2".into()),
        );
        for (key, value) in [
            ("embedding_length", 64),
            ("block_count", 2),
            ("attention.head_count", 4),
            ("attention.head_count_kv", 2),
            ("feed_forward_length", 96),
            ("context_length", 8),
        ] {
            source
                .metadata
                .insert(format!("qwen2.{key}"), MetaValue::Uint32(value));
        }
        source.metadata.insert(
            "qwen2.attention.layer_norm_rms_epsilon".into(),
            MetaValue::Float32(1e-6),
        );
        source
            .metadata
            .insert("qwen2.rope.freq_base".into(), MetaValue::Float32(10_000.0));
        source.metadata.insert(
            "tokenizer.ggml.tokens".into(),
            MetaValue::Array(
                MetaValueType::String,
                (0..8).map(|i| MetaValue::String(format!("t{i}"))).collect(),
            ),
        );
        source.insert("output_norm.weight", &[64], GGMLType::F32, &[1.0; 64]);
        let embedding: Vec<_> = (0..64 * 8)
            .map(|i| ((i * 13 % 61) as f32 - 30.0) / 128.0)
            .collect();
        source.insert("token_embd.weight", &[64, 8], dtype, &embedding);
        for layer in 0..2 {
            for suffix in ["attn_norm.weight", "ffn_norm.weight"] {
                source.insert(
                    &format!("blk.{layer}.{suffix}"),
                    &[64],
                    GGMLType::F32,
                    &[1.0; 64],
                );
            }
            for (suffix, width) in [
                ("attn_q.bias", 64),
                ("attn_k.bias", 32),
                ("attn_v.bias", 32),
            ] {
                let values: Vec<_> = (0..width).map(|i| (i as f32 - 16.0) / 1024.0).collect();
                source.insert(
                    &format!("blk.{layer}.{suffix}"),
                    &[width],
                    GGMLType::F32,
                    &values,
                );
            }
            for (index, (suffix, n_in, n_out)) in [
                ("attn_q.weight", 64, 64),
                ("attn_k.weight", 64, 32),
                ("attn_v.weight", 64, 32),
                ("attn_output.weight", 64, 64),
                ("ffn_gate.weight", 64, 96),
                ("ffn_up.weight", 64, 96),
                ("ffn_down.weight", 96, 64),
            ]
            .into_iter()
            .enumerate()
            {
                let values: Vec<_> = (0..n_in * n_out)
                    .map(|i| {
                        let row = i / n_in;
                        (((i * 7 + row * 3 + index * 5 + layer * 11) % 17) as f32 - 8.0) / 128.0
                    })
                    .collect();
                source.insert(
                    &format!("blk.{layer}.{suffix}"),
                    &[n_in as u64, n_out as u64],
                    dtype,
                    &values,
                );
            }
        }
        source
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                actual.is_finite() && (actual - expected).abs() <= tolerance,
                "element {i}: {actual} != {expected}, tolerance {tolerance}"
            );
        }
    }

    #[test]
    fn native_weights_prefill_decode_match_sequential_and_threads() {
        let embedding: Vec<_> = (0..64).map(|i| (i as f32 - 31.0) / 64.0).collect();
        let rows = [
            LlmInputRow::Token(2),
            LlmInputRow::Embedding(&embedding),
            LlmInputRow::Token(5),
        ];
        let mut dense_reference = Vec::new();
        for dtype in [GGMLType::BF16, GGMLType::F32, GGMLType::Q8_0, GGMLType::F16] {
            let mut threaded_reference = Vec::new();
            for threads in [1, 3] {
                let source = Arc::new(tiny_source(dtype));
                let source_lifetime = Arc::downgrade(&source);
                let model =
                    DotsLlm::from_source(source.clone(), Arc::new(ComputePool::new(threads)))
                        .unwrap();
                drop(source);
                assert!(source_lifetime.upgrade().is_some());
                assert_eq!(model.token_embedding.ggml_type, dtype);
                for layer in &model.layers {
                    for weight in [
                        &layer.wq,
                        &layer.wk,
                        &layer.wv,
                        &layer.wo,
                        &layer.w_gate,
                        &layer.w_up,
                        &layer.w_down,
                    ] {
                        assert_eq!(weight.ggml_type, dtype);
                    }
                }
                let mut batch = model.new_session().unwrap();
                let q8_buffer = batch.input_q8.as_ptr();
                let scale_buffer = batch.input_scales.as_ptr();
                let mut actual = batch.prefill_rows(&rows).unwrap();
                actual.extend(batch.step_row(LlmInputRow::Token(7)).unwrap());
                assert_eq!(batch.position(), 4);
                assert_eq!(batch.last_hidden(), &actual[3 * 64..]);
                assert_eq!(batch.input_q8.as_ptr(), q8_buffer);
                assert_eq!(batch.input_scales.as_ptr(), scale_buffer);
                let mut sequential = model.new_session().unwrap();
                let mut expected = Vec::new();
                for row in [
                    LlmInputRow::Token(2),
                    LlmInputRow::Embedding(&embedding),
                    LlmInputRow::Token(5),
                    LlmInputRow::Token(7),
                ] {
                    expected.extend(sequential.step_row(row).unwrap());
                }
                assert_close(&actual, &expected, 0.0);
                if threads == 1 {
                    threaded_reference = actual.clone();
                }
                assert_close(&actual, &threaded_reference, 1e-6);
                if dtype == GGMLType::BF16 && threads == 1 {
                    dense_reference = actual.clone();
                }
                // Q8 weights are exactly represented; activation quantization adds bounded error.
                assert_close(
                    &actual,
                    &dense_reference,
                    if dtype == GGMLType::Q8_0 { 0.025 } else { 1e-6 },
                );
                drop(sequential);
                drop(batch);
                drop(model);
                assert!(source_lifetime.upgrade().is_none());
            }
        }
    }

    #[test]
    fn weight_loading_rejects_malformed_matrices_before_forward() {
        for name in [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.1.ffn_down.weight",
        ] {
            for dtype in [GGMLType::BF16, GGMLType::Q8_0] {
                for bad_shape in [false, true] {
                    let mut source = tiny_source(dtype);
                    let (info, bytes) = source.tensors.get_mut(name).unwrap();
                    if bad_shape {
                        info.dims[0] -= 1;
                    } else {
                        bytes.pop();
                    }
                    let error =
                        DotsLlm::from_source(Arc::new(source), Arc::new(ComputePool::new(1)))
                            .err()
                            .expect("malformed matrix accepted");
                    assert!(error.contains(name), "{error}");
                }
            }
        }
    }

    #[test]
    fn invalid_rows_and_exhausted_context_preserve_session() {
        let model = DotsLlm::from_source(
            Arc::new(tiny_source(GGMLType::Q8_0)),
            Arc::new(ComputePool::new(1)),
        )
        .unwrap();
        let mut session = model.new_session().unwrap();
        assert!(session.prefill_rows(&[LlmInputRow::Token(8)]).is_err());
        assert_eq!(session.position(), 0);
        session.prefill_rows(&[LlmInputRow::Token(1)]).unwrap();
        let hidden = session.last_hidden().to_vec();
        assert!(session.step_row(LlmInputRow::Token(8)).is_err());
        assert!(session.step_row(LlmInputRow::Embedding(&[1.0])).is_err());
        assert!(session.prefill_rows(&[LlmInputRow::Token(1)]).is_err());
        assert_eq!(session.last_hidden(), hidden);
        assert_eq!(session.position(), 1);
        for _ in 1..8 {
            session.step_row(LlmInputRow::Token(1)).unwrap();
        }
        let hidden = session.last_hidden().to_vec();
        assert!(session.step_row(LlmInputRow::Token(2)).is_err());
        assert_eq!(session.last_hidden(), hidden);
        assert_eq!(session.position(), 8);
    }

    #[test]
    fn strided_causal_attention_matches_independent_scalar_reference() {
        let (rows, keys, head_dim, q_stride, cache_stride) = (3, 6, 4, 12, 8);
        let q: Vec<_> = (0..rows * q_stride)
            .map(|i| (i as f32 - 13.0) / 32.0)
            .collect();
        let mut k: Vec<_> = (0..keys * cache_stride)
            .map(|i| (i as f32 - 23.0) / 64.0)
            .collect();
        let mut v: Vec<_> = (0..keys * cache_stride)
            .map(|i| ((i * 7 % 19) as f32 - 9.0) / 8.0)
            .collect();
        // The last key is future padding for every query and must never be read.
        k[5 * cache_stride..].fill(f32::NAN);
        v[5 * cache_stride..].fill(f32::NAN);
        for head_offset in [0, 4] {
            let mut output = vec![1234.5; rows * q_stride];
            attention_head(
                &q[4..],
                rows,
                q_stride,
                &k,
                &v,
                keys,
                2,
                head_offset,
                head_dim,
                cache_stride,
                &mut output[8..],
                q_stride,
                &mut vec![0.0; keys + 2 * head_dim],
            )
            .unwrap();
            for row in 0..rows {
                let scores: Vec<_> = (0..row + 3)
                    .map(|key| {
                        (0..head_dim)
                            .map(|d| {
                                q[row * q_stride + 4 + d] as f64
                                    * k[key * cache_stride + head_offset + d] as f64
                            })
                            .sum::<f64>()
                            / (head_dim as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<_> = scores.iter().map(|score| (score - max).exp()).collect();
                let sum: f64 = exp.iter().sum();
                let expected: Vec<_> = (0..head_dim)
                    .map(|d| {
                        exp.iter()
                            .enumerate()
                            .map(|(key, p)| {
                                p / sum * v[key * cache_stride + head_offset + d] as f64
                            })
                            .sum::<f64>() as f32
                    })
                    .collect();
                assert_close(
                    &output[row * q_stride + 8..(row + 1) * q_stride],
                    &expected,
                    2e-6,
                );
                assert_eq!(&output[row * q_stride..row * q_stride + 8], &[1234.5; 8]);
            }
        }
        assert!(attention_head(
            &q,
            1,
            q_stride,
            &k,
            &v,
            keys,
            6,
            0,
            head_dim,
            cache_stride,
            &mut [0.0; 4],
            4,
            &mut [0.0; 14]
        )
        .is_err());
        assert!(attention_head(
            &q,
            1,
            q_stride,
            &k,
            &v,
            keys,
            0,
            usize::MAX,
            head_dim,
            cache_stride,
            &mut [0.0; 4],
            4,
            &mut [0.0; 14]
        )
        .is_err());
    }
}
