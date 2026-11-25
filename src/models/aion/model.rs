use crate::compute::linear::{LinearBinding, LinearExecutor};
use crate::compute::ComputePolicy;
use crate::core::loader::model_config_from_source;
use crate::core::scratchpad::{ExecutionScratchpad, KvCache, KvFormat};
use crate::core::tensor::{load_f32_tensor, load_weight_any, GGMLType, MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::traits::ModelConfig;
use crate::models::llama::trunk::forward::attention_head_f16;
use crate::ops::attention_value::attention_value_reduce;
use crate::ops::kernel::Weight;
use crate::ops::{
    dot_f32, f32_slice_to_f16, rms_norm, rms_norm_inplace, silu_mul_inplace, softmax_inplace,
    vec_add_into,
};
use std::sync::Arc;

const ARCH: &str = "aion_instruct";
const MAX_CONTEXT: usize = 32_768;
const FORMATS: &[GGMLType] = &[
    GGMLType::F32,
    GGMLType::F16,
    GGMLType::BF16,
    GGMLType::Q8_0,
    GGMLType::Q4_0,
    GGMLType::Q4_1,
    GGMLType::Q4K,
    GGMLType::Q5K,
    GGMLType::Q6K,
];

struct Layer<'a> {
    sliding: bool,
    norms: [Vec<f32>; 4],
    // Q, K, V, attention output, gate, up, down.
    weights: [Weight<'a>; 7],
}

pub struct AionModel<'a> {
    config: ModelConfig,
    window: usize,
    rope: [Vec<f32>; 2],
    yarn_scale: f32,
    layers: Vec<Layer<'a>>,
    embedding: Weight<'a>,
    output_norm: Vec<f32>,
    pool: Arc<ComputePool>,
    pub(crate) stop_ids: Vec<u32>,
}

fn matrix<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    input: usize,
    output: usize,
) -> Result<Weight<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [input as u64, output as u64] {
        return Err(format!(
            "{name} expected [{input}, {output}], got {:?}",
            info.dims
        ));
    }
    load_weight_any(source, name, input, output, Some(FORMATS))
}

fn yarn_frequencies(
    dim: usize,
    base: f32,
    factor: f32,
    original: f32,
    fast: f32,
    slow: f32,
) -> Vec<f32> {
    let correction = |rotations: f64| {
        dim as f64 * (f64::from(original) / (rotations * std::f64::consts::TAU)).ln()
            / (2.0 * f64::from(base).ln())
    };
    let low = correction(f64::from(fast)).floor().max(0.0) as f32;
    let high = correction(f64::from(slow)).ceil().min((dim - 1) as f64) as f32;
    let span = if low == high { 0.001 } else { high - low };
    (0..dim / 2)
        .map(|i| {
            let ramp = ((i as f32 - low) / span).clamp(0.0, 1.0);
            let frequency = base.powf(2.0 * i as f32 / dim as f32);
            (1.0 - ramp) / frequency + ramp / (factor * frequency)
        })
        .collect()
}

impl<'a> AionModel<'a> {
    pub fn load(source: &'a dyn TensorSource, pool: Arc<ComputePool>) -> Result<Self, String> {
        if source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            != Some(ARCH)
        {
            return Err("Expected aion_instruct model".into());
        }
        let config = model_config_from_source(source)?;
        let dim = config.n_embd_head;
        // Only the published checkpoint geometry is supported. Check it before
        // multiplying dimensions or allocating buffers owned by the session.
        if [
            config.n_embd,
            config.n_ff,
            config.n_head,
            config.n_head_kv,
            dim,
            config.n_layer,
            config.vocab_size,
            config.n_ctx,
        ] != [2048, 8192, 24, 8, 128, 28, 200_029, MAX_CONTEXT]
            || !config.norm_eps.is_finite()
            || config.norm_eps <= 0.0
        {
            return Err("Invalid Aion Instruct 2.3B geometry or RMSNorm epsilon".into());
        }
        let array = |key: &str| {
            source
                .metadata(key)
                .and_then(MetaValue::to_arr)
                .ok_or_else(|| format!("Missing or invalid metadata: {key}"))
        };
        let schedule = array("aion_instruct.attention.layer_types")?;
        if schedule.len() != config.n_layer {
            return Err("Aion attention schedule length must match layer count".into());
        }
        let window = source
            .metadata("aion_instruct.attention.sliding_window")
            .and_then(MetaValue::to_u64)
            .filter(|&n| n > 0 && n <= MAX_CONTEXT as u64)
            .ok_or("Invalid Aion sliding window")? as usize;
        let scalar = |key: &str, minimum: f32| {
            source
                .metadata(key)
                .and_then(MetaValue::to_f64)
                .map(|n| n as f32)
                .filter(|n| n.is_finite() && *n >= minimum)
                .ok_or_else(|| format!("Missing or invalid metadata: {key}"))
        };
        let base = scalar("aion_instruct.rope.freq_base", 2.0)?;
        let local_base = scalar("aion_instruct.rope.sliding_freq_base", 2.0)?;
        let factor = scalar("aion_instruct.rope.scaling.factor", 1.0)?;
        let original = scalar("aion_instruct.rope.scaling.original_context_length", 1.0)?;
        let fast = scalar(
            "aion_instruct.rope.scaling.yarn_beta_fast",
            f32::MIN_POSITIVE,
        )?;
        let slow = scalar(
            "aion_instruct.rope.scaling.yarn_beta_slow",
            f32::MIN_POSITIVE,
        )?;
        if fast < slow {
            return Err("Invalid Aion YaRN beta range".into());
        }
        let stop_ids = array("aion_instruct.stop_token_ids")?
            .iter()
            .map(|id| {
                id.to_u64()
                    .filter(|&id| id < config.vocab_size as u64)
                    .map(|id| id as u32)
                    .ok_or("Invalid Aion stop token")
            })
            .collect::<Result<Vec<_>, _>>()?;
        if stop_ids.is_empty() {
            return Err("Aion stop tokens are missing".into());
        }
        let width = config.n_embd;
        let q = config.n_head * dim;
        let kv = config.n_head_kv * dim;
        let ff = config.n_ff;
        let embedding = matrix(source, "token_embd.weight", width, config.vocab_size)?;
        let output_norm = load_f32_tensor(source, "output_norm.weight", &[width as u64])?;
        if source.tensor_info("output.weight").is_some() {
            return Err("Aion expects a tied output head and no output.weight".into());
        }
        let mut layers = Vec::with_capacity(config.n_layer);
        for (i, kind) in schedule.iter().enumerate() {
            let sliding = match kind.to_string_val() {
                Some("sliding_attention") => true,
                Some("full_attention") => false,
                _ => return Err(format!("Invalid Aion attention type at layer {i}")),
            };
            let norm =
                |name: &str, n| load_f32_tensor(source, &format!("blk.{i}.{name}.weight"), &[n]);
            let weight = |name: &str, input, output| {
                matrix(source, &format!("blk.{i}.{name}.weight"), input, output)
            };
            layers.push(Layer {
                sliding,
                norms: [
                    norm("attn_norm", width as u64)?,
                    norm("attn_q_norm", dim as u64)?,
                    norm("attn_k_norm", dim as u64)?,
                    norm("ffn_norm", width as u64)?,
                ],
                weights: [
                    weight("attn_q", width, q)?,
                    weight("attn_k", width, kv)?,
                    weight("attn_v", width, kv)?,
                    weight("attn_output", q, width)?,
                    weight("ffn_gate", width, ff)?,
                    weight("ffn_up", width, ff)?,
                    weight("ffn_down", ff, width)?,
                ],
            });
        }
        Ok(Self {
            config,
            window,
            rope: [
                yarn_frequencies(dim, base, factor, original, fast, slow),
                (0..dim / 2)
                    .map(|i| 1.0 / local_base.powf(2.0 * i as f32 / dim as f32))
                    .collect(),
            ],
            yarn_scale: (1.0 + 0.1 * f64::from(factor).ln()) as f32,
            layers,
            embedding,
            output_norm,
            pool,
            stop_ids,
        })
    }

    pub fn context_length(&self) -> usize {
        self.config.n_ctx
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub(crate) fn validate_tokens(&self, tokens: &[u32]) -> Result<(), String> {
        if tokens.is_empty()
            || tokens
                .iter()
                .any(|&id| id as usize >= self.config.vocab_size)
        {
            return Err("Aion requires a nonempty prompt with valid token IDs".into());
        }
        Ok(())
    }
}

struct LayerCache {
    kv: KvCache,
    /// A mirrored ring keeps each sliding window contiguous after wraparound.
    /// One full prefill chunk can be appended before the first query reads it.
    ring: Option<usize>,
}

enum AttentionCache {
    Host(Vec<LayerCache>),
    #[cfg(feature = "vulkan")]
    Device(Vec<crate::vulkan::ops::CachedAttention>),
}

impl LayerCache {
    fn new(
        format: KvFormat,
        capacity: usize,
        width: usize,
        window: Option<usize>,
        batch: usize,
    ) -> Self {
        let ring = window.map(|w| w + batch - 1).filter(|&n| 2 * n < capacity);
        let rows = ring.map_or(capacity, |n| 2 * n);
        let kv = match format {
            KvFormat::F32 => KvCache::new_f32(1, rows, width),
            KvFormat::F16 => KvCache::new_f16(1, rows, width),
        };
        Self { kv, ring }
    }

    fn append(&mut self, position: usize, width: usize, keys: &[f32], values: &[f32]) {
        for (row, (k, v)) in keys
            .chunks_exact(width)
            .zip(values.chunks_exact(width))
            .enumerate()
        {
            let position = position + row;
            let slot = self.slot(position);
            let mut write = |slot| {
                let range = slot * width..(slot + 1) * width;
                match &mut self.kv {
                    KvCache::F32(cache) => {
                        cache.k[range.clone()].copy_from_slice(k);
                        cache.v[range].copy_from_slice(v);
                    }
                    KvCache::F16(cache) => {
                        f32_slice_to_f16(k, &mut cache.k[range.clone()]);
                        f32_slice_to_f16(v, &mut cache.v[range]);
                    }
                }
            };
            write(slot);
            if let Some(ring) = self.ring {
                write(slot + ring);
            }
        }
    }

    fn slot(&self, position: usize) -> usize {
        self.ring.map_or(position, |n| position % n)
    }
}

pub struct AionSession<'m, 'w> {
    model: &'m AionModel<'w>,
    linear: LinearExecutor<'m, 'w>,
    output: LinearExecutor<'m, 'w>,
    cache: AttentionCache,
    policy: ComputePolicy,
    scratch: ExecutionScratchpad,
    max_rows: usize,
    capacity: usize,
    position: usize,
}

impl<'m, 'w> AionSession<'m, 'w> {
    pub fn new(
        model: &'m AionModel<'w>,
        policy: ComputePolicy,
        format: KvFormat,
        capacity: usize,
        max_rows: usize,
    ) -> Result<Self, String> {
        if capacity == 0 || capacity > model.context_length() || max_rows == 0 {
            return Err("Invalid Aion context capacity or prefill batch size".into());
        }
        policy.check_build()?;
        if policy == ComputePolicy::Vulkan && crate::ops::scalar_mode() {
            return Err("Aion cannot use strict Vulkan with scalar reference arithmetic".into());
        }
        let max_rows = max_rows.min(capacity);
        let linear = LinearExecutor::new(
            policy,
            model
                .layers
                .iter()
                .flat_map(|l| l.weights.iter().map(LinearBinding::weight_only))
                .collect(),
            max_rows,
            model.pool.clone(),
        )
        .map_err(|e| e.to_string())?;
        // The tied vocabulary projection only runs on the last row. Keeping
        // it separate avoids allocating vocab × prefill_batch scratch on GPU.
        let output = LinearExecutor::new(
            policy,
            vec![LinearBinding::weight_only(&model.embedding)],
            1,
            model.pool.clone(),
        )
        .map_err(|e| e.to_string())?;
        let cfg = &model.config;
        let kv_width = cfg.n_head_kv * cfg.n_embd_head;
        let host_cache = || {
            AttentionCache::Host(
                model
                    .layers
                    .iter()
                    .map(|l| {
                        LayerCache::new(
                            format,
                            capacity,
                            kv_width,
                            l.sliding.then_some(model.window),
                            max_rows,
                        )
                    })
                    .collect(),
            )
        };
        #[cfg(feature = "vulkan")]
        let cache = if let Some(context) = linear.vulkan_context() {
            let caches = model
                .layers
                .iter()
                .map(|layer| {
                    crate::vulkan::ops::CachedAttention::new(
                        context,
                        capacity,
                        max_rows,
                        layer.sliding.then_some(model.window),
                        cfg.n_head,
                        cfg.n_head_kv,
                        cfg.n_embd_head,
                        format,
                    )
                })
                .collect::<Result<Vec<_>, _>>();
            match caches {
                Ok(caches) => AttentionCache::Device(caches),
                Err(e) if policy == ComputePolicy::Auto => {
                    log::info!("Aion attention using CPU: {e}");
                    host_cache()
                }
                Err(e) => return Err(e.to_string()),
            }
        } else {
            host_cache()
        };
        #[cfg(not(feature = "vulkan"))]
        let cache = host_cache();
        Ok(Self {
            model,
            linear,
            output,
            cache,
            policy,
            scratch: ExecutionScratchpad::new_batched(
                cfg.n_embd,
                cfg.n_head * cfg.n_embd_head,
                kv_width,
                cfg.n_ff,
                cfg.vocab_size,
                model.pool.n_threads(),
                capacity,
                max_rows,
            ),
            max_rows,
            capacity,
            position: 0,
        })
    }

    /// Number of tokens committed by complete prefill or decode chunks.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Append tokens and return the last row of logits. If a later chunk
    /// fails, `position()` identifies the first token to retry. A lost
    /// device requires replaying the prompt in a fresh session.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<&mut [f32], String> {
        self.model.validate_tokens(tokens)?;
        if tokens.len() > self.capacity - self.position {
            return Err(format!(
                "Aion input exceeds context capacity {}",
                self.capacity
            ));
        }
        let mut chunks = tokens.chunks(self.max_rows).peekable();
        while let Some(chunk) = chunks.next() {
            self.forward_chunk(chunk, chunks.peek().is_none())
                .map_err(|e| e.to_string())?;
            self.position += chunk.len();
        }
        Ok(&mut self.scratch.logits)
    }

    fn forward_chunk(
        &mut self,
        tokens: &[u32],
        project: bool,
    ) -> Result<(), crate::compute::ComputeError> {
        let model = self.model;
        let cfg = &model.config;
        let width = cfg.n_embd;
        let dim = cfg.n_embd_head;
        let q_width = cfg.n_head * dim;
        let kv_width = cfg.n_head_kv * dim;
        let rows = tokens.len();
        let s = &mut self.scratch;
        let x = &mut s.x[..rows * width];
        let norm = &mut s.normed[..rows * width];
        let q = &mut s.q[..rows * q_width];
        let k = &mut s.k_new[..rows * kv_width];
        let v = &mut s.v_new[..rows * kv_width];
        let a = &mut s.attn_out[..rows * q_width];
        let gate = &mut s.gate_buf[..rows * cfg.n_ff];
        let up = &mut s.up_buf[..rows * cfg.n_ff];
        let residual = &mut s.attn_proj[..rows * width];
        for (&token, row) in tokens.iter().zip(x.chunks_exact_mut(width)) {
            model.embedding.embedding_lookup(token, row);
        }
        let mut cos = vec![0.0; dim / 2];
        let mut sin = vec![0.0; dim / 2];
        for (i, layer) in model.layers.iter().enumerate() {
            let ids = |index| self.linear.id(i * layer.weights.len() + index);
            let [wq, wk, wv, wo, wg, wu, wd] = [
                ids(0)?,
                ids(1)?,
                ids(2)?,
                ids(3)?,
                ids(4)?,
                ids(5)?,
                ids(6)?,
            ];
            let [attn_norm, q_norm, k_norm, ffn_norm] = &layer.norms;
            normalize_rows(x, norm, attn_norm, cfg.norm_eps);
            self.linear.run_group([wq, wk, wv], norm, rows, [q, k, v])?;
            #[cfg(feature = "parity-trace")]
            trace_projections(
                Some(i),
                self.position,
                rows,
                norm,
                [
                    ("attn_q", q.as_ref()),
                    ("attn_k", k.as_ref()),
                    ("attn_v", v.as_ref()),
                ],
            );
            for (row, (q, k)) in q
                .chunks_exact_mut(q_width)
                .zip(k.chunks_exact_mut(kv_width))
                .enumerate()
            {
                let position = (self.position + row) as f32;
                let scale = if layer.sliding { 1.0 } else { model.yarn_scale };
                for ((cos, sin), &frequency) in cos
                    .iter_mut()
                    .zip(&mut sin)
                    .zip(&model.rope[usize::from(layer.sliding)])
                {
                    let (s, c) = (position * frequency).sin_cos();
                    *cos = c * scale;
                    *sin = s * scale;
                }
                for (values, norm) in [(q, q_norm), (k, k_norm)] {
                    for head in values.chunks_exact_mut(dim) {
                        rms_norm_inplace(head, norm, cfg.norm_eps);
                    }
                    crate::ops::rope::neox::rope_neox_f32_with_table(values, dim, &cos, &sin);
                }
            }
            let backend = match &mut self.cache {
                AttentionCache::Host(caches) => {
                    let cache = &mut caches[i];
                    cache.append(self.position, kv_width, k, v);
                    attention(
                        model,
                        layer,
                        cache,
                        self.position,
                        q,
                        a,
                        &mut s.scores,
                        s.score_stride,
                    );
                    crate::compute::UsedBackend::Cpu
                }
                #[cfg(feature = "vulkan")]
                AttentionCache::Device(caches) => {
                    caches[i].forward(self.position, q, k, v, a)?;
                    crate::compute::UsedBackend::Vulkan
                }
            };
            self.policy.trace("aion_attention_kv", backend, rows);
            self.linear.run(wo, a, rows, residual)?;
            #[cfg(feature = "parity-trace")]
            trace_projections(
                Some(i),
                self.position,
                rows,
                a,
                [("attn_output", residual.as_ref())],
            );
            vec_add_into(residual, x);
            normalize_rows(x, norm, ffn_norm, cfg.norm_eps);
            self.linear.run_group([wg, wu], norm, rows, [gate, up])?;
            #[cfg(feature = "parity-trace")]
            trace_projections(
                Some(i),
                self.position,
                rows,
                norm,
                [("ffn_gate", gate.as_ref()), ("ffn_up", up.as_ref())],
            );
            silu_mul_inplace(gate, up);
            self.linear.run(wd, up, rows, residual)?;
            #[cfg(feature = "parity-trace")]
            trace_projections(
                Some(i),
                self.position,
                rows,
                up,
                [("ffn_down", residual.as_ref())],
            );
            vec_add_into(residual, x);
        }
        if project {
            rms_norm(
                &x[(rows - 1) * width..],
                &model.output_norm,
                &mut norm[..width],
                cfg.norm_eps,
            );
            self.output
                .run(self.output.id(0)?, &norm[..width], 1, &mut s.logits)?;
            #[cfg(feature = "parity-trace")]
            trace_projections(
                None,
                self.position + rows - 1,
                1,
                &norm[..width],
                [("token_embd", s.logits.as_slice())],
            );
        }
        Ok(())
    }
}

#[cfg(feature = "parity-trace")]
fn trace_projections<const N: usize>(
    layer: Option<usize>,
    position: usize,
    rows: usize,
    input: &[f32],
    outputs: [(&str, &[f32]); N],
) {
    if std::env::var_os("RMI_PARITY_TRACE").is_none() {
        return;
    }
    for (name, output) in outputs {
        for (kind, values) in [("input", input), ("output", output)] {
            let width = values.len() / rows;
            let row = &values[(rows - 1) * width..];
            crate::parity_trace::report(crate::parity_trace::checkpoint_at(
                &format!("aion.{name}.{kind}"),
                layer,
                Some(position + rows - 1),
                &[width],
                row,
            ));
        }
    }
}

fn normalize_rows(input: &[f32], output: &mut [f32], weight: &[f32], eps: f32) {
    for (input, output) in input
        .chunks_exact(weight.len())
        .zip(output.chunks_exact_mut(weight.len()))
    {
        rms_norm(input, weight, output, eps);
    }
}

fn visible_start(position: usize, window: Option<usize>) -> usize {
    window.map_or(0, |window| (position + 1).saturating_sub(window))
}

#[allow(clippy::too_many_arguments)]
fn attention(
    model: &AionModel<'_>,
    layer: &Layer<'_>,
    cache: &LayerCache,
    position: usize,
    query: &[f32],
    output: &mut [f32],
    scores: &mut [f32],
    stride: usize,
) {
    let cfg = &model.config;
    let dim = cfg.n_embd_head;
    let kv_width = cfg.n_head_kv * dim;
    let group = cfg.n_head / cfg.n_head_kv;
    let window = layer.sliding.then_some(model.window);
    let scale = 1.0 / (dim as f32).sqrt();
    let heads = query.len() / dim;
    let out_ptr = output.as_mut_ptr();
    let score_ptr = scores.as_mut_ptr();
    model.pool.compute(move |worker, threads| {
        let first = worker * heads / threads;
        let last = (worker + 1) * heads / threads;
        // SAFETY: workers own disjoint output head ranges and score rows.
        // The pool joins all workers before returning to the caller.
        let out = unsafe {
            std::slice::from_raw_parts_mut(out_ptr.add(first * dim), (last - first) * dim)
        };
        let scores =
            unsafe { std::slice::from_raw_parts_mut(score_ptr.add(worker * stride), stride) };
        for (head, out) in (first..last).zip(out.chunks_exact_mut(dim)) {
            let pos = position + head / cfg.n_head;
            let start = visible_start(pos, window);
            let count = pos + 1 - start;
            let offset = cache.slot(start) * kv_width;
            let kv_head = (head % cfg.n_head) / group * dim;
            let q = &query[head * dim..(head + 1) * dim];
            match &cache.kv {
                KvCache::F16(kv) => attention_head_f16(
                    q,
                    out,
                    &kv.k,
                    &kv.v,
                    offset + kv_head,
                    kv_width,
                    count,
                    scale,
                    0.0,
                ),
                KvCache::F32(kv) => {
                    let scores = &mut scores[..count];
                    for (i, score) in scores.iter_mut().enumerate() {
                        let k = offset + kv_head + i * kv_width;
                        *score = dot_f32(q, &kv.k[k..k + dim], dim) * scale;
                    }
                    softmax_inplace(scores);
                    attention_value_reduce(
                        &kv.v, scores, out, offset, kv_width, kv_head, count, dim,
                    );
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn device_cache_preserves_causality_and_grouped_heads_across_wraps() {
        let context = ComputePolicy::Vulkan.context().unwrap().unwrap();
        let (capacity, batch, heads, kv_heads, dim) = (28, 5, 4, 2, 128);
        for window in [Some(3), None] {
            for format in [KvFormat::F32, KvFormat::F16] {
                let mut cache = crate::vulkan::ops::CachedAttention::new(
                    context, capacity, batch, window, heads, kv_heads, dim, format,
                )
                .unwrap();
                let stored = |value: f32| match format {
                    KvFormat::F16 => crate::ops::f16_to_f32(crate::ops::f32_to_f16(value)),
                    KvFormat::F32 => value,
                };
                let key = |p: usize, h: usize| ((p * 5 + h * 3) % 17) as f32 / 7.0;
                let value = |p: usize, h: usize, d: usize| ((p + h * 13 + d) % 19) as f32 / 11.0;
                let mut pos = 0;
                for rows in [5, 1, 4, 5, 3, 5, 5] {
                    let mut q = vec![0.0; rows * heads * dim];
                    let mut k = vec![0.0; rows * kv_heads * dim];
                    let mut v = vec![0.0; k.len()];
                    for r in 0..rows {
                        for h in 0..heads {
                            q[(r * heads + h) * dim] = (h + 1) as f32;
                        }
                        for h in 0..kv_heads {
                            let base = (r * kv_heads + h) * dim;
                            k[base] = key(pos + r, h);
                            for d in 0..dim {
                                v[base + d] = value(pos + r, h, d);
                            }
                        }
                    }
                    let mut output = vec![f32::NAN; q.len()];
                    if pos == 10 {
                        // A later layer may have failed after this layer wrote KV.
                        // Replacing that chunk must preserve its visible prefix.
                        cache
                            .forward(pos, &q, &k, &vec![0.0; v.len()], &mut output)
                            .unwrap();
                    }
                    cache.forward(pos, &q, &k, &v, &mut output).unwrap();
                    for r in 0..rows {
                        let end = pos + r + 1;
                        let start = window.map_or(0, |w| end.saturating_sub(w));
                        for h in 0..heads {
                            let kv_head = h / (heads / kv_heads);
                            let weights: Vec<_> = (start..end)
                                .map(|p| {
                                    (f64::from(stored(key(p, kv_head))) * (h + 1) as f64
                                        / (dim as f64).sqrt())
                                    .exp()
                                })
                                .collect();
                            let sum: f64 = weights.iter().sum();
                            for d in [0, 63, 127] {
                                let expected = (start..end)
                                    .zip(&weights)
                                    .map(|(p, w)| w * f64::from(stored(value(p, kv_head, d))))
                                    .sum::<f64>()
                                    / sum;
                                let actual = output[(r * heads + h) * dim + d];
                                assert!((f64::from(actual) - expected).abs() < 2e-5,
                                    "{format:?} window={window:?} pos={pos} row={r} head={h} dim={d}: {actual} vs {expected}");
                            }
                        }
                    }
                    pos += rows;
                }
            }
        }
    }

    #[test]
    fn sliding_window_keeps_current_token_and_wraps_for_chunks() {
        let mut cache = LayerCache::new(KvFormat::F32, 20, 1, Some(3), 2);
        for pos in (0..20).step_by(2) {
            let chunk = [pos as f32, (pos + 1) as f32];
            cache.append(pos, 1, &chunk, &chunk);
            for absolute in pos..pos + 2 {
                let start = visible_start(absolute, Some(3));
                let offset = cache.slot(start);
                let KvCache::F32(kv) = &cache.kv else {
                    unreachable!()
                };
                let want: Vec<_> = (start..=absolute).map(|n| n as f32).collect();
                assert_eq!(&kv.k[offset..offset + want.len()], want);
            }
        }
    }
}
