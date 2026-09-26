use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::core::scratchpad::{KvArch, KvCache, KvFormat, KvState};
use crate::core::tensor::{load_f32_tensor, GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::ops::kernel::{QuantizedTensor, Weight};
use crate::ops::quant::BlockQ8K;

use super::config::YuE2Config;
use super::nar::Mt19937;
use super::protocol::{
    SamplingConfig, YuE2Protocol, ABC_END, CODEC_OFFSET, CODEC_SIZE, EOD, MUSIC_END,
};

pub(super) struct YuE2Weight {
    fast: Weight<'static>,
    bf16: Option<&'static [u8]>,
    n_in: usize,
    n_out: usize,
}

impl YuE2Weight {
    fn load(
        source: &dyn TensorSource,
        name: &str,
        n_in: usize,
        n_out: usize,
    ) -> Result<Self, String> {
        let bytes = source
            .tensor_slice(name)
            .ok_or_else(|| format!("Missing tensor data: {name}"))?;
        let bytes: &'static [u8] = unsafe { std::mem::transmute(bytes) };
        Ok(Self {
            fast: Weight::from_quantized(QuantizedTensor::from_bytes(
                bytes,
                GGMLType::BF16,
                n_in,
                n_out,
            )),
            bf16: Some(bytes),
            n_in,
            n_out,
        })
    }

    #[cfg(test)]
    pub(super) fn from_f32(values: Vec<f32>, n_in: usize, n_out: usize) -> Self {
        Self {
            fast: Weight::from_quantized(QuantizedTensor::F32 {
                data: values,
                n_in,
                n_out,
            }),
            bf16: None,
            n_in,
            n_out,
        }
    }

    pub(super) fn embedding_lookup(&self, token_id: u32, output: &mut [f32]) {
        if let Some(bytes) = self.bf16 {
            let start = token_id as usize * self.n_in * 2;
            for (index, value) in output.iter_mut().enumerate() {
                let offset = start + index * 2;
                *value =
                    crate::ops::bf16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
            }
            return;
        }
        self.fast.embedding_lookup(token_id, output);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn matmul(
        &self,
        input: &[f32],
        output: &mut [f32],
        pool: &ComputePool,
        q8: &mut [u8],
        scales: &mut [f32],
        q8k: &mut [BlockQ8K],
    ) {
        debug_assert_eq!(input.len(), self.n_in);
        debug_assert_eq!(output.len(), self.n_out);
        if self.bf16.is_some() {
            self.matmul_bf16(input, None, output, pool);
            return;
        }
        self.fast
            .quantize_and_matmul_with_scratch(input, q8k, q8, scales, output, pool);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn matmul_bias(
        &self,
        input: &[f32],
        bias: &[f32],
        output: &mut [f32],
        pool: &ComputePool,
        q8: &mut [u8],
        scales: &mut [f32],
        q8k: &mut [BlockQ8K],
    ) {
        debug_assert_eq!(bias.len(), self.n_out);
        if self.bf16.is_some() {
            self.matmul_bf16(input, Some(bias), output, pool);
        } else {
            self.matmul(input, output, pool, q8, scales, q8k);
            for (value, &bias) in output.iter_mut().zip(bias) {
                *value += bias;
            }
        }
    }

    fn matmul_bf16(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
        pool: &ComputePool,
    ) {
        let bytes = self.bf16.unwrap();
        let output_ptr = output.as_mut_ptr();
        pool.compute(|thread, threads| {
            let (start, end) =
                crate::ops::kernel::bf16::BF16Kernel::row_range(self.n_out, thread, threads);
            if start == end {
                return;
            }
            let output =
                unsafe { std::slice::from_raw_parts_mut(output_ptr.add(start), end - start) };
            torch_bf16_matmul_rows(bytes, input, bias, output, self.n_in, start);
        });
    }
}

pub(super) fn torch_bf16_matmul_rows(
    bytes: &[u8],
    input: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
) {
    for (offset_row, value) in output.iter_mut().enumerate() {
        let row = row_start + offset_row;
        let byte_start = row * n_in * 2;
        let sum = crate::ops::dot_bf16_f32(input, &bytes[byte_start..byte_start + n_in * 2], n_in);
        let sum = bias.map_or(sum, |bias| sum + bias[row]);
        *value = half::bf16::from_f32(sum).to_f32();
    }
}

pub(super) struct YuE2AttentionWeights {
    pub(super) norm: Vec<f32>,
    pub(super) q_norm: Vec<f32>,
    pub(super) k_norm: Vec<f32>,
    pub(super) q: YuE2Weight,
    pub(super) k: YuE2Weight,
    pub(super) v: YuE2Weight,
    pub(super) output: YuE2Weight,
}

pub(super) struct YuE2MlpWeights {
    pub(super) norm: Vec<f32>,
    pub(super) gate: YuE2Weight,
    pub(super) up: YuE2Weight,
    pub(super) down: YuE2Weight,
}

pub(super) struct YuE2LayerWeights {
    pub(super) ar_attention: YuE2AttentionWeights,
    pub(super) ar_mlp: YuE2MlpWeights,
    pub(super) nar_attention: YuE2AttentionWeights,
    pub(super) nar_mlp: YuE2MlpWeights,
}

pub(super) struct YuE2AuxWeights {
    pub(super) llm2vae: YuE2Weight,
    pub(super) llm2vae_bias: Vec<f32>,
    pub(super) vae2llm: YuE2Weight,
    pub(super) vae2llm_bias: Vec<f32>,
    pub(super) time_in: YuE2Weight,
    pub(super) time_in_bias: Vec<f32>,
    pub(super) time_out: YuE2Weight,
    pub(super) time_out_bias: Vec<f32>,
    pub(super) latent_position: YuE2Weight,
}

pub struct YuE2Model {
    source: Option<Arc<dyn TensorSource>>,
    tokenizer: Arc<BPETokenizer>,
    pub(super) pool: Arc<ComputePool>,
    pub(super) config: YuE2Config,
    pub(super) layers: Vec<YuE2LayerWeights>,
    pub(super) token_embedding: YuE2Weight,
    pub(super) final_norm: Vec<f32>,
    lm_head: YuE2Weight,
    pub(super) aux: YuE2AuxWeights,
}

impl fmt::Debug for YuE2Model {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("YuE2Model")
            .field("config", &self.config)
            .field("layers", &self.layers.len())
            .finish_non_exhaustive()
    }
}

impl YuE2Model {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        tokenizer: Arc<BPETokenizer>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let config = YuE2Config::from_source(source.as_ref())?;
        YuE2Protocol::from_source(source.as_ref())?;
        if tokenizer.vocab_size() != config.vocab {
            return Err(format!(
                "YuE2 vocabulary size {} does not match tokenizer vocab {}",
                config.vocab,
                tokenizer.vocab_size()
            ));
        }
        preflight(source.as_ref(), &config)?;

        let q_width = config.q_heads * config.head_dim;
        let kv_width = config.kv_heads * config.head_dim;
        let mut layers = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let base = format!("model.layers.{layer}");
            layers.push(YuE2LayerWeights {
                ar_attention: load_attention(
                    source.as_ref(),
                    &base,
                    "input_layernorm",
                    "self_attn",
                    &config,
                )?,
                ar_mlp: load_mlp(
                    source.as_ref(),
                    &base,
                    "post_attention_layernorm",
                    "mlp",
                    &config,
                )?,
                nar_attention: load_attention(
                    source.as_ref(),
                    &base,
                    "nar_input_layernorm",
                    "nar_self_attn",
                    &config,
                )?,
                nar_mlp: load_mlp(
                    source.as_ref(),
                    &base,
                    "nar_pre_mlp_layernorm",
                    "nar_mlp",
                    &config,
                )?,
            });
        }

        let model = Self {
            token_embedding: YuE2Weight::load(
                source.as_ref(),
                "model.embed_tokens.weight",
                config.hidden,
                config.vocab,
            )?,
            final_norm: load_f32_tensor(
                source.as_ref(),
                "model.norm.weight",
                &[config.hidden as u64],
            )?,
            lm_head: YuE2Weight::load(
                source.as_ref(),
                "lm_head.weight",
                config.hidden,
                config.vocab,
            )?,
            aux: YuE2AuxWeights {
                llm2vae: YuE2Weight::load(
                    source.as_ref(),
                    "llm2vae.weight",
                    config.hidden,
                    config.latent_channels,
                )?,
                llm2vae_bias: load_f32_tensor(
                    source.as_ref(),
                    "llm2vae.bias",
                    &[config.latent_channels as u64],
                )?,
                vae2llm: YuE2Weight::load(
                    source.as_ref(),
                    "vae2llm.weight",
                    config.latent_channels,
                    config.hidden,
                )?,
                vae2llm_bias: load_f32_tensor(
                    source.as_ref(),
                    "vae2llm.bias",
                    &[config.hidden as u64],
                )?,
                time_in: YuE2Weight::load(
                    source.as_ref(),
                    "time_embedder.mlp.0.weight",
                    256,
                    config.hidden,
                )?,
                time_in_bias: load_f32_tensor(
                    source.as_ref(),
                    "time_embedder.mlp.0.bias",
                    &[config.hidden as u64],
                )?,
                time_out: YuE2Weight::load(
                    source.as_ref(),
                    "time_embedder.mlp.2.weight",
                    config.hidden,
                    config.hidden,
                )?,
                time_out_bias: load_f32_tensor(
                    source.as_ref(),
                    "time_embedder.mlp.2.bias",
                    &[config.hidden as u64],
                )?,
                latent_position: YuE2Weight::load(
                    source.as_ref(),
                    "latent_pos_embed.pe",
                    config.hidden,
                    config.context,
                )?,
            },
            source: Some(Arc::clone(&source)),
            tokenizer,
            pool,
            config,
            layers,
        };
        debug_assert_eq!(model.layers[0].ar_attention.q.n_out, q_width);
        debug_assert_eq!(model.layers[0].ar_attention.k.n_out, kv_width);
        Ok(model)
    }

    pub fn config(&self) -> &YuE2Config {
        &self.config
    }

    pub fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }

    pub fn generate_abc(
        &self,
        prefix: &[u32],
        sampling: SamplingConfig,
        seed: u64,
    ) -> Result<Vec<u32>, String> {
        self.generate_phase(prefix, sampling, seed, Phase::Abc)
    }

    pub fn generate_semantic(
        &self,
        prefix: &[u32],
        sampling: SamplingConfig,
        seed: u64,
    ) -> Result<Vec<u32>, String> {
        self.generate_phase(prefix, sampling, seed, Phase::Semantic)
    }

    fn generate_phase(
        &self,
        prefix: &[u32],
        sampling: SamplingConfig,
        seed: u64,
        phase: Phase,
    ) -> Result<Vec<u32>, String> {
        sampling.validate()?;
        let capacity = prefix
            .len()
            .checked_add(sampling.max_tokens)
            .ok_or("YuE2 generation length overflow")?;
        if prefix.is_empty() || capacity > self.config.context {
            return Err(format!(
                "YuE2 prefix plus generation requires {capacity} positions, maximum is {}",
                self.config.context
            ));
        }
        let mut session = YuE2ArSession::new(self, capacity)?;
        let mut logits = session.prefill(prefix)?.to_vec();
        let mut rng = Mt19937::new(seed);
        let mut output = Vec::with_capacity(sampling.max_tokens);
        for step in 0..sampling.max_tokens {
            let token = sample_phase_token(&logits, &output, sampling, step, phase, &mut rng)?;
            if token == phase.end_token() {
                break;
            }
            output.push(token);
            if step + 1 < sampling.max_tokens {
                logits = session.prefill(&[token])?.to_vec();
            }
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::token_ids(phase.trace_name(), &output));
        Ok(output)
    }

    #[cfg(test)]
    pub(super) fn tiny_for_test(tokenizer: Arc<BPETokenizer>, pool: Arc<ComputePool>) -> Self {
        let config = YuE2Config {
            hidden: 4,
            layers: 1,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            ffn: 8,
            vocab: 16,
            context: 8,
            rms_eps: 0.000001,
            rope_base: 1_000_000.0,
            latent_channels: 2,
            timestep_shift: 1.0,
        };
        fn weight(n_in: usize, n_out: usize, salt: usize) -> YuE2Weight {
            let values = (0..n_in * n_out)
                .map(|index| (((index + salt) % 11) as f32 - 5.0) * 0.025)
                .collect();
            YuE2Weight::from_f32(values, n_in, n_out)
        }
        fn attention(config: &YuE2Config, salt: usize) -> YuE2AttentionWeights {
            YuE2AttentionWeights {
                norm: vec![1.0; config.hidden],
                q_norm: vec![1.0; config.head_dim],
                k_norm: vec![1.0; config.head_dim],
                q: weight(config.hidden, config.q_heads * config.head_dim, salt),
                k: weight(config.hidden, config.kv_heads * config.head_dim, salt + 1),
                v: weight(config.hidden, config.kv_heads * config.head_dim, salt + 2),
                output: weight(config.q_heads * config.head_dim, config.hidden, salt + 3),
            }
        }
        fn mlp(config: &YuE2Config, salt: usize) -> YuE2MlpWeights {
            YuE2MlpWeights {
                norm: vec![1.0; config.hidden],
                gate: weight(config.hidden, config.ffn, salt),
                up: weight(config.hidden, config.ffn, salt + 1),
                down: weight(config.ffn, config.hidden, salt + 2),
            }
        }
        let layer = YuE2LayerWeights {
            ar_attention: attention(&config, 1),
            ar_mlp: mlp(&config, 5),
            nar_attention: attention(&config, 9),
            nar_mlp: mlp(&config, 13),
        };
        Self {
            source: None,
            tokenizer,
            pool,
            token_embedding: weight(config.hidden, config.vocab, 17),
            final_norm: vec![1.0; config.hidden],
            lm_head: weight(config.hidden, config.vocab, 19),
            aux: YuE2AuxWeights {
                llm2vae: weight(config.hidden, config.latent_channels, 21),
                llm2vae_bias: vec![0.0; config.latent_channels],
                vae2llm: weight(config.latent_channels, config.hidden, 22),
                vae2llm_bias: vec![0.0; config.hidden],
                time_in: weight(256, config.hidden, 23),
                time_in_bias: vec![0.0; config.hidden],
                time_out: weight(config.hidden, config.hidden, 24),
                time_out_bias: vec![0.0; config.hidden],
                latent_position: weight(config.hidden, config.context, 25),
            },
            config,
            layers: vec![layer],
        }
    }
}

pub struct YuE2ArSession<'model> {
    model: &'model YuE2Model,
    kv: KvState,
    x: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attention: Vec<f32>,
    projected: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    logits: Vec<f32>,
    scores: Vec<f32>,
    q8: Vec<u8>,
    scales: Vec<f32>,
    q8k: Vec<BlockQ8K>,
}

impl<'model> YuE2ArSession<'model> {
    pub fn new(model: &'model YuE2Model, capacity: usize) -> Result<Self, String> {
        if capacity == 0 || capacity > model.config.context {
            return Err(format!(
                "YuE2 session capacity {capacity} must be within 1..={}",
                model.config.context
            ));
        }
        let config = &model.config;
        let q_width = checked_product(config.q_heads, config.head_dim, "YuE2 query width")?;
        let kv_width = checked_product(config.kv_heads, config.head_dim, "YuE2 KV width")?;
        let _ = checked_product(
            checked_product(config.layers, capacity, "YuE2 KV rows")?,
            kv_width,
            "YuE2 KV values",
        )?;
        let max_input = config.ffn.max(config.hidden).max(q_width);
        let arch = Arc::new(KvArch::new(
            config.layers,
            config.kv_heads,
            config.head_dim,
            config.head_dim,
            config.context,
        ));
        Ok(Self {
            model,
            kv: KvState::new(arch, KvFormat::F32, capacity),
            x: vec![0.0; config.hidden],
            normed: vec![0.0; config.hidden],
            q: vec![0.0; q_width],
            k: vec![0.0; kv_width],
            v: vec![0.0; kv_width],
            attention: vec![0.0; q_width],
            projected: vec![0.0; config.hidden],
            gate: vec![0.0; config.ffn],
            up: vec![0.0; config.ffn],
            down: vec![0.0; config.hidden],
            logits: vec![0.0; config.vocab],
            scores: vec![0.0; capacity],
            q8: vec![0; max_input],
            scales: vec![0.0; max_input.div_ceil(32)],
            q8k: vec![
                BlockQ8K {
                    d: 0.0,
                    qs: [0; 256],
                    bsums: [0; 16],
                };
                max_input.div_ceil(256)
            ],
        })
    }

    pub fn position(&self) -> usize {
        self.kv.seq_len
    }

    pub fn prefill(&mut self, token_ids: &[u32]) -> Result<&[f32], String> {
        if token_ids.is_empty() {
            return Err("YuE2 prefill requires at least one token".into());
        }
        let end = self
            .kv
            .seq_len
            .checked_add(token_ids.len())
            .ok_or("YuE2 prefill length overflow")?;
        if end > self.kv.capacity {
            return Err(format!(
                "YuE2 prefill requires {end} positions; session capacity is {}",
                self.kv.capacity
            ));
        }
        validate_token_ids(token_ids, self.model.config.vocab)?;
        for &token in token_ids {
            let position = self.kv.seq_len;
            self.forward_token(token, position)?;
            self.kv.seq_len += 1;
        }
        self.kv.update_access();
        Ok(&self.logits)
    }

    fn forward_token(&mut self, token_id: u32, position: usize) -> Result<(), String> {
        let config = &self.model.config;
        self.model
            .token_embedding
            .embedding_lookup(token_id, &mut self.x);
        trace(
            "yue2.ar.embedding",
            None,
            position,
            &[config.hidden],
            &self.x,
        );

        let q_width = config.q_heads * config.head_dim;
        let kv_width = config.kv_heads * config.head_dim;
        let group_size = config.q_heads / config.kv_heads;
        let scale = (config.head_dim as f32).sqrt().recip();
        for (layer_index, layer) in self.model.layers.iter().enumerate() {
            rms_norm(
                &self.x,
                &layer.ar_attention.norm,
                &mut self.normed,
                config.rms_eps,
            );
            trace(
                "yue2.ar.attn_norm",
                Some(layer_index),
                position,
                &[config.hidden],
                &self.normed,
            );
            layer.ar_attention.q.matmul(
                &self.normed,
                &mut self.q,
                &self.model.pool,
                &mut self.q8,
                &mut self.scales,
                &mut self.q8k,
            );
            layer.ar_attention.k.matmul(
                &self.normed,
                &mut self.k,
                &self.model.pool,
                &mut self.q8,
                &mut self.scales,
                &mut self.q8k,
            );
            layer.ar_attention.v.matmul(
                &self.normed,
                &mut self.v,
                &self.model.pool,
                &mut self.q8,
                &mut self.scales,
                &mut self.q8k,
            );
            rms_norm_heads(
                &mut self.q,
                &layer.ar_attention.q_norm,
                config.head_dim,
                config.rms_eps,
            );
            rms_norm_heads(
                &mut self.k,
                &layer.ar_attention.k_norm,
                config.head_dim,
                config.rms_eps,
            );
            trace(
                "yue2.ar.q",
                Some(layer_index),
                position,
                &[config.q_heads, config.head_dim],
                &self.q,
            );
            trace(
                "yue2.ar.k",
                Some(layer_index),
                position,
                &[config.kv_heads, config.head_dim],
                &self.k,
            );
            trace(
                "yue2.ar.v",
                Some(layer_index),
                position,
                &[config.kv_heads, config.head_dim],
                &self.v,
            );
            rope(&mut self.q, position, config.head_dim, config.rope_base);
            rope(&mut self.k, position, config.head_dim, config.rope_base);
            trace(
                "yue2.ar.rope_q",
                Some(layer_index),
                position,
                &[config.q_heads, config.head_dim],
                &self.q,
            );
            trace(
                "yue2.ar.rope_k",
                Some(layer_index),
                position,
                &[config.kv_heads, config.head_dim],
                &self.k,
            );

            let (key_cache, value_cache) = match &mut self.kv.cache {
                KvCache::F32(cache) => (&mut cache.k, &mut cache.v),
                KvCache::F16(_) => return Err("YuE2 AR requires an F32 KV cache".into()),
            };
            let layer_base = layer_index * self.kv.capacity * kv_width;
            let row_base = layer_base + position * kv_width;
            key_cache[row_base..row_base + kv_width].copy_from_slice(&self.k);
            value_cache[row_base..row_base + kv_width].copy_from_slice(&self.v);
            self.attention.fill(0.0);
            for head in 0..config.q_heads {
                let kv_head = head / group_size;
                let q_start = head * config.head_dim;
                let kv_start = kv_head * config.head_dim;
                attention_head(
                    &self.q[q_start..q_start + config.head_dim],
                    &key_cache[layer_base..],
                    &value_cache[layer_base..],
                    kv_width,
                    kv_start,
                    &mut self.scores[..=position],
                    &mut self.attention[q_start..q_start + config.head_dim],
                    scale,
                );
            }
            trace(
                "yue2.ar.attn",
                Some(layer_index),
                position,
                &[q_width],
                &self.attention,
            );
            layer.ar_attention.output.matmul(
                &self.attention,
                &mut self.projected,
                &self.model.pool,
                &mut self.q8,
                &mut self.scales,
                &mut self.q8k,
            );
            trace(
                "yue2.ar.attn_output",
                Some(layer_index),
                position,
                &[config.hidden],
                &self.projected,
            );
            add_in_place(&mut self.x, &self.projected);
            trace(
                "yue2.ar.attn_residual",
                Some(layer_index),
                position,
                &[config.hidden],
                &self.x,
            );

            rms_norm(
                &self.x,
                &layer.ar_mlp.norm,
                &mut self.normed,
                config.rms_eps,
            );
            trace(
                "yue2.ar.ffn_norm",
                Some(layer_index),
                position,
                &[config.hidden],
                &self.normed,
            );
            layer.ar_mlp.gate.matmul(
                &self.normed,
                &mut self.gate,
                &self.model.pool,
                &mut self.q8,
                &mut self.scales,
                &mut self.q8k,
            );
            trace(
                "yue2.ar.ffn_gate",
                Some(layer_index),
                position,
                &[config.ffn],
                &self.gate,
            );
            layer.ar_mlp.up.matmul(
                &self.normed,
                &mut self.up,
                &self.model.pool,
                &mut self.q8,
                &mut self.scales,
                &mut self.q8k,
            );
            trace(
                "yue2.ar.ffn_up",
                Some(layer_index),
                position,
                &[config.ffn],
                &self.up,
            );
            for (gate, &up) in self.gate.iter_mut().zip(&self.up) {
                let activated = half::bf16::from_f32(silu(*gate)).to_f32();
                *gate = half::bf16::from_f32(activated * up).to_f32();
            }
            layer.ar_mlp.down.matmul(
                &self.gate,
                &mut self.down,
                &self.model.pool,
                &mut self.q8,
                &mut self.scales,
                &mut self.q8k,
            );
            trace(
                "yue2.ar.ffn_down",
                Some(layer_index),
                position,
                &[config.hidden],
                &self.down,
            );
            add_in_place(&mut self.x, &self.down);
            trace(
                "yue2.ar.ffn_residual",
                Some(layer_index),
                position,
                &[config.hidden],
                &self.x,
            );
        }

        rms_norm(
            &self.x,
            &self.model.final_norm,
            &mut self.normed,
            config.rms_eps,
        );
        trace(
            "yue2.ar.final_norm",
            None,
            position,
            &[config.hidden],
            &self.normed,
        );
        self.model.lm_head.matmul(
            &self.normed,
            &mut self.logits,
            &self.model.pool,
            &mut self.q8,
            &mut self.scales,
            &mut self.q8k,
        );
        trace(
            "yue2.ar.logits",
            None,
            position,
            &[config.vocab],
            &self.logits,
        );
        if self.logits.iter().any(|value| !value.is_finite()) {
            return Err("YuE2 AR produced non-finite logits".into());
        }
        Ok(())
    }
}

fn preflight(source: &dyn TensorSource, config: &YuE2Config) -> Result<(), String> {
    for (name, dims) in [
        (
            "model.embed_tokens.weight",
            vec![config.hidden, config.vocab],
        ),
        ("model.norm.weight", vec![config.hidden]),
        ("lm_head.weight", vec![config.hidden, config.vocab]),
        (
            "llm2vae.weight",
            vec![config.hidden, config.latent_channels],
        ),
        ("llm2vae.bias", vec![config.latent_channels]),
        (
            "vae2llm.weight",
            vec![config.latent_channels, config.hidden],
        ),
        ("vae2llm.bias", vec![config.hidden]),
        ("time_embedder.mlp.0.weight", vec![256, config.hidden]),
        ("time_embedder.mlp.0.bias", vec![config.hidden]),
        (
            "time_embedder.mlp.2.weight",
            vec![config.hidden, config.hidden],
        ),
        ("time_embedder.mlp.2.bias", vec![config.hidden]),
        ("latent_pos_embed.pe", vec![config.hidden, config.context]),
    ] {
        require_bf16(source, name, &dims)?;
    }
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    for layer in 0..config.layers {
        let base = format!("model.layers.{layer}");
        for (suffix, dims) in [
            ("input_layernorm.weight", vec![config.hidden]),
            ("self_attn.q_proj.weight", vec![config.hidden, q_width]),
            ("self_attn.k_proj.weight", vec![config.hidden, kv_width]),
            ("self_attn.v_proj.weight", vec![config.hidden, kv_width]),
            ("self_attn.o_proj.weight", vec![q_width, config.hidden]),
            ("self_attn.q_norm.weight", vec![config.head_dim]),
            ("self_attn.k_norm.weight", vec![config.head_dim]),
            ("post_attention_layernorm.weight", vec![config.hidden]),
            ("mlp.gate_proj.weight", vec![config.hidden, config.ffn]),
            ("mlp.up_proj.weight", vec![config.hidden, config.ffn]),
            ("mlp.down_proj.weight", vec![config.ffn, config.hidden]),
            ("nar_input_layernorm.weight", vec![config.hidden]),
            ("nar_self_attn.q_proj.weight", vec![config.hidden, q_width]),
            ("nar_self_attn.k_proj.weight", vec![config.hidden, kv_width]),
            ("nar_self_attn.v_proj.weight", vec![config.hidden, kv_width]),
            ("nar_self_attn.o_proj.weight", vec![q_width, config.hidden]),
            ("nar_self_attn.q_norm.weight", vec![config.head_dim]),
            ("nar_self_attn.k_norm.weight", vec![config.head_dim]),
            ("nar_pre_mlp_layernorm.weight", vec![config.hidden]),
            ("nar_mlp.gate_proj.weight", vec![config.hidden, config.ffn]),
            ("nar_mlp.up_proj.weight", vec![config.hidden, config.ffn]),
            ("nar_mlp.down_proj.weight", vec![config.ffn, config.hidden]),
        ] {
            require_bf16(source, &format!("{base}.{suffix}"), &dims)?;
        }
    }
    Ok(())
}

fn require_bf16(source: &dyn TensorSource, name: &str, dims: &[usize]) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let expected = dims.iter().map(|&value| value as u64).collect::<Vec<_>>();
    if info.dims != expected {
        return Err(format!(
            "Invalid tensor {name} shape {:?}; expected {expected:?}",
            info.dims
        ));
    }
    if info.ggml_type != GGMLType::BF16 {
        return Err(format!(
            "Invalid tensor {name} type {:?}; expected BF16",
            info.ggml_type
        ));
    }
    info.checked_nbytes()
        .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?;
    Ok(())
}

fn load_attention(
    source: &dyn TensorSource,
    base: &str,
    norm: &str,
    attention: &str,
    config: &YuE2Config,
) -> Result<YuE2AttentionWeights, String> {
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    Ok(YuE2AttentionWeights {
        norm: load_f32_tensor(
            source,
            &format!("{base}.{norm}.weight"),
            &[config.hidden as u64],
        )?,
        q_norm: load_f32_tensor(
            source,
            &format!("{base}.{attention}.q_norm.weight"),
            &[config.head_dim as u64],
        )?,
        k_norm: load_f32_tensor(
            source,
            &format!("{base}.{attention}.k_norm.weight"),
            &[config.head_dim as u64],
        )?,
        q: YuE2Weight::load(
            source,
            &format!("{base}.{attention}.q_proj.weight"),
            config.hidden,
            q_width,
        )?,
        k: YuE2Weight::load(
            source,
            &format!("{base}.{attention}.k_proj.weight"),
            config.hidden,
            kv_width,
        )?,
        v: YuE2Weight::load(
            source,
            &format!("{base}.{attention}.v_proj.weight"),
            config.hidden,
            kv_width,
        )?,
        output: YuE2Weight::load(
            source,
            &format!("{base}.{attention}.o_proj.weight"),
            q_width,
            config.hidden,
        )?,
    })
}

fn load_mlp(
    source: &dyn TensorSource,
    base: &str,
    norm: &str,
    mlp: &str,
    config: &YuE2Config,
) -> Result<YuE2MlpWeights, String> {
    Ok(YuE2MlpWeights {
        norm: load_f32_tensor(
            source,
            &format!("{base}.{norm}.weight"),
            &[config.hidden as u64],
        )?,
        gate: YuE2Weight::load(
            source,
            &format!("{base}.{mlp}.gate_proj.weight"),
            config.hidden,
            config.ffn,
        )?,
        up: YuE2Weight::load(
            source,
            &format!("{base}.{mlp}.up_proj.weight"),
            config.hidden,
            config.ffn,
        )?,
        down: YuE2Weight::load(
            source,
            &format!("{base}.{mlp}.down_proj.weight"),
            config.ffn,
            config.hidden,
        )?,
    })
}

#[derive(Clone, Copy)]
pub(super) enum Phase {
    Abc,
    Semantic,
}

impl Phase {
    fn end_token(self) -> u32 {
        match self {
            Self::Abc => ABC_END,
            Self::Semantic => MUSIC_END,
        }
    }

    #[cfg(feature = "parity-trace")]
    fn trace_name(self) -> &'static str {
        match self {
            Self::Abc => "yue2.abc.generated_ids",
            Self::Semantic => "yue2.semantic.generated_ids",
        }
    }

    fn allows(self, token: usize) -> bool {
        match self {
            Self::Abc => token < EOD as usize || token == ABC_END as usize,
            Self::Semantic => {
                (CODEC_OFFSET as usize..CODEC_OFFSET as usize + CODEC_SIZE).contains(&token)
                    || token == MUSIC_END as usize
            }
        }
    }
}

pub(super) fn sample_phase_token(
    logits: &[f32],
    history: &[u32],
    sampling: SamplingConfig,
    step: usize,
    phase: Phase,
    rng: &mut Mt19937,
) -> Result<u32, String> {
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("YuE2 sampling received non-finite logits".into());
    }
    let mut scores = logits.to_vec();
    for (token, score) in scores.iter_mut().enumerate() {
        if !phase.allows(token)
            || (step < sampling.min_tokens && token == phase.end_token() as usize)
        {
            *score = f32::NEG_INFINITY;
        }
    }
    let recent = &history[history.len().saturating_sub(sampling.penalty_window)..];
    let mut counts = HashMap::new();
    for &token in recent {
        *counts.entry(token).or_insert(0) += 1;
    }
    crate::ops::sampling::apply_repetition_penalty(
        &mut scores,
        &counts,
        sampling.repetition_penalty,
    );
    if sampling.temperature == 0.0 {
        return argmax_finite(&scores).map(|token| token as u32);
    }
    if sampling.temperature != 1.0 {
        for score in &mut scores {
            *score /= sampling.temperature;
        }
    }
    let mut candidates = scores
        .into_iter()
        .enumerate()
        .filter(|(_, score)| score.is_finite())
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    candidates.truncate(sampling.top_k.min(candidates.len()));
    if candidates.is_empty() {
        return Err("YuE2 sampling has no allowed tokens".into());
    }
    let max = candidates[0].1;
    let mut total = 0.0f32;
    for (_, value) in &mut candidates {
        *value = (*value - max).exp();
        total += *value;
    }
    if sampling.top_p < 1.0 {
        let mut cumulative = 0.0f32;
        let mut keep = 0;
        for &(_, probability) in &candidates {
            cumulative += probability / total;
            keep += 1;
            if cumulative >= sampling.top_p {
                break;
            }
        }
        candidates.truncate(keep.max(1));
        total = candidates.iter().map(|(_, probability)| probability).sum();
    }
    Ok(torch_multinomial(&mut candidates, total, logits.len(), rng) as u32)
}

fn torch_multinomial(
    candidates: &mut [(usize, f32)],
    total: f32,
    vocabulary: usize,
    rng: &mut Mt19937,
) -> usize {
    candidates.sort_unstable_by_key(|&(token, _)| token);
    let mut candidate = 0;
    let mut best = (0, f32::NEG_INFINITY);
    for token in 0..vocabulary {
        let random = (u64::from(rng.next()) << 32) | u64::from(rng.next());
        if candidate == candidates.len() || candidates[candidate].0 != token {
            continue;
        }
        let uniform = (random & ((1u64 << 53) - 1)) as f64 * (1.0 / (1u64 << 53) as f64);
        let exponential = -(-uniform).ln_1p() as f32;
        let score = (candidates[candidate].1 / total) / exponential;
        if score > best.1 {
            best = (token, score);
        }
        candidate += 1;
    }
    best.0
}

fn argmax_finite(values: &[f32]) -> Result<usize, String> {
    values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|left, right| left.1.total_cmp(right.1).then(right.0.cmp(&left.0)))
        .map(|(index, _)| index)
        .ok_or_else(|| "YuE2 sampling has no allowed tokens".into())
}

fn checked_product(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{name} overflow"))
}

fn validate_token_ids(token_ids: &[u32], vocab: usize) -> Result<(), String> {
    if let Some(token) = token_ids
        .iter()
        .copied()
        .find(|&token| token as usize >= vocab)
    {
        return Err(format!(
            "YuE2 token ID {token} is outside vocabulary {vocab}"
        ));
    }
    Ok(())
}

pub(super) fn rms_norm(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let sum = input.iter().map(|v| v * v).sum::<f32>();
    let scale = (sum / input.len() as f32 + eps).sqrt().recip();
    let scale = half::bf16::from_f32(scale).to_f32();
    for ((output, &input), &weight) in output.iter_mut().zip(input).zip(weight) {
        let scaled = half::bf16::from_f32(input * scale).to_f32();
        *output = half::bf16::from_f32(scaled * weight).to_f32();
    }
}

pub(super) fn rms_norm_heads(values: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
    let mut normalized = vec![0.0; head_dim];
    for head in values.chunks_exact_mut(head_dim) {
        rms_norm(head, weight, &mut normalized, eps);
        head.copy_from_slice(&normalized);
    }
}

pub(super) fn rope(values: &mut [f32], position: usize, head_dim: usize, base: f32) {
    let (mut cos, mut sin) =
        crate::ops::rope::rope_sin_cos_sleef_table_with_threads(&[position], head_dim, base, 1);
    for value in cos.iter_mut().chain(&mut sin) {
        *value = half::bf16::from_f32(*value).to_f32();
    }
    crate::ops::rope::rope_neox_inplace_with_table(values, head_dim, &cos, &sin);
}

pub(super) fn dot(left: &[f32], right: &[f32]) -> f32 {
    crate::ops::dot_f32(left, right, left.len())
}

pub(super) fn attention_head(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    kv_width: usize,
    kv_start: usize,
    scores: &mut [f32],
    output: &mut [f32],
    scale: f32,
) {
    for (cached_position, score) in scores.iter_mut().enumerate() {
        let key_start = cached_position * kv_width + kv_start;
        *score = dot(query, &key_cache[key_start..key_start + query.len()]) * scale;
    }
    if scores.len() <= 512 {
        let inverse_sum = softmax(scores);
        for (dimension, value) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for cached_position in 0..scores.len() {
                sum += scores[cached_position]
                    * value_cache[cached_position * kv_width + kv_start + dimension];
            }
            *value = half::bf16::from_f32(sum * inverse_sum).to_f32();
        }
        return;
    }

    output.fill(0.0);
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0f32;
    for start in (0..scores.len()).step_by(512) {
        let end = scores.len().min(start + 512);
        let block = &mut scores[start..end];
        let block_max = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let next_max = running_max.max(block_max);
        let block_sum = softmax_exp_sum(block, next_max);
        let rescale = (running_max - next_max).exp();
        running_sum = rescale.mul_add(running_sum, block_sum);
        if start > 0 {
            for value in output.iter_mut() {
                *value *= rescale;
            }
        }
        for (dimension, value) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for offset in 0..block.len() {
                sum += block[offset]
                    * value_cache[(start + offset) * kv_width + kv_start + dimension];
            }
            *value += sum;
        }
        running_max = next_max;
    }
    let inverse_sum = running_sum.recip();
    for value in output.iter_mut() {
        *value = half::bf16::from_f32(*value * inverse_sum).to_f32();
    }
}

pub(super) fn softmax(values: &mut [f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    softmax_exp_sum(values, max).recip()
}

fn softmax_exp_sum(values: &mut [f32], max: f32) -> f32 {
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        let exponential = (*value - max).exp();
        sum += exponential;
        *value = half::bf16::from_f32(exponential).to_f32();
    }
    sum
}

pub(super) fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

pub(super) fn add_in_place(target: &mut [f32], update: &[f32]) {
    for (target, &update) in target.iter_mut().zip(update) {
        *target = half::bf16::from_f32(*target + update).to_f32();
    }
}

fn trace(name: &str, layer: Option<usize>, step: usize, shape: &[usize], values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint_at(
        name,
        layer,
        Some(step),
        shape,
        values,
    ));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, layer, step, shape, values);
}
