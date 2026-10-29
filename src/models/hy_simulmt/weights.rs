use super::{check_tensor, validate_decoder, HyConfig};
use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::ops::kernel::{QuantizedTensor, Weight};
use std::sync::Arc;

pub struct HyDecoder {
    pub(super) config: HyConfig,
    pub(super) layers: Vec<HyLayer>,
    pub(super) embedding: Weight<'static>,
    pub(super) output: Weight<'static>,
    pub(super) output_norm: Vec<f32>,
    pub(super) pool: Arc<ComputePool>,
    tokenizer: Arc<BPETokenizer>,
    _source: Arc<dyn TensorSource>,
}

pub(super) struct HyLayer {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub q: Weight<'static>,
    pub k: Weight<'static>,
    pub v: Weight<'static>,
    pub out: Weight<'static>,
    pub ffn: HyFfn,
}

pub(super) enum HyFfn {
    Dense(Expert),
    Moe(HyMoe),
}

pub(super) struct Expert {
    pub gate: Weight<'static>,
    pub up: Weight<'static>,
    pub down: Weight<'static>,
}

pub(super) struct HyMoe {
    pub router: Vec<f32>,
    pub bias: Vec<f32>,
    pub experts: Vec<Expert>,
    pub shared: Expert,
}

fn matrix(
    source: &dyn TensorSource,
    name: &str,
    input: usize,
    output: usize,
) -> Result<Weight<'static>, String> {
    check_tensor(source, name, &[input as u64, output as u64], false)?;
    let info = source.tensor_info(name).unwrap();
    // SAFETY: HyDecoder retains its source until after every borrowed weight is dropped.
    let bytes: &'static [u8] = unsafe { std::mem::transmute(source.tensor_slice(name).unwrap()) };
    Ok(Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        input,
        output,
    )))
}

impl HyDecoder {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        tokenizer: Arc<BPETokenizer>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let config = HyConfig::from_source(source.as_ref())?;
        validate_decoder(source.as_ref(), &config)?;
        if config.vocab != tokenizer.vocab_size() {
            return Err("Hy-SimulMT vocabulary does not match tokenizer".into());
        }
        let s = source.as_ref();
        let h = config.n_embd;
        let q = config
            .n_head
            .checked_mul(config.n_embd_head_k)
            .ok_or("Hy query width overflow")?;
        let kv = config
            .n_head_kv
            .checked_mul(config.n_embd_head_k)
            .ok_or("Hy KV width overflow")?;
        let vector = |name: &str, n| load_f32_tensor(s, name, &[n as u64]);
        let mut layers = Vec::with_capacity(config.n_layer);
        for index in 0..config.n_layer {
            let p = format!("blk.{index}");
            let expert = |suffix: &str, width| -> Result<Expert, String> {
                Ok(Expert {
                    gate: matrix(s, &format!("{p}.ffn_gate{suffix}.weight"), h, width)?,
                    up: matrix(s, &format!("{p}.ffn_up{suffix}.weight"), h, width)?,
                    down: matrix(s, &format!("{p}.ffn_down{suffix}.weight"), width, h)?,
                })
            };
            let ffn = if index == 0 {
                HyFfn::Dense(expert("", config.n_ff)?)
            } else {
                let m = &config.moe;
                let matrices = |name: &str, input, output| -> Vec<Weight<'static>> {
                    let name = format!("{p}.ffn_{name}_exps.weight");
                    let info = s.tensor_info(&name).unwrap();
                    let bytes = s.tensor_slice(&name).unwrap();
                    let stride = bytes.len() / m.expert_count;
                    bytes
                        .chunks_exact(stride)
                        .map(|bytes| {
                            // SAFETY: all slices belong to the source retained by this decoder.
                            let bytes: &'static [u8] = unsafe { std::mem::transmute(bytes) };
                            Weight::from_quantized(QuantizedTensor::from_bytes(
                                bytes,
                                info.ggml_type,
                                input,
                                output,
                            ))
                        })
                        .collect()
                };
                let experts = matrices("gate", h, m.expert_ffn)
                    .into_iter()
                    .zip(matrices("up", h, m.expert_ffn))
                    .zip(matrices("down", m.expert_ffn, h))
                    .map(|((gate, up), down)| Expert { gate, up, down })
                    .collect();
                HyFfn::Moe(HyMoe {
                    router: load_f32_tensor(
                        s,
                        &format!("{p}.ffn_gate_inp.weight"),
                        &[h as u64, m.expert_count as u64],
                    )?,
                    bias: vector(&format!("{p}.exp_probs_b"), m.expert_count)?,
                    experts,
                    shared: expert("_shexp", m.shared_expert_ffn)?,
                })
            };
            layers.push(HyLayer {
                attn_norm: vector(&format!("{p}.attn_norm.weight"), h)?,
                ffn_norm: vector(&format!("{p}.ffn_norm.weight"), h)?,
                q_norm: vector(&format!("{p}.attn_q_norm.weight"), config.n_embd_head_k)?,
                k_norm: vector(&format!("{p}.attn_k_norm.weight"), config.n_embd_head_k)?,
                q: matrix(s, &format!("{p}.attn_q.weight"), h, q)?,
                k: matrix(s, &format!("{p}.attn_k.weight"), h, kv)?,
                v: matrix(s, &format!("{p}.attn_v.weight"), h, kv)?,
                out: matrix(s, &format!("{p}.attn_output.weight"), q, h)?,
                ffn,
            });
        }
        Ok(Self {
            embedding: matrix(s, "token_embd.weight", h, config.vocab)?,
            output: matrix(s, "output.weight", h, config.vocab)?,
            output_norm: vector("output_norm.weight", h)?,
            layers,
            config,
            pool,
            tokenizer,
            _source: source,
        })
    }

    pub fn config(&self) -> &HyConfig {
        &self.config
    }
    pub fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
    pub fn pool(&self) -> Arc<ComputePool> {
        Arc::clone(&self.pool)
    }

    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        if ids.iter().any(|&id| id as usize >= self.config.vocab) {
            return Err("Hy-SimulMT token ID is outside vocabulary".into());
        }
        let len = ids
            .len()
            .checked_mul(self.config.n_embd)
            .ok_or("Hy embedding size overflow")?;
        let mut values = vec![0.0; len];
        for (&id, row) in ids.iter().zip(values.chunks_exact_mut(self.config.n_embd)) {
            self.embedding.embedding_lookup(id, row);
        }
        Ok(values)
    }
}
