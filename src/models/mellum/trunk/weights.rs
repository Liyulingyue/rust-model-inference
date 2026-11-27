//! Mellum weight tables: `MellumModel` plus the per-layer weight stack.
//!
//! Tensor names follow llama.cpp's shared table, so they are the same
//! strings the Qwen3-MoE path already uses (`blk.{i}.ffn_{gate,up,down}_exps`)
//! plus the two per-head norms (`attn_q_norm` / `attn_k_norm`).

use crate::core::loader::load_static_weight;
use crate::core::tensor::{load_f32_tensor, GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::ops::kernel::{QuantizedTensor, Weight};
use std::sync::Arc;

pub use super::config::MellumConfig;

pub struct MellumLayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub wq: Weight<'a>,
    pub wk: Weight<'a>,
    pub wv: Weight<'a>,
    pub wo: Weight<'a>,
    /// Router: `[expert_count, n_embd]` f32 (llama.cpp keeps it F32 because
    /// the routing decision must stay deterministic).
    pub moe_router: Vec<f32>,
    /// One `[expert_ffn, n_embd]` gate / up / down matrix per expert, sliced
    /// out of the fused `ffn_*_exps` tensors.
    pub moe_gate: Vec<Weight<'a>>,
    pub moe_up: Vec<Weight<'a>>,
    pub moe_down: Vec<Weight<'a>>,
}

pub struct MellumModel {
    /// Kept alive for the model's lifetime: every `Weight<'static>` handed out
    /// by [`MellumModel::from_source`] borrows this mmap.
    #[allow(dead_code)]
    pub(crate) source: Arc<dyn TensorSource>,
    pub(crate) tokenizer: Arc<BPETokenizer>,
    pub(crate) pool: Arc<ComputePool>,
    pub(crate) config: MellumConfig,
    pub(crate) layers: Vec<MellumLayerWeights<'static>>,
    pub(crate) output_norm: Vec<f32>,
    pub(crate) token_embedding: Weight<'static>,
    pub(crate) output: Weight<'static>,
}

impl MellumModel {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        tokenizer: Arc<BPETokenizer>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let config = MellumConfig::from_source(source.as_ref())?;
        if config.vocab_size != tokenizer.vocab_size() {
            return Err(format!(
                "{} vocabulary size {} does not match tokenizer vocab {}",
                super::config::ARCH,
                config.vocab_size,
                tokenizer.vocab_size()
            ));
        }
        let view = source.as_ref();

        let n_embd_q = config.n_head * config.n_embd_head_k;
        let n_embd_k = config.n_head_kv * config.n_embd_head_k;
        let n_embd_v = config.n_head_kv * config.n_embd_head_v;
        let expert_ffn = config.expert_ffn;

        let output_norm = load_f32_tensor(view, "output_norm.weight", &[config.n_embd as u64])?;
        let token_embedding =
            static_matrix(view, "token_embd.weight", config.n_embd, config.vocab_size)?;
        let output = static_matrix(view, "output.weight", config.n_embd, config.vocab_size)?;

        let mut layers = Vec::with_capacity(config.n_layer);
        for index in 0..config.n_layer {
            let prefix = format!("blk.{index}");
            let gate = expert_slices(
                view,
                &format!("{prefix}.ffn_gate_exps.weight"),
                config.n_embd,
                expert_ffn,
                config.expert_count,
            )?;
            let up = expert_slices(
                view,
                &format!("{prefix}.ffn_up_exps.weight"),
                config.n_embd,
                expert_ffn,
                config.expert_count,
            )?;
            let down = expert_slices(
                view,
                &format!("{prefix}.ffn_down_exps.weight"),
                expert_ffn,
                config.n_embd,
                config.expert_count,
            )?;
            layers.push(MellumLayerWeights {
                attn_norm: load_f32_tensor(
                    view,
                    &format!("{prefix}.attn_norm.weight"),
                    &[config.n_embd as u64],
                )?,
                ffn_norm: load_f32_tensor(
                    view,
                    &format!("{prefix}.ffn_norm.weight"),
                    &[config.n_embd as u64],
                )?,
                q_norm: load_f32_tensor(
                    view,
                    &format!("{prefix}.attn_q_norm.weight"),
                    &[config.n_embd_head_k as u64],
                )?,
                k_norm: load_f32_tensor(
                    view,
                    &format!("{prefix}.attn_k_norm.weight"),
                    &[config.n_embd_head_k as u64],
                )?,
                wq: load_static_weight(
                    view,
                    &format!("{prefix}.attn_q.weight"),
                    config.n_embd,
                    n_embd_q,
                ),
                wk: load_static_weight(
                    view,
                    &format!("{prefix}.attn_k.weight"),
                    config.n_embd,
                    n_embd_k,
                ),
                wv: load_static_weight(
                    view,
                    &format!("{prefix}.attn_v.weight"),
                    config.n_embd,
                    n_embd_v,
                ),
                wo: load_static_weight(
                    view,
                    &format!("{prefix}.attn_output.weight"),
                    n_embd_q,
                    config.n_embd,
                ),
                moe_router: load_f32_tensor(
                    view,
                    &format!("{prefix}.ffn_gate_inp.weight"),
                    &[config.n_embd as u64, config.expert_count as u64],
                )?,
                moe_gate: gate
                    .into_iter()
                    .map(|bytes| {
                        weight_from_bytes(
                            bytes,
                            config.n_embd,
                            expert_ffn,
                            view,
                            &format!("{prefix}.ffn_gate_exps.weight"),
                        )
                    })
                    .collect(),
                moe_up: up
                    .into_iter()
                    .map(|bytes| {
                        weight_from_bytes(
                            bytes,
                            config.n_embd,
                            expert_ffn,
                            view,
                            &format!("{prefix}.ffn_up_exps.weight"),
                        )
                    })
                    .collect(),
                moe_down: down
                    .into_iter()
                    .map(|bytes| {
                        weight_from_bytes(
                            bytes,
                            expert_ffn,
                            config.n_embd,
                            view,
                            &format!("{prefix}.ffn_down_exps.weight"),
                        )
                    })
                    .collect(),
            });
        }

        Ok(Self {
            source,
            tokenizer,
            pool,
            config,
            layers,
            output_norm,
            token_embedding,
            output,
        })
    }

    pub fn config(&self) -> &MellumConfig {
        &self.config
    }

    pub fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }

    pub fn pool(&self) -> Arc<ComputePool> {
        Arc::clone(&self.pool)
    }

    pub fn layers(&self) -> &[MellumLayerWeights<'_>] {
        &self.layers
    }

    pub fn output_norm(&self) -> &[f32] {
        &self.output_norm
    }

    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        let vocab = self.config.vocab_size;
        for &token_id in token_ids {
            if token_id as usize >= vocab {
                return Err(format!("token id {token_id} is outside vocabulary {vocab}"));
            }
        }
        let mut embeddings = vec![0.0; token_ids.len() * self.config.n_embd];
        for (row, &token_id) in embeddings
            .chunks_exact_mut(self.config.n_embd)
            .zip(token_ids)
        {
            self.token_embedding.embedding_lookup(token_id, row);
        }
        Ok(embeddings)
    }
}

fn weight_from_bytes(
    bytes: &'static [u8],
    n_in: usize,
    n_out: usize,
    view: &dyn TensorSource,
    name: &str,
) -> Weight<'static> {
    let ggml_type = view
        .tensor_info(name)
        .map(|info| info.ggml_type)
        .unwrap_or(GGMLType::F32);
    Weight::from_quantized(QuantizedTensor::from_bytes(bytes, ggml_type, n_in, n_out))
}

/// Split the fused `ffn_*_exps.weight` tensor (`[n_in, n_ff_exp, n_expert]`)
/// into one contiguous slice per expert. GGUF stores the experts back to
/// back, so the split is a pure view.
fn expert_slices(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
) -> Result<Vec<&'static [u8]>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let expected = [n_in as u64, n_out as u64, expert_count as u64];
    if info.dims != expected {
        return Err(format!(
            "Invalid {name} shape {:?}; expected {expected:?}",
            info.dims
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let total = info
        .checked_nbytes()
        .ok_or_else(|| format!("Invalid tensor byte size: {name}"))? as usize;
    if bytes.len() != total || !total.is_multiple_of(expert_count) {
        return Err(format!("Invalid {name} byte length: {}", bytes.len()));
    }
    let per = total / expert_count;
    let bytes: &'static [u8] = unsafe { std::mem::transmute(bytes) };
    (0..expert_count)
        .map(|expert| {
            bytes
                .get(expert * per..(expert + 1) * per)
                .ok_or_else(|| format!("{name} expert slice overflow"))
        })
        .collect()
}

fn static_matrix(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Weight<'static>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [n_in as u64, n_out as u64] {
        return Err(format!(
            "Invalid {name} shape {:?}; expected [{n_in}, {n_out}]",
            info.dims
        ));
    }
    Ok(load_static_weight(source, name, n_in, n_out))
}
