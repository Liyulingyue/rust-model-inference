use super::config::{Gemma4Config, PER_LAYER, VOCAB};
use crate::core::tensor::{load_f32_tensor, GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{QuantizedTensor, Weight};
use std::sync::Arc;

pub struct Gemma4Model {
    pub(super) _source: Arc<dyn TensorSource>,
    pub config: Gemma4Config,
    pub(super) pool: Arc<ComputePool>,
    pub(super) token_embedding: Weight<'static>,
    /// Optional per-layer token projection. Only present when
    /// `config.use_per_layer_projection()` is true (E2B/E4B). For 12B
    /// this is `Weight::default()` and is unused.
    pub(super) per_layer_token_embedding: Option<Weight<'static>>,
    pub(super) per_layer_model_proj: Option<Weight<'static>>,
    pub(super) per_layer_proj_norm: Option<Vec<f32>>,
    pub(super) output_norm: Vec<f32>,
    pub(super) rope_freqs: Vec<f32>,
    pub(super) layers: Vec<Gemma4Layer>,
}

pub(super) struct Gemma4Layer {
    /// Q head count, uniform across layers (taken from `config.n_heads`).
    pub(super) q_heads: usize,
    /// KV head count, per-layer (`config.kv_heads_per_layer[layer]`).
    pub(super) kv_heads: usize,
    /// Per-layer head dim (256 SWA, 512 full).
    pub(super) head_dim: usize,
    /// `true` when this layer's V matrix is shared with K (12B
    /// full-attn MQA layers). When true, `attn_v` is `None` and the
    /// runtime reuses the K output as V.
    pub(super) kv_shared_with_k: bool,
    pub(super) attn_norm: Vec<f32>,
    pub(super) attn_q: Weight<'static>,
    pub(super) attn_k: Weight<'static>,
    /// `None` when V is shared with K (12B MQA). The runtime
    /// reuses the freshly-matmuled K as V in that case.
    pub(super) attn_v: Option<Weight<'static>>,
    pub(super) attn_output: Weight<'static>,
    pub(super) attn_q_norm: Vec<f32>,
    pub(super) attn_k_norm: Option<Vec<f32>>,
    pub(super) post_attention_norm: Vec<f32>,
    pub(super) ffn_norm: Vec<f32>,
    pub(super) ffn_gate: Weight<'static>,
    pub(super) ffn_up: Weight<'static>,
    pub(super) ffn_down: Weight<'static>,
    pub(super) post_ffw_norm: Vec<f32>,
    /// `None` when the per-layer projection is disabled (12B).
    pub(super) inp_gate: Option<Weight<'static>>,
    pub(super) proj: Option<Weight<'static>>,
    pub(super) post_norm: Option<Vec<f32>>,
    pub(super) output_scale: f32,
}

impl Gemma4Model {
    pub fn from_source(source: Arc<dyn TensorSource>, threads: usize) -> Result<Self, String> {
        let config = Gemma4Config::from_source(source.as_ref())?;
        let embd = config.embd;
        let per_layer_all = config.per_layer_all();
        let token_embedding = load_weight_any(
            source.as_ref(),
            "token_embd.weight",
            &[embd as u64, VOCAB as u64],
            &[GGMLType::Q8_0, GGMLType::Q4K],
        )?;

        // Per-layer projection tensors are optional (E2B/E4B: on, 12B: off).
        let (per_layer_token_embedding, per_layer_model_proj, per_layer_proj_norm) =
            if config.use_per_layer_projection() {
                let pe = load_weight_any(
                    source.as_ref(),
                    "per_layer_token_embd.weight",
                    &[per_layer_all as u64, VOCAB as u64],
                    &[GGMLType::Q8_0, GGMLType::Q5K],
                )?;
                let pm = load_weight(
                    source.as_ref(),
                    "per_layer_model_proj.weight",
                    &[embd as u64, per_layer_all as u64],
                    GGMLType::BF16,
                )?;
                let pn = load_f32(
                    source.as_ref(),
                    "per_layer_proj_norm.weight",
                    &[config.per_layer_width as u64],
                )?;
                (Some(pe), Some(pm), Some(pn))
            } else {
                (None, None, None)
            };

        let output_norm = load_f32(source.as_ref(), "output_norm.weight", &[embd as u64])?;
        let rope_freqs_dim = (config.full_head_dim / 2) as u64;
        let rope_freqs_dim_swa = (config.swa_head_dim / 2) as u64;
        let rope_info = source
            .tensor_info("rope_freqs.weight")
            .ok_or_else(|| "Missing tensor: rope_freqs.weight".to_string())?;
        let rope_len = if rope_info.dims.first().copied() == Some(rope_freqs_dim) {
            rope_freqs_dim
        } else if rope_info.dims.first().copied() == Some(rope_freqs_dim_swa) {
            rope_freqs_dim_swa
        } else {
            return Err(format!(
                "Invalid tensor rope_freqs.weight: shape {:?}; expected [{rope_freqs_dim}] or [{rope_freqs_dim_swa}]",
                rope_info.dims
            ));
        };
        let rope_freqs = load_f32(source.as_ref(), "rope_freqs.weight", &[rope_len])?;

        let mut layers = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            layers.push(load_layer(source.as_ref(), &config, layer)?);
        }

        Ok(Self {
            _source: source,
            config,
            pool: Arc::new(ComputePool::new(threads.max(1))),
            token_embedding,
            per_layer_token_embedding,
            per_layer_model_proj,
            per_layer_proj_norm,
            output_norm,
            rope_freqs,
            layers,
        })
    }

    pub fn pool(&self) -> &ComputePool {
        &self.pool
    }
}

pub(super) fn kv_source_layer(cfg: &Gemma4Config, layer: usize) -> usize {
    debug_assert!(layer < cfg.layers);
    let base = cfg.base_kv_layers();
    if layer < base {
        layer
    } else if cfg.is_swa(layer) {
        base - 2
    } else {
        base - 1
    }
}

fn load_layer(
    source: &dyn TensorSource,
    cfg: &Gemma4Config,
    layer: usize,
) -> Result<Gemma4Layer, String> {
    let prefix = format!("blk.{layer}");
    let head_dim = cfg.head_dim(layer);
    let q_heads = cfg.n_heads;
    let kv_heads = cfg.kv_heads(layer);
    let q_dim = q_heads * head_dim;
    let kv_dim = kv_heads * head_dim;
    let ffn = cfg.ffn_per_layer[layer];
    let embd = cfg.embd;
    let k_quant = [
        GGMLType::Q8_0,
        GGMLType::Q4K,
        GGMLType::Q6K,
        GGMLType::Q4_0,
        GGMLType::Q4_1,
        GGMLType::Q5_0,
        GGMLType::Q5_1,
    ];
    // V is required for kv_heads > 1; for kv_heads == 1 we fall back
    // to V := K (12B MQA sharing).
    let (attn_v, kv_shared_with_k) = if kv_heads > 1 {
        let v = load_weight_any(
            source,
            &format!("{prefix}.attn_v.weight"),
            &[embd as u64, kv_dim as u64],
            &k_quant,
        )?;
        (Some(v), false)
    } else if source.tensor_info(&format!("{prefix}.attn_v.weight")).is_some() {
        let v = load_weight_any(
            source,
            &format!("{prefix}.attn_v.weight"),
            &[embd as u64, kv_dim as u64],
            &k_quant,
        )?;
        (Some(v), false)
    } else {
        // V is shared with K (12B MQA). Runtime will reuse K as V.
        (None, true)
    };
    // attn_k_norm is required by E2B/E4B and present in 12B; load if
    // available. If missing, use a unit-norm fallback (effectively
    // a no-op since rms_norm with weight=1 is identity).
    let attn_k_norm = if source
        .tensor_info(&format!("{prefix}.attn_k_norm.weight"))
        .is_some()
    {
        load_f32(
            source,
            &format!("{prefix}.attn_k_norm.weight"),
            &[head_dim as u64],
        )?
        .into()
    } else {
        None
    };
    // Per-layer projection is optional (12B disables it).
    let (inp_gate, proj, post_norm) = if cfg.use_per_layer_projection() {
        let ig = load_weight(
            source,
            &format!("{prefix}.inp_gate.weight"),
            &[embd as u64, PER_LAYER as u64],
            GGMLType::F32,
        )?;
        let p = load_weight(
            source,
            &format!("{prefix}.proj.weight"),
            &[PER_LAYER as u64, embd as u64],
            GGMLType::F32,
        )?;
        let pn = load_f32(
            source,
            &format!("{prefix}.post_norm.weight"),
            &[embd as u64],
        )?;
        (Some(ig), Some(p), Some(pn))
    } else {
        (None, None, None)
    };
    Ok(Gemma4Layer {
        q_heads,
        kv_heads,
        head_dim,
        kv_shared_with_k,
        attn_norm: load_f32(
            source,
            &format!("{prefix}.attn_norm.weight"),
            &[embd as u64],
        )?,
        attn_q: load_weight_any(
            source,
            &format!("{prefix}.attn_q.weight"),
            &[embd as u64, q_dim as u64],
            &k_quant,
        )?,
        attn_k: load_weight_any(
            source,
            &format!("{prefix}.attn_k.weight"),
            &[embd as u64, kv_dim as u64],
            &k_quant,
        )?,
        attn_v,
        attn_output: load_weight_any(
            source,
            &format!("{prefix}.attn_output.weight"),
            &[q_dim as u64, embd as u64],
            &k_quant,
        )?,
        attn_q_norm: load_f32(
            source,
            &format!("{prefix}.attn_q_norm.weight"),
            &[head_dim as u64],
        )?,
        attn_k_norm,
        post_attention_norm: load_f32(
            source,
            &format!("{prefix}.post_attention_norm.weight"),
            &[embd as u64],
        )?,
        ffn_norm: load_f32(source, &format!("{prefix}.ffn_norm.weight"), &[embd as u64])?,
        ffn_gate: load_weight_any(
            source,
            &format!("{prefix}.ffn_gate.weight"),
            &[embd as u64, ffn as u64],
            &k_quant,
        )?,
        ffn_up: load_weight_any(
            source,
            &format!("{prefix}.ffn_up.weight"),
            &[embd as u64, ffn as u64],
            &k_quant,
        )?,
        ffn_down: load_weight_any(
            source,
            &format!("{prefix}.ffn_down.weight"),
            &[ffn as u64, embd as u64],
            &k_quant,
        )?,
        post_ffw_norm: load_f32(
            source,
            &format!("{prefix}.post_ffw_norm.weight"),
            &[embd as u64],
        )?,
        inp_gate,
        proj,
        post_norm,
        output_scale: load_f32(source, &format!("{prefix}.layer_output_scale.weight"), &[1])?[0],
    })
}

pub(super) fn load_weight(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
    ggml_type: GGMLType,
) -> Result<Weight<'static>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims || info.ggml_type != ggml_type {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} {:?}",
            info.dims, info.ggml_type, dims, ggml_type
        ));
    }
    let expected = info
        .checked_nbytes()
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    let n_in = usize::try_from(dims[0]).map_err(|_| format!("{name} input width overflow"))?;
    let n_out = usize::try_from(dims[1]).map_err(|_| format!("{name} output width overflow"))?;
    // SAFETY: Gemma4Model retains the immutable TensorSource Arc for at least as
    // long as these weights, matching the repository's existing model loaders.
    let bytes = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes) };
    let mut weight =
        Weight::from_quantized(QuantizedTensor::from_bytes(bytes, ggml_type, n_in, n_out));
    weight.n_in = n_in;
    weight.n_out = n_out;
    Ok(weight)
}

pub(super) fn load_weight_any(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
    allowed_types: &[GGMLType],
) -> Result<Weight<'static>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims || !allowed_types.contains(&info.ggml_type) {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} one of {allowed_types:?}",
            info.dims, info.ggml_type, dims
        ));
    }
    let expected = info
        .checked_nbytes()
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    let n_in = usize::try_from(dims[0]).map_err(|_| format!("{name} input width overflow"))?;
    let n_out = usize::try_from(dims[1]).map_err(|_| format!("{name} output width overflow"))?;
    let ggml_type = info.ggml_type;
    // SAFETY: Gemma4Model retains the immutable TensorSource Arc for at least as
    // long as these weights, matching the repository's existing model loaders.
    let bytes = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes) };
    let mut weight =
        Weight::from_quantized(QuantizedTensor::from_bytes(bytes, ggml_type, n_in, n_out));
    weight.n_in = n_in;
    weight.n_out = n_out;
    Ok(weight)
}

fn load_f32(source: &dyn TensorSource, name: &str, dims: &[u64]) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims || info.ggml_type != GGMLType::F32 {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} F32",
            info.dims, info.ggml_type, dims
        ));
    }
    load_f32_tensor(source, name, dims)
}
