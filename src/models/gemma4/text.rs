use super::Gemma4Config;
use crate::core::scratchpad::KvFormat;
use crate::core::tensor::{load_f32_tensor, GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{QuantizedTensor, Weight};
use crate::ops::{
    attention_value_f32, dot_f32, gelu_inplace, quantize_q8_0_into, rms_norm, rms_norm_inplace,
    rope_neox, softmax_inplace,
};
use std::sync::Arc;

const LAYERS: usize = 35;
const BASE_KV_LAYERS: usize = 15;
const EMBED: usize = 1536;
const HEADS: usize = 8;
const FULL_HEAD_DIM: usize = 512;
const SWA_HEAD_DIM: usize = 256;
const BASE_FFN_LAYERS: usize = 15;
const MAX_FFN: usize = 12_288;
const VOCAB: usize = 262_144;
const PER_LAYER: usize = 256;
const PER_LAYER_ALL: usize = LAYERS * PER_LAYER;
const CONTEXT: usize = 131_072;
const EPS: f32 = 1e-6;

#[derive(Debug, Clone, PartialEq)]
pub enum Gemma4InputRow {
    Token(u32),
    Raw {
        values: Vec<f32>,
        per_layer_token: u32,
    },
}

#[derive(Debug)]
enum InputValues {
    Token(u32),
    Raw(Vec<f32>),
}

#[derive(Debug)]
struct AssembledInputRow {
    values: InputValues,
    scale_token_embedding: bool,
    per_layer_token: u32,
}

pub struct Gemma4Model {
    _source: Arc<dyn TensorSource>,
    pub config: Gemma4Config,
    pool: Arc<ComputePool>,
    token_embedding: Weight<'static>,
    per_layer_token_embedding: Weight<'static>,
    per_layer_model_proj: Weight<'static>,
    per_layer_proj_norm: Vec<f32>,
    output_norm: Vec<f32>,
    rope_freqs: Vec<f32>,
    layers: Vec<Gemma4Layer>,
}

struct Gemma4Layer {
    head_dim: usize,
    attn_norm: Vec<f32>,
    attn_q: Weight<'static>,
    attn_k: Weight<'static>,
    attn_v: Weight<'static>,
    attn_output: Weight<'static>,
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: Weight<'static>,
    ffn_up: Weight<'static>,
    ffn_down: Weight<'static>,
    post_ffw_norm: Vec<f32>,
    inp_gate: Weight<'static>,
    proj: Weight<'static>,
    post_norm: Vec<f32>,
    output_scale: f32,
}

pub struct Gemma4Session<'model> {
    model: &'model Gemma4Model,
    kv: Vec<KvLayer>,
    scratch: Gemma4Scratch,
    seq_len: usize,
}

struct KvLayer {
    head_dim: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

struct Gemma4Scratch {
    x: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    projected: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    per_layer: Vec<f32>,
    per_layer_projected: Vec<f32>,
    per_layer_gate: Vec<f32>,
    q8: Vec<u8>,
    scales: Vec<f32>,
    scores: Vec<f32>,
    attention_values: Vec<f32>,
    v_norm_weight: Vec<f32>,
    logits: Vec<f32>,
}

impl Gemma4Scratch {
    fn new() -> Self {
        let max_input = MAX_FFN.max(HEADS * FULL_HEAD_DIM);
        Self {
            x: vec![0.0; EMBED],
            normed: vec![0.0; EMBED],
            q: vec![0.0; HEADS * FULL_HEAD_DIM],
            k: vec![0.0; FULL_HEAD_DIM],
            v: vec![0.0; FULL_HEAD_DIM],
            attn: vec![0.0; HEADS * FULL_HEAD_DIM],
            projected: vec![0.0; EMBED],
            gate: vec![0.0; MAX_FFN],
            up: vec![0.0; MAX_FFN],
            down: vec![0.0; EMBED],
            per_layer: vec![0.0; PER_LAYER_ALL],
            per_layer_projected: vec![0.0; PER_LAYER_ALL],
            per_layer_gate: vec![0.0; PER_LAYER],
            q8: vec![0; max_input],
            scales: vec![0.0; max_input.div_ceil(32)],
            scores: Vec::new(),
            attention_values: Vec::new(),
            v_norm_weight: vec![1.0; FULL_HEAD_DIM],
            logits: vec![0.0; VOCAB],
        }
    }
}

impl Gemma4Model {
    pub fn from_source(source: Arc<dyn TensorSource>, threads: usize) -> Result<Self, String> {
        let config = Gemma4Config::from_source(source.as_ref())?;
        let token_embedding = load_weight(
            source.as_ref(),
            "token_embd.weight",
            &[EMBED as u64, VOCAB as u64],
            GGMLType::Q8_0,
        )?;
        let per_layer_token_embedding = load_weight(
            source.as_ref(),
            "per_layer_token_embd.weight",
            &[PER_LAYER_ALL as u64, VOCAB as u64],
            GGMLType::Q8_0,
        )?;
        let per_layer_model_proj = load_weight(
            source.as_ref(),
            "per_layer_model_proj.weight",
            &[EMBED as u64, PER_LAYER_ALL as u64],
            GGMLType::BF16,
        )?;
        let per_layer_proj_norm = load_f32(
            source.as_ref(),
            "per_layer_proj_norm.weight",
            &[PER_LAYER as u64],
        )?;
        let output_norm = load_f32(source.as_ref(), "output_norm.weight", &[EMBED as u64])?;
        let rope_freqs = load_f32(
            source.as_ref(),
            "rope_freqs.weight",
            &[(FULL_HEAD_DIM / 2) as u64],
        )?;

        let mut layers = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            layers.push(load_layer(source.as_ref(), layer)?);
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

impl<'model> Gemma4Session<'model> {
    pub fn new(model: &'model Gemma4Model, kv_format: KvFormat) -> Result<Self, String> {
        require_f32_kv(kv_format)?;
        let kv = (0..BASE_KV_LAYERS)
            .map(|layer| KvLayer {
                head_dim: head_dim(layer),
                keys: Vec::new(),
                values: Vec::new(),
            })
            .collect();
        Ok(Self {
            model,
            kv,
            scratch: Gemma4Scratch::new(),
            seq_len: 0,
        })
    }

    pub fn forward_rows(&mut self, rows: &[Gemma4InputRow]) -> Result<Vec<f32>, String> {
        let rows = assemble_input_rows(rows)?;
        let end = self
            .seq_len
            .checked_add(rows.len())
            .ok_or_else(|| "Gemma4 input length overflow".to_string())?;
        if end > CONTEXT {
            return Err(format!(
                "Gemma4 input length {end} exceeds context {CONTEXT}"
            ));
        }

        for row in &rows {
            self.forward_row(row)?;
            self.seq_len += 1;
        }
        Ok(self.scratch.logits.clone())
    }

    pub fn len(&self) -> usize {
        self.seq_len
    }

    fn forward_row(&mut self, row: &AssembledInputRow) -> Result<(), String> {
        let model = self.model;
        let scratch = &mut self.scratch;
        match &row.values {
            InputValues::Token(token) => {
                model
                    .token_embedding
                    .embedding_lookup(*token, &mut scratch.x);
            }
            InputValues::Raw(values) => scratch.x.copy_from_slice(values),
        }
        if row.scale_token_embedding {
            let scale = (EMBED as f32).sqrt();
            for value in &mut scratch.x {
                *value *= scale;
            }
        }
        ensure_finite("gemma4.input", &scratch.x)?;

        model
            .per_layer_token_embedding
            .embedding_lookup(row.per_layer_token, &mut scratch.per_layer);
        let token_scale = (PER_LAYER as f32).sqrt();
        for value in &mut scratch.per_layer {
            *value *= token_scale;
        }
        matmul(
            "per_layer_model_proj.weight",
            &model.per_layer_model_proj,
            &scratch.x,
            &mut scratch.per_layer_projected,
            model.pool(),
            &mut scratch.q8,
            &mut scratch.scales,
        )?;
        let projection_scale = 1.0 / (EMBED as f32).sqrt();
        let merge_scale = 1.0 / 2.0_f32.sqrt();
        for layer in 0..LAYERS {
            let start = layer * PER_LAYER;
            let end = start + PER_LAYER;
            let projected = &mut scratch.per_layer_projected[start..end];
            for value in projected.iter_mut() {
                *value *= projection_scale;
            }
            rms_norm_inplace(projected, &model.per_layer_proj_norm, EPS);
            for (target, projected) in scratch.per_layer[start..end].iter_mut().zip(projected) {
                *target = (*target + *projected) * merge_scale;
            }
        }
        ensure_finite("gemma4.per_layer_input", &scratch.per_layer)?;

        let position = self.seq_len;
        for layer_index in 0..LAYERS {
            let layer = &model.layers[layer_index];
            let dim = layer.head_dim;
            let q_width = HEADS * dim;
            let ffn = layer.ffn_gate.n_out;

            checked_rms_norm(
                &format!("blk.{layer_index}.attn_norm.weight"),
                &scratch.x,
                &layer.attn_norm,
                &mut scratch.normed,
            )?;
            matmul(
                &format!("blk.{layer_index}.attn_q.weight"),
                &layer.attn_q,
                &scratch.normed,
                &mut scratch.q[..q_width],
                model.pool(),
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            for query in scratch.q[..q_width].chunks_exact_mut(dim) {
                rms_norm_inplace(query, &layer.attn_q_norm, EPS);
                apply_rope(query, position, dim, layer_index, &model.rope_freqs)?;
            }

            if layer_index < BASE_KV_LAYERS {
                matmul(
                    &format!("blk.{layer_index}.attn_k.weight"),
                    &layer.attn_k,
                    &scratch.normed,
                    &mut scratch.k[..dim],
                    model.pool(),
                    &mut scratch.q8,
                    &mut scratch.scales,
                )?;
                matmul(
                    &format!("blk.{layer_index}.attn_v.weight"),
                    &layer.attn_v,
                    &scratch.normed,
                    &mut scratch.v[..dim],
                    model.pool(),
                    &mut scratch.q8,
                    &mut scratch.scales,
                )?;
                rms_norm_inplace(&mut scratch.k[..dim], &layer.attn_k_norm, EPS);
                rms_norm_inplace(&mut scratch.v[..dim], &scratch.v_norm_weight[..dim], EPS);
                apply_rope(
                    &mut scratch.k[..dim],
                    position,
                    dim,
                    layer_index,
                    &model.rope_freqs,
                )?;
                self.kv[layer_index].append(
                    layer_index,
                    position,
                    &scratch.k[..dim],
                    &scratch.v[..dim],
                )?;
            }

            let cache_layer = kv_source_layer(layer_index);
            attend(
                layer_index,
                position,
                &scratch.q[..q_width],
                &self.kv[cache_layer],
                is_swa(layer_index),
                &mut scratch.attn[..q_width],
                &mut scratch.scores,
                &mut scratch.attention_values,
            )?;
            matmul(
                &format!("blk.{layer_index}.attn_output.weight"),
                &layer.attn_output,
                &scratch.attn[..q_width],
                &mut scratch.projected,
                model.pool(),
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            checked_rms_norm(
                &format!("blk.{layer_index}.post_attention_norm.weight"),
                &scratch.projected,
                &layer.post_attention_norm,
                &mut scratch.down,
            )?;
            for (hidden, attention) in scratch.x.iter_mut().zip(&scratch.down) {
                *hidden += *attention;
            }
            ensure_finite(&format!("gemma4.layer.{layer_index}.attn_out"), &scratch.x)?;
            trace_layer("attn_out", layer_index, &scratch.x);

            checked_rms_norm(
                &format!("blk.{layer_index}.ffn_norm.weight"),
                &scratch.x,
                &layer.ffn_norm,
                &mut scratch.normed,
            )?;
            matmul(
                &format!("blk.{layer_index}.ffn_gate.weight"),
                &layer.ffn_gate,
                &scratch.normed,
                &mut scratch.gate[..ffn],
                model.pool(),
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            matmul(
                &format!("blk.{layer_index}.ffn_up.weight"),
                &layer.ffn_up,
                &scratch.normed,
                &mut scratch.up[..ffn],
                model.pool(),
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            gelu_inplace(&mut scratch.gate[..ffn]);
            for (gate, up) in scratch.gate[..ffn].iter_mut().zip(&scratch.up[..ffn]) {
                *gate *= *up;
            }
            matmul(
                &format!("blk.{layer_index}.ffn_down.weight"),
                &layer.ffn_down,
                &scratch.gate[..ffn],
                &mut scratch.down,
                model.pool(),
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            checked_rms_norm(
                &format!("blk.{layer_index}.post_ffw_norm.weight"),
                &scratch.down,
                &layer.post_ffw_norm,
                &mut scratch.projected,
            )?;
            for (hidden, ffn) in scratch.x.iter_mut().zip(&scratch.projected) {
                *hidden += *ffn;
            }
            ensure_finite(&format!("gemma4.layer.{layer_index}.ffn_out"), &scratch.x)?;
            trace_layer("ffn_out", layer_index, &scratch.x);

            matmul(
                &format!("blk.{layer_index}.inp_gate.weight"),
                &layer.inp_gate,
                &scratch.x,
                &mut scratch.per_layer_gate,
                model.pool(),
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            gelu_inplace(&mut scratch.per_layer_gate);
            let per_start = layer_index * PER_LAYER;
            for (gate, input) in scratch
                .per_layer_gate
                .iter_mut()
                .zip(&scratch.per_layer[per_start..per_start + PER_LAYER])
            {
                *gate *= *input;
            }
            matmul(
                &format!("blk.{layer_index}.proj.weight"),
                &layer.proj,
                &scratch.per_layer_gate,
                &mut scratch.down,
                model.pool(),
                &mut scratch.q8,
                &mut scratch.scales,
            )?;
            checked_rms_norm(
                &format!("blk.{layer_index}.post_norm.weight"),
                &scratch.down,
                &layer.post_norm,
                &mut scratch.projected,
            )?;
            for (hidden, per_layer) in scratch.x.iter_mut().zip(&scratch.projected) {
                *hidden = (*hidden + *per_layer) * layer.output_scale;
            }
            ensure_finite(
                &format!("gemma4.layer.{layer_index}.per_layer_out"),
                &scratch.x,
            )?;
            trace_layer("per_layer_out", layer_index, &scratch.x);
        }

        checked_rms_norm(
            "output_norm.weight",
            &scratch.x,
            &model.output_norm,
            &mut scratch.normed,
        )?;
        ensure_finite("gemma4.final.norm", &scratch.normed)?;
        trace("gemma4.final.norm", None, &scratch.normed);
        matmul(
            "token_embd.weight (tied output)",
            &model.token_embedding,
            &scratch.normed,
            &mut scratch.logits,
            model.pool(),
            &mut scratch.q8,
            &mut scratch.scales,
        )?;
        for logit in &mut scratch.logits {
            *logit = softcap(*logit, model.config.logit_softcap);
        }
        ensure_finite("gemma4.logits", &scratch.logits)?;
        trace("gemma4.logits", None, &scratch.logits);
        Ok(())
    }
}

impl KvLayer {
    fn append(
        &mut self,
        layer: usize,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), String> {
        if key.len() != self.head_dim || value.len() != self.head_dim {
            return Err(format!(
                "blk.{layer} KV row length mismatch: key {}, value {}, expected {}",
                key.len(),
                value.len(),
                self.head_dim
            ));
        }
        let expected = position
            .checked_mul(self.head_dim)
            .ok_or_else(|| format!("blk.{layer} KV length overflow"))?;
        if self.keys.len() != expected || self.values.len() != expected {
            return Err(format!(
                "blk.{layer} KV context mismatch at position {position}: key {}, value {}, expected {expected}",
                self.keys.len(),
                self.values.len()
            ));
        }
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        Ok(())
    }
}

fn assemble_input_rows(rows: &[Gemma4InputRow]) -> Result<Vec<AssembledInputRow>, String> {
    if rows.is_empty() {
        return Err("Gemma4 input rows are empty".into());
    }
    rows.iter()
        .enumerate()
        .map(|(index, row)| match row {
            Gemma4InputRow::Token(token) => {
                validate_token(index, "token", *token)?;
                Ok(AssembledInputRow {
                    values: InputValues::Token(*token),
                    scale_token_embedding: true,
                    per_layer_token: *token,
                })
            }
            Gemma4InputRow::Raw {
                values,
                per_layer_token,
            } => {
                if values.len() != EMBED {
                    return Err(format!(
                        "Gemma4 raw row {index} has length {}; expected {EMBED}",
                        values.len()
                    ));
                }
                if let Some((value_index, value)) = values
                    .iter()
                    .enumerate()
                    .find(|(_, value)| !value.is_finite())
                {
                    return Err(format!(
                        "Gemma4 raw row {index} has non-finite value {value:?} at index {value_index}"
                    ));
                }
                validate_token(index, "per-layer token", *per_layer_token)?;
                Ok(AssembledInputRow {
                    values: InputValues::Raw(values.clone()),
                    scale_token_embedding: false,
                    per_layer_token: *per_layer_token,
                })
            }
        })
        .collect()
}

fn validate_token(row: usize, kind: &str, token: u32) -> Result<(), String> {
    if token as usize >= VOCAB {
        return Err(format!(
            "Gemma4 row {row} {kind} ID {token} exceeds vocabulary {VOCAB}"
        ));
    }
    Ok(())
}

fn require_f32_kv(kv_format: KvFormat) -> Result<(), String> {
    if kv_format != KvFormat::F32 {
        return Err("Gemma4 incremental session requires an F32 KV cache".into());
    }
    Ok(())
}

fn is_swa(layer: usize) -> bool {
    layer % 5 != 4
}

fn head_dim(layer: usize) -> usize {
    if is_swa(layer) {
        SWA_HEAD_DIM
    } else {
        FULL_HEAD_DIM
    }
}

fn kv_source_layer(layer: usize) -> usize {
    debug_assert!(layer < LAYERS);
    if layer < BASE_KV_LAYERS {
        layer
    } else if is_swa(layer) {
        BASE_KV_LAYERS - 2
    } else {
        BASE_KV_LAYERS - 1
    }
}

fn load_layer(source: &dyn TensorSource, layer: usize) -> Result<Gemma4Layer, String> {
    let prefix = format!("blk.{layer}");
    let dim = head_dim(layer);
    let ffn = if layer < BASE_FFN_LAYERS {
        6144
    } else {
        MAX_FFN
    };
    Ok(Gemma4Layer {
        head_dim: dim,
        attn_norm: load_f32(
            source,
            &format!("{prefix}.attn_norm.weight"),
            &[EMBED as u64],
        )?,
        attn_q: load_weight(
            source,
            &format!("{prefix}.attn_q.weight"),
            &[EMBED as u64, (HEADS * dim) as u64],
            GGMLType::Q8_0,
        )?,
        attn_k: load_weight(
            source,
            &format!("{prefix}.attn_k.weight"),
            &[EMBED as u64, dim as u64],
            GGMLType::Q8_0,
        )?,
        attn_v: load_weight(
            source,
            &format!("{prefix}.attn_v.weight"),
            &[EMBED as u64, dim as u64],
            GGMLType::Q8_0,
        )?,
        attn_output: load_weight(
            source,
            &format!("{prefix}.attn_output.weight"),
            &[(HEADS * dim) as u64, EMBED as u64],
            GGMLType::Q8_0,
        )?,
        attn_q_norm: load_f32(
            source,
            &format!("{prefix}.attn_q_norm.weight"),
            &[dim as u64],
        )?,
        attn_k_norm: load_f32(
            source,
            &format!("{prefix}.attn_k_norm.weight"),
            &[dim as u64],
        )?,
        post_attention_norm: load_f32(
            source,
            &format!("{prefix}.post_attention_norm.weight"),
            &[EMBED as u64],
        )?,
        ffn_norm: load_f32(
            source,
            &format!("{prefix}.ffn_norm.weight"),
            &[EMBED as u64],
        )?,
        ffn_gate: load_weight(
            source,
            &format!("{prefix}.ffn_gate.weight"),
            &[EMBED as u64, ffn as u64],
            GGMLType::Q8_0,
        )?,
        ffn_up: load_weight(
            source,
            &format!("{prefix}.ffn_up.weight"),
            &[EMBED as u64, ffn as u64],
            GGMLType::Q8_0,
        )?,
        ffn_down: load_weight(
            source,
            &format!("{prefix}.ffn_down.weight"),
            &[ffn as u64, EMBED as u64],
            GGMLType::Q8_0,
        )?,
        post_ffw_norm: load_f32(
            source,
            &format!("{prefix}.post_ffw_norm.weight"),
            &[EMBED as u64],
        )?,
        inp_gate: load_weight(
            source,
            &format!("{prefix}.inp_gate.weight"),
            &[EMBED as u64, PER_LAYER as u64],
            GGMLType::F32,
        )?,
        proj: load_weight(
            source,
            &format!("{prefix}.proj.weight"),
            &[PER_LAYER as u64, EMBED as u64],
            GGMLType::F32,
        )?,
        post_norm: load_f32(
            source,
            &format!("{prefix}.post_norm.weight"),
            &[EMBED as u64],
        )?,
        output_scale: load_f32(source, &format!("{prefix}.layer_output_scale.weight"), &[1])?[0],
    })
}

fn load_weight(
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
    // QuantizedTensor's owned F32 variant carries values but not matrix shape.
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

#[allow(clippy::too_many_arguments)]
fn matmul(
    name: &str,
    weight: &Weight<'_>,
    input: &[f32],
    output: &mut [f32],
    pool: &ComputePool,
    q8: &mut [u8],
    scales: &mut [f32],
) -> Result<(), String> {
    if input.len() != weight.n_in || output.len() != weight.n_out {
        return Err(format!(
            "Invalid {name} matmul lengths: input {}, output {}; expected {}, {}",
            input.len(),
            output.len(),
            weight.n_in,
            weight.n_out
        ));
    }
    if weight.ggml_type == GGMLType::F32 {
        weight
            .kernel
            .forward(input, output, weight.n_in, weight.n_out);
    } else {
        let blocks = weight.n_in.div_ceil(32);
        if q8.len() < weight.n_in || scales.len() < blocks {
            return Err(format!("Invalid {name} activation scratch length"));
        }
        quantize_q8_0_into(
            input,
            weight.n_in,
            &mut q8[..weight.n_in],
            &mut scales[..blocks],
        );
        let input_ptr = input.as_ptr();
        let q8_ptr = q8.as_ptr();
        let scales_ptr = scales.as_ptr();
        let output_ptr = output.as_mut_ptr();
        pool.compute(|thread, threads| unsafe {
            weight.kernel.forward_prepared(
                std::slice::from_raw_parts(input_ptr, weight.n_in),
                std::slice::from_raw_parts(q8_ptr, weight.n_in),
                std::slice::from_raw_parts(scales_ptr, blocks),
                None,
                std::slice::from_raw_parts_mut(output_ptr, weight.n_out),
                weight.n_in,
                weight.n_out,
                thread,
                threads,
            );
        });
    }
    ensure_finite(name, output)
}

fn checked_rms_norm(
    name: &str,
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    if input.len() != weight.len() || input.len() != output.len() {
        return Err(format!(
            "Invalid {name} RMS lengths: input {}, weight {}, output {}",
            input.len(),
            weight.len(),
            output.len()
        ));
    }
    rms_norm(input, weight, output, EPS);
    ensure_finite(name, output)
}

fn apply_rope(
    values: &mut [f32],
    position: usize,
    dim: usize,
    layer: usize,
    full_freq_factors: &[f32],
) -> Result<(), String> {
    if values.len() % dim != 0 {
        return Err(format!(
            "blk.{layer} RoPE length {} is not divisible by head width {dim}",
            values.len()
        ));
    }
    if is_swa(layer) {
        rope_neox(values, position, dim, 10_000.0);
        return Ok(());
    }
    if full_freq_factors.len() != dim / 2 {
        return Err(format!(
            "rope_freqs.weight length {}; expected {} for blk.{layer}",
            full_freq_factors.len(),
            dim / 2
        ));
    }
    let theta_scale = 1_000_000.0_f32.powf(-2.0 / dim as f32);
    for head in values.chunks_exact_mut(dim) {
        let mut theta = position as f32;
        for pair in 0..dim / 2 {
            let angle = theta / full_freq_factors[pair];
            let (cosine, sine) = crate::ops::rope::rope_sin_cos(angle);
            let first = head[pair];
            let second = head[pair + dim / 2];
            head[pair] = first.mul_add(cosine, second * -sine);
            head[pair + dim / 2] = first.mul_add(sine, second * cosine);
            theta *= theta_scale;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn attend(
    layer: usize,
    position: usize,
    query: &[f32],
    cache: &KvLayer,
    sliding: bool,
    output: &mut [f32],
    scores: &mut Vec<f32>,
    values: &mut Vec<f32>,
) -> Result<(), String> {
    let dim = cache.head_dim;
    if query.len() != HEADS * dim || output.len() != HEADS * dim {
        return Err(format!(
            "blk.{layer} attention length mismatch: query {}, output {}, expected {}",
            query.len(),
            output.len(),
            HEADS * dim
        ));
    }
    let rows = position + 1;
    let expected = rows
        .checked_mul(dim)
        .ok_or_else(|| format!("blk.{layer} KV context length overflow"))?;
    if cache.keys.len() != expected || cache.values.len() != expected {
        return Err(format!(
            "blk.{layer} shared KV context mismatch: key {}, value {}, expected {expected}",
            cache.keys.len(),
            cache.values.len()
        ));
    }
    let first = if sliding { rows.saturating_sub(512) } else { 0 };
    let cached = rows - first;
    let padded = cached.div_ceil(256) * 256;
    scores.resize(padded, f32::NEG_INFINITY);
    values.resize(padded, 0.0);

    for (head, query) in query.chunks_exact(dim).enumerate() {
        scores.fill(f32::NEG_INFINITY);
        for (score, token) in scores[..cached].iter_mut().zip(first..rows) {
            let offset = token * dim;
            *score = dot_f32(query, &cache.keys[offset..offset + dim], dim);
        }
        softmax_inplace(scores);
        for dimension in 0..dim {
            values.fill(0.0);
            for (slot, token) in values[..cached].iter_mut().zip(first..rows) {
                *slot = cache.values[token * dim + dimension];
            }
            output[head * dim + dimension] = attention_value_f32(values, scores, cached, padded);
        }
    }
    ensure_finite(&format!("blk.{layer} attention"), output)
}

fn softcap(value: f32, cap: f32) -> f32 {
    cap * (value / cap).tanh()
}

fn ensure_finite(name: &str, values: &[f32]) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{name} produced non-finite value {value:?} at index {index}"
        ));
    }
    Ok(())
}

fn trace_layer(stage: &str, layer: usize, values: &[f32]) {
    trace(
        &format!("gemma4.layer.{layer}.{stage}"),
        Some(layer),
        values,
    );
}

#[cfg(feature = "parity-trace")]
fn trace(name: &str, layer: Option<usize>, values: &[f32]) {
    crate::parity_trace::report(crate::parity_trace::checkpoint(
        name,
        layer,
        &[1, values.len()],
        values,
    ));
}

#[cfg(not(feature = "parity-trace"))]
fn trace(_name: &str, _layer: Option<usize>, _values: &[f32]) {}

#[cfg(test)]
mod tests {
    use super::{
        assemble_input_rows, kv_source_layer, load_weight, require_f32_kv, softcap, Gemma4InputRow,
    };
    use crate::core::scratchpad::KvFormat;
    use crate::core::tensor::{GGMLType, TensorInfo, TensorSource};

    #[test]
    fn raw_rows_are_not_embedding_scaled_and_use_padding_layer_id() {
        let rows = assemble_input_rows(&[
            Gemma4InputRow::Token(7),
            Gemma4InputRow::Raw {
                values: vec![2.0; 1536],
                per_layer_token: 0,
            },
        ])
        .unwrap();
        assert!(rows[0].scale_token_embedding);
        assert!(!rows[1].scale_token_embedding);
        assert_eq!(rows[1].per_layer_token, 0);
    }

    #[test]
    fn softcap_uses_the_declared_f32_formula() {
        assert_eq!(
            softcap(60.0, 30.0).to_bits(),
            (30.0_f32 * 2.0_f32.tanh()).to_bits()
        );
    }

    #[test]
    fn input_rows_reject_empty_invalid_and_nonfinite_values() {
        assert!(assemble_input_rows(&[]).unwrap_err().contains("empty"));
        assert!(assemble_input_rows(&[Gemma4InputRow::Token(262_144)])
            .unwrap_err()
            .contains("token"));
        assert!(assemble_input_rows(&[Gemma4InputRow::Raw {
            values: vec![0.0; 1535],
            per_layer_token: 0,
        }])
        .unwrap_err()
        .contains("1536"));
        assert!(assemble_input_rows(&[Gemma4InputRow::Raw {
            values: {
                let mut values = vec![0.0; 1536];
                values[7] = f32::NAN;
                values
            },
            per_layer_token: 0,
        }])
        .unwrap_err()
        .contains("non-finite"));
    }

    #[test]
    fn shared_kv_layers_map_by_attention_kind() {
        assert_eq!(kv_source_layer(0), 0);
        assert_eq!(kv_source_layer(14), 14);
        assert_eq!(kv_source_layer(15), 13);
        assert_eq!(kv_source_layer(19), 14);
        assert_eq!(kv_source_layer(34), 14);
    }

    #[test]
    fn incremental_session_is_f32_only() {
        assert!(require_f32_kv(KvFormat::F32).is_ok());
        assert!(require_f32_kv(KvFormat::F16).unwrap_err().contains("F32"));
    }

    #[test]
    fn f32_matrix_loader_preserves_declared_shape() {
        struct F32Matrix {
            info: TensorInfo,
            bytes: Vec<u8>,
        }
        impl TensorSource for F32Matrix {
            fn metadata(&self, _key: &str) -> Option<&crate::core::tensor::MetaValue> {
                None
            }

            fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
                (name == "matrix.weight").then_some(&self.info)
            }

            fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
                (name == "matrix.weight").then_some(self.bytes.as_slice())
            }
        }
        let source = F32Matrix {
            info: TensorInfo {
                name: "matrix.weight".into(),
                dims: vec![2, 3],
                ggml_type: GGMLType::F32,
                offset: 0,
            },
            bytes: vec![0; 2 * 3 * 4],
        };
        let weight = load_weight(&source, "matrix.weight", &[2, 3], GGMLType::F32).unwrap();
        assert_eq!((weight.n_in, weight.n_out), (2, 3));
    }

    #[test]
    #[ignore = "requires RMI_GEMMA4_MODEL"]
    fn actual_model_one_token_produces_finite_logits() {
        let path = std::env::var_os("RMI_GEMMA4_MODEL").expect("RMI_GEMMA4_MODEL");
        let source = std::sync::Arc::new(crate::core::loader::GGUFLoader::from_file(path).unwrap());
        for (layer, expected_ffn) in [(14, 6144), (15, 12_288), (34, 12_288)] {
            assert_eq!(
                source
                    .tensor_info(&format!("blk.{layer}.ffn_gate.weight"))
                    .unwrap()
                    .dims,
                [1536, expected_ffn]
            );
        }
        let model = super::Gemma4Model::from_source(source, 4).unwrap();
        let mut session = super::Gemma4Session::new(&model, KvFormat::F32).unwrap();
        let logits = session.forward_rows(&[Gemma4InputRow::Token(2)]).unwrap();
        assert_eq!(logits.len(), 262_144);
        assert!(logits.iter().all(|value| value.is_finite()));
    }
}
