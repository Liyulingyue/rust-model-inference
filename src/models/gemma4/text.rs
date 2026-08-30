use super::Gemma4Config;
use crate::core::scratchpad::KvFormat;
use crate::core::tensor::{load_f32_tensor, GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{QuantizedTensor, Weight};
use crate::ops::{
    attention_value_f32, bf16_to_f32, dot_f32, f16_to_f32, f32_to_bf16, f32_to_f16,
    quantize_q8_0_into, rms_norm, rms_norm_inplace, rope_neox, softmax_ggml_inplace,
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
            let kv_lengths = self
                .kv
                .iter()
                .map(|layer| (layer.keys.len(), layer.values.len()))
                .collect::<Vec<_>>();
            if let Err(error) = self.forward_row(row) {
                for (layer, (key_len, value_len)) in self.kv.iter_mut().zip(kv_lengths) {
                    layer.keys.truncate(key_len);
                    layer.values.truncate(value_len);
                }
                return Err(error);
            }
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
            ggml_geglu_fp16_inplace(&mut scratch.gate[..ffn], &scratch.up[..ffn]);
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
            let per_start = layer_index * PER_LAYER;
            ggml_geglu_fp16_inplace(
                &mut scratch.per_layer_gate,
                &scratch.per_layer[per_start..per_start + PER_LAYER],
            );
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
    if name == "per_layer_model_proj.weight" {
        if weight.ggml_type != GGMLType::BF16 {
            return Err(format!("{name} requires BF16 weight"));
        }
        let bytes = weight
            .kernel
            .bf16_bytes()
            .ok_or_else(|| format!("Invalid {name} BF16 kernel"))?;
        gemma4_bf16_projection_matmul(bytes, input, output, pool, q8)?;
    } else if weight.ggml_type == GGMLType::F32 {
        let values = weight
            .kernel
            .f32_slice()
            .ok_or_else(|| format!("Invalid {name} F32 kernel"))?;
        let expected = weight
            .n_in
            .checked_mul(weight.n_out)
            .ok_or_else(|| format!("Invalid {name} F32 weight size"))?;
        if values.len() != expected {
            return Err(format!(
                "Invalid {name} F32 weight length: expected {expected}, got {}",
                values.len()
            ));
        }
        #[cfg(target_arch = "aarch64")]
        {
            for (result, row) in output.iter_mut().zip(values.chunks_exact(weight.n_in)) {
                *result = dot_f32(row, input, weight.n_in);
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            weight
                .kernel
                .forward(input, output, weight.n_in, weight.n_out);
        }
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

fn gemma4_bf16_projection_matmul(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    pool: &ComputePool,
    input_bf16: &mut [u8],
) -> Result<(), String> {
    let input_bytes = input
        .len()
        .checked_mul(2)
        .ok_or_else(|| "Gemma4 BF16 projection input byte size overflow".to_owned())?;
    let weight_bytes = input_bytes
        .checked_mul(output.len())
        .ok_or_else(|| "Gemma4 BF16 projection weight byte size overflow".to_owned())?;
    if input_bf16.len() < input_bytes {
        return Err("Invalid Gemma4 BF16 projection storage length".to_owned());
    }
    if weight.len() != weight_bytes {
        return Err(format!(
            "Invalid Gemma4 BF16 projection weight storage length: expected {weight_bytes} bytes, got {}",
            weight.len()
        ));
    }

    for (bytes, value) in input_bf16[..input_bytes].chunks_exact_mut(2).zip(input) {
        bytes.copy_from_slice(&f32_to_bf16(*value).to_le_bytes());
    }

    let n_in = input.len();
    let n_out = output.len();
    let weight_ptr = weight.as_ptr();
    let input_ptr = input_bf16.as_ptr();
    let output_ptr = output.as_mut_ptr();
    pool.compute(|thread, threads| unsafe {
        let start = n_out * thread / threads;
        let end = n_out * (thread + 1) / threads;
        for row in start..end {
            let mut sum = 0.0f64;
            for column in 0..n_in {
                let weight_offset = (row * n_in + column) * 2;
                let input_offset = column * 2;
                let weight_bits = u16::from_le_bytes([
                    *weight_ptr.add(weight_offset),
                    *weight_ptr.add(weight_offset + 1),
                ]);
                let input_bits = u16::from_le_bytes([
                    *input_ptr.add(input_offset),
                    *input_ptr.add(input_offset + 1),
                ]);
                let product = bf16_to_f32(weight_bits) * bf16_to_f32(input_bits);
                sum += f64::from(product);
            }
            *output_ptr.add(row) = sum as f32;
        }
    });
    Ok(())
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
        softmax_ggml_inplace(scores);
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
    cap * (value * (1.0 / cap)).tanh()
}

fn ggml_geglu_fp16_inplace(gate: &mut [f32], up: &[f32]) {
    const GELU_COEF_A: f32 = 0.044715;
    const SQRT_2_OVER_PI: f32 = 0.79788456080286535587989211986876;

    assert_eq!(gate.len(), up.len());
    for (gate, up) in gate.iter_mut().zip(up) {
        let x = *gate;
        *gate = if x <= -10.0 {
            0.0
        } else if x >= 10.0 {
            x * up
        } else {
            let x = f16_to_f32(f32_to_f16(x));
            let gelu =
                0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * x.mul_add(GELU_COEF_A * x, 1.0)).tanh());
            f16_to_f32(f32_to_f16(gelu)) * up
        };
    }
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
        assemble_input_rows, attend, head_dim, kv_source_layer, load_weight, matmul,
        require_f32_kv, softcap, Gemma4InputRow, Gemma4Layer, Gemma4Model, KvLayer,
        BASE_FFN_LAYERS, EMBED, FULL_HEAD_DIM, HEADS, LAYERS, MAX_FFN, PER_LAYER, PER_LAYER_ALL,
        SWA_HEAD_DIM, VOCAB,
    };
    use crate::core::scratchpad::KvFormat;
    use crate::core::tensor::{GGMLType, TensorInfo, TensorSource};
    use crate::core::thread_pool::ComputePool;
    use crate::models::gemma4::Gemma4Config;
    use crate::ops::kernel::{Kernel, QuantizedTensor, Weight};
    use std::sync::Arc;

    struct EmptySource;

    impl TensorSource for EmptySource {
        fn metadata(&self, _key: &str) -> Option<&crate::core::tensor::MetaValue> {
            None
        }

        fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
            None
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    struct ZeroKernel;

    impl Kernel for ZeroKernel {
        fn forward_prequantized(
            &self,
            _input_q8: &[u8],
            _input_scales: &[f32],
            output: &mut [f32],
            _n_in: usize,
            n_out: usize,
            _ith: usize,
            _nth: usize,
        ) {
            output[..n_out].fill(0.0);
        }

        fn embedding_lookup(&self, _token_id: u32, n_embd: usize, output: &mut [f32]) {
            assert_eq!(output.len(), n_embd);
            output.fill(0.0);
        }
    }

    struct ZeroBf16Kernel {
        bytes: Vec<u8>,
    }

    impl Kernel for ZeroBf16Kernel {
        fn bf16_bytes(&self) -> Option<&[u8]> {
            Some(&self.bytes)
        }

        fn forward_prequantized(
            &self,
            _input_q8: &[u8],
            _input_scales: &[f32],
            output: &mut [f32],
            _n_in: usize,
            n_out: usize,
            _ith: usize,
            _nth: usize,
        ) {
            output[..n_out].fill(0.0);
        }
    }

    fn zero_weight(n_in: usize, n_out: usize) -> Weight<'static> {
        Weight {
            kernel: Box::new(ZeroKernel),
            ggml_type: GGMLType::F32,
            n_in,
            n_out,
        }
    }

    fn zero_q8_weight(n_in: usize, n_out: usize) -> Weight<'static> {
        Weight {
            kernel: Box::new(ZeroKernel),
            ggml_type: GGMLType::Q8_0,
            n_in,
            n_out,
        }
    }

    fn zero_bf16_weight(n_in: usize, n_out: usize) -> Weight<'static> {
        Weight {
            kernel: Box::new(ZeroBf16Kernel {
                bytes: vec![0; n_in * n_out * 2],
            }),
            ggml_type: GGMLType::BF16,
            n_in,
            n_out,
        }
    }

    fn zero_layer(layer: usize) -> Gemma4Layer {
        let dim = head_dim(layer);
        let ffn = if layer < BASE_FFN_LAYERS {
            6144
        } else {
            MAX_FFN
        };
        Gemma4Layer {
            head_dim: dim,
            attn_norm: vec![1.0; EMBED],
            attn_q: zero_q8_weight(EMBED, HEADS * dim),
            attn_k: zero_q8_weight(EMBED, dim),
            attn_v: zero_q8_weight(EMBED, dim),
            attn_output: zero_q8_weight(HEADS * dim, EMBED),
            attn_q_norm: vec![1.0; dim],
            attn_k_norm: vec![1.0; dim],
            post_attention_norm: vec![1.0; EMBED],
            ffn_norm: vec![1.0; EMBED],
            ffn_gate: zero_q8_weight(EMBED, ffn),
            ffn_up: zero_q8_weight(EMBED, ffn),
            ffn_down: zero_q8_weight(ffn, EMBED),
            post_ffw_norm: vec![1.0; EMBED],
            inp_gate: zero_weight(EMBED, PER_LAYER),
            proj: zero_weight(PER_LAYER, EMBED),
            post_norm: vec![1.0; EMBED],
            output_scale: 1.0,
        }
    }

    fn post_kv_failure_model() -> Gemma4Model {
        let mut layers = (0..LAYERS).map(zero_layer).collect::<Vec<_>>();
        layers[0].attn_output.n_in += 1;
        Gemma4Model {
            _source: Arc::new(EmptySource),
            config: Gemma4Config {
                layers: LAYERS,
                embd: EMBED,
                heads: HEADS,
                kv_heads: 1,
                vocab: VOCAB,
                full_head_dim: FULL_HEAD_DIM,
                swa_head_dim: SWA_HEAD_DIM,
                shared_kv_layers: 20,
                per_layer_width: PER_LAYER,
                sliding_window: 512,
                logit_softcap: 30.0,
            },
            pool: Arc::new(ComputePool::new(1)),
            token_embedding: zero_weight(EMBED, VOCAB),
            per_layer_token_embedding: zero_weight(PER_LAYER_ALL, VOCAB),
            per_layer_model_proj: zero_bf16_weight(EMBED, PER_LAYER_ALL),
            per_layer_proj_norm: vec![1.0; PER_LAYER],
            output_norm: vec![1.0; EMBED],
            rope_freqs: vec![1.0; FULL_HEAD_DIM / 2],
            layers,
        }
    }

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
    fn softcap_matches_pinned_reciprocal_scale_bits() {
        // Pinned llama.cpp 3173a56471c, first text raw logit at index 1.
        let raw = f32::from_bits(0x417c_38d8);
        assert_eq!(softcap(raw, 30.0).to_bits(), 0x4167_507f);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn layer_12_attention_softmax_matches_pinned_neon_bits() {
        // Occurrence 7 head-0 KQ words and output are independently pinned from llama.cpp 3173a56471c.
        let keys = [
            0x40b4_85b2,
            0x3ffc_c0c2,
            0x4079_1edf,
            0x4027_f5cc,
            0x407c_44ba,
            0x4078_0503,
            0xbec0_388c,
            0x405a_25f4,
        ]
        .map(f32::from_bits);
        let cache = KvLayer {
            head_dim: 1,
            keys: keys.to_vec(),
            values: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].to_vec(),
        };
        let mut output = [0.0; HEADS];

        attend(
            12,
            7,
            &[1.0; HEADS],
            &cache,
            true,
            &mut output,
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .unwrap();

        assert_eq!(output.map(f32::to_bits), [0x3f15_89fe; HEADS]);
    }

    #[test]
    fn ggml_geglu_rounds_gate_and_gelu_through_f16() {
        let mut gate = [0.0; 8];
        let mut up = [1.0; 8];
        gate[0] = f32::from_bits(0x3f12_598e);
        up[0] = f32::from_bits(0xbed7_8765);
        gate[1] = f32::from_bits(0xbfff_e000);

        super::ggml_geglu_fp16_inplace(&mut gate, &up);

        assert_eq!(gate[0].to_bits(), 0xbe30_7c3e);
        assert_eq!(gate[1].to_bits(), 0xbd3a_6000);
    }

    #[test]
    fn f32_projection_rejects_missing_or_wrong_backing_storage() {
        let cases = [
            (zero_weight(2, 1), "F32 kernel"),
            (
                Weight {
                    kernel: Box::new(crate::ops::kernel::f32::F32Kernel::new(vec![0.0])),
                    ggml_type: GGMLType::F32,
                    n_in: 2,
                    n_out: 1,
                },
                "expected 2, got 1",
            ),
        ];

        for (weight, expected_error) in cases {
            let mut output = [7.0];
            let mut q8 = [0; 2];
            let mut scales = [0.0];
            let error = matmul(
                "blk.0.inp_gate.weight",
                &weight,
                &[1.0, 2.0],
                &mut output,
                &ComputePool::new(1),
                &mut q8,
                &mut scales,
            )
            .unwrap_err();

            assert!(error.contains(expected_error), "{error}");
            assert_eq!(output, [7.0]);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn per_layer_f32_projection_matches_pinned_neon_dot_bits() {
        // Pinned llama.cpp 3173a56471c, blk.0.inp_gate.weight row 0 and
        // layer-0 FFN output occurrence 0. The first 16 real operands already
        // distinguish its four FMA accumulators from sequential F32 addition.
        let weights = [
            0x3a89_0000,
            0x39fb_0000,
            0xb7e5_0000,
            0xb7a2_0000,
            0x39f0_0000,
            0x3a16_0000,
            0xba2b_0000,
            0xba3c_0000,
            0xb906_0000,
            0x3748_0000,
            0xba28_0000,
            0xb9c4_0000,
            0x377b_0000,
            0xba2a_0000,
            0xb983_0000,
            0xb987_0000,
        ]
        .map(f32::from_bits);
        let input = [
            0xc116_ef77,
            0x413c_c829,
            0x3e4c_d214,
            0xc180_6c96,
            0x400b_f0a2,
            0xc03c_6f04,
            0xbe76_9592,
            0x3cec_5d40,
            0x3f80_3150,
            0x401d_94ed,
            0x3e9a_5ed4,
            0x4093_33e6,
            0x3f1c_0cfe,
            0xc0af_a2b5,
            0xc026_7f9c,
            0xbf27_ece8,
        ]
        .map(f32::from_bits);
        let weight = Weight {
            kernel: Box::new(crate::ops::kernel::f32::F32Kernel::new(weights.to_vec())),
            ggml_type: GGMLType::F32,
            n_in: input.len(),
            n_out: 1,
        };
        let mut output = [0.0];
        let mut q8 = [0; 16];
        let mut scales = [0.0];

        matmul(
            "blk.0.inp_gate.weight",
            &weight,
            &input,
            &mut output,
            &ComputePool::new(1),
            &mut q8,
            &mut scales,
        )
        .unwrap();

        assert_eq!(output[0].to_bits(), 0xbb08_36fd);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn per_layer_f32_projection_matches_pinned_neon_long_rows() {
        const WIDTH: usize = 1536;
        const ROWS: usize = 3;
        let input = (0_u32..WIDTH as u32)
            .map(|index| {
                let mixed = index.wrapping_mul(0x9e37_79b9).wrapping_add(0x243f_6a88);
                f32::from_bits((mixed & 0x8000_0000) | 0x3e00_0000 | ((mixed >> 1) & 0x007f_ffff))
            })
            .collect::<Vec<_>>();
        let weights = (0_u32..ROWS as u32)
            .flat_map(|row| {
                (0_u32..WIDTH as u32).map(move |index| {
                    let mixed = index
                        .wrapping_mul(0x85eb_ca6b)
                        .wrapping_add((row + 1).wrapping_mul(0xc2b2_ae35));
                    f32::from_bits(
                        (mixed & 0x8000_0000) | 0x3d80_0000 | ((mixed >> 1) & 0x007f_ffff),
                    )
                })
            })
            .collect::<Vec<_>>();
        let weight = Weight {
            kernel: Box::new(crate::ops::kernel::f32::F32Kernel::new(weights)),
            ggml_type: GGMLType::F32,
            n_in: WIDTH,
            n_out: ROWS,
        };
        let mut output = [0.0; ROWS];
        let mut q8 = vec![0; WIDTH];
        let mut scales = vec![0.0; WIDTH.div_ceil(32)];

        matmul(
            "blk.0.inp_gate.weight",
            &weight,
            &input,
            &mut output,
            &ComputePool::new(1),
            &mut q8,
            &mut scales,
        )
        .unwrap();

        // Independent literals from pinned llama.cpp ggml_vec_dot_f32.
        assert_eq!(
            output.map(f32::to_bits),
            [0xbe74_d6c6, 0x3df2_a865, 0x3ed8_e80e]
        );
    }

    #[test]
    fn per_layer_bf16_projection_matches_pinned_scalar_dot_bits() {
        // Pinned llama.cpp 3173a56471c, Gemma4 text projection, first 16
        // operands from real rows 0, 1, 2, 3, and 5. Its arm64 BF16 dot rounds
        // the activation to BF16, forms F32 products, accumulates them in
        // ggml_float (F64), then casts once. The pinned row-3 and row-5 F32
        // accumulator words are respectively 0x3daed280 and 0xbdef03c0, so
        // those rows make an F32-accumulation mutation observable.
        let input = [
            0xbfd0_8482,
            0xbfc2_eb2b,
            0x3e47_739e,
            0xbfbe_62b9,
            0xbf7d_d8f7,
            0xbd11_0e44,
            0xbee2_a64a,
            0x3e87_fd60,
            0xbfa9_fcb8,
            0x3f7d_d8f7,
            0xbf1a_1f28,
            0xbfa9_fcb8,
            0xbeeb_b72f,
            0x3ee2_a64a,
            0xbf8a_4199,
            0xbebe_62b9,
        ]
        .map(f32::from_bits);
        let weight_rows = [
            [
                0x3d37_u16, 0x3d04, 0xbc50, 0x3d77, 0x3bc7, 0x3cd1, 0xbcdb, 0xbdae, 0xbbe5, 0x3b39,
                0xbbcd, 0x3c9e, 0x3cde, 0x3d16, 0xbd82, 0x3c63,
            ],
            [
                0x3c47, 0x3b92, 0x3ca0, 0xbd46, 0xbd80, 0x3d89, 0x3ce9, 0xbcef, 0xbc48, 0xbcbf,
                0xbd18, 0x3ce0, 0x3d43, 0xbd9e, 0x3c35, 0xbcae,
            ],
            [
                0x3ca8, 0xbc5d, 0x3d50, 0xbd1e, 0xbc40, 0x3da4, 0x3ba4, 0xbc8f, 0x3d2c, 0x3cac,
                0xbd3c, 0x3b94, 0x3d03, 0x3c49, 0x3d79, 0x3c83,
            ],
            [
                0xbb1e, 0xbd4b, 0xbac3, 0xbd35, 0x3cc6, 0x3c9b, 0x3c2c, 0x3d72, 0xbd09, 0xbcf5,
                0xbcb6, 0xbbdb, 0xbc6d, 0xbcea, 0x3d91, 0x3a98,
            ],
            [
                0x3cd0, 0x3d18, 0x3bf3, 0xbb41, 0xbbea, 0xbb32, 0x3c12, 0xbd5e, 0x3afa, 0xbc7c,
                0x39ab, 0x3d39, 0xbc9d, 0x3d0e, 0xbd33, 0x3c94,
            ],
        ];
        let rows = weight_rows.len();
        let weight = weight_rows
            .into_iter()
            .flatten()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let weight = Weight::from_quantized(QuantizedTensor::from_bytes(
            &weight,
            GGMLType::BF16,
            input.len(),
            rows,
        ));
        let mut output = [0.0; 5];
        let mut q8 = vec![0; input.len() * 2];
        let mut scales = vec![0.0; input.len().div_ceil(32)];

        matmul(
            "per_layer_model_proj.weight",
            &weight,
            &input,
            &mut output,
            &ComputePool::new(3),
            &mut q8,
            &mut scales,
        )
        .unwrap();

        assert_eq!(
            output.map(f32::to_bits),
            [
                0xbe32_95aa,
                0x3be9_7100,
                0xbd1b_8ce0,
                0x3dae_d27f,
                0xbdef_03c1
            ]
        );
    }

    #[test]
    fn per_layer_projection_rejects_non_bf16_weight() {
        let weight = zero_weight(2, 1);
        let mut output = [7.0];
        let mut input_bf16 = [0; 4];
        let mut scales = [0.0];

        let error = matmul(
            "per_layer_model_proj.weight",
            &weight,
            &[1.0, 2.0],
            &mut output,
            &ComputePool::new(1),
            &mut input_bf16,
            &mut scales,
        )
        .unwrap_err();

        assert!(error.contains("requires BF16"), "{error}");
        assert_eq!(output, [7.0]);
    }

    #[test]
    fn per_layer_projection_rejects_wrong_bf16_storage_length() {
        for byte_len in [6, 2] {
            let bytes = vec![0; byte_len];
            let weight =
                Weight::from_quantized(QuantizedTensor::from_bytes(&bytes, GGMLType::BF16, 2, 1));
            let mut output = [7.0];
            let mut input_bf16 = [0; 4];
            let mut scales = [0.0];

            let error = matmul(
                "per_layer_model_proj.weight",
                &weight,
                &[1.0, 2.0],
                &mut output,
                &ComputePool::new(1),
                &mut input_bf16,
                &mut scales,
            )
            .unwrap_err();

            assert!(error.contains("expected 4 bytes"), "{error}");
            assert!(error.contains(&format!("got {byte_len}")), "{error}");
            assert_eq!(output, [7.0]);
        }
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
    fn post_kv_failure_leaves_session_state_unchanged() {
        let model = post_kv_failure_model();
        let mut session = super::Gemma4Session::new(&model, KvFormat::F32).unwrap();
        let rows = [Gemma4InputRow::Raw {
            values: vec![0.0; EMBED],
            per_layer_token: 0,
        }];

        for _ in 0..2 {
            let error = session.forward_rows(&rows).unwrap_err();
            assert!(error.contains("blk.0.attn_output.weight"), "{error}");
            assert_eq!(session.len(), 0);
            assert!(session
                .kv
                .iter()
                .all(|layer| layer.keys.is_empty() && layer.values.is_empty()));
        }
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
