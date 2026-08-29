//! # LFM2 Shared Skeleton
//!
//! Layer weights, configuration and loaders for LFM2 (Liquid Foundation
//! Model 2) hybrid architectures. Each layer is either:
//!
//! - **Attention** (`is_attn == true`): standard multi-head attention with
//!   Q/K norms, RoPE, and a per-layer KV head count > 0.
//! - **Recurrent** (`is_attn == false`): a short convolution over a
//!   persistent state tensor (LFM2's "shortconv" block) — there is no Q/K/V
//!   projection and no KV cache; instead the layer maintains a per-channel
//!   state of shape `[d_conv, n_embd]` where `d_conv = l_cache - 1`. The
//!   conv kernel itself has `l_cache` rows.
//!
//! FFN tensors (gate / up / down) exist for both layer types.
//!
//! Matmul weights are loaded via [`Weight::from_quantized`], which auto
//! dispatches to the per-dtype kernel (`Q8Kernel` for Q8_0 weights, etc.).

use crate::core::tensor::TensorSource;
use crate::ops::kernel::{QuantizedTensor, Weight};

/// Per-layer weights for LFM2. The `wq`/`wk`/`wv`/`wo`/`q_norm`/`k_norm`
/// fields are populated only for attention layers; `shortconv_*` fields are
/// populated only for recurrent layers.
pub struct Lfm2LayerWeights<'a> {
    pub is_attn: bool,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub wq: Option<Weight<'a>>,
    pub wk: Option<Weight<'a>>,
    pub wv: Option<Weight<'a>>,
    pub wo: Option<Weight<'a>>,
    pub w_gate: Weight<'a>,
    pub w_up: Weight<'a>,
    pub w_down: Weight<'a>,
    pub shortconv_in: Option<Weight<'a>>,
    pub shortconv_out: Option<Weight<'a>>,
    pub shortconv_conv: Option<Vec<f32>>,
}

/// LFM2 model configuration derived from GGUF metadata.
pub struct Lfm2Config {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub n_ff: usize,
    pub n_ctx: usize,
    pub vocab_size: usize,
    pub rope_freq_base: f32,
    pub norm_eps: f32,
    /// Per-layer head count KV (length = n_layer). A value of 0 marks the
    /// layer as recurrent (shortconv); non-zero marks attention.
    pub n_head_kv_per_layer: Vec<usize>,
    /// `d_conv = l_cache - 1` — the length of the conv state per channel.
    pub d_conv: usize,
    /// `l_cache` from metadata — the size of the conv kernel.
    pub l_cache: usize,
}

pub fn get_f32_tensor<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected_len: usize,
) -> Vec<f32> {
    crate::core::tensor::load_f32_tensor(source, name, &[expected_len as u64])
        .unwrap_or_else(|e| panic!("{e}"))
}

fn read_i32_array<S: TensorSource + ?Sized>(
    source: &S,
    key: &str,
    expected_len: usize,
) -> Result<Vec<i32>, String> {
    let value = source
        .metadata(key)
        .ok_or_else(|| format!("Missing metadata: {key}"))?;
    let crate::core::tensor::MetaValue::Array(_, items) = value else {
        return Err(format!("{key} is not an array"));
    };
    if items.len() != expected_len {
        return Err(format!(
            "{key} expected {expected_len} entries, got {}",
            items.len()
        ));
    }
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let v = match item {
            crate::core::tensor::MetaValue::Int32(v) => *v,
            crate::core::tensor::MetaValue::Uint32(v) => *v as i32,
            _ => return Err(format!("{key} has non-integer entries")),
        };
        out.push(v);
    }
    Ok(out)
}

impl Lfm2Config {
    pub fn from_source<S: TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        let get_u32 = |key: &str| -> Result<u32, String> {
            source
                .metadata(key)
                .and_then(crate::core::tensor::MetaValue::to_u64)
                .ok_or_else(|| format!("Missing metadata: {key}"))
                .and_then(|v| u32::try_from(v).map_err(|_| format!("{key} does not fit u32")))
        };
        let get_f32 = |key: &str| -> Result<f32, String> {
            source
                .metadata(key)
                .and_then(crate::core::tensor::MetaValue::to_f64)
                .ok_or_else(|| format!("Missing metadata: {key}"))
                .map(|v| v as f32)
        };
        let get_f32_opt = |key: &str, default: f32| -> Result<f32, String> {
            Ok(source
                .metadata(key)
                .and_then(crate::core::tensor::MetaValue::to_f64)
                .map(|v| v as f32)
                .unwrap_or(default))
        };

        let n_embd = get_u32("lfm2.embedding_length")? as usize;
        let n_layer = get_u32("lfm2.block_count")? as usize;
        let n_head = get_u32("lfm2.attention.head_count")? as usize;
        let n_ff = get_u32("lfm2.feed_forward_length")? as usize;
        let n_ctx = get_u32("lfm2.context_length")? as usize;
        let rope_freq_base = get_f32_opt("lfm2.rope.freq_base", 1_000_000.0)?;
        let norm_eps = get_f32("lfm2.attention.layer_norm_rms_epsilon")?;

        let n_embd_head_k = source
            .metadata("lfm2.attention.key_length")
            .and_then(crate::core::tensor::MetaValue::to_u64)
            .map(|v| v as usize)
            .unwrap_or(n_embd / n_head);
        let n_embd_head_v = source
            .metadata("lfm2.attention.value_length")
            .and_then(crate::core::tensor::MetaValue::to_u64)
            .map(|v| v as usize)
            .unwrap_or(n_embd_head_k);

        let head_kv_arr = read_i32_array(source, "lfm2.attention.head_count_kv", n_layer)?;
        let n_head_kv_per_layer: Vec<usize> =
            head_kv_arr.iter().map(|&v| v.max(0) as usize).collect();

        let l_cache = get_u32("lfm2.shortconv.l_cache")? as usize;
        let d_conv = l_cache.saturating_sub(1).max(1);

        let vocab_size = match get_u32("lfm2.vocab_size") {
            Ok(value) => value as usize,
            Err(_) => source
                .metadata("tokenizer.ggml.tokens")
                .and_then(crate::core::tensor::MetaValue::to_arr)
                .map(Vec::len)
                .unwrap_or(0),
        };

        Ok(Lfm2Config {
            n_embd,
            n_layer,
            n_head,
            n_embd_head_k,
            n_embd_head_v,
            n_ff,
            n_ctx,
            vocab_size,
            rope_freq_base,
            norm_eps,
            n_head_kv_per_layer,
            d_conv,
            l_cache,
        })
    }
}

fn quant_weight<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Weight<'a>, String> {
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("tensor {name} not found"))?;
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("tensor info {name} not found"))?;
    Ok(Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        n_in,
        n_out,
    )))
}

pub fn load_layers<'a>(
    source: &'a dyn TensorSource,
    cfg: &Lfm2Config,
) -> Result<Vec<Lfm2LayerWeights<'a>>, String> {
    let mut layers = Vec::with_capacity(cfg.n_layer);
    for l in 0..cfg.n_layer {
        let is_attn = cfg.n_head_kv_per_layer[l] > 0;
        let attn_norm = get_f32_tensor_checked(
            source,
            &format!("blk.{l}.attn_norm.weight"),
            cfg.n_embd,
        )?;
        let ffn_norm =
            get_f32_tensor_checked(source, &format!("blk.{l}.ffn_norm.weight"), cfg.n_embd)?;

        let (q_norm, k_norm, wq, wk, wv, wo) = if is_attn {
            let n_embd_q = cfg.n_head * cfg.n_embd_head_k;
            let n_embd_kv = cfg.n_head_kv_per_layer[l] * cfg.n_embd_head_k;
            (
                Some(get_f32_tensor_checked(
                    source,
                    &format!("blk.{l}.attn_q_norm.weight"),
                    cfg.n_embd_head_k,
                )?),
                Some(get_f32_tensor_checked(
                    source,
                    &format!("blk.{l}.attn_k_norm.weight"),
                    cfg.n_embd_head_k,
                )?),
                Some(quant_weight(
                    source,
                    &format!("blk.{l}.attn_q.weight"),
                    cfg.n_embd,
                    n_embd_q,
                )?),
                Some(quant_weight(
                    source,
                    &format!("blk.{l}.attn_k.weight"),
                    cfg.n_embd,
                    n_embd_kv,
                )?),
                Some(quant_weight(
                    source,
                    &format!("blk.{l}.attn_v.weight"),
                    cfg.n_embd,
                    n_embd_kv,
                )?),
                Some(quant_weight(
                    source,
                    &format!("blk.{l}.attn_output.weight"),
                    n_embd_q,
                    cfg.n_embd,
                )?),
            )
        } else {
            (None, None, None, None, None, None)
        };

        let w_gate = quant_weight(
            source,
            &format!("blk.{l}.ffn_gate.weight"),
            cfg.n_embd,
            cfg.n_ff,
        )?;
        let w_up =
            quant_weight(source, &format!("blk.{l}.ffn_up.weight"), cfg.n_embd, cfg.n_ff)?;
        let w_down = quant_weight(
            source,
            &format!("blk.{l}.ffn_down.weight"),
            cfg.n_ff,
            cfg.n_embd,
        )?;

        let shortconv_in: Option<Weight<'a>>;
        let shortconv_out: Option<Weight<'a>>;
        let shortconv_conv: Option<Vec<f32>>;
        if !is_attn {
            let in_name = format!("blk.{l}.shortconv.in_proj.weight");
            let out_name = format!("blk.{l}.shortconv.out_proj.weight");
            shortconv_in = Some(quant_weight(
                source,
                &in_name,
                cfg.n_embd,
                3 * cfg.n_embd,
            )?);
            shortconv_out = Some(quant_weight(
                source,
                &out_name,
                cfg.n_embd,
                cfg.n_embd,
            )?);
            shortconv_conv = Some(get_f32_tensor_checked(
                source,
                &format!("blk.{l}.shortconv.conv.weight"),
                cfg.l_cache * cfg.n_embd,
            )?);
        } else {
            shortconv_in = None;
            shortconv_out = None;
            shortconv_conv = None;
        }

        layers.push(Lfm2LayerWeights {
            is_attn,
            attn_norm,
            ffn_norm,
            q_norm,
            k_norm,
            wq,
            wk,
            wv,
            wo,
            w_gate,
            w_up,
            w_down,
            shortconv_in,
            shortconv_out,
            shortconv_conv,
        });
    }
    Ok(layers)
}

fn get_f32_tensor_checked<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>, String> {
    crate::core::tensor::load_f32_tensor(source, name, &[expected_len as u64])
}