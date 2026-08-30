//! LFM2-MoE layer weights + load helpers

use crate::core::tensor::TensorSource;
use crate::ops::kernel::{QuantizedTensor, Weight};

use super::config::Lfm2MoeConfig;

/// Per-layer weights for LFM2-MoE. The `wq`/`wk`/`wv`/`wo`/`q_norm`/`k_norm`
/// fields are populated only for attention layers; `shortconv_*` fields only
/// for recurrent layers. The dense `w_gate`/`w_up`/`w_down` fields are
/// populated for the leading dense-FFN blocks; the `router`/`exp_probs_b`/
/// `experts_*` fields only for MoE blocks.
pub struct Lfm2MoeLayerWeights<'a> {
    pub is_attn: bool,
    pub is_moe: bool,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub wq: Option<Weight<'a>>,
    pub wk: Option<Weight<'a>>,
    pub wv: Option<Weight<'a>>,
    pub wo: Option<Weight<'a>>,
    pub w_gate: Option<Weight<'a>>,
    pub w_up: Option<Weight<'a>>,
    pub w_down: Option<Weight<'a>>,
    pub shortconv_in: Option<Weight<'a>>,
    pub shortconv_out: Option<Weight<'a>>,
    pub shortconv_conv: Option<Vec<f32>>,
    /// Router projection `[n_embd * n_expert]`, row-major per expert
    /// (`ffn_gate_inp.weight`, F32).
    pub router: Vec<f32>,
    /// Expert-selection bias added to the router probabilities.
    pub exp_probs_b: Vec<f32>,
    /// Per-expert gate projections, sliced out of the `[n_embd, n_ff_exp,
    /// n_expert]` `ffn_gate_exps` tensor (each expert is a contiguous span).
    pub experts_gate: Vec<Weight<'a>>,
    pub experts_up: Vec<Weight<'a>>,
    pub experts_down: Vec<Weight<'a>>,
}

pub fn get_f32_tensor<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected_len: usize,
) -> Vec<f32> {
    crate::core::tensor::load_f32_tensor(source, name, &[expected_len as u64])
        .unwrap_or_else(|e| panic!("{e}"))
}

fn get_f32_tensor_checked<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>, String> {
    crate::core::tensor::load_f32_tensor(source, name, &[expected_len as u64])
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

/// Split a 3-D expert tensor `[n_in, n_ff_exp, n_expert]` into per-expert 2-D
/// weights. GGML layouts flatten ne[0] fastest, so each expert occupies one
/// contiguous span of `n_in * n_ff_exp` elements.
fn expert_weights<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    n_expert: usize,
    n_in: usize,
    n_ff_exp: usize,
) -> Result<Vec<Weight<'a>>, String> {
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("tensor {name} not found"))?;
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("tensor info {name} not found"))?;
    if bytes.len() % n_expert != 0 {
        return Err(format!("{name}: size not divisible by {n_expert} experts"));
    }
    let per_expert = bytes.len() / n_expert;
    Ok((0..n_expert)
        .map(|e| {
            Weight::from_quantized(QuantizedTensor::from_bytes(
                &bytes[e * per_expert..(e + 1) * per_expert],
                info.ggml_type,
                n_in,
                n_ff_exp,
            ))
        })
        .collect())
}

pub fn load_layers<'a>(
    source: &'a dyn TensorSource,
    cfg: &Lfm2MoeConfig,
) -> Result<Vec<Lfm2MoeLayerWeights<'a>>, String> {
    let mut layers = Vec::with_capacity(cfg.n_layer);
    for l in 0..cfg.n_layer {
        let is_attn = cfg.n_head_kv_per_layer[l] > 0;
        let is_moe = l >= cfg.n_layer_dense_lead;
        let attn_norm =
            get_f32_tensor_checked(source, &format!("blk.{l}.attn_norm.weight"), cfg.n_embd)?;
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

        let (w_gate, w_up, w_down, router, exp_probs_b, experts_gate, experts_up, experts_down) =
            if is_moe {
                (
                    None,
                    None,
                    None,
                    crate::core::tensor::load_f32_tensor(
                        source,
                        &format!("blk.{l}.ffn_gate_inp.weight"),
                        // [n_embd, n_expert]; flat order = e*n_embd + i,
                        // matching the forward's per-expert row indexing.
                        &[cfg.n_embd as u64, cfg.n_expert as u64],
                    )?,
                    get_f32_tensor_checked(
                        source,
                        &format!("blk.{l}.exp_probs_b.bias"),
                        cfg.n_expert,
                    )?,
                    expert_weights(
                        source,
                        &format!("blk.{l}.ffn_gate_exps.weight"),
                        cfg.n_expert,
                        cfg.n_embd,
                        cfg.n_ff_exp,
                    )?,
                    expert_weights(
                        source,
                        &format!("blk.{l}.ffn_up_exps.weight"),
                        cfg.n_expert,
                        cfg.n_embd,
                        cfg.n_ff_exp,
                    )?,
                    expert_weights(
                        source,
                        &format!("blk.{l}.ffn_down_exps.weight"),
                        cfg.n_expert,
                        cfg.n_ff_exp,
                        cfg.n_embd,
                    )?,
                )
            } else {
                (
                    Some(quant_weight(
                        source,
                        &format!("blk.{l}.ffn_gate.weight"),
                        cfg.n_embd,
                        cfg.n_ff,
                    )?),
                    Some(quant_weight(
                        source,
                        &format!("blk.{l}.ffn_up.weight"),
                        cfg.n_embd,
                        cfg.n_ff,
                    )?),
                    Some(quant_weight(
                        source,
                        &format!("blk.{l}.ffn_down.weight"),
                        cfg.n_ff,
                        cfg.n_embd,
                    )?),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            };

        let (shortconv_in, shortconv_out, shortconv_conv) = if !is_attn {
            (
                Some(quant_weight(
                    source,
                    &format!("blk.{l}.shortconv.in_proj.weight"),
                    cfg.n_embd,
                    3 * cfg.n_embd,
                )?),
                Some(quant_weight(
                    source,
                    &format!("blk.{l}.shortconv.out_proj.weight"),
                    cfg.n_embd,
                    cfg.n_embd,
                )?),
                Some(crate::core::tensor::load_f32_tensor(
                    source,
                    &format!("blk.{l}.shortconv.conv.weight"),
                    // This arch stores the conv kernel 2-D [l_cache, n_embd];
                    // the flat return order (ne[0] fastest = c*l_cache + k)
                    // matches the forward's kernel indexing.
                    &[cfg.l_cache as u64, cfg.n_embd as u64],
                )?),
            )
        } else {
            (None, None, None)
        };

        layers.push(Lfm2MoeLayerWeights {
            is_attn,
            is_moe,
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
            router,
            exp_probs_b,
            experts_gate,
            experts_up,
            experts_down,
        });
    }
    Ok(layers)
}
