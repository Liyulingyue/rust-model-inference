//! LFM3 layer weights + load helpers

use crate::core::tensor::TensorSource;
use crate::ops::kernel::{QuantizedTensor, Weight};

use super::config::Lfm3Config;

/// Per-layer weights for LFM3. The `wq`/`wk`/`wv`/`wo`/`q_norm`/`k_norm`
/// fields are populated only for attention layers; `shortconv_*` fields are
/// populated only for recurrent layers.
pub struct Lfm3LayerWeights<'a> {
    pub is_attn: bool,
    pub attn_norm: Vec<f32>,
    pub post_attention_norm: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub post_ffw_norm: Option<Vec<f32>>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub wq: Option<Weight<'a>>,
    pub wk: Option<Weight<'a>>,
    pub wv: Option<Weight<'a>>,
    pub wo: Option<Weight<'a>>,
    pub attn_gate: Option<Weight<'a>>,
    pub rope_freqs: Option<Vec<f32>>,
    pub w_gate: Weight<'a>,
    pub w_up: Weight<'a>,
    pub w_down: Weight<'a>,
    pub shortconv_in: Option<Weight<'a>>,
    pub shortconv_out: Option<Weight<'a>>,
    pub shortconv_conv: Option<Vec<f32>>,
}

fn get_f32_tensor_checked<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>, String> {
    crate::core::tensor::load_f32_tensor(source, name, &[expected_len as u64])
}

fn load_optional_norm<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected_len: usize,
) -> Result<Option<Vec<f32>>, String> {
    if source.tensor_info(name).is_some() {
        get_f32_tensor_checked(source, name, expected_len).map(Some)
    } else {
        Ok(None)
    }
}

pub(crate) fn quant_weight<'a>(
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
    if info.dims != [n_in as u64, n_out as u64] {
        return Err(format!(
            "{name} expected [{n_in}, {n_out}], got {:?}",
            info.dims
        ));
    }
    if info.checked_nbytes() != Some(bytes.len() as u64) {
        return Err(format!("{name} has invalid block alignment or byte length"));
    }
    use crate::core::tensor::GGMLType;
    if !matches!(
        info.ggml_type,
        GGMLType::F32
            | GGMLType::F16
            | GGMLType::BF16
            | GGMLType::Q8_0
            | GGMLType::Q4_0
            | GGMLType::Q4_1
            | GGMLType::Q4K
            | GGMLType::Q5K
            | GGMLType::Q6K
    ) {
        return Err(format!(
            "{name} uses unsupported weight format {:?}",
            info.ggml_type
        ));
    }
    Ok(Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        n_in,
        n_out,
    )))
}

pub fn load_layers<'a>(
    source: &'a dyn TensorSource,
    cfg: &Lfm3Config,
) -> Result<Vec<Lfm3LayerWeights<'a>>, String> {
    let mut layers = Vec::with_capacity(cfg.n_layer);
    for l in 0..cfg.n_layer {
        let is_attn = cfg.n_head_kv_per_layer[l] > 0;
        let attn_norm =
            get_f32_tensor_checked(source, &format!("blk.{l}.attn_norm.weight"), cfg.n_embd)?;
        let post_attention_norm = load_optional_norm(
            source,
            &format!("blk.{l}.post_attention_norm.weight"),
            cfg.n_embd,
        )?;
        let ffn_norm =
            get_f32_tensor_checked(source, &format!("blk.{l}.ffn_norm.weight"), cfg.n_embd)?;
        let post_ffw_norm =
            load_optional_norm(source, &format!("blk.{l}.post_ffw_norm.weight"), cfg.n_embd)?;
        if post_attention_norm.is_some() != post_ffw_norm.is_some() {
            return Err(format!(
                "layer {l} must provide both post_attention_norm and post_ffw_norm"
            ));
        }

        let (q_norm, k_norm, wq, wk, wv, wo, attn_gate) = if is_attn {
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
                Some(quant_weight(
                    source,
                    &format!("blk.{l}.attn_gate.weight"),
                    cfg.n_embd,
                    cfg.n_head,
                )?),
            )
        } else {
            (None, None, None, None, None, None, None)
        };

        let w_gate = quant_weight(
            source,
            &format!("blk.{l}.ffn_gate.weight"),
            cfg.n_embd,
            cfg.n_ff,
        )?;
        let w_up = quant_weight(
            source,
            &format!("blk.{l}.ffn_up.weight"),
            cfg.n_embd,
            cfg.n_ff,
        )?;
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
            shortconv_in = Some(quant_weight(source, &in_name, cfg.n_embd, 3 * cfg.n_embd)?);
            shortconv_out = Some(quant_weight(source, &out_name, cfg.n_embd, cfg.n_embd)?);
            // The conv kernel ships either 1-D [l_cache * n_embd] or 2-D
            // [l_cache, n_embd]; both flatten to the same c*l_cache + k order.
            let conv_name = format!("blk.{l}.shortconv.conv.weight");
            let conv = match get_f32_tensor_checked(source, &conv_name, cfg.l_cache * cfg.n_embd) {
                Ok(v) => v,
                Err(_) => crate::core::tensor::load_f32_tensor(
                    source,
                    &conv_name,
                    &[cfg.l_cache as u64, cfg.n_embd as u64],
                )?,
            };
            shortconv_conv = Some(conv);
        } else {
            shortconv_in = None;
            shortconv_out = None;
            shortconv_conv = None;
        }

        layers.push(Lfm3LayerWeights {
            is_attn,
            attn_norm,
            post_attention_norm,
            ffn_norm,
            post_ffw_norm,
            q_norm,
            k_norm,
            wq,
            wk,
            wv,
            wo,
            attn_gate,
            rope_freqs: if is_attn && !cfg.sliding_window_pattern[l] {
                let freqs =
                    get_f32_tensor_checked(source, "rope_freqs.weight", cfg.n_embd_head_k / 2)?;
                if freqs.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                    return Err("rope_freqs.weight must contain positive finite divisors".into());
                }
                Some(freqs)
            } else {
                None
            },
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
