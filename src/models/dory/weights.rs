//! Dory GGUF tensor loading and shape validation.

use crate::core::tensor::{load_f32_tensor, GGMLType, TensorSource};
use crate::ops::kernel::{QuantizedTensor, Weight};

use super::config::DoryConfig;

pub enum DoryLayerWeights<'a> {
    Attention {
        residual_alpha: Vec<f32>,
        gate_scale: Vec<f32>,
        qk_scale: Vec<f32>,
        w_gate: Weight<'a>,
        wk: Weight<'a>,
        wo: Weight<'a>,
        wq: Weight<'a>,
        wv: Weight<'a>,
    },
    Mlp {
        residual_alpha: Vec<f32>,
        suv_gate: Vec<f32>,
        suv_up: Vec<f32>,
        w_down: Weight<'a>,
        w_gate: Weight<'a>,
        w_up: Weight<'a>,
    },
}

fn supported(t: GGMLType) -> bool {
    matches!(
        t,
        GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 | GGMLType::Q5K | GGMLType::Q6K
    )
}

pub fn load_weight<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Weight<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [n_in as u64, n_out as u64] || !supported(info.ggml_type) {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected [{n_in}, {n_out}] F32/F16/BF16/Q5K/Q6K",
            info.dims, info.ggml_type
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if Some(bytes.len() as u64) != info.checked_nbytes() {
        return Err(format!("Invalid tensor byte length: {name}"));
    }
    let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        n_in,
        n_out,
    ));
    // Recurrent activation quantization amplifies SIMD reduction rounding.
    weight.kernel.use_scalar_reference();
    Ok(weight)
}

fn vec(source: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    load_f32_tensor(source, name, &[len as u64])
}

pub fn load_layers<'a>(
    source: &'a dyn TensorSource,
    cfg: &DoryConfig,
) -> Result<Vec<DoryLayerWeights<'a>>, String> {
    let q = cfg.n_head * cfg.head_dim;
    let kv = cfg.n_head_kv * cfg.head_dim;
    let mut layers = Vec::with_capacity(cfg.n_layer);
    for layer in 0..cfg.n_layer {
        let p = format!("blk.{layer}");
        let residual_alpha = vec(source, &format!("{p}.residual_alpha.weight"), cfg.n_embd)?;
        if cfg.is_attention(layer) {
            layers.push(DoryLayerWeights::Attention {
                residual_alpha,
                gate_scale: vec(source, &format!("{p}.attn_gate_scale.weight"), q)?,
                qk_scale: vec(source, &format!("{p}.attn_qk_scale.weight"), q)?,
                w_gate: load_weight(source, &format!("{p}.attn_gate.weight"), cfg.n_embd, q)?,
                wk: load_weight(source, &format!("{p}.attn_k.weight"), cfg.n_embd, kv)?,
                wo: load_weight(source, &format!("{p}.attn_output.weight"), q, cfg.n_embd)?,
                wq: load_weight(source, &format!("{p}.attn_q.weight"), cfg.n_embd, q)?,
                wv: load_weight(source, &format!("{p}.attn_v.weight"), cfg.n_embd, kv)?,
            });
        } else {
            layers.push(DoryLayerWeights::Mlp {
                residual_alpha,
                suv_gate: vec(source, &format!("{p}.ffn_gate_scale.weight"), cfg.n_ff)?,
                suv_up: vec(source, &format!("{p}.ffn_up_scale.weight"), cfg.n_ff)?,
                w_down: load_weight(
                    source,
                    &format!("{p}.ffn_down.weight"),
                    cfg.n_ff,
                    cfg.n_embd,
                )?,
                w_gate: load_weight(
                    source,
                    &format!("{p}.ffn_gate.weight"),
                    cfg.n_embd,
                    cfg.n_ff,
                )?,
                w_up: load_weight(source, &format!("{p}.ffn_up.weight"), cfg.n_embd, cfg.n_ff)?,
            });
        }
    }
    Ok(layers)
}
