//! GGUF → Qwen35Model weight loading.
//!
//! `Qwen35Model::from_source` reads a GGUF `TensorSource` and builds
//! `Qwen35LayerWeights` rows, one per layer. Recurrent (Mamba) layers only
//! fill the SSM field group; dense (attention) layers only fill the
//! attention field group — see `config.is_recurrent`.
//!
//! `load_weight` and `load_weight_f32` are the per-tensor helpers used
//! by `from_source`. They are `pub(crate)` because they are only useful
//! inside this module's loading path.

use crate::core::tensor::{GGMLType, TensorSource};
use crate::models::qwen35::clip_config::Qwen35Config;
use crate::models::qwen35::util::f16_at;
use crate::ops::kernel::{QuantizedTensor, Weight};
use crate::ops::quant;
use crate::models::qwen35::Qwen35LayerWeights;

/// Load a quantized weight (F32/F16/Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K) into a
/// `Weight` borrowing the GGUF bytes. Returns `None` if the tensor is
/// missing or the dtype is not supported (with a stderr warning).
pub(crate) fn load_weight<'a, S: TensorSource + ?Sized>(source: &'a S, name: &str) -> Option<Weight<'a>> {
    let ti = source.tensor_info(name)?;
    let data = source.tensor_slice(name)?;
    let n_cols = ti.dims[0] as usize;
    let n_rows = if ti.dims.len() >= 2 { ti.dims[1] as usize } else { 1 };

    match ti.ggml_type {
        GGMLType::F32 | GGMLType::F16 | GGMLType::Q8_0 | GGMLType::Q4_0
        | GGMLType::Q4_1 | GGMLType::Q4K | GGMLType::Q5K | GGMLType::Q6K => {
            Some(Weight::from_quantized(QuantizedTensor::from_bytes(data, ti.ggml_type, n_cols, n_rows)))
        }
        _ => {
            eprintln!("WARNING: unsupported quant type {:?} for tensor {}", ti.ggml_type, name);
            None
        }
    }
}

/// Load an F32-or-F16 norm/bias tensor into an owned `Vec<f32>`. Used for
/// non-matmul tensors (norms, biases, conv1d, dt.bias, A-log).
pub(crate) fn load_weight_f32<S: TensorSource + ?Sized>(source: &S, name: &str) -> Option<Vec<f32>> {
    let ti = source.tensor_info(name)?;
    let data = source.tensor_slice(name)?;
    let n_el = ti.n_elements();
    match ti.ggml_type {
        GGMLType::F32 => {
            let mut out = Vec::with_capacity(n_el);
            for i in 0..n_el {
                let off = i * 4;
                if off + 4 <= data.len() {
                    out.push(f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]));
                } else { out.push(0.0); }
            }
            Some(out)
        }
        GGMLType::F16 => {
            let mut out = Vec::with_capacity(n_el);
            for i in 0..n_el { out.push(f16_at(data, i)); }
            Some(out)
        }
        _ => None,
    }
}

impl<'a> crate::models::qwen35::Qwen35Model<'a> {
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        let config = Qwen35Config::from_source(source)?;

        let tok_embd = {
            let ti = source.tensor_info("token_embd.weight").ok_or("Missing token_embd.weight")?;
            let data = source.tensor_slice("token_embd.weight").unwrap();
            let n_cols = ti.dims[0] as usize;
            let n_rows = ti.dims[1] as usize;
            match ti.ggml_type {
                GGMLType::F16 => (0..n_cols * n_rows).map(|i| f16_at(data, i)).collect(),
                GGMLType::Q8_0 => quant::dequant_q80_weight(data, n_cols, n_rows),
                GGMLType::Q6K => quant::dequant_q6k_weight(data, n_cols, n_rows),
                _ => return Err("Unsupported token_embd type".into()),
            }
        };

        let output_norm = load_weight_f32(source, "output_norm.weight").ok_or("Missing output_norm.weight")?;

        let output_weight = {
            let name = if source.tensor_info("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
            load_weight(source, name).ok_or("Missing output weight")?
        };

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let attn_norm = load_weight_f32(source, &format!("blk.{}.attn_norm.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.attn_norm.weight", i))?;
            let attn_post_norm = load_weight_f32(source, &format!("blk.{}.post_attention_norm.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.post_attention_norm.weight", i))?;
            let is_recr = config.is_recurrent[i];

            let (wq, wk, wv, wo, attn_q_norm, attn_k_norm) = if !is_recr {
                (
                    load_weight(source, &format!("blk.{}.attn_q.weight", i)),
                    load_weight(source, &format!("blk.{}.attn_k.weight", i)),
                    load_weight(source, &format!("blk.{}.attn_v.weight", i)),
                    load_weight(source, &format!("blk.{}.attn_output.weight", i)),
                    load_weight_f32(source, &format!("blk.{}.attn_q_norm.weight", i)),
                    load_weight_f32(source, &format!("blk.{}.attn_k_norm.weight", i)),
                )
            } else { (None, None, None, None, None, None) };

            let (wqkv, wqkv_gate, ssm_conv1d, ssm_dt, ssm_a, ssm_beta, ssm_alpha, ssm_norm, ssm_out) = if is_recr {
                (
                    load_weight(source, &format!("blk.{}.attn_qkv.weight", i)),
                    load_weight(source, &format!("blk.{}.attn_gate.weight", i)),
                    load_weight_f32(source, &format!("blk.{}.ssm_conv1d.weight", i)),
                    load_weight_f32(source, &format!("blk.{}.ssm_dt.bias", i)),
                    load_weight_f32(source, &format!("blk.{}.ssm_a", i)),
                    load_weight(source, &format!("blk.{}.ssm_beta.weight", i)),
                    load_weight(source, &format!("blk.{}.ssm_alpha.weight", i)),
                    load_weight_f32(source, &format!("blk.{}.ssm_norm.weight", i)),
                    load_weight(source, &format!("blk.{}.ssm_out.weight", i)),
                )
            } else { (None, None, None, None, None, None, None, None, None) };

            let ffn_gate = load_weight(source, &format!("blk.{}.ffn_gate.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.ffn_gate.weight", i))?;
            let ffn_up = load_weight(source, &format!("blk.{}.ffn_up.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.ffn_up.weight", i))?;
            let ffn_down = load_weight(source, &format!("blk.{}.ffn_down.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.ffn_down.weight", i))?;
            layers.push(Qwen35LayerWeights {
                attn_norm, attn_post_norm, wq, wk, wv, wo,
                attn_q_norm, attn_k_norm,
                wqkv, wqkv_gate, ssm_conv1d, ssm_dt, ssm_a, ssm_beta, ssm_alpha, ssm_norm, ssm_out,
                ffn_gate, ffn_up, ffn_down,
            });
        }

        Ok(Self { config, tok_embd, output_norm, output_weight, layers })
    }
}