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

use super::config::Qwen35Config;
use super::util::f16_at;
use crate::core::tensor::GGMLType;
use crate::core::tensor::TensorSource;
use crate::ops::kernel::{QuantizedTensor, Weight};

// =============================================================================
// Model + Layer-weight structs
// =============================================================================

/// All weights for a single Qwen3.5 layer.
///
/// The `Option` fields distinguish dense-attention layers (which fill
/// `wq`/`wk`/`wv`/`wo`/`attn_q_norm`/`attn_k_norm`) from recurrent (Mamba
/// SSM) layers (which fill `wqkv`/`wqkv_gate`/`ssm_*`). `config.is_recurrent`
/// selects which group is active.
pub struct Qwen35LayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub attn_post_norm: Vec<f32>,
    pub wq: Option<Weight<'a>>,
    pub wk: Option<Weight<'a>>,
    pub wv: Option<Weight<'a>>,
    pub wo: Option<Weight<'a>>,
    pub attn_q_norm: Option<Vec<f32>>,
    pub attn_k_norm: Option<Vec<f32>>,
    pub wqkv: Option<Weight<'a>>,
    pub wqkv_gate: Option<Weight<'a>>,
    pub ssm_conv1d: Option<Vec<f32>>,
    pub ssm_dt: Option<Vec<f32>>,
    pub ssm_a: Option<Vec<f32>>,
    pub ssm_beta: Option<Weight<'a>>,
    pub ssm_alpha: Option<Weight<'a>>,
    pub ssm_norm: Option<Vec<f32>>,
    pub ssm_out: Option<Weight<'a>>,
    pub ffn_gate: Weight<'a>,
    pub ffn_up: Weight<'a>,
    pub ffn_down: Weight<'a>,
}

/// Loaded Qwen3.5 model weights + parsed config.
///
/// `from_source` is defined in `weights.rs`. `forward` and friends are
/// defined in `forward.rs`. This struct is the source of truth shared by
/// `Qwen35Session` and the existing `app/text.rs` / `bin/server.rs`
/// call sites.
pub struct Qwen35Model<'a> {
    pub config: Qwen35Config,
    pub tok_embd: Weight<'a>,
    pub output_norm: Vec<f32>,
    pub output_weight: Weight<'a>,
    pub layers: Vec<Qwen35LayerWeights<'a>>,
}

// Convenience alias so that `impl Qwen35Model { fn from_source(...) }` in
// `weights.rs` and `impl Qwen35Model { fn forward(...) }` in `forward.rs`
// can refer to a common TensorSource without redundant imports.
pub(crate) type Source<'a> = &'a dyn TensorSource;

/// Load a quantized weight (F32/F16/Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K) into a
/// `Weight` borrowing the GGUF bytes. Returns `None` if the tensor is
/// missing or the dtype is not supported (with a stderr warning).
pub(crate) fn load_weight<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
) -> Option<Weight<'a>> {
    let ti = source.tensor_info(name)?;
    let data = source.tensor_slice(name)?;
    let n_cols = ti.dims[0] as usize;
    let n_rows = if ti.dims.len() >= 2 {
        ti.dims[1] as usize
    } else {
        1
    };

    match ti.ggml_type {
        GGMLType::F32
        | GGMLType::F16
        | GGMLType::BF16
        | GGMLType::Q8_0
        | GGMLType::Q4_0
        | GGMLType::Q4_1
        | GGMLType::Q4K
        | GGMLType::Q5K
        | GGMLType::Q6K => {
            let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
                data,
                ti.ggml_type,
                n_cols,
                n_rows,
            ));
            weight.n_in = n_cols;
            weight.n_out = n_rows;
            Some(weight)
        }
        _ => {
            eprintln!(
                "WARNING: unsupported quant type {:?} for tensor {}",
                ti.ggml_type, name
            );
            None
        }
    }
}

/// Load an F32-or-F16 norm/bias tensor into an owned `Vec<f32>`. Used for
/// non-matmul tensors (norms, biases, conv1d, dt.bias, A-log).
pub(crate) fn load_weight_f32<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
) -> Option<Vec<f32>> {
    let ti = source.tensor_info(name)?;
    let data = source.tensor_slice(name)?;
    let n_el = ti.n_elements();
    match ti.ggml_type {
        GGMLType::F32 => {
            let mut out = Vec::with_capacity(n_el);
            for i in 0..n_el {
                let off = i * 4;
                if off + 4 <= data.len() {
                    out.push(f32::from_le_bytes([
                        data[off],
                        data[off + 1],
                        data[off + 2],
                        data[off + 3],
                    ]));
                } else {
                    out.push(0.0);
                }
            }
            Some(out)
        }
        GGMLType::F16 => {
            let mut out = Vec::with_capacity(n_el);
            for i in 0..n_el {
                out.push(f16_at(data, i));
            }
            Some(out)
        }
        GGMLType::BF16 => {
            let mut out = Vec::with_capacity(n_el);
            for i in 0..n_el {
                let off = i * 2;
                if off + 2 <= data.len() {
                    out.push(crate::ops::bf16_to_f32(u16::from_le_bytes([
                        data[off],
                        data[off + 1],
                    ])));
                } else {
                    out.push(0.0);
                }
            }
            Some(out)
        }
        _ => None,
    }
}

impl<'a> Qwen35Model<'a> {
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        let config = Qwen35Config::from_source(source)?;

        let token_info = source
            .tensor_info("token_embd.weight")
            .ok_or("Missing token_embd.weight")?;
        let actual = token_info
            .dims
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>();
        let expected = vec![config.n_embd, config.vocab_size];
        if actual != expected {
            return Err(format!(
                "token_embd.weight shape mismatch: expected {expected:?}, got {actual:?}, dtype={:?}",
                token_info.ggml_type
            ));
        }
        let tok_embd = load_weight(source, "token_embd.weight").ok_or_else(|| {
            format!(
                "Unsupported token_embd.weight dtype: {:?}",
                token_info.ggml_type
            )
        })?;

        let output_norm =
            load_weight_f32(source, "output_norm.weight").ok_or("Missing output_norm.weight")?;

        let output_weight = {
            let name = if source.tensor_info("output.weight").is_some() {
                "output.weight"
            } else {
                "token_embd.weight"
            };
            load_weight(source, name).ok_or("Missing output weight")?
        };

        let n_layers_impl = config.n_layer_impl();
        let mut layers = Vec::with_capacity(n_layers_impl);
        for i in 0..n_layers_impl {
            let attn_norm = load_weight_f32(source, &format!("blk.{}.attn_norm.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.attn_norm.weight", i))?;
            let attn_post_norm =
                load_weight_f32(source, &format!("blk.{}.post_attention_norm.weight", i))
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
            } else {
                (None, None, None, None, None, None)
            };

            let (
                wqkv,
                wqkv_gate,
                ssm_conv1d,
                ssm_dt,
                ssm_a,
                ssm_beta,
                ssm_alpha,
                ssm_norm,
                ssm_out,
            ) = if is_recr {
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
            } else {
                (None, None, None, None, None, None, None, None, None)
            };

            let ffn_gate = load_weight(source, &format!("blk.{}.ffn_gate.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.ffn_gate.weight", i))?;
            let ffn_up = load_weight(source, &format!("blk.{}.ffn_up.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.ffn_up.weight", i))?;
            let ffn_down = load_weight(source, &format!("blk.{}.ffn_down.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.ffn_down.weight", i))?;
            layers.push(Qwen35LayerWeights {
                attn_norm,
                attn_post_norm,
                wq,
                wk,
                wv,
                wo,
                attn_q_norm,
                attn_k_norm,
                wqkv,
                wqkv_gate,
                ssm_conv1d,
                ssm_dt,
                ssm_a,
                ssm_beta,
                ssm_alpha,
                ssm_norm,
                ssm_out,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        Ok(Self {
            config,
            tok_embd,
            output_norm,
            output_weight,
            layers,
        })
    }

    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        if let Some(&token_id) = token_ids
            .iter()
            .find(|&&token_id| token_id as usize >= self.tok_embd.n_out)
        {
            return Err(format!(
                "Qwen3.5 token id {token_id} out of range (vocab={})",
                self.tok_embd.n_out
            ));
        }
        let len = token_ids
            .len()
            .checked_mul(self.config.n_embd)
            .ok_or("Qwen3.5 token embedding length overflow")?;
        let mut embeddings = vec![0.0; len];
        for (row, &token_id) in embeddings
            .chunks_exact_mut(self.config.n_embd)
            .zip(token_ids)
        {
            self.tok_embd.embedding_lookup(token_id, row);
        }
        Ok(embeddings)
    }
}
