//! # Qwen3 Weights — `Qwen3LayerWeights` + load helpers + `Qwen3Model` struct
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2.1:
//! - `Qwen3Model` struct lives here because its fields are weight tables
//!   (`token_embedding`, `output`, `layers`, `output_norm`).
//! - `Qwen3Model::from_source` + stateless accessors live here.
//! - The `text_encode` *method* lives in `forward.rs` (forward-loop concern);
//!   the `text_encode` *free function* lives in `forward.rs`.
//! - `Qwen3Input` / `Qwen3GenerateOptions` / `Qwen3Generation` are in `forward.rs`.
//! - `Qwen3Session` struct lives in `session.rs`.

use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::ops::bf16_to_f32;
use crate::ops::kernel::{Kernel, QuantizedTensor, Weight};
use std::sync::Arc;

pub use super::config::Qwen3Config;

// =============================================================================
// Qwen3Model struct (weight tables)
// =============================================================================

pub struct Qwen3Model {
    pub(crate) source: Arc<dyn TensorSource>,
    pub(crate) tokenizer: Arc<BPETokenizer>,
    pub(crate) pool: Arc<ComputePool>,
    pub(crate) config: Qwen3Config,
    pub(crate) layers: Vec<Qwen3LayerWeights<'static>>,
    pub(crate) output_norm: Vec<f32>,
    pub(crate) token_embedding: Weight<'static>,
    pub(crate) output: Weight<'static>,
}

// =============================================================================
// Qwen3LayerWeights (per-layer weight stack)
// =============================================================================

pub struct Qwen3LayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    pub moe_router: Option<Vec<f32>>,
    pub moe_gate: Option<Vec<Weight<'a>>>,
    pub moe_up: Option<Vec<Weight<'a>>>,
    pub moe_down: Option<Vec<Weight<'a>>>,
    pub wq: Weight<'a>,
    pub wk: Weight<'a>,
    pub wv: Weight<'a>,
    pub wo: Weight<'a>,
    pub w_gate: Weight<'a>,
    pub w_up: Weight<'a>,
    pub w_down: Weight<'a>,
}

// =============================================================================
// Load helpers
// =============================================================================

pub fn get_f32_tensor<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected_len: usize,
) -> Vec<f32> {
    let info = source
        .tensor_info(name)
        .unwrap_or_else(|| panic!("tensor {name} not found"));
    let bytes = source
        .tensor_slice(name)
        .unwrap_or_else(|| panic!("slice {name} not found"));
    let mut output = vec![0.0; expected_len];
    if info.ggml_type == GGMLType::F32 {
        for (value, chunk) in output.iter_mut().zip(bytes.chunks_exact(4)) {
            *value = f32::from_le_bytes(chunk.try_into().unwrap());
        }
    } else if info.ggml_type == GGMLType::BF16 {
        for (value, chunk) in output.iter_mut().zip(bytes.chunks_exact(2)) {
            *value = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
pub fn load_layers<'a>(
    source: &'a dyn TensorSource,
    n_layer: usize,
    n_embd: usize,
    n_embd_q: usize,
    n_embd_gqa: usize,
    n_ff: usize,
    n_embd_head_k: usize,
    has_qk_norm: bool,
) -> Vec<Qwen3LayerWeights<'a>> {
    (0..n_layer)
        .map(|l| Qwen3LayerWeights {
            attn_norm: get_f32_tensor(source, &format!("blk.{}.attn_norm.weight", l), n_embd),
            ffn_norm: get_f32_tensor(source, &format!("blk.{}.ffn_norm.weight", l), n_embd),
            q_norm: if has_qk_norm {
                Some(get_f32_tensor(
                    source,
                    &format!("blk.{}.attn_q_norm.weight", l),
                    n_embd_head_k,
                ))
            } else {
                None
            },
            k_norm: if has_qk_norm {
                Some(get_f32_tensor(
                    source,
                    &format!("blk.{}.attn_k_norm.weight", l),
                    n_embd_head_k,
                ))
            } else {
                None
            },
            q_bias: None,
            k_bias: None,
            v_bias: None,
            moe_router: None,
            moe_gate: None,
            moe_up: None,
            moe_down: None,
            wq: Weight::from_quantized(QuantizedTensor::from_bytes(
                source
                    .tensor_slice(&format!("blk.{}.attn_q.weight", l))
                    .unwrap(),
                source
                    .tensor_info(&format!("blk.{}.attn_q.weight", l))
                    .unwrap()
                    .ggml_type,
                n_embd,
                n_embd_q,
            )),
            wk: Weight::from_quantized(QuantizedTensor::from_bytes(
                source
                    .tensor_slice(&format!("blk.{}.attn_k.weight", l))
                    .unwrap(),
                source
                    .tensor_info(&format!("blk.{}.attn_k.weight", l))
                    .unwrap()
                    .ggml_type,
                n_embd,
                n_embd_gqa,
            )),
            wv: Weight::from_quantized(QuantizedTensor::from_bytes(
                source
                    .tensor_slice(&format!("blk.{}.attn_v.weight", l))
                    .unwrap(),
                source
                    .tensor_info(&format!("blk.{}.attn_v.weight", l))
                    .unwrap()
                    .ggml_type,
                n_embd,
                n_embd_gqa,
            )),
            wo: Weight::from_quantized(QuantizedTensor::from_bytes(
                source
                    .tensor_slice(&format!("blk.{}.attn_output.weight", l))
                    .unwrap(),
                source
                    .tensor_info(&format!("blk.{}.attn_output.weight", l))
                    .unwrap()
                    .ggml_type,
                n_embd_q,
                n_embd,
            )),
            w_gate: Weight::from_quantized(QuantizedTensor::from_bytes(
                source
                    .tensor_slice(&format!("blk.{}.ffn_gate.weight", l))
                    .unwrap(),
                source
                    .tensor_info(&format!("blk.{}.ffn_gate.weight", l))
                    .unwrap()
                    .ggml_type,
                n_embd,
                n_ff,
            )),
            w_up: Weight::from_quantized(QuantizedTensor::from_bytes(
                source
                    .tensor_slice(&format!("blk.{}.ffn_up.weight", l))
                    .unwrap(),
                source
                    .tensor_info(&format!("blk.{}.ffn_up.weight", l))
                    .unwrap()
                    .ggml_type,
                n_embd,
                n_ff,
            )),
            w_down: Weight::from_quantized(QuantizedTensor::from_bytes(
                source
                    .tensor_slice(&format!("blk.{}.ffn_down.weight", l))
                    .unwrap(),
                source
                    .tensor_info(&format!("blk.{}.ffn_down.weight", l))
                    .unwrap()
                    .ggml_type,
                n_ff,
                n_embd,
            )),
        })
        .collect()
}

pub fn static_weight(
    source: &dyn TensorSource,
    name: &str,
    rows: usize,
    cols: usize,
) -> Weight<'static> {
    let bytes = source
        .tensor_slice(name)
        .unwrap_or_else(|| panic!("tensor {name} not found"));
    let info = source
        .tensor_info(name)
        .unwrap_or_else(|| panic!("tensor info {name} not found"));
    let ggml_type = info.ggml_type;
    let bytes_static: &'static [u8] = unsafe { std::mem::transmute(bytes) };
    Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes_static,
        ggml_type,
        rows,
        cols,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn load_layers_static(
    source: Arc<dyn TensorSource>,
    n_layer: usize,
    n_embd: usize,
    n_embd_q: usize,
    n_embd_gqa: usize,
    n_ff: usize,
    n_embd_head_k: usize,
    has_qk_norm: bool,
    has_qkv_bias: bool,
    moe: Option<crate::core::loader::Qwen3MoeConfig>,
) -> Result<Vec<Qwen3LayerWeights<'static>>, String> {
    let source = source.as_ref();
    let mut layers = Vec::with_capacity(n_layer);
    for l in 0..n_layer {
        let q_bias = load_optional_bias(
            source,
            &format!("blk.{l}.attn_q.bias"),
            n_embd_q,
            has_qkv_bias,
        )?;
        let k_bias = load_optional_bias(
            source,
            &format!("blk.{l}.attn_k.bias"),
            n_embd_gqa,
            has_qkv_bias,
        )?;
        let v_bias = load_optional_bias(
            source,
            &format!("blk.{l}.attn_v.bias"),
            n_embd_gqa,
            has_qkv_bias,
        )?;
        let (w_gate, w_up, w_down, moe_router, moe_gate, moe_up, moe_down) = if let Some(moe) = moe
        {
            if moe.shared_expert_ffn != 0 {
                return Err("Qwen3VL-MoE shared experts are not supported".into());
            }
            let gate_slices = expert_slices(
                source,
                &format!("blk.{l}.ffn_gate_exps.weight"),
                n_embd,
                moe.expert_ffn,
                moe.expert_count,
            )?;
            let up_slices = expert_slices(
                source,
                &format!("blk.{l}.ffn_up_exps.weight"),
                n_embd,
                moe.expert_ffn,
                moe.expert_count,
            )?;
            let down_slices = expert_slices(
                source,
                &format!("blk.{l}.ffn_down_exps.weight"),
                moe.expert_ffn,
                n_embd,
                moe.expert_count,
            )?;
            let gate = gate_slices
                .iter()
                .map(|bytes| {
                    weight_from_bytes(
                        bytes,
                        source
                            .tensor_info(&format!("blk.{l}.ffn_gate_exps.weight"))
                            .unwrap()
                            .ggml_type,
                        n_embd,
                        moe.expert_ffn,
                    )
                })
                .collect::<Vec<_>>();
            let up = up_slices
                .iter()
                .map(|bytes| {
                    weight_from_bytes(
                        bytes,
                        source
                            .tensor_info(&format!("blk.{l}.ffn_up_exps.weight"))
                            .unwrap()
                            .ggml_type,
                        n_embd,
                        moe.expert_ffn,
                    )
                })
                .collect::<Vec<_>>();
            let down = down_slices
                .iter()
                .map(|bytes| {
                    weight_from_bytes(
                        bytes,
                        source
                            .tensor_info(&format!("blk.{l}.ffn_down_exps.weight"))
                            .unwrap()
                            .ggml_type,
                        moe.expert_ffn,
                        n_embd,
                    )
                })
                .collect::<Vec<_>>();
            let router = crate::core::tensor::load_f32_tensor(
                source,
                &format!("blk.{l}.ffn_gate_inp.weight"),
                &[n_embd as u64, moe.expert_count as u64],
            )?;
            (
                weight_from_bytes(
                    gate_slices[0],
                    source
                        .tensor_info(&format!("blk.{l}.ffn_gate_exps.weight"))
                        .unwrap()
                        .ggml_type,
                    n_embd,
                    moe.expert_ffn,
                ),
                weight_from_bytes(
                    up_slices[0],
                    source
                        .tensor_info(&format!("blk.{l}.ffn_up_exps.weight"))
                        .unwrap()
                        .ggml_type,
                    n_embd,
                    moe.expert_ffn,
                ),
                weight_from_bytes(
                    down_slices[0],
                    source
                        .tensor_info(&format!("blk.{l}.ffn_down_exps.weight"))
                        .unwrap()
                        .ggml_type,
                    moe.expert_ffn,
                    n_embd,
                ),
                Some(router),
                Some(gate),
                Some(up),
                Some(down),
            )
        } else {
            (
                static_weight(source, &format!("blk.{}.ffn_gate.weight", l), n_embd, n_ff),
                static_weight(source, &format!("blk.{}.ffn_up.weight", l), n_embd, n_ff),
                static_weight(source, &format!("blk.{}.ffn_down.weight", l), n_ff, n_embd),
                None,
                None,
                None,
                None,
            )
        };
        layers.push(Qwen3LayerWeights {
            attn_norm: get_f32_tensor(source, &format!("blk.{}.attn_norm.weight", l), n_embd),
            ffn_norm: get_f32_tensor(source, &format!("blk.{}.ffn_norm.weight", l), n_embd),
            q_norm: if has_qk_norm {
                Some(get_f32_tensor(
                    source,
                    &format!("blk.{}.attn_q_norm.weight", l),
                    n_embd_head_k,
                ))
            } else {
                None
            },
            q_bias,
            k_bias,
            v_bias,
            k_norm: if has_qk_norm {
                Some(get_f32_tensor(
                    source,
                    &format!("blk.{}.attn_k_norm.weight", l),
                    n_embd_head_k,
                ))
            } else {
                None
            },
            wq: static_weight(
                source,
                &format!("blk.{}.attn_q.weight", l),
                n_embd,
                n_embd_q,
            ),
            wk: static_weight(
                source,
                &format!("blk.{}.attn_k.weight", l),
                n_embd,
                n_embd_gqa,
            ),
            wv: static_weight(
                source,
                &format!("blk.{}.attn_v.weight", l),
                n_embd,
                n_embd_gqa,
            ),
            wo: static_weight(
                source,
                &format!("blk.{}.attn_output.weight", l),
                n_embd_q,
                n_embd,
            ),
            w_gate,
            w_up,
            w_down,
            moe_router,
            moe_gate,
            moe_up,
            moe_down,
        });
    }
    Ok(layers)
}

fn weight_from_bytes(
    bytes: &'static [u8],
    ggml_type: GGMLType,
    n_in: usize,
    n_out: usize,
) -> Weight<'static> {
    Weight::from_quantized(QuantizedTensor::from_bytes(bytes, ggml_type, n_in, n_out))
}

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
    if bytes.len() != total || total % expert_count != 0 {
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

fn load_optional_bias(
    source: &dyn TensorSource,
    name: &str,
    expected_len: usize,
    required: bool,
) -> Result<Option<Vec<f32>>, String> {
    if source.tensor_info(name).is_none() {
        return if required {
            Err(format!("Missing tensor: {name}"))
        } else {
            Ok(None)
        };
    }
    let dims = [u64::try_from(expected_len).map_err(|_| format!("{name} length overflow"))?];
    crate::core::tensor::load_f32_tensor(source, name, &dims).map(Some)
}

// =============================================================================
// Qwen3Model impl (constructor + stateless accessors)
// =============================================================================

impl Qwen3Model {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        tokenizer: Arc<BPETokenizer>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        use super::util::{
            check_allocation, checked_product, load_f32_tensor, usize_to_u64, validate_token_ids,
        };

        let config = Qwen3Config::from_source(source.as_ref())?;
        if config.vocab != tokenizer.vocab_size() {
            return Err(format!(
                "{} vocabulary size {} does not match tokenizer vocab {}",
                config.architecture,
                config.vocab,
                tokenizer.vocab_size()
            ));
        }
        let _n_embd_q = checked_product("query width", config.n_head, config.n_embd_head_k)?;
        let _n_embd_k = checked_product("key width", config.n_head_kv, config.n_embd_head_k)?;
        let _n_embd_v = checked_product("value width", config.n_head_kv, config.n_embd_head_v)?;

        let output_norm = load_f32_tensor(
            source.as_ref(),
            "output_norm.weight",
            &[usize_to_u64(config.n_embd, "embedding width")?],
        )?;
        let token_embedding_info = source
            .tensor_info("token_embd.weight")
            .expect("no token_embd.weight");
        let token_embedding_bytes = source.tensor_slice("token_embd.weight").expect("no embd");
        let token_embedding_bytes_static: &'static [u8] =
            unsafe { std::mem::transmute(token_embedding_bytes) };
        let token_embedding = Weight::from_quantized(QuantizedTensor::from_bytes(
            token_embedding_bytes_static,
            token_embedding_info.ggml_type,
            config.n_embd,
            config.vocab,
        ));

        let output_info = source
            .tensor_info("output.weight")
            .unwrap_or(token_embedding_info);
        let output_bytes = source
            .tensor_slice("output.weight")
            .unwrap_or(token_embedding_bytes);
        let output_bytes_static: &'static [u8] = unsafe { std::mem::transmute(output_bytes) };
        let output = Weight::from_quantized(QuantizedTensor::from_bytes(
            output_bytes_static,
            output_info.ggml_type,
            config.n_embd,
            config.vocab,
        ));

        let layers: Vec<Qwen3LayerWeights<'static>> = load_layers_static(
            Arc::clone(&source),
            config.n_layer,
            config.n_embd,
            checked_product("query width", config.n_head, config.n_embd_head_k)?,
            checked_product("key width", config.n_head_kv, config.n_embd_head_k)?,
            config.n_ff,
            config.n_embd_head_k,
            config.has_qk_norm,
            config.has_qkv_bias,
            config.moe,
        )?;

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

    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    pub fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }

    pub fn pool(&self) -> Arc<ComputePool> {
        Arc::clone(&self.pool)
    }

    pub fn layers(&self) -> &Vec<Qwen3LayerWeights> {
        &self.layers
    }

    pub fn output_norm(&self) -> &Vec<f32> {
        &self.output_norm
    }

    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        use super::util::{check_allocation, checked_product, validate_token_ids};

        validate_token_ids(token_ids, self.config.vocab)?;
        let len = checked_product(
            "token embedding values",
            token_ids.len(),
            self.config.n_embd,
        )?;
        check_allocation("token embeddings", len, std::mem::size_of::<f32>())?;
        let mut embeddings = vec![0.0; len];
        for (row, &token_id) in embeddings
            .chunks_exact_mut(self.config.n_embd)
            .zip(token_ids)
        {
            self.token_embedding.embedding_lookup(token_id, row);
        }
        Ok(embeddings)
    }
}
