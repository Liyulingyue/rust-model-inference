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
use crate::ops::kernel::mlx_affine::MlxAffineKernel;
use crate::ops::kernel::{QuantizedTensor, Weight};
#[cfg(feature = "vulkan")]
use crate::vulkan::qwen35::Qwen35VulkanSession;

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
    pub edge0_moe: Option<Vec<Edge0MoeWeights<'a>>>,
    /// Lazily-initialized Vulkan session. Built on the first decode token
    /// after `--gpu` enables the global Vulkan context. `None` means CPU
    /// fallback (no eligible GPU or Vulkan init failed).
    #[cfg(feature = "vulkan")]
    pub(crate) gpu: Option<Qwen35VulkanSession>,
}

pub struct Edge0MoeWeights<'a> {
    pub router: Weight<'a>,
    pub shared_gate: Weight<'a>,
    pub gate: Vec<Weight<'a>>,
    pub up: Vec<Weight<'a>>,
    pub down: Vec<Weight<'a>>,
    pub used: usize,
}

fn load_affine<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    expert: Option<usize>,
) -> Result<Weight<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("missing {name}"))?;
    let scale_name = name.replace(".weight", ".scales");
    let bias_name = name.replace(".weight", ".biases");
    let scale_info = source
        .tensor_info(&scale_name)
        .ok_or_else(|| format!("missing {scale_name}"))?;
    let bias_info = source
        .tensor_info(&bias_name)
        .ok_or_else(|| format!("missing {bias_name}"))?;
    if info.ggml_type != GGMLType::I32
        || scale_info.ggml_type != GGMLType::BF16
        || bias_info.ggml_type != GGMLType::BF16
        || scale_info.dims != bias_info.dims
        || info.dims.len() != scale_info.dims.len()
        || !matches!(info.dims.len(), 2 | 3)
        || info.dims[1..] != scale_info.dims[1..]
    {
        return Err(format!("invalid Edge0 affine tensor contract: {name}"));
    }
    let experts = info.dims.get(2).copied().unwrap_or(1) as usize;
    if expert.is_some() != (info.dims.len() == 3) || expert.unwrap_or(0) >= experts {
        return Err(format!("invalid Edge0 expert index for {name}"));
    }
    let n_out = info.dims[1] as usize;
    let groups = scale_info.dims[0] as usize;
    let n_in = groups.checked_mul(64).ok_or("Edge0 input width overflow")?;
    let packed_cols = info.dims[0] as usize;
    if n_in == 0 || packed_cols == 0 || packed_cols * 32 % n_in != 0 {
        return Err(format!("invalid Edge0 packed width for {name}"));
    }
    let bits = packed_cols * 32 / n_in;
    let packed = source
        .tensor_slice(name)
        .ok_or_else(|| format!("missing {name} data"))?;
    let scales = source
        .tensor_slice(&scale_name)
        .ok_or_else(|| format!("missing {scale_name} data"))?;
    let biases = source
        .tensor_slice(&bias_name)
        .ok_or_else(|| format!("missing {bias_name} data"))?;
    let packed_stride = packed_cols * n_out * 4;
    let scale_stride = groups * n_out * 2;
    let index = expert.unwrap_or(0);
    let packed = packed
        .get(index * packed_stride..(index + 1) * packed_stride)
        .ok_or_else(|| format!("truncated {name}"))?;
    let scales = scales
        .get(index * scale_stride..(index + 1) * scale_stride)
        .ok_or_else(|| format!("truncated {scale_name}"))?;
    let biases = biases
        .get(index * scale_stride..(index + 1) * scale_stride)
        .ok_or_else(|| format!("truncated {bias_name}"))?;
    let lora_a_name = name.replace(".weight", ".lora_A");
    let lora_b_name = name.replace(".weight", ".lora_B");
    let lora = match (
        source.tensor_info(&lora_a_name),
        source.tensor_info(&lora_b_name),
    ) {
        (None, None) => None,
        (Some(a), Some(b))
            if expert.is_none()
                && a.ggml_type == GGMLType::F16
                && b.ggml_type == GGMLType::F16
                && a.dims.len() == 2
                && b.dims.len() == 2
                && a.dims[0] as usize == n_in
                && b.dims[0] == a.dims[1]
                && b.dims[1] as usize == n_out =>
        {
            let scale = source
                .metadata("edge0.lora.scale")
                .and_then(|v| v.to_f64())
                .ok_or("missing edge0.lora.scale")? as f32;
            Some((
                source
                    .tensor_slice(&lora_a_name)
                    .ok_or("missing LoRA A data")?,
                source
                    .tensor_slice(&lora_b_name)
                    .ok_or("missing LoRA B data")?,
                a.dims[1] as usize,
                scale,
            ))
        }
        _ => return Err(format!("invalid Edge0 LoRA pair for {name}")),
    };
    let kernel = MlxAffineKernel::new(packed, scales, biases, n_in, n_out, bits, lora)?;
    Ok(Weight {
        kernel: Box::new(kernel),
        ggml_type: GGMLType::I32,
        n_in,
        n_out,
    })
}

fn load_edge0_moe<'a>(
    source: &'a dyn TensorSource,
    config: &Qwen35Config,
) -> Result<Vec<Edge0MoeWeights<'a>>, String> {
    let count = source
        .metadata("edge0.expert_count")
        .and_then(|v| v.to_u64())
        .unwrap_or(0) as usize;
    let used = source
        .metadata("edge0.expert_used_count")
        .and_then(|v| v.to_u64())
        .unwrap_or(0) as usize;
    let width = source
        .metadata("edge0.expert_feed_forward_length")
        .and_then(|v| v.to_u64())
        .unwrap_or(0) as usize;
    if count != 256 || used != 4 || width != 512 || config.n_ff != 512 {
        return Err("unsupported Edge0 MoE contract".into());
    }
    (0..config.n_layer_impl())
        .map(|layer| {
            let prefix = format!("blk.{layer}");
            let router = load_affine(source, &format!("{prefix}.ffn_gate_inp.weight"), None)?;
            let shared_gate =
                load_affine(source, &format!("{prefix}.ffn_shared_gate.weight"), None)?;
            if (
                router.n_in,
                router.n_out,
                shared_gate.n_in,
                shared_gate.n_out,
            ) != (config.n_embd, count, config.n_embd, 1)
            {
                return Err(format!("Edge0 router shape mismatch at layer {layer}"));
            }
            let load_experts =
                |kind: &str, input: usize, output: usize| -> Result<Vec<Weight<'a>>, String> {
                    let name = format!("{prefix}.ffn_{kind}_exps.weight");
                    let info = source
                        .tensor_info(&name)
                        .ok_or_else(|| format!("missing {name}"))?;
                    if info.dims.len() != 3 || info.dims[2] as usize != count {
                        return Err(format!("Edge0 expert count mismatch for {name}"));
                    }
                    (0..count)
                        .map(|expert| {
                            let weight = load_affine(source, &name, Some(expert))?;
                            if (weight.n_in, weight.n_out) != (input, output) {
                                return Err(format!("Edge0 expert shape mismatch for {name}"));
                            }
                            Ok(weight)
                        })
                        .collect()
                };
            Ok(Edge0MoeWeights {
                router,
                shared_gate,
                gate: load_experts("gate", config.n_embd, width)?,
                up: load_experts("up", config.n_embd, width)?,
                down: load_experts("down", width, config.n_embd)?,
                used,
            })
        })
        .collect()
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
        GGMLType::I32 => match load_affine(source, name, None) {
            Ok(weight) => Some(weight),
            Err(error) => {
                eprintln!("WARNING: {error}");
                None
            }
        },
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
            if ti.ggml_type == GGMLType::BF16 {
                weight.kernel =
                    Box::new(crate::ops::kernel::bf16::BF16Kernel::with_bf16_input(data));
            }
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
        let edge0 = source
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val())
            == Some("edge0");

        let token_info = source
            .tensor_info("token_embd.weight")
            .ok_or("Missing token_embd.weight")?;
        let actual = token_info
            .dims
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>();
        let expected = vec![
            if edge0 {
                config.n_embd / 8
            } else {
                config.n_embd
            },
            config.vocab_size,
        ];
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
        if edge0
            && (
                tok_embd.n_in,
                tok_embd.n_out,
                output_weight.n_in,
                output_weight.n_out,
            ) != (
                config.n_embd,
                config.vocab_size,
                config.n_embd,
                config.vocab_size,
            )
        {
            return Err("Edge0 embedding or output shape mismatch".into());
        }

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
            let mut ssm_a = ssm_a;
            if edge0 {
                if let Some(values) = &mut ssm_a {
                    for value in values {
                        *value = -value.exp();
                    }
                }
            }

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

        if edge0 {
            let require =
                |name: &str, weight: Option<&Weight<'_>>, input: usize, output: usize| match weight
                {
                    Some(weight) if (weight.n_in, weight.n_out) == (input, output) => Ok(()),
                    _ => Err(format!("Edge0 {name} shape mismatch or missing")),
                };
            for (index, layer) in layers.iter().enumerate() {
                let prefix = format!("layer {index}");
                require(
                    &format!("{prefix} shared gate"),
                    Some(&layer.ffn_gate),
                    config.n_embd,
                    config.n_ff,
                )?;
                require(
                    &format!("{prefix} shared up"),
                    Some(&layer.ffn_up),
                    config.n_embd,
                    config.n_ff,
                )?;
                require(
                    &format!("{prefix} shared down"),
                    Some(&layer.ffn_down),
                    config.n_ff,
                    config.n_embd,
                )?;
                if config.is_recurrent[index] {
                    require(
                        &format!("{prefix} qkv"),
                        layer.wqkv.as_ref(),
                        config.n_embd,
                        8192,
                    )?;
                    require(
                        &format!("{prefix} z"),
                        layer.wqkv_gate.as_ref(),
                        config.n_embd,
                        4096,
                    )?;
                    require(
                        &format!("{prefix} alpha"),
                        layer.ssm_alpha.as_ref(),
                        config.n_embd,
                        32,
                    )?;
                    require(
                        &format!("{prefix} beta"),
                        layer.ssm_beta.as_ref(),
                        config.n_embd,
                        32,
                    )?;
                    require(
                        &format!("{prefix} out"),
                        layer.ssm_out.as_ref(),
                        4096,
                        config.n_embd,
                    )?;
                    if layer.ssm_conv1d.as_ref().map(Vec::len) != Some(8192 * 4)
                        || layer.ssm_dt.as_ref().map(Vec::len) != Some(32)
                        || layer.ssm_a.as_ref().map(Vec::len) != Some(32)
                        || layer.ssm_norm.as_ref().map(Vec::len) != Some(128)
                    {
                        return Err(format!("Edge0 {prefix} recurrent tensor shape mismatch"));
                    }
                } else {
                    require(
                        &format!("{prefix} q"),
                        layer.wq.as_ref(),
                        config.n_embd,
                        8192,
                    )?;
                    require(
                        &format!("{prefix} k"),
                        layer.wk.as_ref(),
                        config.n_embd,
                        512,
                    )?;
                    require(
                        &format!("{prefix} v"),
                        layer.wv.as_ref(),
                        config.n_embd,
                        512,
                    )?;
                    require(
                        &format!("{prefix} o"),
                        layer.wo.as_ref(),
                        4096,
                        config.n_embd,
                    )?;
                    if layer.attn_q_norm.as_ref().map(Vec::len) != Some(256)
                        || layer.attn_k_norm.as_ref().map(Vec::len) != Some(256)
                    {
                        return Err(format!("Edge0 {prefix} attention norm shape mismatch"));
                    }
                }
            }
        }

        let edge0_moe = if edge0 {
            Some(load_edge0_moe(source, &config)?)
        } else {
            None
        };

        Ok(Self {
            config,
            tok_embd,
            output_norm,
            output_weight,
            layers,
            edge0_moe,
            #[cfg(feature = "vulkan")]
            gpu: None,
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
