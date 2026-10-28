//! GGUF → shared hybrid trunk weight loading.
//!
//! `Qwen35Model::from_source` accepts Qwen3.5 only. `Edge0Model` loads the
//! same attention/SSM trunk and keeps its MoE weights in `models::edge0`.
//! The shared loader reads a GGUF `TensorSource` and builds
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
use crate::models::edge0::weights::load_affine;
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
    /// Dense FFN, absent on `qwen35moe` where `moe` carries the experts.
    pub ffn_gate: Option<Weight<'a>>,
    pub ffn_up: Option<Weight<'a>>,
    pub ffn_down: Option<Weight<'a>>,
    pub moe: Option<Qwen35MoeWeights<'a>>,
}

/// MoE weights for `qwen35moe`.
///
/// Experts are kept as one borrowed slice per projection rather than 3 *
/// `n_expert` eager `Weight`s: a 40-layer 256-expert model would otherwise build
/// 30 720 kernels up front for a path that only ever touches `n_expert_used`
/// of them per token. Each expert's matrix is a contiguous block inside the 3-D
/// GGUF tensor, so an expert is a sub-slice at a fixed stride.
pub struct Qwen35MoeWeights<'a> {
    /// Router: `[n_expert, n_embd]` (ggml dims `[n_embd, n_expert]`).
    pub gate_inp: Weight<'a>,
    pub up_exps: &'a [u8],
    pub gate_exps: &'a [u8],
    pub down_exps: &'a [u8],
    pub up_type: GGMLType,
    pub gate_type: GGMLType,
    pub down_type: GGMLType,
    pub n_expert: usize,
    pub n_ff_exp: usize,
    pub n_embd: usize,
    /// Bytes per expert for each projection.
    pub stride_up: usize,
    pub stride_gate: usize,
    pub stride_down: usize,
    /// Shared expert; `None` when the checkpoint has none.
    pub gate_inp_shexp: Option<Vec<f32>>,
    pub up_shexp: Option<Weight<'a>>,
    pub gate_shexp: Option<Weight<'a>>,
    pub down_shexp: Option<Weight<'a>>,
}

impl<'a> Qwen35MoeWeights<'a> {
    /// `[n_ff_exp, n_embd]` for expert `e`, matching ggml's
    /// `ffn_up_exps` (`[n_embd, n_ff_exp, n_expert]`).
    pub fn up(&self, e: usize) -> Weight<'a> {
        self.view(
            self.up_exps,
            self.up_type,
            self.stride_up,
            e,
            self.n_embd,
            self.n_ff_exp,
        )
    }

    pub fn gate(&self, e: usize) -> Weight<'a> {
        self.view(
            self.gate_exps,
            self.gate_type,
            self.stride_gate,
            e,
            self.n_embd,
            self.n_ff_exp,
        )
    }

    /// `[n_embd, n_ff_exp]` for expert `e`, matching ggml's
    /// `ffn_down_exps` (`[n_ff_exp, n_embd, n_expert]`).
    pub fn down(&self, e: usize) -> Weight<'a> {
        self.view(
            self.down_exps,
            self.down_type,
            self.stride_down,
            e,
            self.n_ff_exp,
            self.n_embd,
        )
    }

    fn view(
        &self,
        data: &'a [u8],
        ggml_type: GGMLType,
        stride: usize,
        e: usize,
        n_in: usize,
        n_out: usize,
    ) -> Weight<'a> {
        let start = e * stride;
        let slice = data.get(start..start + stride).unwrap_or(&data[..0]);
        Weight::from_quantized(QuantizedTensor::from_bytes(slice, ggml_type, n_in, n_out))
    }
}

/// Shared Qwen3.5/Edge0 attention and SSM weights + parsed config.
///
/// Qwen3.5 exposes this as `Qwen35Model`; Edge0 composes it with MoE weights.
pub struct HybridTrunk<'a> {
    pub config: Qwen35Config,
    pub tok_embd: Weight<'a>,
    pub output_norm: Vec<f32>,
    pub output_weight: Weight<'a>,
    pub layers: Vec<Qwen35LayerWeights<'a>>,
    /// Optional classification head (cross-encoder / NLI), loaded when the
    /// GGUF carries `cls.output.weight` (and optionally `cls.output.bias`).
    /// The shape is `[n_embd, num_labels]` so the same `Weight` matmul used
    /// by `qwen3::trunk::weights::score_logits` works here; the bias, when
    /// present, is a separate `Vec<f32>` of length `num_labels`.
    pub cls_score: Option<Weight<'a>>,
    pub cls_score_bias: Vec<f32>,
    /// Lazily-initialized Vulkan session. Built on the first decode token
    /// after `--gpu` enables the global Vulkan context. `None` means CPU
    /// fallback (no eligible GPU or Vulkan init failed).
    #[cfg(feature = "vulkan")]
    pub(crate) gpu: Option<Qwen35VulkanSession>,
}

/// Qwen3.5 uses the shared hybrid trunk without architecture-specific weights.
pub type Qwen35Model<'a> = HybridTrunk<'a>;

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
/// Bind one layer's MoE tensors.
///
/// Each projection is a 3-D GGUF tensor with the expert axis last, so expert
/// `e` is a contiguous block at `e * bytes_per_expert`.
fn load_moe_weights<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    layer: usize,
    moe: super::config::Qwen35Moe,
    n_embd_cfg: usize,
) -> Result<Qwen35MoeWeights<'a>, String> {
    let need = |name: &str| -> Result<&'a [u8], String> {
        source
            .tensor_slice(name)
            .ok_or_else(|| format!("Missing {name}"))
    };
    let info = |name: &str| -> Result<&'a crate::core::tensor::TensorInfo, String> {
        source
            .tensor_info(name)
            .ok_or_else(|| format!("Missing {name}"))
    };

    let up_name = format!("blk.{layer}.ffn_up_exps.weight");
    let gate_name = format!("blk.{layer}.ffn_gate_exps.weight");
    let down_name = format!("blk.{layer}.ffn_down_exps.weight");

    let up_info = info(&up_name)?;
    if up_info.dims.len() != 3 {
        return Err(format!(
            "{up_name}: expected [n_embd, n_ff_exp, n_expert], got {:?}",
            up_info.dims
        ));
    }
    let n_embd = up_info.dims[0] as usize;
    let n_ff_exp = up_info.dims[1] as usize;
    let n_expert = up_info.dims[2] as usize;
    if n_embd != n_embd_cfg || n_ff_exp != moe.n_ff_exp || n_expert != moe.n_expert {
        return Err(format!(
            "{up_name}: got [{n_embd}, {n_ff_exp}, {n_expert}], config says \
             [{n_embd_cfg}, {}, {}]",
            moe.n_ff_exp, moe.n_expert
        ));
    }
    // ggml stores the expert axis last and contiguous, so a 3-D tensor is
    // `n_expert` back-to-back copies of one expert's matrix.
    let gate_info = info(&gate_name)?;
    let down_info = info(&down_name)?;
    for (name, tensor, dims) in [
        (
            &up_name,
            up_info,
            [n_embd as u64, n_ff_exp as u64, n_expert as u64],
        ),
        (
            &gate_name,
            gate_info,
            [n_embd as u64, n_ff_exp as u64, n_expert as u64],
        ),
        (
            &down_name,
            down_info,
            [n_ff_exp as u64, n_embd as u64, n_expert as u64],
        ),
    ] {
        if tensor.dims != dims || tensor.checked_nbytes() != Some(need(name)?.len() as u64) {
            return Err(format!(
                "{name}: invalid expert shape or payload; expected {dims:?}"
            ));
        }
    }
    let stride_up = up_info.ggml_type.nbytes(n_embd * n_ff_exp);
    let stride_gate = gate_info.ggml_type.nbytes(n_embd * n_ff_exp);
    let stride_down = down_info.ggml_type.nbytes(n_ff_exp * n_embd);

    let gate_inp = load_weight(source, &format!("blk.{layer}.ffn_gate_inp.weight"))
        .ok_or_else(|| format!("Missing blk.{layer}.ffn_gate_inp.weight"))?;
    if gate_inp.n_in != n_embd || gate_inp.n_out != n_expert {
        return Err(format!(
            "blk.{layer}.ffn_gate_inp.weight: got [{}, {}], expected [{n_embd}, {n_expert}]",
            gate_inp.n_in, gate_inp.n_out
        ));
    }

    let has_shexp = moe.n_ff_shexp > 0;
    let shared_weight = |suffix: &str,
                         n_in: usize,
                         n_out: usize|
     -> Result<Option<Weight<'a>>, String> {
        if !has_shexp {
            return Ok(None);
        }
        let name = format!("blk.{layer}.ffn_{suffix}_shexp.weight");
        let tensor = info(&name)?;
        if tensor.dims != [n_in as u64, n_out as u64]
            || tensor.checked_nbytes() != Some(need(&name)?.len() as u64)
        {
            return Err(format!("Invalid shared expert tensor {name}: expected [{n_in}, {n_out}] with complete payload"));
        }
        load_weight(source, &name)
            .map(Some)
            .ok_or_else(|| format!("Unsupported shared expert tensor: {name}"))
    };
    let gate_inp_shexp = if has_shexp {
        let name = [
            format!("blk.{layer}.ffn_gate_inp_shexp.weight"),
            format!("blk.{layer}.ffn_gate_inp_shexp"),
        ]
        .into_iter()
        .find(|name| source.tensor_info(name).is_some())
        .ok_or_else(|| format!("Missing blk.{layer}.ffn_gate_inp_shexp.weight"))?;
        let tensor = info(&name)?;
        if tensor.dims != [n_embd as u64] && tensor.dims != [n_embd as u64, 1] {
            return Err(format!(
                "Invalid shared expert gate {name}: shape {:?}",
                tensor.dims
            ));
        }
        crate::core::tensor::load_f32_tensor(source, &name, &tensor.dims)?
    } else {
        Vec::new()
    };

    Ok(Qwen35MoeWeights {
        gate_inp,
        up_exps: need(&up_name)?,
        gate_exps: need(&gate_name)?,
        down_exps: need(&down_name)?,
        up_type: up_info.ggml_type,
        gate_type: gate_info.ggml_type,
        down_type: down_info.ggml_type,
        n_expert,
        n_ff_exp,
        n_embd,
        stride_up,
        stride_gate,
        stride_down,
        gate_inp_shexp: has_shexp.then_some(gate_inp_shexp),
        up_shexp: shared_weight("up", n_embd, moe.n_ff_shexp)?,
        gate_shexp: shared_weight("gate", n_embd, moe.n_ff_shexp)?,
        down_shexp: shared_weight("down", moe.n_ff_shexp, n_embd)?,
    })
}

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

impl<'a> HybridTrunk<'a> {
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        let arch = source
            .metadata("general.architecture")
            .and_then(|value| value.to_string_val());
        if arch == Some("edge0") {
            return Err("Edge0 architecture requires Edge0Model::from_source".into());
        }
        let config = Qwen35Config::from_source(source)?;
        Self::load_trunk(source, config, false)
    }

    pub(crate) fn load_trunk(
        source: &'a dyn TensorSource,
        config: Qwen35Config,
        edge0: bool,
    ) -> Result<Self, String> {
        let token_info = source
            .tensor_info("token_embd.weight")
            .ok_or("Missing token_embd.weight")?;
        let actual = token_info
            .dims
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>();
        // The lossless export stores packed 4-bit codes, so the embedding is
        // one eighth as wide; the expanded modes store whole values instead.
        let expanded = crate::models::edge0::weights::is_expanded(source);
        let expected = vec![
            if edge0 && !expanded {
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
            && !expanded
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
        // Every mode, packed or expanded, must present the same logical matrix,
        // so the expanded path checks the widths the kernel will actually use.
        if edge0
            && expanded
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
            return Err("Edge0 expanded embedding or output shape mismatch".into());
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

            let moe = match config.moe {
                None => None,
                Some(moe) => Some(load_moe_weights(source, i, moe, config.n_embd)?),
            };
            let dense = |name: &str| -> Result<Option<Weight<'a>>, String> {
                if config.moe.is_some() {
                    // qwen35moe carries no dense FFN; the per-expert tensors
                    // stand in for it. Absent rather than mis-bound, so the
                    // dispatch cannot silently run a dense block on a MoE model.
                    return Ok(None);
                }
                load_weight(source, name)
                    .map(Some)
                    .ok_or_else(|| format!("Missing {name}"))
            };
            let ffn_gate = dense(&format!("blk.{i}.ffn_gate.weight"))?;
            let ffn_up = dense(&format!("blk.{i}.ffn_up.weight"))?;
            let ffn_down = dense(&format!("blk.{i}.ffn_down.weight"))?;
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
                moe,
            });
        }

        Ok(Self {
            config,
            tok_embd,
            output_norm,
            output_weight,
            layers,
            cls_score: load_weight(source, "cls.output.weight"),
            cls_score_bias: load_weight_f32(source, "cls.output.bias").unwrap_or_default(),
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

    /// True when the GGUF carried a `cls.output.weight` head. Mirrors
    /// `Qwen3Model::is_rerank` so callers (server / rerank CLI) can probe
    /// the model the same way for both backbones.
    pub fn is_classifier(&self) -> bool {
        self.cls_score.is_some()
    }

    /// Project `last_hidden` (length `n_embd`) through the classification
    /// head, returning one logit per class plus the optional bias. Returns
    /// an error if the head wasn't loaded.
    ///
    /// Same contract as `Qwen3Model::score_logits` but with a `cls_score_bias`
    /// added on top of the matmul output. The HF `score.weight` is stored
    /// as `[num_labels, n_embd]` in safetensors; the converter transposes
    /// it to `[n_embd, num_labels]` for the llama.cpp rerank-packer layout,
    /// and we add a zero bias when the HF head had no bias (`openjev` does
    /// not have one; `jina-bert-v2` does).
    pub fn score_logits(&self, last_hidden: &[f32]) -> Result<Vec<f32>, String> {
        let weight = self
            .cls_score
            .as_ref()
            .ok_or_else(|| "Qwen3.5 model has no cls.output.weight head".to_string())?;
        if last_hidden.len() != self.config.n_embd {
            return Err(format!(
                "score_logits: hidden size {} does not match n_embd {}",
                last_hidden.len(),
                self.config.n_embd
            ));
        }
        let n_in = self.config.n_embd;
        let n_cls = weight.n_out;
        let mut out = vec![0.0f32; n_cls];
        weight
            .kernel
            .forward_prepared(last_hidden, &[], &[], None, &mut out, n_in, n_cls, 0, 1);
        if !self.cls_score_bias.is_empty() {
            if self.cls_score_bias.len() != n_cls {
                return Err(format!(
                    "cls.output.bias length {} does not match num_labels {}",
                    self.cls_score_bias.len(),
                    n_cls
                ));
            }
            for (slot, bias) in out.iter_mut().zip(self.cls_score_bias.iter()) {
                *slot += *bias;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod moe_tests {
    use super::*;
    use crate::core::tensor::{MetaValue, TensorInfo};
    use std::collections::HashMap;

    struct Source(HashMap<String, (TensorInfo, Vec<u8>)>);
    impl TensorSource for Source {
        fn metadata(&self, _: &str) -> Option<&MetaValue> {
            None
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.0.get(name).map(|v| &v.0)
        }
        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.0.get(name).map(|v| v.1.as_slice())
        }
    }

    fn source() -> (Source, super::super::config::Qwen35Moe) {
        let mut tensors = HashMap::new();
        for (name, dims, ggml_type) in [
            ("ffn_up_exps.weight", vec![512, 256, 2], GGMLType::Q4K),
            ("ffn_gate_exps.weight", vec![512, 256, 2], GGMLType::Q5K),
            ("ffn_down_exps.weight", vec![256, 512, 2], GGMLType::Q6K),
            ("ffn_gate_inp.weight", vec![512, 2], GGMLType::F32),
        ] {
            let name = format!("blk.0.{name}");
            let bytes = vec![0; ggml_type.nbytes(dims.iter().product::<u64>() as usize)];
            tensors.insert(
                name.clone(),
                (
                    TensorInfo {
                        name,
                        dims,
                        ggml_type,
                        offset: 0,
                    },
                    bytes,
                ),
            );
        }
        (
            Source(tensors),
            super::super::config::Qwen35Moe {
                n_expert: 2,
                n_expert_used: 1,
                n_ff_exp: 256,
                n_ff_shexp: 0,
                weights_scale: 1.0,
            },
        )
    }

    #[test]
    fn qwen35moe_preserves_each_expert_projection_type() {
        let (source, config) = source();
        let weights = load_moe_weights(&source, 0, config, 512).unwrap();
        for expert in 0..2 {
            assert_eq!(weights.up(expert).ggml_type, GGMLType::Q4K);
            assert_eq!(weights.gate(expert).ggml_type, GGMLType::Q5K);
            assert_eq!(weights.down(expert).ggml_type, GGMLType::Q6K);
        }
    }

    #[test]
    fn qwen35moe_rejects_invalid_expert_shapes_and_payloads() {
        for name in [
            "ffn_up_exps.weight",
            "ffn_gate_exps.weight",
            "ffn_down_exps.weight",
        ] {
            for bad_shape in [true, false] {
                let (mut source, config) = source();
                let tensor = source.0.get_mut(&format!("blk.0.{name}")).unwrap();
                if bad_shape {
                    tensor.0.dims[2] = 1;
                } else {
                    tensor.1.pop();
                }
                assert!(
                    load_moe_weights(&source, 0, config, 512).is_err(),
                    "{name}, bad_shape={bad_shape}"
                );
            }
        }
    }
}

/// The shared expert is a separate path from the routed experts: it has
/// three projections of its own, and the loader must reject a missing,
/// truncated or misshaped one. The routed-expert tests live in `moe_tests`;
/// this one covers the shared branch.
#[cfg(test)]
mod moe_shared_expert_tests {
    use super::*;
    use crate::core::tensor::{MetaValue, TensorInfo};
    use std::collections::HashMap;

    const N_EMBD: usize = 3;
    const N_FF_EXP: usize = 2;
    const N_EXPERT: usize = 2;
    const N_FF_SHEXP: usize = 2;

    struct Source(HashMap<String, (TensorInfo, Vec<u8>)>);
    impl TensorSource for Source {
        fn metadata(&self, _: &str) -> Option<&MetaValue> {
            None
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.0.get(name).map(|v| &v.0)
        }
        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.0.get(name).map(|v| v.1.as_slice())
        }
    }

    fn insert(source: &mut Source, name: &str, dims: Vec<u64>) {
        let len = dims.iter().product::<u64>() as usize * 4;
        source.0.insert(
            name.to_string(),
            (
                TensorInfo {
                    name: name.to_string(),
                    dims,
                    ggml_type: GGMLType::F32,
                    offset: 0,
                },
                vec![0; len],
            ),
        );
    }

    fn source() -> Source {
        let mut source = Source(HashMap::new());
        for (suffix, dims) in [
            ("up_exps.weight", vec![N_EMBD, N_FF_EXP, N_EXPERT]),
            ("gate_exps.weight", vec![N_EMBD, N_FF_EXP, N_EXPERT]),
            ("down_exps.weight", vec![N_FF_EXP, N_EMBD, N_EXPERT]),
            ("gate_inp.weight", vec![N_EMBD, N_EXPERT]),
        ] {
            insert(&mut source, &format!("blk.0.ffn_{suffix}"), dims);
        }
        for (suffix, dims) in [
            ("up_shexp.weight", vec![N_EMBD, N_FF_SHEXP]),
            ("gate_shexp.weight", vec![N_EMBD, N_FF_SHEXP]),
            ("down_shexp.weight", vec![N_FF_SHEXP, N_EMBD]),
        ] {
            insert(&mut source, &format!("blk.0.ffn_{suffix}"), dims);
        }
        source
    }

    fn config() -> super::super::config::Qwen35Moe {
        super::super::config::Qwen35Moe {
            n_expert: N_EXPERT,
            n_expert_used: 1,
            n_ff_exp: N_FF_EXP,
            n_ff_shexp: N_FF_SHEXP,
            weights_scale: 1.0,
        }
    }

    #[test]
    fn shared_expert_requires_all_projections_and_valid_payloads() {
        let mut source = source();
        assert!(load_moe_weights(&source, 0, config(), N_EMBD).is_ok());
        for suffix in ["up", "gate", "down"] {
            let name = format!("blk.0.ffn_{suffix}_shexp.weight");
            let saved = source.0.remove(&name).unwrap();
            assert!(
                load_moe_weights(&source, 0, config(), N_EMBD).is_err(),
                "missing {suffix}"
            );
            source.0.insert(name.clone(), saved);
            source.0.get_mut(&name).unwrap().1.pop();
            assert!(
                load_moe_weights(&source, 0, config(), N_EMBD).is_err(),
                "truncated {suffix}"
            );
            source.0.get_mut(&name).unwrap().1.push(0);
        }
        source
            .0
            .get_mut("blk.0.ffn_up_shexp.weight")
            .unwrap()
            .0
            .dims = vec![N_FF_SHEXP, N_EMBD];
        assert!(load_moe_weights(&source, 0, config(), N_EMBD).is_err());
    }
}
