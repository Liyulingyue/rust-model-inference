//! Nemotron-3 Nano weight loading.
//!
//! Hybrid Mamba-Transformer: every layer carries both attention and SSM
//! weights. Each block loads its FFN, attention, or Mamba2 branch.

use crate::core::tensor::{load_f32_tensor, GGMLType, TensorSource};
use crate::ops::kernel::QuantizedTensor;
use crate::ops::kernel::Weight;

use super::config::{NemotronConfig, NemotronMoeConfig};

fn usize_to_u64(v: usize, name: &str) -> Result<u64, String> {
    u64::try_from(v).map_err(|_| format!("{name} does not fit u64"))
}

/// Mixture-of-experts weights for one Nemotron-3-Nano 30B-A3B layer.
///
/// The experts are kept as one borrowed slice per projection rather than
/// `n_expert` eager `Weight`s: a 23-layer 128-expert model would
/// otherwise build 2944 kernels up front for a path that only ever
/// touches `n_expert_used` of them per token. Each expert's matrix is a
/// contiguous block inside the 3-D GGUF tensor, so an expert is a
/// sub-slice at a fixed stride.
pub struct NemotronMoeWeights<'a> {
    /// Router projection: ggml dims `[n_embd, n_expert]`, F32.
    pub gate_inp: Weight<'a>,
    /// Expert up projections: ggml dims `[n_embd, n_ff_exp, n_expert]`.
    pub up_exps: &'a [u8],
    /// Expert down projections: ggml dims `[n_ff_exp, n_embd, n_expert]`.
    pub down_exps: &'a [u8],
    pub up_type: GGMLType,
    pub down_type: GGMLType,
    pub n_expert: usize,
    pub n_ff_exp: usize,
    pub n_embd: usize,
    /// Bytes per expert for each projection.
    pub stride_up: usize,
    pub stride_down: usize,
    /// Shared expert up: ggml dims `[n_embd, n_ff_shexp]`.
    pub up_shexp: Weight<'a>,
    /// Shared expert down: ggml dims `[n_ff_shexp, n_embd]`.
    pub down_shexp: Weight<'a>,
    /// Router selection bias (`ffn_exp_probs_b.bias`). It only takes
    /// part in *ranking* the experts; the routing weights themselves are
    /// the unbiased sigmoid probabilities.
    pub exp_probs_b: Vec<f32>,
}

impl<'a> NemotronMoeWeights<'a> {
    /// `[n_embd, n_ff_exp]` for expert `e`, matching ggml's
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

    /// `[n_ff_exp, n_embd]` for expert `e`, matching ggml's
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

pub struct NemotronLayerWeights<'a> {
    /// Pre-attention / pre-MLP RMSNorm. Loaded for every layer (even SSM
    /// layers keep a norm on the input).
    pub attn_norm: Vec<f32>,
    /// Attention branch (only on the 4 attention-only layers).
    pub wq: Option<Weight<'a>>,
    pub wk: Option<Weight<'a>>,
    pub wv: Option<Weight<'a>>,
    pub wo: Option<Weight<'a>>,
    pub attn_q_norm: Option<Vec<f32>>,
    pub attn_k_norm: Option<Vec<f32>>,
    /// FFN branch (only on the 17 FFN-only layers). Plain 2-layer FFN
    /// (`up → activation → down`); no gate.
    pub ffn_norm: Option<Vec<f32>>,
    pub w_up: Option<Weight<'a>>,
    pub w_down: Option<Weight<'a>>,
    /// Mixture-of-experts branch (only on the MoE layers of the 30B
    /// variant). `None` for the dense architecture.
    pub moe: Option<NemotronMoeWeights<'a>>,
    /// Mamba2 SSM branch (only on the 21 SSM-only layers). None for the
    /// other 21 layers.
    pub ssm_in: Option<Weight<'a>>,
    pub ssm_conv1d_w: Option<Vec<f32>>,
    pub ssm_conv1d_b: Option<Vec<f32>>,
    pub ssm_dt_bias: Option<Vec<f32>>,
    pub ssm_a_log: Option<Vec<f32>>,
    pub ssm_d: Option<Vec<f32>>,
    pub ssm_norm: Option<Vec<f32>>,
    pub ssm_out: Option<Weight<'a>>,
}

/// Single source of truth for the GGML types the kernel layer accepts
/// for matmul. Mirrors `QuantizedTensor::from_bytes`
/// (`src/ops/kernel/quantized_tensor.rs`). When that list grows (e.g.
/// IQ4_XL), update this match — no caller needs to know.
fn is_supported_weight_type(t: GGMLType) -> bool {
    use GGMLType::*;
    matches!(
        t,
        F32 | F16
            | BF16
            | Q8_0
            | Q4_0
            | Q4_1
            | Q5_0
            | Q5_1
            | Q2K
            | Q3K
            | Q4K
            | Q5K
            | Q6K
            | IQ4_NL
            | IQ2_XXS
            | IQ2_XS
            | IQ3_XXS
            | IQ1_S
            | IQ3_S
            | IQ2_S
            | IQ4_XS
            | IQ1_M
    )
}

pub(crate) fn load_weight(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Weight<'static>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [n_in as u64, n_out as u64] || !is_supported_weight_type(info.ggml_type) {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected [{n_in}, {n_out}] F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/Q2K/Q3K/Q4K/Q5K/Q6K/IQ*",
            info.dims, info.ggml_type
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if Some(bytes.len() as u64) != info.checked_nbytes() {
        return Err(format!("Invalid tensor byte length: {name}"));
    }
    let bytes_static: &'static [u8] = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes) };
    Ok(Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes_static,
        info.ggml_type,
        n_in,
        n_out,
    )))
}

/// Load the mixture-of-experts weights of one layer.
///
/// Tensor names follow the upstream reference
/// (`src/models/nemotron-h.cpp::build_ffn_layer`):
///
/// - `ffn_gate_inp.weight` — router, `[n_embd, n_expert]`, F32
/// - `ffn_up_exps.weight`   — expert up, `[n_embd, n_ff_exp, n_expert]`
/// - `ffn_down_exps.weight` — expert down, `[n_ff_exp, n_embd, n_expert]`
/// - `ffn_up_shexp.weight`  — shared expert up, `[n_embd, n_ff_shexp]`
/// - `ffn_down_shexp.weight`— shared expert down, `[n_ff_shexp, n_embd]`
/// - `exp_probs_b.bias`     — router selection bias, `[n_expert]`
///
/// Unlike Qwen3.5 there is **no** `ffn_gate_exps` (the experts are not
/// gated) and **no** `ffn_gate_inp_shexp` (the shared expert is not
/// gated either).
fn load_moe_weights(
    source: &dyn TensorSource,
    layer: usize,
    moe: &NemotronMoeConfig,
    n_embd_cfg: usize,
) -> Result<NemotronMoeWeights<'static>, String> {
    let need = |name: &str| -> Result<&'static [u8], String> {
        let bytes = source
            .tensor_slice(name)
            .ok_or_else(|| format!("Missing {name}"))?;
        // Same escape hatch as `load_weight`: the GGUF bytes outlive the
        // loader because they come from an mmap'd file.
        Ok(unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes) })
    };
    let info = |name: &str| -> Result<&'static crate::core::tensor::TensorInfo, String> {
        let info = source
            .tensor_info(name)
            .ok_or_else(|| format!("Missing {name}"))?;
        Ok(unsafe {
            std::mem::transmute::<
                &crate::core::tensor::TensorInfo,
                &'static crate::core::tensor::TensorInfo,
            >(info)
        })
    };

    let up_name = format!("blk.{layer}.ffn_up_exps.weight");
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
    let down_info = info(&down_name)?;
    for (name, tensor, dims) in [
        (
            &up_name,
            up_info,
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
    let stride_down = down_info.ggml_type.nbytes(n_ff_exp * n_embd);

    let gate_inp = load_weight(
        source,
        &format!("blk.{layer}.ffn_gate_inp.weight"),
        n_embd,
        n_expert,
    )?;
    if gate_inp.n_in != n_embd || gate_inp.n_out != n_expert {
        return Err(format!(
            "blk.{layer}.ffn_gate_inp.weight: got [{}, {}], expected [{n_embd}, {n_expert}]",
            gate_inp.n_in, gate_inp.n_out
        ));
    }

    // The shared expert has no gate: `build_ffn(..., NULL /* no gate */, ...)`.
    let up_shexp = load_weight(
        source,
        &format!("blk.{layer}.ffn_up_shexp.weight"),
        n_embd,
        moe.n_ff_shexp,
    )?;
    let down_shexp = load_weight(
        source,
        &format!("blk.{layer}.ffn_down_shexp.weight"),
        moe.n_ff_shexp,
        n_embd,
    )?;

    // The router selection bias is F32 and only participates in ranking.
    let exp_probs_b = load_f32_tensor(
        source,
        &format!("blk.{layer}.exp_probs_b.bias"),
        &[n_expert as u64],
    )?;

    Ok(NemotronMoeWeights {
        gate_inp,
        up_exps: need(&up_name)?,
        down_exps: need(&down_name)?,
        up_type: up_info.ggml_type,
        down_type: down_info.ggml_type,
        n_expert,
        n_ff_exp,
        n_embd,
        stride_up,
        stride_down,
        up_shexp,
        down_shexp,
        exp_probs_b,
    })
}

pub fn load_layers(
    source: &dyn TensorSource,
    config: &NemotronConfig,
) -> Result<Vec<NemotronLayerWeights<'static>>, String> {
    let mut layers = Vec::with_capacity(config.n_layer);
    let n_attn_q = config.n_head * config.n_embd_head_k;
    let n_attn_k = config.n_head_kv * config.n_embd_head_k;
    let n_attn_v = config.n_head_kv * config.n_embd_head_v;
    let n_attn_out = config.n_head * config.n_embd_head_v;
    let head_dim = config.n_embd_head_k;
    let layer_has_attn = |idx: usize| -> bool {
        source
            .tensor_info(&format!("blk.{idx}.attn_q.weight"))
            .is_some()
            && source
                .tensor_info(&format!("blk.{idx}.attn_output.weight"))
                .is_some()
    };
    let layer_has_ssm = |idx: usize| -> bool {
        source
            .tensor_info(&format!("blk.{idx}.ssm_in.weight"))
            .is_some()
    };
    let layer_has_ffn = |idx: usize| -> bool {
        source
            .tensor_info(&format!("blk.{idx}.ffn_up.weight"))
            .is_some()
    };
    // The MoE layers carry `ffn_up_exps` instead of `ffn_up`.
    let layer_has_moe = |idx: usize| -> bool {
        source
            .tensor_info(&format!("blk.{idx}.ffn_up_exps.weight"))
            .is_some()
    };
    let is_moe_arch = config.is_moe();
    for layer_idx in 0..config.n_layer {
        let prefix = format!("blk.{layer_idx}");
        let attn_norm = load_f32_tensor(
            source,
            &format!("{prefix}.attn_norm.weight"),
            &[config.n_embd as u64],
        )?;
        let this_has_attn = layer_has_attn(layer_idx);
        let this_has_ssm = layer_has_ssm(layer_idx);
        let has_layer_ffn = layer_has_ffn(layer_idx);
        let has_layer_moe = layer_has_moe(layer_idx);
        let expected_attn = config.attention_layers.contains(&layer_idx);
        let expected_ffn = config.ffn_layers.contains(&layer_idx);
        // The feed-forward branch of an FFN layer is dense for the 4B
        // and routed for the 30B; the wrong one must not load silently.
        let ffn_kind_ok = if expected_ffn {
            if is_moe_arch {
                has_layer_moe && !has_layer_ffn
            } else {
                has_layer_ffn && !has_layer_moe
            }
        } else {
            !has_layer_ffn && !has_layer_moe
        };
        if this_has_attn != expected_attn
            || !ffn_kind_ok
            || this_has_ssm != (!expected_attn && !expected_ffn)
        {
            return Err(format!(
                "Nemotron layer {layer_idx} tensors do not match layer metadata"
            ));
        }
        let (wq, wk, wv, wo, attn_q_norm, attn_k_norm) = if this_has_attn {
            let wq = load_weight(
                source,
                &format!("{prefix}.attn_q.weight"),
                config.n_embd,
                n_attn_q,
            )?;
            let wk = load_weight(
                source,
                &format!("{prefix}.attn_k.weight"),
                config.n_embd,
                n_attn_k,
            )?;
            let wv = load_weight(
                source,
                &format!("{prefix}.attn_v.weight"),
                config.n_embd,
                n_attn_v,
            )?;
            let wo = load_weight(
                source,
                &format!("{prefix}.attn_output.weight"),
                n_attn_out,
                config.n_embd,
            )?;
            let qn = if source
                .tensor_info(&format!("{prefix}.attn_q_norm.weight"))
                .is_some()
            {
                Some(load_f32_tensor(
                    source,
                    &format!("{prefix}.attn_q_norm.weight"),
                    &[head_dim as u64],
                )?)
            } else {
                None
            };
            let kn = if source
                .tensor_info(&format!("{prefix}.attn_k_norm.weight"))
                .is_some()
            {
                Some(load_f32_tensor(
                    source,
                    &format!("{prefix}.attn_k_norm.weight"),
                    &[head_dim as u64],
                )?)
            } else {
                None
            };
            (Some(wq), Some(wk), Some(wv), Some(wo), qn, kn)
        } else {
            // Per-layer attn tensor is absent in this block; skip the
            // attention branch entirely (forward_layer checks Option).
            (None, None, None, None, None, None)
        };
        // FFN is optional per layer (some Nemotron-H blocks skip FFN).
        // When present, ffn_norm is sometimes stored explicitly and
        // sometimes reuses attn_norm; the forward path always falls back
        // to attn_norm if ffn_norm is absent. FFN is a plain 2-layer
        // (`up → activation → down`); no `ffn_gate` weight.
        let ffn_norm_present = has_layer_ffn
            && source
                .tensor_info(&format!("{prefix}.ffn_norm.weight"))
                .is_some();
        let (ffn_norm, w_up, w_down) = if has_layer_ffn {
            let ffn_norm = if ffn_norm_present {
                Some(load_f32_tensor(
                    source,
                    &format!("{prefix}.ffn_norm.weight"),
                    &[config.n_embd as u64],
                )?)
            } else {
                None
            };
            let wu = load_weight(
                source,
                &format!("{prefix}.ffn_up.weight"),
                config.n_embd,
                config.n_ff,
            )?;
            let wd = load_weight(
                source,
                &format!("{prefix}.ffn_down.weight"),
                config.n_ff,
                config.n_embd,
            )?;
            (ffn_norm, Some(wu), Some(wd))
        } else {
            (None, None, None)
        };
        // MoE branch: routed experts plus one shared expert.
        let moe = if has_layer_moe {
            let moe_cfg = config
                .moe
                .as_ref()
                .ok_or_else(|| "Nemotron MoE tensors without MoE metadata".to_string())?;
            Some(load_moe_weights(source, layer_idx, moe_cfg, config.n_embd)?)
        } else {
            None
        };
        // Mamba2 SSM branch.
        let (ssm_in, ssm_conv1d_w, ssm_conv1d_b, ssm_dt_bias, ssm_a_log, ssm_d, ssm_norm, ssm_out) =
            if this_has_ssm {
                // ssm_in.shape is (n_embd, d_in_proj) where
                // d_in_proj = 2*d_inner + 2*n_group*d_state + dt_rank
                // (matches llama.cpp nemotron-h.cpp). Earlier read attempts
                // swapped the args and treated ssm_in as (d_inner, n_embd),
                // which corrupted the dispatch and produced degenerate output.
                let d_in_proj = 2 * config.ssm_inner_size
                    + 2 * config.ssm_state_size * config.ssm_group_count
                    + config.ssm_time_step_rank;
                let ssm_in = load_weight(
                    source,
                    &format!("{prefix}.ssm_in.weight"),
                    config.n_embd,
                    d_in_proj,
                )?;
                // Mamba2 state and convolution tensors in this checkpoint are F32.
                // Layout per the reference:
                //   ssm_conv1d.weight (4, 9728) — fused causal conv + b + c outputs
                //   ssm_conv1d.bias   (9728,)   — bias for the same fused tensor
                //   ssm_dt.bias       (dt_rank,) — dt bias
                //   ssm_a             (1, dt_rank) — decay factor
                //   ssm_d             (1, dt_rank) — D skip
                //   ssm_norm.weight   (per_group, n_groups) — group RMSNorm
                //   ssm_out.weight    (inner, n_embd) — Q8_0 out_proj
                let ssm_conv1d_w = load_f32_tensor(
                    source,
                    &format!("{prefix}.ssm_conv1d.weight"),
                    &[
                        usize_to_u64(config.ssm_conv_kernel, "ssm conv1d kernel")?,
                        usize_to_u64(
                            config.ssm_inner_size
                                + 2 * config.ssm_state_size * config.ssm_group_count,
                            "ssm conv1d cols",
                        )?,
                    ],
                )?;
                let ssm_conv1d_b = load_f32_tensor(
                    source,
                    &format!("{prefix}.ssm_conv1d.bias"),
                    &[usize_to_u64(
                        config.ssm_inner_size + 2 * config.ssm_state_size * config.ssm_group_count,
                        "ssm conv1d bias",
                    )?],
                )?;
                // ssm_dt.bias is per SSM head (96).
                let ssm_dt_bias = load_f32_tensor(
                    source,
                    &format!("{prefix}.ssm_dt.bias"),
                    &[usize_to_u64(config.ssm_time_step_rank, "ssm dt bias")?],
                )?;
                // ssm_a is shape [1, time_step_rank=96].
                let ssm_a_log = load_f32_tensor(
                    source,
                    &format!("{prefix}.ssm_a"),
                    &[1, usize_to_u64(config.ssm_time_step_rank, "ssm a log")?],
                )?;
                // ssm_d is [1, time_step_rank] in this checkpoint.
                let ssm_d = load_f32_tensor(
                    source,
                    &format!("{prefix}.ssm_d"),
                    &[1, usize_to_u64(config.ssm_time_step_rank, "ssm d")?],
                )?;
                // ssm_norm.weight is (per_group, n_groups). per_group is
                // 960 (= inner_size 7680 / n_groups 8) in this checkpoint.
                let ssm_norm = load_f32_tensor(
                    source,
                    &format!("{prefix}.ssm_norm.weight"),
                    &[
                        usize_to_u64(
                            config.ssm_inner_size / config.ssm_group_count,
                            "ssm norm per-group",
                        )?,
                        usize_to_u64(config.ssm_group_count, "ssm norm groups")?,
                    ],
                )?;
                let ssm_out = load_weight(
                    source,
                    &format!("{prefix}.ssm_out.weight"),
                    config.ssm_inner_size,
                    config.n_embd,
                )?;
                (
                    Some(ssm_in),
                    Some(ssm_conv1d_w),
                    Some(ssm_conv1d_b),
                    Some(ssm_dt_bias),
                    Some(ssm_a_log),
                    Some(ssm_d),
                    Some(ssm_norm),
                    Some(ssm_out),
                )
            } else {
                (None, None, None, None, None, None, None, None)
            };
        let _ = this_has_attn;
        layers.push(NemotronLayerWeights {
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            attn_q_norm,
            attn_k_norm,
            ffn_norm,
            w_up,
            w_down,
            moe,
            ssm_in,
            ssm_conv1d_w,
            ssm_conv1d_b,
            ssm_dt_bias,
            ssm_a_log,
            ssm_d,
            ssm_norm,
            ssm_out,
        });
    }
    Ok(layers)
}
