//! Nemotron-3 Nano weight loading.
//!
//! Hybrid Mamba-Transformer: every layer carries both attention and SSM
//! weights. We load both branches up front. (Mamba2 SSM weights are
//! stored as `blk.X.ssm_*` per the reference.)

use crate::core::loader::load_static_weight;
use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::ops::kernel::QuantizedTensor;
use crate::ops::kernel::Weight;

use super::config::NemotronConfig;

fn usize_to_u64(v: usize, name: &str) -> Result<u64, String> {
    u64::try_from(v).map_err(|_| format!("{name} does not fit u64"))
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

pub(crate) fn static_q8_into_weight(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Weight<'static> {
    let bytes = source
        .tensor_slice(name)
        .unwrap_or_else(|| panic!("tensor {name} not found"));
    let info = source
        .tensor_info(name)
        .unwrap_or_else(|| panic!("tensor info {name} not found"));
    let bytes_static: &'static [u8] =
        unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes) };
    Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes_static,
        info.ggml_type,
        n_in,
        n_out,
    ))
}

pub fn load_layers(
    source: &dyn TensorSource,
    config: &NemotronConfig,
) -> Result<Vec<NemotronLayerWeights<'static>>, String> {
    let mut layers = Vec::with_capacity(config.n_layer);
    let n_attn_q = config.n_head * config.n_embd_head_k;
    let n_attn_k = config.n_head_kv * config.n_embd_head_k;
    let n_attn_v = config.n_head_kv * config.n_embd_head_v;
    let head_dim = config.head_dim();
    let has_qk_norm = source
        .tensor_info(&format!("blk.0.attn_q_norm.weight"))
        .is_some();
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
    for layer_idx in 0..config.n_layer {
        let prefix = format!("blk.{layer_idx}");
        let attn_norm = load_f32_tensor(
            source,
            &format!("{prefix}.attn_norm.weight"),
            &[config.n_embd as u64],
        )?;
        let this_has_attn = layer_has_attn(layer_idx);
        let this_has_ssm = layer_has_ssm(layer_idx);
        let (wq, wk, wv, wo, attn_q_norm, attn_k_norm) = if this_has_attn {
            let wq = load_static_weight(
                source,
                &format!("{prefix}.attn_q.weight"),
                n_attn_q,
                config.n_embd,
            );
            let wk = load_static_weight(
                source,
                &format!("{prefix}.attn_k.weight"),
                n_attn_k,
                config.n_embd,
            );
            let wv = load_static_weight(
                source,
                &format!("{prefix}.attn_v.weight"),
                n_attn_v,
                config.n_embd,
            );
            let wo = load_static_weight(
                source,
                &format!("{prefix}.attn_output.weight"),
                config.n_embd,
                n_attn_v,
            );
            let qn = if has_qk_norm {
                Some(load_f32_tensor(
                    source,
                    &format!("{prefix}.attn_q_norm.weight"),
                    &[head_dim as u64],
                )?)
            } else {
                None
            };
            let kn = if has_qk_norm {
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
        let has_layer_ffn = layer_has_ffn(layer_idx);
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
            let wu = load_static_weight(
                source,
                &format!("{prefix}.ffn_up.weight"),
                config.n_ff,
                config.n_embd,
            );
            let wd = load_static_weight(
                source,
                &format!("{prefix}.ffn_down.weight"),
                config.n_embd,
                config.n_ff,
            );
            (ffn_norm, Some(wu), Some(wd))
        } else {
            (None, None, None)
        };
        // Mamba2 SSM branch — present in ~21 of 42 layers.
        let (
            ssm_in,
            ssm_conv1d_w,
            ssm_conv1d_b,
            ssm_dt_bias,
            ssm_a_log,
            ssm_d,
            ssm_norm,
            ssm_out,
        ) = if this_has_ssm {
            // ssm_in.shape is (n_embd, d_in_proj) where
            // d_in_proj = 2*d_inner + 2*n_group*d_state + dt_rank
            // (matches llama.cpp nemotron-h.cpp). Earlier read attempts
            // swapped the args and treated ssm_in as (d_inner, n_embd),
            // which corrupted the dispatch and produced degenerate output.
            let d_in_proj =
                2 * config.ssm_inner_size
                    + 2 * config.ssm_state_size * config.ssm_group_count
                    + config.ssm_time_step_rank;
            let ssm_in = static_q8_into_weight(
                source,
                &format!("{prefix}.ssm_in.weight"),
                d_in_proj,
                config.n_embd,
            );
            // Mamba2 SSM tensors in this checkpoint are F32 (not quantized).
            // Layout per the reference:
            //   ssm_conv1d.weight (4, 9728) — fused causal conv + b + c outputs
            //   ssm_conv1d.bias   (9728,)   — bias for the same fused tensor
            //   ssm_dt.bias       (inner,)  — dt bias
            //   ssm_a             (1, inner) — A_log
            //   ssm_d             (1, inner) — D skip
            //   ssm_norm.weight   (per_group, n_groups) — group RMSNorm
            //   ssm_out.weight    (n_embd, inner) — out_proj (Q5_K, quantized)
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
                    config.ssm_inner_size
                        + 2 * config.ssm_state_size * config.ssm_group_count,
                    "ssm conv1d bias",
                )?],
            )?;
            // ssm_dt.bias is per-time_step_rank (96), not per-inner_size.
            // A proper Mamba2 would project (B, L, dt_rank) → (B, L,
            // inner_size); this checkpoint appears to broadcast dt to
            // per-channel via the time_step_rank dim. The dt is
            // effectively a per-timestep-rank bias, not a per-channel
            // weight.
            let ssm_dt_bias = load_f32_tensor(
                source,
                &format!("{prefix}.ssm_dt.bias"),
                &[usize_to_u64(config.ssm_time_step_rank, "ssm dt bias")?],
            )?;
            // ssm_a is shape [1, time_step_rank=96]. The A_log is the
            // negative exponential of state decay.
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
            // 960 (= inner_size 7680 / n_groups 8) in this checkpoint;
            // the per-group dim doesn't follow a standard formula
            // (it's not d_state, dt_rank, or anything conventional).
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
            let ssm_out = static_q8_into_weight(
                source,
                &format!("{prefix}.ssm_out.weight"),
                config.n_embd,
                config.ssm_inner_size,
            );
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
