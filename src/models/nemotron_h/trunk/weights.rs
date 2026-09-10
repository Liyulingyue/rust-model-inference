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
    pub ssm_conv1d: Option<Weight<'a>>,
    pub ssm_dt: Option<Weight<'a>>,
    pub ssm_a: Option<Weight<'a>>,
    pub ssm_d: Option<Weight<'a>>,
    pub ssm_norm: Option<Weight<'a>>,
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
        let (ssm_in, ssm_conv1d, ssm_dt, ssm_a, ssm_d, ssm_norm, ssm_out) = if this_has_ssm {
            let ssm_in = static_q8_into_weight(
                source,
                &format!("{prefix}.ssm_in.weight"),
                config.ssm_inner_size,
                2 * config.n_embd,
            );
            let ssm_conv1d = static_q8_into_weight(
                source,
                &format!("{prefix}.ssm_conv1d.weight"),
                config.ssm_conv_kernel,
                config.ssm_inner_size + 2 * config.ssm_group_count,
            );
            // ssm_dt is stored as a bias vector (F32) of length inner_size.
            // Real Mamba2: dt = softplus(linear(x) + dt_bias). For now the
            // forward pass is a no-op, so we just load the bytes for future
            // use.
            let _ssm_dt_bytes: &[u8] = source
                .tensor_slice(&format!("{prefix}.ssm_dt.bias"))
                .expect("ssm_dt.bias not found");
            // Modeled as a 1×1 weight to keep the field type uniform.
            let ssm_dt = static_q8_into_weight(
                source,
                &format!("{prefix}.ssm_dt.bias"),
                1,
                1,
            );
            let ssm_a = static_q8_into_weight(
                source,
                &format!("{prefix}.ssm_a"),
                config.ssm_state_size,
                config.ssm_inner_size,
            );
            let ssm_d = static_q8_into_weight(
                source,
                &format!("{prefix}.ssm_d"),
                config.ssm_inner_size,
                1,
            );
            let ssm_norm = static_q8_into_weight(
                source,
                &format!("{prefix}.ssm_norm.weight"),
                config.ssm_inner_size,
                1,
            );
            let ssm_out = static_q8_into_weight(
                source,
                &format!("{prefix}.ssm_out.weight"),
                config.n_embd,
                config.ssm_inner_size,
            );
            (
                Some(ssm_in),
                Some(ssm_conv1d),
                Some(ssm_dt),
                Some(ssm_a),
                Some(ssm_d),
                Some(ssm_norm),
                Some(ssm_out),
            )
        } else {
            (None, None, None, None, None, None, None)
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
            ssm_conv1d,
            ssm_dt,
            ssm_a,
            ssm_d,
            ssm_norm,
            ssm_out,
        });
    }
    Ok(layers)
}
