//! Falcon-H1 weight loading.
//!
//! Every layer carries BOTH an attention branch and a Mamba2 SSM branch
//! (parallel hybrid). Unlike Nemotron-H, all branches are mandatory on all
//! layers and the FFN is gated (SwiGLU). Note the GGUF naming quirks
//! confirmed against the 1.5B checkpoint: `ffn_norm`, `ssm_a` and `ssm_d`
//! have no `.weight` suffix; `attn_norm` and `ssm_norm` do.

use crate::core::tensor::{load_f32_tensor, GGMLType, TensorSource};
use crate::ops::kernel::QuantizedTensor;
use crate::ops::kernel::Weight;

use super::config::FalconH1Config;

fn usize_to_u64(v: usize, name: &str) -> Result<u64, String> {
    u64::try_from(v).map_err(|_| format!("{name} does not fit u64"))
}

pub struct FalconH1LayerWeights<'a> {
    /// Pre-attention AND pre-SSM RMSNorm (the same normed input feeds both
    /// branches, mirroring llama.cpp's double `build_norm(inpL, attn_norm)`).
    pub attn_norm: Vec<f32>,
    /// Attention branch (GQA + RoPE).
    pub wq: Weight<'a>,
    pub wk: Weight<'a>,
    pub wv: Weight<'a>,
    pub wo: Weight<'a>,
    /// Raw Q8_0 bytes for the Q8_0 attention weights (used by the
    /// falcon-local matmul in `forward.rs` to match llama.cpp's
    /// `vec_dot_q8_0_q8_0` reduction order). Empty for non-Q8_0
    /// types — the falcon path only supports Q8_0 activations, so
    /// mixed-precision tensors are handled via the generic kernel.
    pub wq_bytes: &'a [u8],
    pub wk_bytes: &'a [u8],
    pub wv_bytes: &'a [u8],
    pub wo_bytes: &'a [u8],
    /// FFN branch (SwiGLU). `ffn_norm` is stored WITHOUT a `.weight`
    /// suffix in falcon-h1 GGUFs.
    pub ffn_norm: Vec<f32>,
    pub w_gate: Weight<'a>,
    pub w_up: Weight<'a>,
    pub w_down: Weight<'a>,
    pub w_gate_bytes: &'a [u8],
    pub w_up_bytes: &'a [u8],
    pub w_down_bytes: &'a [u8],
    /// Mamba2 SSM branch.
    pub ssm_in: Weight<'a>,
    pub ssm_conv1d_w: Vec<f32>,
    pub ssm_conv1d_b: Vec<f32>,
    pub ssm_dt_bias: Vec<f32>,
    pub ssm_a: Vec<f32>,
    pub ssm_d: Vec<f32>,
    pub ssm_norm: Vec<f32>,
    pub ssm_out: Weight<'a>,
    pub ssm_in_bytes: &'a [u8],
    pub ssm_out_bytes: &'a [u8],
}

/// Single source of truth for the GGML types the kernel layer accepts
/// for matmul. Mirrors `QuantizedTensor::from_bytes`
/// (`src/ops/kernel/quantized_tensor.rs`).
fn is_supported_weight_type(t: GGMLType) -> bool {
    use GGMLType::*;
    matches!(
        t,
        F32 | F16
            | BF16
            | Q8_0
            | Q4_0
            | Q4_1
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
            "Invalid tensor {name}: shape {:?} type {:?}; expected [{n_in}, {n_out}] F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q2K/Q3K/Q4K/Q5K/Q6K/IQ*",
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

/// Same as `load_weight` but also returns the raw tensor bytes (for
/// falcon-local kernels that need direct access, e.g. the hsum-ggml
/// matmul variant). The returned byte slice has lifetime `'static`,
/// transmuted from the GGUF mmap source.
pub(crate) fn load_weight_and_bytes(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<(Weight<'static>, &'static [u8]), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [n_in as u64, n_out as u64] || !is_supported_weight_type(info.ggml_type) {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected [{n_in}, {n_out}] F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q2K/Q3K/Q4K/Q5K/Q6K/IQ*",
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
    let weight = Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes_static,
        info.ggml_type,
        n_in,
        n_out,
    ));
    Ok((weight, bytes_static))
}

pub fn load_layers(
    source: &dyn TensorSource,
    config: &FalconH1Config,
) -> Result<Vec<FalconH1LayerWeights<'static>>, String> {
    let mut layers = Vec::with_capacity(config.n_layer);
    let n_attn_q = config.n_head * config.n_embd_head_k;
    let n_attn_kv = config.n_head_kv * config.n_embd_head_k;
    let n_attn_v = config.n_head_kv * config.n_embd_head_v;
    let n_attn_out = config.n_head * config.n_embd_head_v;
    for layer_idx in 0..config.n_layer {
        let prefix = format!("blk.{layer_idx}");
        let expect = |name: &str| -> String { format!("layer {layer_idx}: {name}") };
        let attn_norm = load_f32_tensor(
            source,
            &format!("{prefix}.attn_norm.weight"),
            &[config.n_embd as u64],
        )
        .map_err(|e| format!("attn_norm: {e}"))?;
        let wq = super::weights::load_weight_and_bytes(
            source,
            &format!("{prefix}.attn_q.weight"),
            config.n_embd,
            n_attn_q,
        )
        .map_err(|e| expect("attn_q") + &format!(": {e}"))?;
        let wk = super::weights::load_weight_and_bytes(
            source,
            &format!("{prefix}.attn_k.weight"),
            config.n_embd,
            n_attn_kv,
        )
        .map_err(|e| expect("attn_k") + &format!(": {e}"))?;
        let wv = super::weights::load_weight_and_bytes(
            source,
            &format!("{prefix}.attn_v.weight"),
            config.n_embd,
            n_attn_v,
        )
        .map_err(|e| expect("attn_v") + &format!(": {e}"))?;
        let wo = super::weights::load_weight_and_bytes(
            source,
            &format!("{prefix}.attn_output.weight"),
            n_attn_out,
            config.n_embd,
        )
        .map_err(|e| expect("attn_output") + &format!(": {e}"))?;
        let ffn_norm = load_f32_tensor(
            source,
            &format!("{prefix}.ffn_norm"),
            &[config.n_embd as u64],
        )
        .map_err(|e| format!("ffn_norm: {e}"))?;
        let w_gate = super::weights::load_weight_and_bytes(
            source,
            &format!("{prefix}.ffn_gate.weight"),
            config.n_embd,
            config.n_ff,
        )
        .map_err(|e| expect("ffn_gate") + &format!(": {e}"))?;
        let w_up = super::weights::load_weight_and_bytes(
            source,
            &format!("{prefix}.ffn_up.weight"),
            config.n_embd,
            config.n_ff,
        )
        .map_err(|e| expect("ffn_up") + &format!(": {e}"))?;
        let w_down = super::weights::load_weight_and_bytes(
            source,
            &format!("{prefix}.ffn_down.weight"),
            config.n_ff,
            config.n_embd,
        )
        .map_err(|e| expect("ffn_down") + &format!(": {e}"))?;
        // Mamba2 SSM branch. Tensor layouts match llama.cpp
        // mamba-base.cpp `build_mamba2_layer`:
        //   ssm_in.weight      (n_embd, d_in_proj)
        //   ssm_conv1d.weight  (d_conv, d_inner + 2*n_group*d_state)
        //   ssm_conv1d.bias    (d_inner + 2*n_group*d_state,)
        //   ssm_dt.bias        (n_head,)
        //   ssm_a / ssm_d      (1, n_head)  -- no `.weight` suffix
        //   ssm_norm.weight    (d_inner / n_group, n_group)
        //   ssm_out.weight     (d_inner, n_embd)
        let ssm_in = super::weights::load_weight_and_bytes(
            source,
            &format!("{prefix}.ssm_in.weight"),
            config.n_embd,
            config.ssm_in_proj_dim(),
        )
        .map_err(|e| expect("ssm_in") + &format!(": {e}"))?;
        let ssm_conv1d_w = load_f32_tensor(
            source,
            &format!("{prefix}.ssm_conv1d.weight"),
            &[
                usize_to_u64(config.ssm_conv_kernel, "ssm conv1d kernel")?,
                usize_to_u64(config.ssm_conv_cols(), "ssm conv1d cols")?,
            ],
        )
        .map_err(|e| format!("conv1d: {e}"))?;
        let ssm_conv1d_b = load_f32_tensor(
            source,
            &format!("{prefix}.ssm_conv1d.bias"),
            &[usize_to_u64(config.ssm_conv_cols(), "ssm conv1d bias")?],
        )
        .map_err(|e| format!("conv1d bias: {e}"))?;
        let ssm_dt_bias = load_f32_tensor(
            source,
            &format!("{prefix}.ssm_dt.bias"),
            &[usize_to_u64(config.ssm_n_head(), "ssm dt bias")?],
        )
        .map_err(|e| format!("dt bias: {e}"))?;
        let ssm_a = load_f32_tensor(
            source,
            &format!("{prefix}.ssm_a"),
            &[1, usize_to_u64(config.ssm_n_head(), "ssm a")?],
        )
        .map_err(|e| format!("ssm_a: {e}"))?;
        let ssm_d = load_f32_tensor(
            source,
            &format!("{prefix}.ssm_d"),
            &[1, usize_to_u64(config.ssm_n_head(), "ssm d")?],
        )
        .map_err(|e| format!("ssm_d: {e}"))?;
        // GGUF drops trailing size-1 dims, so ssm_norm may arrive as the
        // logical 2-D [d_inner/n_group, n_group] or (n_group == 1) as a
        // flat [d_inner]. Accept both.
        let ssm_norm_name = format!("{prefix}.ssm_norm.weight");
        let ssm_norm_2d = vec![
            (config.ssm_inner_size / config.ssm_group_count) as u64,
            config.ssm_group_count as u64,
        ];
        let ssm_norm_1d = vec![config.ssm_inner_size as u64];
        let ssm_norm_info = source
            .tensor_info(&ssm_norm_name)
            .ok_or_else(|| format!("Missing tensor: {ssm_norm_name}"))?;
        if ssm_norm_info.dims != ssm_norm_2d && ssm_norm_info.dims != ssm_norm_1d
            || ssm_norm_info.ggml_type != GGMLType::F32
        {
            return Err(format!(
                "Invalid tensor {ssm_norm_name}: shape {:?} type {:?}; expected [{}] F32 or [{}, {}] F32",
                ssm_norm_info.dims,
                ssm_norm_info.ggml_type,
                config.ssm_inner_size,
                config.ssm_inner_size / config.ssm_group_count,
                config.ssm_group_count,
            ));
        }
        let ssm_norm = load_f32_tensor(
            source,
            &ssm_norm_name,
            &ssm_norm_1d,
        )
        .map_err(|e| format!("ssm_norm: {e}"))?;
        let ssm_out = super::weights::load_weight_and_bytes(
            source,
            &format!("{prefix}.ssm_out.weight"),
            config.ssm_inner_size,
            config.n_embd,
        )
        .map_err(|e| expect("ssm_out") + &format!(": {e}"))?;
        let (wq, wq_bytes) = wq;
        let (wk, wk_bytes) = wk;
        let (wv, wv_bytes) = wv;
        let (wo, wo_bytes) = wo;
        let (w_gate, w_gate_bytes) = w_gate;
        let (w_up, w_up_bytes) = w_up;
        let (w_down, w_down_bytes) = w_down;
        let (ssm_in, ssm_in_bytes) = ssm_in;
        let (ssm_out, ssm_out_bytes) = ssm_out;
        layers.push(FalconH1LayerWeights {
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            wq_bytes,
            wk_bytes,
            wv_bytes,
            wo_bytes,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
            w_gate_bytes,
            w_up_bytes,
            w_down_bytes,
            ssm_in,
            ssm_conv1d_w,
            ssm_conv1d_b,
            ssm_dt_bias,
            ssm_a,
            ssm_d,
            ssm_norm,
            ssm_out,
            ssm_in_bytes,
            ssm_out_bytes,
        });
    }
    Ok(layers)
}
