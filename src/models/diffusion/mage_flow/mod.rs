//! Microsoft Mage-Flow NR-MMDiT transformer.
//!
//! The ModelScope checkpoints all share this contract.  This module consumes
//! the lossless BF16 GGUF emitted by `tools/converter/mage_flow` and exposes a
//! deterministic DiT forward for parity harnesses.  Text encoding and Mage-VAE
//! are intentionally separate components; the raw transformer path does not
//! pretend to generate pixels without those inputs.

use crate::core::tensor::{GGMLType, TensorSource};

pub mod dit;
pub mod text;
pub mod vae;

pub use dit::MageFlowDit;

pub const IN_CHANNELS: usize = 128;
pub const OUT_CHANNELS: usize = 128;
pub const CONTEXT_DIM: usize = 2560;
pub const HIDDEN: usize = 3072;
pub const HEADS: usize = 24;
pub const HEAD_DIM: usize = 128;
pub const LAYERS: usize = 12;
pub const FFN: usize = 12_288;
pub const AXES_DIM: [usize; 3] = [16, 56, 56];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MageFlowConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub context_dim: usize,
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub ffn: usize,
}

pub const CONFIG: MageFlowConfig = MageFlowConfig {
    in_channels: IN_CHANNELS,
    out_channels: OUT_CHANNELS,
    context_dim: CONTEXT_DIM,
    hidden: HIDDEN,
    heads: HEADS,
    layers: LAYERS,
    ffn: FFN,
};

pub fn matches_signature(source: &dyn TensorSource) -> bool {
    source
        .metadata("general.architecture")
        .and_then(crate::core::tensor::MetaValue::to_string_val)
        == Some("mage_flow")
}

pub fn validate_dit(source: &dyn TensorSource) -> Result<(), String> {
    if !matches_signature(source) {
        return Err("Expected general.architecture=mage_flow".into());
    }
    let variant = source
        .metadata("mage_flow.variant")
        .and_then(crate::core::tensor::MetaValue::to_string_val);
    if !matches!(
        variant,
        Some("base" | "flow" | "turbo" | "edit-base" | "edit" | "edit-turbo")
    ) {
        return Err("Missing or unsupported mage_flow.variant".into());
    }
    let matrix = |name: &str, dims: &[u64]| -> Result<(), String> {
        let info = source
            .tensor_info(name)
            .ok_or_else(|| format!("Missing Mage-Flow tensor: {name}"))?;
        if info.dims != dims || info.ggml_type != GGMLType::BF16 {
            return Err(format!(
                "Invalid Mage-Flow tensor {name}: expected BF16 {dims:?}, got {:?} {:?}",
                info.ggml_type, info.dims
            ));
        }
        let bytes = source
            .tensor_slice(name)
            .ok_or_else(|| format!("Missing Mage-Flow data: {name}"))?;
        if Some(bytes.len() as u64) != info.checked_nbytes() {
            return Err(format!("Invalid Mage-Flow tensor payload: {name}"));
        }
        Ok(())
    };
    let vector = |name: &str, len: usize| matrix(name, &[len as u64]);
    matrix("img_in.weight", &[IN_CHANNELS as u64, HIDDEN as u64])?;
    vector("img_in.bias", HIDDEN)?;
    matrix("txt_in.weight", &[CONTEXT_DIM as u64, HIDDEN as u64])?;
    vector("txt_in.bias", HIDDEN)?;
    vector("txt_norm.weight", CONTEXT_DIM)?;
    matrix(
        "norm_out.linear.weight",
        &[HIDDEN as u64, (2 * HIDDEN) as u64],
    )?;
    vector("norm_out.linear.bias", 2 * HIDDEN)?;
    matrix("proj_out.weight", &[HIDDEN as u64, OUT_CHANNELS as u64])?;
    vector("proj_out.bias", OUT_CHANNELS)?;
    matrix(
        "time_text_embed.timestep_embedder.linear_1.weight",
        &[256, HIDDEN as u64],
    )?;
    vector("time_text_embed.timestep_embedder.linear_1.bias", HIDDEN)?;
    matrix(
        "time_text_embed.timestep_embedder.linear_2.weight",
        &[HIDDEN as u64, HIDDEN as u64],
    )?;
    vector("time_text_embed.timestep_embedder.linear_2.bias", HIDDEN)?;
    for layer in 0..LAYERS {
        let p = format!("transformer_blocks.{layer}");
        for (name, dims) in [
            (
                format!("{p}.img_mod.1.weight"),
                [HIDDEN as u64, (6 * HIDDEN) as u64],
            ),
            (
                format!("{p}.txt_mod.1.weight"),
                [HIDDEN as u64, (6 * HIDDEN) as u64],
            ),
            (
                format!("{p}.attn.to_q.weight"),
                [HIDDEN as u64, HIDDEN as u64],
            ),
            (
                format!("{p}.attn.to_k.weight"),
                [HIDDEN as u64, HIDDEN as u64],
            ),
            (
                format!("{p}.attn.to_v.weight"),
                [HIDDEN as u64, HIDDEN as u64],
            ),
            (
                format!("{p}.attn.add_q_proj.weight"),
                [HIDDEN as u64, HIDDEN as u64],
            ),
            (
                format!("{p}.attn.add_k_proj.weight"),
                [HIDDEN as u64, HIDDEN as u64],
            ),
            (
                format!("{p}.attn.add_v_proj.weight"),
                [HIDDEN as u64, HIDDEN as u64],
            ),
            (
                format!("{p}.attn.to_out.0.weight"),
                [HIDDEN as u64, HIDDEN as u64],
            ),
            (
                format!("{p}.attn.to_add_out.weight"),
                [HIDDEN as u64, HIDDEN as u64],
            ),
            (
                format!("{p}.img_mlp.net.0.proj.weight"),
                [HIDDEN as u64, FFN as u64],
            ),
            (
                format!("{p}.img_mlp.net.2.weight"),
                [FFN as u64, HIDDEN as u64],
            ),
            (
                format!("{p}.txt_mlp.net.0.proj.weight"),
                [HIDDEN as u64, FFN as u64],
            ),
            (
                format!("{p}.txt_mlp.net.2.weight"),
                [FFN as u64, HIDDEN as u64],
            ),
        ] {
            matrix(&name, &dims)?;
        }
        for name in [
            "img_mod.1.bias",
            "txt_mod.1.bias",
            "attn.to_q.bias",
            "attn.to_k.bias",
            "attn.to_v.bias",
            "attn.add_q_proj.bias",
            "attn.add_k_proj.bias",
            "attn.add_v_proj.bias",
            "attn.to_out.0.bias",
            "attn.to_add_out.bias",
            "img_mlp.net.0.proj.bias",
            "img_mlp.net.2.bias",
            "txt_mlp.net.0.proj.bias",
            "txt_mlp.net.2.bias",
        ] {
            let len = if name.contains("mod") {
                6 * HIDDEN
            } else if name.contains("net.0") {
                FFN
            } else {
                HIDDEN
            };
            vector(&format!("{p}.{name}"), len)?;
        }
        for name in [
            "attn.norm_q.weight",
            "attn.norm_k.weight",
            "attn.norm_added_q.weight",
            "attn.norm_added_k.weight",
        ] {
            vector(&format!("{p}.{name}"), HEAD_DIM)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_constants_match_modelscope_config() {
        assert_eq!(
            CONFIG,
            MageFlowConfig {
                in_channels: 128,
                out_channels: 128,
                context_dim: 2560,
                hidden: 3072,
                heads: 24,
                layers: 12,
                ffn: 12288
            }
        );
        assert_eq!(AXES_DIM.iter().sum::<usize>(), HEAD_DIM);
    }
}
