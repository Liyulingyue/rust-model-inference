//! Qwen-Image-2.1 diffusion transformer (7B unified-stream DiT).
//!
//! Adapted against the pinned stable-diffusion.cpp oracle
//! (tools/oracle/qwen_image_2_1) with per-checkpoint bit parity. The GGUF
//! carries no metadata (kv=0): the architecture is identified by tensor
//! names and the config is detected from tensor shapes, matching the
//! oracle's `QwenImage21Config::detect_from_weights`.

use crate::core::tensor::{GGMLType, TensorInfo, TensorSource};

pub mod dit;

pub(crate) use dit::QwenImage21Dit;

/// Prefix every tensor in the diffusion GGUF carries.
pub(crate) const PREFIX: &str = "model.diffusion_model";

/// Detects the Qwen-Image-2.1 signature from tensor names alone, mirroring
/// the oracle's version sniffing (`model.diffusion_model.txt_in.text_norm.weight`).
pub fn matches_signature(source: &dyn TensorSource) -> bool {
    source
        .tensor_info(&format!("{PREFIX}.txt_in.text_norm.weight"))
        .is_some()
}

/// Public wrapper for the app layer; the full contract is checked by
/// [`validate_dit`].
pub fn config_from_source(source: &dyn TensorSource) -> Result<QwenImage21Config, String> {
    QwenImage21Config::detect_from_source(source)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenImage21Config {
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_size: usize,
    pub context_dim: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
}

impl QwenImage21Config {
    /// Derives the config from weight shapes exactly like the oracle.
    pub(crate) fn detect_from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let find = |suffix: &str| source.tensor_info(&format!("{PREFIX}.{suffix}"));
        let img_in = find("img_in.weight").ok_or("Missing tensor: img_in.weight")?;
        let proj_out = find("proj_out.weight").ok_or("Missing tensor: proj_out.weight")?;
        let txt_in =
            find("txt_in.in_layer.weight").ok_or("Missing tensor: txt_in.in_layer.weight")?;
        let norm_q = find("transformer_blocks.0.attn.norm_q.weight")
            .ok_or("Missing tensor: transformer_blocks.0.attn.norm_q.weight")?;
        let gate_up = find("transformer_blocks.0.img_mlp.gate_up.weight")
            .ok_or("Missing tensor: transformer_blocks.0.img_mlp.gate_up.weight")?;
        for (name, info, rank) in [
            ("img_in.weight", img_in, 2),
            ("proj_out.weight", proj_out, 2),
            ("txt_in.in_layer.weight", txt_in, 2),
            ("transformer_blocks.0.attn.norm_q.weight", norm_q, 1),
            ("transformer_blocks.0.img_mlp.gate_up.weight", gate_up, 2),
        ] {
            if info.dims.len() != rank || info.dims.contains(&0) {
                return Err(format!("Invalid {name} dimensions: {:?}", info.dims));
            }
        }
        if gate_up.dims[1] % 2 != 0 {
            return Err(format!(
                "Invalid transformer_blocks.0.img_mlp.gate_up.weight dimensions: {:?}",
                gate_up.dims
            ));
        }

        let mut num_layers = 0usize;
        while source
            .tensor_info(&format!(
                "{PREFIX}.transformer_blocks.{num_layers}.attn.to_q.weight"
            ))
            .is_some()
        {
            num_layers += 1;
        }
        if num_layers == 0 {
            return Err("Model declares no transformer blocks".into());
        }

        Ok(Self {
            in_channels: img_in.dims[0] as usize,
            hidden_size: img_in.dims[1] as usize,
            out_channels: proj_out.dims[1] as usize,
            context_dim: txt_in.dims[0] as usize,
            head_dim: norm_q.dims[0] as usize,
            intermediate_size: gate_up.dims[1] as usize / 2,
            num_layers,
        })
    }
}

pub fn validate_dit(source: &dyn TensorSource) -> Result<(), String> {
    let config = QwenImage21Config::detect_from_source(source)?;
    if config.hidden_size != config.context_dim {
        return Err(format!(
            "Qwen-Image-2.1 requires hidden_size == context_dim, got {} and {}",
            config.hidden_size, config.context_dim
        ));
    }
    if config.hidden_size % config.head_dim != 0 {
        return Err(format!(
            "Qwen-Image-2.1 hidden_size {} not divisible by head_dim {}",
            config.hidden_size, config.head_dim
        ));
    }

    let mut expected: Vec<(String, [u64; 2], GGMLType)> = Vec::new();
    let mut matrix = |name: String, dims: [u64; 2], ggml_type: GGMLType| {
        expected.push((name, dims, ggml_type));
    };
    matrix(
        format!("{PREFIX}.img_in.weight"),
        [config.in_channels as u64, config.hidden_size as u64],
        GGMLType::BF16,
    );
    matrix(
        format!("{PREFIX}.txt_in.in_layer.weight"),
        [config.context_dim as u64, config.hidden_size as u64],
        GGMLType::BF16,
    );
    matrix(
        format!("{PREFIX}.txt_in.out_layer.weight"),
        [config.hidden_size as u64, config.hidden_size as u64],
        GGMLType::BF16,
    );
    matrix(
        format!("{PREFIX}.time_text_embed.timestep_embedder.linear_1.weight"),
        [256, config.hidden_size as u64],
        GGMLType::Q8_0,
    );
    matrix(
        format!("{PREFIX}.time_text_embed.timestep_embedder.linear_2.weight"),
        [config.hidden_size as u64, config.hidden_size as u64],
        GGMLType::Q8_0,
    );
    matrix(
        format!("{PREFIX}.modulation.1.weight"),
        [config.hidden_size as u64, (4 * config.hidden_size) as u64],
        GGMLType::Q8_0,
    );
    matrix(
        format!("{PREFIX}.norm_out.linear.weight"),
        [config.hidden_size as u64, config.hidden_size as u64],
        GGMLType::F32,
    );
    matrix(
        format!("{PREFIX}.proj_out.weight"),
        [config.hidden_size as u64, config.out_channels as u64],
        GGMLType::Q8_0,
    );
    for layer in 0..config.num_layers {
        let prefix = format!("{PREFIX}.transformer_blocks.{layer}");
        matrix(
            format!("{prefix}.attn.to_q.weight"),
            [config.hidden_size as u64, config.hidden_size as u64],
            GGMLType::Q8_0,
        );
        matrix(
            format!("{prefix}.attn.to_k.weight"),
            [config.hidden_size as u64, config.hidden_size as u64],
            GGMLType::Q8_0,
        );
        matrix(
            format!("{prefix}.attn.to_v.weight"),
            [config.hidden_size as u64, config.hidden_size as u64],
            GGMLType::Q8_0,
        );
        matrix(
            format!("{prefix}.attn.to_out.0.weight"),
            [config.hidden_size as u64, config.hidden_size as u64],
            GGMLType::Q8_0,
        );
        matrix(
            format!("{prefix}.img_mlp.gate_up.weight"),
            [
                config.hidden_size as u64,
                (2 * config.intermediate_size) as u64,
            ],
            GGMLType::Q8_0,
        );
        matrix(
            format!("{prefix}.img_mlp.out.weight"),
            [config.intermediate_size as u64, config.hidden_size as u64],
            GGMLType::Q8_0,
        );
    }
    for (name, dims, ggml_type) in expected {
        require_tensor(source, &name, &dims, ggml_type)?;
    }
    for layer in 0..config.num_layers {
        let prefix = format!("{PREFIX}.transformer_blocks.{layer}");
        for name in [
            format!("{prefix}.attn.norm_q.weight"),
            format!("{prefix}.attn.norm_k.weight"),
        ] {
            require_vector(source, &name, config.head_dim as u64, GGMLType::F32)?;
        }
    }
    require_vector(
        source,
        &format!("{PREFIX}.txt_in.text_norm.weight"),
        config.context_dim as u64,
        GGMLType::BF16,
    )?;
    Ok(())
}

fn require_tensor(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64; 2],
    ggml_type: GGMLType,
) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims[..] != dims[..] {
        return Err(format!(
            "Invalid {name} dimensions: expected {dims:?}, got {:?}",
            info.dims
        ));
    }
    if info.ggml_type != ggml_type {
        return Err(format!(
            "Invalid {name} type: expected {ggml_type:?}, got {:?}",
            info.ggml_type
        ));
    }
    require_data(source, name, info)
}

fn require_vector(
    source: &dyn TensorSource,
    name: &str,
    len: u64,
    ggml_type: GGMLType,
) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims.first() != Some(&len)
        || info.dims.len() > 2
        || info.dims.iter().skip(1).any(|&dimension| dimension != 1)
    {
        return Err(format!(
            "Invalid {name} dimensions: expected [{len}], got {:?}",
            info.dims
        ));
    }
    if info.ggml_type != ggml_type {
        return Err(format!(
            "Invalid {name} type: expected {ggml_type:?}, got {:?}",
            info.ggml_type
        ));
    }
    require_data(source, name, info)
}

fn require_data(source: &dyn TensorSource, name: &str, info: &TensorInfo) -> Result<(), String> {
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("Invalid {name} byte size"))?,
    )
    .map_err(|_| format!("Invalid {name} byte size"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!("Invalid {name} byte length"));
    }
    Ok(())
}
