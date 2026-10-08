//! Qwen-Image-2.1 diffusion transformer (7B unified-stream DiT).
//!
//! Adapted against the pinned stable-diffusion.cpp oracle
//! (tools/oracle/qwen_image_2_1) with per-checkpoint bit parity. The GGUF
//! carries no metadata (kv=0): the architecture is identified by its complete
//! tensor contract and the config is detected from tensor shapes, matching
//! the oracle's `QwenImage21Config::detect_from_weights`.

use crate::core::tensor::{GGMLType, TensorInfo, TensorSource};

pub mod dit;
mod pipeline;
pub mod text;
pub mod vae;

pub(crate) use pipeline::sample;
pub use pipeline::{flow_sigmas, rgba_bytes};

pub(crate) use dit::QwenImage21Dit;

pub(crate) struct Condition {
    pub values: Vec<f32>,
    pub image_slots: Vec<usize>,
}

pub(crate) struct ReferenceLatent {
    pub values: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

/// Prefix every tensor in the diffusion GGUF carries.
pub(crate) const PREFIX: &str = "model.diffusion_model";

const EXPECTED_IN_CHANNELS: usize = 64;
const EXPECTED_OUT_CHANNELS: usize = 64;
const EXPECTED_HIDDEN_SIZE: usize = 4096;
const EXPECTED_HEAD_DIM: usize = 128;
const EXPECTED_INTERMEDIATE_SIZE: usize = 12288;
const EXPECTED_NUM_LAYERS: usize = 32;

pub const DEFAULT_LATENT_SIDE: usize = 16;
pub const DEFAULT_TIMESTEP: f32 = 500.0;
const DEFAULT_CONTEXT_LEN: usize = 128;
const DEFAULT_SEED: u32 = 1_234_567;

/// Prepare the DiT-only inputs, matching the oracle's latent-then-context LCG.
pub(crate) fn prepare_dit_inputs(
    config: &QwenImage21Config,
    latent: Option<Vec<f32>>,
    context: Option<Vec<f32>>,
    width: usize,
    height: usize,
    timestep: f32,
) -> Result<(Vec<f32>, Vec<f32>, usize), String> {
    if width == 0 || height == 0 {
        return Err("Qwen-Image-2.1 latent width and height must be positive".into());
    }
    if !timestep.is_finite() {
        return Err("Qwen-Image-2.1 timestep must be finite".into());
    }
    let image_tokens = width
        .checked_mul(height)
        .ok_or("Qwen-Image-2.1 latent dimensions overflow")?;
    let expected_latent = config
        .in_channels
        .checked_mul(image_tokens)
        .ok_or("Qwen-Image-2.1 latent dimensions overflow")?;
    let default_context = config
        .context_dim
        .checked_mul(DEFAULT_CONTEXT_LEN)
        .ok_or("Qwen-Image-2.1 context dimensions overflow")?;
    let mut state = DEFAULT_SEED;
    let mut next_default = |count: usize| -> Vec<f32> {
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) & 0xffff) as f32 / 32_768.0 - 1.0
            })
            .collect()
    };
    let latent = latent.unwrap_or_else(|| next_default(expected_latent));
    let context = context.unwrap_or_else(|| next_default(default_context));
    if latent.len() != expected_latent {
        return Err(format!(
            "Qwen-Image-2.1 latent must hold {}x{width}x{height} values, got {}",
            config.in_channels,
            latent.len()
        ));
    }
    if context.is_empty() || context.len() % config.context_dim != 0 {
        return Err(format!(
            "Qwen-Image-2.1 context must be rows of {} values, got {}",
            config.context_dim,
            context.len()
        ));
    }
    if !latent.iter().all(|value| value.is_finite())
        || !context.iter().all(|value| value.is_finite())
    {
        return Err("Qwen-Image-2.1 latent and context must contain only finite values".into());
    }
    let context_len = context.len() / config.context_dim;
    Ok((latent, context, context_len))
}

#[cfg(test)]
mod input_tests {
    use super::*;

    #[test]
    fn default_inputs_follow_oracle_order_and_reject_invalid_dimensions() {
        let config = QwenImage21Config {
            in_channels: 1,
            out_channels: 1,
            hidden_size: 2,
            context_dim: 2,
            head_dim: 1,
            intermediate_size: 1,
            num_layers: 1,
        };
        let (latent, context, context_len) =
            prepare_dit_inputs(&config, None, None, 1, 1, DEFAULT_TIMESTEP).unwrap();
        assert_eq!(latent[0].to_bits(), 0xbf66_bc00);
        assert_eq!(context[0].to_bits(), 0xbe7d_a000);
        assert_eq!(context_len, DEFAULT_CONTEXT_LEN);
        assert!(prepare_dit_inputs(&config, None, None, 0, 1, DEFAULT_TIMESTEP).is_err());
    }
}

/// Recognizes this no-metadata DiT family; `validate_dit` then checks the
/// complete contract and reports a useful error for damaged GGUFs.
pub fn matches_signature(source: &dyn TensorSource) -> bool {
    source
        .tensor_info(&format!("{PREFIX}.txt_in.text_norm.weight"))
        .is_some()
        && source
            .tensor_info(&format!("{PREFIX}.img_in.weight"))
            .is_some()
        && source
            .tensor_info(&format!("{PREFIX}.transformer_blocks.0.attn.norm_q.weight"))
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

        let dimension = |name: &str, value: u64| {
            usize::try_from(value).map_err(|_| format!("Invalid {name} dimension: {value}"))
        };
        let in_channels = dimension("img_in.weight", img_in.dims[0])?;
        let hidden_size = dimension("img_in.weight", img_in.dims[1])?;
        let out_channels = dimension("proj_out.weight", proj_out.dims[1])?;
        let context_dim = dimension("txt_in.in_layer.weight", txt_in.dims[0])?;
        let head_dim = dimension("transformer_blocks.0.attn.norm_q.weight", norm_q.dims[0])?;
        let intermediate_size = dimension(
            "transformer_blocks.0.img_mlp.gate_up.weight",
            gate_up.dims[1],
        )? / 2;

        Ok(Self {
            in_channels,
            hidden_size,
            out_channels,
            context_dim,
            head_dim,
            intermediate_size,
            num_layers: EXPECTED_NUM_LAYERS,
        })
    }
}

pub fn validate_dit(source: &dyn TensorSource) -> Result<(), String> {
    let config = QwenImage21Config::detect_from_source(source)?;
    if config.in_channels != EXPECTED_IN_CHANNELS
        || config.out_channels != EXPECTED_OUT_CHANNELS
        || config.hidden_size != EXPECTED_HIDDEN_SIZE
        || config.context_dim != EXPECTED_HIDDEN_SIZE
        || config.head_dim != EXPECTED_HEAD_DIM
        || config.intermediate_size != EXPECTED_INTERMEDIATE_SIZE
        || config.num_layers != EXPECTED_NUM_LAYERS
    {
        return Err(format!(
            "Unsupported Qwen-Image-2.1 DiT config: expected {}x{} in/out, hidden/context {}, head {}, intermediate {}, layers {}; got {:?}",
            EXPECTED_IN_CHANNELS,
            EXPECTED_OUT_CHANNELS,
            EXPECTED_HIDDEN_SIZE,
            EXPECTED_HEAD_DIM,
            EXPECTED_INTERMEDIATE_SIZE,
            EXPECTED_NUM_LAYERS,
            config
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
    if info.dims.as_slice() != &[len] {
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
