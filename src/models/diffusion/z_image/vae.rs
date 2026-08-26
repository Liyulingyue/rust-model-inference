use super::{validate_component, Component, ZImageRgb};
use crate::core::tensor::TensorSource;
use crate::ops::{dot_f16_f16_bytes, f32_to_f16, silu_approx_inplace, softmax_inplace};
use std::sync::Arc;

const LATENT_CHANNELS: usize = 16;
const GROUPS: usize = 32;
const GROUP_NORM_EPSILON: f32 = 1e-6;

struct VaeConv {
    weight: String,
    bias: Vec<f32>,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
}

impl VaeConv {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: format!("{prefix}.weight"),
            bias: load_f32(source, &format!("{prefix}.bias"), output_channels)?,
            input_channels,
            output_channels,
            kernel,
        })
    }
}

struct VaeNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl VaeNorm {
    fn load(source: &dyn TensorSource, prefix: &str, channels: usize) -> Result<Self, String> {
        Ok(Self {
            weight: load_f32(source, &format!("{prefix}.weight"), channels)?,
            bias: load_f32(source, &format!("{prefix}.bias"), channels)?,
        })
    }
}

struct VaeResidualBlock {
    input_channels: usize,
    output_channels: usize,
    norm1: VaeNorm,
    conv1: VaeConv,
    norm2: VaeNorm,
    conv2: VaeConv,
    shortcut: Option<VaeConv>,
}

impl VaeResidualBlock {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            input_channels,
            output_channels,
            norm1: VaeNorm::load(source, &format!("{prefix}.norm1"), input_channels)?,
            conv1: VaeConv::load(
                source,
                &format!("{prefix}.conv1"),
                input_channels,
                output_channels,
                3,
            )?,
            norm2: VaeNorm::load(source, &format!("{prefix}.norm2"), output_channels)?,
            conv2: VaeConv::load(
                source,
                &format!("{prefix}.conv2"),
                output_channels,
                output_channels,
                3,
            )?,
            shortcut: (input_channels != output_channels)
                .then(|| {
                    VaeConv::load(
                        source,
                        &format!("{prefix}.nin_shortcut"),
                        input_channels,
                        output_channels,
                        1,
                    )
                })
                .transpose()?,
        })
    }
}

struct VaeAttention {
    norm: VaeNorm,
    q: VaeConv,
    k: VaeConv,
    v: VaeConv,
    proj_out: VaeConv,
}

impl VaeAttention {
    fn load(source: &dyn TensorSource, prefix: &str, channels: usize) -> Result<Self, String> {
        Ok(Self {
            norm: VaeNorm::load(source, &format!("{prefix}.norm"), channels)?,
            q: VaeConv::load(source, &format!("{prefix}.q"), channels, channels, 1)?,
            k: VaeConv::load(source, &format!("{prefix}.k"), channels, channels, 1)?,
            v: VaeConv::load(source, &format!("{prefix}.v"), channels, channels, 1)?,
            proj_out: VaeConv::load(source, &format!("{prefix}.proj_out"), channels, channels, 1)?,
        })
    }
}

struct DecoderStage {
    index: usize,
    output_channels: usize,
    blocks: Vec<VaeResidualBlock>,
    upsample: Option<VaeConv>,
}

struct VaeScratch {
    first: Vec<f32>,
    second: Vec<f32>,
    conv_patch: Vec<u16>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    scores: Vec<f32>,
}

impl VaeScratch {
    fn new() -> Self {
        Self {
            first: Vec::new(),
            second: Vec::new(),
            conv_patch: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            scores: Vec::new(),
        }
    }

    fn prepare_features(&mut self, len: usize) -> Result<(), String> {
        resize_f32(&mut self.first, "VAE first feature map", len)?;
        resize_f32(&mut self.second, "VAE second feature map", len)
    }

    fn prepare_attention(&mut self, feature_len: usize, spatial: usize) -> Result<(), String> {
        resize_f32(&mut self.q, "VAE attention query", feature_len)?;
        resize_f32(&mut self.k, "VAE attention key", feature_len)?;
        resize_f32(&mut self.v, "VAE attention value", feature_len)?;
        resize_f32(&mut self.scores, "VAE attention scores", spatial)
    }
}

pub(crate) struct FluxVae {
    source: Arc<dyn TensorSource>,
    conv_in: VaeConv,
    mid_block_1: VaeResidualBlock,
    mid_attention: VaeAttention,
    mid_block_2: VaeResidualBlock,
    stages: Vec<DecoderStage>,
    norm_out: VaeNorm,
    conv_out: VaeConv,
}

impl FluxVae {
    pub(crate) fn load(source: Arc<dyn TensorSource>) -> Result<Self, String> {
        validate_component(source.as_ref(), Component::Vae)?;
        let conv_in = VaeConv::load(source.as_ref(), "decoder.conv_in", 16, 512, 3)?;
        let mid_block_1 = VaeResidualBlock::load(source.as_ref(), "decoder.mid.block_1", 512, 512)?;
        let mid_attention = VaeAttention::load(source.as_ref(), "decoder.mid.attn_1", 512)?;
        let mid_block_2 = VaeResidualBlock::load(source.as_ref(), "decoder.mid.block_2", 512, 512)?;
        let mut stages = Vec::with_capacity(4);
        for (index, input_channels, output_channels) in
            [(3, 512, 512), (2, 512, 512), (1, 512, 256), (0, 256, 128)]
        {
            let mut blocks = Vec::with_capacity(3);
            for block in 0..3 {
                blocks.push(VaeResidualBlock::load(
                    source.as_ref(),
                    &format!("decoder.up.{index}.block.{block}"),
                    if block == 0 {
                        input_channels
                    } else {
                        output_channels
                    },
                    output_channels,
                )?);
            }
            stages.push(DecoderStage {
                index,
                output_channels,
                blocks,
                upsample: (index != 0)
                    .then(|| {
                        VaeConv::load(
                            source.as_ref(),
                            &format!("decoder.up.{index}.upsample.conv"),
                            output_channels,
                            output_channels,
                            3,
                        )
                    })
                    .transpose()?,
            });
        }
        let norm_out = VaeNorm::load(source.as_ref(), "decoder.norm_out", 128)?;
        let conv_out = VaeConv::load(source.as_ref(), "decoder.conv_out", 128, 3, 3)?;
        Ok(Self {
            source,
            conv_in,
            mid_block_1,
            mid_attention,
            mid_block_2,
            stages,
            norm_out,
            conv_out,
        })
    }

    pub(crate) fn decode_rgb(
        &self,
        diffusion_latent: &[f32],
        latent_side: usize,
    ) -> Result<ZImageRgb, String> {
        if latent_side == 0 {
            return Err("Z-Image VAE latent side must be positive".into());
        }
        let latent_spatial = checked_spatial(latent_side, "VAE latent")?;
        let expected_latent = checked_feature_len(LATENT_CHANNELS, latent_spatial, "VAE latent")?;
        if diffusion_latent.len() != expected_latent {
            return Err(format!(
                "Invalid Z-Image VAE latent length: expected {expected_latent}, got {}",
                diffusion_latent.len()
            ));
        }
        if diffusion_latent.iter().any(|value| !value.is_finite()) {
            return Err("Z-Image VAE latent contains NaN or infinity".into());
        }
        let output_side = latent_side
            .checked_mul(8)
            .ok_or_else(|| "Z-Image VAE output side overflow".to_string())?;
        let width = u32::try_from(output_side)
            .map_err(|_| "Z-Image VAE output width does not fit u32".to_string())?;
        let output_spatial = checked_spatial(output_side, "VAE output")?;
        checked_feature_len(3, output_spatial, "VAE RGB output")?;

        let mut current = reserve_f32("VAE mapped latent", expected_latent)?;
        for (output, input) in current.iter_mut().zip(diffusion_latent) {
            *output = diffusion_to_vae(*input);
            if !output.is_finite() {
                return Err("Z-Image VAE mapped latent contains NaN or infinity".into());
            }
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "z_image.vae.mapped_latent",
            None,
            &[latent_side, latent_side, LATENT_CHANNELS],
            &current,
        ));
        let mut scratch = VaeScratch::new();
        let mid_len = checked_feature_len(512, latent_spatial, "VAE middle feature")?;
        resize_f32(&mut scratch.first, "VAE convolution input output", mid_len)?;
        run_conv(
            self.source.as_ref(),
            &self.conv_in,
            &current,
            latent_side,
            &mut scratch.first,
            &mut scratch.conv_patch,
        )?;
        std::mem::swap(&mut current, &mut scratch.first);
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "z_image.vae.conv_in",
            None,
            &[latent_side, latent_side, 512],
            &current,
        ));

        scratch.prepare_features(mid_len)?;
        self.run_residual_block(&self.mid_block_1, &mut current, latent_side, &mut scratch)?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "z_image.vae.mid.block_1",
            None,
            &[latent_side, latent_side, 512],
            &current,
        ));
        scratch.prepare_attention(mid_len, latent_spatial)?;
        self.run_attention(&mut current, latent_side, &mut scratch)?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "z_image.vae.mid.attention",
            None,
            &[latent_side, latent_side, 512],
            &current,
        ));
        self.run_residual_block(&self.mid_block_2, &mut current, latent_side, &mut scratch)?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "z_image.vae.mid",
            None,
            &[latent_side, latent_side, 512],
            &current,
        ));

        let mut side = latent_side;
        for stage in &self.stages {
            if stage.upsample.is_some() != (stage.index != 0) {
                return Err("Invalid loaded VAE decoder stage".into());
            }
            let spatial = checked_spatial(side, "VAE decoder stage")?;
            let stage_channels = stage
                .blocks
                .iter()
                .flat_map(|block| [block.input_channels, block.output_channels])
                .max()
                .ok_or_else(|| "VAE decoder stage has no blocks".to_string())?;
            let stage_len = checked_feature_len(stage_channels, spatial, "VAE decoder stage")?;
            scratch.prepare_features(stage_len)?;
            for block in &stage.blocks {
                self.run_residual_block(block, &mut current, side, &mut scratch)?;
            }
            if let Some(upsample) = &stage.upsample {
                let next_side = side
                    .checked_mul(2)
                    .ok_or_else(|| "VAE upsample side overflow".to_string())?;
                let next_spatial = checked_spatial(next_side, "VAE upsample")?;
                let next_len =
                    checked_feature_len(stage.output_channels, next_spatial, "VAE upsample")?;
                scratch.prepare_features(next_len)?;
                upsample_nearest_into(&current, stage.output_channels, side, &mut scratch.first)?;
                run_conv(
                    self.source.as_ref(),
                    upsample,
                    &scratch.first,
                    next_side,
                    &mut scratch.second,
                    &mut scratch.conv_patch,
                )?;
                std::mem::swap(&mut current, &mut scratch.second);
                side = next_side;
            }
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                &format!("z_image.vae.up.{}", stage.index),
                Some(stage.index),
                &[side, side, stage.output_channels],
                &current,
            ));
        }
        if side != output_side {
            return Err("Invalid Z-Image VAE spatial factor".into());
        }

        let final_len = checked_feature_len(128, output_spatial, "VAE final feature")?;
        scratch.prepare_features(final_len)?;
        group_norm_32_into(
            &current,
            128,
            output_side,
            &self.norm_out.weight,
            &self.norm_out.bias,
            &mut scratch.first,
        )?;
        silu_inplace_checked(&mut scratch.first)?;
        let rgb_len = checked_feature_len(3, output_spatial, "VAE RGB output")?;
        resize_f32(&mut scratch.second, "VAE RGB channels", rgb_len)?;
        run_conv(
            self.source.as_ref(),
            &self.conv_out,
            &scratch.first,
            output_side,
            &mut scratch.second,
            &mut scratch.conv_patch,
        )?;
        std::mem::swap(&mut current, &mut scratch.second);
        if current.len() != rgb_len || current.iter().any(|value| !value.is_finite()) {
            return Err("Invalid Z-Image VAE RGB channel output".into());
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "z_image.vae.rgb_channels",
            None,
            &[output_side, output_side, 3],
            &current,
        ));
        let bytes = rgb_bytes_from_channels(&current, output_side)?;
        let expected_bytes = output_spatial
            .checked_mul(3)
            .ok_or_else(|| "VAE RGB byte length overflow".to_string())?;
        if bytes.len() != expected_bytes {
            return Err("Invalid Z-Image VAE RGB byte length".into());
        }
        Ok(ZImageRgb {
            width,
            height: width,
            bytes,
        })
    }

    fn run_residual_block(
        &self,
        block: &VaeResidualBlock,
        current: &mut Vec<f32>,
        side: usize,
        scratch: &mut VaeScratch,
    ) -> Result<(), String> {
        let spatial = checked_spatial(side, "VAE residual")?;
        let input_len = checked_feature_len(block.input_channels, spatial, "VAE residual input")?;
        let output_len =
            checked_feature_len(block.output_channels, spatial, "VAE residual output")?;
        if current.len() != input_len {
            return Err("Invalid VAE residual input length".into());
        }
        resize_f32(&mut scratch.first, "VAE normalized feature", input_len)?;
        group_norm_32_into(
            current,
            block.input_channels,
            side,
            &block.norm1.weight,
            &block.norm1.bias,
            &mut scratch.first,
        )?;
        silu_inplace_checked(&mut scratch.first)?;
        resize_f32(&mut scratch.second, "VAE first convolution", output_len)?;
        run_conv(
            self.source.as_ref(),
            &block.conv1,
            &scratch.first,
            side,
            &mut scratch.second,
            &mut scratch.conv_patch,
        )?;
        resize_f32(
            &mut scratch.first,
            "VAE second normalized feature",
            output_len,
        )?;
        group_norm_32_into(
            &scratch.second,
            block.output_channels,
            side,
            &block.norm2.weight,
            &block.norm2.bias,
            &mut scratch.first,
        )?;
        silu_inplace_checked(&mut scratch.first)?;
        run_conv(
            self.source.as_ref(),
            &block.conv2,
            &scratch.first,
            side,
            &mut scratch.second,
            &mut scratch.conv_patch,
        )?;

        if let Some(shortcut) = &block.shortcut {
            resize_f32(&mut scratch.first, "VAE projected shortcut", output_len)?;
            let weights = self
                .source
                .tensor_slice(&shortcut.weight)
                .ok_or_else(|| format!("Missing tensor data: {}", shortcut.weight))?;
            add_shortcut_residual_into(
                current,
                &scratch.second,
                block.input_channels,
                block.output_channels,
                side,
                weights,
                Some(&shortcut.bias),
                &mut scratch.first,
                &mut scratch.conv_patch,
            )?;
            std::mem::swap(current, &mut scratch.first);
        } else {
            if current.len() != scratch.second.len() {
                return Err("Invalid VAE identity residual length".into());
            }
            for (branch, residual) in scratch.second.iter_mut().zip(current.iter()) {
                *branch += residual;
                if !branch.is_finite() {
                    return Err("Non-finite VAE residual output".into());
                }
            }
            std::mem::swap(current, &mut scratch.second);
        }
        Ok(())
    }

    fn run_attention(
        &self,
        current: &mut Vec<f32>,
        side: usize,
        scratch: &mut VaeScratch,
    ) -> Result<(), String> {
        let spatial = checked_spatial(side, "VAE attention")?;
        let feature_len = checked_feature_len(512, spatial, "VAE attention")?;
        if current.len() != feature_len {
            return Err("Invalid VAE attention input length".into());
        }
        group_norm_32_into(
            current,
            512,
            side,
            &self.mid_attention.norm.weight,
            &self.mid_attention.norm.bias,
            &mut scratch.first,
        )?;
        run_conv(
            self.source.as_ref(),
            &self.mid_attention.q,
            &scratch.first,
            side,
            &mut scratch.q,
            &mut scratch.conv_patch,
        )?;
        run_conv(
            self.source.as_ref(),
            &self.mid_attention.k,
            &scratch.first,
            side,
            &mut scratch.k,
            &mut scratch.conv_patch,
        )?;
        run_conv(
            self.source.as_ref(),
            &self.mid_attention.v,
            &scratch.first,
            side,
            &mut scratch.v,
            &mut scratch.conv_patch,
        )?;
        one_head_spatial_attention_into(
            &scratch.q,
            &scratch.k,
            &scratch.v,
            512,
            spatial,
            &mut scratch.first,
            &mut scratch.scores,
        )?;
        run_conv(
            self.source.as_ref(),
            &self.mid_attention.proj_out,
            &scratch.first,
            side,
            &mut scratch.second,
            &mut scratch.conv_patch,
        )?;
        for (projected, residual) in scratch.second.iter_mut().zip(current.iter()) {
            *projected += residual;
            if !projected.is_finite() {
                return Err("Non-finite VAE attention residual".into());
            }
        }
        std::mem::swap(current, &mut scratch.second);
        Ok(())
    }
}

pub(crate) fn diffusion_to_vae(value: f32) -> f32 {
    value / 0.3611 + 0.1159
}

pub(crate) fn to_rgb_byte(value: f32) -> u8 {
    (((value.clamp(-1.0, 1.0) + 1.0) * 127.5).round()).clamp(0.0, 255.0) as u8
}

fn checked_spatial(side: usize, name: &str) -> Result<usize, String> {
    side.checked_mul(side)
        .ok_or_else(|| format!("{name} spatial size overflow"))
}

fn checked_feature_len(channels: usize, spatial: usize, name: &str) -> Result<usize, String> {
    channels
        .checked_mul(spatial)
        .ok_or_else(|| format!("{name} feature size overflow"))
}

fn reserve_f32(name: &str, len: usize) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|error| format!("Failed to allocate {name}: {error}"))?;
    values.resize(len, 0.0);
    Ok(values)
}

fn resize_f32(values: &mut Vec<f32>, name: &str, len: usize) -> Result<(), String> {
    if values.capacity() < len {
        let additional = len
            .checked_sub(values.len())
            .ok_or_else(|| format!("Invalid {name} length"))?;
        values
            .try_reserve_exact(additional)
            .map_err(|error| format!("Failed to allocate {name}: {error}"))?;
    }
    values.resize(len, 0.0);
    Ok(())
}

fn load_f32(source: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len()
        != len
            .checked_mul(4)
            .ok_or_else(|| format!("Invalid {name} byte size"))?
    {
        return Err(format!("Invalid {name} byte length"));
    }
    let mut values = reserve_f32(name, len)?;
    for (output, bytes) in values.iter_mut().zip(bytes.chunks_exact(4)) {
        *output = f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("Non-finite tensor: {name}"));
    }
    Ok(values)
}

fn run_conv(
    source: &dyn TensorSource,
    conv: &VaeConv,
    input: &[f32],
    side: usize,
    output: &mut [f32],
    patch: &mut Vec<u16>,
) -> Result<(), String> {
    let weights = source
        .tensor_slice(&conv.weight)
        .ok_or_else(|| format!("Missing tensor data: {}", conv.weight))?;
    conv_f16_into(
        input,
        conv.input_channels,
        side,
        weights,
        conv.output_channels,
        conv.kernel,
        Some(&conv.bias),
        output,
        patch,
    )
}

fn conv_f16_into(
    input: &[f32],
    input_channels: usize,
    side: usize,
    weights: &[u8],
    output_channels: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
    patch: &mut Vec<u16>,
) -> Result<(), String> {
    if side == 0 || !matches!(kernel, 1 | 3) {
        return Err("Invalid VAE convolution shape".into());
    }
    let spatial = checked_spatial(side, "VAE convolution")?;
    let input_len = checked_feature_len(input_channels, spatial, "VAE convolution input")?;
    let output_len = checked_feature_len(output_channels, spatial, "VAE convolution output")?;
    let weight_elements = kernel
        .checked_mul(kernel)
        .and_then(|value| value.checked_mul(input_channels))
        .and_then(|value| value.checked_mul(output_channels))
        .ok_or_else(|| "VAE convolution weight shape overflow".to_string())?;
    let weight_len = weight_elements
        .checked_mul(2)
        .ok_or_else(|| "VAE convolution weight byte size overflow".to_string())?;
    if input.len() != input_len || output.len() != output_len || weights.len() != weight_len {
        return Err("Invalid VAE convolution buffer length".into());
    }
    if bias.is_some_and(|values| values.len() != output_channels) {
        return Err("Invalid VAE convolution bias length".into());
    }
    if input.iter().any(|value| !value.is_finite())
        || bias.is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err("Non-finite VAE convolution input".into());
    }

    let patch_len = kernel
        .checked_mul(kernel)
        .and_then(|value| value.checked_mul(input_channels))
        .ok_or_else(|| "VAE convolution patch size overflow".to_string())?;
    if patch.capacity() < patch_len {
        let additional = patch_len
            .checked_sub(patch.len())
            .ok_or_else(|| "Invalid VAE convolution patch length".to_string())?;
        patch
            .try_reserve_exact(additional)
            .map_err(|error| format!("Failed to allocate VAE convolution patch: {error}"))?;
    }
    patch.resize(patch_len, 0);
    let padding = kernel / 2;
    for output_y in 0..side {
        for output_x in 0..side {
            patch.fill(0);
            for input_channel in 0..input_channels {
                let input_plane = &input[input_channel * spatial..(input_channel + 1) * spatial];
                for kernel_y in 0..kernel {
                    for kernel_x in 0..kernel {
                        let Some(input_y) = output_y
                            .checked_add(kernel_y)
                            .and_then(|value| value.checked_sub(padding))
                        else {
                            continue;
                        };
                        let Some(input_x) = output_x
                            .checked_add(kernel_x)
                            .and_then(|value| value.checked_sub(padding))
                        else {
                            continue;
                        };
                        if input_y >= side || input_x >= side {
                            continue;
                        }
                        let patch_index = kernel_x + kernel * (kernel_y + kernel * input_channel);
                        patch[patch_index] = f32_to_f16(input_plane[input_y * side + input_x]);
                    }
                }
            }

            let output_position = output_y * side + output_x;
            for output_channel in 0..output_channels {
                let row_start = output_channel * patch_len * 2;
                let dot = dot_f16_f16_bytes(
                    &patch,
                    &weights[row_start..row_start + patch_len * 2],
                    patch_len,
                );
                output[output_channel * spatial + output_position] =
                    dot + bias.map_or(0.0, |values| values[output_channel]);
            }
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err("Non-finite VAE convolution output".into());
    }
    Ok(())
}

fn padded_conv_f16_into(
    input: &[f32],
    input_channels: usize,
    side: usize,
    weights: &[u8],
    output_channels: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
    patch: &mut Vec<u16>,
) -> Result<(), String> {
    conv_f16_into(
        input,
        input_channels,
        side,
        weights,
        output_channels,
        3,
        bias,
        output,
        patch,
    )
}

fn group_norm_32_into(
    input: &[f32],
    channels: usize,
    side: usize,
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    if channels == 0 || channels % GROUPS != 0 || side == 0 {
        return Err("Invalid VAE GroupNorm shape".into());
    }
    let spatial = checked_spatial(side, "VAE GroupNorm")?;
    let feature_len = checked_feature_len(channels, spatial, "VAE GroupNorm")?;
    if input.len() != feature_len
        || output.len() != feature_len
        || weight.len() != channels
        || bias.len() != channels
    {
        return Err("Invalid VAE GroupNorm buffer length".into());
    }
    if input.iter().any(|value| !value.is_finite())
        || weight.iter().any(|value| !value.is_finite())
        || bias.iter().any(|value| !value.is_finite())
    {
        return Err("Non-finite VAE GroupNorm input".into());
    }
    let channels_per_group = channels / GROUPS;
    let values_per_group = channels_per_group
        .checked_mul(spatial)
        .ok_or_else(|| "VAE GroupNorm group size overflow".to_string())?;
    for group in 0..GROUPS {
        let channel_start = group * channels_per_group;
        let channel_end = channel_start + channels_per_group;
        let mut sum = 0.0f64;
        for channel in channel_start..channel_end {
            let channel_values = &input[channel * spatial..(channel + 1) * spatial];
            for row in channel_values.chunks_exact(side) {
                let mut row_sum = 0.0f64;
                for &value in row {
                    row_sum += f64::from(value);
                }
                sum += row_sum;
            }
        }
        let mean = (sum / values_per_group as f64) as f32;
        let mut sum_squared = 0.0f64;
        for channel in channel_start..channel_end {
            for row in 0..side {
                let row_start = channel * spatial + row * side;
                let mut row_sum_squared = 0.0f64;
                for position in 0..side {
                    let index = row_start + position;
                    let centered = input[index] - mean;
                    output[index] = centered;
                    row_sum_squared += f64::from(centered * centered);
                }
                sum_squared += row_sum_squared;
            }
        }
        let variance = (sum_squared / values_per_group as f64) as f32;
        let inverse_std = (variance + GROUP_NORM_EPSILON).sqrt().recip();
        if !inverse_std.is_finite() {
            return Err("Non-finite VAE GroupNorm statistics".into());
        }
        for channel in channel_start..channel_end {
            for position in 0..spatial {
                let index = channel * spatial + position;
                output[index] *= inverse_std;
            }
        }
        for channel in channel_start..channel_end {
            for position in 0..spatial {
                output[channel * spatial + position] *= weight[channel];
            }
        }
        for channel in channel_start..channel_end {
            for position in 0..spatial {
                let value = &mut output[channel * spatial + position];
                *value += bias[channel];
                if !value.is_finite() {
                    return Err("Non-finite VAE GroupNorm output".into());
                }
            }
        }
    }
    Ok(())
}

fn silu_inplace_checked(values: &mut [f32]) -> Result<(), String> {
    silu_approx_inplace(values);
    if values.iter().any(|value| !value.is_finite()) {
        return Err("Non-finite VAE SiLU output".into());
    }
    Ok(())
}

fn add_shortcut_residual_into(
    input: &[f32],
    residual_branch: &[f32],
    input_channels: usize,
    output_channels: usize,
    side: usize,
    weights: &[u8],
    bias: Option<&[f32]>,
    output: &mut [f32],
    patch: &mut Vec<u16>,
) -> Result<(), String> {
    conv_f16_into(
        input,
        input_channels,
        side,
        weights,
        output_channels,
        1,
        bias,
        output,
        patch,
    )?;
    if residual_branch.len() != output.len() {
        return Err("Invalid VAE shortcut residual length".into());
    }
    for (output, branch) in output.iter_mut().zip(residual_branch) {
        *output += branch;
        if !output.is_finite() {
            return Err("Non-finite VAE shortcut residual".into());
        }
    }
    Ok(())
}

fn one_head_spatial_attention_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    channels: usize,
    spatial: usize,
    output: &mut [f32],
    scores: &mut [f32],
) -> Result<(), String> {
    let feature_len = checked_feature_len(channels, spatial, "VAE attention")?;
    if channels == 0
        || spatial == 0
        || q.len() != feature_len
        || k.len() != feature_len
        || v.len() != feature_len
        || output.len() != feature_len
        || scores.len() != spatial
    {
        return Err("Invalid VAE attention buffer length".into());
    }
    if q.iter().chain(k).chain(v).any(|value| !value.is_finite()) {
        return Err("Non-finite VAE attention input".into());
    }
    output.fill(0.0);
    let scale = 1.0 / (channels as f32).sqrt();
    for query_position in 0..spatial {
        for key_position in 0..spatial {
            let mut score = 0.0f32;
            for channel in 0..channels {
                score +=
                    q[channel * spatial + query_position] * k[channel * spatial + key_position];
            }
            score *= scale;
            if !score.is_finite() {
                return Err("Non-finite VAE attention score".into());
            }
            scores[key_position] = score;
        }
        vae_softmax_inplace(scores);
        if scores.iter().any(|value| !value.is_finite()) {
            return Err("Non-finite VAE attention probability".into());
        }
        for channel in 0..channels {
            let mut value = 0.0f32;
            for key_position in 0..spatial {
                value += scores[key_position] * v[channel * spatial + key_position];
            }
            if !value.is_finite() {
                return Err("Non-finite VAE attention output".into());
            }
            output[channel * spatial + query_position] = value;
        }
    }
    Ok(())
}

#[inline]
fn vae_softmax_inplace(values: &mut [f32]) {
    softmax_inplace(values);
}

fn upsample_nearest_into(
    input: &[f32],
    channels: usize,
    side: usize,
    output: &mut [f32],
) -> Result<(), String> {
    if side == 0 {
        return Err("Invalid VAE upsample shape".into());
    }
    let output_side = side
        .checked_mul(2)
        .ok_or_else(|| "VAE upsample side overflow".to_string())?;
    let input_spatial = checked_spatial(side, "VAE upsample input")?;
    let output_spatial = checked_spatial(output_side, "VAE upsample output")?;
    let input_len = checked_feature_len(channels, input_spatial, "VAE upsample input")?;
    let output_len = checked_feature_len(channels, output_spatial, "VAE upsample output")?;
    if input.len() != input_len || output.len() != output_len {
        return Err("Invalid VAE upsample buffer length".into());
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err("Non-finite VAE upsample input".into());
    }
    for channel in 0..channels {
        for y in 0..output_side {
            for x in 0..output_side {
                output[channel * output_spatial + y * output_side + x] =
                    input[channel * input_spatial + (y / 2) * side + x / 2];
            }
        }
    }
    Ok(())
}

pub(crate) fn upsample_nearest_then_conv(
    input: &[f32],
    channels: usize,
    side: usize,
    weights: &[u8],
    bias: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    let output_side = side
        .checked_mul(2)
        .ok_or_else(|| "VAE upsample shape overflow".to_string())?;
    let output_spatial = checked_spatial(output_side, "VAE upsample")?;
    let output_len = checked_feature_len(channels, output_spatial, "VAE upsample")?;
    let mut nearest = reserve_f32("VAE nearest upsample", output_len)?;
    upsample_nearest_into(input, channels, side, &mut nearest)?;
    let mut output = reserve_f32("VAE learned upsample", output_len)?;
    let mut patch = Vec::new();
    padded_conv_f16_into(
        &nearest,
        channels,
        output_side,
        weights,
        channels,
        bias,
        &mut output,
        &mut patch,
    )?;
    Ok(output)
}

fn rgb_bytes_from_channels(values: &[f32], side: usize) -> Result<Vec<u8>, String> {
    if side == 0 {
        return Err("Invalid VAE RGB shape".into());
    }
    let spatial = checked_spatial(side, "VAE RGB")?;
    let expected = checked_feature_len(3, spatial, "VAE RGB")?;
    if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
        return Err("Invalid VAE RGB channel output".into());
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(expected)
        .map_err(|error| format!("Failed to allocate VAE RGB bytes: {error}"))?;
    for position in 0..spatial {
        bytes.push(to_rgb_byte(values[position]));
        bytes.push(to_rgb_byte(values[spatial + position]));
        bytes.push(to_rgb_byte(values[2 * spatial + position]));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{GGMLType, MetaValue, TensorInfo, TensorSource};
    use half::f16;
    use std::collections::HashMap;
    use std::sync::Arc;

    struct DecoderSource {
        tensors: HashMap<String, TensorInfo>,
        zeroes: Vec<u8>,
    }

    impl TensorSource for DecoderSource {
        fn metadata(&self, _key: &str) -> Option<&MetaValue> {
            None
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.tensors.get(name)
        }

        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            let len = self.tensors.get(name)?.nbytes();
            self.zeroes.get(..len)
        }
    }

    fn decoder_source_without(missing: &str) -> DecoderSource {
        let mut tensors = HashMap::new();
        let mut add = |name: String, dims: &[u64], ggml_type| {
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dims: dims.to_vec(),
                    ggml_type,
                    offset: 0,
                },
            );
        };
        for (name, dims, ggml_type) in [
            ("decoder.conv_in.bias", &[512][..], GGMLType::F32),
            (
                "decoder.conv_in.weight",
                &[3, 3, 16, 512][..],
                GGMLType::F16,
            ),
            ("decoder.conv_out.bias", &[3][..], GGMLType::F32),
            (
                "decoder.conv_out.weight",
                &[3, 3, 128, 3][..],
                GGMLType::F16,
            ),
            ("decoder.norm_out.bias", &[128][..], GGMLType::F32),
            ("decoder.norm_out.weight", &[128][..], GGMLType::F32),
        ] {
            add(name.into(), dims, ggml_type);
        }
        add_attention(&mut add, "decoder.mid.attn_1", 512);
        add_block(&mut add, "decoder.mid.block_1", 512, 512);
        add_block(&mut add, "decoder.mid.block_2", 512, 512);
        for (stage, input_channels, output_channels) in
            [(0, 256, 128), (1, 512, 256), (2, 512, 512), (3, 512, 512)]
        {
            for block in 0..3 {
                add_block(
                    &mut add,
                    &format!("decoder.up.{stage}.block.{block}"),
                    if block == 0 {
                        input_channels
                    } else {
                        output_channels
                    },
                    output_channels,
                );
            }
            if stage != 0 {
                add(
                    format!("decoder.up.{stage}.upsample.conv.weight"),
                    &[3, 3, output_channels, output_channels],
                    GGMLType::F16,
                );
                add(
                    format!("decoder.up.{stage}.upsample.conv.bias"),
                    &[output_channels],
                    GGMLType::F32,
                );
            }
        }
        tensors.remove(missing);
        DecoderSource {
            tensors,
            zeroes: vec![0; 3 * 3 * 512 * 512 * 2],
        }
    }

    fn add_attention(add: &mut impl FnMut(String, &[u64], GGMLType), prefix: &str, channels: u64) {
        for projection in ["k", "proj_out", "q", "v"] {
            add(
                format!("{prefix}.{projection}.weight"),
                &[1, 1, channels, channels],
                GGMLType::F16,
            );
            add(
                format!("{prefix}.{projection}.bias"),
                &[channels],
                GGMLType::F32,
            );
        }
        for affine in ["weight", "bias"] {
            add(
                format!("{prefix}.norm.{affine}"),
                &[channels],
                GGMLType::F32,
            );
        }
    }

    fn add_block(
        add: &mut impl FnMut(String, &[u64], GGMLType),
        prefix: &str,
        input_channels: u64,
        output_channels: u64,
    ) {
        for (conv, input) in [("conv1", input_channels), ("conv2", output_channels)] {
            add(
                format!("{prefix}.{conv}.weight"),
                &[3, 3, input, output_channels],
                GGMLType::F16,
            );
            add(
                format!("{prefix}.{conv}.bias"),
                &[output_channels],
                GGMLType::F32,
            );
        }
        for (norm, channels) in [("norm1", input_channels), ("norm2", output_channels)] {
            for affine in ["weight", "bias"] {
                add(
                    format!("{prefix}.{norm}.{affine}"),
                    &[channels],
                    GGMLType::F32,
                );
            }
        }
        if input_channels != output_channels {
            add(
                format!("{prefix}.nin_shortcut.weight"),
                &[1, 1, input_channels, output_channels],
                GGMLType::F16,
            );
            add(
                format!("{prefix}.nin_shortcut.bias"),
                &[output_channels],
                GGMLType::F32,
            );
        }
    }

    fn identity_center_kernel() -> Vec<u8> {
        (0..9)
            .flat_map(|index| {
                f16::from_f32(if index == 4 { 1.0 } else { 0.0 })
                    .to_bits()
                    .to_le_bytes()
            })
            .collect()
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    #[test]
    fn conv_f16_reuses_caller_owned_patch_buffer() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weights = f16_bytes(&[1.0]);
        let mut output = [0.0; 4];
        let mut patch = Vec::new();

        conv_f16_into(&input, 1, 2, &weights, 1, 1, None, &mut output, &mut patch).unwrap();
        let patch_ptr = patch.as_ptr();

        conv_f16_into(&input, 1, 2, &weights, 1, 1, None, &mut output, &mut patch).unwrap();

        assert_eq!(patch.len(), 1);
        assert_eq!(patch.as_ptr(), patch_ptr);
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[test]
    fn conv_f16_matches_pinned_ggml_im2col_dot_and_post_bias() {
        let input = [
            0x403b_7bad,
            0x4008_55cd,
            0x3f93_53de,
            0x4000_6fa8,
            0xc027_ed45,
            0xbdfa_667d,
            0x3f2a_0050,
            0xbfdf_e6bd,
            0xbee7_1555,
            0xbf23_6130,
            0xc020_627c,
            0xc03c_40ba,
            0xbe16_4c3a,
            0xbf17_c63e,
            0x3f74_e6b4,
            0x3f86_eba9,
            0xbf66_5ae1,
            0x3f85_74b8,
            0xbe97_b12d,
            0xbf7b_7473,
            0xbf1d_af46,
            0x3d89_389d,
            0xbe50_4cbc,
            0xbf5b_d04c,
            0xbf98_0dc3,
            0xc01c_508c,
            0xc028_7b93,
            0xc03b_c8b2,
            0x3fe2_02e8,
            0x3e94_b97e,
            0x3d41_656c,
            0x3ff6_37e2,
            0xbfb8_2bf1,
            0xc01b_6836,
            0xc07a_29d8,
            0xbebe_92b2,
            0xbf44_353d,
            0x3f30_f267,
            0xbf5f_a659,
            0x3fd2_6b29,
            0x3d06_ae4c,
            0x3f52_a60a,
            0x3f6d_0935,
            0x3d7c_956e,
            0x3f38_02f2,
            0xbfb7_4b77,
            0xc01a_389b,
            0xbf6c_3a61,
            0x3f91_d9e8,
            0xbf69_a40b,
            0xc01d_d891,
            0xbf99_8256,
            0x3fd0_f493,
            0x4019_0727,
            0x4027_bc25,
            0x4078_b217,
            0xbfba_b1a2,
            0xbfbc_fafe,
            0x3f3b_8f4f,
            0x3d21_48c6,
            0x3f31_73ef,
            0x3f27_3161,
            0xbfcc_849e,
            0xbfb3_6697,
        ]
        .map(f32::from_bits);
        let weights = [
            0xa038u16, 0xac61, 0xa446, 0x9b8d, 0xb039, 0xa2ff, 0xa226, 0xa5e2, 0x9dc3, 0x2694,
            0xae51, 0xa7fd, 0xa3ab, 0xa45e, 0x2865, 0x9ea0, 0xa563, 0x231e, 0xa807, 0xa51b, 0xa9df,
            0xabcc, 0x2a8e, 0xa3d8, 0x2a59, 0xa9e6, 0x1572, 0x2a6e, 0x2ba8, 0x2997, 0x299e, 0x3256,
            0x2c2f, 0x20a6, 0x9320, 0x27ed, 0x9c45, 0xad9d, 0x2351, 0xac46, 0x2aae, 0x1d00, 0x22f5,
            0x9ef1, 0xaa74, 0xaa1e, 0x2c44, 0xa749, 0xa45c, 0x2c25, 0xa493, 0x26da, 0xa6cf, 0xa561,
            0x24f3, 0xa8e8, 0x2bbf, 0x275c, 0x2d08, 0xac48, 0xa1fb, 0x2654, 0xa763, 0x27e7, 0x9893,
            0xabf6, 0x281d, 0x2847, 0xa491, 0xaceb, 0x9a96, 0xa983, 0x28c3, 0xa6f4, 0x2c85, 0x2c1a,
            0xab51, 0xa7c8, 0xa80a, 0xa008, 0xa3cc, 0x0d4e, 0x282b, 0xa6e0, 0x280e, 0x2c15, 0xa5af,
            0xab10, 0x2c1b, 0x28bb, 0xa840, 0x2c1b, 0x2b09, 0x2a17, 0xad3f, 0xa9b4, 0x28c1, 0x2ca1,
            0x2a1d, 0x2425, 0xa894, 0xab0f, 0x2b2b, 0xb002, 0x9b4f, 0xab1f, 0xa01a, 0x23cc, 0x22a1,
            0xa793, 0x2082, 0xaa1a, 0xae56, 0xa9f7, 0xa641, 0x2613, 0xa3f1, 0x23ab, 0xa8d3, 0x28bd,
            0xaa78, 0x2f62, 0x97a1, 0x9d67, 0xa9d2, 0xa65b, 0xaac7, 0x2881, 0x2674, 0x2cdb, 0xadc0,
            0x290c, 0x21dc, 0x2275, 0x26bc, 0x2364, 0x241a, 0xab79, 0xa8b1, 0xb030, 0x2bb3, 0xa85b,
            0xa53c, 0xa5e2,
        ]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
        let mut output = [0.0; 4];
        let mut patch = Vec::new();

        conv_f16_into(
            &input,
            16,
            2,
            &weights,
            1,
            3,
            Some(&[f32::from_bits(0xbd69_17db)]),
            &mut output,
            &mut patch,
        )
        .unwrap();

        assert_eq!(output[0].to_bits(), 0xbeb5_c582);
        assert_eq!(output[1].to_bits(), 0x3e7d_324b);
    }

    #[test]
    fn diffusion_latent_uses_flux_scale_and_shift() {
        assert_eq!(diffusion_to_vae(0.3611), 1.1159);
    }

    #[test]
    fn default_vae_softmax_uses_the_oracle_f64_sum_order() {
        let mut values = [0x40ff_22d2, 0xc075_0e57, 0x4098_49bb].map(f32::from_bits);

        super::vae_softmax_inplace(&mut values);

        assert_eq!(
            values.map(f32::to_bits),
            [0x3f76_1b16, 0x36f1_9850, 0x3d1e_470c]
        );
    }

    #[test]
    fn learned_upsample_is_nearest_then_padded_conv() {
        let output = upsample_nearest_then_conv(
            &[1.0, 2.0, 3.0, 4.0],
            1,
            2,
            &identity_center_kernel(),
            None,
        )
        .unwrap();
        assert_eq!(
            output,
            vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,]
        );
    }

    #[test]
    fn missing_mid_attention_is_a_load_error() {
        assert!(FluxVae::load(Arc::new(decoder_source_without(
            "decoder.mid.attn_1.q.weight"
        )))
        .is_err());
    }

    #[test]
    fn group_norm_uses_each_groups_channels_and_spatial_values() {
        let input = (0..32)
            .flat_map(|channel| {
                let offset = channel as f32 * 10.0;
                [offset + 1.0, offset + 2.0, offset + 3.0, offset + 4.0]
            })
            .collect::<Vec<_>>();
        let mut output = vec![0.0; input.len()];
        group_norm_32_into(&input, 32, 2, &[1.0; 32], &[0.0; 32], &mut output).unwrap();
        let expected = [-1.3416402, -0.4472134, 0.4472134, 1.3416402];
        for channel in 0..32 {
            for spatial in 0..4 {
                assert!((output[channel * 4 + spatial] - expected[spatial]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn group_norm_combines_two_channels_in_each_group() {
        let mut input = [0.0; 64 * 4];
        for group in 0..32 {
            let first = group * 2 * 4;
            let second = first + 4;
            input[first..first + 4].copy_from_slice(&[1.0, 1.0, 3.0, 3.0]);
            input[second..second + 4].copy_from_slice(&[5.0, 5.0, 7.0, 7.0]);
        }
        let mut output = [0.0; 64 * 4];
        group_norm_32_into(&input, 64, 2, &[1.0; 64], &[0.0; 64], &mut output).unwrap();

        let expected_first = [-1.3416407, -1.3416407, -0.44721356, -0.44721356];
        let expected_second = [0.44721356, 0.44721356, 1.3416407, 1.3416407];
        for group in 0..32 {
            let first = group * 2 * 4;
            let second = first + 4;
            for position in 0..4 {
                assert!((output[first + position] - expected_first[position]).abs() < 1e-6);
                assert!((output[second + position] - expected_second[position]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn group_norm_matches_pinned_ggml_double_statistics_and_staged_affine() {
        let group = [
            0xbeb5_c582,
            0x3e9b_14b4,
            0xbe1e_2320,
            0xbebc_655b,
            0xbf48_edd1,
            0x3d85_35da,
            0x3ebf_b47c,
            0xbe9e_b9e5,
        ]
        .map(f32::from_bits);
        let input = (0..32).flat_map(|_| group).collect::<Vec<_>>();
        let mut output = vec![0.0; input.len()];

        group_norm_32_into(&input, 64, 2, &[1.0; 64], &[0.0; 64], &mut output).unwrap();

        let actual = output[..8]
            .iter()
            .copied()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            [
                0xbf0e_9a5c,
                0x3fa1_c256,
                0xbaf9_7dfc,
                0xbf17_c4ff,
                0xbfdf_9307,
                0x3f1b_01c6,
                0x3fbb_193a,
                0xbedd_6d7b,
            ]
        );
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[test]
    fn silu_matches_pinned_ggml_neon_vector_path() {
        let mut values = [
            0xbf0e_9a5c,
            0x3fa1_c256,
            0xbaf9_7dfc,
            0xbf17_c4ff,
            0xbfdf_9307,
            0x3f1b_01c6,
            0x3fbb_193a,
            0xbedd_6d7b,
        ]
        .map(f32::from_bits);

        silu_inplace_checked(&mut values).unwrap();

        assert_eq!(
            values.map(f32::to_bits),
            [
                0xbe4f_c323,
                0x3f7c_3cc7,
                0xba79_4131,
                0xbe58_1bc2,
                0xbe84_c615,
                0x3ec8_8d48,
                0x3f97_e2aa,
                0xbe2e_4779,
            ]
        );
    }

    #[test]
    fn group_norm_adds_epsilon_to_variance_before_square_root() {
        let input = (0..32)
            .flat_map(|_| [0.0, 0.0, 0.002, 0.002])
            .collect::<Vec<_>>();
        let mut output = vec![0.0; input.len()];
        group_norm_32_into(&input, 32, 2, &[1.0; 32], &[0.0; 32], &mut output).unwrap();
        for channel in 0..32 {
            assert!((output[channel * 4] + 0.7071068).abs() < 1e-6);
            assert!((output[channel * 4 + 1] + 0.7071068).abs() < 1e-6);
            assert!((output[channel * 4 + 2] - 0.7071068).abs() < 1e-6);
            assert!((output[channel * 4 + 3] - 0.7071068).abs() < 1e-6);
        }
    }

    #[test]
    fn residual_shortcut_projects_input_before_adding_branch() {
        let mut output = [0.0];
        let mut patch = Vec::new();
        add_shortcut_residual_into(
            &[1.0, 2.0],
            &[7.0],
            2,
            1,
            1,
            &f16_bytes(&[2.0, 3.0]),
            Some(&[5.0]),
            &mut output,
            &mut patch,
        )
        .unwrap();
        assert_eq!(output, [20.0]);
    }

    #[test]
    fn mid_attention_softmax_is_per_query_and_spatially_non_uniform() {
        let q = [2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let k = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = [3.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut output = [0.0; 8];
        let mut scores = [0.0; 2];
        one_head_spatial_attention_into(&q, &k, &v, 4, 2, &mut output, &mut scores).unwrap();
        assert!((output[0] - 4.0757656).abs() < 1e-6);
        assert!((output[1] - 3.4768117).abs() < 1e-6);
        assert_eq!(&output[2..], &[0.0; 6]);
    }

    #[test]
    fn rgb_bytes_round_clamp_and_interleave_channel_major_output() {
        let bytes = rgb_bytes_from_channels(
            &[
                -1.0, -0.5, 0.0, 0.5, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, -0.5, -1.0,
            ],
            2,
        )
        .unwrap();
        assert_eq!(
            bytes,
            vec![0, 255, 128, 64, 128, 255, 128, 0, 64, 191, 128, 0]
        );
    }

    #[test]
    fn decode_rgb_rejects_wrong_or_zero_latent_shape() {
        let vae = FluxVae::load(Arc::new(decoder_source_without(""))).unwrap();
        assert!(vae.decode_rgb(&[0.0; 15], 1).is_err());
        assert!(vae.decode_rgb(&[], 0).is_err());
    }

    #[test]
    fn decoder_stages_load_in_oracle_order_with_three_blocks_each() {
        let vae = FluxVae::load(Arc::new(decoder_source_without(""))).unwrap();
        assert_eq!(
            vae.stages
                .iter()
                .map(|stage| stage.index)
                .collect::<Vec<_>>(),
            vec![3, 2, 1, 0]
        );
        assert_eq!(
            vae.stages
                .iter()
                .map(|stage| stage.blocks.len())
                .collect::<Vec<_>>(),
            vec![3, 3, 3, 3]
        );
        assert_eq!(
            vae.stages
                .iter()
                .map(|stage| stage.upsample.is_some())
                .collect::<Vec<_>>(),
            vec![true, true, true, false]
        );
    }

    #[test]
    fn convolution_reads_gguf_kw_kh_input_output_layout() {
        let mut weights = vec![0.0; 3 * 3 * 2 * 2];
        for (output_channel, values) in [[1.0, 2.0], [3.0, 4.0]].into_iter().enumerate() {
            for (input_channel, value) in values.into_iter().enumerate() {
                weights[4 + 9 * (input_channel + 2 * output_channel)] = value;
            }
        }
        let mut output = [0.0; 2];
        let mut patch = Vec::new();
        padded_conv_f16_into(
            &[5.0, 6.0],
            2,
            1,
            &f16_bytes(&weights),
            2,
            None,
            &mut output,
            &mut patch,
        )
        .unwrap();
        assert_eq!(output, [17.0, 39.0]);
    }

    #[test]
    fn convolution_uses_zero_padding_at_image_edges() {
        let mut output = [0.0];
        let mut patch = Vec::new();
        padded_conv_f16_into(
            &[2.0],
            1,
            1,
            &f16_bytes(&[1.0; 9]),
            1,
            None,
            &mut output,
            &mut patch,
        )
        .unwrap();
        assert_eq!(output, [2.0]);
    }

    #[test]
    fn non_finite_convolution_weight_is_fatal() {
        let mut weights = vec![0.0; 9];
        weights[4] = f32::INFINITY;
        let mut output = [0.0];
        let mut patch = Vec::new();
        assert!(padded_conv_f16_into(
            &[1.0],
            1,
            1,
            &f16_bytes(&weights),
            1,
            None,
            &mut output,
            &mut patch,
        )
        .is_err());
    }

    #[test]
    #[ignore = "requires Z_IMAGE_VAE"]
    fn flux_vae_loader_accepts_complete_supplied_decoder() {
        let source = crate::core::loader::GGUFLoader::from_file(
            std::env::var("Z_IMAGE_VAE").expect("missing Z_IMAGE_VAE"),
        )
        .unwrap();
        FluxVae::load(Arc::new(source)).unwrap();
    }
}
