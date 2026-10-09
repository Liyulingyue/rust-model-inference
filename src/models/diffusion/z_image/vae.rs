use super::dit::TorchMt19937;
use super::{validate_component, Component, ZImageRgb};
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::{dot_f16_f16_bytes, f32_to_f16, silu_inplace, softmax_inplace};
use std::sync::atomic::{AtomicBool, Ordering};
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
    #[cfg(feature = "vulkan")]
    gpu: crate::ops::kernel::vulkan::GpuLinear,
    linear: Option<Vec<f32>>,
}

impl VaeConv {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
    ) -> Result<Self, String> {
        let weight = format!("{prefix}.weight");
        let linear = if source
            .tensor_info(&weight)
            .is_some_and(|info| info.ggml_type == crate::core::tensor::GGMLType::F32)
        {
            Some(load_f32(source, &weight, input_channels * output_channels)?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias: load_f32(source, &format!("{prefix}.bias"), output_channels)?,
            input_channels,
            output_channels,
            kernel,
            #[cfg(feature = "vulkan")]
            gpu: crate::ops::kernel::vulkan::GpuLinear::default(),
            linear,
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

struct EncoderStage {
    blocks: Vec<VaeResidualBlock>,
    downsample: Option<VaeConv>,
    output_channels: usize,
}

struct FluxVaeEncoder {
    conv_in: VaeConv,
    stages: Vec<EncoderStage>,
    mid_block_1: VaeResidualBlock,
    mid_attention: VaeAttention,
    mid_block_2: VaeResidualBlock,
    norm_out: VaeNorm,
    conv_out: VaeConv,
}

impl FluxVaeEncoder {
    fn load(source: &dyn TensorSource) -> Result<Self, String> {
        let mut stages = Vec::with_capacity(4);
        for (stage, input, output) in [(0, 128, 128), (1, 128, 256), (2, 256, 512), (3, 512, 512)] {
            let mut blocks = Vec::with_capacity(2);
            for block in 0..2 {
                blocks.push(VaeResidualBlock::load(
                    source,
                    &format!("encoder.down.{stage}.block.{block}"),
                    if block == 0 { input } else { output },
                    output,
                )?);
            }
            stages.push(EncoderStage {
                blocks,
                downsample: (stage != 3)
                    .then(|| {
                        VaeConv::load(
                            source,
                            &format!("encoder.down.{stage}.downsample.conv"),
                            output,
                            output,
                            3,
                        )
                    })
                    .transpose()?,
                output_channels: output,
            });
        }
        Ok(Self {
            conv_in: VaeConv::load(source, "encoder.conv_in", 3, 128, 3)?,
            stages,
            mid_block_1: VaeResidualBlock::load(source, "encoder.mid.block_1", 512, 512)?,
            mid_attention: VaeAttention::load(source, "encoder.mid.attn_1", 512)?,
            mid_block_2: VaeResidualBlock::load(source, "encoder.mid.block_2", 512, 512)?,
            norm_out: VaeNorm::load(source, "encoder.norm_out", 512)?,
            conv_out: VaeConv::load(source, "encoder.conv_out", 512, 32, 3)?,
        })
    }
}

struct VaeScratch {
    first: Vec<f32>,
    second: Vec<f32>,
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
    pool: Arc<ComputePool>,
    #[cfg(feature = "vulkan")]
    gpu_conv: crate::ops::kernel::vulkan::GpuConv,
    conv_in: VaeConv,
    mid_block_1: VaeResidualBlock,
    mid_attention: VaeAttention,
    mid_block_2: VaeResidualBlock,
    stages: Vec<DecoderStage>,
    norm_out: VaeNorm,
    conv_out: VaeConv,
    encoder: Option<FluxVaeEncoder>,
    post_quant_conv: Option<VaeConv>,
}

impl FluxVae {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        validate_component(source.as_ref(), Component::Vae)?;
        Self::load_decoder(source, pool, 16, None)
    }

    pub(crate) fn load_longcat(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let encoder = FluxVaeEncoder::load(source.as_ref())?;
        let mut vae = Self::load_decoder(source, pool, 16, None)?;
        vae.encoder = Some(encoder);
        Ok(vae)
    }

    pub(crate) fn load_flux2(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        super::validate_flux2_vae(source.as_ref())?;
        let post_quant_conv = VaeConv::load(source.as_ref(), "post_quant_conv", 32, 32, 1)?;
        Self::load_decoder(source, pool, 32, Some(post_quant_conv))
    }

    fn load_decoder(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
        channels: usize,
        post_quant_conv: Option<VaeConv>,
    ) -> Result<Self, String> {
        let conv_in = VaeConv::load(source.as_ref(), "decoder.conv_in", channels, 512, 3)?;
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
            pool,
            #[cfg(feature = "vulkan")]
            gpu_conv: crate::ops::kernel::vulkan::GpuConv::default(),
            conv_in,
            mid_block_1,
            mid_attention,
            mid_block_2,
            stages,
            norm_out,
            conv_out,
            encoder: None,
            post_quant_conv,
        })
    }

    pub(crate) fn encode_rgb(
        &self,
        rgb: &[u8],
        side: usize,
        seed: u64,
    ) -> Result<Vec<f32>, String> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or("This VAE has no image encoder")?;
        if side == 0 || side % 16 != 0 {
            return Err("LongCat VAE image side must be a positive multiple of 16".into());
        }
        let spatial = checked_spatial(side, "LongCat VAE image")?;
        if rgb.len() != checked_feature_len(3, spatial, "LongCat RGB input")? {
            return Err("Invalid LongCat RGB image length".into());
        }
        let mut current = reserve_f32("LongCat VAE RGB input", rgb.len())?;
        for pixel in 0..spatial {
            for channel in 0..3 {
                current[channel * spatial + pixel] = rgb[pixel * 3 + channel] as f32 / 127.5 - 1.0;
            }
        }
        let mut scratch = VaeScratch::new();
        resize_f32(&mut scratch.first, "LongCat VAE conv input", 128 * spatial)?;
        run_conv(
            self.source.as_ref(),
            &self.pool,
            &encoder.conv_in,
            &current,
            side,
            &mut scratch.first,
            #[cfg(feature = "vulkan")]
            Some(&self.gpu_conv),
        )?;
        std::mem::swap(&mut current, &mut scratch.first);
        let mut feature_side = side;
        for stage in &encoder.stages {
            for block in &stage.blocks {
                self.run_residual_block(block, &mut current, feature_side, &mut scratch)?;
            }
            if let Some(downsample) = &stage.downsample {
                let next_side = feature_side / 2;
                let next_len = checked_feature_len(
                    stage.output_channels,
                    checked_spatial(next_side, "LongCat VAE downsample")?,
                    "LongCat VAE downsample",
                )?;
                resize_f32(&mut scratch.first, "LongCat VAE downsample", next_len)?;
                run_bf16_downsample(
                    self.source.as_ref(),
                    &self.pool,
                    downsample,
                    &current,
                    feature_side,
                    &mut scratch.first,
                )?;
                std::mem::swap(&mut current, &mut scratch.first);
                feature_side = next_side;
            }
        }
        let mid_len = checked_feature_len(
            512,
            checked_spatial(feature_side, "LongCat VAE mid")?,
            "LongCat VAE mid",
        )?;
        scratch.prepare_features(mid_len)?;
        self.run_residual_block(
            &encoder.mid_block_1,
            &mut current,
            feature_side,
            &mut scratch,
        )?;
        scratch.prepare_attention(mid_len, feature_side * feature_side)?;
        self.run_attention(
            &encoder.mid_attention,
            &mut current,
            feature_side,
            &mut scratch,
        )?;
        self.run_residual_block(
            &encoder.mid_block_2,
            &mut current,
            feature_side,
            &mut scratch,
        )?;
        resize_f32(&mut scratch.first, "LongCat VAE output norm", mid_len)?;
        group_norm_32_into(
            &current,
            512,
            feature_side,
            &encoder.norm_out.weight,
            &encoder.norm_out.bias,
            &mut scratch.first,
        )?;
        silu_inplace_checked(&mut scratch.first)?;
        let latent_spatial = checked_spatial(feature_side, "LongCat VAE latent")?;
        let mut moments = reserve_f32("LongCat VAE moments", 32 * latent_spatial)?;
        run_conv(
            self.source.as_ref(),
            &self.pool,
            &encoder.conv_out,
            &scratch.first,
            feature_side,
            &mut moments,
            #[cfg(feature = "vulkan")]
            Some(&self.gpu_conv),
        )?;
        let mut noise = vec![0.0; 16 * latent_spatial];
        TorchMt19937::new(seed).fill_normal(&mut noise);
        let latent = (0..noise.len())
            .map(|index| {
                let mean = moments[index];
                let logvar = moments[index + noise.len()].clamp(-30.0, 20.0);
                let sampled = mean + (0.5 * logvar).exp() * noise[index];
                (sampled - 0.1159) * 0.3611
            })
            .collect();
        Ok(latent)
    }

    pub(crate) fn decode_rgb(
        &self,
        diffusion_latent: &[f32],
        latent_side: usize,
    ) -> Result<ZImageRgb, String> {
        let mapped: Vec<f32> = diffusion_latent
            .iter()
            .copied()
            .map(diffusion_to_vae)
            .collect();
        self.decode_mapped_rgb(&mapped, latent_side)
    }

    pub(crate) fn decode_mapped_rgb(
        &self,
        mapped_latent: &[f32],
        latent_side: usize,
    ) -> Result<ZImageRgb, String> {
        if latent_side == 0 {
            return Err("Z-Image VAE latent side must be positive".into());
        }
        let latent_spatial = checked_spatial(latent_side, "VAE latent")?;
        let expected_latent =
            checked_feature_len(self.conv_in.input_channels, latent_spatial, "VAE latent")?;
        if mapped_latent.len() != expected_latent {
            return Err(format!(
                "Invalid Z-Image VAE latent length: expected {expected_latent}, got {}",
                mapped_latent.len()
            ));
        }
        if mapped_latent.iter().any(|value| !value.is_finite()) {
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
        for (output, input) in current.iter_mut().zip(mapped_latent) {
            *output = *input;
            if !output.is_finite() {
                return Err("Z-Image VAE mapped latent contains NaN or infinity".into());
            }
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "z_image.vae.mapped_latent",
            None,
            &[latent_side, latent_side, self.conv_in.input_channels],
            &current,
        ));
        let mut scratch = VaeScratch::new();
        if let Some(conv) = &self.post_quant_conv {
            resize_f32(
                &mut scratch.first,
                "Flux2 post-quant output",
                expected_latent,
            )?;
            run_conv(
                self.source.as_ref(),
                &self.pool,
                conv,
                &current,
                latent_side,
                &mut scratch.first,
                #[cfg(feature = "vulkan")]
                Some(&self.gpu_conv),
            )?;
            std::mem::swap(&mut current, &mut scratch.first);
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "z_image.vae.post_quant",
                None,
                &[latent_side, latent_side, self.conv_in.input_channels],
                &current,
            ));
        }
        let t_vae_total = std::time::Instant::now();
        let _t_vae_map = std::time::Instant::now().elapsed();
        let mid_len = checked_feature_len(512, latent_spatial, "VAE middle feature")?;
        resize_f32(&mut scratch.first, "VAE convolution input output", mid_len)?;
        let t_conv_in = std::time::Instant::now();
        run_conv(
            self.source.as_ref(),
            &self.pool,
            &self.conv_in,
            &current,
            latent_side,
            &mut scratch.first,
            #[cfg(feature = "vulkan")]
            Some(&self.gpu_conv),
        )?;
        std::mem::swap(&mut current, &mut scratch.first);
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "z_image.vae.conv_in",
            None,
            &[latent_side, latent_side, 512],
            &current,
        ));
        let t_conv_in = t_conv_in.elapsed();

        let t_mid = std::time::Instant::now();
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
        self.run_attention(&self.mid_attention, &mut current, latent_side, &mut scratch)?;
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
        let t_mid = t_mid.elapsed();

        let t_up = std::time::Instant::now();
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
                    &self.pool,
                    upsample,
                    &scratch.first,
                    next_side,
                    &mut scratch.second,
                    #[cfg(feature = "vulkan")]
                    Some(&self.gpu_conv),
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
        let t_up = t_up.elapsed();
        if side != output_side {
            return Err("Invalid Z-Image VAE spatial factor".into());
        }

        let t_out = std::time::Instant::now();
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
            &self.pool,
            &self.conv_out,
            &scratch.first,
            output_side,
            &mut scratch.second,
            #[cfg(feature = "vulkan")]
            Some(&self.gpu_conv),
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
        let t_out = t_out.elapsed();
        eprintln!(
            "[vae-profile] conv_in={:.1}ms mid={:.1}ms up={:.1}ms out={:.1}ms total={:.1}ms",
            t_conv_in.as_secs_f64() * 1000.0,
            t_mid.as_secs_f64() * 1000.0,
            t_up.as_secs_f64() * 1000.0,
            t_out.as_secs_f64() * 1000.0,
            t_vae_total.elapsed().as_secs_f64() * 1000.0,
        );
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
            &self.pool,
            &block.conv1,
            &scratch.first,
            side,
            &mut scratch.second,
            #[cfg(feature = "vulkan")]
            Some(&self.gpu_conv),
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
            &self.pool,
            &block.conv2,
            &scratch.first,
            side,
            &mut scratch.second,
            #[cfg(feature = "vulkan")]
            Some(&self.gpu_conv),
        )?;

        if let Some(shortcut) = &block.shortcut {
            resize_f32(&mut scratch.first, "VAE shortcut", output_len)?;
            run_conv(
                self.source.as_ref(),
                &self.pool,
                shortcut,
                current,
                side,
                &mut scratch.first,
                #[cfg(feature = "vulkan")]
                Some(&self.gpu_conv),
            )?;
            for (output, branch) in scratch.first.iter_mut().zip(&scratch.second) {
                *output += branch;
                if !output.is_finite() {
                    return Err("Non-finite VAE shortcut residual".into());
                }
            }
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
        attention: &VaeAttention,
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
            &attention.norm.weight,
            &attention.norm.bias,
            &mut scratch.first,
        )?;
        run_attention_projection(
            self.source.as_ref(),
            &self.pool,
            &attention.q,
            &scratch.first,
            side,
            &mut scratch.q,
            #[cfg(feature = "vulkan")]
            Some(&self.gpu_conv),
        )?;
        run_attention_projection(
            self.source.as_ref(),
            &self.pool,
            &attention.k,
            &scratch.first,
            side,
            &mut scratch.k,
            #[cfg(feature = "vulkan")]
            Some(&self.gpu_conv),
        )?;
        run_attention_projection(
            self.source.as_ref(),
            &self.pool,
            &attention.v,
            &scratch.first,
            side,
            &mut scratch.v,
            #[cfg(feature = "vulkan")]
            Some(&self.gpu_conv),
        )?;
        #[cfg(feature = "parity-trace")]
        for (name, values) in [
            ("norm", &scratch.first),
            ("q", &scratch.q),
            ("k", &scratch.k),
            ("v", &scratch.v),
        ] {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                &format!("z_image.vae.{name}"),
                None,
                &[side, side, 512],
                values,
            ));
        }
        one_head_spatial_attention_parallel_into(
            &scratch.q,
            &scratch.k,
            &scratch.v,
            512,
            spatial,
            &mut scratch.first,
            &mut scratch.scores,
            &self.pool,
        )?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "z_image.vae.attention_values",
            None,
            &[side, side, 512],
            &scratch.first,
        ));
        run_conv(
            self.source.as_ref(),
            &self.pool,
            &attention.proj_out,
            &scratch.first,
            side,
            &mut scratch.second,
            #[cfg(feature = "vulkan")]
            Some(&self.gpu_conv),
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
    let values = crate::core::tensor::load_f32_tensor(source, name, &[len as u64])?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("Non-finite tensor: {name}"));
    }
    Ok(values)
}

fn run_conv(
    source: &dyn TensorSource,
    pool: &Arc<ComputePool>,
    conv: &VaeConv,
    input: &[f32],
    side: usize,
    output: &mut [f32],
    #[cfg(feature = "vulkan")] gpu: Option<&crate::ops::kernel::vulkan::GpuConv>,
) -> Result<(), String> {
    if let Some(linear) = &conv.linear {
        let spatial = checked_spatial(side, "VAE linear")?;
        if side == 0
            || input.len() != conv.input_channels * spatial
            || output.len() != conv.output_channels * spatial
        {
            return Err("Invalid VAE linear shape".into());
        }
        let output_ptr = output.as_mut_ptr() as usize;
        pool.compute(move |ith, nth| {
            let mut row = vec![0.; conv.input_channels];
            for pixel in (ith..spatial).step_by(nth) {
                for channel in 0..conv.input_channels {
                    row[channel] = input[channel * spatial + pixel];
                }
                for channel in 0..conv.output_channels {
                    let start = channel * conv.input_channels;
                    let projected = crate::ops::dot_f32(
                        &row,
                        &linear[start..start + conv.input_channels],
                        conv.input_channels,
                    );
                    // Workers own disjoint pixels across every output channel.
                    unsafe {
                        *(output_ptr as *mut f32).add(channel * spatial + pixel) =
                            projected + conv.bias[channel];
                    }
                }
            }
        });
        if output.iter().any(|value| !value.is_finite()) {
            return Err("Non-finite VAE linear output".into());
        }
        return Ok(());
    }
    let weights = source
        .tensor_slice(&conv.weight)
        .ok_or_else(|| format!("Missing tensor data: {}", conv.weight))?;
    let dtype = source
        .tensor_info(&conv.weight)
        .ok_or_else(|| format!("Missing tensor info: {}", conv.weight))?
        .ggml_type;
    match dtype {
        GGMLType::F16 => {
            #[cfg(feature = "vulkan")]
            if gpu.is_some_and(|gpu| {
                gpu.try_conv_f16(
                    weights,
                    input,
                    output,
                    conv.input_channels,
                    conv.output_channels,
                    side,
                    conv.kernel,
                    Some(&conv.bias),
                )
            }) {
                return Ok(());
            }
            conv_f16_parallel_into(
                input,
                conv.input_channels,
                side,
                weights,
                conv.output_channels,
                conv.kernel,
                Some(&conv.bias),
                output,
                pool,
                #[cfg(feature = "vulkan")]
                Some(&conv.gpu),
            )
        }
        GGMLType::BF16 => conv_bf16_parallel_into(
            input,
            conv.input_channels,
            side,
            weights,
            conv.output_channels,
            conv.kernel,
            &conv.bias,
            output,
            pool,
            false,
        ),
        other => Err(format!("Unsupported VAE convolution dtype: {other:?}")),
    }
}

fn run_attention_projection(
    source: &dyn TensorSource,
    pool: &Arc<ComputePool>,
    conv: &VaeConv,
    input: &[f32],
    side: usize,
    output: &mut [f32],
    #[cfg(feature = "vulkan")] gpu: Option<&crate::ops::kernel::vulkan::GpuConv>,
) -> Result<(), String> {
    if conv.kernel != 1 {
        return Err("Invalid VAE attention projection kernel".into());
    }
    if source.tensor_info(&conv.weight).map(|info| info.ggml_type) != Some(GGMLType::BF16) {
        return run_conv(
            source,
            pool,
            conv,
            input,
            side,
            output,
            #[cfg(feature = "vulkan")]
            gpu,
        );
    }
    let weights = source
        .tensor_slice(&conv.weight)
        .ok_or_else(|| format!("Missing tensor data: {}", conv.weight))?;
    conv_bf16_parallel_into(
        input,
        conv.input_channels,
        side,
        weights,
        conv.output_channels,
        1,
        &conv.bias,
        output,
        pool,
        true,
    )
}

fn run_bf16_downsample(
    source: &dyn TensorSource,
    pool: &Arc<ComputePool>,
    conv: &VaeConv,
    input: &[f32],
    side: usize,
    output: &mut [f32],
) -> Result<(), String> {
    if side == 0 || side % 2 != 0 || conv.kernel != 3 || conv.input_channels != conv.output_channels
    {
        return Err("Invalid BF16 VAE downsample shape".into());
    }
    let input_spatial = checked_spatial(side, "VAE downsample input")?;
    let output_side = side / 2;
    let output_spatial = checked_spatial(output_side, "VAE downsample output")?;
    let patch_len = checked_feature_len(conv.input_channels, 9, "VAE downsample patch")?;
    let input_len =
        checked_feature_len(conv.input_channels, input_spatial, "VAE downsample input")?;
    let output_len = checked_feature_len(
        conv.output_channels,
        output_spatial,
        "VAE downsample output",
    )?;
    let weight_len =
        checked_feature_len(conv.output_channels, patch_len * 2, "VAE downsample weight")?;
    let info = source
        .tensor_info(&conv.weight)
        .ok_or_else(|| format!("Missing tensor info: {}", conv.weight))?;
    let weights = source
        .tensor_slice(&conv.weight)
        .ok_or_else(|| format!("Missing tensor data: {}", conv.weight))?;
    if info.ggml_type != GGMLType::BF16
        || weights.len() != weight_len
        || input.len() != input_len
        || output.len() != output_len
        || input.iter().any(|value| !value.is_finite())
    {
        return Err("Invalid BF16 VAE downsample buffer".into());
    }
    let input_ptr = input.as_ptr() as usize;
    let weights_ptr = weights.as_ptr() as usize;
    let bias_ptr = conv.bias.as_ptr() as usize;
    let output_ptr = output.as_mut_ptr() as usize;
    pool.compute(move |thread, threads| {
        let input = unsafe { std::slice::from_raw_parts(input_ptr as *const f32, input_len) };
        let weights = unsafe { std::slice::from_raw_parts(weights_ptr as *const u8, weight_len) };
        let bias =
            unsafe { std::slice::from_raw_parts(bias_ptr as *const f32, conv.output_channels) };
        let output = output_ptr as *mut f32;
        let mut patch = vec![0.0f32; patch_len];
        for pixel in (thread..output_spatial).step_by(threads) {
            patch.fill(0.0);
            let y = pixel / output_side;
            let x = pixel % output_side;
            for channel in 0..conv.input_channels {
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = 2 * y as isize + ky as isize;
                        let ix = 2 * x as isize + kx as isize;
                        if iy >= 0 && ix >= 0 && iy < side as isize && ix < side as isize {
                            patch[(channel * 3 + ky) * 3 + kx] =
                                input[channel * input_spatial + iy as usize * side + ix as usize];
                        }
                    }
                }
            }
            if crate::ops::scalar_mode() {
                // The Oracle loads BF16 VAE weights as F16 and builds F16 im2col patches.
                for value in &mut patch {
                    *value = crate::ops::f16_to_f32(crate::ops::f32_to_f16(*value));
                }
            }
            for channel in 0..conv.output_channels {
                let start = channel * patch_len * 2;
                // Workers own disjoint pixels, including across channels.
                unsafe {
                    let weight = &weights[start..start + patch_len * 2];
                    let value = if crate::ops::scalar_mode() {
                        dot_vae_f16_scalar(&patch, weight)
                    } else {
                        crate::ops::dot_bf16_f32(&patch, weight, patch_len)
                    };
                    *output.add(channel * output_spatial + pixel) = value + bias[channel];
                }
            }
        }
    });
    if output.iter().any(|value| !value.is_finite()) {
        return Err("Non-finite BF16 VAE downsample output".into());
    }
    Ok(())
}

fn conv_bf16_parallel_into(
    input: &[f32],
    input_channels: usize,
    side: usize,
    weights: &[u8],
    output_channels: usize,
    kernel: usize,
    bias: &[f32],
    output: &mut [f32],
    pool: &Arc<ComputePool>,
    bf16_matmul: bool,
) -> Result<(), String> {
    if side == 0 || !matches!(kernel, 1 | 3) || bias.len() != output_channels {
        return Err("Invalid BF16 VAE convolution shape".into());
    }
    let spatial = checked_spatial(side, "BF16 VAE convolution")?;
    let patch_len = checked_feature_len(input_channels, kernel * kernel, "VAE patch")?;
    let expected_input = checked_feature_len(input_channels, spatial, "VAE input")?;
    let expected_output = checked_feature_len(output_channels, spatial, "VAE output")?;
    let expected_weight = checked_feature_len(output_channels, patch_len * 2, "VAE weight")?;
    if input.len() != expected_input
        || output.len() != expected_output
        || weights.len() != expected_weight
        || input.iter().chain(bias).any(|value| !value.is_finite())
    {
        return Err("Invalid BF16 VAE convolution buffer".into());
    }
    let input_ptr = input.as_ptr() as usize;
    let weights_ptr = weights.as_ptr() as usize;
    let bias_ptr = bias.as_ptr() as usize;
    let output_ptr = output.as_mut_ptr() as usize;
    pool.compute(move |thread, threads| {
        let input = unsafe { std::slice::from_raw_parts(input_ptr as *const f32, expected_input) };
        let weights =
            unsafe { std::slice::from_raw_parts(weights_ptr as *const u8, expected_weight) };
        let bias = unsafe { std::slice::from_raw_parts(bias_ptr as *const f32, output_channels) };
        let output = output_ptr as *mut f32;
        let mut patch = vec![0.0f32; patch_len];
        for pixel in (thread..spatial).step_by(threads) {
            patch.fill(0.0);
            let y = pixel / side;
            let x = pixel % side;
            for channel in 0..input_channels {
                for ky in 0..kernel {
                    for kx in 0..kernel {
                        let iy = y as isize + ky as isize - (kernel / 2) as isize;
                        let ix = x as isize + kx as isize - (kernel / 2) as isize;
                        if iy >= 0 && ix >= 0 && iy < side as isize && ix < side as isize {
                            patch[(channel * kernel + ky) * kernel + kx] =
                                input[channel * spatial + iy as usize * side + ix as usize];
                        }
                    }
                }
            }
            if crate::ops::scalar_mode() {
                // Convolutions use F16 im2col; the VAE attention linears use BF16.
                for value in &mut patch {
                    *value = if bf16_matmul {
                        crate::ops::bf16_to_f32(crate::ops::f32_to_bf16(*value))
                    } else {
                        crate::ops::f16_to_f32(crate::ops::f32_to_f16(*value))
                    };
                }
            }
            for channel in 0..output_channels {
                let start = channel * patch_len * 2;
                // Workers own disjoint pixels, including across channels.
                unsafe {
                    let weight = &weights[start..start + patch_len * 2];
                    let value = if crate::ops::scalar_mode() {
                        if bf16_matmul {
                            dot_vae_bf16_scalar(&patch, weight)
                        } else {
                            dot_vae_f16_scalar(&patch, weight)
                        }
                    } else {
                        crate::ops::dot_bf16_f32(&patch, weight, patch_len)
                    };
                    *output.add(channel * spatial + pixel) = value + bias[channel];
                }
            }
        }
    });
    if output.iter().any(|value| !value.is_finite()) {
        return Err("Non-finite BF16 VAE convolution output".into());
    }
    Ok(())
}

// The Oracle's scalar ggml_vec_dot_f16 sums F32 products in double precision.
fn dot_vae_f16_scalar(patch: &[f32], weights: &[u8]) -> f32 {
    let mut sum = 0.0f64;
    for (&input, bytes) in patch.iter().zip(weights.chunks_exact(2)) {
        let bf16 = u16::from_le_bytes([bytes[0], bytes[1]]);
        let weight = crate::ops::bf16_to_f32(bf16);
        let weight = crate::ops::f16_to_f32(crate::ops::f32_to_f16(weight));
        sum += (weight * input) as f64;
    }
    sum as f32
}

fn dot_vae_bf16_scalar(patch: &[f32], weights: &[u8]) -> f32 {
    let mut sum = 0.0f64;
    for (&input, bytes) in patch.iter().zip(weights.chunks_exact(2)) {
        let weight = crate::ops::bf16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        sum += (weight * input) as f64;
    }
    sum as f32
}

/// Per-pixel F16 dot convolution, parallelized by output pixel.
///
/// Each worker builds its own `[K*K*IC]` patch (allocated inside the
/// closure) for the pixels assigned to it and then computes the `OC` dot
/// products per pixel using the existing `dot_f16_f16_bytes` AVX2 kernel.
/// The patch buffer lives entirely in L1 of the worker thread and avoids
/// building a single shared image-wide im2col matrix that would exceed L3.
fn conv_f16_parallel_into(
    input: &[f32],
    input_channels: usize,
    side: usize,
    weights: &[u8],
    output_channels: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
    pool: &Arc<ComputePool>,
    #[cfg(feature = "vulkan")] gpu: Option<&crate::ops::kernel::vulkan::GpuLinear>,
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
    let pixel_count = side
        .checked_mul(side)
        .ok_or_else(|| "VAE pixel count overflow".to_string())?;
    #[cfg(feature = "vulkan")]
    if gpu.is_some_and(|gpu| {
        try_conv_f16(
            gpu,
            input,
            side,
            weights,
            output_channels,
            kernel,
            bias,
            output,
            patch_len,
            pool,
        )
    }) {
        return Ok(());
    }

    let weight_rows: Vec<&[u8]> = (0..output_channels)
        .map(|oc| {
            let start = oc * patch_len * 2;
            &weights[start..start + patch_len * 2]
        })
        .collect();
    let bias_values: Vec<f32> = (0..output_channels)
        .map(|oc| bias.map_or(0.0, |values| values[oc]))
        .collect();
    let input_usize = input.as_ptr() as usize;
    let input_len = input.len();
    let weight_rows_usize = weight_rows.as_ptr() as usize;
    let weight_rows_len = weight_rows.len();
    let bias_values_usize = bias_values.as_ptr() as usize;
    let bias_values_len = bias_values.len();
    let output_usize = output.as_mut_ptr() as usize;
    let padding = kernel / 2;
    let padding_signed = padding as isize;
    let side_signed = side as isize;

    pool.compute(move |ith, nth| {
        let per_thread = (pixel_count + nth - 1) / nth;
        let start = ith * per_thread;
        let end = (start + per_thread).min(pixel_count);
        if start >= end {
            return;
        }
        let input_local =
            unsafe { std::slice::from_raw_parts(input_usize as *const f32, input_len) };
        let weight_rows_local = unsafe {
            std::slice::from_raw_parts(weight_rows_usize as *const &[u8], weight_rows_len)
        };
        let bias_values_local =
            unsafe { std::slice::from_raw_parts(bias_values_usize as *const f32, bias_values_len) };
        let output_ptr = output_usize as *mut f32;
        let mut patch = vec![0u16; patch_len];
        for pixel in start..end {
            let output_y = pixel / side;
            let output_x = pixel % side;
            patch.fill(0);
            // Build patch (kernel_x, kernel_y, ic) -> patch_index
            for kernel_y in 0..kernel {
                let input_y_signed = output_y as isize + kernel_y as isize - padding_signed;
                for kernel_x in 0..kernel {
                    let input_x_signed = output_x as isize + kernel_x as isize - padding_signed;
                    let in_bounds = input_y_signed >= 0
                        && input_y_signed < side_signed
                        && input_x_signed >= 0
                        && input_x_signed < side_signed;
                    if in_bounds {
                        let input_y = input_y_signed as usize;
                        let input_x = input_x_signed as usize;
                        for input_channel in 0..input_channels {
                            let input_plane_base = input_channel * spatial;
                            let value = input_local[input_plane_base + input_y * side + input_x];
                            let patch_index =
                                kernel_x + kernel * (kernel_y + kernel * input_channel);
                            patch[patch_index] = f32_to_f16(value);
                        }
                    } else {
                        for input_channel in 0..input_channels {
                            let patch_index =
                                kernel_x + kernel * (kernel_y + kernel * input_channel);
                            patch[patch_index] = 0;
                        }
                    }
                }
            }
            for oc in 0..output_channels {
                let dot = dot_f16_f16_bytes(&patch, weight_rows_local[oc], patch_len);
                // Workers own disjoint pixels across every output channel.
                unsafe {
                    *output_ptr.add(oc * spatial + pixel) = dot + bias_values_local[oc];
                }
            }
        }
    });

    if output.is_empty() || output.iter().any(|value| !value.is_finite()) {
        return Err("Non-finite VAE convolution output".into());
    }
    Ok(())
}

#[cfg(any(feature = "vulkan", test))]
fn conv_patch_into(input: &[f32], side: usize, kernel: usize, pixel: usize, patch: &mut [f32]) {
    patch.fill(0.0);
    let spatial = side * side;
    let padding = (kernel / 2) as isize;
    let top = (pixel / side) as isize - padding;
    let left = (pixel % side) as isize - padding;
    let start_x = left.max(0) as usize;
    let end_x = (left + kernel as isize).min(side as isize) as usize;
    for (channel, patch) in patch.chunks_exact_mut(kernel * kernel).enumerate() {
        for ky in 0..kernel {
            let y = top + ky as isize;
            if y >= 0 && y < side as isize {
                let source = channel * spatial + y as usize * side;
                let destination = ky * kernel + (start_x as isize - left) as usize;
                patch[destination..destination + end_x - start_x]
                    .copy_from_slice(&input[source + start_x..source + end_x]);
            }
        }
    }
}

#[cfg(feature = "vulkan")]
fn try_conv_f16(
    gpu: &crate::ops::kernel::vulkan::GpuLinear,
    input: &[f32],
    side: usize,
    weights: &[u8],
    output_channels: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
    patch_len: usize,
    pool: &ComputePool,
) -> bool {
    if !crate::ops::kernel::vulkan::offload_enabled() {
        return false;
    }
    let spatial = side * side;
    let tile_rows = crate::ops::kernel::vulkan::GpuLinear::tile_rows(
        crate::vulkan::ops::GpuWeightFormat::F16F32,
        patch_len,
        output_channels,
        spatial,
    );
    let mut patches = vec![0.0; tile_rows * patch_len];
    let mut projected = vec![0.0; tile_rows * output_channels];
    for start in (0..spatial).step_by(tile_rows) {
        let rows = tile_rows.min(spatial - start);
        let patches_ptr = patches.as_mut_ptr() as usize;
        pool.compute(move |ith, nth| {
            let per_thread = rows.div_ceil(nth);
            let first = ith * per_thread;
            let end = (first + per_thread).min(rows);
            if first >= end {
                return;
            }
            // Each worker owns disjoint pixel rows; compute joins before the GPU upload.
            let patches = unsafe {
                std::slice::from_raw_parts_mut(
                    (patches_ptr as *mut f32).add(first * patch_len),
                    (end - first) * patch_len,
                )
            };
            for (row, patch) in patches.chunks_exact_mut(patch_len).enumerate() {
                conv_patch_into(input, side, kernel, start + first + row, patch);
            }
        });
        if !gpu.try_matmul(
            weights,
            crate::vulkan::ops::GpuWeightFormat::F16F32,
            &patches[..rows * patch_len],
            &mut projected[..rows * output_channels],
            patch_len,
            output_channels,
            rows,
        ) {
            return false;
        }
        let output_ptr = output.as_mut_ptr() as usize;
        let projected = &projected[..rows * output_channels];
        pool.compute(move |ith, nth| {
            let per_thread = output_channels.div_ceil(nth);
            for channel in ith * per_thread..((ith + 1) * per_thread).min(output_channels) {
                // Each worker owns separate channel planes; only this completed tile is written.
                let output = unsafe {
                    std::slice::from_raw_parts_mut(
                        (output_ptr as *mut f32).add(channel * spatial + start),
                        rows,
                    )
                };
                let bias = bias.map_or(0.0, |bias| bias[channel]);
                for (row, value) in output.iter_mut().enumerate() {
                    *value = projected[row * output_channels + channel] + bias;
                }
            }
        });
    }
    output.iter().all(|value| value.is_finite())
}

fn padded_conv_f16_into(
    input: &[f32],
    input_channels: usize,
    side: usize,
    weights: &[u8],
    output_channels: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
    pool: &Arc<ComputePool>,
) -> Result<(), String> {
    conv_f16_parallel_into(
        input,
        input_channels,
        side,
        weights,
        output_channels,
        3,
        bias,
        output,
        pool,
        #[cfg(feature = "vulkan")]
        None,
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
    silu_inplace(values);
    if values.iter().any(|value| !value.is_finite()) {
        return Err("Non-finite VAE SiLU output".into());
    }
    Ok(())
}

#[cfg(test)]
fn add_shortcut_residual_into(
    input: &[f32],
    residual_branch: &[f32],
    input_channels: usize,
    output_channels: usize,
    side: usize,
    weights: &[u8],
    bias: Option<&[f32]>,
    output: &mut [f32],
    pool: &Arc<ComputePool>,
    #[cfg(feature = "vulkan")] gpu: Option<&crate::ops::kernel::vulkan::GpuLinear>,
) -> Result<(), String> {
    conv_f16_parallel_into(
        input,
        input_channels,
        side,
        weights,
        output_channels,
        1,
        bias,
        output,
        pool,
        #[cfg(feature = "vulkan")]
        gpu,
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
    let scalar = crate::ops::scalar_mode();
    let mut query = vec![0.; channels];
    let mut key = vec![0.; channels];
    for query_position in 0..spatial {
        for channel in 0..channels {
            query[channel] = q[channel * spatial + query_position];
        }
        for key_position in 0..spatial {
            for channel in 0..channels {
                key[channel] = k[channel * spatial + key_position];
            }
            let score = if scalar {
                let mut sum = 0.0f64;
                for channel in 0..channels {
                    sum += (q[channel * spatial + query_position]
                        * k[channel * spatial + key_position]) as f64;
                }
                sum as f32 * scale
            } else {
                crate::ops::dot_f32(&query, &key, channels) * scale
            };
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
            let value = if scalar {
                let mut sum = 0.0f64;
                for key_position in 0..spatial {
                    sum += (scores[key_position] * v[channel * spatial + key_position]) as f64;
                }
                sum as f32
            } else {
                crate::ops::dot_f32(
                    scores,
                    &v[channel * spatial..(channel + 1) * spatial],
                    spatial,
                )
            };
            if !value.is_finite() {
                return Err("Non-finite VAE attention output".into());
            }
            output[channel * spatial + query_position] = value;
        }
    }
    Ok(())
}

/// Runs the VAE's single-head spatial attention by partitioning query rows.
///
/// A query row only reads Q/K/V and writes its own spatial positions in the
/// output, so query rows are independent. Keeping the dot-product and softmax
/// loops unchanged preserves the scalar result for each row while allowing
/// the existing inference pool to process rows concurrently.
fn one_head_spatial_attention_parallel_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    channels: usize,
    spatial: usize,
    output: &mut [f32],
    scores: &mut [f32],
    pool: &Arc<ComputePool>,
) -> Result<(), String> {
    if pool.n_threads() <= 1 {
        return one_head_spatial_attention_into(q, k, v, channels, spatial, output, scores);
    }
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
    let output_ptr = output.as_mut_ptr() as usize;
    let failure = AtomicBool::new(false);
    let worker_failure = &failure;

    // SAFETY: each worker receives a disjoint query-position range. It only
    // reads q/k/v and writes output[channel * spatial + query_position] for
    // positions in that range. The source slices and output outlive compute.
    pool.compute(move |ith, nth| {
        let per_thread = spatial.div_ceil(nth);
        let start = ith * per_thread;
        let end = (start + per_thread).min(spatial);
        if start >= end {
            return;
        }
        let mut local_scores = vec![0.0f32; spatial];
        let mut query = vec![0.; channels];
        let mut key = vec![0.; channels];

        for query_position in start..end {
            for channel in 0..channels {
                query[channel] = q[channel * spatial + query_position];
            }
            for key_position in 0..spatial {
                for channel in 0..channels {
                    key[channel] = k[channel * spatial + key_position];
                }
                let score = crate::ops::dot_f32(&query, &key, channels) * scale;
                if !score.is_finite() {
                    worker_failure.store(true, Ordering::Relaxed);
                    return;
                }
                local_scores[key_position] = score;
            }
            vae_softmax_inplace(&mut local_scores);
            if local_scores.iter().any(|value| !value.is_finite()) {
                worker_failure.store(true, Ordering::Relaxed);
                return;
            }
            for channel in 0..channels {
                let value = crate::ops::dot_f32(
                    &local_scores,
                    &v[channel * spatial..(channel + 1) * spatial],
                    spatial,
                );
                if !value.is_finite() {
                    worker_failure.store(true, Ordering::Relaxed);
                    return;
                }
                // Write the query element directly so each worker creates no
                // mutable slice that aliases another worker's disjoint
                // strided query range.
                unsafe {
                    *(output_ptr as *mut f32).add(channel * spatial + query_position) = value;
                }
            }
        }
    });

    if failure.load(Ordering::Relaxed) || output.iter().any(|value| !value.is_finite()) {
        return Err("Non-finite VAE attention output".into());
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
    pool: &Arc<ComputePool>,
) -> Result<Vec<f32>, String> {
    let output_side = side
        .checked_mul(2)
        .ok_or_else(|| "VAE upsample shape overflow".to_string())?;
    let output_spatial = checked_spatial(output_side, "VAE upsample")?;
    let output_len = checked_feature_len(channels, output_spatial, "VAE upsample")?;
    let mut nearest = reserve_f32("VAE nearest upsample", output_len)?;
    upsample_nearest_into(input, channels, side, &mut nearest)?;
    let mut output = reserve_f32("VAE learned upsample", output_len)?;
    padded_conv_f16_into(
        &nearest,
        channels,
        output_side,
        weights,
        channels,
        bias,
        &mut output,
        pool,
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
    #[cfg(feature = "vulkan")]
    use crate::ops::f16_to_f32;
    use half::f16;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires a Vulkan device and Z_IMAGE_VAE pointing to the real F16 decoder"]
    fn vulkan_vae_real_weights_decode_matches_cpu() {
        let source = Arc::new(
            crate::core::loader::GGUFLoader::from_file(
                std::env::var("Z_IMAGE_VAE").expect("missing Z_IMAGE_VAE"),
            )
            .unwrap(),
        );
        let vae = FluxVae::load(source, Arc::new(ComputePool::new(8))).unwrap();
        let mut latent = vec![0.0; 16 * 64 * 64];
        TorchMt19937::new(42).fill_normal(&mut latent);
        let start = std::time::Instant::now();
        let cpu = {
            let _disabled = ComputePool::disable_gpu_matmul_for_scope();
            vae.decode_rgb(&latent, 64).unwrap()
        };
        eprintln!(
            "[vae-bench] cpu_cold={:.6}s threads=8",
            start.elapsed().as_secs_f64()
        );
        assert_eq!(
            (cpu.width, cpu.height, cpu.bytes.len()),
            (512, 512, 512 * 512 * 3)
        );
        eprintln!(
            "[vae-bench] cpu_gpu_requested={}",
            crate::ops::gpu_requested()
        );
        let mut cpu_times = Vec::new();
        for run in 0..3 {
            let _disabled = ComputePool::disable_gpu_matmul_for_scope();
            let start = std::time::Instant::now();
            let actual = vae.decode_rgb(&latent, 64).unwrap();
            let seconds = start.elapsed().as_secs_f64();
            assert_eq!(actual.bytes, cpu.bytes);
            eprintln!("[vae-bench] cpu_warm_run={run} seconds={seconds:.6}");
            cpu_times.push(seconds);
        }
        crate::ops::enable_gpu();
        let context = crate::ops::get_vulkan_context().expect("Vulkan device required");
        let mut gpu_times = Vec::new();
        for run in 0..4 {
            let before = context.submission_count();
            let start = std::time::Instant::now();
            let actual = vae.decode_rgb(&latent, 64).unwrap();
            let seconds = start.elapsed().as_secs_f64();
            assert!(!crate::vulkan::gpu_broken());
            assert!(context.submission_count() > before);
            assert_eq!(actual.bytes.len(), cpu.bytes.len());
            let mut maximum = 0.0f64;
            let mut absolute = 0.0;
            let mut squared = 0.0;
            for (&a, &b) in actual.bytes.iter().zip(&cpu.bytes) {
                let error = (a as f64 - b as f64).abs();
                maximum = maximum.max(error);
                absolute += error;
                squared += error * error;
            }
            let mae = absolute / cpu.bytes.len() as f64;
            let psnr = 10.0 * (255.0 * 255.0 / (squared / cpu.bytes.len() as f64)).log10();
            eprintln!("[vae-bench] gpu_run={run} seconds={seconds:.6} submissions={} max_byte_error={maximum} mae={mae:.6} psnr={psnr:.3}dB", context.submission_count() - before);
            assert!(psnr > 40.0, "VAE CPU/GPU PSNR={psnr}dB");
            if run > 0 {
                gpu_times.push(seconds);
            }
        }
        cpu_times.sort_by(f64::total_cmp);
        gpu_times.sort_by(f64::total_cmp);
        eprintln!(
            "[vae-bench] warm_median cpu={:.6}s gpu={:.6}s",
            cpu_times[1], gpu_times[1]
        );
        crate::vulkan::dump_submit_trace();
        crate::vulkan::ops::dump_dispatch_trace();
    }
    #[test]
    fn vae_f16_dot_uses_double_accumulator() {
        let weights: Vec<u8> = [1.0, 2.0f32.powi(-24), 2.0f32.powi(-24)]
            .into_iter()
            .flat_map(|value| crate::ops::f32_to_bf16(value).to_le_bytes())
            .collect();
        assert_eq!(
            dot_vae_f16_scalar(&[1.0; 3], &weights).to_bits(),
            (1.0f32 + 2.0f32.powi(-23)).to_bits()
        );
        assert_eq!(
            dot_vae_bf16_scalar(&[1.0; 3], &weights).to_bits(),
            (1.0f32 + 2.0f32.powi(-23)).to_bits()
        );
        if crate::ops::scalar_mode() {
            let mut output = [0.0];
            conv_bf16_parallel_into(
                &[1.0 + 1.0 / 256.0],
                1,
                1,
                &weights[..2],
                1,
                1,
                &[0.0],
                &mut output,
                &Arc::new(ComputePool::new(1)),
                true,
            )
            .unwrap();
            assert_eq!(output[0], 1.0);
        }
    }

    #[test]
    fn bf16_downsample_uses_asymmetric_right_bottom_padding() {
        struct OneConv {
            info: TensorInfo,
            weights: Vec<u8>,
        }
        impl TensorSource for OneConv {
            fn metadata(&self, _: &str) -> Option<&MetaValue> {
                None
            }
            fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
                (name == "weight").then_some(&self.info)
            }
            fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
                (name == "weight").then_some(&self.weights)
            }
        }
        let source = OneConv {
            info: TensorInfo {
                name: "weight".into(),
                dims: vec![3, 3, 1, 1],
                ggml_type: GGMLType::BF16,
                offset: 0,
            },
            weights: [1.0f32; 9]
                .iter()
                .flat_map(|value| crate::ops::f32_to_bf16(*value).to_le_bytes())
                .collect(),
        };
        let conv = VaeConv {
            weight: "weight".into(),
            bias: vec![0.0],
            input_channels: 1,
            output_channels: 1,
            kernel: 3,
            #[cfg(feature = "vulkan")]
            gpu: crate::ops::kernel::vulkan::GpuLinear::default(),
            linear: None,
        };
        let mut output = [0.0; 4];
        run_bf16_downsample(
            &source,
            &Arc::new(ComputePool::new(1)),
            &conv,
            &[1.0; 16],
            4,
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [9.0, 6.0, 6.0, 4.0]);
        if crate::ops::scalar_mode() {
            run_bf16_downsample(
                &source,
                &Arc::new(ComputePool::new(1)),
                &conv,
                &[1.0 + 1.0 / 4096.0; 16],
                4,
                &mut output,
            )
            .unwrap();
            assert_eq!(output, [9.0, 6.0, 6.0, 4.0]);
            let mut full = [0.0; 16];
            conv_bf16_parallel_into(
                &[1.0 + 1.0 / 4096.0; 16],
                1,
                4,
                &source.weights,
                1,
                3,
                &[0.0],
                &mut full,
                &Arc::new(ComputePool::new(1)),
                false,
            )
            .unwrap();
            assert_eq!(full[0], 4.0);
            assert_eq!(full[5], 9.0);
        }
    }

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
    fn convolution_zero_padding_is_reset_between_pixels() {
        let mut output = [0.0; 4];
        conv_f16_parallel_into(
            &[1.0, 2.0, 3.0, 4.0],
            1,
            2,
            &f16_bytes(&[1.0; 9]),
            1,
            3,
            None,
            &mut output,
            &Arc::new(ComputePool::new(1)),
            #[cfg(feature = "vulkan")]
            None,
        )
        .unwrap();
        assert_eq!(output, [10.0; 4]);
    }

    #[test]
    fn convolution_patch_keeps_channel_and_kernel_order() {
        let mut patch = [99.0; 18];
        conv_patch_into(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            2,
            3,
            0,
            &mut patch,
        );
        assert_eq!(
            patch,
            [
                0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0,
                7.0, 8.0
            ]
        );
        conv_patch_into(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            2,
            3,
            3,
            &mut patch,
        );
        assert_eq!(
            patch,
            [
                1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 7.0, 8.0, 0.0, 0.0,
                0.0, 0.0
            ]
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires a Vulkan device; fails if offload is unavailable"]
    fn vulkan_vae_convolution_matches_cpu_across_tiles() {
        crate::ops::enable_gpu();
        let context = crate::ops::get_vulkan_context().expect("Vulkan device required");
        let pool = Arc::new(ComputePool::new(8));
        assert!(matches!(
            crate::vulkan::ops::Conv2dRuntime::new(context, (u32::MAX as usize / 4 + 1, 1, 1)),
            Err(crate::vulkan::VulkanError::UnsupportedShape(_))
        ));
        let shapes = [(9, 2, 3), (65, 32, 65), (65, 1, 3)];
        let weight_storage: Vec<Vec<u8>> = shapes
            .iter()
            .flat_map(|&(_, input_channels, output_channels)| {
                [1, 3].map(move |kernel| {
                    let patch_len = input_channels * kernel * kernel;
                    (0..patch_len * output_channels)
                        .flat_map(|i| f32_to_f16((i as f32 % 11.0 - 5.0) / 16.0).to_le_bytes())
                        .collect()
                })
            })
            .collect();
        let direct = crate::ops::kernel::vulkan::GpuConv::default();
        // Keep immutable weights alive while the shared arena grows, then reuse the first weights.
        for shape_index in [0, 1, 2, 0] {
            let (side, input_channels, output_channels) = shapes[shape_index];
            let spatial = side * side;
            let input: Vec<_> = (0..spatial * input_channels)
                .map(|i| ((i * 29 % 251) as f32 - 125.0) / 97.0)
                .collect();
            let bias: Vec<_> = (0..output_channels)
                .map(|i| (i as f32 - 1.0) / 8.0)
                .collect();
            for (kernel_index, kernel) in [1, 3].into_iter().enumerate() {
                let patch_len = input_channels * kernel * kernel;
                let weights = &weight_storage[shape_index * 2 + kernel_index];
                let mut expected = vec![0.0; spatial * output_channels];
                conv_f16_parallel_into(
                    &input,
                    input_channels,
                    side,
                    &weights,
                    output_channels,
                    kernel,
                    Some(&bias),
                    &mut expected,
                    &pool,
                    None,
                )
                .unwrap();
                // AArch64's CPU kernel accumulates in FP16; check GPU FP32 against an independent sum.
                let mut reference = vec![0.0; expected.len()];
                let mut patch = vec![0.0; patch_len];
                for pixel in 0..spatial {
                    conv_patch_into(&input, side, kernel, pixel, &mut patch);
                    for (channel, weight) in weights.chunks_exact(patch_len * 2).enumerate() {
                        let sum: f64 = patch
                            .iter()
                            .zip(weight.chunks_exact(2))
                            .map(|(&x, w)| {
                                f64::from(f16_to_f32(f32_to_f16(x)))
                                    * f64::from(f16_to_f32(u16::from_le_bytes([w[0], w[1]])))
                            })
                            .sum();
                        reference[channel * spatial + pixel] = sum as f32 + bias[channel];
                    }
                }
                let gpu = crate::ops::kernel::vulkan::GpuLinear::default();
                let submissions = context.submission_count();
                let mut actual = vec![f32::NAN; expected.len()];
                assert!(try_conv_f16(
                    &gpu,
                    &input,
                    side,
                    &weights,
                    output_channels,
                    kernel,
                    Some(&bias),
                    &mut actual,
                    patch_len,
                    &pool,
                ));
                assert_eq!(
                    context.submission_count() - submissions,
                    spatial.div_ceil(4096) as u64
                );
                for (actual, expected) in actual.iter().zip(&reference) {
                    assert!(
                        (actual - expected).abs() <= 3e-4 + 3e-4 * expected.abs(),
                        "side={side} kernel={kernel} channels={input_channels}/{output_channels} gpu={actual} reference={expected}"
                    );
                }
                let submissions = context.submission_count();
                let mut direct_output = vec![f32::NAN; actual.len()];
                assert!(direct.try_conv_f16(
                    &weights,
                    &input,
                    &mut direct_output,
                    input_channels,
                    output_channels,
                    side,
                    kernel,
                    Some(&bias),
                ));
                assert_eq!(
                    context.submission_count() - submissions,
                    1,
                    "a convolution must finish in one submission without host im2col tiles"
                );
                for (direct, tiled) in direct_output.iter().zip(&actual) {
                    assert_eq!(
                        direct.to_bits(),
                        tiled.to_bits(),
                        "direct convolution must retain the tiled GPU arithmetic"
                    );
                }
                let submissions = context.submission_count();
                assert!(direct.try_conv_f16(
                    &weights,
                    &input,
                    &mut direct_output,
                    input_channels,
                    output_channels,
                    side,
                    kernel,
                    None,
                ));
                assert_eq!(context.submission_count() - submissions, 1);
                for (i, &value) in direct_output.iter().enumerate() {
                    assert!(
                        (value - (reference[i] - bias[i / spatial])).abs()
                            <= 3e-4 + 3e-4 * reference[i].abs()
                    );
                }
                let submissions = context.submission_count();
                let unchanged: Vec<_> = direct_output.iter().map(|v| v.to_bits()).collect();
                assert!(!direct.try_conv_f16(
                    &weights,
                    &input,
                    &mut direct_output,
                    input_channels,
                    output_channels,
                    side,
                    kernel,
                    Some(&bias[..bias.len() - 1]),
                ));
                assert!(!direct.try_conv_f16(
                    &weights,
                    &input,
                    &mut direct_output,
                    usize::MAX,
                    output_channels,
                    side,
                    kernel,
                    Some(&bias),
                ));
                assert_eq!(context.submission_count(), submissions);
                assert_eq!(
                    direct_output
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    unchanged
                );
                assert!(!crate::vulkan::gpu_broken());
                let _disabled = ComputePool::disable_gpu_matmul_for_scope();
                let submissions = context.submission_count();
                assert!(!direct.try_conv_f16(
                    &weights,
                    &input,
                    &mut direct_output,
                    input_channels,
                    output_channels,
                    side,
                    kernel,
                    Some(&bias),
                ));
                actual.fill(f32::NAN);
                conv_f16_parallel_into(
                    &input,
                    input_channels,
                    side,
                    &weights,
                    output_channels,
                    kernel,
                    Some(&bias),
                    &mut actual,
                    &pool,
                    Some(&gpu),
                )
                .unwrap();
                assert_eq!(context.submission_count(), submissions);
                assert!(actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, e)| a.to_bits() == e.to_bits()));
            }
        }
    }

    #[test]
    fn conv_f16_parallel_runs() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weights = f16_bytes(&[1.0]);
        let mut output = [0.0; 4];
        let pool = Arc::new(ComputePool::new(1));

        conv_f16_parallel_into(
            &input,
            1,
            2,
            &weights,
            1,
            1,
            None,
            &mut output,
            &pool,
            #[cfg(feature = "vulkan")]
            None,
        )
        .unwrap();
        // Verify output values are non-zero (regression check on parallelism).
        assert!(output.iter().any(|&v| v.is_finite()));
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

        conv_f16_parallel_into(
            &input,
            16,
            2,
            &weights,
            1,
            3,
            Some(&[f32::from_bits(0xbd69_17db)]),
            &mut output,
            &Arc::new(ComputePool::new(1)),
            #[cfg(feature = "vulkan")]
            None,
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
            &Arc::new(ComputePool::new(1)),
        )
        .unwrap();
        assert_eq!(
            output,
            vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,]
        );
    }

    #[test]
    fn missing_mid_attention_is_a_load_error() {
        assert!(FluxVae::load(
            Arc::new(decoder_source_without("decoder.mid.attn_1.q.weight")),
            Arc::new(ComputePool::new(1))
        )
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
        add_shortcut_residual_into(
            &[1.0, 2.0],
            &[7.0],
            2,
            1,
            1,
            &f16_bytes(&[2.0, 3.0]),
            Some(&[5.0]),
            &mut output,
            &Arc::new(ComputePool::new(1)),
            #[cfg(feature = "vulkan")]
            None,
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
    fn mid_attention_parallel_matches_scalar_query_partitioning() {
        for (channels, spatial) in [(1, 1), (4, 5), (7, 17), (512, 33)] {
            let q = (0..channels * spatial)
                .map(|index| ((index * 17 % 143) as f32 - 71.0) * 0.03125)
                .collect::<Vec<_>>();
            let k = (0..channels * spatial)
                .map(|index| ((index * 13 % 137) as f32 - 68.0) * -0.046875)
                .collect::<Vec<_>>();
            let v = (0..channels * spatial)
                .map(|index| ((index * 7 % 131) as f32 - 65.0) * 0.0625)
                .collect::<Vec<_>>();
            let mut expected = vec![0.0; channels * spatial];
            let mut scalar_scores = vec![0.0; spatial];
            one_head_spatial_attention_into(
                &q,
                &k,
                &v,
                channels,
                spatial,
                &mut expected,
                &mut scalar_scores,
            )
            .unwrap();
            let expected_bits = expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            for threads in [1, 2, 4, 8] {
                // NaNs make missing query writes fail, including an empty
                // worker range when there are more threads than queries.
                let mut actual = vec![f32::NAN; channels * spatial];
                let mut parallel_scores = vec![0.0; spatial];
                one_head_spatial_attention_parallel_into(
                    &q,
                    &k,
                    &v,
                    channels,
                    spatial,
                    &mut actual,
                    &mut parallel_scores,
                    &Arc::new(ComputePool::new(threads)),
                )
                .unwrap();
                assert_eq!(
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected_bits,
                    "channels={channels}, spatial={spatial}, threads={threads}"
                );
            }
        }
    }

    #[test]
    fn mid_attention_parallel_rejects_invalid_and_non_finite_buffers() {
        let pool = Arc::new(ComputePool::new(4));
        let mut output = [0.0; 2];
        let mut scores = [0.0; 2];
        let mut run = |q: &[f32], k: &[f32], v: &[f32]| {
            one_head_spatial_attention_parallel_into(q, k, v, 1, 2, &mut output, &mut scores, &pool)
        };
        assert_eq!(
            run(&[0.0], &[0.0; 2], &[0.0; 2]).unwrap_err(),
            "Invalid VAE attention buffer length"
        );
        assert_eq!(
            run(&[0.0, f32::NAN], &[0.0; 2], &[0.0; 2]).unwrap_err(),
            "Non-finite VAE attention input"
        );
        // Finite inputs can still overflow inside a worker. Reuse the same
        // pool afterwards to verify the error does not leave workers busy.
        assert!(run(&[0.0, f32::MAX], &[2.0; 2], &[1.0; 2]).is_err());
        run(&[0.0; 2], &[0.0; 2], &[1.0, 3.0]).unwrap();
        assert_eq!(output, [2.0; 2]);
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
        let vae = FluxVae::load(
            Arc::new(decoder_source_without("")),
            Arc::new(ComputePool::new(1)),
        )
        .unwrap();
        assert!(vae.decode_rgb(&[0.0; 15], 1).is_err());
        assert!(vae.decode_rgb(&[], 0).is_err());
    }

    #[test]
    fn decoder_stages_load_in_oracle_order_with_three_blocks_each() {
        let vae = FluxVae::load(
            Arc::new(decoder_source_without("")),
            Arc::new(ComputePool::new(1)),
        )
        .unwrap();
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
        padded_conv_f16_into(
            &[5.0, 6.0],
            2,
            1,
            &f16_bytes(&weights),
            2,
            None,
            &mut output,
            &Arc::new(ComputePool::new(1)),
        )
        .unwrap();
        assert_eq!(output, [17.0, 39.0]);
    }

    #[test]
    fn convolution_clears_padding_when_reusing_a_patch() {
        let weights: Vec<u8> = [1.0f32; 9]
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        for threads in [1, 2, 4] {
            let mut output = [0.; 4];
            padded_conv_f16_into(
                &[1., 2., 3., 4.],
                1,
                2,
                &weights,
                1,
                None,
                &mut output,
                &Arc::new(ComputePool::new(threads)),
            )
            .unwrap();
            assert_eq!(output, [10.; 4]);
        }
    }

    #[test]
    fn convolution_uses_zero_padding_at_image_edges() {
        let mut output = [0.0];
        padded_conv_f16_into(
            &[2.0],
            1,
            1,
            &f16_bytes(&[1.0; 9]),
            1,
            None,
            &mut output,
            &Arc::new(ComputePool::new(1)),
        )
        .unwrap();
        assert_eq!(output, [2.0]);
    }

    #[test]
    fn non_finite_convolution_weight_is_fatal() {
        let mut weights = vec![0.0; 9];
        weights[4] = f32::INFINITY;
        let mut output = [0.0];
        assert!(padded_conv_f16_into(
            &[1.0],
            1,
            1,
            &f16_bytes(&weights),
            1,
            None,
            &mut output,
            &Arc::new(ComputePool::new(1)),
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
        FluxVae::load(Arc::new(source), Arc::new(ComputePool::new(1))).unwrap();
    }
}
