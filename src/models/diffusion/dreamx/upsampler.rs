use std::sync::Arc;

use super::kernels::{conv2d, conv3d_with_options, load_float_values, Conv3dSpec};
use super::video_vae::VideoLatent;
use super::LatentUpsampleKind;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::ops::silu_inplace;

const FLASH_PREFIX: &str = "dreamx.refiner.upsampler.flash";
const CAUSAL_PREFIX: &str = "dreamx.refiner.upsampler.causal2d";

pub enum LatentUpsampler {
    Bilinear,
    Flash(FlashUpsampler),
    Causal2d(Causal2dUpsampler),
}

impl std::fmt::Debug for LatentUpsampler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Bilinear => "LatentUpsampler::Bilinear",
            Self::Flash(_) => "LatentUpsampler::Flash",
            Self::Causal2d(_) => "LatentUpsampler::Causal2d",
        })
    }
}

impl LatentUpsampler {
    pub fn load(
        kind: LatentUpsampleKind,
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        Ok(match kind {
            LatentUpsampleKind::Bilinear => Self::Bilinear,
            LatentUpsampleKind::Flash => Self::Flash(FlashUpsampler::load(source, pool)?),
            LatentUpsampleKind::Causal2d => Self::Causal2d(Causal2dUpsampler::load(source, pool)?),
        })
    }

    pub fn upsample(&self, latent: &VideoLatent) -> Result<VideoLatent, String> {
        match self {
            Self::Bilinear => bilinear_2x(latent),
            Self::Flash(model) => model.upsample(latent),
            Self::Causal2d(model) => model.upsample(latent),
        }
    }

    #[cfg(test)]
    fn testing(_kind: LatentUpsampleKind) -> Self {
        Self::Bilinear
    }
}

struct Conv2 {
    weight: Vec<f32>,
    bias: Vec<f32>,
    shape: [usize; 4],
    padding: [usize; 2],
}

impl Conv2 {
    fn load(
        source: &dyn TensorSource,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        padding: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: load(
                source,
                &format!("{name}.weight"),
                &[output, input, kernel, kernel],
            )?,
            bias: load(source, &format!("{name}.bias"), &[output])?,
            shape: [output, input, kernel, kernel],
            padding: [padding; 2],
        })
    }

    fn forward(&self, pool: &ComputePool, input: &Image) -> Result<Image, String> {
        let (data, shape) = conv2d(
            pool,
            &input.data,
            input.shape,
            &self.weight,
            self.shape,
            Some(&self.bias),
            [1; 2],
            self.padding,
            [1; 2],
            1,
        )?;
        Image::new(data, shape)
    }
}

struct Dense {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
}

impl Dense {
    fn load(
        source: &dyn TensorSource,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: load(source, &format!("{name}.weight"), &[output, input])?,
            bias: load(source, &format!("{name}.bias"), &[output])?,
            input,
            output,
        })
    }

    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.len() != self.input {
            return Err("Invalid DreamX upsampler dense input".into());
        }
        let mut output = self.bias.clone();
        for (row, value) in output.iter_mut().enumerate() {
            let weight = &self.weight[row * self.input..(row + 1) * self.input];
            *value += weight.iter().zip(input).map(|(a, b)| a * b).sum::<f32>();
        }
        if output.len() != self.output {
            return Err("Invalid DreamX upsampler dense output".into());
        }
        Ok(output)
    }
}

#[derive(Clone)]
struct Image {
    data: Vec<f32>,
    shape: [usize; 3],
}

impl Image {
    fn new(data: Vec<f32>, shape: [usize; 3]) -> Result<Self, String> {
        if data.len() != checked_product(&shape)? || data.iter().any(|value| !value.is_finite()) {
            return Err("Invalid DreamX upsampler image feature".into());
        }
        Ok(Self { data, shape })
    }
}

#[derive(Clone)]
struct Feature {
    data: Vec<f32>,
    shape: [usize; 4],
}

impl Feature {
    fn new(data: Vec<f32>, shape: [usize; 4]) -> Result<Self, String> {
        if data.len() != checked_product(&shape)? || data.iter().any(|value| !value.is_finite()) {
            return Err("Invalid DreamX upsampler feature".into());
        }
        Ok(Self { data, shape })
    }

    fn frame(&self, frame: usize) -> Result<Image, String> {
        let [channels, frames, height, width] = self.shape;
        if frame >= frames {
            return Err("DreamX upsampler frame is out of bounds".into());
        }
        let plane = height * width;
        let mut data = vec![0.0; channels * plane];
        for channel in 0..channels {
            let source = (channel * frames + frame) * plane;
            data[channel * plane..(channel + 1) * plane]
                .copy_from_slice(&self.data[source..source + plane]);
        }
        Image::new(data, [channels, height, width])
    }

    fn from_frames(frames: Vec<Image>) -> Result<Self, String> {
        let first = frames.first().ok_or("DreamX upsampler requires frames")?;
        let [channels, height, width] = first.shape;
        if frames.iter().any(|frame| frame.shape != first.shape) {
            return Err("DreamX upsampler frame shapes differ".into());
        }
        let depth = frames.len();
        let plane = height * width;
        let mut data = vec![0.0; channels * depth * plane];
        for (frame_index, frame) in frames.into_iter().enumerate() {
            for channel in 0..channels {
                let target = (channel * depth + frame_index) * plane;
                data[target..target + plane]
                    .copy_from_slice(&frame.data[channel * plane..(channel + 1) * plane]);
            }
        }
        Self::new(data, [channels, depth, height, width])
    }
}

struct FlashBlock {
    first: Conv2,
    second: Conv2,
    third: Conv2,
}

impl FlashBlock {
    fn load(source: &dyn TensorSource, index: usize) -> Result<Self, String> {
        let prefix = format!("{FLASH_PREFIX}.blocks.{index}.conv");
        Ok(Self {
            first: Conv2::load(source, &format!("{prefix}.0"), 256, 128, 3, 1)?,
            second: Conv2::load(source, &format!("{prefix}.2"), 128, 128, 3, 1)?,
            third: Conv2::load(source, &format!("{prefix}.4"), 128, 128, 3, 1)?,
        })
    }

    fn forward(&self, pool: &ComputePool, input: &Image, past: &Image) -> Result<Image, String> {
        if input.shape != past.shape || input.shape[0] != 128 {
            return Err("Invalid DreamX Flash memory shape".into());
        }
        let mut joined = input.data.clone();
        joined.extend_from_slice(&past.data);
        let mut hidden = self.first.forward(
            pool,
            &Image::new(joined, [256, input.shape[1], input.shape[2]])?,
        )?;
        relu_inplace(&mut hidden.data);
        hidden = self.second.forward(pool, &hidden)?;
        relu_inplace(&mut hidden.data);
        hidden = self.third.forward(pool, &hidden)?;
        for (value, residual) in hidden.data.iter_mut().zip(&input.data) {
            *value = (*value + residual).max(0.0);
        }
        Ok(hidden)
    }
}

pub struct FlashUpsampler {
    pool: Arc<ComputePool>,
    pre_shuffle: Conv2,
    blocks: Vec<FlashBlock>,
    final_projection: Conv2,
}

impl FlashUpsampler {
    fn load(source: Arc<dyn TensorSource>, pool: Arc<ComputePool>) -> Result<Self, String> {
        let pre_shuffle = Conv2::load(
            source.as_ref(),
            &format!("{FLASH_PREFIX}.pre_shuffle.0"),
            48,
            512,
            3,
            1,
        )?;
        let blocks = (0..8)
            .map(|index| FlashBlock::load(source.as_ref(), index))
            .collect::<Result<Vec<_>, _>>()?;
        let final_projection = Conv2::load(
            source.as_ref(),
            &format!("{FLASH_PREFIX}.final"),
            128,
            48,
            3,
            1,
        )?;
        Ok(Self {
            pool,
            pre_shuffle,
            blocks,
            final_projection,
        })
    }

    fn upsample(&self, latent: &VideoLatent) -> Result<VideoLatent, String> {
        let feature = Feature::new(latent.as_slice().to_vec(), latent.shape())?;
        let mut caches: Vec<Option<Image>> = vec![None; self.blocks.len()];
        let mut output = Vec::with_capacity(feature.shape[1]);
        for frame in 0..feature.shape[1] {
            let projected = self
                .pre_shuffle
                .forward(&self.pool, &feature.frame(frame)?)?;
            let mut hidden = pixel_shuffle_2x(projected, 128)?;
            relu_inplace(&mut hidden.data);
            for (index, block) in self.blocks.iter().enumerate() {
                let past = caches[index].as_ref().unwrap_or(&hidden);
                let next = block.forward(&self.pool, &hidden, past)?;
                caches[index] = Some(hidden);
                hidden = next;
            }
            output.push(self.final_projection.forward(&self.pool, &hidden)?);
        }
        let output = Feature::from_frames(output)?;
        VideoLatent::new(output.data, output.shape)
    }
}

struct GroupNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    groups: usize,
}

impl GroupNorm {
    fn load(
        source: &dyn TensorSource,
        name: &str,
        channels: usize,
        groups: usize,
    ) -> Result<Self, String> {
        if channels % groups != 0 {
            return Err("Invalid DreamX GroupNorm groups".into());
        }
        Ok(Self {
            weight: load(source, &format!("{name}.weight"), &[channels])?,
            bias: load(source, &format!("{name}.bias"), &[channels])?,
            groups,
        })
    }

    fn forward_image(&self, input: &Image) -> Result<Image, String> {
        let [channels, height, width] = input.shape;
        if channels != self.weight.len() {
            return Err("Invalid DreamX GroupNorm channels".into());
        }
        let group_channels = channels / self.groups;
        let plane = height * width;
        let count = group_channels * plane;
        let mut output = input.data.clone();
        for group in 0..self.groups {
            let start = group * group_channels * plane;
            let end = start + count;
            let mean = input.data[start..end]
                .iter()
                .map(|&value| value as f64)
                .sum::<f64>()
                / count as f64;
            let variance = input.data[start..end]
                .iter()
                .map(|&value| {
                    let delta = value as f64 - mean;
                    delta * delta
                })
                .sum::<f64>()
                / count as f64;
            let inverse = (variance + 1e-5).sqrt().recip() as f32;
            for channel in group * group_channels..(group + 1) * group_channels {
                for offset in 0..plane {
                    let index = channel * plane + offset;
                    output[index] =
                        (input.data[index] - mean as f32) * inverse * self.weight[channel]
                            + self.bias[channel];
                }
            }
        }
        Image::new(output, input.shape)
    }

    fn forward_feature(&self, input: &Feature) -> Result<Feature, String> {
        let frames = (0..input.shape[1])
            .map(|frame| self.forward_image(&input.frame(frame)?))
            .collect::<Result<Vec<_>, String>>()?;
        Feature::from_frames(frames)
    }
}

struct CausalBlock {
    input_norm: GroupNorm,
    input_conv: Conv2,
    modulation: Dense,
    output_norm: GroupNorm,
    output_conv: Conv2,
}

impl CausalBlock {
    fn load(source: &dyn TensorSource, phase: &str, index: usize) -> Result<Self, String> {
        let prefix = format!("{CAUSAL_PREFIX}.{phase}_blocks.{index}");
        Ok(Self {
            input_norm: GroupNorm::load(source, &format!("{prefix}.in_layers.0"), 512, 32)?,
            input_conv: Conv2::load(source, &format!("{prefix}.in_layers.2"), 512, 512, 3, 1)?,
            modulation: Dense::load(source, &format!("{prefix}.emb_layers.1"), 64, 1024)?,
            output_norm: GroupNorm::load(source, &format!("{prefix}.out_norm"), 512, 32)?,
            output_conv: Conv2::load(source, &format!("{prefix}.out_layers.2"), 512, 512, 3, 1)?,
        })
    }

    fn forward(
        &self,
        pool: &ComputePool,
        input: &Feature,
        embedding: &[f32],
    ) -> Result<Feature, String> {
        let mut conditioned = embedding.to_vec();
        silu_inplace(&mut conditioned);
        let modulation = self.modulation.forward(&conditioned)?;
        let (scale, shift) = modulation.split_at(512);
        let mut frames = Vec::with_capacity(input.shape[1]);
        for frame in 0..input.shape[1] {
            let residual = input.frame(frame)?;
            let mut hidden = self.input_norm.forward_image(&residual)?;
            silu_inplace(&mut hidden.data);
            hidden = self.input_conv.forward(pool, &hidden)?;
            hidden = self.output_norm.forward_image(&hidden)?;
            let plane = hidden.shape[1] * hidden.shape[2];
            for channel in 0..512 {
                for value in &mut hidden.data[channel * plane..(channel + 1) * plane] {
                    *value = *value * (1.0 + scale[channel]) + shift[channel];
                }
            }
            silu_inplace(&mut hidden.data);
            hidden = self.output_conv.forward(pool, &hidden)?;
            for (value, residual) in hidden.data.iter_mut().zip(&residual.data) {
                *value += residual;
            }
            frames.push(hidden);
        }
        Feature::from_frames(frames)
    }
}

struct TemporalConv {
    norm: GroupNorm,
    depthwise_weight: Vec<f32>,
    depthwise_bias: Vec<f32>,
    pointwise_weight: Vec<f32>,
    pointwise_bias: Vec<f32>,
}

impl TemporalConv {
    fn load(source: &dyn TensorSource, phase: &str) -> Result<Self, String> {
        let prefix = format!("{CAUSAL_PREFIX}.temporal_{phase}");
        Ok(Self {
            norm: GroupNorm::load(source, &format!("{prefix}.norm"), 512, 32)?,
            depthwise_weight: load(
                source,
                &format!("{prefix}.dwconv.weight"),
                &[512, 1, 3, 1, 1],
            )?,
            depthwise_bias: load(source, &format!("{prefix}.dwconv.bias"), &[512])?,
            pointwise_weight: load(
                source,
                &format!("{prefix}.pwconv.weight"),
                &[512, 512, 1, 1, 1],
            )?,
            pointwise_bias: load(source, &format!("{prefix}.pwconv.bias"), &[512])?,
        })
    }

    fn forward(&self, pool: &ComputePool, input: &Feature) -> Result<Feature, String> {
        let mut normalized = self.norm.forward_feature(input)?;
        silu_inplace(&mut normalized.data);
        let [channels, frames, height, width] = normalized.shape;
        let plane = height * width;
        let mut padded = vec![0.0; channels * (frames + 2) * plane];
        for channel in 0..channels {
            padded[(channel * (frames + 2) + 2) * plane..(channel + 1) * (frames + 2) * plane]
                .copy_from_slice(
                    &normalized.data[channel * frames * plane..(channel + 1) * frames * plane],
                );
        }
        let (depthwise, shape) = conv3d_with_options(
            pool,
            &padded,
            [512, frames + 2, height, width],
            &self.depthwise_weight,
            [512, 1, 3, 1, 1],
            Some(&self.depthwise_bias),
            Conv3dSpec {
                groups: 512,
                ..Conv3dSpec::default()
            },
        )?;
        let (mut output, shape) = conv3d_with_options(
            pool,
            &depthwise,
            shape,
            &self.pointwise_weight,
            [512, 512, 1, 1, 1],
            Some(&self.pointwise_bias),
            Conv3dSpec::default(),
        )?;
        if shape != input.shape {
            return Err("Invalid DreamX causal temporal output shape".into());
        }
        for (value, residual) in output.iter_mut().zip(&input.data) {
            *value += residual;
        }
        Feature::new(output, shape)
    }
}

pub struct Causal2dUpsampler {
    pool: Arc<ComputePool>,
    input: Conv2,
    embed_input: Dense,
    embed_output: Dense,
    input_blocks: Vec<CausalBlock>,
    output_blocks: Vec<CausalBlock>,
    temporal_input: TemporalConv,
    temporal_output: TemporalConv,
    output_norm: GroupNorm,
    output: Conv2,
}

impl Causal2dUpsampler {
    fn load(source: Arc<dyn TensorSource>, pool: Arc<ComputePool>) -> Result<Self, String> {
        let input = Conv2::load(
            source.as_ref(),
            &format!("{CAUSAL_PREFIX}.conv_in"),
            48,
            512,
            3,
            1,
        )?;
        let embed_input = Dense::load(source.as_ref(), &format!("{CAUSAL_PREFIX}.embed.0"), 1, 64)?;
        let embed_output =
            Dense::load(source.as_ref(), &format!("{CAUSAL_PREFIX}.embed.2"), 64, 64)?;
        let input_blocks = (0..12)
            .map(|index| CausalBlock::load(source.as_ref(), "in", index))
            .collect::<Result<Vec<_>, _>>()?;
        let output_blocks = (0..12)
            .map(|index| CausalBlock::load(source.as_ref(), "out", index))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            pool,
            input,
            embed_input,
            embed_output,
            input_blocks,
            output_blocks,
            temporal_input: TemporalConv::load(source.as_ref(), "in")?,
            temporal_output: TemporalConv::load(source.as_ref(), "out")?,
            output_norm: GroupNorm::load(
                source.as_ref(),
                &format!("{CAUSAL_PREFIX}.norm_out"),
                512,
                32,
            )?,
            output: Conv2::load(
                source.as_ref(),
                &format!("{CAUSAL_PREFIX}.conv_out"),
                512,
                48,
                3,
                1,
            )?,
        })
    }

    fn upsample(&self, latent: &VideoLatent) -> Result<VideoLatent, String> {
        let input = Feature::new(latent.as_slice().to_vec(), latent.shape())?;
        let mut embedding = self.embed_input.forward(&[1.0])?;
        silu_inplace(&mut embedding);
        embedding = self.embed_output.forward(&embedding)?;
        let mut hidden = conv_frames(&self.input, &self.pool, &input)?;
        hidden = self.run_phase(&self.input_blocks, &self.temporal_input, hidden, &embedding)?;
        hidden = bilinear_feature(&hidden, 2)?;
        hidden = self.run_phase(
            &self.output_blocks,
            &self.temporal_output,
            hidden,
            &embedding,
        )?;
        hidden = self.output_norm.forward_feature(&hidden)?;
        silu_inplace(&mut hidden.data);
        hidden = conv_frames(&self.output, &self.pool, &hidden)?;
        let base = bilinear_feature(&input, 2)?;
        for (value, residual) in hidden.data.iter_mut().zip(base.data) {
            *value += residual;
        }
        VideoLatent::new(hidden.data, hidden.shape)
    }

    fn run_phase(
        &self,
        blocks: &[CausalBlock],
        temporal: &TemporalConv,
        mut hidden: Feature,
        embedding: &[f32],
    ) -> Result<Feature, String> {
        for (index, block) in blocks.iter().enumerate() {
            hidden = block.forward(&self.pool, &hidden, embedding)?;
            if index % 2 == 0 {
                hidden = temporal.forward(&self.pool, &hidden)?;
            }
        }
        Ok(hidden)
    }
}

fn conv_frames(conv: &Conv2, pool: &ComputePool, input: &Feature) -> Result<Feature, String> {
    Feature::from_frames(
        (0..input.shape[1])
            .map(|frame| conv.forward(pool, &input.frame(frame)?))
            .collect::<Result<Vec<_>, String>>()?,
    )
}

fn pixel_shuffle_2x(input: Image, output_channels: usize) -> Result<Image, String> {
    let [channels, height, width] = input.shape;
    if channels != output_channels * 4 {
        return Err("Invalid DreamX PixelShuffle channels".into());
    }
    let mut output = vec![0.0; output_channels * height * 2 * width * 2];
    for channel in 0..output_channels {
        for y in 0..height {
            for x in 0..width {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let source_channel = channel * 4 + dy * 2 + dx;
                        let source = (source_channel * height + y) * width + x;
                        let target = (channel * height * 2 + y * 2 + dy) * width * 2 + x * 2 + dx;
                        output[target] = input.data[source];
                    }
                }
            }
        }
    }
    Image::new(output, [output_channels, height * 2, width * 2])
}

fn bilinear_2x(input: &VideoLatent) -> Result<VideoLatent, String> {
    let feature = Feature::new(input.as_slice().to_vec(), input.shape())?;
    let output = bilinear_feature(&feature, 2)?;
    VideoLatent::new(output.data, output.shape)
}

fn bilinear_feature(input: &Feature, scale: usize) -> Result<Feature, String> {
    let [channels, frames, height, width] = input.shape;
    if frames == 0 || height == 0 || width == 0 || scale == 0 {
        return Err("DreamX upsampler dimensions must be nonzero".into());
    }
    let output_height = height
        .checked_mul(scale)
        .ok_or("DreamX upsample height overflow")?;
    let output_width = width
        .checked_mul(scale)
        .ok_or("DreamX upsample width overflow")?;
    let mut output = vec![0.0; checked_product(&[channels, frames, output_height, output_width])?];
    let input_plane = height * width;
    let output_plane = output_height * output_width;
    for channel in 0..channels {
        for frame in 0..frames {
            for y in 0..output_height {
                let source_y = (y as f32 + 0.5) / scale as f32 - 0.5;
                let y0 = source_y.floor().max(0.0) as usize;
                let y1 = (y0 + 1).min(height - 1);
                let wy = source_y.clamp(0.0, (height - 1) as f32) - y0 as f32;
                for x in 0..output_width {
                    let source_x = (x as f32 + 0.5) / scale as f32 - 0.5;
                    let x0 = source_x.floor().max(0.0) as usize;
                    let x1 = (x0 + 1).min(width - 1);
                    let wx = source_x.clamp(0.0, (width - 1) as f32) - x0 as f32;
                    let base = (channel * frames + frame) * input_plane;
                    let top = input.data[base + y0 * width + x0] * (1.0 - wx)
                        + input.data[base + y0 * width + x1] * wx;
                    let bottom = input.data[base + y1 * width + x0] * (1.0 - wx)
                        + input.data[base + y1 * width + x1] * wx;
                    output[(channel * frames + frame) * output_plane + y * output_width + x] =
                        top * (1.0 - wy) + bottom * wy;
                }
            }
        }
    }
    Feature::new(output, [channels, frames, output_height, output_width])
}

fn relu_inplace(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

fn load(source: &dyn TensorSource, name: &str, source_shape: &[usize]) -> Result<Vec<f32>, String> {
    let expected: Vec<u64> = source_shape
        .iter()
        .rev()
        .map(|&value| value as u64)
        .collect();
    load_float_values(source, name, &expected)
}

fn checked_product(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |length, &dimension| {
        length
            .checked_mul(dimension)
            .ok_or_else(|| "DreamX upsampler shape overflow".into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{MetaValue, TensorInfo, TensorSource};
    use crate::core::thread_pool::ComputePool;
    use std::sync::Arc;

    struct EmptySource;

    impl TensorSource for EmptySource {
        fn metadata(&self, _key: &str) -> Option<&MetaValue> {
            None
        }

        fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
            None
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    #[test]
    fn unavailable_selected_upsampler_does_not_fallback() {
        let error = LatentUpsampler::load(
            LatentUpsampleKind::Flash,
            Arc::new(EmptySource),
            Arc::new(ComputePool::new(1)),
        )
        .unwrap_err();
        assert!(error.contains("dreamx.refiner.upsampler.flash"));
    }

    #[test]
    fn every_upsampler_doubles_spatial_shape() {
        let latent = VideoLatent::new(vec![0.0; 48 * 3 * 4 * 6], [48, 3, 4, 6]).unwrap();
        for kind in [
            LatentUpsampleKind::Bilinear,
            LatentUpsampleKind::Flash,
            LatentUpsampleKind::Causal2d,
        ] {
            let upsampler = LatentUpsampler::testing(kind);
            assert_eq!(upsampler.upsample(&latent).unwrap().shape(), [48, 3, 8, 12]);
        }
    }
}
