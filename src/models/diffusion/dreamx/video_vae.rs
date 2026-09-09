use std::sync::Arc;

use image::{Rgb, RgbImage};

use super::kernels::{
    attention_online, checked_len, conv2d, conv3d_with_options, AttentionSpec, Conv3dSpec,
};
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::silu_inplace;

const PREFIX: &str = "dreamx.video_vae";
const LATENT_CHANNELS: usize = 48;
const PATCH_SIZE: usize = 2;
const SPATIAL_FACTOR: usize = 16;
const CACHE_FRAMES: usize = 2;
const TILE_SIZE: [usize; 2] = [34, 34];
const TILE_STRIDE: [usize; 2] = [18, 16];

const LATENT_MEAN: [f32; LATENT_CHANNELS] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813,
    0.0891, 0.1570, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108,
    -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230, -0.0313,
    -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944,
    0.3573, -0.0548, -0.1681, -0.0667,
];

const LATENT_STD: [f32; LATENT_CHANNELS] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

#[derive(Clone, Debug)]
pub struct VideoLatent {
    values: Vec<f32>,
    shape: [usize; 4],
}

impl VideoLatent {
    pub fn new(values: Vec<f32>, shape: [usize; 4]) -> Result<Self, String> {
        if shape[0] != LATENT_CHANNELS
            || values.len() != checked_len("DreamX video latent", &shape)?
            || values.iter().any(|value| !value.is_finite())
        {
            return Err("Invalid DreamX video latent".into());
        }
        Ok(Self { values, shape })
    }

    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

pub fn normalize_latent(raw: &[f32]) -> Result<Vec<f32>, String> {
    map_latent(raw, |value, channel| {
        (value - LATENT_MEAN[channel]) / LATENT_STD[channel]
    })
}

pub fn denormalize_latent(normalized: &[f32]) -> Result<Vec<f32>, String> {
    map_latent(normalized, |value, channel| {
        value * LATENT_STD[channel] + LATENT_MEAN[channel]
    })
}

fn map_latent(values: &[f32], map: impl Fn(f32, usize) -> f32) -> Result<Vec<f32>, String> {
    if values.is_empty()
        || !values.len().is_multiple_of(LATENT_CHANNELS)
        || values.iter().any(|value| !value.is_finite())
    {
        return Err("Invalid DreamX latent normalization input".into());
    }
    let plane = values.len() / LATENT_CHANNELS;
    let mut output = Vec::with_capacity(values.len());
    for channel in 0..LATENT_CHANNELS {
        output.extend(
            values[channel * plane..(channel + 1) * plane]
                .iter()
                .map(|&value| map(value, channel)),
        );
    }
    Ok(output)
}

#[derive(Clone)]
struct Feature {
    data: Vec<f32>,
    shape: [usize; 4],
}

impl Feature {
    fn new(data: Vec<f32>, shape: [usize; 4]) -> Result<Self, String> {
        if data.len() != checked_len("DreamX VAE feature", &shape)? {
            return Err("Invalid DreamX VAE feature length".into());
        }
        Ok(Self { data, shape })
    }

    fn zeros(shape: [usize; 4]) -> Result<Self, String> {
        Ok(Self {
            data: vec![0.0; checked_len("DreamX VAE feature", &shape)?],
            shape,
        })
    }
}

enum CacheEntry {
    Data(Feature),
    Repeat,
}

#[derive(Default)]
struct CausalCache {
    slots: Vec<Option<CacheEntry>>,
    cursor: usize,
}

impl CausalCache {
    fn begin_chunk(&mut self) {
        self.cursor = 0;
    }

    fn take(&mut self) -> (usize, Option<CacheEntry>) {
        let index = self.cursor;
        self.cursor += 1;
        if index == self.slots.len() {
            self.slots.push(None);
        }
        (index, self.slots[index].take())
    }

    fn put(&mut self, index: usize, entry: CacheEntry) {
        self.slots[index] = Some(entry);
    }

    fn finish_chunk(&self) -> Result<(), String> {
        if self.cursor != self.slots.len() {
            return Err("DreamX VAE causal cache graph changed between chunks".into());
        }
        Ok(())
    }
}

struct Conv3 {
    weight: String,
    weight_shape: [usize; 5],
    bias: Vec<f32>,
    stride: [usize; 3],
    padding: [usize; 3],
}

impl Conv3 {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
    ) -> Result<Self, String> {
        let weight = format!("{PREFIX}.{prefix}.weight");
        let weight_shape = [
            output_channels,
            input_channels,
            kernel[0],
            kernel[1],
            kernel[2],
        ];
        validate_f32_tensor(source, &weight, &weight_shape)?;
        let bias = load_f32_values(
            source,
            &format!("{PREFIX}.{prefix}.bias"),
            &[output_channels],
        )?;
        Ok(Self {
            weight,
            weight_shape,
            bias,
            stride,
            padding,
        })
    }

    fn raw(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        input: &Feature,
    ) -> Result<Feature, String> {
        let weight = f32_tensor(source, &self.weight)?;
        let (data, shape) = conv3d_with_options(
            pool,
            &input.data,
            input.shape,
            weight,
            self.weight_shape,
            Some(&self.bias),
            Conv3dSpec {
                stride: self.stride,
                padding: [0, self.padding[1], self.padding[2]],
                dilation: [1; 3],
                groups: 1,
            },
        )?;
        Feature::new(data, shape)
    }

    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        input: &Feature,
        cache: Option<&mut CausalCache>,
    ) -> Result<Feature, String> {
        let front = self.padding[0] * 2;
        match cache {
            Some(cache) => {
                let (slot, previous) = cache.take();
                let previous = match previous {
                    Some(CacheEntry::Data(feature)) => Some(feature),
                    Some(CacheEntry::Repeat) => {
                        return Err("Invalid DreamX VAE causal cache entry".into())
                    }
                    None => None,
                };
                let padded = prepend_context(previous.as_ref(), input, front)?;
                let next = tail_joined(previous.as_ref(), input, CACHE_FRAMES)?;
                let output = self.raw(source, pool, &padded)?;
                cache.put(slot, CacheEntry::Data(next));
                Ok(output)
            }
            None => self.raw(source, pool, &prepend_context(None, input, front)?),
        }
    }
}

struct Conv2 {
    weight: String,
    weight_shape: [usize; 4],
    bias: Vec<f32>,
    stride: [usize; 2],
    padding: [usize; 2],
}

impl Conv2 {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Self, String> {
        let weight = format!("{PREFIX}.{prefix}.weight");
        let weight_shape = [output_channels, input_channels, kernel[0], kernel[1]];
        validate_f32_tensor(source, &weight, &weight_shape)?;
        let bias = load_f32_values(
            source,
            &format!("{PREFIX}.{prefix}.bias"),
            &[output_channels],
        )?;
        Ok(Self {
            weight,
            weight_shape,
            bias,
            stride,
            padding,
        })
    }

    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        input: &Feature,
    ) -> Result<Feature, String> {
        let [channels, depth, height, width] = input.shape;
        if channels != self.weight_shape[1] {
            return Err("Invalid DreamX Conv2D input channels".into());
        }
        let weight = f32_tensor(source, &self.weight)?;
        let mut output = Vec::new();
        let mut output_shape = None;
        let input_plane = height * width;
        for frame in 0..depth {
            let mut frame_data = vec![0.0; channels * input_plane];
            for channel in 0..channels {
                let source_start = (channel * depth + frame) * input_plane;
                frame_data[channel * input_plane..(channel + 1) * input_plane]
                    .copy_from_slice(&input.data[source_start..source_start + input_plane]);
            }
            let (frame_output, shape) = conv2d(
                pool,
                &frame_data,
                [channels, height, width],
                weight,
                self.weight_shape,
                Some(&self.bias),
                self.stride,
                self.padding,
                [1; 2],
                1,
            )?;
            if output_shape.replace(shape).is_some_and(|old| old != shape) {
                return Err("DreamX Conv2D output shape changed across frames".into());
            }
            output.push(frame_output);
        }
        let [output_channels, output_height, output_width] =
            output_shape.ok_or("DreamX Conv2D requires frames")?;
        let output_plane = output_height * output_width;
        let mut data = vec![0.0; output_channels * depth * output_plane];
        for (frame, frame_output) in output.into_iter().enumerate() {
            for channel in 0..output_channels {
                let target = (channel * depth + frame) * output_plane;
                data[target..target + output_plane].copy_from_slice(
                    &frame_output[channel * output_plane..(channel + 1) * output_plane],
                );
            }
        }
        Feature::new(data, [output_channels, depth, output_height, output_width])
    }
}

struct ResidualBlock {
    norm1: Vec<f32>,
    conv1: Conv3,
    norm2: Vec<f32>,
    conv2: Conv3,
    shortcut: Option<Conv3>,
}

impl ResidualBlock {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            norm1: load_f32_values(
                source,
                &format!("{PREFIX}.{prefix}.residual.0.gamma"),
                &[input_channels, 1, 1, 1],
            )?,
            conv1: Conv3::load(
                source,
                &format!("{prefix}.residual.2"),
                input_channels,
                output_channels,
                [3; 3],
                [1; 3],
                [1; 3],
            )?,
            norm2: load_f32_values(
                source,
                &format!("{PREFIX}.{prefix}.residual.3.gamma"),
                &[output_channels, 1, 1, 1],
            )?,
            conv2: Conv3::load(
                source,
                &format!("{prefix}.residual.6"),
                output_channels,
                output_channels,
                [3; 3],
                [1; 3],
                [1; 3],
            )?,
            shortcut: (input_channels != output_channels)
                .then(|| {
                    Conv3::load(
                        source,
                        &format!("{prefix}.shortcut"),
                        input_channels,
                        output_channels,
                        [1; 3],
                        [1; 3],
                        [0; 3],
                    )
                })
                .transpose()?,
        })
    }

    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        input: &Feature,
        cache: &mut CausalCache,
    ) -> Result<Feature, String> {
        let shortcut = match &self.shortcut {
            Some(shortcut) => shortcut.forward(source, pool, input, None)?,
            None => input.clone(),
        };
        let mut data = channel_rms_norm(input, &self.norm1)?;
        silu_inplace(&mut data.data);
        data = self.conv1.forward(source, pool, &data, Some(cache))?;
        data = channel_rms_norm(&data, &self.norm2)?;
        silu_inplace(&mut data.data);
        data = self.conv2.forward(source, pool, &data, Some(cache))?;
        add_features(data, &shortcut)
    }
}

struct AttentionBlock {
    norm: Vec<f32>,
    qkv: Conv2,
    projection: Conv2,
}

impl AttentionBlock {
    fn load(source: &dyn TensorSource, prefix: &str, channels: usize) -> Result<Self, String> {
        Ok(Self {
            norm: load_f32_values(
                source,
                &format!("{PREFIX}.{prefix}.norm.gamma"),
                &[channels, 1, 1],
            )?,
            qkv: Conv2::load(
                source,
                &format!("{prefix}.to_qkv"),
                channels,
                channels * 3,
                [1; 2],
                [1; 2],
                [0; 2],
            )?,
            projection: Conv2::load(
                source,
                &format!("{prefix}.proj"),
                channels,
                channels,
                [1; 2],
                [1; 2],
                [0; 2],
            )?,
        })
    }

    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        input: &Feature,
    ) -> Result<Feature, String> {
        let normalized = channel_rms_norm(input, &self.norm)?;
        let qkv = self.qkv.forward(source, pool, &normalized)?;
        let [channels, depth, height, width] = input.shape;
        let spatial = height * width;
        let qkv_plane = depth * spatial;
        let mut attended = vec![0.0; input.data.len()];
        for frame in 0..depth {
            let mut query = vec![0.0; spatial * channels];
            let mut key = vec![0.0; spatial * channels];
            let mut value = vec![0.0; spatial * channels];
            for token in 0..spatial {
                for channel in 0..channels {
                    query[token * channels + channel] =
                        qkv.data[channel * qkv_plane + frame * spatial + token];
                    key[token * channels + channel] =
                        qkv.data[(channels + channel) * qkv_plane + frame * spatial + token];
                    value[token * channels + channel] =
                        qkv.data[(channels * 2 + channel) * qkv_plane + frame * spatial + token];
                }
            }
            let output = attention_online(
                &query,
                &key,
                &value,
                AttentionSpec {
                    query_tokens: spatial,
                    key_tokens: spatial,
                    query_heads: 1,
                    key_value_heads: 1,
                    head_dim: channels,
                    causal: false,
                    scale: 1.0 / (channels as f32).sqrt(),
                },
            )?;
            for token in 0..spatial {
                for channel in 0..channels {
                    attended[(channel * depth + frame) * spatial + token] =
                        output[token * channels + channel];
                }
            }
        }
        let projected =
            self.projection
                .forward(source, pool, &Feature::new(attended, input.shape)?)?;
        add_features(projected, input)
    }
}

struct DownBlock {
    residuals: Vec<ResidualBlock>,
    resample: Option<Downsample>,
    input_channels: usize,
    output_channels: usize,
    temporal: bool,
    spatial: bool,
}

struct Downsample {
    spatial: Conv2,
    temporal: Option<Conv3>,
}

impl DownBlock {
    fn load(
        source: &dyn TensorSource,
        index: usize,
        input_channels: usize,
        output_channels: usize,
        residual_count: usize,
        temporal: bool,
        spatial: bool,
    ) -> Result<Self, String> {
        let prefix = format!("encoder.downsamples.{index}.downsamples");
        let mut residuals = Vec::with_capacity(residual_count);
        for block in 0..residual_count {
            residuals.push(ResidualBlock::load(
                source,
                &format!("{prefix}.{block}"),
                if block == 0 {
                    input_channels
                } else {
                    output_channels
                },
                output_channels,
            )?);
        }
        let resample = spatial
            .then(|| {
                let layer = residual_count;
                Ok::<_, String>(Downsample {
                    spatial: Conv2::load(
                        source,
                        &format!("{prefix}.{layer}.resample.1"),
                        output_channels,
                        output_channels,
                        [3; 2],
                        [2; 2],
                        [0; 2],
                    )?,
                    temporal: temporal
                        .then(|| {
                            Conv3::load(
                                source,
                                &format!("{prefix}.{layer}.time_conv"),
                                output_channels,
                                output_channels,
                                [3, 1, 1],
                                [2, 1, 1],
                                [0; 3],
                            )
                        })
                        .transpose()?,
                })
            })
            .transpose()?;
        Ok(Self {
            residuals,
            resample,
            input_channels,
            output_channels,
            temporal,
            spatial,
        })
    }

    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        input: &Feature,
        cache: &mut CausalCache,
    ) -> Result<Feature, String> {
        let shortcut = avg_down3d(
            input,
            self.input_channels,
            self.output_channels,
            if self.temporal { 2 } else { 1 },
            if self.spatial { 2 } else { 1 },
        )?;
        let mut main = input.clone();
        for block in &self.residuals {
            main = block.forward(source, pool, &main, cache)?;
        }
        if let Some(resample) = &self.resample {
            main = resample.forward(source, pool, &main, cache)?;
        }
        add_features(main, &shortcut)
    }
}

impl Downsample {
    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        input: &Feature,
        cache: &mut CausalCache,
    ) -> Result<Feature, String> {
        let padded = pad_right_bottom(input)?;
        let spatial = self.spatial.forward(source, pool, &padded)?;
        let Some(temporal) = &self.temporal else {
            return Ok(spatial);
        };
        let (slot, previous) = cache.take();
        match previous {
            None => {
                cache.put(slot, CacheEntry::Data(tail_joined(None, &spatial, 1)?));
                Ok(spatial)
            }
            Some(CacheEntry::Data(previous)) => {
                let joined = prepend_context(Some(&previous), &spatial, 1)?;
                let output = temporal.raw(source, pool, &joined)?;
                cache.put(slot, CacheEntry::Data(tail_joined(None, &spatial, 1)?));
                Ok(output)
            }
            Some(CacheEntry::Repeat) => Err("Invalid DreamX downsample cache".into()),
        }
    }
}

struct UpBlock {
    residuals: Vec<ResidualBlock>,
    resample: Option<Upsample>,
    input_channels: usize,
    output_channels: usize,
    temporal: bool,
}

struct Upsample {
    spatial: Conv2,
    temporal: Option<Conv3>,
}

impl UpBlock {
    fn load(
        source: &dyn TensorSource,
        index: usize,
        input_channels: usize,
        output_channels: usize,
        residual_count: usize,
        temporal: bool,
        spatial: bool,
    ) -> Result<Self, String> {
        let prefix = format!("decoder.upsamples.{index}.upsamples");
        let mut residuals = Vec::with_capacity(residual_count);
        for block in 0..residual_count {
            residuals.push(ResidualBlock::load(
                source,
                &format!("{prefix}.{block}"),
                if block == 0 {
                    input_channels
                } else {
                    output_channels
                },
                output_channels,
            )?);
        }
        let resample = spatial
            .then(|| {
                let layer = residual_count;
                Ok::<_, String>(Upsample {
                    spatial: Conv2::load(
                        source,
                        &format!("{prefix}.{layer}.resample.1"),
                        output_channels,
                        output_channels,
                        [3; 2],
                        [1; 2],
                        [1; 2],
                    )?,
                    temporal: temporal
                        .then(|| {
                            Conv3::load(
                                source,
                                &format!("{prefix}.{layer}.time_conv"),
                                output_channels,
                                output_channels * 2,
                                [3, 1, 1],
                                [1; 3],
                                [1, 0, 0],
                            )
                        })
                        .transpose()?,
                })
            })
            .transpose()?;
        Ok(Self {
            residuals,
            resample,
            input_channels,
            output_channels,
            temporal,
        })
    }

    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        input: &Feature,
        cache: &mut CausalCache,
        first_chunk: bool,
    ) -> Result<Feature, String> {
        let shortcut = self
            .resample
            .as_ref()
            .map(|_| {
                dup_up3d(
                    input,
                    self.input_channels,
                    self.output_channels,
                    if self.temporal { 2 } else { 1 },
                    2,
                    first_chunk,
                )
            })
            .transpose()?;
        let mut main = input.clone();
        for block in &self.residuals {
            main = block.forward(source, pool, &main, cache)?;
        }
        if let Some(resample) = &self.resample {
            main = resample.forward(source, pool, &main, cache)?;
        }
        match shortcut {
            Some(shortcut) => add_features(main, &shortcut),
            None => Ok(main),
        }
    }
}

impl Upsample {
    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        input: &Feature,
        cache: &mut CausalCache,
    ) -> Result<Feature, String> {
        let temporal = if let Some(time_conv) = &self.temporal {
            let (slot, previous) = cache.take();
            match previous {
                None => {
                    cache.put(slot, CacheEntry::Repeat);
                    input.clone()
                }
                Some(CacheEntry::Repeat) => {
                    let output = time_conv.forward(source, pool, input, None)?;
                    cache.put(
                        slot,
                        CacheEntry::Data(cache_tail_with_zeros(input, CACHE_FRAMES)?),
                    );
                    interleave_time_channels(output)?
                }
                Some(CacheEntry::Data(previous)) => {
                    let padded = prepend_context(Some(&previous), input, time_conv.padding[0] * 2)?;
                    let output = time_conv.raw(source, pool, &padded)?;
                    cache.put(
                        slot,
                        CacheEntry::Data(tail_joined(Some(&previous), input, CACHE_FRAMES)?),
                    );
                    interleave_time_channels(output)?
                }
            }
        } else {
            input.clone()
        };
        self.spatial
            .forward(source, pool, &nearest_spatial(&temporal, 2)?)
    }
}

struct Encoder {
    input: Conv3,
    stages: Vec<DownBlock>,
    middle1: ResidualBlock,
    attention: AttentionBlock,
    middle2: ResidualBlock,
    norm: Vec<f32>,
    output: Conv3,
}

impl Encoder {
    fn load(source: &dyn TensorSource, dimensions: VaeDimensions) -> Result<Self, String> {
        let dims = dimensions.encoder_dims();
        let input = Conv3::load(source, "encoder.conv1", 12, dims[0], [3; 3], [1; 3], [1; 3])?;
        let mut stages = Vec::with_capacity(4);
        for index in 0..4 {
            stages.push(DownBlock::load(
                source,
                index,
                dims[index],
                dims[index + 1],
                dimensions.residual_blocks,
                dimensions
                    .temporal_down
                    .get(index)
                    .copied()
                    .unwrap_or(false),
                index != 3,
            )?);
        }
        let channels = dims[4];
        Ok(Self {
            input,
            stages,
            middle1: ResidualBlock::load(source, "encoder.middle.0", channels, channels)?,
            attention: AttentionBlock::load(source, "encoder.middle.1", channels)?,
            middle2: ResidualBlock::load(source, "encoder.middle.2", channels, channels)?,
            norm: load_f32_values(
                source,
                &format!("{PREFIX}.encoder.head.0.gamma"),
                &[channels, 1, 1, 1],
            )?,
            output: Conv3::load(
                source,
                "encoder.head.2",
                channels,
                dimensions.latent_channels * 2,
                [3; 3],
                [1; 3],
                [1; 3],
            )?,
        })
    }

    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        input: &Feature,
        cache: &mut CausalCache,
    ) -> Result<Feature, String> {
        let mut data = self.input.forward(source, pool, input, Some(cache))?;
        for stage in &self.stages {
            data = stage.forward(source, pool, &data, cache)?;
        }
        data = self.middle1.forward(source, pool, &data, cache)?;
        data = self.attention.forward(source, pool, &data)?;
        data = self.middle2.forward(source, pool, &data, cache)?;
        data = channel_rms_norm(&data, &self.norm)?;
        silu_inplace(&mut data.data);
        self.output.forward(source, pool, &data, Some(cache))
    }
}

struct Decoder {
    input: Conv3,
    middle1: ResidualBlock,
    attention: AttentionBlock,
    middle2: ResidualBlock,
    stages: Vec<UpBlock>,
    norm: Vec<f32>,
    output: Conv3,
}

impl Decoder {
    fn load(source: &dyn TensorSource, dimensions: VaeDimensions) -> Result<Self, String> {
        let dims = dimensions.decoder_dims();
        Self::load_with_dims(
            source,
            dims,
            dimensions.latent_channels,
            dimensions.residual_blocks,
            dimensions.temporal_down,
        )
    }

    fn load_with_dims(
        source: &dyn TensorSource,
        dims: [usize; 5],
        latent_channels: usize,
        residual_blocks: usize,
        temporal_down: [bool; 3],
    ) -> Result<Self, String> {
        let channels = dims[0];
        let mut stages = Vec::with_capacity(4);
        for index in 0..4 {
            stages.push(UpBlock::load(
                source,
                index,
                dims[index],
                dims[index + 1],
                residual_blocks + 1,
                temporal_down
                    .iter()
                    .rev()
                    .copied()
                    .nth(index)
                    .unwrap_or(false),
                index != 3,
            )?);
        }
        Ok(Self {
            input: Conv3::load(
                source,
                "decoder.conv1",
                latent_channels,
                channels,
                [3; 3],
                [1; 3],
                [1; 3],
            )?,
            middle1: ResidualBlock::load(source, "decoder.middle.0", channels, channels)?,
            attention: AttentionBlock::load(source, "decoder.middle.1", channels)?,
            middle2: ResidualBlock::load(source, "decoder.middle.2", channels, channels)?,
            stages,
            norm: load_f32_values(
                source,
                &format!("{PREFIX}.decoder.head.0.gamma"),
                &[dims[4], 1, 1, 1],
            )?,
            output: Conv3::load(
                source,
                "decoder.head.2",
                dims[4],
                12,
                [3; 3],
                [1; 3],
                [1; 3],
            )?,
        })
    }

    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        input: &Feature,
        cache: &mut CausalCache,
        first_chunk: bool,
    ) -> Result<Feature, String> {
        let mut data = self.input.forward(source, pool, input, Some(cache))?;
        data = self.middle1.forward(source, pool, &data, cache)?;
        data = self.attention.forward(source, pool, &data)?;
        data = self.middle2.forward(source, pool, &data, cache)?;
        for stage in &self.stages {
            data = stage.forward(source, pool, &data, cache, first_chunk)?;
        }
        data = channel_rms_norm(&data, &self.norm)?;
        silu_inplace(&mut data.data);
        self.output.forward(source, pool, &data, Some(cache))
    }
}

struct PrefixSource {
    source: Arc<dyn TensorSource>,
    actual_prefix: &'static str,
}

impl PrefixSource {
    fn mapped(&self, name: &str) -> String {
        name.strip_prefix(PREFIX)
            .map(|suffix| format!("{}{suffix}", self.actual_prefix))
            .unwrap_or_else(|| name.to_owned())
    }
}

impl TensorSource for PrefixSource {
    fn metadata(&self, key: &str) -> Option<&crate::core::tensor::MetaValue> {
        self.source.metadata(key)
    }

    fn tensor_info(&self, name: &str) -> Option<&crate::core::tensor::TensorInfo> {
        self.source.tensor_info(&self.mapped(name))
    }

    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.source.tensor_slice(&self.mapped(name))
    }
}

pub(crate) struct LightVaeDecoderCore {
    source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    latent_input: Conv3,
    decoder: Decoder,
}

impl LightVaeDecoderCore {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
        dims: [usize; 5],
    ) -> Result<Self, String> {
        let source: Arc<dyn TensorSource> = Arc::new(PrefixSource {
            source,
            actual_prefix: "dreamx.refiner.lightvae",
        });
        let latent_input = Conv3::load(
            source.as_ref(),
            "conv2",
            LATENT_CHANNELS,
            LATENT_CHANNELS,
            [1; 3],
            [1; 3],
            [0; 3],
        )?;
        let decoder = Decoder::load_with_dims(
            source.as_ref(),
            dims,
            LATENT_CHANNELS,
            2,
            [false, true, true],
        )?;
        Ok(Self {
            source,
            pool,
            latent_input,
            decoder,
        })
    }

    pub(crate) fn decode_frames(&self, latent: &VideoLatent) -> Result<Vec<RgbImage>, String> {
        let [channels, depth, height, width] = latent.shape;
        let raw = Feature::new(
            denormalize_latent(&latent.values)?,
            [channels, depth, height, width],
        )?;
        let input = self
            .latent_input
            .forward(self.source.as_ref(), &self.pool, &raw, None)?;
        let mut cache = CausalCache::default();
        let mut decoded: Option<Feature> = None;
        for frame in 0..depth {
            cache.begin_chunk();
            let output = self.decoder.forward(
                self.source.as_ref(),
                &self.pool,
                &slice_time(&input, frame, frame + 1)?,
                &mut cache,
                frame == 0,
            )?;
            cache.finish_chunk()?;
            decoded = Some(match decoded {
                Some(ref current) => concat_time(current, &output)?,
                None => output,
            });
        }
        feature_to_rgb(&unpatchify(
            &decoded.ok_or("DreamX LightVAE produced no frames")?,
        )?)
    }
}

#[derive(Clone, Copy)]
struct VaeDimensions {
    encoder_dim: usize,
    decoder_dim: usize,
    latent_channels: usize,
    multipliers: [usize; 4],
    residual_blocks: usize,
    temporal_down: [bool; 3],
}

impl VaeDimensions {
    const RELEASED: Self = Self {
        encoder_dim: 160,
        decoder_dim: 256,
        latent_channels: LATENT_CHANNELS,
        multipliers: [1, 2, 4, 4],
        residual_blocks: 2,
        temporal_down: [false, true, true],
    };

    fn encoder_dims(self) -> [usize; 5] {
        [
            self.encoder_dim,
            self.encoder_dim * self.multipliers[0],
            self.encoder_dim * self.multipliers[1],
            self.encoder_dim * self.multipliers[2],
            self.encoder_dim * self.multipliers[3],
        ]
    }

    fn decoder_dims(self) -> [usize; 5] {
        [
            self.decoder_dim * self.multipliers[3],
            self.decoder_dim * self.multipliers[3],
            self.decoder_dim * self.multipliers[2],
            self.decoder_dim * self.multipliers[1],
            self.decoder_dim * self.multipliers[0],
        ]
    }
}

struct VaeGraph {
    encoder: Encoder,
    posterior: Conv3,
    latent_input: Conv3,
    decoder: Decoder,
}

impl VaeGraph {
    fn load(source: &dyn TensorSource) -> Result<Self, String> {
        let dimensions = VaeDimensions::RELEASED;
        Ok(Self {
            encoder: Encoder::load(source, dimensions)?,
            posterior: Conv3::load(
                source,
                "conv1",
                LATENT_CHANNELS * 2,
                LATENT_CHANNELS * 2,
                [1; 3],
                [1; 3],
                [0; 3],
            )?,
            latent_input: Conv3::load(
                source,
                "conv2",
                LATENT_CHANNELS,
                LATENT_CHANNELS,
                [1; 3],
                [1; 3],
                [0; 3],
            )?,
            decoder: Decoder::load(source, dimensions)?,
        })
    }
}

pub struct Wan22Vae {
    source: Option<Arc<dyn TensorSource>>,
    pool: Arc<ComputePool>,
    graph: Option<VaeGraph>,
}

impl Wan22Vae {
    pub fn load(source: Arc<dyn TensorSource>, pool: Arc<ComputePool>) -> Result<Self, String> {
        let graph = VaeGraph::load(source.as_ref())?;
        Ok(Self {
            source: Some(source),
            pool,
            graph: Some(graph),
        })
    }

    pub fn encode_first_frame(&self, frame: &RgbImage) -> Result<VideoLatent, String> {
        self.encode_frames(std::slice::from_ref(frame))
    }

    pub fn encode_frames(&self, frames: &[RgbImage]) -> Result<VideoLatent, String> {
        validate_frames(frames)?;
        #[cfg(test)]
        if self.graph.is_none() {
            let height = frames[0].height() as usize / SPATIAL_FACTOR;
            let width = frames[0].width() as usize / SPATIAL_FACTOR;
            return VideoLatent::new(
                vec![0.0; LATENT_CHANNELS * height * width],
                [LATENT_CHANNELS, 1, height, width],
            );
        }
        let latent_height = frames[0].height() as usize / SPATIAL_FACTOR;
        let latent_width = frames[0].width() as usize / SPATIAL_FACTOR;
        if latent_height > TILE_SIZE[0] || latent_width > TILE_SIZE[1] {
            self.encode_frames_tiled(frames)
        } else {
            self.encode_frames_core(frames)
        }
    }

    fn encode_frames_core(&self, frames: &[RgbImage]) -> Result<VideoLatent, String> {
        validate_frames(frames)?;
        if frames.len() != 1 && !(frames.len() - 1).is_multiple_of(4) {
            return Err("DreamX VAE frame count must be one plus a multiple of four".into());
        }
        let source = self.source()?;
        let graph = self.graph()?;
        let input = patchify_frames(frames)?;
        let mut cache = CausalCache::default();
        let mut encoded: Option<Feature> = None;
        let mut start = 0;
        while start < frames.len() {
            let end = if start == 0 {
                1
            } else {
                (start + 4).min(frames.len())
            };
            cache.begin_chunk();
            let chunk = slice_time(&input, start, end)?;
            let output = graph
                .encoder
                .forward(source, &self.pool, &chunk, &mut cache)?;
            cache.finish_chunk()?;
            encoded = Some(match encoded {
                Some(ref current) => concat_time(current, &output)?,
                None => output,
            });
            start = end;
        }
        let posterior = graph.posterior.forward(
            source,
            &self.pool,
            &encoded.ok_or("DreamX VAE produced no latent")?,
            None,
        )?;
        let [channels, depth, height, width] = posterior.shape;
        if channels != LATENT_CHANNELS * 2 {
            return Err("Invalid DreamX VAE posterior channels".into());
        }
        let plane = depth * height * width;
        let mean = normalize_latent(&posterior.data[..LATENT_CHANNELS * plane])?;
        VideoLatent::new(mean, [LATENT_CHANNELS, depth, height, width])
    }

    pub fn decode_frames(&self, latent: &VideoLatent) -> Result<Vec<RgbImage>, String> {
        let feature = if latent.shape[2] > TILE_SIZE[0] || latent.shape[3] > TILE_SIZE[1] {
            self.decode_frames_tiled(latent)?
        } else {
            self.decode_frames_core(latent)?
        };
        feature_to_rgb(&feature)
    }

    fn decode_frames_core(&self, latent: &VideoLatent) -> Result<Feature, String> {
        let source = self.source()?;
        let graph = self.graph()?;
        let [channels, depth, height, width] = latent.shape;
        if channels != LATENT_CHANNELS {
            return Err("Invalid DreamX VAE latent channels".into());
        }
        let raw = Feature::new(
            denormalize_latent(&latent.values)?,
            [channels, depth, height, width],
        )?;
        let input = graph.latent_input.forward(source, &self.pool, &raw, None)?;
        let mut cache = CausalCache::default();
        let mut decoded: Option<Feature> = None;
        for frame in 0..depth {
            cache.begin_chunk();
            let chunk = slice_time(&input, frame, frame + 1)?;
            let output =
                graph
                    .decoder
                    .forward(source, &self.pool, &chunk, &mut cache, frame == 0)?;
            cache.finish_chunk()?;
            decoded = Some(match decoded {
                Some(ref current) => concat_time(current, &output)?,
                None => output,
            });
        }
        unpatchify(&decoded.ok_or("DreamX VAE produced no frames")?)
    }

    fn encode_frames_tiled(&self, frames: &[RgbImage]) -> Result<VideoLatent, String> {
        let latent_height = frames[0].height() as usize / SPATIAL_FACTOR;
        let latent_width = frames[0].width() as usize / SPATIAL_FACTOR;
        let depth = 1 + (frames.len() - 1) / 4;
        let shape = [LATENT_CHANNELS, depth, latent_height, latent_width];
        let mut values = vec![0.0; checked_len("DreamX tiled latent", &shape)?];
        let mut weights = vec![0.0; latent_height * latent_width];
        for top in tile_starts(latent_height, TILE_SIZE[0], TILE_STRIDE[0])? {
            let tile_height = TILE_SIZE[0].min(latent_height - top);
            for left in tile_starts(latent_width, TILE_SIZE[1], TILE_STRIDE[1])? {
                let tile_width = TILE_SIZE[1].min(latent_width - left);
                let pixel_top = top * SPATIAL_FACTOR;
                let pixel_left = left * SPATIAL_FACTOR;
                let pixel_height = tile_height * SPATIAL_FACTOR;
                let pixel_width = tile_width * SPATIAL_FACTOR;
                let cropped: Vec<RgbImage> = frames
                    .iter()
                    .map(|frame| {
                        image::imageops::crop_imm(
                            frame,
                            pixel_left as u32,
                            pixel_top as u32,
                            pixel_width as u32,
                            pixel_height as u32,
                        )
                        .to_image()
                    })
                    .collect();
                let tile = self.encode_frames_core(&cropped)?;
                blend_tile(
                    tile.as_slice(),
                    tile.shape(),
                    &mut values,
                    shape,
                    &mut weights,
                    top,
                    left,
                    [TILE_SIZE[0] - TILE_STRIDE[0], TILE_SIZE[1] - TILE_STRIDE[1]],
                )?;
            }
        }
        divide_blended(&mut values, shape, &weights)?;
        VideoLatent::new(values, shape)
    }

    fn decode_frames_tiled(&self, latent: &VideoLatent) -> Result<Feature, String> {
        let [_, latent_depth, latent_height, latent_width] = latent.shape;
        let output_depth = latent_depth * 4 - 3;
        let output_height = latent_height * SPATIAL_FACTOR;
        let output_width = latent_width * SPATIAL_FACTOR;
        let shape = [3, output_depth, output_height, output_width];
        let mut values = vec![0.0; checked_len("DreamX tiled video", &shape)?];
        let mut weights = vec![0.0; output_height * output_width];
        for top in tile_starts(latent_height, TILE_SIZE[0], TILE_STRIDE[0])? {
            let tile_height = TILE_SIZE[0].min(latent_height - top);
            for left in tile_starts(latent_width, TILE_SIZE[1], TILE_STRIDE[1])? {
                let tile_width = TILE_SIZE[1].min(latent_width - left);
                let tile = crop_latent(latent, top, left, tile_height, tile_width)?;
                let decoded = self.decode_frames_core(&tile)?;
                blend_tile(
                    &decoded.data,
                    decoded.shape,
                    &mut values,
                    shape,
                    &mut weights,
                    top * SPATIAL_FACTOR,
                    left * SPATIAL_FACTOR,
                    [
                        (TILE_SIZE[0] - TILE_STRIDE[0]) * SPATIAL_FACTOR,
                        (TILE_SIZE[1] - TILE_STRIDE[1]) * SPATIAL_FACTOR,
                    ],
                )?;
            }
        }
        divide_blended(&mut values, shape, &weights)?;
        Feature::new(values, shape)
    }

    fn source(&self) -> Result<&dyn TensorSource, String> {
        self.source
            .as_deref()
            .ok_or_else(|| "DreamX VAE has no tensor source".into())
    }

    fn graph(&self) -> Result<&VaeGraph, String> {
        self.graph
            .as_ref()
            .ok_or_else(|| "DreamX VAE graph is unavailable".into())
    }

    #[cfg(test)]
    fn testing() -> Self {
        Self {
            source: None,
            pool: Arc::new(ComputePool::new(1)),
            graph: None,
        }
    }
}

fn validate_frames(frames: &[RgbImage]) -> Result<(), String> {
    let first = frames
        .first()
        .ok_or("DreamX VAE requires at least one frame")?;
    let width = first.width() as usize;
    let height = first.height() as usize;
    if width == 0
        || height == 0
        || !width.is_multiple_of(SPATIAL_FACTOR)
        || !height.is_multiple_of(SPATIAL_FACTOR)
        || frames
            .iter()
            .any(|frame| frame.width() != first.width() || frame.height() != first.height())
    {
        return Err("DreamX VAE frames must share dimensions divisible by 16".into());
    }
    Ok(())
}

fn patchify_frames(frames: &[RgbImage]) -> Result<Feature, String> {
    validate_frames(frames)?;
    let depth = frames.len();
    let height = frames[0].height() as usize / PATCH_SIZE;
    let width = frames[0].width() as usize / PATCH_SIZE;
    let mut output = Feature::zeros([12, depth, height, width])?;
    let plane = depth * height * width;
    for (frame_index, frame) in frames.iter().enumerate() {
        for y in 0..height {
            for x in 0..width {
                for channel in 0..3 {
                    for inner_x in 0..PATCH_SIZE {
                        for inner_y in 0..PATCH_SIZE {
                            let output_channel =
                                (channel * PATCH_SIZE + inner_x) * PATCH_SIZE + inner_y;
                            let value = frame.get_pixel(
                                (x * PATCH_SIZE + inner_x) as u32,
                                (y * PATCH_SIZE + inner_y) as u32,
                            )[channel];
                            output.data[output_channel * plane
                                + frame_index * height * width
                                + y * width
                                + x] = value as f32 / 127.5 - 1.0;
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

fn unpatchify(input: &Feature) -> Result<Feature, String> {
    let [channels, depth, height, width] = input.shape;
    if channels != 12 {
        return Err("DreamX VAE decoder must produce 12 patch channels".into());
    }
    let output_height = height * PATCH_SIZE;
    let output_width = width * PATCH_SIZE;
    let plane = depth * height * width;
    let output_plane = depth * output_height * output_width;
    let mut output = vec![0.0; 3 * output_plane];
    for channel in 0..3 {
        for frame in 0..depth {
            for y in 0..output_height {
                for x in 0..output_width {
                    let source_x = x / PATCH_SIZE;
                    let source_y = y / PATCH_SIZE;
                    let inner_x = x % PATCH_SIZE;
                    let inner_y = y % PATCH_SIZE;
                    let source_channel = (channel * PATCH_SIZE + inner_x) * PATCH_SIZE + inner_y;
                    output[(channel * depth + frame) * output_height * output_width
                        + y * output_width
                        + x] = input.data[source_channel * plane
                        + frame * height * width
                        + source_y * width
                        + source_x];
                }
            }
        }
    }
    Feature::new(output, [3, depth, output_height, output_width])
}

fn feature_to_rgb(input: &Feature) -> Result<Vec<RgbImage>, String> {
    let [channels, depth, height, width] = input.shape;
    if channels != 3 {
        return Err("DreamX VAE RGB feature must have three channels".into());
    }
    let image_width = u32::try_from(width).map_err(|_| "DreamX video width exceeds u32")?;
    let image_height = u32::try_from(height).map_err(|_| "DreamX video height exceeds u32")?;
    let spatial = height * width;
    let mut frames = Vec::with_capacity(depth);
    for frame in 0..depth {
        frames.push(RgbImage::from_fn(image_width, image_height, |x, y| {
            let pixel = y as usize * width + x as usize;
            Rgb(std::array::from_fn(|channel| {
                let value =
                    input.data[(channel * depth + frame) * spatial + pixel].clamp(-1.0, 1.0);
                ((value + 1.0) * 127.5).round() as u8
            }))
        }));
    }
    Ok(frames)
}

fn tile_starts(length: usize, size: usize, stride: usize) -> Result<Vec<usize>, String> {
    if length == 0 || size == 0 || stride == 0 || stride > size {
        return Err("Invalid DreamX VAE tile dimensions".into());
    }
    let mut starts = Vec::new();
    let mut start = 0;
    while start < length {
        if start >= stride && start - stride + size >= length {
            break;
        }
        starts.push(start);
        start = start
            .checked_add(stride)
            .ok_or("DreamX VAE tile offset overflow")?;
    }
    Ok(starts)
}

fn crop_latent(
    latent: &VideoLatent,
    top: usize,
    left: usize,
    height: usize,
    width: usize,
) -> Result<VideoLatent, String> {
    let [channels, depth, source_height, source_width] = latent.shape;
    if top + height > source_height || left + width > source_width {
        return Err("DreamX VAE latent crop is out of bounds".into());
    }
    let mut output = vec![0.0; channels * depth * height * width];
    let source_plane = source_height * source_width;
    let output_plane = height * width;
    for channel in 0..channels {
        for frame in 0..depth {
            for row in 0..height {
                let source =
                    (channel * depth + frame) * source_plane + (top + row) * source_width + left;
                let target = (channel * depth + frame) * output_plane + row * width;
                output[target..target + width]
                    .copy_from_slice(&latent.values[source..source + width]);
            }
        }
    }
    VideoLatent::new(output, [channels, depth, height, width])
}

#[allow(clippy::too_many_arguments)]
fn blend_tile(
    tile: &[f32],
    tile_shape: [usize; 4],
    output: &mut [f32],
    output_shape: [usize; 4],
    weights: &mut [f32],
    top: usize,
    left: usize,
    border: [usize; 2],
) -> Result<(), String> {
    let [channels, depth, tile_height, tile_width] = tile_shape;
    let [output_channels, output_depth, output_height, output_width] = output_shape;
    if channels != output_channels
        || depth != output_depth
        || top + tile_height > output_height
        || left + tile_width > output_width
        || tile.len() != checked_len("DreamX VAE tile", &tile_shape)?
        || output.len() != checked_len("DreamX VAE tile output", &output_shape)?
        || weights.len() != output_height * output_width
    {
        return Err("Invalid DreamX VAE tile blend".into());
    }
    let tile_spatial = tile_height * tile_width;
    let output_spatial = output_height * output_width;
    for y in 0..tile_height {
        let vertical = blend_axis(y, tile_height, top, output_height, border[0]);
        for x in 0..tile_width {
            let weight = vertical * blend_axis(x, tile_width, left, output_width, border[1]);
            let output_pixel = (top + y) * output_width + left + x;
            weights[output_pixel] += weight;
            for channel in 0..channels {
                for frame in 0..depth {
                    output[(channel * depth + frame) * output_spatial + output_pixel] += tile
                        [(channel * depth + frame) * tile_spatial + y * tile_width + x]
                        * weight;
                }
            }
        }
    }
    Ok(())
}

fn blend_axis(index: usize, length: usize, offset: usize, total: usize, border: usize) -> f32 {
    if border == 0 {
        return 1.0;
    }
    let ramp = border.min(length);
    let mut weight = 1.0f32;
    if offset > 0 && index < ramp {
        weight = weight.min((index + 1) as f32 / ramp as f32);
    }
    if offset + length < total && index >= length - ramp {
        weight = weight.min((length - index) as f32 / ramp as f32);
    }
    weight
}

fn divide_blended(output: &mut [f32], shape: [usize; 4], weights: &[f32]) -> Result<(), String> {
    let [channels, depth, height, width] = shape;
    let spatial = height * width;
    if output.len() != channels * depth * spatial
        || weights.len() != spatial
        || weights.iter().any(|&weight| weight <= 0.0)
    {
        return Err("DreamX VAE tiles did not cover the output".into());
    }
    for channel in 0..channels {
        for frame in 0..depth {
            let plane = &mut output
                [(channel * depth + frame) * spatial..(channel * depth + frame + 1) * spatial];
            for (value, weight) in plane.iter_mut().zip(weights) {
                *value /= weight;
            }
        }
    }
    Ok(())
}

fn channel_rms_norm(input: &Feature, gamma: &[f32]) -> Result<Feature, String> {
    let [channels, depth, height, width] = input.shape;
    if gamma.len() != channels {
        return Err("Invalid DreamX channel RMSNorm gamma".into());
    }
    let plane = depth * height * width;
    let mut output = vec![0.0; input.data.len()];
    for index in 0..plane {
        let mut sum = 0.0f64;
        for channel in 0..channels {
            let value = input.data[channel * plane + index];
            sum += (value * value) as f64;
        }
        let scale = (channels as f64).sqrt() as f32 / (sum.sqrt() as f32).max(1e-12);
        for channel in 0..channels {
            output[channel * plane + index] =
                input.data[channel * plane + index] * scale * gamma[channel];
        }
    }
    Feature::new(output, input.shape)
}

fn add_features(mut left: Feature, right: &Feature) -> Result<Feature, String> {
    if left.shape != right.shape {
        return Err(format!(
            "DreamX VAE residual shape mismatch: {:?} != {:?}",
            left.shape, right.shape
        ));
    }
    for (left, right) in left.data.iter_mut().zip(&right.data) {
        *left += right;
    }
    Ok(left)
}

fn slice_time(input: &Feature, start: usize, end: usize) -> Result<Feature, String> {
    let [channels, depth, height, width] = input.shape;
    if start >= end || end > depth {
        return Err("Invalid DreamX VAE temporal slice".into());
    }
    let spatial = height * width;
    let output_depth = end - start;
    let mut output = vec![0.0; channels * output_depth * spatial];
    for channel in 0..channels {
        let source = (channel * depth + start) * spatial;
        let target = channel * output_depth * spatial;
        output[target..target + output_depth * spatial]
            .copy_from_slice(&input.data[source..source + output_depth * spatial]);
    }
    Feature::new(output, [channels, output_depth, height, width])
}

fn concat_time(left: &Feature, right: &Feature) -> Result<Feature, String> {
    let [channels, left_depth, height, width] = left.shape;
    if right.shape[0] != channels || right.shape[2..] != [height, width] {
        return Err("Invalid DreamX VAE temporal concatenation".into());
    }
    let right_depth = right.shape[1];
    let spatial = height * width;
    let mut output = vec![0.0; channels * (left_depth + right_depth) * spatial];
    for channel in 0..channels {
        let target = channel * (left_depth + right_depth) * spatial;
        output[target..target + left_depth * spatial].copy_from_slice(
            &left.data[channel * left_depth * spatial..(channel + 1) * left_depth * spatial],
        );
        output[target + left_depth * spatial..target + (left_depth + right_depth) * spatial]
            .copy_from_slice(
                &right.data[channel * right_depth * spatial..(channel + 1) * right_depth * spatial],
            );
    }
    Feature::new(output, [channels, left_depth + right_depth, height, width])
}

fn prepend_context(
    previous: Option<&Feature>,
    input: &Feature,
    context: usize,
) -> Result<Feature, String> {
    let [channels, depth, height, width] = input.shape;
    let previous_depth = previous.map_or(0, |feature| feature.shape[1].min(context));
    if previous.is_some_and(|feature| {
        feature.shape[0] != channels || feature.shape[2..] != [height, width]
    }) {
        return Err("Invalid DreamX VAE causal context shape".into());
    }
    let zero_depth = context - previous_depth;
    let output_depth = context + depth;
    let spatial = height * width;
    let mut output = vec![0.0; channels * output_depth * spatial];
    for channel in 0..channels {
        let target = channel * output_depth * spatial;
        if let Some(previous) = previous {
            let source_depth = previous.shape[1];
            let source = (channel * source_depth + source_depth - previous_depth) * spatial;
            output[target + zero_depth * spatial..target + context * spatial]
                .copy_from_slice(&previous.data[source..source + previous_depth * spatial]);
        }
        output[target + context * spatial..target + output_depth * spatial].copy_from_slice(
            &input.data[channel * depth * spatial..(channel + 1) * depth * spatial],
        );
    }
    Feature::new(output, [channels, output_depth, height, width])
}

fn tail_joined(
    previous: Option<&Feature>,
    input: &Feature,
    frames: usize,
) -> Result<Feature, String> {
    if frames == 0 {
        return Err("DreamX VAE cache size must be positive".into());
    }
    let joined = match previous {
        Some(previous) => concat_time(previous, input)?,
        None => input.clone(),
    };
    let start = joined.shape[1].saturating_sub(frames);
    slice_time(&joined, start, joined.shape[1])
}

fn cache_tail_with_zeros(input: &Feature, frames: usize) -> Result<Feature, String> {
    if input.shape[1] >= frames {
        return slice_time(input, input.shape[1] - frames, input.shape[1]);
    }
    prepend_context(None, input, frames - input.shape[1])
}

fn pad_right_bottom(input: &Feature) -> Result<Feature, String> {
    let [channels, depth, height, width] = input.shape;
    let mut output = Feature::zeros([channels, depth, height + 1, width + 1])?;
    let source_plane = height * width;
    let target_plane = (height + 1) * (width + 1);
    for channel in 0..channels {
        for frame in 0..depth {
            for row in 0..height {
                let source = (channel * depth + frame) * source_plane + row * width;
                let target = (channel * depth + frame) * target_plane + row * (width + 1);
                output.data[target..target + width]
                    .copy_from_slice(&input.data[source..source + width]);
            }
        }
    }
    Ok(output)
}

fn nearest_spatial(input: &Feature, factor: usize) -> Result<Feature, String> {
    let [channels, depth, height, width] = input.shape;
    let output_height = height
        .checked_mul(factor)
        .ok_or("DreamX VAE upsample height overflow")?;
    let output_width = width
        .checked_mul(factor)
        .ok_or("DreamX VAE upsample width overflow")?;
    let mut output = Feature::zeros([channels, depth, output_height, output_width])?;
    let input_plane = height * width;
    let output_plane = output_height * output_width;
    for channel in 0..channels {
        for frame in 0..depth {
            for y in 0..output_height {
                for x in 0..output_width {
                    output.data[(channel * depth + frame) * output_plane + y * output_width + x] =
                        input.data[(channel * depth + frame) * input_plane
                            + (y / factor) * width
                            + x / factor];
                }
            }
        }
    }
    Ok(output)
}

fn avg_down3d(
    input: &Feature,
    input_channels: usize,
    output_channels: usize,
    factor_t: usize,
    factor_s: usize,
) -> Result<Feature, String> {
    let [channels, depth, height, width] = input.shape;
    if channels != input_channels
        || height % factor_s != 0
        || width % factor_s != 0
        || !(input_channels * factor_t * factor_s * factor_s).is_multiple_of(output_channels)
    {
        return Err("Invalid DreamX average-downsample shape".into());
    }
    let pad_t = (factor_t - depth % factor_t) % factor_t;
    let output_depth = (depth + pad_t) / factor_t;
    let output_height = height / factor_s;
    let output_width = width / factor_s;
    let factor = factor_t * factor_s * factor_s;
    let group_size = input_channels * factor / output_channels;
    let mut output = Feature::zeros([output_channels, output_depth, output_height, output_width])?;
    let input_plane = depth * height * width;
    let output_plane = output_depth * output_height * output_width;
    for output_channel in 0..output_channels {
        for output_t in 0..output_depth {
            for output_y in 0..output_height {
                for output_x in 0..output_width {
                    let mut sum = 0.0;
                    for group in 0..group_size {
                        let mut composite = output_channel * group_size + group;
                        let inner_x = composite % factor_s;
                        composite /= factor_s;
                        let inner_y = composite % factor_s;
                        composite /= factor_s;
                        let inner_t = composite % factor_t;
                        let input_channel = composite / factor_t;
                        let padded_t = output_t * factor_t + inner_t;
                        if padded_t >= pad_t {
                            let input_t = padded_t - pad_t;
                            let input_y = output_y * factor_s + inner_y;
                            let input_x = output_x * factor_s + inner_x;
                            sum += input.data[input_channel * input_plane
                                + input_t * height * width
                                + input_y * width
                                + input_x];
                        }
                    }
                    output.data[output_channel * output_plane
                        + output_t * output_height * output_width
                        + output_y * output_width
                        + output_x] = sum / group_size as f32;
                }
            }
        }
    }
    Ok(output)
}

fn dup_up3d(
    input: &Feature,
    input_channels: usize,
    output_channels: usize,
    factor_t: usize,
    factor_s: usize,
    first_chunk: bool,
) -> Result<Feature, String> {
    let [channels, depth, height, width] = input.shape;
    let factor = factor_t * factor_s * factor_s;
    if channels != input_channels || !(output_channels * factor).is_multiple_of(input_channels) {
        return Err("Invalid DreamX duplicate-upsample shape".into());
    }
    let repeats = output_channels * factor / input_channels;
    let full_depth = depth * factor_t;
    let output_depth = if first_chunk {
        full_depth - (factor_t - 1)
    } else {
        full_depth
    };
    let output_height = height * factor_s;
    let output_width = width * factor_s;
    let mut output = Feature::zeros([output_channels, output_depth, output_height, output_width])?;
    let input_plane = depth * height * width;
    let output_plane = output_depth * output_height * output_width;
    for output_channel in 0..output_channels {
        for input_t in 0..depth {
            for inner_t in 0..factor_t {
                let full_t = input_t * factor_t + inner_t;
                if first_chunk && full_t < factor_t - 1 {
                    continue;
                }
                let output_t = full_t - if first_chunk { factor_t - 1 } else { 0 };
                for input_y in 0..height {
                    for inner_y in 0..factor_s {
                        for input_x in 0..width {
                            for inner_x in 0..factor_s {
                                let composite =
                                    (((output_channel * factor_t + inner_t) * factor_s + inner_y)
                                        * factor_s)
                                        + inner_x;
                                let input_channel = composite / repeats;
                                let output_y = input_y * factor_s + inner_y;
                                let output_x = input_x * factor_s + inner_x;
                                output.data[output_channel * output_plane
                                    + output_t * output_height * output_width
                                    + output_y * output_width
                                    + output_x] = input.data[input_channel * input_plane
                                    + input_t * height * width
                                    + input_y * width
                                    + input_x];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

fn interleave_time_channels(input: Feature) -> Result<Feature, String> {
    let [double_channels, depth, height, width] = input.shape;
    if double_channels % 2 != 0 {
        return Err("Invalid DreamX temporal upsample channels".into());
    }
    let channels = double_channels / 2;
    let spatial = height * width;
    let mut output = vec![0.0; input.data.len()];
    for channel in 0..channels {
        for frame in 0..depth {
            for branch in 0..2 {
                let source = ((branch * channels + channel) * depth + frame) * spatial;
                let target = (channel * depth * 2 + frame * 2 + branch) * spatial;
                output[target..target + spatial]
                    .copy_from_slice(&input.data[source..source + spatial]);
            }
        }
    }
    Feature::new(output, [channels, depth * 2, height, width])
}

fn validate_f32_tensor(
    source: &dyn TensorSource,
    name: &str,
    source_shape: &[usize],
) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let expected: Vec<u64> = source_shape
        .iter()
        .rev()
        .map(|&value| value as u64)
        .collect();
    if info.dims != expected {
        return Err(format!(
            "Invalid tensor {name}: shape {:?}; expected {:?}",
            info.dims, expected
        ));
    }
    if info.ggml_type != GGMLType::F32 {
        return Err(format!(
            "Invalid tensor {name}: expected F32, got {:?}",
            info.ggml_type
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let expected_bytes = checked_len("DreamX F32 tensor", source_shape)?
        .checked_mul(4)
        .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected_bytes}",
            bytes.len()
        ));
    }
    Ok(())
}

fn f32_tensor<'a>(source: &'a dyn TensorSource, name: &str) -> Result<&'a [f32], String> {
    if cfg!(target_endian = "big") {
        return Err("DreamX F32 mmap weights require a little-endian target".into());
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let (prefix, values, suffix) = unsafe { bytes.align_to::<f32>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(format!("Unaligned DreamX F32 tensor: {name}"));
    }
    Ok(values)
}

fn load_f32_values(
    source: &dyn TensorSource,
    name: &str,
    source_shape: &[usize],
) -> Result<Vec<f32>, String> {
    validate_f32_tensor(source, name, source_shape)?;
    Ok(f32_tensor(source, name)?.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_values(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| (index as f32 - len as f32 * 0.5) / 17.0)
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: expected {expected}, got {actual}"
            );
        }
    }

    fn rgb_fixture(height: u32, width: u32) -> RgbImage {
        RgbImage::from_fn(width, height, |x, y| {
            Rgb([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8])
        })
    }

    #[test]
    fn wan_latent_normalization_round_trips() {
        let raw = deterministic_values(48 * 2);
        let normalized = normalize_latent(&raw).unwrap();
        assert_close(&denormalize_latent(&normalized).unwrap(), &raw, 1e-6);
    }

    #[test]
    fn first_frame_encode_has_48_channels_and_spatial_ratio_16() {
        let latent = Wan22Vae::testing()
            .encode_first_frame(&rgb_fixture(64, 96))
            .unwrap();
        assert_eq!(latent.shape(), [48, 1, 4, 6]);
    }

    #[test]
    fn initial_temporal_upsample_cache_zero_fills_two_frames() {
        let input = Feature::new(vec![3.0], [1, 1, 1, 1]).unwrap();
        let cached = cache_tail_with_zeros(&input, 2).unwrap();
        assert_eq!(cached.shape, [1, 2, 1, 1]);
        assert_eq!(cached.data, [0.0, 3.0]);
    }

    #[test]
    fn tiling_skips_redundant_trailing_tiles() {
        assert_eq!(tile_starts(40, 34, 18).unwrap(), [0, 18]);
        assert_eq!(tile_starts(34, 34, 18).unwrap(), [0]);
        assert_eq!(tile_starts(12, 34, 18).unwrap(), [0]);
    }
}
