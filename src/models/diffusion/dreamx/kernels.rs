use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{QuantizedTensor, Weight};
use crate::ops::quant::{BlockQ8K, QK_K};
use crate::ops::{dot_f32, sum_f32, sum_sq_centered_f32, vec_add_into};

pub use crate::ops::{
    gelu_approx_inplace, gelu_inplace, rms_norm, rms_norm_inplace, rope_neox, rope_neox_partial,
    silu_approx_inplace, silu_inplace, silu_mul_approx_inplace, silu_mul_inplace,
};

pub fn checked_len(name: &str, dimensions: &[usize]) -> Result<usize, String> {
    if dimensions.is_empty() || dimensions.contains(&0) {
        return Err(format!("{name} dimensions must be non-zero"));
    }
    dimensions.iter().try_fold(1usize, |length, &dimension| {
        length
            .checked_mul(dimension)
            .ok_or_else(|| format!("{name} shape overflows usize"))
    })
}

pub struct Linear<'a> {
    weight: Weight<'a>,
    bias: Option<Vec<f32>>,
}

impl<'a> Linear<'a> {
    pub fn from_weight(weight: Weight<'a>, bias: Option<Vec<f32>>) -> Result<Self, String> {
        if weight.n_in == 0 || weight.n_out == 0 {
            return Err("DreamX linear dimensions must be non-zero".into());
        }
        if bias.as_ref().is_some_and(|bias| bias.len() != weight.n_out) {
            return Err("DreamX linear bias width mismatch".into());
        }
        Ok(Self { weight, bias })
    }

    pub fn from_source(
        source: &'a dyn TensorSource,
        weight_name: &str,
        bias_name: Option<&str>,
        n_in: usize,
        n_out: usize,
    ) -> Result<Self, String> {
        let info = source
            .tensor_info(weight_name)
            .ok_or_else(|| format!("Missing tensor: {weight_name}"))?;
        let expected_dims = [n_in as u64, n_out as u64];
        if info.dims != expected_dims {
            return Err(format!(
                "Invalid tensor {weight_name}: shape {:?}; expected {:?}",
                info.dims, expected_dims
            ));
        }
        match info.ggml_type {
            GGMLType::F32
            | GGMLType::F16
            | GGMLType::BF16
            | GGMLType::Q8_0
            | GGMLType::Q6K
            | GGMLType::Q2K
            | GGMLType::Q3K
            | GGMLType::IQ4_NL
            | GGMLType::IQ2_XXS
            | GGMLType::IQ2_S
            | GGMLType::IQ2_XS
            | GGMLType::IQ3_XXS
            | GGMLType::IQ3_S
            | GGMLType::IQ4_XS
            | GGMLType::IQ1_M
            | GGMLType::IQ1_S
            | GGMLType::Q4_0
            | GGMLType::Q4_1
            | GGMLType::Q4K
            | GGMLType::Q5K => {}
            other => return Err(format!("Unsupported DreamX linear type {other:?}")),
        }
        let bytes = source
            .tensor_slice(weight_name)
            .ok_or_else(|| format!("Missing tensor data: {weight_name}"))?;
        let expected_bytes = info
            .checked_nbytes()
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| format!("Invalid tensor byte size: {weight_name}"))?;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "Invalid tensor data length for {weight_name}: {}; expected {expected_bytes}",
                bytes.len()
            ));
        }
        let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
            bytes,
            info.ggml_type,
            n_in,
            n_out,
        ));
        weight.n_in = n_in;
        weight.n_out = n_out;
        let bias = bias_name
            .map(|name| load_float_tensor(source, name, n_out))
            .transpose()?;
        Self::from_weight(weight, bias)
    }

    pub fn n_in(&self) -> usize {
        self.weight.n_in
    }

    pub fn n_out(&self) -> usize {
        self.weight.n_out
    }

    pub fn forward(
        &self,
        pool: &ComputePool,
        input: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>, String> {
        if rows == 0 || input.len() != checked_len("DreamX linear input", &[rows, self.n_in()])? {
            return Err("Invalid DreamX linear input shape".into());
        }
        let mut output = vec![0.0; checked_len("DreamX linear output", &[rows, self.n_out()])?];
        if let Some(weight) = self.weight.kernel.bf16_bytes() {
            linear_bf16(
                pool,
                weight,
                input,
                rows,
                self.n_in(),
                self.n_out(),
                self.bias.as_deref(),
                &mut output,
            );
            return Ok(output);
        }
        let mut q8 = vec![0; self.n_in()];
        let mut scales = vec![0.0; self.n_in().div_ceil(32)];
        let mut q8k = vec![
            BlockQ8K {
                d: 0.0,
                qs: [0; QK_K],
                bsums: [0; QK_K / 16],
            };
            self.n_in().div_ceil(QK_K)
        ];
        for row in 0..rows {
            let input_row = &input[row * self.n_in()..(row + 1) * self.n_in()];
            let output_row = &mut output[row * self.n_out()..(row + 1) * self.n_out()];
            self.weight.quantize_and_matmul_with_scratch(
                input_row,
                &mut q8k,
                &mut q8,
                &mut scales,
                output_row,
                pool,
            );
            if let Some(bias) = &self.bias {
                vec_add_into(bias, output_row);
            }
        }
        Ok(output)
    }
}

fn load_float_tensor(
    source: &dyn TensorSource,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [expected_len as u64] {
        return Err(format!(
            "Invalid tensor {name}: shape {:?}; expected [{expected_len}]",
            info.dims
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let expected_bytes = info
        .checked_nbytes()
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected_bytes}",
            bytes.len()
        ));
    }
    let values: Vec<f32> = match info.ggml_type {
        GGMLType::F32 => bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect(),
        GGMLType::F16 => bytes
            .chunks_exact(2)
            .map(|bytes| crate::ops::f16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())))
            .collect(),
        GGMLType::BF16 => bytes
            .chunks_exact(2)
            .map(|bytes| crate::ops::bf16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())))
            .collect(),
        other => {
            return Err(format!(
                "Unsupported DreamX float tensor type {other:?}: {name}"
            ))
        }
    };
    if values.len() != expected_len {
        return Err(format!("Invalid tensor data length for {name}"));
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn linear_bf16(
    pool: &ComputePool,
    weight: &[u8],
    input: &[f32],
    rows: usize,
    n_in: usize,
    n_out: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    let output_address = output.as_mut_ptr() as usize;
    let output_len = rows * n_out;
    pool.compute(|thread, threads| {
        for index in (thread..output_len).step_by(threads) {
            let row = index / n_out;
            let column = index % n_out;
            let input = &input[row * n_in..(row + 1) * n_in];
            let weight = &weight[column * n_in * 2..(column + 1) * n_in * 2];
            let value = dot_bf16_f32(weight, input) + bias.map_or(0.0, |bias| bias[column]);
            unsafe { (output_address as *mut f32).add(index).write(value) };
        }
    });
}

fn dot_bf16_f32(weight: &[u8], input: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if crate::ops::has_neon() {
        return unsafe { dot_bf16_f32_neon(weight, input) };
    }
    weight
        .chunks_exact(2)
        .zip(input)
        .map(|(bytes, &input)| {
            crate::ops::bf16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())) * input
        })
        .sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_bf16_f32_neon(weight: &[u8], input: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let mut accumulator = vdupq_n_f32(0.0);
    let mut index = 0;
    while index + 4 <= input.len() {
        let packed = vld1_u16(weight.as_ptr().add(index * 2).cast());
        let bits = vshlq_n_u32(vmovl_u16(packed), 16);
        let values = vreinterpretq_f32_u32(bits);
        accumulator = vfmaq_f32(accumulator, values, vld1q_f32(input.as_ptr().add(index)));
        index += 4;
    }
    let mut sum = vaddvq_f32(accumulator);
    while index < input.len() {
        let offset = index * 2;
        let value =
            crate::ops::bf16_to_f32(u16::from_le_bytes([weight[offset], weight[offset + 1]]));
        sum += value * input[index];
        index += 1;
    }
    sum
}

pub fn rms_norm_rows(
    input: &[f32],
    rows: usize,
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    let width = weight.len();
    if rows == 0
        || width == 0
        || input.len() != checked_len("DreamX RMSNorm input", &[rows, width])?
        || !epsilon.is_finite()
        || epsilon < 0.0
    {
        return Err("Invalid DreamX RMSNorm tensors".into());
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let range = row * width..(row + 1) * width;
        rms_norm(&input[range.clone()], weight, &mut output[range], epsilon);
    }
    Ok(output)
}

pub fn layer_norm_rows(
    input: &[f32],
    rows: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    let width = weight
        .map(<[f32]>::len)
        .or_else(|| bias.map(<[f32]>::len))
        .ok_or("DreamX LayerNorm requires a weight or bias")?;
    if rows == 0
        || width == 0
        || weight.is_some_and(|values| values.len() != width)
        || bias.is_some_and(|values| values.len() != width)
        || input.len() != checked_len("DreamX LayerNorm input", &[rows, width])?
        || !epsilon.is_finite()
        || epsilon < 0.0
    {
        return Err("Invalid DreamX LayerNorm tensors".into());
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let range = row * width..(row + 1) * width;
        let values = &input[range.clone()];
        let mean = (sum_f32(values) / width as f64) as f32;
        let variance = (sum_sq_centered_f32(values, mean) / width as f64) as f32;
        let inverse = 1.0 / (variance + epsilon).sqrt();
        for index in 0..width {
            output[range.start + index] =
                (values[index] - mean) * inverse * weight.map_or(1.0, |weight| weight[index])
                    + bias.map_or(0.0, |bias| bias[index]);
        }
    }
    Ok(output)
}

pub fn group_norm_ncthw(
    input: &[f32],
    shape: [usize; 4],
    groups: usize,
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    let [channels, depth, height, width] = shape;
    if groups == 0
        || channels % groups != 0
        || weight.len() != channels
        || bias.len() != channels
        || input.len() != checked_len("DreamX GroupNorm input", &shape)?
        || !epsilon.is_finite()
        || epsilon < 0.0
    {
        return Err("Invalid DreamX GroupNorm tensors".into());
    }
    let plane = checked_len("DreamX GroupNorm plane", &[depth, height, width])?;
    let channels_per_group = channels / groups;
    let values_per_group = checked_len("DreamX GroupNorm group", &[channels_per_group, plane])?;
    let mut output = vec![0.0; input.len()];
    for group in 0..groups {
        let start = group * values_per_group;
        let values = &input[start..start + values_per_group];
        let mean = (sum_f32(values) / values_per_group as f64) as f32;
        let variance = (sum_sq_centered_f32(values, mean) / values_per_group as f64) as f32;
        let inverse = 1.0 / (variance + epsilon).sqrt();
        for channel_offset in 0..channels_per_group {
            let channel = group * channels_per_group + channel_offset;
            let channel_start = channel * plane;
            for index in 0..plane {
                output[channel_start + index] =
                    (input[channel_start + index] - mean) * inverse * weight[channel]
                        + bias[channel];
            }
        }
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionSpec {
    pub query_tokens: usize,
    pub key_tokens: usize,
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub scale: f32,
}

impl AttentionSpec {
    fn validate(self, query: &[f32], key: &[f32], value: &[f32]) -> Result<(), String> {
        if self.query_tokens == 0
            || self.key_tokens == 0
            || self.query_heads == 0
            || self.key_value_heads == 0
            || self.head_dim == 0
            || self.query_heads % self.key_value_heads != 0
            || !self.scale.is_finite()
            || query.len()
                != checked_len(
                    "DreamX attention query",
                    &[self.query_tokens, self.query_heads, self.head_dim],
                )?
            || key.len()
                != checked_len(
                    "DreamX attention key",
                    &[self.key_tokens, self.key_value_heads, self.head_dim],
                )?
            || value.len() != key.len()
            || (self.causal && self.query_tokens > self.key_tokens)
        {
            return Err("Invalid DreamX attention tensors".into());
        }
        Ok(())
    }

    fn key_visible(self, query: usize, key: usize) -> bool {
        !self.causal || key <= query + (self.key_tokens - self.query_tokens)
    }
}

pub fn attention_scalar(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    spec: AttentionSpec,
) -> Result<Vec<f32>, String> {
    spec.validate(query, key, value)?;
    let mut output = vec![0.0; query.len()];
    let mut scores = vec![f32::NEG_INFINITY; spec.key_tokens];
    let group_size = spec.query_heads / spec.key_value_heads;
    for query_token in 0..spec.query_tokens {
        for query_head in 0..spec.query_heads {
            let query_start = (query_token * spec.query_heads + query_head) * spec.head_dim;
            let key_head = query_head / group_size;
            let mut maximum = f32::NEG_INFINITY;
            for key_token in 0..spec.key_tokens {
                if !spec.key_visible(query_token, key_token) {
                    scores[key_token] = f32::NEG_INFINITY;
                    continue;
                }
                let key_start = (key_token * spec.key_value_heads + key_head) * spec.head_dim;
                let score = dot_f32(
                    &query[query_start..query_start + spec.head_dim],
                    &key[key_start..key_start + spec.head_dim],
                    spec.head_dim,
                ) * spec.scale;
                scores[key_token] = score;
                maximum = maximum.max(score);
            }
            let mut normalizer = 0.0;
            for &score in &scores {
                if score.is_finite() {
                    normalizer += (score - maximum).exp();
                }
            }
            let output_row = &mut output[query_start..query_start + spec.head_dim];
            for (key_token, &score) in scores.iter().enumerate() {
                if !score.is_finite() {
                    continue;
                }
                let probability = (score - maximum).exp() / normalizer;
                let value_start = (key_token * spec.key_value_heads + key_head) * spec.head_dim;
                for dimension in 0..spec.head_dim {
                    output_row[dimension] += probability * value[value_start + dimension];
                }
            }
        }
    }
    Ok(output)
}

pub fn attention_online(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    spec: AttentionSpec,
) -> Result<Vec<f32>, String> {
    spec.validate(query, key, value)?;
    let mut output = vec![0.0; query.len()];
    let group_size = spec.query_heads / spec.key_value_heads;
    for query_token in 0..spec.query_tokens {
        for query_head in 0..spec.query_heads {
            let query_start = (query_token * spec.query_heads + query_head) * spec.head_dim;
            let output_row = &mut output[query_start..query_start + spec.head_dim];
            let key_head = query_head / group_size;
            let mut running_max = f32::NEG_INFINITY;
            let mut normalizer = 0.0;
            for key_token in 0..spec.key_tokens {
                if !spec.key_visible(query_token, key_token) {
                    continue;
                }
                let key_start = (key_token * spec.key_value_heads + key_head) * spec.head_dim;
                let score = dot_f32(
                    &query[query_start..query_start + spec.head_dim],
                    &key[key_start..key_start + spec.head_dim],
                    spec.head_dim,
                ) * spec.scale;
                let new_max = running_max.max(score);
                let old_scale = (running_max - new_max).exp();
                let new_scale = (score - new_max).exp();
                normalizer = normalizer * old_scale + new_scale;
                let value_start = (key_token * spec.key_value_heads + key_head) * spec.head_dim;
                for dimension in 0..spec.head_dim {
                    output_row[dimension] = output_row[dimension] * old_scale
                        + new_scale * value[value_start + dimension];
                }
                running_max = new_max;
            }
            for value in output_row {
                *value /= normalizer;
            }
        }
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug)]
pub struct Conv3dSpec {
    pub stride: [usize; 3],
    pub padding: [usize; 3],
    pub dilation: [usize; 3],
    pub groups: usize,
}

impl Default for Conv3dSpec {
    fn default() -> Self {
        Self {
            stride: [1; 3],
            padding: [0; 3],
            dilation: [1; 3],
            groups: 1,
        }
    }
}

pub fn conv3d(
    pool: &ComputePool,
    input: &[f32],
    input_shape: [usize; 4],
    weight: &[f32],
    weight_shape: [usize; 5],
    stride: [usize; 3],
) -> Result<Vec<f32>, String> {
    conv3d_with_options(
        pool,
        input,
        input_shape,
        weight,
        weight_shape,
        None,
        Conv3dSpec {
            stride,
            ..Conv3dSpec::default()
        },
    )
    .map(|(output, _)| output)
}

pub fn conv3d_with_options(
    pool: &ComputePool,
    input: &[f32],
    input_shape: [usize; 4],
    weight: &[f32],
    weight_shape: [usize; 5],
    bias: Option<&[f32]>,
    spec: Conv3dSpec,
) -> Result<(Vec<f32>, [usize; 4]), String> {
    let [input_channels, depth, height, width] = input_shape;
    let [output_channels, weight_channels, kernel_depth, kernel_height, kernel_width] =
        weight_shape;
    if spec.groups == 0
        || input_channels % spec.groups != 0
        || output_channels % spec.groups != 0
        || weight_channels != input_channels / spec.groups
        || spec.stride.contains(&0)
        || spec.dilation.contains(&0)
        || bias.is_some_and(|bias| bias.len() != output_channels)
        || input.len() != checked_len("DreamX Conv3D input", &input_shape)?
        || weight.len() != checked_len("DreamX Conv3D weight", &weight_shape)?
    {
        return Err("Invalid DreamX Conv3D tensors".into());
    }
    let output_depth = conv_output(
        depth,
        kernel_depth,
        spec.stride[0],
        spec.padding[0],
        spec.dilation[0],
    )?;
    let output_height = conv_output(
        height,
        kernel_height,
        spec.stride[1],
        spec.padding[1],
        spec.dilation[1],
    )?;
    let output_width = conv_output(
        width,
        kernel_width,
        spec.stride[2],
        spec.padding[2],
        spec.dilation[2],
    )?;
    let output_shape = [output_channels, output_depth, output_height, output_width];
    let output_len = checked_len("DreamX Conv3D output", &output_shape)?;
    let mut output = vec![0.0; output_len];
    let output_address = output.as_mut_ptr() as usize;
    let output_channels_per_group = output_channels / spec.groups;
    pool.compute(|thread, threads| {
        for output_index in (thread..output_len).step_by(threads) {
            let ow = output_index % output_width;
            let rest = output_index / output_width;
            let oh = rest % output_height;
            let rest = rest / output_height;
            let od = rest % output_depth;
            let oc = rest / output_depth;
            let group = oc / output_channels_per_group;
            let mut sum = bias.map_or(0.0, |bias| bias[oc]);
            for local_channel in 0..weight_channels {
                let input_channel = group * weight_channels + local_channel;
                for kd in 0..kernel_depth {
                    let padded_depth = od * spec.stride[0] + kd * spec.dilation[0];
                    if padded_depth < spec.padding[0] {
                        continue;
                    }
                    let input_depth = padded_depth - spec.padding[0];
                    if input_depth >= depth {
                        continue;
                    }
                    for kh in 0..kernel_height {
                        let padded_height = oh * spec.stride[1] + kh * spec.dilation[1];
                        if padded_height < spec.padding[1] {
                            continue;
                        }
                        let input_height = padded_height - spec.padding[1];
                        if input_height >= height {
                            continue;
                        }
                        let input_row =
                            ((input_channel * depth + input_depth) * height + input_height) * width;
                        let weight_row = (((oc * weight_channels + local_channel) * kernel_depth
                            + kd)
                            * kernel_height
                            + kh)
                            * kernel_width;
                        if spec.dilation[2] == 1 {
                            let base = ow * spec.stride[2];
                            let first = spec.padding[2].saturating_sub(base).min(kernel_width);
                            let last = kernel_width
                                .min(width.saturating_add(spec.padding[2]).saturating_sub(base));
                            if first < last {
                                let input_width = base + first - spec.padding[2];
                                sum += dot_f32(
                                    &input[input_row + input_width
                                        ..input_row + input_width + last - first],
                                    &weight[weight_row + first..weight_row + last],
                                    last - first,
                                );
                            }
                        } else {
                            for kw in 0..kernel_width {
                                let padded_width = ow * spec.stride[2] + kw * spec.dilation[2];
                                if padded_width < spec.padding[2] {
                                    continue;
                                }
                                let input_width = padded_width - spec.padding[2];
                                if input_width < width {
                                    sum += input[input_row + input_width] * weight[weight_row + kw];
                                }
                            }
                        }
                    }
                }
            }
            unsafe { (output_address as *mut f32).add(output_index).write(sum) };
        }
    });
    Ok((output, output_shape))
}

pub fn conv2d(
    pool: &ComputePool,
    input: &[f32],
    input_shape: [usize; 3],
    weight: &[f32],
    weight_shape: [usize; 4],
    bias: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Result<(Vec<f32>, [usize; 3]), String> {
    let [channels, height, width] = input_shape;
    let [output_channels, weight_channels, kernel_height, kernel_width] = weight_shape;
    let (output, shape) = conv3d_with_options(
        pool,
        input,
        [channels, 1, height, width],
        weight,
        [
            output_channels,
            weight_channels,
            1,
            kernel_height,
            kernel_width,
        ],
        bias,
        Conv3dSpec {
            stride: [1, stride[0], stride[1]],
            padding: [0, padding[0], padding[1]],
            dilation: [1, dilation[0], dilation[1]],
            groups,
        },
    )?;
    Ok((output, [shape[0], shape[2], shape[3]]))
}

pub fn conv1d(
    pool: &ComputePool,
    input: &[f32],
    input_shape: [usize; 2],
    weight: &[f32],
    weight_shape: [usize; 3],
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(Vec<f32>, [usize; 2]), String> {
    let [channels, width] = input_shape;
    let [output_channels, weight_channels, kernel_width] = weight_shape;
    let (output, shape) = conv3d_with_options(
        pool,
        input,
        [channels, 1, 1, width],
        weight,
        [output_channels, weight_channels, 1, 1, kernel_width],
        bias,
        Conv3dSpec {
            stride: [1, 1, stride],
            padding: [0, 0, padding],
            dilation: [1, 1, dilation],
            groups,
        },
    )?;
    Ok((output, [shape[0], shape[3]]))
}

pub fn depthwise_conv2d(
    pool: &ComputePool,
    input: &[f32],
    input_shape: [usize; 3],
    weight: &[f32],
    weight_shape: [usize; 4],
    bias: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> Result<(Vec<f32>, [usize; 3]), String> {
    conv2d(
        pool,
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        stride,
        padding,
        [1; 2],
        input_shape[0],
    )
}

#[derive(Clone, Copy, Debug)]
pub struct ConvTranspose3dSpec {
    pub stride: [usize; 3],
    pub padding: [usize; 3],
    pub output_padding: [usize; 3],
    pub dilation: [usize; 3],
    pub groups: usize,
}

pub fn conv_transpose3d(
    pool: &ComputePool,
    input: &[f32],
    input_shape: [usize; 4],
    weight: &[f32],
    weight_shape: [usize; 5],
    bias: Option<&[f32]>,
    spec: ConvTranspose3dSpec,
) -> Result<(Vec<f32>, [usize; 4]), String> {
    let [input_channels, depth, height, width] = input_shape;
    let [weight_input_channels, output_channels_per_group, kernel_depth, kernel_height, kernel_width] =
        weight_shape;
    let output_channels = output_channels_per_group
        .checked_mul(spec.groups)
        .ok_or("DreamX transposed-convolution channel overflow")?;
    if spec.groups == 0
        || input_channels != weight_input_channels
        || input_channels % spec.groups != 0
        || spec.stride.contains(&0)
        || spec.dilation.contains(&0)
        || spec
            .output_padding
            .iter()
            .zip(spec.stride)
            .any(|(&output_padding, stride)| output_padding >= stride)
        || bias.is_some_and(|bias| bias.len() != output_channels)
        || input.len() != checked_len("DreamX ConvTranspose3D input", &input_shape)?
        || weight.len() != checked_len("DreamX ConvTranspose3D weight", &weight_shape)?
    {
        return Err("Invalid DreamX ConvTranspose3D tensors".into());
    }
    let output_depth = conv_transpose_output(
        depth,
        kernel_depth,
        spec.stride[0],
        spec.padding[0],
        spec.output_padding[0],
        spec.dilation[0],
    )?;
    let output_height = conv_transpose_output(
        height,
        kernel_height,
        spec.stride[1],
        spec.padding[1],
        spec.output_padding[1],
        spec.dilation[1],
    )?;
    let output_width = conv_transpose_output(
        width,
        kernel_width,
        spec.stride[2],
        spec.padding[2],
        spec.output_padding[2],
        spec.dilation[2],
    )?;
    let output_shape = [output_channels, output_depth, output_height, output_width];
    let output_len = checked_len("DreamX ConvTranspose3D output", &output_shape)?;
    let mut output = vec![0.0; output_len];
    let output_address = output.as_mut_ptr() as usize;
    let input_channels_per_group = input_channels / spec.groups;
    pool.compute(|thread, threads| {
        for output_index in (thread..output_len).step_by(threads) {
            let ow = output_index % output_width;
            let rest = output_index / output_width;
            let oh = rest % output_height;
            let rest = rest / output_height;
            let od = rest % output_depth;
            let oc = rest / output_depth;
            let group = oc / output_channels_per_group;
            let output_channel = oc % output_channels_per_group;
            let mut sum = bias.map_or(0.0, |bias| bias[oc]);
            for local_input_channel in 0..input_channels_per_group {
                let input_channel = group * input_channels_per_group + local_input_channel;
                for kd in 0..kernel_depth {
                    let Some(input_depth) = transpose_input_index(
                        od,
                        kd,
                        depth,
                        spec.stride[0],
                        spec.padding[0],
                        spec.dilation[0],
                    ) else {
                        continue;
                    };
                    for kh in 0..kernel_height {
                        let Some(input_height) = transpose_input_index(
                            oh,
                            kh,
                            height,
                            spec.stride[1],
                            spec.padding[1],
                            spec.dilation[1],
                        ) else {
                            continue;
                        };
                        for kw in 0..kernel_width {
                            let Some(input_width) = transpose_input_index(
                                ow,
                                kw,
                                width,
                                spec.stride[2],
                                spec.padding[2],
                                spec.dilation[2],
                            ) else {
                                continue;
                            };
                            let input_index = (((input_channel * depth + input_depth) * height
                                + input_height)
                                * width)
                                + input_width;
                            let weight_index = ((((input_channel * output_channels_per_group
                                + output_channel)
                                * kernel_depth
                                + kd)
                                * kernel_height
                                + kh)
                                * kernel_width)
                                + kw;
                            sum += input[input_index] * weight[weight_index];
                        }
                    }
                }
            }
            unsafe { (output_address as *mut f32).add(output_index).write(sum) };
        }
    });
    Ok((output, output_shape))
}

fn conv_output(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize, String> {
    let effective_kernel = dilation
        .checked_mul(kernel.saturating_sub(1))
        .and_then(|value| value.checked_add(1))
        .ok_or("DreamX convolution kernel overflow")?;
    let padded = padding
        .checked_mul(2)
        .and_then(|padding| input.checked_add(padding))
        .ok_or("DreamX convolution padding overflow")?;
    if kernel == 0 || stride == 0 || dilation == 0 || padded < effective_kernel {
        return Err("Invalid DreamX convolution geometry".into());
    }
    Ok((padded - effective_kernel) / stride + 1)
}

fn conv_transpose_output(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> Result<usize, String> {
    if input == 0 || kernel == 0 || stride == 0 || dilation == 0 || output_padding >= stride {
        return Err("Invalid DreamX transposed-convolution geometry".into());
    }
    let length = (input - 1)
        .checked_mul(stride)
        .and_then(|value| value.checked_add(dilation.checked_mul(kernel - 1)?))
        .and_then(|value| value.checked_add(output_padding))
        .and_then(|value| value.checked_add(1))
        .ok_or("DreamX transposed-convolution shape overflow")?;
    length
        .checked_sub(padding.checked_mul(2).ok_or("DreamX padding overflow")?)
        .filter(|&length| length > 0)
        .ok_or_else(|| "Invalid DreamX transposed-convolution output shape".into())
}

fn transpose_input_index(
    output: usize,
    kernel: usize,
    input_length: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<usize> {
    let padded_output = output.checked_add(padding)?;
    let kernel_offset = kernel.checked_mul(dilation)?;
    let numerator = padded_output.checked_sub(kernel_offset)?;
    if numerator % stride != 0 {
        return None;
    }
    let input = numerator / stride;
    (input < input_length).then_some(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::thread_pool::ComputePool;

    fn values(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| ((index * 17 % 29) as f32 - 14.0) / 11.0)
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {index}: {actual} != {expected}"
            );
        }
    }

    fn conv3d_reference(
        input: &[f32],
        input_shape: [usize; 4],
        weight: &[f32],
        weight_shape: [usize; 5],
        stride: [usize; 3],
    ) -> Vec<f32> {
        let [channels, depth, height, width] = input_shape;
        let [output_channels, weight_channels, kernel_depth, kernel_height, kernel_width] =
            weight_shape;
        assert_eq!(channels, weight_channels);
        let output_depth = (depth - kernel_depth) / stride[0] + 1;
        let output_height = (height - kernel_height) / stride[1] + 1;
        let output_width = (width - kernel_width) / stride[2] + 1;
        let mut output = vec![0.0; output_channels * output_depth * output_height * output_width];
        for oc in 0..output_channels {
            for od in 0..output_depth {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let mut sum = 0.0;
                        for ic in 0..channels {
                            for kd in 0..kernel_depth {
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let input_index = (((ic * depth + od * stride[0] + kd)
                                            * height
                                            + oh * stride[1]
                                            + kh)
                                            * width)
                                            + ow * stride[2]
                                            + kw;
                                        let weight_index =
                                            ((((oc * channels + ic) * kernel_depth + kd)
                                                * kernel_height
                                                + kh)
                                                * kernel_width)
                                                + kw;
                                        sum += input[input_index] * weight[weight_index];
                                    }
                                }
                            }
                        }
                        let output_index =
                            (((oc * output_depth + od) * output_height + oh) * output_width) + ow;
                        output[output_index] = sum;
                    }
                }
            }
        }
        output
    }

    #[test]
    fn conv3d_matches_scalar_reference() {
        let input = values(2 * 3 * 4 * 5);
        let weight = values(4 * 2 * 3 * 3 * 3);
        let expected = conv3d_reference(&input, [2, 3, 4, 5], &weight, [4, 2, 3, 3, 3], [1, 1, 1]);
        let actual = conv3d(
            &ComputePool::new(2),
            &input,
            [2, 3, 4, 5],
            &weight,
            [4, 2, 3, 3, 3],
            [1, 1, 1],
        )
        .unwrap();
        assert_close(&actual, &expected, 1e-5);
    }

    #[test]
    fn online_attention_matches_materialized_softmax() {
        let q = values(2 * 2 * 4);
        let k = values(3 * 2 * 4);
        let v = values(3 * 2 * 4);
        let spec = AttentionSpec {
            query_tokens: 2,
            key_tokens: 3,
            query_heads: 2,
            key_value_heads: 2,
            head_dim: 4,
            causal: false,
            scale: 0.5,
        };
        let expected = attention_scalar(&q, &k, &v, spec).unwrap();
        let actual = attention_online(&q, &k, &v, spec).unwrap();
        assert_close(&actual, &expected, 1e-5);
    }

    #[test]
    fn linear_applies_rows_and_bias() {
        let raw: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let mut weight =
            Weight::from_quantized(QuantizedTensor::from_bytes(&raw, GGMLType::F32, 3, 2));
        weight.n_in = 3;
        weight.n_out = 2;
        let linear = Linear::from_weight(weight, Some(vec![0.5, -0.5])).unwrap();
        let output = linear
            .forward(&ComputePool::new(2), &[1.0, 0.0, -1.0, 2.0, 1.0, 0.0], 2)
            .unwrap();
        assert_close(&output, &[-1.5, -2.5, 4.5, 12.5], 1e-6);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn bf16_dot_neon_matches_scalar_values() {
        let input = values(19);
        let weight = values(19);
        let bytes: Vec<u8> = weight
            .iter()
            .flat_map(|&value| crate::ops::f32_to_bf16(value).to_le_bytes())
            .collect();
        let rounded: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|bytes| crate::ops::bf16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())))
            .collect();
        let expected = rounded.iter().zip(&input).map(|(a, b)| a * b).sum::<f32>();
        let actual = unsafe { dot_bf16_f32_neon(&bytes, &input) };
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }

    #[test]
    fn transposed_convolution_uses_pytorch_weight_layout() {
        let (output, shape) = conv_transpose3d(
            &ComputePool::new(2),
            &[1.0, 2.0],
            [1, 1, 1, 2],
            &[1.0, 10.0, 100.0],
            [1, 1, 1, 1, 3],
            None,
            ConvTranspose3dSpec {
                stride: [1, 1, 2],
                padding: [0, 0, 1],
                output_padding: [0, 0, 1],
                dilation: [1; 3],
                groups: 1,
            },
        )
        .unwrap();
        assert_eq!(shape, [1, 1, 1, 4]);
        assert_eq!(output, [10.0, 102.0, 20.0, 200.0]);
    }
}
