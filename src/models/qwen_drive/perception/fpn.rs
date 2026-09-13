use super::ops::{resize_bilinear_aligned, round_bf16, voxel_pool_depth, Tensor4, VoxelPoolInput};
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::diffusion::dreamx::kernels::{
    conv2d, conv3d_with_options, group_norm_ncthw, Conv3dSpec,
};
use crate::models::qwen_drive::config::PerceptionConfig;
use serde::Deserialize;
use std::collections::BTreeMap;

unsafe extern "C" {
    fn erff(value: f32) -> f32;
}

fn checked_len(dims: &[usize], label: &str) -> Result<usize, String> {
    dims.iter().try_fold(1usize, |size, &dim| {
        size.checked_mul(dim)
            .ok_or_else(|| format!("{label} shape overflow"))
    })
}

fn finite(values: &[f32], label: &str) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} contains non-finite values"));
    }
    Ok(())
}

pub fn layer_norm_2d(
    input: &Tensor4,
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> Result<Tensor4, String> {
    let [batch, channels, height, width] = input.shape();
    if channels == 0
        || weight.len() != channels
        || bias.len() != channels
        || !epsilon.is_finite()
        || epsilon < 0.0
    {
        return Err("invalid Qwen-Drive LayerNorm2d parameters".into());
    }
    finite(weight, "LayerNorm2d weight")?;
    finite(bias, "LayerNorm2d bias")?;
    let plane = height * width;
    let mut output = vec![0.0f32; input.values().len()];
    let mut centered = vec![0.0f32; channels];
    for n in 0..batch {
        for position in 0..plane {
            let mut sum = 0.0f32;
            for channel in 0..channels {
                sum += input.values()[(n * channels + channel) * plane + position];
            }
            let mean = sum / channels as f32;
            let mut sum_square = 0.0f32;
            for channel in 0..channels {
                let value = input.values()[(n * channels + channel) * plane + position] - mean;
                centered[channel] = value;
                sum_square += value * value;
            }
            let variance = sum_square / channels as f32;
            let denominator = (variance + epsilon).sqrt();
            for channel in 0..channels {
                let normalized = centered[channel] / denominator;
                let scaled = normalized * weight[channel];
                output[(n * channels + channel) * plane + position] =
                    round_bf16(scaled + bias[channel]);
            }
        }
    }
    Tensor4::new(output, input.shape())
}

pub fn conv_transpose2d(
    input: &Tensor4,
    weight: &[f32],
    weight_shape: [usize; 4],
    bias: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> Result<Tensor4, String> {
    let [batch, input_channels, input_height, input_width] = input.shape();
    let [weight_input, output_channels, kernel_height, kernel_width] = weight_shape;
    if weight_input != input_channels
        || output_channels == 0
        || kernel_height == 0
        || kernel_width == 0
        || stride.contains(&0)
        || weight.len() != checked_len(&weight_shape, "transpose weight")?
        || bias.is_some_and(|values| values.len() != output_channels)
    {
        return Err("invalid Qwen-Drive transposed convolution parameters".into());
    }
    finite(weight, "transpose weight")?;
    if let Some(bias) = bias {
        finite(bias, "transpose bias")?;
    }
    let output_height = (input_height - 1)
        .checked_mul(stride[0])
        .and_then(|value| value.checked_add(kernel_height))
        .and_then(|value| value.checked_sub(padding[0] * 2))
        .ok_or("invalid transpose output height")?;
    let output_width = (input_width - 1)
        .checked_mul(stride[1])
        .and_then(|value| value.checked_add(kernel_width))
        .and_then(|value| value.checked_sub(padding[1] * 2))
        .ok_or("invalid transpose output width")?;
    let output_plane = output_height * output_width;
    let mut output = vec![
        0.0f32;
        checked_len(
            &[batch, output_channels, output_height, output_width],
            "transpose output"
        )?
    ];
    if let Some(bias) = bias {
        for n in 0..batch {
            for (channel, &value) in bias.iter().enumerate() {
                output[(n * output_channels + channel) * output_plane
                    ..(n * output_channels + channel + 1) * output_plane]
                    .fill(value);
            }
        }
    }
    let input_plane = input_height * input_width;
    for n in 0..batch {
        for input_channel in 0..input_channels {
            for input_y in 0..input_height {
                for input_x in 0..input_width {
                    let value = input.values()[(n * input_channels + input_channel) * input_plane
                        + input_y * input_width
                        + input_x];
                    for output_channel in 0..output_channels {
                        for kernel_y in 0..kernel_height {
                            let padded_y = input_y * stride[0] + kernel_y;
                            if padded_y < padding[0] || padded_y - padding[0] >= output_height {
                                continue;
                            }
                            let output_y = padded_y - padding[0];
                            for kernel_x in 0..kernel_width {
                                let padded_x = input_x * stride[1] + kernel_x;
                                if padded_x < padding[1] || padded_x - padding[1] >= output_width {
                                    continue;
                                }
                                let output_x = padded_x - padding[1];
                                let weight_index = (((input_channel * output_channels
                                    + output_channel)
                                    * kernel_height
                                    + kernel_y)
                                    * kernel_width)
                                    + kernel_x;
                                let output_index = (n * output_channels + output_channel)
                                    * output_plane
                                    + output_y * output_width
                                    + output_x;
                                output[output_index] += value * weight[weight_index];
                            }
                        }
                    }
                }
            }
        }
    }
    output
        .iter_mut()
        .for_each(|value| *value = round_bf16(*value));
    Tensor4::new(
        output,
        [batch, output_channels, output_height, output_width],
    )
}

pub fn depth_softmax(input: &Tensor4) -> Result<Tensor4, String> {
    let [batch, channels, height, width] = input.shape();
    let plane = height * width;
    let mut output = vec![0.0f32; input.values().len()];
    let mut values = vec![0.0f32; channels];
    for n in 0..batch {
        for position in 0..plane {
            for channel in 0..channels {
                values[channel] = input.values()[(n * channels + channel) * plane + position];
            }
            crate::ops::softmax_inplace(&mut values);
            for channel in 0..channels {
                output[(n * channels + channel) * plane + position] = round_bf16(values[channel]);
            }
        }
    }
    Tensor4::new(output, input.shape())
}

pub struct ViewGeometry<'a> {
    pub frustum_range: [f32; 6],
    pub frustum_size: [f32; 3],
    pub pc_range: [f32; 6],
    pub voxel_size: [f32; 3],
    pub voxel_shape: [usize; 3],
    pub lidar2img: &'a [[f32; 16]],
    pub lidar2ego: &'a [[f32; 16]],
}

fn axis(start: f32, end: f32, step: f32) -> Result<Vec<f32>, String> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || step <= 0.0 || end <= start {
        return Err("invalid frustum axis".into());
    }
    let mut values = Vec::new();
    let mut value = start;
    while value < end {
        values.push(value);
        value += step;
        if values.len() > 1_000_000 {
            return Err("frustum axis is too large".into());
        }
    }
    Ok(values)
}

pub(crate) fn inverse_4x4(matrix: &[f32; 16]) -> Result<[f32; 16], String> {
    finite(matrix, "calibration matrix")?;
    let mut rows = [[0.0f32; 8]; 4];
    for row in 0..4 {
        rows[row][..4].copy_from_slice(&matrix[row * 4..row * 4 + 4]);
        rows[row][4 + row] = 1.0;
    }
    for column in 0..4 {
        let pivot = (column..4)
            .max_by(|&left, &right| {
                rows[left][column]
                    .abs()
                    .total_cmp(&rows[right][column].abs())
            })
            .expect("non-empty pivot range");
        if rows[pivot][column].abs() <= f32::EPSILON {
            return Err("singular lidar2img matrix".into());
        }
        rows.swap(column, pivot);
        let divisor = rows[column][column];
        for value in &mut rows[column] {
            *value /= divisor;
        }
        for row in 0..4 {
            if row == column {
                continue;
            }
            let scale = rows[row][column];
            for index in 0..8 {
                rows[row][index] -= scale * rows[column][index];
            }
        }
    }
    let mut inverse = [0.0f32; 16];
    for row in 0..4 {
        inverse[row * 4..row * 4 + 4].copy_from_slice(&rows[row][4..]);
    }
    finite(&inverse, "inverse calibration matrix")?;
    Ok(inverse)
}

#[inline]
fn mat4_vector(matrix: &[f32; 16], vector: [f32; 4]) -> [f32; 4] {
    let mut output = [0.0f32; 4];
    for row in 0..4 {
        for column in 0..4 {
            output[row] += matrix[row * 4 + column] * vector[column];
        }
    }
    output
}

pub fn frustum_voxel_coordinates(
    geometry: &ViewGeometry<'_>,
) -> Result<(Vec<[usize; 4]>, Vec<usize>), String> {
    finite(&geometry.frustum_range, "frustum range")?;
    finite(&geometry.frustum_size, "frustum size")?;
    finite(&geometry.pc_range, "point-cloud range")?;
    finite(&geometry.voxel_size, "voxel size")?;
    if geometry.lidar2ego.is_empty()
        || geometry.voxel_shape.contains(&0)
        || geometry.voxel_size.iter().any(|&value| value <= 0.0)
        || geometry.lidar2img.len() % geometry.lidar2ego.len() != 0
    {
        return Err("invalid view geometry dimensions".into());
    }
    let batch = geometry.lidar2ego.len();
    let cameras = geometry.lidar2img.len() / batch;
    let xs = axis(
        geometry.frustum_range[0],
        geometry.frustum_range[3],
        geometry.frustum_size[0],
    )?;
    let ys = axis(
        geometry.frustum_range[1],
        geometry.frustum_range[4],
        geometry.frustum_size[1],
    )?;
    let depths = axis(
        geometry.frustum_range[2],
        geometry.frustum_range[5],
        geometry.frustum_size[2],
    )?;
    let points_per_camera = checked_len(&[depths.len(), ys.len(), xs.len()], "frustum")?;
    let mut coords = Vec::new();
    let mut point_indices = Vec::new();
    for batch_index in 0..batch {
        finite(&geometry.lidar2ego[batch_index], "lidar2ego")?;
        for camera in 0..cameras {
            let matrix_index = batch_index * cameras + camera;
            let inverse = inverse_4x4(&geometry.lidar2img[matrix_index])?;
            for (depth_index, &depth) in depths.iter().enumerate() {
                for (y_index, &y) in ys.iter().enumerate() {
                    for (x_index, &x) in xs.iter().enumerate() {
                        let lidar = mat4_vector(&inverse, [x * depth, y * depth, depth, 1.0]);
                        let ego = mat4_vector(&geometry.lidar2ego[batch_index], lidar);
                        if ego.iter().any(|value| !value.is_finite()) {
                            return Err("non-finite frustum projection".into());
                        }
                        let mut voxel = [0isize; 3];
                        for dimension in 0..3 {
                            voxel[dimension] = ((ego[dimension] - geometry.pc_range[dimension])
                                / geometry.voxel_size[dimension])
                                as isize;
                        }
                        if voxel.iter().enumerate().all(|(dimension, &value)| {
                            value >= 0 && value < geometry.voxel_shape[dimension] as isize
                        }) {
                            coords.push([
                                batch_index,
                                voxel[0] as usize,
                                voxel[1] as usize,
                                voxel[2] as usize,
                            ]);
                            point_indices.push(
                                (batch_index * cameras + camera) * points_per_camera
                                    + depth_index * ys.len() * xs.len()
                                    + y_index * xs.len()
                                    + x_index,
                            );
                        }
                    }
                }
            }
        }
    }
    Ok((coords, point_indices))
}

pub fn voxel_to_bev_tokens(
    values: &[f32],
    shape: [usize; 5],
    weight: &[f32],
    weight_shape: [usize; 2],
    bias: &[f32],
) -> Result<Vec<f32>, String> {
    let [batch, channels, depth, height, width] = shape;
    let [output_channels, input_channels] = weight_shape;
    if [batch, channels, depth, height, width, output_channels].contains(&0)
        || input_channels != channels * depth
        || values.len() != checked_len(&shape, "voxel BEV input")?
        || weight.len() != checked_len(&weight_shape, "voxel BEV weight")?
        || bias.len() != output_channels
    {
        return Err("invalid voxel-to-BEV projection".into());
    }
    finite(values, "voxel BEV input")?;
    finite(weight, "voxel BEV weight")?;
    finite(bias, "voxel BEV bias")?;
    let positions = height * width;
    let mut projected = vec![0.0f32; batch * output_channels * positions];
    for batch_index in 0..batch {
        for output_channel in 0..output_channels {
            for position in 0..positions {
                let mut sum = bias[output_channel];
                for channel in 0..channels {
                    for depth_index in 0..depth {
                        let input_channel = channel * depth + depth_index;
                        let input_index = (((batch_index * channels + channel) * depth
                            + depth_index)
                            * positions)
                            + position;
                        sum += values[input_index]
                            * weight[output_channel * input_channels + input_channel];
                    }
                }
                projected
                    [(batch_index * output_channels + output_channel) * positions + position] =
                    round_bf16(sum);
            }
        }
    }
    let mut output = vec![0.0f32; positions * output_channels];
    for position in 0..positions {
        for channel in 0..output_channels {
            let mut sum = 0.0f32;
            for batch_index in 0..batch {
                sum += projected[(batch_index * output_channels + channel) * positions + position];
            }
            output[position * output_channels + channel] = round_bf16(sum / batch as f32);
        }
    }
    Ok(output)
}

#[derive(Deserialize)]
struct SourceManifest {
    components: SourceComponents,
}

#[derive(Deserialize)]
struct SourceComponents {
    perception: SourceComponent,
}

#[derive(Deserialize)]
struct SourceComponent {
    tensors: Vec<TensorContract>,
}

#[derive(Clone, Deserialize)]
pub(crate) struct TensorContract {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
}

pub(crate) fn perception_contracts() -> Result<Vec<TensorContract>, String> {
    serde_json::from_str::<SourceManifest>(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tools/qwen_drive/source-tensors.json"
    )))
    .map(|manifest| manifest.components.perception.tensors)
    .map_err(|error| format!("Invalid embedded Qwen-Drive tensor manifest: {error}"))
}

pub(crate) struct F32Tensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) values: Vec<f32>,
}

impl F32Tensor {
    fn load<S: TensorSource + ?Sized>(
        source: &S,
        contract: &TensorContract,
    ) -> Result<Self, String> {
        let name = format!(
            "qwen_drive_perception.{}",
            contract
                .name
                .strip_prefix("bev_modeling.")
                .ok_or_else(|| format!("invalid perception tensor contract: {}", contract.name))?
        );
        let info = source
            .tensor_info(&name)
            .ok_or_else(|| format!("Missing tensor: {name}"))?;
        let expected_dims = if contract.shape.is_empty() {
            vec![1]
        } else {
            contract
                .shape
                .iter()
                .rev()
                .map(|&dimension| dimension as u64)
                .collect::<Vec<_>>()
        };
        if info.ggml_type != GGMLType::F32 || info.dims != expected_dims {
            return Err(format!(
                "Invalid tensor {name}: shape {:?} type {:?}; expected {expected_dims:?} F32",
                info.dims, info.ggml_type
            ));
        }
        let elements = checked_len(&contract.shape, &name)?;
        let bytes = source
            .tensor_slice(&name)
            .ok_or_else(|| format!("Missing tensor data: {name}"))?;
        if bytes.len() != elements * 4 {
            return Err(format!(
                "Invalid tensor data length for {name}: {}; expected {}",
                bytes.len(),
                elements * 4
            ));
        }
        let values = bytes
            .chunks_exact(4)
            .map(|bytes| round_bf16(f32::from_le_bytes(bytes.try_into().unwrap())))
            .collect::<Vec<_>>();
        finite(&values, &name)?;
        Ok(Self {
            shape: contract.shape.clone(),
            values,
        })
    }
}

pub(crate) struct ComponentWeights {
    pub(crate) tensors: BTreeMap<String, F32Tensor>,
}

impl ComponentWeights {
    pub(crate) fn load<S: TensorSource + ?Sized>(
        source: &S,
        contracts: &[TensorContract],
        prefixes: &[&str],
    ) -> Result<Self, String> {
        let tensors = contracts
            .iter()
            .filter(|contract| {
                prefixes
                    .iter()
                    .any(|prefix| contract.name.starts_with(prefix))
            })
            .map(|contract| {
                F32Tensor::load(source, contract).map(|tensor| (contract.name.clone(), tensor))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        if tensors.is_empty() {
            return Err(format!(
                "No tensors matched perception prefixes {prefixes:?}"
            ));
        }
        Ok(Self { tensors })
    }

    pub(crate) fn tensor(&self, name: &str, shape: &[usize]) -> Result<&F32Tensor, String> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("Missing perception tensor: {name}"))?;
        if tensor.shape != shape {
            return Err(format!(
                "Invalid perception tensor {name}: shape {:?}; expected {shape:?}",
                tensor.shape
            ));
        }
        Ok(tensor)
    }
}

pub struct SimpleFpn {
    prefix: String,
    weights: ComponentWeights,
}

impl SimpleFpn {
    fn load<S: TensorSource + ?Sized>(
        source: &S,
        contracts: &[TensorContract],
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            prefix: prefix.to_owned(),
            weights: ComponentWeights::load(source, contracts, &[prefix])?,
        })
    }

    fn name(&self, suffix: &str) -> String {
        format!("{}{suffix}", self.prefix)
    }

    fn conv_transpose(
        &self,
        input: &Tensor4,
        stage: usize,
        layer: usize,
    ) -> Result<Tensor4, String> {
        let [_, input_channels, _, _] = input.shape();
        let weight_name = self.name(&format!("stages.{stage}.{layer}.weight"));
        let weight = self
            .weights
            .tensors
            .get(&weight_name)
            .ok_or_else(|| format!("Missing perception tensor: {weight_name}"))?;
        let [weight_input, output_channels, kernel_height, kernel_width] = weight
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| format!("Invalid transpose weight shape: {weight_name}"))?;
        if weight_input != input_channels {
            return Err(format!("Invalid transpose input channels: {weight_name}"));
        }
        let bias_name = self.name(&format!("stages.{stage}.{layer}.bias"));
        let bias = self.weights.tensor(&bias_name, &[output_channels])?;
        conv_transpose2d(
            input,
            &weight.values,
            [weight_input, output_channels, kernel_height, kernel_width],
            Some(&bias.values),
            [2, 2],
            [0, 0],
        )
    }

    fn conv(
        &self,
        input: &Tensor4,
        stage: usize,
        layer: usize,
        padding: usize,
        pool: &ComputePool,
    ) -> Result<Tensor4, String> {
        let [batch, input_channels, input_height, input_width] = input.shape();
        let weight_name = self.name(&format!("stages.{stage}.{layer}.weight"));
        let weight = self
            .weights
            .tensors
            .get(&weight_name)
            .ok_or_else(|| format!("Missing perception tensor: {weight_name}"))?;
        let [output_channels, weight_input, kernel_height, kernel_width] = weight
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| format!("Invalid convolution weight shape: {weight_name}"))?;
        if weight_input != input_channels {
            return Err(format!("Invalid convolution input channels: {weight_name}"));
        }
        let input_stride = input_channels * input_height * input_width;
        let mut output = Vec::new();
        let mut output_shape = None;
        for values in input.values().chunks_exact(input_stride).take(batch) {
            let (mut values, shape) = conv2d(
                pool,
                values,
                [input_channels, input_height, input_width],
                &weight.values,
                [output_channels, weight_input, kernel_height, kernel_width],
                None,
                [1, 1],
                [padding, padding],
                [1, 1],
                1,
            )?;
            values
                .iter_mut()
                .for_each(|value| *value = round_bf16(*value));
            output.extend(values);
            output_shape = Some(shape);
        }
        let [channels, height, width] = output_shape.ok_or("Empty FPN batch")?;
        Tensor4::new(output, [batch, channels, height, width])
    }

    fn norm(&self, input: &Tensor4, stage: usize, layer: usize) -> Result<Tensor4, String> {
        let channels = input.shape()[1];
        let weight = self.weights.tensor(
            &self.name(&format!("stages.{stage}.{layer}.weight")),
            &[channels],
        )?;
        let bias = self.weights.tensor(
            &self.name(&format!("stages.{stage}.{layer}.bias")),
            &[channels],
        )?;
        layer_norm_2d(input, &weight.values, &bias.values, 1e-6)
    }

    pub fn forward(
        &self,
        input: &Tensor4,
        scales: &[f32],
        pool: &ComputePool,
    ) -> Result<Vec<Tensor4>, String> {
        let mut outputs = Vec::with_capacity(scales.len());
        for (stage, &scale) in scales.iter().enumerate() {
            let (mut output, first_conv) = if scale == 4.0 {
                let mut output = self.conv_transpose(input, stage, 0)?;
                output = self.norm(&output, stage, 1)?;
                output = Tensor4::new(
                    output
                        .values()
                        .iter()
                        .map(|&value| {
                            round_bf16(
                                0.5 * value
                                    * (1.0
                                        + unsafe { erff(value * std::f32::consts::FRAC_1_SQRT_2) }),
                            )
                        })
                        .collect(),
                    output.shape(),
                )?;
                (self.conv_transpose(&output, stage, 3)?, 4)
            } else if scale == 2.0 {
                (self.conv_transpose(input, stage, 0)?, 1)
            } else if scale == 1.0 {
                (input.clone(), 0)
            } else if scale == 0.5 {
                (max_pool_2d(input)?, 1)
            } else {
                return Err(format!("Unsupported Qwen-Drive FPN scale: {scale}"));
            };
            output = self.conv(&output, stage, first_conv, 0, pool)?;
            output = self.norm(&output, stage, first_conv + 1)?;
            output = self.conv(&output, stage, first_conv + 2, 1, pool)?;
            outputs.push(self.norm(&output, stage, first_conv + 3)?);
        }
        Ok(outputs)
    }
}

fn max_pool_2d(input: &Tensor4) -> Result<Tensor4, String> {
    let [batch, channels, height, width] = input.shape();
    let output_height = height / 2;
    let output_width = width / 2;
    if output_height == 0 || output_width == 0 {
        return Err("Qwen-Drive FPN max-pool input is too small".into());
    }
    let mut output = Vec::with_capacity(batch * channels * output_height * output_width);
    let input_plane = height * width;
    for n in 0..batch {
        for channel in 0..channels {
            let base = (n * channels + channel) * input_plane;
            for y in 0..output_height {
                for x in 0..output_width {
                    let top = base + y * 2 * width + x * 2;
                    output.push(
                        input.values()[top]
                            .max(input.values()[top + 1])
                            .max(input.values()[top + width])
                            .max(input.values()[top + width + 1]),
                    );
                }
            }
        }
    }
    Tensor4::new(output, [batch, channels, output_height, output_width])
}

pub struct DepthNet {
    weights: ComponentWeights,
}

impl DepthNet {
    fn tensor(&self, suffix: &str) -> Result<&F32Tensor, String> {
        let name = format!("bev_modeling.depth_net.{suffix}");
        self.weights
            .tensors
            .get(&name)
            .ok_or_else(|| format!("Missing perception tensor: {name}"))
    }

    fn conv(
        &self,
        input: &Tensor4,
        suffix: &str,
        bias: bool,
        padding: usize,
        dilation: usize,
        pool: &ComputePool,
    ) -> Result<Tensor4, String> {
        let weight = self.tensor(&format!("{suffix}.weight"))?;
        let bias = if bias {
            Some(self.tensor(&format!("{suffix}.bias"))?)
        } else {
            None
        };
        conv2d_bf16(input, weight, bias, padding, dilation, pool)
    }

    fn norm(&self, input: &Tensor4, suffix: &str) -> Result<Tensor4, String> {
        group_norm_2d_bf16(
            input,
            self.tensor(&format!("{suffix}.weight"))?,
            self.tensor(&format!("{suffix}.bias"))?,
            32,
            1e-5,
        )
    }

    fn block(&self, input: Tensor4, index: usize, pool: &ComputePool) -> Result<Tensor4, String> {
        let mut output = self.conv(
            &input,
            &format!("depth_conv.{index}.conv1"),
            false,
            1,
            1,
            pool,
        )?;
        output = relu(self.norm(&output, &format!("depth_conv.{index}.gn1"))?);
        output = self.conv(
            &output,
            &format!("depth_conv.{index}.conv2"),
            false,
            1,
            1,
            pool,
        )?;
        output = self.norm(&output, &format!("depth_conv.{index}.gn2"))?;
        add_relu_bf16(&output, &input)
    }

    fn aspp_branch(
        &self,
        input: &Tensor4,
        index: usize,
        padding: usize,
        dilation: usize,
        pool: &ComputePool,
    ) -> Result<Tensor4, String> {
        let mut output = self.conv(
            input,
            &format!("depth_conv.3.aspp{index}.atrous_conv"),
            false,
            padding,
            dilation,
            pool,
        )?;
        output = self.norm(&output, &format!("depth_conv.3.aspp{index}.bn"))?;
        Ok(relu(output))
    }

    pub fn forward(&self, input: &Tensor4, pool: &ComputePool) -> Result<Tensor4, String> {
        let mut output = self.conv(input, "reduce_conv.0", true, 1, 1, pool)?;
        output = relu(self.norm(&output, "reduce_conv.1")?);
        for index in 0..3 {
            output = self.block(output, index, pool)?;
        }

        let [_, _, height, width] = output.shape();
        let mut branches = vec![
            self.aspp_branch(&output, 1, 0, 1, pool)?,
            self.aspp_branch(&output, 2, 6, 6, pool)?,
            self.aspp_branch(&output, 3, 12, 12, pool)?,
            self.aspp_branch(&output, 4, 18, 18, pool)?,
        ];
        let mut global = adaptive_average_pool_1x1(&output)?;
        global = self.conv(&global, "depth_conv.3.global_avg_pool.1", false, 0, 1, pool)?;
        global = relu(self.norm(&global, "depth_conv.3.global_avg_pool.2")?);
        branches.push(resize_bilinear_aligned(&global, height, width, true)?);

        output = concatenate_channels(&branches)?;
        output = self.conv(&output, "depth_conv.3.conv1", false, 0, 1, pool)?;
        output = relu(self.norm(&output, "depth_conv.3.bn1")?);
        self.conv(&output, "depth_conv.4", true, 0, 1, pool)
    }
}

fn conv2d_bf16(
    input: &Tensor4,
    weight: &F32Tensor,
    bias: Option<&F32Tensor>,
    padding: usize,
    dilation: usize,
    pool: &ComputePool,
) -> Result<Tensor4, String> {
    let [batch, input_channels, input_height, input_width] = input.shape();
    let [output_channels, weight_input, kernel_height, kernel_width] = weight
        .shape
        .as_slice()
        .try_into()
        .map_err(|_| "Invalid Qwen-Drive convolution weight shape")?;
    if weight_input != input_channels
        || bias.is_some_and(|bias| bias.shape.as_slice() != [output_channels])
    {
        return Err("Invalid Qwen-Drive convolution channels".into());
    }
    let input_stride = input_channels * input_height * input_width;
    let mut output = Vec::new();
    let mut output_shape = None;
    for values in input.values().chunks_exact(input_stride).take(batch) {
        let (mut values, shape) = conv2d(
            pool,
            values,
            [input_channels, input_height, input_width],
            &weight.values,
            [output_channels, weight_input, kernel_height, kernel_width],
            bias.map(|bias| bias.values.as_slice()),
            [1, 1],
            [padding, padding],
            [dilation, dilation],
            1,
        )?;
        values
            .iter_mut()
            .for_each(|value| *value = round_bf16(*value));
        output.extend(values);
        output_shape = Some(shape);
    }
    let [channels, height, width] = output_shape.ok_or("Empty convolution batch")?;
    Tensor4::new(output, [batch, channels, height, width])
}

fn group_norm_2d_bf16(
    input: &Tensor4,
    weight: &F32Tensor,
    bias: &F32Tensor,
    groups: usize,
    epsilon: f32,
) -> Result<Tensor4, String> {
    let [batch, channels, height, width] = input.shape();
    if weight.shape.as_slice() != [channels] || bias.shape.as_slice() != [channels] {
        return Err("Invalid Qwen-Drive GroupNorm weights".into());
    }
    let stride = channels * height * width;
    let mut output = Vec::with_capacity(input.values().len());
    for values in input.values().chunks_exact(stride).take(batch) {
        let mut values = group_norm_ncthw(
            values,
            [channels, 1, height, width],
            groups,
            &weight.values,
            &bias.values,
            epsilon,
        )?;
        values
            .iter_mut()
            .for_each(|value| *value = round_bf16(*value));
        output.extend(values);
    }
    Tensor4::new(output, input.shape())
}

fn relu(input: Tensor4) -> Tensor4 {
    Tensor4::new(
        input.values().iter().map(|&value| value.max(0.0)).collect(),
        input.shape(),
    )
    .unwrap()
}

fn add_relu_bf16(left: &Tensor4, right: &Tensor4) -> Result<Tensor4, String> {
    if left.shape() != right.shape() {
        return Err("Qwen-Drive residual shapes differ".into());
    }
    Tensor4::new(
        left.values()
            .iter()
            .zip(right.values())
            .map(|(&left, &right)| round_bf16(left + right).max(0.0))
            .collect(),
        left.shape(),
    )
}

fn adaptive_average_pool_1x1(input: &Tensor4) -> Result<Tensor4, String> {
    let [batch, channels, height, width] = input.shape();
    let plane = height * width;
    let mut output = Vec::with_capacity(batch * channels);
    for values in input.values().chunks_exact(plane) {
        output.push(round_bf16(
            values.iter().copied().sum::<f32>() / plane as f32,
        ));
    }
    Tensor4::new(output, [batch, channels, 1, 1])
}

fn concatenate_channels(inputs: &[Tensor4]) -> Result<Tensor4, String> {
    let [batch, _, height, width] = inputs
        .first()
        .ok_or("Cannot concatenate an empty tensor list")?
        .shape();
    if inputs.iter().any(|input| {
        [input.shape()[0], input.shape()[2], input.shape()[3]] != [batch, height, width]
    }) {
        return Err("Qwen-Drive ASPP shapes differ".into());
    }
    let channels = inputs.iter().map(|input| input.shape()[1]).sum();
    let mut output = Vec::with_capacity(batch * channels * height * width);
    for n in 0..batch {
        for input in inputs {
            let stride = input.shape()[1] * height * width;
            output.extend_from_slice(&input.values()[n * stride..(n + 1) * stride]);
        }
    }
    Tensor4::new(output, [batch, channels, height, width])
}

pub struct ViewTransform {
    weights: ComponentWeights,
}

impl ViewTransform {
    fn tensor(&self, suffix: &str) -> Result<&F32Tensor, String> {
        let name = format!("bev_modeling.{suffix}");
        self.weights
            .tensors
            .get(&name)
            .ok_or_else(|| format!("Missing perception tensor: {name}"))
    }

    fn conv3d(
        &self,
        input: &[f32],
        shape: [usize; 5],
        index: usize,
        pool: &ComputePool,
    ) -> Result<(Vec<f32>, [usize; 5]), String> {
        let weight = self.tensor(&format!("view_trans.conv_layer.{index}.0.weight"))?;
        let bias = self.tensor(&format!("view_trans.conv_layer.{index}.0.bias"))?;
        let weight_shape: [usize; 5] = weight
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| "Invalid Qwen-Drive 3D convolution weight shape")?;
        let [batch, channels, depth, height, width] = shape;
        if weight_shape[1] != channels || bias.shape.as_slice() != [weight_shape[0]] {
            return Err("Invalid Qwen-Drive 3D convolution channels".into());
        }
        let stride = channels * depth * height * width;
        let mut output = Vec::new();
        let mut output_shape = None;
        for values in input.chunks_exact(stride).take(batch) {
            let (mut values, shape) = conv3d_with_options(
                pool,
                values,
                [channels, depth, height, width],
                &weight.values,
                weight_shape,
                Some(&bias.values),
                Conv3dSpec {
                    padding: [1, 1, 1],
                    ..Conv3dSpec::default()
                },
            )?;
            values
                .iter_mut()
                .for_each(|value| *value = round_bf16(*value));
            output.extend(values);
            output_shape = Some(shape);
        }
        let [channels, depth, height, width] = output_shape.ok_or("Empty 3D convolution batch")?;
        Ok((output, [batch, channels, depth, height, width]))
    }

    fn batch_norm_relu(
        &self,
        mut input: Vec<f32>,
        shape: [usize; 5],
        index: usize,
    ) -> Result<Vec<f32>, String> {
        let weight = self.tensor(&format!("view_trans.conv_layer.{index}.1.weight"))?;
        let bias = self.tensor(&format!("view_trans.conv_layer.{index}.1.bias"))?;
        let mean = self.tensor(&format!("view_trans.conv_layer.{index}.1.running_mean"))?;
        let variance = self.tensor(&format!("view_trans.conv_layer.{index}.1.running_var"))?;
        let channels = shape[1];
        if [weight, bias, mean, variance]
            .iter()
            .any(|tensor| tensor.shape.as_slice() != [channels])
        {
            return Err("Invalid Qwen-Drive BatchNorm3d weights".into());
        }
        let plane = shape[2] * shape[3] * shape[4];
        for batch in 0..shape[0] {
            for channel in 0..channels {
                let inverse = 1.0 / (variance.values[channel] + 1e-5).sqrt();
                let start = (batch * channels + channel) * plane;
                for value in &mut input[start..start + plane] {
                    *value = round_bf16(
                        ((*value - mean.values[channel]) * inverse * weight.values[channel]
                            + bias.values[channel])
                            .max(0.0),
                    );
                }
            }
        }
        Ok(input)
    }

    pub fn forward(
        &self,
        image_features: &Tensor4,
        image_depth: &Tensor4,
        geometry: &ViewGeometry<'_>,
        pool: &ComputePool,
    ) -> Result<(Vec<f32>, [usize; 5], Vec<f32>), String> {
        let batch = geometry.lidar2ego.len();
        if batch == 0 || geometry.lidar2img.len() % batch != 0 {
            return Err("Invalid Qwen-Drive camera geometry".into());
        }
        let cameras = geometry.lidar2img.len() / batch;
        let [images, channels, height, width] = image_features.shape();
        let [depth_images, depth, depth_height, depth_width] = image_depth.shape();
        let depth_bins = axis(
            geometry.frustum_range[2],
            geometry.frustum_range[5],
            geometry.frustum_size[2],
        )?
        .len();
        if images != batch * cameras
            || depth_images != images
            || [depth_height, depth_width] != [height, width]
            || depth != depth_bins
        {
            return Err("Qwen-Drive view feature shapes do not match geometry".into());
        }
        let (coords, point_indices) = frustum_voxel_coordinates(geometry)?;
        let [x, y, z] = geometry.voxel_shape;
        let pooled = voxel_pool_depth(&VoxelPoolInput {
            img_feats: image_features.values(),
            img_depth: image_depth.values(),
            coords: &coords,
            point_indices: &point_indices,
            batch,
            sweeps: 1,
            cameras,
            x,
            y,
            z,
            depth,
            height,
            width,
            channels,
        })?;

        let mut voxel = vec![0.0f32; batch * channels * z * y * x];
        for batch_index in 0..batch {
            for channel in 0..channels {
                for depth_index in 0..z {
                    for row in 0..y {
                        for column in 0..x {
                            let mut sum = 0.0f32;
                            for camera in 0..cameras {
                                let source =
                                    (((((batch_index * cameras + camera) * x + column) * y + row)
                                        * z
                                        + depth_index)
                                        * channels)
                                        + channel;
                                sum += pooled[source];
                            }
                            let destination =
                                ((((batch_index * channels + channel) * z + depth_index) * y
                                    + row)
                                    * x)
                                    + column;
                            voxel[destination] = round_bf16(sum);
                        }
                    }
                }
            }
        }
        let mut voxel_shape = [batch, channels, z, y, x];
        for index in 0..3 {
            (voxel, voxel_shape) = self.conv3d(&voxel, voxel_shape, index, pool)?;
            voxel = self.batch_norm_relu(voxel, voxel_shape, index)?;
        }

        let weight = self.tensor("uvtr_query_proj.weight")?;
        let bias = self.tensor("uvtr_query_proj.bias")?;
        let [output_channels, input_channels, kernel_height, kernel_width]: [usize; 4] = weight
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| "Invalid Qwen-Drive UVTR weight shape")?;
        if [kernel_height, kernel_width] != [1, 1] || bias.shape.as_slice() != [output_channels] {
            return Err("Invalid Qwen-Drive UVTR projection".into());
        }
        let bev = voxel_to_bev_tokens(
            &voxel,
            voxel_shape,
            &weight.values,
            [output_channels, input_channels],
            &bias.values,
        )?;
        Ok((voxel, voxel_shape, bev))
    }
}

pub struct ViewOutput {
    pub llm_levels: Vec<Tensor4>,
    pub voxel: Vec<f32>,
    pub voxel_shape: [usize; 5],
    pub bev_tokens: Vec<f32>,
}

pub struct ViewBackbone {
    config: PerceptionConfig,
    adaptor: SimpleFpn,
    vit_neck: SimpleFpn,
    depth: DepthNet,
    view: ViewTransform,
}

impl ViewBackbone {
    pub fn from_source<S: TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        let config = PerceptionConfig::from_source(source)?;
        let contracts = perception_contracts()?;
        Ok(Self {
            config,
            adaptor: SimpleFpn::load(source, &contracts, "bev_modeling.adaptor.")?,
            vit_neck: SimpleFpn::load(source, &contracts, "bev_modeling.vit_neck.")?,
            depth: DepthNet {
                weights: ComponentWeights::load(source, &contracts, &["bev_modeling.depth_net."])?,
            },
            view: ViewTransform {
                weights: ComponentWeights::load(
                    source,
                    &contracts,
                    &["bev_modeling.view_trans.", "bev_modeling.uvtr_query_proj."],
                )?,
            },
        })
    }

    pub fn config(&self) -> &PerceptionConfig {
        &self.config
    }

    pub fn forward(
        &self,
        vit_features: &Tensor4,
        llm_features: &Tensor4,
        geometry: &ViewGeometry<'_>,
        pool: &ComputePool,
    ) -> Result<ViewOutput, String> {
        let llm_levels = self
            .adaptor
            .forward(llm_features, &[4.0, 2.0, 1.0, 0.5], pool)?;
        let vit_level = self
            .vit_neck
            .forward(vit_features, &[1.0], pool)?
            .pop()
            .ok_or("Qwen-Drive ViT neck produced no feature level")?;
        let depth = depth_softmax(&self.depth.forward(&vit_level, pool)?)?;
        let (voxel, voxel_shape, bev_tokens) =
            self.view.forward(&vit_level, &depth, geometry, pool)?;
        Ok(ViewOutput {
            llm_levels,
            voxel,
            voxel_shape,
            bev_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::thread_pool::ComputePool;

    #[derive(Deserialize)]
    struct FixtureRoot {
        fpn: FpnFixture,
        depth_net: DepthNetFixture,
        view_transform: ViewTransformFixture,
        geometry: GeometryFixture,
    }

    #[derive(Deserialize)]
    struct FpnFixture {
        input: TensorFixture,
        weights: Vec<WeightFixture>,
        outputs: Vec<TensorFixture>,
    }

    #[derive(Deserialize)]
    struct DepthNetFixture {
        input: TensorFixture,
        weights: Vec<WeightFixture>,
        logits: TensorFixture,
        probabilities: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct ViewTransformFixture {
        features: TensorFixture,
        depth: TensorFixture,
        weights: Vec<WeightFixture>,
        voxel: Tensor5Fixture,
        bev: FlatTensorFixture,
    }

    #[derive(Deserialize)]
    struct Tensor5Fixture {
        shape: [usize; 5],
        values: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct FlatTensorFixture {
        shape: Vec<usize>,
        values: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct GeometryFixture {
        frustum_range: [u32; 6],
        frustum_size: [u32; 3],
        pc_range: [u32; 6],
        voxel_size: [u32; 3],
        voxel_shape: [usize; 3],
        lidar2img: Vec<[u32; 16]>,
        lidar2ego: Vec<[u32; 16]>,
    }

    #[derive(Deserialize)]
    struct TensorFixture {
        shape: [usize; 4],
        values: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct WeightFixture {
        name: String,
        shape: Vec<usize>,
        values: Vec<u32>,
    }

    #[test]
    fn simple_fpn_matches_official_bf16_words() {
        let fixture: FixtureRoot = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen_drive/perception-view.json"
        )))
        .unwrap();
        let tensors = fixture
            .fpn
            .weights
            .into_iter()
            .map(|weight| {
                (
                    weight.name,
                    F32Tensor {
                        shape: weight.shape,
                        values: weight.values.into_iter().map(f32::from_bits).collect(),
                    },
                )
            })
            .collect();
        let model = SimpleFpn {
            prefix: String::new(),
            weights: ComponentWeights { tensors },
        };
        let input = Tensor4::new(
            fixture
                .fpn
                .input
                .values
                .into_iter()
                .map(f32::from_bits)
                .collect(),
            fixture.fpn.input.shape,
        )
        .unwrap();

        let output = model
            .forward(&input, &[4.0, 2.0, 1.0, 0.5], &ComputePool::new(2))
            .unwrap();

        assert_eq!(output.len(), fixture.fpn.outputs.len());
        for (level, (actual, expected)) in output.iter().zip(&fixture.fpn.outputs).enumerate() {
            assert_eq!(actual.shape(), expected.shape, "level {level} shape");
            assert_eq!(
                actual
                    .values()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected.values,
                "level {level} words"
            );
        }
    }

    #[test]
    fn depth_net_matches_official_bf16_words() {
        let fixture: FixtureRoot = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen_drive/perception-view.json"
        )))
        .unwrap();
        let tensors = fixture
            .depth_net
            .weights
            .into_iter()
            .map(|weight| {
                (
                    format!("bev_modeling.depth_net.{}", weight.name),
                    F32Tensor {
                        shape: weight.shape,
                        values: weight.values.into_iter().map(f32::from_bits).collect(),
                    },
                )
            })
            .collect();
        let model = DepthNet {
            weights: ComponentWeights { tensors },
        };
        let input = Tensor4::new(
            fixture
                .depth_net
                .input
                .values
                .into_iter()
                .map(f32::from_bits)
                .collect(),
            fixture.depth_net.input.shape,
        )
        .unwrap();

        let logits = model.forward(&input, &ComputePool::new(2)).unwrap();
        assert_eq!(logits.shape(), fixture.depth_net.logits.shape);
        assert_eq!(
            logits
                .values()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            fixture.depth_net.logits.values,
            "depth logits"
        );
        assert_eq!(
            depth_softmax(&logits)
                .unwrap()
                .values()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            fixture.depth_net.probabilities,
            "depth probabilities"
        );
    }

    #[test]
    fn view_transform_matches_official_bf16_words() {
        let fixture: FixtureRoot = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen_drive/perception-view.json"
        )))
        .unwrap();
        let tensors = fixture
            .view_transform
            .weights
            .into_iter()
            .map(|weight| {
                (
                    format!("bev_modeling.{}", weight.name),
                    F32Tensor {
                        shape: weight.shape,
                        values: weight.values.into_iter().map(f32::from_bits).collect(),
                    },
                )
            })
            .collect();
        let model = ViewTransform {
            weights: ComponentWeights { tensors },
        };
        let features = Tensor4::new(
            fixture
                .view_transform
                .features
                .values
                .into_iter()
                .map(f32::from_bits)
                .collect(),
            fixture.view_transform.features.shape,
        )
        .unwrap();
        let depth = Tensor4::new(
            fixture
                .view_transform
                .depth
                .values
                .into_iter()
                .map(f32::from_bits)
                .collect(),
            fixture.view_transform.depth.shape,
        )
        .unwrap();
        let lidar2img = fixture
            .geometry
            .lidar2img
            .iter()
            .map(|matrix| matrix.map(f32::from_bits))
            .collect::<Vec<_>>();
        let lidar2ego = fixture
            .geometry
            .lidar2ego
            .iter()
            .map(|matrix| matrix.map(f32::from_bits))
            .collect::<Vec<_>>();
        let geometry = ViewGeometry {
            frustum_range: fixture.geometry.frustum_range.map(f32::from_bits),
            frustum_size: fixture.geometry.frustum_size.map(f32::from_bits),
            pc_range: fixture.geometry.pc_range.map(f32::from_bits),
            voxel_size: fixture.geometry.voxel_size.map(f32::from_bits),
            voxel_shape: fixture.geometry.voxel_shape,
            lidar2img: &lidar2img,
            lidar2ego: &lidar2ego,
        };

        let (voxel, voxel_shape, bev) = model
            .forward(&features, &depth, &geometry, &ComputePool::new(2))
            .unwrap();
        assert_eq!(voxel_shape, fixture.view_transform.voxel.shape);
        assert_eq!(
            voxel
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            fixture.view_transform.voxel.values,
            "view voxel"
        );
        assert_eq!(vec![bev.len() / 3, 3], fixture.view_transform.bev.shape);
        assert_eq!(
            bev.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            fixture.view_transform.bev.values,
            "view BEV"
        );
    }
}
