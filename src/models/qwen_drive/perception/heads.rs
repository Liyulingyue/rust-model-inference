use super::bev::{layer_norm_rows, refine_reference, sigmoid, BevFormer, BevOutput, Linear};
use super::fpn::{inverse_4x4, perception_contracts, ComponentWeights, F32Tensor};
use super::ops::{grid_sample_bilinear_aligned, resize_bilinear_aligned, Tensor4};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::models::diffusion::dreamx::kernels::{
    conv2d, conv3d_with_options, group_norm_ncthw, Conv3dSpec,
};
use crate::models::qwen_drive::config::PerceptionConfig;
use serde::Serialize;

struct ClassificationBranch<'a> {
    first: Linear<'a>,
    second: Linear<'a>,
    output: Linear<'a>,
}

pub struct PerceptionHeads<'a> {
    config: PerceptionConfig,
    weights: ComponentWeights,
    classification: Vec<ClassificationBranch<'a>>,
    occ_first: Linear<'a>,
    occ_output: Linear<'a>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Detection {
    pub boxes: Vec<[f32; 9]>,
    pub scores: Vec<f32>,
    pub labels: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Occupancy {
    pub shape: [usize; 3],
    pub values: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MapSegmentation {
    pub shape: [usize; 2],
    pub values: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerceptionResult {
    pub detections: Detection,
    pub occupancy: Occupancy,
    pub map: MapSegmentation,
}

#[derive(Clone)]
struct Tensor5 {
    values: Vec<f32>,
    shape: [usize; 5],
}

impl Tensor5 {
    fn new(values: Vec<f32>, shape: [usize; 5]) -> Result<Self, String> {
        let expected = shape.iter().try_fold(1usize, |size, &dimension| {
            size.checked_mul(dimension)
                .ok_or_else(|| "Qwen-Drive volume shape overflow".to_string())
        })?;
        if shape.contains(&0)
            || values.len() != expected
            || values.iter().any(|value| !value.is_finite())
        {
            return Err("Invalid Qwen-Drive volume".into());
        }
        Ok(Self { values, shape })
    }
}

fn sample_axis(position: usize, input: usize, output: usize, align_corners: bool) -> f32 {
    if align_corners && output > 1 {
        position as f32 * (input - 1) as f32 / (output - 1) as f32
    } else {
        (position as f32 + 0.5) * input as f32 / output as f32 - 0.5
    }
}

fn resize_trilinear(
    input: &Tensor5,
    output_shape: [usize; 3],
    align_corners: bool,
) -> Result<Tensor5, String> {
    let [batch, channels, input_depth, input_height, input_width] = input.shape;
    let [output_depth, output_height, output_width] = output_shape;
    if output_shape.contains(&0) {
        return Err("Invalid Qwen-Drive trilinear output shape".into());
    }
    let mut output = vec![0.0; batch * channels * output_depth * output_height * output_width];
    for n in 0..batch {
        for channel in 0..channels {
            for z in 0..output_depth {
                let source_z = sample_axis(z, input_depth, output_depth, align_corners);
                let z0 = source_z.floor() as isize;
                let z1 = z0 + 1;
                let fz = source_z - z0 as f32;
                for y in 0..output_height {
                    let source_y = sample_axis(y, input_height, output_height, align_corners);
                    let y0 = source_y.floor() as isize;
                    let y1 = y0 + 1;
                    let fy = source_y - y0 as f32;
                    for x in 0..output_width {
                        let source_x = sample_axis(x, input_width, output_width, align_corners);
                        let x0 = source_x.floor() as isize;
                        let x1 = x0 + 1;
                        let fx = source_x - x0 as f32;
                        let mut value = 0.0f32;
                        for (source_z, wz) in [(z0, 1.0 - fz), (z1, fz)] {
                            let source_z = source_z.clamp(0, input_depth as isize - 1) as usize;
                            for (source_y, wy) in [(y0, 1.0 - fy), (y1, fy)] {
                                let source_y =
                                    source_y.clamp(0, input_height as isize - 1) as usize;
                                for (source_x, wx) in [(x0, 1.0 - fx), (x1, fx)] {
                                    let source_x =
                                        source_x.clamp(0, input_width as isize - 1) as usize;
                                    let source = ((((n * channels + channel) * input_depth
                                        + source_z)
                                        * input_height
                                        + source_y)
                                        * input_width)
                                        + source_x;
                                    value += input.values[source] * wz * wy * wx;
                                }
                            }
                        }
                        let destination =
                            ((((n * channels + channel) * output_depth + z) * output_height + y)
                                * output_width)
                                + x;
                        output[destination] = super::ops::round_bf16(value);
                    }
                }
            }
        }
    }
    Tensor5::new(
        output,
        [batch, channels, output_depth, output_height, output_width],
    )
}

fn concatenate_volumes(left: &Tensor5, right: &Tensor5) -> Result<Tensor5, String> {
    if left.shape[0] != right.shape[0] || left.shape[2..] != right.shape[2..] {
        return Err("Qwen-Drive volume shapes cannot be concatenated".into());
    }
    let plane = left.shape[2] * left.shape[3] * left.shape[4];
    let mut values = Vec::with_capacity(left.values.len() + right.values.len());
    for batch in 0..left.shape[0] {
        let left_start = batch * left.shape[1] * plane;
        values.extend_from_slice(&left.values[left_start..left_start + left.shape[1] * plane]);
        let right_start = batch * right.shape[1] * plane;
        values.extend_from_slice(&right.values[right_start..right_start + right.shape[1] * plane]);
    }
    Tensor5::new(
        values,
        [
            left.shape[0],
            left.shape[1] + right.shape[1],
            left.shape[2],
            left.shape[3],
            left.shape[4],
        ],
    )
}

fn sample_volume_at(
    input: &Tensor5,
    xs: &[f32],
    ys: &[f32],
    zs: &[f32],
) -> Result<Tensor5, String> {
    if xs.is_empty() || ys.is_empty() || zs.is_empty() {
        return Err("Invalid Qwen-Drive occupancy grid".into());
    }
    let mut output = vec![0.0; input.shape[0] * input.shape[1] * zs.len() * ys.len() * xs.len()];
    for batch in 0..input.shape[0] {
        for channel in 0..input.shape[1] {
            for (z, &source_z) in zs.iter().enumerate() {
                let z0 = source_z.floor() as usize;
                let z1 = (z0 + 1).min(input.shape[2] - 1);
                let fz = source_z - z0 as f32;
                for (y, &source_y) in ys.iter().enumerate() {
                    let y0 = source_y.floor() as usize;
                    let y1 = (y0 + 1).min(input.shape[3] - 1);
                    let fy = source_y - y0 as f32;
                    for (x, &source_x) in xs.iter().enumerate() {
                        let x0 = source_x.floor() as usize;
                        let x1 = (x0 + 1).min(input.shape[4] - 1);
                        let fx = source_x - x0 as f32;
                        let mut value = 0.0f32;
                        for (source_z, wz) in [(z0, 1.0 - fz), (z1, fz)] {
                            for (source_y, wy) in [(y0, 1.0 - fy), (y1, fy)] {
                                for (source_x, wx) in [(x0, 1.0 - fx), (x1, fx)] {
                                    let source = ((((batch * input.shape[1] + channel)
                                        * input.shape[2]
                                        + source_z)
                                        * input.shape[3]
                                        + source_y)
                                        * input.shape[4])
                                        + source_x;
                                    value += input.values[source] * wz * wy * wx;
                                }
                            }
                        }
                        let destination =
                            ((((batch * input.shape[1] + channel) * zs.len() + z) * ys.len() + y)
                                * xs.len())
                                + x;
                        output[destination] = super::ops::round_bf16(value);
                    }
                }
            }
        }
    }
    Tensor5::new(
        output,
        [input.shape[0], input.shape[1], zs.len(), ys.len(), xs.len()],
    )
}

fn adapt_occ_volume(
    input: &Tensor5,
    source_range: [f32; 6],
    target_range: [f32; 6],
    target_voxel: [f32; 3],
    target_depth: usize,
) -> Result<Tensor5, String> {
    let target_width = ((target_range[3] - target_range[0]) / target_voxel[0]).round() as usize;
    let target_height = ((target_range[4] - target_range[1]) / target_voxel[1]).round() as usize;
    if target_width == 0 || target_height == 0 || target_depth == 0 {
        return Err("Invalid Qwen-Drive occupancy target".into());
    }
    let source_ranges = [
        (source_range[0], source_range[3]),
        (source_range[1], source_range[4]),
        (source_range[2], source_range[5]),
    ];
    let target_ranges = [
        (target_range[0], target_range[3]),
        (target_range[1], target_range[4]),
        (target_range[2], target_range[5]),
    ];
    let source_sizes = [input.shape[4], input.shape[3], input.shape[2]];
    let target_sizes = [target_width, target_height, target_depth];
    let coordinates = (0..3)
        .map(|axis| {
            let (source_min, source_max) = source_ranges[axis];
            let (target_min, target_max) = target_ranges[axis];
            let source_step = (source_max - source_min) / source_sizes[axis] as f32;
            let source_center = source_min + 0.5 * source_step;
            let target_step = (target_max - target_min) / target_sizes[axis] as f32;
            (0..target_sizes[axis])
                .map(|index| {
                    let center = target_min + (index as f32 + 0.5) * target_step;
                    ((center - source_center) / source_step)
                        .clamp(0.0, source_sizes[axis].saturating_sub(1) as f32)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    sample_volume_at(input, &coordinates[0], &coordinates[1], &coordinates[2])
}

fn map_crop_grid(
    start: &[f32],
    xbound: [f32; 3],
    ybound: [f32; 3],
) -> Result<(Vec<f32>, [usize; 2]), String> {
    if start.len() != 3 || start[0] == 0.0 || start[1] == 0.0 {
        return Err("Invalid Qwen-Drive map crop position".into());
    }
    let [x0, x1, dx] = xbound;
    let [y0, y1, dy] = ybound;
    let width = ((x1 - x0) / dx).round() as usize;
    let height = ((y1 - y0) / dy).round() as usize;
    let mut grid = Vec::with_capacity(height * width * 2);
    for y in 0..height {
        let normalized_y = super::ops::round_bf16((y0 + dy * 0.5 + y as f32 * dy) / -start[1]);
        for x in 0..width {
            grid.push(super::ops::round_bf16(
                (x0 + dx * 0.5 + x as f32 * dx) / -start[0],
            ));
            grid.push(normalized_y);
        }
    }
    Ok((grid, [height, width]))
}

fn softplus_inplace(values: &mut [f32]) {
    values.iter_mut().for_each(|value| {
        *value = super::ops::round_bf16(if *value > 20.0 {
            *value
        } else {
            (1.0 + value.exp()).ln()
        })
    });
}

fn argmax_rows(values: &[f32], classes: usize) -> Result<Vec<u8>, String> {
    if classes == 0 || values.len() % classes != 0 {
        return Err("Invalid Qwen-Drive occupancy logits".into());
    }
    Ok(values
        .chunks_exact(classes)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .expect("non-empty class row")
                .0 as u8
        })
        .collect())
}

fn exported(suffix: &str) -> String {
    format!("qwen_drive_perception.head.{suffix}")
}

fn apply_detection_references(
    coordinates: &mut [f32],
    references: &[f32],
    queries: usize,
    code_size: usize,
    pc_range: [f32; 6],
) -> Result<(), String> {
    if coordinates.len() != queries * code_size || references.len() != queries * 3 || code_size < 5
    {
        return Err("Invalid Qwen-Drive detection reference tensors".into());
    }
    for query in 0..queries {
        let coord = &mut coordinates[query * code_size..(query + 1) * code_size];
        coord[0] = refine_reference(references[query * 3], coord[0]);
        coord[1] = refine_reference(references[query * 3 + 1], coord[1]);
        coord[4] = refine_reference(references[query * 3 + 2], coord[4]);
        for (index, min, max) in [
            (0, pc_range[0], pc_range[3]),
            (1, pc_range[1], pc_range[4]),
            (4, pc_range[2], pc_range[5]),
        ] {
            coord[index] = super::ops::round_bf16(
                super::ops::round_bf16(coord[index] * (max - min)) + super::ops::round_bf16(min),
            );
        }
    }
    Ok(())
}

impl<'a> PerceptionHeads<'a> {
    pub fn from_source<S: TensorSource + ?Sized>(source: &'a S) -> Result<Self, String> {
        let config = PerceptionConfig::from_source(source)?;
        let contracts = perception_contracts()?;
        let weights = ComponentWeights::load(
            source,
            &contracts,
            &[
                "bev_modeling.head.cls_branches.",
                "bev_modeling.head.seg_decoder.",
                "bev_modeling.head.transformer.occ_",
                "bev_modeling.head.transformer.uvtr_occ_",
                "bev_modeling.head.feat_cropper.",
            ],
        )?;
        let mut classification = Vec::with_capacity(config.decoder_layers);
        for layer in 0..config.decoder_layers {
            let root = exported(&format!("cls_branches.{layer}"));
            classification.push(ClassificationBranch {
                first: Linear::load(
                    source,
                    &format!("{root}.0"),
                    config.embed_dim,
                    config.embed_dim,
                )?,
                second: Linear::load(
                    source,
                    &format!("{root}.3"),
                    config.embed_dim,
                    config.embed_dim,
                )?,
                output: Linear::load(
                    source,
                    &format!("{root}.6"),
                    config.embed_dim,
                    config.det_classes,
                )?,
            });
        }
        Ok(Self {
            occ_first: Linear::load(
                source,
                &exported("transformer.occ_pred_head.0"),
                config.occ_dim,
                config.occ_dim * 2,
            )?,
            occ_output: Linear::load(
                source,
                &exported("transformer.occ_pred_head.2"),
                config.occ_dim * 2,
                config.occ_classes,
            )?,
            config,
            weights,
            classification,
        })
    }

    fn tensor(&self, suffix: &str) -> Result<&F32Tensor, String> {
        let name = format!("bev_modeling.head.{suffix}");
        self.weights
            .tensors
            .get(&name)
            .ok_or_else(|| format!("Missing perception tensor: {name}"))
    }

    fn classification(
        &self,
        layer: usize,
        input: &[f32],
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        let branch = self
            .classification
            .get(layer)
            .ok_or("Invalid Qwen-Drive classification layer")?;
        let rows = self.config.num_queries;
        let root = format!("cls_branches.{layer}");
        let mut output = branch.first.forward(input, rows, pool)?;
        output = layer_norm_rows(
            &output,
            rows,
            self.config.embed_dim,
            self.tensor(&format!("{root}.1.weight"))?,
            self.tensor(&format!("{root}.1.bias"))?,
        )?;
        output.iter_mut().for_each(|value| *value = value.max(0.0));
        output = branch.second.forward(&output, rows, pool)?;
        output = layer_norm_rows(
            &output,
            rows,
            self.config.embed_dim,
            self.tensor(&format!("{root}.4.weight"))?,
            self.tensor(&format!("{root}.4.bias"))?,
        )?;
        output.iter_mut().for_each(|value| *value = value.max(0.0));
        branch.output.forward(&output, rows, pool)
    }

    pub fn detections(
        &self,
        transformer: &BevFormer<'_>,
        output: &BevOutput,
        lidar2ego: &[f32; 16],
        box_coord_system_ego: bool,
        pool: &ComputePool,
    ) -> Result<Detection, String> {
        if output.decoder_states.len() != self.config.decoder_layers
            || output.decoder_references.len() != self.config.decoder_layers
        {
            return Err("Invalid Qwen-Drive decoder output".into());
        }
        let final_layer = self.config.decoder_layers - 1;
        let classes =
            self.classification(final_layer, &output.decoder_states[final_layer], pool)?;
        let mut coordinates =
            transformer.regression(final_layer, &output.decoder_states[final_layer], pool)?;
        let references = if final_layer == 0 {
            &output.initial_references
        } else {
            &output.decoder_references[final_layer - 1]
        };
        apply_detection_references(
            &mut coordinates,
            references,
            self.config.num_queries,
            self.config.code_size,
            self.config.det_pc_range,
        )?;
        decode_detections(
            &classes,
            &coordinates,
            self.config.num_queries,
            self.config.det_classes,
            self.config.code_size,
            lidar2ego,
            box_coord_system_ego,
        )
    }

    fn conv3d(
        &self,
        input: &Tensor5,
        suffix: &str,
        stride: [usize; 3],
        padding: [usize; 3],
        bias: bool,
        pool: &ComputePool,
    ) -> Result<Tensor5, String> {
        let weight = self.tensor(&format!("{suffix}.weight"))?;
        let weight_shape: [usize; 5] = weight
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| format!("Invalid Qwen-Drive Conv3d weight: {suffix}"))?;
        if weight_shape[1] != input.shape[1] {
            return Err(format!("Invalid Qwen-Drive Conv3d channels: {suffix}"));
        }
        let bias = if bias {
            Some(self.tensor(&format!("{suffix}.bias"))?)
        } else {
            None
        };
        let per_batch = input.shape[1] * input.shape[2] * input.shape[3] * input.shape[4];
        let mut values = Vec::new();
        let mut shape = None;
        for batch in input.values.chunks_exact(per_batch) {
            let (mut output, current) = conv3d_with_options(
                pool,
                batch,
                [
                    input.shape[1],
                    input.shape[2],
                    input.shape[3],
                    input.shape[4],
                ],
                &weight.values,
                weight_shape,
                bias.map(|bias| bias.values.as_slice()),
                Conv3dSpec {
                    stride,
                    padding,
                    ..Conv3dSpec::default()
                },
            )?;
            output
                .iter_mut()
                .for_each(|value| *value = super::ops::round_bf16(*value));
            values.extend(output);
            shape = Some(current);
        }
        let [channels, depth, height, width] =
            shape.ok_or("Qwen-Drive Conv3d received an empty batch")?;
        Tensor5::new(values, [input.shape[0], channels, depth, height, width])
    }

    fn batch_norm3d(
        &self,
        mut input: Tensor5,
        suffix: &str,
        relu: bool,
    ) -> Result<Tensor5, String> {
        let channels = input.shape[1];
        let weight = self.tensor(&format!("{suffix}.weight"))?;
        let bias = self.tensor(&format!("{suffix}.bias"))?;
        let mean = self.tensor(&format!("{suffix}.running_mean"))?;
        let variance = self.tensor(&format!("{suffix}.running_var"))?;
        if [weight, bias, mean, variance]
            .iter()
            .any(|tensor| tensor.shape.as_slice() != [channels])
        {
            return Err(format!("Invalid Qwen-Drive BatchNorm3d: {suffix}"));
        }
        let plane = input.shape[2] * input.shape[3] * input.shape[4];
        for batch in 0..input.shape[0] {
            for channel in 0..channels {
                let inverse = 1.0 / (variance.values[channel] + 1e-5).sqrt();
                let start = (batch * channels + channel) * plane;
                for value in &mut input.values[start..start + plane] {
                    let normalized = (*value - mean.values[channel]) * inverse;
                    *value =
                        super::ops::round_bf16(
                            (normalized * weight.values[channel] + bias.values[channel])
                                .max(if relu { 0.0 } else { f32::NEG_INFINITY }),
                        );
                }
            }
        }
        Ok(input)
    }

    fn block3d(
        &self,
        input: Tensor5,
        root: &str,
        stride: [usize; 3],
        pool: &ComputePool,
    ) -> Result<Tensor5, String> {
        let mut output = self.conv3d(
            &input,
            &format!("{root}.conv1"),
            stride,
            [1; 3],
            false,
            pool,
        )?;
        output = self.batch_norm3d(output, &format!("{root}.norm1"), true)?;
        output = self.conv3d(
            &output,
            &format!("{root}.conv2"),
            [1; 3],
            [1; 3],
            false,
            pool,
        )?;
        output = self.batch_norm3d(output, &format!("{root}.norm2"), false)?;
        let identity = if self
            .weights
            .tensors
            .contains_key(&format!("bev_modeling.head.{root}.downsample.0.weight"))
        {
            let identity = self.conv3d(
                &input,
                &format!("{root}.downsample.0"),
                stride,
                [0; 3],
                false,
                pool,
            )?;
            self.batch_norm3d(identity, &format!("{root}.downsample.1"), false)?
        } else {
            input
        };
        if output.shape != identity.shape {
            return Err(format!("Invalid Qwen-Drive 3D residual shape: {root}"));
        }
        for (value, residual) in output.values.iter_mut().zip(identity.values) {
            *value = super::ops::round_bf16((*value + residual).max(0.0));
        }
        Ok(output)
    }

    fn stage3d(
        &self,
        mut input: Tensor5,
        stage: &str,
        stride: [usize; 3],
        pool: &ComputePool,
    ) -> Result<Tensor5, String> {
        input = self.block3d(input, &format!("{stage}.blocks.0"), stride, pool)?;
        self.block3d(input, &format!("{stage}.blocks.1"), [1; 3], pool)
    }

    fn adapt_occ_volume(
        &self,
        input: &Tensor5,
        range: [f32; 6],
        voxel: [f32; 3],
    ) -> Result<Tensor5, String> {
        adapt_occ_volume(
            input,
            self.config.det_pc_range,
            range,
            voxel,
            self.config.occ_pillar_h,
        )
    }

    pub fn occupancy(
        &self,
        output: &BevOutput,
        uvtr: &[f32],
        uvtr_shape: [usize; 5],
        dataset_type: &str,
        pool: &ComputePool,
    ) -> Result<Occupancy, String> {
        let (range, voxel) = match dataset_type {
            "nuscenes" => (
                self.config.nuscenes_occ_pc_range,
                self.config.nuscenes_occ_voxel_size,
            ),
            "nuplan" => (
                self.config.nuplan_occ_pc_range,
                self.config.nuplan_occ_voxel_size,
            ),
            value => return Err(format!("Unsupported Qwen-Drive dataset type: {value}")),
        };
        let [bev_h, bev_w] = self.config.bev;
        let middle = self.config.embed_dim / self.config.occ_pillar_h;
        let mut bev_values = vec![0.0; output.bev.len()];
        for channel in 0..middle {
            for z in 0..self.config.occ_pillar_h {
                let source_channel = channel * self.config.occ_pillar_h + z;
                for y in 0..bev_h {
                    for x in 0..bev_w {
                        let source = (y * bev_w + x) * self.config.embed_dim + source_channel;
                        let destination =
                            ((channel * self.config.occ_pillar_h + z) * bev_h + y) * bev_w + x;
                        bev_values[destination] = output.bev[source];
                    }
                }
            }
        }
        let bev = Tensor5::new(
            bev_values,
            [1, middle, self.config.occ_pillar_h, bev_h, bev_w],
        )?;
        let mut bev = self.adapt_occ_volume(&bev, range, voxel)?;
        let uvtr = Tensor5::new(uvtr.to_vec(), uvtr_shape)?;
        let uvtr = self.adapt_occ_volume(&uvtr, range, voxel)?;
        let uvtr = self.conv3d(
            &uvtr,
            "transformer.uvtr_occ_proj",
            [1; 3],
            [0; 3],
            true,
            pool,
        )?;
        let fused = concatenate_volumes(&bev, &uvtr)?;
        let mut fused = self.conv3d(
            &fused,
            "transformer.uvtr_occ_fuse.conv",
            [1; 3],
            [0; 3],
            false,
            pool,
        )?;
        fused = self.batch_norm3d(fused, "transformer.uvtr_occ_fuse.bn", true)?;
        for (value, residual) in bev.values.iter_mut().zip(fused.values) {
            *value = super::ops::round_bf16(*value + residual);
        }

        let root = "transformer.occ_decoder";
        let skip0 = self.stage3d(bev, &format!("{root}.input_proj"), [1; 3], pool)?;
        let skip1 = self.stage3d(skip0.clone(), &format!("{root}.enc1"), [1, 2, 2], pool)?;
        let skip2 = self.stage3d(skip1.clone(), &format!("{root}.enc2"), [1, 2, 2], pool)?;
        let mut occ = self.stage3d(skip2.clone(), &format!("{root}.enc3"), [1, 2, 2], pool)?;
        occ = self.stage3d(occ, &format!("{root}.bottleneck"), [1; 3], pool)?;
        occ = resize_trilinear(
            &occ,
            [skip2.shape[2], skip2.shape[3], skip2.shape[4]],
            false,
        )?;
        occ = self.stage3d(
            concatenate_volumes(&occ, &skip2)?,
            &format!("{root}.dec2"),
            [1; 3],
            pool,
        )?;
        occ = resize_trilinear(
            &occ,
            [skip1.shape[2], skip1.shape[3], skip1.shape[4]],
            false,
        )?;
        occ = self.stage3d(
            concatenate_volumes(&occ, &skip1)?,
            &format!("{root}.dec1"),
            [1; 3],
            pool,
        )?;
        occ = resize_trilinear(
            &occ,
            [skip0.shape[2], skip0.shape[3], skip0.shape[4]],
            false,
        )?;
        occ = self.stage3d(
            concatenate_volumes(&occ, &skip0)?,
            &format!("{root}.dec0"),
            [1; 3],
            pool,
        )?;
        occ = self.stage3d(occ, &format!("{root}.out_block"), [1; 3], pool)?;
        occ = self.conv3d(
            &occ,
            &format!("{root}.out_proj.0"),
            [1; 3],
            [1; 3],
            false,
            pool,
        )?;
        occ = self.batch_norm3d(occ, &format!("{root}.out_proj.1"), true)?;

        let positions = occ.shape[2] * occ.shape[3] * occ.shape[4];
        let mut rows = vec![0.0; positions * self.config.occ_dim];
        for x in 0..occ.shape[4] {
            for y in 0..occ.shape[3] {
                for z in 0..occ.shape[2] {
                    let position = (x * occ.shape[3] + y) * occ.shape[2] + z;
                    for channel in 0..self.config.occ_dim {
                        let source =
                            ((channel * occ.shape[2] + z) * occ.shape[3] + y) * occ.shape[4] + x;
                        rows[position * self.config.occ_dim + channel] = occ.values[source];
                    }
                }
            }
        }
        let mut logits = self.occ_first.forward(&rows, positions, pool)?;
        softplus_inplace(&mut logits);
        logits = self.occ_output.forward(&logits, positions, pool)?;
        let values = argmax_rows(&logits, self.config.occ_classes)?;
        Ok(Occupancy {
            shape: [occ.shape[4], occ.shape[3], occ.shape[2]],
            values,
        })
    }

    fn conv2d(
        &self,
        input: &Tensor4,
        suffix: &str,
        stride: [usize; 2],
        padding: [usize; 2],
        bias: bool,
        pool: &ComputePool,
    ) -> Result<Tensor4, String> {
        let weight = self.tensor(&format!("{suffix}.weight"))?;
        let weight_shape: [usize; 4] = weight
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| format!("Invalid Qwen-Drive Conv2d weight: {suffix}"))?;
        let [batch, channels, height, width] = input.shape();
        if weight_shape[1] != channels {
            return Err(format!("Invalid Qwen-Drive Conv2d channels: {suffix}"));
        }
        let bias = if bias {
            Some(self.tensor(&format!("{suffix}.bias"))?)
        } else {
            None
        };
        let per_batch = channels * height * width;
        let mut values = Vec::new();
        let mut shape = None;
        for image in input.values().chunks_exact(per_batch) {
            let (mut output, current) = conv2d(
                pool,
                image,
                [channels, height, width],
                &weight.values,
                weight_shape,
                bias.map(|bias| bias.values.as_slice()),
                stride,
                padding,
                [1; 2],
                1,
            )?;
            output
                .iter_mut()
                .for_each(|value| *value = super::ops::round_bf16(*value));
            values.extend(output);
            shape = Some(current);
        }
        let [channels, height, width] = shape.ok_or("Qwen-Drive Conv2d received an empty batch")?;
        Tensor4::new(values, [batch, channels, height, width])
    }

    fn group_norm2d(&self, input: Tensor4, suffix: &str, relu: bool) -> Result<Tensor4, String> {
        let [batch, channels, height, width] = input.shape();
        let weight = self.tensor(&format!("{suffix}.weight"))?;
        let bias = self.tensor(&format!("{suffix}.bias"))?;
        if weight.shape.as_slice() != [channels] || bias.shape.as_slice() != [channels] {
            return Err(format!("Invalid Qwen-Drive GroupNorm: {suffix}"));
        }
        let stride = channels * height * width;
        let mut values = Vec::with_capacity(input.values().len());
        for image in input.values().chunks_exact(stride) {
            let mut output = group_norm_ncthw(
                image,
                [channels, 1, height, width],
                32.min(channels),
                &weight.values,
                &bias.values,
                1e-5,
            )?;
            output.iter_mut().for_each(|value| {
                *value =
                    super::ops::round_bf16(value.max(if relu { 0.0 } else { f32::NEG_INFINITY }))
            });
            values.extend(output);
        }
        Tensor4::new(values, [batch, channels, height, width])
    }

    fn block2d(
        &self,
        input: Tensor4,
        root: &str,
        stride: [usize; 2],
        pool: &ComputePool,
    ) -> Result<Tensor4, String> {
        let mut output = self.conv2d(
            &input,
            &format!("{root}.conv1"),
            stride,
            [1; 2],
            false,
            pool,
        )?;
        output = self.group_norm2d(output, &format!("{root}.bn1"), true)?;
        output = self.conv2d(
            &output,
            &format!("{root}.conv2"),
            [1; 2],
            [1; 2],
            false,
            pool,
        )?;
        output = self.group_norm2d(output, &format!("{root}.bn2"), false)?;
        let identity = if self
            .weights
            .tensors
            .contains_key(&format!("bev_modeling.head.{root}.downsample.0.weight"))
        {
            let identity = self.conv2d(
                &input,
                &format!("{root}.downsample.0"),
                stride,
                [0; 2],
                false,
                pool,
            )?;
            self.group_norm2d(identity, &format!("{root}.downsample.1"), false)?
        } else {
            input
        };
        if output.shape() != identity.shape() {
            return Err(format!("Invalid Qwen-Drive map residual: {root}"));
        }
        let values = output
            .values()
            .iter()
            .zip(identity.values())
            .map(|(&value, &residual)| super::ops::round_bf16((value + residual).max(0.0)))
            .collect();
        Tensor4::new(values, output.shape())
    }

    pub fn map(&self, output: &BevOutput, pool: &ComputePool) -> Result<MapSegmentation, String> {
        let [bev_h, bev_w] = self.config.bev;
        let mut bev = vec![0.0; output.bev.len()];
        for y in 0..bev_h {
            for x in 0..bev_w {
                for channel in 0..self.config.embed_dim {
                    bev[(channel * bev_h + y) * bev_w + x] =
                        output.bev[(y * bev_w + x) * self.config.embed_dim + channel];
                }
            }
        }
        let bev = Tensor4::new(bev, [1, self.config.embed_dim, bev_h, bev_w])?;
        let start = self.tensor("feat_cropper.bev_start_position")?;
        if start.shape.as_slice() != [3] {
            return Err("Invalid Qwen-Drive map crop position".into());
        }
        let (grid, [output_height, output_width]) = map_crop_grid(
            &start.values,
            self.config.map_xbound,
            self.config.map_ybound,
        )?;
        let mut map =
            grid_sample_bilinear_aligned(&bev, &grid, [1, output_height, output_width], true)?;
        map = self.conv2d(&map, "seg_decoder.conv1", [2; 2], [3; 2], false, pool)?;
        map = self.group_norm2d(map, "seg_decoder.bn1", true)?;
        let mut skip1 = self.block2d(map, "seg_decoder.layer1.0", [1; 2], pool)?;
        skip1 = self.block2d(skip1, "seg_decoder.layer1.1", [1; 2], pool)?;
        map = self.block2d(skip1.clone(), "seg_decoder.layer2.0", [2; 2], pool)?;
        map = self.block2d(map, "seg_decoder.layer2.1", [1; 2], pool)?;
        map = self.block2d(map, "seg_decoder.layer3.0", [2; 2], pool)?;
        map = self.block2d(map, "seg_decoder.layer3.1", [1; 2], pool)?;
        map = resize_bilinear_aligned(&map, skip1.shape()[2], skip1.shape()[3], true)?;
        let [batch, channels, height, width] = map.shape();
        let skip_channels = skip1.shape()[1];
        let plane = height * width;
        let mut joined = Vec::with_capacity(batch * (channels + skip_channels) * plane);
        for n in 0..batch {
            joined.extend_from_slice(
                &skip1.values()[n * skip_channels * plane..(n + 1) * skip_channels * plane],
            );
            joined
                .extend_from_slice(&map.values()[n * channels * plane..(n + 1) * channels * plane]);
        }
        map = Tensor4::new(joined, [batch, channels + skip_channels, height, width])?;
        map = self.conv2d(&map, "seg_decoder.up1.conv.0", [1; 2], [1; 2], false, pool)?;
        map = self.group_norm2d(map, "seg_decoder.up1.conv.1", true)?;
        map = self.conv2d(&map, "seg_decoder.up1.conv.3", [1; 2], [1; 2], false, pool)?;
        map = self.group_norm2d(map, "seg_decoder.up1.conv.4", true)?;
        map = resize_bilinear_aligned(&map, output_height, output_width, true)?;
        map = self.conv2d(&map, "seg_decoder.up2.1", [1; 2], [1; 2], false, pool)?;
        map = self.group_norm2d(map, "seg_decoder.up2.2", true)?;
        map = self.conv2d(&map, "seg_decoder.up2.4", [1; 2], [0; 2], true, pool)?;
        let values = (0..output_height * output_width)
            .map(|position| {
                (0..self.config.map_classes)
                    .max_by(|&left, &right| {
                        map.values()[left * output_height * output_width + position].total_cmp(
                            &map.values()[right * output_height * output_width + position],
                        )
                    })
                    .unwrap() as u8
            })
            .collect();
        Ok(MapSegmentation {
            shape: [output_height, output_width],
            values,
        })
    }

    pub fn forward(
        &self,
        transformer: &BevFormer<'_>,
        output: &BevOutput,
        uvtr: &[f32],
        uvtr_shape: [usize; 5],
        dataset_type: &str,
        lidar2ego: &[f32; 16],
        box_coord_system_ego: bool,
        pool: &ComputePool,
    ) -> Result<PerceptionResult, String> {
        Ok(PerceptionResult {
            detections: self.detections(
                transformer,
                output,
                lidar2ego,
                box_coord_system_ego,
                pool,
            )?,
            occupancy: self.occupancy(output, uvtr, uvtr_shape, dataset_type, pool)?,
            map: self.map(output, pool)?,
        })
    }
}

fn decode_detections(
    classes: &[f32],
    coordinates: &[f32],
    queries: usize,
    classes_count: usize,
    code_size: usize,
    lidar2ego: &[f32; 16],
    box_coord_system_ego: bool,
) -> Result<Detection, String> {
    if classes.len() != queries * classes_count || coordinates.len() != queries * code_size {
        return Err("Invalid Qwen-Drive detection tensors".into());
    }
    let mut ranked = classes
        .iter()
        .enumerate()
        .map(|(index, &score)| (index, sigmoid(score)))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let ego2lidar = box_coord_system_ego
        .then(|| inverse_4x4(lidar2ego))
        .transpose()?;
    let mut boxes = Vec::new();
    let mut scores = Vec::new();
    let mut labels = Vec::new();
    for &(index, score) in ranked.iter().take(300) {
        let query = index / classes_count;
        let values = &coordinates[query * code_size..(query + 1) * code_size];
        let mut bbox = [
            values[0],
            values[1],
            values[4],
            super::ops::round_bf16(values[2].exp()),
            super::ops::round_bf16(values[3].exp()),
            super::ops::round_bf16(values[5].exp()),
            super::ops::round_bf16(values[6].atan2(values[7])),
            values[8],
            values[9],
        ];
        if bbox[0] < -61.2
            || bbox[1] < -61.2
            || bbox[2] < -10.0
            || bbox[0] > 61.2
            || bbox[1] > 61.2
            || bbox[2] > 10.0
        {
            continue;
        }
        if let Some(matrix) = ego2lidar {
            let [x, y, z] = [bbox[0], bbox[1], bbox[2]];
            bbox[0] = matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3];
            bbox[1] = matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7];
            bbox[2] = matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11];
            bbox[6] += matrix[4].atan2(matrix[0]);
            let [vx, vy] = [bbox[7], bbox[8]];
            bbox[7] = matrix[0] * vx + matrix[1] * vy;
            bbox[8] = matrix[4] * vx + matrix[5] * vy;
        }
        bbox[2] = super::ops::round_bf16(bbox[2] - bbox[5] * 0.5);
        boxes.push(bbox);
        scores.push(score);
        labels.push((index % classes_count) as u8);
    }
    Ok(Detection {
        boxes,
        scores,
        labels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct BitsFixture {
        input: Vec<u32>,
        output: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct Tensor4Fixture {
        shape: [usize; 4],
        values: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct Tensor5Fixture {
        shape: [usize; 5],
        values: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct DetectionFixture {
        classes: Vec<u32>,
        coordinates: Vec<u32>,
        boxes: Vec<u32>,
        scores: Vec<u32>,
        labels: Vec<u8>,
    }

    #[derive(Deserialize)]
    struct ReferenceRefineFixture {
        references: Vec<u32>,
        regression: Vec<u32>,
        refined: Vec<u32>,
        scaled: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct MapCropFixture {
        input: Tensor4Fixture,
        grid: Vec<u32>,
        output: Tensor4Fixture,
    }

    #[derive(Deserialize)]
    struct OccupancyCropFixture {
        input: Tensor5Fixture,
        output: Tensor5Fixture,
    }

    #[derive(Deserialize)]
    struct SoftplusFixture {
        input: Vec<u32>,
        output: Vec<u32>,
        argmax: Vec<u8>,
    }

    #[derive(Deserialize)]
    struct HeadsFixture {
        inverse_sigmoid: BitsFixture,
        reference_refine: ReferenceRefineFixture,
        detection: DetectionFixture,
        map_crop: MapCropFixture,
        occupancy_crop: OccupancyCropFixture,
        softplus: SoftplusFixture,
    }

    fn values(bits: &[u32]) -> Vec<f32> {
        bits.iter().copied().map(f32::from_bits).collect()
    }

    fn assert_bits(label: &str, actual: &[f32], expected: &[u32]) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected,
                "{label}[{index}] rust={:08x} oracle={expected:08x}",
                actual.to_bits()
            );
        }
    }

    #[test]
    fn perception_heads_match_official_bf16_words() {
        let fixture: HeadsFixture = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen_drive/perception-heads.json"
        )))
        .unwrap();

        let inverse = values(&fixture.inverse_sigmoid.input)
            .into_iter()
            .map(super::super::bev::inverse_sigmoid)
            .collect::<Vec<_>>();
        assert_bits("inverse sigmoid", &inverse, &fixture.inverse_sigmoid.output);

        let references = values(&fixture.reference_refine.references);
        let regression = values(&fixture.reference_refine.regression);
        let mut refined = Vec::with_capacity(6);
        for row in 0..2 {
            refined.push(refine_reference(references[row * 3], regression[row * 10]));
            refined.push(refine_reference(
                references[row * 3 + 1],
                regression[row * 10 + 1],
            ));
            refined.push(refine_reference(
                references[row * 3 + 2],
                regression[row * 10 + 4],
            ));
        }
        assert_bits(
            "reference refine",
            &refined,
            &fixture.reference_refine.refined,
        );
        let mut scaled = regression.clone();
        apply_detection_references(
            &mut scaled,
            &references,
            2,
            10,
            [-51.2, -51.2, -5.0, 51.2, 51.2, 5.4],
        )
        .unwrap();
        assert_bits("reference scale", &scaled, &fixture.reference_refine.scaled);

        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let detection = decode_detections(
            &values(&fixture.detection.classes),
            &values(&fixture.detection.coordinates),
            2,
            2,
            10,
            &identity,
            false,
        )
        .unwrap();
        assert_bits(
            "detection boxes",
            &detection
                .boxes
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            &fixture.detection.boxes,
        );
        assert_bits(
            "detection scores",
            &detection.scores,
            &fixture.detection.scores,
        );
        assert_eq!(detection.labels, fixture.detection.labels);

        let (grid, shape) =
            map_crop_grid(&[-1.5, -1.5, 0.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]).unwrap();
        assert_bits("map grid", &grid, &fixture.map_crop.grid);
        let crop = grid_sample_bilinear_aligned(
            &Tensor4::new(
                values(&fixture.map_crop.input.values),
                fixture.map_crop.input.shape,
            )
            .unwrap(),
            &grid,
            [1, shape[0], shape[1]],
            true,
        )
        .unwrap();
        assert_eq!(crop.shape(), fixture.map_crop.output.shape);
        assert_bits("map crop", crop.values(), &fixture.map_crop.output.values);

        let input = Tensor5::new(
            values(&fixture.occupancy_crop.input.values),
            fixture.occupancy_crop.input.shape,
        )
        .unwrap();
        let cropped = adapt_occ_volume(
            &input,
            [-2.0, -2.0, -1.0, 2.0, 2.0, 1.0],
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            2,
        )
        .unwrap();
        assert_eq!(cropped.shape, fixture.occupancy_crop.output.shape);
        assert_bits(
            "occupancy crop",
            &cropped.values,
            &fixture.occupancy_crop.output.values,
        );

        let mut softplus = values(&fixture.softplus.input);
        softplus_inplace(&mut softplus);
        assert_bits("softplus", &softplus, &fixture.softplus.output);
        assert_eq!(argmax_rows(&softplus, 3).unwrap(), fixture.softplus.argmax);
    }

    #[test]
    fn detection_decode_is_stable_and_shifts_center_to_bottom() {
        let classes = [0.0, 1.0, 1.0, -1.0];
        let mut coordinates = vec![0.0; 20];
        coordinates[5] = 2.0f32.ln();
        coordinates[15] = 4.0f32.ln();
        coordinates[7] = 1.0;
        coordinates[17] = 1.0;
        let output = decode_detections(
            &classes,
            &coordinates,
            2,
            2,
            10,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            false,
        )
        .unwrap();
        assert_eq!(output.labels[..2], [1, 0]);
        assert_eq!(output.boxes[0][2].to_bits(), (-1.0f32).to_bits());
        assert_eq!(output.boxes[1][2].to_bits(), (-2.0f32).to_bits());
    }
}
