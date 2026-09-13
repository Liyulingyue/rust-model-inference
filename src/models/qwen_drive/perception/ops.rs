use half::bf16;

fn checked_len(dims: &[usize], label: &str) -> Result<usize, String> {
    dims.iter().try_fold(1usize, |size, &dim| {
        size.checked_mul(dim)
            .ok_or_else(|| format!("{label} shape overflow"))
    })
}

fn require_finite(values: &[f32], label: &str) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} contains non-finite values"));
    }
    Ok(())
}

#[inline]
pub(crate) fn round_bf16(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor4 {
    values: Vec<f32>,
    shape: [usize; 4],
}

impl Tensor4 {
    pub fn new(values: Vec<f32>, shape: [usize; 4]) -> Result<Self, String> {
        if shape.contains(&0) {
            return Err("Tensor4 dimensions must be greater than zero".into());
        }
        let expected = checked_len(&shape, "Tensor4")?;
        if values.len() != expected {
            return Err(format!(
                "Tensor4 shape {shape:?} requires {expected} values, got {}",
                values.len()
            ));
        }
        require_finite(&values, "Tensor4")?;
        Ok(Self { values, shape })
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }

    #[inline]
    fn at(&self, n: usize, c: usize, h: usize, w: usize) -> f32 {
        let [_, channels, height, width] = self.shape;
        self.values[((n * channels + c) * height + h) * width + w]
    }
}

pub fn resize_bilinear(
    input: &Tensor4,
    output_height: usize,
    output_width: usize,
) -> Result<Tensor4, String> {
    resize_bilinear_aligned(input, output_height, output_width, false)
}

pub fn resize_bilinear_aligned(
    input: &Tensor4,
    output_height: usize,
    output_width: usize,
    align_corners: bool,
) -> Result<Tensor4, String> {
    if output_height == 0 || output_width == 0 {
        return Err("resize output dimensions must be greater than zero".into());
    }
    let [batch, channels, input_height, input_width] = input.shape;
    let mut output =
        vec![0.0; checked_len(&[batch, channels, output_height, output_width], "resize")?];
    let y_scale = if align_corners && output_height > 1 {
        (input_height - 1) as f32 / (output_height - 1) as f32
    } else {
        input_height as f32 / output_height as f32
    };
    let x_scale = if align_corners && output_width > 1 {
        (input_width - 1) as f32 / (output_width - 1) as f32
    } else {
        input_width as f32 / output_width as f32
    };
    for n in 0..batch {
        for c in 0..channels {
            for y in 0..output_height {
                let source_y = if align_corners {
                    y as f32 * y_scale
                } else {
                    (y as f32 + 0.5) * y_scale - 0.5
                };
                let y_low_raw = source_y.floor() as isize;
                let y_high_raw = y_low_raw + 1;
                let y_lerp = source_y - y_low_raw as f32;
                let y_low = y_low_raw.clamp(0, input_height as isize - 1) as usize;
                let y_high = y_high_raw.clamp(0, input_height as isize - 1) as usize;
                for x in 0..output_width {
                    let source_x = if align_corners {
                        x as f32 * x_scale
                    } else {
                        (x as f32 + 0.5) * x_scale - 0.5
                    };
                    let x_low_raw = source_x.floor() as isize;
                    let x_high_raw = x_low_raw + 1;
                    let x_lerp = source_x - x_low_raw as f32;
                    let x_low = x_low_raw.clamp(0, input_width as isize - 1) as usize;
                    let x_high = x_high_raw.clamp(0, input_width as isize - 1) as usize;
                    let top = input.at(n, c, y_low, x_low) * (1.0 - x_lerp)
                        + input.at(n, c, y_low, x_high) * x_lerp;
                    let bottom = input.at(n, c, y_high, x_low) * (1.0 - x_lerp)
                        + input.at(n, c, y_high, x_high) * x_lerp;
                    output[((n * channels + c) * output_height + y) * output_width + x] =
                        round_bf16(top * (1.0 - y_lerp) + bottom * y_lerp);
                }
            }
        }
    }
    Tensor4::new(output, [batch, channels, output_height, output_width])
}

pub fn grid_sample_bilinear(
    input: &Tensor4,
    grid: &[f32],
    grid_shape: [usize; 3],
) -> Result<Tensor4, String> {
    grid_sample_bilinear_aligned(input, grid, grid_shape, false)
}

pub fn grid_sample_bilinear_aligned(
    input: &Tensor4,
    grid: &[f32],
    grid_shape: [usize; 3],
    align_corners: bool,
) -> Result<Tensor4, String> {
    let [batch, output_height, output_width] = grid_shape;
    if batch != input.shape[0] || output_height == 0 || output_width == 0 {
        return Err(format!(
            "invalid grid shape {grid_shape:?} for input {:?}",
            input.shape
        ));
    }
    let expected = checked_len(&[batch, output_height, output_width, 2], "grid")?;
    if grid.len() != expected {
        return Err(format!(
            "grid requires {expected} values, got {}",
            grid.len()
        ));
    }
    require_finite(grid, "grid")?;
    let [_, channels, input_height, input_width] = input.shape;
    let mut output = vec![
        0.0;
        checked_len(
            &[batch, channels, output_height, output_width],
            "grid output"
        )?
    ];
    for n in 0..batch {
        for y in 0..output_height {
            for x in 0..output_width {
                let grid_index = ((n * output_height + y) * output_width + x) * 2;
                let source_x = if align_corners {
                    round_bf16(round_bf16(grid[grid_index] + 1.0) * (input_width - 1) as f32) * 0.5
                } else {
                    ((grid[grid_index] + 1.0) * input_width as f32 - 1.0) * 0.5
                };
                let source_y = if align_corners {
                    round_bf16(round_bf16(grid[grid_index + 1] + 1.0) * (input_height - 1) as f32)
                        * 0.5
                } else {
                    ((grid[grid_index + 1] + 1.0) * input_height as f32 - 1.0) * 0.5
                };
                let x_low = source_x.floor() as isize;
                let y_low = source_y.floor() as isize;
                let x_high = x_low + 1;
                let y_high = y_low + 1;
                let x_lerp = source_x - x_low as f32;
                let y_lerp = source_y - y_low as f32;
                for c in 0..channels {
                    let mut value = 0.0f32;
                    for (source_h, h_weight) in [(y_low, 1.0 - y_lerp), (y_high, y_lerp)] {
                        for (source_w, w_weight) in [(x_low, 1.0 - x_lerp), (x_high, x_lerp)] {
                            if source_h >= 0
                                && source_h < input_height as isize
                                && source_w >= 0
                                && source_w < input_width as isize
                            {
                                let sample = input.at(n, c, source_h as usize, source_w as usize);
                                if align_corners {
                                    let h_weight = round_bf16(h_weight);
                                    let w_weight = round_bf16(w_weight);
                                    value = round_bf16(
                                        value
                                            + round_bf16(round_bf16(h_weight * w_weight) * sample),
                                    );
                                } else {
                                    value += h_weight * w_weight * sample;
                                }
                            }
                        }
                    }
                    output[((n * channels + c) * output_height + y) * output_width + x] =
                        round_bf16(value);
                }
            }
        }
    }
    Tensor4::new(output, [batch, channels, output_height, output_width])
}

pub fn scatter_add_ordered(
    output: &mut [f32],
    indices: &[usize],
    values: &[f32],
) -> Result<(), String> {
    if indices.len() != values.len() {
        return Err("scatter indices and values have different lengths".into());
    }
    require_finite(values, "scatter values")?;
    for (&index, &value) in indices.iter().zip(values) {
        let destination = output
            .get_mut(index)
            .ok_or_else(|| format!("scatter index {index} out of bounds"))?;
        *destination += value;
    }
    Ok(())
}

pub struct VoxelPoolInput<'a> {
    pub img_feats: &'a [f32],
    pub img_depth: &'a [f32],
    pub coords: &'a [[usize; 4]],
    pub point_indices: &'a [usize],
    pub batch: usize,
    pub sweeps: usize,
    pub cameras: usize,
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub depth: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}

pub fn voxel_pool_depth(input: &VoxelPoolInput<'_>) -> Result<Vec<f32>, String> {
    let dims = [
        input.batch,
        input.sweeps,
        input.cameras,
        input.x,
        input.y,
        input.z,
        input.depth,
        input.height,
        input.width,
        input.channels,
    ];
    if dims.contains(&0) {
        return Err("voxel dimensions must be greater than zero".into());
    }
    if input.coords.len() != input.point_indices.len() {
        return Err("voxel coords and point indices have different lengths".into());
    }
    let images = checked_len(&[input.batch, input.sweeps, input.cameras], "voxel images")?;
    let feature_len = checked_len(
        &[images, input.channels, input.height, input.width],
        "voxel features",
    )?;
    let depth_len = checked_len(
        &[images, input.depth, input.height, input.width],
        "voxel depth",
    )?;
    if input.img_feats.len() != feature_len || input.img_depth.len() != depth_len {
        return Err(format!(
            "invalid voxel input lengths: features {} expected {feature_len}, depth {} expected {depth_len}",
            input.img_feats.len(),
            input.img_depth.len()
        ));
    }
    require_finite(input.img_feats, "voxel features")?;
    require_finite(input.img_depth, "voxel depth")?;
    let points_per_image = checked_len(&[input.depth, input.height, input.width], "voxel points")?;
    let total_points = checked_len(&[images, points_per_image], "voxel points")?;
    let output_len = checked_len(
        &[
            input.batch,
            input.cameras,
            input.x,
            input.y,
            input.z,
            input.channels,
        ],
        "voxel output",
    )?;
    let mut output = vec![0.0f32; output_len];
    for (&point_index, coord) in input.point_indices.iter().zip(input.coords) {
        if point_index >= total_points
            || coord[0] >= input.batch
            || coord[1] >= input.x
            || coord[2] >= input.y
            || coord[3] >= input.z
        {
            return Err(format!("invalid voxel point {point_index} at {coord:?}"));
        }
        let mut point = point_index;
        let w = point % input.width;
        point /= input.width;
        let h = point % input.height;
        point /= input.height;
        let d = point % input.depth;
        point /= input.depth;
        let camera = point % input.cameras;
        point /= input.cameras;
        let sweep = point % input.sweeps;
        let point_batch = point / input.sweeps;
        if point_batch != coord[0] {
            return Err(format!(
                "voxel point {point_index} belongs to batch {point_batch}, coord uses {}",
                coord[0]
            ));
        }
        let image = (point_batch * input.sweeps + sweep) * input.cameras + camera;
        let depth_index = ((image * input.depth + d) * input.height + h) * input.width + w;
        for channel in 0..input.channels {
            let feature_index =
                ((image * input.channels + channel) * input.height + h) * input.width + w;
            let output_index = (((((coord[0] * input.cameras + camera) * input.x + coord[1])
                * input.y
                + coord[2])
                * input.z
                + coord[3])
                * input.channels)
                + channel;
            output[output_index] += input.img_feats[feature_index] * input.img_depth[depth_index];
        }
    }
    output
        .iter_mut()
        .for_each(|value| *value = round_bf16(*value));
    Ok(output)
}

pub struct DeformAttentionInput<'a> {
    pub value: &'a [f32],
    pub spatial_shapes: &'a [[usize; 2]],
    pub level_start_index: &'a [usize],
    pub sampling_locations: &'a [f32],
    pub attention_weights: &'a [f32],
    pub batch: usize,
    pub queries: usize,
    pub heads: usize,
    pub channels: usize,
    pub points: usize,
}

pub fn ms_deform_attn(input: &DeformAttentionInput<'_>) -> Result<Vec<f32>, String> {
    let levels = input.spatial_shapes.len();
    if [
        input.batch,
        input.queries,
        input.heads,
        input.channels,
        input.points,
        levels,
    ]
    .contains(&0)
        || input.level_start_index.len() != levels
    {
        return Err("invalid deformable-attention dimensions".into());
    }
    let mut spatial_size = 0usize;
    for (level, &[height, width]) in input.spatial_shapes.iter().enumerate() {
        if height == 0 || width == 0 || input.level_start_index[level] != spatial_size {
            return Err(format!("invalid deformable-attention level {level}"));
        }
        spatial_size = spatial_size
            .checked_add(checked_len(&[height, width], "deform level")?)
            .ok_or_else(|| "deform spatial size overflow".to_string())?;
    }
    let value_len = checked_len(
        &[input.batch, spatial_size, input.heads, input.channels],
        "deform values",
    )?;
    let samples = checked_len(
        &[
            input.batch,
            input.queries,
            input.heads,
            levels,
            input.points,
        ],
        "deform samples",
    )?;
    if input.value.len() != value_len
        || input.sampling_locations.len() != samples * 2
        || input.attention_weights.len() != samples
    {
        return Err("invalid deformable-attention input lengths".into());
    }
    require_finite(input.value, "deform values")?;
    require_finite(input.sampling_locations, "deform sampling locations")?;
    require_finite(input.attention_weights, "deform attention weights")?;
    let mut output = vec![
        0.0f32;
        checked_len(
            &[input.batch, input.queries, input.heads, input.channels],
            "deform output"
        )?
    ];
    let qid_stride = input.heads * input.channels;
    for batch in 0..input.batch {
        let value_batch = batch * spatial_size * qid_stride;
        for query in 0..input.queries {
            for head in 0..input.heads {
                for channel in 0..input.channels {
                    let mut acc = 0.0f32;
                    for (level, &[height, width]) in input.spatial_shapes.iter().enumerate() {
                        let value_level = value_batch
                            + input.level_start_index[level] * qid_stride
                            + head * input.channels
                            + channel;
                        for point in 0..input.points {
                            let sample = ((((batch * input.queries + query) * input.heads + head)
                                * levels
                                + level)
                                * input.points)
                                + point;
                            let loc_w = round_bf16(input.sampling_locations[sample * 2]);
                            let loc_h = round_bf16(input.sampling_locations[sample * 2 + 1]);
                            let weight = round_bf16(input.attention_weights[sample]);
                            let h_image = loc_h * height as f32 - 0.5;
                            let w_image = loc_w * width as f32 - 0.5;
                            if h_image > -1.0
                                && w_image > -1.0
                                && h_image < height as f32
                                && w_image < width as f32
                            {
                                let h_low = h_image.floor() as isize;
                                let w_low = w_image.floor() as isize;
                                let h_high = h_low + 1;
                                let w_high = w_low + 1;
                                let low_h_weight = 1.0 - (h_image - h_low as f32);
                                let low_w_weight = 1.0 - (w_image - w_low as f32);
                                let high_h_weight = 1.0 - low_h_weight;
                                let high_w_weight = 1.0 - low_w_weight;
                                let mut sampled = 0.0f32;
                                for (h, h_weight) in
                                    [(h_low, low_h_weight), (h_high, high_h_weight)]
                                {
                                    for (w, w_weight) in
                                        [(w_low, low_w_weight), (w_high, high_w_weight)]
                                    {
                                        if h >= 0
                                            && h < height as isize
                                            && w >= 0
                                            && w < width as isize
                                        {
                                            let value_index = value_level
                                                + (h as usize * width + w as usize) * qid_stride;
                                            sampled += h_weight
                                                * w_weight
                                                * round_bf16(input.value[value_index]);
                                        }
                                    }
                                }
                                acc += sampled * weight;
                            }
                        }
                    }
                    let output_index = ((batch * input.queries + query) * input.heads + head)
                        * input.channels
                        + channel;
                    output[output_index] = round_bf16(acc);
                }
            }
        }
    }
    Ok(output)
}
