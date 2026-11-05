use super::fpn::{perception_contracts, ComponentWeights, F32Tensor, ViewGeometry};
use super::ops::{ms_deform_attn, round_bf16, DeformAttentionInput, Tensor4};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::models::diffusion::dreamx::kernels::{attention_scalar, AttentionSpec};
use crate::models::qwen_drive::config::PerceptionConfig;
use crate::models::qwen_drive::weights::{HeadLinear, HeadLinearScratch};

const HEADS: usize = 8;
const ENCODER_LEVELS: usize = 4;
const ENCODER_POINTS: usize = 8;
const PILLAR_POINTS: usize = 4;
const DECODER_POINTS: usize = 4;

pub(crate) struct Linear<'a> {
    inner: HeadLinear<'a>,
    output: usize,
}

impl<'a> Linear<'a> {
    pub(crate) fn load<S: TensorSource + ?Sized>(
        source: &'a S,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: HeadLinear::load_perception(source, name, input, output, true)?,
            output,
        })
    }

    fn load_slice<S: TensorSource + ?Sized>(
        source: &'a S,
        name: &str,
        input: usize,
        full_output: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Self, String> {
        let output = rows.end - rows.start;
        Ok(Self {
            inner: HeadLinear::load_perception_slice(source, name, input, full_output, rows)?,
            output,
        })
    }

    pub(crate) fn forward(
        &self,
        input: &[f32],
        rows: usize,
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0; rows * self.output];
        self.inner.forward_rows(
            input,
            rows,
            pool,
            &mut output,
            &mut HeadLinearScratch::default(),
        )?;
        Ok(output)
    }
}

struct EncoderLayer<'a> {
    temporal_offsets: Linear<'a>,
    temporal_weights: Linear<'a>,
    temporal_value: Linear<'a>,
    temporal_output: Linear<'a>,
    spatial_offsets: Linear<'a>,
    spatial_weights: Linear<'a>,
    spatial_value: Linear<'a>,
    spatial_output: Linear<'a>,
    ffn_up: Linear<'a>,
    ffn_down: Linear<'a>,
}

struct DecoderLayer<'a> {
    self_query: Linear<'a>,
    self_key: Linear<'a>,
    self_value: Linear<'a>,
    self_output: Linear<'a>,
    cross_offsets: Linear<'a>,
    cross_weights: Linear<'a>,
    cross_value: Linear<'a>,
    cross_output: Linear<'a>,
    ffn_up: Linear<'a>,
    ffn_down: Linear<'a>,
}

struct RegBranch<'a> {
    first: Linear<'a>,
    second: Linear<'a>,
    output: Linear<'a>,
}

pub struct BevFormer<'a> {
    config: PerceptionConfig,
    weights: ComponentWeights,
    encoder: Vec<EncoderLayer<'a>>,
    decoder: Vec<DecoderLayer<'a>>,
    reg_branches: Vec<RegBranch<'a>>,
    reference_points: Linear<'a>,
}

pub struct BevInput<'a> {
    pub levels: &'a [Tensor4],
    pub uvtr_bev: &'a [f32],
    pub geometry: &'a ViewGeometry<'a>,
}

pub struct BevOutput {
    pub bev: Vec<f32>,
    pub decoder_states: Vec<Vec<f32>>,
    pub initial_references: Vec<f32>,
    pub decoder_references: Vec<Vec<f32>>,
}

fn name(suffix: &str) -> String {
    format!("qwen_drive_perception.head.{suffix}")
}

impl<'a> BevFormer<'a> {
    pub fn from_source<S: TensorSource + ?Sized>(source: &'a S) -> Result<Self, String> {
        let config = PerceptionConfig::from_source(source)?;
        let contracts = perception_contracts()?;
        let weights = ComponentWeights::load(
            source,
            &contracts,
            &[
                "bev_modeling.head.bev_embedding.",
                "bev_modeling.head.query_embedding.",
                "bev_modeling.head.positional_encoding.",
                "bev_modeling.head.transformer.level_embeds",
                "bev_modeling.head.transformer.encoder.layers.",
                "bev_modeling.head.transformer.decoder.layers.",
            ],
        )?;
        let mut encoder = Vec::with_capacity(config.encoder_layers);
        for layer in 0..config.encoder_layers {
            let root = format!("transformer.encoder.layers.{layer}");
            encoder.push(EncoderLayer {
                temporal_offsets: Linear::load(
                    source,
                    &name(&format!("{root}.attentions.0.sampling_offsets")),
                    config.embed_dim * 2,
                    HEADS * 2 * DECODER_POINTS * 2,
                )?,
                temporal_weights: Linear::load(
                    source,
                    &name(&format!("{root}.attentions.0.attention_weights")),
                    config.embed_dim * 2,
                    HEADS * 2 * DECODER_POINTS,
                )?,
                temporal_value: Linear::load(
                    source,
                    &name(&format!("{root}.attentions.0.value_proj")),
                    config.embed_dim,
                    config.embed_dim,
                )?,
                temporal_output: Linear::load(
                    source,
                    &name(&format!("{root}.attentions.0.output_proj")),
                    config.embed_dim,
                    config.embed_dim,
                )?,
                spatial_offsets: Linear::load(
                    source,
                    &name(&format!(
                        "{root}.attentions.1.deformable_attention.sampling_offsets"
                    )),
                    config.embed_dim,
                    HEADS * ENCODER_LEVELS * ENCODER_POINTS * 2,
                )?,
                spatial_weights: Linear::load(
                    source,
                    &name(&format!(
                        "{root}.attentions.1.deformable_attention.attention_weights"
                    )),
                    config.embed_dim,
                    HEADS * ENCODER_LEVELS * ENCODER_POINTS,
                )?,
                spatial_value: Linear::load(
                    source,
                    &name(&format!(
                        "{root}.attentions.1.deformable_attention.value_proj"
                    )),
                    config.embed_dim,
                    config.embed_dim,
                )?,
                spatial_output: Linear::load(
                    source,
                    &name(&format!("{root}.attentions.1.output_proj")),
                    config.embed_dim,
                    config.embed_dim,
                )?,
                ffn_up: Linear::load(
                    source,
                    &name(&format!("{root}.ffns.0.layers.0.0")),
                    config.embed_dim,
                    config.embed_dim * 2,
                )?,
                ffn_down: Linear::load(
                    source,
                    &name(&format!("{root}.ffns.0.layers.1")),
                    config.embed_dim * 2,
                    config.embed_dim,
                )?,
            });
        }
        let mut decoder = Vec::with_capacity(config.decoder_layers);
        let mut reg_branches = Vec::with_capacity(config.decoder_layers);
        for layer in 0..config.decoder_layers {
            let root = format!("transformer.decoder.layers.{layer}");
            let in_proj = name(&format!("{root}.attentions.0.attn.in_proj"));
            decoder.push(DecoderLayer {
                self_query: Linear::load_slice(
                    source,
                    &in_proj,
                    config.embed_dim,
                    config.embed_dim * 3,
                    0..config.embed_dim,
                )?,
                self_key: Linear::load_slice(
                    source,
                    &in_proj,
                    config.embed_dim,
                    config.embed_dim * 3,
                    config.embed_dim..config.embed_dim * 2,
                )?,
                self_value: Linear::load_slice(
                    source,
                    &in_proj,
                    config.embed_dim,
                    config.embed_dim * 3,
                    config.embed_dim * 2..config.embed_dim * 3,
                )?,
                self_output: Linear::load(
                    source,
                    &name(&format!("{root}.attentions.0.attn.out_proj")),
                    config.embed_dim,
                    config.embed_dim,
                )?,
                cross_offsets: Linear::load(
                    source,
                    &name(&format!("{root}.attentions.1.sampling_offsets")),
                    config.embed_dim,
                    HEADS * DECODER_POINTS * 2,
                )?,
                cross_weights: Linear::load(
                    source,
                    &name(&format!("{root}.attentions.1.attention_weights")),
                    config.embed_dim,
                    HEADS * DECODER_POINTS,
                )?,
                cross_value: Linear::load(
                    source,
                    &name(&format!("{root}.attentions.1.value_proj")),
                    config.embed_dim,
                    config.embed_dim,
                )?,
                cross_output: Linear::load(
                    source,
                    &name(&format!("{root}.attentions.1.output_proj")),
                    config.embed_dim,
                    config.embed_dim,
                )?,
                ffn_up: Linear::load(
                    source,
                    &name(&format!("{root}.ffns.0.layers.0.0")),
                    config.embed_dim,
                    config.embed_dim * 2,
                )?,
                ffn_down: Linear::load(
                    source,
                    &name(&format!("{root}.ffns.0.layers.1")),
                    config.embed_dim * 2,
                    config.embed_dim,
                )?,
            });
            let root = name(&format!("reg_branches.{layer}"));
            reg_branches.push(RegBranch {
                first: Linear::load(
                    source,
                    &format!("{root}.0"),
                    config.embed_dim,
                    config.embed_dim,
                )?,
                second: Linear::load(
                    source,
                    &format!("{root}.2"),
                    config.embed_dim,
                    config.embed_dim,
                )?,
                output: Linear::load(
                    source,
                    &format!("{root}.4"),
                    config.embed_dim,
                    config.code_size,
                )?,
            });
        }
        Ok(Self {
            reference_points: Linear::load(
                source,
                &name("transformer.reference_points"),
                config.embed_dim,
                3,
            )?,
            config,
            weights,
            encoder,
            decoder,
            reg_branches,
        })
    }

    fn tensor(&self, suffix: &str) -> Result<&F32Tensor, String> {
        let name = format!("bev_modeling.head.{suffix}");
        self.weights
            .tensors
            .get(&name)
            .ok_or_else(|| format!("Missing perception tensor: {name}"))
    }

    pub fn forward(&self, input: &BevInput<'_>, pool: &ComputePool) -> Result<BevOutput, String> {
        let encoder = self.prepare_encoder_input(input)?;
        let rows = self.config.bev[0] * self.config.bev[1];
        let mut bev = encoder.query.clone();
        for (index, layer) in self.encoder.iter().enumerate() {
            bev = self.temporal_attention(
                layer,
                &bev,
                &encoder.position,
                &encoder.reference_2d,
                pool,
            )?;
            bev = self.norm("encoder", index, 0, &bev, rows)?;
            bev = self.spatial_attention(layer, &bev, &encoder, pool)?;
            bev = self.norm("encoder", index, 1, &bev, rows)?;
            let identity = bev.clone();
            let mut ffn = layer.ffn_up.forward(&bev, rows, pool)?;
            ffn.iter_mut().for_each(|value| *value = value.max(0.0));
            ffn = layer.ffn_down.forward(&ffn, rows, pool)?;
            bev = add_round(&identity, &ffn)?;
            bev = self.norm("encoder", index, 2, &bev, rows)?;
        }

        let query_embedding = self.tensor("query_embedding.weight")?;
        let query_rows = self.config.num_queries;
        let width = self.config.embed_dim;
        if query_embedding.shape.as_slice() != [query_rows, width * 2] {
            return Err("Invalid Qwen-Drive object query embedding".into());
        }
        let mut query_position = Vec::with_capacity(query_rows * width);
        let mut query = Vec::with_capacity(query_rows * width);
        for row in query_embedding.values.chunks_exact(width * 2) {
            query_position.extend_from_slice(&row[..width]);
            query.extend_from_slice(&row[width..]);
        }
        let mut references = self
            .reference_points
            .forward(&query_position, query_rows, pool)?;
        references
            .iter_mut()
            .for_each(|value| *value = sigmoid(*value));
        let initial_references = references.clone();
        let mut decoder_states = Vec::with_capacity(self.decoder.len());
        let mut decoder_references = Vec::with_capacity(self.decoder.len());
        for (index, layer) in self.decoder.iter().enumerate() {
            query = self.decoder_self_attention(layer, &query, &query_position, pool)?;
            query = self.norm("decoder", index, 0, &query, query_rows)?;
            query = self.decoder_cross_attention(
                layer,
                &query,
                &query_position,
                &references,
                &bev,
                pool,
            )?;
            query = self.norm("decoder", index, 1, &query, query_rows)?;
            let identity = query.clone();
            let mut ffn = layer.ffn_up.forward(&query, query_rows, pool)?;
            ffn.iter_mut().for_each(|value| *value = value.max(0.0));
            ffn = layer.ffn_down.forward(&ffn, query_rows, pool)?;
            query = add_round(&identity, &ffn)?;
            query = self.norm("decoder", index, 2, &query, query_rows)?;

            let regression = self.regression(index, &query, pool)?;
            for row in 0..query_rows {
                references[row * 3] =
                    refine_reference(references[row * 3], regression[row * self.config.code_size]);
                references[row * 3 + 1] = refine_reference(
                    references[row * 3 + 1],
                    regression[row * self.config.code_size + 1],
                );
                references[row * 3 + 2] = refine_reference(
                    references[row * 3 + 2],
                    regression[row * self.config.code_size + 4],
                );
            }
            decoder_states.push(query.clone());
            decoder_references.push(references.clone());
        }
        Ok(BevOutput {
            bev,
            decoder_states,
            initial_references,
            decoder_references,
        })
    }

    pub(crate) fn regression(
        &self,
        layer: usize,
        input: &[f32],
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        let branch = self
            .reg_branches
            .get(layer)
            .ok_or("Invalid Qwen-Drive regression layer")?;
        let rows = input.len() / self.config.embed_dim;
        let mut output = branch.first.forward(input, rows, pool)?;
        output.iter_mut().for_each(|value| *value = value.max(0.0));
        output = branch.second.forward(&output, rows, pool)?;
        output.iter_mut().for_each(|value| *value = value.max(0.0));
        branch.output.forward(&output, rows, pool)
    }
}

fn add_round(left: &[f32], right: &[f32]) -> Result<Vec<f32>, String> {
    if left.len() != right.len() {
        return Err("Qwen-Drive residual shapes differ".into());
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(&left, &right)| round_bf16(left + right))
        .collect())
}

pub(crate) fn layer_norm_rows(
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &F32Tensor,
    bias: &F32Tensor,
) -> Result<Vec<f32>, String> {
    if input.len() != rows * width
        || weight.shape.as_slice() != [width]
        || bias.shape.as_slice() != [width]
    {
        return Err("Invalid Qwen-Drive LayerNorm shape".into());
    }
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(width) {
        let mean = row.iter().copied().sum::<f32>() / width as f32;
        let variance = row
            .iter()
            .map(|&value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / width as f32;
        let inverse = 1.0 / (variance + 1e-5).sqrt();
        output.extend((0..width).map(|index| {
            round_bf16((row[index] - mean) * inverse * weight.values[index] + bias.values[index])
        }));
    }
    Ok(output)
}

fn softmax_rows(values: &mut [f32], width: usize) {
    for row in values.chunks_exact_mut(width) {
        crate::ops::softmax_inplace(row);
        row.iter_mut().for_each(|value| *value = round_bf16(*value));
    }
}

pub(crate) fn inverse_sigmoid(value: f32) -> f32 {
    let value = round_bf16(value.clamp(0.0, 1.0));
    let lower = value.max(round_bf16(1e-5));
    let upper = round_bf16(1.0 - value).max(round_bf16(1e-5));
    round_bf16(round_bf16(lower / upper).ln())
}

pub(crate) fn sigmoid(value: f32) -> f32 {
    round_bf16(1.0 / (1.0 + (-value).exp()))
}

pub(crate) fn refine_reference(reference: f32, delta: f32) -> f32 {
    sigmoid(round_bf16(delta + round_bf16(inverse_sigmoid(reference))))
}

struct EncoderInput {
    features: Vec<Vec<f32>>,
    spatial_shapes: Vec<[usize; 2]>,
    query: Vec<f32>,
    position: Vec<f32>,
    reference_2d: Vec<f32>,
    reference_cam: Vec<f32>,
    visible: Vec<bool>,
    cameras: usize,
}

impl BevFormer<'_> {
    fn prepare_encoder_input(&self, input: &BevInput<'_>) -> Result<EncoderInput, String> {
        if input.levels.len() != ENCODER_LEVELS {
            return Err(format!(
                "Qwen-Drive BEV encoder requires {ENCODER_LEVELS} feature levels"
            ));
        }
        let width = self.config.embed_dim;
        let queries = self.config.bev[0] * self.config.bev[1];
        if input.uvtr_bev.len() != queries * width {
            return Err("Invalid Qwen-Drive UVTR BEV token shape".into());
        }
        if input.geometry.lidar2ego.len() != 1 {
            return Err("Qwen-Drive perception currently requires one frame per call".into());
        }
        let cameras = input.geometry.lidar2img.len();
        if cameras == 0 {
            return Err("Qwen-Drive perception requires at least one camera".into());
        }
        let level_embeds = self.tensor("transformer.level_embeds")?;
        if level_embeds.shape.as_slice() != [ENCODER_LEVELS, width] {
            return Err("Invalid Qwen-Drive level embeddings".into());
        }

        let mut spatial_shapes = Vec::with_capacity(ENCODER_LEVELS);
        let total_positions = input.levels.iter().try_fold(0usize, |total, level| {
            let [images, channels, height, level_width] = level.shape();
            if images != cameras || channels != width {
                return Err("Invalid Qwen-Drive camera feature shape".to_string());
            }
            spatial_shapes.push([height, level_width]);
            total
                .checked_add(height * level_width)
                .ok_or_else(|| "Qwen-Drive feature shape overflow".to_string())
        })?;
        let mut features = (0..cameras)
            .map(|_| Vec::with_capacity(total_positions * width))
            .collect::<Vec<_>>();
        for (level_index, level) in input.levels.iter().enumerate() {
            let [_, _, height, level_width] = level.shape();
            let plane = height * level_width;
            for (camera, camera_features) in features.iter_mut().enumerate() {
                for position in 0..plane {
                    for channel in 0..width {
                        let value = level.values()[(camera * width + channel) * plane + position];
                        camera_features.push(round_bf16(
                            value + level_embeds.values[level_index * width + channel],
                        ));
                    }
                }
            }
        }

        let bev_embedding = self.tensor("bev_embedding.weight")?;
        if bev_embedding.shape.as_slice() != [queries, width] {
            return Err("Invalid Qwen-Drive BEV query embedding".into());
        }
        let query = add_round(&bev_embedding.values, input.uvtr_bev)?;
        let row = self.tensor("positional_encoding.row_embed.weight")?;
        let col = self.tensor("positional_encoding.col_embed.weight")?;
        let half = width / 2;
        if width % 2 != 0
            || row.shape.as_slice() != [self.config.bev[0], half]
            || col.shape.as_slice() != [self.config.bev[1], half]
        {
            return Err("Invalid Qwen-Drive learned positional encoding".into());
        }
        let mut position = Vec::with_capacity(query.len());
        for y in 0..self.config.bev[0] {
            for x in 0..self.config.bev[1] {
                position.extend_from_slice(&col.values[x * half..(x + 1) * half]);
                position.extend_from_slice(&row.values[y * half..(y + 1) * half]);
            }
        }
        let reference_2d = (0..self.config.bev[0])
            .flat_map(|y| {
                (0..self.config.bev[1]).flat_map(move |x| {
                    [
                        (x as f32 + 0.5) / self.config.bev[1] as f32,
                        (y as f32 + 0.5) / self.config.bev[0] as f32,
                    ]
                })
            })
            .map(round_bf16)
            .collect::<Vec<_>>();
        let (reference_cam, visible) = self.project_reference_points(input.geometry, cameras)?;
        Ok(EncoderInput {
            features,
            spatial_shapes,
            query,
            position,
            reference_2d,
            reference_cam,
            visible,
            cameras,
        })
    }

    fn norm(
        &self,
        kind: &str,
        layer: usize,
        norm: usize,
        input: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>, String> {
        let root = format!("transformer.{kind}.layers.{layer}.norms.{norm}");
        layer_norm_rows(
            input,
            rows,
            self.config.embed_dim,
            self.tensor(&format!("{root}.weight"))?,
            self.tensor(&format!("{root}.bias"))?,
        )
    }

    fn temporal_attention(
        &self,
        layer: &EncoderLayer<'_>,
        query: &[f32],
        position: &[f32],
        reference: &[f32],
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        let rows = query.len() / self.config.embed_dim;
        let width = self.config.embed_dim;
        let channels = width / HEADS;
        let with_position = add_round(query, position)?;
        let mut paired = Vec::with_capacity(rows * width * 2);
        for row in 0..rows {
            paired.extend_from_slice(&query[row * width..(row + 1) * width]);
            paired.extend_from_slice(&with_position[row * width..(row + 1) * width]);
        }
        let offsets = layer.temporal_offsets.forward(&paired, rows, pool)?;
        let mut weights = layer.temporal_weights.forward(&paired, rows, pool)?;
        softmax_rows(&mut weights, DECODER_POINTS);
        let projected = layer.temporal_value.forward(query, rows, pool)?;
        let mut values = Vec::with_capacity(projected.len() * 2);
        values.extend_from_slice(&projected);
        values.extend_from_slice(&projected);
        let mut locations = vec![0.0; 2 * rows * HEADS * DECODER_POINTS * 2];
        let mut attention = vec![0.0; 2 * rows * HEADS * DECODER_POINTS];
        for queue in 0..2 {
            for row in 0..rows {
                for head in 0..HEADS {
                    for point in 0..DECODER_POINTS {
                        let source = ((row * HEADS + head) * 2 + queue) * DECODER_POINTS + point;
                        let destination =
                            ((queue * rows + row) * HEADS + head) * DECODER_POINTS + point;
                        locations[destination * 2] = round_bf16(
                            reference[row * 2] + offsets[source * 2] / self.config.bev[1] as f32,
                        );
                        locations[destination * 2 + 1] = round_bf16(
                            reference[row * 2 + 1]
                                + offsets[source * 2 + 1] / self.config.bev[0] as f32,
                        );
                        attention[destination] = weights[source];
                    }
                }
            }
        }
        let sampled = ms_deform_attn(&DeformAttentionInput {
            value: &values,
            spatial_shapes: &[self.config.bev],
            level_start_index: &[0],
            sampling_locations: &locations,
            attention_weights: &attention,
            batch: 2,
            queries: rows,
            heads: HEADS,
            channels,
            points: DECODER_POINTS,
        })?;
        let mut averaged = vec![0.0; query.len()];
        for index in 0..averaged.len() {
            averaged[index] = round_bf16((sampled[index] + sampled[query.len() + index]) * 0.5);
        }
        let output = layer.temporal_output.forward(&averaged, rows, pool)?;
        add_round(&output, query)
    }

    fn spatial_attention(
        &self,
        layer: &EncoderLayer<'_>,
        query: &[f32],
        encoder: &EncoderInput,
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        let rows = query.len() / self.config.embed_dim;
        let width = self.config.embed_dim;
        let channels = width / HEADS;
        let offsets = layer.spatial_offsets.forward(query, rows, pool)?;
        let mut weights = layer.spatial_weights.forward(query, rows, pool)?;
        softmax_rows(&mut weights, ENCODER_LEVELS * ENCODER_POINTS);
        let spatial = encoder.features[0].len() / width;
        let mut values = Vec::with_capacity(encoder.cameras * spatial * width);
        for camera in &encoder.features {
            values.extend(layer.spatial_value.forward(camera, spatial, pool)?);
        }
        let samples = encoder.cameras * rows * HEADS * ENCODER_LEVELS * ENCODER_POINTS;
        let mut locations = vec![0.0; samples * 2];
        let mut attention = vec![0.0; samples];
        for camera in 0..encoder.cameras {
            for row in 0..rows {
                for head in 0..HEADS {
                    for level in 0..ENCODER_LEVELS {
                        let [height, level_width] = encoder.spatial_shapes[level];
                        for point in 0..ENCODER_POINTS {
                            let source = ((row * HEADS + head) * ENCODER_LEVELS + level)
                                * ENCODER_POINTS
                                + point;
                            let destination =
                                (((camera * rows + row) * HEADS + head) * ENCODER_LEVELS + level)
                                    * ENCODER_POINTS
                                    + point;
                            let anchor = point % PILLAR_POINTS;
                            let reference = (camera * rows + row) * PILLAR_POINTS + anchor;
                            locations[destination * 2] = round_bf16(
                                encoder.reference_cam[reference * 2]
                                    + offsets[source * 2] / level_width as f32,
                            );
                            locations[destination * 2 + 1] = round_bf16(
                                encoder.reference_cam[reference * 2 + 1]
                                    + offsets[source * 2 + 1] / height as f32,
                            );
                            attention[destination] = weights[source];
                        }
                    }
                }
            }
        }
        let starts = encoder
            .spatial_shapes
            .iter()
            .scan(0usize, |offset, &[height, width]| {
                let current = *offset;
                *offset += height * width;
                Some(current)
            })
            .collect::<Vec<_>>();
        let sampled = ms_deform_attn(&DeformAttentionInput {
            value: &values,
            spatial_shapes: &encoder.spatial_shapes,
            level_start_index: &starts,
            sampling_locations: &locations,
            attention_weights: &attention,
            batch: encoder.cameras,
            queries: rows,
            heads: HEADS,
            channels,
            points: ENCODER_POINTS,
        })?;
        let mut slots = vec![0.0; query.len()];
        for row in 0..rows {
            let mut count = 0usize;
            for camera in 0..encoder.cameras {
                let visible = (0..PILLAR_POINTS)
                    .any(|anchor| encoder.visible[(camera * rows + row) * PILLAR_POINTS + anchor]);
                if visible {
                    count += 1;
                    let source = (camera * rows + row) * width;
                    for channel in 0..width {
                        slots[row * width + channel] += sampled[source + channel];
                    }
                }
            }
            let divisor = count.max(1) as f32;
            for channel in 0..width {
                slots[row * width + channel] = round_bf16(slots[row * width + channel] / divisor);
            }
        }
        let output = layer.spatial_output.forward(&slots, rows, pool)?;
        add_round(&output, query)
    }

    fn decoder_self_attention(
        &self,
        layer: &DecoderLayer<'_>,
        query: &[f32],
        position: &[f32],
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        let rows = self.config.num_queries;
        let width = self.config.embed_dim;
        let input = add_round(query, position)?;
        let q = layer.self_query.forward(&input, rows, pool)?;
        let k = layer.self_key.forward(&input, rows, pool)?;
        let v = layer.self_value.forward(query, rows, pool)?;
        let mut output = attention_scalar(
            &q,
            &k,
            &v,
            AttentionSpec {
                query_tokens: rows,
                key_tokens: rows,
                query_heads: HEADS,
                key_value_heads: HEADS,
                head_dim: width / HEADS,
                causal: false,
                scale: 1.0 / ((width / HEADS) as f32).sqrt(),
            },
        )?;
        output
            .iter_mut()
            .for_each(|value| *value = round_bf16(*value));
        output = layer.self_output.forward(&output, rows, pool)?;
        add_round(&output, query)
    }

    fn decoder_cross_attention(
        &self,
        layer: &DecoderLayer<'_>,
        query: &[f32],
        position: &[f32],
        references: &[f32],
        bev: &[f32],
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        let rows = self.config.num_queries;
        let width = self.config.embed_dim;
        let channels = width / HEADS;
        if references.len() != rows * 3 {
            return Err("Invalid Qwen-Drive decoder reference shape".into());
        }
        let input = add_round(query, position)?;
        let offsets = layer.cross_offsets.forward(&input, rows, pool)?;
        let mut weights = layer.cross_weights.forward(&input, rows, pool)?;
        softmax_rows(&mut weights, DECODER_POINTS);
        let bev_rows = self.config.bev[0] * self.config.bev[1];
        let value = layer.cross_value.forward(bev, bev_rows, pool)?;
        let mut locations = vec![0.0; rows * HEADS * DECODER_POINTS * 2];
        for row in 0..rows {
            for head in 0..HEADS {
                for point in 0..DECODER_POINTS {
                    let index = (row * HEADS + head) * DECODER_POINTS + point;
                    locations[index * 2] = round_bf16(
                        references[row * 3] + offsets[index * 2] / self.config.bev[1] as f32,
                    );
                    locations[index * 2 + 1] = round_bf16(
                        references[row * 3 + 1]
                            + offsets[index * 2 + 1] / self.config.bev[0] as f32,
                    );
                }
            }
        }
        let mut output = ms_deform_attn(&DeformAttentionInput {
            value: &value,
            spatial_shapes: &[self.config.bev],
            level_start_index: &[0],
            sampling_locations: &locations,
            attention_weights: &weights,
            batch: 1,
            queries: rows,
            heads: HEADS,
            channels,
            points: DECODER_POINTS,
        })?;
        output = layer.cross_output.forward(&output, rows, pool)?;
        add_round(&output, query)
    }

    fn project_reference_points(
        &self,
        geometry: &ViewGeometry<'_>,
        cameras: usize,
    ) -> Result<(Vec<f32>, Vec<bool>), String> {
        let ego2lidar = super::fpn::inverse_4x4(&geometry.lidar2ego[0])?;
        let mut projections = Vec::with_capacity(cameras);
        for lidar2img in geometry.lidar2img {
            let mut matrix = [0.0f32; 16];
            for row in 0..4 {
                for column in 0..4 {
                    matrix[row * 4 + column] = (0..4)
                        .map(|inner| lidar2img[row * 4 + inner] * ego2lidar[inner * 4 + column])
                        .sum();
                }
            }
            projections.push(matrix);
        }
        let [x0, y0, z0, x1, y1, z1] = self.config.det_pc_range;
        let queries = self.config.bev[0] * self.config.bev[1];
        let z_size = z1 - z0;
        let mut points = vec![0.0; cameras * queries * PILLAR_POINTS * 2];
        let mut visible = vec![false; cameras * queries * PILLAR_POINTS];
        for camera in 0..cameras {
            for y in 0..self.config.bev[0] {
                for x in 0..self.config.bev[1] {
                    let query = y * self.config.bev[1] + x;
                    let ego_x = (x as f32 + 0.5) / self.config.bev[1] as f32 * (x1 - x0) + x0;
                    let ego_y = (y as f32 + 0.5) / self.config.bev[0] as f32 * (y1 - y0) + y0;
                    for anchor in 0..PILLAR_POINTS {
                        let fraction = if PILLAR_POINTS == 1 {
                            0.5
                        } else {
                            (0.5 + anchor as f32 * (z_size - 1.0) / (PILLAR_POINTS - 1) as f32)
                                / z_size
                        };
                        let ego_z = fraction * z_size + z0;
                        let matrix = &projections[camera];
                        let image_x =
                            matrix[0] * ego_x + matrix[1] * ego_y + matrix[2] * ego_z + matrix[3];
                        let image_y =
                            matrix[4] * ego_x + matrix[5] * ego_y + matrix[6] * ego_z + matrix[7];
                        let depth =
                            matrix[8] * ego_x + matrix[9] * ego_y + matrix[10] * ego_z + matrix[11];
                        let point = (camera * queries + query) * PILLAR_POINTS + anchor;
                        let denominator = depth.max(1e-5);
                        let normalized_x = image_x / denominator / self.config.image_size[0] as f32;
                        let normalized_y = image_y / denominator / self.config.image_size[1] as f32;
                        points[point * 2] = round_bf16(normalized_x);
                        points[point * 2 + 1] = round_bf16(normalized_y);
                        visible[point] = depth > 1e-5
                            && normalized_x > 0.0
                            && normalized_x < 1.0
                            && normalized_y > 0.0
                            && normalized_y < 1.0;
                    }
                }
            }
        }
        Ok((points, visible))
    }
}
