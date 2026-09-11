use crate::models::dots::speaker::exp::torch28_exp;
use crate::models::qwen35::Qwen35DenseKvSnapshot;
use crate::ops::{bf16_to_f32, f32_to_bf16, rope_sin_cos_sleef};
use crate::{core::tensor::GGMLType, core::tensor::TensorSource};

use super::config::PlannerConfig;
use super::rng::TorchNormalRng;
use super::scene::PlanningScene;
use super::weights::{HeadLinear, HeadLinearScratch};
use crate::core::thread_pool::ComputePool;

fn bf16(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(value))
}

fn finite(values: &[f32], label: &str) -> Result<(), String> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(format!("{label} contains a non-finite value"))
    }
}

#[cfg(feature = "parity-trace")]
fn trace(name: &str, layer: Option<usize>, step: Option<usize>, shape: &[usize], values: &[f32]) {
    crate::parity_trace::report(crate::parity_trace::checkpoint_at(
        name, layer, step, shape, values,
    ));
}

#[cfg(not(feature = "parity-trace"))]
fn trace(
    _name: &str,
    _layer: Option<usize>,
    _step: Option<usize>,
    _shape: &[usize],
    _values: &[f32],
) {
}

pub fn time_embedding(times: &[f32], dim: usize, scale: f32) -> Result<Vec<f32>, String> {
    if dim < 4 || dim % 2 != 0 || !scale.is_finite() {
        return Err("Invalid Qwen-Drive time embedding parameters".into());
    }
    finite(times, "flow time")?;
    let half = dim / 2;
    let decay = 10_000.0f32.ln() / (half - 1) as f32;
    let frequencies: Vec<f32> = (0..half)
        .map(|index| torch28_exp(-(index as f32) * decay))
        .collect();
    let mut output = Vec::with_capacity(times.len() * dim);
    for &time in times {
        for &frequency in &frequencies {
            output.push(rope_sin_cos_sleef(scale * time * frequency).1);
        }
        for &frequency in &frequencies {
            output.push(rope_sin_cos_sleef(scale * time * frequency).0);
        }
    }
    Ok(output)
}

pub fn fourier_features(
    points: &[f32],
    point_dim: usize,
    num_features: usize,
    max_frequency: f32,
) -> Result<Vec<f32>, String> {
    if point_dim == 0
        || num_features != 16
        || points.len() % point_dim != 0
        || max_frequency.to_bits() != 16.0f32.to_bits()
    {
        return Err("Invalid Qwen-Drive Fourier feature parameters".into());
    }
    finite(points, "waypoints")?;
    let frequencies = [
        0x3f80_0000,
        0x3f9a_0000,
        0x3fb9_0000,
        0x3fdf_0000,
        0x4006_0000,
        0x4021_0000,
        0x4042_0000,
        0x4069_0000,
        0x408c_0000,
        0x40a9_0000,
        0x40cb_0000,
        0x40f4_0000,
        0x4113_0000,
        0x4131_0000,
        0x4154_0000,
        0x417f_0000,
    ]
    .map(f32::from_bits);
    let mut output = Vec::with_capacity(points.len() * num_features * 2);
    for point in points.chunks_exact(point_dim) {
        for &coordinate in point {
            for &frequency in &frequencies[..num_features] {
                output.push(bf16(
                    rope_sin_cos_sleef(coordinate * frequency * std::f32::consts::TAU).1,
                ));
            }
            for &frequency in &frequencies[..num_features] {
                output.push(bf16(
                    rope_sin_cos_sleef(coordinate * frequency * std::f32::consts::TAU).0,
                ));
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn waypoint_mrope(
    query: &[f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    sections: [usize; 3],
    anchor: [usize; 3],
    theta: f32,
) -> Result<Vec<f32>, String> {
    let pairs = rotary_dim / 2;
    if tokens == 0
        || heads == 0
        || rotary_dim == 0
        || rotary_dim % 2 != 0
        || rotary_dim > head_dim
        || sections.iter().sum::<usize>() != pairs
        || sections
            .iter()
            .enumerate()
            .any(|(axis, &count)| count != (axis..pairs).step_by(3).count())
        || query.len() != tokens.saturating_mul(heads).saturating_mul(head_dim)
        || !theta.is_finite()
        || theta <= 0.0
    {
        return Err("Invalid Qwen-Drive waypoint mRoPE parameters".into());
    }
    finite(query, "waypoint query")?;
    let inverse: Vec<f32> = (0..pairs)
        .map(|pair| bf16(1.0 / theta.powf((2 * pair) as f32 / rotary_dim as f32)))
        .collect();
    let mut output = query.to_vec();
    for token in 0..tokens {
        let mut cosine = vec![0.0; pairs];
        let mut sine = vec![0.0; pairs];
        for pair in 0..pairs {
            let axis = pair % 3;
            let position = bf16((anchor[axis] + token + 1) as f32);
            let angle = bf16(position * inverse[pair]);
            let (cos, sin) = rope_sin_cos_sleef(angle);
            cosine[pair] = bf16(cos);
            sine[pair] = bf16(sin);
        }
        for head in 0..heads {
            let start = (token * heads + head) * head_dim;
            let source = &query[start..start + rotary_dim];
            for index in 0..rotary_dim {
                let pair = index % pairs;
                let rotated = if index < pairs {
                    -source[index + pairs]
                } else {
                    source[index - pairs]
                };
                let left = bf16(source[index] * cosine[pair]);
                let right = bf16(rotated * sine[pair]);
                output[start + index] = bf16(left + right);
            }
        }
    }
    Ok(output)
}

fn wrap_heading(value: f32) -> f32 {
    (value + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

pub fn normalize_history(
    history: &[f32],
    points: usize,
    scale: [f32; 3],
) -> Result<Vec<f32>, String> {
    if points < 2
        || history.len() != points.saturating_mul(3)
        || scale
            .iter()
            .any(|value| !value.is_finite() || *value == 0.0)
    {
        return Err("Invalid Qwen-Drive history parameters".into());
    }
    finite(history, "history")?;
    let origin = &history[..3];
    let mut output = Vec::with_capacity((points - 1) * 3);
    for point in history.chunks_exact(3).skip(1) {
        output.push((point[0] - origin[0]) / scale[0]);
        output.push((point[1] - origin[1]) / scale[1]);
        let relative = wrap_heading(point[2] - origin[2]);
        output.push(wrap_heading(relative) / scale[2]);
    }
    Ok(output)
}

pub fn euler_update(
    values: &[f32],
    endpoint: &[f32],
    time: f64,
    min_one_minus_t: f64,
    step: f64,
) -> Result<Vec<f32>, String> {
    if values.len() != endpoint.len()
        || values.is_empty()
        || !time.is_finite()
        || !min_one_minus_t.is_finite()
        || min_one_minus_t <= 0.0
        || !step.is_finite()
        || step <= 0.0
    {
        return Err("Invalid Qwen-Drive Euler update".into());
    }
    finite(values, "waypoints")?;
    finite(endpoint, "endpoint")?;
    let remaining = (1.0 - time).max(min_one_minus_t) as f32;
    let step = step as f32;
    Ok(values
        .iter()
        .zip(endpoint)
        .map(|(&value, &endpoint)| value + (endpoint - value) / remaining * step)
        .collect())
}

fn checked_product(values: &[usize], label: &str) -> Result<usize, String> {
    values.iter().try_fold(1usize, |product, &value| {
        product
            .checked_mul(value)
            .ok_or_else(|| format!("Qwen-Drive {label} size overflow"))
    })
}

fn load_bf16<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    dims: &[u64],
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims || info.ggml_type != GGMLType::BF16 {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {dims:?} BF16",
            info.dims, info.ggml_type
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let expected = info
        .checked_nbytes()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    let values: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|bytes| bf16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
        .collect();
    finite(&values, name)?;
    Ok(values)
}

fn linear_rows(
    linear: &HeadLinear<'_>,
    input: &[f32],
    rows: usize,
    pool: &ComputePool,
    scratch: &mut HeadLinearScratch,
) -> Result<Vec<f32>, String> {
    let output_len = rows
        .checked_mul(linear.output())
        .ok_or("Qwen-Drive linear output size overflow")?;
    let mut output = vec![0.0; output_len];
    linear.forward_rows(input, rows, pool, &mut output, scratch)?;
    Ok(output)
}

fn silu_bf16(value: f32) -> f32 {
    bf16(value / (1.0 + torch28_exp(-value)))
}

fn sigmoid_bf16(value: f32) -> f32 {
    bf16(1.0 / (1.0 + torch28_exp(-value)))
}

// PyTorch 2.8's AArch64 float sum uses SumKernel.cpp's four-level cascade.
fn torch28_sum_squares(values: &[f32]) -> f32 {
    const LANES: usize = 4;
    const ILP: usize = 4;
    const LEVELS: usize = 4;

    let vector_count = values.len() / LANES;
    let cascade_count = vector_count / ILP;
    let level_power = if cascade_count <= 1 {
        4
    } else {
        ((usize::BITS - (cascade_count - 1).leading_zeros()) as usize / LEVELS).max(4)
    };
    let level_step = 1usize << level_power;
    let level_mask = level_step - 1;
    let mut cascade = [[[0.0f32; LANES]; ILP]; LEVELS];
    let mut index = 0;
    while index + level_step <= cascade_count {
        for _ in 0..level_step {
            let base = index * ILP * LANES;
            for row in 0..ILP {
                for lane in 0..LANES {
                    let value = values[base + row * LANES + lane];
                    cascade[0][row][lane] += value * value;
                }
            }
            index += 1;
        }
        for level in 1..LEVELS {
            for row in 0..ILP {
                for lane in 0..LANES {
                    cascade[level][row][lane] += cascade[level - 1][row][lane];
                    cascade[level - 1][row][lane] = 0.0;
                }
            }
            if index & (level_mask << (level * level_power)) != 0 {
                break;
            }
        }
    }
    while index < cascade_count {
        let base = index * ILP * LANES;
        for row in 0..ILP {
            for lane in 0..LANES {
                let value = values[base + row * LANES + lane];
                cascade[0][row][lane] += value * value;
            }
        }
        index += 1;
    }
    for level in 1..LEVELS {
        for row in 0..ILP {
            for lane in 0..LANES {
                cascade[0][row][lane] += cascade[level][row][lane];
            }
        }
    }

    for vector in cascade_count * ILP..vector_count {
        for lane in 0..LANES {
            let value = values[vector * LANES + lane];
            cascade[0][0][lane] += value * value;
        }
    }
    for row in 1..ILP {
        for lane in 0..LANES {
            cascade[0][0][lane] += cascade[0][row][lane];
        }
    }
    let mut sum = 0.0f32;
    for &value in &values[vector_count * LANES..] {
        sum += value * value;
    }
    for lane in 0..LANES {
        sum += cascade[0][0][lane];
    }
    sum
}

fn torch28_softmax_inplace(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut lanes = [0.0f32; 4];
    for (index, value) in values.iter_mut().enumerate() {
        *value = torch28_exp(*value - maximum);
        lanes[index % 4] += *value;
    }
    let inverse = 1.0 / ((lanes[0] + lanes[2]) + (lanes[1] + lanes[3]));
    values.iter_mut().for_each(|value| *value *= inverse);
}

fn torch28_fma_accumulate(output: &mut [f32], scale: f32, values: &[f32]) {
    for (output, &value) in output.iter_mut().zip(values) {
        *output = scale.mul_add(value, *output);
    }
}

#[allow(clippy::too_many_arguments)]
fn planner_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    query_tokens: usize,
    key_tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    layer: usize,
) -> Vec<f32> {
    debug_assert_eq!(query.len(), query_tokens * query_heads * head_dim);
    debug_assert_eq!(key.len(), key_tokens * key_value_heads * head_dim);
    debug_assert_eq!(value.len(), key.len());
    let mut output = vec![0.0; query.len()];
    let mut scores = vec![0.0; key_tokens];
    #[cfg(feature = "parity-trace")]
    let mut traced_scores = Vec::with_capacity(query_tokens * query_heads * key_tokens);
    #[cfg(feature = "parity-trace")]
    let mut traced_probabilities = Vec::with_capacity(query_tokens * query_heads * key_tokens);
    let heads_per_group = query_heads / key_value_heads;
    let scale = 1.0 / (head_dim as f32).sqrt().sqrt();
    for query_token in 0..query_tokens {
        for query_head in 0..query_heads {
            let query_start = (query_token * query_heads + query_head) * head_dim;
            let key_head = query_head / heads_per_group;
            for (key_token, score) in scores.iter_mut().enumerate() {
                let key_start = (key_token * key_value_heads + key_head) * head_dim;
                let mut sum = 0.0f32;
                for dimension in 0..head_dim {
                    sum += (query[query_start + dimension] * scale)
                        * (key[key_start + dimension] * scale);
                }
                *score = sum;
            }
            #[cfg(feature = "parity-trace")]
            traced_scores.extend_from_slice(&scores);
            torch28_softmax_inplace(&mut scores);
            #[cfg(feature = "parity-trace")]
            traced_probabilities.extend_from_slice(&scores);
            let output_row = &mut output[query_start..query_start + head_dim];
            for (key_token, &probability) in scores.iter().enumerate() {
                let value_start = (key_token * key_value_heads + key_head) * head_dim;
                torch28_fma_accumulate(
                    output_row,
                    probability,
                    &value[value_start..value_start + head_dim],
                );
            }
        }
    }
    #[cfg(feature = "parity-trace")]
    {
        trace(
            "qwen_drive.planner.attention_scores",
            Some(layer),
            None,
            &[1, query_tokens, query_heads, key_tokens],
            &traced_scores,
        );
        trace(
            "qwen_drive.planner.attention_probabilities",
            Some(layer),
            None,
            &[1, query_tokens, query_heads, key_tokens],
            &traced_probabilities,
        );
    }
    #[cfg(not(feature = "parity-trace"))]
    let _ = layer;
    output
}

fn rms_norm_rows(
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, String> {
    if input.len() != checked_product(&[rows, width], "RMSNorm input")? || weight.len() != width {
        return Err("Invalid Qwen-Drive RMSNorm shape".into());
    }
    let mut output = vec![0.0; input.len()];
    for (input, output) in input
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        let sum = torch28_sum_squares(input);
        let scale = 1.0 / (sum / width as f32 + eps).sqrt();
        for ((output, &value), &weight) in output.iter_mut().zip(input).zip(weight) {
            *output = bf16((value * scale) * weight);
        }
    }
    Ok(output)
}

struct Mlp<'a> {
    first: HeadLinear<'a>,
    second: HeadLinear<'a>,
}

impl<'a> Mlp<'a> {
    fn load<S: TensorSource + ?Sized>(
        source: &'a S,
        name: &str,
        input: usize,
        hidden: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            first: HeadLinear::load(source, &format!("{name}.0"), input, hidden, true)?,
            second: HeadLinear::load(source, &format!("{name}.2"), hidden, hidden, true)?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        rows: usize,
        trace_shape: &[usize],
        pool: &ComputePool,
        scratch: &mut HeadLinearScratch,
    ) -> Result<Vec<f32>, String> {
        let mut hidden = linear_rows(&self.first, input, rows, pool, scratch)?;
        trace(
            "qwen_drive.planner.mlp_linear",
            None,
            None,
            trace_shape,
            &hidden,
        );
        hidden
            .iter_mut()
            .for_each(|value| *value = silu_bf16(*value));
        trace(
            "qwen_drive.planner.mlp_silu",
            None,
            None,
            trace_shape,
            &hidden,
        );
        linear_rows(&self.second, &hidden, rows, pool, scratch)
    }
}

struct PlanningLayer<'a> {
    index: usize,
    input_norm: Vec<f32>,
    qkv: HeadLinear<'a>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    output: HeadLinear<'a>,
    post_attention_norm: Vec<f32>,
    gate_up: HeadLinear<'a>,
    down: HeadLinear<'a>,
    adaln: HeadLinear<'a>,
}

impl<'a> PlanningLayer<'a> {
    fn load<S: TensorSource + ?Sized>(
        source: &'a S,
        index: usize,
        config: &PlannerConfig,
    ) -> Result<Self, String> {
        let name = format!("qwen_drive_planner.layers.{index}");
        let attention = config
            .heads
            .checked_mul(config.head_dim)
            .ok_or("Qwen-Drive attention size overflow")?;
        let qkv = (attention * 2)
            .checked_add(config.kv_heads * config.head_dim * 2)
            .ok_or("Qwen-Drive QKV size overflow")?;
        Ok(Self {
            index,
            input_norm: load_bf16(
                source,
                &format!("{name}.input_layernorm.weight"),
                &[config.hidden as u64],
            )?,
            qkv: HeadLinear::load(
                source,
                &format!("{name}.qkv_proj"),
                config.hidden,
                qkv,
                false,
            )?,
            q_norm: load_bf16(
                source,
                &format!("{name}.q_norm.weight"),
                &[config.head_dim as u64],
            )?,
            k_norm: load_bf16(
                source,
                &format!("{name}.k_norm.weight"),
                &[config.head_dim as u64],
            )?,
            output: HeadLinear::load(
                source,
                &format!("{name}.o_proj"),
                attention,
                config.hidden,
                false,
            )?,
            post_attention_norm: load_bf16(
                source,
                &format!("{name}.post_attention_layernorm.weight"),
                &[config.hidden as u64],
            )?,
            gate_up: HeadLinear::load(
                source,
                &format!("{name}.gate_up_proj"),
                config.hidden,
                config.intermediate * 2,
                false,
            )?,
            down: HeadLinear::load(
                source,
                &format!("{name}.down_proj"),
                config.intermediate,
                config.hidden,
                false,
            )?,
            adaln: HeadLinear::load(
                source,
                &format!("{name}.adaln_modulation.1"),
                config.hidden,
                config.hidden * 6,
                true,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: &[f32],
        batch: usize,
        tokens: usize,
        scene: &Qwen35DenseKvSnapshot,
        anchor: [usize; 3],
        condition: &[f32],
        config: &PlannerConfig,
        pool: &ComputePool,
        scratch: &mut HeadLinearScratch,
    ) -> Result<Vec<f32>, String> {
        let rows = checked_product(&[batch, tokens], "planner rows")?;
        if hidden.len() != checked_product(&[rows, config.hidden], "planner hidden")?
            || condition.len() != checked_product(&[batch, config.hidden], "planner condition")?
        {
            return Err("Invalid Qwen-Drive planner layer input shape".into());
        }
        let mut adaln_input = condition.to_vec();
        adaln_input
            .iter_mut()
            .for_each(|value| *value = silu_bf16(*value));
        let modulation = linear_rows(&self.adaln, &adaln_input, batch, pool, scratch)?;
        trace(
            "qwen_drive.planner.modulation",
            Some(self.index),
            None,
            &[batch, 6 * config.hidden],
            &modulation,
        );

        let normalized = rms_norm_rows(
            hidden,
            rows,
            config.hidden,
            &self.input_norm,
            config.rms_eps,
        )?;
        trace(
            "qwen_drive.planner.attention_norm",
            Some(self.index),
            None,
            &[batch, tokens, config.hidden],
            &normalized,
        );
        let mut attention_input = vec![0.0; hidden.len()];
        for row in 0..rows {
            let sample = row / tokens;
            let modulation =
                &modulation[sample * 6 * config.hidden..(sample + 1) * 6 * config.hidden];
            for dimension in 0..config.hidden {
                let scaled = bf16(
                    normalized[row * config.hidden + dimension]
                        * bf16(1.0 + modulation[config.hidden + dimension]),
                );
                attention_input[row * config.hidden + dimension] =
                    bf16(scaled + modulation[dimension]);
            }
        }

        trace(
            "qwen_drive.planner.qkv_input",
            Some(self.index),
            None,
            &[batch, tokens, config.hidden],
            &attention_input,
        );
        let fused = linear_rows(&self.qkv, &attention_input, rows, pool, scratch)?;
        trace(
            "qwen_drive.planner.qkv",
            Some(self.index),
            None,
            &[batch, tokens, self.qkv.output()],
            &fused,
        );
        let attention_width = config.heads * config.head_dim;
        let kv_width = config.kv_heads * config.head_dim;
        let fused_group = (config.heads / config.kv_heads * 2 + 2) * config.head_dim;
        let mut query = vec![0.0; rows * attention_width];
        let mut gate = vec![0.0; query.len()];
        let mut key = vec![0.0; rows * kv_width];
        let mut value = vec![0.0; key.len()];
        for row in 0..rows {
            let fused = &fused[row * self.qkv.output()..(row + 1) * self.qkv.output()];
            for group in 0..config.kv_heads {
                let source = &fused[group * fused_group..(group + 1) * fused_group];
                let query_group = attention_width / config.kv_heads;
                let query_start = row * attention_width + group * query_group;
                query[query_start..query_start + query_group]
                    .copy_from_slice(&source[..query_group]);
                gate[query_start..query_start + query_group]
                    .copy_from_slice(&source[query_group..query_group * 2]);
                let kv_start = row * kv_width + group * config.head_dim;
                key[kv_start..kv_start + config.head_dim]
                    .copy_from_slice(&source[query_group * 2..query_group * 2 + config.head_dim]);
                value[kv_start..kv_start + config.head_dim].copy_from_slice(
                    &source
                        [query_group * 2 + config.head_dim..query_group * 2 + config.head_dim * 2],
                );
            }
        }
        query = rms_norm_rows(
            &query,
            rows * config.heads,
            config.head_dim,
            &self.q_norm,
            config.rms_eps,
        )?;
        key = rms_norm_rows(
            &key,
            rows * config.kv_heads,
            config.head_dim,
            &self.k_norm,
            config.rms_eps,
        )?;
        trace(
            "qwen_drive.planner.query_norm",
            Some(self.index),
            None,
            &[batch, tokens, config.heads, config.head_dim],
            &query,
        );
        trace(
            "qwen_drive.planner.key_norm",
            Some(self.index),
            None,
            &[batch, tokens, config.kv_heads, config.head_dim],
            &key,
        );

        let rotary_dim = config.mrope_sections.iter().sum::<usize>() * 2;
        let mut attended = vec![0.0; query.len()];
        let scene_width = scene.kv_heads * scene.head_dim;
        for sample in 0..batch {
            let query_range =
                sample * tokens * attention_width..(sample + 1) * tokens * attention_width;
            let key_range = sample * tokens * kv_width..(sample + 1) * tokens * kv_width;
            let rotated_query = waypoint_mrope(
                &query[query_range.clone()],
                tokens,
                config.heads,
                config.head_dim,
                rotary_dim,
                config.mrope_sections,
                anchor,
                config.rope_theta,
            )?;
            let rotated_key = waypoint_mrope(
                &key[key_range.clone()],
                tokens,
                config.kv_heads,
                config.head_dim,
                rotary_dim,
                config.mrope_sections,
                anchor,
                config.rope_theta,
            )?;
            let mut joint_key = Vec::with_capacity(scene.key.len() + rotated_key.len());
            joint_key.extend(scene.key.iter().copied().map(bf16));
            joint_key.extend_from_slice(&rotated_key);
            let mut joint_value = Vec::with_capacity(scene.value.len() + tokens * kv_width);
            joint_value.extend(scene.value.iter().copied().map(bf16));
            joint_value.extend_from_slice(&value[key_range]);
            debug_assert_eq!(scene_width, kv_width);
            trace(
                "qwen_drive.planner.rotary_query",
                Some(self.index),
                None,
                &[1, tokens, config.heads, config.head_dim],
                &rotated_query,
            );
            trace(
                "qwen_drive.planner.joint_key",
                Some(self.index),
                None,
                &[1, scene.tokens + tokens, config.kv_heads, config.head_dim],
                &joint_key,
            );
            trace(
                "qwen_drive.planner.joint_value",
                Some(self.index),
                None,
                &[1, scene.tokens + tokens, config.kv_heads, config.head_dim],
                &joint_value,
            );
            let mut output = planner_attention(
                &rotated_query,
                &joint_key,
                &joint_value,
                tokens,
                scene.tokens + tokens,
                config.heads,
                config.kv_heads,
                config.head_dim,
                self.index,
            );
            output.iter_mut().for_each(|value| *value = bf16(*value));
            trace(
                "qwen_drive.planner.attention_raw",
                Some(self.index),
                None,
                &[1, tokens, config.heads, config.head_dim],
                &output,
            );
            attended[query_range].copy_from_slice(&output);
        }
        for (value, &gate) in attended.iter_mut().zip(&gate) {
            *value = bf16(*value * sigmoid_bf16(gate));
        }
        trace(
            "qwen_drive.planner.attention",
            Some(self.index),
            None,
            &[batch, tokens, attention_width],
            &attended,
        );
        let projected = linear_rows(&self.output, &attended, rows, pool, scratch)?;
        let mut hidden_after_attention = vec![0.0; hidden.len()];
        for row in 0..rows {
            let sample = row / tokens;
            let modulation =
                &modulation[sample * 6 * config.hidden..(sample + 1) * 6 * config.hidden];
            for dimension in 0..config.hidden {
                let gated = bf16(
                    projected[row * config.hidden + dimension]
                        * bf16(1.0 + modulation[2 * config.hidden + dimension]),
                );
                hidden_after_attention[row * config.hidden + dimension] =
                    bf16(hidden[row * config.hidden + dimension] + gated);
            }
        }

        let normalized = rms_norm_rows(
            &hidden_after_attention,
            rows,
            config.hidden,
            &self.post_attention_norm,
            config.rms_eps,
        )?;
        let mut ffn_input = vec![0.0; normalized.len()];
        for row in 0..rows {
            let sample = row / tokens;
            let modulation =
                &modulation[sample * 6 * config.hidden..(sample + 1) * 6 * config.hidden];
            for dimension in 0..config.hidden {
                let scaled = bf16(
                    normalized[row * config.hidden + dimension]
                        * bf16(1.0 + modulation[4 * config.hidden + dimension]),
                );
                ffn_input[row * config.hidden + dimension] =
                    bf16(scaled + modulation[3 * config.hidden + dimension]);
            }
        }
        let gate_up = linear_rows(&self.gate_up, &ffn_input, rows, pool, scratch)?;
        let mut intermediate = vec![0.0; rows * config.intermediate];
        for row in 0..rows {
            let gate_start = row * config.intermediate * 2;
            for dimension in 0..config.intermediate {
                intermediate[row * config.intermediate + dimension] = bf16(
                    silu_bf16(gate_up[gate_start + dimension])
                        * gate_up[gate_start + config.intermediate + dimension],
                );
            }
        }
        let projected = linear_rows(&self.down, &intermediate, rows, pool, scratch)?;
        let mut output = hidden_after_attention.clone();
        for row in 0..rows {
            let sample = row / tokens;
            let modulation =
                &modulation[sample * 6 * config.hidden..(sample + 1) * 6 * config.hidden];
            for dimension in 0..config.hidden {
                let gated = bf16(
                    projected[row * config.hidden + dimension]
                        * bf16(1.0 + modulation[5 * config.hidden + dimension]),
                );
                output[row * config.hidden + dimension] =
                    bf16(output[row * config.hidden + dimension] + gated);
            }
        }
        trace(
            "qwen_drive.planner.layer_output",
            Some(self.index),
            None,
            &[batch, tokens, config.hidden],
            &output,
        );
        Ok(output)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryBatch {
    pub samples: usize,
    pub points: usize,
    pub values: Vec<f32>,
}

pub struct PlanningExpert<'a> {
    config: PlannerConfig,
    trajectory: HeadLinear<'a>,
    fourier: Mlp<'a>,
    waypoint_embedding: Vec<f32>,
    time: Mlp<'a>,
    nav: Mlp<'a>,
    ego: Mlp<'a>,
    history: Mlp<'a>,
    velocity: Mlp<'a>,
    acceleration: Mlp<'a>,
    query_fusion: Mlp<'a>,
    layers: Vec<PlanningLayer<'a>>,
    final_norm: Vec<f32>,
    output: HeadLinear<'a>,
}

pub(crate) fn validate_sample_request(
    config: &PlannerConfig,
    scene_cache: &[Qwen35DenseKvSnapshot],
    scene: &PlanningScene,
    samples: usize,
    steps: usize,
) -> Result<(), String> {
    let expected_caches = config.layers / config.layers_per_kv;
    if scene_cache.len() != expected_caches {
        return Err(format!(
            "Expected {expected_caches} Qwen-Drive scene caches, got {}",
            scene_cache.len()
        ));
    }
    for (index, cache) in scene_cache.iter().enumerate() {
        if cache.layer != index * config.layers_per_kv + (config.layers_per_kv - 1)
            || cache.tokens == 0
            || cache.kv_heads != config.kv_heads
            || cache.head_dim != config.head_dim
            || cache.key.len()
                != checked_product(
                    &[cache.tokens, cache.kv_heads, cache.head_dim],
                    "scene cache",
                )?
            || cache.value.len() != cache.key.len()
        {
            return Err(format!(
                "Invalid Qwen-Drive scene cache {index}: layer {} shape {:?}",
                cache.layer,
                cache.shape()
            ));
        }
        finite(&cache.key, "scene key")?;
        finite(&cache.value, "scene value")?;
    }
    if samples == 0 || steps == 0 {
        return Err("Qwen-Drive samples and steps must be greater than zero".into());
    }
    if scene.history.len() != config.history_points
        || scene.history_velocity.len() != config.history_points
        || scene.history_acceleration.len() != config.history_points
        || scene.nav_command >= config.nav_classes
        || scene.ego_status.iter().any(|value| !value.is_finite())
    {
        return Err("Invalid Qwen-Drive planning scene features".into());
    }
    finite(
        &scene.history.iter().flatten().copied().collect::<Vec<_>>(),
        "history",
    )?;
    finite(
        &scene
            .history_velocity
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        "history velocity",
    )?;
    finite(
        &scene
            .history_acceleration
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        "history acceleration",
    )
}

impl<'a> PlanningExpert<'a> {
    pub fn from_source<S: TensorSource + ?Sized>(source: &'a S) -> Result<Self, String> {
        let config = PlannerConfig::from_source(source)?;
        let name = "qwen_drive_planner";
        let fourier_input = config.point_dim * config.fourier_features * 2;
        let history_input = (config.history_points - 1) * config.point_dim + config.nav_classes;
        let dynamics_input = config.history_points * config.history_dynamics_dim;
        let layers = (0..config.layers)
            .map(|index| PlanningLayer::load(source, index, &config))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            trajectory: HeadLinear::load(
                source,
                &format!("{name}.trajectory_proj"),
                config.point_dim,
                config.hidden,
                true,
            )?,
            fourier: Mlp::load(
                source,
                &format!("{name}.fourier_encoder.net"),
                fourier_input,
                config.hidden,
            )?,
            waypoint_embedding: load_bf16(
                source,
                &format!("{name}.waypoint_embed.weight"),
                &[config.hidden as u64, config.future_points as u64],
            )?,
            time: Mlp::load(
                source,
                &format!("{name}.time_mlp"),
                config.time_embed_dim,
                config.hidden,
            )?,
            nav: Mlp::load(
                source,
                &format!("{name}.nav_mlp"),
                config.nav_classes,
                config.hidden,
            )?,
            ego: Mlp::load(
                source,
                &format!("{name}.ego_mlp"),
                config.ego_status_dim,
                config.hidden,
            )?,
            history: Mlp::load(
                source,
                &format!("{name}.history_encoder"),
                history_input,
                config.hidden,
            )?,
            velocity: Mlp::load(
                source,
                &format!("{name}.history_velocity_encoder"),
                dynamics_input,
                config.hidden,
            )?,
            acceleration: Mlp::load(
                source,
                &format!("{name}.history_acceleration_encoder"),
                dynamics_input,
                config.hidden,
            )?,
            query_fusion: Mlp::load(
                source,
                &format!("{name}.query_fusion"),
                config.hidden * 7,
                config.hidden,
            )?,
            final_norm: load_bf16(
                source,
                &format!("{name}.final_layernorm.weight"),
                &[config.hidden as u64],
            )?,
            output: HeadLinear::load(
                source,
                &format!("{name}.out_proj"),
                config.hidden,
                config.point_dim,
                true,
            )?,
            layers,
            config,
        })
    }

    pub fn config(&self) -> &PlannerConfig {
        &self.config
    }

    fn one_hot(&self, command: usize) -> Vec<f32> {
        let mut values = vec![0.0; self.config.nav_classes];
        if command < values.len() {
            values[command] = 1.0;
        }
        values
    }

    fn encode_history(
        &self,
        scene: &PlanningScene,
        pool: &ComputePool,
        scratch: &mut HeadLinearScratch,
    ) -> Result<[Vec<f32>; 3], String> {
        let history: Vec<f32> = scene.history.iter().flatten().copied().collect();
        let normalized = normalize_history(
            &history,
            self.config.history_points,
            self.config.trajectory_scale,
        )?;
        let mut pose = normalized.into_iter().map(bf16).collect::<Vec<_>>();
        pose.extend(self.one_hot(scene.nav_command));
        let velocity: Vec<f32> = scene
            .history_velocity
            .iter()
            .flatten()
            .copied()
            .map(bf16)
            .collect();
        let acceleration: Vec<f32> = scene
            .history_acceleration
            .iter()
            .flatten()
            .copied()
            .map(bf16)
            .collect();
        let shape = [1, self.config.hidden];
        let history = self.history.forward(&pose, 1, &shape, pool, scratch)?;
        trace("qwen_drive.planner.history", None, None, &shape, &history);
        let velocity = self.velocity.forward(&velocity, 1, &shape, pool, scratch)?;
        trace("qwen_drive.planner.velocity", None, None, &shape, &velocity);
        let acceleration = self
            .acceleration
            .forward(&acceleration, 1, &shape, pool, scratch)?;
        trace(
            "qwen_drive.planner.acceleration",
            None,
            None,
            &shape,
            &acceleration,
        );
        Ok([history, velocity, acceleration])
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_endpoint(
        &self,
        waypoints: &[f32],
        batch: usize,
        time: f32,
        history: &[Vec<f32>; 3],
        scene_cache: &[Qwen35DenseKvSnapshot],
        anchor: [usize; 3],
        nav_command: usize,
        ego_status: &[f32; 8],
        pool: &ComputePool,
        scratch: &mut HeadLinearScratch,
    ) -> Result<Vec<f32>, String> {
        let tokens = self.config.future_points;
        let rows = checked_product(&[batch, tokens], "planner endpoint rows")?;
        let waypoints_bf16: Vec<f32> = waypoints.iter().copied().map(bf16).collect();
        let time_embedding = time_embedding(
            &[time; 1],
            self.config.time_embed_dim,
            self.config.time_embed_scale,
        )?
        .into_iter()
        .map(bf16)
        .collect::<Vec<_>>();
        let condition_shape = [1, self.config.hidden];
        let time_condition =
            self.time
                .forward(&time_embedding, 1, &condition_shape, pool, scratch)?;
        trace(
            "qwen_drive.planner.time_condition",
            None,
            None,
            &condition_shape,
            &time_condition,
        );
        let trajectory = linear_rows(&self.trajectory, &waypoints_bf16, rows, pool, scratch)?;
        trace(
            "qwen_drive.planner.trajectory_embedding",
            None,
            None,
            &[batch, tokens, self.config.hidden],
            &trajectory,
        );
        let fourier = fourier_features(
            &waypoints_bf16,
            self.config.point_dim,
            self.config.fourier_features,
            self.config.fourier_max_frequency,
        )?;
        let token_shape = [batch, tokens, self.config.hidden];
        let fourier = self
            .fourier
            .forward(&fourier, rows, &token_shape, pool, scratch)?;
        trace(
            "qwen_drive.planner.fourier_embedding",
            None,
            None,
            &[batch, tokens, self.config.hidden],
            &fourier,
        );

        let mut fused = Vec::with_capacity(rows * self.config.hidden * 7);
        for row in 0..rows {
            let waypoint = row % tokens;
            fused.extend_from_slice(
                &trajectory[row * self.config.hidden..(row + 1) * self.config.hidden],
            );
            fused.extend_from_slice(
                &fourier[row * self.config.hidden..(row + 1) * self.config.hidden],
            );
            fused.extend_from_slice(&time_condition);
            fused.extend_from_slice(&history[0]);
            fused.extend_from_slice(
                &self.waypoint_embedding
                    [waypoint * self.config.hidden..(waypoint + 1) * self.config.hidden],
            );
            fused.extend_from_slice(&history[1]);
            fused.extend_from_slice(&history[2]);
        }
        let mut hidden = self
            .query_fusion
            .forward(&fused, rows, &token_shape, pool, scratch)?;
        trace(
            "qwen_drive.planner.query",
            None,
            None,
            &[batch, tokens, self.config.hidden],
            &hidden,
        );
        let nav = self.nav.forward(
            &self.one_hot(nav_command),
            1,
            &condition_shape,
            pool,
            scratch,
        )?;
        let ego = self.ego.forward(
            &ego_status.iter().copied().map(bf16).collect::<Vec<_>>(),
            1,
            &condition_shape,
            pool,
            scratch,
        )?;
        let mut condition = Vec::with_capacity(batch * self.config.hidden);
        for _ in 0..batch {
            for dimension in 0..self.config.hidden {
                condition.push(bf16(
                    bf16(time_condition[dimension] + nav[dimension]) + ego[dimension],
                ));
            }
        }
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(
                &hidden,
                batch,
                tokens,
                &scene_cache[index / self.config.layers_per_kv],
                anchor,
                &condition,
                &self.config,
                pool,
                scratch,
            )?;
        }
        let normalized = rms_norm_rows(
            &hidden,
            rows,
            self.config.hidden,
            &self.final_norm,
            self.config.rms_eps,
        )?;
        trace(
            "qwen_drive.planner.final_norm",
            None,
            None,
            &[batch, tokens, self.config.hidden],
            &normalized,
        );
        let endpoint = linear_rows(&self.output, &normalized, rows, pool, scratch)?;
        trace(
            "qwen_drive.planner.endpoint",
            None,
            None,
            &[batch, tokens, self.config.point_dim],
            &endpoint,
        );
        Ok(endpoint)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        scene_cache: &[Qwen35DenseKvSnapshot],
        position_anchor: [usize; 3],
        scene: &PlanningScene,
        samples: usize,
        steps: usize,
        seed: u64,
        pool: &ComputePool,
    ) -> Result<TrajectoryBatch, String> {
        validate_sample_request(&self.config, scene_cache, scene, samples, steps)?;
        let values = checked_product(
            &[samples, self.config.future_points, self.config.point_dim],
            "trajectory",
        )?;
        let mut waypoints = Vec::with_capacity(values);
        for sample in 0..samples {
            waypoints.extend(
                TorchNormalRng::normal_f32(seed.wrapping_add(sample as u64), values / samples)
                    .into_iter()
                    .map(|value| value * self.config.noise_init_std),
            );
        }
        trace(
            "qwen_drive.planner.noise",
            None,
            None,
            &[samples, self.config.future_points, self.config.point_dim],
            &waypoints,
        );
        let mut scratch = HeadLinearScratch::default();
        let history = self.encode_history(scene, pool, &mut scratch)?;
        let step = 1.0 / steps as f64;
        for index in 0..steps {
            let time = index as f64 * step;
            let endpoint = self.predict_endpoint(
                &waypoints,
                samples,
                time as f32,
                &history,
                scene_cache,
                position_anchor,
                scene.nav_command,
                &scene.ego_status,
                pool,
                &mut scratch,
            )?;
            waypoints = euler_update(
                &waypoints,
                &endpoint,
                time,
                self.config.min_one_minus_t as f64,
                step,
            )?;
            trace(
                "qwen_drive.planner.euler",
                None,
                Some(index),
                &[samples, self.config.future_points, self.config.point_dim],
                &waypoints,
            );
        }
        for point in waypoints.chunks_exact_mut(self.config.point_dim) {
            for (value, scale) in point.iter_mut().zip(self.config.trajectory_scale) {
                *value *= scale;
            }
            point[2] = wrap_heading(point[2]);
        }
        trace(
            "qwen_drive.planner.trajectory",
            None,
            None,
            &[samples, self.config.future_points, self.config.point_dim],
            &waypoints,
        );
        Ok(TrajectoryBatch {
            samples,
            points: self.config.future_points,
            values: waypoints,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{torch28_fma_accumulate, torch28_softmax_inplace, torch28_sum_squares};

    #[test]
    fn torch28_attention_value_dot_uses_fused_multiply_add() {
        let probability = [
            0x378f_3dc3, 0x37c5_84aa, 0x37be_b637, 0x3d54_1116, 0x3d9c_6d2d,
            0x3db9_f2c2, 0x3df8_4ffe, 0x3dde_ec2d, 0x3dbe_cdaa, 0x3da0_1117,
            0x3d80_90dd, 0x3d34_f463, 0x3cef_c69a, 0x3cad_2431, 0x3c7e_8027,
            0x3c25_f86e, 0x3bfa_8aab, 0x3bd0_c3ab, 0x3bae_0b51, 0x3b74_de75,
            0x3b78_dfc5, 0x3b61_2095, 0x3b5a_d315, 0x3b3c_b799, 0x3b62_19a2,
            0x3b7c_f65e, 0x3b8b_5607, 0x3b8f_4d78, 0x3b93_f546, 0x3ba3_e8e6,
            0x3b91_118e, 0x3bc4_b931, 0x3bac_d572, 0x3b7f_93f5, 0x3b91_7d19,
            0x3b8d_ec74, 0x3b8f_a709, 0x3b84_430c, 0x3b7e_caba, 0x3b78_f01d,
            0x3b8c_85b6, 0x3bc2_fe1f, 0x3bd7_1f84, 0x3c09_8973, 0x3c28_d0a0,
            0x3c29_86fe, 0x3c08_d3e1, 0x3bff_21cf, 0x3c0a_0cdf, 0x3bde_4a47,
            0x3bbe_9c80, 0x3b3c_da3c, 0x3b89_99db,
        ]
        .map(f32::from_bits);
        let value = [
            0xbec4_0000, 0xbebc_0000, 0xbeb4_0000, 0x3e89_0000, 0xc03a_0000,
            0xbd5b_0000, 0xbfaa_0000, 0xbe13_0000, 0xbf75_0000, 0xbf55_0000,
            0x4021_0000, 0x3efc_0000, 0x3dbe_0000, 0xc02b_0000, 0x402c_0000,
            0x3f2a_0000, 0x401d_0000, 0xbf61_0000, 0x3f6d_0000, 0x3fb0_0000,
            0x4036_0000, 0xbea3_0000, 0x4004_0000, 0xbfff_0000, 0x4010_0000,
            0x3fc3_0000, 0x405b_0000, 0x3f9f_0000, 0x3e57_0000, 0x3ffb_0000,
            0x3f84_0000, 0x400d_0000, 0xbf9d_0000, 0xc07e_0000, 0xc00c_0000,
            0xbfc8_0000, 0xbffe_0000, 0xc022_0000, 0xc021_0000, 0xc063_0000,
            0xbf53_0000, 0x3f93_0000, 0x400d_0000, 0x4039_0000, 0x407c_0000,
            0x40de_0000, 0x4078_0000, 0x400f_0000, 0x4084_0000, 0x409d_0000,
            0x40fb_0000, 0x4074_0000, 0x40d0_0000,
        ]
        .map(f32::from_bits);

        let mut output = [0.0];
        for (&probability, &value) in probability.iter().zip(&value) {
            torch28_fma_accumulate(&mut output, probability, &[value]);
        }
        assert_eq!(output[0].to_bits(), 0x3b14_7f07);
    }

    #[test]
    fn torch28_sum_squares_uses_cpu_cascade_order() {
        let mut state = 42_u32;
        let values = (0..1024)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f32::from_bits(0x3f00_0000 | (state & 0x007f_ffff))
            })
            .collect::<Vec<_>>();

        assert_eq!(torch28_sum_squares(&values).to_bits(), 0x4417_9c1f);
    }

    #[test]
    fn torch28_softmax_uses_neon_lane_reduction() {
        let mut state = 42_u32;
        let mut values = (0..53)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f32::from_bits(0x3f00_0000 | (state & 0x007f_ffff))
            })
            .collect::<Vec<_>>();

        torch28_softmax_inplace(&mut values);
        assert_eq!(values[0].to_bits(), 0x3c84_6566);
        assert_eq!(values[17].to_bits(), 0x3c95_9c2b);
        assert_eq!(values[52].to_bits(), 0x3cb0_1d13);
    }
}
