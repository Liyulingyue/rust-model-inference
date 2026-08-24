use std::sync::Arc;

use half::f16;

use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::{dot_f32, rms_norm, rms_norm_inplace, silu, softmax_inplace};

use super::{linear_into, validate_component, Component, Q8Scratch, ZImageOptions};

const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;
const PATCH_SIZE: usize = 2;
const ROPE_THETA: f32 = 256.0;
const ROPE_AXES: [usize; 3] = [32, 48, 48];
const ROPE_HEAD_WIDTH: usize = 128;
const SEQUENCE_MULTIPLE: usize = 32;
const LATENT_CHANNELS: usize = 16;
const PATCH_WIDTH: usize = LATENT_CHANNELS * PATCH_SIZE * PATCH_SIZE;
const CAP_WIDTH: usize = 2_560;
const HIDDEN: usize = 3_840;
const HEADS: usize = 30;
const QKV_WIDTH: usize = HIDDEN * 3;
const FFN_WIDTH: usize = 10_240;
const TIME_WIDTH: usize = 256;
const TIME_HIDDEN: usize = 1_024;
const MAIN_LAYERS: usize = 30;
const REFINER_LAYERS: usize = 2;
const RMS_EPSILON: f32 = 1e-5;
const QK_RMS_EPSILON: f32 = 1e-6;
const FINAL_NORM_EPSILON: f32 = 1e-6;

struct ModulationWeights {
    matrix: String,
    bias: Vec<f32>,
}

struct BlockWeights {
    qkv: String,
    out: String,
    w1: String,
    w2: String,
    w3: String,
    attention_norm1: Vec<f32>,
    attention_norm2: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn_norm1: Vec<f32>,
    ffn_norm2: Vec<f32>,
    modulation: Option<ModulationWeights>,
}

pub(crate) struct ZImageDit {
    source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    cap_norm: Vec<f32>,
    cap_bias: Vec<f32>,
    x_bias: Vec<f32>,
    cap_pad_token: Vec<f32>,
    x_pad_token: Vec<f32>,
    time_0_bias: Vec<f32>,
    time_2_bias: Vec<f32>,
    context_refiners: Vec<BlockWeights>,
    noise_refiners: Vec<BlockWeights>,
    layers: Vec<BlockWeights>,
    final_modulation_bias: Vec<f32>,
    final_bias: Vec<f32>,
}

pub(crate) struct DitScratch {
    text: Vec<f32>,
    image: Vec<f32>,
    tokens: Vec<f32>,
    qkv: Vec<f32>,
    attention: Vec<f32>,
    ffn: Vec<f32>,
    scores: Vec<f32>,
    modulation: Vec<f32>,
    rope: Vec<f32>,
    patches: Vec<f32>,
    velocity: Vec<f32>,
    time_frequency: [f32; TIME_WIDTH],
    time_hidden: [f32; TIME_HIDDEN],
    time: [f32; TIME_WIDTH],
    q8: Q8Scratch,
}

impl DitScratch {
    pub(crate) fn new() -> Self {
        Self {
            text: Vec::new(),
            image: Vec::new(),
            tokens: Vec::new(),
            qkv: Vec::new(),
            attention: Vec::new(),
            ffn: Vec::new(),
            scores: Vec::new(),
            modulation: Vec::new(),
            rope: Vec::new(),
            patches: Vec::new(),
            velocity: Vec::new(),
            time_frequency: [0.0; TIME_WIDTH],
            time_hidden: [0.0; TIME_HIDDEN],
            time: [0.0; TIME_WIDTH],
            q8: Q8Scratch::new(FFN_WIDTH),
        }
    }

    fn prepare(
        &mut self,
        text_tokens: usize,
        image_tokens: usize,
        latent_values: usize,
    ) -> Result<(), String> {
        let total_tokens = text_tokens
            .checked_add(image_tokens)
            .ok_or("Z-Image token count overflow")?;
        resize_zeroed(
            &mut self.text,
            checked_product(text_tokens, HIDDEN, "text tokens")?,
            "Z-Image text tokens",
        )?;
        resize_zeroed(
            &mut self.image,
            checked_product(image_tokens, HIDDEN, "image tokens")?,
            "Z-Image image tokens",
        )?;
        resize_zeroed(
            &mut self.tokens,
            checked_product(total_tokens, HIDDEN, "joint tokens")?,
            "Z-Image joint tokens",
        )?;
        resize_zeroed(
            &mut self.qkv,
            checked_product(total_tokens, QKV_WIDTH, "QKV")?,
            "Z-Image QKV",
        )?;
        resize_zeroed(
            &mut self.attention,
            checked_product(total_tokens, HIDDEN, "attention")?,
            "Z-Image attention",
        )?;
        resize_zeroed(
            &mut self.ffn,
            checked_product(total_tokens, FFN_WIDTH, "FFN")?,
            "Z-Image FFN",
        )?;
        resize_zeroed(&mut self.scores, total_tokens, "Z-Image attention scores")?;
        resize_zeroed(&mut self.modulation, HIDDEN * 4, "Z-Image AdaLN modulation")?;
        resize_zeroed(
            &mut self.rope,
            checked_product(total_tokens, ROPE_HEAD_WIDTH, "RoPE")?,
            "Z-Image RoPE",
        )?;
        resize_zeroed(
            &mut self.patches,
            checked_product(image_tokens, PATCH_WIDTH, "patches")?,
            "Z-Image patches",
        )?;
        resize_zeroed(&mut self.velocity, latent_values, "Z-Image velocity")
    }
}

fn checked_product(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Z-Image {name} shape overflow"))
}

fn require_finite(values: &[f32], name: &str) -> Result<(), String> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(format!("Non-finite Z-Image {name}"))
    }
}

fn resize_zeroed(values: &mut Vec<f32>, len: usize, name: &str) -> Result<(), String> {
    if values.capacity() < len {
        values
            .try_reserve_exact(len - values.len())
            .map_err(|error| format!("Failed to allocate {name}: {error}"))?;
    }
    values.resize(len, 0.0);
    Ok(())
}

fn load_f32_vector(source: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [len as u64] || info.ggml_type != GGMLType::F32 {
        return Err(format!("Invalid vector tensor: {name}"));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != checked_product(len, 4, name)? {
        return Err(format!("Invalid {name} byte length"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn load_f16_vector(source: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [len as u64, 1] || info.ggml_type != GGMLType::F16 {
        return Err(format!("Invalid vector tensor: {name}"));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != checked_product(len, 2, name)? {
        return Err(format!("Invalid {name} byte length"));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32())
        .collect())
}

fn load_block(
    source: &dyn TensorSource,
    prefix: String,
    modulated: bool,
) -> Result<BlockWeights, String> {
    let vector = |suffix: &str, len| load_f32_vector(source, &format!("{prefix}.{suffix}"), len);
    Ok(BlockWeights {
        qkv: format!("{prefix}.attention.qkv.weight"),
        out: format!("{prefix}.attention.out.weight"),
        w1: format!("{prefix}.feed_forward.w1.weight"),
        w2: format!("{prefix}.feed_forward.w2.weight"),
        w3: format!("{prefix}.feed_forward.w3.weight"),
        attention_norm1: vector("attention_norm1.weight", HIDDEN)?,
        attention_norm2: vector("attention_norm2.weight", HIDDEN)?,
        q_norm: vector("attention.q_norm.weight", ROPE_HEAD_WIDTH)?,
        k_norm: vector("attention.k_norm.weight", ROPE_HEAD_WIDTH)?,
        ffn_norm1: vector("ffn_norm1.weight", HIDDEN)?,
        ffn_norm2: vector("ffn_norm2.weight", HIDDEN)?,
        modulation: if modulated {
            Some(ModulationWeights {
                matrix: format!("{prefix}.adaLN_modulation.0.weight"),
                bias: vector("adaLN_modulation.0.bias", HIDDEN * 4)?,
            })
        } else {
            None
        },
    })
}

pub(crate) fn pad_rows_to_32(
    values: &mut Vec<f32>,
    rows: usize,
    pad_token: &[f32],
) -> Result<(), String> {
    if pad_token.is_empty() || values.len() != checked_product(rows, pad_token.len(), "rows")? {
        return Err("Invalid Z-Image rows for padding".into());
    }
    let padded_rows = padded_to_sequence_multiple(rows)?;
    let target = checked_product(padded_rows, pad_token.len(), "padded rows")?;
    values
        .try_reserve_exact(target.saturating_sub(values.len()))
        .map_err(|error| format!("Failed to allocate Z-Image padding: {error}"))?;
    while values.len() < target {
        values.extend_from_slice(pad_token);
    }
    Ok(())
}

pub(crate) fn euler_flow_step(
    latent: &mut [f32],
    velocity: &[f32],
    sigma: f32,
    sigma_next: f32,
) -> Result<(), String> {
    if latent.len() != velocity.len() {
        return Err(format!(
            "Invalid Z-Image Euler buffer lengths: latent {}, velocity {}",
            latent.len(),
            velocity.len()
        ));
    }
    for (x, v) in latent.iter_mut().zip(velocity) {
        *x += (sigma_next - sigma) * *v;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct AdaLnModulation<'a> {
    scale_msa: &'a [f32],
    gate_msa: &'a [f32],
    scale_mlp: &'a [f32],
    gate_mlp: &'a [f32],
}

fn split_adaln_modulation(values: &[f32], hidden: usize) -> Result<AdaLnModulation<'_>, String> {
    let expected = hidden
        .checked_mul(4)
        .ok_or("Z-Image AdaLN width overflow")?;
    if hidden == 0 || values.len() != expected {
        return Err(format!(
            "Invalid Z-Image AdaLN length: expected {expected}, got {}",
            values.len()
        ));
    }
    let (scale_msa, rest) = values.split_at(hidden);
    let (gate_msa, rest) = rest.split_at(hidden);
    let (scale_mlp, gate_mlp) = rest.split_at(hidden);
    Ok(AdaLnModulation {
        scale_msa,
        gate_msa,
        scale_mlp,
        gate_mlp,
    })
}

fn scale_modulated_branch(values: &mut [f32], scales: Option<&[f32]>) -> Result<(), String> {
    let Some(scales) = scales else {
        return Ok(());
    };
    if values.len() != scales.len() {
        return Err("Invalid Z-Image AdaLN scale buffers".into());
    }
    for (value, scale) in values.iter_mut().zip(scales) {
        *value *= 1.0 + *scale;
    }
    Ok(())
}

fn add_modulated_residual(
    tokens: &mut [f32],
    residual: &[f32],
    gates: Option<&[f32]>,
) -> Result<(), String> {
    if tokens.len() != residual.len() || gates.is_some_and(|values| values.len() != tokens.len()) {
        return Err("Invalid Z-Image AdaLN residual buffers".into());
    }
    match gates {
        Some(gates) => {
            for ((token, residual), gate) in tokens.iter_mut().zip(residual).zip(gates) {
                *token += *residual * gate.tanh();
            }
        }
        None => {
            for (token, residual) in tokens.iter_mut().zip(residual) {
                *token += *residual;
            }
        }
    }
    Ok(())
}

fn real_image_row<'a>(
    tokens: &'a [f32],
    padded_text_rows: usize,
    real_image_rows: usize,
    padded_image_rows: usize,
    image_row: usize,
    width: usize,
) -> Result<&'a [f32], String> {
    if width == 0 || real_image_rows > padded_image_rows {
        return Err("Invalid Z-Image final image-row shape".into());
    }
    let total_rows = padded_text_rows
        .checked_add(padded_image_rows)
        .ok_or("Z-Image final row count overflow")?;
    let expected = checked_product(total_rows, width, "final token rows")?;
    if tokens.len() != expected {
        return Err(format!(
            "Invalid Z-Image final token length: expected {expected}, got {}",
            tokens.len()
        ));
    }
    if image_row >= real_image_rows {
        return Err(format!(
            "Invalid Z-Image real image row: {image_row} >= {real_image_rows}"
        ));
    }
    let row = padded_text_rows
        .checked_add(image_row)
        .ok_or("Z-Image final row offset overflow")?;
    let start = checked_product(row, width, "final row offset")?;
    Ok(&tokens[start..start + width])
}

fn sign_and_unpatchify_image(
    patches: &mut [f32],
    bias: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    output: &mut [f32],
) -> Result<(), String> {
    let (patch_height, patch_width, patch_width_channels) = patch_shape(channels, height, width)?;
    let patch_rows = checked_product(patch_height, patch_width, "output patch rows")?;
    let expected_patches =
        checked_product(patch_rows, patch_width_channels, "output patch values")?;
    let expected_output = channels
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or("Z-Image latent shape overflow")?;
    if patches.len() != expected_patches
        || bias.len() != patch_width_channels
        || output.len() != expected_output
    {
        return Err("Invalid Z-Image final output buffers".into());
    }

    for patch in patches.chunks_exact_mut(patch_width_channels) {
        for (value, bias) in patch.iter_mut().zip(bias) {
            *value = -(*value + *bias);
        }
    }
    unpatchify_latent_into(patches, channels, height, width, output)
}

fn timestep_embedding(timestep: f32, output: &mut [f32]) -> Result<(), String> {
    if output.is_empty() || output.len() % 2 != 0 {
        return Err("Z-Image timestep embedding width must be positive and even".into());
    }
    let half = output.len() / 2;
    for index in 0..half {
        let frequency = (-10_000.0f32.ln() * index as f32 / half as f32).exp();
        let angle = timestep * frequency;
        output[index] = angle.cos();
        output[index + half] = angle.sin();
    }
    Ok(())
}

fn rotate_interleaved_inplace(values: &mut [f32], rope: &[f32]) -> Result<(), String> {
    if values.len() != rope.len() || values.len() % 2 != 0 {
        return Err("Invalid compact Z-Image RoPE slice".into());
    }
    for (value, rotation) in values.chunks_exact_mut(2).zip(rope.chunks_exact(2)) {
        let first = value[0];
        let second = value[1];
        value[0] = first * rotation[0] - second * rotation[1];
        value[1] = first * rotation[1] + second * rotation[0];
    }
    Ok(())
}

fn attention_into(
    qkv: &[f32],
    tokens: usize,
    heads: usize,
    head_width: usize,
    scores: &mut [f32],
    output: &mut [f32],
) -> Result<(), String> {
    let hidden = checked_product(heads, head_width, "attention hidden")?;
    let qkv_width = checked_product(hidden, 3, "attention QKV")?;
    if qkv.len() != checked_product(tokens, qkv_width, "attention QKV rows")?
        || output.len() != checked_product(tokens, hidden, "attention output")?
        || scores.len() < tokens
    {
        return Err("Invalid Z-Image attention buffers".into());
    }
    let scale = 1.0 / (head_width as f32).sqrt();
    for query in 0..tokens {
        for head in 0..heads {
            let query_start = query * qkv_width + head * head_width;
            let query_values = &qkv[query_start..query_start + head_width];
            for key in 0..tokens {
                let key_start = key * qkv_width + hidden + head * head_width;
                scores[key] = dot_f32(
                    query_values,
                    &qkv[key_start..key_start + head_width],
                    head_width,
                ) * scale;
            }
            softmax_inplace(&mut scores[..tokens]);
            let output_start = query * hidden + head * head_width;
            for dimension in 0..head_width {
                let mut value = 0.0f32;
                for key in 0..tokens {
                    value += scores[key]
                        * qkv[key * qkv_width + hidden * 2 + head * head_width + dimension];
                }
                output[output_start + dimension] = value;
            }
        }
    }
    Ok(())
}

fn layer_norm_no_affine(input: &[f32], output: &mut [f32], eps: f32) -> Result<(), String> {
    if input.is_empty() || input.len() != output.len() {
        return Err("Invalid Z-Image final LayerNorm buffers".into());
    }
    let mean =
        (input.iter().map(|value| f64::from(*value)).sum::<f64>() / input.len() as f64) as f32;
    let variance = (input
        .iter()
        .map(|value| {
            let centered = *value - mean;
            f64::from(centered * centered)
        })
        .sum::<f64>()
        / input.len() as f64) as f32;
    let inverse = 1.0 / (variance + eps).sqrt();
    for (output, input) in output.iter_mut().zip(input) {
        *output = (*input - mean) * inverse;
    }
    Ok(())
}

impl ZImageDit {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        validate_component(source.as_ref(), Component::Dit)?;
        let context_refiners = (0..REFINER_LAYERS)
            .map(|layer| load_block(source.as_ref(), format!("context_refiner.{layer}"), false))
            .collect::<Result<Vec<_>, _>>()?;
        let noise_refiners = (0..REFINER_LAYERS)
            .map(|layer| load_block(source.as_ref(), format!("noise_refiner.{layer}"), true))
            .collect::<Result<Vec<_>, _>>()?;
        let layers = (0..MAIN_LAYERS)
            .map(|layer| load_block(source.as_ref(), format!("layers.{layer}"), true))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            cap_norm: load_f32_vector(source.as_ref(), "cap_embedder.0.weight", CAP_WIDTH)?,
            cap_bias: load_f32_vector(source.as_ref(), "cap_embedder.1.bias", HIDDEN)?,
            x_bias: load_f32_vector(source.as_ref(), "x_embedder.bias", HIDDEN)?,
            cap_pad_token: load_f16_vector(source.as_ref(), "cap_pad_token", HIDDEN)?,
            x_pad_token: load_f16_vector(source.as_ref(), "x_pad_token", HIDDEN)?,
            time_0_bias: load_f32_vector(source.as_ref(), "t_embedder.mlp.0.bias", TIME_HIDDEN)?,
            time_2_bias: load_f32_vector(source.as_ref(), "t_embedder.mlp.2.bias", TIME_WIDTH)?,
            context_refiners,
            noise_refiners,
            layers,
            final_modulation_bias: load_f32_vector(
                source.as_ref(),
                "final_layer.adaLN_modulation.1.bias",
                HIDDEN,
            )?,
            final_bias: load_f32_vector(source.as_ref(), "final_layer.linear.bias", PATCH_WIDTH)?,
            source,
            pool,
        })
    }

    pub(crate) fn predict_flow(
        &self,
        latent: &[f32],
        latent_side: usize,
        context: &[f32],
        context_tokens: usize,
        sigma: f32,
        scratch: &mut DitScratch,
    ) -> Result<(), String> {
        if latent_side == 0 || context_tokens == 0 {
            return Err("Z-Image latent side and context token count must be positive".into());
        }
        if !sigma.is_finite() || !(0.0..=1.0).contains(&sigma) {
            return Err("Z-Image sigma must be finite and within [0, 1]".into());
        }
        let latent_values = checked_product(
            LATENT_CHANNELS,
            checked_product(latent_side, latent_side, "latent spatial")?,
            "latent",
        )?;
        if latent.len() != latent_values {
            return Err(format!(
                "Invalid Z-Image latent length: expected {latent_values}, got {}",
                latent.len()
            ));
        }
        let context_values = checked_product(context_tokens, CAP_WIDTH, "context")?;
        if context.len() != context_values {
            return Err(format!(
                "Invalid Z-Image context length: expected {context_values}, got {}",
                context.len()
            ));
        }
        require_finite(latent, "latent")?;
        require_finite(context, "context")?;

        let (patch_height, patch_width, _) =
            patch_shape(LATENT_CHANNELS, latent_side, latent_side)?;
        let image_tokens = checked_product(patch_height, patch_width, "image tokens")?;
        let padded_text = padded_to_sequence_multiple(context_tokens)?;
        let padded_image = padded_to_sequence_multiple(image_tokens)?;
        scratch.prepare(padded_text, padded_image, latent_values)?;

        timestep_embedding(sigma * 1_000.0, &mut scratch.time_frequency)?;
        linear_into(
            self.source.as_ref(),
            "t_embedder.mlp.0.weight",
            TIME_WIDTH,
            TIME_HIDDEN,
            &scratch.time_frequency,
            &mut scratch.time_hidden,
            &mut scratch.q8,
            self.pool.as_ref(),
        )?;
        for (value, bias) in scratch.time_hidden.iter_mut().zip(&self.time_0_bias) {
            *value = silu(*value + *bias);
        }
        linear_into(
            self.source.as_ref(),
            "t_embedder.mlp.2.weight",
            TIME_HIDDEN,
            TIME_WIDTH,
            &scratch.time_hidden,
            &mut scratch.time,
            &mut scratch.q8,
            self.pool.as_ref(),
        )?;
        for (value, bias) in scratch.time.iter_mut().zip(&self.time_2_bias) {
            *value += *bias;
        }

        for row in 0..context_tokens {
            let normalized = &mut scratch.attention[..CAP_WIDTH];
            rms_norm(
                &context[row * CAP_WIDTH..(row + 1) * CAP_WIDTH],
                &self.cap_norm,
                normalized,
                RMS_EPSILON,
            );
            linear_into(
                self.source.as_ref(),
                "cap_embedder.1.weight",
                CAP_WIDTH,
                HIDDEN,
                normalized,
                &mut scratch.text[row * HIDDEN..(row + 1) * HIDDEN],
                &mut scratch.q8,
                self.pool.as_ref(),
            )?;
            for (value, bias) in scratch.text[row * HIDDEN..(row + 1) * HIDDEN]
                .iter_mut()
                .zip(&self.cap_bias)
            {
                *value += *bias;
            }
        }

        patchify_latent_into(
            latent,
            LATENT_CHANNELS,
            latent_side,
            latent_side,
            &mut scratch.patches,
        )?;
        for row in 0..image_tokens {
            linear_into(
                self.source.as_ref(),
                "x_embedder.weight",
                PATCH_WIDTH,
                HIDDEN,
                &scratch.patches[row * PATCH_WIDTH..(row + 1) * PATCH_WIDTH],
                &mut scratch.image[row * HIDDEN..(row + 1) * HIDDEN],
                &mut scratch.q8,
                self.pool.as_ref(),
            )?;
            for (value, bias) in scratch.image[row * HIDDEN..(row + 1) * HIDDEN]
                .iter_mut()
                .zip(&self.x_bias)
            {
                *value += *bias;
            }
        }

        scratch.text.truncate(context_tokens * HIDDEN);
        pad_rows_to_32(&mut scratch.text, context_tokens, &self.cap_pad_token)?;
        scratch.image.truncate(image_tokens * HIDDEN);
        pad_rows_to_32(&mut scratch.image, image_tokens, &self.x_pad_token)?;
        z_image_rope_into(context_tokens, latent_side, latent_side, &mut scratch.rope)?;

        let text_hidden = padded_text * HIDDEN;
        let image_hidden = padded_image * HIDDEN;
        let text_rope = padded_text * ROPE_HEAD_WIDTH;
        let image_rope = padded_image * ROPE_HEAD_WIDTH;
        for block in &self.context_refiners {
            run_block(
                self.source.as_ref(),
                self.pool.as_ref(),
                block,
                padded_text,
                &mut scratch.text[..text_hidden],
                &scratch.rope[..text_rope],
                None,
                &mut scratch.qkv,
                &mut scratch.attention,
                &mut scratch.ffn,
                &mut scratch.scores,
                &mut scratch.modulation,
                &mut scratch.q8,
            )?;
        }
        for block in &self.noise_refiners {
            run_block(
                self.source.as_ref(),
                self.pool.as_ref(),
                block,
                padded_image,
                &mut scratch.image[..image_hidden],
                &scratch.rope[text_rope..text_rope + image_rope],
                Some(&scratch.time),
                &mut scratch.qkv,
                &mut scratch.attention,
                &mut scratch.ffn,
                &mut scratch.scores,
                &mut scratch.modulation,
                &mut scratch.q8,
            )?;
        }

        scratch.tokens[..text_hidden].copy_from_slice(&scratch.text[..text_hidden]);
        scratch.tokens[text_hidden..text_hidden + image_hidden]
            .copy_from_slice(&scratch.image[..image_hidden]);
        let total_tokens = padded_text + padded_image;
        let total_hidden = text_hidden + image_hidden;
        let total_rope = text_rope + image_rope;
        for block in &self.layers {
            run_block(
                self.source.as_ref(),
                self.pool.as_ref(),
                block,
                total_tokens,
                &mut scratch.tokens[..total_hidden],
                &scratch.rope[..total_rope],
                Some(&scratch.time),
                &mut scratch.qkv,
                &mut scratch.attention,
                &mut scratch.ffn,
                &mut scratch.scores,
                &mut scratch.modulation,
                &mut scratch.q8,
            )?;
        }

        for (output, input) in scratch.time_frequency.iter_mut().zip(&scratch.time) {
            *output = silu(*input);
        }
        linear_into(
            self.source.as_ref(),
            "final_layer.adaLN_modulation.1.weight",
            TIME_WIDTH,
            HIDDEN,
            &scratch.time_frequency,
            &mut scratch.modulation[..HIDDEN],
            &mut scratch.q8,
            self.pool.as_ref(),
        )?;
        for (value, bias) in scratch.modulation[..HIDDEN]
            .iter_mut()
            .zip(&self.final_modulation_bias)
        {
            *value += *bias;
        }

        resize_zeroed(
            &mut scratch.patches,
            checked_product(image_tokens, PATCH_WIDTH, "output patches")?,
            "Z-Image output patches",
        )?;
        for row in 0..image_tokens {
            let normalized = &mut scratch.attention[..HIDDEN];
            layer_norm_no_affine(
                real_image_row(
                    &scratch.tokens[..total_hidden],
                    padded_text,
                    image_tokens,
                    padded_image,
                    row,
                    HIDDEN,
                )?,
                normalized,
                FINAL_NORM_EPSILON,
            )?;
            for (value, scale) in normalized.iter_mut().zip(&scratch.modulation[..HIDDEN]) {
                *value *= 1.0 + *scale;
            }
            linear_into(
                self.source.as_ref(),
                "final_layer.linear.weight",
                HIDDEN,
                PATCH_WIDTH,
                normalized,
                &mut scratch.patches[row * PATCH_WIDTH..(row + 1) * PATCH_WIDTH],
                &mut scratch.q8,
                self.pool.as_ref(),
            )?;
        }

        sign_and_unpatchify_image(
            &mut scratch.patches,
            &self.final_bias,
            LATENT_CHANNELS,
            latent_side,
            latent_side,
            &mut scratch.velocity,
        )?;
        require_finite(&scratch.velocity, "predicted flow")
    }

    pub(crate) fn denoise(
        &self,
        context: &[f32],
        context_tokens: usize,
        options: &ZImageOptions,
    ) -> Result<Vec<f32>, String> {
        if options.resolution == 0 || options.resolution % 8 != 0 {
            return Err("Z-Image resolution must be a positive multiple of 8".into());
        }
        let sigmas = z_image_sigmas(options.steps)?;
        let latent_side = options.resolution / 8;
        let latent_len = checked_product(
            LATENT_CHANNELS,
            checked_product(latent_side, latent_side, "latent spatial")?,
            "latent",
        )?;
        let mut latent = zeroed_f32("Z-Image initial noise", latent_len)?;
        TorchMt19937::new(options.seed as u64).fill_normal(&mut latent);
        let mut scratch = DitScratch::new();
        for pair in sigmas.windows(2) {
            let sigma = pair[0];
            let sigma_next = pair[1];
            self.predict_flow(
                &latent,
                latent_side,
                context,
                context_tokens,
                sigma,
                &mut scratch,
            )?;
            euler_flow_step(&mut latent, &scratch.velocity, sigma, sigma_next)?;
            require_finite(&latent, "Euler latent")?;
        }
        Ok(latent)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_block(
    source: &dyn TensorSource,
    pool: &ComputePool,
    block: &BlockWeights,
    rows: usize,
    tokens: &mut [f32],
    rope: &[f32],
    time: Option<&[f32]>,
    qkv: &mut [f32],
    attention: &mut [f32],
    ffn: &mut [f32],
    scores: &mut [f32],
    modulation: &mut [f32],
    q8: &mut Q8Scratch,
) -> Result<(), String> {
    let hidden_len = checked_product(rows, HIDDEN, "block tokens")?;
    let qkv_len = checked_product(rows, QKV_WIDTH, "block QKV")?;
    let ffn_len = checked_product(rows, FFN_WIDTH, "block FFN")?;
    if tokens.len() != hidden_len
        || rope.len() != checked_product(rows, ROPE_HEAD_WIDTH, "block RoPE")?
        || qkv.len() < qkv_len
        || attention.len() < hidden_len
        || ffn.len() < ffn_len
        || scores.len() < rows
        || modulation.len() < HIDDEN * 4
    {
        return Err("Invalid Z-Image transformer scratch".into());
    }

    let modulations = if let Some(weights) = &block.modulation {
        let time = time.ok_or("Missing Z-Image AdaLN input")?;
        linear_into(
            source,
            &weights.matrix,
            TIME_WIDTH,
            HIDDEN * 4,
            time,
            &mut modulation[..HIDDEN * 4],
            q8,
            pool,
        )?;
        for (value, bias) in modulation[..HIDDEN * 4].iter_mut().zip(&weights.bias) {
            *value += *bias;
        }
        Some(split_adaln_modulation(&modulation[..HIDDEN * 4], HIDDEN)?)
    } else {
        if time.is_some() {
            return Err("Unexpected Z-Image AdaLN input".into());
        }
        None
    };

    for row in 0..rows {
        let token = &tokens[row * HIDDEN..(row + 1) * HIDDEN];
        let normalized = &mut attention[row * HIDDEN..(row + 1) * HIDDEN];
        rms_norm(token, &block.attention_norm1, normalized, RMS_EPSILON);
        scale_modulated_branch(normalized, modulations.map(|values| values.scale_msa))?;
        linear_into(
            source,
            &block.qkv,
            HIDDEN,
            QKV_WIDTH,
            normalized,
            &mut qkv[row * QKV_WIDTH..(row + 1) * QKV_WIDTH],
            q8,
            pool,
        )?;
    }

    for row in 0..rows {
        let rotation = &rope[row * ROPE_HEAD_WIDTH..(row + 1) * ROPE_HEAD_WIDTH];
        let row_qkv = &mut qkv[row * QKV_WIDTH..(row + 1) * QKV_WIDTH];
        for head in 0..HEADS {
            let start = head * ROPE_HEAD_WIDTH;
            let query = &mut row_qkv[start..start + ROPE_HEAD_WIDTH];
            rms_norm_inplace(query, &block.q_norm, QK_RMS_EPSILON);
            rotate_interleaved_inplace(query, rotation)?;

            let key_start = HIDDEN + start;
            let key = &mut row_qkv[key_start..key_start + ROPE_HEAD_WIDTH];
            rms_norm_inplace(key, &block.k_norm, QK_RMS_EPSILON);
            rotate_interleaved_inplace(key, rotation)?;
        }
    }

    attention_into(
        &qkv[..qkv_len],
        rows,
        HEADS,
        ROPE_HEAD_WIDTH,
        scores,
        &mut attention[..hidden_len],
    )?;

    for row in 0..rows {
        linear_into(
            source,
            &block.out,
            HIDDEN,
            HIDDEN,
            &attention[row * HIDDEN..(row + 1) * HIDDEN],
            &mut qkv[row * HIDDEN..(row + 1) * HIDDEN],
            q8,
            pool,
        )?;
        let projected = &mut qkv[row * HIDDEN..(row + 1) * HIDDEN];
        rms_norm_inplace(projected, &block.attention_norm2, RMS_EPSILON);
        add_modulated_residual(
            &mut tokens[row * HIDDEN..(row + 1) * HIDDEN],
            projected,
            modulations.map(|values| values.gate_msa),
        )?;
    }

    for row in 0..rows {
        let token = &tokens[row * HIDDEN..(row + 1) * HIDDEN];
        let normalized = &mut attention[row * HIDDEN..(row + 1) * HIDDEN];
        rms_norm(token, &block.ffn_norm1, normalized, RMS_EPSILON);
        scale_modulated_branch(normalized, modulations.map(|values| values.scale_mlp))?;
        linear_into(
            source,
            &block.w1,
            HIDDEN,
            FFN_WIDTH,
            normalized,
            &mut qkv[row * FFN_WIDTH..(row + 1) * FFN_WIDTH],
            q8,
            pool,
        )?;
        linear_into(
            source,
            &block.w3,
            HIDDEN,
            FFN_WIDTH,
            normalized,
            &mut ffn[row * FFN_WIDTH..(row + 1) * FFN_WIDTH],
            q8,
            pool,
        )?;
        for index in row * FFN_WIDTH..(row + 1) * FFN_WIDTH {
            qkv[index] = silu(qkv[index]) * ffn[index];
        }
        linear_into(
            source,
            &block.w2,
            FFN_WIDTH,
            HIDDEN,
            &qkv[row * FFN_WIDTH..(row + 1) * FFN_WIDTH],
            normalized,
            q8,
            pool,
        )?;
        rms_norm_inplace(normalized, &block.ffn_norm2, RMS_EPSILON);
        add_modulated_residual(
            &mut tokens[row * HIDDEN..(row + 1) * HIDDEN],
            normalized,
            modulations.map(|values| values.gate_mlp),
        )?;
    }
    Ok(())
}

pub(crate) struct TorchMt19937 {
    state: [u32; MT_N],
    left: usize,
    next: usize,
    has_next_gauss: bool,
    next_gauss: f64,
}

impl TorchMt19937 {
    pub(crate) fn new(seed: u64) -> Self {
        let mut state = [0; MT_N];
        state[0] = seed as u32;
        for index in 1..MT_N {
            let previous = state[index - 1];
            state[index] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(index as u32);
        }
        Self {
            state,
            left: 1,
            next: 0,
            has_next_gauss: false,
            next_gauss: 0.0,
        }
    }

    fn twist(u: u32, v: u32) -> u32 {
        (((u & MT_UPPER_MASK) | (v & MT_LOWER_MASK)) >> 1)
            ^ if v & 1 == 0 { 0 } else { MT_MATRIX_A }
    }

    fn next_state(&mut self) {
        self.left = MT_N;
        self.next = 0;
        for index in 0..MT_N - MT_M {
            self.state[index] =
                self.state[index + MT_M] ^ Self::twist(self.state[index], self.state[index + 1]);
        }
        for index in MT_N - MT_M..MT_N - 1 {
            self.state[index] = self.state[index + MT_M - MT_N]
                ^ Self::twist(self.state[index], self.state[index + 1]);
        }
        self.state[MT_N - 1] =
            self.state[MT_M - 1] ^ Self::twist(self.state[MT_N - 1], self.state[0]);
    }

    fn rand_u32(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.next_state();
        }
        let mut value = self.state[self.next];
        self.next += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    fn rand_u64(&mut self) -> u64 {
        (u64::from(self.rand_u32()) << 32) | u64::from(self.rand_u32())
    }

    fn uniform_f32(value: u32) -> f32 {
        (value & 0x00ff_ffff) as f32 * (1.0f32 / (1u32 << 24) as f32)
    }

    fn uniform_f64(value: u64) -> f64 {
        (value & ((1u64 << 53) - 1)) as f64 * (1.0f64 / (1u64 << 53) as f64)
    }

    fn normal_double(&mut self) -> f64 {
        if self.has_next_gauss {
            self.has_next_gauss = false;
            return self.next_gauss;
        }
        let u1 = Self::uniform_f64(self.rand_u64());
        let u2 = Self::uniform_f64(self.rand_u64());
        let radius = (-2.0 * (-u2).ln_1p()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u1;
        self.next_gauss = radius * theta.sin();
        self.has_next_gauss = true;
        radius * theta.cos()
    }

    fn normal_fill_16(values: &mut [f32]) {
        debug_assert_eq!(values.len(), 16);
        for index in 0..8 {
            let u1 = 1.0 - values[index];
            let u2 = values[index + 8];
            let radius = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            values[index] = radius * theta.cos();
            values[index + 8] = radius * theta.sin();
        }
    }

    pub(crate) fn fill_normal(&mut self, output: &mut [f32]) {
        if output.len() < 16 {
            for value in output {
                *value = self.normal_double() as f32;
            }
            return;
        }

        for value in output.iter_mut() {
            *value = Self::uniform_f32(self.rand_u32());
        }
        for start in (0..output.len() - 15).step_by(16) {
            Self::normal_fill_16(&mut output[start..start + 16]);
        }
        if output.len() % 16 != 0 {
            let tail = output.len() - 16;
            for value in &mut output[tail..] {
                *value = Self::uniform_f32(self.rand_u32());
            }
            Self::normal_fill_16(&mut output[tail..]);
        }
    }
}

pub(crate) fn time_snr_shift(alpha: f32, t: f32) -> f32 {
    if alpha == 1.0 {
        t
    } else {
        alpha * t / (1.0 + (alpha - 1.0) * t)
    }
}

pub(crate) fn z_image_sigmas(steps: usize) -> Result<Vec<f32>, String> {
    if steps == 0 {
        return Err("Z-Image steps must be positive".into());
    }
    if steps == 1 {
        return Ok(vec![time_snr_shift(3.0, 1.0), 0.0]);
    }
    let stride = 999.0 / (steps - 1) as f32;
    let mut result = (0..steps)
        .map(|index| time_snr_shift(3.0, (1000.0 - stride * index as f32) / 1000.0))
        .collect::<Vec<_>>();
    result.push(0.0);
    Ok(result)
}

fn zeroed_f32(name: &str, len: usize) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|error| format!("Failed to allocate {name}: {error}"))?;
    values.resize(len, 0.0);
    Ok(values)
}

fn patch_shape(
    channels: usize,
    height: usize,
    width: usize,
) -> Result<(usize, usize, usize), String> {
    if channels == 0 || height == 0 || width == 0 {
        return Err("Z-Image latent dimensions must be positive".into());
    }
    let patch_height = height
        .checked_add(PATCH_SIZE - 1)
        .ok_or("Z-Image latent shape overflow")?
        / PATCH_SIZE;
    let patch_width = width
        .checked_add(PATCH_SIZE - 1)
        .ok_or("Z-Image latent shape overflow")?
        / PATCH_SIZE;
    let patch_width_channels = channels
        .checked_mul(PATCH_SIZE * PATCH_SIZE)
        .ok_or("Z-Image patch width overflow")?;
    Ok((patch_height, patch_width, patch_width_channels))
}

pub(crate) fn patchify_latent(
    latent: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    let mut output = Vec::new();
    patchify_latent_into(latent, channels, height, width, &mut output)?;
    Ok(output)
}

fn patchify_latent_into(
    latent: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    output: &mut Vec<f32>,
) -> Result<(), String> {
    let expected = channels
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or("Z-Image latent shape overflow")?;
    if latent.len() != expected {
        return Err(format!(
            "Invalid Z-Image latent length: expected {expected}, got {}",
            latent.len()
        ));
    }
    let (patch_height, patch_width, patch_width_channels) = patch_shape(channels, height, width)?;
    let output_len = patch_height
        .checked_mul(patch_width)
        .and_then(|value| value.checked_mul(patch_width_channels))
        .ok_or("Z-Image patch shape overflow")?;
    resize_zeroed(output, output_len, "Z-Image patches")?;
    output.fill(0.0);

    for patch_y in 0..patch_height {
        for patch_x in 0..patch_width {
            let token = patch_y * patch_width + patch_x;
            for inner_y in 0..PATCH_SIZE {
                let y = patch_y * PATCH_SIZE + inner_y;
                if y >= height {
                    continue;
                }
                for inner_x in 0..PATCH_SIZE {
                    let x = patch_x * PATCH_SIZE + inner_x;
                    if x >= width {
                        continue;
                    }
                    let patch_offset = (inner_y * PATCH_SIZE + inner_x) * channels;
                    for channel in 0..channels {
                        let source = (channel * height + y) * width + x;
                        output[token * patch_width_channels + patch_offset + channel] =
                            latent[source];
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn unpatchify_latent(
    patches: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    let (patch_height, patch_width, patch_width_channels) = patch_shape(channels, height, width)?;
    let expected = patch_height
        .checked_mul(patch_width)
        .and_then(|value| value.checked_mul(patch_width_channels))
        .ok_or("Z-Image patch shape overflow")?;
    if patches.len() != expected {
        return Err(format!(
            "Invalid Z-Image patch length: expected {expected}, got {}",
            patches.len()
        ));
    }
    let output_len = channels
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or("Z-Image latent shape overflow")?;
    let mut output = zeroed_f32("Z-Image latent", output_len)?;
    unpatchify_latent_into(patches, channels, height, width, &mut output)?;
    Ok(output)
}

fn unpatchify_latent_into(
    patches: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    output: &mut [f32],
) -> Result<(), String> {
    let (patch_height, patch_width, patch_width_channels) = patch_shape(channels, height, width)?;
    let expected = patch_height
        .checked_mul(patch_width)
        .and_then(|value| value.checked_mul(patch_width_channels))
        .ok_or("Z-Image patch shape overflow")?;
    if patches.len() != expected {
        return Err(format!(
            "Invalid Z-Image patch length: expected {expected}, got {}",
            patches.len()
        ));
    }
    let output_len = channels
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or("Z-Image latent shape overflow")?;
    if output.len() != output_len {
        return Err(format!(
            "Invalid Z-Image latent output length: expected {output_len}, got {}",
            output.len()
        ));
    }

    for channel in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let token = (y / PATCH_SIZE) * patch_width + x / PATCH_SIZE;
                let patch_offset = ((y % PATCH_SIZE) * PATCH_SIZE + x % PATCH_SIZE) * channels;
                output[(channel * height + y) * width + x] =
                    patches[token * patch_width_channels + patch_offset + channel];
            }
        }
    }
    Ok(())
}

fn padded_to_sequence_multiple(value: usize) -> Result<usize, String> {
    value
        .checked_add(SEQUENCE_MULTIPLE - 1)
        .map(|value| value / SEQUENCE_MULTIPLE * SEQUENCE_MULTIPLE)
        .ok_or_else(|| "Z-Image sequence length overflow".into())
}

pub(crate) fn z_image_rope(
    text_tokens: usize,
    image_width: usize,
    image_height: usize,
) -> Result<Vec<f32>, String> {
    let mut output = Vec::new();
    z_image_rope_into(text_tokens, image_width, image_height, &mut output)?;
    Ok(output)
}

fn z_image_rope_into(
    text_tokens: usize,
    image_width: usize,
    image_height: usize,
    output: &mut Vec<f32>,
) -> Result<(), String> {
    if text_tokens == 0 || image_width == 0 || image_height == 0 {
        return Err("Z-Image RoPE dimensions must be positive".into());
    }
    let axes_sum = ROPE_AXES.iter().try_fold(0usize, |sum, axis| {
        if axis % 2 != 0 {
            return Err("Z-Image RoPE axes must be even".to_string());
        }
        sum.checked_add(*axis)
            .ok_or_else(|| "Z-Image RoPE head width overflow".into())
    })?;
    if axes_sum != ROPE_HEAD_WIDTH {
        return Err("Z-Image RoPE axes must match the attention head width".into());
    }

    let patch_width = image_width
        .checked_add(PATCH_SIZE - 1)
        .ok_or("Z-Image image shape overflow")?
        / PATCH_SIZE;
    let patch_height = image_height
        .checked_add(PATCH_SIZE - 1)
        .ok_or("Z-Image image shape overflow")?
        / PATCH_SIZE;
    let image_tokens = patch_width
        .checked_mul(patch_height)
        .ok_or("Z-Image image token count overflow")?;
    let padded_text = padded_to_sequence_multiple(text_tokens)?;
    let padded_image = padded_to_sequence_multiple(image_tokens)?;
    let position_count = padded_text
        .checked_add(padded_image)
        .ok_or("Z-Image position count overflow")?;
    let output_len = position_count
        .checked_mul(ROPE_HEAD_WIDTH)
        .ok_or("Z-Image RoPE output size overflow")?;
    resize_zeroed(output, output_len, "Z-Image RoPE")?;

    for position_index in 0..position_count {
        let positions = if position_index < padded_text {
            [(position_index + 1) as f32, 0.0, 0.0]
        } else {
            let image_index = position_index - padded_text;
            if image_index < image_tokens {
                [
                    (padded_text + 1) as f32,
                    (image_index / patch_width) as f32,
                    (image_index % patch_width) as f32,
                ]
            } else {
                [0.0, 0.0, 0.0]
            }
        };

        let mut output_index = position_index * ROPE_HEAD_WIDTH;
        for (axis, dimension) in ROPE_AXES.iter().copied().enumerate() {
            let half = dimension / 2;
            let end = (dimension as f32 - 2.0) / dimension as f32;
            let step = end / (half - 1) as f32;
            for frequency in 0..half {
                let scale = frequency as f32 * step;
                let omega = 1.0 / ROPE_THETA.powf(scale);
                let angle = positions[axis] * omega;
                output[output_index] = angle.cos();
                output[output_index + 1] = angle.sin();
                output_index += 2;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        add_modulated_residual, attention_into, euler_flow_step, layer_norm_no_affine,
        pad_rows_to_32, patchify_latent, patchify_latent_into, real_image_row, require_finite,
        rotate_interleaved_inplace, scale_modulated_branch, sign_and_unpatchify_image,
        split_adaln_modulation, time_snr_shift, timestep_embedding, unpatchify_latent,
        z_image_rope, z_image_sigmas, TorchMt19937, ZImageDit,
    };
    use crate::core::thread_pool::ComputePool;
    use std::sync::Arc;

    fn expected_seed_42_20_bits() -> Vec<u32> {
        vec![
            0x3ff6_a52a,
            0x3fbe_5f53,
            0x3f66_9567,
            0xc006_c0db,
            0xbf42_14e2,
            0x3f8a_0650,
            0x3f4d_0143,
            0x3fd7_1e93,
            0x3eb6_3345,
            0xbf2f_c686,
            0xbefc_9934,
            0x3e77_4894,
            0xbe6d_2eed,
            0x3d2b_0c00,
            0xbe80_ce7a,
            0x3f5c_1fb0,
            0xbe9e_9482,
            0xbeca_9a91,
            0x3f4d_ac3c,
            0xbf1f_20e0,
        ]
    }

    #[test]
    fn torch_mt19937_recomputes_the_final_sixteen_values() {
        let mut values = vec![0.0; 20];
        TorchMt19937::new(42).fill_normal(&mut values);
        assert_eq!(
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected_seed_42_20_bits()
        );
    }

    #[test]
    fn torch_mt19937_uses_the_double_fallback_for_short_vectors() {
        let mut values = vec![0.0; 5];
        TorchMt19937::new(42).fill_normal(&mut values);
        assert_eq!(
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [
                0x3eac_62ae,
                0x3e03_e69d,
                0x3e70_16e7,
                0x3e6b_dc6c,
                0xbf8f_b9c2
            ]
        );
    }

    #[test]
    fn discrete_flow_schedule_has_eight_steps_and_a_zero_tail() {
        let sigmas = z_image_sigmas(8).unwrap();
        assert_eq!(sigmas.len(), 9);
        assert_eq!(sigmas[0].to_bits(), time_snr_shift(3.0, 1.0).to_bits());
        assert_eq!(sigmas[8].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn discrete_flow_schedule_rejects_zero_steps() {
        assert_eq!(
            z_image_sigmas(0).unwrap_err(),
            "Z-Image steps must be positive"
        );
    }

    #[test]
    fn z_image_rope_axes_sum_to_the_128_wide_head() {
        assert_eq!(z_image_rope(32, 64, 64).unwrap().len(), (32 + 1024) * 128);
    }

    #[test]
    fn z_image_rope_pads_text_and_image_positions_independently() {
        let rope = z_image_rope(1, 2, 2).unwrap();
        assert_eq!(rope.len(), 64 * 128);
        assert!((rope[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((rope[1] - 1.0f32.sin()).abs() < 1e-6);

        let image = 32 * 128;
        assert!((rope[image] - 33.0f32.cos()).abs() < 1e-6);
        assert!((rope[image + 1] - 33.0f32.sin()).abs() < 1e-6);

        let image_padding = 33 * 128;
        assert_eq!(&rope[image_padding..image_padding + 2], &[1.0, 0.0]);
    }

    #[test]
    fn z_image_rope_rejects_invalid_shapes() {
        assert!(z_image_rope(0, 64, 64).is_err());
        assert!(z_image_rope(1, 0, 64).is_err());
        assert!(z_image_rope(1, usize::MAX, 64).is_err());
    }

    #[test]
    fn patch_layout_keeps_channels_inside_each_spatial_position() {
        let latent = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        assert_eq!(
            patchify_latent(&latent, 2, 2, 2).unwrap(),
            [0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0]
        );
    }

    #[test]
    fn latent_patch_round_trip_preserves_spatial_and_channel_order() {
        let latent = (0..16 * 64 * 64)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let patches = patchify_latent(&latent, 16, 64, 64).unwrap();
        assert_eq!(patches.len(), 1024 * 64);
        assert_eq!(unpatchify_latent(&patches, 16, 64, 64).unwrap(), latent);
    }

    #[test]
    fn patching_rejects_bad_lengths_and_overflowing_shapes() {
        assert!(patchify_latent(&[], 16, 64, 64).is_err());
        assert!(patchify_latent(&[], usize::MAX, 2, 2).is_err());
        assert!(unpatchify_latent(&[], 16, 64, 64).is_err());
    }

    #[test]
    fn reused_patch_buffer_zeros_out_of_bounds_spatial_slots() {
        let mut patches = Vec::new();
        patchify_latent_into(&[1.0, 2.0, 3.0, 4.0], 1, 2, 2, &mut patches).unwrap();
        patchify_latent_into(&[9.0], 1, 1, 1, &mut patches).unwrap();
        assert_eq!(patches, [9.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn padding_uses_the_learned_token_not_zero_context() {
        let mut values = vec![1.0; 31 * 3840];
        pad_rows_to_32(&mut values, 31, &[2.0; 3840]).unwrap();
        assert_eq!(&values[31 * 3840..32 * 3840], &[2.0; 3840]);
    }

    #[test]
    fn euler_flow_update_consumes_the_next_sigma_once() {
        let mut latent = [2.0, -3.0];
        euler_flow_step(&mut latent, &[0.5, -0.25], 0.9, 0.4).unwrap();
        assert_eq!(latent, [1.75, -2.875]);
    }

    #[test]
    fn euler_flow_update_rejects_mismatched_lengths_before_mutation() {
        let mut latent = [2.0, -3.0];
        let before = latent;
        assert_eq!(
            euler_flow_step(&mut latent, &[0.5], 0.9, 0.4).unwrap_err(),
            "Invalid Z-Image Euler buffer lengths: latent 2, velocity 1"
        );
        assert_eq!(latent, before);
    }

    #[test]
    fn final_image_rows_use_the_padded_text_offset_and_skip_image_padding() {
        let tokens = [
            1.0, 2.0, // padded text row 0
            3.0, 4.0, // padded text row 1
            10.0, 11.0, // real image row 0
            20.0, 21.0, // real image row 1
            90.0, 91.0, // padded image row
        ];
        assert_eq!(
            real_image_row(&tokens, 2, 2, 3, 0, 2).unwrap(),
            &[10.0, 11.0]
        );
        assert_eq!(
            real_image_row(&tokens, 2, 2, 3, 1, 2).unwrap(),
            &[20.0, 21.0]
        );
        assert!(real_image_row(&tokens, 2, 2, 3, 2, 2).is_err());
    }

    #[test]
    fn final_sign_bias_and_unpatchify_use_channel_major_layout() {
        let mut patches = [0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0];
        let bias = [100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0];
        let mut velocity = [0.0; 8];
        sign_and_unpatchify_image(&mut patches, &bias, 2, 2, 2, &mut velocity).unwrap();
        assert_eq!(
            velocity,
            [-100.0, -101.0, -102.0, -103.0, -210.0, -211.0, -212.0, -213.0]
        );
    }

    #[test]
    fn final_image_helpers_reject_invalid_lengths() {
        assert!(real_image_row(&[0.0; 9], 2, 2, 3, 0, 2).is_err());
        assert!(real_image_row(&[0.0; 10], 2, 4, 3, 0, 2).is_err());

        let mut short_patches = [0.0; 7];
        let mut velocity = [0.0; 8];
        assert!(
            sign_and_unpatchify_image(&mut short_patches, &[0.0; 8], 2, 2, 2, &mut velocity)
                .is_err()
        );

        let mut patches = [0.0; 8];
        assert!(
            sign_and_unpatchify_image(&mut patches, &[0.0; 7], 2, 2, 2, &mut velocity).is_err()
        );
        assert!(
            sign_and_unpatchify_image(&mut patches, &[0.0; 8], 2, 2, 2, &mut [0.0; 7]).is_err()
        );
    }

    #[test]
    fn adaln_chunk_order_applies_attention_before_feed_forward() {
        let gate_msa = 0.5f32.atanh();
        let gate_mlp = 0.25f32.atanh();
        let values = [1.0, 2.0, gate_msa, gate_msa, 3.0, 4.0, gate_mlp, gate_mlp];
        let modulation = split_adaln_modulation(&values, 2).unwrap();
        assert_eq!(modulation.scale_msa, &[1.0, 2.0]);
        assert_eq!(modulation.gate_msa, &[gate_msa, gate_msa]);
        assert_eq!(modulation.scale_mlp, &[3.0, 4.0]);
        assert_eq!(modulation.gate_mlp, &[gate_mlp, gate_mlp]);

        let mut attention_branch = [10.0, 20.0];
        scale_modulated_branch(&mut attention_branch, Some(modulation.scale_msa)).unwrap();
        assert_eq!(attention_branch, [20.0, 60.0]);

        let mut tokens = [100.0, 200.0];
        add_modulated_residual(&mut tokens, &[8.0, 4.0], Some(modulation.gate_msa)).unwrap();
        assert_eq!(tokens, [104.0, 202.0]);

        let mut feed_forward_branch = tokens;
        scale_modulated_branch(&mut feed_forward_branch, Some(modulation.scale_mlp)).unwrap();
        assert_eq!(feed_forward_branch, [416.0, 1010.0]);

        add_modulated_residual(&mut tokens, &[4.0, 8.0], Some(modulation.gate_mlp)).unwrap();
        assert_eq!(tokens, [105.0, 204.0]);
    }

    #[test]
    fn adaln_helpers_reject_invalid_lengths() {
        assert!(split_adaln_modulation(&[0.0; 7], 2).is_err());
        assert!(scale_modulated_branch(&mut [1.0, 2.0], Some(&[1.0])).is_err());
        assert!(add_modulated_residual(&mut [1.0, 2.0], &[1.0], Some(&[0.0, 0.0])).is_err());
    }

    #[test]
    fn timestep_embedding_uses_pinned_cos_then_sin_layout() {
        let mut embedding = [0.0; 4];
        timestep_embedding(1.0, &mut embedding).unwrap();
        assert!((embedding[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((embedding[1] - 0.01f32.cos()).abs() < 1e-6);
        assert!((embedding[2] - 1.0f32.sin()).abs() < 1e-6);
        assert!((embedding[3] - 0.01f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn compact_rope_rotates_interleaved_values_once() {
        let mut values = [1.0, 2.0, 3.0, 4.0];
        rotate_interleaved_inplace(&mut values, &[0.0, 1.0, 1.0, 0.0]).unwrap();
        assert_eq!(values, [-2.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn attention_reads_qkv_rows_and_softmaxes_each_query() {
        let qkv = [
            0.0, 0.0, 0.0, 0.0, 2.0, 4.0, // token 0: q, k, v
            0.0, 0.0, 0.0, 0.0, 6.0, 8.0, // token 1: q, k, v
        ];
        let mut scores = [0.0; 2];
        let mut output = [0.0; 4];
        attention_into(&qkv, 2, 1, 2, &mut scores, &mut output).unwrap();
        assert_eq!(output, [4.0, 6.0, 4.0, 6.0]);
    }

    #[test]
    fn final_layer_norm_uses_population_variance_without_affine() {
        let mut output = [0.0; 2];
        layer_norm_no_affine(&[1.0, 3.0], &mut output, 0.0).unwrap();
        assert_eq!(output, [-1.0, 1.0]);
    }

    #[test]
    fn dit_boundaries_reject_non_finite_values() {
        assert!(require_finite(&[0.0, -1.0], "context").is_ok());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                require_finite(&[0.0, value], "context").unwrap_err(),
                "Non-finite Z-Image context"
            );
        }
    }

    #[test]
    #[ignore = "requires Z_IMAGE_DIT"]
    fn dit_loader_accepts_supplied_tensor_signature() {
        let source = Arc::new(
            crate::core::loader::GGUFLoader::from_file(
                std::env::var("Z_IMAGE_DIT").expect("missing Z_IMAGE_DIT"),
            )
            .unwrap(),
        );
        ZImageDit::load(source, Arc::new(ComputePool::new(1))).unwrap();
    }
}
