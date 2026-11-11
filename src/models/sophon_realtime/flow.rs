//! CosyVoice3 token/voice conditioning and four-step flow matching.

use crate::core::tensor::{load_f32_tensor, require_tensor_type, GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::Weight;
use crate::ops::nn::{self, AttentionSpec, Linear};

const MEL: usize = 80;
const DIM: usize = 1024;
const VOCAB: usize = 6561;
const LOOKAHEAD: usize = 3;
const DEPTH: usize = 22;
const MAX_FRAMES: usize = 15_000;
const PREFIX: &str = "decoder.estimator";

/// Channels-first mel and noise, with a 192-dimensional voice embedding.
/// The caller supplies the noise so identical Oracle and runtime inputs are possible.
pub struct FlowRequest<'a> {
    pub tokens: &'a [u32],
    pub prompt_tokens: &'a [u32],
    pub prompt_mel: &'a [f32],
    pub speaker: &'a [f32],
    pub initial_noise: &'a [f32],
    pub streaming: bool,
    pub finalize: bool,
    pub steps: usize,
}

impl<'a> FlowRequest<'a> {
    pub fn new(tokens: &'a [u32], speaker: &'a [f32], initial_noise: &'a [f32]) -> Self {
        Self {
            tokens,
            prompt_tokens: &[],
            prompt_mel: &[],
            speaker,
            initial_noise,
            streaming: true,
            finalize: true,
            steps: 4,
        }
    }
}

pub struct FlowOutput {
    /// Channels-first [80, frames], excluding the prompt mel prefix.
    pub mel: Vec<f32>,
    pub frames: usize,
}

/// Prepared channels-first inputs, also exposed for estimator checkpoint comparisons.
pub struct FlowConditioning {
    pub mu: Vec<f32>,
    pub cond: Vec<f32>,
    pub speaker: Vec<f32>,
    pub frames: usize,
    pub prompt_frames: usize,
}

pub struct Flow<'a> {
    embedding: Linear<'a>,
    speaker: Linear<'a>,
    lookahead1: Conv,
    lookahead2: Conv,
    time1: Linear<'a>,
    time2: Linear<'a>,
    input: Linear<'a>,
    pos1: Conv,
    pos2: Conv,
    inv_freq: Vec<f32>,
    blocks: Vec<Block<'a>>,
    norm_out: Linear<'a>,
    proj_out: Linear<'a>,
}

impl<'a> Flow<'a> {
    pub fn load(source: &'a dyn TensorSource) -> Result<Self, String> {
        super::require_string(source, "general.architecture", "sophon_realtime_flow")?;
        super::require_string(source, "sophon_realtime.component", "flow")?;
        super::require_string(
            source,
            "sophon_realtime.source_revision",
            super::SOURCE_REVISION,
        )?;
        for (key, value) in [
            ("input_size", 80),
            ("output_size", 80),
            ("spk_embed_dim", 192),
            ("vocab_size", 6561),
            ("sample_rate", 24000),
            ("input_frame_rate", 25),
            ("token_mel_ratio", 2),
            ("pre_lookahead_len", 3),
            ("pre_lookahead_channels", 1024),
            ("dim", 1024),
            ("depth", 22),
            ("heads", 16),
            ("dim_head", 64),
            ("ff_mult", 2),
            ("time_embed_dim", 256),
            ("conv_pos_groups", 16),
            ("conv_pos_kernel", 31),
            ("static_chunk_size", 50),
            ("tag_default_steps", 4),
        ] {
            super::require_u64(source, &format!("sophon_realtime.flow.{key}"), value)?;
        }
        for (key, value) in [
            ("ff_activation", "gelu_tanh"),
            ("rotary_layout", "adjacent_pairs"),
            ("solver", "euler"),
            ("t_scheduler", "cosine"),
        ] {
            super::require_string(source, &format!("sophon_realtime.flow.{key}"), value)?;
        }
        for (key, expected) in [
            ("norm_eps", 1e-6f32),
            ("sigma_min", 1e-6),
            ("yaml_inference_cfg_rate", 0.7),
            ("tag_default_cfg", 0.0),
        ] {
            if source
                .metadata(&format!("sophon_realtime.flow.{key}"))
                .and_then(crate::core::tensor::MetaValue::to_f64)
                .map(|v| v as f32)
                != Some(expected)
            {
                return Err(format!("unsupported Sophon flow {key}"));
            }
        }
        if source
            .metadata("sophon_realtime.flow.num_decoding_left_chunks")
            .and_then(crate::core::tensor::MetaValue::to_f64)
            != Some(-1.0)
        {
            return Err("Sophon flow requires all left chunks".into());
        }
        Ok(Self {
            embedding: load_linear(source, "input_embedding", MEL, VOCAB, false)?,
            speaker: load_linear(source, "spk_embed_affine_layer", 192, MEL, true)?,
            lookahead1: Conv::load(source, "pre_lookahead_layer.conv1", MEL, DIM, 4, 1)?,
            lookahead2: Conv::load(source, "pre_lookahead_layer.conv2", DIM, MEL, 3, 1)?,
            time1: load_linear(
                source,
                &format!("{PREFIX}.time_embed.time_mlp.0"),
                256,
                DIM,
                true,
            )?,
            time2: load_linear(
                source,
                &format!("{PREFIX}.time_embed.time_mlp.2"),
                DIM,
                DIM,
                true,
            )?,
            input: load_linear(
                source,
                &format!("{PREFIX}.input_embed.proj"),
                MEL * 4,
                DIM,
                true,
            )?,
            pos1: Conv::load(
                source,
                &format!("{PREFIX}.input_embed.conv_pos_embed.conv1.0"),
                DIM,
                DIM,
                31,
                16,
            )?,
            pos2: Conv::load(
                source,
                &format!("{PREFIX}.input_embed.conv_pos_embed.conv2.0"),
                DIM,
                DIM,
                31,
                16,
            )?,
            inv_freq: load_f32(source, &format!("{PREFIX}.rotary_embed.inv_freq"), &[32])?,
            blocks: (0..DEPTH)
                .map(|i| Block::load(source, i))
                .collect::<Result<_, _>>()?,
            norm_out: load_linear(
                source,
                &format!("{PREFIX}.norm_out.linear"),
                DIM,
                DIM * 2,
                true,
            )?,
            proj_out: load_linear(source, &format!("{PREFIX}.proj_out"), DIM, MEL, true)?,
        })
    }

    pub fn condition(
        &self,
        request: &FlowRequest<'_>,
        pool: &ComputePool,
    ) -> Result<FlowConditioning, String> {
        let token_count = request
            .prompt_tokens
            .len()
            .checked_add(request.tokens.len())
            .ok_or("flow token length overflow")?;
        let valid_tokens = token_count
            .checked_sub(if request.finalize { 0 } else { LOOKAHEAD })
            .filter(|&n| n > 0)
            .ok_or("flow needs tokens beyond its lookahead")?;
        let frames = valid_tokens
            .checked_mul(2)
            .filter(|&n| n <= MAX_FRAMES)
            .ok_or("flow exceeds the pinned 15000-frame noise window")?;
        if request.speaker.len() != 192 || request.speaker.iter().any(|v| !v.is_finite()) {
            return Err("flow requires a finite 192-dimensional speaker embedding".into());
        }
        let (cond, prompt_frames) = prompt_condition(request.prompt_mel, frames)?;
        let norm = crate::ops::dot_f32(request.speaker, request.speaker, 192)
            .sqrt()
            .max(1e-12);
        let speaker: Vec<f32> = request.speaker.iter().map(|v| v / norm).collect();
        let speaker = self.speaker.forward(pool, &speaker, 1)?;
        let mut embedded = vec![0.0; token_count * MEL];
        for (row, &token) in embedded
            .chunks_exact_mut(MEL)
            .zip(request.prompt_tokens.iter().chain(request.tokens))
        {
            self.embedding.embedding_lookup(token, row)?;
        }
        let embedded = transpose(&embedded, token_count, MEL);
        // With a non-final chunk, the last three tokens supply context but yield no output.
        let (mut hidden, n) = self.lookahead1.forward(
            &embedded,
            token_count,
            0,
            if request.finalize { LOOKAHEAD } else { 0 },
            1,
            1,
            pool,
        )?;
        for v in &mut hidden {
            if *v < 0.0 {
                *v *= 0.01;
            }
        }
        let (mut hidden, n2) = self.lookahead2.forward(&hidden, n, 2, 0, 1, 1, pool)?;
        if n != valid_tokens || n2 != valid_tokens {
            return Err("invalid flow lookahead shape".into());
        }
        for c in 0..MEL {
            for t in 0..valid_tokens {
                hidden[c * valid_tokens + t] += embedded[c * token_count + t];
            }
        }
        let mut mu = vec![0.0; MEL * frames];
        for c in 0..MEL {
            for t in 0..valid_tokens {
                mu[c * frames + 2 * t] = hidden[c * valid_tokens + t];
                mu[c * frames + 2 * t + 1] = hidden[c * valid_tokens + t];
            }
        }
        Ok(FlowConditioning {
            mu,
            cond,
            speaker,
            frames,
            prompt_frames,
        })
    }

    /// Euler integration with CFG=0: exactly one conditioned estimator call per step.
    pub fn infer(
        &self,
        request: &FlowRequest<'_>,
        pool: &ComputePool,
    ) -> Result<FlowOutput, String> {
        self.infer_until(request, pool, &|| false)
    }

    pub(crate) fn infer_until(
        &self,
        request: &FlowRequest<'_>,
        pool: &ComputePool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<FlowOutput, String> {
        super::check_cancelled(cancelled)?;
        if request.steps == 0 {
            return Err("Euler requires at least one step".into());
        }
        let conditioning = self.condition(request, pool)?;
        validate_mel(request.initial_noise, conditioning.frames, "initial noise")?;
        trace("mu", &[MEL, conditioning.frames], &conditioning.mu);
        trace("cond", &[MEL, conditioning.frames], &conditioning.cond);
        trace("speaker", &[MEL], &conditioning.speaker);
        let mut step = 0;
        let mel = cosine_euler(request.initial_noise, request.steps, |x, t| {
            let velocity =
                self.velocity_until(x, t, &conditioning, request.streaming, pool, cancelled)?;
            trace(
                &format!("velocity.{step}"),
                &[MEL, conditioning.frames],
                &velocity,
            );
            step += 1;
            Ok(velocity)
        })?;
        let output = FlowOutput {
            mel: crop_prompt(&mel, conditioning.frames, conditioning.prompt_frames),
            frames: conditioning.frames - conditioning.prompt_frames,
        };
        trace("mel", &[MEL, output.frames], &output.mel);
        Ok(output)
    }

    /// DiT velocity on channels-first [80, frames] tensors, before Euler integration.
    pub fn velocity(
        &self,
        noisy: &[f32],
        time: f32,
        conditioning: &FlowConditioning,
        streaming: bool,
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        self.velocity_until(noisy, time, conditioning, streaming, pool, &|| false)
    }

    #[allow(clippy::too_many_arguments)]
    fn velocity_until(
        &self,
        noisy: &[f32],
        time: f32,
        conditioning: &FlowConditioning,
        streaming: bool,
        pool: &ComputePool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>, String> {
        super::check_cancelled(cancelled)?;
        let frames = conditioning.frames;
        validate_mel(noisy, frames, "noisy mel")?;
        validate_mel(&conditioning.mu, frames, "mu")?;
        validate_mel(&conditioning.cond, frames, "prompt condition")?;
        if !time.is_finite()
            || !(0.0..=1.0).contains(&time)
            || conditioning.speaker.len() != MEL
            || conditioning.speaker.iter().any(|v| !v.is_finite())
        {
            return Err("invalid flow time or projected speaker".into());
        }
        let mut time = self.time1.forward(pool, &timestep_features(time), 1)?;
        nn::silu_inplace(&mut time);
        let mut time = self.time2.forward(pool, &time, 1)?;
        // Every adaptive layer normalizer applies SiLU to the same time embedding.
        nn::silu_inplace(&mut time);
        let mut input = vec![0.0; frames * MEL * 4];
        for t in 0..frames {
            for c in 0..MEL {
                input[t * MEL * 4 + c] = noisy[c * frames + t];
                input[t * MEL * 4 + MEL + c] = conditioning.cond[c * frames + t];
                input[t * MEL * 4 + MEL * 2 + c] = conditioning.mu[c * frames + t];
                input[t * MEL * 4 + MEL * 3 + c] = conditioning.speaker[c];
            }
        }
        let mut x = self.input.forward(pool, &input, frames)?;
        let position = transpose(&x, frames, DIM);
        let (mut position, _) = self.pos1.forward(&position, frames, 30, 0, 1, 1, pool)?;
        mish(&mut position);
        let (mut position, _) = self.pos2.forward(&position, frames, 30, 0, 1, 1, pool)?;
        mish(&mut position);
        let position = transpose(&position, DIM, frames);
        for (x, p) in x.iter_mut().zip(position) {
            *x += p;
        }
        let ones = vec![1.0; DIM];
        for block in &self.blocks {
            super::check_cancelled(cancelled)?;
            x = block.forward(x, &time, frames, &ones, &self.inv_freq, streaming, pool)?;
        }
        let modulation = self.norm_out.forward(pool, &time, 1)?;
        let x = modulated_norm(&x, frames, &ones, &modulation[DIM..], &modulation[..DIM])?;
        let mel = self.proj_out.forward(pool, &x, frames)?;
        let mel = transpose(&mel, frames, MEL);
        validate_mel(&mel, frames, "flow velocity")?;
        Ok(mel)
    }
}

fn trace(name: &str, shape: &[usize], values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(
        &format!("sophon.flow.{name}"),
        None,
        shape,
        values,
    ));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, shape, values);
}

struct Block<'a> {
    modulation: Linear<'a>,
    q: Linear<'a>,
    k: Linear<'a>,
    v: Linear<'a>,
    out: Linear<'a>,
    ff1: Linear<'a>,
    ff2: Linear<'a>,
}

impl<'a> Block<'a> {
    fn load(source: &'a dyn TensorSource, index: usize) -> Result<Self, String> {
        let prefix = format!("{PREFIX}.transformer_blocks.{index}");
        let linear = |name: &str, n_in, n_out| {
            load_linear(source, &format!("{prefix}.{name}"), n_in, n_out, true)
        };
        Ok(Self {
            modulation: linear("attn_norm.linear", DIM, DIM * 6)?,
            q: linear("attn.to_q", DIM, DIM)?,
            k: linear("attn.to_k", DIM, DIM)?,
            v: linear("attn.to_v", DIM, DIM)?,
            out: linear("attn.to_out.0", DIM, DIM)?,
            ff1: linear("ff.ff.0.0", DIM, DIM * 2)?,
            ff2: linear("ff.ff.2", DIM * 2, DIM)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        mut x: Vec<f32>,
        time: &[f32],
        frames: usize,
        ones: &[f32],
        inv_freq: &[f32],
        streaming: bool,
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        let modulation = self.modulation.forward(pool, time, 1)?;
        let m: Vec<&[f32]> = modulation.chunks_exact(DIM).collect();
        let norm = modulated_norm(&x, frames, ones, m[0], m[1])?;
        let mut q = self.q.forward(pool, &norm, frames)?;
        let mut k = self.k.forward(pool, &norm, frames)?;
        let v = self.v.forward(pool, &norm, frames)?;
        // The pinned implementation rotates before the head reshape: only head zero.
        rotate_prefix(&mut q, DIM, inv_freq);
        rotate_prefix(&mut k, DIM, inv_freq);
        let attention = chunk_attention(&q, &k, &v, frames, streaming)?;
        let attention = self.out.forward(pool, &attention, frames)?;
        gated_residual(&mut x, &attention, m[2]);
        let norm = modulated_norm(&x, frames, ones, m[3], m[4])?;
        let mut ff = self.ff1.forward(pool, &norm, frames)?;
        nn::gelu_inplace(&mut ff);
        let ff = self.ff2.forward(pool, &ff, frames)?;
        gated_residual(&mut x, &ff, m[5]);
        Ok(x)
    }
}

fn chunk_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    frames: usize,
    streaming: bool,
) -> Result<Vec<f32>, String> {
    let mut output = Vec::with_capacity(q.len());
    let chunk = if streaming { 50 } else { frames };
    for start in (0..frames).step_by(chunk) {
        let end = (start + chunk).min(frames);
        // All left chunks and the complete current chunk are visible, without a token-causal mask.
        output.extend(nn::attention_scalar(
            &q[start * DIM..end * DIM],
            &k[..end * DIM],
            &v[..end * DIM],
            AttentionSpec {
                query_tokens: end - start,
                key_tokens: end,
                query_heads: 16,
                key_value_heads: 16,
                head_dim: 64,
                causal: false,
                scale: 0.125,
            },
        )?);
    }
    Ok(output)
}

fn modulated_norm(
    input: &[f32],
    frames: usize,
    ones: &[f32],
    shift: &[f32],
    scale: &[f32],
) -> Result<Vec<f32>, String> {
    let mut norm = nn::layer_norm_rows(input, frames, Some(ones), None, 1e-6)?;
    for row in norm.chunks_exact_mut(DIM) {
        for c in 0..DIM {
            row[c] = row[c] * (1.0 + scale[c]) + shift[c];
        }
    }
    Ok(norm)
}

fn gated_residual(x: &mut [f32], update: &[f32], gate: &[f32]) {
    for (row, update) in x.chunks_exact_mut(DIM).zip(update.chunks_exact(DIM)) {
        for c in 0..DIM {
            row[c] += gate[c] * update[c];
        }
    }
}

fn mish(input: &mut [f32]) {
    for x in input {
        *x *= (x.max(0.0) + (-x.abs()).exp().ln_1p()).tanh();
    }
}

fn transpose(input: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        for col in 0..columns {
            output[col * rows + row] = input[row * columns + col];
        }
    }
    output
}

fn validate_mel(mel: &[f32], frames: usize, name: &str) -> Result<(), String> {
    if frames == 0
        || frames > MAX_FRAMES
        || frames.checked_mul(MEL) != Some(mel.len())
        || mel.iter().any(|v| !v.is_finite())
    {
        return Err(format!("invalid finite flow {name} [80,{frames}]"));
    }
    Ok(())
}

fn load_linear<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
    bias: bool,
) -> Result<Linear<'a>, String> {
    let weight_name = format!("{name}.weight");
    let types = [
        GGMLType::F32,
        GGMLType::F16,
        GGMLType::BF16,
        GGMLType::Q8_0,
        GGMLType::Q4_0,
        GGMLType::Q4_1,
        GGMLType::Q4K,
        GGMLType::Q5K,
        GGMLType::Q6K,
    ];
    let kind = require_tensor_type(
        source,
        &weight_name,
        &[n_in as u64, n_out as u64],
        if n_in * n_out < 65536 {
            &types[..1]
        } else {
            &types
        },
    )?;
    let bias_name = format!("{name}.bias");
    let bias_values = if bias {
        Some(load_f32(source, &bias_name, &[n_out as u64])?)
    } else {
        None
    };
    if kind == GGMLType::F32 {
        let bytes = source
            .tensor_slice(&weight_name)
            .ok_or_else(|| format!("missing {weight_name}"))?;
        let weight = Weight::from_f32_bytes(bytes, n_in, n_out)
            .ok_or_else(|| format!("invalid {weight_name}"))?;
        Linear::from_weight(weight, bias_values)
    } else {
        Linear::from_source(
            source,
            &weight_name,
            bias.then_some(bias_name.as_str()),
            n_in,
            n_out,
        )
    }
}

pub(crate) fn load_f32(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
) -> Result<Vec<f32>, String> {
    require_tensor_type(source, name, dims, &[GGMLType::F32])?;
    let values = load_f32_tensor(source, name, dims)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(format!("nonfinite {name}"));
    }
    Ok(values)
}

/// F32 Conv1d OIK weights; all arithmetic and dispatch belong to the shared operator.
pub(crate) struct Conv {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
    kernel: usize,
    groups: usize,
}

impl Conv {
    pub(crate) fn load(
        source: &dyn TensorSource,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        groups: usize,
    ) -> Result<Self, String> {
        if input == 0
            || output == 0
            || kernel == 0
            || groups == 0
            || input % groups != 0
            || output % groups != 0
        {
            return Err(format!("invalid Conv1d dimensions for {name}"));
        }
        Ok(Self {
            weight: load_f32(
                source,
                &format!("{name}.weight"),
                &[kernel as u64, (input / groups) as u64, output as u64],
            )?,
            bias: load_f32(source, &format!("{name}.bias"), &[output as u64])?,
            input,
            output,
            kernel,
            groups,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        input: &[f32],
        frames: usize,
        left_pad: usize,
        right_pad: usize,
        stride: usize,
        dilation: usize,
        pool: &ComputePool,
    ) -> Result<(Vec<f32>, usize), String> {
        let (padded, padded_frames) = pad_channels(input, self.input, frames, left_pad, right_pad)?;
        let (output, shape) = nn::conv1d(
            pool,
            &padded,
            [self.input, padded_frames],
            &self.weight,
            [self.output, self.input / self.groups, self.kernel],
            Some(&self.bias),
            stride,
            0,
            dilation,
            self.groups,
        )?;
        Ok((output, shape[1]))
    }
}

fn cosine_euler(
    noise: &[f32],
    steps: usize,
    mut velocity: impl FnMut(&[f32], f32) -> Result<Vec<f32>, String>,
) -> Result<Vec<f32>, String> {
    if steps == 0 {
        return Err("Euler requires at least one step".into());
    }
    let mut x = noise.to_vec();
    let mut t = 0.0;
    for step in 1..=steps {
        let next = 1.0 - ((step as f32 / steps as f32) * std::f32::consts::FRAC_PI_2).cos();
        let delta = next - t;
        let update = velocity(&x, t)?;
        if update.len() != x.len() || update.iter().any(|v| !v.is_finite()) {
            return Err("invalid flow velocity".into());
        }
        for (x, update) in x.iter_mut().zip(update) {
            *x += delta * update;
        }
        t = next;
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite integrated mel".into());
    }
    Ok(x)
}

fn timestep_features(time: f32) -> Vec<f32> {
    let mut features = vec![0.0; 256];
    // Python computes this constant in double precision before the F32 tensor multiply.
    let exponent = -(10000.0f64.ln() / 127.0) as f32;
    for i in 0..128 {
        let phase = time * 1000.0 * (i as f32 * exponent).exp();
        features[i] = phase.sin();
        features[128 + i] = phase.cos();
    }
    features
}

fn rotate_prefix(input: &mut [f32], width: usize, inv_freq: &[f32]) {
    for (time, row) in input.chunks_exact_mut(width).enumerate() {
        for (pair, &freq) in inv_freq.iter().enumerate() {
            let (sin, cos) = (time as f32 * freq).sin_cos();
            let a = row[pair * 2];
            let b = row[pair * 2 + 1];
            row[pair * 2] = a * cos - b * sin;
            row[pair * 2 + 1] = b * cos + a * sin;
        }
    }
}

fn pad_channels(
    input: &[f32],
    channels: usize,
    frames: usize,
    left: usize,
    right: usize,
) -> Result<(Vec<f32>, usize), String> {
    if channels == 0 || frames == 0 || channels.checked_mul(frames) != Some(input.len()) {
        return Err("invalid channels-first convolution input".into());
    }
    let length = frames
        .checked_add(left)
        .and_then(|v| v.checked_add(right))
        .ok_or("padding overflow")?;
    let size = channels
        .checked_mul(length)
        .ok_or("padding size overflow")?;
    let mut output = vec![0.0; size];
    for channel in 0..channels {
        output[channel * length + left..channel * length + left + frames]
            .copy_from_slice(&input[channel * frames..(channel + 1) * frames]);
    }
    Ok((output, length))
}

fn prompt_condition(prompt: &[f32], frames: usize) -> Result<(Vec<f32>, usize), String> {
    if frames == 0
        || frames > MAX_FRAMES
        || prompt.len() % MEL != 0
        || prompt.len() / MEL > frames
        || prompt.iter().any(|v| !v.is_finite())
    {
        return Err("invalid prompt mel shape".into());
    }
    let prefix = prompt.len() / MEL;
    let mut cond = vec![0.0; MEL * frames];
    for c in 0..MEL {
        cond[c * frames..c * frames + prefix]
            .copy_from_slice(&prompt[c * prefix..(c + 1) * prefix]);
    }
    Ok((cond, prefix))
}

fn crop_prompt(mel: &[f32], frames: usize, prefix: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(MEL * (frames - prefix));
    for c in 0..MEL {
        output.extend_from_slice(&mel[c * frames + prefix..(c + 1) * frames]);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_euler_uses_previous_time_and_four_nonuniform_steps() {
        let mut times = Vec::new();
        let out = cosine_euler(&[2.0], 4, |_, t| {
            times.push(t);
            Ok(vec![t])
        })
        .unwrap();
        let expected = [0.0, 0.07612047, 0.29289323, 0.6173166];
        assert_eq!(times.len(), expected.len());
        for (a, b) in times.iter().zip(expected) {
            assert!((a - b).abs() < 2e-7);
        }
        let mut integrated = 2.0;
        for i in 0..4 {
            integrated += (if i == 3 { 1.0 } else { expected[i + 1] } - expected[i]) * expected[i];
        }
        assert!((out[0] - integrated).abs() < 1e-6);
        assert!(cosine_euler(&[0.0], 0, |_, _| Ok(vec![1.0])).is_err());
        assert!(cosine_euler(&[0.0], 1, |_, _| Ok(vec![])).is_err());
    }

    #[test]
    fn timestep_features_are_sine_then_cosine_at_scale_1000() {
        let zero = timestep_features(0.0);
        assert_eq!(&zero[..128], &[0.0; 128]);
        assert_eq!(&zero[128..], &[1.0; 128]);
        let time = timestep_features(0.001);
        assert!((time[0] - 1.0f32.sin()).abs() < 1e-7);
        assert!((time[128] - 1.0f32.cos()).abs() < 1e-7);
        assert!((time[127] - 0.0001).abs() < 1e-9);
    }

    #[test]
    fn rope_rotates_adjacent_pairs_only_in_flattened_prefix() {
        let mut x = vec![1.0, 2.0, 9.0, 10.0, 3.0, 4.0, 11.0, 12.0];
        rotate_prefix(&mut x, 4, &[std::f32::consts::FRAC_PI_2]);
        assert_eq!(&x[..4], &[1.0, 2.0, 9.0, 10.0]);
        assert!((x[4] + 4.0).abs() < 1e-6);
        assert!((x[5] - 3.0).abs() < 1e-6);
        assert_eq!(&x[6..], &[11.0, 12.0]);
    }

    #[test]
    fn causal_and_lookahead_padding_preserve_channel_boundaries() {
        let x = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            pad_channels(&x, 2, 2, 1, 0).unwrap(),
            (vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0], 3)
        );
        assert_eq!(
            pad_channels(&x, 2, 2, 0, 1).unwrap(),
            (vec![1.0, 2.0, 0.0, 3.0, 4.0, 0.0], 3)
        );
        assert!(pad_channels(&x[..3], 2, 2, 0, 0).is_err());
    }

    #[test]
    fn prompt_condition_and_output_crop_use_channels_first() {
        let prompt: Vec<f32> = (0..MEL * 2).map(|i| i as f32).collect();
        let (cond, prefix) = prompt_condition(&prompt, 5).unwrap();
        assert_eq!(prefix, 2);
        for c in 0..MEL {
            assert_eq!(
                &cond[c * 5..c * 5 + 5],
                &[prompt[c * 2], prompt[c * 2 + 1], 0.0, 0.0, 0.0]
            );
        }
        let mel: Vec<f32> = (0..MEL * 5).map(|i| i as f32).collect();
        let cropped = crop_prompt(&mel, 5, prefix);
        for c in 0..MEL {
            assert_eq!(&cropped[c * 3..c * 3 + 3], &mel[c * 5 + 2..c * 5 + 5]);
        }
        assert!(prompt_condition(&prompt, 1).is_err());
        assert!(prompt_condition(&[0.0], 5).is_err());
    }
}
