use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::kernels::{
    attention_online, checked_len, layer_norm_rows, load_float_values, rms_norm_rows,
    AttentionSpec, Linear,
};
use super::text::TextConditioning;
use super::video_vae::VideoLatent;
use super::DreamXOptions;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::ops::{gelu_inplace, silu_inplace};

const VIDEO_PREFIX: &str = "dreamx.creator.video";
const AUDIO_PREFIX: &str = "dreamx.creator.audio";
const JOINT_PREFIX: &str = "dreamx.creator.joint";
const VIDEO_DIM: usize = 3072;
const VIDEO_FFN: usize = 14336;
const VIDEO_HEADS: usize = 24;
const VIDEO_CHANNELS: usize = 48;
const AUDIO_DIM: usize = 1536;
const AUDIO_FFN: usize = 8960;
const AUDIO_HEADS: usize = 12;
const AUDIO_CHANNELS: usize = 128;
const HEAD_DIM: usize = 128;
const TEXT_DIM: usize = 4096;
const TEXT_TOKENS: usize = 512;
const TIME_DIM: usize = 256;
const LAYERS: usize = 30;
const FIRST_JOINT_LAYER: usize = 15;
const EPSILON: f32 = 1e-6;
const AUDIO_FPS: f32 = 50.0;
const VAE_TEMPORAL_STRIDE: f32 = 4.0;
const FLOW_SHIFT: f32 = 5.0;
const TEXT_GUIDANCE: f32 = 5.0;
const VIDEO_BRIDGE_GUIDANCE: f32 = 3.5;
const AUDIO_BRIDGE_GUIDANCE: f32 = 3.5;

#[derive(Clone, Debug)]
pub struct CreatorOutput {
    pub video: VideoLatent,
    /// Channel-major continuous DAC latent `[128, frames]`.
    pub audio: Vec<f32>,
    pub audio_frames: usize,
}

#[derive(Clone, Copy)]
struct BranchDimensions {
    dim: usize,
    ffn: usize,
    heads: usize,
    channels: usize,
}

impl BranchDimensions {
    const VIDEO: Self = Self {
        dim: VIDEO_DIM,
        ffn: VIDEO_FFN,
        heads: VIDEO_HEADS,
        channels: VIDEO_CHANNELS,
    };
    const AUDIO: Self = Self {
        dim: AUDIO_DIM,
        ffn: AUDIO_FFN,
        heads: AUDIO_HEADS,
        channels: AUDIO_CHANNELS,
    };
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BranchKind {
    Video,
    Audio,
}

struct Projection {
    weight: String,
    bias: Option<String>,
    n_in: usize,
    n_out: usize,
}

impl Projection {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        n_in: usize,
        n_out: usize,
        bias: bool,
    ) -> Result<Self, String> {
        let weight = format!("{prefix}.weight");
        let bias = bias.then(|| format!("{prefix}.bias"));
        Linear::from_source(source, &weight, bias.as_deref(), n_in, n_out)?;
        Ok(Self {
            weight,
            bias,
            n_in,
            n_out,
        })
    }

    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        input: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>, String> {
        Linear::from_source(
            source,
            &self.weight,
            self.bias.as_deref(),
            self.n_in,
            self.n_out,
        )?
        .forward(pool, input, rows)
    }
}

struct Attention {
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    heads: usize,
}

#[derive(Clone, Copy)]
enum Rope<'a> {
    None,
    Sequential,
    Positions(&'a [f32]),
    Video([usize; 3]),
}

impl Attention {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        q_dim: usize,
        kv_dim: usize,
        heads: usize,
    ) -> Result<Self, String> {
        if q_dim / heads != HEAD_DIM || q_dim % heads != 0 {
            return Err("Invalid DreamX Creator attention dimensions".into());
        }
        Ok(Self {
            q: Projection::load(source, &format!("{prefix}.q"), q_dim, q_dim, true)?,
            k: Projection::load(source, &format!("{prefix}.k"), kv_dim, q_dim, true)?,
            v: Projection::load(source, &format!("{prefix}.v"), kv_dim, q_dim, true)?,
            o: Projection::load(source, &format!("{prefix}.o"), q_dim, q_dim, true)?,
            norm_q: load_float_values(source, &format!("{prefix}.norm_q.weight"), &[q_dim as u64])?,
            norm_k: load_float_values(source, &format!("{prefix}.norm_k.weight"), &[q_dim as u64])?,
            heads,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        query_input: &[f32],
        query_tokens: usize,
        key_value_input: &[f32],
        key_tokens: usize,
        query_rope: Rope<'_>,
        key_rope: Rope<'_>,
    ) -> Result<Vec<f32>, String> {
        let mut query = self.q.forward(source, pool, query_input, query_tokens)?;
        let mut key = self.k.forward(source, pool, key_value_input, key_tokens)?;
        let value = self.v.forward(source, pool, key_value_input, key_tokens)?;
        query = rms_norm_rows(&query, query_tokens, &self.norm_q, EPSILON)?;
        key = rms_norm_rows(&key, key_tokens, &self.norm_k, EPSILON)?;
        apply_rope(&mut query, query_tokens, self.heads, query_rope)?;
        apply_rope(&mut key, key_tokens, self.heads, key_rope)?;
        let context = attention_online(
            &query,
            &key,
            &value,
            AttentionSpec {
                query_tokens,
                key_tokens,
                query_heads: self.heads,
                key_value_heads: self.heads,
                head_dim: HEAD_DIM,
                causal: false,
                scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            },
        )?;
        self.o.forward(source, pool, &context, query_tokens)
    }
}

struct TransformerBlock {
    self_attention: Attention,
    text_attention: Attention,
    norm3_weight: Vec<f32>,
    norm3_bias: Vec<f32>,
    ffn_in: Projection,
    ffn_out: Projection,
    modulation: Vec<f32>,
    dimensions: BranchDimensions,
}

impl TransformerBlock {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        dimensions: BranchDimensions,
    ) -> Result<Self, String> {
        Ok(Self {
            self_attention: Attention::load(
                source,
                &format!("{prefix}.self_attn"),
                dimensions.dim,
                dimensions.dim,
                dimensions.heads,
            )?,
            text_attention: Attention::load(
                source,
                &format!("{prefix}.cross_attn"),
                dimensions.dim,
                dimensions.dim,
                dimensions.heads,
            )?,
            norm3_weight: load_float_values(
                source,
                &format!("{prefix}.norm3.weight"),
                &[dimensions.dim as u64],
            )?,
            norm3_bias: load_float_values(
                source,
                &format!("{prefix}.norm3.bias"),
                &[dimensions.dim as u64],
            )?,
            ffn_in: Projection::load(
                source,
                &format!("{prefix}.ffn.0"),
                dimensions.dim,
                dimensions.ffn,
                true,
            )?,
            ffn_out: Projection::load(
                source,
                &format!("{prefix}.ffn.2"),
                dimensions.ffn,
                dimensions.dim,
                true,
            )?,
            modulation: load_float_values(
                source,
                &format!("{prefix}.modulation"),
                &[dimensions.dim as u64, 6, 1],
            )?,
            dimensions,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn self_attention_and_text(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        hidden: &mut [f32],
        tokens: usize,
        time_modulation: &[f32],
        time_rows: usize,
        self_rope: Rope<'_>,
        context: &[f32],
    ) -> Result<(), String> {
        let normalized = modulated_layer_norm(
            hidden,
            tokens,
            self.dimensions.dim,
            &self.modulation,
            time_modulation,
            time_rows,
            0,
            1,
        )?;
        let residual = self.self_attention.forward(
            source,
            pool,
            &normalized,
            tokens,
            &normalized,
            tokens,
            self_rope,
            self_rope,
        )?;
        gated_residual(
            hidden,
            &residual,
            tokens,
            self.dimensions.dim,
            &self.modulation,
            time_modulation,
            time_rows,
            2,
        )?;

        let normalized = layer_norm_rows(
            hidden,
            tokens,
            Some(&self.norm3_weight),
            Some(&self.norm3_bias),
            EPSILON,
        )?;
        let residual = self.text_attention.forward(
            source,
            pool,
            &normalized,
            tokens,
            context,
            TEXT_TOKENS,
            Rope::None,
            Rope::None,
        )?;
        add_residual(hidden, &residual)
    }

    fn feed_forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        hidden: &mut [f32],
        tokens: usize,
        time_modulation: &[f32],
        time_rows: usize,
    ) -> Result<(), String> {
        let normalized = modulated_layer_norm(
            hidden,
            tokens,
            self.dimensions.dim,
            &self.modulation,
            time_modulation,
            time_rows,
            3,
            4,
        )?;
        let mut residual = self.ffn_in.forward(source, pool, &normalized, tokens)?;
        gelu_inplace(&mut residual);
        let residual = self.ffn_out.forward(source, pool, &residual, tokens)?;
        gated_residual(
            hidden,
            &residual,
            tokens,
            self.dimensions.dim,
            &self.modulation,
            time_modulation,
            time_rows,
            5,
        )
    }
}

struct OutputHead {
    projection: Projection,
    modulation: Vec<f32>,
    dim: usize,
}

impl OutputHead {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        dim: usize,
        output: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            projection: Projection::load(
                source,
                &format!("{prefix}.head.head"),
                dim,
                output,
                true,
            )?,
            modulation: load_float_values(
                source,
                &format!("{prefix}.head.modulation"),
                &[dim as u64, 2, 1],
            )?,
            dim,
        })
    }

    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        hidden: &[f32],
        tokens: usize,
        time_embedding: &[f32],
        time_rows: usize,
    ) -> Result<Vec<f32>, String> {
        if !matches!(time_rows, 1) && time_rows != tokens {
            return Err("Invalid DreamX Creator output time rows".into());
        }
        let mut normalized = layer_norm_no_affine(hidden, tokens, self.dim)?;
        for token in 0..tokens {
            let time_row = if time_rows == 1 { 0 } else { token };
            for dimension in 0..self.dim {
                let shift =
                    self.modulation[dimension] + time_embedding[time_row * self.dim + dimension];
                let scale = self.modulation[self.dim + dimension]
                    + time_embedding[time_row * self.dim + dimension];
                let index = token * self.dim + dimension;
                normalized[index] = normalized[index] * (1.0 + scale) + shift;
            }
        }
        self.projection.forward(source, pool, &normalized, tokens)
    }
}

struct Branch {
    kind: BranchKind,
    dimensions: BranchDimensions,
    patch_weight: Vec<f32>,
    patch_bias: Vec<f32>,
    text_in: Projection,
    text_out: Projection,
    time_in: Projection,
    time_out: Projection,
    time_projection: Projection,
    blocks: Vec<TransformerBlock>,
    head: OutputHead,
}

impl Branch {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        kind: BranchKind,
        dimensions: BranchDimensions,
    ) -> Result<Self, String> {
        let patch_dims = match kind {
            BranchKind::Video => vec![2, 2, 1, dimensions.channels as u64, dimensions.dim as u64],
            BranchKind::Audio => vec![1, dimensions.channels as u64, dimensions.dim as u64],
        };
        let patch_weight = load_float_values(
            source,
            &format!("{prefix}.patch_embedding.weight"),
            &patch_dims,
        )?;
        let patch_bias = load_float_values(
            source,
            &format!("{prefix}.patch_embedding.bias"),
            &[dimensions.dim as u64],
        )?;
        let mut blocks = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            blocks.push(TransformerBlock::load(
                source,
                &format!("{prefix}.blocks.{layer}"),
                dimensions,
            )?);
        }
        let output_width = match kind {
            BranchKind::Video => dimensions.channels * 4,
            BranchKind::Audio => dimensions.channels,
        };
        Ok(Self {
            kind,
            dimensions,
            patch_weight,
            patch_bias,
            text_in: Projection::load(
                source,
                &format!("{prefix}.text_embedding.0"),
                TEXT_DIM,
                dimensions.dim,
                true,
            )?,
            text_out: Projection::load(
                source,
                &format!("{prefix}.text_embedding.2"),
                dimensions.dim,
                dimensions.dim,
                true,
            )?,
            time_in: Projection::load(
                source,
                &format!("{prefix}.time_embedding.0"),
                TIME_DIM,
                dimensions.dim,
                true,
            )?,
            time_out: Projection::load(
                source,
                &format!("{prefix}.time_embedding.2"),
                dimensions.dim,
                dimensions.dim,
                true,
            )?,
            time_projection: Projection::load(
                source,
                &format!("{prefix}.time_projection.1"),
                dimensions.dim,
                dimensions.dim * 6,
                true,
            )?,
            blocks,
            head: OutputHead::load(source, prefix, dimensions.dim, output_width)?,
        })
    }

    fn embed_context(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        context: &[f32],
    ) -> Result<Vec<f32>, String> {
        if context.len() != TEXT_TOKENS * TEXT_DIM {
            return Err("Invalid DreamX Creator text context".into());
        }
        let mut hidden = self.text_in.forward(source, pool, context, TEXT_TOKENS)?;
        gelu_inplace(&mut hidden);
        self.text_out.forward(source, pool, &hidden, TEXT_TOKENS)
    }

    fn time_embeddings(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        timesteps: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let rows = timesteps.len();
        let sinusoidal = sinusoidal_embedding(timesteps, TIME_DIM)?;
        let mut embedding = self.time_in.forward(source, pool, &sinusoidal, rows)?;
        silu_inplace(&mut embedding);
        let embedding = self.time_out.forward(source, pool, &embedding, rows)?;
        let mut projected_input = embedding.clone();
        silu_inplace(&mut projected_input);
        let modulation = self
            .time_projection
            .forward(source, pool, &projected_input, rows)?;
        Ok((embedding, modulation))
    }

    fn patch_video(
        &self,
        pool: &ComputePool,
        latent: &[f32],
        shape: [usize; 4],
    ) -> Result<Vec<f32>, String> {
        let [channels, frames, height, width] = shape;
        if self.kind != BranchKind::Video
            || channels != self.dimensions.channels
            || !height.is_multiple_of(2)
            || !width.is_multiple_of(2)
            || latent.len() != checked_len("DreamX Creator video latent", &shape)?
        {
            return Err("Invalid DreamX Creator video patch input".into());
        }
        let patch_height = height / 2;
        let patch_width = width / 2;
        let tokens = frames * patch_height * patch_width;
        let dim = self.dimensions.dim;
        let mut output = vec![0.0; checked_len("DreamX Creator video patches", &[tokens, dim])?];
        let output_address = output.as_mut_ptr() as usize;
        pool.compute(|thread, threads| {
            for index in (thread..tokens * dim).step_by(threads) {
                let token = index / dim;
                let output_channel = index % dim;
                let patch_x = token % patch_width;
                let patch_y = (token / patch_width) % patch_height;
                let frame = token / (patch_width * patch_height);
                let mut value = self.patch_bias[output_channel];
                for input_channel in 0..channels {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let input_index =
                                ((input_channel * frames + frame) * height + patch_y * 2 + dy)
                                    * width
                                    + patch_x * 2
                                    + dx;
                            let weight_index =
                                (((output_channel * channels + input_channel) * 2 + dy) * 2) + dx;
                            value += latent[input_index] * self.patch_weight[weight_index];
                        }
                    }
                }
                unsafe { (output_address as *mut f32).add(index).write(value) };
            }
        });
        Ok(output)
    }

    fn patch_audio(
        &self,
        pool: &ComputePool,
        latent: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, String> {
        if self.kind != BranchKind::Audio
            || frames == 0
            || latent.len() != self.dimensions.channels * frames
        {
            return Err("Invalid DreamX Creator audio patch input".into());
        }
        let dim = self.dimensions.dim;
        let channels = self.dimensions.channels;
        let mut output = vec![0.0; frames * dim];
        let output_address = output.as_mut_ptr() as usize;
        pool.compute(|thread, threads| {
            for index in (thread..frames * dim).step_by(threads) {
                let frame = index / dim;
                let output_channel = index % dim;
                let mut value = self.patch_bias[output_channel];
                for input_channel in 0..channels {
                    value += latent[input_channel * frames + frame]
                        * self.patch_weight[output_channel * channels + input_channel];
                }
                unsafe { (output_address as *mut f32).add(index).write(value) };
            }
        });
        Ok(output)
    }

    fn unpatch_video(&self, patches: &[f32], shape: [usize; 4]) -> Result<VideoLatent, String> {
        let [channels, frames, height, width] = shape;
        let patch_height = height / 2;
        let patch_width = width / 2;
        let output_width = channels * 4;
        let tokens = frames * patch_height * patch_width;
        if self.kind != BranchKind::Video || patches.len() != tokens * output_width {
            return Err("Invalid DreamX Creator video head output".into());
        }
        let mut values = vec![0.0; channels * frames * height * width];
        for frame in 0..frames {
            for patch_y in 0..patch_height {
                for patch_x in 0..patch_width {
                    let token = (frame * patch_height + patch_y) * patch_width + patch_x;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            for channel in 0..channels {
                                let patch_index = ((dy * 2 + dx) * channels) + channel;
                                let output_index =
                                    ((channel * frames + frame) * height + patch_y * 2 + dy)
                                        * width
                                        + patch_x * 2
                                        + dx;
                                values[output_index] = patches[token * output_width + patch_index];
                            }
                        }
                    }
                }
            }
        }
        VideoLatent::new(values, shape)
    }

    fn unpatch_audio(&self, patches: &[f32], frames: usize) -> Result<Vec<f32>, String> {
        let channels = self.dimensions.channels;
        if self.kind != BranchKind::Audio || patches.len() != frames * channels {
            return Err("Invalid DreamX Creator audio head output".into());
        }
        let mut values = vec![0.0; channels * frames];
        for frame in 0..frames {
            for channel in 0..channels {
                values[channel * frames + frame] = patches[frame * channels + channel];
            }
        }
        Ok(values)
    }
}

struct GatedCrossModalAttention {
    attention: Attention,
    norm_x_weight: Vec<f32>,
    norm_x_bias: Vec<f32>,
    norm_y_weight: Vec<f32>,
    norm_y_bias: Vec<f32>,
    gate_hidden: Projection,
    gate_context_norm_weight: Vec<f32>,
    gate_context_norm_bias: Vec<f32>,
    gate_context: Projection,
    gate_bias: Vec<f32>,
    q_dim: usize,
    kv_dim: usize,
    heads: usize,
}

impl GatedCrossModalAttention {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        q_dim: usize,
        kv_dim: usize,
        heads: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            attention: Attention::load(source, prefix, q_dim, kv_dim, heads)?,
            norm_x_weight: load_float_values(
                source,
                &format!("{prefix}.norm_x.weight"),
                &[q_dim as u64],
            )?,
            norm_x_bias: load_float_values(
                source,
                &format!("{prefix}.norm_x.bias"),
                &[q_dim as u64],
            )?,
            norm_y_weight: load_float_values(
                source,
                &format!("{prefix}.norm.weight"),
                &[kv_dim as u64],
            )?,
            norm_y_bias: load_float_values(
                source,
                &format!("{prefix}.norm.bias"),
                &[kv_dim as u64],
            )?,
            gate_hidden: Projection::load(
                source,
                &format!("{prefix}.gate_hidden"),
                q_dim,
                heads,
                false,
            )?,
            gate_context_norm_weight: load_float_values(
                source,
                &format!("{prefix}.gate_context_norm.weight"),
                &[HEAD_DIM as u64],
            )?,
            gate_context_norm_bias: load_float_values(
                source,
                &format!("{prefix}.gate_context_norm.bias"),
                &[HEAD_DIM as u64],
            )?,
            gate_context: Projection::load(
                source,
                &format!("{prefix}.gate_context"),
                HEAD_DIM,
                1,
                false,
            )?,
            gate_bias: load_float_values(source, &format!("{prefix}.gate_bias"), &[heads as u64])?,
            q_dim,
            kv_dim,
            heads,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        primary: &[f32],
        primary_tokens: usize,
        context: &[f32],
        context_tokens: usize,
        primary_positions: &[f32],
        context_positions: &[f32],
    ) -> Result<Vec<f32>, String> {
        if primary.len() != primary_tokens * self.q_dim
            || context.len() != context_tokens * self.kv_dim
        {
            return Err("Invalid DreamX cross-modal hidden states".into());
        }
        let primary_norm = layer_norm_rows(
            primary,
            primary_tokens,
            Some(&self.norm_x_weight),
            Some(&self.norm_x_bias),
            EPSILON,
        )?;
        let context_norm = layer_norm_rows(
            context,
            context_tokens,
            Some(&self.norm_y_weight),
            Some(&self.norm_y_bias),
            EPSILON,
        )?;

        let mut query = self
            .attention
            .q
            .forward(source, pool, &primary_norm, primary_tokens)?;
        let mut key = self
            .attention
            .k
            .forward(source, pool, &context_norm, context_tokens)?;
        let value = self
            .attention
            .v
            .forward(source, pool, &context_norm, context_tokens)?;
        query = rms_norm_rows(&query, primary_tokens, &self.attention.norm_q, EPSILON)?;
        key = rms_norm_rows(&key, context_tokens, &self.attention.norm_k, EPSILON)?;
        apply_rope(
            &mut query,
            primary_tokens,
            self.heads,
            Rope::Positions(primary_positions),
        )?;
        apply_rope(
            &mut key,
            context_tokens,
            self.heads,
            Rope::Positions(context_positions),
        )?;
        let mut attended = attention_online(
            &query,
            &key,
            &value,
            AttentionSpec {
                query_tokens: primary_tokens,
                key_tokens: context_tokens,
                query_heads: self.heads,
                key_value_heads: self.heads,
                head_dim: HEAD_DIM,
                causal: false,
                scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            },
        )?;

        let hidden_gate = self
            .gate_hidden
            .forward(source, pool, &primary_norm, primary_tokens)?;
        let normalized_context = layer_norm_rows(
            &attended,
            primary_tokens * self.heads,
            Some(&self.gate_context_norm_weight),
            Some(&self.gate_context_norm_bias),
            EPSILON,
        )?;
        let context_gate = self.gate_context.forward(
            source,
            pool,
            &normalized_context,
            primary_tokens * self.heads,
        )?;
        for token in 0..primary_tokens {
            for head in 0..self.heads {
                let gate_index = token * self.heads + head;
                let gate = sigmoid(
                    hidden_gate[gate_index] + context_gate[gate_index] + self.gate_bias[head],
                );
                let start = gate_index * HEAD_DIM;
                for value in &mut attended[start..start + HEAD_DIM] {
                    *value *= gate;
                }
            }
        }
        self.attention
            .o
            .forward(source, pool, &attended, primary_tokens)
    }
}

struct JointLayer {
    video_from_audio: GatedCrossModalAttention,
    audio_from_video: GatedCrossModalAttention,
}

impl JointLayer {
    fn load(source: &dyn TensorSource, layer: usize) -> Result<Self, String> {
        let prefix = format!("{JOINT_PREFIX}.joint_blocks.{layer}");
        Ok(Self {
            video_from_audio: GatedCrossModalAttention::load(
                source,
                &format!("{prefix}.video_cross_attn_audio"),
                VIDEO_DIM,
                AUDIO_DIM,
                VIDEO_HEADS,
            )?,
            audio_from_video: GatedCrossModalAttention::load(
                source,
                &format!("{prefix}.audio_cross_attn_video"),
                AUDIO_DIM,
                VIDEO_DIM,
                AUDIO_HEADS,
            )?,
        })
    }
}

struct PreparedBranch {
    hidden: Vec<f32>,
    time_embedding: Vec<f32>,
    time_modulation: Vec<f32>,
    time_rows: usize,
    tokens: usize,
    grid: [usize; 3],
}

struct EmbeddedContexts {
    video_positive: Vec<f32>,
    video_negative: Vec<f32>,
    audio_positive: Vec<f32>,
    audio_negative: Vec<f32>,
}

struct ModelPrediction {
    video: Vec<f32>,
    audio: Vec<f32>,
}

pub struct CreatorModel {
    source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    video: Branch,
    audio: Branch,
    joint: Vec<JointLayer>,
}

impl CreatorModel {
    pub fn load(source: Arc<dyn TensorSource>, pool: Arc<ComputePool>) -> Result<Self, String> {
        let video = Branch::load(
            source.as_ref(),
            VIDEO_PREFIX,
            BranchKind::Video,
            BranchDimensions::VIDEO,
        )?;
        let audio = Branch::load(
            source.as_ref(),
            AUDIO_PREFIX,
            BranchKind::Audio,
            BranchDimensions::AUDIO,
        )?;
        let mut joint = Vec::with_capacity(LAYERS - FIRST_JOINT_LAYER);
        for layer in FIRST_JOINT_LAYER..LAYERS {
            joint.push(JointLayer::load(source.as_ref(), layer)?);
        }
        Ok(Self {
            source,
            pool,
            video,
            audio,
            joint,
        })
    }

    pub fn denoise(
        &self,
        first_frame: VideoLatent,
        conditioning: &TextConditioning,
        options: &DreamXOptions,
    ) -> Result<CreatorOutput, String> {
        if options.steps == 0 || options.fps == 0 || !options.duration_seconds.is_finite() {
            return Err("Invalid DreamX Creator denoise options".into());
        }
        let [channels, first_frames, height, width] = first_frame.shape();
        if channels != VIDEO_CHANNELS
            || first_frames != 1
            || !height.is_multiple_of(2)
            || !width.is_multiple_of(2)
        {
            return Err("Invalid DreamX Creator first-frame latent".into());
        }
        let requested_frames = (options.duration_seconds as f64 * options.fps as f64) as usize;
        let requested_frames = requested_frames.max(1);
        let output_frames = ((requested_frames - 1) / 4) * 4 + 1;
        let latent_frames = (output_frames - 1) / 4 + 1;
        let duration = output_frames as f32 / options.fps as f32;
        let audio_frames = ((duration * AUDIO_FPS).ceil() as usize).max(1);
        let video_shape = [VIDEO_CHANNELS, latent_frames, height, width];

        let contexts = self.embed_contexts(conditioning)?;
        let mut rng = StdRng::seed_from_u64(options.seed as u64);
        let mut video_latent =
            gaussian_values(&mut rng, checked_len("DreamX video noise", &video_shape)?);
        restore_first_frame(&mut video_latent, video_shape, first_frame.as_slice())?;
        let mut audio_latent = gaussian_values(&mut rng, AUDIO_CHANNELS * audio_frames);

        let video_schedule = flow_match_schedule(options.steps, FLOW_SHIFT)?;
        let audio_schedule = flow_match_schedule(options.steps, FLOW_SHIFT)?;
        if video_schedule.len() != audio_schedule.len() {
            return Err("DreamX video/audio timestep count mismatch".into());
        }
        let video_tokens = latent_frames * (height / 2) * (width / 2);
        let first_frame_tokens = (height / 2) * (width / 2);
        let video_positions =
            video_temporal_positions(latent_frames, height / 2, width / 2, options.fps)?;
        let audio_positions: Vec<f32> = (0..audio_frames).map(|value| value as f32).collect();

        for (video_step, audio_step) in video_schedule.iter().zip(&audio_schedule) {
            let video_times =
                video_token_timesteps(video_step.timestep, first_frame_tokens, video_tokens)?;
            let video_prepared = PreparedBranch {
                hidden: self
                    .video
                    .patch_video(&self.pool, &video_latent, video_shape)?,
                time_embedding: Vec::new(),
                time_modulation: Vec::new(),
                time_rows: video_tokens,
                tokens: video_tokens,
                grid: [latent_frames, height / 2, width / 2],
            };
            let (video_time_embedding, video_time_modulation) =
                self.video
                    .time_embeddings(self.source.as_ref(), &self.pool, &video_times)?;
            let video_prepared = PreparedBranch {
                time_embedding: video_time_embedding,
                time_modulation: video_time_modulation,
                ..video_prepared
            };
            let (audio_time_embedding, audio_time_modulation) = self.audio.time_embeddings(
                self.source.as_ref(),
                &self.pool,
                &[audio_step.timestep],
            )?;
            let audio_prepared = PreparedBranch {
                hidden: self
                    .audio
                    .patch_audio(&self.pool, &audio_latent, audio_frames)?,
                time_embedding: audio_time_embedding,
                time_modulation: audio_time_modulation,
                time_rows: 1,
                tokens: audio_frames,
                grid: [audio_frames, 1, 1],
            };

            let no_bridge = self.forward_prepared(
                &video_prepared,
                &audio_prepared,
                &contexts.video_negative,
                &contexts.audio_negative,
                &video_positions,
                &audio_positions,
                false,
            )?;
            let negative_bridge = self.forward_prepared(
                &video_prepared,
                &audio_prepared,
                &contexts.video_negative,
                &contexts.audio_negative,
                &video_positions,
                &audio_positions,
                true,
            )?;
            let positive_bridge = self.forward_prepared(
                &video_prepared,
                &audio_prepared,
                &contexts.video_positive,
                &contexts.audio_positive,
                &video_positions,
                &audio_positions,
                true,
            )?;
            let video_velocity = multimodal_cfg(
                &no_bridge.video,
                &negative_bridge.video,
                &positive_bridge.video,
                VIDEO_BRIDGE_GUIDANCE,
                TEXT_GUIDANCE,
            )?;
            let audio_velocity = multimodal_cfg(
                &no_bridge.audio,
                &negative_bridge.audio,
                &positive_bridge.audio,
                AUDIO_BRIDGE_GUIDANCE,
                TEXT_GUIDANCE,
            )?;
            euler_step(&mut video_latent, &video_velocity, video_step.delta_sigma)?;
            restore_first_frame(&mut video_latent, video_shape, first_frame.as_slice())?;
            euler_step(&mut audio_latent, &audio_velocity, audio_step.delta_sigma)?;
        }

        Ok(CreatorOutput {
            video: VideoLatent::new(video_latent, video_shape)?,
            audio: audio_latent,
            audio_frames,
        })
    }

    fn embed_contexts(&self, conditioning: &TextConditioning) -> Result<EmbeddedContexts, String> {
        if conditioning.positive_len > TEXT_TOKENS || conditioning.negative_len > TEXT_TOKENS {
            return Err("Invalid DreamX Creator text lengths".into());
        }
        Ok(EmbeddedContexts {
            video_positive: self.video.embed_context(
                self.source.as_ref(),
                &self.pool,
                &conditioning.positive,
            )?,
            video_negative: self.video.embed_context(
                self.source.as_ref(),
                &self.pool,
                &conditioning.negative,
            )?,
            audio_positive: self.audio.embed_context(
                self.source.as_ref(),
                &self.pool,
                &conditioning.positive,
            )?,
            audio_negative: self.audio.embed_context(
                self.source.as_ref(),
                &self.pool,
                &conditioning.negative,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_prepared(
        &self,
        video: &PreparedBranch,
        audio: &PreparedBranch,
        video_context: &[f32],
        audio_context: &[f32],
        video_positions: &[f32],
        audio_positions: &[f32],
        bridge: bool,
    ) -> Result<ModelPrediction, String> {
        let mut video_hidden = video.hidden.clone();
        let mut audio_hidden = audio.hidden.clone();
        for layer in 0..LAYERS {
            self.video.blocks[layer].self_attention_and_text(
                self.source.as_ref(),
                &self.pool,
                &mut video_hidden,
                video.tokens,
                &video.time_modulation,
                video.time_rows,
                Rope::Video(video.grid),
                video_context,
            )?;
            self.audio.blocks[layer].self_attention_and_text(
                self.source.as_ref(),
                &self.pool,
                &mut audio_hidden,
                audio.tokens,
                &audio.time_modulation,
                audio.time_rows,
                Rope::Sequential,
                audio_context,
            )?;

            if bridge && layer >= FIRST_JOINT_LAYER {
                let joint = &self.joint[layer - FIRST_JOINT_LAYER];
                (video_hidden, audio_hidden) = joint_update_from_snapshot(
                    video_hidden,
                    audio_hidden,
                    |video_snapshot, audio_snapshot| {
                        joint.video_from_audio.forward(
                            self.source.as_ref(),
                            &self.pool,
                            video_snapshot,
                            video.tokens,
                            audio_snapshot,
                            audio.tokens,
                            video_positions,
                            audio_positions,
                        )
                    },
                    |audio_snapshot, video_snapshot| {
                        joint.audio_from_video.forward(
                            self.source.as_ref(),
                            &self.pool,
                            audio_snapshot,
                            audio.tokens,
                            video_snapshot,
                            video.tokens,
                            audio_positions,
                            video_positions,
                        )
                    },
                )?;
            }

            self.video.blocks[layer].feed_forward(
                self.source.as_ref(),
                &self.pool,
                &mut video_hidden,
                video.tokens,
                &video.time_modulation,
                video.time_rows,
            )?;
            self.audio.blocks[layer].feed_forward(
                self.source.as_ref(),
                &self.pool,
                &mut audio_hidden,
                audio.tokens,
                &audio.time_modulation,
                audio.time_rows,
            )?;
        }
        let video_patches = self.video.head.forward(
            self.source.as_ref(),
            &self.pool,
            &video_hidden,
            video.tokens,
            &video.time_embedding,
            video.time_rows,
        )?;
        let audio_patches = self.audio.head.forward(
            self.source.as_ref(),
            &self.pool,
            &audio_hidden,
            audio.tokens,
            &audio.time_embedding,
            audio.time_rows,
        )?;
        Ok(ModelPrediction {
            video: self
                .video
                .unpatch_video(
                    &video_patches,
                    [
                        VIDEO_CHANNELS,
                        video.grid[0],
                        video.grid[1] * 2,
                        video.grid[2] * 2,
                    ],
                )?
                .into_values(),
            audio: self.audio.unpatch_audio(&audio_patches, audio.tokens)?,
        })
    }
}

fn joint_update_from_snapshot<F, G>(
    mut video: Vec<f32>,
    mut audio: Vec<f32>,
    video_update: F,
    audio_update: G,
) -> Result<(Vec<f32>, Vec<f32>), String>
where
    F: FnOnce(&[f32], &[f32]) -> Result<Vec<f32>, String>,
    G: FnOnce(&[f32], &[f32]) -> Result<Vec<f32>, String>,
{
    let video_residual = video_update(&video, &audio)?;
    let audio_residual = audio_update(&audio, &video)?;
    add_residual(&mut video, &video_residual)?;
    add_residual(&mut audio, &audio_residual)?;
    Ok((video, audio))
}

fn video_token_timesteps(
    timestep: f32,
    first_frame_tokens: usize,
    total_tokens: usize,
) -> Result<Vec<f32>, String> {
    if !timestep.is_finite() || first_frame_tokens == 0 || total_tokens < first_frame_tokens {
        return Err("Invalid DreamX Creator video token timesteps".into());
    }
    let mut values = vec![timestep; total_tokens];
    values[..first_frame_tokens].fill(0.0);
    Ok(values)
}

fn layer_norm_no_affine(input: &[f32], rows: usize, width: usize) -> Result<Vec<f32>, String> {
    if rows == 0 || width == 0 || input.len() != rows * width {
        return Err("Invalid DreamX Creator LayerNorm input".into());
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let start = row * width;
        let values = &input[start..start + width];
        let mean = values.iter().map(|&value| value as f64).sum::<f64>() / width as f64;
        let variance = values
            .iter()
            .map(|&value| {
                let centered = value as f64 - mean;
                centered * centered
            })
            .sum::<f64>()
            / width as f64;
        let inverse = (variance + EPSILON as f64).sqrt().recip() as f32;
        for dimension in 0..width {
            output[start + dimension] = (values[dimension] - mean as f32) * inverse;
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn modulated_layer_norm(
    input: &[f32],
    rows: usize,
    width: usize,
    block_modulation: &[f32],
    time_modulation: &[f32],
    time_rows: usize,
    shift_part: usize,
    scale_part: usize,
) -> Result<Vec<f32>, String> {
    if block_modulation.len() != 6 * width
        || !matches!(time_rows, 1) && time_rows != rows
        || time_modulation.len() != time_rows * 6 * width
    {
        return Err("Invalid DreamX Creator modulation".into());
    }
    let mut output = layer_norm_no_affine(input, rows, width)?;
    for row in 0..rows {
        let time_row = if time_rows == 1 { 0 } else { row };
        for dimension in 0..width {
            let shift = block_modulation[shift_part * width + dimension]
                + time_modulation[(time_row * 6 + shift_part) * width + dimension];
            let scale = block_modulation[scale_part * width + dimension]
                + time_modulation[(time_row * 6 + scale_part) * width + dimension];
            let index = row * width + dimension;
            output[index] = output[index] * (1.0 + scale) + shift;
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn gated_residual(
    hidden: &mut [f32],
    residual: &[f32],
    rows: usize,
    width: usize,
    block_modulation: &[f32],
    time_modulation: &[f32],
    time_rows: usize,
    gate_part: usize,
) -> Result<(), String> {
    if hidden.len() != rows * width
        || residual.len() != hidden.len()
        || block_modulation.len() != 6 * width
        || time_modulation.len() != time_rows * 6 * width
        || !matches!(time_rows, 1) && time_rows != rows
    {
        return Err("Invalid DreamX Creator gated residual".into());
    }
    for row in 0..rows {
        let time_row = if time_rows == 1 { 0 } else { row };
        for dimension in 0..width {
            let gate = block_modulation[gate_part * width + dimension]
                + time_modulation[(time_row * 6 + gate_part) * width + dimension];
            let index = row * width + dimension;
            hidden[index] += gate * residual[index];
        }
    }
    Ok(())
}

fn add_residual(output: &mut [f32], residual: &[f32]) -> Result<(), String> {
    if output.len() != residual.len() {
        return Err("DreamX Creator residual length mismatch".into());
    }
    for (output, &residual) in output.iter_mut().zip(residual) {
        *output += residual;
    }
    Ok(())
}

fn sinusoidal_embedding(positions: &[f32], dim: usize) -> Result<Vec<f32>, String> {
    if positions.is_empty()
        || !dim.is_multiple_of(2)
        || positions.iter().any(|value| !value.is_finite())
    {
        return Err("Invalid DreamX Creator sinusoidal positions".into());
    }
    let half = dim / 2;
    let mut output = vec![0.0; positions.len() * dim];
    for (row, &position) in positions.iter().enumerate() {
        for index in 0..half {
            let frequency = 10000.0f64.powf(-(index as f64) / half as f64);
            let angle = position as f64 * frequency;
            output[row * dim + index] = angle.cos() as f32;
            output[row * dim + half + index] = angle.sin() as f32;
        }
    }
    Ok(output)
}

fn apply_rope(
    values: &mut [f32],
    tokens: usize,
    heads: usize,
    rope: Rope<'_>,
) -> Result<(), String> {
    if matches!(rope, Rope::None) {
        return Ok(());
    }
    if values.len() != tokens * heads * HEAD_DIM {
        return Err("Invalid DreamX Creator RoPE tensor".into());
    }
    if let Rope::Positions(positions) = rope {
        if positions.len() != tokens {
            return Err("Invalid DreamX Creator temporal positions".into());
        }
    }
    if let Rope::Video([frames, height, width]) = rope {
        if frames * height * width != tokens {
            return Err("Invalid DreamX Creator video RoPE grid".into());
        }
    }
    for token in 0..tokens {
        for head in 0..heads {
            let base = (token * heads + head) * HEAD_DIM;
            for pair in 0..HEAD_DIM / 2 {
                let angle = match rope {
                    Rope::None => unreachable!(),
                    Rope::Sequential => {
                        token as f64 * 10000.0f64.powf(-((2 * pair) as f64) / HEAD_DIM as f64)
                    }
                    Rope::Positions(positions) => {
                        positions[token] as f64
                            * 10000.0f64.powf(-((2 * pair) as f64) / HEAD_DIM as f64)
                    }
                    Rope::Video([_, height, width]) => {
                        let x = token % width;
                        let y = (token / width) % height;
                        let frame = token / (height * width);
                        if pair < 22 {
                            frame as f64 * 10000.0f64.powf(-((2 * pair) as f64) / 44.0)
                        } else if pair < 43 {
                            let axis_pair = pair - 22;
                            y as f64 * 10000.0f64.powf(-((2 * axis_pair) as f64) / 42.0)
                        } else {
                            let axis_pair = pair - 43;
                            x as f64 * 10000.0f64.powf(-((2 * axis_pair) as f64) / 42.0)
                        }
                    }
                };
                let (sin, cos) = angle.sin_cos();
                let first = values[base + 2 * pair];
                let second = values[base + 2 * pair + 1];
                values[base + 2 * pair] = first * cos as f32 - second * sin as f32;
                values[base + 2 * pair + 1] = first * sin as f32 + second * cos as f32;
            }
        }
    }
    Ok(())
}

fn video_temporal_positions(
    frames: usize,
    height: usize,
    width: usize,
    video_fps: usize,
) -> Result<Vec<f32>, String> {
    if frames == 0 || height == 0 || width == 0 || video_fps == 0 {
        return Err("Invalid DreamX Creator temporal grid".into());
    }
    let spatial = height * width;
    let scale = AUDIO_FPS / (video_fps as f32 / VAE_TEMPORAL_STRIDE);
    Ok((0..frames * spatial)
        .map(|token| (token / spatial) as f32 * scale)
        .collect())
}

#[derive(Clone, Copy)]
struct FlowStep {
    timestep: f32,
    delta_sigma: f32,
}

fn flow_match_schedule(steps: usize, shift: f32) -> Result<Vec<FlowStep>, String> {
    if steps == 0 || !shift.is_finite() || shift <= 0.0 {
        return Err("Invalid DreamX FlowMatch schedule".into());
    }
    let mut sigmas = Vec::with_capacity(steps + 1);
    for index in 0..steps {
        let fraction = if steps == 1 {
            0.0
        } else {
            index as f32 / (steps - 1) as f32
        };
        let sigma = 1.0 + fraction * (0.001 - 1.0);
        sigmas.push(shift * sigma / (1.0 + (shift - 1.0) * sigma));
    }
    sigmas.push(0.0);
    Ok((0..steps)
        .map(|index| FlowStep {
            timestep: sigmas[index] * 1000.0,
            delta_sigma: sigmas[index + 1] - sigmas[index],
        })
        .collect())
}

fn multimodal_cfg(
    no_bridge: &[f32],
    negative_bridge: &[f32],
    positive_bridge: &[f32],
    bridge_scale: f32,
    text_scale: f32,
) -> Result<Vec<f32>, String> {
    if no_bridge.len() != negative_bridge.len()
        || no_bridge.len() != positive_bridge.len()
        || !bridge_scale.is_finite()
        || !text_scale.is_finite()
    {
        return Err("Invalid DreamX multimodal CFG predictions".into());
    }
    Ok(no_bridge
        .iter()
        .zip(negative_bridge)
        .zip(positive_bridge)
        .map(|((&d00, &d0b), &dtb)| d00 + bridge_scale * (d0b - d00) + text_scale * (dtb - d0b))
        .collect())
}

fn euler_step(sample: &mut [f32], velocity: &[f32], delta_sigma: f32) -> Result<(), String> {
    if sample.len() != velocity.len() || !delta_sigma.is_finite() {
        return Err("Invalid DreamX FlowMatch Euler step".into());
    }
    for (sample, &velocity) in sample.iter_mut().zip(velocity) {
        *sample += delta_sigma * velocity;
    }
    Ok(())
}

fn restore_first_frame(
    latent: &mut [f32],
    shape: [usize; 4],
    first_frame: &[f32],
) -> Result<(), String> {
    let [channels, frames, height, width] = shape;
    let plane = height * width;
    if channels != VIDEO_CHANNELS
        || frames == 0
        || latent.len() != channels * frames * plane
        || first_frame.len() != channels * plane
    {
        return Err("Invalid DreamX first-frame restoration".into());
    }
    for channel in 0..channels {
        latent[channel * frames * plane..channel * frames * plane + plane]
            .copy_from_slice(&first_frame[channel * plane..(channel + 1) * plane]);
    }
    Ok(())
}

fn gaussian_values<R: Rng + ?Sized>(rng: &mut R, len: usize) -> Vec<f32> {
    (0..len)
        .map(|_| {
            let u1 = rng.gen::<f32>().max(1e-9);
            let u2 = rng.gen::<f32>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect()
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_block_uses_the_same_pre_update_snapshot_both_ways() {
        let (video, audio) = joint_update_from_snapshot(
            vec![1.0],
            vec![2.0],
            |_, audio| Ok(audio.to_vec()),
            |_, video| Ok(video.to_vec()),
        )
        .unwrap();
        assert_eq!(video, vec![3.0]);
        assert_eq!(audio, vec![3.0]);
    }

    #[test]
    fn first_frame_timestep_is_zero_at_every_step() {
        let times = video_token_timesteps(750.0, 4, 16).unwrap();
        assert_eq!(&times[..4], &[0.0; 4]);
        assert!(times[4..].iter().all(|&value| value == 750.0));
    }
}
