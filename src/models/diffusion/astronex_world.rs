//! Astronex-World (Wan2.2-TI2V-5B) video pipeline.
//!
//! This module intentionally has its own tensor namespace and model graph.
//! DreamX is a different audio/video model and must not be used as an
//! architecture fallback.  The graph keeps activations in F32 so the strict
//! `RMI_SCALAR=1` route can be used for the parity checkpoints; the released
//! BF16 weights are converted only at the weight boundary.

use std::sync::Arc;

use image::RgbImage;

use super::dreamx::kernels::{
    attention_scalar_f32, checked_len, conv3d_with_options, gelu_approx_inplace, load_float_values,
    AttentionSpec, Conv3dSpec, Linear,
};
use super::dreamx::video_vae::Wan22Vae;
use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;

const DIM: usize = 3072;
const FFN: usize = 14336;
const HEADS: usize = 24;
const HEAD_DIM: usize = DIM / HEADS;
const LAYERS: usize = 30;
const TEXT_DIM: usize = 4096;
const TEXT_LEN: usize = 512;
const FREQ_DIM: usize = 256;
const LATENT_CHANNELS: usize = 48;
const PATCH: [usize; 3] = [1, 2, 2];
const SOURCE_REVISION: &str = "dec4ec502f7113a1ab07b78ae40a7ac1b88b2eeb";
const ORACLE_COMMIT: &str = "27584bf1a89da01ef35a03b48d83bc818d2adf2b";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AstronexMode {
    TextToVideo,
    ImageToVideo,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AstronexOptions {
    pub mode: AstronexMode,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps: usize,
    pub steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
}

impl Default for AstronexOptions {
    fn default() -> Self {
        Self {
            mode: AstronexMode::TextToVideo,
            width: 832,
            height: 480,
            frames: 9,
            fps: 24,
            steps: 8,
            guidance_scale: 3.0,
            seed: 42,
        }
    }
}

impl AstronexOptions {
    pub fn validate(&self) -> Result<(), String> {
        if self.width == 0
            || self.height == 0
            || !self.width.is_multiple_of(16)
            || !self.height.is_multiple_of(16)
        {
            return Err("Astronex width and height must be positive multiples of 16".into());
        }
        if self.frames == 0 || (self.frames - 1) % 4 != 0 {
            return Err("Astronex frame count must be 1 plus a multiple of 4".into());
        }
        if self.fps == 0 || self.steps == 0 {
            return Err("Astronex fps and steps must be positive".into());
        }
        if self.steps != 8 {
            return Err(
                "Astronex currently supports the validated 8-step UniPC budget only".into(),
            );
        }
        if !self.guidance_scale.is_finite() || self.guidance_scale <= 0.0 {
            return Err("Astronex guidance scale must be finite and positive".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AstronexConfig {
    pub pair_id: String,
    pub source_revision: String,
    pub oracle_commit: String,
    pub layers: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub heads: usize,
    pub latent_channels: usize,
    pub text_len: usize,
}

impl AstronexConfig {
    pub fn from_sources(
        main: &dyn TensorSource,
        components: &dyn TensorSource,
    ) -> Result<Self, String> {
        let architecture = string_meta(main, "general.architecture")?;
        if architecture != "astronex_world" {
            return Err(format!(
                "Astronex main architecture mismatch: {architecture:?}"
            ));
        }
        let role = string_meta(main, "astronex.file_role")?;
        if role != "main" {
            return Err(format!("Astronex main file role mismatch: {role:?}"));
        }
        if string_meta(components, "astronex.file_role")? != "components" {
            return Err("Astronex component file role mismatch".into());
        }
        let pair_id = string_meta(main, "astronex.pair_id")?;
        if string_meta(components, "astronex.pair_id")? != pair_id {
            return Err("Astronex main/components pair_id mismatch".into());
        }
        let source_revision = string_meta(main, "astronex.source_revision")?;
        let oracle_commit = string_meta(main, "astronex.oracle_commit")?;
        if source_revision != SOURCE_REVISION {
            return Err(format!(
                "Astronex source revision mismatch: {source_revision:?}"
            ));
        }
        if oracle_commit != ORACLE_COMMIT {
            return Err(format!(
                "Astronex Oracle commit mismatch: {oracle_commit:?}"
            ));
        }
        let config = Self {
            pair_id,
            source_revision,
            oracle_commit,
            layers: LAYERS,
            hidden: DIM,
            ffn: FFN,
            heads: HEADS,
            latent_channels: LATENT_CHANNELS,
            text_len: TEXT_LEN,
        };
        validate_required_tensors(main, components)?;
        Ok(config)
    }
}

fn string_meta(source: &dyn TensorSource, key: &str) -> Result<String, String> {
    source
        .metadata(key)
        .and_then(MetaValue::to_string_val)
        .map(str::to_owned)
        .ok_or_else(|| format!("Astronex metadata {key:?} must be a string"))
}

fn validate_required_tensors(
    main: &dyn TensorSource,
    components: &dyn TensorSource,
) -> Result<(), String> {
    let required = [
        ("astronex.patch_embedding.weight", &[2, 2, 1, 48, 3072][..]),
        ("astronex.patch_embedding.bias", &[3072][..]),
        ("astronex.text_embedding.0.weight", &[4096, 3072][..]),
        ("astronex.text_embedding.2.weight", &[3072, 3072][..]),
        ("astronex.time_embedding.0.weight", &[256, 3072][..]),
        ("astronex.time_embedding.2.weight", &[3072, 3072][..]),
        ("astronex.time_projection.1.weight", &[3072, 18432][..]),
        ("astronex.head.head.weight", &[3072, 192][..]),
    ];
    for (name, dims) in required {
        let info = main
            .tensor_info(name)
            .ok_or_else(|| format!("Missing Astronex tensor {name}"))?;
        if info.dims != dims {
            return Err(format!(
                "Astronex tensor {name} has GGUF shape {:?}; expected {dims:?}",
                info.dims
            ));
        }
    }
    let token_embedding = components
        .tensor_info("astronex.text.token_embedding.weight")
        .ok_or("Missing Astronex UMT5 token embedding")?;
    if token_embedding.dims != [TEXT_DIM as u64, 256384] {
        return Err(format!(
            "Astronex token embedding shape {:?} is not [4096, 256384]",
            token_embedding.dims
        ));
    }
    if components
        .tensor_info("astronex.vae.conv1.weight")
        .is_none()
    {
        return Err("Missing Astronex Wan VAE tensors".into());
    }
    Ok(())
}

struct Block {
    norm3_weight: Vec<f32>,
    norm3_bias: Vec<f32>,
    modulation: Vec<f32>,
    self_q: Linear<'static>,
    self_k: Linear<'static>,
    self_v: Linear<'static>,
    self_o: Linear<'static>,
    self_norm_q: Vec<f32>,
    self_norm_k: Vec<f32>,
    cross_q: Linear<'static>,
    cross_k: Linear<'static>,
    cross_v: Linear<'static>,
    cross_o: Linear<'static>,
    cross_norm_q: Vec<f32>,
    cross_norm_k: Vec<f32>,
    ffn_up: Linear<'static>,
    ffn_down: Linear<'static>,
}

struct Denoiser {
    _source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    patch_weight: Vec<f32>,
    patch_bias: Vec<f32>,
    text_0: Linear<'static>,
    text_2: Linear<'static>,
    time_0: Linear<'static>,
    time_2: Linear<'static>,
    time_projection: Linear<'static>,
    blocks: Vec<Block>,
    head: Linear<'static>,
    head_modulation: Vec<f32>,
}

fn original_dims(dims: &[usize]) -> Vec<u64> {
    dims.iter().rev().map(|&v| v as u64).collect()
}

fn values(source: &dyn TensorSource, name: &str, dims: &[usize]) -> Result<Vec<f32>, String> {
    load_float_values(source, name, &original_dims(dims))
}

fn linear(
    source: &'static dyn TensorSource,
    prefix: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Linear<'static>, String> {
    Linear::from_source(
        source,
        &format!("{prefix}.weight"),
        Some(&format!("{prefix}.bias")),
        n_in,
        n_out,
    )
}

impl Denoiser {
    fn load(source: Arc<dyn TensorSource>, pool: Arc<ComputePool>) -> Result<Self, String> {
        let source_ref: &'static dyn TensorSource = unsafe { &*Arc::as_ptr(&source) };
        let patch_weight = values(
            source_ref,
            "astronex.patch_embedding.weight",
            &[DIM, LATENT_CHANNELS, 1, 2, 2],
        )?;
        let patch_bias = values(source_ref, "astronex.patch_embedding.bias", &[DIM])?;
        let text_0 = linear(source_ref, "astronex.text_embedding.0", TEXT_DIM, DIM)?;
        let text_2 = linear(source_ref, "astronex.text_embedding.2", DIM, DIM)?;
        let time_0 = linear(source_ref, "astronex.time_embedding.0", FREQ_DIM, DIM)?;
        let time_2 = linear(source_ref, "astronex.time_embedding.2", DIM, DIM)?;
        let time_projection = linear(source_ref, "astronex.time_projection.1", DIM, DIM * 6)?;
        let head = linear(source_ref, "astronex.head.head", DIM, LATENT_CHANNELS * 4)?;
        let head_modulation = values(source_ref, "astronex.head.modulation", &[1, 2, DIM])?;
        let mut blocks = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            let prefix = format!("astronex.blocks.{index}");
            blocks.push(Block {
                norm3_weight: values(source_ref, &format!("{prefix}.norm3.weight"), &[DIM])?,
                norm3_bias: values(source_ref, &format!("{prefix}.norm3.bias"), &[DIM])?,
                modulation: values(source_ref, &format!("{prefix}.modulation"), &[1, 6, DIM])?,
                self_q: linear(source_ref, &format!("{prefix}.self_attn.q"), DIM, DIM)?,
                self_k: linear(source_ref, &format!("{prefix}.self_attn.k"), DIM, DIM)?,
                self_v: linear(source_ref, &format!("{prefix}.self_attn.v"), DIM, DIM)?,
                self_o: linear(source_ref, &format!("{prefix}.self_attn.o"), DIM, DIM)?,
                self_norm_q: values(
                    source_ref,
                    &format!("{prefix}.self_attn.norm_q.weight"),
                    &[DIM],
                )?,
                self_norm_k: values(
                    source_ref,
                    &format!("{prefix}.self_attn.norm_k.weight"),
                    &[DIM],
                )?,
                cross_q: linear(source_ref, &format!("{prefix}.cross_attn.q"), DIM, DIM)?,
                cross_k: linear(source_ref, &format!("{prefix}.cross_attn.k"), DIM, DIM)?,
                cross_v: linear(source_ref, &format!("{prefix}.cross_attn.v"), DIM, DIM)?,
                cross_o: linear(source_ref, &format!("{prefix}.cross_attn.o"), DIM, DIM)?,
                cross_norm_q: values(
                    source_ref,
                    &format!("{prefix}.cross_attn.norm_q.weight"),
                    &[DIM],
                )?,
                cross_norm_k: values(
                    source_ref,
                    &format!("{prefix}.cross_attn.norm_k.weight"),
                    &[DIM],
                )?,
                ffn_up: linear(source_ref, &format!("{prefix}.ffn.0"), DIM, FFN)?,
                ffn_down: linear(source_ref, &format!("{prefix}.ffn.2"), FFN, DIM)?,
            });
        }
        Ok(Self {
            _source: source,
            pool,
            patch_weight,
            patch_bias,
            text_0,
            text_2,
            time_0,
            time_2,
            time_projection,
            blocks,
            head,
            head_modulation,
        })
    }

    fn patch(&self, latent: &[f32], shape: [usize; 4]) -> Result<(Vec<f32>, [usize; 3]), String> {
        let (data, out) = conv3d_with_options(
            &self.pool,
            latent,
            shape,
            &self.patch_weight,
            [DIM, LATENT_CHANNELS, 1, 2, 2],
            Some(&self.patch_bias),
            Conv3dSpec {
                stride: PATCH,
                ..Conv3dSpec::default()
            },
        )?;
        let [_, frames, height, width] = out;
        let tokens = frames * height * width;
        let mut hidden = vec![0.0; tokens * DIM];
        for f in 0..frames {
            for h in 0..height {
                for w in 0..width {
                    let row = (f * height + h) * width + w;
                    for c in 0..DIM {
                        hidden[row * DIM + c] =
                            data[(c * frames + f) * height * width + h * width + w];
                    }
                }
            }
        }
        Ok((hidden, [frames, height, width]))
    }

    fn embed_text(&self, pool: &ComputePool, context: &[f32]) -> Result<Vec<f32>, String> {
        if context.len() != TEXT_LEN * TEXT_DIM {
            return Err(format!(
                "Astronex context must have {} values",
                TEXT_LEN * TEXT_DIM
            ));
        }
        let mut hidden = self.text_0.forward(pool, context, TEXT_LEN)?;
        gelu_approx_inplace(&mut hidden);
        self.text_2.forward(pool, &hidden, TEXT_LEN)
    }

    fn embed_time(
        &self,
        pool: &ComputePool,
        timestep: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut sinusoid = vec![0.0; FREQ_DIM];
        let half = FREQ_DIM / 2;
        for i in 0..half {
            let frequency = 10000.0f64.powf(-(i as f64) / half as f64);
            let value = timestep as f64 * frequency;
            sinusoid[i] = value.cos() as f32;
            sinusoid[half + i] = value.sin() as f32;
        }
        let mut time = self.time_0.forward(pool, &sinusoid, 1)?;
        for value in &mut time {
            *value = *value / (1.0 + (-*value).exp());
        }
        let mut time = self.time_2.forward(pool, &time, 1)?;
        for value in &mut time {
            *value = *value / (1.0 + (-*value).exp());
        }
        let projected = self.time_projection.forward(pool, &time, 1)?;
        Ok((time, projected))
    }

    fn forward(
        &self,
        latent: &[f32],
        shape: [usize; 4],
        timestep: f32,
        context: &[f32],
        context_tokens: usize,
    ) -> Result<Vec<f32>, String> {
        let (mut x, grid) = self.patch(latent, shape)?;
        let tokens = x.len() / DIM;
        let context = self.embed_text(&self.pool, context)?;
        if context_tokens == 0 || context_tokens > TEXT_LEN {
            return Err("Astronex text context length is out of range".into());
        }
        // The official Wan T2V graph pads UMT5 to all 512 positions and calls
        // cross-attention without a key-length mask. Padding embeddings are
        // zeroed by the text encoder, but those zero keys still participate in
        // the softmax denominator. Keep the full context here for bit-level
        // graph parity; `context_tokens` remains a contract check.
        let context = &context[..TEXT_LEN * DIM];
        let (time, time_mod) = self.embed_time(&self.pool, timestep)?;
        let (frames, height, width) = (grid[0], grid[1], grid[2]);
        for block in &self.blocks {
            x = block.forward(&self.pool, x, tokens, grid, &time_mod, context, TEXT_LEN)?;
        }
        let mut modulated = x.clone();
        // Head::forward consumes the unprojected time embedding `e`, while
        // blocks consume the six-way `time_projection(e)` modulation.
        let hmod = &self.head_modulation[..DIM];
        let smod = &self.head_modulation[DIM..2 * DIM];
        let normed = norm_no_affine(&x, tokens, 1e-6)?;
        for row in 0..tokens {
            for c in 0..DIM {
                modulated[row * DIM + c] =
                    normed[row * DIM + c] * (1.0 + smod[c]) + hmod[c] + time[c];
            }
        }
        let output = self.head.forward(&self.pool, &modulated, tokens)?;
        let mut video = vec![0.0; checked_len("Astronex denoiser output", &shape)?];
        for f in 0..frames {
            for h in 0..height {
                for w in 0..width {
                    let row = (f * height + h) * width + w;
                    for pf in 0..PATCH[0] {
                        for ph in 0..PATCH[1] {
                            for pw in 0..PATCH[2] {
                                let patch_index = (pf * PATCH[1] + ph) * PATCH[2] + pw;
                                let ff = f * PATCH[0] + pf;
                                let hh = h * PATCH[1] + ph;
                                let ww = w * PATCH[2] + pw;
                                for channel in 0..LATENT_CHANNELS {
                                    video[(channel * shape[1] + ff) * shape[2] * shape[3]
                                        + hh * shape[3]
                                        + ww] = output[row * (LATENT_CHANNELS * 4)
                                        + patch_index * LATENT_CHANNELS
                                        + channel];
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(video)
    }
}

impl Block {
    fn forward(
        &self,
        pool: &ComputePool,
        mut x: Vec<f32>,
        tokens: usize,
        grid: [usize; 3],
        time: &[f32],
        context: &[f32],
        context_tokens: usize,
    ) -> Result<Vec<f32>, String> {
        let mut norm = norm_no_affine(&x, tokens, 1e-6)?;
        let mut e = self.modulation.clone();
        for i in 0..6 * DIM {
            e[i] += time[i];
        }
        let mut q_in = norm.clone();
        for row in 0..tokens {
            for c in 0..DIM {
                q_in[row * DIM + c] = norm[row * DIM + c] * (1.0 + e[DIM + c]) + e[c];
            }
        }
        let mut q = self.self_q.forward(pool, &q_in, tokens)?;
        let mut k = self.self_k.forward(pool, &q_in, tokens)?;
        let v = self.self_v.forward(pool, &q_in, tokens)?;
        rms_norm_in_rows(&mut q, tokens, &self.self_norm_q, 1e-6);
        rms_norm_in_rows(&mut k, tokens, &self.self_norm_k, 1e-6);
        apply_rope(&mut q, &mut k, grid);
        // The bidirectional Wan graph uses the complete clip as its key/value
        // set (`window_size=(-1, -1)`).  A local temporal window belongs to the
        // separate causal pipeline and silently changes the released graph.
        let attended = attention_scalar_f32(
            &q,
            &k,
            &v,
            AttentionSpec {
                query_tokens: tokens,
                key_tokens: tokens,
                query_heads: HEADS,
                key_value_heads: HEADS,
                head_dim: HEAD_DIM,
                causal: false,
                scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            },
        )?;
        let attn = self.self_o.forward(pool, &attended, tokens)?;
        for row in 0..tokens {
            for c in 0..DIM {
                x[row * DIM + c] += attn[row * DIM + c] * e[2 * DIM + c];
            }
        }

        norm = layer_norm_affine(&x, tokens, &self.norm3_weight, &self.norm3_bias, 1e-6)?;
        let q = self.cross_q.forward(pool, &norm, tokens)?;
        let mut k = self.cross_k.forward(pool, context, context_tokens)?;
        let v = self.cross_v.forward(pool, context, context_tokens)?;
        let mut q = q;
        rms_norm_in_rows(&mut q, tokens, &self.cross_norm_q, 1e-6);
        rms_norm_in_rows(&mut k, context_tokens, &self.cross_norm_k, 1e-6);
        let cross = attention_scalar_f32(
            &q,
            &k,
            &v,
            AttentionSpec {
                query_tokens: tokens,
                key_tokens: context_tokens,
                query_heads: HEADS,
                key_value_heads: HEADS,
                head_dim: HEAD_DIM,
                causal: false,
                scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            },
        )?;
        let cross = self.cross_o.forward(pool, &cross, tokens)?;
        for row in 0..tokens {
            for c in 0..DIM {
                x[row * DIM + c] += cross[row * DIM + c];
            }
        }

        norm = norm_no_affine(&x, tokens, 1e-6)?;
        for row in 0..tokens {
            for c in 0..DIM {
                norm[row * DIM + c] = norm[row * DIM + c] * (1.0 + e[4 * DIM + c]) + e[3 * DIM + c];
            }
        }
        let mut ff = self.ffn_up.forward(pool, &norm, tokens)?;
        gelu_approx_inplace(&mut ff);
        let ff = self.ffn_down.forward(pool, &ff, tokens)?;
        for row in 0..tokens {
            for c in 0..DIM {
                x[row * DIM + c] += ff[row * DIM + c] * e[5 * DIM + c];
            }
        }
        Ok(x)
    }
}

fn norm_no_affine(input: &[f32], rows: usize, eps: f32) -> Result<Vec<f32>, String> {
    if rows == 0 || input.len() != checked_len("Astronex LayerNorm input", &[rows, DIM])? {
        return Err("Invalid Astronex LayerNorm tensors".into());
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let slice = &input[row * DIM..(row + 1) * DIM];
        let mean = slice.iter().copied().sum::<f32>() / DIM as f32;
        let variance = slice
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f32>()
            / DIM as f32;
        let inv = 1.0 / (variance + eps).sqrt();
        for c in 0..DIM {
            output[row * DIM + c] = (slice[c] - mean) * inv;
        }
    }
    Ok(output)
}

fn layer_norm_affine(
    input: &[f32],
    rows: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Result<Vec<f32>, String> {
    let mut output = norm_no_affine(input, rows, eps)?;
    for row in 0..rows {
        for c in 0..DIM {
            output[row * DIM + c] = output[row * DIM + c] * weight[c] + bias[c];
        }
    }
    Ok(output)
}

fn rms_scalar(values: &[f32], eps: f32) -> f32 {
    (values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32 + eps).sqrt()
}

fn rms_norm_in_rows(values: &mut [f32], rows: usize, weight: &[f32], eps: f32) {
    for row in 0..rows {
        let start = row * DIM;
        let scale = 1.0 / rms_scalar(&values[start..start + DIM], eps);
        for c in 0..DIM {
            values[start + c] *= scale * weight[c];
        }
    }
}

fn apply_rope(q: &mut [f32], k: &mut [f32], grid: [usize; 3]) {
    let (frames, height, width) = (grid[0], grid[1], grid[2]);
    let splits = [22usize, 21, 21];
    for f in 0..frames {
        for h in 0..height {
            for w in 0..width {
                let token = (f * height + h) * width + w;
                let positions = [f, h, w];
                let mut offset = 0;
                for axis in 0..3 {
                    for i in 0..splits[axis] {
                        let dim = offset + i;
                        // Wan computes the complex RoPE multiplier in F64 and
                        // casts back to the activation dtype. Keep this
                        // separate from the intentionally approximate F32
                        // generation kernels; it is the only RoPE path used
                        // by the scalar parity trace.
                        let inv = 10000.0f64.powf(-(i as f64) / splits[axis] as f64);
                        let angle = positions[axis] as f64 * inv;
                        let (sin, cos) = angle.sin_cos();
                        for head in 0..HEADS {
                            let base = token * DIM + head * HEAD_DIM + 2 * dim;
                            for data in [&mut *q, &mut *k] {
                                let a = data[base] as f64;
                                let b = data[base + 1] as f64;
                                data[base] = (a * cos - b * sin) as f32;
                                data[base + 1] = (a * sin + b * cos) as f32;
                            }
                        }
                    }
                    offset += splits[axis];
                }
            }
        }
    }
}

pub struct AstronexPipeline {
    pub config: AstronexConfig,
    text: super::dreamx::text::DreamXTextEncoder,
    denoiser: Denoiser,
    vae: Wan22Vae,
}

/// Flow-matching UniPC (order 2) used by the pinned Astronex reference.
///
/// This is deliberately kept local to Astronex instead of pretending to be a
/// general diffusion scheduler: the model is trained for the eight-step
/// `predict_x0`/`bh2` contract below.  All tensor arithmetic stays F32, just
/// like the reference scheduler tensors.  The scheduler is not part of the
/// operator parity gate because platform `ln`/`exp_m1` implementations are
/// allowed to differ by a last bit.
struct FlowUniPc {
    sigmas: Vec<f32>,
    timesteps: Vec<u32>,
    step_index: usize,
    model_outputs: [Option<Vec<f32>>; 2],
    lower_order_nums: usize,
    this_order: usize,
    last_sample: Option<Vec<f32>>,
}

impl FlowUniPc {
    fn new(steps: usize, shift: f32) -> Result<Self, String> {
        if steps == 0 || !shift.is_finite() || shift <= 0.0 {
            return Err("Astronex UniPC requires positive finite steps and shift".into());
        }
        // The official scheduler constructs this linspace in F64, applies the
        // flow shift, then stores the sigma table as F32. Preserve that order
        // so timestep selection does not move at the boundary.
        let sigma_max = 1.0f64 - 1.0 / 1000.0;
        let mut sigmas = Vec::with_capacity(steps + 1);
        let mut timesteps = Vec::with_capacity(steps);
        for index in 0..steps {
            let base = sigma_max * (1.0 - index as f64 / steps as f64);
            let shifted = shift as f64 * base / (1.0 + (shift as f64 - 1.0) * base);
            sigmas.push(shifted as f32);
            timesteps.push((shifted * 1000.0) as u32);
        }
        sigmas.push(0.0);
        Ok(Self {
            sigmas,
            timesteps,
            step_index: 0,
            model_outputs: std::array::from_fn(|_| None),
            lower_order_nums: 0,
            this_order: 1,
            last_sample: None,
        })
    }

    fn step(
        &mut self,
        model_output: &[f32],
        timestep: u32,
        sample: &[f32],
    ) -> Result<Vec<f32>, String> {
        if self.step_index >= self.timesteps.len() {
            return Err("Astronex UniPC was stepped after completion".into());
        }
        if self.timesteps[self.step_index] != timestep {
            return Err(format!(
                "Astronex UniPC timestep mismatch: got {timestep}, expected {}",
                self.timesteps[self.step_index]
            ));
        }
        if model_output.is_empty() || model_output.len() != sample.len() {
            return Err("Astronex UniPC sample/model shape mismatch".into());
        }

        // The official scheduler converts the flow prediction before the
        // UniC corrector.  Passing the raw flow here changes both the history
        // residual and every later latent.
        let converted = self.convert_model_output(model_output, sample)?;
        let mut current = sample.to_vec();
        if self.step_index > 0 && self.last_sample.is_some() {
            current = self.corrector(&converted, &current)?;
        }

        self.model_outputs[0] = self.model_outputs[1].take();
        self.model_outputs[1] = Some(converted);
        let order = 2
            .min(self.timesteps.len() - self.step_index)
            .min(self.lower_order_nums + 1);
        self.this_order = order;
        self.last_sample = Some(current.clone());
        let next = self.predictor(&current, order)?;
        self.lower_order_nums = (self.lower_order_nums + 1).min(2);
        self.step_index += 1;
        if next.iter().any(|value| !value.is_finite()) {
            return Err("Astronex UniPC produced a non-finite latent".into());
        }
        Ok(next)
    }

    fn convert_model_output(
        &self,
        model_output: &[f32],
        sample: &[f32],
    ) -> Result<Vec<f32>, String> {
        let sigma = self.sigmas[self.step_index];
        let mut x0 = Vec::with_capacity(sample.len());
        for (&x, &flow) in sample.iter().zip(model_output) {
            x0.push(x - sigma * flow);
        }
        Ok(x0)
    }

    fn predictor(&self, sample: &[f32], order: usize) -> Result<Vec<f32>, String> {
        let index = self.step_index;
        let m0 = self.model_outputs[1]
            .as_deref()
            .ok_or("Astronex UniPC predictor has no current model output")?;
        let (alpha_t, sigma_t) = alpha_sigma(self.sigmas[index + 1]);
        let (alpha_s0, sigma_s0) = alpha_sigma(self.sigmas[index]);
        let h = lambda(alpha_t, sigma_t) - lambda(alpha_s0, sigma_s0);
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = hh.exp_m1();
        let mut residual = vec![0.0; sample.len()];
        if order == 2 {
            let previous = self.model_outputs[0]
                .as_deref()
                .ok_or("Astronex UniPC order-2 predictor has no history")?;
            let sigma_si = self.sigmas[index - 1];
            let (alpha_si, sigma_si) = alpha_sigma(sigma_si);
            let rk = (lambda(alpha_si, sigma_si) - lambda(alpha_s0, sigma_s0)) / h;
            if !rk.is_finite() || rk == 0.0 {
                return Err("Astronex UniPC predictor has a degenerate history ratio".into());
            }
            for ((value, &older), &current) in residual.iter_mut().zip(previous).zip(m0) {
                *value = 0.5 * (older - current) / rk;
            }
        }
        let mut out = Vec::with_capacity(sample.len());
        for ((&x, &current), &residual) in sample.iter().zip(m0).zip(&residual) {
            let base = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * current;
            out.push(base - alpha_t * b_h * residual);
        }
        Ok(out)
    }

    fn corrector(&self, model_output: &[f32], sample: &[f32]) -> Result<Vec<f32>, String> {
        let index = self.step_index;
        let previous_sample = self
            .last_sample
            .as_deref()
            .ok_or("Astronex UniPC corrector has no previous sample")?;
        let m0 = self.model_outputs[1]
            .as_deref()
            .ok_or("Astronex UniPC corrector has no previous model output")?;
        let (alpha_t, sigma_t) = alpha_sigma(self.sigmas[index]);
        let (alpha_s0, sigma_s0) = alpha_sigma(self.sigmas[index - 1]);
        let h = lambda(alpha_t, sigma_t) - lambda(alpha_s0, sigma_s0);
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = hh.exp_m1();
        let order = self.this_order;
        let mut residuals: Vec<Vec<f32>> = Vec::new();
        let rhos = if order == 1 {
            vec![0.5]
        } else {
            let previous = self.model_outputs[0]
                .as_deref()
                .ok_or("Astronex UniPC order-2 corrector has no history")?;
            let sigma_si = self.sigmas[index - 2];
            let (alpha_si, sigma_si) = alpha_sigma(sigma_si);
            let rk = (lambda(alpha_si, sigma_si) - lambda(alpha_s0, sigma_s0)) / h;
            if !rk.is_finite() || rk == 0.0 {
                return Err("Astronex UniPC corrector has a degenerate history ratio".into());
            }
            let h_phi_k = h_phi_1 / hh - 1.0;
            let h_phi_k2 = h_phi_k / hh - 0.5;
            let b0 = h_phi_k / b_h;
            let b1 = h_phi_k2 * 2.0 / b_h;
            let determinant = 1.0 - rk;
            if determinant == 0.0 || !determinant.is_finite() {
                return Err("Astronex UniPC corrector linear system is singular".into());
            }
            residuals.push(
                previous
                    .iter()
                    .zip(m0)
                    .map(|(&older, &current)| (older - current) / rk)
                    .collect(),
            );
            vec![(b0 - rk * b1) / determinant, (b1 - b0) / determinant]
        };
        let mut out = Vec::with_capacity(sample.len());
        for index in 0..sample.len() {
            let correction = residuals
                .first()
                .map_or(0.0, |values| rhos[0] * values[index])
                + rhos[order - 1] * (model_output[index] - m0[index]);
            let base = sigma_t / sigma_s0 * previous_sample[index] - alpha_t * h_phi_1 * m0[index];
            out.push(base - alpha_t * b_h * correction);
        }
        Ok(out)
    }
}

fn alpha_sigma(sigma: f32) -> (f32, f32) {
    (1.0 - sigma, sigma)
}

fn lambda(alpha: f32, sigma: f32) -> f32 {
    alpha.ln() - sigma.ln()
}

impl AstronexPipeline {
    pub fn load(
        main: Arc<dyn TensorSource>,
        components: Arc<dyn TensorSource>,
        n_threads: usize,
    ) -> Result<Self, String> {
        if n_threads == 0 {
            return Err("Astronex thread count must be non-zero".into());
        }
        let config = AstronexConfig::from_sources(main.as_ref(), components.as_ref())?;
        let pool = Arc::new(ComputePool::new(n_threads));
        let text = super::dreamx::text::DreamXTextEncoder::load_astronex(
            components.clone(),
            pool.clone(),
        )?;
        let denoiser = Denoiser::load(main.clone(), pool.clone())?;
        let vae = Wan22Vae::load_astronex(components.clone(), pool.clone())?;
        Ok(Self {
            config,
            text,
            denoiser,
            vae,
        })
    }

    pub fn generate(
        &self,
        prompt: &str,
        image: Option<&RgbImage>,
        options: &AstronexOptions,
    ) -> Result<Vec<RgbImage>, String> {
        options.validate()?;
        if options.mode == AstronexMode::ImageToVideo && image.is_none() {
            return Err("Astronex image-to-video requires an input image".into());
        }
        let latent_height = options.height / 16;
        let latent_width = options.width / 16;
        let latent_shape = [
            LATENT_CHANNELS,
            options.frames.div_ceil(4),
            latent_height,
            latent_width,
        ];
        let mut latent = gaussian(options.seed, checked_len("Astronex latent", &latent_shape)?)?;
        let mut reference = None;
        if let Some(frame) = image {
            let encoded = self.vae.encode_first_frame(frame)?;
            if encoded.shape() != [LATENT_CHANNELS, 1, latent_height, latent_width] {
                return Err("Astronex reference image latent shape mismatch".into());
            }
            for c in 0..LATENT_CHANNELS {
                for h in 0..latent_height {
                    for w in 0..latent_width {
                        latent[(c * latent_shape[1]) * latent_height * latent_width
                            + h * latent_width
                            + w] = encoded.as_slice()[(c * latent_height + h) * latent_width + w];
                    }
                }
            }
            reference = Some(encoded);
        }
        let positive = self.text.encode(prompt, "")?;
        let negative = self.text.encode("", "")?;
        let positive_len = positive.positive_len;
        let negative_len = negative.negative_len;
        let positive = positive.positive;
        let negative = negative.negative;
        let mut scheduler = FlowUniPc::new(options.steps, 5.0)?;
        for step in 0..options.steps {
            let timestep = scheduler.timesteps[step];
            let t = timestep as f32;
            let model_input = if let Some(ref frame) = reference {
                let mut input = latent.clone();
                let plane = latent_height * latent_width;
                for c in 0..LATENT_CHANNELS {
                    input[c * latent_shape[1] * plane..c * latent_shape[1] * plane + plane]
                        .copy_from_slice(&frame.as_slice()[c * plane..(c + 1) * plane]);
                }
                input
            } else {
                latent.clone()
            };
            let cond =
                self.denoiser
                    .forward(&model_input, latent_shape, t, &positive, positive_len)?;
            let uncond =
                self.denoiser
                    .forward(&model_input, latent_shape, t, &negative, negative_len)?;
            let mut flow = vec![0.0; latent.len()];
            for ((value, &unconditional), &conditional) in flow.iter_mut().zip(&uncond).zip(&cond) {
                *value = unconditional + options.guidance_scale * (conditional - unconditional);
            }
            latent = scheduler.step(&flow, timestep, &latent)?;
            if let Some(ref frame) = reference {
                let plane = latent_height * latent_width;
                for c in 0..LATENT_CHANNELS {
                    latent[c * latent_shape[1] * plane..c * latent_shape[1] * plane + plane]
                        .copy_from_slice(&frame.as_slice()[c * plane..(c + 1) * plane]);
                }
            }
        }
        let latent = super::dreamx::video_vae::VideoLatent::new(latent, latent_shape)?;
        self.vae.decode_frames(&latent)
    }
}

fn gaussian(seed: u64, len: usize) -> Result<Vec<f32>, String> {
    let mut state = seed.max(1);
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u1 = ((state >> 11) as f64 / ((1u64 << 53) as f64)).max(1e-12);
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u2 = (state >> 11) as f64 / ((1u64 << 53) as f64);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        out.push((radius * angle.cos()) as f32);
        if out.len() < len {
            out.push((radius * angle.sin()) as f32);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_reject_non_contract_frame_boundaries() {
        let mut options = AstronexOptions::default();
        options.frames = 10;
        assert!(options.validate().is_err());
    }

    #[test]
    fn rope_axes_cover_head_dimension() {
        assert_eq!((22 + 21 + 21) * 2, HEAD_DIM);
    }

    #[test]
    fn unipc_uses_official_eight_step_timesteps() {
        let scheduler = FlowUniPc::new(8, 5.0).unwrap();
        assert_eq!(
            scheduler.timesteps,
            [999, 972, 937, 892, 833, 749, 624, 416]
        );
        assert_eq!(scheduler.sigmas.last().copied(), Some(0.0));
    }
}
