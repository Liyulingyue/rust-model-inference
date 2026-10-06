//! ERNIE-Image / ERNIE-Image-Turbo DiT core.
//!
//! Reference: `references/stable-diffusion.cpp/src/model/diffusion/ernie_image.hpp`
//! (commit `3f8527a`). The forward loop is:
//!
//! 1. Conv2d `x_embedder.proj` (kernel=patch_size, stride=patch_size) over the
//!    latent to produce image tokens `[N, image_tokens, hidden]`.
//! 2. `text_proj` (3072 → hidden, no bias) on the text encoder output to
//!    produce `[N, text_tokens, hidden]`.
//! 3. Concat image + text tokens along axis=1 → `[N, total_tokens, hidden]`.
//! 4. Time embedding: `timestep_embedding_sin_cos(t)` → `time_embedding.mlp(c)`
//!    produces conditioning `c` of shape `[N, hidden]`.
//! 5. Shared AdaLN: `silu(c) @ adaLN_modulation.1.weight^T + adaLN_modulation.1.bias`
//!    produces `[N, 6*hidden]` which is chunked into 6 modulation tensors of
//!    shape `[N, 1, hidden]` each: `(shift_msa, scale_msa, gate_msa, shift_mlp,
//!    scale_mlp, gate_mlp)`.
//! 6. For each of [`NUM_LAYERS`]: AdaLN block forward (see
//!    [`ErnieImageBlock::forward`]). Each block reuses the same 6-way
//!    modulation tensors.
//! 7. `final_norm` is `AdaLNContinuous`: independent `norm + linear(c)`, where
//!    `c` is the conditioning from step 4. Produces `shift, scale`, applies
//!    `norm(x) * (1 + scale) + shift`.
//! 8. `final_linear(hidden) → [N, image_tokens, patch_size² * out_channels]`
//!    then `unpatchify` to `[N, out_channels, H, W]`.
//!
//! The whole step runs at flow-matching sigma in `[0, 1]`; the denoise loop
//! (see [`ErnieImageDit::denoise`]) is Euler flow-matching.

use half::f16;
use std::sync::Arc;

use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::rope::neox::rope_sin_cos;
use crate::ops::{rms_norm, rms_norm_inplace, silu_inplace};

use super::super::z_image::dit::{layer_norm_no_affine, TorchMt19937};
use super::{linear_into, Q8Scratch};

// === DiT architecture constants (ERNIE-Image, Baidu) ===

/// Channels in the latent (`vae.encode` output, also `vae.decode` input).
pub(crate) const LATENT_CHANNELS: usize = 128;

/// Channels out of `x_embedder.proj` (= main hidden size).
pub(crate) const HIDDEN: usize = 4096;

/// Heads per attention layer. `head_dim = HIDDEN / HEADS = 128`.
pub(crate) const HEADS: usize = 32;

pub(crate) const HEAD_DIM: usize = HIDDEN / HEADS;

/// Inner attention dim (`to_q/k/v` output channels).
pub(crate) const INNER_DIM: usize = HEADS * HEAD_DIM;

/// FFN hidden width (GELU(gate) * up, followed by linear_fc2).
pub(crate) const FFN_WIDTH: usize = 12_288;

pub(crate) const NUM_LAYERS: usize = 36;

/// Patch size for the latent → token grid. ERNIE-Image uses 1 (every latent
/// pixel is a token). Keep this in sync with `x_embedder.proj.weight`
/// tensor dimensions.
pub(crate) const PATCH_SIZE: usize = 1;
pub(crate) const PATCH_AREA: usize = PATCH_SIZE * PATCH_SIZE;

/// Channels in the latent input to the DiT. For `patch_size = 1`, the Conv2d
/// input dim is `IN_CHANNELS * PATCH_AREA`.
pub(crate) const IN_CHANNELS: usize = 128;

/// Channels out of the DiT (matches VAE input).
pub(crate) const OUT_CHANNELS: usize = 128;

/// RoPE theta (frequency base).
pub(crate) const ROPE_THETA: f32 = 256.0;

/// Per-axis RoPE dimensions: temporal / height / width.
pub(crate) const ROPE_AXES: [usize; 3] = [32, 48, 48];

/// First ROPE_AXES dimensions are rotated; remainder is pass-through. This must
/// equal HEAD_DIM, but ERNIE-Image's axes happen to sum exactly to it.
pub(crate) const ROPE_HEAD_WIDTH: usize = 128;

/// RMS epsilon for the per-block norms and Q/K norms.
pub(crate) const RMS_EPSILON: f32 = 1e-6;

/// === Text encoder constants (Ministral-3-3B-Instruct-2512) ===
pub(crate) const TEXT_IN_DIM: usize = 3072;
pub(crate) const TEXT_INNER_DIM: usize = TEXT_IN_DIM;
pub(crate) const TEXT_NUM_HEADS: usize = 32;
pub(crate) const TEXT_NUM_KV_HEADS: usize = 8;
pub(crate) const TEXT_HEAD_DIM: usize = 128;
pub(crate) const TEXT_FFN: usize = 9216;
pub(crate) const TEXT_NUM_LAYERS: usize = 26;
pub(crate) const TEXT_VOCAB: usize = 131_072;

/// Per-block tensor names and norm weights.
pub(crate) struct ErnieImageBlock {
    pub(crate) adaLN_sa_ln: Vec<f32>,
    pub(crate) adaLN_mlp_ln: Vec<f32>,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) to_q: String,
    pub(crate) to_k: String,
    pub(crate) to_v: String,
    pub(crate) to_out: String,
    pub(crate) gate_proj: String,
    pub(crate) up_proj: String,
    pub(crate) linear_fc2: String,
}

pub(crate) struct ErnieImageDit {
    source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    /// Shared 6-way AdaLN modulation weights: hidden → 6 * hidden.
    adaLN_modulation_weight: String,
    adaLN_modulation_bias: Vec<f32>,
    /// Time embedding MLP: linear_1 (hidden → hidden), linear_2 (hidden → hidden).
    time_linear_1_weight: String,
    time_linear_1_bias: Vec<f32>,
    time_linear_2_weight: String,
    time_linear_2_bias: Vec<f32>,
    /// x_embedder Conv2d (IN_CHANNELS*PATCH_AREA → hidden, kernel=PATCH_SIZE).
    x_embedder_weight: String,
    x_embedder_bias: Vec<f32>,
    /// text_proj (TEXT_IN_DIM → hidden, no bias) when text_in_dim != hidden.
    text_proj_weight: Option<String>,
    /// Final `AdaLNContinuous` (norm + linear(c) → scale, shift).
    final_norm_linear_weight: String,
    final_norm_linear_bias: Vec<f32>,
    /// Final linear: hidden → out_channels * patch_area.
    final_linear_weight: String,
    final_linear_bias: Vec<f32>,
    /// 36 main blocks.
    layers: Vec<ErnieImageBlock>,
}

impl ErnieImageDit {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let source_ref = source.as_ref();
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for layer in 0..NUM_LAYERS {
            layers.push(load_block(source_ref, layer)?);
        }
        let text_proj_weight = if source_ref.tensor_info("text_proj.weight").is_some() {
            Some("text_proj.weight".into())
        } else {
            None
        };
        let final_norm_linear_bias =
            load_f32_vector(source_ref, "final_norm.linear.bias", HIDDEN * 2)?;
        let final_linear_bias =
            load_f32_vector(source_ref, "final_linear.bias", OUT_CHANNELS * PATCH_AREA)?;
        let x_embedder_bias = load_f32_vector(source_ref, "x_embedder.proj.bias", HIDDEN)?;
        let adaLN_modulation_bias =
            load_f32_vector(source_ref, "adaLN_modulation.1.bias", HIDDEN * 6)?;
        let time_linear_1_bias =
            load_f32_vector(source_ref, "time_embedding.linear_1.bias", HIDDEN)?;
        let time_linear_2_bias =
            load_f32_vector(source_ref, "time_embedding.linear_2.bias", HIDDEN)?;
        Ok(Self {
            source,
            pool,
            adaLN_modulation_weight: "adaLN_modulation.1.weight".into(),
            adaLN_modulation_bias,
            time_linear_1_weight: "time_embedding.linear_1.weight".into(),
            time_linear_1_bias,
            time_linear_2_weight: "time_embedding.linear_2.weight".into(),
            time_linear_2_bias,
            x_embedder_weight: "x_embedder.proj.weight".into(),
            x_embedder_bias,
            text_proj_weight,
            final_norm_linear_weight: "final_norm.linear.weight".into(),
            final_norm_linear_bias,
            final_linear_weight: "final_linear.weight".into(),
            final_linear_bias,
            layers,
        })
    }

    /// Run the diffusion loop for `options.steps` Euler steps and return the
    /// final latent, using the reference discrete schedule with flow shift 4.
    pub(crate) fn denoise(
        &self,
        context: &[f32],
        context_tokens: usize,
        unconditional: Option<&[f32]>,
        options: &super::ErnieImageOptions,
    ) -> Result<Vec<f32>, String> {
        if context_tokens == 0 {
            return Err("ERNIE-Image context token count must be positive".into());
        }
        let latent_side = options.resolution / 16;
        if latent_side == 0 || latent_side % PATCH_SIZE != 0 {
            return Err("ERNIE-Image resolution must be a positive multiple of 16".into());
        }
        let latent_values = LATENT_CHANNELS * latent_side * latent_side;
        let mut latent = vec![0.0_f32; latent_values];
        // Seed the latent with N(0, 1) * sigma_max (sigma_max = 1).
        TorchMt19937::new(options.seed as u64).fill_normal_sd_cpp(&mut latent);
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "ernie_image.initial_latent",
            None,
            &[latent_side, latent_side, LATENT_CHANNELS],
            &latent,
        ));
        let steps = options.steps;
        let mut scratch = ErnieScratch::new(context_tokens, latent_side)?;
        let mut velocity = vec![0.0_f32; latent_values];
        let mut unconditional_velocity = vec![0.0_f32; latent_values];

        for step in 0..steps {
            let sigma = flow_sigma(step, steps);
            let sigma_next = flow_sigma(step + 1, steps);
            self.predict_flow(
                &mut latent,
                latent_side,
                context,
                context_tokens,
                sigma,
                &mut scratch,
                &mut velocity,
            )?;
            if let Some(unconditional) = unconditional {
                let tokens = super::context_token_count(unconditional)?;
                self.predict_flow(
                    &latent,
                    latent_side,
                    unconditional,
                    tokens,
                    sigma,
                    &mut scratch,
                    &mut unconditional_velocity,
                )?;
                for (conditional, unconditional) in velocity.iter_mut().zip(&unconditional_velocity)
                {
                    *conditional =
                        *unconditional + options.cfg_scale * (*conditional - *unconditional);
                }
            }
            euler_flow_step(&mut latent, &velocity, sigma, sigma_next)?;
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "ernie_image.sample",
                Some(step),
                &[latent_side, latent_side, LATENT_CHANNELS],
                &latent,
            ));
            eprintln!("[ernie-image] step {}/{}", step + 1, steps);
        }
        Ok(latent)
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_flow(
        &self,
        latent: &[f32],
        latent_side: usize,
        context: &[f32],
        context_tokens: usize,
        sigma: f32,
        scratch: &mut ErnieScratch,
        velocity: &mut [f32],
    ) -> Result<(), String> {
        if !sigma.is_finite() || !(0.0..=1.0).contains(&sigma) {
            return Err("ERNIE-Image sigma must be finite and within [0, 1]".into());
        }
        require_finite(latent, "latent")?;
        require_finite(context, "context")?;
        let latent_values = LATENT_CHANNELS * latent_side * latent_side;
        if latent.len() != latent_values {
            return Err("Invalid ERNIE-Image latent length".into());
        }

        // ERNIE-Image keeps the image tokens first and the text tokens last.
        // The reference graph accepts the exact joint length; adding zero
        // padding here would make those artificial tokens participate in
        // attention and change the model output.
        let image_token_count = latent_side * latent_side;
        let total_tokens = image_token_count
            .checked_add(context_tokens)
            .ok_or("ERNIE-Image sequence length overflow")?;
        scratch.prepare(total_tokens)?;

        // Time embedding: c ∈ [hidden]
        timestep_embedding(sigma * 1000.0, &mut scratch.time_frequency);
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "ernie_image.time_frequency",
            None,
            &[HIDDEN],
            &scratch.time_frequency,
        ));
        linear_into(
            self.source.as_ref(),
            &self.time_linear_1_weight,
            HIDDEN,
            HIDDEN,
            &scratch.time_frequency,
            &mut scratch.time_hidden,
            &mut scratch.q8,
            self.pool.as_ref(),
        )?;
        for (v, b) in scratch.time_hidden.iter_mut().zip(&self.time_linear_1_bias) {
            *v += *b;
        }
        silu_inplace(&mut scratch.time_hidden);
        linear_into(
            self.source.as_ref(),
            &self.time_linear_2_weight,
            HIDDEN,
            HIDDEN,
            &scratch.time_hidden,
            &mut scratch.time,
            &mut scratch.q8,
            self.pool.as_ref(),
        )?;
        for (v, b) in scratch.time.iter_mut().zip(&self.time_linear_2_bias) {
            *v += *b;
        }
        require_finite(&scratch.time, "time conditioning")?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "ernie_image.time",
            None,
            &[HIDDEN],
            &scratch.time,
        ));

        // Shared AdaLN: silu(c) @ W^T + b -> 6 * hidden, then chunk into 6.
        for v in scratch.time_hidden.iter_mut() {
            *v = 0.0;
        }
        // Reuse scratch.time as a scratch buffer for silu(c):
        for (dst, src) in scratch.time_hidden.iter_mut().zip(&scratch.time) {
            *dst = *src;
        }
        silu_inplace(&mut scratch.time_hidden);
        linear_into(
            self.source.as_ref(),
            &self.adaLN_modulation_weight,
            HIDDEN,
            6 * HIDDEN,
            &scratch.time_hidden,
            &mut scratch.modulation,
            &mut scratch.q8,
            self.pool.as_ref(),
        )?;
        for (v, b) in scratch
            .modulation
            .iter_mut()
            .zip(&self.adaLN_modulation_bias)
        {
            *v += *b;
        }
        // Split into 6 chunks of HIDDEN.
        let chunks = scratch.modulation.chunks_exact(HIDDEN);
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "ernie_image.modulation",
            None,
            &[6 * HIDDEN],
            &scratch.modulation,
        ));
        assert_eq!(chunks.len(), 6);
        let mod_shift_msa = &scratch.modulation[0..HIDDEN];
        let mod_scale_msa = &scratch.modulation[HIDDEN..2 * HIDDEN];
        let mod_gate_msa = &scratch.modulation[2 * HIDDEN..3 * HIDDEN];
        let mod_shift_mlp = &scratch.modulation[3 * HIDDEN..4 * HIDDEN];
        let mod_scale_mlp = &scratch.modulation[4 * HIDDEN..5 * HIDDEN];
        let mod_gate_mlp = &scratch.modulation[5 * HIDDEN..6 * HIDDEN];

        // x_embedder Conv2d: latent [N=1, IN_CHANNELS, H, W] -> tokens [N, image_tokens, hidden]
        // We write into a local buffer because `scratch.image` is sized for
        // the padded total sequence (incl. text tokens), not just image tokens.
        let image_token_count = latent_side * latent_side;
        let mut image_local = vec![0.0_f32; image_token_count * HIDDEN];
        run_x_embedder_into(
            self.source.as_ref(),
            &self.x_embedder_weight,
            &self.x_embedder_bias,
            latent,
            latent_side,
            &mut image_local,
            self.pool.as_ref(),
            &mut scratch.q8,
        )?;

        // text_proj: context [N, text_tokens, TEXT_IN_DIM] -> [N, text_tokens, hidden]
        if let Some(text_proj) = &self.text_proj_weight {
            let text_tokens = context_tokens;
            let mut projected = vec![0.0_f32; text_tokens * HIDDEN];
            for token in 0..text_tokens {
                let input = &context[token * TEXT_IN_DIM..(token + 1) * TEXT_IN_DIM];
                let output = &mut projected[token * HIDDEN..(token + 1) * HIDDEN];
                linear_into(
                    self.source.as_ref(),
                    text_proj,
                    TEXT_IN_DIM,
                    HIDDEN,
                    input,
                    output,
                    &mut scratch.q8,
                    self.pool.as_ref(),
                )?;
            }
            for (dst, src) in scratch
                .text
                .chunks_exact_mut(HIDDEN)
                .zip(projected.chunks_exact(HIDDEN))
            {
                dst.copy_from_slice(src);
            }
        } else {
            // text_in_dim == hidden: copy straight through.
            for (dst, src) in scratch
                .text
                .chunks_exact_mut(HIDDEN)
                .zip(context.chunks_exact(TEXT_IN_DIM))
            {
                dst.copy_from_slice(src);
            }
        }

        // Concat image + text tokens along axis=1 (image first, then text).
        for token in 0..image_token_count {
            scratch.joint[token * HIDDEN..(token + 1) * HIDDEN]
                .copy_from_slice(&image_local[token * HIDDEN..(token + 1) * HIDDEN]);
        }
        for token in 0..context_tokens {
            let dst = image_token_count + token;
            scratch.joint[dst * HIDDEN..(dst + 1) * HIDDEN]
                .copy_from_slice(&scratch.text[token * HIDDEN..(token + 1) * HIDDEN]);
        }

        // 3D RoPE cache.
        ernie_image_rope_into(context_tokens, latent_side, latent_side, &mut scratch.rope)?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "ernie_image.prelude",
            None,
            &[total_tokens, HIDDEN],
            &scratch.joint,
        ));

        // Forward the 36 blocks.
        for (layer_index, block) in self.layers.iter().enumerate() {
            run_block(
                self.source.as_ref(),
                block,
                &mut scratch.joint,
                total_tokens,
                &scratch.rope,
                mod_shift_msa,
                mod_scale_msa,
                mod_gate_msa,
                mod_shift_mlp,
                mod_scale_mlp,
                mod_gate_mlp,
                &mut scratch.qkv,
                &mut scratch.attention,
                &mut scratch.ffn,
                &mut scratch.ffn_up,
                &mut scratch.scores,
                &mut scratch.mlp_out,
                &mut scratch.q8,
                self.pool.as_ref(),
                layer_index,
            )?;
            require_finite(&scratch.joint, "ERNIE-Image block output")?;
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "ernie_image.block",
                Some(layer_index),
                &[total_tokens, HIDDEN],
                &scratch.joint,
            ));
        }

        // AdaLNContinuous final norm: scale, shift from c.
        let mut final_norm_out = vec![0.0_f32; total_tokens * HIDDEN];
        linear_into(
            self.source.as_ref(),
            &self.final_norm_linear_weight,
            HIDDEN,
            2 * HIDDEN,
            &scratch.time,
            &mut scratch.modulation[..2 * HIDDEN],
            &mut scratch.q8,
            self.pool.as_ref(),
        )?;
        for (v, b) in scratch.modulation[..2 * HIDDEN]
            .iter_mut()
            .zip(&self.final_norm_linear_bias)
        {
            *v += *b;
        }
        let final_scale = &scratch.modulation[..HIDDEN];
        let final_shift = &scratch.modulation[HIDDEN..2 * HIDDEN];
        for token in 0..total_tokens {
            let normalized = &mut scratch.attention[token * HIDDEN..(token + 1) * HIDDEN];
            let source = &scratch.joint[token * HIDDEN..(token + 1) * HIDDEN];
            layer_norm_no_affine(source, normalized, RMS_EPSILON)?;
            // modulate: norm * (1 + scale) + shift
            for ((v, s), sh) in normalized
                .iter_mut()
                .zip(&final_scale[..HIDDEN])
                .zip(&final_shift[..HIDDEN])
            {
                *v = (*v + *v * *s) + *sh;
            }
            final_norm_out[token * HIDDEN..(token + 1) * HIDDEN].copy_from_slice(normalized);
        }
        // Overwrite scratch.joint with the final normalized tensor.
        scratch.joint.copy_from_slice(&final_norm_out);

        // final_linear: hidden -> out_channels * patch_area, slice to image tokens.
        let mut patches = vec![0.0_f32; image_token_count * OUT_CHANNELS * PATCH_AREA];
        for token in 0..image_token_count {
            let input = &scratch.joint[token * HIDDEN..(token + 1) * HIDDEN];
            let output = &mut patches
                [token * OUT_CHANNELS * PATCH_AREA..(token + 1) * OUT_CHANNELS * PATCH_AREA];
            linear_into(
                self.source.as_ref(),
                &self.final_linear_weight,
                HIDDEN,
                OUT_CHANNELS * PATCH_AREA,
                input,
                output,
                &mut scratch.q8,
                self.pool.as_ref(),
            )?;
            for (v, b) in output.iter_mut().zip(&self.final_linear_bias) {
                *v += *b;
            }
        }

        // Unpatchify: rearrange [image_token_count, OUT_CHANNELS, PATCH_AREA] ->
        // [OUT_CHANNELS, latent_side, latent_side].
        for token in 0..image_token_count {
            let patch_y = token / latent_side;
            let patch_x = token % latent_side;
            for c in 0..OUT_CHANNELS {
                for p in 0..PATCH_AREA {
                    let py = p / PATCH_SIZE;
                    let px = p % PATCH_SIZE;
                    let y = patch_y * PATCH_SIZE + py;
                    let x = patch_x * PATCH_SIZE + px;
                    let dst = (c * latent_side + y) * latent_side + x;
                    let src = token * OUT_CHANNELS * PATCH_AREA + c * PATCH_AREA + p;
                    velocity[dst] = patches[src];
                }
            }
        }

        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "ernie_image.velocity",
            None,
            &[OUT_CHANNELS, latent_side, latent_side],
            velocity,
        ));
        Ok(())
    }
}

// === Helpers ===

fn load_f32_vector(source: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let expected_dims = [len as u64];
    if info.dims != expected_dims {
        return Err(format!("Invalid {name} dimensions"));
    }
    if !matches!(
        info.ggml_type,
        GGMLType::F32 | GGMLType::BF16 | GGMLType::F16
    ) {
        return Err(format!(
            "Invalid {name} type {:?}: expected F32 / F16 / BF16",
            info.ggml_type
        ));
    }
    let expected = info
        .checked_nbytes()
        .ok_or_else(|| format!("Invalid {name} byte size"))?;
    let expected = usize::try_from(expected)
        .map_err(|_| format!("Tensor byte size does not fit usize: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!("Invalid {name} byte length"));
    }
    let mut values = vec![0.0_f32; len];
    match info.ggml_type {
        GGMLType::F32 => {
            for (dst, chunk) in values.iter_mut().zip(bytes.chunks_exact(4)) {
                *dst = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        GGMLType::F16 | GGMLType::BF16 => {
            for (dst, chunk) in values.iter_mut().zip(bytes.chunks_exact(2)) {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                *dst = if info.ggml_type == GGMLType::BF16 {
                    half::bf16::from_bits(bits).to_f32()
                } else {
                    f16::from_bits(bits).to_f32()
                };
            }
        }
        _ => unreachable!(),
    }
    Ok(values)
}

fn load_block(source: &dyn TensorSource, layer: usize) -> Result<ErnieImageBlock, String> {
    let prefix = format!("layers.{layer}");
    Ok(ErnieImageBlock {
        adaLN_sa_ln: load_f32_vector(source, &format!("{prefix}.adaLN_sa_ln.weight"), HIDDEN)?,
        adaLN_mlp_ln: load_f32_vector(source, &format!("{prefix}.adaLN_mlp_ln.weight"), HIDDEN)?,
        q_norm: load_f32_vector(
            source,
            &format!("{prefix}.self_attention.norm_q.weight"),
            HEAD_DIM,
        )?,
        k_norm: load_f32_vector(
            source,
            &format!("{prefix}.self_attention.norm_k.weight"),
            HEAD_DIM,
        )?,
        to_q: format!("{prefix}.self_attention.to_q.weight"),
        to_k: format!("{prefix}.self_attention.to_k.weight"),
        to_v: format!("{prefix}.self_attention.to_v.weight"),
        to_out: format!("{prefix}.self_attention.to_out.0.weight"),
        gate_proj: format!("{prefix}.mlp.gate_proj.weight"),
        up_proj: format!("{prefix}.mlp.up_proj.weight"),
        linear_fc2: format!("{prefix}.mlp.linear_fc2.weight"),
    })
}

fn require_finite(values: &[f32], name: &str) -> Result<(), String> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(format!("Non-finite {name}"))
    }
}

fn flow_sigma(step: usize, steps: usize) -> f32 {
    if step == steps {
        return 0.;
    }
    let t = if steps == 1 {
        999.
    } else {
        999. - (999. / (steps - 1) as f32) * step as f32
    };
    let t = (t + 1.) / 1000.;
    4. * t / (1. + 3. * t)
}

/// Preserve the reference's denoised -> derivative -> Euler floating-point order.
pub(crate) fn euler_flow_step(
    latent: &mut [f32],
    velocity: &[f32],
    sigma: f32,
    sigma_next: f32,
) -> Result<(), String> {
    if latent.len() != velocity.len() {
        return Err("Invalid ERNIE-Image Euler buffer lengths".into());
    }
    if !sigma.is_finite()
        || !sigma_next.is_finite()
        || sigma <= 0.
        || sigma > 1.
        || sigma_next < 0.
        || sigma_next > sigma
    {
        return Err("Invalid ERNIE-Image Euler sigma interval".into());
    }
    let step = sigma_next - sigma;
    for (x, v) in latent.iter_mut().zip(velocity) {
        let denoised = *v * -sigma + *x;
        let derivative = (*x - denoised) / sigma;
        *x += derivative * step;
    }
    Ok(())
}

/// Generate the sinusoidal timestep embedding coefficients (length = hidden)
/// for a single scalar timestep.
fn timestep_embedding(t: f32, out: &mut [f32; HIDDEN]) {
    let half = HIDDEN / 2;
    let log_theta = (1e4_f32).ln();
    for i in 0..half {
        let freq_exp = (i as f32) / half as f32;
        let omega = (freq_exp * -log_theta).exp();
        let angle = t * omega;
        let (cosine, sine) = rope_sin_cos(angle);
        // ggml_timestep_embedding emits [sin(freq), cos(freq)] halves and
        // ERNIE's helper keeps that ordering before the MLP.
        out[i] = sine;
        out[half + i] = cosine;
    }
}

/// Per-call scratch buffers reused across timesteps.
pub(crate) struct ErnieScratch {
    /// Hidden-dim time embedding coefficients.
    time_frequency: [f32; HIDDEN],
    time_hidden: [f32; HIDDEN],
    /// Final conditioning `c` of shape `[HIDDEN]`.
    time: [f32; HIDDEN],
    /// Image tokens `[image_tokens, HIDDEN]`.
    image: Vec<f32>,
    /// Text tokens projected to `[text_tokens, HIDDEN]` (caller passes
    /// `context_tokens` for sizing).
    text: Vec<f32>,
    /// Joint sequence `[total_tokens, HIDDEN]`.
    joint: Vec<f32>,
    /// Per-layer QKV cache `[total_tokens, INNER_DIM]` (Q + K + V concatenated).
    qkv: Vec<f32>,
    /// Attention output `[total_tokens, HIDDEN]`.
    attention: Vec<f32>,
    /// FFN intermediate `[total_tokens, FFN_WIDTH]` (gelu(gate) * up).
    ffn: Vec<f32>,
    /// Up-projection output `[total_tokens, FFN_WIDTH]` (separate from
    /// `ffn` so we can hold both gate and up before multiplying).
    ffn_up: Vec<f32>,
    /// MLP output `[total_tokens, HIDDEN]` before residual add.
    mlp_out: Vec<f32>,
    /// Attention score buffer `[total_tokens]`.
    scores: Vec<f32>,
    /// Modulation chunks `[6 * HIDDEN]`.
    modulation: Vec<f32>,
    /// RoPE cache `[total_tokens, ROPE_HEAD_WIDTH]` (interleaved cos, sin).
    rope: Vec<f32>,
    /// Per-tensor Q8 staging, plus per-thread F16 staging.
    q8: Q8Scratch,
}

impl ErnieScratch {
    fn new(context_tokens: usize, latent_side: usize) -> Result<Self, String> {
        if context_tokens == 0 {
            return Err("ERNIE-Image scratch: context tokens must be positive".into());
        }
        let _ = latent_side;
        Ok(Self {
            time_frequency: [0.0; HIDDEN],
            time_hidden: [0.0; HIDDEN],
            time: [0.0; HIDDEN],
            image: Vec::new(),
            text: Vec::new(),
            joint: Vec::new(),
            qkv: Vec::new(),
            attention: Vec::new(),
            ffn: Vec::new(),
            ffn_up: Vec::new(),
            mlp_out: Vec::new(),
            scores: Vec::new(),
            modulation: Vec::new(),
            rope: Vec::new(),
            q8: Q8Scratch::new(FFN_WIDTH.max(HIDDEN)),
        })
    }

    fn prepare(&mut self, total_tokens: usize) -> Result<(), String> {
        resize_zeroed(
            &mut self.image,
            total_tokens * HIDDEN,
            "ERNIE-Image image tokens",
        )?;
        resize_zeroed(
            &mut self.text,
            total_tokens * HIDDEN,
            "ERNIE-Image text tokens",
        )?;
        resize_zeroed(
            &mut self.joint,
            total_tokens * HIDDEN,
            "ERNIE-Image joint tokens",
        )?;
        resize_zeroed(
            &mut self.qkv,
            total_tokens * 3 * INNER_DIM,
            "ERNIE-Image QKV",
        )?;
        resize_zeroed(
            &mut self.attention,
            total_tokens * HIDDEN,
            "ERNIE-Image attention",
        )?;
        resize_zeroed(&mut self.ffn, total_tokens * FFN_WIDTH, "ERNIE-Image FFN")?;
        resize_zeroed(
            &mut self.ffn_up,
            total_tokens * FFN_WIDTH,
            "ERNIE-Image FFN up",
        )?;
        resize_zeroed(&mut self.mlp_out, total_tokens * HIDDEN, "ERNIE-Image MLP")?;
        resize_zeroed(&mut self.scores, total_tokens, "ERNIE-Image scores")?;
        resize_zeroed(&mut self.modulation, HIDDEN * 6, "ERNIE-Image modulation")?;
        resize_zeroed(
            &mut self.rope,
            total_tokens * ROPE_HEAD_WIDTH,
            "ERNIE-Image RoPE",
        )?;
        Ok(())
    }
}

fn resize_zeroed(dst: &mut Vec<f32>, len: usize, name: &str) -> Result<(), String> {
    dst.clear();
    dst.try_reserve_exact(len)
        .map_err(|e| format!("Failed to allocate {name}: {e}"))?;
    dst.resize(len, 0.0);
    Ok(())
}

/// Compute the x_embedder Conv2d: a 1×1 conv over `IN_CHANNELS = 128` channels
/// at the same resolution as the latent. With `PATCH_SIZE = 1`, every spatial
/// position maps to one token, and the output is `[IN_CHANNELS, hidden]` per
/// pixel — i.e. `[image_tokens, hidden]` per sample.
#[allow(clippy::too_many_arguments)]
fn run_x_embedder_into(
    source: &dyn TensorSource,
    weight: &str,
    bias: &[f32],
    latent: &[f32],
    latent_side: usize,
    output: &mut [f32],
    pool: &ComputePool,
    q8: &mut Q8Scratch,
) -> Result<(), String> {
    let image_tokens = latent_side * latent_side;
    if output.len() != image_tokens * HIDDEN {
        return Err("ERNIE-Image x_embedder output length mismatch".into());
    }
    let info = source
        .tensor_info(weight)
        .ok_or_else(|| format!("Missing tensor: {weight}"))?;
    let bytes = source
        .tensor_slice(weight)
        .ok_or_else(|| format!("Missing tensor data: {weight}"))?;
    // Oracle Conv2d loads its weights as F16, even from BF16 storage.
    let converted: Vec<u8> = if info.ggml_type == GGMLType::BF16 {
        bytes
            .chunks_exact(2)
            .flat_map(|b| {
                let value =
                    half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32();
                crate::ops::f32_to_f16(value).to_le_bytes()
            })
            .collect()
    } else {
        Vec::new()
    };
    let f16_bytes = if converted.is_empty() {
        bytes
    } else {
        &converted
    };
    let mut prepared = Vec::new();
    for token in 0..image_tokens {
        let mut input = [0.; IN_CHANNELS];
        for c in 0..IN_CHANNELS {
            input[c] = latent[c * image_tokens + token];
        }
        let out = &mut output[token * HIDDEN..(token + 1) * HIDDEN];
        if info.dims.len() == 4 {
            if !matches!(info.ggml_type, GGMLType::F16 | GGMLType::BF16) {
                return Err(format!("Unsupported x_embedder type {:?}", info.ggml_type));
            }
            crate::ops::kernel::f16::F16Kernel::new(f16_bytes).forward_scaled(
                &input,
                out,
                IN_CHANNELS,
                HIDDEN,
                1.,
                &mut prepared,
            );
        } else {
            linear_into(source, weight, IN_CHANNELS, HIDDEN, &input, out, q8, pool)?;
        }
        for (v, b) in out.iter_mut().zip(bias) {
            *v += *b;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_block(
    source: &dyn TensorSource,
    block: &ErnieImageBlock,
    tokens: &mut [f32],
    total_tokens: usize,
    rope: &[f32],
    shift_msa: &[f32],
    scale_msa: &[f32],
    gate_msa: &[f32],
    shift_mlp: &[f32],
    scale_mlp: &[f32],
    gate_mlp: &[f32],
    qkv: &mut [f32],
    attention: &mut [f32],
    ffn_buf: &mut [f32],
    ffn_up: &mut [f32],
    scores: &mut [f32],
    mlp_out: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
    layer_index: usize,
) -> Result<(), String> {
    // 1. attention branch
    for token in 0..total_tokens {
        modulated_rms_norm(
            &tokens[token * HIDDEN..(token + 1) * HIDDEN],
            &block.adaLN_sa_ln,
            scale_msa,
            shift_msa,
            &mut attention[token * HIDDEN..(token + 1) * HIDDEN],
        );
    }

    #[cfg(feature = "parity-trace")]
    if layer_index == 0 {
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "ernie_image.norm",
            None,
            &[total_tokens, HIDDEN],
            attention,
        ));
    }
    // 2. to_q / to_k / to_v: three matmuls per row
    for token in 0..total_tokens {
        let input = &attention[token * HIDDEN..(token + 1) * HIDDEN];
        let row_qkv = &mut qkv[token * 3 * INNER_DIM..(token + 1) * 3 * INNER_DIM];
        linear_into(
            source,
            &block.to_q,
            HIDDEN,
            INNER_DIM,
            input,
            &mut row_qkv[0..INNER_DIM],
            q8,
            pool,
        )?;
        linear_into(
            source,
            &block.to_k,
            HIDDEN,
            INNER_DIM,
            input,
            &mut row_qkv[INNER_DIM..2 * INNER_DIM],
            q8,
            pool,
        )?;
        linear_into(
            source,
            &block.to_v,
            HIDDEN,
            INNER_DIM,
            input,
            &mut row_qkv[2 * INNER_DIM..3 * INNER_DIM],
            q8,
            pool,
        )?;
    }

    // 3. Q/K RMS norm + RoPE
    for token in 0..total_tokens {
        let rotation = &rope[token * ROPE_HEAD_WIDTH..(token + 1) * ROPE_HEAD_WIDTH];
        let row_qkv = &mut qkv[token * 3 * INNER_DIM..(token + 1) * 3 * INNER_DIM];
        for head in 0..HEADS {
            let start = head * HEAD_DIM;
            let query = &mut row_qkv[start..start + HEAD_DIM];
            rms_norm_inplace(query, &block.q_norm, RMS_EPSILON);
            rotate_neox_inplace(query, rotation)?;

            let key_start = INNER_DIM + start;
            let key = &mut row_qkv[key_start..key_start + HEAD_DIM];
            rms_norm_inplace(key, &block.k_norm, RMS_EPSILON);
            rotate_neox_inplace(key, rotation)?;
        }
    }

    #[cfg(feature = "parity-trace")]
    if layer_index == 0 {
        for (component, name) in ["ernie_image.q_rot", "ernie_image.k_rot", "ernie_image.v"]
            .iter()
            .enumerate()
        {
            if crate::parity_trace::enabled(name) {
                let values: Vec<f32> = qkv
                    .chunks_exact(3 * INNER_DIM)
                    .flat_map(|row| {
                        row[component * INNER_DIM..(component + 1) * INNER_DIM]
                            .iter()
                            .copied()
                    })
                    .collect();
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    name,
                    None,
                    &[total_tokens, INNER_DIM],
                    &values,
                ));
            }
        }
    }
    // Pack V columns once so attention uses the shared F32 dot contract.
    for d in 0..HIDDEN {
        for token in 0..total_tokens {
            ffn_buf[d * total_tokens + token] = qkv[token * 3 * INNER_DIM + 2 * INNER_DIM + d];
        }
    }
    let scale_attn = 1.0 / (HEAD_DIM as f32).sqrt();
    for head in 0..HEADS {
        let head_offset = head * HEAD_DIM;
        for query_idx in 0..total_tokens {
            let q_offset = query_idx * 3 * INNER_DIM + head_offset;
            let q = &qkv[q_offset..q_offset + HEAD_DIM];
            for key_idx in 0..total_tokens {
                let k_offset = key_idx * 3 * INNER_DIM + INNER_DIM + head_offset;
                scores[key_idx] =
                    crate::ops::dot_f32(q, &qkv[k_offset..k_offset + HEAD_DIM], HEAD_DIM)
                        * scale_attn;
            }
            crate::ops::softmax_inplace(&mut scores[..total_tokens]);
            for d in 0..HEAD_DIM {
                let column = (head_offset + d) * total_tokens;
                attention[query_idx * HIDDEN + head_offset + d] = crate::ops::dot_f32(
                    &scores[..total_tokens],
                    &ffn_buf[column..column + total_tokens],
                    total_tokens,
                );
            }
        }
    }

    #[cfg(feature = "parity-trace")]
    if layer_index == 0 {
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "ernie_image.attention_values",
            None,
            &[total_tokens, HIDDEN],
            attention,
        ));
    }
    // 5. to_out + residual + gate
    for token in 0..total_tokens {
        let input = &attention[token * HIDDEN..(token + 1) * HIDDEN];
        let out = &mut ffn_buf[token * HIDDEN..(token + 1) * HIDDEN];
        linear_into(
            source,
            &block.to_out,
            INNER_DIM,
            HIDDEN,
            input,
            out,
            q8,
            pool,
        )?;
        for ((token_v, proj_v), g) in tokens[token * HIDDEN..(token + 1) * HIDDEN]
            .iter_mut()
            .zip(out.iter())
            .zip(gate_msa)
        {
            *token_v += proj_v * *g;
        }
    }

    #[cfg(feature = "parity-trace")]
    if layer_index == 0 {
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "ernie_image.attn_residual",
            None,
            &[total_tokens, HIDDEN],
            tokens,
        ));
    }
    // 6. MLP branch
    for token in 0..total_tokens {
        let token_slice = &tokens[token * HIDDEN..(token + 1) * HIDDEN];
        let token_mlp = &mut mlp_out[token * HIDDEN..(token + 1) * HIDDEN];
        modulated_rms_norm(
            token_slice,
            &block.adaLN_mlp_ln,
            scale_mlp,
            shift_mlp,
            token_mlp,
        );
        // gate_proj → ffn_buf (gate), up_proj → ffn_up (up). Both buffers
        // are sized `total_tokens * FFN_WIDTH` and disjoint.
        linear_into(
            source,
            &block.gate_proj,
            HIDDEN,
            FFN_WIDTH,
            &mlp_out[token * HIDDEN..(token + 1) * HIDDEN],
            &mut ffn_buf[token * FFN_WIDTH..(token + 1) * FFN_WIDTH],
            q8,
            pool,
        )?;
        linear_into(
            source,
            &block.up_proj,
            HIDDEN,
            FFN_WIDTH,
            &mlp_out[token * HIDDEN..(token + 1) * HIDDEN],
            &mut ffn_up[token * FFN_WIDTH..(token + 1) * FFN_WIDTH],
            q8,
            pool,
        )?;
        // gelu(gate) * up -> linear_fc2. Result lands in `mlp_out`.
        let gate = &mut ffn_buf[token * FFN_WIDTH..(token + 1) * FFN_WIDTH];
        for g in gate.iter_mut() {
            *g = crate::ops::gelu_ggml_f16(*g);
        }
        for (g, u) in gate
            .iter_mut()
            .zip(ffn_up[token * FFN_WIDTH..(token + 1) * FFN_WIDTH].iter())
        {
            *g *= *u;
        }
        linear_into(
            source,
            &block.linear_fc2,
            FFN_WIDTH,
            HIDDEN,
            gate,
            &mut mlp_out[token * HIDDEN..(token + 1) * HIDDEN],
            q8,
            pool,
        )?;
        // residual add with gate_mlp
        for ((token_v, mlp_v), g) in tokens[token * HIDDEN..(token + 1) * HIDDEN]
            .iter_mut()
            .zip(mlp_out[token * HIDDEN..(token + 1) * HIDDEN].iter())
            .zip(gate_mlp)
        {
            *token_v += mlp_v * *g;
        }
    }

    Ok(())
}

fn modulated_rms_norm(
    input: &[f32],
    weight: &[f32],
    scale: &[f32],
    shift: &[f32],
    output: &mut [f32],
) {
    rms_norm(input, weight, output, RMS_EPSILON);
    for ((value, scale), shift) in output.iter_mut().zip(scale).zip(shift) {
        *value = (*value + *value * *scale) + *shift;
    }
}

fn rotate_neox_inplace(values: &mut [f32], rope: &[f32]) -> Result<(), String> {
    if values.len() != rope.len() || values.len() % 2 != 0 {
        return Err("Invalid ERNIE-Image rotary buffers".into());
    }
    let (first, second) = values.split_at_mut(values.len() / 2);
    let half = first.len();
    for (i, (first, second)) in first.iter_mut().zip(second).enumerate() {
        let x0 = *first;
        let x1 = *second;
        // The reference repeats each frequency in adjacent lanes, while
        // rotate_half pairs the first and second halves of the head.
        let a = (i / 2) * 2;
        let b = ((i + half) / 2) * 2;
        *first = x0 * rope[a] + (-x1) * rope[a + 1];
        *second = x1 * rope[b] + x0 * rope[b + 1];
    }
    Ok(())
}

pub(crate) fn ernie_image_rope(
    text_tokens: usize,
    image_width: usize,
    image_height: usize,
) -> Result<Vec<f32>, String> {
    let mut output = Vec::new();
    ernie_image_rope_into(text_tokens, image_width, image_height, &mut output)?;
    Ok(output)
}

fn ernie_image_rope_into(
    text_tokens: usize,
    image_width: usize,
    image_height: usize,
    output: &mut Vec<f32>,
) -> Result<(), String> {
    if text_tokens == 0 || image_width == 0 || image_height == 0 {
        return Err("ERNIE-Image RoPE dimensions must be positive".into());
    }
    let axes_sum = ROPE_AXES.iter().try_fold(0usize, |sum, axis| {
        if axis % 2 != 0 {
            return Err("ERNIE-Image RoPE axes must be even".to_string());
        }
        sum.checked_add(*axis)
            .ok_or_else(|| "ERNIE-Image RoPE head width overflow".into())
    })?;
    if axes_sum != ROPE_HEAD_WIDTH {
        return Err("ERNIE-Image RoPE axes must match the attention head width".into());
    }

    let patch_width = image_width / PATCH_SIZE;
    let patch_height = image_height / PATCH_SIZE;
    let image_tokens = patch_width
        .checked_mul(patch_height)
        .ok_or("ERNIE-Image image token count overflow")?;
    let position_count = image_tokens
        .checked_add(text_tokens)
        .ok_or("ERNIE-Image position count overflow")?;
    let output_len = position_count
        .checked_mul(ROPE_HEAD_WIDTH)
        .ok_or("ERNIE-Image RoPE output size overflow")?;
    resize_zeroed(output, output_len, "ERNIE-Image RoPE")?;

    for position_index in 0..position_count {
        let positions = if position_index < image_tokens {
            [
                text_tokens as f32,
                (position_index / patch_width) as f32,
                (position_index % patch_width) as f32,
            ]
        } else {
            [(position_index - image_tokens) as f32, 0.0, 0.0]
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
                let (cosine, sine) = rope_sin_cos(angle);
                output[output_index] = cosine;
                output[output_index + 1] = sine;
                output_index += 2;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the real GGUF and pinned Oracle F32 fixtures"]
    fn oracle_flow_fixture() {
        let model = std::env::var("RMI_ERNIE_IMAGE_DIT_GGUF").unwrap();
        let fixtures = std::env::var("RMI_ERNIE_ORACLE_FIXTURES").unwrap();
        let read = |name: &str| {
            let bytes = std::fs::read(format!("{fixtures}/{name}.f32")).unwrap();
            assert_eq!(bytes.len() % 4, 0);
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let latent = read("rmi.ernie.input");
        let context = read("rmi.ernie.context");
        let side = ((latent.len() / LATENT_CHANNELS) as f64).sqrt() as usize;
        assert_eq!(latent.len(), LATENT_CHANNELS * side * side);
        let tokens = super::super::context_token_count(&context).unwrap();
        let source = crate::format::ggufrs::open_model_source(
            std::path::Path::new(&model),
            crate::format::ggufrs::ComponentRole::Llm,
        )
        .unwrap();
        let dit = ErnieImageDit::load(Arc::from(source), Arc::new(ComputePool::new(1))).unwrap();
        let mut scratch = ErnieScratch::new(tokens, side).unwrap();
        let mut velocity = vec![0.; latent.len()];
        dit.predict_flow(
            &latent,
            side,
            &context,
            tokens,
            1.,
            &mut scratch,
            &mut velocity,
        )
        .unwrap();
        let expected = read("rmi.ernie.velocity");
        assert_eq!(velocity.len(), expected.len());
        let mismatch = velocity
            .iter()
            .zip(&expected)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert_eq!(mismatch, None, "first raw-bit velocity difference");
    }

    #[test]
    fn discrete_schedule_uses_shift_four_and_appends_zero() {
        assert_eq!(flow_sigma(0, 2), 1.);
        assert_eq!(
            flow_sigma(1, 2).to_bits(),
            (4f32 * 0.001 / (1. + 3. * 0.001)).to_bits()
        );
        assert_eq!(flow_sigma(2, 2), 0.);
        assert_eq!([flow_sigma(0, 1), flow_sigma(1, 1)], [1., 0.]);
    }

    #[test]
    fn euler_integrates_velocity_instead_of_treating_it_as_denoised() {
        let mut latent = [4.];
        euler_flow_step(&mut latent, &[1.], 1., 0.).unwrap();
        assert_eq!(latent, [3.]);
        assert!(euler_flow_step(&mut latent, &[1.], 0., 0.).is_err());
    }

    #[test]
    fn timestep_zero_is_sine_then_cosine_halves() {
        let mut output = [0.; HIDDEN];
        timestep_embedding(0., &mut output);
        assert!(output[..HIDDEN / 2].iter().all(|v| *v == 0.));
        assert!(output[HIDDEN / 2..].iter().all(|v| *v == 1.));
    }

    #[test]
    fn rope_uses_exact_image_then_text_sequence() {
        let rope = ernie_image_rope(2, 2, 1).unwrap();
        assert_eq!(rope.len(), 4 * HEAD_DIM);
        let (cos, sin) = rope_sin_cos(2.);
        assert_eq!(&rope[..2], &[cos, sin]);
        assert_eq!(&rope[2 * HEAD_DIM..2 * HEAD_DIM + 2], &[1., 0.]);
        assert_eq!(&rope[HEAD_DIM + 80..HEAD_DIM + 82], &{
            let (cos, sin) = rope_sin_cos(1.);
            [cos, sin]
        });
    }

    #[test]
    fn rotary_pairs_first_and_second_halves() {
        let mut values = [1., 2., 3., 4.];
        rotate_neox_inplace(&mut values, &[0., 1., 1., 0.]).unwrap();
        assert_eq!(values, [-3., -4., 3., 4.]);
    }

    #[test]
    fn branch_norm_preserves_residual_and_writes_the_entire_row() {
        let input = [2.; 4];
        let mut output = [f32::NAN; 4];
        modulated_rms_norm(&input, &[1.; 4], &[2.; 4], &[4.; 4], &mut output);
        assert_eq!(input, [2.; 4]);
        assert!(output.iter().all(|v| (v - 7.).abs() < 1e-5));
    }
}
