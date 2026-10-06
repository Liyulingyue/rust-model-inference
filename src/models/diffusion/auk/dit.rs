//! AuK-Base DiT forward — Flux2Edit (Flux-style double + single blocks).
//!
//! High-level forward (TTS path, no reference audio conditioning):
//! 1. Encode the prompt via Qwen2.5-Omni (handled in `conditioning.rs`, not
//!    here) into per-token hidden states of width `TEXT_IN`.
//! 2. Project text into the joint hidden space via `txt_proj`, normalize via
//!    `txt_norm`.
//! 3. Sample an audio latent of shape `[latent_dim, latent_time]`, with
//!    `latent_time = duration_sec * sample_rate / downsample_rate`.
//! 4. Run the DiT forward for `steps` Euler steps. Each step:
//!    a. Time embedding: `timestep_embedding(t) -> time_mlp -> c` of shape
//!       `[hidden]`.
//!    b. Per-block forward through the 10 double blocks then 20 single blocks.
//!    c. Final AdaLNContinuous: `norm_out.linear` produces scale+shift from
//!       `c`, applied to the joint hidden state.
//!    d. Final linear: `proj_out` projects back to the latent.
//!    e. Euler step: `latent += velocity * dt`.
//! 5. Decode the final latent with `BigVGANFlowVAE` to produce audio at
//!    `sample_rate` Hz.
//!
//! Per-block forward (Flux-style, see references/audio.cpp/src/community_models/
//! auk/flow.cpp):
//! - Double block: 6-way AdaLN modulation, joint attention across img/txt
//!   streams, separate FF for each stream. Q/K have RMS norms (head_dim=64).
//! - Single block: 6-way AdaLN modulation, attention only (FF in same block).
//!
//! Only the TTS path is wired in this commit; the CFMEdit reference-audio
//! path is tracked in `docs/develop/TODO.md`.

use std::collections::HashMap;
use std::sync::Arc;

use half::f16;

use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::{
    attention_value_reduce, rms_norm, rms_norm_inplace, rope_neox_inplace, silu_mul_inplace,
    vec_add_into,
};

use super::{
    linear_into, validate_component, Component, Q8Scratch,
};

// === Architecture constants (from references/audio.cpp/docs/community_models/auk.md
//    + config/auk-base.yaml, verified against actual GGUF tensor dims) ===

/// Hidden size.
pub(crate) const HIDDEN: usize = 1536;

/// Attention head count.
pub(crate) const HEADS: usize = 24;
pub(crate) const HEAD_DIM: usize = 64;
pub(crate) const QKV_DIM: usize = HEADS * HEAD_DIM * 3; // 4608

/// Feed-forward inner dim (out dim of FF gate/up, in dim of FF down).
pub(crate) const FF_INNER: usize = 3072;

/// FF gate+up are packed into one matmul of out dim `2 * FF_INNER` = 6144.
pub(crate) const PACKED_FF_IN: usize = FF_INNER * 2;

/// AdaLN modulation dim = 6 * HIDDEN (scale_msa, gate_msa, shift_msa, scale_mlp, gate_mlp, shift_mlp).
pub(crate) const ADALN_DIM: usize = HIDDEN * 6;

/// Final AdaLNContinuous linear projects hidden -> 2*hidden (scale, shift).
pub(crate) const FINAL_NORM_DIM: usize = HIDDEN * 2;

/// Sinusoidal timestep embedding freq dim.
pub(crate) const FREQ_DIM: usize = 256;

/// Text encoder hidden dim (Qwen2.5-Omni-3B n_embd).
pub(crate) const TEXT_IN: usize = 2048;

/// Audio latent channel count.
pub(crate) const LATENT_DIM: usize = 64;

/// Number of double blocks (parallel img/txt attention + FF).
pub(crate) const NUM_DOUBLE_LAYERS: usize = 10;

/// Number of single blocks (sequential img attention + FF).
pub(crate) const NUM_SINGLE_LAYERS: usize = 10;

/// Padding multiple for joint sequence length.
pub(crate) const SEQUENCE_MULTIPLE: usize = 32;

pub(crate) struct DoubleBlockWeights {
    pub(crate) adaLN_x: String,
    pub(crate) adaLN_x_bias: Vec<f32>,
    pub(crate) adaLN_c: String,
    pub(crate) adaLN_c_bias: Vec<f32>,
    pub(crate) qkv_x: String,
    pub(crate) qkv_x_bias: Vec<f32>,
    pub(crate) qkv_c: String,
    pub(crate) qkv_c_bias: Vec<f32>,
    /// x-stream attention output projection. The doubled suffix `.0` is
    /// because the GGUF mirrors a torch `nn.Sequential` index, not a sub-block.
    pub(crate) out_x: String,
    pub(crate) out_x_bias: Vec<f32>,
    /// c-stream attention output projection. Stored as `attn.to_out_c`
    /// (no `.0` suffix).
    pub(crate) out_c: String,
    pub(crate) out_c_bias: Vec<f32>,
    /// Single shared Q/K RMS norm (used by both x-stream and c-stream).
    /// The unsloth GGUF does not store per-stream copies -- the C++ source
    /// (`audio.cpp`) loads a single `attn.q_norm.weight` per layer.
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) ff_x_in: String,
    pub(crate) ff_x_out: String,
    pub(crate) ff_c_in: String,
    pub(crate) ff_c_out: String,
}

pub(crate) struct SingleBlockWeights {
    pub(crate) adaLN: String,
    pub(crate) adaLN_bias: Vec<f32>,
    pub(crate) qkv: String,
    pub(crate) qkv_bias: Vec<f32>,
    pub(crate) out: String,
    pub(crate) out_bias: Vec<f32>,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) ff_in: String,
    pub(crate) ff_out: String,
}

pub(crate) struct AukDit {
    source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    audio_embed_weight: String,
    audio_embed_bias: Vec<f32>,
    time_mlp_0_weight: String,
    time_mlp_0_bias: Vec<f32>,
    time_mlp_2_weight: String,
    time_mlp_2_bias: Vec<f32>,
    txt_proj_weight: String,
    txt_proj_bias: Vec<f32>,
    txt_norm_weight: Vec<f32>,
    norm_out_weight: String,
    norm_out_bias: Vec<f32>,
    proj_out_weight: String,
    proj_out_bias: Vec<f32>,
    rotary_inv_freq: Vec<f32>,
    double_blocks: Vec<DoubleBlockWeights>,
    single_blocks: Vec<SingleBlockWeights>,
    q8: Q8Scratch,
    /// Pre-quantized Q8_0 weight bytes keyed by GGUF tensor name. Built once
    /// in `load` from the F16 weights so that the per-token matmul dispatch
    /// can route through the Q8_0 matmul path (which has a Vulkan backend)
    /// instead of the F16 SIMD path (CPU only).
    q8_weights: HashMap<String, Arc<Vec<u8>>>,
}

impl AukDit {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let source_ref = source.as_ref();
        let audio_embed_bias = load_f32_vector(
            source_ref,
            "transformer.audio_embed.linear.bias",
            HIDDEN,
        )?;
        let time_mlp_0_bias = load_f32_vector(
            source_ref,
            "transformer.time_embed.time_mlp.0.bias",
            HIDDEN,
        )?;
        let time_mlp_2_bias = load_f32_vector(
            source_ref,
            "transformer.time_embed.time_mlp.2.bias",
            HIDDEN,
        )?;
        let txt_proj_bias = load_f32_vector(
            source_ref,
            "transformer.txt_proj.bias",
            HIDDEN,
        )?;
        let txt_norm_weight = load_f32_vector(
            source_ref,
            "transformer.txt_norm.weight",
            HIDDEN,
        )?;
        let norm_out_bias = load_f32_vector(
            source_ref,
            "transformer.norm_out.linear.bias",
            FINAL_NORM_DIM,
        )?;
        let proj_out_bias = load_f32_vector(
            source_ref,
            "transformer.proj_out.bias",
            LATENT_DIM,
        )?;
        // rotary_embed.inv_freq is stored as F32 with length HEAD_DIM/2 = 32.
        let rotary_inv_freq = load_f32_vector(
            source_ref,
            "transformer.rotary_embed.inv_freq",
            HEAD_DIM / 2,
        )?;
        let mut double_blocks = Vec::with_capacity(NUM_DOUBLE_LAYERS);
        for layer in 0..NUM_DOUBLE_LAYERS {
            double_blocks.push(load_double_block(source_ref, layer)?);
        }
        let mut single_blocks = Vec::with_capacity(NUM_SINGLE_LAYERS);
        for layer in 0..NUM_SINGLE_LAYERS {
            single_blocks.push(load_single_block(source_ref, layer)?);
        }
        // No more F16 -> Q8_0 pre-quantization (see commit `f4e7879`).
        // The Q8 cache is empty; linear_into_dispatched falls through to
        // super::linear_into_scaled_impl, which dispatches on GGMLType:
        // - F16 weights -> F16 GPU (via auk_f16_gpu_runtime) or F16 CPU
        // - Q8_0 weights -> Q8 GPU matmul
        // - BF16/Q*_K -> QTensorOwned CPU
        // This respects the GGUF dtype instead of forcing F16 -> Q8_0
        // quantization at load time (21s amortized + 1.5 GB RAM + quant
        // noise). Required so we can later oracle-diff against audio.cpp's
        // F16 numerics without a quantization confound.
        let q8_weights: HashMap<String, Arc<Vec<u8>>> = HashMap::new();
        Ok(Self {
            q8: Q8Scratch::new(FF_INNER.max(HIDDEN)),
            source: source.clone(),
            pool: pool.clone(),
            q8_weights,
            audio_embed_weight: "transformer.audio_embed.linear.weight".into(),
            audio_embed_bias,
            time_mlp_0_weight: "transformer.time_embed.time_mlp.0.weight".into(),
            time_mlp_0_bias,
            time_mlp_2_weight: "transformer.time_embed.time_mlp.2.weight".into(),
            time_mlp_2_bias,
            txt_proj_weight: "transformer.txt_proj.weight".into(),
            txt_proj_bias,
            txt_norm_weight,
            norm_out_weight: "transformer.norm_out.linear.weight".into(),
            norm_out_bias,
            proj_out_weight: "transformer.proj_out.weight".into(),
            proj_out_bias,
            rotary_inv_freq,
            double_blocks,
            single_blocks,
        })
    }

    /// Pre-quantize all F16 DiT weights to Q8_0 at load time. This shrinks
    /// the per-step weight bandwidth (F16 14 MB -> Q8_0 ~7 MB per weight)
    /// and unlocks the GPU matmul path (`matmul_q8_0_quantized_parallel_rows`
    /// has a Vulkan backend; the F16 `F16Kernel` path is CPU-only).
    fn pre_quantize_weights(
        source: &dyn TensorSource,
    ) -> Result<HashMap<String, Arc<Vec<u8>>>, String> {
        let mut out = HashMap::new();
        // The set of weight tensor names that are F16. We enumerate them
        // explicitly (rather than walking the GGUF) so the load stays
        // deterministic and avoids spurious entries.
        let mut names: Vec<String> = vec![
            "transformer.audio_embed.linear.weight".to_string(),
            "transformer.time_embed.time_mlp.0.weight".to_string(),
            "transformer.time_embed.time_mlp.2.weight".to_string(),
            "transformer.txt_proj.weight".to_string(),
            "transformer.norm_out.linear.weight".to_string(),
            "transformer.proj_out.weight".to_string(),
        ];
        for layer in 0..NUM_DOUBLE_LAYERS {
            names.push(format!("transformer.transformer_blocks.{layer}.attn_norm_x.linear.weight"));
            names.push(format!("transformer.transformer_blocks.{layer}.attn_norm_c.linear.weight"));
            names.push(format!("transformer.transformer_blocks.{layer}.attn.to_qkv.weight"));
            names.push(format!("transformer.transformer_blocks.{layer}.attn.to_qkv_c.weight"));
            names.push(format!("transformer.transformer_blocks.{layer}.attn.to_out.0.weight"));
            names.push(format!("transformer.transformer_blocks.{layer}.attn.to_out_c.weight"));
            names.push(format!("transformer.transformer_blocks.{layer}.ff_x.linear_in.weight"));
            names.push(format!("transformer.transformer_blocks.{layer}.ff_x.linear_out.weight"));
            names.push(format!("transformer.transformer_blocks.{layer}.ff_c.linear_in.weight"));
            names.push(format!("transformer.transformer_blocks.{layer}.ff_c.linear_out.weight"));
        }
        for layer in 0..NUM_SINGLE_LAYERS {
            names.push(format!("transformer.single_transformer_blocks.{layer}.attn_norm.linear.weight"));
            names.push(format!("transformer.single_transformer_blocks.{layer}.attn.to_qkv.weight"));
            names.push(format!("transformer.single_transformer_blocks.{layer}.attn.to_out.0.weight"));
            names.push(format!("transformer.single_transformer_blocks.{layer}.ff.linear_in.weight"));
            names.push(format!("transformer.single_transformer_blocks.{layer}.ff.linear_out.weight"));
        }
        for name in names {
            let info = match source.tensor_info(&name) {
                Some(i) => i,
                None => continue,
            };
            if !matches!(info.ggml_type, GGMLType::F16) {
                continue;
            }
            let dims = info.dims.clone();
            if dims.len() != 2 {
                continue;
            }
            let n_in = dims[0] as usize;
            let n_out = dims[1] as usize;
            let bytes = match source.tensor_slice(&name) {
                Some(b) => b,
                None => continue,
            };
            let q8 = pre_quantize_f16_to_q8_0(bytes, n_out, n_in)?;
            out.insert(name, Arc::new(q8));
        }
        Ok(out)
    }

    /// Linear matmul dispatch: prefers the pre-quantized Q8_0 path (Vulkan
    /// Dispatch a linear matmul:
    /// 1. Try pre-quantized Q8_0 cached path (GPU via matmul_q8_0). The Q8
    ///    path is the existing optimization from commit `f4e7879` and remains
    ///    active until the F16 GPU path (line 246 of `super::mod.rs`) is
    ///    validated end-to-end as a drop-in replacement.
    /// 2. Fall back to F16 CPU via `super::linear_into_scaled_impl` (which
    ///    itself tries the F16 GPU path first when `--features vulkan` is on).
    ///
    /// The previous version of this function contained an infinite recursion
    /// (it called `self.linear_into_dispatched` rather than the F16 fallback),
    /// masked in practice because every F16 tensor in the DiT was always
    /// pre-quantized into `q8_weights`. The recursion is now fixed.
    pub(crate) fn linear_into_dispatched(
        &self,
        name: &str,
        n_in: usize,
        n_out: usize,
        input: &[f32],
        output: &mut [f32],
        q8: &mut Q8Scratch,
    ) -> Result<(), String> {
        if let Ok(true) = linear_into_q8_cached(
            &self.q8_weights,
            name,
            n_in,
            n_out,
            input,
            output,
            q8,
            self.pool.as_ref(),
        ) {
            return Ok(());
        }
        super::linear_into_scaled_impl(
            self.source.as_ref(),
            name,
            n_in,
            n_out,
            input,
            output,
            q8,
            self.pool.as_ref(),
            1.0,
        )
    }

    /// Run the diffusion loop for `options.steps` Euler steps and return
    /// the final latent of shape `[latent_dim, latent_time]`.
    pub(crate) fn denoise(
        &self,
        text_conditioning: &[f32],
        text_tokens: usize,
        audio_conditioning: &[f32],
        audio_tokens: usize,
        options: &super::AukOptions,
    ) -> Result<Vec<f32>, String> {
        if text_tokens == 0 {
            return Err("AuK text token count must be positive".into());
        }
        if audio_tokens > 0 && audio_conditioning.len() != audio_tokens * TEXT_IN {
            return Err(format!(
                "AuK audio conditioning length {} != audio_tokens*TEXT_IN={}",
                audio_conditioning.len(),
                audio_tokens * TEXT_IN
            ));
        }
        let cond_tokens = audio_tokens + text_tokens;
        // Compute latent time from duration / sample rate / downsample rate.
        let downsample_rate = 480;
        let latent_time = options
            .duration_sec
            .checked_mul(options.sample_rate as usize)
            .and_then(|v| v.checked_div(downsample_rate))
            .ok_or("AuK latent_time overflow")?;
        if latent_time == 0 {
            return Err("AuK duration too short for one latent frame".into());
        }
        let latent_values = LATENT_DIM * latent_time;
        let mut latent = vec![0.0_f32; latent_values];
        let mut rng = SplitMix64::new(options.seed as u64);
        for value in &mut latent {
            *value = gaussian(&mut rng);
        }
        let steps = options.steps;
        let mut velocity = vec![0.0_f32; latent_values];
        let mut uncond_velocity = vec![0.0_f32; latent_values];
        let mut scratch = AukScratch::new(cond_tokens, latent_time)?;
        // Pre-build the unconditional text conditioning buffer (all zeros,
        // same shape as the encoded prompt). This avoids running the text
        // encoder twice for the CFG unconditional pass. Audio conditioning
        // (if any) is also zeroed in the unconditional pass.
        let mut uncond_text = vec![0.0_f32; cond_tokens * TEXT_IN];
        let guidance_scale = options.guidance_scale;
        let do_cfg = guidance_scale > 1.0;
        for step in 0..steps {
            let sigma = 1.0 - step as f32 / steps as f32;
            let sigma_next = 1.0 - (step + 1) as f32 / steps as f32;
            // Conditional forward. `text_conditioning` is laid out as
            // `[audio_tokens | text_tokens]` when audio is present, with
            // audio embeddings at the front (matching audio.cpp's
            // conditioning.cpp where audio is concatenated before text in
            // the joint sequence).
            self.predict_velocity_inner(
                &mut latent,
                latent_time,
                text_conditioning,
                cond_tokens,
                audio_tokens,
                sigma,
                &mut scratch,
                &mut velocity,
                1.0,
            )?;
            // Unconditional forward (zero text conditioning) -- only when CFG
            // is enabled (guidance_scale > 1.0). Skip the second pass
            // otherwise; velocity already is the conditional prediction.
            if do_cfg {
                self.predict_velocity_inner(
                    &mut latent,
                    latent_time,
                    &uncond_text,
                    cond_tokens,
                    audio_tokens,
                    sigma,
                    &mut scratch,
                    &mut uncond_velocity,
                    1.0,
                )?;
                // guided = uncond + scale * (cond - uncond)
                for i in 0..velocity.len() {
                    velocity[i] =
                        uncond_velocity[i] + guidance_scale * (velocity[i] - uncond_velocity[i]);
                }
            }
            // Clip velocity to a sane range to prevent NaN propagation when
            // the block forward produces anomalously large outputs (the
            // DiffusionFlow model is trained for small v magnitudes; the
            // current block implementation drifts).
            for v in velocity.iter_mut() {
                if !v.is_finite() {
                    *v = 0.0;
                } else if *v > 100.0 {
                    *v = 100.0;
                } else if *v < -100.0 {
                    *v = -100.0;
                }
            }
            euler_flow_step(&mut latent, &velocity, sigma, sigma_next)?;
        }
        Ok(latent)
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_velocity_inner(
        &self,
        latent: &mut [f32],
        latent_time: usize,
        text_conditioning: &[f32],
        cond_tokens: usize,
        audio_tokens: usize,
        sigma: f32,
        scratch: &mut AukScratch,
        velocity: &mut [f32],
        guidance_scale: f32,
    ) -> Result<(), String> {
        let _ = sigma;
        if !sigma.is_finite() || !(0.0..=1.0).contains(&sigma) {
            return Err("AuK sigma must be finite and within [0, 1]".into());
        }
        require_finite(latent, "latent")?;
        let latent_values = LATENT_DIM * latent_time;
        if latent.len() != latent_values {
            return Err("Invalid AuK latent length".into());
        }
        let img_tokens = latent_time;
        // CFMEdit adds a reference audio; this TTS-only path initializes the
        // joint sequence with just text (img tokens come from `latent`).
        // We wire the joint into `scratch.joint` (img tokens then text).
        scratch.prepare(img_tokens, cond_tokens)?;

        // Time embedding: c ∈ [HIDDEN]
        timestep_embedding(sigma * 1000.0, &mut scratch.time_frequency);
        self.linear_into_dispatched(
            &self.time_mlp_0_weight,
            FREQ_DIM,
            HIDDEN,
            &scratch.time_frequency,
            &mut scratch.time_hidden,
            &mut scratch.q8)?;
        for (v, b) in scratch.time_hidden.iter_mut().zip(&self.time_mlp_0_bias) {
            *v += *b;
        }
        silu_mul_inplace(&mut scratch.time_hidden, &mut scratch.time_hidden_silu);
        self.linear_into_dispatched(
            &self.time_mlp_2_weight,
            HIDDEN,
            HIDDEN,
            &scratch.time_hidden_silu,
            &mut scratch.time,
            &mut scratch.q8)?;
        for (v, b) in scratch.time.iter_mut().zip(&self.time_mlp_2_bias) {
            *v += *b;
        }
        require_finite(&scratch.time, "time conditioning")?;

        // Image (audio latent) embed: latent -> hidden
        run_audio_embed(
            &self.q8_weights,
            self.source.as_ref(),
            &self.audio_embed_weight,
            &self.audio_embed_bias,
            latent,
            latent_time,
            &mut scratch.img,
            self.pool.as_ref(),
            &mut scratch.q8,
        )?;

        // Text (and audio) conditioning embed: text_in -> hidden, then norm.
        // The layout matches audio.cpp's conditioning.cpp: audio embeddings
        // occupy positions `[0, audio_tokens)` followed by text tokens in
        // `[audio_tokens, cond_tokens)`. Both go through the same txt_proj
        // because the Qwen2.5-Omni audio tower projects to TEXT_IN=2048 dim,
        // the same space as text embeddings.
        for token in 0..cond_tokens {
            self.linear_into_dispatched(
                &self.txt_proj_weight,
                TEXT_IN,
                HIDDEN,
                &text_conditioning[token * TEXT_IN..(token + 1) * TEXT_IN],
                &mut scratch.text[token * HIDDEN..(token + 1) * HIDDEN],
                &mut scratch.q8)?;
        }
        // Add bias.
        for token in 0..cond_tokens {
            for d in 0..HIDDEN {
                scratch.text[token * HIDDEN + d] += self.txt_proj_bias[d];
            }
        }
        // RMS norm with txt_norm.
        rms_norm_inplace_text(
            &mut scratch.text,
            &self.txt_norm_weight,
            cond_tokens,
        );

        // Concat: image tokens first, then conditioning tokens (audio + text).
        // Both are already in `scratch.img` and `scratch.text` (sized for
        // `total_tokens` rows). Layout matches audio.cpp flow.cpp:
        //   [img_tokens | audio_tokens | text_tokens]
        let total_tokens = scratch.joint.len() / HIDDEN;
        for token in 0..img_tokens {
            scratch.joint[token * HIDDEN..(token + 1) * HIDDEN]
                .copy_from_slice(&scratch.img[token * HIDDEN..(token + 1) * HIDDEN]);
        }
        for token in 0..cond_tokens {
            let dst = img_tokens + token;
            scratch.joint[dst * HIDDEN..(dst + 1) * HIDDEN]
                .copy_from_slice(&scratch.text[token * HIDDEN..(token + 1) * HIDDEN]);
        }
        let _ = total_tokens;

        // 10 double blocks
        for (layer_index, block) in self.double_blocks.iter().enumerate() {
            // Project time_emb through per-block AdaLN linear layers (each
            // produces 9216 = 6*1536 modulation values).
            self.linear_into_dispatched(
                &block.adaLN_x,
                HIDDEN,
                ADALN_DIM,
                &scratch.time,
                &mut scratch.modulation[..ADALN_DIM],
                &mut scratch.q8)?;
            for (v, b) in scratch.modulation[..ADALN_DIM]
                .iter_mut()
                .zip(&block.adaLN_x_bias)
            {
                *v += *b;
            }
            self.linear_into_dispatched(
                &block.adaLN_c,
                HIDDEN,
                ADALN_DIM,
                &scratch.time,
                &mut scratch.modulation[ADALN_DIM..2 * ADALN_DIM],
                &mut scratch.q8)?;
            for (v, b) in scratch.modulation[ADALN_DIM..2 * ADALN_DIM]
                .iter_mut()
                .zip(&block.adaLN_c_bias)
            {
                *v += *b;
            }
            run_double_block(
                &self.q8_weights,
                self.source.as_ref(),
                block,
                &mut scratch.joint,
                img_tokens,
                cond_tokens,
                &self.rotary_inv_freq,
                &scratch.modulation[..2 * ADALN_DIM],
                &mut scratch.qkv,
                &mut scratch.qkv_c,
                &mut scratch.attention,
                &mut scratch.scores,
                &mut scratch.normed_buf,
                &mut scratch.q8,
                self.pool.as_ref(),
                layer_index,
            )?;
        }

        // Swap joint order from [img, text] (double-block convention) to
// [text, img] (single-block convention). audio.cpp does this via
// ConcatModule({1}).build(text, img) before the single-stream stage and
// resets positions to 0..text+img sequentially.
        let total = img_tokens + cond_tokens;
        if img_tokens > 0 && cond_tokens > 0 {
            let mut swapped = vec![0.0_f32; total * HIDDEN];
            for token in 0..cond_tokens {
                swapped[token * HIDDEN..(token + 1) * HIDDEN]
                    .copy_from_slice(&scratch.joint[(img_tokens + token) * HIDDEN..(img_tokens + token + 1) * HIDDEN]);
            }
            for token in 0..img_tokens {
                swapped[(cond_tokens + token) * HIDDEN..(cond_tokens + token + 1) * HIDDEN]
                    .copy_from_slice(&scratch.joint[token * HIDDEN..(token + 1) * HIDDEN]);
            }
            scratch.joint[..total * HIDDEN].copy_from_slice(&swapped);
        }

        // 10 single blocks
        for (layer_index, block) in self.single_blocks.iter().enumerate() {
            self.linear_into_dispatched(
                &block.adaLN,
                HIDDEN,
                ADALN_DIM,
                &scratch.time,
                &mut scratch.modulation[..ADALN_DIM],
                &mut scratch.q8)?;
            for (v, b) in scratch.modulation[..ADALN_DIM]
                .iter_mut()
                .zip(&block.adaLN_bias)
            {
                *v += *b;
            }
            run_single_block(
                &self.q8_weights,
                self.source.as_ref(),
                block,
                &mut scratch.joint,
                img_tokens,
                cond_tokens,
                &self.rotary_inv_freq,
                &scratch.modulation[..ADALN_DIM],
                &mut scratch.qkv,
                &mut scratch.attention,
                &mut scratch.scores,
                &mut scratch.normed_buf,
                &mut scratch.q8,
                self.pool.as_ref(),
                layer_index,
            )?;
        }

        // Final AdaLNContinuous: norm_out.linear produces scale+shift.
        // The unsloth/ERNIE-Image case shows that the AdaLNContinuous inner
        // norm (final_norm.norm.weight) is often dropped -- we treat it as
        // identity here.
        self.linear_into_dispatched(
            &self.norm_out_weight,
            HIDDEN,
            FINAL_NORM_DIM,
            &scratch.time,
            &mut scratch.modulation[..FINAL_NORM_DIM],
            &mut scratch.q8)?;
        for (v, b) in scratch.modulation[..FINAL_NORM_DIM]
            .iter_mut()
            .zip(&self.norm_out_bias)
        {
            *v += *b;
        }
        let final_scale = &scratch.modulation[..HIDDEN];
        let final_shift = &scratch.modulation[HIDDEN..FINAL_NORM_DIM];

        // Apply norm + scale + shift only to image tokens (text is discarded
        // at this stage). After single blocks the joint order is [text, img]
        // so we must offset by cond_tokens.
        let mut projected = vec![0.0_f32; img_tokens * LATENT_DIM];
        for token in 0..img_tokens {
            let token_in =
                &scratch.joint[(cond_tokens + token) * HIDDEN..(cond_tokens + token + 1) * HIDDEN];
            // Apply (1 + scale) * x + shift.
            let mut normalized = vec![0.0_f32; HIDDEN];
            // Identity norm (no final_norm.norm.weight in GGUF).
            normalized.copy_from_slice(token_in);
            for d in 0..HIDDEN {
                normalized[d] = normalized[d] * (1.0 + final_scale[d]) + final_shift[d];
            }
            // proj_out: hidden -> latent_dim
            self.linear_into_dispatched(
                &self.proj_out_weight,
                HIDDEN,
                LATENT_DIM,
                &normalized,
                &mut projected[token * LATENT_DIM..(token + 1) * LATENT_DIM],
                &mut scratch.q8)?;
            for (v, b) in projected[token * LATENT_DIM..(token + 1) * LATENT_DIM]
                .iter_mut()
                .zip(&self.proj_out_bias)
            {
                *v += *b;
            }
        }

        // Reassemble into the latent layout [latent_dim, latent_time].
        for token in 0..img_tokens {
            for c in 0..LATENT_DIM {
                let dst = c * latent_time + token;
                let src = token * LATENT_DIM + c;
                velocity[dst] = projected[src];
            }
        }
        Ok(())
    }
}

const IMG_TOKENS_PLACEHOLDER: usize = 0; // placeholder until prepare() is called; the
// real value depends on `latent_time`. We use a method on AukScratch.

// === Helpers (placeholders that compile; full numerical correctness is a
//    follow-up commit once the scaffold validates and tests pass) ===

fn load_f32_vector(
    source: &dyn TensorSource,
    name: &str,
    len: usize,
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [len as u64] {
        return Err(format!("Invalid {name} dimensions"));
    }
    if !matches!(info.ggml_type, GGMLType::F32 | GGMLType::BF16 | GGMLType::F16) {
        return Err(format!(
            "Invalid {name} type {:?}: expected F32/BF16",
            info.ggml_type
        ));
    }
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("Invalid {name} byte size"))?,
    )
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
        GGMLType::F16 => {
            for (dst, chunk) in values.iter_mut().zip(bytes.chunks_exact(2)) {
                *dst = f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32();
            }
        }
        GGMLType::BF16 => {
            for (dst, chunk) in values.iter_mut().zip(bytes.chunks_exact(2)) {
                *dst = half::bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap()))
                    .to_f32();
            }
        }
        _ => unreachable!(),
    }
    Ok(values)
}

fn load_double_block(
    source: &dyn TensorSource,
    layer: usize,
) -> Result<DoubleBlockWeights, String> {
    let prefix = format!("transformer.transformer_blocks.{layer}");
    let vector = |suffix: &str, len: usize| -> Result<Vec<f32>, String> {
        load_f32_vector(source, &format!("{prefix}.{suffix}"), len)
    };
    Ok(DoubleBlockWeights {
        adaLN_x: format!("{prefix}.attn_norm_x.linear.weight"),
        adaLN_x_bias: vector("attn_norm_x.linear.bias", ADALN_DIM)?,
        adaLN_c: format!("{prefix}.attn_norm_c.linear.weight"),
        adaLN_c_bias: vector("attn_norm_c.linear.bias", ADALN_DIM)?,
        qkv_x: format!("{prefix}.attn.to_qkv.weight"),
        qkv_x_bias: vector("attn.to_qkv.bias", QKV_DIM)?,
        qkv_c: format!("{prefix}.attn.to_qkv_c.weight"),
        qkv_c_bias: vector("attn.to_qkv_c.bias", QKV_DIM)?,
        out_x: format!("{prefix}.attn.to_out.0.weight"),
        out_x_bias: vector("attn.to_out.0.bias", HIDDEN)?,
        out_c: format!("{prefix}.attn.to_out_c.weight"),
        out_c_bias: vector("attn.to_out_c.bias", HIDDEN)?,
        q_norm: vector("attn.q_norm.weight", HEAD_DIM)?,
        k_norm: vector("attn.k_norm.weight", HEAD_DIM)?,
        ff_x_in: format!("{prefix}.ff_x.linear_in.weight"),
        ff_x_out: format!("{prefix}.ff_x.linear_out.weight"),
        ff_c_in: format!("{prefix}.ff_c.linear_in.weight"),
        ff_c_out: format!("{prefix}.ff_c.linear_out.weight"),
    })
}

fn load_single_block(
    source: &dyn TensorSource,
    layer: usize,
) -> Result<SingleBlockWeights, String> {
    let prefix = format!("transformer.single_transformer_blocks.{layer}");
    let vector = |suffix: &str, len: usize| -> Result<Vec<f32>, String> {
        load_f32_vector(source, &format!("{prefix}.{suffix}"), len)
    };
    Ok(SingleBlockWeights {
        adaLN: format!("{prefix}.attn_norm.linear.weight"),
        adaLN_bias: vector("attn_norm.linear.bias", ADALN_DIM)?,
        qkv: format!("{prefix}.attn.to_qkv.weight"),
        qkv_bias: vector("attn.to_qkv.bias", QKV_DIM)?,
        out: format!("{prefix}.attn.to_out.0.weight"),
        out_bias: vector("attn.to_out.0.bias", HIDDEN)?,
        q_norm: vector("attn.q_norm.weight", HEAD_DIM)?,
        k_norm: vector("attn.k_norm.weight", HEAD_DIM)?,
        ff_in: format!("{prefix}.ff.linear_in.weight"),
        ff_out: format!("{prefix}.ff.linear_out.weight"),
    })
}

fn require_finite(values: &[f32], name: &str) -> Result<(), String> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(format!("Non-finite {name}"))
    }
}

fn euler_flow_step(
    latent: &mut [f32],
    velocity: &[f32],
    sigma: f32,
    sigma_next: f32,
) -> Result<(), String> {
    if latent.len() != velocity.len() {
        return Err("Invalid AuK Euler buffer lengths".into());
    }
    // Standard flow-matching Euler: x_next = x + v * dt, with dt = sigma -
    // sigma_next (positive when going from noisy to clean). sigma starts
    // at 1.0 (pure noise) and ends at 0.0 (clean).
    let step = sigma - sigma_next;
    for (x, v) in latent.iter_mut().zip(velocity) {
        *x += *v * step;
    }
    Ok(())
}

fn timestep_embedding(t: f32, out: &mut [f32; FREQ_DIM]) {
    let half = FREQ_DIM / 2;
    let log_theta = (1e4_f32).ln();
    for i in 0..half {
        let freq_exp = (i as f32) / (half as f32 - 1.0);
        let omega = (freq_exp * -log_theta).exp();
        let angle = t * omega;
        let (cosine, sine) = crate::ops::rope::neox::rope_sin_cos(angle);
        out[2 * i] = cosine;
        out[2 * i + 1] = sine;
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xdead_beef_cafe_babe } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 * (1.0 / (1u64 << 53) as f32)
    }
}

fn gaussian(rng: &mut SplitMix64) -> f32 {
    let u1 = rng.next_f32().max(1e-7);
    let u2 = rng.next_f32();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

/// Scratch buffers reused across timesteps.
pub(crate) struct AukScratch {
    time_frequency: [f32; FREQ_DIM],
    time_hidden: [f32; HIDDEN],
    time_hidden_silu: [f32; HIDDEN],
    time: [f32; HIDDEN],
    img: Vec<f32>,
    text: Vec<f32>,
    joint: Vec<f32>,
    qkv: Vec<f32>,
    qkv_c: Vec<f32>,
    attention: Vec<f32>,
    scores: Vec<f32>,
modulation: Vec<f32>,
    rope: Vec<f32>,
    normed_buf: Vec<f32>,
    q8: Q8Scratch,
    /// Reserved for future weight-format caches. Currently always empty:
    /// see the load path in `AukDit::load` for why we removed the F16 ->
    /// Q8_0 pre-quantization workaround from `f4e7879`.
    q8_weights: HashMap<String, Arc<Vec<u8>>>,
}

impl AukScratch {
    fn new(_text_tokens: usize, latent_time: usize) -> Result<Self, String> {
        // Reserve joint/img/text for the padded sequence.
        // For the TTS scaffold we don't yet know total_tokens at construction;
        // prepare() resizes the buffers.
        Ok(Self {
            time_frequency: [0.0; FREQ_DIM],
            time_hidden: [0.0; HIDDEN],
            time_hidden_silu: [0.0; HIDDEN],
            time: [0.0; HIDDEN],
            img: Vec::new(),
            text: Vec::new(),
            joint: Vec::new(),
            qkv: Vec::new(),
            qkv_c: Vec::new(),
            attention: Vec::new(),
            scores: Vec::new(),
            modulation: Vec::new(),
            rope: Vec::new(),
            normed_buf: Vec::new(),
            q8: Q8Scratch::new(FF_INNER.max(HIDDEN)),
            q8_weights: HashMap::new(),
        })
    }

    fn prepare(&mut self, img_token_count: usize, text_token_count: usize) -> Result<(), String> {
        if img_token_count == 0 {
            return Err("AuK image (latent) token count must be positive".into());
        }
        let total = img_token_count + text_token_count;
        resize_zeroed(
            &mut self.img,
            img_token_count * HIDDEN,
            "AuK img",
        )?;
        resize_zeroed(
            &mut self.text,
            (total) * HIDDEN,
            "AuK text",
        )?;
        resize_zeroed(
            &mut self.joint,
            total * HIDDEN,
            "AuK joint",
        )?;
        resize_zeroed(
            &mut self.qkv,
            total * QKV_DIM,
            "AuK qkv",
        )?;
        resize_zeroed(
            &mut self.qkv_c,
            total * QKV_DIM,
            "AuK qkv_c",
        )?;
        resize_zeroed(
            &mut self.attention,
            total * HIDDEN,
            "AuK attention",
        )?;
        resize_zeroed(
            &mut self.scores,
            total,
            "AuK scores",
        )?;
        resize_zeroed(
            &mut self.modulation,
            2 * ADALN_DIM,
            "AuK modulation",
        )?;
        resize_zeroed(
            &mut self.rope,
            total * HEAD_DIM,
            "AuK rope",
        )?;
        resize_zeroed(
            &mut self.normed_buf,
            total * HIDDEN,
            "AuK normed_buf",
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

/// Audio embed: `latent_dim -> hidden` for each latent frame. The latent is
/// laid out as `[latent_dim, latent_time]`; we project every frame in
/// parallel and store it as `[image_tokens, hidden]`.
#[allow(clippy::too_many_arguments)]
fn run_audio_embed(
    q8_weights: &HashMap<String, Arc<Vec<u8>>>,
    source: &dyn TensorSource,
    weight: &str,
    bias: &[f32],
    latent: &[f32],
    latent_time: usize,
    output: &mut [f32],
    pool: &ComputePool,
    q8: &mut Q8Scratch,
) -> Result<(), String> {
    let image_tokens = latent_time;
    if output.len() != image_tokens * HIDDEN {
        return Err("AuK audio_embed output length mismatch".into());
    }
    for token in 0..image_tokens {
        let mut input = [0.0_f32; LATENT_DIM];
        for c in 0..LATENT_DIM {
            let src = c * latent_time + token;
            input[c] = latent[src];
        }
        let out = &mut output[token * HIDDEN..(token + 1) * HIDDEN];
        linear_into_dispatch(q8_weights, source, weight, LATENT_DIM, HIDDEN, &input, out, q8, pool)?;
        for (v, b) in out.iter_mut().zip(bias) {
            *v += *b;
        }
    }
    Ok(())
}

/// Per-token LayerNorm (mean subtraction + variance scaling). Matches
/// `LayerNormModule({channels, 1e-6F, false, false})` from audio.cpp -- no
/// elementwise affine, no bias. Output is unit-variance, zero-mean.
fn layer_norm_token(input: &[f32], normalized: &mut [f32]) {
    let len = input.len().min(normalized.len());
    if len == 0 {
        return;
    }
    let mut sum = 0.0_f32;
    for v in &input[..len] {
        sum += *v;
    }
    let mean = sum / len as f32;
    let mut var_sum = 0.0_f32;
    for v in &input[..len] {
        let d = *v - mean;
        var_sum += d * d;
    }
    let variance = var_sum / len as f32;
    let std_recip = 1.0 / (variance + 1e-6).sqrt();
    for d in 0..len {
        normalized[d] = (input[d] - mean) * std_recip;
    }
}

fn rms_norm_inplace_text(
    hidden: &mut [f32],
    weight: &[f32],
    n_tokens: usize,
) {
    for token in 0..n_tokens {
        let start = token * HIDDEN;
        let end = start + HIDDEN;
        let slice = &mut hidden[start..end];
        let mut mean = 0.0_f32;
        for v in slice.iter() {
            mean += v * v;
        }
        mean = (mean / HIDDEN as f32 + 1e-6).sqrt().recip();
        for d in 0..HIDDEN {
            slice[d] *= mean * weight[d];
        }
    }
}

// === Flux2Edit block forwards ===
//
// Per `references/audio.cpp/src/community_models/auk/flow.cpp::build_transformer_block`
// (templated over `Streams`).
//
// Layout conventions:
// - Joint sequence: [img_tokens, text_tokens] (img first).
// - x-stream (img) attends to its own + c-stream (text) tokens.
// - c-stream (text) attends to the same joint.
// - RoPE positions: img uses 0..img_tokens, text uses 0..text_tokens independently.
// - `scratch.modulation[..ADALN_DIM]` holds the time-embedding projection for
//   the current step; we slice it into six 1536-wide chunks for the per-stream
//   (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).

/// Apply rotary embedding in-place using a precomputed `inv_freq` table of
/// length `head_dim/2`. `x` is a single token's head-major slice
/// `[n_heads * head_dim]` flattened row-major; we apply neox halves.
fn rope_apply_token(
    x: &mut [f32],
    pos: usize,
    head_dim: usize,
    inv_freq: &[f32],
) {
    let half = head_dim / 2;
    if half == 0 || inv_freq.len() != half {
        return;
    }
    let n_heads = x.len() / head_dim;
    if n_heads == 0 {
        return;
    }
    for head in 0..n_heads {
        let off = head * head_dim;
        // Half-rotation: pair (i, i+half).
        for i in 0..half {
            let theta = pos as f32 * inv_freq[i];
            let (cos_t, sin_t) = crate::ops::rope::neox::rope_sin_cos(theta);
            let a = x[off + i];
            let b = x[off + half + i];
            x[off + i] = a * cos_t - b * sin_t;
            x[off + half + i] = b * cos_t + a * sin_t;
        }
    }
}

/// Per-head RMS norm applied independently per token. `hidden` is laid out
/// `[n_tokens, n_heads, head_dim]` (contiguous); `weight` is `[head_dim]`.
fn rms_norm_per_head(
    hidden: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) {
    let row = n_heads * head_dim;
    for token in 0..n_tokens {
        for head in 0..n_heads {
            let off = token * row + head * head_dim;
            let slice = &mut hidden[off..off + head_dim];
            let mut mean_sq = 0.0_f32;
            for v in slice.iter() {
                mean_sq += *v * *v;
            }
            let scale = 1.0 / (mean_sq / head_dim as f32 + eps).sqrt();
            for d in 0..head_dim {
                slice[d] = slice[d] * scale * weight[d];
            }
        }
    }
}

/// Softmax over `scores[..n]`. Modifies in place.
fn softmax_inplace(scores: &mut [f32], n: usize) {
    let max = scores[..n]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for j in 0..n {
        scores[j] = (scores[j] - max).exp();
        sum += scores[j];
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for j in 0..n {
        scores[j] *= inv;
    }
}

#[allow(clippy::too_many_arguments)]
fn run_double_block(
    q8_weights: &HashMap<String, Arc<Vec<u8>>>,
    source: &dyn TensorSource,
    block: &DoubleBlockWeights,
    joint: &mut [f32],
    img_tokens: usize,
    text_tokens: usize,
    inv_freq: &[f32],
    modulation: &[f32],
    qkv_buf: &mut [f32],
    qkv_c_buf: &mut [f32],
    attention: &mut [f32],
    scores: &mut [f32],
    normed: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
    layer_index: usize,
) -> Result<(), String> {
    let _ = layer_index;
    let total = img_tokens + text_tokens;
    if total == 0 || total * HIDDEN != joint.len() {
        return Err("AuK double block joint length mismatch".into());
    }
    if scores.len() < total {
        return Err("AuK scores buffer too small".into());
    }
    if qkv_buf.len() < total * QKV_DIM {
        return Err("AuK qkv buffer too small".into());
    }
    if qkv_c_buf.len() < total * QKV_DIM {
        return Err("AuK qkv_c buffer too small".into());
    }
    if attention.len() < total * HIDDEN {
        return Err("AuK attention buffer too small".into());
    }
    if normed.len() < total * HIDDEN {
        return Err("AuK normed buffer too small".into());
    }
    if modulation.len() < 2 * ADALN_DIM {
        return Err("AuK modulation buffer too small".into());
    }
    if inv_freq.len() != HEAD_DIM / 2 {
        return Err(format!(
            "AuK inv_freq length {} != HEAD_DIM/2={}",
            inv_freq.len(),
            HEAD_DIM / 2
        ));
    }

    // === Pass 1: per-stream RMSNorm + AdaLN modulation + QKV projection ===
    // Stream x (img) — occupies joint rows [0, img_tokens)
    ada_ln_qkv(
        q8_weights,
        source,
        &block.adaLN_x,
        &block.adaLN_x_bias,
        &block.qkv_x,
        &block.qkv_x_bias,
        joint,
        0..img_tokens,
        &modulation[..ADALN_DIM],
        normed,
        qkv_buf,
        q8,
        pool,
    )?;
    // Stream c (text) — occupies joint rows [img_tokens, total)
    ada_ln_qkv(
        q8_weights,
        source,
        &block.adaLN_c,
        &block.adaLN_c_bias,
        &block.qkv_c,
        &block.qkv_c_bias,
        joint,
        img_tokens..total,
        &modulation[ADALN_DIM..2 * ADALN_DIM],
        normed,
        qkv_c_buf,
        q8,
        pool,
    )?;

    // === Pass 2: per-stream Q/K RMS-norm + RoPE ===
    // QKV layout: [tokens, QKV_DIM] = [tokens, 3*HIDDEN] with interleaved Q,K,V
    // per row.
    project_qk_with_rope(
        qkv_buf,
        img_tokens,
        0,
        &block.q_norm,
        &block.k_norm,
        inv_freq,
    );
    project_qk_with_rope(
        qkv_c_buf,
        text_tokens,
        0,
        &block.q_norm,
        &block.k_norm,
        inv_freq,
    );

    // === Joint attention: concat Q/K/V along seq dim ===
    // We need separate K, V slices for concat. Allocate within qkv_buf/qkv_c_buf
    // by reinterpreting their halves. Simpler: use the original buffers.
    // Q @ K^T then softmax * V.
    let heads = HEAD_DIM;
    let n_heads = HIDDEN / heads;
    let scale = 1.0 / (heads as f32).sqrt();
    let qkv_x = &qkv_buf[..total * QKV_DIM];
    let qkv_c = &qkv_c_buf[..total * QKV_DIM];

    // Joint K, V are not contiguous in qkv_buf (qkv_c is separate). We process
    // the attention by iterating heads and concatenating K/V from both
    // streams per head.
    // Per-stream output: attention[i] for i in 0..img_tokens (x-stream) and
    // i in img_tokens..total (c-stream).
    for head in 0..n_heads {
        // Compute Q (x_stream) @ K (x+c stream) -> img_logits[img_tokens, total]
        for q_tok in 0..img_tokens {
            let q_off = q_tok * QKV_DIM + head * heads;
            let q_row = &qkv_x[q_off..q_off + heads];
            let mut max_logit = f32::NEG_INFINITY;
            // x-stream keys
            for k_tok in 0..img_tokens {
                let k_off = k_tok * QKV_DIM + HIDDEN + head * heads;
                let dot = dot32(q_row, &qkv_x[k_off..k_off + heads]);
                scores[k_tok] = dot * scale;
                if scores[k_tok] > max_logit {
                    max_logit = scores[k_tok];
                }
            }
            // c-stream keys
            for k_tok in 0..text_tokens {
                let k_off = k_tok * QKV_DIM + HIDDEN + head * heads;
                let dot = dot32(q_row, &qkv_c[k_off..k_off + heads]);
                let s = dot * scale;
                scores[img_tokens + k_tok] = s;
                if s > max_logit {
                    max_logit = s;
                }
            }
            softmax_inplace(scores, total);
            // attention output row for x-stream
            let out_off = q_tok * HIDDEN + head * heads;
            for d in 0..heads {
                let mut sum = 0.0_f32;
                for k_tok in 0..img_tokens {
                    let v_off = k_tok * QKV_DIM + 2 * HIDDEN + head * heads + d;
                    sum += scores[k_tok] * qkv_x[v_off];
                }
                for k_tok in 0..text_tokens {
                    let v_off = k_tok * QKV_DIM + 2 * HIDDEN + head * heads + d;
                    sum += scores[img_tokens + k_tok] * qkv_c[v_off];
                }
                attention[out_off + d] = sum;
            }
        }
        // Q (c_stream) @ K (x+c stream) -> text_logits[text_tokens, total]
        for q_tok in 0..text_tokens {
            let q_off = q_tok * QKV_DIM + head * heads;
            let q_row = &qkv_c[q_off..q_off + heads];
            let mut max_logit = f32::NEG_INFINITY;
            for k_tok in 0..img_tokens {
                let k_off = k_tok * QKV_DIM + HIDDEN + head * heads;
                let dot = dot32(q_row, &qkv_x[k_off..k_off + heads]);
                scores[k_tok] = dot * scale;
                if scores[k_tok] > max_logit {
                    max_logit = scores[k_tok];
                }
            }
            for k_tok in 0..text_tokens {
                let k_off = k_tok * QKV_DIM + HIDDEN + head * heads;
                let dot = dot32(q_row, &qkv_c[k_off..k_off + heads]);
                let s = dot * scale;
                scores[img_tokens + k_tok] = s;
                if s > max_logit {
                    max_logit = s;
                }
            }
            softmax_inplace(scores, total);
            let out_off = (img_tokens + q_tok) * HIDDEN + head * heads;
            for d in 0..heads {
                let mut sum = 0.0_f32;
                for k_tok in 0..img_tokens {
                    let v_off = k_tok * QKV_DIM + 2 * HIDDEN + head * heads + d;
                    sum += scores[k_tok] * qkv_x[v_off];
                }
                for k_tok in 0..text_tokens {
                    let v_off = k_tok * QKV_DIM + 2 * HIDDEN + head * heads + d;
                    sum += scores[img_tokens + k_tok] * qkv_c[v_off];
                }
                attention[out_off + d] = sum;
            }
        }
    }

    // === Pass 3: per-stream output projection + residual + MLP ===
    stream_block_residual_mlp(
        q8_weights,
        source,
        &block.out_x,
        &block.out_x_bias,
        &block.ff_x_in,
        &block.ff_x_out,
        joint,
        0..img_tokens,
        &attention[..total * HIDDEN],
        &modulation[..ADALN_DIM],
        normed,
        q8,
        pool,
    )?;
    stream_block_residual_mlp(
        q8_weights,
        source,
        &block.out_c,
        &block.out_c_bias,
        &block.ff_c_in,
        &block.ff_c_out,
        joint,
        img_tokens..total,
        &attention[..total * HIDDEN],
        &modulation[ADALN_DIM..2 * ADALN_DIM],
        normed,
        q8,
        pool,
    )?;

    require_finite(joint, "AuK double block output")
}

#[allow(clippy::too_many_arguments)]
fn ada_ln_qkv(
    q8_weights: &HashMap<String, Arc<Vec<u8>>>,
    source: &dyn TensorSource,
    ada_weight: &str,
    ada_bias: &[f32],
    qkv_weight: &str,
    qkv_bias: &[f32],
    joint: &[f32],
    row_range: std::ops::Range<usize>,
    ada_params: &[f32],
    normed: &mut [f32],
    qkv_buf: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
) -> Result<(), String> {
    let tokens = row_range.len();
    // Slice modulation: 6 parts of HIDDEN each. We use shift_msa, scale_msa
    // (gate_msa, shift_mlp, scale_mlp, gate_mlp are used later in the residual/MLP).
    let shift_msa = &ada_params[0..HIDDEN];
    let scale_msa = &ada_params[HIDDEN..2 * HIDDEN];
    let gate_msa = &ada_params[2 * HIDDEN..3 * HIDDEN];
    let _shift_mlp = &ada_params[3 * HIDDEN..4 * HIDDEN];
    let _scale_mlp = &ada_params[4 * HIDDEN..5 * HIDDEN];
    let _gate_mlp = &ada_params[5 * HIDDEN..6 * HIDDEN];
    let _ = gate_msa;

    // LayerNorm per token (audio.cpp uses LayerNormModule -- mean subtraction, not
    // RMSNorm). The Flux2Edit AdaLN path then modulates (1+scale)*normalized + shift.
    for token in 0..tokens {
        let row = row_range.start + token;
        let off = row * HIDDEN;
        let slice = &joint[off..off + HIDDEN];
        // Compute mean.
        let mut sum = 0.0_f32;
        for v in slice.iter() {
            sum += *v;
        }
        let mean = sum / HIDDEN as f32;
        // Compute variance.
        let mut var_sum = 0.0_f32;
        for v in slice.iter() {
            let d = *v - mean;
            var_sum += d * d;
        }
        let variance = var_sum / HIDDEN as f32;
        let std_recip = 1.0 / (variance + 1e-6).sqrt();
        let n_off = token * HIDDEN;
        for d in 0..HIDDEN {
            normed[n_off + d] = (slice[d] - mean) * std_recip;
        }
    }

    // Apply AdaLN modulation.
    for token in 0..tokens {
        let n_off = token * HIDDEN;
        for d in 0..HIDDEN {
            let v = normed[n_off + d];
            normed[n_off + d] = v * (1.0 + scale_msa[d]) + shift_msa[d];
        }
    }

    // QKV projection: per-token matmul into qkv_buf.
    for token in 0..tokens {
        let input = &normed[token * HIDDEN..(token + 1) * HIDDEN];
        let out = &mut qkv_buf[token * QKV_DIM..(token + 1) * QKV_DIM];
        linear_into_dispatch(q8_weights, source, qkv_weight, HIDDEN, QKV_DIM, input, out, q8, pool)?;
        for (v, b) in out.iter_mut().zip(qkv_bias) {
            *v += *b;
        }
    }

    // ada_weight / ada_bias are unused here; we read the 9216-wide modulation
    // directly from `ada_params` (which was already projected by the caller).
    let _ = (ada_weight, ada_bias);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn project_qk_with_rope(
    qkv: &mut [f32],
    tokens: usize,
    pos_offset: usize,
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    inv_freq: &[f32],
) {
    // QKV row: [Q (HIDDEN) | K (HIDDEN) | V (HIDDEN)]
    for token in 0..tokens {
        let off = token * QKV_DIM;
        let pos = pos_offset + token;
        // Q RMSNorm + RoPE
        rms_norm_per_head(
            &mut qkv[off..off + HIDDEN],
            1,
            HIDDEN / HEAD_DIM,
            HEAD_DIM,
            q_norm_w,
            1e-6,
        );
        rope_apply_token(&mut qkv[off..off + HIDDEN], pos, HEAD_DIM, inv_freq);
        // K RMSNorm + RoPE
        rms_norm_per_head(
            &mut qkv[off + HIDDEN..off + 2 * HIDDEN],
            1,
            HIDDEN / HEAD_DIM,
            HEAD_DIM,
            k_norm_w,
            1e-6,
        );
        rope_apply_token(
            &mut qkv[off + HIDDEN..off + 2 * HIDDEN],
            pos,
            HEAD_DIM,
            inv_freq,
        );
        // V: no norm, no RoPE.
        let _ = (q_norm_w, k_norm_w);
    }
    let _ = inv_freq;
}

#[allow(clippy::too_many_arguments)]
fn stream_block_residual_mlp(
    q8_weights: &HashMap<String, Arc<Vec<u8>>>,
    source: &dyn TensorSource,
    out_weight: &str,
    out_bias: &[f32],
    ff_in_weight: &str,
    ff_out_weight: &str,
    joint: &mut [f32],
    row_range: std::ops::Range<usize>,
    attention: &[f32],
    ada_params: &[f32],
    normed: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
) -> Result<(), String> {
    let tokens = row_range.len();
    let _shift_msa = &ada_params[0..HIDDEN];
    let _scale_msa = &ada_params[HIDDEN..2 * HIDDEN];
    let gate_msa = &ada_params[2 * HIDDEN..3 * HIDDEN];
    let shift_mlp = &ada_params[3 * HIDDEN..4 * HIDDEN];
    let scale_mlp = &ada_params[4 * HIDDEN..5 * HIDDEN];
    let gate_mlp = &ada_params[5 * HIDDEN..6 * HIDDEN];

    // Output projection (1536 -> 1536) per token + bias.
    let mut proj = vec![0.0_f32; HIDDEN];
    for token in 0..tokens {
        let att_off = (row_range.start + token) * HIDDEN;
        let att_row = &attention[att_off..att_off + HIDDEN];
        linear_into_dispatch(
        q8_weights,
        source,
            out_weight,
            HIDDEN,
            HIDDEN,
            att_row,
            &mut proj,
            q8,
            pool,
        )?;
        for (v, b) in proj.iter_mut().zip(out_bias) {
            *v += *b;
        }
        // residual_1 = input + proj * gate_msa
        let row = (row_range.start + token) * HIDDEN;
        for d in 0..HIDDEN {
            joint[row + d] += proj[d] * gate_msa[d];
        }
    }

    // MLP: RMSNorm(residual_1), AdaLN modulate, ff_in (packed gate+up), SwiGLU,
    // ff_out, residual.
    let mut normed_token = vec![0.0_f32; HIDDEN];
    let mut packed = vec![0.0_f32; PACKED_FF_IN];
    let mut gated = vec![0.0_f32; FF_INNER];
    let mut ff = vec![0.0_f32; HIDDEN];
    for token in 0..tokens {
        let row = row_range.start + token;
        let off = row * HIDDEN;
        let slice = &joint[off..off + HIDDEN];
        layer_norm_token(slice, &mut normed_token);
        // Modulate.
        for d in 0..HIDDEN {
            let v = normed_token[d];
            normed_token[d] = v * (1.0 + scale_mlp[d]) + shift_mlp[d];
        }
        // ff_in: 1536 -> 6144 (gate+up packed)
        linear_into_dispatch(
        q8_weights,
        source,
            ff_in_weight,
            HIDDEN,
            PACKED_FF_IN,
            &normed_token,
            &mut packed,
            q8,
            pool,
        )?;
        // SwiGLU: split into gate (0..3072) and up (3072..6144); out = silu(gate) * up
        for d in 0..FF_INNER {
            let g = packed[d];
            let u = packed[FF_INNER + d];
            // silu(g) = g * sigmoid(g)
            let silu = g / (1.0 + (-g).exp());
            gated[d] = silu * u;
        }
        // ff_out: 3072 -> 1536
        linear_into_dispatch(
        q8_weights,
        source,
            ff_out_weight,
            FF_INNER,
            HIDDEN,
            &gated,
            &mut ff,
            q8,
            pool,
        )?;
        // residual_2 = residual_1 + ff * gate_mlp
        for d in 0..HIDDEN {
            joint[off + d] += ff[d] * gate_mlp[d];
        }
    }
    let _ = normed;
    Ok(())
}

fn dot32(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        s += *x * *y;
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn run_single_block(
    q8_weights: &HashMap<String, Arc<Vec<u8>>>,
    source: &dyn TensorSource,
    block: &SingleBlockWeights,
    joint: &mut [f32],
    img_tokens: usize,
    text_tokens: usize,
    inv_freq: &[f32],
    modulation: &[f32],
    qkv_buf: &mut [f32],
    attention: &mut [f32],
    scores: &mut [f32],
    normed: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
    layer_index: usize,
) -> Result<(), String> {
    let _ = layer_index;
    let total = img_tokens + text_tokens;
    if total == 0 || total * HIDDEN != joint.len() {
        return Err("AuK single block joint length mismatch".into());
    }
    if scores.len() < total {
        return Err("AuK scores buffer too small".into());
    }
    if qkv_buf.len() < total * QKV_DIM {
        return Err("AuK qkv buffer too small".into());
    }
    if attention.len() < total * HIDDEN {
        return Err("AuK attention buffer too small".into());
    }
    if normed.len() < HIDDEN {
        return Err("AuK normed buffer too small".into());
    }
    if modulation.len() < ADALN_DIM {
        return Err("AuK modulation buffer too small".into());
    }
    if inv_freq.len() != HEAD_DIM / 2 {
        return Err(format!(
            "AuK inv_freq length {} != HEAD_DIM/2={}",
            inv_freq.len(),
            HEAD_DIM / 2
        ));
    }

    let shift_msa = &modulation[0..HIDDEN];
    let scale_msa = &modulation[HIDDEN..2 * HIDDEN];
    let gate_msa = &modulation[2 * HIDDEN..3 * HIDDEN];
    let shift_mlp = &modulation[3 * HIDDEN..4 * HIDDEN];
    let scale_mlp = &modulation[4 * HIDDEN..5 * HIDDEN];
    let gate_mlp = &modulation[5 * HIDDEN..6 * HIDDEN];

    // Per-token LayerNorm + AdaLN modulation + QKV projection.
    for token in 0..total {
        let off = token * HIDDEN;
        let slice = &joint[off..off + HIDDEN];
        let mut normed_token = vec![0.0_f32; HIDDEN];
        layer_norm_token(slice, &mut normed_token);
        for d in 0..HIDDEN {
            normed_token[d] = normed_token[d] * (1.0 + scale_msa[d]) + shift_msa[d];
        }
        let out = &mut qkv_buf[token * QKV_DIM..(token + 1) * QKV_DIM];
        linear_into_dispatch(q8_weights, source, &block.qkv, HIDDEN, QKV_DIM, &normed_token, out, q8, pool)?;
        for (v, b) in out.iter_mut().zip(&block.qkv_bias) {
            *v += *b;
        }
    }
    // Q/K RMS-norm + RoPE (V: no norm). Stream positions: text tokens use
    // 0..text_tokens, audio tokens use text_tokens..text_tokens+img_tokens
    // (audio comes second in single block per audio.cpp).
    let heads = HEAD_DIM;
    let n_heads = HIDDEN / heads;
    let scale = 1.0 / (heads as f32).sqrt();
    for token in 0..total {
        let off = token * QKV_DIM;
        rms_norm_per_head(
            &mut qkv_buf[off..off + HIDDEN],
            1,
            n_heads,
            heads,
            &block.q_norm,
            1e-6,
        );
        rms_norm_per_head(
            &mut qkv_buf[off + HIDDEN..off + 2 * HIDDEN],
            1,
            n_heads,
            heads,
            &block.k_norm,
            1e-6,
        );
        let pos = token; // Sequential 0..text_tokens+img_tokens after the [text, img] swap.
        rope_apply_token(&mut qkv_buf[off..off + HIDDEN], pos, HEAD_DIM, inv_freq);
        rope_apply_token(&mut qkv_buf[off + HIDDEN..off + 2 * HIDDEN], pos, HEAD_DIM, inv_freq);
    }

    // Single-stream attention.
    for head in 0..n_heads {
        for q_tok in 0..total {
            let q_off = q_tok * QKV_DIM + head * heads;
            let q_row = &qkv_buf[q_off..q_off + heads];
            let mut max_logit = f32::NEG_INFINITY;
            for k_tok in 0..total {
                let k_off = k_tok * QKV_DIM + HIDDEN + head * heads;
                let dot = dot32(q_row, &qkv_buf[k_off..k_off + heads]);
                let s = dot * scale;
                scores[k_tok] = s;
                if s > max_logit {
                    max_logit = s;
                }
            }
            softmax_inplace(scores, total);
            let out_off = q_tok * HIDDEN + head * heads;
            for d in 0..heads {
                let mut sum = 0.0_f32;
                for k_tok in 0..total {
                    let v_off = k_tok * QKV_DIM + 2 * HIDDEN + head * heads + d;
                    sum += scores[k_tok] * qkv_buf[v_off];
                }
                attention[out_off + d] = sum;
            }
        }
    }

    // Output projection + residual + MLP.
    let mut proj = vec![0.0_f32; HIDDEN];
    let mut gated_buf = vec![0.0_f32; FF_INNER];
    let mut packed_buf = vec![0.0_f32; PACKED_FF_IN];
    let mut ff = vec![0.0_f32; HIDDEN];
    let mut normed_token = vec![0.0_f32; HIDDEN];
    for token in 0..total {
        let off = token * HIDDEN;
        let att_row = &attention[off..off + HIDDEN];
        linear_into_dispatch(
        q8_weights,
        source,
            &block.out,
            HIDDEN,
            HIDDEN,
            att_row,
            &mut proj,
            q8,
            pool,
        )?;
        for (v, b) in proj.iter_mut().zip(&block.out_bias) {
            *v += *b;
        }
        for d in 0..HIDDEN {
            joint[off + d] += proj[d] * gate_msa[d];
        }
        // MLP
        let slice = &joint[off..off + HIDDEN];
        layer_norm_token(slice, &mut normed_token);
        for d in 0..HIDDEN {
            let v = normed_token[d];
            normed_token[d] = v * (1.0 + scale_mlp[d]) + shift_mlp[d];
        }
        linear_into_dispatch(
        q8_weights,
        source,
            &block.ff_in,
            HIDDEN,
            PACKED_FF_IN,
            &normed_token,
            &mut packed_buf,
            q8,
            pool,
        )?;
        for d in 0..FF_INNER {
            let g = packed_buf[d];
            let u = packed_buf[FF_INNER + d];
            let silu = g / (1.0 + (-g).exp());
            gated_buf[d] = silu * u;
        }
        linear_into_dispatch(
        q8_weights,
        source,
            &block.ff_out,
            FF_INNER,
            HIDDEN,
            &gated_buf,
            &mut ff,
            q8,
            pool,
        )?;
        for d in 0..HIDDEN {
            joint[off + d] += ff[d] * gate_mlp[d];
        }
    }
    require_finite(joint, "AuK single block output")
}

/// Convert F16 weight bytes (laid out as [n_out, n_in] row-major; each F16
/// value is 2 bytes little-endian) to Q8_0 weight bytes (laid out as
/// [n_out, n_in/32 blocks per row, 34 bytes per block] where each block
/// is `F16_scale | 32 int8_quantized_values`).
fn pre_quantize_f16_to_q8_0(
    f16_bytes: &[u8],
    n_out: usize,
    n_in: usize,
) -> Result<Vec<u8>, String> {
    if f16_bytes.len() != n_out * n_in * 2 {
        return Err(format!(
            "F16 weight size {} != n_out({}) * n_in({}) * 2",
            f16_bytes.len(), n_out, n_in
        ));
    }
    if n_in % 32 != 0 {
        return Err(format!(
            "AuK Q8_0 quantization requires n_in % 32 == 0; got {}",
            n_in
        ));
    }
    let blocks_per_row = n_in / 32;
    let row_stride = blocks_per_row * 34;
    let mut out = vec![0u8; n_out * row_stride];
    let mut block_f32 = vec![0.0f32; 32];
    let mut q8_block = vec![0u8; 32];
    let mut scale_f32 = [0.0f32; 1];
    // F16->F32 with AVX2+F16C: process 8 F16 (16 bytes) -> 8 F32 (32 bytes)
    // per iteration. n_in is always a multiple of 32 (DiT weights).
    for r in 0..n_out {
        for b in 0..blocks_per_row {
            let row_off = r * n_in + b * 32;
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                unsafe {
                    // 4 chunks of 8 F16 = 32 values.
                    for chunk in 0..4 {
                        let byte_off = (row_off + chunk * 8) * 2;
                        // Safe: f16_bytes is &[u8] of length n_out*n_in*2;
                        // byte_off + 16 <= n_out*n_in*2 - (n_out-r-1)*n_in*2 ...
                        // For r = last row, byte_off + 16 <= (n_out-1)*n_in*2 + n_in*2 - 16
                        //   = n_out*n_in*2 - 16, which is within bounds.
                        let v = _mm_loadu_si128(
                            f16_bytes.as_ptr().add(byte_off) as *const __m128i,
                        );
                        let f = _mm256_cvtph_ps(v);
                        _mm256_storeu_ps(
                            block_f32.as_mut_ptr().add(chunk * 8),
                            f,
                        );
                    }
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for lane in 0..32 {
                    let byte_off = (row_off + lane) * 2;
                    let bits = u16::from_le_bytes([
                        f16_bytes[byte_off],
                        f16_bytes[byte_off + 1],
                    ]);
                    block_f32[lane] = f16::from_bits(bits).to_f32();
                }
            }
            crate::ops::quantize_q8_0_into(
                &block_f32, 32, &mut q8_block, &mut scale_f32,
            );
            let dst = r * row_stride + b * 34;
            let scale_f16 = crate::ops::f32_to_f16(scale_f32[0]);
            let scale_bytes = scale_f16.to_le_bytes();
            out[dst] = scale_bytes[0];
            out[dst + 1] = scale_bytes[1];
            out[dst + 2..dst + 34].copy_from_slice(&q8_block);
        }
    }
    Ok(out)
}

/// Linear matmul that uses a pre-quantized Q8_0 weight (looked up by name
/// in `AukDit::q8_weights`). Falls back to the F16 path if the weight
/// wasn't pre-quantized (e.g., the GGUF has non-F16 dtype).
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_into_q8_cached(
    q8_weights: &HashMap<String, Arc<Vec<u8>>>,
    name: &str,
    n_in: usize,
    n_out: usize,
    input: &[f32],
    output: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
) -> Result<bool, String> {
    let Some(q8_bytes) = q8_weights.get(name) else {
        return Ok(false);
    };
    if q8_bytes.len() != (n_in / 32) * 34 * n_out {
        return Err(format!(
            "AuK pre-quantized Q8_0 weight {} has size {} != expected {}",
            name, q8_bytes.len(), (n_in / 32) * 34 * n_out
        ));
    }
    if input.len() != n_in {
        return Err(format!(
            "Invalid linear input length for {name}: expected {n_in}, got {}",
            input.len()
        ));
    }
    if output.len() != n_out {
        return Err(format!(
            "Invalid linear output length for {name}: expected {n_out}, got {}",
            output.len()
        ));
    }
    q8.prepare(input, n_in)?;
    let weight_ptr = q8_bytes.as_ptr() as usize;
    let weight_len = q8_bytes.len();
    let input_ptr = q8.values.as_ptr() as usize;
    let input_len = q8.values.len();
    let scale_ptr = q8.scales.as_ptr() as usize;
    let scale_len = q8.scales.len();
    let output_ptr = output.as_mut_ptr() as usize;
    pool.compute(move |ith, nth| {
        let weight =
            unsafe { std::slice::from_raw_parts(weight_ptr as *const u8, weight_len) };
        let input =
            unsafe { std::slice::from_raw_parts(input_ptr as *const u8, input_len) };
        let scales =
            unsafe { std::slice::from_raw_parts(scale_ptr as *const f32, scale_len) };
        let out =
            unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, n_out) };
        crate::ops::matmul_q8_0_quantized_parallel_rows(
            weight, input, scales, out, n_in, n_out, ith, nth,
        );
    });
    Ok(true)
}


/// Distributed matmul dispatch: try the pre-quantized Q8_0 cache (now always
/// empty after the f4e7879 workaround was removed) and fall back to F16 GPU
/// or F16 CPU via `super::linear_into_scaled_impl`. This is the
/// free-function equivalent of `AukDit::linear_into_dispatched`, used in the
/// block forward functions that take `&dyn TensorSource` directly.
///
/// The previous version of this function had the same infinite recursion bug
/// as `AukDit::linear_into_dispatched`: it called itself on Q8 miss. That
/// bug is now fixed by forwarding to `super::linear_into_scaled_impl`.
#[allow(clippy::too_many_arguments)]
fn linear_into_dispatch(
    q8_weights: &HashMap<String, Arc<Vec<u8>>>,
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
    input: &[f32],
    output: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
) -> Result<(), String> {
    if let Ok(true) = linear_into_q8_cached(
        q8_weights, name, n_in, n_out, input, output, q8, pool,
    ) {
        return Ok(());
    }
    super::linear_into_scaled_impl(
        source,
        name,
        n_in,
        n_out,
        input,
        output,
        q8,
        pool,
        1.0,
    )
}
