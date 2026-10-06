//! AuK-Base 1.5B speech DiT port.
//!
//! Reference: `references/audio.cpp/src/community_models/auk/{flow,conditioning,
//! vae,audio_conditioning,session}.cpp` (116 KB C++ implementation by
//! 0xShug0). Architecture: `Flux2Edit` (Flux-style double + single blocks with
//! fused QKV, packed FF gate+up, 6-way AdaLN modulation, joint img/txt
//! attention in double blocks).
//!
//! Architecture constants (verified against `auk-base-f16.gguf`):
//! - `dim=1536, heads=24, head_dim=64`
//! - `ff_inner=3072, packed_ff_in=6144 = 2*ff_inner` (gate+up packed)
//! - `qkv_dim=4608 = 3*hidden` (fused Q+K+V)
//! - `AdaLN out=9216 = 6*hidden` (6-way modulation, like Flux)
//! - `freq_dim=256` for timestep sinusoidal embedding
//! - 10 double blocks + 20 single blocks
//! - `text_in=2048` (Qwen2.5-Omni-3B n_embd)
//! - `latent_dim=64`, `downsample_rate=480`, `sample_rate=24000 Hz`
//!
//! Tensor layout (F16/BF16 GGUF): weights stored as `[in_dim, out_dim]`,
//! matching our other repos. `linear_into(n_in, n_out)` reads the 2-D layout
//! `[n_in, n_out]` directly -- the audio.cpp C++ uses `(out, in)` argument
//! order but the GGUF is `[in, out]` so the matmul direction agrees.
//!
//! This scaffold implements the TTS-only path first (no reference audio
//! conditioning); the CFMEdit reference-audio path is tracked in
//! `docs/develop/TODO.md` for a follow-up commit.

use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{Kernel, QTensorOwned};
use crate::ops::matmul_q8_0_quantized_parallel_rows;
#[cfg(feature = "vulkan")]
use crate::vulkan::ops::{BatchedLinearRuntime, GpuWeightFormat};
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanContext;
use dit::TEXT_IN;
use std::sync::Arc;

pub(crate) mod dit;
pub(crate) mod text;
pub(crate) mod vae;

pub struct AukAudio {
    pub sample_rate: u32,
    pub samples: Vec<f64>,
    pub channels: u32,
}

impl AukAudio {
    /// Pitch shift by `semitones`. **Changes both pitch and duration by
    /// the same factor** (output duration = input duration / `2^(semitones/12)`).
    /// This is the same primitive as [`speed_change`](Self::speed_change)
    /// with `rate = 2^(semitones/12)` -- a true "preserve-duration" pitch
    /// shift would need a phase vocoder which we don't have. Output
    /// sample_rate is preserved; if you want to play the result at the
    /// original duration, just resample up by `2^(semitones/12)` in your
    /// WAV writer.
    pub fn pitch_shift(&self, semitones: f32) -> AukAudio {
        self.speed_change((semitones / 12.0).exp2())
    }

    /// Speed change by `rate` (1.0 = unchanged, 1.5 = 1.5x faster).
    /// Output duration is divided by `rate`; sample_rate is preserved.
    /// Uses linear interpolation.
    pub fn speed_change(&self, rate: f32) -> AukAudio {
        if rate == 1.0 || self.samples.is_empty() {
            return AukAudio {
                sample_rate: self.sample_rate,
                samples: self.samples.clone(),
                channels: self.channels,
            };
        }
        let input = &self.samples;
        let output_len = ((input.len() as f64) / rate as f64).round() as usize;
        let mut output = Vec::with_capacity(output_len);
        for i in 0..output_len {
            let src_pos = i as f64 * rate as f64;
            let lo = src_pos.floor() as usize;
            let hi = (lo + 1).min(input.len() - 1);
            let frac = (src_pos - lo as f64) as f32;
            let sample = input[lo] as f32 * (1.0 - frac) + input[hi] as f32 * frac;
            output.push(sample as f64);
        }
        AukAudio {
            sample_rate: self.sample_rate,
            samples: output,
            channels: self.channels,
        }
    }

    /// Volume change by `db` decibels (0 = unchanged, +10 = 10x amplitude).
    /// Output duration and sample_rate are preserved.
    pub fn volume_change(&self, db: f32) -> AukAudio {
        if db == 0.0 || self.samples.is_empty() {
            return AukAudio {
                sample_rate: self.sample_rate,
                samples: self.samples.clone(),
                channels: self.channels,
            };
        }
        let factor = (10.0f32).powf(db / 20.0);
        let output: Vec<f64> = self.samples.iter().map(|&s| s * factor as f64).collect();
        AukAudio {
            sample_rate: self.sample_rate,
            samples: output,
            channels: self.channels,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AukOptions {
    pub steps: usize,
    pub sample_rate: u32,
    pub duration_sec: usize,
    pub seed: i64,
    pub guidance_scale: f32,
    /// Optional instruct string for instruct-TTS. When `Some`, the prompt
    /// template is rebuilt as `<instruct=...>\n<text>` per audio.cpp's
    /// instruct-TTS path. When `None`, the prompt is the raw `text`
    /// passed to `generate_audio` (TTS without instruct, matching
    /// audio.cpp's `<|no_prompt_audio|>` fallback).
    pub instruct: Option<String>,
}

pub struct AukPipeline {
    dit: dit::AukDit,
    vae: vae::BigVGANFlowVae,
    text: Option<text::AukTextEncoder>,
    /// Optional Qwen2.5-Omni audio tower for CFMEdit reference-audio conditioning.
    /// Loaded lazily from a separate BF16 Qwen GGUF (which contains the audio
    /// tower in addition to the text trunk). Required only for end-to-end
    /// CFMEdit (WAV -> mel -> audio tower -> DiT). The Q8_0 Qwen GGUF is
    /// insufficient since it omits the audio tower.
    audio_tower: Option<crate::models::qwen3::omni_audio::AudioTowerModel>,
}

impl AukPipeline {
    pub fn load(
        diffusion: Arc<dyn TensorSource>,
        vae_source: Arc<dyn TensorSource>,
        text_source: Option<Arc<dyn TensorSource>>,
        n_threads: usize,
    ) -> Result<Self, String> {
        Self::load_with_audio_tower(diffusion, vae_source, text_source, None, n_threads)
    }

    /// Load pipeline with optional Qwen2.5-Omni audio tower (BF16 GGUF).
    /// When `qwen_omni_bf16_source` is provided, the audio tower is loaded
    /// from it and `generate_audio_with_reference_wav` becomes available.
    /// When it's `None`, the pipeline is TTS-only (use `generate_audio` or
    /// `generate_audio_with_audio` with pre-computed audio embeddings).
    pub fn load_with_audio_tower(
        diffusion: Arc<dyn TensorSource>,
        vae_source: Arc<dyn TensorSource>,
        text_source: Option<Arc<dyn TensorSource>>,
        qwen_omni_bf16_source: Option<Arc<dyn TensorSource>>,
        n_threads: usize,
    ) -> Result<Self, String> {
        validate_component(diffusion.as_ref(), Component::Dit)?;
        let pool = Arc::new(ComputePool::new(n_threads.max(1)));
        let text = match text_source {
            Some(src) => Some(text::AukTextEncoder::load(src, Arc::clone(&pool))?),
            None => None,
        };
        let audio_tower = match qwen_omni_bf16_source {
            Some(src) => {
                eprintln!("[AukDiT] loading Qwen2.5-Omni audio tower (BF16 GGUF)...");
                let t = std::time::Instant::now();
                let tower = crate::models::qwen3::omni_audio::AudioTowerModel::from_source(src)?;
                eprintln!(
                    "[AukDiT] audio tower loaded in {:.1}s",
                    t.elapsed().as_secs_f64()
                );
                Some(tower)
            }
            None => None,
        };
        Ok(Self {
            dit: dit::AukDit::load(diffusion, Arc::clone(&pool))?,
            vae: vae::BigVGANFlowVae::load(vae_source, pool)?,
            text,
            audio_tower,
        })
    }

    pub fn generate_audio(&self, prompt: &str, options: &AukOptions) -> Result<AukAudio, String> {
        let total_start = std::time::Instant::now();
        // Apply instruct prefix to the bare prompt (matches audio.cpp's
        // instruct-TTS path). We delegate to encode_with_audio with
        // audio_count=0 so the template + tokenization stays centralized.
        let (text_conditioning, text_tokens) = match self.text.as_ref() {
            Some(encoder) => {
                let t = std::time::Instant::now();
                let instruct_ref = options.instruct.as_deref();
                let hidden = encoder.encode_with_audio(prompt, 0, &[], instruct_ref)?;
                let tokens = encoder.last_token_count(&hidden);
                eprintln!(
                    "[auk-stage-profile] text_encode={:.1}ms  n_tokens={}  instruct={}",
                    t.elapsed().as_secs_f64() * 1000.0,
                    tokens,
                    if instruct_ref.is_some() { "yes" } else { "no" },
                );
                (hidden, tokens)
            }
            None => return Err("AuK requires --text-encoder (Qwen2.5-Omni-3B) for TTS".into()),
        };
        // No reference audio conditioning in this path (audio_conditioning
        // is exposed via `generate_audio_with_audio`).
        let t = std::time::Instant::now();
        let latent = self
            .dit
            .denoise(&text_conditioning, text_tokens, &[], 0, options)?;
        let t_denoise = t.elapsed();
        let t = std::time::Instant::now();
        let audio = {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut sum_sq = 0.0_f64;
            for v in &latent {
                if *v < min {
                    min = *v;
                }
                if *v > max {
                    max = *v;
                }
                sum_sq += (*v as f64) * (*v as f64);
            }
            let rms = (sum_sq / latent.len() as f64).sqrt();
            eprintln!(
                "[auk-stage-profile] latent_stats min={:.3} max={:.3} rms={:.3}",
                min, max, rms
            );
            self.vae.decode(&latent, options.sample_rate)?
        };
        let t_vae = t.elapsed();
        eprintln!(
            "[auk-stage-profile] denoise={:.1}ms  vae_decode={:.1}ms  total={:.1}ms",
            t_denoise.as_secs_f64() * 1000.0,
            t_vae.as_secs_f64() * 1000.0,
            total_start.elapsed().as_secs_f64() * 1000.0,
        );
        Ok(audio)
    }

    /// Generate audio with optional CFMEdit reference-audio conditioning.
    ///
    /// `audio_conditioning` is a flat `audio_tokens * TEXT_IN` f32 buffer in
    /// Qwen2.5-Omni's hidden space (TEXT_IN=2048). When non-empty, these
    /// embeddings are prepended to the text encoder output and form the
    /// conditioning sequence the DiT attends to. Layout:
    ///
    /// ```text
    /// joint = [img_tokens | audio_tokens | text_tokens]
    /// ```
    ///
    /// The audio embeddings are expected to come from Qwen2.5-Omni-3B's
    /// audio tower (thinker.audio_tower.*), but the audio tower encoder
    /// itself is NOT yet ported here. To use this API today, supply
    /// pre-computed audio embeddings (e.g. exported from a Python run of
    /// the Qwen audio tower on a 16 kHz reference WAV).
    ///
    /// For unconditional reference-audio conditioning, pass `&[]` with
    /// `audio_tokens = 0` (equivalent to TTS without reference audio).
    pub fn generate_audio_with_audio(
        &self,
        prompt: &str,
        audio_conditioning: &[f32],
        audio_tokens: usize,
        options: &AukOptions,
    ) -> Result<AukAudio, String> {
        let total_start = std::time::Instant::now();
        let instruct_ref = options.instruct.as_deref();
        let (text_conditioning, text_tokens) = match self.text.as_ref() {
            Some(encoder) => {
                let t = std::time::Instant::now();
                let hidden = encoder.encode_with_audio(prompt, 0, &[], instruct_ref)?;
                let tokens = encoder.last_token_count(&hidden);
                eprintln!(
                    "[auk-stage-profile] text_encode={:.1}ms  n_text_tokens={} n_audio_tokens={}",
                    t.elapsed().as_secs_f64() * 1000.0,
                    tokens,
                    audio_tokens,
                );
                (hidden, tokens)
            }
            None => return Err("AuK requires --text-encoder (Qwen2.5-Omni-3B) for TTS".into()),
        };
        // Concatenate audio + text embeddings: audio first, then text.
        // audio_conditioning is in Qwen2.5-Omni's 2048-dim hidden space
        // (same as text encoder output), so the two concatenate cleanly.
        let combined;
        let combined_slice: &[f32] = if audio_tokens > 0 {
            if audio_conditioning.len() != audio_tokens * TEXT_IN {
                return Err(format!(
                    "AuK audio conditioning length {} != audio_tokens*TEXT_IN={}",
                    audio_conditioning.len(),
                    audio_tokens * TEXT_IN
                ));
            }
            combined = {
                let mut buf = Vec::with_capacity((audio_tokens + text_tokens) * TEXT_IN);
                buf.extend_from_slice(audio_conditioning);
                buf.extend_from_slice(&text_conditioning);
                buf
            };
            &combined
        } else {
            &text_conditioning
        };
        let t = std::time::Instant::now();
        let latent = self.dit.denoise(
            combined_slice,
            text_tokens,
            audio_conditioning,
            audio_tokens,
            options,
        )?;
        let t_denoise = t.elapsed();
        let t = std::time::Instant::now();
        let audio = {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut sum_sq = 0.0_f64;
            for v in &latent {
                if *v < min {
                    min = *v;
                }
                if *v > max {
                    max = *v;
                }
                sum_sq += (*v as f64) * (*v as f64);
            }
            let rms = (sum_sq / latent.len() as f64).sqrt();
            eprintln!(
                "[auk-stage-profile] latent_stats min={:.3} max={:.3} rms={:.3}",
                min, max, rms
            );
            self.vae.decode(&latent, options.sample_rate)?
        };
        let t_vae = t.elapsed();
        eprintln!(
            "[auk-stage-profile] denoise={:.1}ms  vae_decode={:.1}ms  total={:.1}ms",
            t_denoise.as_secs_f64() * 1000.0,
            t_vae.as_secs_f64() * 1000.0,
            total_start.elapsed().as_secs_f64() * 1000.0,
        );
        Ok(audio)
    }

    /// End-to-end CFMEdit: take a reference WAV file, run it through the
    /// Qwen2.5-Omni audio tower, then have the Qwen text encoder run with
    /// the audio embeddings substituted for the `<|AUDIO|>` token positions
    /// (text and audio attend to each other inside the Qwen trunk --
    /// matching audio.cpp's `conditioning.cpp::build_text_conditioning`).
    /// The fused per-token hidden states then drive the DiT.
    ///
    /// `wav_samples_16k_mono` must be 16 kHz mono PCM f32 (use
    /// `crate::app::media::decode_audio` to load + resample any input
    /// format). For very long audio, the audio tower chunks into 200-frame
    /// windows internally.
    ///
    /// Requires the audio tower to be loaded (via `load_with_audio_tower`).
    pub fn generate_audio_with_reference_wav(
        &self,
        prompt: &str,
        wav_samples_16k_mono: &[f32],
        options: &AukOptions,
    ) -> Result<AukAudio, String> {
        let tower = self.audio_tower.as_ref().ok_or_else(|| {
            "AukPipeline::generate_audio_with_reference_wav requires the audio tower; \
             load the pipeline via AukPipeline::load_with_audio_tower with a Qwen2.5-Omni \
             BF16 GGUF (the Q8_0 GGUF omits the audio tower)"
                .to_string()
        })?;
        if wav_samples_16k_mono.is_empty() {
            return Err("Reference WAV is empty".into());
        }
        let t = std::time::Instant::now();
        let (audio_embeddings, audio_tokens) = tower
            .encode_pcm(wav_samples_16k_mono)
            .map_err(|e| format!("Qwen2.5-Omni audio tower encode: {e}"))?;
        eprintln!(
            "[auk-stage-profile] audio_tower_encode={:.1}ms  n_audio_tokens={}",
            t.elapsed().as_secs_f64() * 1000.0,
            audio_tokens,
        );
        self.generate_audio_cfmedit(prompt, &audio_embeddings, audio_tokens, options)
    }

    /// CFMEdit core: build the CFMEdit prompt template (system + user +
    /// audio_bos + audio placeholders + audio_eos + assistant), substitute
    /// the audio tower's per-frame embeddings for the `<|AUDIO|>` positions,
    /// run the Qwen transformer (text attends to audio via self-attention),
    /// then drive the DiT with the fused hidden states.
    pub fn generate_audio_cfmedit(
        &self,
        prompt: &str,
        audio_embeddings: &[f32],
        audio_tokens: usize,
        options: &AukOptions,
    ) -> Result<AukAudio, String> {
        let total_start = std::time::Instant::now();
        let encoder = self.text.as_ref().ok_or_else(|| {
            "AuK CFMEdit requires the Qwen2.5-Omni text encoder (--text-encoder)".to_string()
        })?;

        let t = std::time::Instant::now();
        let instruct_ref = options.instruct.as_deref();
        let (cond_hidden, total_cond_tokens) = if audio_tokens == 0 {
            // Pure TTS path (no reference audio) -- use the bare-prompt
            // encoder so we get the same conditioning as generate_audio().
            let hidden = encoder.encode_with_audio(prompt, 0, &[], instruct_ref)?;
            let tokens = encoder.last_token_count(&hidden);
            (hidden, tokens)
        } else {
            let hidden =
                encoder.encode_with_audio(prompt, audio_tokens, audio_embeddings, instruct_ref)?;
            let tokens = encoder.last_token_count(&hidden);
            (hidden, tokens)
        };
        eprintln!(
            "[auk-stage-profile] qwen_cfmedit_encode={:.1}ms  total_cond_tokens={}",
            t.elapsed().as_secs_f64() * 1000.0,
            total_cond_tokens,
        );

        let t = std::time::Instant::now();
        let latent = self
            .dit
            .denoise(&cond_hidden, total_cond_tokens, &[], 0, options)?;
        let t_denoise = t.elapsed();
        let t = std::time::Instant::now();
        let audio = {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut sum_sq = 0.0_f64;
            for v in &latent {
                if *v < min {
                    min = *v;
                }
                if *v > max {
                    max = *v;
                }
                sum_sq += (*v as f64) * (*v as f64);
            }
            let rms = (sum_sq / latent.len() as f64).sqrt();
            eprintln!(
                "[auk-stage-profile] latent_stats min={:.3} max={:.3} rms={:.3}",
                min, max, rms
            );
            self.vae.decode(&latent, options.sample_rate)?
        };
        let t_vae = t.elapsed();
        eprintln!(
            "[auk-stage-profile] denoise={:.1}ms  vae_decode={:.1}ms  total={:.1}ms",
            t_denoise.as_secs_f64() * 1000.0,
            t_vae.as_secs_f64() * 1000.0,
            total_start.elapsed().as_secs_f64() * 1000.0,
        );
        Ok(audio)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Component {
    Dit,
    Vae,
}

pub(crate) fn validate_component(
    source: &dyn TensorSource,
    component: Component,
) -> Result<(), String> {
    match component {
        Component::Dit => validate_dit(source),
        Component::Vae => validate_vae(source),
    }
}

fn validate_dit(source: &dyn TensorSource) -> Result<(), String> {
    let hidden = dit::HIDDEN as u64;
    let qkv = dit::QKV_DIM as u64;
    let ff_packed = dit::PACKED_FF_IN as u64;
    let ff_inner = dit::FF_INNER as u64;
    let mod_dim = dit::ADALN_DIM as u64;
    // Audio embed: latent 64 -> hidden 1536
    require_matrix(
        source,
        "transformer.audio_embed.linear.weight",
        &[64, hidden],
    )?;
    // Time embed: freq_dim 256 -> hidden -> hidden
    require_matrix(
        source,
        "transformer.time_embed.time_mlp.0.weight",
        &[dit::FREQ_DIM as u64, hidden],
    )?;
    require_matrix(
        source,
        "transformer.time_embed.time_mlp.2.weight",
        &[hidden, hidden],
    )?;
    // Text in: text_hidden 2048 -> hidden
    require_matrix(
        source,
        "transformer.txt_proj.weight",
        &[dit::TEXT_IN as u64, hidden],
    )?;
    // txt_norm: 1-D, hidden (F16 in the unsloth F16 GGUF)
    require_norm_f16(source, "transformer.txt_norm.weight", hidden)?;
    // Final norm: AdaLNContinuous linear (hidden -> 2*hidden) + norm_out.norm
    // is not stored (norm is identity in the GGUF).
    require_matrix(
        source,
        "transformer.norm_out.linear.weight",
        &[hidden, 2 * hidden],
    )?;
    require_norm_f16(source, "transformer.norm_out.linear.bias", 2 * hidden)?;
    // Final projection: hidden -> latent_dim
    require_matrix(
        source,
        "transformer.proj_out.weight",
        &[hidden, dit::LATENT_DIM as u64],
    )?;
    require_norm_f16(source, "transformer.proj_out.bias", dit::LATENT_DIM as u64)?;
    // Rotary inv_freq: head_dim/2 = 32 (just the inv_freq, not the expanded form;
    // F16 in the unsloth F16 GGUF).
    require_norm_f16(source, "transformer.rotary_embed.inv_freq", 32)?;
    // Per-block tensors
    for layer in 0..dit::NUM_DOUBLE_LAYERS {
        let prefix = format!("transformer.transformer_blocks.{layer}");
        // AdaLN modulation (hidden -> 6*hidden), with bias.
        require_matrix(
            source,
            &format!("{prefix}.attn_norm_x.linear.weight"),
            &[hidden, mod_dim],
        )?;
        require_norm_f16(
            source,
            &format!("{prefix}.attn_norm_x.linear.bias"),
            mod_dim,
        )?;
        require_matrix(
            source,
            &format!("{prefix}.attn_norm_c.linear.weight"),
            &[hidden, mod_dim],
        )?;
        require_norm_f16(
            source,
            &format!("{prefix}.attn_norm_c.linear.bias"),
            mod_dim,
        )?;
        // QKV for x-stream (no suffix)
        require_matrix(
            source,
            &format!("{prefix}.attn.to_qkv.weight"),
            &[hidden, qkv],
        )?;
        require_norm_f16(source, &format!("{prefix}.attn.to_qkv.bias"), qkv)?;
        // QKV for c-stream (_c suffix)
        require_matrix(
            source,
            &format!("{prefix}.attn.to_qkv_c.weight"),
            &[hidden, qkv],
        )?;
        require_norm_f16(source, &format!("{prefix}.attn.to_qkv_c.bias"), qkv)?;
        // Output projection for c-stream
        require_matrix(
            source,
            &format!("{prefix}.attn.to_out_c.weight"),
            &[hidden, hidden],
        )?;
        require_norm_f16(source, &format!("{prefix}.attn.to_out_c.bias"), hidden)?;
        // Output projection for x-stream (`.0` suffix)
        require_matrix(
            source,
            &format!("{prefix}.attn.to_out.0.weight"),
            &[hidden, hidden],
        )?;
        require_norm_f16(source, &format!("{prefix}.attn.to_out.0.bias"), hidden)?;
        // Q/K RMS norms
        require_norm_f16(
            source,
            &format!("{prefix}.attn.q_norm.weight"),
            dit::HEAD_DIM as u64,
        )?;
        require_norm_f16(
            source,
            &format!("{prefix}.attn.k_norm.weight"),
            dit::HEAD_DIM as u64,
        )?;
        // FF gate+up packed for x-stream
        require_matrix(
            source,
            &format!("{prefix}.ff_x.linear_in.weight"),
            &[hidden, ff_packed],
        )?;
        require_matrix(
            source,
            &format!("{prefix}.ff_x.linear_out.weight"),
            &[ff_inner, hidden],
        )?;
        // FF gate+up packed for c-stream
        require_matrix(
            source,
            &format!("{prefix}.ff_c.linear_in.weight"),
            &[hidden, ff_packed],
        )?;
        require_matrix(
            source,
            &format!("{prefix}.ff_c.linear_out.weight"),
            &[ff_inner, hidden],
        )?;
    }
    for layer in 0..dit::NUM_SINGLE_LAYERS {
        let prefix = format!("transformer.single_transformer_blocks.{layer}");
        require_matrix(
            source,
            &format!("{prefix}.attn_norm.linear.weight"),
            &[hidden, mod_dim],
        )?;
        require_norm_f16(source, &format!("{prefix}.attn_norm.linear.bias"), mod_dim)?;
        require_matrix(
            source,
            &format!("{prefix}.attn.to_qkv.weight"),
            &[hidden, qkv],
        )?;
        require_norm_f16(source, &format!("{prefix}.attn.to_qkv.bias"), qkv)?;
        // to_out.0 (special: 0 suffix even for single block)
        require_matrix(
            source,
            &format!("{prefix}.attn.to_out.0.weight"),
            &[hidden, hidden],
        )?;
        require_norm_f16(source, &format!("{prefix}.attn.to_out.0.bias"), hidden)?;
        require_norm_f16(
            source,
            &format!("{prefix}.attn.q_norm.weight"),
            dit::HEAD_DIM as u64,
        )?;
        require_norm_f16(
            source,
            &format!("{prefix}.attn.k_norm.weight"),
            dit::HEAD_DIM as u64,
        )?;
        require_matrix(
            source,
            &format!("{prefix}.ff.linear_in.weight"),
            &[hidden, ff_packed],
        )?;
        require_matrix(
            source,
            &format!("{prefix}.ff.linear_out.weight"),
            &[ff_inner, hidden],
        )?;
    }
    Ok(())
}

/// Accept either F16 or F32 for a 1-D vector tensor. The AuK F16 GGUF
/// stores biases and rotary inv_freq as F16; the F32 GGUF would store
/// them as F32. Both round-trip cleanly through the F32 normalization in
/// `load_f32_vector`.
fn require_norm_f16(source: &dyn TensorSource, name: &str, len: u64) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [len] {
        return Err(format!("Invalid {name} dimensions"));
    }
    if !matches!(
        info.ggml_type,
        GGMLType::F32 | GGMLType::BF16 | GGMLType::F16
    ) {
        return Err(format!(
            "Invalid {name} type {:?}: expected F32/F16/BF16",
            info.ggml_type
        ));
    }
    Ok(())
}

fn validate_vae(_source: &dyn TensorSource) -> Result<(), String> {
    // VAE validation deferred to the VAE module's own probe.
    Ok(())
}

fn require_tensor(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
    ggml_type: GGMLType,
) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims {
        return Err(format!("Invalid {name} dimensions"));
    }
    if info.ggml_type != ggml_type {
        return Err(format!(
            "Invalid {name} type: expected {ggml_type:?}, got {:?}",
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
    Ok(())
}

/// Accept F16 / BF16 / Q8_0 / Q*_K dtypes. The actual matmul dispatch in
/// `linear_into_scaled_impl` covers the full set.
fn require_matrix(source: &dyn TensorSource, name: &str, dims: &[u64]) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims {
        return Err(format!("Invalid {name} dimensions"));
    }
    if !matches!(
        info.ggml_type,
        GGMLType::F16
            | GGMLType::BF16
            | GGMLType::Q8_0
            | GGMLType::Q4K
            | GGMLType::Q5K
            | GGMLType::Q6K
            | GGMLType::Q4_0
            | GGMLType::Q4_1
            | GGMLType::Q5_0
            | GGMLType::Q5_1
            | GGMLType::Q8_1
            | GGMLType::Q2K
            | GGMLType::Q3K
            | GGMLType::Q8K
    ) {
        return Err(format!(
            "Invalid {name} type {:?}: expected F16/BF16/Q8_0/Q*_K",
            info.ggml_type
        ));
    }
    Ok(())
}

pub(crate) struct Q8Scratch {
    scaled: Vec<f32>,
    force_f32_row: Vec<f32>,
    f16_input: Vec<u16>,
    f16_inputs: Vec<Vec<u16>>,
    values: Vec<u8>,
    scales: Vec<f32>,
}

impl Q8Scratch {
    pub(crate) fn new(n_in: usize) -> Self {
        Self {
            scaled: Vec::new(),
            force_f32_row: Vec::new(),
            f16_input: Vec::new(),
            f16_inputs: Vec::new(),
            values: vec![0; n_in],
            scales: vec![0.0; n_in.div_ceil(32)],
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_into(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
    input: &[f32],
    output: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
) -> Result<(), String> {
    linear_into_scaled_impl(source, name, n_in, n_out, input, output, q8, pool, 1.0)
}

#[allow(clippy::too_many_arguments)]
fn linear_into_scaled_impl(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
    input: &[f32],
    output: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
    scale: f32,
) -> Result<(), String> {
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
    n_in.checked_mul(n_out)
        .ok_or_else(|| format!("Invalid {name} dimensions"))?;
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [n_in as u64, n_out as u64] {
        return Err(format!(
            "Invalid {name} dimensions: expected [{}, {}], got {:?}",
            n_in, n_out, info.dims
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
    match info.ggml_type {
        GGMLType::F16 => {
            // Try F16 GPU path first. Single-row `matmul_rows` reuses the
            // F16-aware Vulkan shader stack (see `auk_f16_gpu_runtime`).
            // Synchronous dispatch -- `matmul_rows` fences its own submission.
            #[cfg(feature = "vulkan")]
            {
                match f16_gpu_linear_into(bytes, input, output, n_in, n_out) {
                    Ok(true) => {
                        if scale != 1.0 {
                            for v in output.iter_mut() {
                                *v *= scale;
                            }
                        }
                        return Ok(());
                    }
                    Ok(false) => {} // Vulkan disabled / runtime uninit -> CPU
                    Err(e) => eprintln!("[AukDiT] {}", e),
                }
            }
            let weight_ptr = bytes.as_ptr() as usize;
            let weight_len = bytes.len();
            let input_ptr = input.as_ptr() as usize;
            let input_len = input.len();
            let output_ptr = output.as_mut_ptr() as usize;
            let output_len = output.len();
            let scale_copy = scale;
            let threads = pool.n_threads();
            if q8.f16_inputs.len() < threads {
                q8.f16_inputs.resize_with(threads, Vec::new);
            }
            let staging = &q8.f16_inputs[..threads];
            let staging_ptr = staging.as_ptr() as usize;
            pool.compute(move |ith, nth| {
                let weight =
                    unsafe { std::slice::from_raw_parts(weight_ptr as *const u8, weight_len) };
                let values =
                    unsafe { std::slice::from_raw_parts(input_ptr as *const f32, input_len) };
                let out =
                    unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, output_len) };
                let buffer = unsafe { &mut *((staging_ptr as *mut Vec<u16>).add(ith)) };
                crate::ops::kernel::f16::F16Kernel::new(weight)
                    .forward_scaled_rows(values, out, n_in, n_out, scale_copy, buffer, ith, nth);
            });
            Ok(())
        }
        GGMLType::BF16 | GGMLType::Q4K | GGMLType::Q5K | GGMLType::Q6K => {
            // For non-F16/Q8_0 weights, build an owned QTensor and dispatch
            // through its parallel matmul (parallels the ERNIE-Image path).
            let tensor = QTensorOwned::from_bytes_owned(bytes, info.ggml_type, n_in, n_out);
            let input_ptr = input.as_ptr() as usize;
            let input_len = input.len();
            let output_ptr = output.as_mut_ptr() as usize;
            let scale_copy = scale;
            pool.compute(move |ith, nth| {
                let values =
                    unsafe { std::slice::from_raw_parts(input_ptr as *const f32, input_len) };
                let out = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, n_out) };
                tensor.forward_prepared(values, &[], &[], None, out, n_in, n_out, ith, nth);
                if scale_copy != 1.0 {
                    for v in out.iter_mut() {
                        *v *= scale_copy;
                    }
                }
            });
            Ok(())
        }
        GGMLType::Q8_0 => {
            q8.prepare(input, n_in)?;
            let weight_ptr = bytes.as_ptr() as usize;
            let weight_len = bytes.len();
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
                let out = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, n_out) };
                matmul_q8_0_quantized_parallel_rows(
                    weight, input, scales, out, n_in, n_out, ith, nth,
                );
            });
            Ok(())
        }
        _ => Err(format!(
            "Unsupported {name} dtype {:?}: expected F16/BF16/Q8_0/Q*_K",
            info.ggml_type
        )),
    }
}

impl Q8Scratch {
    fn prepare(&mut self, input: &[f32], n_in: usize) -> Result<(), String> {
        if input.len() != n_in {
            return Err("Invalid linear input length".into());
        }
        self.values.resize(n_in, 0);
        self.scales.resize(n_in.div_ceil(32), 0.0);
        crate::ops::quantize_q8_0_into(input, n_in, &mut self.values, &mut self.scales);
        Ok(())
    }
}

/// Lazily-initialized `BatchedLinearRuntime` configured for single-row F16
/// GPU matmul. The runtime owns an arena sized for the largest single matmul
/// in the AuK DiT (AdaLN `1536 -> 9216`) and a descriptor pool that holds
/// entries keyed by `(weight_ptr, weight_len)`.
///
/// We use `BatchedLinearRuntime::matmul_rows` with `rows=1` and
/// `format=F16` so the existing F16-aware Vulkan shader handles the matmul
/// on GPU. This eliminates the F16 -> Q8_0 dynamic-quantization workaround
/// (`f4e7879`) at the cost of one new global `OnceLock`.
///
/// Falls back to the caller's CPU path on any error so this layer never
/// forces the model into a hard failure mode.
#[cfg(feature = "vulkan")]
static AUK_F16_GPU_RUNTIME: std::sync::OnceLock<Option<std::sync::Mutex<BatchedLinearRuntime>>> =
    std::sync::OnceLock::new();

#[cfg(feature = "vulkan")]
fn auk_f16_gpu_runtime(
    ctx: &'static VulkanContext,
) -> Option<&'static std::sync::Mutex<BatchedLinearRuntime>> {
    AUK_F16_GPU_RUNTIME
        .get_or_init(|| {
            // AuK DiT max shapes verified against the GGUF:
            // - AdaLN mod `1536 -> 9216` (n_out = 6 * HIDDEN)
            // - FF linear_out `3072 -> 1536` (n_in = FF_INNER; HIDDEN
            //   packed gate+up projections only need n_in=HIDDEN, but the
            //   linear_out is FF_INNER -> HIDDEN and is the largest n_in).
            // 4096 gives slack for hypothetical future variants.
            const MAX_ROWS: usize = 1;
            const MAX_N_IN: usize = 4096;
            const MAX_N_OUT: usize = 9216;
            // ~hundreds of unique weight tensors expected across the DiT;
            // 4096 descriptor slots is plenty.
            const DESCRIPTOR_CAPACITY: usize = 4096;
            unsafe {
                match BatchedLinearRuntime::new(
                    ctx,
                    MAX_ROWS,
                    MAX_N_IN,
                    MAX_N_OUT,
                    DESCRIPTOR_CAPACITY,
                ) {
                    Ok(rt) => {
                        eprintln!(
                            "[AukDiT] F16 GPU matmul runtime ready (max_rows={MAX_ROWS}, \
                             max_n_in={MAX_N_IN}, max_n_out={MAX_N_OUT}, \
                             descriptor_capacity={DESCRIPTOR_CAPACITY})"
                        );
                        Some(std::sync::Mutex::new(rt))
                    }
                    Err(e) => {
                        eprintln!("[AukDiT] F16 GPU matmul runtime init failed: {:?}", e);
                        None
                    }
                }
            }
        })
        .as_ref()
}

/// Try to dispatch a single-row F16 GPU matmul. Returns:
/// - `Ok(true)` if the matmul succeeded on GPU (caller should skip CPU path).
/// - `Ok(false)` if Vulkan is disabled or the runtime failed to init (caller
///   should fall back to CPU without surfacing an error).
/// - `Err(msg)` if the GPU dispatch itself failed at runtime; the caller
///   should also fall back, but the message will be logged.
#[cfg(feature = "vulkan")]
fn f16_gpu_linear_into(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
) -> Result<bool, String> {
    let Some(ctx) = crate::ops::get_vulkan_context() else {
        return Ok(false);
    };
    let Some(runtime_mutex) = auk_f16_gpu_runtime(ctx) else {
        return Ok(false);
    };
    if crate::core::thread_pool::gpu_matmul_disabled() || crate::vulkan::gpu_broken() {
        return Ok(false);
    }
    let mut runtime = runtime_mutex
        .lock()
        .map_err(|e| format!("AuK F16 GPU runtime lock poisoned: {}", e))?;
    runtime
        .matmul_rows(
            weight,
            GpuWeightFormat::F16,
            input,
            /*rows=*/ 1,
            n_in,
            n_out,
            output,
        )
        .map_err(|e| format!("AuK F16 GPU matmul failed: {:?}", e))?;
    Ok(true)
}
