//! dots.tts generation orchestration: schedule → LLM prefill (with optional
//! prompt audio) → autoregressive loop (EOS head, DiT flow-matching patch,
//! patch-encoder feedback into the LLM) → vocoder → 48 kHz mono waveform.
//!
//! Reference: `DotsTtsModel._generate_latents_stream` + `runtime.py` with the
//! default sampling contract (euler, NFE=10, guidance=1.2, speaker_scale=1.5,
//! eos_threshold=0.8).

use std::sync::Arc;

use rand::Rng;

use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::models::dots::config::DotsTtsConfig;
use crate::models::dots::dit::DiT;
use crate::models::dots::llm::{DotsLlm, DotsLlmSession, LlmInputRow};
use crate::models::dots::patch_encoder::{linear_forward, load_f16_f32, PatchEncoder};
use crate::models::dots::schedule::{DotsSchedule, build_generation_schedule};
use crate::models::dots::speaker::{kaldi_fbank, CamPlus, Resampler};
use crate::models::dots::vocoder::Vocoder;

pub const DEFAULT_NFE: usize = 10;
pub const DEFAULT_GUIDANCE: f32 = 1.2;
pub const DEFAULT_SPEAKER_SCALE: f32 = 1.5;
pub const DEFAULT_EOS_THRESHOLD: f32 = 0.8;
pub const LN_EPS: f32 = 1e-5;

pub struct DotsTtsModel {
    pub config: DotsTtsConfig,
    pub llm: DotsLlm,
    pub patch_encoder: PatchEncoder,
    pub dit: DiT,
    pub speaker: CamPlus,
    pub speaker_resample: Resampler,
    pub vocoder: Vocoder,
    pub hidden_proj: (Vec<f32>, Vec<f32>),
    pub latent_proj: (Vec<f32>, Vec<f32>),
    pub coordinate_proj: (Vec<f32>, Vec<f32>),
    pub xvec_proj: (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), // lin w/b, norm w/b
    pub eos_proj: (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),  // l0 w/b, l2 w/b
    pub latent_mean: Vec<f32>,
    pub latent_var: Vec<f32>,
}

impl DotsTtsModel {
    pub fn from_sources(
        llm_source: Arc<dyn TensorSource>,
        mmproj_source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let config = DotsTtsConfig::from_source(mmproj_source.as_ref())?;
        let w = |name: &str, dims: &[u64]| load_f16_f32(mmproj_source.as_ref(), name, dims);
        let ln_var = config.latent_dim as u64;
        let fm = config.fm_hidden_size as u64;
        let llm_h = config.llm_hidden_size as u64;
        let xvec = config.xvec_dim as u64;
        Ok(Self {
            config: config.clone(),
            llm: DotsLlm::from_source(llm_source, pool)?,
            patch_encoder: PatchEncoder::from_source(mmproj_source.as_ref(), config.clone())?,
            dit: DiT::from_source(mmproj_source.as_ref(), config.clone())?,
            speaker: CamPlus::from_source(mmproj_source.as_ref())?,
            speaker_resample: Resampler::from_kernel(&w(
                "dotstts.speaker.resample_kernel",
                &[41, 1, 1],
            )?)?,
            vocoder: Vocoder::from_source(mmproj_source.as_ref())?,
            hidden_proj: (
                w("dotstts.hidden_proj.weight", &[llm_h, fm])?,
                w("dotstts.hidden_proj.bias", &[fm])?,
            ),
            latent_proj: (
                w("dotstts.latent_proj.weight", &[ln_var, fm])?,
                w("dotstts.latent_proj.bias", &[fm])?,
            ),
            coordinate_proj: (
                w("dotstts.coordinate_proj.weight", &[ln_var, fm])?,
                w("dotstts.coordinate_proj.bias", &[fm])?,
            ),
            xvec_proj: (
                w("dotstts.xvec_proj.0.weight", &[xvec, fm])?,
                w("dotstts.xvec_proj.0.bias", &[fm])?,
                w("dotstts.xvec_proj.1.weight", &[fm])?,
                w("dotstts.xvec_proj.1.bias", &[fm])?,
            ),
            eos_proj: (
                w("dotstts.eos_proj.0.weight", &[llm_h; 2])?,
                w("dotstts.eos_proj.0.bias", &[llm_h])?,
                w("dotstts.eos_proj.2.weight", &[llm_h, 2])?,
                w("dotstts.eos_proj.2.bias", &[2])?,
            ),
            latent_mean: w("dotstts.latent_stats.mean", &[ln_var])?,
            latent_var: w("dotstts.latent_stats.var", &[ln_var])?,
        })
    }

    pub fn normalize(&self, x: &mut [f32]) {
        for (value, (&mean, &var)) in x
            .iter_mut()
            .zip(self.latent_mean.iter().zip(self.latent_var.iter()))
        {
            *value = (*value - mean) / var.sqrt();
        }
    }

    pub fn denormalize(&self, x: &[f32]) -> Vec<f32> {
        // the latent stats are per-dimension (128); process row-wise chunks
        let mut out = Vec::with_capacity(x.len());
        for chunk in x.chunks_exact(self.config.latent_dim) {
            for (&value, (&mean, &var)) in chunk
                .iter()
                .zip(self.latent_mean.iter().zip(self.latent_var.iter()))
            {
                out.push(value * var.sqrt() + mean);
            }
        }
        out
    }

    /// xvec_proj(speaker_embedding × scale) → 1024-dim g_cond.
    pub fn speaker_condition(&self, xvec: &[f32], scale: f32) -> Result<Vec<f32>, String> {
        if xvec.len() != self.config.xvec_dim {
            return Err("speaker x-vector width mismatch".into());
        }
        let scaled: Vec<f32> = xvec.iter().map(|&v| v * scale).collect();
        let mut out = vec![0.0f32; self.config.fm_hidden_size];
        linear_forward(
            &self.xvec_proj.0,
            Some(&self.xvec_proj.1),
            &scaled,
            self.config.xvec_dim,
            self.config.fm_hidden_size,
            &mut out,
        );
        // LayerNorm
        let mut mean = 0.0f64;
        for &v in out.iter() {
            mean += v as f64;
        }
        mean /= out.len() as f64;
        let mut var = 0.0f64;
        for &v in out.iter() {
            let d = v as f64 - mean;
            var += d * d;
        }
        var /= out.len() as f64;
        let inv = 1.0 / (var + LN_EPS as f64).sqrt();
        for i in 0..out.len() {
            out[i] = ((out[i] as f64 - mean) * inv) as f32 * self.xvec_proj.2[i]
                + self.xvec_proj.3[i];
        }
        Ok(out)
    }

    pub fn eos_probability(&self, hidden: &[f32]) -> Result<f32, String> {
        if hidden.len() != self.config.llm_hidden_size {
            return Err("eos hidden width mismatch".into());
        }
        let mut l0 = vec![0.0f32; self.config.llm_hidden_size];
        linear_forward(
            &self.eos_proj.0,
            Some(&self.eos_proj.1),
            hidden,
            self.config.llm_hidden_size,
            self.config.llm_hidden_size,
            &mut l0,
        );
        for v in l0.iter_mut() {
            *v = crate::ops::silu(*v);
        }
        let mut logits = vec![0.0f32; 2];
        linear_forward(&self.eos_proj.2, Some(&self.eos_proj.3), &l0, self.config.llm_hidden_size, 2, &mut logits);
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e0 = (logits[0] - max).exp();
        let e1 = (logits[1] - max).exp();
        Ok(e1 / (e0 + e1))
    }
}

/// Prompt audio conditioning (voice cloning).
pub struct PromptConditioning {
    /// Normalized prompt patches `[P, 4, 128]` for the FM history.
    pub patches: Vec<f32>,
    /// g_cond from the speaker x-vector.
    pub g_cond: Vec<f32>,
}

impl DotsTtsModel {
    /// Extract the speaker x-vector from a 48 kHz mono prompt waveform.
    pub fn encode_speaker(&self, wav48k: &[f32]) -> Result<Vec<f32>, String> {
        let wav16k = self.speaker_resample.resample(wav48k);
        if wav16k.len() < 400 {
            return Err("prompt audio too short for the speaker encoder".into());
        }
        let mel = kaldi_fbank(&wav16k);
        self.speaker.encode(&mel)
    }

    /// Full prompt conditioning: speaker vector + sampled prompt latents.
    pub fn prepare_prompt_conditioning<R: Rng + ?Sized>(
        &self,
        wav48k: &[f32],
        speaker_scale: f32,
        rng: &mut R,
    ) -> Result<PromptConditioning, String> {
        let xvec = self.encode_speaker(wav48k)?;
        let g_cond = self.speaker_condition(&xvec, speaker_scale)?;
        // pad to whole patches
        let samples_per_patch = self.config.samples_per_patch();
        let target = wav48k.len().div_ceil(samples_per_patch) * samples_per_patch;
        let mut padded = wav48k.to_vec();
        padded.resize(target, 0.0);
        let dist = self.vocoder.extract_latent_distribution(&padded)?;
        let frames = target / self.config.hop_size;
        let mut sampled = vec![0.0f32; frames * self.config.latent_dim];
        for t in 0..frames {
            for c in 0..self.config.latent_dim {
                let mean = dist[c * frames + t];
                let log_std = dist[(self.config.latent_dim + c) * frames + t];
                sampled[t * self.config.latent_dim + c] = mean + gaussian(rng) * log_std.exp();
            }
        }
        // drop the last patch_size frames, then normalize → [P, 4, 128]
        let keep = frames.saturating_sub(self.config.patch_size);
        let patch_frames = self.config.patch_size;
        let p_count = keep / patch_frames;
        let mut patches = vec![0.0f32; p_count * patch_frames * self.config.latent_dim];
        for (dst, src) in patches
            .chunks_exact_mut(self.config.latent_dim)
            .zip(sampled.chunks_exact(self.config.latent_dim))
        {
            dst.copy_from_slice(src);
        }
        // normalize in place
        for chunk in patches.chunks_exact_mut(self.config.latent_dim) {
            self.normalize(chunk);
        }
        Ok(PromptConditioning { patches, g_cond })
    }
}

fn gaussian<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    // Box-Muller
    let u1 = rng.gen::<f32>().max(1e-9);
    let u2 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// One decoded latent patch in normalized space plus its raw (denormalized) form.
pub struct GenerateOptions {
    pub max_patches: usize,
    pub temperature: f32,
    pub nfe: usize,
    pub guidance: f32,
    pub speaker_scale: f32,
    pub eos_threshold: f32,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_patches: 64,
            temperature: 0.9,
            nfe: DEFAULT_NFE,
            guidance: DEFAULT_GUIDANCE,
            speaker_scale: DEFAULT_SPEAKER_SCALE,
            eos_threshold: DEFAULT_EOS_THRESHOLD,
        }
    }
}

/// Streaming generation state: the FM sequence buffer and the LLM session.
pub struct DotsGenerateSession<'a> {
    model: &'a DotsTtsModel,
    pub llm: DotsLlmSession<'a>,
    pub fm: Vec<f32>,
    pub fm_cfg: Vec<f32>,
    pub fm_seq_len: usize,
}

impl<'a> DotsGenerateSession<'a> {
    pub fn new(model: &'a DotsTtsModel, capacity_patches: usize) -> Result<Self, String> {
        let capacity = capacity_patches * model.config.unit_len() + 8;
        Ok(Self {
            model,
            llm: model.llm.new_session()?,
            fm: vec![0.0; capacity * model.config.fm_hidden_size],
            fm_cfg: vec![0.0; capacity * model.config.fm_hidden_size],
            fm_seq_len: 0,
        })
    }

    fn append_hidden_chunk(&mut self, hidden: &[f32]) -> Result<(), String> {
        let fm = self.model.config.fm_hidden_size;
        let mut projected = vec![0.0f32; fm];
        linear_forward(
            &self.model.hidden_proj.0,
            Some(&self.model.hidden_proj.1),
            hidden,
            self.model.config.llm_hidden_size,
            fm,
            &mut projected,
        );
        let start = self.fm_seq_len * fm;
        self.fm[start..start + fm].copy_from_slice(&projected);
        // CFG branch: null projection (zeros)
        let start_cfg = self.fm_seq_len * fm;
        self.fm_cfg[start_cfg..start_cfg + fm].fill(0.0);
        self.fm_seq_len += 1;
        Ok(())
    }

    fn append_history_chunk(&mut self, latents: &[f32]) -> Result<(), String> {
        // latents: [patch_size, latent_dim] (normalized space)
        let fm = self.model.config.fm_hidden_size;
        let p = self.model.config.patch_size;
        let d = self.model.config.latent_dim;
        for i in 0..p {
            let mut projected = vec![0.0f32; fm];
            linear_forward(
                &self.model.latent_proj.0,
                Some(&self.model.latent_proj.1),
                &latents[i * d..(i + 1) * d],
                d,
                fm,
                &mut projected,
            );
            let start = self.fm_seq_len * fm;
            self.fm[start..start + fm].copy_from_slice(&projected);
            self.fm_cfg[start..start + fm].copy_from_slice(&projected);
            self.fm_seq_len += 1;
        }
        Ok(())
    }

    fn decode_next_patch<R: Rng + ?Sized>(&mut self, g_cond: &[f32], options: &GenerateOptions, rng: &mut R) -> Result<Vec<f32>, String> {
        // noise z0: [patch_size, latent_dim]
        let p = self.model.config.patch_size;
        let d = self.model.config.latent_dim;
        let mut z0 = vec![0.0f32; p * d];
        for value in z0.iter_mut() {
            *value = gaussian(rng);
        }
        let seq = &self.fm[..self.fm_seq_len * self.model.config.fm_hidden_size];
        let cfg = &self.fm_cfg[..self.fm_seq_len * self.model.config.fm_hidden_size];
        let mut patch = vec![0.0f32; p * d];
        self.model.dit.solve_patch(
            seq,
            cfg,
            self.fm_seq_len,
            g_cond,
            &self.model.coordinate_proj.0,
            &self.model.coordinate_proj.1,
            options.guidance,
            options.nfe,
            &z0,
            &mut patch,
        )?;
        Ok(patch)
    }
}

/// Run the TTS pipeline for a schedule and produce raw latent patches
/// (denormalized, ready for the vocoder).
pub fn generate_latents<R: Rng + ?Sized>(
    model: &DotsTtsModel,
    tokenizer: &crate::core::tokenizer::BPETokenizer,
    text: &str,
    prompt: Option<&PromptConditioning>,
    options: &GenerateOptions,
    rng: &mut R,
) -> Result<Vec<f32>, String> {
    let prompt_patch_count = prompt.map_or(0, |p| p.patches.len() / (4 * model.config.latent_dim));
    // the LLM schedule covers the prompt (reference) spans too, so the
    // generation span budget sits *after* the prompt prefill
    let schedule =
        build_generation_schedule(tokenizer, text, options.max_patches + prompt_patch_count)?;
    let span_ids = DotsSchedule::audio_span_ids(tokenizer)?;
    let span_positions = &schedule.span_positions;

    // prompt prefill = up to the first *generated* span
    // schedule must contain prompt_patch_count prompt spans + ≥1 generation span
    let prefill_end = if span_positions.len() > prompt_patch_count {
        span_positions[prompt_patch_count]
    } else {
        return Err(format!(
            "generation schedule provides {} spans; prompt prefill requires {} + 1",
            span_positions.len(),
            prompt_patch_count
        ));
    };

    let mut session = DotsGenerateSession::new(model, options.max_patches + prompt_patch_count)?;

    // ---- LLM prefill -------------------------------------------------- //
    let mut hiddens: Vec<Vec<f32>> = Vec::new();
    let mut prompt_embeds: Option<Vec<f32>> = None;
    if let Some(p) = prompt {
        // patch-encoder embeddings for the prompt spans
        let mut pe_state = model.patch_encoder.new_state(prompt_patch_count * 2 + 8);
        let raw_patches = denormalize_patches(model, &p.patches);
        let embeddings = model
            .patch_encoder
            .prefill(&raw_patches, &mut pe_state)?;
        prompt_embeds = Some(embeddings);
    }
    for pos in 0..prefill_end {
        let id = schedule.ids[pos];
        let row: LlmInputRow<'_> = if let Some(ref emb) = prompt_embeds {
            if let Some(span_idx) = span_positions[..prompt_patch_count]
                .iter()
                .position(|&s| s == pos)
            {
                LlmInputRow::Embedding(
                    &emb[span_idx * model.config.llm_hidden_size
                        ..(span_idx + 1) * model.config.llm_hidden_size],
                )
            } else {
                LlmInputRow::Token(id)
            }
        } else {
            LlmInputRow::Token(id)
        };
        hiddens.push(session.llm.step_row(row)?);
    }

    // ---- FM buffer assembly from the prefill --------------------------- //
    let mut cursor = 0usize;
    if let Some(p) = prompt {
        let patches_per_span = model.config.patch_size * model.config.latent_dim;
        for (span_idx, &span_position) in span_positions[..prompt_patch_count].iter().enumerate() {
            if span_position > cursor {
                session.append_hidden_chunk(&hiddens[span_position - 1])?;
            }
            let patch = &p.patches
                [span_idx * patches_per_span..(span_idx + 1) * patches_per_span];
            session.append_history_chunk(patch)?;
            if span_position + 1 < schedule.ids.len() && span_ids.contains(&schedule.ids[span_position + 1]) {
                session.append_hidden_chunk(&hiddens[span_position])?;
            }
            cursor = span_position + 1;
        }
    }
    if prefill_end > cursor {
        session.append_hidden_chunk(&hiddens[prefill_end - 1])?;
    }

    // ---- decode loop --------------------------------------------------- //
    let g_cond = match prompt {
        Some(p) => p.g_cond.clone(),
        None => vec![0.0f32; model.config.fm_hidden_size],
    };
    let mut raw_patches: Vec<f32> = Vec::new();
    let mut pe_state = model.patch_encoder.new_state(options.max_patches * 2 + 8);
    let mut emitted_patches = 0usize;
    let suppress_first_eos = prompt_patch_count > 0;
    let mut position = prefill_end;
    let mut span_cursor = prompt_patch_count;

    while position < schedule.ids.len() {
        let id = schedule.ids[position];
        if span_ids.contains(&id) {
            let should_check = !(suppress_first_eos && emitted_patches == 0);
            let stop_after = if should_check {
                let h = session.llm.last_hidden().to_vec();
                model.eos_probability(&h)? > options.eos_threshold
            } else {
                false
            };
            let patch = session.decode_next_patch(&g_cond, options, rng)?;
            // consume: history + patch-encoder → LLM
            session.append_history_chunk(&patch)?;
            let raw = model.denormalize(&patch);
            let embedding = model.patch_encoder.encode_patch(&raw, &mut pe_state)?;
            session.llm.step_row(LlmInputRow::Embedding(&embedding))?;
            raw_patches.extend_from_slice(&raw);
            emitted_patches += 1;
            position += 1;
            span_cursor += 1;
            if position < schedule.ids.len() && span_ids.contains(&schedule.ids[position]) {
                let hidden = session.llm.last_hidden().to_vec();
                session.append_hidden_chunk(&hidden)?;
            }
            if stop_after {
                break;
            }
            continue;
        }
        // text run until the next span
        let next_audio = if span_cursor < span_positions.len() {
            span_positions[span_cursor]
        } else {
            schedule.ids.len()
        };
        while position < next_audio {
            let id = schedule.ids[position];
            session.llm.step_row(LlmInputRow::Token(id))?;
            position += 1;
        }
        let hidden = session.llm.last_hidden().to_vec();
        session.append_hidden_chunk(&hidden)?;
    }
    if raw_patches.is_empty() {
        return Err("generation produced no latent patches (EOS before the first patch)".into());
    }
    Ok(raw_patches)
}

fn denormalize_patches(model: &DotsTtsModel, patches: &[f32]) -> Vec<f32> {
    model.denormalize(patches)
}

/// Full synthesis: text → latent patches → 48 kHz mono waveform.
pub fn synthesize<R: Rng + ?Sized>(
    model: &DotsTtsModel,
    tokenizer: &crate::core::tokenizer::BPETokenizer,
    text: &str,
    prompt: Option<&PromptConditioning>,
    options: &GenerateOptions,
    rng: &mut R,
) -> Result<Vec<f32>, String> {
    let latents = generate_latents(model, tokenizer, text, prompt, options, rng)?;
    // latents: [frames, 128] raw
    if latents.len() % model.config.latent_dim != 0 {
        return Err("latent stream width mismatch".into());
    }
    model.vocoder.decode_latents(&latents)
}