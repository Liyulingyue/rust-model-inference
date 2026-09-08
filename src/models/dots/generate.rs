//! dots.tts generation orchestration: schedule → LLM prefill (with optional
//! prompt audio) → autoregressive loop (EOS head, DiT flow-matching patch,
//! patch-encoder feedback into the LLM) → vocoder → 48 kHz mono waveform.
//!
//! Reference: `DotsTtsModel._generate_latents_stream` + `runtime.py` with the
//! default sampling contract (euler, NFE=10, guidance=1.2, speaker_scale=1.5,
//! eos_threshold=0.8).

use std::sync::Arc;

use rand::Rng;

#[cfg(any(
    all(feature = "accelerate", target_os = "macos"),
    all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
))]
use super::blas::sys;

use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::models::dots::config::DotsTtsConfig;
use crate::models::dots::dit::DiT;
use crate::models::dots::llm::{DotsLlm, DotsLlmSession, LlmInputRow};
use crate::models::dots::patch_encoder::{
    linear_forward, linear_forward_transposed_input_then_bias, load_f16_f32, PatchEncoder,
    PatchEncoderState,
};
use crate::models::dots::schedule::{
    build_edit_generation_schedule, build_generation_schedule, DotsSchedule,
};
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
        Ok(speaker_condition_forward(
            xvec,
            scale,
            &self.xvec_proj.0,
            &self.xvec_proj.1,
            &self.xvec_proj.2,
            &self.xvec_proj.3,
        ))
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
        linear_forward(
            &self.eos_proj.2,
            Some(&self.eos_proj.3),
            &l0,
            self.config.llm_hidden_size,
            2,
            &mut logits,
        );
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e0 = (logits[0] - max).exp();
        let e1 = (logits[1] - max).exp();
        Ok(e1 / (e0 + e1))
    }
}

fn combine_moments(
    incoming_count: i32,
    incoming_mean: f32,
    incoming_m2: f32,
    count: &mut i32,
    mean: &mut f32,
    m2: &mut f32,
) {
    let total = *count + incoming_count;
    let factor = if total == 0 {
        0.0
    } else {
        incoming_count as f32 / total as f32
    };
    let delta = incoming_mean - *mean;
    *mean += factor * delta;
    *m2 += incoming_m2 + delta * delta * factor * *count as f32;
    *count = total;
}

fn sample_latent_distribution(
    distribution: &[f32],
    noise: &[f32],
    frames: usize,
    latent_dim: usize,
) -> Result<Vec<f32>, String> {
    let sampled_len = frames
        .checked_mul(latent_dim)
        .ok_or_else(|| "prompt latent sample length overflow".to_string())?;
    if distribution.len() != sampled_len * 2 || noise.len() != sampled_len {
        return Err("prompt latent sampling shape mismatch".into());
    }
    let mut sampled = vec![0.0f32; sampled_len];
    for t in 0..frames {
        for c in 0..latent_dim {
            let mean = distribution[c * frames + t];
            let log_std = distribution[(latent_dim + c) * frames + t];
            sampled[t * latent_dim + c] =
                mean + noise[c * frames + t] * super::speaker::exp::torch28_exp(log_std);
        }
    }
    Ok(sampled)
}

fn torch28_rowwise_moments_1024(input: &[f32]) -> (f32, f32) {
    debug_assert_eq!(input.len(), 1024);
    let mut counts = [[0i32; 4]; 4];
    let mut means = [[0.0f32; 4]; 4];
    let mut m2s = [[0.0f32; 4]; 4];
    for block in 0..16 {
        let mut block_mean = [0.0f32; 4];
        let mut block_m2 = [0.0f32; 4];
        for index in 0..16 {
            let reciprocal = 1.0 / (index + 1) as f32;
            for lane in 0..4 {
                let value = input[block * 64 + index * 4 + lane];
                let delta = value - block_mean[lane];
                block_mean[lane] += delta * reciprocal;
                block_m2[lane] += delta * (value - block_mean[lane]);
            }
        }
        for lane in 0..4 {
            combine_moments(
                16,
                block_mean[lane],
                block_m2[lane],
                &mut counts[0][lane],
                &mut means[0][lane],
                &mut m2s[0][lane],
            );
        }
        let mut mask = block + 1;
        for depth in 1..4 {
            if mask & 1 != 0 {
                break;
            }
            for lane in 0..4 {
                combine_moments(
                    counts[depth - 1][lane],
                    means[depth - 1][lane],
                    m2s[depth - 1][lane],
                    &mut counts[depth][lane],
                    &mut means[depth][lane],
                    &mut m2s[depth][lane],
                );
            }
            counts[depth - 1] = [0; 4];
            means[depth - 1] = [0.0; 4];
            m2s[depth - 1] = [0.0; 4];
            mask >>= 1;
        }
    }
    for depth in 1..4 {
        for lane in 0..4 {
            combine_moments(
                counts[depth][lane],
                means[depth][lane],
                m2s[depth][lane],
                &mut counts[0][lane],
                &mut means[0][lane],
                &mut m2s[0][lane],
            );
        }
    }
    let (mut count, mut mean, mut m2) = (0i32, 0.0f32, 0.0f32);
    for lane in 0..4 {
        combine_moments(
            256,
            means[0][lane],
            m2s[0][lane],
            &mut count,
            &mut mean,
            &mut m2,
        );
    }
    (mean, m2 / 1024.0)
}

fn speaker_condition_forward(
    xvector: &[f32],
    scale: f32,
    weight: &[f32],
    bias: &[f32],
    norm_weight: &[f32],
    norm_bias: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(xvector.len(), 512);
    debug_assert_eq!(weight.len(), 1024 * 512);
    debug_assert_eq!(bias.len(), 1024);
    let scaled = xvector
        .iter()
        .map(|&value| value * scale)
        .collect::<Vec<_>>();
    let mut output = bias.to_vec();
    #[cfg(any(
        all(feature = "accelerate", target_os = "macos"),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    ))]
    unsafe {
        sys::cblas_sgemm(
            101,
            111,
            111,
            1024,
            1,
            512,
            1.0,
            weight.as_ptr(),
            512,
            scaled.as_ptr(),
            1,
            1.0,
            output.as_mut_ptr(),
            1,
        );
    }
    #[cfg(not(any(
        all(feature = "accelerate", target_os = "macos"),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    )))]
    for out in 0..1024 {
        for input in 0..512 {
            output[out] = weight[out * 512 + input].mul_add(scaled[input], output[out]);
        }
    }
    let (mean, variance) = torch28_rowwise_moments_1024(&output);
    let reciprocal_std = 1.0 / (variance + LN_EPS).sqrt();
    for index in 0..1024 {
        output[index] =
            ((output[index] - mean) * reciprocal_std) * norm_weight[index] + norm_bias[index];
    }
    output
}

/// Prompt audio conditioning (voice cloning).
pub struct PromptConditioning {
    /// Raw prompt patches `[P, 4, 128]` for PatchEncoder prefill.
    pub patches: Vec<f32>,
    /// g_cond from the speaker x-vector.
    pub g_cond: Vec<f32>,
}

pub enum GenerationRequest<'a> {
    Base {
        text: &'a str,
        prompt: Option<&'a PromptConditioning>,
    },
    Edit {
        source_text: &'a str,
        instruction: &'a str,
        target_text: &'a str,
        source: &'a PromptConditioning,
    },
}

impl DotsTtsModel {
    /// Extract the speaker x-vector from a 48 kHz mono prompt waveform.
    pub fn encode_speaker(&self, wav48k: &[f32]) -> Result<Vec<f32>, String> {
        let wav16k = self.speaker_resample.resample(wav48k);
        if wav16k.len() < 400 {
            return Err("prompt audio too short for the speaker encoder".into());
        }
        let mel = kaldi_fbank(&wav16k);
        #[cfg(feature = "parity-trace")]
        {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.speaker.input16k",
                None,
                &[1, wav16k.len()],
                &wav16k,
            ));
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.speaker.fbank",
                None,
                &[1, mel.len() / 80, 80],
                &mel,
            ));
        }
        let xvector = self.speaker.encode(&mel)?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.speaker.xvector",
            None,
            &[1, xvector.len()],
            &xvector,
        ));
        Ok(xvector)
    }

    /// Full prompt conditioning: speaker vector + sampled prompt latents.
    pub fn prepare_prompt_conditioning<R: Rng + ?Sized>(
        &self,
        wav48k: &[f32],
        speaker_scale: f32,
        use_xvector: bool,
        drop_tail_patch_count: usize,
        rng: &mut R,
    ) -> Result<PromptConditioning, String> {
        self.prepare_prompt_conditioning_inner(
            wav48k,
            speaker_scale,
            use_xvector,
            drop_tail_patch_count,
            |noise| {
                for value in noise {
                    *value = gaussian(rng);
                }
                Ok(())
            },
        )
    }

    #[cfg(feature = "parity-trace")]
    pub fn prepare_prompt_conditioning_with_noise(
        &self,
        wav48k: &[f32],
        speaker_scale: f32,
        use_xvector: bool,
        drop_tail_patch_count: usize,
        latent_noise: &[f32],
    ) -> Result<PromptConditioning, String> {
        self.prepare_prompt_conditioning_inner(
            wav48k,
            speaker_scale,
            use_xvector,
            drop_tail_patch_count,
            |noise| {
                if noise.len() != latent_noise.len() {
                    return Err(format!(
                        "prompt latent noise length mismatch: expected {}, got {}",
                        noise.len(),
                        latent_noise.len()
                    ));
                }
                noise.copy_from_slice(latent_noise);
                Ok(())
            },
        )
    }

    fn prepare_prompt_conditioning_inner<F>(
        &self,
        wav48k: &[f32],
        speaker_scale: f32,
        use_xvector: bool,
        drop_tail_patch_count: usize,
        mut fill_noise: F,
    ) -> Result<PromptConditioning, String>
    where
        F: FnMut(&mut [f32]) -> Result<(), String>,
    {
        if wav48k.is_empty() {
            return Err("prompt waveform must not be empty".into());
        }
        if !wav48k.iter().any(|value| value.is_finite()) {
            return Err("prompt waveform contains no finite samples".into());
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.audio.input48k",
            None,
            &[1, wav48k.len()],
            wav48k,
        ));
        let samples_per_patch = self.config.samples_per_patch();
        let target = wav48k
            .len()
            .div_ceil(samples_per_patch)
            .checked_mul(samples_per_patch)
            .ok_or_else(|| "prompt waveform padded length overflow".to_string())?;
        let mut padded = wav48k.to_vec();
        padded.resize(target, 0.0);
        let g_cond = if use_xvector {
            let xvec = self.encode_speaker(&padded)?;
            self.speaker_condition(&xvec, speaker_scale)?
        } else {
            vec![0.0; self.config.fm_hidden_size]
        };
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.condition.g_cond",
            None,
            &[1, g_cond.len()],
            &g_cond,
        ));
        let dist = self.vocoder.extract_latent_distribution(&padded)?;
        let frames = target / self.config.hop_size;
        let sampled_len = frames
            .checked_mul(self.config.latent_dim)
            .ok_or_else(|| "prompt latent shape overflow".to_string())?;
        if dist.len() != sampled_len * 2 {
            return Err(format!(
                "prompt latent distribution length mismatch: expected {}, got {}",
                sampled_len * 2,
                dist.len()
            ));
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.prompt.distribution",
            None,
            &[1, self.config.latent_dim * 2, frames],
            &dist,
        ));
        let mut noise = vec![0.0f32; sampled_len];
        fill_noise(&mut noise)?;
        let sampled = sample_latent_distribution(&dist, &noise, frames, self.config.latent_dim)?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.prompt.latents",
            None,
            &[1, frames, self.config.latent_dim],
            &sampled,
        ));
        let drop_frames = drop_tail_patch_count
            .checked_mul(self.config.patch_size)
            .ok_or_else(|| "prompt tail-drop length overflow".to_string())?;
        if drop_frames >= frames {
            return Err(format!(
                "drop_tail_patch_count removes all prompt latents: drop={drop_tail_patch_count} frames={frames}"
            ));
        }
        let keep = frames - drop_frames;
        let patch_frames = self.config.patch_size;
        let p_count = keep / patch_frames;
        let mut patches = vec![0.0f32; p_count * patch_frames * self.config.latent_dim];
        for (dst, src) in patches
            .chunks_exact_mut(self.config.latent_dim)
            .zip(sampled.chunks_exact(self.config.latent_dim))
        {
            dst.copy_from_slice(src);
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

fn fixed_noise_patch(
    noise: &[f32],
    patch_index: usize,
    patch_len: usize,
) -> Result<&[f32], String> {
    if patch_len == 0 {
        return Err("fixed DiT noise patch length must be positive".into());
    }
    let start = patch_index
        .checked_mul(patch_len)
        .ok_or_else(|| "fixed DiT noise offset overflow".to_string())?;
    let end = start
        .checked_add(patch_len)
        .ok_or_else(|| "fixed DiT noise offset overflow".to_string())?;
    noise
        .get(start..end)
        .ok_or_else(|| "fixed DiT noise is shorter than the decoded patch count".into())
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

impl GenerateOptions {
    pub fn for_model(config: &DotsTtsConfig) -> Self {
        Self {
            max_patches: 64,
            temperature: 0.9,
            nfe: config.default_nfe,
            guidance: config.default_guidance,
            speaker_scale: config.default_speaker_scale,
            eos_threshold: config.default_eos_threshold,
        }
    }
}

#[derive(Clone, Copy)]
enum FillPolicy {
    None,
    BasePrompt,
    EditSource,
}

impl FillPolicy {
    fn base(fill_patch_count: usize) -> Self {
        if fill_patch_count == 0 {
            Self::None
        } else {
            Self::BasePrompt
        }
    }

    fn fill_fm_history(self) -> bool {
        matches!(self, Self::BasePrompt)
    }

    fn drop_generated_head_patches(self) -> usize {
        usize::from(matches!(self, Self::BasePrompt))
    }

    fn decode_plan(self, target_patch_count: usize) -> Result<DecodePlan, String> {
        let drop_count = self.drop_generated_head_patches();
        let scheduled_patch_count = target_patch_count
            .checked_add(drop_count)
            .ok_or_else(|| "dots decode patch count overflow".to_string())?;
        Ok(DecodePlan {
            scheduled_patch_count,
            drop_count,
        })
    }
}

struct DecodePlan {
    scheduled_patch_count: usize,
    drop_count: usize,
}

struct DecodeStep {
    noise_patch_index: usize,
    should_check_eos: bool,
    should_emit: bool,
}

impl DecodePlan {
    fn step(&self, decoded_index: usize) -> DecodeStep {
        let is_payload = decoded_index >= self.drop_count;
        DecodeStep {
            noise_patch_index: decoded_index,
            should_check_eos: is_payload,
            should_emit: is_payload,
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
    patch_encoder_state: PatchEncoderState,
}

impl<'a> DotsGenerateSession<'a> {
    pub fn new(model: &'a DotsTtsModel, capacity_patches: usize) -> Result<Self, String> {
        let capacity = capacity_patches
            .checked_mul(model.config.unit_len())
            .and_then(|value| value.checked_add(8))
            .ok_or_else(|| "dots generation capacity overflow".to_string())?;
        let fm_capacity = capacity
            .checked_mul(model.config.fm_hidden_size)
            .ok_or_else(|| "dots FM buffer size overflow".to_string())?;
        let patch_encoder_capacity = capacity_patches
            .checked_mul(2)
            .and_then(|value| value.checked_add(8))
            .ok_or_else(|| "dots patch-encoder capacity overflow".to_string())?;
        Ok(Self {
            model,
            llm: model.llm.new_session()?,
            fm: vec![0.0; fm_capacity],
            fm_cfg: vec![0.0; fm_capacity],
            fm_seq_len: 0,
            patch_encoder_state: model.patch_encoder.new_state(patch_encoder_capacity),
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
        // CFG branch: hidden_proj(zeros) is the projection bias.
        let start_cfg = self.fm_seq_len * fm;
        self.fm_cfg[start_cfg..start_cfg + fm].copy_from_slice(&self.model.hidden_proj.1);
        self.fm_seq_len += 1;
        Ok(())
    }

    fn append_history_chunk(&mut self, latents: &[f32]) -> Result<(), String> {
        // latents: [patch_size, latent_dim] (normalized space)
        let fm = self.model.config.fm_hidden_size;
        let p = self.model.config.patch_size;
        let d = self.model.config.latent_dim;
        let mut projected = vec![0.0f32; p * fm];
        linear_forward(
            &self.model.latent_proj.0,
            Some(&self.model.latent_proj.1),
            latents,
            d,
            fm,
            &mut projected,
        );
        self.append_projected_history(&projected);
        Ok(())
    }

    fn append_prompt_history_chunk(&mut self, latents: &[f32]) -> Result<(), String> {
        let fm = self.model.config.fm_hidden_size;
        let p = self.model.config.patch_size;
        let d = self.model.config.latent_dim;
        let mut transposed = vec![0.0f32; p * d];
        for frame in 0..p {
            for channel in 0..d {
                transposed[channel * p + frame] = latents[frame * d + channel];
            }
        }
        let mut projected = vec![0.0f32; p * fm];
        linear_forward_transposed_input_then_bias(
            &self.model.latent_proj.0,
            &self.model.latent_proj.1,
            &transposed,
            p,
            d,
            fm,
            &mut projected,
        );
        self.append_projected_history(&projected);
        Ok(())
    }

    fn append_projected_history(&mut self, projected: &[f32]) {
        let fm = self.model.config.fm_hidden_size;
        let rows = projected.len() / fm;
        let start = self.fm_seq_len * fm;
        let end = start + projected.len();
        self.fm[start..end].copy_from_slice(projected);
        self.fm_cfg[start..end].copy_from_slice(projected);
        self.fm_seq_len += rows;
    }

    fn decode_next_patch(
        &mut self,
        g_cond: &[f32],
        options: &GenerateOptions,
        z0: &[f32],
    ) -> Result<Vec<f32>, String> {
        let p = self.model.config.patch_size;
        let d = self.model.config.latent_dim;
        let seq = &self.fm[..self.fm_seq_len * self.model.config.fm_hidden_size];
        let cfg = &self.fm_cfg[..self.fm_seq_len * self.model.config.fm_hidden_size];
        let mut patch = vec![0.0f32; p * d];
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.fm.sequence",
            None,
            &[1, self.fm_seq_len, self.model.config.fm_hidden_size],
            seq,
        ));
        self.model.dit.solve_patch(
            seq,
            cfg,
            self.fm_seq_len,
            g_cond,
            &self.model.coordinate_proj.0,
            &self.model.coordinate_proj.1,
            options.guidance,
            options.nfe,
            z0,
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
    request: GenerationRequest<'_>,
    options: &GenerateOptions,
    rng: &mut R,
) -> Result<Vec<f32>, String> {
    generate_latents_inner(model, tokenizer, request, options, |_, noise| {
        for value in noise {
            *value = gaussian(rng);
        }
        Ok(())
    })
}

#[cfg(feature = "parity-trace")]
pub fn generate_latents_with_noise<R: Rng + ?Sized>(
    model: &DotsTtsModel,
    tokenizer: &crate::core::tokenizer::BPETokenizer,
    request: GenerationRequest<'_>,
    options: &GenerateOptions,
    _rng: &mut R,
    fixed_noise: &[f32],
) -> Result<Vec<f32>, String> {
    crate::models::dots::dit::reset_internal_trace();
    let patch_len = model
        .config
        .patch_size
        .checked_mul(model.config.latent_dim)
        .ok_or_else(|| "dots latent patch size overflow".to_string())?;
    if patch_len == 0 || fixed_noise.len() % patch_len != 0 {
        return Err(format!(
            "fixed DiT noise length must be a multiple of {patch_len}, got {}",
            fixed_noise.len()
        ));
    }
    generate_latents_inner(model, tokenizer, request, options, |patch_index, noise| {
        noise.copy_from_slice(fixed_noise_patch(fixed_noise, patch_index, noise.len())?);
        Ok(())
    })
}

fn generate_latents_inner<F>(
    model: &DotsTtsModel,
    tokenizer: &crate::core::tokenizer::BPETokenizer,
    request: GenerationRequest<'_>,
    options: &GenerateOptions,
    mut fill_noise: F,
) -> Result<Vec<f32>, String>
where
    F: FnMut(usize, &mut [f32]) -> Result<(), String>,
{
    let patch_len = model
        .config
        .patch_size
        .checked_mul(model.config.latent_dim)
        .ok_or_else(|| "dots latent patch size overflow".to_string())?;
    if patch_len == 0 {
        return Err("dots latent patch size must be positive".into());
    }
    let (conditioning, fill_policy, decode_plan, schedule) = match request {
        GenerationRequest::Base { text, prompt } => {
            let fill_patch_count = match prompt {
                Some(prompt) => {
                    if prompt.patches.len() % patch_len != 0 {
                        return Err("base prompt conditioning is not patch-sized".into());
                    }
                    prompt.patches.len() / patch_len
                }
                None => 0,
            };
            let fill_policy = FillPolicy::base(fill_patch_count);
            let decode_plan = fill_policy.decode_plan(options.max_patches)?;
            let schedule = build_generation_schedule(
                tokenizer,
                text,
                fill_patch_count,
                decode_plan.scheduled_patch_count,
            )?;
            (prompt, fill_policy, decode_plan, schedule)
        }
        GenerationRequest::Edit {
            source_text,
            instruction,
            target_text,
            source,
        } => {
            if source.patches.len() % patch_len != 0 {
                return Err("edit source conditioning is not patch-sized".into());
            }
            let fill_patch_count = source.patches.len() / patch_len;
            let fill_policy = FillPolicy::EditSource;
            let decode_plan = fill_policy.decode_plan(options.max_patches)?;
            let schedule = build_edit_generation_schedule(
                tokenizer,
                source_text,
                instruction,
                target_text,
                fill_patch_count,
                decode_plan.scheduled_patch_count,
            )?;
            (Some(source), fill_policy, decode_plan, schedule)
        }
    };
    let fill_patch_count = conditioning.map_or(0, |value| value.patches.len() / patch_len);
    let span_ids = DotsSchedule::audio_span_ids(tokenizer)?;
    let fill_span_positions = &schedule.fill_span_positions;
    let decode_span_positions = &schedule.decode_span_positions;
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::token_ids(
        "dots.schedule.ids",
        &schedule.ids,
    ));

    let prefill_end = *decode_span_positions
        .first()
        .ok_or_else(|| "generation schedule provides no decode spans".to_string())?;
    if fill_span_positions.len() != fill_patch_count {
        return Err(format!(
            "generation schedule provides {} fill spans; prompt requires {}",
            fill_span_positions.len(),
            fill_patch_count
        ));
    }

    let capacity_patches = fill_span_positions
        .len()
        .checked_add(decode_span_positions.len())
        .ok_or_else(|| "generation schedule patch count overflow".to_string())?;
    let mut session = DotsGenerateSession::new(model, capacity_patches)?;

    // ---- LLM prefill -------------------------------------------------- //
    let mut hiddens: Vec<Vec<f32>> = Vec::with_capacity(prefill_end);
    let prompt_embeds = if let Some(conditioning) = conditioning {
        model
            .patch_encoder
            .prefill(&conditioning.patches, &mut session.patch_encoder_state)?
    } else {
        Vec::new()
    };
    let mut fill_cursor = 0usize;
    let mut prefill_rows = Vec::with_capacity(prefill_end);
    for pos in 0..prefill_end {
        let id = schedule.ids[pos];
        let row =
            if fill_cursor < fill_span_positions.len() && fill_span_positions[fill_cursor] == pos {
                let start = fill_cursor * model.config.llm_hidden_size;
                fill_cursor += 1;
                LlmInputRow::Embedding(&prompt_embeds[start..start + model.config.llm_hidden_size])
            } else {
                LlmInputRow::Token(id)
            };
        prefill_rows.push(row);
    }
    let prefill_hidden = session.llm.prefill_rows(&prefill_rows)?;
    for row in prefill_hidden.chunks_exact(model.config.llm_hidden_size) {
        hiddens.push(row.to_vec());
    }

    // ---- FM buffer assembly from the prefill --------------------------- //
    let mut cursor = 0usize;
    if fill_policy.fill_fm_history() {
        if let Some(conditioning) = conditioning {
            for (span_idx, &span_position) in fill_span_positions.iter().enumerate() {
                if span_position > cursor {
                    session.append_hidden_chunk(&hiddens[span_position - 1])?;
                }
                let patch = &conditioning.patches[span_idx * patch_len..(span_idx + 1) * patch_len];
                let mut normalized = patch.to_vec();
                for frame in normalized.chunks_exact_mut(model.config.latent_dim) {
                    model.normalize(frame);
                }
                session.append_prompt_history_chunk(&normalized)?;
                if span_position + 1 < schedule.ids.len()
                    && span_ids.contains(&schedule.ids[span_position + 1])
                {
                    session.append_hidden_chunk(&hiddens[span_position])?;
                }
                cursor = span_position + 1;
            }
        }
    }
    if prefill_end > cursor {
        session.append_hidden_chunk(&hiddens[prefill_end - 1])?;
    }

    // ---- decode loop --------------------------------------------------- //
    let g_cond = match conditioning {
        Some(conditioning) => conditioning.g_cond.clone(),
        None => vec![0.0f32; model.config.fm_hidden_size],
    };
    if g_cond.len() != model.config.fm_hidden_size {
        return Err("dots speaker conditioning width mismatch".into());
    }
    let mut raw_patches: Vec<f32> = Vec::new();
    let mut position = prefill_end;
    for (decoded_index, &decode_position) in decode_span_positions.iter().enumerate() {
        let decode_step = decode_plan.step(decoded_index);
        if decode_position < position {
            return Err("dots decode span positions are not ascending".into());
        }
        while position < decode_position {
            let id = schedule.ids[position];
            session.llm.step_row(LlmInputRow::Token(id))?;
            position += 1;
        }
        if !span_ids.contains(&schedule.ids[decode_position]) {
            return Err("dots decode position is not an audio span".into());
        }
        if decoded_index > 0 {
            let hidden = session.llm.last_hidden().to_vec();
            session.append_hidden_chunk(&hidden)?;
        }
        let stop_after = decode_step.should_check_eos
            && model.eos_probability(session.llm.last_hidden())? > options.eos_threshold;
        let mut z0 = vec![0.0f32; patch_len];
        fill_noise(decode_step.noise_patch_index, &mut z0)?;
        let patch = session.decode_next_patch(&g_cond, options, &z0)?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.latent.consumed",
            None,
            &[1, model.config.patch_size, model.config.latent_dim],
            &patch,
        ));
        session.append_history_chunk(&patch)?;
        let raw = model.denormalize(&patch);
        let embedding = model
            .patch_encoder
            .encode_patch(&raw, &mut session.patch_encoder_state)?;
        session.llm.step_row(LlmInputRow::Embedding(&embedding))?;
        if decode_step.should_emit {
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.latent.payload",
                None,
                &[1, model.config.patch_size, model.config.latent_dim],
                &raw,
            ));
            raw_patches.extend_from_slice(&raw);
        }
        position += 1;
        if stop_after {
            break;
        }
    }
    if raw_patches.is_empty() {
        return Err("generation produced no latent patches (EOS before the first patch)".into());
    }
    Ok(raw_patches)
}

/// Full synthesis: text → latent patches → 48 kHz mono waveform.
pub fn synthesize_request<R: Rng + ?Sized>(
    model: &DotsTtsModel,
    tokenizer: &crate::core::tokenizer::BPETokenizer,
    request: GenerationRequest<'_>,
    options: &GenerateOptions,
    rng: &mut R,
) -> Result<Vec<f32>, String> {
    let latents = generate_latents(model, tokenizer, request, options, rng)?;
    // latents: [frames, 128] raw
    if latents.len() % model.config.latent_dim != 0 {
        return Err("latent stream width mismatch".into());
    }
    model.vocoder.decode_latents(&latents)
}

#[cfg(feature = "parity-trace")]
pub fn synthesize_request_with_noise<R: Rng + ?Sized>(
    model: &DotsTtsModel,
    tokenizer: &crate::core::tokenizer::BPETokenizer,
    request: GenerationRequest<'_>,
    options: &GenerateOptions,
    rng: &mut R,
    fixed_noise: &[f32],
) -> Result<Vec<f32>, String> {
    let latents =
        generate_latents_with_noise(model, tokenizer, request, options, rng, fixed_noise)?;
    if latents.len() % model.config.latent_dim != 0 {
        return Err("latent stream width mismatch".into());
    }
    model.vocoder.decode_latents(&latents)
}

#[cfg(feature = "parity-trace")]
#[doc(hidden)]
pub fn read_dots_wav_for_parity(
    path: &std::path::Path,
    edit: bool,
    samples_per_patch: usize,
) -> Result<Vec<f32>, String> {
    crate::app::dots::read_dots_wav_for_parity(path, edit, samples_per_patch)
}

#[cfg(test)]
mod tests {
    use super::{
        fixed_noise_patch, sample_latent_distribution, speaker_condition_forward, FillPolicy,
        GenerateOptions, GenerationRequest, PromptConditioning,
    };
    use crate::models::dots::config::DotsTtsConfig;

    #[test]
    #[ignore = "requires DOTS_PROMPT_DISTRIBUTION, DOTS_PROMPT_NOISE, and DOTS_PROMPT_LATENTS"]
    fn prompt_latent_sampling_matches_pinned_torch_bitwise() {
        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let distribution = read("DOTS_PROMPT_DISTRIBUTION");
        let noise = read("DOTS_PROMPT_NOISE");
        let expected = read("DOTS_PROMPT_LATENTS");

        let actual = sample_latent_distribution(&distribution, &noise, 148, 128).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "dots.prompt.latents[{index}]"
            );
        }
    }

    #[test]
    #[ignore = "requires DOTS_GCOND_MMPROJ, DOTS_GCOND_XVECTOR, and DOTS_GCOND_ORACLE"]
    fn real_xvector_projection_matches_pinned_oracle_bitwise() {
        use crate::{open_model_source, ComponentRole};

        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let path = std::path::PathBuf::from(std::env::var_os("DOTS_GCOND_MMPROJ").unwrap());
        let source = open_model_source(&path, ComponentRole::Mmproj).unwrap();
        let load =
            |name: &str, dims: &[u64]| super::load_f16_f32(source.as_ref(), name, dims).unwrap();
        let xvector = read("DOTS_GCOND_XVECTOR");
        let expected = read("DOTS_GCOND_ORACLE");
        let actual = speaker_condition_forward(
            &xvector,
            1.5,
            &load("dotstts.xvec_proj.0.weight", &[512, 1024]),
            &load("dotstts.xvec_proj.0.bias", &[1024]),
            &load("dotstts.xvec_proj.1.weight", &[1024]),
            &load("dotstts.xvec_proj.1.bias", &[1024]),
        );
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "g_cond[{index}]");
        }
    }

    #[test]
    fn generation_requests_borrow_the_exact_base_and_edit_inputs() {
        let source = PromptConditioning {
            patches: Vec::new(),
            g_cond: Vec::new(),
        };
        let base = GenerationRequest::Base {
            text: "[EN]hello",
            prompt: Some(&source),
        };
        let edit = GenerationRequest::Edit {
            source_text: "old",
            instruction: "<sub target=\"new\">old</sub>",
            target_text: "new",
            source: &source,
        };
        assert!(matches!(
            base,
            GenerationRequest::Base {
                text: "[EN]hello",
                ..
            }
        ));
        assert!(matches!(
            edit,
            GenerationRequest::Edit {
                target_text: "new",
                ..
            }
        ));
    }

    #[test]
    fn model_sampling_defaults_seed_generate_options() {
        let config = DotsTtsConfig {
            patch_size: 4,
            latent_dim: 128,
            hop_size: 1920,
            sample_rate: 48_000,
            fm_hidden_size: 1024,
            llm_hidden_size: 1536,
            xvec_dim: 512,
            patch_encoder_layers: 24,
            dit_layers: 18,
            dit_heads: 16,
            default_nfe: 7,
            default_guidance: 1.3,
            default_speaker_scale: 1.6,
            default_eos_threshold: 0.75,
        };
        let options = GenerateOptions::for_model(&config);
        assert_eq!(options.nfe, 7);
        assert_eq!(options.guidance, 1.3);
        assert_eq!(options.speaker_scale, 1.6);
        assert_eq!(options.eos_threshold, 0.75);
    }

    #[test]
    fn decode_plan_preserves_target_budget_and_consumes_noise_for_every_solved_patch() {
        let fixed_noise = [10.0, 11.0, 20.0, 21.0];

        let conditioned_base = FillPolicy::BasePrompt.decode_plan(1).unwrap();
        assert_eq!(conditioned_base.scheduled_patch_count, 2);
        let head = conditioned_base.step(0);
        assert_eq!((head.should_check_eos, head.should_emit), (false, false));
        assert_eq!(
            fixed_noise_patch(&fixed_noise, head.noise_patch_index, 2).unwrap(),
            &[10.0, 11.0]
        );
        let payload = conditioned_base.step(1);
        assert_eq!(
            (payload.should_check_eos, payload.should_emit),
            (true, true)
        );
        assert_eq!(
            fixed_noise_patch(&fixed_noise, payload.noise_patch_index, 2).unwrap(),
            &[20.0, 21.0]
        );

        for policy in [FillPolicy::EditSource, FillPolicy::None] {
            let plan = policy.decode_plan(1).unwrap();
            assert_eq!(plan.scheduled_patch_count, 1);
            let payload = plan.step(0);
            assert_eq!(
                (payload.should_check_eos, payload.should_emit),
                (true, true)
            );
            assert_eq!(
                fixed_noise_patch(&fixed_noise, payload.noise_patch_index, 2).unwrap(),
                &[10.0, 11.0]
            );
        }
    }

    #[test]
    fn base_and_edit_fill_policies_are_not_interchangeable() {
        assert!(!FillPolicy::None.fill_fm_history());
        assert_eq!(FillPolicy::None.drop_generated_head_patches(), 0);
        assert!(FillPolicy::BasePrompt.fill_fm_history());
        assert_eq!(FillPolicy::BasePrompt.drop_generated_head_patches(), 1);
        assert!(!FillPolicy::EditSource.fill_fm_history());
        assert_eq!(FillPolicy::EditSource.drop_generated_head_patches(), 0);
    }

    #[test]
    fn base_policy_requires_llm_fill_patches() {
        assert!(!FillPolicy::base(0).fill_fm_history());
        assert_eq!(FillPolicy::base(0).drop_generated_head_patches(), 0);
        assert!(FillPolicy::base(1).fill_fm_history());
        assert_eq!(FillPolicy::base(1).drop_generated_head_patches(), 1);
    }
}
