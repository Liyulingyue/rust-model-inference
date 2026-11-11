//! Linum P-JiT 21K: Qwen3.5 caption features, pixel-space DiT and APG/Euler.
mod dit;
#[cfg(test)]
mod tests;
use crate::{
    compute::ComputePolicy,
    core::{
        tensor::TensorSource,
        thread_pool::ComputePool,
        tokenizer::{BPETokenizer, EncodeOptions},
    },
};
pub use dit::PjitDit;
use std::sync::Arc;
const HIDDEN: usize = 2944;
const TEXT_WIDTH: usize = 7680;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct PjitOptions {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub seed: u64,
    pub guidance: f32,
    pub negative_prompt: String,
}
impl Default for PjitOptions {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            steps: 50,
            seed: 0,
            guidance: 15.,
            negative_prompt: String::new(),
        }
    }
}
impl PjitOptions {
    pub fn validate(&self) -> Result<(), String> {
        validate_shape(self.width, self.height)?;
        if !(1..=1000).contains(&self.steps)
            || !self.guidance.is_finite()
            || self.guidance < 0.
            || self.guidance > 100.
        {
            return Err("P-JiT requires steps 1..=1000 and finite guidance 0..=100".into());
        }
        Ok(())
    }
}
fn validate_shape(width: usize, height: usize) -> Result<(), String> {
    if width == 0
        || height == 0
        || width > 2048
        || height > 2048
        || width.checked_mul(height).is_none_or(|n| n > 2_097_152)
    {
        return Err("P-JiT size must be positive, at most 2048 per side and 2097152 pixels".into());
    }
    Ok(())
}
fn architecture(source: &dyn TensorSource) -> &str {
    source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or("")
}
fn finite(x: &[f32], name: &str) -> Result<(), String> {
    if x.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(format!("Non-finite P-JiT {name}"))
    }
}

pub struct PjitPipeline {
    dit: PjitDit,
    text: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
}
impl PjitPipeline {
    pub fn load(
        dit: Arc<dyn TensorSource>,
        text: Arc<dyn TensorSource>,
        threads: usize,
    ) -> Result<Self, String> {
        let config = crate::models::qwen35::Qwen35Config::from_source(text.as_ref())?;
        if architecture(text.as_ref()) != "qwen35"
            || config.n_embd != 2560
            || config.n_layer_impl() != 32
            || config.n_ff != 9216
            || config.n_head != 16
            || config.n_head_kv != 4
            || config.vocab_size != 248320
            || config.key_length != 256
            || config
                .is_recurrent
                .iter()
                .enumerate()
                .any(|(i, &recurrent)| recurrent != ((i + 1) % 4 != 0))
        {
            return Err("P-JiT requires the Qwen3.5-4B text component (2560-wide, 32 blocks, dense blocks 7/15/27)".into());
        }
        Ok(Self {
            dit: PjitDit::load(dit, threads)?,
            text,
            pool: Arc::new(ComputePool::new(threads)),
        })
    }
    pub fn encode(&self, prompt: &str) -> Result<Vec<f32>, String> {
        // The existing resident GPU decoder exposes normalized final states only.
        // shortcut: recompute three CPU prefixes, replace when shared multi-layer feature extraction exists.
        let _scope = ComputePolicy::Cpu.cpu_scope();
        let ids = caption_ids(self.text.as_ref(), prompt)?;
        if ids.is_empty() {
            return Err("P-JiT caption template produced no tokens".into());
        }
        trace(
            "pjit.token_ids",
            &[ids.len()],
            &ids.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        )?;
        let mut model = crate::models::qwen35::Qwen35Model::from_source(self.text.as_ref())?;
        model.config.n_ctx = 256;
        model.config.n_nextn = 0;
        let mut output = vec![0.; ids.len() * TEXT_WIDTH];
        for (slot, count) in [8, 16, 28].into_iter().enumerate() {
            model.config.n_layer = count;
            let mut scratch =
                crate::models::qwen35::Qwen35Scratchpad::new(&model.config, ids.len());
            let mut kv =
                crate::core::scratchpad::KvCache::new_f32(count, 256, model.config.n_embd_gqa());
            scratch.x.copy_from_slice(&model.embed_tokens(&ids)?);
            let positions: Vec<_> = (0..ids.len()).map(|i| [i; 4]).collect();
            model.forward(ids.len(), &mut kv, &mut scratch, &self.pool, &positions)?;
            for (dst, src) in output
                .chunks_exact_mut(TEXT_WIDTH)
                .zip(scratch.x.chunks_exact(2560))
            {
                for i in 0..2560 {
                    dst[slot * 2560 + i] = half::bf16::from_f32(src[i]).to_f32();
                }
            }
        }
        trace("pjit.text", &[ids.len(), TEXT_WIDTH], &output)?;
        finite(&output, "caption features")?;
        Ok(output)
    }
    pub fn generate(
        &self,
        prompt: &str,
        options: &PjitOptions,
        noise: Option<&[f32]>,
    ) -> Result<super::DiffusionRgb, String> {
        options.validate()?;
        if prompt.trim().is_empty() {
            return Err("P-JiT prompt must not be empty".into());
        }
        let text = self.encode(prompt)?;
        let negative = self.encode(&options.negative_prompt)?;
        let count = 3 * options.width * options.height;
        let mut x = if let Some(noise) = noise {
            if noise.len() != count {
                return Err("P-JiT noise shape mismatch".into());
            }
            finite(noise, "noise")?;
            noise.to_vec()
        } else {
            let mut noise = vec![0.; count];
            super::z_image::dit::TorchMt19937::new(options.seed).fill_normal(&mut noise);
            for x in &mut noise {
                *x *= 4.;
            }
            noise
        };
        trace(
            "pjit.initial_noise",
            &[3, options.height, options.width],
            &x,
        )?;
        let mut momentum = vec![0.; count];
        let mut session = self.dit.session()?;
        for i in 0..options.steps {
            let t = sampling_time(i, options.steps);
            let next = sampling_time(i + 1, options.steps);
            let input: Vec<_> = x
                .iter()
                .map(|&v| half::bf16::from_f32(v).to_f32())
                .collect();
            let timestep = (t * 1000.) as usize;
            let conditional =
                session.predict(&input, options.width, options.height, &text, timestep)?;
            let other =
                session.predict(&input, options.width, options.height, &negative, timestep)?;
            let vc: Vec<_> = input
                .iter()
                .zip(conditional)
                .map(|(&x, p)| (x - p) / t)
                .collect();
            let vu: Vec<_> = input.iter().zip(other).map(|(&x, p)| (x - p) / t).collect();
            for v in &mut momentum {
                *v = half::bf16::from_f32(*v).to_f32();
            }
            let guided = apg(&vc, &vu, &mut momentum, options.guidance);
            for j in 0..count {
                let predicted = input[j] - t * guided[j];
                x[j] += (next - t) * ((x[j] - predicted) / t);
            }
            finite(&x, "sampler state")?;
            trace("pjit.euler", &[3, options.height, options.width], &x)?;
            eprintln!("P-JiT step {}/{}", i + 1, options.steps);
        }
        let mut bytes = vec![0; count];
        for i in 0..options.width * options.height {
            for c in 0..3 {
                bytes[i * 3 + c] = ((x[c * options.width * options.height + i].clamp(-1., 1.) + 1.)
                    * 127.5)
                    .round_ties_even() as u8;
            }
        }
        Ok(super::DiffusionRgb {
            width: options.width as u32,
            height: options.height as u32,
            bytes,
        })
    }
}
fn render_caption(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n")
}
fn caption_ids(source: &dyn TensorSource, prompt: &str) -> Result<Vec<u32>, String> {
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|e| e.to_string())?;
    let mut ids = tokenizer.encode(
        &render_caption(prompt),
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    ids.truncate(256);
    Ok(ids)
}
fn sampling_time(i: usize, steps: usize) -> f32 {
    // torch.linspace uses symmetric endpoints and fused multiply-add; integer timesteps depend on it.
    let step = -1.0 / steps as f32;
    if i < (steps + 1) / 2 {
        step.mul_add(i as f32, 1.)
    } else {
        (-step).mul_add((steps - i) as f32, 0.)
    }
}
fn apg(cond: &[f32], other: &[f32], momentum: &mut [f32], scale: f32) -> Vec<f32> {
    for i in 0..cond.len() {
        momentum[i] = cond[i] - other[i] - 0.75 * momentum[i];
    }
    let norm = momentum
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let clip = (10. / norm).min(1.) as f32;
    let diff: Vec<_> = momentum.iter().map(|&v| v * clip).collect();
    let norm = cond
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        .max(1e-24);
    let projection = diff
        .iter()
        .zip(cond)
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum::<f64>()
        / norm;
    cond.iter()
        .zip(diff)
        .map(|(&c, d)| c + scale * (d as f64 - projection * c as f64) as f32)
        .collect()
}
fn patchify(image: &[f32], width: usize, height: usize) -> Vec<f32> {
    let gw = width.div_ceil(32);
    let gh = height.div_ceil(32);
    let mut out = vec![0.; gw * gh * 3072];
    for py in 0..gh {
        for px in 0..gw {
            for c in 0..3 {
                for y in 0..32 {
                    for x in 0..32 {
                        if py * 32 + y < height && px * 32 + x < width {
                            out[((py * gw + px) * 3 + c) * 1024 + y * 32 + x] =
                                image[(c * height + py * 32 + y) * width + px * 32 + x];
                        }
                    }
                }
            }
        }
    }
    out
}
fn unpatchify(patches: &[f32], width: usize, height: usize) -> Vec<f32> {
    let gw = width.div_ceil(32);
    let mut out = vec![0.; 3 * width * height];
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                out[(c * height + y) * width + x] =
                    patches[((y / 32 * gw + x / 32) * 1024 + (y % 32) * 32 + x % 32) * 3 + c];
            }
        }
    }
    out
}
fn trace(_name: &str, _shape: &[usize], _values: &[f32]) -> Result<(), String> {
    #[cfg(feature = "parity-trace")]
    if crate::parity_trace::enabled(_name) {
        crate::parity_trace::checkpoint(_name, None, _shape, _values).map_err(|e| e.to_string())?;
    }
    Ok(())
}
