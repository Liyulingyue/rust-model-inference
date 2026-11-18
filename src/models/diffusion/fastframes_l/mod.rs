//! Native FastFrames-L video generation.

#[derive(Clone, Debug)]
pub struct Options {
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps: usize,
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            width: 832,
            height: 480,
            frames: 81,
            fps: 16,
            seed: 0,
        }
    }
}

impl Options {
    pub fn validate(&self) -> Result<(), String> {
        if self.width == 0
            || self.width > 832
            || !self.width.is_multiple_of(16)
            || self.height == 0
            || self.height > 480
            || !self.height.is_multiple_of(16)
            || !(1..=81).contains(&self.frames)
            || !(1..=60).contains(&self.fps)
        {
            return Err(
                "FastFrames-L requires multiples of 16 up to 832x480, 1..=81 frames and fps 1..=60"
                    .into(),
            );
        }
        Ok(())
    }

    pub fn latent_frames(&self) -> usize {
        (self.frames + 3).div_ceil(4)
    }
}

fn sigma(timestep: f32) -> f32 {
    (0..1000)
        .map(|i| {
            let base = if i < 500 {
                (-0.001f32).mul_add(i as f32, 1.)
            } else {
                0.001f32 * (1000 - i) as f32
            };
            5. * base / (1. + 4. * base)
        })
        .min_by(|a, b| {
            (1000. * a - timestep)
                .abs()
                .total_cmp(&(1000. * b - timestep).abs())
        })
        .unwrap()
}

fn patchify(latent: &[f32], height: usize, width: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(latent.len());
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            for c in 0..16 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        result.push(latent[(c * height + y + dy) * width + x + dx]);
                    }
                }
            }
        }
    }
    result
}

fn unpatchify(tokens: &[f32], height: usize, width: usize) -> Vec<f32> {
    let mut result = vec![0.; 16 * height * width];
    for y in 0..height {
        for x in 0..width {
            for c in 0..16 {
                result[(c * height + y) * width + x] =
                    tokens[((y / 2 * (width / 2) + x / 2) * 4 + y % 2 * 2 + x % 2) * 16 + c];
            }
        }
    }
    result
}

#[cfg(test)]
mod tests;

mod decoder;
mod dit;
mod math;
mod refiner;
mod text;
mod weights;

use crate::compute::ComputePolicy;
use crate::core::{tensor::TensorSource, thread_pool::ComputePool};
use crate::models::diffusion::z_image::dit::TorchMt19937;
use std::sync::Arc;
use weights::Graph;

pub use weights::matches_model;

pub struct Engine {
    source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    tokenizer: tokenizers::Tokenizer,
}

pub struct Video {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps: usize,
}

impl Engine {
    pub fn load(source: Arc<dyn TensorSource>, threads: usize) -> Result<Self, String> {
        if threads == 0 {
            return Err("FastFrames-L thread count must be positive".into());
        }
        Graph::load(source.as_ref())?;
        let tokenizer = text::tokenizer(source.as_ref())?;
        Ok(Self {
            source,
            pool: Arc::new(ComputePool::new(threads)),
            tokenizer,
        })
    }

    pub fn generate(
        &self,
        prompt: &str,
        options: &Options,
        policy: ComputePolicy,
        noise: Option<&[f32]>,
    ) -> Result<Video, String> {
        options.validate()?;
        let _scope = policy.enter_legacy_scope();
        if prompt.len() > 65536 {
            return Err("FastFrames-L prompt exceeds 64 KiB".into());
        }
        let ids = text::ids(&self.tokenizer, prompt)?;
        let h = options.height / 8;
        let w = options.width / 8;
        let tokens = h * w / 4;
        let frames = options.latent_frames();
        let count = frames * 16 * h * w;
        if let Some(noise) = noise {
            if noise.len() != 4 * count || noise.iter().any(|v| !v.is_finite()) {
                return Err(format!("FastFrames-L shared noise needs {} finite F32 values (initial + 3 transitions)",4*count));
            }
        }
        let g = Graph::load(self.source.as_ref())?;
        let mut ex = g.executor(policy, tokens.max(512), Arc::clone(&self.pool))?;
        eprintln!("FastFrames-L: requested_compute={policy:?}; activations/KV=F32; text relative-bias attention and normalization on CPU");
        let embeds = text::encode(&g, &mut ex, &ids)?;
        let text_cache = dit::text_cache(&g, &mut ex, &embeds)?;
        let noise = if let Some(noise) = noise {
            noise.to_vec()
        } else {
            let mut noise = vec![0.; 4 * count];
            TorchMt19937::new(options.seed).fill_normal(&mut noise);
            noise
        };
        math::trace("fastframes.noise", &noise);
        let mut cache = (0..40)
            .map(|_| dit::Cache {
                k: vec![0.; frames * tokens * 5120],
                v: vec![0.; frames * tokens * 5120],
            })
            .collect::<Vec<_>>();
        let mut features = Vec::with_capacity(frames * tokens * 5120);
        let mut modulation = Vec::with_capacity(frames * 5120);
        let times = [1000., 937., 833., 625., 0.]
            .into_iter()
            .map(|t| dit::time(&g, &mut ex, t).map(|time| (t as usize, time)))
            .collect::<Result<std::collections::BTreeMap<_, _>, _>>()?;
        for frame in 0..frames {
            let schedule: &[usize] = if frame == 0 {
                &[1000, 937, 833, 625]
            } else {
                &[1000, 833]
            };
            let start = frame * 16 * h * w;
            let end = start + 16 * h * w;
            let mut x = noise[start..end].to_vec();
            for (step, &t) in schedule.iter().enumerate() {
                let (flow, feature) = dit::predict(
                    &g,
                    &mut ex,
                    &x,
                    h,
                    w,
                    frame,
                    &times[&t],
                    &text_cache,
                    &mut cache,
                    policy,
                    &self.pool,
                )?;
                let current_sigma = sigma(t as f32) as f64;
                for (x, f) in x.iter_mut().zip(flow) {
                    *x = (*x as f64 - current_sigma * f as f64) as f32;
                }
                if x.iter().any(|v| !v.is_finite()) {
                    return Err(format!(
                        "non-finite FastFrames-L latent at frame {frame}, timestep {t}"
                    ));
                }
                math::trace("fastframes.x0", &x);
                if let Some(&next) = schedule.get(step + 1) {
                    let s = sigma(next as f32);
                    let eps = &noise[(step + 1) * count + start..(step + 1) * count + end];
                    for (x, eps) in x.iter_mut().zip(eps) {
                        *x = (1. - s) * *x + s * eps;
                    }
                } else {
                    features.extend(feature);
                    modulation.extend_from_slice(&times[&t].head);
                }
            }
            if frame == 0 {
                dit::predict(
                    &g,
                    &mut ex,
                    &x,
                    h,
                    w,
                    frame,
                    &times[&0],
                    &text_cache,
                    &mut cache,
                    policy,
                    &self.pool,
                )?;
            } else {
                let stem = dit::embed(&g, &mut ex, &x, h, w)?;
                refiner::refine(&g, &mut ex, &stem, frame, &mut cache)?;
            }
            eprintln!("FastFrames-L: latent frame {}/{}", frame + 1, frames);
        }
        drop(cache);
        drop(text_cache);
        let rgb = decoder::decode(&g, &mut ex, &self.pool, &features, &modulation, options)?;
        Ok(Video {
            rgb,
            width: options.width,
            height: options.height,
            frames: options.frames,
            fps: options.fps,
        })
    }
}
