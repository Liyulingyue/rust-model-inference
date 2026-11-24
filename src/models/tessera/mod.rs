//! Tessera-small, a CLIP-conditioned discrete diffusion model for palette sprites.
mod denoiser;
mod palette;
mod sampler;
mod text;

pub use palette::Pixel;
pub use sampler::{Options, PixelOptions};

use crate::{
    compute::ComputePolicy,
    core::{
        tensor::{MetaValue, TensorSource},
        thread_pool::ComputePool,
    },
};
use std::sync::Arc;

const COLORS: usize = 512;
const MASK: u16 = COLORS as u16;
pub const GRID: usize = 16;
const PIXELS: usize = GRID * GRID;

pub fn matches(source: &dyn TensorSource) -> bool {
    source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        == Some("tessera")
}

fn check_contract(source: &dyn TensorSource) -> Result<(), String> {
    if !matches(source)
        || source
            .metadata("tessera.text_encoder")
            .and_then(MetaValue::to_string_val)
            != Some("openai/clip-vit-base-patch32")
    {
        return Err("expected a Tessera-small GGUF with the CLIP ViT-B/32 text tower".into());
    }
    let cfg: serde_json::Value = serde_json::from_str(
        source
            .metadata("tessera.model_config")
            .and_then(MetaValue::to_string_val)
            .ok_or("missing tessera.model_config")?,
    )
    .map_err(|e| format!("invalid tessera.model_config: {e}"))?;
    for (name, val) in [
        ("grid_size", 16.),
        ("dim", 512.),
        ("vocab_size", 512.),
        ("n_heads", 8.),
        ("depth_in", 2.),
        ("depth_core", 8.),
        ("depth_out", 2.),
        ("patch", 2.),
        ("mlp_ratio", 3.),
        ("n_registers", 4.),
        ("text_dim", 512.),
        ("text_len", 24.),
        ("rope_theta", 100.),
        ("color_freqs", 6.),
    ] {
        if cfg[name].as_f64() != Some(val) {
            return Err(format!(
                "unsupported tessera.model_config.{name} (expected {val})"
            ));
        }
    }
    for name in ["conv_ffn", "self_cond"] {
        if cfg[name].as_bool() != Some(true) {
            return Err(format!("Tessera-small requires {name}"));
        }
    }
    Ok(())
}

/// The tower and denoiser borrow the same GGUF mapping for the lifetime of a request.
pub struct Pipeline {
    source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    policy: ComputePolicy,
}

/// CLIP encodes an explicit empty prompt as BOS, EOS and padding.
pub fn validate_prompt(prompt: &str) -> Result<&str, String> {
    if prompt.len() > 32768 {
        return Err("prompt exceeds 32768 bytes".into());
    }
    Ok(prompt)
}

impl Pipeline {
    pub fn load(
        source: Arc<dyn TensorSource>,
        threads: usize,
        policy: ComputePolicy,
    ) -> Result<Self, String> {
        check_contract(source.as_ref())?;
        // Reject incomplete or incompatible bundles at server startup.
        text::TextEncoder::load(source.as_ref())?;
        denoiser::Weights::load(source.as_ref())?;
        Ok(Self {
            source,
            pool: Arc::new(ComputePool::new(threads.max(1))),
            policy,
        })
    }

    pub fn generate(&self, prompt: &str, options: &Options) -> Result<image::RgbaImage, String> {
        options.validate()?;
        validate_prompt(prompt)?;
        let weights = denoiser::Weights::load(self.source.as_ref())?;
        let encoder = text::TextEncoder::load(self.source.as_ref())?;
        let (states, pooled) = encoder.encode(prompt, self.policy, self.pool.clone())?;
        let mut net = denoiser::Denoiser::new(&weights, self.pool.clone(), self.policy)?;
        let canvas =
            palette::initial_canvas(options.pixel.canvas.as_deref(), weights.palette_rgb());
        let ids = sampler::sample(options, canvas, |ids, sc, guidance| {
            let cond = net.forward(ids, &states, &pooled, sc, false)?;
            if (guidance - 1.).abs() <= 1e-6 {
                return Ok(cond);
            }
            let mut uncond = net.forward(ids, &states, &pooled, sc, true)?;
            for (u, c) in uncond.iter_mut().zip(cond) {
                *u += guidance * (c - *u);
            }
            Ok(uncond)
        })?;
        let (cpu, gpu) = net.counts();
        eprintln!("Tessera denoiser projections: cpu={cpu} vulkan={gpu}");
        let palette = weights.palette_rgb();
        Ok(image::RgbaImage::from_fn(
            GRID as u32,
            GRID as u32,
            |x, y| {
                let id = ids[(y as usize) * GRID + x as usize] as usize;
                let [r, g, b] = palette[id];
                image::Rgba([r, g, b, if id == 0 { 0 } else { 255 }])
            },
        ))
    }
}
