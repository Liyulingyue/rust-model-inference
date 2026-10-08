//! Diffusion model abstractions shared across image-gen pipelines.
//!
//! Each pipeline family (Z-Image, ERNIE-Image, Mage-Flow, Qwen-Image-2.1,
//! LongCat) wraps a DiT + text encoder + VAE. They differ in:
//! - DiT architecture (must stay per-model)
//! - text encoder (must stay per-model)
//! - sigma schedule / latent packing (per-model where it matters)
//!
//! What they share is the *shape* of the public surface: a
//! [`DiffusionPipeline`] trait that takes three GGUF sources + a prompt +
//! per-model options and produces an RGB image buffer. A single
//! `crate::app::diffusion::run_diffusion_cli` helper then handles the
//! boilerplate (component loading, stage profiling, PNG writing) for every
//! implementor.
//!
//! LongCat intentionally does NOT implement the trait yet -- it is an edit
//! model (needs reference image + latent encode), not a pure text-to-image
//! pipeline. Mage-Flow and Qwen-Image-2.1 still expose their own bespoke
//! `run_*_cli` functions because their CLI surfaces carry extra knobs
//! (vision references, DiT-only debug mode). The trait is the default path
//! for new pipelines.

pub mod auk;
pub mod dreamx;
pub mod ernie_image;
pub mod longcat;
pub mod mage_flow;
pub mod pig;
pub mod qwen_image_2_1;
pub(crate) mod z_image;

use std::sync::Arc;

use crate::core::tensor::TensorSource;

/// Decoded RGB image bytes (channel-first, R then G then B per pixel).
///
/// `width * height * 3 == bytes.len()` is enforced by
/// `crate::app::diffusion::write_png_atomically`. New pipelines should
/// return this type directly; it is a type alias of the legacy
/// `ZImageRgb` for source compatibility with existing call sites.
pub type DiffusionRgb = crate::models::diffusion::z_image::ZImageRgb;

/// A text-to-image diffusion pipeline: DiT + text encoder + VAE, callable
/// from the CLI with a uniform `load` / `generate_rgb` surface.
///
/// Implementors are responsible for the model-specific pieces (DiT
/// architecture, text encoder, scheduler, latent packing). The trait
/// abstracts the parts every CLI dispatcher cares about: opening three
/// GGUF sources, running the pipeline, returning RGB bytes.
pub trait DiffusionPipeline {
    /// Per-pipeline knobs (steps, resolution, CFG, ...). `Clone` so the
    /// dispatcher can keep a copy for logging / error messages.
    type Options: Clone;

    /// Load the three model components from their GGUF sources.
    ///
    /// `n_threads` is the upper bound for the pipeline's worker pool. The
    /// pipeline is free to use fewer threads internally.
    fn load(
        diffusion: Arc<dyn TensorSource>,
        text: Arc<dyn TensorSource>,
        vae: Arc<dyn TensorSource>,
        n_threads: usize,
    ) -> Result<Self, String>
    where
        Self: Sized;

    /// Encode `prompt`, denoise, decode, and return the RGB image.
    fn generate_rgb(
        &self,
        prompt: &str,
        options: &Self::Options,
    ) -> Result<DiffusionRgb, String>;
}
