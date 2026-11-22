//! Image-generation HTTP transport for the diffusion models.
//!
//! The models layer keeps the computation: Z-Image and ERNIE-Image both
//! implement [`DiffusionPipeline`](crate::models::diffusion::DiffusionPipeline),
//! which loads the three components and returns the RGB image. This module
//! only maps a request onto the pipeline's options and hands the image
//! back, so the HTTP path and the CLI path run the same code.
//!
//! Only the models whose pipeline is a plain text-to-image one are here.
//! LongCat-Image-Edit edits a given image and Qwen-Image-2.1 drives its
//! own latent/context orchestration, so neither has a text-to-image request
//! to answer yet; their entries in `docs/MODEL_LIST.md` say so.

use super::image_generation::Request;
use crate::{
    app::CliOptions,
    compute::ComputePolicy,
    core::tensor::TensorSource,
    format::ggufrs::{open_model_source, ComponentRole},
    models::diffusion::{
        ernie_image::{ErnieImageOptions, ErnieImagePipeline},
        z_image::{ZImageOptions, ZImagePipeline},
        DiffusionPipeline, DiffusionRgb,
    },
};
use std::sync::Arc;

/// Which of the served diffusion models a DiT GGUF is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DiffusionKind {
    ZImage,
    Ernie,
}

/// The diffusion models the image-generation route serves.
pub(super) enum DiffusionBackend {
    ZImage {
        pipeline: ZImagePipeline,
        defaults: ZImageOptions,
    },
    Ernie {
        pipeline: ErnieImagePipeline,
        defaults: ErnieImageOptions,
        /// Whether the DiT's main stack is stored in the K-quant family.
        k_quant: bool,
    },
}

/// The options one request resolved to.
#[derive(Debug, Clone, Copy)]
pub(super) enum DiffusionPlan {
    ZImage(ZImageOptions),
    Ernie(ErnieImageOptions),
}

/// The request defaults of the loaded model, without the pipeline.
///
/// Split out so the request mapping can be tested without loading three
/// multi-gigabyte files.
#[derive(Debug, Clone, Copy)]
enum Defaults {
    ZImage(ZImageOptions),
    Ernie(ErnieImageOptions),
}

/// Map a request onto the pipeline's options.
///
/// The request shape is checked here, before the generation slot is taken,
/// so a bad request fails fast instead of queueing behind a running
/// generation.
fn plan(defaults: &Defaults, request: &Request) -> Result<DiffusionPlan, String> {
    request.single_image()?;
    match defaults {
        Defaults::ZImage(defaults) => {
            if request.guidance.is_some() {
                return Err("Z-Image has no CFG branch; guidance is not supported".into());
            }
            let mut options = *defaults;
            if let Some(resolution) = request.resolution()? {
                options.resolution = resolution;
            }
            if let Some(steps) = request.steps {
                options.steps = steps;
            }
            if let Some(seed) = request.seed {
                options.seed = seed as i64;
            }
            if options.steps == 0 || options.resolution == 0 || options.resolution % 16 != 0 {
                return Err("Z-Image needs positive steps and a resolution divisible by 16".into());
            }
            Ok(DiffusionPlan::ZImage(options))
        }
        Defaults::Ernie(defaults) => {
            let mut options = *defaults;
            if let Some(resolution) = request.resolution()? {
                options.resolution = resolution;
            }
            if let Some(steps) = request.steps {
                options.steps = steps;
            }
            if let Some(seed) = request.seed {
                options.seed = seed as i64;
            }
            if let Some(guidance) = request.guidance {
                if !guidance.is_finite() || guidance <= 0.0 {
                    return Err("guidance must be a finite positive number".into());
                }
                options.cfg_scale = guidance;
            }
            if options.steps == 0 || options.resolution == 0 || options.resolution % 16 != 0 {
                return Err(
                    "ERNIE-Image needs positive steps and a resolution divisible by 16".into(),
                );
            }
            Ok(DiffusionPlan::Ernie(options))
        }
    }
}

/// Validate the command line for the image-generation server.
///
/// The models need their three components on the command line, but the
/// prompt arrives with the request, so the CLI-only prompt and output
/// flags are rejected instead of being silently ignored.
pub(super) fn validate_server_options(options: &CliOptions) -> Result<(), String> {
    if options.prompt.is_some()
        || options.out.is_some()
        || options.overwrite
        || options.noise.is_some()
    {
        return Err(
            "HTTP image generation does not accept --prompt, --noise, --out or --overwrite; \
             supply prompts per request"
                .into(),
        );
    }
    if options
        .text_encoder
        .as_deref()
        .is_none_or(|path| path.as_os_str().is_empty())
    {
        return Err("image generation over HTTP requires --text-encoder".into());
    }
    if options
        .vae
        .as_deref()
        .is_none_or(|path| path.as_os_str().is_empty())
    {
        return Err("image generation over HTTP requires --vae".into());
    }
    if options.steps == Some(0) {
        return Err("--steps must be at least 1".into());
    }
    if let Some(resolution) = options.resolution {
        if resolution == 0 || resolution % 16 != 0 {
            return Err("--resolution must be positive and divisible by 16".into());
        }
    }
    Ok(())
}

impl DiffusionKind {
    /// Which model this DiT GGUF holds, if it is one of the served ones.
    ///
    /// The probe lives in the model layer because it is a fact about the
    /// weights, not about the entry point: the CLI and the server must
    /// agree on it, and the published ERNIE export mis-tags its
    /// `general.architecture` as `wan`.
    pub(super) fn detect(source: &dyn TensorSource) -> Option<Self> {
        if crate::models::diffusion::z_image::matches_signature(source) {
            return Some(Self::ZImage);
        }
        if crate::models::diffusion::ernie_image::matches_signature(source) {
            return Some(Self::Ernie);
        }
        None
    }
}

/// The compute policy this model runs under.
///
/// A K-quant ERNIE DiT measures 2.2x slower on the device than on the CPU
/// (TODO-073), so the CLI pins such a model to the CPU and the HTTP path
/// has to make the same choice: the same weights must not answer
/// differently depending on the entry point.
pub(super) fn policy_for(
    kind: DiffusionKind,
    source: &dyn TensorSource,
    requested: ComputePolicy,
) -> ComputePolicy {
    match kind {
        DiffusionKind::Ernie if crate::models::diffusion::ernie_image::is_k_quant(source) => {
            ComputePolicy::Cpu
        }
        _ => requested,
    }
}

impl DiffusionBackend {
    /// Load the pipeline and take the request defaults from the command line.
    ///
    /// The text encoder and the VAE are separate files for both models, so
    /// the server needs `--text-encoder` and `--vae` next to `--model`.
    pub(super) fn build(
        options: &CliOptions,
        kind: DiffusionKind,
        diffusion: Arc<dyn TensorSource>,
        threads: usize,
    ) -> Result<Self, String> {
        let text_path = options
            .text_encoder
            .as_deref()
            .filter(|path| !path.as_os_str().is_empty())
            .ok_or("image generation over HTTP requires --text-encoder")?;
        let vae_path = options
            .vae
            .as_deref()
            .filter(|path| !path.as_os_str().is_empty())
            .ok_or("image generation over HTTP requires --vae")?;
        let text =
            Arc::from(open_model_source(text_path, ComponentRole::Llm).map_err(|e| e.to_string())?);
        let vae =
            Arc::from(open_model_source(vae_path, ComponentRole::Llm).map_err(|e| e.to_string())?);
        match kind {
            DiffusionKind::ZImage => {
                let defaults = ZImageOptions {
                    steps: options.steps.unwrap_or(8),
                    resolution: options.resolution.unwrap_or(512),
                    seed: options.seed.unwrap_or(0),
                };
                let pipeline = ZImagePipeline::load(diffusion, text, vae, threads)?;
                Ok(Self::ZImage { pipeline, defaults })
            }
            DiffusionKind::Ernie => {
                // The published ERNIE export is the Turbo one, and the CLI
                // picks the step count and the CFG scale from the same
                // "turbo" in the file name.
                let turbo = options
                    .model
                    .to_string_lossy()
                    .to_lowercase()
                    .contains("turbo");
                let defaults = ErnieImageOptions {
                    steps: options.steps.unwrap_or(if turbo { 8 } else { 32 }),
                    resolution: options.resolution.unwrap_or(512),
                    seed: options.seed.unwrap_or(0),
                    cfg_scale: options.cfg_scale.unwrap_or(if turbo { 1.0 } else { 5.0 }),
                };
                let k_quant = crate::models::diffusion::ernie_image::is_k_quant(diffusion.as_ref());
                let pipeline = ErnieImagePipeline::load(diffusion, text, vae, threads)?;
                Ok(Self::Ernie {
                    pipeline,
                    defaults,
                    k_quant,
                })
            }
        }
    }

    /// The name this backend reports in the server's mode label.
    pub(super) fn label(&self) -> &'static str {
        match self {
            Self::ZImage { .. } => "z-image",
            Self::Ernie { .. } => "ernie-image",
        }
    }

    /// The compute policy this backend actually runs under.
    ///
    /// A K-quant ERNIE DiT measures 2.2x slower on the device than on the
    /// CPU (TODO-073), so the CLI pins such a model to the CPU; the HTTP
    /// path has to make the same choice, otherwise the same weights would
    /// answer differently depending on the entry point.
    pub(super) fn compute_policy(&self, requested: ComputePolicy) -> ComputePolicy {
        match self {
            Self::Ernie { k_quant: true, .. } => ComputePolicy::Cpu,
            _ => requested,
        }
    }

    /// The defaults this backend serves requests with.
    fn defaults(&self) -> Defaults {
        match self {
            Self::ZImage { defaults, .. } => Defaults::ZImage(*defaults),
            Self::Ernie { defaults, .. } => Defaults::Ernie(*defaults),
        }
    }

    /// Resolve a request into the pipeline's options.
    ///
    /// Done before the generation slot is taken so a bad request does not
    /// queue behind a running one.
    pub(super) fn plan(&self, request: &Request) -> Result<DiffusionPlan, String> {
        plan(&self.defaults(), request)
    }

    /// Generate the image the plan describes.
    pub(super) fn generate(
        &self,
        plan: &DiffusionPlan,
        prompt: &str,
    ) -> Result<DiffusionRgb, String> {
        match (self, plan) {
            (Self::ZImage { pipeline, .. }, DiffusionPlan::ZImage(options)) => {
                pipeline.generate_rgb(prompt, options)
            }
            (Self::Ernie { pipeline, .. }, DiffusionPlan::Ernie(options)) => {
                pipeline.generate_rgb(prompt, options)
            }
            _ => Err("the plan does not match the loaded model".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{plan, Defaults};
    use crate::models::diffusion::{ernie_image::ErnieImageOptions, z_image::ZImageOptions};

    use super::super::image_generation::Request;

    fn request(json: &str) -> Request {
        serde_json::from_str(json).expect("the request must deserialize")
    }

    fn z_image_defaults() -> Defaults {
        Defaults::ZImage(ZImageOptions {
            steps: 8,
            resolution: 512,
            seed: 0,
        })
    }

    fn ernie_defaults() -> Defaults {
        Defaults::Ernie(ErnieImageOptions {
            steps: 32,
            resolution: 512,
            seed: 0,
            cfg_scale: 5.0,
        })
    }

    #[test]
    fn a_request_without_knobs_keeps_the_server_defaults() {
        let plan = plan(&z_image_defaults(), &request(r#"{"prompt":"a red panda"}"#))
            .expect("the default request is valid");
        let super::DiffusionPlan::ZImage(options) = plan else {
            panic!("Z-Image must stay on the Z-Image pipeline");
        };
        assert_eq!(options.steps, 8);
        assert_eq!(options.resolution, 512);
        assert_eq!(options.seed, 0);
    }

    #[test]
    fn the_request_knobs_replace_the_defaults() {
        let plan = plan(
            &z_image_defaults(),
            &request(r#"{"prompt":"a red panda","size":"256x256","steps":3,"seed":7}"#),
        )
        .expect("the request is valid");
        let super::DiffusionPlan::ZImage(options) = plan else {
            panic!("Z-Image must stay on the Z-Image pipeline");
        };
        assert_eq!(options.steps, 3);
        assert_eq!(options.resolution, 256);
        assert_eq!(options.seed, 7);
    }

    #[test]
    fn a_non_square_request_is_refused() {
        let error = plan(
            &z_image_defaults(),
            &request(r#"{"prompt":"a red panda","size":"512x256"}"#),
        )
        .expect_err("a non-square image is not supported");
        assert!(error.contains("square"), "{error}");
    }

    #[test]
    fn a_resolution_that_the_latent_shape_cannot_hold_is_refused() {
        let error = plan(
            &z_image_defaults(),
            &request(r#"{"prompt":"a red panda","size":"127x127"}"#),
        )
        .expect_err("the latent shape needs a multiple of 16");
        assert!(error.contains("divisible by 16"), "{error}");
    }

    #[test]
    fn z_image_has_no_cfg_branch() {
        let error = plan(
            &z_image_defaults(),
            &request(r#"{"prompt":"a red panda","guidance":3.5}"#),
        )
        .expect_err("Z-Image does not take a guidance scale");
        assert!(error.contains("guidance"), "{error}");
    }

    #[test]
    fn ernie_takes_the_guidance_scale_as_its_cfg_scale() {
        let plan = plan(
            &ernie_defaults(),
            &request(r#"{"prompt":"a red panda","guidance":2.5}"#),
        )
        .expect("ERNIE-Image has a CFG branch");
        let super::DiffusionPlan::Ernie(options) = plan else {
            panic!("ERNIE-Image must stay on the ERNIE-Image pipeline");
        };
        assert_eq!(options.cfg_scale, 2.5);
    }

    #[test]
    fn a_non_positive_guidance_scale_is_refused() {
        let error = plan(
            &ernie_defaults(),
            &request(r#"{"prompt":"a red panda","guidance":0}"#),
        )
        .expect_err("a CFG scale of zero would disable the branch");
        assert!(error.contains("positive"), "{error}");
    }

    #[test]
    fn more_than_one_image_per_request_is_refused() {
        let error = plan(
            &z_image_defaults(),
            &request(r#"{"prompt":"a red panda","n":2}"#),
        )
        .expect_err("the route returns one image");
        assert!(error.contains("n=1"), "{error}");
    }

    #[test]
    fn a_url_response_format_is_refused() {
        let error = plan(
            &z_image_defaults(),
            &request(r#"{"prompt":"a red panda","response_format":"url"}"#),
        )
        .expect_err("the route returns base64");
        assert!(error.contains("b64_json"), "{error}");
    }
}
