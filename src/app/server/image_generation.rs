//! Image-generation HTTP transport. Model computation stays in models/diffusion.
use super::{compute_task, AppState, Backend};
use crate::{
    app::{diffusion::pjit_options, CliOptions},
    core::tensor::TensorSource,
    format::ggufrs::{open_model_source, ComponentRole},
    models::{
        diffusion::pjit::{PjitOptions, PjitPipeline},
        tessera,
    },
};
use axum::{extract::State, http::StatusCode, Json};
use base64::Engine;
use image::ImageEncoder;
use serde::Deserialize;
use std::sync::Arc;

pub(super) struct PixelBackend {
    pipeline: tessera::Pipeline,
    defaults: tessera::Options,
}
pub(super) fn build_pixel(
    options: &CliOptions,
    source: Arc<dyn TensorSource>,
) -> Result<PixelBackend, String> {
    let defaults = crate::app::pixel_art::options(options, true)?;
    let threads = crate::app::resolve_thread_count(
        options.threads,
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
    );
    Ok(PixelBackend {
        pipeline: tessera::Pipeline::load(source, threads, options.compute_policy())?,
        defaults,
    })
}

pub(super) struct PjitBackend {
    pipeline: PjitPipeline,
    defaults: PjitOptions,
}
pub(super) fn build(
    options: &CliOptions,
    source: Arc<dyn TensorSource>,
) -> Result<PjitBackend, String> {
    let defaults = server_defaults(options)?;
    let text = options
        .text_encoder
        .as_deref()
        .ok_or("P-JiT requires --text-encoder")?;
    let text = Arc::from(open_model_source(text, ComponentRole::Llm).map_err(|e| e.to_string())?);
    let available = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let threads = crate::app::resolve_thread_count(options.threads, available);
    Ok(PjitBackend {
        pipeline: PjitPipeline::load(source, text, threads)?,
        defaults,
    })
}
fn server_defaults(options: &CliOptions) -> Result<PjitOptions, String> {
    let defaults = pjit_options(options)?;
    if options.noise.is_some()
        || options.out.is_some()
        || options.prompt.is_some()
        || options.overwrite
    {
        return Err("HTTP image generation does not accept --prompt, --noise, --out or --overwrite; supply prompts per request".into());
    }
    Ok(defaults)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub(super) prompt: String,
    pub(super) model: Option<String>,
    pub(super) size: Option<String>,
    pub(super) n: Option<usize>,
    pub(super) response_format: Option<String>,
    pub(super) width: Option<usize>,
    pub(super) height: Option<usize>,
    pub(super) steps: Option<usize>,
    pub(super) seed: Option<u64>,
    pub(super) guidance: Option<f32>,
    pub(super) negative_prompt: Option<String>,
    temperature: Option<f32>,
    pixel_options: Option<tessera::PixelOptions>,
}
type ApiError = (StatusCode, Json<serde_json::Value>);
fn error(status: StatusCode, message: impl ToString) -> ApiError {
    (
        status,
        Json(serde_json::json!({"error":{"message":message.to_string()}})),
    )
}
impl Request {
    /// The square resolution the request asks for, if any.
    ///
    /// The diffusion models generate square images, so a request naming two
    /// different sides is a bad request rather than something to round.
    pub(super) fn resolution(&self) -> Result<Option<usize>, String> {
        let (width, height) = match (&self.size, self.width, self.height) {
            (Some(size), None, None) => {
                let (width, height) = size.split_once('x').ok_or("size must be WIDTHxHEIGHT")?;
                (
                    width.parse().map_err(|_| "invalid width")?,
                    height.parse().map_err(|_| "invalid height")?,
                )
            }
            (None, Some(width), Some(height)) => (width, height),
            (None, Some(width), None) => (width, width),
            (None, None, Some(height)) => (height, height),
            (Some(_), _, _) => return Err("size conflicts with width/height".into()),
            (None, None, None) => return Ok(None),
        };
        if width != height {
            return Err("this model generates square images; width and height must match".into());
        }
        if width == 0 {
            return Err("resolution must be positive".into());
        }
        Ok(Some(width))
    }

    /// The prompt, if it is one the models can encode.
    ///
    /// Checked before the generation slot is taken, so an empty prompt
    /// fails fast instead of queueing behind a running generation.
    pub(super) fn prompt_text(&self) -> Result<&str, String> {
        if self.prompt.trim().is_empty() || self.prompt.len() > 32768 {
            return Err("prompt must contain 1..=32768 bytes".into());
        }
        Ok(&self.prompt)
    }

    /// The shared request shape every image-generation backend accepts.
    ///
    /// One image per request, returned as base64: the route has a single
    /// PNG to give back.
    pub(super) fn single_image(&self) -> Result<(), String> {
        if self.n.is_some_and(|n| n != 1)
            || self
                .response_format
                .as_deref()
                .is_some_and(|f| f != "b64_json")
        {
            return Err("image generation supports n=1 and response_format=b64_json".into());
        }
        Ok(())
    }

    fn pixel_options(&self, defaults: &tessera::Options) -> Result<tessera::Options, String> {
        self.single_image()?;
        if self.resolution()?.is_some_and(|v| v != tessera::GRID) || self.negative_prompt.is_some()
        {
            return Err(
                "Tessera-small requires 16x16 and uses learned unconditional guidance".into(),
            );
        }
        let options = tessera::Options {
            steps: self.steps.unwrap_or(defaults.steps),
            seed: self.seed.unwrap_or(defaults.seed),
            guidance: self.guidance.unwrap_or(defaults.guidance),
            temperature: self.temperature.unwrap_or(defaults.temperature),
            pixel: self
                .pixel_options
                .clone()
                .unwrap_or_else(|| defaults.pixel.clone()),
        };
        options.validate()?;
        Ok(options)
    }

    fn options(&self, defaults: &PjitOptions) -> Result<PjitOptions, String> {
        self.prompt_text()?;
        self.single_image()?;
        let mut options = defaults.clone();
        if let Some(size) = &self.size {
            if self.width.is_some() || self.height.is_some() {
                return Err("size conflicts with width/height".into());
            }
            let (w, h) = size.split_once('x').ok_or("size must be WIDTHxHEIGHT")?;
            options.width = w.parse().map_err(|_| "invalid width")?;
            options.height = h.parse().map_err(|_| "invalid height")?;
        }
        if let Some(v) = self.width {
            options.width = v
        }
        if let Some(v) = self.height {
            options.height = v
        }
        if let Some(v) = self.steps {
            options.steps = v
        }
        if let Some(v) = self.seed {
            options.seed = v
        }
        if let Some(v) = self.guidance {
            options.guidance = v
        }
        if let Some(v) = &self.negative_prompt {
            options.negative_prompt = v.clone()
        }
        if options.negative_prompt.len() > 32768 {
            return Err("negative_prompt exceeds 32768 bytes".into());
        }
        options.validate()?;
        Ok(options)
    }
}
/// What the request resolved to, per backend.
enum Plan {
    Tessera(tessera::Options),
    Pjit(PjitOptions),
    Diffusion(super::diffusion::DiffusionPlan),
}

pub(super) async fn generate(
    State(state): State<AppState>,
    Json(request): Json<Request>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if request
        .model
        .as_deref()
        .is_some_and(|m| m != state.model_name)
    {
        return Err(error(StatusCode::BAD_REQUEST, "Unknown model"));
    }
    // Resolve the request before taking the slot, so a bad request never
    // queues behind a running generation.
    if (request.pixel_options.is_some() || request.temperature.is_some())
        && !matches!(state.model.as_ref(), Backend::Tessera(_))
    {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "unsupported pixel sampling options for this model",
        ));
    }
    let plan = match state.model.as_ref() {
        Backend::Tessera(backend) => Plan::Tessera(
            request
                .pixel_options(&backend.defaults)
                .map_err(|e| error(StatusCode::BAD_REQUEST, e))?,
        ),
        Backend::Pjit(backend) => Plan::Pjit(
            request
                .options(&backend.defaults)
                .map_err(|e| error(StatusCode::BAD_REQUEST, e))?,
        ),
        Backend::Diffusion(backend) => Plan::Diffusion(
            backend
                .plan(&request)
                .map_err(|e| error(StatusCode::BAD_REQUEST, e))?,
        ),
        _ => return Err(error(StatusCode::NOT_FOUND, "Image generation unavailable")),
    };
    // The prompt is checked with the rest of the request shape, before the
    // generation slot is taken.
    let prompt = match &plan {
        Plan::Tessera(_) => tessera::validate_prompt(&request.prompt),
        _ => request.prompt_text(),
    }
    .map_err(|e| error(StatusCode::BAD_REQUEST, e))?
    .to_string();
    let permit = state
        .generation_slot
        .clone()
        .try_acquire_owned()
        .map_err(|_| error(StatusCode::TOO_MANY_REQUESTS, "Image generation is busy"))?;
    let model = state.model.clone();
    let data = compute_task(state.compute, move || {
        let _permit = permit;
        if let (Backend::Tessera(backend), Plan::Tessera(options)) = (model.as_ref(), &plan) {
            let png = crate::app::pixel_art::png(backend.pipeline.generate(&prompt, options)?)?;
            return Ok(base64::engine::general_purpose::STANDARD.encode(png));
        }
        let rgb = match (model.as_ref(), &plan) {
            (Backend::Pjit(backend), Plan::Pjit(options)) => {
                backend.pipeline.generate(&prompt, options, None)?
            }
            (Backend::Diffusion(backend), Plan::Diffusion(plan)) => {
                backend.generate(plan, &prompt)?
            }
            _ => unreachable!(),
        };
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(
                &rgb.bytes,
                rgb.width,
                rgb.height,
                image::ColorType::Rgb8.into(),
            )
            .map_err(|e| e.to_string())?;
        Ok::<_, String>(base64::engine::general_purpose::STANDARD.encode(png))
    })
    .await
    .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?
    .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(
        serde_json::json!({"created":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),"data":[{"b64_json":data}]}),
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pixel_requests_validate_the_fixed_canvas_and_sampler_before_the_slot() {
        for fields in [
            serde_json::json!({"prompt":"x","size":"32x32"}),
            serde_json::json!({"prompt":"x","negative_prompt":"blur"}),
            serde_json::json!({"prompt":"x","steps":0}),
            serde_json::json!({"prompt":"x","pixel_options":{"canvas":[512]}}),
            serde_json::json!({"prompt":"x","pixel_options":{"canvas":vec![512;256]}}),
        ] {
            let request: Request = serde_json::from_value(fields).unwrap();
            assert!(request.pixel_options(&tessera::Options::default()).is_err());
        }
    }
    #[test]
    fn rejects_cli_only_server_options_before_loading() {
        for extra in [
            vec!["--noise", "noise.f32"],
            vec!["--out", "output.png"],
            vec!["--prompt", "cat"],
            vec!["--overwrite"],
        ] {
            let args: Vec<_> = [vec!["server", "--model", "pjit.gguf"], extra]
                .concat()
                .into_iter()
                .map(String::from)
                .collect();
            let options = crate::app::parse_cli_options(&args).unwrap();
            assert!(server_defaults(&options)
                .unwrap_err()
                .contains("HTTP image generation"));
        }
        let args = ["server", "--model", "pjit.gguf"].map(String::from);
        assert!(server_defaults(&crate::app::parse_cli_options(&args).unwrap()).is_ok());
    }
    #[test]
    fn rejects_invalid_image_requests_before_loading() {
        let request: Request =
            serde_json::from_value(serde_json::json!({"prompt":"fox","size":"33x35","steps":2}))
                .unwrap();
        let options = request.options(&PjitOptions::default()).unwrap();
        assert_eq!((options.width, options.height), (33, 35));
        for fields in [
            serde_json::json!({"prompt":" "}),
            serde_json::json!({"prompt":"x","n":2}),
            serde_json::json!({"prompt":"x","size":"99999x1"}),
            serde_json::json!({"prompt":"x","size":"32x32","width":32}),
        ] {
            let request: Request = serde_json::from_value(fields).unwrap();
            assert!(request.options(&PjitOptions::default()).is_err());
        }
        assert!(serde_json::from_value::<Request>(
            serde_json::json!({"prompt":"x","unexpected":true})
        )
        .is_err());
    }
}
