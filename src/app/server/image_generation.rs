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
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use base64::Engine;
use serde::Deserialize;
use std::sync::Arc;

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
#[serde(untagged)]
enum Prompts {
    One(String),
    Many(Vec<String>),
}
impl Prompts {
    fn as_slice(&self) -> &[String] {
        match self {
            Self::One(s) => std::slice::from_ref(s),
            Self::Many(v) => v,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    prompt: Prompts,
    stream: Option<bool>,
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
    pub(super) temperature: Option<f32>,
    pub(super) pixel_options: Option<tessera::PixelOptions>,
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

    /// Validate inputs to the RGB pipelines. Tessera accepts empty text and
    /// defines its own pixel sampler options.
    pub(super) fn validate_rgb_inputs(&self) -> Result<(), String> {
        if self.pixel_options.is_some() || self.temperature.is_some() {
            return Err("unsupported pixel sampling options for this model".into());
        }
        if self.stream.is_some() {
            return Err("this image model does not support stream".into());
        }
        let Prompts::One(prompt) = &self.prompt else {
            return Err("this image model requires a single prompt".into());
        };
        if prompt.trim().is_empty() || prompt.len() > 32768 {
            return Err("prompt must contain 1..=32768 bytes".into());
        }
        Ok(())
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

    pub(super) fn validate_pixel_inputs(&self) -> Result<(), String> {
        if self
            .response_format
            .as_deref()
            .is_some_and(|f| f != "b64_json")
        {
            return Err("Tessera supports response_format=b64_json".into());
        }
        let prompts = self.prompt.as_slice();
        tessera::validate_batch_size(prompts.len().saturating_mul(self.n.unwrap_or(1)))?;
        for prompt in prompts {
            tessera::validate_prompt(prompt)?;
        }
        Ok(())
    }

    fn options(&self, defaults: &PjitOptions) -> Result<PjitOptions, String> {
        self.validate_rgb_inputs()?;
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
    Pjit(PjitOptions),
    Diffusion(super::diffusion::DiffusionPlan),
}

pub(super) async fn generate(
    State(state): State<AppState>,
    Json(request): Json<Request>,
) -> Result<Response, ApiError> {
    if request
        .model
        .as_deref()
        .is_some_and(|m| m != state.model_name)
    {
        return Err(error(StatusCode::BAD_REQUEST, "Unknown model"));
    }
    // Resolve the request before taking the slot, so a bad request never
    // queues behind a running generation.
    let plan = match state.model.as_ref() {
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
    let stream = request.stream.unwrap_or(false);
    let permit = state
        .generation_slot
        .clone()
        .try_acquire_owned()
        .map_err(|_| error(StatusCode::TOO_MANY_REQUESTS, "Image generation is busy"))?;
    let model = state.model.clone();
    let generate = move |preview: Option<&mut tessera::Preview<'_>>| {
        let _permit = permit;
        let prompts: Vec<_> = request
            .prompt
            .as_slice()
            .iter()
            .flat_map(|s| std::iter::repeat_n(s.as_str(), request.n.unwrap_or(1)))
            .collect();
        let images = match (model.as_ref(), &plan) {
            (Backend::Pjit(backend), Plan::Pjit(options)) => {
                vec![crate::app::diffusion::into_rgb_image(
                    backend.pipeline.generate(prompts[0], options, None)?,
                )?]
            }
            (Backend::Diffusion(backend), Plan::Diffusion(plan)) => {
                backend.generate(plan, &prompts, preview)?
            }
            _ => unreachable!(),
        };
        let data = images
            .into_iter()
            .map(|image| Ok(serde_json::json!({"b64_json":encode_png(&image)?})))
            .collect::<Result<Vec<_>, String>>()?;
        Ok::<_, String>(
            serde_json::json!({"created":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),"data":data}),
        )
    };
    if stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(2);
        let previews = tx.clone();
        let job = compute_task(state.compute, move || {
            generate(Some(&mut |step, index, image| {
                let payload = serde_json::json!({"step":step.index, "total_steps":step.total, "index":index,
                    "phase":if step.polish {"polish"} else {"decode"}, "b64_json":encode_png(&image.into())?});
                previews
                    .blocking_send(Ok(Event::default()
                        .event("image.preview")
                        .data(payload.to_string())))
                    .map_err(|_| "client disconnected".to_owned())
            }))
        });
        tokio::spawn(async move {
            let (event, payload) = match job.await.map_err(|e| e.to_string()).and_then(|r| r) {
                Ok(data) => ("image.completed", data),
                Err(e) => ("error", serde_json::json!({"error":{"message":e}})),
            };
            let _ = tx
                .send(Ok(Event::default().event(event).data(payload.to_string())))
                .await;
        });
        return Ok(Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx))
            .keep_alive(KeepAlive::default())
            .into_response());
    }
    let data = compute_task(state.compute, move || generate(None))
        .await
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(data).into_response())
}

fn encode_png(image: &image::DynamicImage) -> Result<String, String> {
    Ok(base64::engine::general_purpose::STANDARD
        .encode(crate::app::diffusion::encode_image_png(image)?))
}
#[cfg(test)]
mod tests {
    use super::*;
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
