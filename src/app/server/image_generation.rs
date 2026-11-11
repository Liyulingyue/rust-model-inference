//! Image-generation HTTP transport. Model computation stays in models/diffusion.
use super::{compute_task, AppState, Backend};
use crate::{
    app::{diffusion::pjit_options, CliOptions},
    core::tensor::TensorSource,
    format::ggufrs::{open_model_source, ComponentRole},
    models::diffusion::pjit::{PjitOptions, PjitPipeline},
};
use axum::{extract::State, http::StatusCode, Json};
use base64::Engine;
use image::ImageEncoder;
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
    let defaults = pjit_options(options)?;
    if options.noise.is_some() {
        return Err("HTTP image generation does not accept a server-local --noise file".into());
    }
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
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    prompt: String,
    model: Option<String>,
    size: Option<String>,
    n: Option<usize>,
    response_format: Option<String>,
    width: Option<usize>,
    height: Option<usize>,
    steps: Option<usize>,
    seed: Option<u64>,
    guidance: Option<f32>,
    negative_prompt: Option<String>,
}
type ApiError = (StatusCode, Json<serde_json::Value>);
fn error(status: StatusCode, message: impl ToString) -> ApiError {
    (
        status,
        Json(serde_json::json!({"error":{"message":message.to_string()}})),
    )
}
impl Request {
    fn options(&self, defaults: &PjitOptions) -> Result<PjitOptions, String> {
        if self.prompt.trim().is_empty() || self.prompt.len() > 32768 {
            return Err("prompt must contain 1..=32768 bytes".into());
        }
        if self.n.is_some_and(|n| n != 1)
            || self
                .response_format
                .as_deref()
                .is_some_and(|f| f != "b64_json")
        {
            return Err("P-JiT supports n=1 and response_format=b64_json".into());
        }
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
pub(super) async fn generate(
    State(state): State<AppState>,
    Json(request): Json<Request>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let Backend::Pjit(backend) = state.model.as_ref() else {
        return Err(error(StatusCode::NOT_FOUND, "Image generation unavailable"));
    };
    if request
        .model
        .as_deref()
        .is_some_and(|m| m != state.model_name)
    {
        return Err(error(StatusCode::BAD_REQUEST, "Unknown model"));
    }
    let options = request
        .options(&backend.defaults)
        .map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    let permit = state
        .generation_slot
        .clone()
        .try_acquire_owned()
        .map_err(|_| error(StatusCode::TOO_MANY_REQUESTS, "Image generation is busy"))?;
    let model = state.model.clone();
    let data = compute_task(state.compute, move || {
        let _permit = permit;
        let Backend::Pjit(backend) = model.as_ref() else {
            unreachable!()
        };
        let rgb = backend.pipeline.generate(&request.prompt, &options, None)?;
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
