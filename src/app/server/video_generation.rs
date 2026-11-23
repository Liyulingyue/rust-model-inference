//! Capability-named video generation transport.
use super::{compute_task, AppState, Backend};
use crate::{
    app::CliOptions,
    core::tensor::TensorSource,
    models::diffusion::fastframes_l::{Engine, Options},
};
use axum::{
    extract::State,
    http::{header, StatusCode},
    Json,
};
use serde::Deserialize;
use std::sync::Arc;

pub(super) struct VideoBackend {
    engine: Engine,
    defaults: Options,
}
pub(super) fn build(
    cli: &CliOptions,
    source: Arc<dyn TensorSource>,
) -> Result<VideoBackend, String> {
    let defaults = crate::app::video_generation::options(cli, true)?;
    // The only route this backend mounts writes an MP4 through ffmpeg.
    // Check it before the weight load so a machine without it fails at
    // startup with the real reason instead of on every request after the
    // full rollout (the same rule the CLI video path applies).
    crate::models::diffusion::dreamx::media::require_ffmpeg()?;
    let available = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    Ok(VideoBackend {
        engine: Engine::load(
            source,
            crate::app::resolve_thread_count(cli.threads, available),
        )?,
        defaults,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    prompt: String,
    model: Option<String>,
    width: Option<usize>,
    height: Option<usize>,
    frames: Option<usize>,
    fps: Option<usize>,
    seed: Option<u64>,
}
impl Request {
    fn options(&self, defaults: &Options) -> Result<Options, String> {
        if self.prompt.trim().is_empty() || self.prompt.len() > 65536 {
            return Err("prompt must contain 1..=65536 bytes".into());
        }
        let options = Options {
            width: self.width.unwrap_or(defaults.width),
            height: self.height.unwrap_or(defaults.height),
            frames: self.frames.unwrap_or(defaults.frames),
            fps: self.fps.unwrap_or(defaults.fps),
            seed: self.seed.unwrap_or(defaults.seed),
        };
        options.validate()?;
        Ok(options)
    }
}
type ApiError = (StatusCode, Json<serde_json::Value>);
fn error(status: StatusCode, message: impl ToString) -> ApiError {
    (
        status,
        Json(serde_json::json!({"error":{"message":message.to_string()}})),
    )
}

pub(super) async fn generate(
    State(state): State<AppState>,
    Json(request): Json<Request>,
) -> Result<([(header::HeaderName, &'static str); 2], Vec<u8>), ApiError> {
    let Backend::Video(backend) = state.model.as_ref() else {
        return Err(error(StatusCode::NOT_FOUND, "Video generation unavailable"));
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
        .map_err(|_| error(StatusCode::TOO_MANY_REQUESTS, "Video generation is busy"))?;
    let model = state.model.clone();
    let policy = state.compute;
    let bytes = compute_task(policy, move || {
        let _permit = permit;
        let Backend::Video(backend) = model.as_ref() else {
            unreachable!()
        };
        let video = backend
            .engine
            .generate(&request.prompt, &options, policy, None)?;
        crate::app::video_generation::encode_mp4(&video)
    })
    .await
    .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?
    .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok((
        [
            (header::CONTENT_TYPE, "video/mp4"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        bytes,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_shapes_and_rejects_unavailable_controls() {
        for body in [
            r#"{"prompt":"","frames":5}"#,
            r#"{"prompt":"a","width":17}"#,
            r#"{"prompt":"a","frames":82}"#,
        ] {
            let request: Request = serde_json::from_str(body).unwrap();
            assert!(request.options(&Options::default()).is_err());
        }
        assert!(serde_json::from_str::<Request>(r#"{"prompt":"a","steps":1}"#).is_err());
        let request: Request =
            serde_json::from_str(r#"{"prompt":"a","width":16,"height":16,"frames":5}"#).unwrap();
        assert_eq!(request.options(&Options::default()).unwrap().frames, 5);
    }
}
