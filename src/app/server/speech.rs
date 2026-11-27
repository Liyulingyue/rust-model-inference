//! Speech synthesis for self-contained models.
use super::{compute_task, AppState, Backend, ErrorResponse};
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;

/// The PiVo backend: the model plus the CLI defaults the HTTP requests
/// inherit.
pub(super) struct PivoBackend {
    pub(super) model: Arc<crate::models::pivo::Model>,
    pub(super) defaults: crate::models::pivo::SpeechRequest,
}

pub(super) struct UnderdogWoofBackend {
    pub(super) model: Arc<crate::models::underdog_woof::Model>,
    pub(super) defaults: crate::models::underdog_woof::SpeechRequest,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    model: Option<String>,
    input: String,
    voice: Option<String>,
    speed: Option<f32>,
    response_format: Option<String>,
    overrides: Option<HashMap<String, Vec<String>>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct UnderdogWoofRequest {
    model: Option<String>,
    input: String,
    voice: Option<String>,
    seed: Option<u32>,
    temperature: Option<f32>,
    normalize: Option<bool>,
    response_format: Option<String>,
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: message.into(),
        }),
    )
        .into_response()
}

pub(super) async fn synthesize(
    State(state): State<AppState>,
    Json(req): Json<Request>,
) -> Response {
    let Backend::Pivo(backend) = state.model.as_ref() else {
        return error(
            StatusCode::BAD_REQUEST,
            "Server is not running a self-contained TTS model",
        );
    };
    let model = backend.model.clone();
    let request = crate::models::pivo::SpeechRequest {
        input: req.input,
        voice: req.voice.unwrap_or_else(|| backend.defaults.voice.clone()),
        speed: req.speed.unwrap_or(backend.defaults.speed),
        overrides: req
            .overrides
            .unwrap_or_else(|| backend.defaults.overrides.clone()),
    };
    synthesize_response(
        state,
        req.model,
        req.response_format,
        crate::models::pivo::SAMPLE_RATE,
        move || {
            model.synthesize(&request).map_err(|e| {
                let status = match e {
                    crate::models::pivo::SynthesisError::InvalidInput(_) => StatusCode::BAD_REQUEST,
                    crate::models::pivo::SynthesisError::Numerical(_) => {
                        StatusCode::INTERNAL_SERVER_ERROR
                    }
                };
                (status, e.to_string())
            })
        },
    )
    .await
}

pub(super) async fn synthesize_underdog_woof(
    State(state): State<AppState>,
    Json(req): Json<UnderdogWoofRequest>,
) -> Response {
    let Backend::UnderdogWoof(backend) = state.model.as_ref() else {
        return error(
            StatusCode::BAD_REQUEST,
            "Server is not running an Underdog Woof TTS model",
        );
    };
    let model = backend.model.clone();
    let request = crate::models::underdog_woof::SpeechRequest {
        input: req.input,
        voice: req.voice.unwrap_or_else(|| backend.defaults.voice.clone()),
        seed: req.seed.unwrap_or(backend.defaults.seed),
        temperature: req.temperature.unwrap_or(backend.defaults.temperature),
        normalize: req.normalize.unwrap_or(backend.defaults.normalize),
    };
    synthesize_response(
        state,
        req.model,
        req.response_format,
        crate::models::underdog_woof::SAMPLE_RATE,
        move || {
            model.synthesize(&request).map_err(|e| {
                let status = match e {
                    crate::models::underdog_woof::SynthesisError::InvalidInput(_) => {
                        StatusCode::BAD_REQUEST
                    }
                    crate::models::underdog_woof::SynthesisError::Numerical(_) => {
                        StatusCode::INTERNAL_SERVER_ERROR
                    }
                };
                (status, e.to_string())
            })
        },
    )
    .await
}

async fn synthesize_response<F>(
    state: AppState,
    requested_model: Option<String>,
    response_format: Option<String>,
    sample_rate: u32,
    synthesize: F,
) -> Response
where
    F: FnOnce() -> Result<Vec<f32>, (StatusCode, String)> + Send + 'static,
{
    if requested_model
        .as_deref()
        .is_some_and(|m| m != state.model_name)
    {
        return error(
            StatusCode::BAD_REQUEST,
            "Requested speech model does not match the loaded model",
        );
    }
    let format = response_format.as_deref().unwrap_or("wav");
    if !matches!(format, "wav" | "pcm") {
        return error(
            StatusCode::BAD_REQUEST,
            "response_format must be wav or pcm",
        );
    }
    let pcm = format == "pcm";
    let Ok(permit) = state.generation_slot.clone().try_acquire_owned() else {
        return error(StatusCode::TOO_MANY_REQUESTS, "Speech model is busy");
    };
    match compute_task(state.compute, move || {
        let _permit = permit;
        let samples = synthesize()?;
        crate::format::wav::encode_wav_pcm16_channels(&samples, sample_rate, 1)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))
    })
    .await
    {
        Ok(Ok(wav)) => {
            let content_type = if pcm { "audio/pcm" } else { "audio/wav" };
            let body = if pcm { wav[44..].to_vec() } else { wav };
            (StatusCode::OK, [(header::CONTENT_TYPE, content_type)], body).into_response()
        }
        Ok(Err((status, e))) => error(status, e),
        Err(e) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Speech worker failed: {e}"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn woof_http_request_accepts_native_controls_and_rejects_unrelated_fields() {
        let request: UnderdogWoofRequest = serde_json::from_value(serde_json::json!({
            "input": "Hi.", "voice": "mira", "seed": 42, "temperature": 0.25,
            "normalize": false, "model": "woof-f32", "response_format": "pcm"
        }))
        .unwrap();
        assert_eq!(request.seed, Some(42));
        assert_eq!(request.temperature, Some(0.25));
        assert_eq!(request.normalize, Some(false));
        for field in ["speed", "overrides", "reference_audio", "stream", "top_p"] {
            let mut value = serde_json::json!({"input": "Hi."});
            value[field] = serde_json::Value::Null;
            assert!(
                serde_json::from_value::<UnderdogWoofRequest>(value).is_err(),
                "{field}"
            );
        }
        for seed in [-1_i64, 4294967296] {
            assert!(
                serde_json::from_value::<UnderdogWoofRequest>(serde_json::json!({
                    "input": "Hi.", "seed": seed
                }))
                .is_err()
            );
        }
        assert!(serde_json::from_value::<Request>(serde_json::json!({
            "input": "Hi.", "seed": 42
        }))
        .is_err());
    }
}
