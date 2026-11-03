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
    if req.model.as_deref().is_some_and(|m| m != state.model_name) {
        return error(
            StatusCode::BAD_REQUEST,
            "Requested speech model does not match the loaded model",
        );
    }
    let format = req.response_format.as_deref().unwrap_or("wav");
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
    let model = backend.model.clone();
    let request = crate::models::pivo::SpeechRequest {
        input: req.input,
        voice: req.voice.unwrap_or_else(|| backend.defaults.voice.clone()),
        speed: req.speed.unwrap_or(backend.defaults.speed),
        overrides: req
            .overrides
            .unwrap_or_else(|| backend.defaults.overrides.clone()),
    };
    match compute_task(state.compute, move || {
        let _permit = permit;
        let samples = model.synthesize(&request).map_err(|e| {
            let status = match e {
                crate::models::pivo::SynthesisError::InvalidInput(_) => StatusCode::BAD_REQUEST,
                crate::models::pivo::SynthesisError::Numerical(_) => {
                    StatusCode::INTERNAL_SERVER_ERROR
                }
            };
            (status, e.to_string())
        })?;
        crate::format::wav::encode_wav_pcm16_channels(&samples, crate::models::pivo::SAMPLE_RATE, 1)
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
