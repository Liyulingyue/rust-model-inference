//! Binary PCM16 WAV input/output for audio enhancement.
use super::{AppState, Backend, ErrorResponse};
use axum::{
    body::Bytes,
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::IntoResponse,
    Json,
};

pub(super) async fn enhance(
    State(state): State<AppState>,
    headers: HeaderMap,
    bytes: Bytes,
) -> axum::response::Response {
    if !headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            let media = v.split(';').next().unwrap_or("").trim();
            ["audio/wav", "audio/x-wav", "audio/wave"]
                .iter()
                .any(|expected| media.eq_ignore_ascii_case(expected))
        })
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "expected Content-Type: audio/wav and a PCM16 WAV body".into(),
            }),
        )
            .into_response();
    }
    let Backend::AudioEnhancement(model) = state.model.as_ref() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let permit = match state.generation_slot.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: "audio enhancement is busy".into(),
                }),
            )
                .into_response()
        }
    };
    let model = model.clone();
    match super::compute_task(state.compute, move || {
        let _permit = permit;
        model.enhance_wav(&bytes)
    })
    .await
    {
        Ok(Ok(wav)) => ([(header::CONTENT_TYPE, "audio/wav")], wav).into_response(),
        Ok(Err(error)) => (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("enhancement worker failed: {error}"),
            }),
        )
            .into_response(),
    }
}
