use super::{compute_task, AppState, Backend};
use crate::app::wevisone_vl::{predict, PredictRequest};
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};

pub(super) async fn prediction(
    State(state): State<AppState>,
    body: Result<Json<PredictRequest>, axum::extract::rejection::JsonRejection>,
) -> axum::response::Response {
    let request = match body {
        Ok(Json(r)) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.body_text()),
    };
    if let Err(e) = request.validate() {
        return error(StatusCode::BAD_REQUEST, e);
    }
    let Backend::Text(backend) = state.model.as_ref() else {
        return error(
            StatusCode::BAD_REQUEST,
            "WeVisOne text backend required".into(),
        );
    };
    if backend
        .source
        .metadata("wevisone_vl.source_revision")
        .is_none()
        || backend.mmproj.is_none()
    {
        return error(
            StatusCode::BAD_REQUEST,
            "WeVisOne text and vision components required".into(),
        );
    }
    if matches!(
        request.task.trim().to_lowercase().as_str(),
        "segmentation" | "matting"
    ) && backend.wevisone_decoder.is_none()
    {
        return error(
            StatusCode::BAD_REQUEST,
            "server requires --wevisone-decoder for alpha masks".into(),
        );
    }
    let Some(options) = backend.wevisone_options else {
        return error(
            StatusCode::BAD_REQUEST,
            "WeVisOne configuration required".into(),
        );
    };
    let source = backend.source.clone();
    let vision = backend.mmproj.clone().unwrap();
    let decoder = backend.wevisone_decoder.clone();
    let pool = backend.pool.clone();
    let tokenizer = backend.tokenizer.clone();
    let permit = match state.generation_slot.clone().acquire_owned().await {
        Ok(p) => p,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    match compute_task(state.compute, move || {
        let _permit = permit;
        let image_ref = super::api::image_input::decode_image_source(
            &request.image,
            super::api::image_input::allow_remote(),
        )?;
        let image = super::api::image_input::decode_image_bytes(&image_ref.bytes)?;
        predict(
            source,
            vision.as_ref(),
            decoder.as_deref(),
            pool,
            tokenizer,
            options,
            &request,
            &image,
        )
    })
    .await
    {
        Ok(Ok(result)) => Json(result).into_response(),
        Ok(Err(e)) => error(StatusCode::BAD_REQUEST, e),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}
fn error(status: StatusCode, message: String) -> axum::response::Response {
    (status, Json(serde_json::json!({"error":message}))).into_response()
}
