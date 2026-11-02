//! `/v1/prediction` HTTP endpoint for image prediction with structured output.
//!
//! The endpoint answers a task over one image and returns the model's text
//! together with the structured geometry it produced. The model behind it is
//! WeVisOne-VL-4B, a Qwen3-VL release with a coordinate vocabulary and a
//! separate pixel decoder, so the task list and the `<OBJ>`/alpha-mask
//! contract below are that model's shape rather than a generic one.
//!
//! Request schema (the same one the `--wevisone-request` file takes):
//! ```json
//! {
//!   "image": "image.png",           // local path, data URL or http URL
//!   "task": "segmentation",         // detection, referring, point, count_box,
//!                                   // count_point, segmentation, matting, ocr,
//!                                   // ocr_grounding, visual_prompt, freeform
//!   "query": "the red rectangle",   // task-specific text
//!   "max_new_tokens": 32
//! }
//! ```
//! `visual_prompt` takes `reference_boxes` as pixel-space xyxy arrays and
//! ignores `query`; the other tasks reject `reference_boxes`.
//!
//! Response:
//! ```json
//! {
//!   "response": "<ref>the red rectangle</ref><box>...</box>",
//!   "token_ids": [152670, 1782],
//!   "boxes": [[0.258, 0.17, 0.773, 0.799]],
//!   "alpha_png": ["data:image/png;base64,..."],
//!   "width": 128,
//!   "height": 128
//! }
//! ```
//! `boxes` holds the normalized `<OBJ>` boxes and `alpha_png` one grayscale
//! PNG data URI per box. Both stay empty unless the response contains
//! `<OBJ>`, which needs the `--wevisone-decoder` component. The route is
//! only mounted for a model that speaks this contract.

use super::{compute_task, AppState, Backend};
use crate::models::wevisone_vl::{predict, PredictRequest};
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
    let Backend::WeVisOne(backend) = state.model.as_ref() else {
        return error(
            StatusCode::BAD_REQUEST,
            "WeVisOne text backend required".into(),
        );
    };
    let text = &backend.text;
    if text.source.metadata("wevisone_vl.source_revision").is_none() || text.mmproj.is_none() {
        return error(
            StatusCode::BAD_REQUEST,
            "WeVisOne text and vision components required".into(),
        );
    }
    if matches!(
        request.task.trim().to_lowercase().as_str(),
        "segmentation" | "matting"
    ) && backend.decoder.is_none()
    {
        return error(
            StatusCode::BAD_REQUEST,
            "server requires --wevisone-decoder for alpha masks".into(),
        );
    }
    let options = backend.options;
    let source = text.source.clone();
    let vision = text.mmproj.clone().unwrap();
    let decoder = backend.decoder.clone();
    let pool = text.pool.clone();
    let tokenizer = text.tokenizer.clone();
    // One active generation protects shared GPU/model state and bounds
    // server memory, exactly like the chat and JEV routes.
    let permit = match state.generation_slot.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return error(
                StatusCode::TOO_MANY_REQUESTS,
                "Another generation is running; retry after it completes".into(),
            )
        }
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
