//! `/v1/ocr` — page OCR for the OCRia backend.
//!
//! Named by capability, not by model: the route reads a page image and returns
//! the line list, and the model behind it is whatever the server was built with
//! (today the OCRia backend).
//!
//! The request carries one page image and nothing else; the response is the
//! line list the CLI writes, so the two entry points return the same JSON for
//! the same page.
//!
//! ```json
//! { "image": "data:image/png;base64,iVBORw0KGgo..." }
//! ```
//!
//! ```json
//! {
//!   "width": 1240,
//!   "height": 1754,
//!   "lines": [
//!     { "text": "Invoice", "bbox": [96, 120, 240, 148], "confidence": 0.98,
//!       "polygon": [[96, 120], [240, 120], [240, 148], [96, 148]],
//!       "faint": false }
//!   ]
//! }
//! ```
//!
//! `image` accepts a `data:` URL or bare base64. Remote URLs are refused: the
//! upstream pipeline is a local page reader, and fetching a caller-supplied URL
//! server-side would make the endpoint a request forwarder.

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use serde::Serialize;
use std::sync::Arc;

use super::api::base64_decode;
use super::{AppState, Backend};
use crate::models::ocria::OptionsOverride;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OcrRequest {
    /// A `data:<mime>;base64,<data>` URL or bare base64.
    pub image: String,
    /// Optional pipeline overrides for this page; absent fields keep the
    /// server's defaults (which come from the model's contract and the
    /// `--ocr-options` the server was started with).
    #[serde(default)]
    pub options: Option<OptionsOverride>,
}

#[derive(Serialize)]
struct OcrError {
    error: String,
}

/// Decodes a `data:` URL or bare base64 page image.
fn decode_page(image: &str) -> Result<Vec<u8>, String> {
    if image.starts_with("http://") || image.starts_with("https://") {
        return Err("remote image URLs are not supported; send a base64 data URL".into());
    }
    match image.strip_prefix("data:") {
        Some(rest) => {
            let (header, body) = rest
                .split_once(',')
                .ok_or("malformed data URL: missing comma")?;
            if !header.contains(";base64") {
                return Err(
                    "only base64 data URLs are supported (data:<mime>;base64,<data>)".into(),
                );
            }
            base64_decode(body)
        }
        None => base64_decode(image),
    }
}

pub async fn ocr(
    State(state): State<AppState>,
    body: Result<Json<OcrRequest>, axum::extract::rejection::JsonRejection>,
) -> impl IntoResponse {
    let Json(request) = match body {
        Ok(body) => body,
        Err(rejection) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(OcrError {
                    error: rejection.body_text(),
                }),
            )
                .into_response();
        }
    };
    let Backend::Ocria(model) = state.model.as_ref() else {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(OcrError {
                error: "server is not configured for OCR".into(),
            }),
        )
            .into_response();
    };
    let bytes = match decode_page(&request.image) {
        Ok(bytes) => bytes,
        Err(error) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(OcrError { error }),
            )
                .into_response();
        }
    };
    let decoded = match crate::app::media::decode_image_bytes(&bytes) {
        Ok(image) => image,
        Err(error) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(OcrError { error }),
            )
                .into_response();
        }
    };
    let options = match request.options.as_ref() {
        Some(overrides) => match overrides.apply(model.options()) {
            Ok(options) => options,
            Err(error) => {
                return (
                    axum::http::StatusCode::BAD_REQUEST,
                    Json(OcrError { error }),
                )
                    .into_response();
            }
        },
        None => model.options(),
    };
    let rgb = decoded.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    // `ComputePool::compute` publishes one closure into a single shared slot,
    // so two pages must not run through the same pool at once: the second would
    // overwrite the first one's work. The slot also caps a server at one page
    // per process, which is what the other single-pool backends do.
    let Ok(permit) = state.generation_slot.clone().try_acquire_owned() else {
        return (
            axum::http::StatusCode::TOO_MANY_REQUESTS,
            Json(OcrError {
                error: "OCR model is busy".into(),
            }),
        )
            .into_response();
    };
    let page = Arc::clone(model);
    let result = match super::compute_task(state.compute, move || {
        let _permit = permit;
        page.run_with(options, rgb.as_raw(), width, height)
    })
    .await
    {
        Ok(result) => result,
        Err(error) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(OcrError {
                    error: format!("OCR task failed: {error}"),
                }),
            )
                .into_response();
        }
    };
    match result {
        Ok(page) => Json(page).into_response(),
        // The model's own checks are about the request (an empty or truncated
        // page); the arithmetic failures come back as 500.
        Err(crate::models::ocria::OcrError::Invalid(error)) => (
            axum::http::StatusCode::BAD_REQUEST,
            Json(OcrError { error }),
        )
            .into_response(),
        Err(error) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(OcrError {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

/// The route a caller can probe for OCR support.
pub(super) const ROUTE: &str = "/v1/ocr";
