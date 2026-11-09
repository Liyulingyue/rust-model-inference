//! Native decisions and the existing JEV request shapes on a decision backend.

use std::sync::Arc;

use axum::extract::{rejection::JsonRejection, DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{AppState, Backend};
use crate::app::decision::JevPlan;
use crate::app::{JevGroupedQuestionInput, JevInputs, JevMode, JevQuestionInput};

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/decide", post(decide))
        .route("/v1/decide/batch", post(batch))
        .route("/v1/jev/score", post(jev_score))
        .route("/v1/jev/grouped", post(jev_grouped))
        .route("/v1/jev/image", post(jev_image))
        .route("/v1/jev/image_grouped", post(jev_image_grouped))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    let kind = if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    (
        status,
        Json(json!({"error":{"type":kind,"message":message.into()}})),
    )
        .into_response()
}

// shortcut: Engine exposes string errors; use typed request errors if that API changes.
fn inference_error_status(message: &str) -> StatusCode {
    if (message.starts_with("THX-02 question needs ") && message.contains(", over context "))
        || message == "THX-02 thinker prompt exceeds context"
        || message == "THX-02 invalid forward dimensions or context exceeded"
        || message == "THX-02 missing thinker"
        || message.starts_with("THX-02 think requires the matching merged thinker GGUF")
        || message.starts_with("THX-02 input contains non-finite")
    {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

fn parse_body(body: Result<Json<Value>, JsonRejection>) -> Result<Value, Response> {
    body.map(|Json(value)| value).map_err(|error_value| {
        error(
            StatusCode::BAD_REQUEST,
            format!("invalid JSON: {error_value}"),
        )
    })
}

fn prepare_request(mut request: Value) -> Result<Value, String> {
    crate::models::thx02::validate_request(&request)?;
    if let Some(images) = request.get_mut("images").and_then(Value::as_array_mut) {
        for source in images {
            let text = source.as_str().ok_or("images must contain strings")?.trim();
            // The shared decoder also accepts file paths; HTTP callers may only supply URLs.
            let data_url = text.starts_with("data:image/")
                && text
                    .split_once(',')
                    .is_some_and(|(metadata, _)| metadata.contains(";base64"));
            if !data_url && !text.starts_with("http://") && !text.starts_with("https://") {
                return Err("HTTP decision images require data URLs or enabled remote URLs; local files are CLI-only".into());
            }
            let image = super::api::image_input::decode_image_source(
                text,
                super::api::image_input::allow_remote(),
            )?;
            crate::app::media::decode_image_bytes(&image.bytes)?;
            let format = image::guess_format(&image.bytes).map_err(|error| error.to_string())?;
            *source = json!(format!(
                "data:{};base64,{}",
                format.to_mime_type(),
                base64::engine::general_purpose::STANDARD.encode(&image.bytes)
            ));
        }
    }
    Ok(request)
}

async fn execute(
    state: AppState,
    requests: Vec<Value>,
    batch: bool,
    plan: Option<JevPlan>,
) -> Response {
    if !matches!(state.model.as_ref(), Backend::Decision(_)) {
        return error(
            StatusCode::NOT_FOUND,
            "this server is not a decision backend",
        );
    }
    let permit = match Arc::clone(&state.generation_slot).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return error(
                StatusCode::TOO_MANY_REQUESTS,
                "a decision request is already running",
            )
        }
    };
    let model = Arc::clone(&state.model);
    let result = super::compute_task(
        state.compute,
        move || -> Result<Value, (StatusCode, String)> {
            // Cancellation must retain the permit until the shared model pool is idle.
            let _permit = permit;
            let Backend::Decision(engine) = model.as_ref() else {
                unreachable!()
            };
            // Remote images use a blocking client; validate every batch item before forwarding.
            let requests = requests
                .into_iter()
                .map(prepare_request)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
            let results = requests
                .iter()
                .map(|request| engine.decide(request))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|message| (inference_error_status(&message), message))?;
            if let Some(plan) = plan {
                return plan
                    .format(&results[0])
                    .map_err(|message| (StatusCode::INTERNAL_SERVER_ERROR, message));
            }
            if batch {
                Ok(json!({"model":"THX-02","results":results}))
            } else {
                results.into_iter().next().ok_or_else(|| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "missing decision request".into(),
                    )
                })
            }
        },
    )
    .await;
    match result {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err((status, message))) => error(status, message),
        Err(join_error) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("decision task failed: {join_error}"),
        ),
    }
}

async fn decide(
    State(state): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let request = match parse_body(body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    execute(state, vec![request], false, None).await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchRequest {
    requests: Vec<Value>,
}

async fn batch(
    State(state): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let requests = match parse_body(body).and_then(|value| {
        let batch: BatchRequest = serde_json::from_value(value)
            .map_err(|message| error(StatusCode::BAD_REQUEST, message.to_string()))?;
        if batch.requests.is_empty() {
            return Err(error(StatusCode::BAD_REQUEST, "requests must not be empty"));
        }
        Ok(batch.requests)
    }) {
        Ok(requests) => requests,
        Err(response) => return response,
    };
    execute(state, requests, true, None).await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SingleRequest {
    #[serde(default)]
    context: String,
    questions: Vec<JevQuestionInput>,
    #[serde(default)]
    positive: Option<String>,
    #[serde(default)]
    image_url: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupedRequest {
    #[serde(default)]
    context: String,
    questions: Vec<JevGroupedQuestionInput>,
    #[serde(default = "default_grouped_mode")]
    mode: String,
    #[serde(default)]
    image_url: Option<String>,
}

fn default_grouped_mode() -> String {
    "multi_select".into()
}

fn prepare_jev(value: Value, grouped: bool, image: bool) -> Result<JevPlan, String> {
    let (inputs, image_url) = if grouped {
        let request: GroupedRequest =
            serde_json::from_value(value).map_err(|error| error.to_string())?;
        let mode = match request.mode.as_str() {
            "multi_select" => JevMode::MultiSelect,
            "block_choice" => JevMode::BlockChoice,
            _ => return Err("mode must be multi_select or block_choice".into()),
        };
        (
            JevInputs::Grouped {
                context: request.context,
                questions: request.questions,
                mode,
            },
            request.image_url,
        )
    } else {
        let request: SingleRequest =
            serde_json::from_value(value).map_err(|error| error.to_string())?;
        (
            JevInputs::Single {
                context: request.context,
                questions: request.questions,
                positive: request.positive,
            },
            request.image_url,
        )
    };
    if image
        && image_url
            .as_deref()
            .is_none_or(|source| source.trim().is_empty())
    {
        return Err("image_url is required for image JEV".into());
    }
    if !image && image_url.is_some() {
        return Err("image_url requires an image JEV route".into());
    }
    JevPlan::new(inputs, image_url.into_iter().collect())
}

async fn jev(
    state: AppState,
    body: Result<Json<Value>, JsonRejection>,
    grouped: bool,
    image: bool,
) -> Response {
    let plan = match parse_body(body).and_then(|value| {
        prepare_jev(value, grouped, image)
            .map_err(|message| error(StatusCode::BAD_REQUEST, message))
    }) {
        Ok(plan) => plan,
        Err(response) => return response,
    };
    execute(state, vec![plan.request.clone()], false, Some(plan)).await
}

async fn jev_score(
    State(state): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    jev(state, body, false, false).await
}
async fn jev_grouped(
    State(state): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    jev(state, body, true, false).await
}
async fn jev_image(
    State(state): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    jev(state, body, false, true).await
}
async fn jev_image_grouped(
    State(state): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    jev(state, body, true, true).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> Value {
        json!({"state":"hello","questions":{"q":{"type":"choice","instructions":"pick","criteria":{"a":"A","b":"B"}}}})
    }

    #[test]
    fn decision_http_rejects_bad_schema_streaming_and_local_files() {
        assert!(prepare_request(json!({"state":"hello","questions":[]})).is_err());
        for source in [
            "/etc/passwd",
            "file:///etc/passwd",
            "images/cat.png",
            "data:local.png",
            "http://127.0.0.1/image.png",
        ] {
            let mut value = request();
            value["images"] = json!([source]);
            assert!(prepare_request(value).is_err(), "{source}");
        }
        let mut value = request();
        value["stream"] = json!(true);
        assert!(prepare_request(value).is_err());
        let mut value = request();
        value["stream"] = json!(false);
        assert!(prepare_request(value).is_ok());
    }

    #[test]
    fn decision_http_normalizes_data_images_with_the_shared_decoder() {
        let png = crate::app::server::api::fixtures::synthetic_png_b64();
        let mut value = request();
        value["images"] = json!([format!("data:image/png;base64,{png}")]);
        assert!(prepare_request(value).unwrap()["images"][0]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
    }

    #[test]
    fn decision_http_rejects_truncated_images_before_inference() {
        let mut value = request();
        // A valid PNG signature is insufficient without an image body.
        value["images"] = json!(["data:image/png;base64,iVBORw0KGgo="]);
        let error = prepare_request(value)
            .expect_err("truncated PNG must fail media validation before inference");
        assert!(error.starts_with("Failed to decode image:"), "{error}");
    }

    #[test]
    fn decision_jev_validates_before_any_forward() {
        assert!(prepare_jev(json!({"questions":[]}), false, false).is_err());
        assert!(prepare_jev(
            json!({"questions":[{"text":"pick","options":["a","b"]}]}),
            false,
            true
        )
        .is_err());
        assert!(prepare_jev(
            json!({"questions":[{"text":"pick","options":["a","b"]}],"positive":"Z"}),
            false,
            false
        )
        .is_err());
        let plan = prepare_jev(
            json!({"context":"hello","questions":[{"text":"pick","options":["a","b"]}]}),
            false,
            false,
        )
        .unwrap();
        assert_eq!(
            plan.request["questions"]["q0"]["criteria"],
            json!({"A":"a","B":"b"})
        );
    }

    #[test]
    fn decision_runtime_errors_distinguish_bad_requests_from_model_failures() {
        for message in [
            "THX-02 question needs 123 tokens, over context 64; nothing was truncated",
            "THX-02 thinker prompt exceeds context",
            "THX-02 think requires the matching merged thinker GGUF (--decision-thinker)",
            "THX-02 missing thinker",
            "THX-02 input contains non-finite values",
        ] {
            assert_eq!(
                inference_error_status(message),
                StatusCode::BAD_REQUEST,
                "{message}"
            );
        }
        for message in [
            "THX-02 produced non-finite decision logits",
            "THX-02 missing decision head",
        ] {
            assert_eq!(
                inference_error_status(message),
                StatusCode::INTERNAL_SERVER_ERROR,
                "{message}"
            );
        }
    }

    #[tokio::test]
    async fn decision_json_rejects_nonfinite_numbers_as_bad_requests() {
        use axum::extract::FromRequest;
        let request = axum::http::Request::builder()
            .method("POST")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(r#"{"state":1e999,"questions":{}}"#))
            .unwrap();
        let body = Json::<Value>::from_request(request, &()).await;
        assert_eq!(
            parse_body(body).err().unwrap().status(),
            StatusCode::BAD_REQUEST
        );
    }
}
