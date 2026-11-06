//! Native image-to-3D generation; mounted only for the geometry backend.
use super::{compute_task, AppState, Backend};
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    image: String,
    #[serde(default)]
    options: crate::models::idol::Options,
}
pub(super) async fn generate(
    State(state): State<AppState>,
    body: Result<Json<Request>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let req = match body {
        Ok(Json(r)) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.body_text()),
    };
    if let Err(e) = req.options.validate() {
        return error(StatusCode::BAD_REQUEST, e);
    }
    let Backend::Idol(backend) = state.model.as_ref() else {
        return error(
            StatusCode::BAD_REQUEST,
            "Image-to-3D backend required".into(),
        );
    };
    let permit = match state.generation_slot.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return error(
                StatusCode::TOO_MANY_REQUESTS,
                "Another generation is running".into(),
            )
        }
    };
    let model = backend.model.clone();
    match compute_task(state.compute, move || {
        let _permit = permit;
        let image_ref = super::api::image_input::decode_image_source(
            &req.image,
            super::api::image_input::allow_remote(),
        )
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
        let image = super::api::image_input::decode_image_bytes(&image_ref.bytes)
            .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
        let output = model
            .generate(image, &req.options)
            .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, e))?;
        let bytes = crate::app::geometry::ply_bytes(&output.gaussians)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
        let camera = serde_json::to_string(&output.camera)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok::<_, (StatusCode, String)>((bytes, camera))
    })
    .await
    {
        Ok(Ok((bytes, camera))) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "application/octet-stream".to_string()),
                (header::HeaderName::from_static("x-camera"), camera),
            ],
            bytes,
        )
            .into_response(),
        Ok(Err((status, e))) => error(status, e),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}
fn error(status: StatusCode, message: String) -> Response {
    (status, Json(serde_json::json!({"error":message}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::Request;

    #[test]
    fn request_options_are_strict_and_default_to_official_schedule() {
        let request: Request = serde_json::from_str(r#"{"image":"object.png"}"#).unwrap();
        assert_eq!(request.options.steps, 50);
        assert!(request.options.validate().is_ok());
        for body in [
            r#"{"image":"object.png","prompt":"ignored"}"#,
            r#"{"image":"object.png","options":{"temperature":1}}"#,
            r#"{"image":"object.png","options":{"seed":-1}}"#,
        ] {
            assert!(serde_json::from_str::<Request>(body).is_err());
        }
        let request: Request =
            serde_json::from_str(r#"{"image":"object.png","options":{"num_gaussians":32769}}"#)
                .unwrap();
        assert!(request.options.validate().is_err());
    }
}
