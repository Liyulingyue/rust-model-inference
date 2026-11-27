//! `/v1/forecast` HTTP endpoint for Timestral time-series forecasting.
//!
//! Request schema (the same one the `--timestral-request` file takes):
//! ```json
//! {
//!   "context": [1.0, 2.0, null, 4.0],          // univariate
//!   "prediction_length": 12,
//!   "observation_mask": [[true, true, false, true]],
//!   "past_covariates": [[1.0, 2.0, 3.0, 4.0]],
//!   "known_covariates": [[5.0, 6.0, 7.0, 8.0]],
//!   "symmetric_averaging": true,
//!   "clip_nonnegative": true,
//!   "max_context": 2048,
//!   "per_shot_horizon_patches": 32
//! }
//! ```
//!
//! A multivariate target sends a list of series instead:
//! ```json
//! { "context": [[1.0, 2.0], [3.0, 4.0]], "prediction_length": 12 }
//! ```
//!
//! Response:
//! ```json
//! {
//!   "quantile_levels": [0.01, 0.05, ...],
//!   "quantiles": [[[...], ...], ...]
//! }
//! ```
//!
//! `quantiles` is indexed `[target variate][horizon step][quantile]`, the
//! same shape the CLI prints.

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;
use std::sync::Arc;

use crate::app::server::{AppState, Backend};
use crate::models::timestral::ForecastRequest;

/// The error body of a rejected request.
#[derive(Serialize)]
struct ForecastError {
    error: String,
}

/// The forecast the caller receives.
#[derive(Serialize)]
struct ForecastResponse {
    /// The probability each quantile level stands for.
    quantile_levels: Vec<f32>,
    /// `[target variate][horizon step][quantile]`.
    quantiles: Vec<Vec<Vec<f32>>>,
}

pub async fn forecast(
    State(state): State<AppState>,
    body: Result<Json<ForecastRequest>, axum::extract::rejection::JsonRejection>,
) -> impl axum::response::IntoResponse {
    let Json(req) = match body {
        Ok(body) => body,
        Err(rejection) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(ForecastError {
                    error: rejection.body_text(),
                }),
            )
                .into_response();
        }
    };
    let Backend::Timestral(backend) = state.model.as_ref() else {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(ForecastError {
                error: "server is not configured for forecasting".into(),
            }),
        )
            .into_response();
    };

    // The forecast is a blocking computation, so it runs on the blocking
    // pool instead of a worker thread. The compute scope belongs to that
    // thread, which is what `compute_task` sets up.
    let model = Arc::clone(&backend.model);
    let result = match super::compute_task(state.compute, move || model.forecast(&req)).await {
        Ok(result) => result,
        Err(error) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(ForecastError {
                    error: format!("forecast task failed: {error}"),
                }),
            )
                .into_response();
        }
    };

    match result {
        Ok(forecast) => Json(ForecastResponse {
            quantile_levels: forecast.quantile_levels,
            quantiles: forecast.quantiles,
        })
        .into_response(),
        // The model's own checks are all about the request: a missing
        // series, a mask of the wrong shape, a value that is not finite.
        Err(error) => (
            axum::http::StatusCode::BAD_REQUEST,
            Json(ForecastError { error }),
        )
            .into_response(),
    }
}
