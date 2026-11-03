//! Paired native speech generation at the existing OpenAI speech endpoint.
use super::{compute_task, AppState, ErrorResponse};
use crate::app::cli::{resolve_thread_count, CliOptions};
use crate::app::tts::{arktts_sampling, reference_wav_at_rate};
use crate::core::tensor::TensorSource;
use crate::format::ggufrs::ComponentRole;
use crate::format::wav::encode_wav_pcm16_channels;
use crate::models::arktts::{codec::Codec, Model, Sampling, SAMPLE_RATE};
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::Engine as _;
use serde::Deserialize;
use std::sync::Arc;

pub(super) struct Backend {
    source: Arc<dyn TensorSource>,
    codec: Arc<dyn TensorSource>,
    sampling: Sampling,
    threads: usize,
    max_frames: usize,
}

impl Backend {
    pub(super) fn load(
        options: &CliOptions,
        source: Arc<dyn TensorSource>,
    ) -> Result<Self, String> {
        let sampling = arktts_sampling(options)?;
        if options.ref_audio.is_some() || options.ref_text.is_some() {
            return Err(
                "server voice cloning takes voice and reference_text in the speech request".into(),
            );
        }
        let path = options
            .mmproj
            .as_deref()
            .ok_or("Audio8 TTS requires an arktts_codec --mmproj")?;
        let codec: Arc<dyn TensorSource> =
            Arc::from(crate::app::open_or_exit(path, ComponentRole::Mmproj));
        crate::models::arktts::validate_pair(source.as_ref(), codec.as_ref())?;
        let threads = resolve_thread_count(
            options.threads,
            std::thread::available_parallelism().map_or(1, |n| n.get()),
        );
        Model::from_source(source.as_ref(), threads)?;
        Codec::from_source(codec.as_ref())?;
        let max_frames = options.max_tokens.unwrap_or(128);
        if !(1..2048).contains(&max_frames) {
            return Err("Audio8 max frames must be in 1..2048".into());
        }
        Ok(Self {
            source,
            codec,
            sampling,
            threads,
            max_frames,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    #[serde(rename = "model")]
    _model: String,
    input: String,
    voice: Option<String>,
    reference_text: Option<String>,
    response_format: Option<String>,
    speed: Option<f32>,
}

impl Request {
    fn validate(&self) -> Result<(), String> {
        if self.input.trim().is_empty() {
            return Err("input must not be empty".into());
        }
        if !matches!(self.response_format.as_deref(), None | Some("wav" | "pcm")) {
            return Err("response_format must be wav or pcm".into());
        }
        if self.speed.is_some_and(|speed| speed != 1.0) {
            return Err("Audio8 TTS supports speed=1 only".into());
        }
        let voice = self.voice.as_deref().is_some_and(|v| !v.is_empty());
        if voice
            != self
                .reference_text
                .as_deref()
                .is_some_and(|v| !v.trim().is_empty())
        {
            return Err("voice cloning requires both voice and nonempty reference_text".into());
        }
        Ok(())
    }
}

fn error(status: StatusCode, error: impl Into<String>) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: error.into(),
        }),
    )
        .into_response()
}

pub(super) async fn speech(
    State(state): State<AppState>,
    Json(request): Json<Request>,
) -> Response {
    let super::Backend::ArkTts(backend) = state.model.as_ref() else {
        return error(
            StatusCode::BAD_REQUEST,
            "server is not running an Audio8 TTS model",
        );
    };
    if let Err(message) = request.validate() {
        return error(StatusCode::BAD_REQUEST, message);
    }
    let reference = match request.voice.as_deref().filter(|v| !v.is_empty()) {
        None => Ok(None),
        Some(voice) => {
            if let Some(path) = voice.strip_prefix("file://") {
                std::fs::metadata(path)
                    .map_err(|e| format!("failed to inspect reference WAV: {e}"))
                    .and_then(|metadata| {
                        if !metadata.is_file() || metadata.len() > 16 * 1024 * 1024 {
                            return Err(
                                "reference WAV must be a regular file of at most 16 MiB".into()
                            );
                        }
                        std::fs::read(path)
                            .map(Some)
                            .map_err(|e| format!("failed to read reference WAV: {e}"))
                    })
            } else if let Some(payload) = voice.strip_prefix("data:audio/wav;base64,") {
                base64::engine::general_purpose::STANDARD
                    .decode(payload)
                    .map(Some)
                    .map_err(|e| format!("invalid reference base64: {e}"))
            } else {
                Err("voice must be file://path or data:audio/wav;base64,...".into())
            }
        }
    };
    let reference = match reference {
        Ok(value) => value,
        Err(message) => return error(StatusCode::BAD_REQUEST, message),
    };
    let permit = match state.generation_slot.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return error(
                StatusCode::TOO_MANY_REQUESTS,
                "a speech generation is already running",
            )
        }
    };
    let (source, audio) = (backend.source.clone(), backend.codec.clone());
    let (threads, sampling, max_frames) = (backend.threads, backend.sampling, backend.max_frames);
    let pcm = request.response_format.as_deref() == Some("pcm");
    let result = compute_task(state.compute, move || -> Result<Vec<u8>, String> {
        // Blocking work continues after HTTP cancellation, so it must own the generation slot.
        let _permit = permit;
        let reference = reference
            .as_deref()
            .map(|bytes| reference_wav_at_rate(bytes, SAMPLE_RATE))
            .transpose()?;
        let model = Model::from_source(source.as_ref(), threads)?;
        let codec = Codec::from_source(audio.as_ref())?;
        let waveform = model.synthesize(
            &codec,
            &request.input,
            request.reference_text.as_deref().zip(reference.as_deref()),
            max_frames,
            sampling,
        )?;
        let wav = encode_wav_pcm16_channels(&waveform, SAMPLE_RATE, 1)?;
        Ok(if pcm { wav[44..].to_vec() } else { wav })
    })
    .await;
    match result {
        Ok(Ok(bytes)) => (
            StatusCode::OK,
            [(
                header::CONTENT_TYPE,
                if pcm { "audio/pcm" } else { "audio/wav" },
            )],
            bytes,
        )
            .into_response(),
        Ok(Err(message)) => error(StatusCode::UNPROCESSABLE_ENTITY, message),
        Err(message) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("speech worker failed: {message}"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn speech_rejects_ignored_controls_and_incomplete_reference() {
        let mut request: Request =
            serde_json::from_value(serde_json::json!({"model":"audio8", "input":"hello"})).unwrap();
        assert!(request.validate().is_ok());
        request.response_format = Some("mp3".into());
        assert!(request.validate().is_err());
        request.response_format = Some("pcm".into());
        request.speed = Some(2.0);
        assert!(request.validate().is_err());
        request.speed = None;
        request.voice = Some("file://voice.wav".into());
        assert!(request.validate().is_err());
        request.reference_text = Some("hello".into());
        assert!(request.validate().is_ok());
        assert!(serde_json::from_value::<Request>(
            serde_json::json!({"model":"audio8", "input":"hello", "temperature":0.5})
        )
        .is_err());
    }
}
