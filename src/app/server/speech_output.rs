//! HTTP speech output and PCM streaming over the model's shared synthesis engine.

use super::{AppState, Backend};
use crate::models::sophon_realtime::speech::{SpeechEngine, SpeechRequest};
use axum::{
    extract::{rejection::JsonRejection, State},
    http::{header, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::json;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

pub(super) struct SpeechBackend {
    engine: Arc<Mutex<SpeechEngine>>,
    voice: String,
    max_tokens: usize,
    temperature: f32,
    seed: u64,
}

pub(super) fn build(
    options: &crate::app::CliOptions,
    source: Arc<dyn crate::core::tensor::TensorSource>,
) -> Result<SpeechBackend, String> {
    crate::app::tts::validate_tag_speech_options(options)?;
    let path = options
        .mmproj
        .as_deref()
        .ok_or("speech requires --mmproj")?;
    build_from_source(options, source, path, options.max_tokens.unwrap_or(128))
}

pub(super) fn build_postprocessor(
    options: &crate::app::CliOptions,
) -> Result<SpeechBackend, String> {
    crate::app::tts::validate_tag_speech_options(options)?;
    let talker = options
        .tts_model
        .as_deref()
        .ok_or("speech requires --tts-model")?;
    let codec = options
        .tts_mmproj
        .as_deref()
        .ok_or("--tts-model requires --tts-mmproj")?;
    let source = Arc::from(
        crate::format::ggufrs::open_model_source(talker, crate::format::ggufrs::ComponentRole::Llm)
            .map_err(|e| e.to_string())?,
    );
    build_from_source(
        options,
        source,
        codec,
        options
            .max_tokens
            .unwrap_or(128)
            .saturating_mul(4)
            .clamp(128, 1250),
    )
}

fn build_from_source(
    options: &crate::app::CliOptions,
    source: Arc<dyn crate::core::tensor::TensorSource>,
    path: &std::path::Path,
    max_tokens: usize,
) -> Result<SpeechBackend, String> {
    let code2wav = Arc::from(
        crate::format::ggufrs::open_model_source(
            path,
            crate::format::ggufrs::ComponentRole::Mmproj,
        )
        .map_err(|e| e.to_string())?,
    );
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let engine = SpeechEngine::new(
        source,
        code2wav,
        crate::app::resolve_thread_count(options.threads, available),
        options.compute_policy(),
    )?;
    let backend = SpeechBackend {
        engine: Arc::new(Mutex::new(engine)),
        voice: options.voice.clone().unwrap_or_else(|| "brand_zh".into()),
        max_tokens,
        temperature: options.temperature.unwrap_or(0.0),
        seed: options.seed.unwrap_or(0) as u64,
    };
    backend
        .engine
        .lock()
        .map_err(|e| e.to_string())?
        .validate(&SpeechRequest {
            text: "probe",
            voice: &backend.voice,
            max_tokens: backend.max_tokens,
            temperature: backend.temperature,
            seed: backend.seed,
            stream: false,
        })?;
    Ok(backend)
}

impl SpeechBackend {
    pub(super) fn synthesize(
        &self,
        text: &str,
        stream: bool,
        emit: impl FnMut(&[f32]) -> bool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<crate::models::sophon_realtime::speech::SpeechOutput, String> {
        self.engine
            .lock()
            .map_err(|e| e.to_string())?
            .synthesize_until(
                &SpeechRequest {
                    text,
                    voice: &self.voice,
                    max_tokens: self.max_tokens,
                    temperature: self.temperature,
                    seed: self.seed,
                    stream,
                },
                emit,
                cancelled,
            )
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    #[serde(default, rename = "model")]
    _model: Option<String>,
    input: String,
    voice: Option<String>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    seed: Option<u64>,
    speed: Option<f32>,
    response_format: Option<String>,
    #[serde(default)]
    stream: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Wav,
    Pcm,
}
impl Request {
    fn format(&self) -> Result<Format, String> {
        if self.speed.is_some_and(|v| v != 1.0) {
            return Err("speech supports speed=1 only".into());
        }
        let format =
            match self
                .response_format
                .as_deref()
                .unwrap_or(if self.stream { "pcm" } else { "wav" })
            {
                "wav" => Format::Wav,
                "pcm" => Format::Pcm,
                _ => return Err("response_format must be wav or pcm".into()),
            };
        if self.stream && format != Format::Pcm {
            return Err("streaming speech requires response_format=pcm".into());
        }
        Ok(format)
    }
}

fn bytes(samples: &[f32], format: Format) -> Result<Vec<u8>, String> {
    let wav = crate::format::wav::encode_wav_pcm16_channels(samples, 24_000, 1)?;
    if format == Format::Wav {
        return Ok(wav);
    }
    if wav.get(36..40) != Some(b"data") {
        return Err("unexpected PCM WAV layout".into());
    }
    Ok(wav[44..].to_vec())
}
fn error(status: StatusCode, message: impl ToString) -> Response {
    (status, Json(json!({"error":message.to_string()}))).into_response()
}

pub(super) async fn handle(
    State(state): State<AppState>,
    body: Result<Json<Request>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(value)) => value,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.body_text()),
    };
    let backend = match state.model.as_ref() {
        Backend::TagSpeech(backend) => backend,
        Backend::AudioChat(backend) => &backend.speech,
        _ => {
            return error(
                StatusCode::BAD_REQUEST,
                "server has no speech synthesis runtime",
            )
        }
    };
    let format = match request.format() {
        Ok(value) => value,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let permit = match state.generation_slot.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, e),
    };
    let voice = request
        .voice
        .clone()
        .unwrap_or_else(|| backend.voice.clone());
    let max_tokens = request.max_tokens.unwrap_or(backend.max_tokens);
    let temperature = request.temperature.unwrap_or(backend.temperature);
    let seed = request.seed.unwrap_or(backend.seed);
    let resolved = SpeechRequest {
        text: &request.input,
        voice: &voice,
        max_tokens,
        temperature,
        seed,
        stream: request.stream,
    };
    let validation = backend
        .engine
        .lock()
        .map_err(|e| e.to_string())
        .and_then(|engine| engine.validate(&resolved));
    if let Err(e) = validation {
        return error(StatusCode::BAD_REQUEST, e);
    }
    let engine = Arc::clone(&backend.engine);
    if !request.stream {
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut cancellation_guard = super::api::CancellationGuard::new(cancelled.clone());
        let result = super::compute_task(state.compute, move || -> Result<Vec<u8>, String> {
            let _permit = permit;
            let resolved = SpeechRequest {
                text: &request.input,
                voice: &voice,
                max_tokens,
                temperature,
                seed,
                stream: false,
            };
            let output = engine.lock().map_err(|e| e.to_string())?.synthesize_until(
                &resolved,
                |_| true,
                &|| cancelled.load(Ordering::Relaxed),
            )?;
            bytes(&output.waveform, format)
        })
        .await;
        cancellation_guard.disarm();
        return match result {
            Ok(Ok(bytes)) => (
                [(
                    header::CONTENT_TYPE,
                    if format == Format::Wav {
                        "audio/wav"
                    } else {
                        "audio/pcm"
                    },
                )],
                bytes,
            )
                .into_response(),
            Ok(Err(e)) => error(StatusCode::UNPROCESSABLE_ENTITY, e),
            Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
        };
    }
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, axum::Error>>(8);
    tokio::spawn(async move {
        let result_tx = tx.clone();
        let result = super::compute_task(state.compute, move || -> Result<serde_json::Value,String> {
            let _permit = permit;
            let resolved = SpeechRequest {text:&request.input,voice:&voice,max_tokens,temperature,seed,stream:true};
            let mut encoding_error = None;
            let output = engine.lock().map_err(|e| e.to_string())?.synthesize_until(&resolved, |chunk| {
                let pcm = match bytes(chunk,Format::Pcm) { Ok(value) => value, Err(e) => {encoding_error=Some(e); return false;} };
                tx.blocking_send(Ok(Event::default().event("audio").data(json!({"audio":base64::engine::general_purpose::STANDARD.encode(pcm),"format":"pcm16","sample_rate":24000,"channels":1}).to_string()))).is_ok()
            }, &|| tx.is_closed())?;
            if let Some(error) = encoding_error { return Err(error); }
            Ok(json!({"speech_tokens":output.speech_tokens.len(),"samples":output.waveform.len(),"terminated":output.terminated,"cancelled":output.cancelled}))
        }).await;
        let (event, data) = match result {
            Ok(Ok(value)) => ("done", value),
            Ok(Err(e)) => ("error", json!({"error":e})),
            Err(e) => ("error", json!({"error":e.to_string()})),
        };
        let _ = result_tx
            .send(Ok(Event::default().event(event).data(data.to_string())))
            .await;
    });
    Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn speech_format_validation_and_pcm_payload_are_explicit() {
        let request = |value| serde_json::from_value::<Request>(value).unwrap();
        assert_eq!(
            request(json!({"input":"hello"})).format().unwrap(),
            Format::Wav
        );
        assert_eq!(
            request(json!({"input":"hello","stream":true}))
                .format()
                .unwrap(),
            Format::Pcm
        );
        for value in [
            json!({"input":"a","stream":true,"response_format":"wav"}),
            json!({"input":"a","response_format":"mp3"}),
            json!({"input":"a","speed":2}),
        ] {
            assert!(request(value).format().is_err());
        }
        assert_eq!(
            bytes(&[0., 0.5, -0.5], Format::Pcm).unwrap(),
            [0, 0, 255, 63, 1, 192]
        );
        assert!(
            serde_json::from_value::<Request>(json!({"input":"a","path":"/tmp/voice"})).is_err()
        );
    }
}
