use super::{AppState, Backend};
use crate::ops::generation_runtime::{CollectSink, Flow, GenerationRequest, TokenSink};
use axum::{
    extract::{rejection::JsonRejection, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

pub(super) struct AudioChatBackend {
    pub(super) text: super::TextBackend,
    pub(super) speech: super::speech_output::SpeechBackend,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    audio: String,
    #[serde(default)]
    prompt: String,
    #[serde(default = "default_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: f32,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    output_audio: bool,
}

fn default_tokens() -> usize {
    128
}

impl Request {
    fn bytes(&self) -> Result<Vec<u8>, String> {
        if self.max_tokens == 0
            || !self.temperature.is_finite()
            || !(0.0..=2.0).contains(&self.temperature)
        {
            return Err(
                "max_tokens must be positive and temperature must be finite in [0,2]".into(),
            );
        }
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&self.audio)
            .map_err(|e| format!("audio must be raw base64 encoded audio: {e}"))?;
        if bytes.is_empty() {
            return Err("audio must be nonempty".into());
        }
        Ok(bytes)
    }
}

fn error(status: StatusCode, message: impl ToString) -> Response {
    (status, Json(json!({"error": message.to_string()}))).into_response()
}

fn generate(
    model: &Backend,
    request: &GenerationRequest,
    audio: &[Vec<f32>],
    sink: &mut dyn TokenSink,
    output_audio: bool,
    stream: Option<&tokio::sync::mpsc::Sender<Result<Event, axum::Error>>>,
    cancelled: &dyn Fn() -> bool,
) -> Result<Value, String> {
    if cancelled() {
        return Err("audio conversation cancelled".into());
    }
    struct CancellableSink<'a> {
        sink: &'a mut dyn TokenSink,
        cancelled: &'a dyn Fn() -> bool,
    }
    impl TokenSink for CancellableSink<'_> {
        fn push_text(&mut self, text: &str) -> Flow {
            if (self.cancelled)() {
                Flow::Stop
            } else {
                self.sink.push_text(text)
            }
        }
    }
    let runtime = model
        .text_backend()
        .and_then(|text| text.runtime.as_ref())
        .ok_or("server has no audio conversation runtime")?;
    let result = runtime.lock().map_err(|e| e.to_string())?.generate_audio(
        request,
        audio,
        &mut CancellableSink { sink, cancelled },
    )?;
    if cancelled() {
        return Err("audio conversation cancelled".into());
    }
    let mut response = json!({"text": result.text, "token_ids": result.token_ids});
    if output_audio {
        let Backend::AudioChat(backend) = model else {
            return Err("audio output requires --tts-model and --tts-mmproj".into());
        };
        let mut encoding_error = None;
        let output = backend.speech.synthesize(
            &result.text,
            stream.is_some(),
            |chunk| {
                let Some(tx) = stream else {
                    return true;
                };
                let wav = match crate::format::wav::encode_wav_pcm16_channels(chunk, 24_000, 1) {
                    Ok(value) => value,
                    Err(e) => {
                        encoding_error = Some(e);
                        return false;
                    }
                };
                tx.blocking_send(Ok(Event::default().event("audio").data(
                    json!({
                        "audio":base64::engine::general_purpose::STANDARD.encode(&wav[44..]),
                        "format":"pcm16","sample_rate":24000,"channels":1,
                    })
                    .to_string(),
                )))
                .is_ok()
            },
            cancelled,
        )?;
        if let Some(error) = encoding_error {
            return Err(error);
        }
        response["speech_tokens"] = json!(output.speech_tokens.len());
        response["samples"] = json!(output.waveform.len());
        if stream.is_none() {
            let wav = crate::format::wav::encode_wav_pcm16_channels(&output.waveform, 24_000, 1)?;
            response["audio"] = json!({"data":base64::engine::general_purpose::STANDARD.encode(wav),"format":"wav","sample_rate":24000,"channels":1});
        }
    }
    Ok(response)
}

struct StreamSink(tokio::sync::mpsc::Sender<Result<Event, axum::Error>>);
impl TokenSink for StreamSink {
    fn push_text(&mut self, text: &str) -> Flow {
        if self
            .0
            .blocking_send(Ok(Event::default()
                .event("delta")
                .data(json!({"text": text}).to_string())))
            .is_ok()
        {
            Flow::Continue
        } else {
            Flow::Stop
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct JevRequest {
    audio: String,
    #[serde(default)]
    context: String,
    questions: Vec<JevQuestion>,
    #[serde(default)]
    positive: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JevQuestion {
    text: String,
    options: Vec<String>,
}

pub(super) fn score_media(
    model: &Backend,
    audio: &[Vec<f32>],
    images: &[Vec<u8>],
    context: &str,
    questions: &[crate::app::jev::JevQuestionInput],
    positive: Option<&str>,
) -> Result<Vec<crate::app::jev::JevResult>, (StatusCode, String)> {
    let invalid = |e: String| (StatusCode::BAD_REQUEST, e);
    let internal = |e: String| (StatusCode::INTERNAL_SERVER_ERROR, e);
    let text = model
        .text_backend()
        .ok_or_else(|| internal("server has no text runtime".into()))?;
    let template = text
        .chat_template
        .resolve(&|key| text.source.metadata(key).cloned())
        .map_err(internal)?
        .ok_or_else(|| internal("candidate scoring requires a chat template".into()))?;
    let mut runtime = text
        .runtime
        .as_ref()
        .ok_or_else(|| internal("server has no text runtime".into()))?
        .lock()
        .map_err(|e| internal(e.to_string()))?;
    crate::app::audio_chat::validate_score(
        runtime.as_ref(),
        &text.tokenizer,
        &template,
        audio,
        images,
        context,
        questions,
        positive,
    )
    .map_err(invalid)?;
    crate::app::audio_chat::score(
        runtime.as_mut(),
        &text.tokenizer,
        &template,
        audio,
        images,
        context,
        questions,
        positive,
    )
    .map_err(internal)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn score_grouped_media(
    model: &Backend,
    audio: &[Vec<f32>],
    images: &[Vec<u8>],
    context: &str,
    questions: &[crate::app::jev::JevGroupedQuestionInput],
    mode: crate::app::jev::JevMode,
) -> Result<Vec<crate::app::jev::types::JevGroupedResult>, (StatusCode, String)> {
    let invalid = |e: String| (StatusCode::BAD_REQUEST, e);
    let internal = |e: String| (StatusCode::INTERNAL_SERVER_ERROR, e);
    let text = model
        .text_backend()
        .ok_or_else(|| internal("server has no text runtime".into()))?;
    let template = text
        .compiled_chat_template
        .as_ref()
        .as_ref()
        .ok_or_else(|| internal("candidate scoring requires a chat template".into()))?;
    let mut runtime = text
        .runtime
        .as_ref()
        .ok_or_else(|| internal("server has no text runtime".into()))?
        .lock()
        .map_err(|e| internal(e.to_string()))?;
    crate::app::audio_chat::validate_grouped_score(
        runtime.as_ref(),
        &text.tokenizer,
        template,
        audio,
        images,
        context,
        questions,
        mode,
    )
    .map_err(invalid)?;
    crate::app::audio_chat::score_grouped(
        text.source.clone(),
        runtime.as_mut(),
        &text.tokenizer,
        template,
        audio,
        images,
        context,
        questions,
        mode,
    )
    .map_err(internal)
}

pub(super) async fn jev(
    State(state): State<AppState>,
    body: Result<Json<JevRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.body_text()),
    };
    let permit = match state.generation_slot.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, e),
    };
    let result = super::compute_task(state.compute, move || -> Result<_, (StatusCode, String)> {
        fn invalid(message: impl ToString) -> (StatusCode, String) {
            (StatusCode::BAD_REQUEST, message.to_string())
        }
        let _permit = permit;
        let text = state
            .model
            .text_backend()
            .ok_or_else(|| invalid("server has no text runtime"))?;
        let questions = request
            .questions
            .into_iter()
            .map(|q| crate::app::jev::JevQuestionInput {
                text: q.text,
                options: q.options,
            })
            .collect::<Vec<_>>();
        crate::app::jev::prepare_jev_questions(&questions, request.positive.as_deref())
            .map_err(invalid)?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&request.audio)
            .map_err(|e| invalid(format!("audio must be raw base64 encoded audio: {e}")))?;
        let max_samples = text.context_length.saturating_mul(2560).min(16_000 * 600);
        let audio = vec![
            crate::app::media::decode_audio_bytes_bounded(&bytes, max_samples).map_err(invalid)?,
        ];
        let results = score_media(
            state.model.as_ref(),
            &audio,
            &[],
            &request.context,
            &questions,
            request.positive.as_deref(),
        )?;
        Ok(json!({"mode":"single","context":request.context,"results":results}))
    })
    .await;
    match result {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err((status, e))) => error(status, e),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

pub(super) async fn handle(
    State(state): State<AppState>,
    body: Result<Json<Request>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(value)) => value,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.body_text()),
    };
    let stream = request.stream;
    let output_audio = request.output_audio;
    if output_audio && !matches!(state.model.as_ref(), Backend::AudioChat(_)) {
        return error(
            StatusCode::BAD_REQUEST,
            "audio output requires --tts-model and --tts-mmproj",
        );
    }
    let permit = match state.generation_slot.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, e),
    };
    let model = Arc::clone(&state.model);
    let prepared = super::compute_task(state.compute, move || -> Result<_, String> {
        let text = model
            .text_backend()
            .ok_or("server is not running a text model")?;
        let runtime = text.runtime.as_ref().ok_or("server has no text runtime")?;
        let generation = crate::app::audio_chat::request(
            text.source.as_ref(),
            &text.tokenizer,
            &text.chat_template,
            &request.prompt,
            1,
            request.max_tokens,
            request.temperature,
        )?;
        let available = text
            .context_length
            .checked_sub(generation.token_ids.len().saturating_sub(1))
            .and_then(|n| n.checked_sub(request.max_tokens))
            .filter(|&n| n > 0)
            .ok_or("audio prompt plus output budget exceeds context")?;
        // Each 2560 waveform samples consumes at least one audio token.
        let max_samples = available
            .checked_mul(2560)
            .ok_or("audio sample budget overflow")?
            .min(16_000 * 600);
        let audio = vec![crate::app::media::decode_audio_bytes_bounded(
            &request.bytes()?,
            max_samples,
        )?];
        runtime
            .lock()
            .map_err(|e| e.to_string())?
            .validate_audio_request(&generation, &audio)?;
        Ok((generation, audio, permit))
    })
    .await;
    let (request, audio, permit) = match prepared {
        Ok(Ok(value)) => value,
        Ok(Err(e)) => return error(StatusCode::BAD_REQUEST, e),
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    if !stream {
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut cancellation_guard = super::api::CancellationGuard::new(cancelled.clone());
        let result = super::compute_task(state.compute, move || {
            let _permit = permit;
            generate(
                state.model.as_ref(),
                &request,
                &audio,
                &mut CollectSink::new(),
                output_audio,
                None,
                &|| cancelled.load(Ordering::Relaxed),
            )
        })
        .await;
        cancellation_guard.disarm();
        return match result {
            Ok(Ok(value)) => Json(value).into_response(),
            Ok(Err(e)) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
            Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
        };
    }
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    tokio::spawn(async move {
        let result_tx = tx.clone();
        let result = super::compute_task(state.compute, move || {
            let _permit = permit;
            generate(
                state.model.as_ref(),
                &request,
                &audio,
                &mut StreamSink(tx.clone()),
                output_audio,
                Some(&tx),
                &|| tx.is_closed(),
            )
        })
        .await;
        let (event, value) = match result {
            Ok(Ok(value)) => ("done", value),
            Ok(Err(e)) => ("error", json!({"error": e})),
            Err(e) => ("error", json!({"error": e.to_string()})),
        };
        let _ = result_tx
            .send(Ok(Event::default().event(event).data(value.to_string())))
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
    fn validates_budget_and_rejects_file_or_url_inputs() {
        for value in [
            json!({"audio":""}),
            json!({"audio":"file:///tmp/a.wav"}),
            json!({"audio":"https://example.com/a.wav"}),
            json!({"audio":"YQ==", "max_tokens":0}),
            json!({"audio":"YQ==", "temperature":3}),
        ] {
            assert!(serde_json::from_value::<Request>(value)
                .unwrap()
                .bytes()
                .is_err());
        }
        assert_eq!(
            serde_json::from_value::<Request>(json!({"audio":"YQ=="}))
                .unwrap()
                .bytes()
                .unwrap(),
            b"a"
        );
        assert!(
            serde_json::from_value::<Request>(json!({"audio":"YQ==", "file":"/tmp/a.wav"}))
                .is_err()
        );
    }
}
