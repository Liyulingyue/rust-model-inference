pub mod api;
mod rerank;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::{header, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

use crate::app::cli::{
    normalize_tts_language, parse_cli_options, validate_cli_options, CliOptions, KvFormat,
};
use crate::app::{compute_embedding, open_or_exit};
use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::ComponentRole;
use crate::format::wav::encode_wav_pcm16_channels;
use crate::models::audio8::streaming as audio8_streaming;
use crate::models::qwen3::asr::model::{
    open_bundled_audio_source, AsrRuntime, TranscriptionOptions,
};
use crate::models::qwen3::tts::codec::{Code2WavDecoder, CodePredictor, WAVEFORM_SAMPLE_RATE};
use crate::models::qwen3::tts::speaker::{reference_wav_to_mel, Qwen3TtsSpeakerEncoder};
use crate::models::qwen3::tts::{predictor_top_k, Qwen3TtsTalker, TtsPrompt, TtsSession};
use crate::models::qwen3::Qwen3Model;

const USAGE: &str = "Usage: rust-model-server --model <path.gguf-or-ggufrs> [--mmproj ...] [--audio ...] [--image ...] [--tts] [--embedding] [--host 0.0.0.0] [--port 8080] [--threads 4] [--prefill-batch-size N (default 64)] [--allow-remote-images]";

// =============================================================================
// Backend types
// =============================================================================

#[derive(Clone)]
struct AppState {
    compute: crate::compute::ComputePolicy,
    model: Arc<Backend>,
    model_name: String,
    responses: Arc<Mutex<api::ResponsesStore>>,
    generation_slot: Arc<tokio::sync::Semaphore>,
}

fn compute_task<T: Send + 'static>(
    policy: crate::compute::ComputePolicy,
    work: impl FnOnce() -> T + Send + 'static,
) -> tokio::task::JoinHandle<T> {
    tokio::task::spawn_blocking(move || {
        let _scope = policy.enter_legacy_scope();
        work()
    })
}

enum Backend {
    Text(TextBackend),
    Embedding(EmbeddingBackend),
    Asr(AsrBackend),
    Audio8(Audio8Backend),
    Tts(TtsBackend),
    Rerank(RerankBackend),
    Clm(ClmBackend),
    Gliner2(Gliner2Backend),
    Gliner2Boundary(Gliner2BoundaryBackend),
}

/// GLiNER2.5-Decide backend: a DeBERTa-v3 encoder with the classifier head in
/// the same GGUF, named by `--model` plus `--gliner2-decide`.  It scores a
/// caller-supplied label set, so it exposes the JEV score route and nothing
/// else.  The tokenizer is built once here; the model itself is
/// zero-copy views over the mapping and is cheap to rebuild per request.
struct Gliner2Backend {
    source: Box<dyn TensorSource>,
    tokenizer: crate::models::gliner::ModelTokenizer,
    n_threads: usize,
}

/// GLiNER2.5 BoundaryExtractor backend: the multi-task head (document pool,
/// pair scorer, classifier, abstention) in one DeBERTa-v3 GGUF, named by
/// `--model` plus `--gliner2-boundary`.
///
/// The mapping is owned here and the model is rebuilt per request from a
/// borrow of it, the same way the Decide backend does: `BoundaryModel` is
/// zero-copy views over the mapping plus a settings parse, so there is nothing
/// to cache, and no `'static` leak is needed.
struct Gliner2BoundaryBackend {
    source: Box<dyn TensorSource>,
    n_threads: usize,
}

/// CLM backend: a Qwen3 encoder plus the projection-head GGUF named by
/// `--clm-head`.  Scores candidates by cosine in projection space, so it
/// exposes the JEV score route and nothing else.
struct ClmBackend {
    /// Kept so the JEV plumbing can hand out an `Arc<dyn TensorSource>`
    /// the same way it does for `TextBackend`.
    source: Arc<dyn TensorSource>,
    /// Leaked for the same reason as `RerankBackend::model`:
    /// `Qwen3Session` wants a `&'static` model.
    model: Arc<&'static Qwen3Model>,
    /// Owned, no borrow: the loader copies the weights out.
    heads: crate::models::clm::ClmHeads,
    tokenizer: Arc<BPETokenizer>,
    prefill_batch_size: usize,
    context_length: usize,
}

/// The two cross-encoder rerank families this server understands.
///
/// `Qwen3` reuses the qwen3 trunk (causal prefill + last-token
/// classification head). `JinaBertV2` reuses the bert-encoder forward loop
/// (bidirectional CLS token + `cls.weight` linear projection), which is
/// structurally different: no causal mask, no Qwen3Session.
enum RerankKind {
    Qwen3,
    JinaBertV2,
}

struct RerankBackend {
    /// Qwen3 model loaded with the optional `cls.output.weight` rerank
    /// head. Leaked to `'static` so per-request `Qwen3Session`s can
    /// borrow from it without re-loading. `None` for `JinaBertV2`.
    model: Option<Arc<&'static Qwen3Model>>,
    tokenizer: Arc<BPETokenizer>,
    prefill_batch_size: usize,
    context_length: usize,
    /// jina-bert-v2 backend keeps the raw `TensorSource` and reuses the
    /// shared `bert_family::compute_rerank_score` for scoring. `None` for
    /// `Qwen3` (the qwen3 path doesn't need it).
    jina_source: Option<Arc<dyn TensorSource>>,
    /// Selects the per-doc scoring path in `rerank::score_one_doc`.
    kind: RerankKind,
}

unsafe impl Send for Backend {}
unsafe impl Sync for Backend {}

struct TextBackend {
    arch: String,
    pool: Arc<ComputePool>,
    tokenizer: Arc<BPETokenizer>,
    prefill_batch_size: usize,
    context_length: usize,
    /// Per-arch generation adapter. `None` = arch has no adapter yet; those
    /// models still load but `/v1/chat/completions` answers 501.
    runtime: Option<crate::ops::TextRuntimeHandle>,
    /// `TensorSource` for the loaded LLM. Used by the JEV scoring
    /// endpoints (`/v1/jev/score`, `/v1/jev/grouped`) so they can
    /// run the existing CLI JEV pipeline without re-loading the model
    /// from disk. The pool and tokenizer above are still derived from
    /// this source at startup; this is a duplicate handle, not a
    /// second copy of the model tensors.
    pub(crate) source: Arc<dyn TensorSource>,
    /// On-disk path of the LLM GGUF (kept for the multimodal path
    /// which takes `&Path` for the model).
    pub(crate) model_path: Option<std::path::PathBuf>,
    /// Optional multimodal vision encoder (CLIP / Qwen2.5-Omni /
    /// etc.). Required for `/v1/jev/image` / `/v1/jev/image_grouped`.
    pub(crate) mmproj: Option<Arc<dyn TensorSource>>,
    /// On-disk path of the mmproj blob (kept alongside the in-memory
    /// source because the existing multimodal CLI helper expects
    /// `&Path` for both the model and the vision encoder).
    pub(crate) mmproj_path: Option<std::path::PathBuf>,
}

unsafe impl Send for TextBackend {}
unsafe impl Sync for TextBackend {}

struct EmbeddingBackend {
    source: Arc<dyn TensorSource>,
    mmproj_path: Option<PathBuf>,
}

struct AsrBackend {
    runtime: Arc<AsrRuntime>,
}

/// Audio8 ASR backend: a single-model Voxtral-Realtime GGUF that bundles the
/// audio tower and Qwen2 text decoder. Same wiring as `RerankBackend` —
/// encoder/decoder are leaked to `'static` so per-request
/// `StreamingTranscriber`s can borrow from them for the server lifetime.
struct Audio8Backend {
    encoder: Arc<&'static crate::models::audio8::Audio8Encoder>,
    decoder: Arc<&'static crate::models::audio8::text::Audio8TextDecoder>,
    tokenizer: Arc<BPETokenizer>,
    specials: crate::models::audio8::streaming::SpecialTokens,
    language: &'static str,
    max_tokens: usize,
}

struct TtsBackend {
    talker: Arc<Qwen3TtsTalker>,
    mmproj: Arc<dyn TensorSource>,
    language: &'static str,
    temperature: f32,
    max_tokens: usize,
}

// =============================================================================
// HTTP request/response shapes (subset of OpenAI)
// =============================================================================

#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Deserialize)]
struct EmbeddingRequest {
    #[serde(default)]
    model: Option<String>,
    /// Text inputs for the text-only embedding path. Optional when
    /// `audio`/`image`/`video` is present (multimodal path).
    #[serde(default)]
    input: Option<serde_json::Value>,
    #[serde(default)]
    encoding_format: Option<String>,
    /// Optional base64-encoded audio bytes for multimodal embedding
    /// (jina-v5-omni / qwen3-omni with audio mmproj). When present,
    /// the server routes through `run_omni_embedding` instead of the
    /// text-only path. The byte payload may be raw audio (decoded by
    /// `ffmpeg` via the same WAV/MP3/FLAC/OGG/M4A/OPUS path as the
    /// CLI `--audio` flag).
    #[serde(default)]
    audio: Option<String>,
    /// Same as `Optional base64-encoded image for multimodal
    /// embedding. Accepts raw image bytes (jpg/png/webp/...).
    #[serde(default)]
    image: Option<String>,
    /// Optional base64-encoded video bytes. Decoded by ffprobe/ffmpeg.
    #[serde(default)]
    video: Option<String>,
    /// Optional prompt string used as the LLM-side system turn for
    /// multimodal embedding (e.g. "Query: ..." vs "Respond this
    /// document ..."). Defaults to "" when omitted.
    #[serde(default)]
    prompt: Option<String>,
}

#[derive(Serialize)]
struct EmbeddingResponse {
    object: String,
    data: Vec<EmbeddingObject>,
    model: String,
    usage: EmbeddingUsage,
}

#[derive(Serialize)]
struct EmbeddingObject {
    object: String,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Serialize)]
struct EmbeddingUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Deserialize)]
struct TranscriptionRequest {
    #[serde(default)]
    model: Option<String>,
    /// Base64-encoded WAV bytes (for the JSON endpoint).
    #[serde(default)]
    input: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct TranscriptionResponse {
    text: String,
}

#[derive(Deserialize)]
struct SpeechRequest {
    model: String,
    input: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// =============================================================================
// Helpers
// =============================================================================

// =============================================================================
// Routes
// =============================================================================

async fn health() -> &'static str {
    "ok"
}

async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "local".to_string(),
        }],
    })
}

async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> impl IntoResponse {
    let Backend::Embedding(backend) = state.model.as_ref() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Server is not running an embedding model".into(),
            }),
        )
            .into_response();
    };
    // Multimodal path: any of audio/image/video routes the request
    // through `run_omni_embedding`, which needs mmproj + a temp file
    // path for the media blob. mmproj is required here (matching the
    // CLI `--mmproj` requirement).
    let has_media = req.audio.is_some() || req.image.is_some() || req.video.is_some();
    if has_media {
        let Some(mmproj_path) = backend.mmproj_path.clone() else {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Multimodal embedding requires --mmproj on server startup".into(),
                }),
            )
                .into_response();
        };
        let media_count = usize::from(req.audio.is_some())
            + usize::from(req.image.is_some())
            + usize::from(req.video.is_some());
        if media_count != 1 {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Exactly one of audio/image/video may be set per request".into(),
                }),
            )
                .into_response();
        }
        let media_b64 = req
            .audio
            .as_ref()
            .or(req.image.as_ref())
            .or(req.video.as_ref())
            .cloned()
            .unwrap();
        let ext = if req.audio.is_some() {
            "wav"
        } else if req.image.is_some() {
            "jpg"
        } else {
            "mp4"
        };
        let prompt = req.prompt.unwrap_or_default();
        let source = backend.source.clone();
        let result = match compute_task(state.compute, move || {
            let temp_path = std::env::temp_dir().join(format!(
                "rmi-media-{}.{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0),
                ext
            ));
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(media_b64.trim())
                .map_err(|error| format!("Invalid base64 media payload: {error}"))?;
            std::fs::write(&temp_path, &bytes)
                .map_err(|error| format!("Failed to write media temp file: {error}"))?;
            let image_path = if req.image.is_some() {
                Some(temp_path.as_path())
            } else {
                None
            };
            let video_path = if req.video.is_some() {
                Some(temp_path.as_path())
            } else {
                None
            };
            let audio_path = if req.audio.is_some() {
                Some(temp_path.as_path())
            } else {
                None
            };
            let result = crate::app::run_omni_embedding(
                source.as_ref(),
                &mmproj_path,
                image_path,
                video_path,
                audio_path,
                &prompt,
                0,
            );
            let _ = std::fs::remove_file(&temp_path);
            let embedding = result?;
            Ok::<_, String>(EmbeddingObject {
                object: "embedding".to_string(),
                embedding,
                index: 0,
            })
        })
        .await
        {
            Ok(Ok(object)) => vec![object],
            Ok(Err(error)) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse { error }),
                )
                    .into_response();
            }
            Err(error) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("embedding worker failed: {error}"),
                    }),
                )
                    .into_response();
            }
        };
        let total_tokens: usize = result.len();
        let response = EmbeddingResponse {
            object: "list".to_string(),
            data: result,
            model: state.model_name.clone(),
            usage: EmbeddingUsage {
                prompt_tokens: total_tokens,
                total_tokens,
            },
        };
        return (StatusCode::OK, Json(response)).into_response();
    }
    let inputs: Result<Vec<String>, String> = match req.input {
        Some(serde_json::Value::String(text)) => Ok(vec![text]),
        Some(serde_json::Value::Array(items)) => items
            .into_iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| "embedding input must be string or array of strings".to_string())
            })
            .collect(),
        Some(_) => Err("embedding input must be string or array of strings".into()),
        None => Err("embedding input is required when audio/image/video are not set".into()),
    };
    let inputs = match inputs {
        Ok(value) => value,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response();
        }
    };
    let source = backend.source.clone();
    let result = match compute_task(state.compute, move || {
        let mut data = Vec::with_capacity(inputs.len());
        for (index, prompt) in inputs.iter().enumerate() {
            let embedding = compute_embedding(source.as_ref(), prompt, 0)?;
            data.push(EmbeddingObject {
                object: "embedding".to_string(),
                embedding,
                index,
            });
        }
        Ok::<_, String>(data)
    })
    .await
    {
        Ok(Ok(data)) => data,
        Ok(Err(error)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error }),
            )
                .into_response();
        }
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("embedding worker failed: {error}"),
                }),
            )
                .into_response();
        }
    };
    let total_tokens: usize = result.len();
    let response = EmbeddingResponse {
        object: "list".to_string(),
        data: result,
        model: state.model_name.clone(),
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };
    (StatusCode::OK, Json(response)).into_response()
}

async fn transcriptions(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut language: Option<String> = None;
    let mut prompt: Option<String> = None;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(value)) => value,
            Ok(None) => break,
            Err(error) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!("multipart error: {error}"),
                    }),
                )
                    .into_response();
            }
        };
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => match field.bytes().await {
                Ok(bytes) => file_bytes = Some(bytes.to_vec()),
                Err(error) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!("failed to read upload: {error}"),
                        }),
                    )
                        .into_response();
                }
            },
            "language" => match field.text().await {
                Ok(text) => language = Some(text),
                Err(error) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!("multipart error: {error}"),
                        }),
                    )
                        .into_response();
                }
            },
            "prompt" => match field.text().await {
                Ok(text) => prompt = Some(text),
                Err(error) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!("multipart error: {error}"),
                        }),
                    )
                        .into_response();
                }
            },
            _ => {
                let _ = field.bytes().await;
            }
        }
    }
    let wav = match file_bytes {
        Some(bytes) => bytes,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "missing 'file' field".into(),
                }),
            )
                .into_response();
        }
    };
    let _ = prompt;
    do_transcribe(&state, wav, language).await
}

async fn transcriptions_json(
    State(state): State<AppState>,
    Json(req): Json<TranscriptionRequest>,
) -> impl IntoResponse {
    let input = match req.input.as_deref() {
        Some(value) => value,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "JSON transcription requires 'input' (base64 WAV)".into(),
                }),
            )
                .into_response();
        }
    };
    let wav = match base64::engine::general_purpose::STANDARD.decode(input.trim()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("base64 decode failed: {error}"),
                }),
            )
                .into_response();
        }
    };
    do_transcribe(&state, wav, req.language).await
}

/// Shared transcription worker: branches on backend type so a single
/// call site covers Qwen3-VL ASR (`AsrRuntime`) and Voxtral Realtime
/// (`StreamingTranscriber`). Audio8 ignores `prompt`; the Qwen3 path
/// accepts it as the decoder prompt context.
async fn do_transcribe(
    state: &AppState,
    wav: Vec<u8>,
    language: Option<String>,
) -> axum::response::Response {
    match state.model.as_ref() {
        Backend::Asr(backend) => {
            let options = TranscriptionOptions {
                language: language.clone(),
                prompt: None,
                max_new_tokens: 256,
            };
            let runtime = backend.runtime.clone();
            let wav = wav;
            let result = compute_task(state.compute, move || {
                runtime.transcribe_wav(&wav, &options)
            })
            .await;
            match result {
                Ok(Ok(value)) => (
                    StatusCode::OK,
                    Json(TranscriptionResponse { text: value.text }),
                )
                    .into_response(),
                Ok(Err(error)) => (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(ErrorResponse {
                        error: error.to_string(),
                    }),
                )
                    .into_response(),
                Err(error) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("asr worker failed: {error}"),
                    }),
                )
                    .into_response(),
            }
        }
        Backend::Audio8(backend) => {
            let encoder = backend.encoder.clone();
            let decoder = backend.decoder.clone();
            let specials = backend.specials.clone();
            let tokenizer = backend.tokenizer.clone();
            let max_tokens = backend.max_tokens;
            let default_language = backend.language.to_string();
            let language = language.unwrap_or(default_language);
            let result = compute_task(state.compute, move || {
                let samples = audio8_streaming::decode_samples(&wav)?;
                let stream = audio8_streaming::Schedule::new().padded_stream(&samples);
                let language_token = specials.language(&language)?;
                let mut transcriber = audio8_streaming::StreamingTranscriber::new(
                    &*encoder,
                    &*decoder,
                    stream,
                    specials,
                    language_token,
                    max_tokens,
                )?;
                while transcriber.next_chunk()?.is_some() {}
                let visible = transcriber.finish();
                Ok::<_, String>(tokenizer.decode(&visible, false))
            })
            .await;
            match result {
                Ok(Ok(text)) => {
                    (StatusCode::OK, Json(TranscriptionResponse { text })).into_response()
                }
                Ok(Err(error)) => (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(ErrorResponse { error }),
                )
                    .into_response(),
                Err(error) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("audio8 worker failed: {error}"),
                    }),
                )
                    .into_response(),
            }
        }
        _ => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Server is not running an ASR model".into(),
            }),
        )
            .into_response(),
    }
}

async fn speech(
    State(state): State<AppState>,
    Json(req): Json<SpeechRequest>,
) -> impl IntoResponse {
    let Backend::Tts(backend) = state.model.as_ref() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Server is not running a TTS model".into(),
            }),
        )
            .into_response();
    };
    let ref_wav_bytes: Option<Vec<u8>> = match req.voice.as_deref() {
        None | Some("") => None,
        Some(voice) => {
            if let Some(path) = voice.strip_prefix("file://") {
                match std::fs::read(path) {
                    Ok(bytes) => Some(bytes),
                    Err(error) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(ErrorResponse {
                                error: format!("failed to read reference voice: {error}"),
                            }),
                        )
                            .into_response();
                    }
                }
            } else if voice.starts_with("data:") {
                let payload = voice.splitn(2, ',').nth(1).unwrap_or("");
                match base64::engine::general_purpose::STANDARD.decode(payload) {
                    Ok(bytes) => Some(bytes),
                    Err(error) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(ErrorResponse {
                                error: format!("voice base64 decode failed: {error}"),
                            }),
                        )
                            .into_response();
                    }
                }
            } else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: "voice must be file://path or data:audio/wav;base64,...".into(),
                    }),
                )
                    .into_response();
            }
        }
    };
    let talker = backend.talker.clone();
    let mmproj = backend.mmproj.clone();
    let language = backend.language;
    let temperature = backend.temperature;
    let max_tokens = backend.max_tokens;
    let input = req.input.clone();
    let wav_bytes = match compute_task(state.compute, move || -> Result<Vec<u8>, String> {
        let speaker = if let Some(wav) = ref_wav_bytes.as_deref() {
            let mel = reference_wav_to_mel(wav)?;
            Some(Qwen3TtsSpeakerEncoder::from_source(mmproj.as_ref())?.encode(&mel)?)
        } else {
            None
        };
        let prompt = talker.prepare_prompt(&input, language, speaker.as_deref())?;
        let predictor = CodePredictor::from_source(mmproj.as_ref())?;
        let decoder = Code2WavDecoder::from_source(mmproj.as_ref())?;
        let frames = synthesize_tts_frames(&talker, &predictor, &prompt, max_tokens, temperature)?;
        let waveform = decoder.decode(&frames)?;
        encode_wav_pcm16_channels(&waveform, WAVEFORM_SAMPLE_RATE, 1)
    })
    .await
    {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(error)) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(ErrorResponse { error }),
            )
                .into_response();
        }
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("tts worker failed: {error}"),
                }),
            )
                .into_response();
        }
    };
    let content_type = match req.response_format.as_deref() {
        Some("pcm") => "audio/pcm",
        _ => "audio/wav",
    };
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, content_type)],
        wav_bytes,
    )
        .into_response()
}

fn synthesize_tts_frames(
    talker: &Qwen3TtsTalker,
    predictor: &CodePredictor,
    prompt: &TtsPrompt,
    max_frames: usize,
    temperature: f32,
) -> Result<Vec<[u32; 16]>, String> {
    let mut session = TtsSession::new(talker)?;
    session.prefill_prompt(prompt)?;
    let mut frames = Vec::with_capacity(max_frames);
    let mut rng = rand::thread_rng();
    let mut next_semantic = session.sample_semantic(temperature, &mut rng)?;
    for frame_index in 0..max_frames {
        let Some(semantic) = next_semantic else {
            break;
        };
        let hidden = session.hidden_state().to_vec();
        let (frame, mut feedback) =
            predictor.predict_frame(&hidden, semantic, predictor_top_k(temperature), &mut rng)?;
        let overlay = &prompt.overlay[frame_index.min(prompt.overlay.len() - 1)];
        if feedback.len() != overlay.len() {
            return Err(format!(
                "TTS feedback length {} != overlay length {}",
                feedback.len(),
                overlay.len()
            ));
        }
        for (value, text) in feedback.iter_mut().zip(overlay) {
            *value += *text;
        }
        let position = prompt
            .positions
            .len()
            .checked_add(frame_index)
            .ok_or_else(|| "TTS frame position overflow".to_string())?;
        session.forward_step_with_embedding(&feedback, [position; 4])?;
        next_semantic = session.sample_semantic(temperature, &mut rng)?;
        frames.push(frame);
    }
    Ok(frames)
}

// =============================================================================
// Text generation
// =============================================================================

// =============================================================================
// Backend construction
// =============================================================================

fn build_backend(options: &CliOptions) -> Result<Arc<Backend>, String> {
    options.validate_compute_mode()?;
    options.compute_policy().check_build()?;
    let _scope = options.compute_policy().enter_legacy_scope();
    if options.tts {
        return Ok(Arc::new(Backend::Tts(build_tts(options)?)));
    }
    if options.audio.is_some() {
        // ASR disambiguation: Audio8 ships the audio tower in the LLM GGUF
        // (no `--mmproj`), so the right backend depends on the file's
        // `general.architecture`. Probed here so a single `--audio` flag
        // covers both Qwen3-VL ASR and Voxtral Realtime. See
        // docs/develop/SERVER_BACKEND_SELECTION.md for why this is a probe
        // rather than a flag.
        if is_audio8_gguf(&options.model) {
            return Ok(Arc::new(Backend::Audio8(build_audio8(options)?)));
        }
        return Ok(Arc::new(Backend::Asr(build_asr(options)?)));
    }
    if options.embedding {
        let source = open_or_exit(&options.model, ComponentRole::Llm);
        return Ok(Arc::new(Backend::Embedding(EmbeddingBackend {
            source: Arc::from(source),
            mmproj_path: options.mmproj.clone(),
        })));
    }
    // Cross-encoder rerank detection: a GGUF that carries
    // `pooling_type = 4` and a `cls.output.weight` is a Qwen3-style
    // rerank model; a `jina-bert-v2` arch with `cls.weight` + `cls.bias`
    // is the jina-bert-v2-style rerank model. Both are detected by
    // metadata peek BEFORE the full Text build (which would load
    // unrelated multimodal state).
    //
    // This is the only metadata-probed backend in this function; the others are
    // flag-driven. Why they differ — and why adding a probe is usually the wrong
    // fix — is in docs/develop/SERVER_BACKEND_SELECTION.md.
    if is_rerank_gguf(&options.model) || is_jina_rerank_gguf(&options.model) {
        // The probe outranks the flag-driven backends below, so a caller
        // who explicitly selected one of them would have it silently
        // dropped (their flag ignored, the rerank backend served).
        // Error out instead: the model file and the flag disagree, and
        // neither resolution is obviously what the caller wanted.
        if let Some(error) = rerank_probe_flag_conflict(options) {
            return Err(error);
        }
        return Ok(Arc::new(Backend::Rerank(build_rerank(options)?)));
    }
    // CLM is opted into rather than detected: the encoder is an ordinary
    // Qwen3 GGUF, and it is `--clm-head` saying "score with these heads"
    // that makes it a CLM backend.
    if options.clm_head.is_some() {
        return Ok(Arc::new(Backend::Clm(build_clm(options)?)));
    }
    if options.gliner2_decide {
        return Ok(Arc::new(Backend::Gliner2(build_gliner2(options)?)));
    }
    if options.gliner2_boundary {
        return Ok(Arc::new(Backend::Gliner2Boundary(build_gliner2_boundary(
            options,
        )?)));
    }
    Ok(Arc::new(Backend::Text(build_text(options)?)))
}

/// Returns `true` if the GGUF at `path` looks like a Qwen3 rerank
/// model: `general.architecture = "qwen3"` AND
/// `<arch>.pooling_type = 4` AND a `cls.output.weight` tensor is
/// present. Done as a quick metadata probe without holding the file
/// open.
/// When the rerank metadata probe fires, the caller may still have
/// explicitly selected a different backend with a flag. The probe
/// outranks those backends in `build_backend`, so honoring it would
/// silently drop the flag. Returns the error message to surface
/// instead, or `None` when the flags are compatible with rerank.
///
/// Pure function of `CliOptions` so the dispatch table can be unit
/// tested without a GGUF on disk. See
/// `docs/develop/SERVER_BACKEND_SELECTION.md`.
fn rerank_probe_flag_conflict(options: &CliOptions) -> Option<String> {
    let model = options.model.display();
    let reranker =
        format!("{model} looks like a Qwen3 reranker (pooling_type=4 + cls.output.weight)");
    if let Some(head) = &options.clm_head {
        return Some(format!(
            "--clm-head selects the CLM backend, but {reranker}; the probe wins and {} \
             would be dropped. Pass a plain Qwen3 encoder as --model instead",
            head.display()
        ));
    }
    if options.gliner2_decide {
        return Some(format!(
            "--gliner2-decide selects the GLiNER2 backend, but {reranker}; the probe wins \
             and the flag would be dropped. GLiNER2 needs a DeBERTa GGUF as --model"
        ));
    }
    if options.gliner2_boundary {
        return Some(format!(
            "--gliner2-boundary selects the GLiNER2 boundary backend, but {reranker}; the \
             probe wins and the flag would be dropped. It needs a boundary-variant \
             DeBERTa GGUF as --model"
        ));
    }
    if let Some(mmproj) = &options.mmproj {
        return Some(format!(
            "--mmproj selects multimodal chat, but {reranker}; the probe wins and {} would \
             be dropped. Pass a chat-capable Qwen3 GGUF as --model instead",
            mmproj.display()
        ));
    }
    None
}

fn is_rerank_gguf(path: &std::path::Path) -> bool {
    use crate::MetaValue;
    let loader = match crate::GGUFLoader::from_file(path) {
        Ok(l) => l,
        Err(_) => return false,
    };
    let arch = loader
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .unwrap_or_default();
    if arch != "qwen3" {
        return false;
    }
    let pooling = loader
        .metadata("qwen3.pooling_type")
        .and_then(MetaValue::to_u64)
        .unwrap_or(0);
    if pooling != 4 {
        return false;
    }
    loader.tensor_info("cls.output.weight").is_some()
}

/// Returns `true` when the GGUF at `path` looks like a jina-bert-v2
/// reranker: `arch = jina-bert-v2` AND the `cls.weight` + `cls.bias`
/// classification tensors are present. Parallel probe to
/// [`is_rerank_gguf`] — used by the same dispatcher to choose
/// `RerankKind::JinaBertV2` over `RerankKind::Qwen3`.
fn is_jina_rerank_gguf(path: &std::path::Path) -> bool {
    use crate::MetaValue;
    let Ok(loader) = crate::GGUFLoader::from_file(path) else {
        return false;
    };
    let arch = loader
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .unwrap_or_default();
    if arch != "jina-bert-v2" {
        return false;
    }
    loader.tensor_info("cls.weight").is_some() && loader.tensor_info("cls.bias").is_some()
}

/// Returns `true` when the GGUF at `name` is an `audio8_asr_infinite`
/// model (Voxtral Realtime bundled with its Qwen2 text decoder). The
/// probe fires only inside the `options.audio.is_some()` branch, so it
/// cannot collide with chat / rerank / GLiNER2 dispatches.
fn is_audio8_gguf(path: &std::path::Path) -> bool {
    use crate::MetaValue;
    let Ok(loader) = crate::GGUFLoader::from_file(path) else {
        return false;
    };
    loader
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .is_some_and(|arch| arch == "audio8_asr_infinite")
}

fn build_rerank(options: &CliOptions) -> Result<RerankBackend, String> {
    if is_jina_rerank_gguf(&options.model) {
        // jina-bert-v2 cross-encoder reranker. No qwen3 trunk: the forward
        // loop lives in `bert_family::compute_rerank_score`, which takes
        // `&dyn TensorSource` directly. The tokenizer here is unused by
        // the scoring path itself (compute_rerank_score builds tokens
        // internally) but we still keep one so future endpoints
        // (e.g. /v1/rerank echoes the input) can decode.
        let source: Arc<dyn TensorSource> =
            Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
        let tokenizer = Arc::new(
            BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
                .map_err(|e| format!("init tokenizer: {e}"))?,
        );
        let context_length: usize = source
            .metadata("jina-bert-v2.context_length")
            .and_then(MetaValue::to_u64)
            .map(|v| v as usize)
            .unwrap_or(8192);
        return Ok(RerankBackend {
            model: None,
            tokenizer,
            prefill_batch_size: 0,
            context_length,
            jina_source: Some(source),
            kind: RerankKind::JinaBertV2,
        });
    }

    // Qwen3-style rerank (pooling_type=4 + cls.output.weight).
    let prefill_batch_size = options.effective_prefill_batch_size()?;
    let source: Arc<dyn TensorSource> = Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|k| {
        source.metadata(k).cloned()
    })?);
    let pool = Arc::new(ComputePool::new(options.threads));
    let model = Qwen3Model::from_source(source, tokenizer.clone(), pool)?;
    if !model.is_rerank() {
        return Err(format!(
            "GGUF looks like qwen3 with pooling_type=4 but has no cls.output.weight — not a rerank model"
        ));
    }
    let context_length = model.config().n_ctx;
    // Pin the model in a `Box::leak` so the `&'static` lifetime bound
    // on `Qwen3Session` is satisfied for the server lifetime.
    let model: &'static Qwen3Model = Box::leak(Box::new(model));
    Ok(RerankBackend {
        model: Some(Arc::new(model)),
        tokenizer,
        prefill_batch_size,
        context_length,
        jina_source: None,
        kind: RerankKind::Qwen3,
    })
}

fn build_clm(options: &CliOptions) -> Result<ClmBackend, String> {
    let prefill_batch_size = options.effective_prefill_batch_size()?;
    let source: Arc<dyn TensorSource> = Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
    let tokenizer = Arc::new(
        BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|e| format!("init tokenizer: {e}"))?,
    );
    let pool = Arc::new(ComputePool::new(options.threads));
    let model = Qwen3Model::from_source(Arc::clone(&source), Arc::clone(&tokenizer), pool)
        .map_err(|e| format!("load encoder: {e}"))?;
    let arch = model.config().architecture.clone();
    if arch != "qwen3" {
        return Err(format!(
            "CLM needs a qwen3 encoder, got {arch:?} (the heads were trained on Qwen3-8B)"
        ));
    }
    // TODO(clm): the heads are encoder-locked, so a quantised base encoder
    // shifts every score -- functionally fine, but not comparable to the
    // paper's numbers.  Warn here instead of refusing, since a low-memory
    // setup may legitimately want a Q4_K_M encoder.
    let context_length = model.config().n_ctx;

    let head_path = options.clm_head.clone().ok_or("--clm-head is required")?;
    let head_source: Box<dyn TensorSource> = open_or_exit(&head_path, ComponentRole::Llm);
    let heads = crate::models::clm::ClmHeads::from_source(head_source.as_ref())
        .map_err(|e| format!("load CLM heads from {}: {e}", head_path.display()))?;
    if model.config().n_embd != heads.encoder_dim() {
        return Err(format!(
            "encoder hidden size {} does not match the CLM heads (expected {})",
            model.config().n_embd,
            heads.encoder_dim()
        ));
    }

    let model: &'static Qwen3Model = Box::leak(Box::new(model));
    Ok(ClmBackend {
        source,
        model: Arc::new(model),
        heads,
        tokenizer,
        prefill_batch_size,
        context_length,
    })
}

fn build_gliner2(options: &CliOptions) -> Result<Gliner2Backend, String> {
    let (source, tokenizer) = crate::app::load_gliner2_source(&options.model)?;
    // Validate the whole contract once at startup rather than per request.
    crate::models::gliner::GlinerModel::from_source_with_tokenizer(
        source.as_ref(),
        tokenizer.clone(),
    )
    .map_err(|e| format!("load GLiNER2 model from {}: {e}", options.model.display()))?;
    Ok(Gliner2Backend {
        source,
        tokenizer,
        n_threads: crate::app::resolve_thread_count(
            options.threads,
            std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(4),
        ),
    })
}

fn build_gliner2_boundary(options: &CliOptions) -> Result<Gliner2BoundaryBackend, String> {
    let source: Box<dyn TensorSource> = crate::format::ggufrs::open_model_source(
        &options.model,
        crate::format::ggufrs::ComponentRole::Llm,
    )
    .map_err(|error| {
        format!(
            "open GLiNER2 boundary model ({}): {error}",
            options.model.display()
        )
    })?;
    if !crate::models::gliner_boundary::is_boundary_gguf(source.as_ref()) {
        return Err(format!(
            "{} is not a gliner2 boundary variant (gliner2.variant = \"boundary\"); \
             --gliner2-decide is the classification variant",
            options.model.display()
        ));
    }
    // Validate the whole contract once at startup rather than per request, the
    // same way `build_gliner2` does.
    crate::models::gliner_boundary::BoundaryModel::from_source(source.as_ref()).map_err(
        |error| {
            format!(
                "load GLiNER2 boundary model from {}: {error}",
                options.model.display()
            )
        },
    )?;
    Ok(Gliner2BoundaryBackend {
        source,
        n_threads: crate::app::resolve_thread_count(
            options.threads,
            std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(4),
        ),
    })
}

fn build_text(options: &CliOptions) -> Result<TextBackend, String> {
    let prefill_batch_size = options.effective_prefill_batch_size()?;
    let raw_source: Arc<dyn TensorSource> =
        Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
    let model_path: std::path::PathBuf = options.model.clone();
    // Snapshot the few metadata fields we need before moving `raw_source`
    // into the (optional) Phi3Source wrapper. Reading them through `&raw_source`
    // creates borrows that would otherwise block the move.
    let arch: String = raw_source
        .metadata("general.architecture")
        .and_then(crate::MetaValue::to_string_val)
        .map(str::to_string)
        .unwrap_or_default();
    let phi3_dims = if arch == "phi3" {
        Some((
            raw_source
                .metadata("phi3.embedding_length")
                .and_then(|v| v.to_u64())
                .ok_or_else(|| "missing phi3.embedding_length".to_string())? as usize,
            raw_source
                .metadata("phi3.attention.head_count")
                .and_then(|v| v.to_u64())
                .ok_or_else(|| "missing phi3.attention.head_count".to_string())?
                as usize,
            raw_source
                .metadata("phi3.attention.head_count_kv")
                .and_then(|v| v.to_u64())
                .ok_or_else(|| "missing phi3.attention.head_count_kv".to_string())?
                as usize,
            raw_source
                .metadata("phi3.feed_forward_length")
                .and_then(|v| v.to_u64())
                .ok_or_else(|| "missing phi3.feed_forward_length".to_string())?
                as usize,
        ))
    } else {
        None
    };
    crate::app::reject_incomplete_z_image_architecture(&arch)?;
    // Phi-3 / Phi-4 ships fused `attn_qkv` and `ffn_up` tensors; the llama
    // trunk reads them as separate `attn_q/k/v` and `ffn_gate/up`. Wrap
    // the source here once so the runtime AND every JEV endpoint
    // (`/v1/jev/score`, `/v1/jev/grouped`) see the per-projection views —
    // mirrors what the CLI does in `app::text::generation`. The wrap must
    // happen BEFORE building the tokenizer, since `BPETokenizer::from_gguf_metadata`
    // captures `&source` for the lifetime of the tokenizer.
    let source: Arc<dyn TensorSource> = if let Some((n_embd, n_head, n_head_kv, n_ff)) = phi3_dims {
        let head_dim = n_embd / n_head;
        let n_embd_q = head_dim * n_head;
        let n_embd_gqa = head_dim * n_head_kv;
        Arc::new(crate::models::phi3::Phi3Source::new(
            raw_source, n_embd_q, n_embd_gqa, n_ff,
        ))
    } else {
        raw_source
    };
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|k| {
        source.metadata(k).cloned()
    })?);
    let pool = Arc::new(ComputePool::new(options.threads));
    // Optional CLIP / Omni vision encoder. Only loaded when the user
    // passed `--mmproj`; the multimodal JEV endpoints
    // (`/v1/jev/image`, `/v1/jev/image_grouped`) refuse to operate
    // without it. Held as `Option<Arc<dyn TensorSource>>` rather than
    // a typed `VisionEncoder` because the encoder shape differs across
    // qwen3vl-merge / Qwen2.5-Omni / gemma4 and the multimodal
    // dispatch in `app::text::multimodal` already does the
    // arch-specific opening for us.
    // Load the vision projector before the runtime options so image input can
    // be wired in (the same handle the multimodal JEV endpoints use).
    let mmp: Option<Arc<dyn TensorSource>> = options
        .mmproj
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty())
        .map(|path| Arc::from(open_or_exit(path, ComponentRole::Mmproj)));
    let mmproj = mmp.clone();
    // One dispatch point for every arch (CLI/HTTP unification,
    // docs/develop/TEXT_RUNTIME_UNIFICATION.md). Returns `None` for archs
    // without an adapter — those models load but answer 501 at request time.
    // Canonical constructor + explicit overrides. Defaults (batch 64, KV F16,
    // context 8K) come from `RuntimeOptions::defaults`, which a unit test pins
    // to the CLI's own resolution — so this line cannot drift again.
    let runtime_options = crate::app::text::RuntimeOptions::from_model(
        source.clone(),
        pool.clone(),
        tokenizer.clone(),
    )
    .with_compute(options.compute_policy())
    .with_kv_format(options.kv_format)
    .with_threads(options.threads)
    .with_max_context(options.effective_max_context())
    .with_prefill_batch_size(prefill_batch_size)
    .with_mmproj(mmp);
    let (runtime, context_length) =
        match crate::app::text::build_text_runtime(&arch, runtime_options) {
            Ok(runtime) => {
                let context = runtime.context_length();
                (Some(std::sync::Mutex::new(runtime)), context)
            }
            Err(error) if options.compute_policy() == crate::compute::ComputePolicy::Vulkan => {
                return Err(error)
            }
            Err(error) => {
                log::info!("text runtime unavailable: {error}");
                (None, 0)
            }
        };
    Ok(TextBackend {
        arch: arch.to_string(),
        pool,
        tokenizer,
        prefill_batch_size,
        context_length,
        runtime,
        source,
        model_path: Some(model_path),
        mmproj,
        mmproj_path: options.mmproj.clone().filter(|p| !p.as_os_str().is_empty()),
    })
}

fn build_asr(options: &CliOptions) -> Result<AsrBackend, String> {
    let prefill_batch_size = options.effective_prefill_batch_size()?;
    let llm_source: Arc<dyn TensorSource> =
        Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|k| {
        llm_source.metadata(k).cloned()
    })?);
    let pool = Arc::new(ComputePool::new(options.threads));
    let decoder = Arc::new(Qwen3Model::from_source(
        llm_source.clone(),
        tokenizer,
        pool,
    )?);
    if decoder.config().architecture != "qwen3vl" {
        return Err("--audio requires a qwen3vl decoder".into());
    }
    let audio_source = match options.mmproj.as_deref() {
        Some(path) => Arc::from(open_or_exit(path, ComponentRole::Mmproj)),
        None => {
            open_bundled_audio_source(&options.model)?.ok_or("raw GGUF ASR requires --mmproj")?
        }
    };
    let runtime = AsrRuntime::new(decoder, audio_source, prefill_batch_size)
        .map_err(|error| error.to_string())?;
    Ok(AsrBackend {
        runtime: Arc::new(runtime),
    })
}

fn build_audio8(options: &CliOptions) -> Result<Audio8Backend, String> {
    if options.mmproj.is_some() {
        return Err("Audio8 carries its audio tower in --model; omit --mmproj".into());
    }
    let source: Arc<dyn TensorSource> = Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
    let encoder = crate::models::audio8::Audio8Encoder::from_source(Arc::clone(&source))?;
    let decoder_source = Arc::clone(&source);
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
        source.metadata(key).cloned()
    })?);
    let decoder = crate::models::audio8::text::Audio8TextDecoder::from_source(decoder_source)?;
    // Validate the whole contract once at startup rather than per request.
    let specials = audio8_streaming::SpecialTokens::lookup(&tokenizer)?;
    let language = options.language.as_deref().unwrap_or("zh");
    if !matches!(language, "zh" | "en") {
        return Err("Audio8 --language must be zh or en".into());
    }
    let language: &'static str = if language == "en" { "en" } else { "zh" };
    let max_tokens = options.max_tokens.unwrap_or(512);
    // Pin the encoder and decoder in `Box::leak` so per-request
    // `StreamingTranscriber`s can hold a borrow for the server lifetime.
    let encoder: &'static crate::models::audio8::Audio8Encoder = Box::leak(Box::new(encoder));
    let decoder: &'static crate::models::audio8::text::Audio8TextDecoder =
        Box::leak(Box::new(decoder));
    Ok(Audio8Backend {
        encoder: Arc::new(encoder),
        decoder: Arc::new(decoder),
        tokenizer,
        specials,
        language,
        max_tokens,
    })
}

fn build_tts(options: &CliOptions) -> Result<TtsBackend, String> {
    let source: Arc<dyn TensorSource> = Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|k| {
        source.metadata(k).cloned()
    })?);
    let pool = Arc::new(ComputePool::new(options.threads));
    let talker = Arc::new(Qwen3TtsTalker::from_source(source, tokenizer, pool)?);
    let mmproj_path = options
        .mmproj
        .as_deref()
        .ok_or_else(|| "--tts requires --mmproj".to_string())?;
    let mmproj: Arc<dyn TensorSource> = Arc::from(open_or_exit(mmproj_path, ComponentRole::Mmproj));
    let language = normalize_tts_language(options.language.as_deref())?;
    Ok(TtsBackend {
        talker,
        mmproj,
        language,
        temperature: options.temperature.unwrap_or(0.6),
        max_tokens: options.max_tokens.unwrap_or(128),
    })
}

// =============================================================================
// main
// =============================================================================

fn configure_gpu(options: &CliOptions) -> Result<crate::compute::LegacyComputeScope, String> {
    options.validate_compute_mode()?;
    options.compute_policy().configure_cli()
}

fn reject_unsupported_server_modes(options: &CliOptions) -> Result<(), String> {
    if options.yue2 {
        return Err("--yue2 is not supported by rust-model-server".into());
    }
    if options.dreamx {
        return Err("--dreamx is not supported by rust-model-server".into());
    }
    if options.planner.is_some() || options.perception.is_some() {
        return Err("Qwen-Drive heads are not supported by rust-model-server".into());
    }
    Ok(())
}

pub fn run_server() {
    let _ = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,rust_model_inference=info"),
    )
    .try_init();
    let raw_args: Vec<String> = std::env::args().collect();
    if raw_args
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!("{USAGE}");
        return;
    }

    // Pre-parse --host/--port (server-only) before passing the rest to the
    // shared CLI parser so the rest of the surface stays in lockstep with
    // the main `rust-model-inference` binary.
    let mut host = "0.0.0.0".to_string();
    let mut port: u16 = 8080;
    // Off by default: fetching remote URLs from an inference server is an SSRF
    // surface (internal-network probing, request amplification). When enabled,
    // the download still obeys size / timeout / redirect caps and refuses
    // private addresses.
    let mut allow_remote_images = false;
    let mut cli_args: Vec<String> = Vec::with_capacity(raw_args.len());
    let mut i = 0;
    while i < raw_args.len() {
        let arg = raw_args[i].clone();
        match arg.as_str() {
            "--allow-remote-images" => {
                allow_remote_images = true;
                i += 1;
                continue;
            }
            "--host" => {
                if i + 1 < raw_args.len() {
                    host = raw_args[i + 1].clone();
                    i += 2;
                    continue;
                }
            }
            "--port" => {
                if i + 1 < raw_args.len() {
                    port = raw_args[i + 1].parse().unwrap_or(8080);
                    i += 2;
                    continue;
                }
            }
            _ => {}
        }
        cli_args.push(arg);
        i += 1;
    }

    // Apply the parsed flag before any request can be decoded.
    api::image_input::set_allow_remote(allow_remote_images);

    let options = match parse_cli_options(&cli_args) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = reject_unsupported_server_modes(&options) {
        eprintln!("{error}");
        std::process::exit(2);
    }
    // `--tts` validation requires non-empty `--prompt` and `--out`, but the
    // server receives both over HTTP. Inject placeholders so validation
    // passes; the real values come from `/v1/audio/speech` requests.
    let mut options = options;
    if options.tts {
        if options.prompt.as_deref().is_none_or(str::is_empty) {
            options.prompt = Some("placeholder".to_string());
        }
        if options.out.is_none() {
            options.out = Some(std::path::PathBuf::from("placeholder.wav"));
        }
        if options.max_tokens.is_none() {
            options.max_tokens = Some(128);
        }
    }
    if let Err(error) = validate_cli_options(&options) {
        eprintln!("{error}");
        std::process::exit(2);
    }
    if !options.tts
        && options.audio.is_none()
        && !options.embedding
        && options.model.as_os_str().is_empty()
    {
        eprintln!("{USAGE}");
        std::process::exit(1);
    }

    let _compute_scope = configure_gpu(&options).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let backend = match build_backend(&options) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("Failed to build model backend: {error}");
            std::process::exit(1);
        }
    };
    let model_name = options
        .model
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let mode_label = match backend.as_ref() {
        Backend::Text(_) => "text",
        Backend::Embedding(_) => "embedding",
        Backend::Asr(_) => "asr",
        Backend::Audio8(_) => "audio8",
        Backend::Tts(_) => "tts",
        Backend::Rerank(_) => "rerank",
        Backend::Clm(_) => "clm",
        Backend::Gliner2(_) => "gliner2",
        Backend::Gliner2Boundary(_) => "gliner2-boundary",
    };
    eprintln!(
        "Model '{}' loaded (mode={}, host={}, port={})",
        model_name, mode_label, host, port
    );

    let state = AppState {
        compute: options.compute_policy(),
        model: backend,
        model_name,
        responses: Arc::new(Mutex::new(api::ResponsesStore::default())),
        generation_slot: Arc::new(tokio::sync::Semaphore::new(1)),
    };

    let mut router = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models));
    router = match state.model.as_ref() {
        Backend::Text(_) => router.merge(api::routes()),
        Backend::Embedding(_) => router.route(
            "/v1/embeddings",
            post(embeddings).layer(DefaultBodyLimit::max(64 * 1024 * 1024)),
        ),
        Backend::Asr(_) | Backend::Audio8(_) => router
            .route(
                "/v1/audio/transcriptions",
                post(transcriptions).layer(DefaultBodyLimit::max(64 * 1024 * 1024)),
            )
            .route("/v1/audio/transcriptions_json", post(transcriptions_json)),
        Backend::Tts(_) => router.route("/v1/audio/speech", post(speech)),
        Backend::Rerank(_) => router.route("/v1/rerank", post(rerank::rerank)),
        // CLM scores by cosine, so only the single-question route applies.
        // Grouped does a per-group softmax that has no cosine analogue,
        // and the image routes need a vision encoder the heads never saw.
        // GLiNER2 and CLM both score a caller-supplied label set on one
        // encoder pass, so the single-question route covers them; grouped mode
        // (per-group softmax) and the image routes do not apply.
        Backend::Clm(_) | Backend::Gliner2(_) => {
            router.route("/v1/jev/score", post(api::jev_score))
        }
        // The boundary variant returns spans and classification groups, which
        // `JevResult` cannot hold, so it gets its own request/response shape
        // rather than being bent into the JEV one. Deliberately *not* aliased
        // onto `/v1/jev/score`: a JEV-shaped body there would fail on a missing
        // `schema` field, which reads as a malformed request rather than as
        // "this server does not do that".
        Backend::Gliner2Boundary(_) => router.route("/v1/jev/boundary", post(api::jev_boundary)),
    };
    if options.compute_policy() == crate::compute::ComputePolicy::Vulkan {
        router = router.layer(axum::middleware::from_fn(
            |request: axum::extract::Request, next: axum::middleware::Next| async move {
                if request.uri().path().starts_with("/v1/jev/") {
                    return (
                        StatusCode::BAD_REQUEST,
                        "JEV has no complete Vulkan execution path",
                    )
                        .into_response();
                }
                next.run(request).await
            },
        ));
    }
    let app = router.layer(CorsLayer::permissive()).with_state(state);

    let addr = format!("{}:{}", host, port);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let listener = runtime.block_on(async {
        tokio::net::TcpListener::bind(&addr)
            .await
            .unwrap_or_else(|error| {
                eprintln!("Failed to bind {addr}: {error}");
                std::process::exit(1);
            })
    });
    eprintln!("Server listening on http://{}", addr);
    runtime.block_on(async {
        axum::serve(listener, app).await.unwrap();
    });
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::{configure_gpu, CliOptions};

    #[test]
    fn compute_request_scope_reaches_blocking_workers() {
        use crate::compute::ComputePolicy;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            super::compute_task(ComputePolicy::Cpu, || {
                assert_eq!(ComputePolicy::legacy(), ComputePolicy::Cpu);
                let pool = crate::ComputePool::new(2);
                pool.compute(|_, _| assert!(crate::core::thread_pool::gpu_matmul_disabled()));
            })
            .await
            .unwrap();
            super::compute_task(ComputePolicy::Vulkan, || {
                assert_eq!(ComputePolicy::legacy(), ComputePolicy::Vulkan);
                assert!(!crate::core::thread_pool::gpu_matmul_disabled());
            })
            .await
            .unwrap();
        });
    }

    #[test]
    fn gpu_flag_reaches_shared_switch() {
        let options = CliOptions {
            gpu: true,
            ..CliOptions::default()
        };

        let _scope = configure_gpu(&options).unwrap();

        assert!(crate::ops::float::gpu_requested());
    }
}

#[cfg(test)]
mod server_mode_tests {
    use super::{reject_unsupported_server_modes, rerank_probe_flag_conflict, CliOptions};
    use std::path::PathBuf;

    fn options_with(model: &str) -> CliOptions {
        CliOptions {
            model: PathBuf::from(model),
            ..CliOptions::default()
        }
    }

    #[test]
    fn dreamx_is_rejected_before_backend_construction() {
        let options = CliOptions {
            dreamx: true,
            ..CliOptions::default()
        };

        assert!(reject_unsupported_server_modes(&options)
            .unwrap_err()
            .contains("--dreamx"));
    }

    #[test]
    fn rerank_probe_conflict_is_silent_for_plain_rerank_invocation() {
        // The documented happy path: no other backend flag, so the probe
        // owns the dispatch and nothing is dropped.
        assert!(rerank_probe_flag_conflict(&options_with("Qwen3-Reranker.gguf")).is_none());
    }

    #[test]
    fn rerank_probe_conflict_rejects_clm_head() {
        let options = CliOptions {
            clm_head: Some(PathBuf::from("heads.gguf")),
            ..options_with("Qwen3-Reranker.gguf")
        };
        let error = rerank_probe_flag_conflict(&options).expect("expected conflict");
        assert!(error.contains("--clm-head"), "{error}");
        assert!(error.contains("heads.gguf"), "{error}");
        assert!(error.contains("Qwen3-Reranker.gguf"), "{error}");
    }

    #[test]
    fn rerank_probe_conflict_rejects_gliner2_decide() {
        let options = CliOptions {
            gliner2_decide: true,
            ..options_with("Qwen3-Reranker.gguf")
        };
        let error = rerank_probe_flag_conflict(&options).expect("expected conflict");
        assert!(error.contains("--gliner2-decide"), "{error}");
        assert!(error.contains("DeBERTa"), "{error}");
    }

    #[test]
    fn rerank_probe_conflict_rejects_mmproj() {
        let options = CliOptions {
            mmproj: Some(PathBuf::from("vision.gguf")),
            ..options_with("Qwen3-Reranker.gguf")
        };
        let error = rerank_probe_flag_conflict(&options).expect("expected conflict");
        assert!(error.contains("--mmproj"), "{error}");
        assert!(error.contains("vision.gguf"), "{error}");
    }

    #[test]
    fn rerank_probe_conflict_reports_first_conflicting_flag_only() {
        // clm_head is checked first; a caller who set both should still
        // get one actionable message rather than a concatenated wall.
        let options = CliOptions {
            clm_head: Some(PathBuf::from("heads.gguf")),
            gliner2_decide: true,
            ..options_with("Qwen3-Reranker.gguf")
        };
        let error = rerank_probe_flag_conflict(&options).expect("expected conflict");
        assert!(error.contains("--clm-head"), "{error}");
        assert!(!error.contains("--gliner2-decide"), "{error}");
    }
}
