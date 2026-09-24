mod api;
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
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::ComponentRole;
use crate::models::qwen3::asr::model::{
    open_bundled_audio_source, AsrRuntime, TranscriptionOptions,
};
use crate::models::qwen3::tts::codec::{
    encode_wav_pcm16, Code2WavDecoder, CodePredictor, WAVEFORM_SAMPLE_RATE,
};
use crate::models::qwen3::tts::speaker::{reference_wav_to_mel, Qwen3TtsSpeakerEncoder};
use crate::models::qwen3::tts::{predictor_top_k, Qwen3TtsTalker, TtsPrompt, TtsSession};
use crate::models::qwen3::{Qwen3GenerateOptions, Qwen3Input, Qwen3Model, Qwen3Session};
use crate::models::qwen35::{build_qwen35_positions, Qwen35Model, Qwen35Session};
use crate::KvLifecycle;

const USAGE: &str = "Usage: rust-model-server --model <path.gguf-or-ggufrs> [--mmproj ...] [--audio ...] [--image ...] [--tts] [--embedding] [--host 0.0.0.0] [--port 8080] [--threads 4] [--prefill-batch-size N (default 64)]";

// =============================================================================
// Backend types
// =============================================================================

#[derive(Clone)]
struct AppState {
    model: Arc<Backend>,
    model_name: String,
    responses: Arc<Mutex<api::ResponsesStore>>,
    generation_slot: Arc<tokio::sync::Semaphore>,
}

enum Backend {
    Text(TextBackend),
    Embedding(EmbeddingBackend),
    Asr(AsrBackend),
    Tts(TtsBackend),
}

unsafe impl Send for Backend {}
unsafe impl Sync for Backend {}

struct TextBackend {
    arch: String,
    pool: Arc<ComputePool>,
    tokenizer: Arc<BPETokenizer>,
    prefill_batch_size: usize,
    context_length: usize,
    inner: TextInner,
}

enum TextInner {
    Qwen3 {
        model: Arc<Qwen3Model>,
    },
    Qwen35 {
        // Qwen35Model borrows from its source; we leak the lifetime to 'static.
        // `Mutex` is needed because forward now takes `&mut self` (Vulkan state).
        model: Mutex<Qwen35Model<'static>>,
        _source: Arc<dyn TensorSource>,
    },
    Fallback {
        arch: String,
    },
}

unsafe impl Send for TextBackend {}
unsafe impl Sync for TextBackend {}
unsafe impl Send for TextInner {}
unsafe impl Sync for TextInner {}

struct EmbeddingBackend {
    source: Arc<dyn TensorSource>,
}

struct AsrBackend {
    runtime: Arc<AsrRuntime>,
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
    input: serde_json::Value,
    #[serde(default)]
    encoding_format: Option<String>,
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

fn sample_token_from_logits(logits: &[f32], temperature: f32) -> i32 {
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as i32)
            .unwrap_or(0);
    }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut probs = vec![0.0f32; logits.len()];
    for (i, l) in logits.iter().enumerate() {
        probs[i] = ((l - max_logit) / temperature).exp();
        sum += probs[i];
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let r: f32 = rand::random();
    let mut cumsum = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i as i32;
        }
    }
    (logits.len() - 1) as i32
}

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
    let inputs: Result<Vec<String>, String> = match req.input {
        serde_json::Value::String(text) => Ok(vec![text]),
        serde_json::Value::Array(items) => items
            .into_iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| "embedding input must be string or array of strings".to_string())
            })
            .collect(),
        _ => Err("embedding input must be string or array of strings".into()),
    };
    let inputs = match inputs {
        Ok(value) => value,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response();
        }
    };
    let source = backend.source.clone();
    let result = match tokio::task::spawn_blocking(move || {
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
    let Backend::Asr(backend) = state.model.as_ref() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Server is not running an ASR model".into(),
            }),
        )
            .into_response();
    };
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
    let options = TranscriptionOptions {
        language: language.clone(),
        prompt: prompt.clone(),
        max_new_tokens: 256,
    };
    let runtime = backend.runtime.clone();
    let transcription =
        match tokio::task::spawn_blocking(move || runtime.transcribe_wav(&wav, &options)).await {
            Ok(Ok(value)) => value,
            Ok(Err(error)) => {
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(ErrorResponse {
                        error: error.to_string(),
                    }),
                )
                    .into_response();
            }
            Err(error) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("asr worker failed: {error}"),
                    }),
                )
                    .into_response();
            }
        };
    let response = TranscriptionResponse {
        text: transcription.text,
    };
    (StatusCode::OK, Json(response)).into_response()
}

async fn transcriptions_json(
    State(state): State<AppState>,
    Json(req): Json<TranscriptionRequest>,
) -> impl IntoResponse {
    let Backend::Asr(backend) = state.model.as_ref() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Server is not running an ASR model".into(),
            }),
        )
            .into_response();
    };
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
    let options = TranscriptionOptions {
        language: req.language.clone(),
        prompt: req.prompt.clone(),
        max_new_tokens: 256,
    };
    let runtime = backend.runtime.clone();
    let transcription =
        match tokio::task::spawn_blocking(move || runtime.transcribe_wav(&wav, &options)).await {
            Ok(Ok(value)) => value,
            Ok(Err(error)) => {
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(ErrorResponse {
                        error: error.to_string(),
                    }),
                )
                    .into_response();
            }
            Err(error) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("asr worker failed: {error}"),
                    }),
                )
                    .into_response();
            }
        };
    let response = TranscriptionResponse {
        text: transcription.text,
    };
    (StatusCode::OK, Json(response)).into_response()
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
    let wav_bytes = match tokio::task::spawn_blocking(move || -> Result<Vec<u8>, String> {
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
        encode_wav_pcm16(&waveform, WAVEFORM_SAMPLE_RATE)
            .map_err(|error| format!("WAV encode failed: {error}"))
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
    if options.tts {
        return Ok(Arc::new(Backend::Tts(build_tts(options)?)));
    }
    if options.audio.is_some() {
        return Ok(Arc::new(Backend::Asr(build_asr(options)?)));
    }
    if options.embedding {
        let source = open_or_exit(&options.model, ComponentRole::Llm);
        return Ok(Arc::new(Backend::Embedding(EmbeddingBackend {
            source: Arc::from(source),
        })));
    }
    Ok(Arc::new(Backend::Text(build_text(options)?)))
}

fn build_text(options: &CliOptions) -> Result<TextBackend, String> {
    let prefill_batch_size = options.effective_prefill_batch_size()?;
    let source: Arc<dyn TensorSource> = Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
    let arch = source
        .metadata("general.architecture")
        .and_then(crate::MetaValue::to_string_val)
        .unwrap_or_default();
    crate::app::reject_incomplete_z_image_architecture(&arch)?;
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|k| {
        source.metadata(k).cloned()
    })?);
    let pool = Arc::new(ComputePool::new(options.threads));
    let inner = match &*arch {
        "qwen3" | "qwen3vl" => {
            let model = Qwen3Model::from_source(source.clone(), tokenizer.clone(), pool.clone())?;
            TextInner::Qwen3 {
                model: Arc::new(model),
            }
        }
        "qwen35" => {
            let model = Qwen35Model::from_source(source.as_ref())?;
            // SAFETY: the model borrows from `source`, which is held by the
            // backend for the full server lifetime; we leak the lifetime to
            // satisfy the 'static bound on Arc storage.
            let model: Qwen35Model<'static> = unsafe { std::mem::transmute(model) };
            TextInner::Qwen35 {
                model: Mutex::new(model),
                _source: source.clone(),
            }
        }
        _ => TextInner::Fallback {
            arch: arch.to_string(),
        },
    };
    let context_length = match &inner {
        TextInner::Qwen3 { model } => model.config().n_ctx,
        TextInner::Qwen35 { model, .. } => model.lock().map_err(|e| e.to_string())?.config.n_ctx,
        TextInner::Fallback { .. } => 0,
    };
    Ok(TextBackend {
        arch: arch.to_string(),
        pool,
        tokenizer,
        prefill_batch_size,
        context_length,
        inner,
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

fn configure_gpu(options: &CliOptions) {
    if options.gpu {
        crate::ops::enable_gpu();
    }
}

fn reject_unsupported_server_modes(options: &CliOptions) -> Result<(), String> {
    if options.dreamx {
        return Err("--dreamx is not supported by rust-model-server".into());
    }
    if options.planner.is_some() || options.perception.is_some() {
        return Err("Qwen-Drive heads are not supported by rust-model-server".into());
    }
    Ok(())
}

pub fn run_server() {
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
    let mut cli_args: Vec<String> = Vec::with_capacity(raw_args.len());
    let mut i = 0;
    while i < raw_args.len() {
        let arg = raw_args[i].clone();
        match arg.as_str() {
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

    configure_gpu(&options);
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
        Backend::Tts(_) => "tts",
    };
    eprintln!(
        "Model '{}' loaded (mode={}, host={}, port={})",
        model_name, mode_label, host, port
    );

    let state = AppState {
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
        Backend::Embedding(_) => router.route("/v1/embeddings", post(embeddings)),
        Backend::Asr(_) => router
            .route(
                "/v1/audio/transcriptions",
                post(transcriptions).layer(DefaultBodyLimit::max(64 * 1024 * 1024)),
            )
            .route("/v1/audio/transcriptions_json", post(transcriptions_json)),
        Backend::Tts(_) => router.route("/v1/audio/speech", post(speech)),
    };
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
    fn gpu_flag_reaches_shared_switch() {
        let options = CliOptions {
            gpu: true,
            ..CliOptions::default()
        };

        configure_gpu(&options);

        assert!(crate::ops::float::gpu_requested());
    }
}

#[cfg(test)]
mod server_mode_tests {
    use super::{reject_unsupported_server_modes, CliOptions};

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
}
