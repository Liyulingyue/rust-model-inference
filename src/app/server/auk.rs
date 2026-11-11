//! The AuK HTTP backend.
//!
//! AuK is a flow-matching audio DiT: a DiT GGUF, a BigVGANFlow VAE GGUF and
//! an optional Qwen2.5-Omni text encoder, plus an optional Qwen2.5-Omni
//! audio tower for reference-audio conditioning. The generation itself lives
//! in `src/models/diffusion/auk`; this module only maps requests onto
//! [`AukOptions`] and writes the audio.
//!
//! Two routes are served:
//!
//! * `POST /v1/audio/speech` — the OpenAI-compatible shape (`input`, `voice`,
//!   `response_format`, `speed`) with the AuK knobs as extra optional fields.
//! * `POST /v1/audio/generations` — the full AuK shape. It accepts JSON or
//!   `multipart/form-data`, and takes the reference audio as an upload, a
//!   local path or base64.

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::{FromRequest, Multipart, Request, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::Engine as _;
use serde::{Deserialize, Serialize};

use crate::app::cli::CliOptions;
use crate::app::server::{AppState, Backend, ErrorResponse};
use crate::core::tensor::{MetaValue, TensorSource};
use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::models::diffusion::auk::{AukAudio, AukOptions, AukPipeline, FLASH_TIME_GRID};

/// The AuK pipeline plus the request defaults taken from the CLI flags.
#[derive(Clone)]
pub(super) struct AukBackend {
    pipeline: Arc<AukPipeline>,
    /// The Qwen2.5-Omni audio tower, needed for reference-audio
    /// conditioning. `None` means reference audio is rejected.
    audio_tower: Option<Arc<dyn TensorSource>>,
    defaults: AukDefaults,
}

/// The defaults a request inherits when it does not override them.
#[derive(Clone, Copy, Debug)]
pub(super) struct AukDefaults {
    steps: usize,
    guidance_scale: f32,
    duration_sec: usize,
    sample_rate: u32,
    seed: i64,
    sway_sampling_coef: Option<f32>,
    flash: bool,
}

/// Where the reference audio comes from.
#[derive(Debug, Default)]
enum ReferenceAudio {
    #[default]
    None,
    /// An upload or a base64 payload, already in memory.
    Bytes(Vec<u8>),
    /// A file on the server.
    Path(PathBuf),
}

/// The audio container the caller asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResponseFormat {
    Wav,
    Pcm,
    Json,
}

/// The knobs a request may set. `None` means "inherit the server default".
#[derive(Debug, Default)]
struct AukKnobs {
    steps: Option<usize>,
    guidance_scale: Option<f32>,
    cfg_scale: Option<f32>,
    sway_sampling_coef: Option<f32>,
    flash: Option<bool>,
    duration: Option<f32>,
    seed: Option<i64>,
    sample_rate: Option<u32>,
    response_format: Option<String>,
    speed: Option<f32>,
    pitch_shift: Option<f32>,
    volume_change: Option<f32>,
}

/// One render request, fully resolved.
struct AukJob {
    prompt: String,
    instruct: Option<String>,
    reference: ReferenceAudio,
    options: AukOptions,
    format: ResponseFormat,
    speed: Option<f32>,
    pitch_shift: Option<f32>,
    volume_change: Option<f32>,
}

/// The OpenAI-compatible speech request with the AuK knobs as extras.
#[derive(Debug, Default, Deserialize)]
pub(super) struct AukSpeechRequest {
    /// The server hosts one model, so the name is accepted and ignored.
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
    /// The text to speak.
    pub input: String,
    /// The reference voice: `file://<path>` or `data:audio/wav;base64,...`.
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub response_format: Option<String>,
    /// Playback speed multiplier, applied to the rendered audio.
    #[serde(default)]
    pub speed: Option<f32>,
    /// AuK extras.
    #[serde(default)]
    pub instruct: Option<String>,
    #[serde(default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub guidance_scale: Option<f32>,
    /// Alias for `guidance_scale`, matching the CLI flag name.
    #[serde(default)]
    pub cfg_scale: Option<f32>,
    #[serde(default)]
    pub sway_sampling_coef: Option<f32>,
    #[serde(default)]
    pub flash: Option<bool>,
    #[serde(default)]
    pub duration: Option<f32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub sample_rate: Option<u32>,
}

/// The full AuK request shape.
#[derive(Debug, Default, Deserialize)]
pub(super) struct AukGenerationsRequest {
    /// The text to speak.
    pub prompt: String,
    /// The instruct prefix for instruct-TTS.
    #[serde(default)]
    pub instruct: Option<String>,
    /// A reference audio file on the server.
    #[serde(default)]
    pub reference_audio_path: Option<String>,
    /// A reference audio payload: raw base64 or a `data:` URL.
    #[serde(default)]
    pub reference_audio_base64: Option<String>,
    #[serde(default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub guidance_scale: Option<f32>,
    #[serde(default)]
    pub cfg_scale: Option<f32>,
    #[serde(default)]
    pub sway_sampling_coef: Option<f32>,
    #[serde(default)]
    pub flash: Option<bool>,
    #[serde(default)]
    pub duration: Option<f32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub sample_rate: Option<u32>,
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub speed: Option<f32>,
    /// Pitch shift in semitones. Changes the duration as well.
    #[serde(default)]
    pub pitch_shift: Option<f32>,
    /// Volume change in decibels.
    #[serde(default)]
    pub volume_change: Option<f32>,
}

/// The JSON answer of `response_format = "json"`.
#[derive(Serialize)]
struct AukJsonResponse {
    /// The WAV payload, base64 encoded.
    audio: String,
    sample_rate: u32,
    channels: u16,
    samples: usize,
    duration_sec: f32,
}

/// Does this GGUF hold an AuK DiT?
///
/// The official GGUF calls the architecture `audiocpp`; the tensor probe
/// covers the exports that spell it differently.
pub(super) fn matches_auk(source: &dyn TensorSource) -> bool {
    source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        == Some("audiocpp")
        || source
            .tensor_info("transformer.transformer_blocks.0.attn_norm_x.linear.weight")
            .is_some()
}

/// Open the `--model` file when the invocation is an AuK one.
///
/// The AuK DiT ships with its own VAE GGUF, so `--vae` plus the architecture
/// of the `--model` file identify the invocation. The opened source is
/// returned so the backend does not have to open the file twice.
pub(super) fn probe_invocation(options: &CliOptions) -> Option<Arc<dyn TensorSource>> {
    if options.vae.is_none() {
        return None;
    }
    let source = open_model_source(&options.model, ComponentRole::Llm).ok()?;
    if !matches_auk(source.as_ref()) {
        return None;
    }
    Some(Arc::from(source) as Arc<dyn TensorSource>)
}

/// Check the AuK specific server flags.
///
/// The diff-model validators are skipped for AuK (they want `--out`), so the
/// AuK knobs get their own check.
pub(super) fn validate_server_options(options: &CliOptions) -> Result<(), String> {
    if let Some(scale) = options.cfg_scale {
        if !scale.is_finite() || scale < 0.0 {
            return Err(format!(
                "--cfg-scale must be finite and not negative (got {scale}); \
                 0 disables classifier-free guidance"
            ));
        }
    }
    if let Some(coefficient) = options.auk_sway {
        if !coefficient.is_finite() {
            return Err(format!("--auk-sway must be finite (got {coefficient})"));
        }
    }
    if options.steps == Some(0) {
        return Err("--steps must be at least 1".into());
    }
    if let Some(seconds) = options.duration_seconds {
        if !seconds.is_finite() || seconds <= 0.0 {
            return Err(format!(
                "--duration-seconds must be a positive number of seconds (got {seconds})"
            ));
        }
    }
    Ok(())
}

/// Load the AuK pipeline for the server.
pub(super) fn build_auk(
    options: &CliOptions,
    diffusion: Arc<dyn TensorSource>,
) -> Result<AukBackend, String> {
    let vae_path = options.vae.as_deref().ok_or_else(|| {
        "the AuK backend needs --vae <auk-vae-*.gguf> (the BigVGANFlow decoder)".to_string()
    })?;
    let vae = open_source(vae_path)?;
    let text = match options.text_encoder.as_deref() {
        Some(path) => Some(open_source(path)?),
        None => None,
    };
    let audio_tower = match options.qwen_omni_bf16.as_deref() {
        Some(path) => Some(open_source(path)?),
        None => None,
    };
    let pipeline = AukPipeline::load_with_audio_tower(
        diffusion,
        vae,
        text,
        audio_tower.clone(),
        options.threads,
    )?;
    Ok(AukBackend {
        pipeline: Arc::new(pipeline),
        audio_tower,
        defaults: AukDefaults {
            steps: options.steps.unwrap_or(32),
            guidance_scale: options.cfg_scale.unwrap_or(2.0),
            duration_sec: options
                .duration_seconds
                .map(|seconds| seconds as usize)
                .unwrap_or(1)
                .max(1),
            sample_rate: options.resolution.unwrap_or(24_000) as u32,
            seed: options.seed.unwrap_or(0),
            sway_sampling_coef: options.auk_sway.or(Some(-1.0)),
            flash: options.auk_flash,
        },
    })
}

fn open_source(path: &std::path::Path) -> Result<Arc<dyn TensorSource>, String> {
    open_model_source(path, ComponentRole::Llm)
        .map(|source| Arc::from(source) as Arc<dyn TensorSource>)
        .map_err(|error| format!("failed to load {}: {error}", path.display()))
}

/// `POST /v1/audio/speech`: the OpenAI-compatible route.
pub(super) async fn speech(
    State(state): State<AppState>,
    Json(request): Json<AukSpeechRequest>,
) -> Response {
    let Some(backend) = auk_backend(&state) else {
        return fail(
            StatusCode::BAD_REQUEST,
            "Server is not running an AuK model",
        );
    };
    let reference = match request.voice.as_deref() {
        None | Some("") => ReferenceAudio::None,
        Some(spec) => match reference_from_spec(spec) {
            Ok(reference) => reference,
            Err(message) => return fail(StatusCode::BAD_REQUEST, &message),
        },
    };
    let knobs = AukKnobs {
        steps: request.steps,
        guidance_scale: request.guidance_scale,
        cfg_scale: request.cfg_scale,
        sway_sampling_coef: request.sway_sampling_coef,
        flash: request.flash,
        duration: request.duration,
        seed: request.seed,
        sample_rate: request.sample_rate,
        response_format: request.response_format.clone(),
        speed: request.speed,
        pitch_shift: None,
        volume_change: None,
    };
    let job = match resolve_job(
        &request.input,
        request.instruct.clone(),
        reference,
        knobs,
        &backend.defaults,
    ) {
        Ok(job) => job,
        Err(message) => return fail(StatusCode::BAD_REQUEST, &message),
    };
    render(state, &backend, job).await
}

/// `POST /v1/audio/generations`: the full AuK route, JSON or multipart.
pub(super) async fn generations(State(state): State<AppState>, request: Request) -> Response {
    let Some(backend) = auk_backend(&state) else {
        return fail(
            StatusCode::BAD_REQUEST,
            "Server is not running an AuK model",
        );
    };
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();
    let parsed = if content_type.starts_with("multipart/form-data") {
        parse_multipart(request, &state).await
    } else {
        match Json::<AukGenerationsRequest>::from_request(request, &state).await {
            Ok(Json(body)) => Ok(body),
            Err(rejection) => Err(rejection.body_text()),
        }
    };
    let body = match parsed {
        Ok(body) => body,
        Err(message) => return fail(StatusCode::BAD_REQUEST, &message),
    };
    let reference = match resolve_generations_reference(&body) {
        Ok(reference) => reference,
        Err(message) => return fail(StatusCode::BAD_REQUEST, &message),
    };
    let knobs = AukKnobs {
        steps: body.steps,
        guidance_scale: body.guidance_scale,
        cfg_scale: body.cfg_scale,
        sway_sampling_coef: body.sway_sampling_coef,
        flash: body.flash,
        duration: body.duration,
        seed: body.seed,
        sample_rate: body.sample_rate,
        response_format: body.response_format.clone(),
        speed: body.speed,
        pitch_shift: body.pitch_shift,
        volume_change: body.volume_change,
    };
    let job = match resolve_job(
        &body.prompt,
        body.instruct.clone(),
        reference,
        knobs,
        &backend.defaults,
    ) {
        Ok(job) => job,
        Err(message) => return fail(StatusCode::BAD_REQUEST, &message),
    };
    render(state, &backend, job).await
}

/// The AuK backend, if this server runs one.
fn auk_backend(state: &AppState) -> Option<AukBackend> {
    match state.model.as_ref() {
        Backend::Auk(backend) => Some(backend.clone()),
        _ => None,
    }
}

/// Resolve the reference audio of the generations route: an upload wins over
/// a local path, which wins over a base64 payload.
fn resolve_generations_reference(body: &AukGenerationsRequest) -> Result<ReferenceAudio, String> {
    if let Some(path) = body.reference_audio_path.as_deref() {
        if path.is_empty() {
            return Err("reference_audio_path must not be empty".into());
        }
        return Ok(ReferenceAudio::Path(PathBuf::from(path)));
    }
    if let Some(payload) = body.reference_audio_base64.as_deref() {
        return reference_from_base64(payload);
    }
    Ok(ReferenceAudio::None)
}

/// Parse a `file://` path or a `data:` URL into a reference audio source.
fn reference_from_spec(spec: &str) -> Result<ReferenceAudio, String> {
    if let Some(path) = spec.strip_prefix("file://") {
        if path.is_empty() {
            return Err("voice must name a file after file://".into());
        }
        return Ok(ReferenceAudio::Path(PathBuf::from(path)));
    }
    if spec.starts_with("data:") {
        return reference_from_base64(spec);
    }
    Err("voice must be file://path or data:audio/wav;base64,...".into())
}

/// Decode a base64 payload or a `data:` URL into reference audio bytes.
fn reference_from_base64(payload: &str) -> Result<ReferenceAudio, String> {
    let encoded = match payload.strip_prefix("data:") {
        Some(rest) => match rest.split_once(',') {
            Some((_, data)) => data,
            None => return Err("the data: URL has no payload".into()),
        },
        None => payload,
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded.trim())
        .map_err(|error| format!("reference audio base64 decode failed: {error}"))?;
    if bytes.is_empty() {
        return Err("the reference audio payload is empty".into());
    }
    Ok(ReferenceAudio::Bytes(bytes))
}

/// Read a multipart upload into the generations request shape.
async fn parse_multipart(
    request: Request,
    state: &AppState,
) -> Result<AukGenerationsRequest, String> {
    let mut multipart = match Multipart::from_request(request, state).await {
        Ok(multipart) => multipart,
        Err(rejection) => return Err(rejection.body_text()),
    };
    let mut body = AukGenerationsRequest::default();
    let mut upload: Option<Vec<u8>> = None;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(error) => return Err(format!("multipart error: {error}")),
        };
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "reference_audio" | "file" | "voice" => {
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|error| format!("failed to read the upload: {error}"))?;
                upload = Some(bytes.to_vec());
            }
            _ => {
                let text = field
                    .text()
                    .await
                    .map_err(|error| format!("multipart error: {error}"))?;
                apply_multipart_field(&mut body, &name, &text);
            }
        }
    }
    if body.prompt.is_empty() {
        return Err("the prompt field is required".into());
    }
    if let Some(bytes) = upload {
        body.reference_audio_base64 = Some(encode_base64(&bytes));
    }
    Ok(body)
}

/// Apply one multipart text field to the request shape.
fn apply_multipart_field(body: &mut AukGenerationsRequest, name: &str, text: &str) {
    match name {
        "prompt" | "input" | "text" => body.prompt = text.to_string(),
        "instruct" | "instruction" => body.instruct = Some(text.to_string()),
        "reference_audio_path" => body.reference_audio_path = Some(text.to_string()),
        "steps" => body.steps = text.parse().ok(),
        "guidance_scale" | "cfg_scale" => {
            let value = text.parse::<f32>().ok();
            if name == "cfg_scale" {
                body.cfg_scale = value;
            } else {
                body.guidance_scale = value;
            }
        }
        "sway_sampling_coef" | "sway" => body.sway_sampling_coef = text.parse().ok(),
        "flash" => body.flash = text.parse().ok(),
        "duration" | "duration_sec" => body.duration = text.parse().ok(),
        "seed" => body.seed = text.parse().ok(),
        "sample_rate" => body.sample_rate = text.parse().ok(),
        "response_format" => body.response_format = Some(text.to_string()),
        "speed" => body.speed = text.parse().ok(),
        "pitch_shift" => body.pitch_shift = text.parse().ok(),
        "volume_change" => body.volume_change = text.parse().ok(),
        _ => {}
    }
}

fn encode_base64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Resolve every knob of a request into the pipeline options.
///
/// The Flash recipe is pinned: 4 steps, the official time grid and no sway.
fn resolve_job(
    prompt: &str,
    instruct: Option<String>,
    reference: ReferenceAudio,
    knobs: AukKnobs,
    defaults: &AukDefaults,
) -> Result<AukJob, String> {
    if prompt.trim().is_empty() {
        return Err("the prompt must not be empty".into());
    }
    let flash = knobs.flash.unwrap_or(defaults.flash);
    let guidance_scale = knobs
        .guidance_scale
        .or(knobs.cfg_scale)
        .or(if flash { Some(0.0) } else { None })
        .unwrap_or(defaults.guidance_scale);
    if !guidance_scale.is_finite() || guidance_scale < 0.0 {
        return Err(format!(
            "guidance_scale must be finite and not negative (got {guidance_scale}); \
             0 disables classifier-free guidance"
        ));
    }
    let sway_sampling_coef = if flash {
        None
    } else {
        knobs.sway_sampling_coef.or(defaults.sway_sampling_coef)
    };
    if let Some(coefficient) = sway_sampling_coef {
        if !coefficient.is_finite() {
            return Err(format!(
                "sway_sampling_coef must be finite (got {coefficient})"
            ));
        }
    }
    let steps = if flash {
        4
    } else {
        knobs.steps.unwrap_or(defaults.steps)
    };
    if steps == 0 {
        return Err("steps must be at least 1".into());
    }
    let duration_sec = match knobs.duration {
        Some(seconds) => {
            if !seconds.is_finite() || seconds <= 0.0 {
                return Err(format!(
                    "duration must be a positive number of seconds (got {seconds})"
                ));
            }
            (seconds.round() as usize).max(1)
        }
        None => defaults.duration_sec,
    };
    if let Some(rate) = knobs.speed {
        if !rate.is_finite() || rate <= 0.0 {
            return Err(format!("speed must be a positive number (got {rate})"));
        }
    }
    let format = parse_response_format(knobs.response_format.as_deref())?;
    Ok(AukJob {
        prompt: prompt.to_string(),
        instruct,
        reference,
        options: AukOptions {
            steps,
            sample_rate: knobs.sample_rate.unwrap_or(defaults.sample_rate),
            duration_sec,
            seed: knobs.seed.unwrap_or(defaults.seed),
            guidance_scale,
            instruct: None,
            sway_sampling_coef,
            time_grid: if flash {
                Some(FLASH_TIME_GRID.to_vec())
            } else {
                None
            },
        },
        format,
        speed: knobs.speed,
        pitch_shift: knobs.pitch_shift,
        volume_change: knobs.volume_change,
    })
}

/// The response container the caller asked for.
fn parse_response_format(format: Option<&str>) -> Result<ResponseFormat, String> {
    match format {
        None | Some("") | Some("wav") | Some("wave") => Ok(ResponseFormat::Wav),
        Some("pcm") => Ok(ResponseFormat::Pcm),
        Some("json") => Ok(ResponseFormat::Json),
        Some(other) => Err(format!(
            "unsupported response_format {other:?}; use wav, pcm or json"
        )),
    }
}

/// Run the render and write the answer.
async fn render(state: AppState, backend: &AukBackend, job: AukJob) -> Response {
    if !matches!(job.reference, ReferenceAudio::None) && backend.audio_tower.is_none() {
        return fail(
            StatusCode::BAD_REQUEST,
            "reference audio needs the Qwen2.5-Omni audio tower; start the server with \
             --qwen-omni-bf16 <qwen2.5-omni-*.gguf>",
        );
    }
    // One render at a time: the DiT, the VAE and the device buffers are
    // shared, and a 32 step render already takes a minute.
    let permit = match state.generation_slot.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return fail(
                StatusCode::TOO_MANY_REQUESTS,
                "Another generation is running; retry after it completes",
            )
        }
    };
    let pipeline = Arc::clone(&backend.pipeline);
    let format = job.format;
    let handle = super::compute_task(state.compute, move || {
        let _permit = permit;
        render_sync(&pipeline, &job)
    });
    match handle.await {
        Ok(Ok(audio)) => match encode_response(&audio, format) {
            Ok(response) => response,
            Err(message) => fail(StatusCode::INTERNAL_SERVER_ERROR, &message),
        },
        Ok(Err(message)) => fail(StatusCode::UNPROCESSABLE_ENTITY, &message),
        Err(failure) => fail(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("the AuK worker failed: {failure}"),
        ),
    }
}

/// Render one job. Runs on a blocking worker.
fn render_sync(pipeline: &AukPipeline, job: &AukJob) -> Result<AukAudio, String> {
    let reference = match &job.reference {
        ReferenceAudio::None => None,
        ReferenceAudio::Bytes(bytes) => Some(crate::app::media::decode_audio_bytes(bytes)?),
        ReferenceAudio::Path(path) => Some(crate::app::media::decode_audio(path)?),
    };
    let mut options = job.options.clone();
    options.instruct = job.instruct.clone();
    let mut audio = match reference {
        Some(samples) => {
            pipeline.generate_audio_with_reference_wav(&job.prompt, &samples, &options)?
        }
        None => pipeline.generate_audio(&job.prompt, &options)?,
    };
    if let Some(speed) = job.speed {
        audio = audio.speed_change(speed);
    }
    if let Some(semitones) = job.pitch_shift {
        audio = audio.pitch_shift(semitones);
    }
    if let Some(decibels) = job.volume_change {
        audio = audio.volume_change(decibels);
    }
    Ok(audio)
}

/// Encode the rendered audio in the requested container.
fn encode_response(audio: &AukAudio, format: ResponseFormat) -> Result<Response, String> {
    let wav = crate::format::wav::encode_wav_pcm16_f64(
        &audio.samples,
        audio.sample_rate,
        audio.channels as u16,
    )?;
    match format {
        ResponseFormat::Wav => {
            Ok((StatusCode::OK, [(header::CONTENT_TYPE, "audio/wav")], wav).into_response())
        }
        ResponseFormat::Pcm => {
            let pcm = raw_pcm16(&audio.samples);
            Ok((StatusCode::OK, [(header::CONTENT_TYPE, "audio/pcm")], pcm).into_response())
        }
        ResponseFormat::Json => {
            let body = AukJsonResponse {
                audio: encode_base64(&wav),
                sample_rate: audio.sample_rate,
                channels: audio.channels as u16,
                samples: audio.samples.len(),
                duration_sec: audio.samples.len() as f32 / audio.sample_rate.max(1) as f32,
            };
            Ok(Json(body).into_response())
        }
    }
}

/// The raw little-endian PCM16 payload, without the RIFF header.
fn raw_pcm16(samples: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let scaled = (sample * 32_768.0).clamp(-32_768.0, 32_767.0) as i16;
        bytes.extend_from_slice(&scaled.to_le_bytes());
    }
    bytes
}

/// An error answer in the shape the other routes use.
fn fail(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: message.to_string(),
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> AukDefaults {
        AukDefaults {
            steps: 32,
            guidance_scale: 2.0,
            duration_sec: 1,
            sample_rate: 24_000,
            seed: 0,
            sway_sampling_coef: Some(-1.0),
            flash: false,
        }
    }

    fn job(knobs: AukKnobs) -> Result<AukJob, String> {
        resolve_job("hello", None, ReferenceAudio::None, knobs, &defaults())
    }

    #[test]
    fn defaults_come_from_the_server_flags() {
        let job = job(AukKnobs::default()).unwrap();
        assert_eq!(job.options.steps, 32);
        assert_eq!(job.options.guidance_scale, 2.0);
        assert_eq!(job.options.duration_sec, 1);
        assert_eq!(job.options.sample_rate, 24_000);
        assert_eq!(job.options.seed, 0);
        assert_eq!(job.options.sway_sampling_coef, Some(-1.0));
        assert_eq!(job.options.time_grid, None);
    }

    #[test]
    fn the_request_overrides_the_defaults() {
        let job = job(AukKnobs {
            steps: Some(8),
            guidance_scale: Some(1.5),
            duration: Some(4.0),
            seed: Some(7),
            sway_sampling_coef: Some(0.0),
            ..AukKnobs::default()
        })
        .unwrap();
        assert_eq!(job.options.steps, 8);
        assert_eq!(job.options.guidance_scale, 1.5);
        assert_eq!(job.options.duration_sec, 4);
        assert_eq!(job.options.seed, 7);
        assert_eq!(job.options.sway_sampling_coef, Some(0.0));
    }

    #[test]
    fn the_cfg_scale_alias_is_accepted() {
        let job = job(AukKnobs {
            cfg_scale: Some(0.5),
            ..AukKnobs::default()
        })
        .unwrap();
        assert_eq!(job.options.guidance_scale, 0.5);
    }

    #[test]
    fn flash_pins_the_official_recipe() {
        let job = job(AukKnobs {
            steps: Some(32),
            sway_sampling_coef: Some(-1.0),
            flash: Some(true),
            ..AukKnobs::default()
        })
        .unwrap();
        assert_eq!(job.options.steps, 4);
        assert_eq!(job.options.guidance_scale, 0.0);
        assert_eq!(job.options.sway_sampling_coef, None);
        assert_eq!(
            job.options.time_grid.as_deref(),
            Some(FLASH_TIME_GRID.as_slice())
        );
    }

    #[test]
    fn flash_keeps_an_explicit_guidance_scale() {
        let job = job(AukKnobs {
            guidance_scale: Some(0.5),
            flash: Some(true),
            ..AukKnobs::default()
        })
        .unwrap();
        assert_eq!(job.options.guidance_scale, 0.5);
        assert_eq!(job.options.steps, 4);
    }

    #[test]
    fn invalid_knobs_are_rejected() {
        assert!(job(AukKnobs {
            guidance_scale: Some(-1.0),
            ..AukKnobs::default()
        })
        .is_err());
        assert!(job(AukKnobs {
            guidance_scale: Some(f32::NAN),
            ..AukKnobs::default()
        })
        .is_err());
        assert!(job(AukKnobs {
            steps: Some(0),
            ..AukKnobs::default()
        })
        .is_err());
        assert!(job(AukKnobs {
            sway_sampling_coef: Some(f32::INFINITY),
            ..AukKnobs::default()
        })
        .is_err());
        assert!(job(AukKnobs {
            duration: Some(0.0),
            ..AukKnobs::default()
        })
        .is_err());
        assert!(job(AukKnobs {
            duration: Some(-1.0),
            ..AukKnobs::default()
        })
        .is_err());
        assert!(job(AukKnobs {
            speed: Some(0.0),
            ..AukKnobs::default()
        })
        .is_err());
    }

    #[test]
    fn an_empty_prompt_is_rejected() {
        assert!(resolve_job(
            "   ",
            None,
            ReferenceAudio::None,
            AukKnobs::default(),
            &defaults(),
        )
        .is_err());
    }

    #[test]
    fn the_reference_audio_accepts_a_path_and_a_data_url() {
        assert!(matches!(
            reference_from_spec("file:///tmp/voice.wav").unwrap(),
            ReferenceAudio::Path(_)
        ));
        assert!(matches!(
            reference_from_spec("data:audio/wav;base64,AAAA").unwrap(),
            ReferenceAudio::Bytes(_)
        ));
        assert!(reference_from_spec("/tmp/voice.wav").is_err());
        assert!(reference_from_spec("data:audio/wav;base64").is_err());
    }

    #[test]
    fn the_response_format_is_restricted() {
        assert_eq!(parse_response_format(None).unwrap(), ResponseFormat::Wav);
        assert_eq!(
            parse_response_format(Some("pcm")).unwrap(),
            ResponseFormat::Pcm
        );
        assert_eq!(
            parse_response_format(Some("json")).unwrap(),
            ResponseFormat::Json
        );
        assert!(parse_response_format(Some("mp3")).is_err());
    }

    #[test]
    fn the_multipart_fields_map_onto_the_request() {
        let mut body = AukGenerationsRequest::default();
        apply_multipart_field(&mut body, "prompt", "hello");
        apply_multipart_field(&mut body, "steps", "8");
        apply_multipart_field(&mut body, "cfg_scale", "1.5");
        apply_multipart_field(&mut body, "duration", "3");
        apply_multipart_field(&mut body, "seed", "11");
        apply_multipart_field(&mut body, "flash", "true");
        assert_eq!(body.prompt, "hello");
        assert_eq!(body.steps, Some(8));
        assert_eq!(body.cfg_scale, Some(1.5));
        assert_eq!(body.duration, Some(3.0));
        assert_eq!(body.seed, Some(11));
        assert_eq!(body.flash, Some(true));
    }
}
