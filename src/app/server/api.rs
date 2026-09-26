pub mod protocol;
mod stop;
pub mod tools;

use protocol::{Message, Request};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

struct CancellationGuard {
    flag: Arc<AtomicBool>,
    armed: bool,
}
impl CancellationGuard {
    fn new(flag: Arc<AtomicBool>) -> Self {
        Self { flag, armed: true }
    }
    fn disarm(&mut self) {
        self.armed = false;
    }
}
impl Drop for CancellationGuard {
    fn drop(&mut self) {
        if self.armed {
            self.flag.store(true, Ordering::Relaxed);
        }
    }
}

struct StoredResponse {
    id: String,
    model: String,
    value: Value,
    messages: Vec<Message>,
    expires: Instant,
    bytes: usize,
}

pub struct ResponsesStore {
    records: VecDeque<StoredResponse>,
    bytes: usize,
    max_records: usize,
    max_bytes: usize,
    ttl: Duration,
}
impl Default for ResponsesStore {
    fn default() -> Self {
        Self::new(128, 64 * 1024 * 1024, Duration::from_secs(1800))
    }
}
impl ResponsesStore {
    fn new(max_records: usize, max_bytes: usize, ttl: Duration) -> Self {
        Self {
            records: VecDeque::new(),
            bytes: 0,
            max_records,
            max_bytes,
            ttl,
        }
    }
    fn prune(&mut self) {
        let now = Instant::now();
        self.records.retain(|record| record.expires > now);
        self.bytes = self.records.iter().map(|record| record.bytes).sum();
    }

    fn insert(
        &mut self,
        id: String,
        model: String,
        value: Value,
        messages: Vec<Message>,
    ) -> Result<(), String> {
        let bytes = serde_json::to_vec(&messages)
            .map_err(|e| e.to_string())?
            .len()
            + value.to_string().len();
        if bytes > self.max_bytes || self.max_records == 0 {
            return Err("Response exceeds storage budget".into());
        }
        self.prune();
        self.remove(&id);
        while self.records.len() >= self.max_records || self.bytes + bytes > self.max_bytes {
            if let Some(old) = self.records.pop_front() {
                self.bytes -= old.bytes;
            }
        }
        self.records.push_back(StoredResponse {
            id,
            model,
            value,
            messages,
            expires: Instant::now() + self.ttl,
            bytes,
        });
        self.bytes += bytes;
        Ok(())
    }

    fn get(&mut self, id: &str) -> Option<(String, Value, Vec<Message>)> {
        self.prune();
        // ponytail: linear lookup over at most 128 records; use a map if this cap grows.
        self.records
            .iter()
            .find(|record| record.id == id)
            .map(|record| {
                (
                    record.model.clone(),
                    record.value.clone(),
                    record.messages.clone(),
                )
            })
    }

    fn remove(&mut self, id: &str) -> bool {
        self.prune();
        if let Some(index) = self.records.iter().position(|record| record.id == id) {
            if let Some(old) = self.records.remove(index) {
                self.bytes -= old.bytes;
            }
            true
        } else {
            false
        }
    }
}

use super::{AppState, Backend, TextInner};
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use axum::{
    extract::{rejection::JsonRejection, Path, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use protocol::{Delta, Encoder, Finish, Generation, Protocol, Usage, WireEvent};
use std::collections::{HashMap, HashSet};

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/v1/responses", post(responses))
        .route(
            "/v1/responses/{id}",
            get(get_response).delete(delete_response),
        )
        .route("/v1/jev/score", post(jev_score))
        .route("/v1/jev/grouped", post(jev_grouped))
        .route("/v1/jev/image", post(jev_image_score))
        .route("/v1/jev/image_grouped", post(jev_image_grouped))
        .layer(axum::extract::DefaultBodyLimit::max(4 * 1024 * 1024))
}

async fn chat(State(state): State<AppState>, body: Result<Json<Value>, JsonRejection>) -> Response {
    handle(state, Protocol::Chat, body).await
}
async fn messages(
    State(state): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    handle(state, Protocol::Anthropic, body).await
}
async fn responses(
    State(state): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    handle(state, Protocol::Responses, body).await
}

fn error(protocol: Protocol, status: u16, message: impl AsRef<str>) -> Response {
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(protocol.error(status, message.as_ref())),
    )
        .into_response()
}

fn check_history(messages: &[Message]) -> Result<(), String> {
    let mut pending = HashMap::new();
    let mut ids = HashSet::new();
    for message in messages {
        if message.role == "tool" {
            let id = message
                .call_id
                .as_deref()
                .ok_or("Tool result is missing its call ID")?;
            if pending.remove(id).is_none() {
                return Err(format!(
                    "Tool result references unknown or already answered call {id:?}"
                ));
            }
        } else if !message.calls.is_empty() {
            if message.role != "assistant" {
                return Err("Only assistant messages may contain tool calls".into());
            }
            for call in &message.calls {
                if !ids.insert(call.id.clone()) {
                    return Err(format!("Duplicate tool call ID {:?}", call.id));
                }
                pending.insert(call.id.as_str(), call.name.as_str());
            }
        } else if !pending.is_empty() {
            return Err(
                "Return all pending tool results before continuing the conversation".into(),
            );
        }
    }
    if !pending.is_empty() {
        return Err("Return all pending tool results before generating another response".into());
    }
    Ok(())
}

fn order_tool_results(messages: &mut Vec<Message>) {
    let original = std::mem::take(messages);
    let mut ordered = Vec::with_capacity(original.len());
    let mut i = 0;
    while i < original.len() {
        let message = original[i].clone();
        let ids = message
            .calls
            .iter()
            .map(|call| call.id.clone())
            .collect::<Vec<_>>();
        ordered.push(message);
        i += 1;
        if ids.is_empty() {
            continue;
        }
        let mut results = HashMap::new();
        while i < original.len() && original[i].role == "tool" {
            if let Some(id) = &original[i].call_id {
                results.insert(id.clone(), original[i].clone());
            }
            i += 1;
        }
        for id in ids {
            if let Some(result) = results.remove(&id) {
                ordered.push(result);
            }
        }
    }
    *messages = ordered;
}

fn resolve(
    state: &AppState,
    protocol: Protocol,
    request: &mut Request,
) -> Result<(), (u16, String)> {
    if let Some(model) = &request.model {
        if model != &state.model_name {
            return Err((
                404,
                format!(
                    "Model {model:?} is not loaded; available model is {:?}",
                    state.model_name
                ),
            ));
        }
    }
    if matches!(protocol, Protocol::Responses) {
        if let Some(id) = &request.previous_response_id {
            let (model, _, mut history) = state
                .responses
                .lock()
                .map_err(|e| (500, e.to_string()))?
                .get(id)
                .ok_or_else(|| {
                    (
                        404,
                        format!("Response {id:?} was not stored, expired, or was evicted"),
                    )
                })?;
            if model != state.model_name {
                return Err((400, "Previous response belongs to another model".into()));
            }
            history.append(&mut request.messages);
            request.messages = history;
        }
    }
    check_history(&request.messages).map_err(|e| (400, e))?;
    order_tool_results(&mut request.messages);
    Ok(())
}

async fn prompt(state: &AppState, request: &Request) -> Result<Vec<u32>, (u16, String)> {
    let Backend::Text(text) = state.model.as_ref() else {
        return Err((400, "Server is not running a text model".into()));
    };
    if matches!(text.inner, TextInner::Fallback { .. }) {
        return Err((
            501,
            format!(
                "Architecture {:?} is not supported by the server text endpoints",
                text.arch
            ),
        ));
    }
    let tokenizer = text.tokenizer.clone();
    let arch = text.arch.clone();
    let context = text.context_length;
    let request = request.clone();
    tokio::task::spawn_blocking(move || {
        let mut messages = request.messages.clone();
        if let Some(instructions) = &request.instructions {
            messages.insert(
                0,
                Message {
                    role: "system".into(),
                    text: instructions.clone(),
                    calls: vec![],
                    call_id: None,
                },
            );
        }
        let ids = tools::build_prompt(
            &tokenizer,
            &arch,
            &messages,
            &request.tools,
            &request.choice,
        )
        .map_err(|e| (400, e))?;
        if ids
            .len()
            .checked_add(request.max_tokens)
            .is_none_or(|total| total > context)
        {
            return Err((
                400,
                format!(
                    "Prompt ({} tokens) plus output budget ({}) exceeds model context ({context})",
                    ids.len(),
                    request.max_tokens
                ),
            ));
        }
        Ok(ids)
    })
    .await
    .map_err(|e| (500, format!("Prompt worker failed: {e}")))?
}

fn apply_response_request(value: &mut Value, request: &Request) {
    value["max_output_tokens"] = Value::from(request.max_tokens);
    value["temperature"] = Value::from(request.temperature);
    value["store"] = Value::Bool(request.store);
    value["previous_response_id"] =
        serde_json::to_value(&request.previous_response_id).unwrap_or(Value::Null);
    value["instructions"] = serde_json::to_value(&request.instructions).unwrap_or(Value::Null);
    value["tools"]=Value::Array(request.tools.iter().map(|tool|serde_json::json!({"type":"function","name":tool.name,"description":tool.description,"parameters":tool.parameters,"strict":false})).collect());
    value["tool_choice"] = match &request.choice {
        protocol::ToolChoice::Auto => Value::from("auto"),
        protocol::ToolChoice::None => Value::from("none"),
        protocol::ToolChoice::Required => Value::from("required"),
        protocol::ToolChoice::Named(name) => serde_json::json!({"type":"function","name":name}),
    };
}
fn patch_response_events(events: &mut [WireEvent], request: &Request) {
    for wire in events {
        let Ok(mut value) = serde_json::from_str::<Value>(&wire.data) else {
            continue;
        };
        let Some(response) = value.get_mut("response") else {
            continue;
        };
        apply_response_request(response, request);
        wire.data = value.to_string();
    }
}

fn store_response(
    state: &AppState,
    id: &str,
    request: &Request,
    mut value: Value,
) -> Result<Value, String> {
    apply_response_request(&mut value, request);
    if request.store {
        let mut history = request.messages.clone();
        // Wire items can interleave text and calls; one generated response is one assistant turn.
        let mut assistant = Message {
            role: "assistant".into(),
            text: String::new(),
            calls: vec![],
            call_id: None,
        };
        if let Some(output) = value["output"].as_array() {
            for item in output {
                match item["type"].as_str() {
                    Some("message") => assistant.text.push_str(
                        &item["content"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .filter(|part| part["type"] == "output_text")
                            .filter_map(|part| part["text"].as_str())
                            .collect::<String>(),
                    ),
                    Some("function_call") => {
                        if let (Some(call_id), Some(name), Some(arguments)) = (
                            item["call_id"].as_str(),
                            item["name"].as_str(),
                            item["arguments"].as_str(),
                        ) {
                            assistant.calls.push(protocol::ToolCall {
                                id: call_id.into(),
                                name: name.into(),
                                arguments: serde_json::from_str(arguments)
                                    .map_err(|e| e.to_string())?,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        if !assistant.text.is_empty() || !assistant.calls.is_empty() {
            history.push(assistant);
        }
        state.responses.lock().map_err(|e| e.to_string())?.insert(
            id.into(),
            state.model_name.clone(),
            value.clone(),
            history,
        )?;
    }
    Ok(value)
}
fn make_response(
    state: &AppState,
    protocol: Protocol,
    id: &str,
    request: &Request,
    generation: &Generation,
) -> Result<Value, String> {
    let value = protocol.response(id, &state.model_name, generation);
    if matches!(protocol, Protocol::Responses) {
        store_response(state, id, request, value)
    } else {
        Ok(value)
    }
}

fn send(
    tx: &tokio::sync::mpsc::Sender<Result<Event, axum::Error>>,
    events: Vec<WireEvent>,
) -> bool {
    events.into_iter().all(|wire| {
        let mut event = Event::default().data(wire.data);
        if let Some(name) = wire.event {
            event = event.event(name);
        }
        tx.blocking_send(Ok(event)).is_ok()
    })
}

async fn handle(
    state: AppState,
    protocol: Protocol,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let Json(body) = match body {
        Ok(body) => body,
        Err(rejection) => {
            return error(protocol, rejection.status().as_u16(), rejection.body_text())
        }
    };
    let mut request = match protocol.parse(&body) {
        Ok(r) => r,
        Err(e) => return error(protocol, 400, e),
    };
    if let Err((status, message)) = resolve(&state, protocol, &mut request) {
        return error(protocol, status, message);
    }
    // One active generation protects shared GPU/model state and bounds server memory.
    let permit = match state.generation_slot.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return error(
                protocol,
                429,
                "Another generation is running; retry after it completes",
            )
        }
    };
    let ids = match prompt(&state, &request).await {
        Ok(ids) => ids,
        Err((status, e)) => return error(protocol, status, e),
    };
    let prefix = match protocol {
        Protocol::Chat => "chatcmpl-",
        Protocol::Anthropic => "msg_",
        Protocol::Responses => "resp_",
    };
    let id = format!("{prefix}{:016x}", rand::random::<u64>());
    if request.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, axum::Error>>(32);
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let mut encoder = Encoder::new(protocol, id.clone(), state.model_name.clone());
            let mut start = encoder.start(ids.len());
            if matches!(protocol, Protocol::Responses) {
                patch_response_events(&mut start, &request);
            }
            if !send(&tx, start) {
                return;
            }
            let result = generate(
                &state,
                &request,
                &ids,
                |delta| send(&tx, encoder.delta(&delta)),
                || tx.is_closed(),
            );
            match result {
                Ok(generation) => {
                    let mut events = encoder.finish(&generation, request.include_usage);
                    let persisted = if matches!(protocol, Protocol::Responses) {
                        events
                            .last()
                            .and_then(|last| serde_json::from_str::<Value>(&last.data).ok())
                            .and_then(|event| event.get("response").cloned())
                            .ok_or_else(|| {
                                "Responses stream has no final response object".to_string()
                            })
                            .and_then(|value| store_response(&state, &id, &request, value))
                    } else {
                        Ok(Value::Null)
                    };
                    match persisted {
                        Ok(value) => {
                            if matches!(protocol, Protocol::Responses) {
                                if let Some(last) = events.last_mut() {
                                    if let Ok(mut event) = serde_json::from_str::<Value>(&last.data)
                                    {
                                        event["response"] = value;
                                        last.data = event.to_string();
                                    }
                                }
                            }
                            send(&tx, events);
                        }
                        Err(e) => {
                            send(&tx, encoder.failure(&e));
                        }
                    }
                }
                Err(e) => {
                    if !tx.is_closed() {
                        send(&tx, encoder.failure(&e));
                    }
                }
            }
        });
        return Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx))
            .keep_alive(KeepAlive::default())
            .into_response();
    }
    let cancelled = Arc::new(AtomicBool::new(false));
    let mut cancellation_guard = CancellationGuard::new(cancelled.clone());
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let generation = generate(
            &state,
            &request,
            &ids,
            |_| true,
            || cancelled.load(Ordering::Relaxed),
        )?;
        make_response(&state, protocol, &id, &request, &generation)
    })
    .await;
    cancellation_guard.disarm();
    match result {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(e)) => error(protocol, 500, e),
        Err(e) => error(protocol, 500, format!("Generation worker failed: {e}")),
    }
}

async fn count_tokens(
    State(state): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let Json(mut body) = match body {
        Ok(b) => b,
        Err(e) => return error(Protocol::Anthropic, e.status().as_u16(), e.body_text()),
    };
    if !body.is_object() {
        return error(Protocol::Anthropic, 400, "Expected a JSON object");
    }
    // Count-token requests do not have generation options.
    body["max_tokens"] = Value::from(1);
    let mut request = match Protocol::Anthropic.parse(&body) {
        Ok(r) => r,
        Err(e) => return error(Protocol::Anthropic, 400, e),
    };
    if let Err((status, e)) = resolve(&state, Protocol::Anthropic, &mut request) {
        return error(Protocol::Anthropic, status, e);
    }
    match prompt(&state, &request).await {
        Ok(ids) => Json(serde_json::json!({"input_tokens":ids.len()})).into_response(),
        Err((status, e)) => error(Protocol::Anthropic, status, e),
    }
}

async fn get_response(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match state.responses.lock() {
        Ok(mut store) => match store.get(&id) {
            Some((_, value, _)) => Json(value).into_response(),
            None => error(
                Protocol::Responses,
                404,
                "Response was not stored, expired, or was evicted",
            ),
        },
        Err(e) => error(Protocol::Responses, 500, e.to_string()),
    }
}
async fn delete_response(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match state.responses.lock() {
        Ok(mut store) => {
            if store.remove(&id) {
                Json(serde_json::json!({"id":id,"object":"response.deleted","deleted":true}))
                    .into_response()
            } else {
                error(
                    Protocol::Responses,
                    404,
                    "Response was not stored, expired, or was evicted",
                )
            }
        }
        Err(e) => error(Protocol::Responses, 500, e.to_string()),
    }
}

fn generate(
    state: &AppState,
    request: &Request,
    ids: &[u32],
    mut emit: impl FnMut(Delta) -> bool,
    cancelled: impl Fn() -> bool,
) -> Result<Generation, String> {
    let Backend::Text(text) = state.model.as_ref() else {
        return Err("Server is not running a text model".into());
    };
    if cancelled() {
        return Err("Client disconnected".into());
    }
    let mut parser = tools::OutputParser::new(&text.arch, &request.tools, &request.choice);
    let mut stop = stop::StopFilter::new(request.stop.clone());
    let mut callback_error = None;
    let mut stopped = false;
    let mut on_token = |chunk: &str| -> bool {
        if stopped || cancelled() {
            stopped = true;
            return false;
        }
        let output = stop.push(chunk);
        match parser.push(&output) {
            Ok(deltas) => {
                for delta in deltas {
                    if !emit(delta) {
                        callback_error = Some("Client disconnected".into());
                        stopped = true;
                        return false;
                    }
                }
            }
            Err(e) => {
                callback_error = Some(e);
                stopped = true;
                return false;
            }
        }
        stopped = stop.hit.is_some();
        !stopped
    };
    let completion_tokens = match &text.inner {
        TextInner::Qwen3 { model } => {
            let mut session = super::Qwen3Session::new_with_kv_state(
                model,
                ids.len() + request.max_tokens,
                super::KvFormat::F16,
                super::KvLifecycle::Ephemeral,
            )?;
            let positions: Vec<_> = (0..ids.len()).map(|i| [i, 0, 0, 0]).collect();
            let generation = session.generate_streaming_until(
                super::Qwen3Input {
                    token_ids: ids,
                    positions: &positions,
                    embeddings: None,
                    deepstack_embeddings: None,
                },
                super::Qwen3GenerateOptions {
                    max_new_tokens: request.max_tokens,
                    temperature: request.temperature,
                    prefill_batch_size: text.prefill_batch_size,
                },
                1.0,
                &mut on_token,
            )?;
            generation.token_ids.len()
        }
        TextInner::Qwen35 { model, .. } => {
            let mut model = model.lock().map_err(|e| e.to_string())?;
            if cancelled() {
                return Err("Client disconnected".into());
            }
            let (positions, _) = super::build_qwen35_positions(ids, None, &[])?;
            let mut session = super::Qwen35Session::new_with_prefill_batch_size(
                &mut model,
                ids.len() + request.max_tokens,
                text.prefill_batch_size,
                text.pool.clone(),
            )?;
            let mut decoder = text.tokenizer.streaming_decoder(false);
            let mut generated = Vec::new();
            for step in 0..request.max_tokens {
                if cancelled() {
                    return Err("Client disconnected".into());
                }
                let pos = session.next_position();
                let decode_positions = [[pos, pos, pos, 0]];
                let (tokens, positions) = if step == 0 {
                    (ids, &positions[..])
                } else {
                    (&generated[generated.len() - 1..], &decode_positions[..])
                };
                let logits = session.step_with_tokens(tokens, positions)?;
                let id = u32::try_from(super::sample_token_from_logits(
                    &logits,
                    request.temperature,
                ))
                .map_err(|e| e.to_string())?;
                if text.tokenizer.eos_id() == Some(id)
                    || text.tokenizer.special_token_id("im_end") == Some(id)
                {
                    break;
                }
                generated.push(id);
                if !on_token(&decoder.push(id)) {
                    break;
                }
            }
            let tail = decoder.finish();
            if !tail.is_empty() {
                on_token(&tail);
            }
            generated.len()
        }
        TextInner::Lfm2Moe { session, .. } => {
            let mut session = session.lock().map_err(|e| e.to_string())?;
            session.reset();
            if cancelled() {
                return Err("Client disconnected".into());
            }
            let mut decoder = text.tokenizer.streaming_decoder(false);
            let mut generated = Vec::new();
            let eos = text.tokenizer.eos_id();

            // Prefill: feed prompt tokens one by one, reusing KV cache.
            let mut logits = Vec::new();
            for &token_id in ids {
                logits = session.forward_token(token_id)?;
            }

            // Decode: generate new tokens one by one, reusing KV cache.
            for _step in 0..request.max_tokens {
                if cancelled() {
                    return Err("Client disconnected".into());
                }
                let id = u32::try_from(super::sample_token_from_logits(
                    &logits,
                    request.temperature,
                ))
                .map_err(|e| e.to_string())?;
                if eos == Some(id) {
                    break;
                }
                generated.push(id);
                if !on_token(&decoder.push(id)) {
                    break;
                }
                logits = session.forward_token(id)?;
            }
            let tail = decoder.finish();
            if !tail.is_empty() {
                on_token(&tail);
            }
            generated.len()
        }
        TextInner::Fallback { arch } => {
            return Err(format!("Architecture {arch:?} is unsupported"))
        }
    };
    if let Some(error) = callback_error {
        return Err(error);
    }
    if cancelled() {
        return Err("Client disconnected".into());
    }
    for delta in parser.push(&stop.finish())? {
        if !emit(delta) {
            return Err("Client disconnected".into());
        }
    }
    for delta in parser.finish()? {
        if !emit(delta) {
            return Err("Client disconnected".into());
        }
    }
    let finish = if let Some(sequence) = stop.hit {
        Finish::Sequence(sequence)
    } else if completion_tokens >= request.max_tokens {
        Finish::Limit
    } else if !parser.calls().is_empty() {
        Finish::Tools
    } else {
        Finish::Stop
    };
    Ok(Generation {
        text: parser.text().into(),
        calls: parser.calls().to_vec(),
        usage: Usage {
            input: ids.len(),
            output: completion_tokens,
        },
        finish,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn message() -> Message {
        Message {
            role: "user".into(),
            text: "hello".into(),
            calls: vec![],
            call_id: None,
        }
    }
    #[test]
    fn response_history_survives_lookup_and_eviction() {
        let mut store = ResponsesStore::new(1, 4096, Duration::from_secs(30));
        store
            .insert(
                "r1".into(),
                "local".into(),
                json!({"id":"r1"}),
                vec![message()],
            )
            .unwrap();
        let (model, value, messages) = store.get("r1").unwrap();
        assert_eq!(model, "local");
        assert_eq!(value["id"], "r1");
        assert_eq!(messages[0].text, "hello");
        store
            .insert(
                "r2".into(),
                "local".into(),
                json!({"id":"r2"}),
                vec![message()],
            )
            .unwrap();
        assert!(store.get("r1").is_none());
        assert!(store.remove("r2"));
        assert!(!store.remove("r2"));
        assert_eq!(store.bytes, 0);
    }
    #[test]
    fn tool_results_are_ordered_by_their_calls() {
        let calls = Message {
            role: "assistant".into(),
            text: String::new(),
            calls: vec![
                protocol::ToolCall {
                    id: "a".into(),
                    name: "f".into(),
                    arguments: serde_json::json!({"city":"北京"}),
                },
                protocol::ToolCall {
                    id: "b".into(),
                    name: "f".into(),
                    arguments: serde_json::json!({"city":"上海"}),
                },
            ],
            call_id: None,
        };
        let result = |id: &str, text: &str| Message {
            role: "tool".into(),
            text: text.into(),
            calls: vec![],
            call_id: Some(id.into()),
        };
        let mut messages = vec![
            message(),
            calls,
            result("b", "上海结果"),
            result("a", "北京结果"),
        ];
        order_tool_results(&mut messages);
        assert_eq!(messages[2].call_id.as_deref(), Some("a"));
        assert_eq!(messages[3].call_id.as_deref(), Some("b"));
    }
    #[test]
    fn responses_parallel_items_correlate_with_results() {
        let call = |id: &str| Message {
            role: "assistant".into(),
            text: String::new(),
            calls: vec![protocol::ToolCall {
                id: id.into(),
                name: "f".into(),
                arguments: serde_json::json!({}),
            }],
            call_id: None,
        };
        let result = |id: &str| Message {
            role: "tool".into(),
            text: "ok".into(),
            calls: vec![],
            call_id: Some(id.into()),
        };
        assert!(
            check_history(&[message(), call("a"), call("b"), result("a"), result("b")]).is_ok()
        );
        assert!(check_history(&[message(), call("a"), result("other")]).is_err());
        assert!(check_history(&[message(), call("a"), result("a"), result("a")]).is_err());
    }
    #[test]
    fn expired_and_oversize_responses_are_not_retained() {
        let mut store = ResponsesStore::new(2, 4096, Duration::ZERO);
        store
            .insert("r1".into(), "local".into(), json!({}), vec![message()])
            .unwrap();
        assert!(store.get("r1").is_none());
        assert_eq!(store.bytes, 0);
        let mut store = ResponsesStore::new(2, 8, Duration::from_secs(30));
        assert!(store
            .insert("r1".into(), "local".into(), json!({}), vec![message()])
            .is_err());
    }
}

#[cfg(test)]
mod http_tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        extract::FromRequest,
    };
    fn state() -> AppState {
        AppState {
            model: std::sync::Arc::new(super::super::Backend::Text(super::super::TextBackend {
                arch: "unimplemented".into(),
                pool: std::sync::Arc::new(super::super::ComputePool::new(1)),
                tokenizer: std::sync::Arc::new(
                    super::super::BPETokenizer::from_qwen3_embedded_merges().unwrap(),
                ),
                prefill_batch_size: 1,
                context_length: 1024,
                inner: super::super::TextInner::Fallback {
                    arch: "unimplemented".into(),
                },
            })),
            model_name: "local".into(),
            responses: std::sync::Arc::new(std::sync::Mutex::new(ResponsesStore::default())),
            generation_slot: std::sync::Arc::new(tokio::sync::Semaphore::new(1)),
        }
    }
    #[test]
    fn response_request_metadata_is_reflected() {
        let request = Request {
            model: Some("local".into()),
            messages: vec![Message {
                role: "user".into(),
                text: "x".into(),
                calls: vec![],
                call_id: None,
            }],
            tools: vec![protocol::Tool {
                name: "f".into(),
                description: Some("d".into()),
                parameters: serde_json::json!({"type":"object"}),
            }],
            choice: protocol::ToolChoice::Named("f".into()),
            max_tokens: 77,
            temperature: 0.25,
            stream: false,
            include_usage: false,
            stop: vec![],
            store: false,
            previous_response_id: Some("prev".into()),
            instructions: Some("system".into()),
        };
        let mut value = serde_json::json!({"object":"response"});
        apply_response_request(&mut value, &request);
        assert_eq!(value["max_output_tokens"], 77);
        assert_eq!(value["temperature"], 0.25);
        assert_eq!(value["tools"][0]["name"], "f");
        assert_eq!(value["tool_choice"]["name"], "f");
        assert_eq!(value["previous_response_id"], "prev");
        assert_eq!(value["instructions"], "system");
        assert_eq!(value["store"], false);
    }
    #[test]
    fn response_store_preserves_streamed_output_order() {
        let state = state();
        let request = Request {
            model: None,
            messages: vec![Message {
                role: "user".into(),
                text: "x".into(),
                calls: vec![],
                call_id: None,
            }],
            tools: vec![],
            choice: protocol::ToolChoice::None,
            max_tokens: 1,
            temperature: 0.0,
            stream: true,
            include_usage: false,
            stop: vec![],
            store: true,
            previous_response_id: None,
            instructions: None,
        };
        let value = serde_json::json!({"id":"resp_order","object":"response","output":[{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"message","content":[]}]});
        let stored = store_response(&state, "resp_order", &request, value).unwrap();
        assert_eq!(stored["output"][0]["type"], "function_call");
        let (_, retrieved, messages) = state.responses.lock().unwrap().get("resp_order").unwrap();
        assert_eq!(retrieved["output"][0]["type"], "function_call");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].calls[0].id, "c");
    }
    #[test]
    fn dropping_cancellation_guard_sets_flag() {
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let _guard = CancellationGuard::new(flag.clone());
        }
        assert!(flag.load(std::sync::atomic::Ordering::Relaxed));
    }
    #[tokio::test]
    async fn invalid_json_and_missing_model_have_protocol_error_bodies() {
        let req = axum::http::Request::builder()
            .header("content-type", "application/json")
            .body(Body::from("{"))
            .unwrap();
        let body = Json::<Value>::from_request(req, &()).await;
        let response = handle(state(), Protocol::Anthropic, body).await;
        assert_eq!(response.status(), 400);
        let value: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(value["type"], "error");
        assert_eq!(value["error"]["type"], "invalid_request_error");
        let response = handle(
            state(),
            Protocol::Chat,
            Ok(Json(
                serde_json::json!({"model":"missing","messages":[{"role":"user","content":"Hi"}]}),
            )),
        )
        .await;
        assert_eq!(response.status(), 404);
        let value: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("missing"));
    }
}

// ============================================================================
// JEV scoring endpoints
//
// `/v1/jev/score` and `/v1/jev/grouped` expose the CLI `--jev` mode as
// HTTP. The server already has a text-backend `Qwen3` model loaded;
// these handlers call the existing `run_jev_decision_data` /
// `run_jev_grouped_decision_data` (which internally rebuild a fresh
// `Qwen3Session` per question for ephemeral KV cache) on the same
// `TensorSource`.
//
// Two key differences from the chat completions path:
//   - Single forward pass per question, then argmax over label tokens
//     (A/B/C/...) — not autoregressive generation. This gives a true
//     decision score rather than a "max_tokens=1 hack".
//   - Same `TensorSource` shared with the chat path; weight memory is
//     not duplicated.
//
// Single-mode request (text scoring / multi-choice / binary / score):
//
//   POST /v1/jev/score
//   {
//     "context": "The quick brown fox jumps over the lazy dog.",
//     "questions": [
//       {"text": "Is this text well-written?", "options": ["yes", "no"]},
//       {"text": "Which sentence is this?",
//        "options": ["London", "Paris", "Berlin"]}
//     ],
//     "positive": "yes"     // optional, for binary mode
//   }
//
// Grouped (multi-select / block-choice):
//
//   POST /v1/jev/grouped
//   {
//     "context": "An apple on a wooden table.",
//     "questions": [
//       {"text": "Which way should the camera move?",
//        "groups": [
//          {"label": "left",   "options": ["yes", "no"]},
//          {"label": "right",  "options": ["yes", "no"]},
//          {"label": "up",     "options": ["yes", "no"]},
//          {"label": "down",   "options": ["yes", "no"]}
//        ]}
//     ],
//     "mode": "multi_select"   // or "block_choice"
//   }

#[derive(serde::Deserialize)]
struct JevOptionInput {
    text: String,
    options: Vec<String>,
}

#[derive(serde::Deserialize)]
struct JevGroupInputHttp {
    label: String,
    options: Vec<String>,
}

#[derive(serde::Deserialize)]
struct JevGroupedOptionInput {
    text: String,
    groups: Vec<JevGroupInputHttp>,
}

#[derive(serde::Deserialize)]
struct JevScoreRequest {
    #[serde(default)]
    context: String,
    questions: Vec<JevOptionInput>,
    #[serde(default)]
    positive: Option<String>,
}

#[derive(serde::Deserialize)]
struct JevGroupedRequest {
    #[serde(default)]
    context: String,
    questions: Vec<JevGroupedOptionInput>,
    /// `"multi_select"` or `"block_choice"`. Maps to `JevMode`.
    #[serde(default = "default_grouped_mode")]
    mode: String,
}

fn default_grouped_mode() -> String {
    "multi_select".to_string()
}

async fn jev_score(
    State(state): State<AppState>,
    body: Result<Json<JevScoreRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return jev_error(StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")),
    };
    if req.questions.is_empty() {
        return jev_error(
            StatusCode::BAD_REQUEST,
            "questions must contain at least one item".to_string(),
        );
    }
    let source = match text_source(&state) {
        Ok(s) => s,
        Err(e) => return jev_error(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let questions: Vec<crate::app::JevQuestionInput> = req
        .questions
        .into_iter()
        .map(|q| crate::app::JevQuestionInput {
            text: q.text,
            options: q.options,
        })
        .collect();
    let positive = req.positive.as_deref();

    let threads = jev_threads(&state);
    let prefill_batch_size = jev_prefill_batch_size(&state);

    let results = match crate::app::run_jev_decision_data(
        source,
        &req.context,
        &questions,
        positive,
        threads,
        prefill_batch_size,
    ) {
        Ok(r) => r,
        Err(e) => return jev_error(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    Json(json!({
        "mode": "single",
        "context": req.context,
        "results": results,
    }))
    .into_response()
}

async fn jev_grouped(
    State(state): State<AppState>,
    body: Result<Json<JevGroupedRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return jev_error(StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")),
    };
    if req.questions.is_empty() {
        return jev_error(
            StatusCode::BAD_REQUEST,
            "questions must contain at least one item".to_string(),
        );
    }
    let mode = match req.mode.as_str() {
        "multi_select" => crate::app::JevMode::MultiSelect,
        "block_choice" => crate::app::JevMode::BlockChoice,
        other => {
            return jev_error(
                StatusCode::BAD_REQUEST,
                format!("mode must be 'multi_select' or 'block_choice', got {other:?}"),
            );
        }
    };
    let source = match text_source(&state) {
        Ok(s) => s,
        Err(e) => return jev_error(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let questions: Vec<crate::app::JevGroupedQuestionInput> = req
        .questions
        .into_iter()
        .map(|q| crate::app::JevGroupedQuestionInput {
            text: q.text,
            groups: q
                .groups
                .into_iter()
                .map(|g| crate::app::JevGroupInput {
                    label: g.label,
                    options: g.options,
                })
                .collect(),
        })
        .collect();

    let threads = jev_threads(&state);
    let prefill_batch_size = jev_prefill_batch_size(&state);

    let results = match crate::app::run_jev_grouped_decision_data(
        source,
        &req.context,
        &questions,
        mode,
        threads,
        prefill_batch_size,
    ) {
        Ok(r) => r,
        Err(e) => return jev_error(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    Json(json!({
        "mode": match mode {
            crate::app::JevMode::MultiSelect => "multi_select",
            crate::app::JevMode::BlockChoice => "block_choice",
            _ => "unknown",
        },
        "context": req.context,
        "results": results,
    }))
    .into_response()
}

fn text_source(state: &AppState) -> Result<Arc<dyn crate::core::tensor::TensorSource>, String> {
    match state.model.as_ref() {
        Backend::Text(text) => Ok(Arc::clone(&text.source)),
        other => Err(format!(
            "/v1/jev/* requires a text backend (got {})",
            backend_label(other)
        )),
    }
}

fn backend_label(b: &Backend) -> &'static str {
    match b {
        Backend::Text(_) => "text",
        Backend::Embedding(_) => "embedding",
        Backend::Asr(_) => "asr",
        Backend::Tts(_) => "tts",
        Backend::Rerank(_) => "rerank",
    }
}

fn jev_threads(state: &AppState) -> usize {
    if let Backend::Text(text) = state.model.as_ref() {
        text.pool.n_threads()
    } else {
        1
    }
}

fn jev_prefill_batch_size(state: &AppState) -> usize {
    if let Backend::Text(text) = state.model.as_ref() {
        text.prefill_batch_size
    } else {
        64
    }
}

fn jev_error(status: StatusCode, message: String) -> Response {
    (
        status,
        Json(json!({"error": {"message": message, "type": "invalid_request_error"}})),
    )
        .into_response()
}

// ============================================================================
// Multimodal JEV endpoints (`/v1/jev/image`, `/v1/jev/image_grouped`)
//
// Pragmatic implementation: Qwen3.5 + CLIP/Qwen2.5-Omni multimodal does
// not yet have a logits-only forward pass (only `model.generate()`),
// so true argmax-over-labels scoring is unavailable. We approximate
// it with `max_new_tokens=1` generation: the model sees a `<image>` +
// "Question: ... A. opt1 B. opt2 ... Answer:" prompt and produces
// the most-likely label token as the first generated token. The
// returned `score` / `probs` are derived from the post-generation
// char-matches, not from logits — so the JSON mirrors the text JEV
// shape but the underlying inference is autoregressive.
//
// For grouped (`/v1/jev/image_grouped`), each group is its own
// forward pass with a separate prompt (`Should camera move left?
// yes/no`), so N groups = N forward passes per request. The 0.8B
// model on this 5-second machine takes ~1.4s per pass, so a 4-axis
// request is ~5-6s end-to-end.
//
// Both endpoints require `--mmproj` at server startup; without it
// they return 400 with a clear error message.

#[derive(serde::Deserialize)]
struct JevImageScoreRequest {
    /// Either `context` (string) or `image_url` (data URL or http URL).
    /// If absent, defaults to the same `context` field as text JEV.
    #[serde(default)]
    context: String,
    /// `data:image/png;base64,...` or `https://...`. Decoded and
    /// written to a temp file under the OS temp dir.
    image_url: String,
    questions: Vec<JevOptionInput>,
    #[serde(default)]
    positive: Option<String>,
    /// Default 0.7. Recorded for response metadata; multimodal JEV
    /// uses argmax over label logits so the value does not affect
    /// scoring itself.
    #[serde(default = "default_multimodal_temperature")]
    temperature: f32,
}

async fn jev_image_score(
    State(state): State<AppState>,
    body: Result<Json<JevImageScoreRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return jev_error(StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")),
    };
    if req.image_url.is_empty() {
        return jev_error(
            StatusCode::BAD_REQUEST,
            "image_url is required for /v1/jev/image".to_string(),
        );
    }
    if req.questions.is_empty() {
        return jev_error(
            StatusCode::BAD_REQUEST,
            "questions must contain at least one item".to_string(),
        );
    }

    let image_path = match decode_image_to_tempfile(&req.image_url) {
        Ok(p) => p,
        Err(e) => return jev_error(StatusCode::BAD_REQUEST, e),
    };
    let result = run_image_jev_blocking(
        state,
        image_path.clone(),
        req.context.clone(),
        req.questions,
        req.positive,
        req.temperature,
    )
    .await;
    // Best-effort cleanup of the temp file; if `run_image_jev_blocking`
    // returned early via an error path the file is still dropped on
    // process exit because we used `NamedTempFile`-equivalent paths.
    let _ = std::fs::remove_file(&image_path);

    match result {
        Ok(results) => Json(json!({
            "mode": "single",
            "context": req.context,
            "results": results,
        }))
        .into_response(),
        Err(e) => jev_error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

#[derive(serde::Deserialize)]
struct JevImageGroupedRequest {
    #[serde(default)]
    context: String,
    image_url: String,
    questions: Vec<JevGroupedOptionInput>,
    /// `"multi_select"` or `"block_choice"`. Maps to `JevMode`.
    #[serde(default = "default_grouped_mode")]
    mode: String,
    /// Default 0.7. Recorded for response metadata; multimodal JEV
    /// uses argmax over label logits so the value does not affect
    /// scoring itself.
    #[serde(default = "default_multimodal_temperature")]
    temperature: f32,
}

fn default_multimodal_temperature() -> f32 {
    0.7
}

async fn jev_image_grouped(
    State(state): State<AppState>,
    body: Result<Json<JevImageGroupedRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return jev_error(StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")),
    };
    if req.image_url.is_empty() {
        return jev_error(
            StatusCode::BAD_REQUEST,
            "image_url is required for /v1/jev/image_grouped".to_string(),
        );
    }
    if req.questions.is_empty() {
        return jev_error(
            StatusCode::BAD_REQUEST,
            "questions must contain at least one item".to_string(),
        );
    }
    let mode_label = match req.mode.as_str() {
        "multi_select" => "multi_select",
        "block_choice" => "block_choice",
        other => {
            return jev_error(
                StatusCode::BAD_REQUEST,
                format!("mode must be 'multi_select' or 'block_choice', got {other:?}"),
            );
        }
    };

    let image_path = match decode_image_to_tempfile(&req.image_url) {
        Ok(p) => p,
        Err(e) => return jev_error(StatusCode::BAD_REQUEST, e),
    };

    let grouped_result = run_image_grouped_jev_blocking(
        state,
        image_path.clone(),
        req.context.clone(),
        req.questions,
        req.mode.clone(),
        req.temperature,
    )
    .await;
    let _ = std::fs::remove_file(&image_path);

    match grouped_result {
        Ok(payload) => Json(json!({
            "context": req.context,
            "results": payload.get("results").cloned().unwrap_or(json!([])),
            "mode": payload.get("mode").and_then(|v| v.as_str()).unwrap_or("multi_select"),
            "temperature": payload.get("temperature").cloned().unwrap_or(json!(0.7)),
        }))
        .into_response(),
        Err(e) => jev_error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

/// Per-question image JEV: build a multimodal chat prompt whose
/// response is constrained to A..Z letter labels, run multimodal
/// logits-only forward, and apply softmax + argmax over the
/// per-label token logits. Mirrors the text-only `JEVResult` shape
/// so the response can be consumed uniformly. Temperature is
/// accepted in the request body but does not affect argmax-based
/// scoring; it is recorded for the response metadata only.
async fn run_image_jev_blocking(
    state: AppState,
    image_path: std::path::PathBuf,
    context: String,
    questions: Vec<JevOptionInput>,
    positive: Option<String>,
    temperature: f32,
) -> Result<Vec<serde_json::Value>, String> {
    let tokenizer = text_tokenizer(&state).ok_or_else(|| {
        "/v1/jev/image requires a text backend with a loaded tokenizer".to_string()
    })?;
    let mut results = Vec::with_capacity(questions.len());
    for q in questions {
        let n_options = q.options.len();
        if !(2..=26).contains(&n_options) {
            return Err(format!(
                "Question {:?} has {} options; multimodal JEV supports 2..=26 (A..Z)",
                q.text, n_options
            ));
        }
        let labels: Vec<char> = (b'A'..=(b'A' + n_options as u8 - 1))
            .map(|b| b as char)
            .collect();
        // Reuse the CLI scorer's preparation verbatim: same mode
        // selection, same system wording and, critically, the same payload
        // bytes. `serde_json::json!` orders `candidates` alphabetically while
        // `jev_payload_json` keeps insertion order, and that difference alone
        // shifts the token ids enough to move the label logits.
        let prepared = crate::app::prepare_jev_questions(
            &[crate::app::JevQuestionInput {
                text: q.text.clone(),
                options: q.options.clone(),
            }],
            positive.as_deref(),
        )?
        .into_iter()
        .next()
        .ok_or("multimodal JEV produced no prepared question")?;
        let system_prompt = crate::app::jev_system_prompt(prepared.mode);
        // The user turn holds only the scored payload; `system_prompt` is
        // forwarded separately so the multimodal helper renders it as its own
        // system turn. Folding the two into one string here would put the
        // instructions in the user turn, which is what this endpoint used to do
        // and which scored worse than the CLI.
        let user_payload = crate::app::jev_payload_json(&context, &prepared)?;
        let logits = run_multimodal_text_only(
            state.clone(),
            image_path.clone(),
            user_payload,
            system_prompt.to_string(),
        )
        .await?;
        let label_token_ids: Vec<u32> = labels
            .iter()
            .map(|l| {
                let s = l.to_string();
                tokenizer
                    .encode(
                        &s,
                        crate::core::tokenizer::EncodeOptions {
                            add_special: false,
                            parse_special: false,
                        },
                    )
                    .into_iter()
                    .next()
                    .unwrap_or(0)
            })
            .collect();
        let label_logits: Vec<f32> = label_token_ids
            .iter()
            .map(|&id| {
                logits
                    .get(id as usize)
                    .copied()
                    .unwrap_or(f32::NEG_INFINITY)
            })
            .collect();
        let probs = softmax(&label_logits);
        let chosen_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let chosen = labels[chosen_idx];
        let confidence = probs[chosen_idx];
        let entropy = -probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f32>();
        let second_best = probs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != chosen_idx)
            .map(|(_, &p)| p)
            .fold(f32::NEG_INFINITY, f32::max);
        let margin = confidence - second_best;

        let probs_map: serde_json::Map<String, serde_json::Value> = labels
            .iter()
            .zip(probs.iter())
            .map(|(l, p)| (l.to_string(), serde_json::json!(p)))
            .collect();
        let mut obj = serde_json::json!({
            "mode": if prepared.mode == crate::app::JevMode::Binary {
                "binary"
            } else {
                "choice"
            },
            "question": q.text,
            "labels": labels.iter().map(|l| l.to_string()).collect::<Vec<_>>(),
            "descriptions": q.options,
            "probabilities": probs_map,
            "choice": chosen.to_string(),
            "chosen_index": chosen_idx,
            "confidence": confidence,
            "entropy": entropy,
            "margin": margin,
            "temperature": temperature,
            "method": "multimodal_logits_argmax",
        });
        if prepared.mode == crate::app::JevMode::Binary {
            let pos_ch = positive
                .as_deref()
                .unwrap()
                .chars()
                .next()
                .unwrap_or('A')
                .to_ascii_uppercase();
            let pos_idx = labels.iter().position(|l| *l == pos_ch).unwrap_or(0);
            obj.as_object_mut().unwrap().insert(
                "positive".to_string(),
                serde_json::Value::String(pos_ch.to_string()),
            );
            obj.as_object_mut()
                .unwrap()
                .insert("probability".to_string(), serde_json::json!(probs[pos_idx]));
        }
        results.push(obj);
    }
    Ok(results)
}

/// Logits-only multimodal grouped JEV: build a prompt listing every
/// group's options (labels are continuous A..Z across groups, like
/// the CLI's text JEV grouped path), run multimodal forward once,
/// and apply per-group softmax over the label token logits.
async fn run_image_grouped_jev_blocking(
    state: AppState,
    image_path: std::path::PathBuf,
    context: String,
    grouped_questions: Vec<JevGroupedOptionInput>,
    mode: String,
    temperature: f32,
) -> Result<serde_json::Value, String> {
    let tokenizer = text_tokenizer(&state).ok_or_else(|| {
        "/v1/jev/image_grouped requires a text backend with a loaded tokenizer".to_string()
    })?;
    let mode_label = match mode.as_str() {
        "multi_select" => "multi_select",
        "block_choice" => "block_choice",
        other => {
            return Err(format!(
                "mode must be 'multi_select' or 'block_choice', got {other:?}"
            ));
        }
    };
    let mut results = Vec::with_capacity(grouped_questions.len());
    for q in grouped_questions {
        if q.groups.is_empty() {
            return Err(format!(
                "Question {:?} has no groups; provide at least one group",
                q.text
            ));
        }
        let mut total_options = 0usize;
        let mut next_label: u8 = b'A';
        let mut all_group_labels: Vec<Vec<char>> = Vec::with_capacity(q.groups.len());
        for g in q.groups.iter() {
            if g.options.len() < 2 {
                return Err(format!("Group {:?} needs at least 2 options", g.label));
            }
            total_options += g.options.len();
            if total_options > 26 {
                return Err(format!(
                    "Question {:?} exceeds 26 total options (current: {})",
                    q.text, total_options
                ));
            }
            // Labels are continuous A..Z across groups (matching the
            // CLI's text JEV grouped path). The model sees a prompt
            // listing all groups and emits one label per group in
            // sequence, so the logits at the LAST position encode
            // per-group preferences via the shared label-token
            // vocabulary. Per-group softmax normalises independently.
            let n = g.options.len() as u8;
            let group_letters: Vec<char> =
                (next_label..(next_label + n)).map(|b| b as char).collect();
            next_label += n;
            all_group_labels.push(group_letters);
        }
        // Reuse the CLI grouped scorer's preparation and payload builder so
        // both paths emit identical bytes; `serde_json::json!` here ordered the
        // candidates alphabetically, which shifted the token ids versus the CLI.
        let prepared = crate::app::prepare_jev_grouped_questions(&[
            crate::app::JevGroupedQuestionInput {
                text: q.text.clone(),
                groups: q
                    .groups
                    .iter()
                    .map(|g| crate::app::JevGroupInput {
                        label: g.label.clone(),
                        options: g.options.clone(),
                    })
                    .collect(),
            },
        ], match mode_label {
            "multi_select" => crate::app::JevMode::MultiSelect,
            _ => crate::app::JevMode::BlockChoice,
        })?
        .into_iter()
        .next()
        .ok_or("multimodal grouped JEV produced no prepared question")?;
        let system_prompt = crate::app::build_grouped_system().to_string();
        let user_payload = crate::app::build_grouped_payload(&context, &prepared)?;
        let logits = run_multimodal_text_only(
            state.clone(),
            image_path.clone(),
            user_payload,
            system_prompt,
        )
        .await?;
        let mut group_results = Vec::with_capacity(q.groups.len());
        for (gi, group) in q.groups.iter().enumerate() {
            let labels = &all_group_labels[gi];
            let label_token_ids: Vec<u32> = labels
                .iter()
                .map(|l| {
                    let s = l.to_string();
                    tokenizer
                        .encode(
                            &s,
                            crate::core::tokenizer::EncodeOptions {
                                add_special: false,
                                parse_special: false,
                            },
                        )
                        .into_iter()
                        .next()
                        .unwrap_or(0)
                })
                .collect();
            let group_logits: Vec<f32> = label_token_ids
                .iter()
                .map(|&id| {
                    logits
                        .get(id as usize)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY)
                })
                .collect();
            let probs = softmax(&group_logits);
            let chosen_idx = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let chosen = labels[chosen_idx];
            let confidence = probs[chosen_idx];
            let entropy = -probs
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p * p.ln())
                .sum::<f32>();
            let second_best = probs
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != chosen_idx)
                .map(|(_, &p)| p)
                .fold(f32::NEG_INFINITY, f32::max);
            let margin = confidence - second_best;
            let probs_map: serde_json::Map<String, serde_json::Value> = labels
                .iter()
                .zip(probs.iter())
                .map(|(l, p)| (l.to_string(), serde_json::json!(p)))
                .collect();
            group_results.push(serde_json::json!({
                "label": group.label,
                "options": group.options,
                "labels": labels.iter().map(|l| l.to_string()).collect::<Vec<_>>(),
                "choice": chosen.to_string(),
                "probabilities": probs_map,
                "confidence": confidence,
                "entropy": entropy,
                "margin": margin,
            }));
        }
        results.push(serde_json::json!({
            "text": q.text,
            "groups": group_results,
        }));
    }
    Ok(serde_json::json!({
        "mode": mode_label,
        "results": results,
        "temperature": temperature,
    }))
}

/// Borrow the tokenizer out of the text backend's inner model so we
/// can encode "A"/"B"/... label strings into token ids for JEV
/// argmax scoring.
fn text_tokenizer(state: &AppState) -> Option<Arc<BPETokenizer>> {
    if let Backend::Text(text) = state.model.as_ref() {
        Some(Arc::clone(&text.tokenizer))
    } else {
        None
    }
}

/// Numerically-stable softmax for a slice of logits.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        // Fall back to uniform.
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Spawn a blocking task that runs multimodal generation and returns
/// the generated text. The underlying `run_qwen3_family_multimodal`
/// is sync (FFI-style); we offload it to the blocking pool so the
/// async runtime can keep serving other requests.
async fn run_multimodal_text_only(
    state: AppState,
    image_path: std::path::PathBuf,
    prompt: String,
    // Rendered into the chat template's own system turn. Passing it
    // separately keeps the JEV instructions out of the user content, matching
    // the CLI scorer; folding it into `prompt` puts it in the user turn instead.
    system_prompt: String,
) -> Result<Vec<f32>, String> {
    // Multimodal logits-only forward — dispatch by arch.
    let arch = match state.model.as_ref() {
        Backend::Text(text) => text.arch.clone(),
        other => {
            return Err(format!(
                "/v1/jev/image requires a text backend (got {})",
                backend_label(other)
            ));
        }
    };
    let source = match state.model.as_ref() {
        Backend::Text(text) => Arc::clone(&text.source),
        _ => unreachable!(),
    };
    let mmproj_path = match state.model.as_ref() {
        Backend::Text(text) => text.mmproj_path.clone().ok_or_else(|| {
            "/v1/jev/image requires the server to be started with --mmproj".to_string()
        })?,
        _ => unreachable!(),
    };
    let threads = jev_threads(&state);
    let prefill_batch_size = jev_prefill_batch_size(&state);

    tokio::task::spawn_blocking(move || match arch.as_str() {
        // Same arch set the CLI `--jev --image` gate accepts; see
        // `crate::app::jev::single::image_supported_arch`.
        "qwen3" | "qwen3vl" | "qwen3vlmoe" => crate::app::run_qwen3_family_multimodal_logits(
            source.as_ref(),
            source.clone(),
            mmproj_path.as_path(),
            Some(image_path.as_path()),
            None,
            None,
            &prompt,
            threads,
            prefill_batch_size,
            Some(system_prompt.as_str()),
        ),
        "qwen35" => {
            let max_context = match state_for_max_ctx(&state) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            crate::app::run_qwen35_family_multimodal_logits(
                source.as_ref(),
                mmproj_path.as_path(),
                Some(image_path.as_path()),
                None,
                None,
                &prompt,
                threads,
                prefill_batch_size,
                max_context,
                Some(system_prompt.as_str()),
            )
        }
        other => Err(format!(
            "/v1/jev/image only supports qwen3/qwen3vl/qwen35 multimodal, got {other:?}"
        )),
    })
    .await
    .map_err(|e| format!("multimodal join failed: {e}"))?
    .map(|(logits, _dur)| logits)
}

fn state_for_max_ctx(state: &AppState) -> Result<usize, String> {
    if let Backend::Text(text) = state.model.as_ref() {
        Ok(text.context_length)
    } else {
        Err("max-context unavailable for non-text backend".to_string())
    }
}

/// Decode an `image_url` (data URL or http URL) into a temp file on
/// disk. We need a real file path because the multimodal CLI helper
/// accepts `&Path` for the image. Returns the path; the caller is
/// responsible for cleanup (typically `std::fs::remove_file`).
fn decode_image_to_tempfile(image_url: &str) -> Result<std::path::PathBuf, String> {
    use std::io::Write;
    let bytes = if let Some(rest) = image_url.strip_prefix("data:") {
        // data:[<mediatype>];base64,<data>
        let comma = rest.find(',').ok_or("malformed data URL: missing comma")?;
        let (header, b64) = rest.split_at(comma);
        if !header.contains(";base64") {
            return Err("only base64 data URLs are supported (data:<mime>;base64,<data>)".into());
        }
        // strip leading ',' from b64
        base64_decode(&b64[1..])?
    } else if image_url.starts_with("http://") || image_url.starts_with("https://") {
        // For HTTP URLs we'd need a runtime fetch — punt for now since
        // test payloads use base64 data URLs.
        return Err(
            "http(s) image URLs are not supported; send base64 (data:image/png;base64,...)".into(),
        );
    } else if let Ok(rest) = base64_decode(image_url) {
        // Bare base64 (no data: prefix).
        rest
    } else {
        return Err("image_url must be a base64 data URL (data:image/png;base64,...)".into());
    };

    let suffix = detect_image_suffix(&bytes);
    let tmp = std::env::temp_dir().join(format!(
        "jev_image_{}.{suffix}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let mut f = std::fs::File::create(&tmp).map_err(|e| format!("create tmp file: {e}"))?;
    f.write_all(&bytes)
        .map_err(|e| format!("write tmp file: {e}"))?;
    drop(f);
    Ok(tmp)
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    // Minimal RFC4648 base64 decoder (whitespace-tolerant, no padding
    // required). Avoids pulling in a base64 crate.
    const TABLE: &[u8; 128] = &{
        let mut t = [255u8; 128];
        let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut i = 0;
        while i < alphabet.len() {
            t[alphabet[i] as usize] = i as u8;
            i += 1;
        }
        t
    };
    let cleaned: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    let mut out = Vec::with_capacity(cleaned.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &c in &cleaned {
        let v = if c < 128 { TABLE[c as usize] } else { 255 };
        if v == 255 {
            return Err(format!("invalid base64 char: {c:?}"));
        }
        buf = (buf << 6) | (v as u32);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xff) as u8);
        }
    }
    Ok(out)
}

fn detect_image_suffix(bytes: &[u8]) -> &'static str {
    if bytes.len() >= 8 && &bytes[..8] == b"\x89PNG\r\n\x1a\n" {
        "png"
    } else if bytes.len() >= 3 && &bytes[..3] == b"\xff\xd8\xff" {
        "jpg"
    } else if bytes.len() >= 4 && &bytes[..4] == b"GIF8" {
        "gif"
    } else {
        "bin"
    }
}
