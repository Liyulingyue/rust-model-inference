#[path = "api/protocol.rs"]
pub mod protocol;
#[path = "api/stop.rs"]
mod stop;
#[path = "api/tools.rs"]
pub mod tools;

use protocol::{Message, Request};
use serde_json::Value;
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
