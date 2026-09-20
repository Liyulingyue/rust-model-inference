use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug)]
pub enum Protocol {
    Chat,
    Anthropic,
    Responses,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub text: String,
    pub calls: Vec<ToolCall>,
    pub call_id: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Request {
    pub model: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
    pub choice: ToolChoice,
    pub max_tokens: usize,
    pub temperature: f32,
    pub stream: bool,
    pub include_usage: bool,
    pub stop: Vec<String>,
    pub store: bool,
    pub previous_response_id: Option<String>,
    pub instructions: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Usage {
    pub input: usize,
    pub output: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Finish {
    Stop,
    Limit,
    Sequence(String),
    Tools,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Generation {
    pub text: String,
    pub calls: Vec<ToolCall>,
    pub usage: Usage,
    pub finish: Finish,
}
#[derive(Clone, Debug)]
pub enum Delta {
    Text(String),
    ToolStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolArguments {
        index: usize,
        fragment: String,
    },
    ToolEnd {
        index: usize,
    },
}
#[derive(Clone, Debug)]
pub struct WireEvent {
    pub event: Option<&'static str>,
    pub data: String,
}

impl Protocol {
    pub fn parse(&self, body: &Value) -> Result<Request, String> {
        if !body.is_object() {
            return Err("request must be a JSON object".into());
        }
        validate_options(body, *self)?;
        let model = optional_string(body, "model")?;
        let previous_response_id = optional_string(body, "previous_response_id")?;
        if previous_response_id.is_some() && !matches!(self, Self::Responses) {
            return Err("previous_response_id is only supported by Responses".into());
        }
        let instructions = optional_string(body, "instructions")?;
        let mut messages = match self {
            Self::Chat => chat_messages(body)?,
            Self::Anthropic => anthropic_messages(body)?,
            Self::Responses => response_messages(body)?,
        };
        if messages.is_empty() {
            return Err("at least one input message is required".into());
        }
        validate_history(&messages, previous_response_id.is_some())?;
        let tools = parse_tools(body, *self)?;
        let choice = parse_choice(body, *self, &tools)?;
        let max_tokens = match self {
            Self::Responses => token_limit(body, &["max_output_tokens"]),
            _ => token_limit(body, &["max_completion_tokens", "max_tokens"]),
        }?;
        let temperature = match body.get("temperature").filter(|v| !v.is_null()) {
            None => 0.6,
            Some(v) => v.as_f64().ok_or("temperature must be a number")?,
        };
        let upper = if matches!(self, Self::Anthropic) {
            1.0
        } else {
            2.0
        };
        if !temperature.is_finite() || !(0.0..=upper).contains(&temperature) {
            return Err(format!(
                "temperature must be finite and between 0 and {upper}"
            ));
        }
        let stop_key = if matches!(self, Self::Anthropic) {
            "stop_sequences"
        } else {
            "stop"
        };
        let stop = parse_stop(body.get(stop_key))?;
        let stream = optional_bool(body, "stream", false)?;
        let include_usage = match body.get("stream_options").filter(|v| !v.is_null()) {
            None => false,
            Some(v) if v.is_object() => {
                if v.as_object().unwrap().keys().any(|k| k != "include_usage") {
                    return Err("unsupported stream_options field".into());
                }
                optional_bool(v, "include_usage", false)?
            }
            _ => return Err("stream_options must be an object".into()),
        };
        let store = optional_bool(body, "store", matches!(self, Self::Responses))?;
        // System instructions are already messages for Anthropic; Responses instructions
        // remain separate so the continuation layer can apply the current instruction.
        messages.shrink_to_fit();
        Ok(Request {
            model,
            messages,
            tools,
            choice,
            max_tokens,
            temperature: temperature as f32,
            stream,
            include_usage,
            stop,
            store,
            previous_response_id,
            instructions,
        })
    }

    pub fn response(&self, id: &str, model: &str, generation: &Generation) -> Value {
        match self {
            Self::Chat => {
                let mut message = json!({"role":"assistant", "content":generation.text});
                if !generation.calls.is_empty() {
                    message["tool_calls"] =
                        Value::Array(generation.calls.iter().map(chat_call).collect());
                    if generation.text.is_empty() {
                        message["content"] = Value::Null;
                    }
                }
                json!({"id":id,"object":"chat.completion","created":now(),"model":model,
                    "choices":[{"index":0,"message":message,"finish_reason":chat_finish(&generation.finish),"logprobs":null}],"usage":chat_usage(&generation.usage)})
            }
            Self::Anthropic => json!({"id":id,"type":"message","role":"assistant","model":model,
                "content":anthropic_content(generation),"stop_reason":anthropic_finish(&generation.finish),
                "stop_sequence":match &generation.finish { Finish::Sequence(s) => Some(s), _ => None },
                "usage":{"input_tokens":generation.usage.input,"output_tokens":generation.usage.output}}),
            Self::Responses => response_object(id, model, generation, now()),
        }
    }

    pub fn error(&self, status: u16, message: &str) -> Value {
        let kind = match status {
            401 => "authentication_error",
            403 => "permission_error",
            404 => "not_found_error",
            429 => "rate_limit_error",
            500..=599 => "api_error",
            _ => "invalid_request_error",
        };
        match self {
            Self::Anthropic => json!({"type":"error","error":{"type":kind,"message":message}}),
            _ => json!({"error":{"message":message,"type":kind,"param":null,"code":null}}),
        }
    }
}

fn optional_string(body: &Value, key: &str) -> Result<Option<String>, String> {
    match body.get(key).filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_owned()))
            .ok_or_else(|| format!("{key} must be a string")),
    }
}
fn required_string(body: &Value, key: &str) -> Result<String, String> {
    optional_string(body, key)?
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("{key} must be a nonempty string"))
}
fn optional_bool(body: &Value, key: &str, default: bool) -> Result<bool, String> {
    match body.get(key).filter(|v| !v.is_null()) {
        None => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| format!("{key} must be a boolean")),
    }
}
fn array<'a>(body: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    body.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} must be an array"))
}
fn token_limit(body: &Value, keys: &[&str]) -> Result<usize, String> {
    let mut limit = None;
    for &key in keys {
        if let Some(value) = body.get(key).filter(|v| !v.is_null()) {
            let value = value
                .as_u64()
                .filter(|&v| v > 0)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| {
                    format!(
                        "{key} must be a positive integer; prompt-cache prewarming is unsupported"
                    )
                })?;
            if limit.is_some_and(|old| old != value) {
                return Err("conflicting token limits".into());
            }
            limit = Some(value);
        }
    }
    Ok(limit.unwrap_or(512))
}
fn parse_stop(value: Option<&Value>) -> Result<Vec<String>, String> {
    let result = match value {
        None | Some(Value::Null) => vec![],
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or("stop entries must be strings".to_owned())
            })
            .collect::<Result<_, _>>()?,
        _ => return Err("stop must be a string or an array of strings".into()),
    };
    if result.iter().any(String::is_empty) {
        return Err("stop sequences cannot be empty".into());
    }
    Ok(result)
}
fn validate_options(body: &Value, protocol: Protocol) -> Result<(), String> {
    for (key, default) in [
        ("n", 1.0),
        ("top_p", 1.0),
        ("top_k", 0.0),
        ("presence_penalty", 0.0),
        ("frequency_penalty", 0.0),
    ] {
        if let Some(v) = body.get(key).filter(|v| !v.is_null()) {
            if v.as_f64() != Some(default) {
                return Err(format!(
                    "{key} is unsupported except for its default {default}"
                ));
            }
        }
    }
    for key in [
        "seed",
        "functions",
        "function_call",
        "include",
        "web_search_options",
        "prompt_cache_key",
        "prompt_cache_retention",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "audio",
        "modalities",
        "prediction",
        "service_tier",
        "context_management",
        "mcp_servers",
        "container",
        "betas",
        "prompt",
        "conversation",
    ] {
        if let Some(v) = body.get(key).filter(|v| !v.is_null()) {
            let benign = match key {
                "logprobs" => v == &json!(false),
                "top_logprobs" => v == &json!(0),
                "logit_bias" => v.as_object().is_some_and(|m| m.is_empty()),
                "modalities" => v == &json!(["text"]),
                "service_tier" => matches!(v.as_str(), Some("auto" | "default")),
                "context_management" | "mcp_servers" | "betas" | "include" => {
                    v.as_array().is_some_and(|a| a.is_empty())
                }
                _ => false,
            };
            if !benign {
                return Err(format!("{key} is unsupported by the local text server"));
            }
        }
    }
    for key in ["reasoning_effort", "reasoning", "thinking"] {
        if let Some(v) = body.get(key).filter(|v| !v.is_null()) {
            let disabled = match key {
                "reasoning_effort" => v == &json!("none"),
                "thinking" => v == &json!({"type":"disabled"}),
                _ => v.as_object().is_some_and(|m| m.is_empty()) || v == &json!({"effort":"none"}),
            };
            if !disabled {
                return Err(format!("active {key} is unsupported"));
            }
        }
    }
    if let Some(v) = body.get("response_format").filter(|v| !v.is_null()) {
        if v != &json!({"type":"text"}) {
            return Err("only plain text response_format is supported".into());
        }
    }
    if let Some(v) = body.get("text").filter(|v| !v.is_null()) {
        if !v.is_object()
            || v.as_object().unwrap().keys().any(|k| k != "format")
            || v.get("format")
                .is_some_and(|f| f != &json!({"type":"text"}))
        {
            return Err("only plain text text.format is supported".into());
        }
    }
    if let Some(v) = body.get("parallel_tool_calls").filter(|v| !v.is_null()) {
        if v != &json!(true) {
            return Err("parallel_tool_calls=false is unsupported".into());
        }
    }
    if let Some(v) = body.get("truncation").filter(|v| !v.is_null()) {
        if v != &json!("disabled") {
            return Err("automatic truncation is unsupported".into());
        }
    }
    if optional_bool(body, "background", false)? {
        return Err("background responses are unsupported".into());
    }
    match protocol {
        Protocol::Responses => {
            if body.get("max_tokens").is_some() || body.get("max_completion_tokens").is_some() {
                return Err("use max_output_tokens for Responses".into());
            }
        }
        _ => {
            if body.get("max_output_tokens").is_some() {
                return Err("max_output_tokens is only supported by Responses".into());
            }
            if optional_bool(body, "store", false)? {
                return Err("store:true is only supported by Responses".into());
            }
            if matches!(protocol, Protocol::Anthropic)
                && body.get("max_completion_tokens").is_some()
            {
                return Err("use max_tokens for Anthropic Messages".into());
            }
        }
    }
    Ok(())
}

fn text_content(content: &Value, nullable: bool) -> Result<String, String> {
    match content {
        Value::Null if nullable => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(blocks) => {
            let mut text = String::new();
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text" | "input_text" | "output_text") => {
                        if b.get("cache_control").is_some() {
                            return Err("prompt cache_control is unsupported".into());
                        }
                        text.push_str(
                            b.get("text")
                                .and_then(Value::as_str)
                                .ok_or("text block requires a string text")?,
                        );
                    }
                    other => {
                        return Err(format!(
                            "unsupported content block {other:?}; only text is supported"
                        ))
                    }
                }
            }
            Ok(text)
        }
        _ => Err("content must be a string or an array of text blocks".into()),
    }
}
fn message(role: &str, text: String) -> Message {
    Message {
        role: role.into(),
        text,
        calls: vec![],
        call_id: None,
    }
}
fn json_arguments(value: &Value) -> Result<Value, String> {
    let arguments = if let Some(text) = value.as_str() {
        serde_json::from_str(text).map_err(|e| format!("invalid tool arguments JSON: {e}"))?
    } else {
        value.clone()
    };
    if !arguments.is_object() {
        return Err("tool arguments must be a JSON object".into());
    }
    Ok(arguments)
}
fn chat_messages(body: &Value) -> Result<Vec<Message>, String> {
    let mut result = Vec::new();
    for m in array(body, "messages")? {
        let role = required_string(m, "role")?;
        if !matches!(
            role.as_str(),
            "system" | "developer" | "user" | "assistant" | "tool"
        ) {
            return Err(format!("unsupported message role {role}"));
        }
        for key in ["function_call", "audio", "refusal"] {
            if m.get(key).is_some_and(|v| !v.is_null()) {
                return Err(format!("message {key} is unsupported"));
            }
        }
        let nullable = role == "assistant";
        let mut item = message(
            &role,
            text_content(m.get("content").unwrap_or(&Value::Null), nullable)?,
        );
        if let Some(calls) = m.get("tool_calls").filter(|v| !v.is_null()) {
            if role != "assistant" {
                return Err("tool_calls require an assistant message".into());
            }
            for c in calls.as_array().ok_or("tool_calls must be an array")? {
                if c["type"] != "function" {
                    return Err("only function tool calls are supported".into());
                }
                let f = &c["function"];
                item.calls.push(ToolCall {
                    id: required_string(c, "id")?,
                    name: required_string(f, "name")?,
                    arguments: json_arguments(
                        f.get("arguments").ok_or("tool call requires arguments")?,
                    )?,
                });
            }
        }
        if role == "tool" {
            item.call_id = Some(required_string(m, "tool_call_id")?);
        } else if m.get("tool_call_id").is_some() {
            return Err("tool_call_id requires a tool message".into());
        }
        result.push(item);
    }
    Ok(result)
}
fn anthropic_messages(body: &Value) -> Result<Vec<Message>, String> {
    let mut result = Vec::new();
    if let Some(system) = body.get("system").filter(|v| !v.is_null()) {
        result.push(message("system", text_content(system, false)?));
    }
    for m in array(body, "messages")? {
        let role = required_string(m, "role")?;
        if !matches!(role.as_str(), "user" | "assistant") {
            return Err(
                "Anthropic messages require user or assistant roles; use top-level system".into(),
            );
        }
        let content = m.get("content").ok_or("message requires content")?;
        if !content.is_array() {
            result.push(message(&role, text_content(content, false)?));
            continue;
        }
        let mut item = message(&role, String::new());
        for block in content.as_array().unwrap() {
            if block.get("cache_control").is_some() {
                return Err("prompt cache_control is unsupported".into());
            }
            match block["type"].as_str() {
                Some("text") => item
                    .text
                    .push_str(block["text"].as_str().ok_or("text block requires text")?),
                Some("tool_use") if role == "assistant" => item.calls.push(ToolCall {
                    id: required_string(block, "id")?,
                    name: required_string(block, "name")?,
                    arguments: json_arguments(
                        block.get("input").ok_or("tool_use requires input")?,
                    )?,
                }),
                Some("tool_result") if role == "user" => {
                    if !item.text.is_empty() || !item.calls.is_empty() {
                        result.push(item);
                        item = message(&role, String::new());
                    }
                    let is_error = optional_bool(block, "is_error", false)?;
                    let mut tool = message(
                        "tool",
                        text_content(block.get("content").unwrap_or(&json!("")), false)?,
                    );
                    if is_error {
                        tool.text = format!("Tool error: {}", tool.text);
                    }
                    tool.call_id = Some(required_string(block, "tool_use_id")?);
                    result.push(tool);
                }
                other => return Err(format!("unsupported Anthropic content block {other:?}")),
            }
        }
        if !item.text.is_empty() || !item.calls.is_empty() || content.as_array().unwrap().is_empty()
        {
            result.push(item);
        }
    }
    Ok(result)
}
fn response_messages(body: &Value) -> Result<Vec<Message>, String> {
    let input = body.get("input").ok_or("Responses requires input")?;
    if input.is_string() {
        return Ok(vec![message("user", text_content(input, false)?)]);
    }
    let mut result: Vec<Message> = Vec::new();
    for item in input
        .as_array()
        .ok_or("input must be a string or array of input items")?
    {
        match item.get("type").and_then(Value::as_str).unwrap_or("message") {
            "message"=>{
                let role=required_string(item,"role")?;
                if !matches!(role.as_str(),"user"|"assistant"|"system"|"developer"){return Err(format!("unsupported Responses message role {role}"));}
                let text=text_content(item.get("content").ok_or("message requires content")?,false)?;
                if role=="assistant"&&result.last().is_some_and(|m|m.role=="assistant"&&m.call_id.is_none()) {result.last_mut().unwrap().text.push_str(&text);} else {result.push(message(&role,text));}
            }
            "function_call"=>{
                let call=ToolCall{id:required_string(item,"call_id")?,name:required_string(item,"name")?,arguments:json_arguments(item.get("arguments").ok_or("function_call requires arguments")?)?};
                if result.last().is_some_and(|m|m.role=="assistant"&&m.call_id.is_none()){result.last_mut().unwrap().calls.push(call);}else{let mut m=message("assistant",String::new());m.calls.push(call);result.push(m);}
            }
            "function_call_output"=>{let mut m=message("tool",text_content(item.get("output").ok_or("function_call_output requires output")?,false)?);m.call_id=Some(required_string(item,"call_id")?);result.push(m);}
            other=>return Err(format!("unsupported Responses input item {other}; item references and media are unsupported")),
        }
    }
    Ok(result)
}

fn validate_history(messages: &[Message], continuation: bool) -> Result<(), String> {
    use std::collections::HashSet;
    let mut pending = HashSet::new();
    let mut seen = HashSet::new();
    let mut results = HashSet::new();
    let mut started = false;
    let mut previous = "";
    for m in messages {
        if matches!(m.role.as_str(), "system" | "developer") {
            if started {
                return Err(
                    "system/developer instructions must precede conversation messages".into(),
                );
            }
            continue;
        }
        if !started && m.role == "assistant" && !continuation {
            return Err("conversation must start with a user message".into());
        }
        started = true;
        if m.role == "tool" {
            let id = m.call_id.as_ref().ok_or("tool result requires call ID")?;
            if !results.insert(id.clone()) {
                return Err(format!("duplicate tool result {id}"));
            }
            if !pending.remove(id) && !(continuation && !seen.contains(id)) {
                return Err(format!("tool result {id} has no matching pending call"));
            }
        } else {
            if !pending.is_empty()
                && !(m.role == "assistant" && previous == "assistant" && !m.calls.is_empty())
            {
                return Err(
                    "pending tool calls require matching tool results before another message"
                        .into(),
                );
            }
            for call in &m.calls {
                if !seen.insert(call.id.clone()) || results.contains(&call.id) {
                    return Err(format!("duplicate tool call ID {}", call.id));
                }
                pending.insert(call.id.clone());
            }
        }
        previous = &m.role;
    }
    if !started {
        return Err("conversation requires user input".into());
    }
    Ok(())
}
fn parse_tools(body: &Value, protocol: Protocol) -> Result<Vec<Tool>, String> {
    let mut result = Vec::new();
    let Some(tools) = body.get("tools").filter(|v| !v.is_null()) else {
        return Ok(result);
    };
    for tool in tools.as_array().ok_or("tools must be an array")? {
        let function = match protocol {
            Protocol::Chat => {
                if tool["type"] != "function" {
                    return Err("only client-executed function tools are supported".into());
                }
                &tool["function"]
            }
            Protocol::Responses => {
                if tool["type"] != "function" {
                    return Err("only client-executed function tools are supported".into());
                }
                tool
            }
            Protocol::Anthropic => {
                if tool.get("type").is_some_and(|v| v != "custom") {
                    return Err("cloud-hosted Anthropic tools are unsupported".into());
                }
                if tool.get("cache_control").is_some() {
                    return Err("tool cache_control is unsupported".into());
                }
                tool
            }
        };
        // We validate every emitted call against this schema, so strict clients
        // can use their normal function definition with the local models.
        let _strict = optional_bool(function, "strict", false)?;
        let name = required_string(function, "name")?;
        if result.iter().any(|t: &Tool| t.name == name) {
            return Err(format!("duplicate tool name {name}"));
        }
        let key = if matches!(protocol, Protocol::Anthropic) {
            "input_schema"
        } else {
            "parameters"
        };
        let parameters = function
            .get(key)
            .filter(|v| !v.is_null())
            .cloned()
            .unwrap_or_else(|| json!({"type":"object","properties":{}}));
        if !parameters.is_object() || parameters.get("type").is_some_and(|v| v != "object") {
            return Err(format!("tool {key} must be an object schema"));
        }
        result.push(Tool {
            name,
            description: optional_string(function, "description")?,
            parameters,
        });
    }
    Ok(result)
}
fn parse_choice(body: &Value, protocol: Protocol, tools: &[Tool]) -> Result<ToolChoice, String> {
    let Some(value) = body.get("tool_choice").filter(|v| !v.is_null()) else {
        return Ok(if tools.is_empty() {
            ToolChoice::None
        } else {
            ToolChoice::Auto
        });
    };
    let choice = match value
        .as_str()
        .or_else(|| value.get("type").and_then(Value::as_str))
    {
        Some("auto") => ToolChoice::Auto,
        Some("none") => ToolChoice::None,
        Some("required" | "any") => ToolChoice::Required,
        Some("function") => ToolChoice::Named(required_string(
            if matches!(protocol, Protocol::Chat) {
                &value["function"]
            } else {
                value
            },
            "name",
        )?),
        Some("tool") if matches!(protocol, Protocol::Anthropic) => {
            ToolChoice::Named(required_string(value, "name")?)
        }
        _ => return Err("unsupported tool_choice".into()),
    };
    if value
        .get("disable_parallel_tool_use")
        .is_some_and(|v| v != &json!(false))
    {
        return Err("disable_parallel_tool_use is unsupported".into());
    }
    match &choice {
        ToolChoice::Named(name) if !tools.iter().any(|t| &t.name == name) => {
            return Err(format!("tool_choice names unknown tool {name}"))
        }
        ToolChoice::Required if tools.is_empty() => {
            return Err("required tool_choice needs tools".into())
        }
        _ => {}
    }
    Ok(choice)
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn chat_call(call: &ToolCall) -> Value {
    json!({"id":call.id,"type":"function","function":{"name":call.name,"arguments":call.arguments.to_string()}})
}
fn chat_usage(usage: &Usage) -> Value {
    json!({"prompt_tokens":usage.input,"completion_tokens":usage.output,"total_tokens":usage.input+usage.output})
}
fn chat_finish(finish: &Finish) -> &'static str {
    match finish {
        Finish::Limit => "length",
        Finish::Tools => "tool_calls",
        _ => "stop",
    }
}
fn anthropic_finish(finish: &Finish) -> &'static str {
    match finish {
        Finish::Limit => "max_tokens",
        Finish::Tools => "tool_use",
        Finish::Sequence(_) => "stop_sequence",
        Finish::Stop => "end_turn",
    }
}
fn anthropic_content(g: &Generation) -> Vec<Value> {
    let mut content = Vec::new();
    if !g.text.is_empty() || g.calls.is_empty() {
        content.push(json!({"type":"text","text":g.text}));
    }
    content.extend(g.calls.iter().map(
        |call| json!({"type":"tool_use","id":call.id,"name":call.name,"input":call.arguments}),
    ));
    content
}
fn text_part(text: &str) -> Value {
    json!({"type":"output_text","text":text,"annotations":[],"logprobs":[]})
}
fn response_message(id: &str, text: &str, status: &str) -> Value {
    json!({"id":format!("msg_{id}"),"type":"message","role":"assistant","status":status,"content":[text_part(text)]})
}
fn response_call(call: &ToolCall, status: &str) -> Value {
    json!({"id":format!("fc_{}",call.id),"type":"function_call","status":status,"call_id":call.id,"name":call.name,"arguments":call.arguments.to_string()})
}
fn response_object(id: &str, model: &str, g: &Generation, created: u64) -> Value {
    let limited = matches!(g.finish, Finish::Limit);
    let status = if limited { "incomplete" } else { "completed" };
    let mut output = Vec::new();
    if !g.text.is_empty() || g.calls.is_empty() {
        output.push(response_message(id, &g.text, status));
    }
    output.extend(g.calls.iter().map(|c| response_call(c, "completed")));
    json!({"id":id,"object":"response","created_at":created,"status":status,"background":false,"error":null,
        "incomplete_details":if limited {json!({"reason":"max_output_tokens"})} else {Value::Null},"instructions":null,"max_output_tokens":null,
        "model":model,"output":output,"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},
        "store":true,"temperature":null,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1.0,"truncation":"disabled",
        "usage":{"input_tokens":g.usage.input,"input_tokens_details":{"cached_tokens":0},"output_tokens":g.usage.output,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":g.usage.input+g.usage.output},"metadata":{}})
}

#[derive(Clone, Debug)]
struct StreamCall {
    index: usize,
    wire_index: usize,
    id: String,
    name: String,
    arguments: String,
    closed: bool,
}
#[derive(Clone, Debug)]
pub struct Encoder {
    protocol: Protocol,
    id: String,
    model: String,
    created: u64,
    sequence: usize,
    text: String,
    text_index: Option<usize>,
    text_closed: bool,
    next_index: usize,
    calls: Vec<StreamCall>,
    input: usize,
    failed: bool,
}
impl Encoder {
    pub fn new(protocol: Protocol, id: String, model: String) -> Self {
        Self {
            protocol,
            id,
            model,
            created: now(),
            sequence: 0,
            text: String::new(),
            text_index: None,
            text_closed: false,
            next_index: 0,
            calls: vec![],
            input: 0,
            failed: false,
        }
    }
    fn event(&mut self, name: &'static str, mut value: Value) -> WireEvent {
        value["type"] = json!(name);
        if matches!(self.protocol, Protocol::Responses) {
            value["sequence_number"] = json!(self.sequence);
            self.sequence += 1;
        }
        WireEvent {
            event: Some(name),
            data: value.to_string(),
        }
    }
    fn chunk(&self, delta: Value, finish: Option<&str>) -> WireEvent {
        WireEvent {event:None,data:json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":self.model,"choices":[{"index":0,"delta":delta,"finish_reason":finish,"logprobs":null}]}).to_string()}
    }
    pub fn start(&mut self, input_tokens: usize) -> Vec<WireEvent> {
        self.input = input_tokens;
        match self.protocol {
            Protocol::Chat => vec![self.chunk(json!({"role":"assistant","content":""}),None)],
            Protocol::Anthropic => vec![self.event("message_start",json!({"message":{"id":self.id,"type":"message","role":"assistant","model":self.model,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":input_tokens,"output_tokens":0}}}))],
            Protocol::Responses => {
                let mut response=response_object(&self.id,&self.model,&Generation {text:String::new(),calls:vec![],usage:Usage{input:input_tokens,output:0},finish:Finish::Stop},self.created);
                response["status"]=json!("in_progress"); response["output"]=json!([]);response["usage"]=Value::Null;
                vec![self.event("response.created",json!({"response":response})),self.event("response.in_progress",json!({"response":response}))]
            }
        }
    }
    fn open_text(&mut self) -> Vec<WireEvent> {
        if self.text_index.is_some()
            && !(matches!(self.protocol, Protocol::Anthropic) && self.text_closed)
        {
            return vec![];
        }
        self.text_closed = false;
        let index = self.next_index;
        self.next_index += 1;
        self.text_index = Some(index);
        match self.protocol {
            Protocol::Anthropic => vec![self.event(
                "content_block_start",
                json!({"index":index,"content_block":{"type":"text","text":""}}),
            )],
            Protocol::Responses => {
                let mut item = response_message(&self.id, "", "in_progress");
                item["content"] = json!([]);
                vec![self.event("response.output_item.added",json!({"output_index":index,"item":item})),self.event("response.content_part.added",json!({"item_id":format!("msg_{}",self.id),"output_index":index,"content_index":0,"part":text_part("")}))]
            }
            Protocol::Chat => vec![],
        }
    }
    fn close_text(&mut self, status: &str) -> Vec<WireEvent> {
        let Some(index) = self.text_index else {
            return vec![];
        };
        if self.text_closed {
            return vec![];
        }
        self.text_closed = true;
        match self.protocol {
            Protocol::Anthropic => vec![self.event("content_block_stop",json!({"index":index}))],
            Protocol::Responses => vec![
                self.event("response.output_text.done",json!({"item_id":format!("msg_{}",self.id),"output_index":index,"content_index":0,"text":self.text,"logprobs":[]})),
                self.event("response.content_part.done",json!({"item_id":format!("msg_{}",self.id),"output_index":index,"content_index":0,"part":text_part(&self.text)})),
                self.event("response.output_item.done",json!({"output_index":index,"item":response_message(&self.id,&self.text,status)}))],
            Protocol::Chat => vec![],
        }
    }
    fn close_call(&mut self, index: usize) -> Vec<WireEvent> {
        let Some(call) = self.calls.iter_mut().find(|c| c.index == index) else {
            return vec![];
        };
        if call.closed {
            return vec![];
        }
        call.closed = true;
        let call = call.clone();
        match self.protocol {
            Protocol::Anthropic => vec![self.event("content_block_stop",json!({"index":call.wire_index}))],
            Protocol::Responses => vec![self.event("response.function_call_arguments.done",json!({"item_id":format!("fc_{}",call.id),"output_index":call.wire_index,"arguments":call.arguments,"name":call.name})),
                self.event("response.output_item.done",json!({"output_index":call.wire_index,"item":{"id":format!("fc_{}",call.id),"type":"function_call","status":"completed","call_id":call.id,"name":call.name,"arguments":call.arguments}}))],
            Protocol::Chat => vec![],
        }
    }
    pub fn delta(&mut self, delta: &Delta) -> Vec<WireEvent> {
        if self.failed {
            return vec![];
        }
        if matches!(self.protocol, Protocol::Responses)
            && !self.calls.is_empty()
            && matches!(delta,Delta::Text(text) if !text.trim().is_empty())
        {
            return self.failure("Text after a function call is unsupported");
        }
        match delta {
            Delta::Text(text) => {
                if text.is_empty() {
                    return vec![];
                }
                let mut events = self.open_text();
                self.text.push_str(text);
                let index = self.text_index.unwrap();
                events.push(match self.protocol {
                    Protocol::Chat => self.chunk(json!({"content":text}),None),
                    Protocol::Anthropic => self.event("content_block_delta",json!({"index":index,"delta":{"type":"text_delta","text":text}})),
                    Protocol::Responses => self.event("response.output_text.delta",json!({"item_id":format!("msg_{}",self.id),"output_index":index,"content_index":0,"delta":text,"logprobs":[]})),
                });
                events
            }
            Delta::ToolStart { index, id, name } => {
                let mut events = if matches!(self.protocol, Protocol::Anthropic) {
                    self.close_text("completed")
                } else {
                    vec![]
                };
                let wire_index = self.next_index;
                self.next_index += 1;
                self.calls.push(StreamCall {
                    index: *index,
                    wire_index,
                    id: id.clone(),
                    name: name.clone(),
                    arguments: String::new(),
                    closed: false,
                });
                events.push(match self.protocol {
                    Protocol::Chat => self.chunk(json!({"tool_calls":[{"index":index,"id":id,"type":"function","function":{"name":name,"arguments":""}}]}),None),
                    Protocol::Anthropic => self.event("content_block_start",json!({"index":wire_index,"content_block":{"type":"tool_use","id":id,"name":name,"input":{}}})),
                    Protocol::Responses => self.event("response.output_item.added",json!({"output_index":wire_index,"item":{"id":format!("fc_{id}"),"type":"function_call","status":"in_progress","call_id":id,"name":name,"arguments":""}})),
                });
                events
            }
            Delta::ToolArguments { index, fragment } => {
                let Some(call) = self.calls.iter_mut().find(|c| c.index == *index) else {
                    return self.failure("tool argument fragment without a tool start");
                };
                call.arguments.push_str(fragment);
                let wire_index = call.wire_index;
                let id = call.id.clone();
                vec![match self.protocol {
                    Protocol::Chat => self.chunk(json!({"tool_calls":[{"index":index,"function":{"arguments":fragment}}]}),None),
                    Protocol::Anthropic => self.event("content_block_delta",json!({"index":wire_index,"delta":{"type":"input_json_delta","partial_json":fragment}})),
                    Protocol::Responses => self.event("response.function_call_arguments.delta",json!({"item_id":format!("fc_{id}"),"output_index":wire_index,"delta":fragment})),
                }]
            }
            Delta::ToolEnd { index } => self.close_call(*index),
        }
    }
    pub fn finish(&mut self, generation: &Generation, include_usage: bool) -> Vec<WireEvent> {
        if self.failed {
            return vec![];
        }
        // Even an empty generation has a text item/block in the JSON response.
        let mut events = if self.text_index.is_none() && self.calls.is_empty() {
            self.open_text()
        } else {
            vec![]
        };
        events.extend(
            self.close_text(if matches!(generation.finish, Finish::Limit) {
                "incomplete"
            } else {
                "completed"
            }),
        );
        let indices: Vec<_> = self
            .calls
            .iter()
            .filter(|c| !c.closed)
            .map(|c| c.index)
            .collect();
        for index in indices {
            events.extend(self.close_call(index));
        }
        match self.protocol {
            Protocol::Chat => {
                events.push(self.chunk(json!({}), Some(chat_finish(&generation.finish))));
                if include_usage {
                    events.push(WireEvent {event:None,data:json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":self.model,"choices":[],"usage":chat_usage(&generation.usage)}).to_string()});
                }
                events.push(WireEvent {
                    event: None,
                    data: "[DONE]".into(),
                });
            }
            Protocol::Anthropic => {
                events.push(self.event("message_delta",json!({"delta":{"stop_reason":anthropic_finish(&generation.finish),"stop_sequence":match &generation.finish {Finish::Sequence(s)=>Some(s),_=>None}},"usage":{"output_tokens":generation.usage.output}})));
                events.push(self.event("message_stop", json!({})));
            }
            Protocol::Responses => {
                let mut response = response_object(&self.id, &self.model, generation, self.created);
                // A tool may precede visible text. Keep final output indices identical
                // to the indices announced by output_item.added.
                let mut output = Vec::new();
                if let Some(index) = self.text_index {
                    output.push((
                        index,
                        response_message(
                            &self.id,
                            &generation.text,
                            if matches!(generation.finish, Finish::Limit) {
                                "incomplete"
                            } else {
                                "completed"
                            },
                        ),
                    ));
                }
                for call in &self.calls {
                    output.push((call.wire_index, json!({"id":format!("fc_{}",call.id),"type":"function_call","status":"completed","call_id":call.id,"name":call.name,"arguments":call.arguments})));
                }
                output.sort_by_key(|(index, _)| *index);
                response["output"] =
                    Value::Array(output.into_iter().map(|(_, item)| item).collect());
                events.push(self.event(
                    if matches!(generation.finish, Finish::Limit) {
                        "response.incomplete"
                    } else {
                        "response.completed"
                    },
                    json!({"response":response}),
                ));
            }
        }
        events
    }
    pub fn failure(&mut self, message: &str) -> Vec<WireEvent> {
        if self.failed {
            return vec![];
        }
        self.failed = true;
        match self.protocol {
            Protocol::Chat => vec![
                WireEvent {
                    event: None,
                    data: self.protocol.error(500, message).to_string(),
                },
                WireEvent {
                    event: None,
                    data: "[DONE]".into(),
                },
            ],
            Protocol::Anthropic => vec![self.event("error", self.protocol.error(500, message))],
            Protocol::Responses => {
                let error = self.event(
                    "error",
                    json!({"code":"server_error","message":message,"param":null}),
                );
                let mut response = response_object(
                    &self.id,
                    &self.model,
                    &Generation {
                        text: self.text.clone(),
                        calls: vec![],
                        usage: Usage {
                            input: self.input,
                            output: 0,
                        },
                        finish: Finish::Stop,
                    },
                    self.created,
                );
                response["status"] = json!("failed");
                response["error"] = json!({"code":"server_error","message":message});
                vec![
                    error,
                    self.event("response.failed", json!({"response":response})),
                ]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn generation() -> Generation {
        Generation {
            text: "Let me check.".into(),
            calls: vec![ToolCall {
                id: "call_1".into(),
                name: "weather".into(),
                arguments: json!({"city":"北京"}),
            }],
            usage: Usage {
                input: 12,
                output: 7,
            },
            finish: Finish::Tools,
        }
    }
    fn events(protocol: Protocol, finish: Finish) -> Vec<WireEvent> {
        let mut g = generation();
        g.finish = finish;
        let mut e = Encoder::new(protocol, "resp_1".into(), "local".into());
        let mut events = e.start(12);
        events.extend(e.delta(&Delta::Text(g.text.clone())));
        events.extend(e.delta(&Delta::ToolStart {
            index: 0,
            id: "call_1".into(),
            name: "weather".into(),
        }));
        events.extend(e.delta(&Delta::ToolArguments {
            index: 0,
            fragment: "{\"city\":".into(),
        }));
        events.extend(e.delta(&Delta::ToolArguments {
            index: 0,
            fragment: "\"北京\"}".into(),
        }));
        events.extend(e.delta(&Delta::ToolEnd { index: 0 }));
        events.extend(e.finish(&g, true));
        events
    }
    #[test]
    fn chat_keeps_tool_history_and_parameters() {
        let r = Protocol::Chat.parse(&json!({"model":"local", "messages":[
            {"role":"developer","content":"Be helpful"}, {"role":"user","content":[{"type":"text","text":"Weather?"}]},
            {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"北京\"}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"Sunny"}, {"role":"user","content":"Thanks"}],
            "tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}],"max_completion_tokens":20,"stream":true,"stream_options":{"include_usage":true}})).unwrap();
        assert_eq!(r.messages.len(), 5);
        assert_eq!(r.messages[0].role, "developer");
        assert_eq!(r.messages[2].calls[0].arguments["city"], "北京");
        assert_eq!(r.messages[3].call_id.as_deref(), Some("c1"));
        assert_eq!(r.max_tokens, 20);
        assert!(r.include_usage);
        assert!(matches!(r.choice, ToolChoice::Auto));
    }
    #[test]
    fn anthropic_keeps_system_and_interleaved_tool_blocks() {
        let r = Protocol::Anthropic.parse(&json!({"system":[{"type":"text","text":"System"}],"max_tokens":32,
            "messages":[{"role":"user","content":"Weather?"}, {"role":"assistant","content":[{"type":"text","text":"Checking"},{"type":"tool_use","id":"c1","name":"weather","input":{"city":"北京"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":[{"type":"text","text":"Sunny"}]},{"type":"text","text":"Explain"}]}],
            "tools":[{"name":"weather","input_schema":{"type":"object"}}],"tool_choice":{"type":"tool","name":"weather"}})).unwrap();
        assert_eq!(r.messages[0].role, "system");
        assert_eq!(r.messages[0].text, "System");
        assert_eq!(r.messages[2].calls.len(), 1);
        assert_eq!(r.messages[3].role, "tool");
        assert_eq!(r.messages[4].text, "Explain");
        assert!(matches!(r.choice, ToolChoice::Named(_)));
    }
    #[test]
    fn responses_replays_output_items_and_allows_parent_correlated_results() {
        let g = generation();
        let output = Protocol::Responses.response("r1", "local", &g)["output"].clone();
        let mut input = vec![json!({"role":"user","content":"Weather?"})];
        input.extend(output.as_array().unwrap().clone());
        input.push(json!({"type":"function_call_output","call_id":"call_1","output":"Sunny"}));
        let r = Protocol::Responses
            .parse(&json!({"input":input,"instructions":"System","max_output_tokens":50}))
            .unwrap();
        assert_eq!(r.messages[1].text, "Let me check.");
        assert_eq!(r.messages[1].calls[0].id, "call_1");
        assert_eq!(r.messages[2].role, "tool");
        assert!(r.store);
        assert!(Protocol::Responses.parse(&json!({"input":[{"type":"function_call_output","call_id":"parent_call","output":"ok"}],"previous_response_id":"r0"})).is_ok());
    }
    #[test]
    fn rejects_unsupported_and_malformed_inputs() {
        let valid = json!({"messages":[{"role":"user","content":"Hi"}]});
        for (key, value) in [
            ("n", json!(2)),
            ("temperature", json!(-1)),
            ("max_tokens", json!(0)),
            ("top_p", json!(0.9)),
            ("response_format", json!({"type":"json_object"})),
            ("reasoning_effort", json!("high")),
        ] {
            let mut body = valid.clone();
            body[key] = value;
            assert!(Protocol::Chat.parse(&body).is_err(), "{key}");
        }
        assert!(Protocol::Chat.parse(&json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"a"}}]}]})).is_err());
        assert!(Protocol::Chat
            .parse(&json!({"messages":[{"role":"tool","content":"x","tool_call_id":"missing"}]}))
            .is_err());
        assert!(Protocol::Chat.parse(&json!({"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f","strict":true,"parameters":{"type":"object"}}}]})).is_ok());
        assert!(Protocol::Responses
            .parse(&json!({"input":"x","tools":[{"type":"web_search"}]}))
            .is_err());
        assert!(Protocol::Anthropic
            .parse(&json!({"messages":[{"role":"user","content":"x"}],"temperature":1.2}))
            .is_err());
    }
    #[test]
    fn chat_stream_has_role_tool_fragments_usage_and_done() {
        let e = events(Protocol::Chat, Finish::Tools);
        let first: Value = serde_json::from_str(&e[0].data).unwrap();
        assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(e.last().unwrap().data, "[DONE]");
        let usage: Value = serde_json::from_str(&e[e.len() - 2].data).unwrap();
        assert_eq!(usage["choices"], json!([]));
        assert_eq!(usage["usage"]["total_tokens"], 19);
        let tool: Value = serde_json::from_str(&e[2].data).unwrap();
        assert_eq!(tool["choices"][0]["delta"]["tool_calls"][0]["id"], "call_1");
    }
    #[test]
    fn anthropic_stream_closes_each_block_and_message() {
        let e = events(Protocol::Anthropic, Finish::Tools);
        let names: Vec<_> = e.iter().map(|e| e.event.unwrap()).collect();
        assert_eq!(names.first(), Some(&"message_start"));
        assert_eq!(names.last(), Some(&"message_stop"));
        assert_eq!(
            names
                .iter()
                .filter(|&&n| n == "content_block_start")
                .count(),
            2
        );
        assert_eq!(
            names.iter().filter(|&&n| n == "content_block_stop").count(),
            2
        );
        let final_delta: Value = serde_json::from_str(&e[e.len() - 2].data).unwrap();
        assert_eq!(final_delta["delta"]["stop_reason"], "tool_use");
    }
    #[test]
    fn responses_stream_has_complete_lifecycle_and_incomplete_limit() {
        let e = events(Protocol::Responses, Finish::Tools);
        let names: Vec<_> = e.iter().map(|e| e.event.unwrap()).collect();
        assert_eq!(&names[..2], &["response.created", "response.in_progress"]);
        assert_eq!(names.last(), Some(&"response.completed"));
        for (i, event) in e.iter().enumerate() {
            let v: Value = serde_json::from_str(&event.data).unwrap();
            assert_eq!(v["type"], event.event.unwrap());
            assert_eq!(v["sequence_number"], i);
        }
        for name in [
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        ] {
            assert!(names.contains(&name), "{name}");
        }
        let done: Value = serde_json::from_str(&e.last().unwrap().data).unwrap();
        assert_eq!(
            done["response"]["output"][0]["content"][0]["text"],
            "Let me check."
        );
        let limited = events(Protocol::Responses, Finish::Limit);
        let done: Value = serde_json::from_str(&limited.last().unwrap().data).unwrap();
        assert_eq!(done["type"], "response.incomplete");
        assert_eq!(
            done["response"]["incomplete_details"]["reason"],
            "max_output_tokens"
        );
    }
    #[test]
    fn interleaved_text_uses_fresh_anthropic_blocks_and_response_indices() {
        for protocol in [Protocol::Anthropic] {
            let mut e = Encoder::new(protocol, "r".into(), "m".into());
            let mut events = e.start(1);
            events.extend(e.delta(&Delta::ToolStart {
                index: 0,
                id: "c".into(),
                name: "f".into(),
            }));
            events.extend(e.delta(&Delta::ToolArguments {
                index: 0,
                fragment: "{}".into(),
            }));
            events.extend(e.delta(&Delta::ToolEnd { index: 0 }));
            events.extend(e.delta(&Delta::Text("After".into())));
            events.extend(e.delta(&Delta::ToolStart {
                index: 1,
                id: "c2".into(),
                name: "f".into(),
            }));
            events.extend(e.delta(&Delta::ToolArguments {
                index: 1,
                fragment: "{}".into(),
            }));
            events.extend(e.delta(&Delta::ToolEnd { index: 1 }));
            events.extend(e.delta(&Delta::Text(" more".into())));
            let g = Generation {
                text: "After more".into(),
                calls: vec![
                    ToolCall {
                        id: "c".into(),
                        name: "f".into(),
                        arguments: json!({}),
                    },
                    ToolCall {
                        id: "c2".into(),
                        name: "f".into(),
                        arguments: json!({}),
                    },
                ],
                usage: Usage {
                    input: 1,
                    output: 4,
                },
                finish: Finish::Tools,
            };
            events.extend(e.finish(&g, false));
            if matches!(protocol, Protocol::Anthropic) {
                let mut open = std::collections::HashSet::new();
                for event in events {
                    let v: Value = serde_json::from_str(&event.data).unwrap();
                    let index = v["index"].as_u64();
                    match event.event.unwrap() {
                        "content_block_start" => {
                            assert!(open.insert(index.unwrap()));
                        }
                        "content_block_delta" => {
                            assert!(open.contains(&index.unwrap()), "delta after block stop")
                        }
                        "content_block_stop" => {
                            assert!(open.remove(&index.unwrap()));
                        }
                        _ => {}
                    }
                }
                assert!(open.is_empty());
            } else {
                let final_response: Value =
                    serde_json::from_str(&events.last().unwrap().data).unwrap();
                let output = final_response["response"]["output"].as_array().unwrap();
                for event in events
                    .iter()
                    .filter(|e| e.event == Some("response.output_item.done"))
                {
                    let v: Value = serde_json::from_str(&event.data).unwrap();
                    let index = v["output_index"].as_u64().unwrap() as usize;
                    assert_eq!(output[index]["id"], v["item"]["id"]);
                }
            }
        }
    }
    #[test]
    fn invalid_ids_choices_and_wrong_scalar_types_are_rejected() {
        let valid = json!({"messages":[{"role":"user","content":"Hi"}]});
        for (key, value) in [
            ("stream", json!("true")),
            ("max_tokens", json!(1.5)),
            ("model", json!(3)),
            (
                "tool_choice",
                json!({"type":"function","function":{"name":"missing"}}),
            ),
            ("stop", json!([""])),
        ] {
            let mut b = valid.clone();
            b[key] = value;
            assert!(Protocol::Chat.parse(&b).is_err(), "{key}");
        }
        let call = json!({"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"[]"}}]});
        assert!(Protocol::Chat
            .parse(&json!({"messages":[{"role":"user","content":"x"},call]}))
            .is_err());
        let call = json!({"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}},{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]});
        assert!(Protocol::Chat
            .parse(&json!({"messages":[{"role":"user","content":"x"},call]}))
            .is_err());
        let mut b = valid;
        b["metadata"] = json!({"request":"id"});
        b["top_p"] = json!(1.0);
        b["frequency_penalty"] = json!(0);
        assert!(Protocol::Chat.parse(&b).is_ok());
    }
    #[test]
    fn responses_rejects_text_after_a_tool_item() {
        let mut e = Encoder::new(Protocol::Responses, "r".into(), "m".into());
        e.start(1);
        e.delta(&Delta::ToolStart {
            index: 0,
            id: "c".into(),
            name: "f".into(),
        });
        e.delta(&Delta::ToolEnd { index: 0 });
        let events = e.delta(&Delta::Text("suffix".into()));
        assert_eq!(events.last().unwrap().event, Some("response.failed"));
        let generation = Generation {
            text: "suffix".into(),
            calls: vec![],
            usage: Usage {
                input: 1,
                output: 1,
            },
            finish: Finish::Stop,
        };
        assert!(e.finish(&generation, false).is_empty());
    }
    #[test]
    fn error_shapes_and_stream_terminal_failures_are_standard() {
        for protocol in [Protocol::Chat, Protocol::Anthropic, Protocol::Responses] {
            assert_eq!(protocol.error(400, "bad")["error"]["message"], "bad");
            let mut e = Encoder::new(protocol, "r".into(), "m".into());
            e.start(1);
            let errors = e.failure("broken");
            assert!(!errors.is_empty());
            match protocol {
                Protocol::Chat => assert_eq!(errors.last().unwrap().data, "[DONE]"),
                Protocol::Anthropic => assert_eq!(errors[0].event, Some("error")),
                Protocol::Responses => {
                    assert_eq!(errors.last().unwrap().event, Some("response.failed"))
                }
            }
        }
    }
    #[test]
    fn tool_errors_survive_normalization_and_behavior_is_not_ignored() {
        let r=Protocol::Anthropic.parse(&json!({"messages":[{"role":"user","content":"go"},{"role":"assistant","content":[{"type":"tool_use","id":"c","name":"f","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"c","is_error":true,"content":"failed"}]}]})).unwrap();
        assert!(r.messages.last().unwrap().text.contains("error"));
        let valid = json!({"messages":[{"role":"user","content":"Hi"}]});
        for (key, value) in [
            ("functions", json!([{ "name":"f" }])),
            ("function_call", json!("auto")),
            ("include", json!(["message.output_text.logprobs"])),
            ("max_output_tokens", json!(1)),
            ("store", json!(true)),
        ] {
            let mut body = valid.clone();
            body[key] = value;
            assert!(Protocol::Chat.parse(&body).is_err(), "{key}");
        }
        assert!(Protocol::Responses
            .parse(&json!({"input":"Hi","max_tokens":1}))
            .is_err());
    }
}
