use super::protocol::{Delta, Message, Tool, ToolCall, ToolChoice};
use crate::{
    prompt::{build_qwen_chat_prompt, QwenMessage},
    BPETokenizer,
};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const MARKERS: &[&str] = &[
    "<tool_call",
    "</tool_call",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
];
const MAX_CALL_BYTES: usize = 1024 * 1024;

// Text-only portions of the official Qwen/Qwen3-0.6B and Qwen/Qwen3.5-0.8B
// tokenizer_config.json templates. ChatML controls are supplied by prompt.rs;
// message content and tool schemas remain ordinary text, never special tokens.
const QWEN3_TOOLS: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
const QWEN3_CALL: &str = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
const QWEN35_TOOLS: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const QWEN35_CALL: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

fn is_qwen35(arch: &str) -> Result<bool, String> {
    match arch {
        "qwen3" | "qwen3vl" | "lfm2moe" => Ok(false),
        "qwen35" => Ok(true),
        _ => Err(format!(
            "Tool/chat template is unsupported for architecture {arch}"
        )),
    }
}

fn tag_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.chars().any(|c| matches!(c, '<' | '>' | '\n' | '\r')) {
        return Err("Tool and parameter names cannot contain tag delimiters or newlines".into());
    }
    Ok(())
}

fn validate_tools(tools: &[Tool], choice: &ToolChoice) -> Result<(), String> {
    let mut names = std::collections::HashSet::new();
    for tool in tools {
        tag_name(&tool.name)?;
        if !names.insert(tool.name.as_str()) {
            return Err(format!("Duplicate tool definition: {}", tool.name));
        }
        if let Some(properties) = tool.parameters.get("properties").and_then(Value::as_object) {
            for name in properties.keys() {
                tag_name(name)?;
            }
        }
    }
    match choice {
        ToolChoice::Required if tools.is_empty() => {
            Err("Required tool choice needs a tool definition".into())
        }
        ToolChoice::Named(name) if !names.contains(name.as_str()) => {
            Err(format!("Unknown named tool: {name}"))
        }
        _ => Ok(()),
    }
}

pub fn build_prompt(
    tokenizer: &BPETokenizer,
    arch: &str,
    messages: &[Message],
    tools: &[Tool],
    choice: &ToolChoice,
) -> Result<Vec<u32>, String> {
    let qwen35 = is_qwen35(arch)?;
    if arch == "qwen3vl" && !tools.is_empty() {
        return Err("Function tools are unsupported for Qwen3VL text generation".into());
    }
    validate_tools(tools, choice)?;
    if messages.is_empty() {
        return Err("No messages provided".into());
    }
    let last_user = messages.iter().rposition(|m| m.role == "user");
    if qwen35 && last_user.is_none() {
        return Err("Qwen35 needs a user query in messages".into());
    }
    let mut turns: Vec<(String, String)> = Vec::new();
    let merged_instructions = if tools.is_empty() {
        0
    } else {
        messages
            .iter()
            .take_while(|message| matches!(message.role.as_str(), "system" | "developer"))
            .count()
    };
    if !tools.is_empty() {
        let leading_instructions = messages[..merged_instructions]
            .iter()
            .map(|message| {
                if qwen35 {
                    message.text.trim()
                } else {
                    message.text.as_str()
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        let mut instruction = if qwen35 { QWEN35_TOOLS } else { QWEN3_TOOLS }.to_owned();
        for tool in tools {
            let mut function = json!({"name":tool.name,"parameters":tool.parameters});
            if let Some(description) = &tool.description {
                function["description"] = Value::String(description.clone());
            }
            instruction.push('\n');
            instruction.push_str(&json!({"type":"function","function":function}).to_string());
        }
        instruction.push_str(if qwen35 { QWEN35_CALL } else { QWEN3_CALL });
        match choice {
            ToolChoice::Auto => {}
            ToolChoice::None => {
                instruction.push_str("\n\nDo not call any functions. Answer with ordinary text.")
            }
            ToolChoice::Required => {
                instruction.push_str("\n\nYou must call at least one of the provided functions.")
            }
            ToolChoice::Named(name) => instruction.push_str(&format!(
                "\n\nYou must call the function {} and no other function.",
                Value::String(name.clone())
            )),
        }
        if !leading_instructions.is_empty() {
            if qwen35 {
                instruction.push_str("\n\n");
                instruction.push_str(&leading_instructions);
            } else {
                instruction = format!("{leading_instructions}\n\n{instruction}");
            }
        }
        turns.push(("system".into(), instruction));
    }
    let mut i = 0;
    while i < messages.len() {
        let message = &messages[i];
        if message.role != "assistant" && !message.calls.is_empty() {
            return Err("Only assistant messages may contain tool calls".into());
        }
        if message.role != "tool" && message.call_id.is_some() {
            return Err("Only tool results may contain a call ID".into());
        }
        match message.role.as_str() {
            "system" | "developer" => {
                if i < merged_instructions {
                    i += 1;
                    continue;
                }
                if qwen35 && message.role == "system" && i != 0 {
                    return Err("Qwen35 system message must be at the beginning".into());
                }
                turns.push((
                    "system".into(),
                    if qwen35 {
                        message.text.trim().into()
                    } else {
                        message.text.clone()
                    },
                ));
            }
            "user" => turns.push((
                "user".into(),
                if qwen35 {
                    message.text.trim().into()
                } else {
                    message.text.clone()
                },
            )),
            "assistant" => {
                let text = if qwen35 {
                    message.text.trim()
                } else {
                    message.text.as_str()
                };
                let (reasoning, content) =
                    if let Some((before, after)) = text.split_once("</think>") {
                        (
                            before.rsplit_once("<think>").map_or(before, |(_, v)| v),
                            after.trim_start_matches('\n'),
                        )
                    } else {
                        ("", text)
                    };
                let reasoning = if qwen35 {
                    reasoning.trim()
                } else {
                    reasoning.trim_matches('\n')
                };
                let current = last_user.is_some_and(|last| i > last);
                let mut content =
                    if current && (qwen35 || i + 1 == messages.len() || !reasoning.is_empty()) {
                        format!(
                            "<think>\n{reasoning}\n</think>\n\n{}",
                            content.trim_start_matches('\n')
                        )
                    } else {
                        content.to_owned()
                    };
                for (call_index, call) in message.calls.iter().enumerate() {
                    tag_name(&call.name)?;
                    let arguments = call
                        .arguments
                        .as_object()
                        .ok_or("Tool call arguments must be a JSON object")?;
                    if call_index > 0 {
                        content.push('\n');
                    } else if !content.is_empty() {
                        content.push_str(if qwen35 { "\n\n" } else { "\n" });
                    }
                    if qwen35 {
                        content.push_str(&format!("<tool_call>\n<function={}>\n", call.name));
                        for (name, value) in arguments {
                            tag_name(name)?;
                            let value = match value {
                                Value::String(value) => {
                                    if [
                                        OPEN,
                                        CLOSE,
                                        "</parameter>",
                                        "<parameter=",
                                        "<function=",
                                        "</function>",
                                    ]
                                    .iter()
                                    .any(|tag| value.contains(tag))
                                    {
                                        return Err("Qwen35 string argument contains a reserved tool delimiter".into());
                                    }
                                    value.clone()
                                }
                                Value::Bool(true) => "True".into(),
                                Value::Bool(false) => "False".into(),
                                Value::Null => "None".into(),
                                _ => value.to_string().replace('<', "\\u003c"),
                            };
                            content
                                .push_str(&format!("<parameter={name}>\n{value}\n</parameter>\n"));
                        }
                        content.push_str("</function>\n</tool_call>");
                    } else {
                        content.push_str(&format!(
                            "<tool_call>\n{{\"name\": {}, \"arguments\": {}}}\n</tool_call>",
                            Value::String(call.name.clone()),
                            call.arguments,
                        ));
                    }
                }
                turns.push(("assistant".into(), content));
            }
            "tool" => {
                let mut responses = String::new();
                while i < messages.len() && messages[i].role == "tool" {
                    let result = &messages[i];
                    if result.call_id.is_none() || !result.calls.is_empty() {
                        return Err("Tool results need a call ID and cannot contain calls".into());
                    }
                    if !responses.is_empty() {
                        responses.push('\n');
                    }
                    responses.push_str("<tool_response>\n");
                    responses.push_str(if qwen35 {
                        result.text.trim()
                    } else {
                        &result.text
                    });
                    responses.push_str("\n</tool_response>");
                    i += 1;
                }
                turns.push(("user".into(), responses));
                continue;
            }
            _ => return Err(format!("Unexpected message role: {}", message.role)),
        }
        i += 1;
    }
    let messages: Vec<QwenMessage<'_>> = turns
        .iter()
        .map(|(role, content)| QwenMessage { role, content })
        .collect();
    build_qwen_chat_prompt(tokenizer, &messages, false)
}

pub struct OutputParser {
    qwen35: bool,
    tools: HashMap<String, Value>,
    choice: ToolChoice,
    pending: String,
    in_call: bool,
    scan: usize,
    quoted: bool,
    escaped: bool,
    text: String,
    post_call_whitespace: String,
    calls: Vec<ToolCall>,
    error: Option<String>,
    finished: bool,
}

impl OutputParser {
    pub fn new(arch: &str, tools: &[Tool], choice: &ToolChoice) -> Self {
        let arch = is_qwen35(arch);
        let error = arch
            .as_ref()
            .err()
            .cloned()
            .or_else(|| validate_tools(tools, choice).err());
        Self {
            qwen35: arch.unwrap_or(false),
            tools: tools
                .iter()
                .map(|t| (t.name.clone(), t.parameters.clone()))
                .collect(),
            choice: choice.clone(),
            pending: String::new(),
            in_call: false,
            scan: 0,
            quoted: false,
            escaped: false,
            text: String::new(),
            post_call_whitespace: String::new(),
            calls: vec![],
            error,
            finished: false,
        }
    }

    pub fn push(&mut self, text: &str) -> Result<Vec<Delta>, String> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        if self.finished {
            return Err("Output parser already finished".into());
        }
        self.pending.push_str(text);
        let result = self.drain();
        if let Err(error) = &result {
            self.error = Some(error.clone());
        }
        result
    }

    pub fn finish(&mut self) -> Result<Vec<Delta>, String> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        if self.finished {
            return Ok(vec![]);
        }
        let result = (|| {
            let mut deltas = self.drain()?;
            if self.in_call
                || (self.pending.len() > 1
                    && MARKERS
                        .iter()
                        .any(|tag| tag.starts_with(self.pending.as_str())))
            {
                return Err("truncated tool call or delimiter at end of generation".into());
            }
            if !self.pending.is_empty() {
                let text = std::mem::take(&mut self.pending);
                self.emit_text(text, &mut deltas)?;
            }
            match &self.choice {
                ToolChoice::Required if self.calls.is_empty() => {
                    return Err("Required tool choice produced no tool call".into())
                }
                ToolChoice::Named(name)
                    if self.calls.is_empty()
                        || self.calls.iter().any(|call| &call.name != name) =>
                {
                    return Err(format!(
                        "Named tool choice must produce only calls to {name}"
                    ))
                }
                _ => {}
            }
            self.finished = true;
            Ok(deltas)
        })();
        if let Err(error) = &result {
            self.error = Some(error.clone());
        }
        result
    }

    pub fn text(&self) -> &str {
        &self.text
    }
    pub fn calls(&self) -> &[ToolCall] {
        &self.calls
    }

    fn emit_text(&mut self, text: String, deltas: &mut Vec<Delta>) -> Result<(), String> {
        if !self.calls.is_empty() && text.trim().is_empty() {
            self.post_call_whitespace.push_str(&text);
            return Ok(());
        }
        if !self.calls.is_empty() && !text.trim().is_empty() {
            return Err("Generated non-whitespace text after a tool call".into());
        }
        if !text.is_empty() {
            let text = format!("{}{text}", std::mem::take(&mut self.post_call_whitespace));
            self.text.push_str(&text);
            deltas.push(Delta::Text(text));
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<Vec<Delta>, String> {
        let mut deltas = vec![];
        loop {
            if self.in_call {
                if let Some(end) = self.call_close() {
                    if end > MAX_CALL_BYTES {
                        return Err("Tool call exceeds the 1 MiB parse limit".into());
                    }
                    let (name, arguments) = self.parse_call(&self.pending[..end])?;
                    let index = self.calls.len();
                    let id = format!("call_{}", next_call_id());
                    deltas.push(Delta::ToolStart {
                        index,
                        id: id.clone(),
                        name: name.clone(),
                    });
                    deltas.push(Delta::ToolArguments {
                        index,
                        fragment: Value::Object(arguments.clone()).to_string(),
                    });
                    deltas.push(Delta::ToolEnd { index });
                    self.calls.push(ToolCall {
                        id,
                        name,
                        arguments: Value::Object(arguments),
                    });
                    self.pending.drain(..end + CLOSE.len());
                    self.in_call = false;
                    self.scan = 0;
                    self.quoted = false;
                    self.escaped = false;
                    continue;
                }
                if self.pending.len() > MAX_CALL_BYTES {
                    return Err("Tool call exceeds the 1 MiB parse limit".into());
                }
                break;
            }
            let marker = MARKERS
                .iter()
                .filter_map(|tag| self.pending.find(tag).map(|i| (i, *tag)))
                .min_by_key(|(i, _)| *i);
            if let Some((position, _)) = marker {
                if position > 0 {
                    let text = self.pending.drain(..position).collect();
                    self.emit_text(text, &mut deltas)?;
                }
                if self.pending.starts_with(OPEN) {
                    if self.tools.is_empty() || matches!(self.choice, ToolChoice::None) {
                        return Err("Model generated a tool call while tools are disabled".into());
                    }
                    self.pending.drain(..OPEN.len());
                    self.post_call_whitespace.clear();
                    self.in_call = true;
                    continue;
                }
                if OPEN.starts_with(self.pending.as_str()) {
                    break;
                }
                return Err("Malformed or unexpected tool delimiter outside a tool call".into());
            }
            let hold = (1..=self.pending.len().min(OPEN.len()))
                .rev()
                .find(|&n| {
                    let start = self.pending.len() - n;
                    self.pending.is_char_boundary(start)
                        && MARKERS
                            .iter()
                            .any(|tag| tag.starts_with(&self.pending[start..]))
                })
                .unwrap_or(0);
            let text = self.pending.drain(..self.pending.len() - hold).collect();
            self.emit_text(text, &mut deltas)?;
            break;
        }
        Ok(deltas)
    }

    // Keep quote state and scan position across chunks: a closing-tag literal
    // inside a Qwen3 JSON string is data, and each byte is scanned only once.
    fn call_close(&mut self) -> Option<usize> {
        let bytes = self.pending.as_bytes();
        while self.scan < bytes.len() {
            let i = self.scan;
            if self.quoted {
                if self.escaped {
                    self.escaped = false;
                } else if bytes[i] == b'\\' {
                    self.escaped = true;
                } else if bytes[i] == b'"' {
                    self.quoted = false;
                }
            } else if !self.qwen35 && bytes[i] == b'"' {
                self.quoted = true;
            } else if bytes[i] == b'<' {
                let suffix = &bytes[i..];
                if suffix.starts_with(CLOSE.as_bytes()) {
                    return Some(i);
                }
                if CLOSE.as_bytes().starts_with(suffix) {
                    return None;
                }
            }
            self.scan += 1;
        }
        None
    }

    fn parse_call(&self, body: &str) -> Result<(String, Map<String, Value>), String> {
        let (name, arguments) = if self.qwen35 {
            let body = body
                .trim()
                .strip_prefix("<function=")
                .ok_or("Qwen35 tool call must contain a function tag")?;
            let (name, parameters) = body.split_once('>').ok_or("Malformed function tag")?;
            let parameters = parameters
                .strip_suffix("</function>")
                .ok_or("Missing closing function tag")?;
            let schema = self
                .tools
                .get(name)
                .ok_or_else(|| format!("Unknown tool function: {name}"))?;
            let properties = schema.get("properties").and_then(Value::as_object);
            let mut rest = parameters.trim();
            let mut arguments = Map::new();
            while !rest.is_empty() {
                rest = rest
                    .strip_prefix("<parameter=")
                    .ok_or("Malformed parameter tag")?;
                let (key, after) = rest.split_once('>').ok_or("Malformed parameter name")?;
                let field = properties
                    .and_then(|fields| fields.get(key))
                    .ok_or_else(|| format!("Unknown tool parameter: {key}"))?;
                if arguments.contains_key(key) {
                    return Err(format!("Duplicate tool parameter: {key}"));
                }
                let (raw, after) = after
                    .split_once("</parameter>")
                    .ok_or("Missing closing parameter tag")?;
                // Remove only the template framing, not the string's whitespace.
                let raw = raw.strip_prefix('\n').unwrap_or(raw);
                let raw = raw.strip_suffix('\n').unwrap_or(raw);
                let value = parameter_value(raw, field)?;
                arguments.insert(key.into(), value);
                rest = after.trim();
            }
            (name.to_owned(), arguments)
        } else {
            let value: Value = serde_json::from_str(body)
                .map_err(|error| format!("Malformed Qwen3 tool JSON: {error}"))?;
            let call = value
                .as_object()
                .ok_or("Qwen3 tool call must be a JSON object")?;
            if call.len() != 2 {
                return Err("Qwen3 tool call must contain only name and arguments".into());
            }
            let name = call
                .get("name")
                .and_then(Value::as_str)
                .ok_or("Tool call is missing its function name")?;
            let arguments = call
                .get("arguments")
                .and_then(Value::as_object)
                .ok_or("Tool call arguments must be a JSON object")?;
            (name.to_owned(), arguments.clone())
        };
        let schema = self
            .tools
            .get(&name)
            .ok_or_else(|| format!("Unknown tool function: {name}"))?;
        validate_arguments(&arguments, schema)?;
        Ok((name, arguments))
    }
}

fn next_call_id() -> String {
    // stdlib uniqueness across requests without introducing another dependency.
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{t:x}_{n:x}")
}

fn matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "null" => value.is_null(),
        _ => false,
    }
}

fn valid_type(value: &Value, schema: &Value) -> bool {
    match schema.get("type") {
        Some(Value::String(kind)) => matches_type(value, kind),
        Some(Value::Array(kinds)) => kinds
            .iter()
            .filter_map(Value::as_str)
            .any(|kind| matches_type(value, kind)),
        None => true,
        _ => false,
    }
}

fn parameter_value(raw: &str, schema: &Value) -> Result<Value, String> {
    if schema.get("type").is_none() || schema.get("type").and_then(Value::as_str) == Some("string")
    {
        return Ok(Value::String(raw.into()));
    }
    let json = match raw.trim() {
        "True" => "true",
        "False" => "false",
        "None" => "null",
        value => value,
    };
    if let Ok(value) = serde_json::from_str::<Value>(json) {
        if valid_type(&value, schema) {
            return Ok(value);
        }
    }
    let string = Value::String(raw.into());
    if valid_type(&string, schema) {
        return Ok(string);
    }
    Err(format!(
        "Tool parameter value does not match declared type {}",
        schema.get("type").unwrap_or(&Value::Null)
    ))
}

fn validate_arguments(arguments: &Map<String, Value>, schema: &Value) -> Result<(), String> {
    let properties = schema.get("properties").and_then(Value::as_object);
    for (name, value) in arguments {
        let field = properties
            .and_then(|fields| fields.get(name))
            .ok_or_else(|| format!("Unknown tool parameter: {name}"))?;
        if !valid_type(value, field) {
            return Err(format!(
                "Tool parameter {name} does not match its declared type"
            ));
        }
    }
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for name in required.iter().filter_map(Value::as_str) {
            if !arguments.contains_key(name) {
                return Err(format!("Missing required tool parameter: {name}"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetaValue, MetaValueType};
    use std::collections::HashMap;

    fn tools() -> Vec<Tool> {
        vec![Tool {
            name: "lookup".into(),
            description: Some("Find a city".into()),
            parameters: json!({"type":"object","properties":{"city":{"type":"string"},"count":{"type":"integer"},"ratio":{"type":"number"},"ok":{"type":"boolean"},"filter":{"type":"object"},"items":{"type":"array"},"empty":{"type":"null"}},"required":["city"]}),
        }]
    }

    fn replay(arch: &str, raw: &str, split: usize) -> (String, Vec<ToolCall>) {
        let mut parser = OutputParser::new(arch, &tools(), &ToolChoice::Auto);
        let mut events = parser.push(&raw[..split]).unwrap();
        events.extend(parser.push(&raw[split..]).unwrap());
        events.extend(parser.finish().unwrap());
        let mut text = String::new();
        let mut streamed: Vec<(String, String, String, bool)> = vec![];
        for event in events {
            match event {
                Delta::Text(value) => {
                    assert!(!value.contains("tool_call"));
                    text.push_str(&value);
                }
                Delta::ToolStart { index, id, name } => {
                    assert_eq!(index, streamed.len());
                    streamed.push((id, name, String::new(), false));
                }
                Delta::ToolArguments { index, fragment } => streamed[index].2.push_str(&fragment),
                Delta::ToolEnd { index } => streamed[index].3 = true,
            }
        }
        assert_eq!(text, parser.text());
        assert_eq!(streamed.len(), parser.calls().len());
        for (wire, call) in streamed.iter().zip(parser.calls()) {
            assert_eq!((&wire.0, &wire.1), (&call.id, &call.name));
            assert_eq!(
                serde_json::from_str::<Value>(&wire.2).unwrap(),
                call.arguments
            );
            assert!(wire.3);
        }
        (text, parser.calls().to_vec())
    }

    #[test]
    fn qwen3_calls_do_not_leak_at_any_character_boundary() {
        let raw = "天气：<tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":\"杭州 </tool_call>\"}}</tool_call>\n<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"city\":\"南京\"}}\n</tool_call>";
        for split in raw.char_indices().map(|(i, _)| i).chain([raw.len()]) {
            let (text, calls) = replay("qwen3", raw, split);
            assert_eq!(text, "天气：");
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].arguments["city"], "杭州 </tool_call>");
            assert_ne!(calls[0].id, calls[1].id);
        }
        let mut parser = OutputParser::new("qwen3", &tools(), &ToolChoice::Auto);
        for ch in raw.chars() {
            parser.push(&ch.to_string()).unwrap();
        }
        parser.finish().unwrap();
        assert_eq!(parser.text(), "天气：");
        assert_eq!(parser.calls().len(), 2);
    }

    #[test]
    fn qwen3_rejects_text_after_a_tool_call_at_any_character_boundary() {
        let raw = "prefix<tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":\"杭州\"}}</tool_call> suffix";
        for split in raw
            .char_indices()
            .map(|(index, _)| index)
            .chain([raw.len()])
        {
            let mut parser = OutputParser::new("qwen3", &tools(), &ToolChoice::Auto);
            let result = parser
                .push(&raw[..split])
                .and_then(|_| parser.push(&raw[split..]))
                .and_then(|_| parser.finish());
            assert!(
                result.unwrap_err().contains("after a tool call"),
                "split={split}"
            );
            assert_eq!(parser.text(), "prefix");
        }
    }

    #[test]
    fn qwen35_parameters_keep_schema_types_and_string_whitespace() {
        let raw = "查询：<tool_call>\n<function=lookup>\n<parameter=city>\n 杭州 \n</parameter>\n<parameter=count>2</parameter><parameter=ratio>0.5</parameter><parameter=ok>True</parameter><parameter=filter>{\"x\":1}</parameter><parameter=items>[1,\"a\"]</parameter><parameter=empty>None</parameter></function>\n</tool_call>";
        for split in raw.char_indices().map(|(i, _)| i).chain([raw.len()]) {
            let (text, calls) = replay("qwen35", raw, split);
            assert_eq!(text, "查询：");
            assert_eq!(
                calls[0].arguments,
                json!({"city":" 杭州 ","count":2,"ratio":0.5,"ok":true,"filter":{"x":1},"items":[1,"a"],"empty":null})
            );
        }
        let mut parser = OutputParser::new("qwen35", &tools(), &ToolChoice::Auto);
        for ch in raw.chars() {
            parser.push(&ch.to_string()).unwrap();
        }
        parser.finish().unwrap();
        assert_eq!(parser.calls().len(), 1);
    }

    #[test]
    fn normal_text_streams_before_finish_and_partial_markers_stay_private() {
        let mut parser = OutputParser::new("qwen3", &tools(), &ToolChoice::Auto);
        assert!(matches!(parser.push("你好").unwrap().as_slice(), [Delta::Text(s)] if s == "你好"));
        assert!(parser.push("<tool_").unwrap().is_empty());
        assert!(parser.finish().unwrap_err().contains("truncated"));
        let mut parser = OutputParser::new("qwen3", &[], &ToolChoice::Auto);
        parser.push("a < b").unwrap();
        parser.finish().unwrap();
        assert_eq!(parser.text(), "a < b");
    }

    #[test]
    fn qwen35_multiple_calls_and_bounded_truncation() {
        let call =
            "<tool_call><function=lookup><parameter=city>杭州</parameter></function></tool_call>";
        let raw = format!("{call}\n{call}");
        for split in raw.char_indices().map(|(i, _)| i).chain([raw.len()]) {
            let (text, calls) = replay("qwen35", &raw, split);
            assert_eq!(text, "");
            assert_eq!(calls.len(), 2);
        }
        for (arch, raw) in [
            (
                "qwen3",
                "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":\"杭州\"}}",
            ),
            (
                "qwen35",
                "<tool_call><function=lookup><parameter=city>杭州</parameter></function>",
            ),
        ] {
            let mut parser = OutputParser::new(arch, &tools(), &ToolChoice::Auto);
            parser.push(raw).unwrap();
            assert!(parser.finish().unwrap_err().contains("truncated"));
            assert!(parser.calls().is_empty());
        }
        let mut parser = OutputParser::new("qwen3", &tools(), &ToolChoice::Auto);
        assert!(parser
            .push(&format!("<tool_call>{}", "x".repeat(MAX_CALL_BYTES + 1)))
            .unwrap_err()
            .contains("parse limit"));
    }

    #[test]
    fn malformed_unknown_and_wrong_architecture_calls_fail() {
        let cases = [
            ("qwen3", "<tool_call>{bad}</tool_call>"),
            ("qwen3", "<tool_call>{\"name\":\"unknown\",\"arguments\":{}}</tool_call>"),
            ("qwen3", "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"unknown\":1}}</tool_call>"),
            ("qwen3", "<tool_call>{\"name\":\"lookup\",\"arguments\":{}}</tool_call>"),
            ("qwen3", "<tool_call><function=lookup><parameter=city>x</parameter></function></tool_call>"),
            ("qwen35", "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":\"x\"}}</tool_call>"),
            ("qwen35", "<tool_call><function=lookup><parameter=city>x</parameter><parameter=count>2.5</parameter></function></tool_call>"),
            ("qwen35", "<tool_call><function=lookup><parameter=city>x</parameter><parameter=city>y</parameter></function></tool_call>"),
            ("qwen35", "<tool_call><function=lookup><parameter=wat>x</parameter></function></tool_call>"),
            ("qwen35", "<tool_call><function=lookup><parameter=city>x</parameter></function></tool_call>suffix"),
        ];
        for (arch, raw) in cases {
            let mut parser = OutputParser::new(arch, &tools(), &ToolChoice::Auto);
            assert!(
                parser.push(raw).and_then(|_| parser.finish()).is_err(),
                "{arch}: {raw}"
            );
        }
        for choice in [ToolChoice::Auto, ToolChoice::None] {
            let mut parser = OutputParser::new("qwen3", &[], &choice);
            assert!(parser.push("<tool_call>{}</tool_call>").is_err());
        }
        let mut parser = OutputParser::new("qwen3", &tools(), &ToolChoice::None);
        assert!(parser.push("<tool_call>{}</tool_call>").is_err());
    }

    #[test]
    fn required_and_named_choices_check_actual_calls_at_finish() {
        for choice in [ToolChoice::Required, ToolChoice::Named("lookup".into())] {
            let mut parser = OutputParser::new("qwen3", &tools(), &choice);
            parser.push("normal answer").unwrap();
            assert!(parser.finish().is_err());
        }
        let mut two = tools();
        two.push(Tool {
            name: "other".into(),
            description: None,
            parameters: json!({"type":"object","properties":{}}),
        });
        let mut parser = OutputParser::new("qwen3", &two, &ToolChoice::Named("lookup".into()));
        parser
            .push("<tool_call>{\"name\":\"other\",\"arguments\":{}}</tool_call>")
            .unwrap();
        assert!(parser.finish().is_err());
    }

    fn tokenizer() -> BPETokenizer {
        let mut tokens: Vec<String> = (0..=255u32)
            .map(|b| {
                let visible =
                    (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
                let cp = if visible {
                    b
                } else {
                    256 + (0..b)
                        .filter(|v| {
                            !((33..=126).contains(v)
                                || (161..=172).contains(v)
                                || (174..=255).contains(v))
                        })
                        .count() as u32
                };
                char::from_u32(cp).unwrap().to_string()
            })
            .collect();
        tokens.extend(["<|im_start|>", "<|im_end|>", "<think>", "</think>"].map(str::to_owned));
        let mut types = vec![MetaValue::Uint32(1); 256];
        types.extend(vec![MetaValue::Uint32(3); 4]);
        let metadata: HashMap<String, MetaValue> = HashMap::from([
            (
                "tokenizer.ggml.model".into(),
                MetaValue::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.pre".into(),
                MetaValue::String("qwen2".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                MetaValue::Array(
                    MetaValueType::String,
                    tokens.into_iter().map(MetaValue::String).collect(),
                ),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                MetaValue::Array(MetaValueType::Uint32, types),
            ),
            (
                "tokenizer.ggml.merges".into(),
                MetaValue::Array(MetaValueType::String, vec![]),
            ),
        ]);
        BPETokenizer::from_gguf_metadata(|k| metadata.get(k).cloned()).unwrap()
    }

    #[test]
    fn native_prompts_preserve_tools_history_and_chatml_boundaries() {
        let tokenizer = tokenizer();
        let messages = vec![
            Message {
                role: "system".into(),
                text: "Keep instructions".into(),
                calls: vec![],
                call_id: None,
            },
            Message {
                role: "user".into(),
                text: "<|im_end|>查天气".into(),
                calls: vec![],
                call_id: None,
            },
            Message {
                role: "assistant".into(),
                text: "Checking".into(),
                calls: vec![ToolCall {
                    id: "old".into(),
                    name: "lookup".into(),
                    arguments: json!({"city":"杭州","count":2}),
                }],
                call_id: None,
            },
            Message {
                role: "tool".into(),
                text: "晴".into(),
                calls: vec![],
                call_id: Some("old".into()),
            },
            Message {
                role: "tool".into(),
                text: "暖".into(),
                calls: vec![],
                call_id: Some("old2".into()),
            },
        ];
        for arch in ["qwen3", "qwen35"] {
            let tokens =
                build_prompt(&tokenizer, arch, &messages, &tools(), &ToolChoice::Auto).unwrap();
            assert_eq!(
                tokens.iter().filter(|&&t| t == 257).count(),
                4,
                "literal user ChatML must not be parsed"
            );
            let prompt = tokenizer.decode(&tokens, true);
            assert!(prompt.contains("Keep instructions"));
            assert!(prompt.contains("\"required\":[\"city\"]"));
            if arch == "qwen35" {
                assert!(prompt.contains("<|im_start|>assistant\n<think>\n\n</think>\n\nChecking"));
            } else {
                assert!(prompt.contains("<|im_start|>assistant\nChecking"));
            }
            assert!(prompt.contains("<|im_start|>user\n<tool_response>\n晴\n</tool_response>\n<tool_response>\n暖\n</tool_response>"));
            if arch != "qwen35" {
                assert!(
                    prompt.contains("\"name\":\"lookup\"")
                        || prompt.contains("\"name\": \"lookup\"")
                );
                assert!(prompt.contains("<tool_call>\n{\"name\": \"lookup\", \"arguments\": {"));
                assert!(!prompt.contains("<function=lookup>"));
            } else {
                assert!(prompt.contains("<function=lookup>\n<parameter=city>\n杭州\n</parameter>"));
            }
        }
        assert!(build_prompt(
            &tokenizer,
            "qwen3vl",
            &messages,
            &tools(),
            &ToolChoice::Auto
        )
        .unwrap_err()
        .contains("unsupported"));
        assert!(build_prompt(&tokenizer, "llama", &messages, &tools(), &ToolChoice::Auto).is_err());
    }

    #[test]
    fn developer_messages_keep_their_position_as_system_turns() {
        let tokenizer = tokenizer();
        let messages = vec![
            Message {
                role: "user".into(),
                text: "question".into(),
                calls: vec![],
                call_id: None,
            },
            Message {
                role: "developer".into(),
                text: "later instruction".into(),
                calls: vec![],
                call_id: None,
            },
        ];
        for arch in ["qwen3", "qwen3vl", "qwen35"] {
            let prompt = tokenizer.decode(
                &build_prompt(&tokenizer, arch, &messages, &[], &ToolChoice::Auto).unwrap(),
                true,
            );
            assert!(prompt.contains(
                "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>system\nlater instruction<|im_end|>"
            ));
        }
    }

    #[test]
    fn leading_instructions_merge_into_the_tool_system_turn() {
        let tokenizer = tokenizer();
        for instructions in [
            vec![Message {
                role: "developer".into(),
                text: "developer instruction".into(),
                calls: vec![],
                call_id: None,
            }],
            vec![
                Message {
                    role: "system".into(),
                    text: "system instruction".into(),
                    calls: vec![],
                    call_id: None,
                },
                Message {
                    role: "developer".into(),
                    text: "developer instruction".into(),
                    calls: vec![],
                    call_id: None,
                },
            ],
        ] {
            let mut messages = instructions;
            messages.push(Message {
                role: "user".into(),
                text: "question".into(),
                calls: vec![],
                call_id: None,
            });
            for arch in ["qwen3", "qwen35"] {
                let prompt = tokenizer.decode(
                    &build_prompt(&tokenizer, arch, &messages, &tools(), &ToolChoice::Auto)
                        .unwrap(),
                    true,
                );
                assert_eq!(
                    prompt.matches("<|im_start|>system\n").count(),
                    1,
                    "{arch}: {prompt}"
                );
                let developer = prompt.find("developer instruction").unwrap();
                let user = prompt.find("<|im_start|>user\nquestion").unwrap();
                assert!(developer < user, "{arch}: {prompt}");
                if messages[0].role == "system" {
                    assert!(
                        prompt.find("system instruction").unwrap() < developer,
                        "{arch}: {prompt}"
                    );
                }
            }
        }
    }
}
