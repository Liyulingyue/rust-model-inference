use serde::{Deserialize, Serialize};
use serde_json::ser::{Formatter, Serializer};
use serde_json::{json, Value};
use std::io;

#[derive(Debug, Deserialize)]
pub struct Request {
    pub state: Value,
    pub questions: serde_json::Map<String, Value>,
    pub max_len: Option<usize>,
    pub head_max_len: Option<usize>,
}

#[derive(Debug)]
pub struct PreparedQuestion {
    pub id: String,
    pub kind: String,
    pub labels: Vec<Value>,
    pub criteria: Value,
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: usize,
}

pub fn prepare(
    tokenizer: &tokenizers::Tokenizer,
    request: &Request,
    max_len: usize,
    head_max_len: usize,
) -> Result<Vec<PreparedQuestion>, String> {
    let max_len = request.max_len.unwrap_or(max_len);
    let head_max_len = request.head_max_len.unwrap_or(head_max_len);
    if !(1..=8192).contains(&max_len) || head_max_len == 0 || head_max_len > max_len {
        return Err("Invalid Laya sequence budgets".into());
    }
    let id = |literal: &str| {
        tokenizer
            .token_to_id(literal)
            .ok_or_else(|| format!("Missing Laya tokenizer token {literal}"))
    };
    let (bos, eos, mask) = (id("<bos>")?, id("<eos>")?, id("<mask>")?);
    let encode = |text: &str| {
        tokenizer
            .encode(text, false)
            .map(|value| value.get_ids().to_vec())
            .map_err(|error| error.to_string())
    };
    let state = serialize_state(&request.state)?.replace("<mask>", " ");
    let state_ids = encode(&state)?;
    let mut prepared = Vec::with_capacity(request.questions.len());
    for (qid, definition) in &request.questions {
        let object = definition
            .as_object()
            .ok_or_else(|| format!("question {qid}: expected object"))?;
        let kind = object
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("question {qid}: missing type"))?;
        let qtype = match kind {
            "choice" => 0,
            "score" => 1,
            "noul" => 2,
            _ => return Err(format!("question {qid}: unsupported type {kind}")),
        };
        if kind != "noul" && object.contains_key("labels") {
            return Err(format!("question {qid}: labels only apply to noul"));
        }
        let instructions = object
            .get("instructions")
            .ok_or_else(|| format!("question {qid}: missing instructions"))?;
        let instructions = render_criterion(instructions).replace("<mask>", " ");
        let criteria = object.get("criteria").cloned().unwrap_or(Value::Null);
        let mut labels = Vec::new();
        let mut options = Vec::new();
        match kind {
            "choice" => match &criteria {
                Value::Object(values) if !values.is_empty() => {
                    for (label, desc) in values {
                        labels.push(json!(label));
                        options.push(if desc.is_null() || desc == "" {
                            label.clone()
                        } else {
                            format!("{label}: {}", render_criterion(desc))
                        });
                    }
                }
                Value::Array(values)
                    if !values.is_empty() && values.iter().all(Value::is_string) =>
                {
                    for label in values {
                        labels.push(label.clone());
                        options.push(label.as_str().unwrap().to_owned());
                    }
                }
                _ => return Err(format!("question {qid}: invalid choice criteria")),
            },
            "score" => match &criteria {
                Value::Array(values)
                    if !values.is_empty() && values.iter().all(|v| !v.is_null()) =>
                {
                    for (i, value) in values.iter().enumerate() {
                        labels.push(json!(i));
                        options.push(format!("level {i}: {}", render_criterion(value)));
                    }
                }
                _ => return Err(format!("question {qid}: invalid score criteria")),
            },
            _ => {
                if !criteria.is_null() && !criteria.is_object() {
                    return Err(format!("question {qid}: invalid noul criteria"));
                }
                if let Some(values) = criteria.as_object() {
                    if values.keys().any(|key| key != "false" && key != "true") {
                        return Err(format!("question {qid}: invalid noul criteria key"));
                    }
                }
                let display = object.get("labels").and_then(Value::as_object);
                if object.contains_key("labels")
                    && display.is_none_or(|v| {
                        v.len() != 2 || !v.contains_key("false") || !v.contains_key("true")
                    })
                {
                    return Err(format!("question {qid}: invalid noul labels"));
                }
                for (key, fallback) in [
                    ("false", "no, the statement does not hold"),
                    ("true", "yes, the statement holds"),
                ] {
                    let label = display
                        .and_then(|v| v.get(key))
                        .and_then(Value::as_str)
                        .unwrap_or(key)
                        .trim();
                    if label.is_empty() {
                        return Err(format!("question {qid}: empty noul label"));
                    }
                    let description = criteria
                        .get(key)
                        .filter(|v| !v.is_null() && *v != "")
                        .map(render_criterion)
                        .unwrap_or_else(|| fallback.into());
                    labels.push(json!(label));
                    options.push(format!("{label}: {description}"));
                }
                if labels[0] == labels[1] {
                    return Err(format!("question {qid}: duplicate noul labels"));
                }
            }
        }
        let mut head = encode(&format!("{kind} question: {instructions}"))?;
        let mut option_ids = Vec::with_capacity(options.len());
        for option in &options {
            let mut tokens = encode(&format!(" {}", option.replace("<mask>", " ")))?;
            tokens.truncate(48);
            tokens.insert(0, mask);
            option_ids.push(tokens);
        }
        let mut budget =
            head_max_len.saturating_sub(option_ids.iter().map(Vec::len).sum::<usize>());
        if budget < 16 {
            let per = 4.max(head_max_len.saturating_sub(16) / option_ids.len());
            for option in &mut option_ids {
                option.truncate(per);
            }
            budget = head_max_len.saturating_sub(option_ids.iter().map(Vec::len).sum::<usize>());
        }
        head.truncate(8.max(budget));
        let mut ids = Vec::with_capacity(max_len);
        ids.push(bos);
        ids.extend(head);
        ids.push(eos);
        let mut markers = Vec::with_capacity(options.len());
        for option in option_ids {
            markers.push(ids.len());
            ids.extend(option);
        }
        ids.push(eos);
        let room = max_len.saturating_sub(ids.len() + 1);
        let selected = if request.state.is_array() {
            &state_ids[state_ids.len().saturating_sub(room)..]
        } else {
            &state_ids[..state_ids.len().min(room)]
        };
        if room > 0 {
            ids.extend_from_slice(selected);
        }
        ids.push(eos);
        ids.truncate(max_len);
        markers.retain(|&position| position < ids.len());
        if markers.len() != labels.len() {
            return Err(format!("question {qid}: options exceed max_len"));
        }
        prepared.push(PreparedQuestion {
            id: qid.clone(),
            kind: kind.into(),
            labels,
            criteria,
            ids,
            markers,
            qtype,
        });
    }
    Ok(prepared)
}

fn render_criterion(value: &Value) -> String {
    if let Some(text) = value.as_str() {
        return text.to_owned();
    }
    python_json(value).expect("serializing a JSON value into a byte vector")
}

fn serialize_state(state: &Value) -> Result<String, String> {
    if let Value::String(text) = state {
        return Ok(text.clone());
    }
    if !state.is_object() && !state.is_array() {
        return Err("Laya state must be a string, object or array".into());
    }
    python_json(state)
}

fn python_json(value: &Value) -> Result<String, String> {
    let mut bytes = Vec::new();
    let mut serializer = Serializer::with_formatter(&mut bytes, PythonJsonFormatter);
    value
        .serialize(&mut serializer)
        .map_err(|error| error.to_string())?;
    String::from_utf8(bytes).map_err(|error| error.to_string())
}

struct PythonJsonFormatter;

impl Formatter for PythonJsonFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }
}

#[cfg(test)]
mod tests {
    use super::serialize_state;
    use serde_json::json;

    #[test]
    fn serializes_structured_state_like_python_json() {
        assert_eq!(
            serialize_state(&json!({"a": [1, 2], "中文": true})).unwrap(),
            r#"{"a": [1, 2], "中文": true}"#
        );
    }
}
