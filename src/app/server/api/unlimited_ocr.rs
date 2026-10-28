use super::*;
use crate::{app::jev, models::unlimited_ocr::Model};

pub(super) fn options(body: &Value, request: &mut Request) -> Result<(), String> {
    if body.get("temperature").is_none_or(Value::is_null) {
        request.temperature = 0.0;
    }
    for key in [
        "top_p",
        "top_k",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
    ] {
        if body.get(key).is_some_and(|v| !v.is_null()) {
            return Err(format!("Unlimited-OCR does not support {key}"));
        }
    }
    Ok(())
}

fn instruction(request: &Request) -> Result<(String, Vec<Vec<u8>>), String> {
    if request.temperature != 0.0
        || !request.tools.is_empty()
        || !matches!(
            request.choice,
            protocol::ToolChoice::Auto | protocol::ToolChoice::None
        )
        || request.enable_thinking == Some(true)
    {
        return Err(
            "Unlimited-OCR requires temperature 0 and does not support tools or thinking".into(),
        );
    }
    if request.messages.iter().filter(|m| m.role == "user").count() != 1
        || request.messages.iter().any(|m| {
            !matches!(m.role.as_str(), "system" | "developer" | "user")
                || !m.calls.is_empty()
                || m.call_id.is_some()
                || (m.role != "user" && !m.images.is_empty())
        })
    {
        return Err(
            "Unlimited-OCR accepts one user turn with page images and optional system instructions"
                .into(),
        );
    }
    let mut parts: Vec<_> = request.instructions.iter().map(String::as_str).collect();
    parts.extend(
        request
            .messages
            .iter()
            .filter(|m| !m.text.is_empty())
            .map(|m| m.text.as_str()),
    );
    let images: Vec<_> = request
        .messages
        .iter()
        .flat_map(|m| m.images.iter().map(|i| i.bytes.clone()))
        .collect();
    if images.is_empty() {
        return Err("Unlimited-OCR requires at least one page image".into());
    }
    Ok((parts.join("\n"), images))
}

pub(super) fn prepare(model: &Model, request: &Request) -> Result<PromptResult, String> {
    let (instruction, images) = instruction(request)?;
    let ids = model.tokens(images.len(), &instruction)?;
    model.check_context(&ids, request.max_tokens)?;
    Ok(PromptResult { ids, images })
}

fn images(source: Option<String>) -> Result<Vec<Vec<u8>>, String> {
    source
        .map(|s| {
            image_input::decode_image_source(&s, image_input::allow_remote()).map(|i| vec![i.bytes])
        })
        .unwrap_or_else(|| Ok(vec![]))
}

pub(super) async fn single(
    state: AppState,
    context: String,
    questions: Vec<JevOptionInput>,
    positive: Option<String>,
    image: Option<String>,
) -> Response {
    let Backend::UnlimitedOcr(native) = state.model.as_ref() else {
        unreachable!()
    };
    let model = native.model.clone();
    let questions: Vec<_> = questions
        .into_iter()
        .map(|q| crate::app::JevQuestionInput {
            text: q.text,
            options: q.options,
        })
        .collect();
    if let Err(e) = crate::app::prepare_jev_questions(&questions, positive.as_deref()) {
        return jev_error(StatusCode::BAD_REQUEST, e);
    }
    let permit = match state.generation_slot.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return jev_error(
                StatusCode::TOO_MANY_REQUESTS,
                "Another generation is running".into(),
            )
        }
    };
    let result = super::super::compute_task(state.compute, move || {
        let _permit = permit;
        let images = images(image)?;
        let results =
            jev::unlimited_ocr::score(&model, &context, &questions, positive.as_deref(), &images)?;
        Ok::<_, String>(json!({"mode":"single", "context":context, "results":results}))
    })
    .await;
    response(result)
}

pub(super) async fn grouped(
    state: AppState,
    context: String,
    questions: Vec<JevGroupedOptionInput>,
    mode: crate::app::JevMode,
    image: Option<String>,
) -> Response {
    let Backend::UnlimitedOcr(native) = state.model.as_ref() else {
        unreachable!()
    };
    let model = native.model.clone();
    let questions: Vec<_> = questions
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
    if let Err(e) = crate::app::prepare_jev_grouped_questions(&questions, mode) {
        return jev_error(StatusCode::BAD_REQUEST, e);
    }
    let permit = match state.generation_slot.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return jev_error(
                StatusCode::TOO_MANY_REQUESTS,
                "Another generation is running".into(),
            )
        }
    };
    let result = super::super::compute_task(state.compute, move || {
        let _permit = permit;
        let images = images(image)?;
        let results =
            jev::unlimited_ocr::score_grouped(&model, &context, &questions, mode, &images)?;
        let mode = if mode == crate::app::JevMode::MultiSelect {
            "multi_select"
        } else {
            "block_choice"
        };
        Ok::<_, String>(json!({"mode":mode, "context":context, "results":results}))
    })
    .await;
    response(result)
}

fn response(result: Result<Result<Value, String>, tokio::task::JoinError>) -> Response {
    match result {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(e)) => jev_error(StatusCode::BAD_REQUEST, e),
        Err(e) => jev_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anthropic_token_count_uses_native_greedy_default() {
        let body = json!({"messages":[{"role":"user", "content":"Parse"}], "max_tokens":1});
        let mut request = Protocol::Anthropic.parse(&body).unwrap();
        request.messages[0].images = vec![image_input::ImageRef::new(vec![1], "page")];
        assert!(instruction(&request).is_err());
        options(&body, &mut request).unwrap();
        assert!(instruction(&request).is_ok());
    }
    #[test]
    fn ocr_output_keeps_literal_tool_and_thinking_markers() {
        let mut parser = tools::OutputParser::new(
            crate::models::unlimited_ocr::MODEL_TYPE,
            &[],
            &protocol::ToolChoice::None,
        );
        for text in ["<PAGE>", "<tool_", "call>", "<think>scanned text</think>"] {
            parser.push(text).unwrap();
        }
        parser.finish().unwrap();
        assert_eq!(
            parser.text(),
            "<PAGE><tool_call><think>scanned text</think>"
        );
        assert!(parser.calls().is_empty());
    }
    #[test]
    fn page_order_defaults_and_unsupported_options_are_explicit() {
        let body = json!({"messages":[{"role":"user", "content":"Parse"}]});
        let mut request = Protocol::Chat.parse(&body).unwrap();
        options(&body, &mut request).unwrap();
        assert_eq!(request.temperature, 0.0);
        request.messages[0].images = vec![
            image_input::ImageRef::new(vec![1], "first"),
            image_input::ImageRef::new(vec![2], "second"),
        ];
        let (text, images) = instruction(&request).unwrap();
        assert_eq!(text, "Parse");
        assert_eq!(images, [vec![1], vec![2]]);
        assert!(options(&json!({"top_p":0.9}), &mut request).is_err());
        request.temperature = 0.5;
        assert!(instruction(&request).is_err());
        request.temperature = 0.0;
        let mut assistant = request.messages[0].clone();
        assistant.role = "assistant".into();
        request.messages.push(assistant);
        assert!(instruction(&request).is_err());
    }
}
