use crate::{BPETokenizer, EncodeOptions};

pub struct QwenMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

pub struct HunyuanMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

pub struct Lfm2Message<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

const PLAIN_TEXT: EncodeOptions = EncodeOptions {
    add_special: false,
    parse_special: false,
};

const WITH_SPECIAL: EncodeOptions = EncodeOptions {
    add_special: false,
    parse_special: true,
};

pub fn build_simple_prompt(tokenizer: &BPETokenizer, text: &str) -> Vec<u32> {
    let mut tokens = Vec::new();
    if let Some(bos_id) = tokenizer.bos_id() {
        tokens.push(bos_id);
    }
    tokens.extend(tokenizer.encode(text, PLAIN_TEXT));
    tokens
}

pub fn build_hunyuan_chat_prompt(
    tokenizer: &BPETokenizer,
    messages: &[HunyuanMessage<'_>],
    add_generation_prompt: bool,
) -> Result<Vec<u32>, String> {
    let hy_user = required_control(tokenizer, "hy_user", "<｜hy_User｜>")?;
    let hy_assistant = required_control(tokenizer, "hy_assistant", "<｜hy_Assistant｜>")?;
    let hy_placeholder_2 = required_control(tokenizer, "hy_placeholder_2", "<｜hy_place▁holder▁no▁2｜>")?;
    let hy_placeholder_3 = required_control(tokenizer, "hy_placeholder_3", "<｜hy_place▁holder▁no▁3｜>")?;
    let hy_placeholder_8 = required_control(tokenizer, "hy_placeholder_8", "<｜hy_place▁holder▁no▁8｜>")?;

    let mut output = Vec::new();
    for (i, message) in messages.iter().enumerate() {
        if i == 0 && message.role == "system" {
            output.extend(tokenizer.encode(message.content, PLAIN_TEXT));
            output.push(hy_placeholder_3);
        }
        if message.role == "user" {
            output.push(hy_user);
            output.extend(tokenizer.encode(message.content, PLAIN_TEXT));
            output.push(hy_assistant);
        } else if message.role == "assistant" {
            output.push(hy_assistant);
            output.extend(tokenizer.encode(message.content, PLAIN_TEXT));
            output.push(hy_placeholder_2);
        }
    }
    if add_generation_prompt {
        output.push(hy_assistant);
    } else {
        output.push(hy_placeholder_8);
    }
    Ok(output)
}

fn required_control(tokenizer: &BPETokenizer, name: &str, literal: &str) -> Result<u32, String> {
    tokenizer
        .special_token_id(name)
        .ok_or_else(|| format!("Required ChatML token missing: {literal}"))
}

pub fn build_lfm2_chat_prompt(
    tokenizer: &BPETokenizer,
    messages: &[Lfm2Message<'_>],
) -> Result<Vec<u32>, String> {
    // LFM2 / LFM2.5 chat format is a literal "role\n{content}\n" sequence
    // (no ChatML control tokens). The official Jinja template starts with
    // `bos_token` and ends each turn with a newline, followed by an
    // "assistant\n" prompt for generation. The tokenizer should treat
    // "system", "user", "assistant" as regular tokens (no special tokens).
    let mut out = Vec::new();
    if let Some(bos) = tokenizer.bos_id() {
        out.push(bos);
    }
    for message in messages {
        // The role itself can include a trailing newline so that the role
        // name and the content are separated by a single `\n`. We use
        // `parse_special: false` so the literal "system" / "user" /
        // "assistant" strings get tokenized as ordinary tokens.
        out.extend(tokenizer.encode(
            &format!("{}\n", message.role),
            PLAIN_TEXT,
        ));
        out.extend(tokenizer.encode(message.content, PLAIN_TEXT));
        out.extend(tokenizer.encode("\n", PLAIN_TEXT));
    }
    // Generation prompt: append "assistant\n".
    out.extend(tokenizer.encode("assistant\n", PLAIN_TEXT));
    Ok(out)
}

pub fn build_lfm25_chat_prompt(
    tokenizer: &BPETokenizer,
    messages: &[Lfm2Message<'_>],
) -> Result<Vec<u32>, String> {
    let mut prompt_text = String::new();
    for message in messages {
        prompt_text.push_str("<|im_start|>");
        prompt_text.push_str(message.role);
        prompt_text.push('\n');
        prompt_text.push_str(message.content);
        prompt_text.push('\n');
    }
    prompt_text.push_str("<|im_start|>assistant\n");
    let mut tokens = tokenizer.encode(&prompt_text, WITH_SPECIAL);
    if let Some(bos) = tokenizer.bos_id() {
        tokens.insert(0, bos);
    }
    Ok(tokens)
}

pub fn build_qwen_chat_prompt(
    tokenizer: &BPETokenizer,
    messages: &[QwenMessage<'_>],
    enable_thinking: bool,
) -> Result<Vec<u32>, String> {
    let mut output = Vec::new();
    for message in messages {
        let content = tokenizer.encode(message.content, PLAIN_TEXT);
        append_qwen_message_tokens(&mut output, tokenizer, message.role, &content)?;
    }
    append_qwen_assistant_prefix(&mut output, tokenizer, enable_thinking)?;
    Ok(output)
}

pub fn append_qwen_message_tokens(
    out: &mut Vec<u32>,
    tokenizer: &BPETokenizer,
    role: &str,
    content_tokens: &[u32],
) -> Result<(), String> {
    let im_start = required_control(tokenizer, "im_start", "<|im_start|>")?;
    let im_end = required_control(tokenizer, "im_end", "<|im_end|>")?;
    out.push(im_start);
    out.extend(tokenizer.encode(&format!("{role}\n"), PLAIN_TEXT));
    out.extend_from_slice(content_tokens);
    out.push(im_end);
    out.extend(tokenizer.encode("\n", PLAIN_TEXT));
    Ok(())
}

pub fn append_qwen_assistant_prefix(
    out: &mut Vec<u32>,
    tokenizer: &BPETokenizer,
    enable_thinking: bool,
) -> Result<(), String> {
    out.push(required_control(tokenizer, "im_start", "<|im_start|>")?);
    out.extend(tokenizer.encode("assistant\n", PLAIN_TEXT));
    if !enable_thinking {
        out.extend(tokenizer.encode("<think>\n\n</think>\n\n", PLAIN_TEXT));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetaValue, MetaValueType};
    use std::collections::HashMap;

    fn prompt_tokenizer(include_im_end: bool) -> BPETokenizer {
        let mut tokens = vec!["u", "s", "e", "r", "Ċ", "H", "i", "<|im_start|>"];
        let mut types = vec![1u32; tokens.len()];
        types[7] = 3;
        if include_im_end {
            tokens.push("<|im_end|>");
            types.push(3);
        }
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
                    tokens
                        .into_iter()
                        .map(|value| MetaValue::String(value.into()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                MetaValue::Array(
                    MetaValueType::Uint32,
                    types.into_iter().map(MetaValue::Uint32).collect(),
                ),
            ),
            (
                "tokenizer.ggml.merges".into(),
                MetaValue::Array(MetaValueType::String, vec![]),
            ),
        ]);
        BPETokenizer::from_gguf_metadata(|key| metadata.get(key).cloned()).unwrap()
    }

    #[test]
    fn token_content_uses_the_same_message_envelope() {
        let tokenizer = prompt_tokenizer(true);
        let mut output = Vec::new();
        append_qwen_message_tokens(&mut output, &tokenizer, "user", &[5, 6]).unwrap();
        assert_eq!(output, vec![7, 0, 1, 2, 3, 4, 5, 6, 8, 4]);
    }

    #[test]
    fn missing_chatml_literal_is_an_error() {
        let tokenizer = prompt_tokenizer(false);
        let error = build_qwen_chat_prompt(
            &tokenizer,
            &[QwenMessage {
                role: "user",
                content: "Hi",
            }],
            false,
        )
        .unwrap_err();
        assert!(error.contains("<|im_end|>"), "{error}");
    }
}
