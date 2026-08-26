//! # Hunyuan Text Inference
//!
//! Builds hunyuan-specific chat prompt, then delegates to base skeleton.

use crate::core::tokenizer::BPETokenizer;
use crate::prompt::{build_hunyuan_chat_prompt, HunyuanMessage};
use crate::core::tensor::TensorSource;
use std::sync::Arc;

pub fn run_inference(
    source: Arc<dyn TensorSource>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    profile: bool,
    kv_format: crate::app::cli::KvFormat,
) -> Result<(), String> {
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;

    let input_tokens = build_hunyuan_chat_prompt(
        &tokenizer,
        &[HunyuanMessage {
            role: "user",
            content: prompt,
        }],
        true,
    )?;

    crate::models::qwen3::base::run_inference_tokens(
        source,
        input_tokens,
        max_tokens,
        temperature,
        n_threads_arg,
        false,
        profile,
        kv_format,
    )
}
