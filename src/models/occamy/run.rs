//! CLI entry point for Occamy-1.0.
//!
//! The hybrid trunk and its session are shared with `qwen35`; this only
//! assembles the runtime, encodes the prompt and streams the answer, so the
//! CLI and the HTTP server run the same MoE path.

use crate::app::cli::KvFormat;
use crate::app::text::runtime::{build_text_runtime, RuntimeOptions};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::ops::generation_runtime::{GeneratedText, GenerationRequest, SamplingParams, TokenSink};
use std::sync::Arc;

struct StdoutSink;

impl TokenSink for StdoutSink {
    fn push_text(&mut self, chunk: &str) -> crate::ops::generation_runtime::Flow {
        print!("{chunk}");
        use std::io::Write;
        let _ = std::io::stdout().flush();
        crate::ops::generation_runtime::Flow::Continue
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_inference(
    source: Arc<dyn TensorSource>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    bench: bool,
    kv_format: KvFormat,
    max_context: usize,
    repetition_penalty: f32,
    jinja: &crate::prompt::jinja::Options,
) -> Result<(), String> {
    let _ = kv_format;
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let threads = crate::app::cli::resolve_thread_count(n_threads_arg, available);
    let pool = Arc::new(ComputePool::new(threads));

    let tokenizer = Arc::new(
        BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?,
    );
    let tokens = tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );

    let mut runtime = build_text_runtime(
        "qwen35moe",
        RuntimeOptions {
            threads,
            kv_format,
            max_context,
            prefill_batch_size: 512,
            source,
            pool,
            tokenizer: tokenizer.clone(),
            mmproj: None,
        },
    )?;

    let mut request = GenerationRequest::new(tokens, if bench { 1 } else { max_tokens });
    request.sampling = SamplingParams {
        temperature,
        repetition_penalty,
        ..SamplingParams::default()
    };

    let mut sink = StdoutSink;
    let generated: GeneratedText = runtime.generate(&request, &mut sink)?;
    if bench {
        println!(
            "\nbenchmark completed: {} tokens",
            generated.token_ids.len()
        );
    } else {
        println!("\n{:?}", generated.finish);
    }
    let _ = jinja;
    Ok(())
}
