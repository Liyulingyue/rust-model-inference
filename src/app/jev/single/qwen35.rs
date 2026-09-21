//! JEV single-mode scorer for qwen35.

use super::super::types::{JevMode, JevQuestionInput, JevResult};
use super::compute_jev_result;
use super::build_jev_prompt;
use super::jev_labels;
use super::print_jev_question;
use super::JevScorer;
use crate::app::cli::{resolve_thread_count, KvFormat};
use super::verify_label_tokens_single;
use super::{PreparedQuestion};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::models::qwen35::{build_qwen35_positions, Qwen35Model, Qwen35Session};
use crate::prompt::{append_qwen_message_tokens};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) fn run_jev_decision_qwen35(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevResult>, String> {
    // Qwen3.5's `Qwen35Model<'a>` borrows from the source. Holding it
    // inside the scorer struct would force the scorer itself to be
    // lifetime-parameterized, which complicates the `JevScorer`
    // trait object story. For Qwen3.5 we keep the per-question loop
    // inline here — the scorer pattern works well for the trunks
    // that own their model data (gemma4, llama-family) or wrap a
    // free function (lfm*/spark/nemotron-h/hunyuan).
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
    verify_label_tokens_single(&tokenizer)?;

    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let pool = Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());

    let mut model = crate::models::qwen35::Qwen35Model::from_source(source.as_ref())
        .map_err(|error| format!("Failed to parse Qwen3.5 model: {error}"))?;
    let n_ctx = model.config.n_ctx;

    let mut results: Vec<JevResult> = Vec::with_capacity(per_question.len());
    for q in per_question {
        let labels = jev_labels(q);
        let (token_ids, payload_str) = build_jev_prompt(&tokenizer, context, q, output_json)?;
        if !output_json {
            print_jev_question(q, &labels);
        }
        let _ = payload_str;

        let (positions, _next) = build_qwen35_positions(&token_ids, None, &[])
            .map_err(|e| format!("Failed to build Qwen3.5 positions: {e}"))?;

        // Recreate session per question (Ephemeral KV).
        let mut session = crate::models::qwen35::Qwen35Session::new_with_prefill_batch_size(
            &mut model,
            n_ctx.min(token_ids.len() + 1),
            prefill_batch_size,
            pool.clone(),
        )?;
        let t0 = Instant::now();
        let logits = session
            .forward_logits(&token_ids, &positions)
            .map_err(|e| format!("Qwen3.5 forward_logits failed: {e}"))?;
        let prefill_dur = t0.elapsed();
        let result = compute_jev_result(q, &tokenizer, &labels, &logits, prefill_dur.as_millis());
        results.push(result);
    }
    Ok(results)
}

