//! JEV grouped-mode scorer for qwen35.

use super::super::types::{JevGroupedQuestionInput, JevGroupedResult, PreparedGroupedQuestion};
use super::{allocate_group_labels, build_grouped_payload, build_grouped_system, build_jev_token_ids_for_arch, compute_grouped_jev_result};
use super::super::single::verify_label_tokens_single;
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::models::qwen35::build_qwen35_positions;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use std::sync::Arc;
use std::time::{Duration, Instant};


pub(crate) fn run_jev_grouped_qwen35(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevGroupedResult>, String> {
    // Qwen3.5's `Qwen35Model<'a>` borrows from the source. Keeping
    // it inside a lifetime-parameterized scorer would complicate
    // the trait-object story (same constraint as
    // `run_jev_decision_qwen35`); for now the per-question loop
    // stays inline here.
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
    verify_label_tokens_single(&tokenizer)?;
    let available_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let pool = Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());
    let mut model = crate::models::qwen35::Qwen35Model::from_source(source.as_ref())
        .map_err(|error| format!("Failed to parse Qwen3.5 model: {error}"))?;
    let n_ctx = model.config.n_ctx;
    let mut results = Vec::with_capacity(per_question.len());
    for q in per_question {
        let group_labels = allocate_group_labels(q);
        let system = build_grouped_system();
        let payload = build_grouped_payload(context, q)?;
        let token_ids = build_jev_token_ids_for_arch("qwen35", &tokenizer, system, &payload)?;
        let (positions, _next) = build_qwen35_positions(&token_ids, None, &[])
            .map_err(|e| format!("Failed to build Qwen3.5 positions: {e}"))?;
        let mut session = crate::models::qwen35::Qwen35Session::new_with_prefill_batch_size(
            &mut model, n_ctx.min(token_ids.len() + 1), prefill_batch_size, pool.clone(),
        )?;
        let t0 = Instant::now();
        let logits = session.forward_logits(&token_ids, &positions)
            .map_err(|e| format!("Qwen3.5 forward_logits failed: {e}"))?;
        let prefill_dur = t0.elapsed();
        results.push(compute_grouped_jev_result(q, &tokenizer, &group_labels, &logits, prefill_dur.as_millis()));
    }
    Ok(results)
}

