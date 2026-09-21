//! JEV grouped-mode scorer for qwen3.

use super::super::single::qwen3::Qwen3JevScorer;
use super::super::types::{JevGroupedQuestionInput, JevGroupedResult, PreparedGroupedQuestion};
use super::{JevGroupedScorer, allocate_group_labels, build_grouped_payload, build_grouped_system, build_jev_token_ids_for_arch, run_jev_grouped_core};
use super::super::single::{JevScorer, verify_label_tokens_single};
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use std::sync::Arc;
use std::time::{Duration, Instant};


pub(crate) fn run_jev_grouped_qwen3(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevGroupedResult>, String> {
    let available_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = Qwen3JevGroupedScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads", scorer.pool().n_threads());
    }
    run_jev_grouped_core(source, context, per_question, output_json, &mut scorer)
}

/// Qwen3 JEV grouped scorer — wraps [`qwen3::Qwen3JevScorer`] for the
/// Qwen3Model + Session API.
struct Qwen3JevGroupedScorer {
    inner: super::super::single::qwen3::Qwen3JevScorer,
}

impl Qwen3JevGroupedScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: super::super::single::qwen3::Qwen3JevScorer::new(source, n_threads, prefill_batch_size)?,
        })
    }

    fn pool(&self) -> Arc<ComputePool> {
        self.inner.pool()
    }
}

impl JevGroupedScorer for Qwen3JevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "Qwen3"
    }

    fn build_grouped_prompt(
        &self,
        context: &str,
        q: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String> {
        let group_labels = allocate_group_labels(q);
        let system = build_grouped_system();
        let payload = build_grouped_payload(context, q)?;
        let token_ids = build_jev_token_ids_for_arch(
            "qwen3",
            self.inner.model.tokenizer(),
            system,
            &payload,
        )?;
        Ok((group_labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        // super::super::single::qwen3::Qwen3JevScorer::forward_logits already runs the session,
        // computes positions, and returns (logits, dur). Delegate
        // rather than re-deriving positions / input.
        self.inner
            .forward_logits(token_ids)
            .map_err(|e| format!("Qwen3 grouped forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}

