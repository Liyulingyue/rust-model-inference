//! JEV grouped-mode scorer for lfm2.

use super::super::single::lfm2::Lfm2JevScorer;
use super::super::types::{JevGroupedQuestionInput, JevGroupedResult, PreparedGroupedQuestion};
use super::{JevGroupedScorer, allocate_group_labels, build_grouped_payload, build_grouped_system, build_jev_token_ids_for_arch, run_jev_grouped_core};
use super::super::single::{JevScorer, verify_label_tokens_single};
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use std::sync::Arc;
use std::time::{Duration, Instant};


pub(crate) fn run_jev_grouped_lfm2(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevGroupedResult>, String> {
    let available_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = Lfm2JevGroupedScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (LFM2)", n_threads);
    }
    run_jev_grouped_core(source, context, per_question, output_json, &mut scorer)
}

/// LFM2 JEV grouped scorer — uses the free-function prefill
/// path (`run_forward_logits_lfm2_with_batch`). Chat template
/// dispatched through [`build_jev_token_ids_for_arch`] (the
/// `"lfm2"` arm).
struct Lfm2JevGroupedScorer {
    inner: super::super::single::lfm2::Lfm2JevScorer,
}

impl Lfm2JevGroupedScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: super::super::single::lfm2::Lfm2JevScorer::new(source, n_threads, prefill_batch_size)?,
        })
    }
}

impl JevGroupedScorer for Lfm2JevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "LFM2"
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
            "lfm2",
            self.inner.tokenizer(),
            system,
            &payload,
        )?;
        Ok((group_labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::lfm2::run_forward_logits_lfm2_with_batch(
            self.inner.source.as_ref(),
            &token_ids,
            self.inner.n_threads,
            KvFormat::F16,
            8192,
            self.inner.prefill_batch_size,
        )
        .map_err(|e| format!("LFM2 forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}

