//! JEV grouped-mode scorer for lfm25.

use super::super::single::JevScorer;
use super::super::types::{JevGroupedResult, PreparedGroupedQuestion};
use super::{
    allocate_group_labels, build_grouped_payload, build_grouped_system,
    build_jev_token_ids_for_arch, run_jev_grouped_core, JevGroupedScorer,
};
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::BPETokenizer;
use std::sync::Arc;

pub(crate) fn run_jev_grouped_lfm25(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevGroupedResult>, String> {
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = Lfm25JevGroupedScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (LFM2.5)", n_threads);
    }
    run_jev_grouped_core(source, context, per_question, output_json, &mut scorer)
}

/// LFM2.5 JEV grouped scorer — wraps [`lfm25::Lfm25JevScorer`] for
/// the chat template / tokenizer + free-function forward path.
struct Lfm25JevGroupedScorer {
    inner: super::super::single::lfm25::Lfm25JevScorer,
}

impl Lfm25JevGroupedScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: super::super::single::lfm25::Lfm25JevScorer::new(
                source,
                n_threads,
                prefill_batch_size,
            )?,
        })
    }
}

impl JevGroupedScorer for Lfm25JevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "LFM2.5"
    }

    fn build_grouped_prompt(
        &self,
        context: &str,
        q: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String> {
        let group_labels = allocate_group_labels(q);
        let system = build_grouped_system();
        let payload = build_grouped_payload(context, q)?;
        let token_ids =
            build_jev_token_ids_for_arch("lfm25", self.inner.tokenizer(), system, &payload)?;
        Ok((group_labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::lfm25::run_forward_logits_lfm25_with_batch(
            self.inner.inner.source.as_ref(),
            &token_ids,
            self.inner.inner.n_threads,
            KvFormat::F16,
            8192,
            self.inner.inner.prefill_batch_size,
        )
        .map_err(|e| format!("LFM2.5 forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}
