//! JEV single-mode scorer for lfm25.

use super::super::types::{JevMode, JevQuestionInput, JevResult};
use super::lfm2::Lfm2JevScorer;
use super::run_jev_decision_core;
use super::JevScorer;
use super::PreparedQuestion;
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::lfm2::trunk::forward::run_forward_logits_lfm2_with_batch;
use crate::models::lfm25::trunk::forward::run_forward_logits_lfm25_with_batch;
use crate::prompt::append_qwen_message_tokens;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) fn run_jev_decision_lfm25(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevResult>, String> {
    // LFM2.5 chat format is identical to LFM2 — see [`Lfm2JevScorer`].
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = Lfm25JevScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (LFM2.5)", n_threads);
    }
    run_jev_decision_core(source, context, per_question, output_json, &mut scorer)
}

/// LFM2.5 JEV scorer — chat template + free-function forward path,
/// identical to LFM2 but routed at the `crate::models::lfm25`
/// module instead of `lfm2`.
pub(crate) struct Lfm25JevScorer {
    pub(crate) inner: Lfm2JevScorer,
}

impl Lfm25JevScorer {
    pub(crate) fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: Lfm2JevScorer::new(source, n_threads, prefill_batch_size)?,
        })
    }
}

impl JevScorer for Lfm25JevScorer {
    fn scorer_label(&self) -> &'static str {
        "LFM2.5"
    }

    fn build_prompt(
        &self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        self.inner.build_prompt(context, q)
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::lfm25::run_forward_logits_lfm25_with_batch(
            self.inner.source.as_ref(),
            &token_ids,
            self.inner.n_threads,
            KvFormat::F16,
            8192,
            self.inner.prefill_batch_size,
        )
        .map_err(|e| format!("LFM2.5 forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}
