//! JEV grouped-mode scorer for spark.

use super::super::single::spark::SparkJevScorer;
use super::super::single::{verify_label_tokens_single, JevScorer};
use super::super::types::{JevGroupedQuestionInput, JevGroupedResult, PreparedGroupedQuestion};
use super::{
    allocate_group_labels, build_grouped_payload, build_grouped_system,
    build_jev_token_ids_for_arch, run_jev_grouped_core, JevGroupedScorer,
};
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) fn run_jev_grouped_spark(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    _prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevGroupedResult>, String> {
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = SparkJevGroupedScorer::new(source.clone(), n_threads)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Spark)", n_threads);
    }
    run_jev_grouped_core(source, context, per_question, output_json, &mut scorer)
}

/// Spark JEV grouped scorer — wraps [`spark::SparkJevScorer`] for the
/// session API + tokenizer access.
struct SparkJevGroupedScorer {
    inner: super::super::single::spark::SparkJevScorer,
}

impl SparkJevGroupedScorer {
    fn new(source: Arc<dyn TensorSource>, n_threads: usize) -> Result<Self, String> {
        Ok(Self {
            inner: super::super::single::spark::SparkJevScorer::new(source, n_threads)?,
        })
    }
}

impl JevGroupedScorer for SparkJevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "Spark"
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
            build_jev_token_ids_for_arch("spark2_5", self.inner.tokenizer(), system, &payload)?;
        Ok((group_labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        let t0 = Instant::now();
        let logits = self
            .inner
            .session
            .forward_logits(&token_ids)
            .map_err(|e| format!("Spark forward_logits failed: {e}"))?;
        Ok((logits, t0.elapsed()))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}
