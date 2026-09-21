//! JEV grouped-mode scorer for qwen35.

use super::super::single::verify_label_tokens_single;
use super::super::types::{JevGroupedQuestionInput, JevGroupedResult, PreparedGroupedQuestion};
use super::allocate_group_labels;
use super::build_grouped_payload;
use super::build_grouped_system;
use super::build_jev_token_ids_for_arch;
use super::compute_grouped_jev_result;
use super::run_jev_grouped_core;
use super::JevGroupedScorer;
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::qwen35::run_forward_logits_qwen35_with_batch;
use std::sync::Arc;

pub(crate) fn run_jev_grouped_qwen35(
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
    let mut scorer = Qwen35JevGroupedScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Qwen3.5)", n_threads);
    }
    run_jev_grouped_core(source, context, per_question, output_json, &mut scorer)
}

/// Qwen3.5 JEV grouped scorer — wraps the Qwen3.5 free-function path
/// (zero-copy Qwen35Model<'a> + Qwen35Session per call). Mirrors the
/// shape of the other 8 per-arch grouped scorers.
pub(crate) struct Qwen35JevGroupedScorer {
    pub(crate) tokenizer: BPETokenizer,
    pub(crate) source: Arc<dyn TensorSource>,
    pub(crate) n_threads: usize,
    pub(crate) prefill_batch_size: usize,
}

impl Qwen35JevGroupedScorer {
    pub(crate) fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        Ok(Self {
            tokenizer,
            source,
            n_threads,
            prefill_batch_size,
        })
    }
}

impl JevGroupedScorer for Qwen35JevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "Qwen3.5"
    }

    fn build_grouped_prompt(
        &self,
        context: &str,
        q: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String> {
        let group_labels = allocate_group_labels(q);
        let system = build_grouped_system();
        let payload = build_grouped_payload(context, q)?;
        let token_ids = build_jev_token_ids_for_arch("qwen35", &self.tokenizer, system, &payload)?;
        Ok((group_labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        run_forward_logits_qwen35_with_batch(
            self.source.as_ref(),
            &token_ids,
            self.n_threads,
            KvFormat::F16,
            8192,
            self.prefill_batch_size,
        )
        .map_err(|e| format!("Qwen3.5 grouped forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}