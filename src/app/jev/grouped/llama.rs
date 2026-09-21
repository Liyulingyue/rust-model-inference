//! JEV grouped-mode scorer for llama.

use super::super::single::llama::LlamaJevScorer;
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

pub(crate) fn run_jev_grouped_llama(
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
    let mut scorer = LlamaJevGroupedScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Llama-family)", n_threads);
    }
    run_jev_grouped_core(source, context, per_question, output_json, &mut scorer)
}

/// Llama-family JEV grouped scorer — wraps [`llama::LlamaJevScorer`]
/// for the per-arch chat template + free-function forward path.
struct LlamaJevGroupedScorer {
    inner: super::super::single::llama::LlamaJevScorer,
}

impl LlamaJevGroupedScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: super::super::single::llama::LlamaJevScorer::new(
                source,
                n_threads,
                prefill_batch_size,
            )?,
        })
    }
}

impl JevGroupedScorer for LlamaJevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "Llama-family"
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
            &self.inner.arch,
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
        crate::models::llama::trunk::run_forward_logits_llama_with_batch(
            self.inner.source.as_ref(),
            &token_ids,
            self.inner.n_threads,
            KvFormat::F16,
            8192,
            self.inner.prefill_batch_size,
        )
        .map_err(|e| format!("Llama forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}
