//! JEV grouped-mode scorer for Mellum 2.x.

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

pub(crate) fn run_jev_grouped_mellum(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
    jinja: crate::prompt::jinja::Options,
) -> Result<Vec<JevGroupedResult>, String> {
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer =
        MellumJevGroupedScorer::new(source.clone(), n_threads, prefill_batch_size, &jinja)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Mellum)", n_threads);
    }
    run_jev_grouped_core(source, context, per_question, output_json, &mut scorer)
}

/// Mellum JEV grouped scorer — wraps
/// [`super::super::single::mellum::MellumJevScorer`] for the tokenizer and
/// template state.
struct MellumJevGroupedScorer {
    inner: super::super::single::mellum::MellumJevScorer,
}

impl MellumJevGroupedScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
        jinja: &crate::prompt::jinja::Options,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: super::super::single::mellum::MellumJevScorer::new(
                source,
                n_threads,
                prefill_batch_size,
                jinja,
            )?,
        })
    }
}

impl JevGroupedScorer for MellumJevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "Mellum"
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
            "mellum",
            self.inner.tokenizer(),
            system,
            &payload,
            self.inner.jinja.as_ref(),
        )?;
        Ok((group_labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::mellum::trunk::run::run_forward_logits_mellum_with_batch(
            Arc::clone(&self.inner.source),
            &token_ids,
            self.inner.n_threads,
            KvFormat::F16,
            8192,
            self.inner.prefill_batch_size,
        )
        .map_err(|e| format!("Mellum forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}
