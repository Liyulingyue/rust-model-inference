//! JEV grouped-mode scorer for NVIDIA Nemotron-Labs Dory 1.0.

use super::super::single::JevScorer;
use super::super::types::{JevGroupedResult, PreparedGroupedQuestion};
use super::{
    allocate_group_labels, build_grouped_payload, build_grouped_system,
    build_jev_token_ids_for_arch, run_jev_grouped_core, JevGroupedScorer,
};
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::BPETokenizer;
use std::sync::Arc;

pub(crate) fn run_jev_grouped_dory(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    _prefill_batch_size: usize,
    output_json: bool,
    jinja: crate::prompt::jinja::Options,
) -> Result<Vec<JevGroupedResult>, String> {
    let mut scorer = DoryJevGroupedScorer::new(source.clone(), n_threads_arg, &jinja)?;
    let _ = output_json;
    run_jev_grouped_core(source, context, per_question, false, &mut scorer)
}

/// Dory JEV grouped scorer — wraps
/// [`super::super::single::dory::DoryJevScorer`] so the grouped prompt
/// shape stays the single-mode ChatML one.
struct DoryJevGroupedScorer {
    inner: super::super::single::dory::DoryJevScorer,
}

impl DoryJevGroupedScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        jinja: &crate::prompt::jinja::Options,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: super::super::single::dory::DoryJevScorer::new(source, n_threads, jinja)?,
        })
    }
}

impl JevGroupedScorer for DoryJevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "Dory"
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
            "dory",
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
        crate::models::dory::run_forward_logits_dory(
            self.inner.source.clone(),
            &token_ids,
            self.inner.n_threads,
        )
        .map_err(|e| format!("Dory forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}
