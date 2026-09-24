//! JEV grouped-mode scorer for nemotron_h.

use super::super::single::JevScorer;
use super::super::types::{JevGroupedResult, PreparedGroupedQuestion};
use super::{
    allocate_group_labels, build_grouped_payload, build_grouped_system,
    build_jev_token_ids_for_arch, run_jev_grouped_core, JevGroupedScorer,
};
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::BPETokenizer;
use std::sync::Arc;

pub(crate) fn run_jev_grouped_nemotron_h(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    _prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevGroupedResult>, String> {
    let mut scorer = NemotronHJevGroupedScorer::new(source.clone(), n_threads_arg)?;
    let _ = output_json;
    run_jev_grouped_core(source, context, per_question, false, &mut scorer)
}

/// Nemotron-H JEV grouped scorer — wraps [`nemotron_h::NemotronHJevScorer`]
/// (base model, free-function forward path).
struct NemotronHJevGroupedScorer {
    inner: super::super::single::nemotron_h::NemotronHJevScorer,
}

impl NemotronHJevGroupedScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: super::super::single::nemotron_h::NemotronHJevScorer::new(source, n_threads)?,
        })
    }
}

impl JevGroupedScorer for NemotronHJevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "Nemotron-H"
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
            build_jev_token_ids_for_arch("nemotron_h", self.inner.tokenizer(), system, &payload)?;
        Ok((group_labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::nemotron_h::trunk::run_forward_logits_nemotron_h(
            self.inner.source.clone(),
            &token_ids,
            self.inner.n_threads,
        )
        .map_err(|e| format!("Nemotron-H forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}
