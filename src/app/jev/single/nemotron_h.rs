//! JEV single-mode scorer for nemotron_h.

use super::super::types::JevResult;
use super::jev_labels;
use super::jev_payload_json;
use super::jev_system_prompt;
use super::run_jev_decision_core;
use super::verify_label_tokens_single;
use super::JevScorer;
use super::PreparedQuestion;
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use std::sync::Arc;

pub(crate) fn run_jev_decision_nemotron_h(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevResult>, String> {
    let _ = prefill_batch_size;
    let mut scorer = NemotronHJevScorer::new(source.clone(), n_threads_arg)?;
    let _ = output_json;
    run_jev_decision_core(source, context, per_question, false, &mut scorer)
}

/// Nemotron-H JEV scorer — base model (no chat template). Uses
/// the standalone `load_nemotron_tokenizer` + `run_forward_logits_nemotron_h`
/// free function (Nemotron-H has no Session API). `n_threads_arg` is
/// forwarded to the matmul `ComputePool` so JEV inference benefits
/// from the same row-partitioned Q4_0 kernel as the standalone path;
/// `prefill_batch_size` is ignored because Nemotron-H's per-step
/// body still walks the per-token path.
pub(crate) struct NemotronHJevScorer {
    pub(crate) tokenizer: BPETokenizer,
    pub(crate) source: Arc<dyn TensorSource>,
    pub(crate) n_threads: usize,
}

impl NemotronHJevScorer {
    pub(crate) fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
    ) -> Result<Self, String> {
        let tokenizer = crate::models::nemotron_h::trunk::load_nemotron_tokenizer(source.as_ref())?;
        verify_label_tokens_single(&tokenizer)?;
        Ok(Self {
            tokenizer,
            source,
            n_threads,
        })
    }
}

impl JevScorer for NemotronHJevScorer {
    fn scorer_label(&self) -> &'static str {
        "Nemotron-H"
    }

    fn build_prompt(
        &self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let system = jev_system_prompt(q.mode);
        let payload = jev_payload_json(context, q)?;
        let prompt_text = format!("{system}\n\n{payload}\n\nAnswer:");
        let token_ids = self.tokenizer.encode(
            &prompt_text,
            EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        );
        Ok((labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::nemotron_h::trunk::run_forward_logits_nemotron_h(
            self.source.clone(),
            &token_ids,
            self.n_threads,
        )
        .map_err(|e| format!("Nemotron-H forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}
