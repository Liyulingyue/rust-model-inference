//! JEV single-mode scorer for lfm2.

use super::super::types::{JevMode, JevQuestionInput, JevResult};
use super::jev_system_prompt;
use super::jev_payload_json;
use super::PreparedQuestion;
use super::run_jev_decision_core;
use super::jev_labels;
use super::JevScorer;
use crate::app::cli::{resolve_thread_count, KvFormat};
use super::verify_label_tokens_single;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::lfm2::trunk::forward::run_forward_logits_lfm2_with_batch;
use crate::prompt::append_qwen_message_tokens;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) fn run_jev_decision_lfm2(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevResult>, String> {
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = Lfm2JevScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (LFM2)", n_threads);
    }
    run_jev_decision_core(
        source,
        context,
        per_question,
        output_json,
        &mut scorer,
    )
}

/// LFM2 JEV scorer: uses the free-function prefill path
/// (`run_forward_logits_lfm2_with_batch`). The chat template is
/// `"system\n...\nuser\n{json}\nassistant\n"` which is the LFM2
/// convention shared with LFM2.5 / LFM2-MoE.
pub(crate) struct Lfm2JevScorer {
    pub(crate) tokenizer: BPETokenizer,
    pub(crate) source: Arc<dyn TensorSource>,
    pub(crate) n_threads: usize,
    pub(crate) prefill_batch_size: usize,
}

impl Lfm2JevScorer {
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

impl JevScorer for Lfm2JevScorer {
    fn scorer_label(&self) -> &'static str {
        "LFM2"
    }

    fn build_prompt(
        &self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let system = jev_system_prompt(q.mode);
        let payload = jev_payload_json(context, q)?;
        let mut token_ids = Vec::new();
        if let Some(bos) = self.tokenizer.bos_id() {
            token_ids.push(bos);
        }
        token_ids.extend(self.tokenizer.encode(
            &format!("system\n{system}\n"),
            EncodeOptions {
                add_special: false,
                parse_special: false,
            },
        ));
        token_ids.extend(self.tokenizer.encode(
            &format!("user\n{payload}\n"),
            EncodeOptions {
                add_special: false,
                parse_special: false,
            },
        ));
        token_ids.extend(self.tokenizer.encode(
            "assistant\n",
            EncodeOptions {
                add_special: false,
                parse_special: false,
            },
        ));
        Ok((labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::lfm2::run_forward_logits_lfm2_with_batch(
            self.source.as_ref(),
            &token_ids,
            self.n_threads,
            KvFormat::F16,
            8192,
            self.prefill_batch_size,
        )
        .map_err(|e| format!("LFM2 forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}

