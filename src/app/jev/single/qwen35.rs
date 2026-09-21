//! JEV single-mode scorer for qwen35.

use super::super::types::{JevMode, JevQuestionInput, JevResult};
use super::jev_labels;
use super::jev_payload_json;
use super::jev_system_prompt;
use super::run_jev_decision_core;
use super::verify_label_tokens_single;
use super::JevScorer;
use super::PreparedQuestion;
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::qwen35::run_forward_logits_qwen35_with_batch;
use std::sync::Arc;

pub(crate) fn run_jev_decision_qwen35(
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
    let mut scorer = Qwen35JevScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Qwen3.5)", n_threads);
    }
    run_jev_decision_core(source, context, per_question, output_json, &mut scorer)
}

/// Qwen3.5 JEV scorer. Holds `Arc<dyn TensorSource>` + tokenizer;
/// forwards logits through `run_forward_logits_qwen35_with_batch`,
/// which constructs the `Qwen35Model<'a>` + `Qwen35Session` per call
/// (zero-copy borrow against the source). Mirrors the Llama-family
/// scorer shape so all 9 trunks now follow the same trait path.
pub(crate) struct Qwen35JevScorer {
    pub(crate) tokenizer: BPETokenizer,
    pub(crate) source: Arc<dyn TensorSource>,
    pub(crate) n_threads: usize,
    pub(crate) prefill_batch_size: usize,
}

impl Qwen35JevScorer {
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

impl JevScorer for Qwen35JevScorer {
    fn scorer_label(&self) -> &'static str {
        "Qwen3.5"
    }

    fn build_prompt(
        &self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let system = jev_system_prompt(q.mode);
        let payload = jev_payload_json(context, q)?;
        // Mirrors `qwen3::trunk::forward::run_inference` chat
        // template (system + user + assistant prefix), encoded with
        // BOS prepended.
        let prompt_text = format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{payload}<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        let mut token_ids = self.tokenizer.encode(
            &prompt_text,
            EncodeOptions {
                add_special: false,
                parse_special: true,
            },
        );
        if let Some(bos) = self.tokenizer.bos_id() {
            token_ids.insert(0, bos);
        }
        Ok((labels, token_ids))
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
        .map_err(|e| format!("Qwen3.5 forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}
