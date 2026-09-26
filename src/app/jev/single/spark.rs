//! JEV single-mode scorer for spark.

use super::super::types::JevResult;
use super::jev_labels;
use super::jev_payload_json;
use super::jev_system_prompt;
use super::run_jev_decision_core;
use super::verify_label_tokens_single;
use super::JevScorer;
use super::PreparedQuestion;
use crate::app::cli::resolve_thread_count;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use std::sync::Arc;
use std::time::Instant;

pub(crate) fn run_jev_decision_spark(
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
    let mut scorer = SparkJevScorer::new(source.clone(), n_threads)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Spark)", n_threads);
    }
    let _ = prefill_batch_size;
    run_jev_decision_core(context, per_question, output_json, &mut scorer)
}

/// Spark 2.5 JEV scorer — uses the session API
/// (`SparkSession::forward_logits`) which owns its compute pool
/// internally, so the scorer holds the session rather than the
/// pool + free function.
pub(crate) struct SparkJevScorer {
    pub(crate) tokenizer: BPETokenizer,
    pub(crate) session: crate::models::spark::SparkSession,
}

impl SparkJevScorer {
    pub(crate) fn new(source: Arc<dyn TensorSource>, n_threads: usize) -> Result<Self, String> {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        let session = crate::models::spark::SparkSession::new(
            source.as_ref(),
            Arc::new(ComputePool::new(n_threads)),
            8192,
        )?;
        Ok(Self { tokenizer, session })
    }
}

impl JevScorer for SparkJevScorer {
    fn scorer_label(&self) -> &'static str {
        "Spark"
    }

    fn build_prompt(
        &mut self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let sos = "<｜start▁of▁sentence｜>";
        let eos = "<｜end▁of▁sentence｜>";
        let system = jev_system_prompt(q.mode);
        let payload = jev_payload_json(context, q)?;
        let prompt_text = format!(
            "{sos}<|System|>\n{system}{eos}\
             {sos}<|User|>{payload}{eos}\
             {sos}<|Bot|></think>",
            sos = sos,
            eos = eos,
        );
        let mut token_ids = self.tokenizer.encode(
            &prompt_text,
            EncodeOptions {
                add_special: false,
                parse_special: true,
            },
        );
        if self.tokenizer.add_bos() {
            if let Some(bos) = self.tokenizer.bos_id() {
                token_ids.insert(0, bos);
            }
        }
        Ok((labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        let t0 = Instant::now();
        let logits = self
            .session
            .forward_logits(&token_ids)
            .map_err(|e| format!("Spark forward_logits failed: {e}"))?;
        Ok((logits, t0.elapsed()))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}
