//! JEV single-mode scorer for gemma4.

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
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::gemma4::{Gemma4Model, Gemma4Session};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) fn run_jev_decision_gemma4(
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
    let mut scorer = Gemma4JevScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Gemma4)", n_threads);
    }
    run_jev_decision_core(source, context, per_question, output_json, &mut scorer)
}

/// Gemma4 JEV scorer — uses the session API. The session is
/// recreated per question to mimic the legacy ephemeral-KV
/// behaviour (each question is a fresh prefill).
pub(crate) struct Gemma4JevScorer {
    pub(crate) tokenizer: BPETokenizer,
    pub(crate) model: crate::models::gemma4::Gemma4Model,
    pub(crate) prefill_batch_size: usize,
}

impl Gemma4JevScorer {
    pub(crate) fn new(
        source: Arc<dyn TensorSource>,
        _n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        let model = crate::models::gemma4::Gemma4Model::from_source(source.clone(), _n_threads)
            .map_err(|e| format!("Failed to load Gemma4 model: {e}"))?;
        Ok(Self {
            tokenizer,
            model,
            prefill_batch_size,
        })
    }
}

impl JevScorer for Gemma4JevScorer {
    fn scorer_label(&self) -> &'static str {
        "Gemma4"
    }

    fn build_prompt(
        &self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let system = jev_system_prompt(q.mode);
        let payload = jev_payload_json(context, q)?;
        let prompt_text = format!("{system}\n\n{payload}\n\n<turn|>\n<|turn>model\n");
        let bos = self
            .tokenizer
            .bos_id()
            .ok_or("Gemma4 tokenizer missing BOS")?;
        let mut ids = self.tokenizer.encode(
            &prompt_text,
            EncodeOptions {
                add_special: false,
                parse_special: true,
            },
        );
        if ids.first() != Some(&bos) {
            ids.insert(0, bos);
        }
        Ok((labels, ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        let mut session = crate::models::gemma4::Gemma4Session::new_with_prefill_batch_size(
            &self.model,
            KvFormat::F16,
            self.prefill_batch_size,
        )?;
        let t0 = Instant::now();
        let logits = session
            .forward_logits(&token_ids)
            .map_err(|e| format!("Gemma4 forward_logits failed: {e}"))?;
        Ok((logits, t0.elapsed()))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}
