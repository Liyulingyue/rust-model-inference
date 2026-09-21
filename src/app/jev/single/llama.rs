//! JEV single-mode scorer for llama.

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
use crate::models::llama::trunk::forward::run_forward_logits_llama_with_batch;
use crate::prompt::append_qwen_message_tokens;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) fn run_jev_decision_llama(
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
    let mut scorer = LlamaJevScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Llama-family)", n_threads);
    }
    run_jev_decision_core(source, context, per_question, output_json, &mut scorer)
}

/// Llama-family JEV scorer — covers llama / k2-horizon / granite /
/// nanbeige / qwen2_2 / minicpm. The chat template varies per arch
/// but the forward path is uniform
/// (`run_forward_logits_llama_with_batch`).
pub(crate) struct LlamaJevScorer {
    pub(crate) tokenizer: BPETokenizer,
    pub(crate) source: Arc<dyn TensorSource>,
    pub(crate) arch: String,
    pub(crate) n_threads: usize,
    pub(crate) prefill_batch_size: usize,
}

impl LlamaJevScorer {
    pub(crate) fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        let arch = source
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val())
            .unwrap_or_default()
            .to_string();
        Ok(Self {
            tokenizer,
            source,
            arch,
            n_threads,
            prefill_batch_size,
        })
    }
}

impl JevScorer for LlamaJevScorer {
    fn scorer_label(&self) -> &'static str {
        "Llama-family"
    }

    fn build_prompt(
        &self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let system = jev_system_prompt(q.mode);
        let payload = jev_payload_json(context, q)?;
        // Per-arch chat template; mirrors `llama::trunk::forward::run_inference`.
        let prompt_text = if self.arch == "k2-horizon" {
            format!(
                "<|start_of_role|>system<|end_of_role|>{system}<|end_of_text|>\n\
                 <|start_of_role|>user<|end_of_role|>{payload}<|end_of_text|>\n\
                 <|start_of_role|>assistant<|end_of_role|>"
            )
        } else if self.arch == "granite" {
            format!(
                "<|start_of_role|>system<|end_of_role|>{system}<|end_of_text|>\n\
                 <|start_of_role|>user<|end_of_role|>{payload}<|end_of_text|>\n\
                 <|start_of_role|>assistant<|end_of_role|>"
            )
        } else if self.arch == "nanbeige" {
            format!("{system}\n\n{payload}\n\nAnswer:")
        } else {
            format!("system\n{system}\nuser\n{payload}\nassistant\n")
        };
        let add_special = self.arch == "nanbeige";
        let mut token_ids = self.tokenizer.encode(
            &prompt_text,
            EncodeOptions {
                add_special,
                parse_special: true,
            },
        );
        if !add_special {
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
        crate::models::llama::trunk::run_forward_logits_llama_with_batch(
            self.source.as_ref(),
            &token_ids,
            self.n_threads,
            KvFormat::F16,
            8192,
            self.prefill_batch_size,
        )
        .map_err(|e| format!("Llama forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}
