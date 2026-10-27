//! JEV single-mode scorer for NVIDIA Nemotron-Labs Dory 1.0
//! (`general.architecture = "dory"`).

use super::super::types::JevResult;
use super::{
    jev_labels, jev_payload_json, jev_system_prompt, run_jev_decision_core,
    verify_label_tokens_single,
};
use super::{JevScorer, PreparedQuestion};
use crate::app::cli::resolve_thread_count;
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use std::sync::Arc;

pub(crate) fn run_jev_decision_dory(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
    jinja: crate::prompt::jinja::Options,
) -> Result<Vec<JevResult>, String> {
    let _ = prefill_batch_size;
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = DoryJevScorer::new(source, n_threads, &jinja)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Dory)", n_threads);
    }
    run_jev_decision_core(context, per_question, output_json, &mut scorer)
}

/// Dory JEV scorer.
///
/// Dory is a pure decoder, so the scorer runs one forward pass over the
/// rendered prompt and reads the label logits off the last position — the
/// same shape as the other decoder-based scorers.
///
/// The prompt uses the checkpoint's own ChatML template
/// (`<|im_start|>role\n … <|im_end|>\n` plus an `<|im_start|>assistant\n`
/// generation prompt), so the hand-built fallback below mirrors the
/// Qwen-family ChatML renderer.
pub(crate) struct DoryJevScorer {
    pub(crate) source: Arc<dyn TensorSource>,
    pub(crate) tokenizer: BPETokenizer,
    pub(crate) n_threads: usize,
    pub(crate) jinja: Option<crate::prompt::jinja::JinjaChatTemplate>,
}

impl DoryJevScorer {
    pub(crate) fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        jinja: &crate::prompt::jinja::Options,
    ) -> Result<Self, String> {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to load tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        let jinja = jinja.resolve(&|k| source.metadata(k).cloned())?;
        Ok(Self {
            source,
            tokenizer,
            n_threads,
            jinja,
        })
    }
}

impl JevScorer for DoryJevScorer {
    fn scorer_label(&self) -> &'static str {
        "Dory"
    }

    fn build_prompt(
        &mut self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let system = jev_system_prompt(q.mode);
        let payload = jev_payload_json(context, q)?;
        // `--jinja` renders the model's own template. JEV opens the assistant
        // turn so the next token is the decision being scored
        // (`add_generation_prompt = true`); `thinking` stays off so the scored
        // position does not move behind an injected `<think>` block.
        if let Some(template) = self.jinja.as_ref() {
            let ids = crate::prompt::jinja::render_text_conversation(
                &self.tokenizer,
                template,
                Some(system),
                &payload,
                false,
            )?;
            return Ok((labels, ids));
        }
        // Fallback: the same ChatML shape the checkpoint's template emits,
        // without the optional `<think></think>` block.
        let prompt_text = format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{payload}<|im_end|>\n\
             <|im_start|>assistant\n"
        );
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
        crate::models::dory::run_forward_logits_dory(
            self.source.clone(),
            &token_ids,
            self.n_threads,
        )
        .map_err(|e| format!("Dory forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}
