//! JEV single-mode scorer for Mellum 2.x (`general.architecture = "mellum"`).

use super::super::types::JevResult;
use super::{
    jev_labels, jev_payload_json, jev_system_prompt, run_jev_decision_core,
    verify_label_tokens_single,
};
use super::{JevScorer, PreparedQuestion};
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use std::sync::Arc;

pub(crate) fn run_jev_decision_mellum(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
    jinja: crate::prompt::jinja::Options,
) -> Result<Vec<JevResult>, String> {
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = MellumJevScorer::new(source, n_threads, prefill_batch_size, &jinja)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Mellum)", n_threads);
    }
    run_jev_decision_core(context, per_question, output_json, &mut scorer)
}

/// Mellum JEV scorer.
///
/// The prompt shape is the checkpoint's own ChatML (`role\n content\n` with an
/// `assistant\n` generation prompt, no `<|im_start|>` markers — that is what the
/// `mellum2` GGUF template emits), so the hand-built branch below mirrors
/// [`crate::models::mellum::build_mellum_prompt`] rather than the Qwen3 marker
/// form.
pub(crate) struct MellumJevScorer {
    pub(crate) source: Arc<dyn TensorSource>,
    pub(crate) tokenizer: BPETokenizer,
    pub(crate) n_threads: usize,
    pub(crate) prefill_batch_size: usize,
    pub(crate) jinja: Option<crate::prompt::jinja::JinjaChatTemplate>,
}

impl MellumJevScorer {
    pub(crate) fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
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
            prefill_batch_size,
            jinja,
        })
    }
}

impl JevScorer for MellumJevScorer {
    fn scorer_label(&self) -> &'static str {
        "Mellum"
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
                Some(&system),
                &payload,
                false,
            )?;
            return Ok((labels, ids));
        }
        // `add_bos_token = false` for this checkpoint, so nothing is
        // prepended. `parse_special: false` keeps the ChatML text out of the
        // `mellum2` special-token scan while leaving its digit-splitting and
        // contraction rules in force, which is what generation does too.
        let mut token_ids = Vec::new();
        for chunk in [
            format!("system\n{system}\n"),
            format!("user\n{payload}\n"),
            "assistant\n".to_string(),
        ] {
            token_ids.extend(self.tokenizer.encode(
                &chunk,
                EncodeOptions {
                    add_special: false,
                    parse_special: false,
                },
            ));
        }
        Ok((labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::mellum::trunk::run::run_forward_logits_mellum_with_batch(
            Arc::clone(&self.source),
            &token_ids,
            self.n_threads,
            KvFormat::F16,
            8192,
            self.prefill_batch_size,
        )
        .map_err(|e| format!("Mellum forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}
