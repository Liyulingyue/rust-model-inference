//! JEV single-mode scorer for falcon-h1.
//!
//! Falcon-H1 has no separate `run_forward_logits_*` entry: its trunk
//! exposes `FalconH1Model::prefill(&[u32], &mut FalconH1Scratch)`, which
//! returns the last-position logits after advancing the KV cache **and**
//! the Mamba2 scan state. That is exactly the JEV contract, so the scorer
//! holds one model + one scratch, resets the scratch per question, and
//! reuses both.
//!
//! The chat template is Qwen-flavoured ChatML
//! (`{role}`), the same markers `prompt::build_qwen_chat_prompt`
//! emits for the HTTP path — so both front-ends render the same prompt.

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
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::falcon_h1::trunk::{FalconH1Model, FalconH1Scratch};
use std::sync::Arc;

/// ChatML markers, spelled with `chr` so this file's own prose can quote
/// the literal text without terminating its own doc comment.
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

pub(crate) fn run_jev_decision_falcon_h1(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedQuestion],
    n_threads_arg: usize,
    _prefill_batch_size: usize,
    output_json: bool,
    jinja: crate::prompt::jinja::Options,
) -> Result<Vec<JevResult>, String> {
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = FalconH1JevScorer::new(source.clone(), n_threads, &jinja)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Falcon-H1)", n_threads);
    }
    run_jev_decision_core(context, per_question, output_json, &mut scorer)
}

pub(crate) struct FalconH1JevScorer {
    /// `--jinja` template, resolved once in `new()` where the source is
    /// available; `build_prompt` then renders per question.
    pub(crate) jinja: Option<crate::prompt::jinja::JinjaChatTemplate>,
    model: FalconH1Model,
    scratch: FalconH1Scratch,
    tokenizer: BPETokenizer,
}

impl FalconH1JevScorer {
    pub(crate) fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        jinja: &crate::prompt::jinja::Options,
    ) -> Result<Self, String> {
        let jinja = jinja.resolve(&|k| source.metadata(k).cloned())?;
        let model = FalconH1Model::from_source(source.clone(), n_threads)
            .map_err(|error| format!("Failed to load Falcon-H1 model: {error}"))?;
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        // Sized for the whole run: a question prompt plus a couple of probe
        // tokens is tiny next to the model's own ctx, but capacity must
        // cover the longest prompt.
        let scratch = FalconH1Scratch::new(&model.config, model.config.n_ctx.max(1));
        Ok(Self {
            jinja,
            model,
            scratch,
            tokenizer,
        })
    }
}

impl JevScorer for FalconH1JevScorer {
    fn scorer_label(&self) -> &'static str {
        "Falcon-H1"
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
        // turn so the next token is the decision being scored, i.e.
        // `add_generation_prompt = true`, the same value generation uses.
        // `thinking` off so the scored position does not move into a
        // reasoning block.
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
        // Same turns `prompt::build_qwen_chat_prompt` emits for HTTP:
        // system + user, then a bare assistant prefix.
        let mut prompt_text = String::new();
        prompt_text.push_str(IM_START);
        prompt_text.push_str("system\n");
        prompt_text.push_str(&system);
        prompt_text.push_str(IM_END);
        prompt_text.push('\n');
        prompt_text.push_str(IM_START);
        prompt_text.push_str("user\n");
        prompt_text.push_str(&payload);
        prompt_text.push_str(IM_END);
        prompt_text.push('\n');
        prompt_text.push_str(IM_START);
        prompt_text.push_str("assistant\n");
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
        // `prefill` mutates KV + SSM state, so reset first: JEV runs one
        // independent forward per question.
        self.scratch.reset();
        let started = std::time::Instant::now();
        let logits = self
            .model
            .prefill(&token_ids, &mut self.scratch)
            .map_err(|error| format!("Falcon-H1 forward_logits failed: {error}"))?;
        Ok((logits, started.elapsed()))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}
