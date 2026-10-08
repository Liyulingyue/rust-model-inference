//! JEV single-mode scorer for llama.

use super::super::types::JevResult;
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
use std::sync::Arc;

pub(crate) fn run_jev_decision_llama(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
    jinja: crate::models::chat_template_jinja::Options,
) -> Result<Vec<JevResult>, String> {
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = LlamaJevScorer::new(source.clone(), n_threads, prefill_batch_size, &jinja)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Llama-family)", n_threads);
    }
    run_jev_decision_core(context, per_question, output_json, &mut scorer)
}

/// Llama-family JEV scorer — covers llama / k2-horizon / granite /
/// nanbeige / qwen2_2 / minicpm. The chat template varies per arch
/// but the forward path is uniform
/// (`run_forward_logits_llama_with_batch`).
pub(crate) struct LlamaJevScorer {
    /// `--jinja` template, resolved once in `new()` where the source is
    /// available; `build_prompt` then renders per question.
    pub(crate) jinja: Option<crate::models::chat_template_jinja::JinjaChatTemplate>,
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
        jinja: &crate::models::chat_template_jinja::Options,
    ) -> Result<Self, String> {
        let jinja = jinja.resolve(&|k| source.metadata(k).cloned())?;
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        let arch = source
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val())
            .unwrap_or_default()
            .to_string();
        Ok(Self {
            jinja,
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
            let ids = crate::models::chat_template_jinja::render_text_conversation(
                &self.tokenizer,
                template,
                Some(system),
                &payload,
                false,
            )?;
            return Ok((labels, ids));
        }
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
        } else if self.arch == "exaone" {
            // EXAONE-3.5 instruct template: `[|user|]{payload}[|endofturn|]\n[|assistant|]`
            // (matches ChatTemplate::Exaone / the CLI's llama_turn_text).
            format!(
                "[|system|]{system}[|endofturn|]\n[|user|]{payload}[|endofturn|]\n[|assistant|]"
            )
        } else if self.arch == "phi3" {
            // Phi-3 / Phi-4: single-turn `<|user|>…<|end|><|assistant|>`
            // with no system role. `payload` already folds system +
            // question into the user message.
            format!("<|user|>{payload}<|end|><|assistant|>")
        } else if self.arch == "glm4" {
            // GLM-4: `[gMASK]<sop>` prefix, then `<|user|>\n{payload}<|assistant|>\n`.
            // Mirrors the CLI's `llama_turn_text` and `run_inference_tokens`.
            // `[gMASK]` is the BOS-like sentinel (id 151329); `<sop>` is 151332.
            // Both are special tokens, recognised as single ids because
            // `parse_special=true`. We deliberately emit the bare user block
            // (no system role) because GLM-4's chat template only has user/assistant.
            format!("[gMASK]<sop><|user|>\n{payload}<|assistant|>\n")
        } else if self.arch == "mistral3" {
            // Mistral 3 (`harshatheg/Ministral-3-3B-Instruct-2512-GGUF`).
            //
            // Empirically, the JSON-shaped JEV payload
            // (`{"context": …, "question": …, "candidates": {"A": …, "B": …}}`)
            // produces an ~81% always-A positional bias on this 3B
            // checkpoint even when the JEV path is wired correctly
            // (`[INST] … [/INST]` template, no think-prefix leakage, no
            // token round-trip drift). Direct `--prompt '[INST] Ask a
            // yes/no question, answer with A or B …'` works correctly
            // with the same tokenizer / forward path, so the bias is
            // specific to the JSON instruction shape, not the model or
            // engine.
            //
            // We pin the chat-template branch here so future maintainers
            // notice the gap; until Mistral 3 picks up a JSON-tuned
            // checkpoint, callers who need decision scoring against
            // Ministral 3 should prefer the `--prompt` path (which
            // returns free-form text, not argmax-via-A/B-token) or
            // hand-roll the candidate set into a natural-language prompt.
            // Mistral's canonical `[INST] … [/INST]` with the JEV system
            // prompt folded inside (Mistral-1 style — Mistral 3's
            // chat_template splits system out into `[SYSTEM_PROMPT]`,
            // but folding it back into `[INST]` is closer to what the
            // 3B checkpoint was actually post-trained on for plain
            // question answering and produces sane per-letter
            // probabilities):
            format!("[INST] {system} {payload} [/INST]")
        } else {
            format!("system\n{system}\nuser\n{payload}\nassistant\n")
        };
        let add_special = self.arch == "nanbeige" || self.arch == "phi3";
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
