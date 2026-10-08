//! JEV single-mode scorer for qwen35.

use super::super::types::JevResult;
use super::jev_labels;
use super::jev_payload_json;
use super::jev_system_prompt;
use super::run_jev_decision_core;
use super::verify_label_tokens_single;
use super::JevScorer;
use super::PreparedQuestion;
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::app::text::{encode_qwen35_image, inject_vision_embeddings};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::models::qwen35::vision::VisionGrid;
use crate::models::qwen35::{
    build_qwen35_positions, run_forward_logits_qwen35_with_batch, Qwen35Model, Qwen35Session,
};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

pub(crate) fn run_jev_decision_qwen35(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
    mmproj_path: Option<&Path>,
    image_path: Option<&Path>,
    jinja: crate::prompt::jinja::Options,
) -> Result<Vec<JevResult>, String> {
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = Qwen35JevScorer::new(
        source.clone(),
        n_threads,
        prefill_batch_size,
        mmproj_path,
        image_path,
        &jinja,
    )?;
    if !output_json {
        eprintln!("compute pool: {} threads (Qwen3.5)", n_threads);
    }
    run_jev_decision_core(context, per_question, output_json, &mut scorer)
}

/// Qwen3.5 JEV scorer. Holds `Arc<dyn TensorSource>` + tokenizer;
/// forwards logits through `run_forward_logits_qwen35_with_batch`,
/// which constructs the `Qwen35Model<'a>` + `Qwen35Session` per call
/// (zero-copy borrow against the source). Mirrors the Llama-family
/// scorer shape so all 9 trunks now follow the same trait path.
pub(crate) struct Qwen35JevScorer {
    /// `--jinja` template, resolved once in `new()` where the source is
    /// available; `build_prompt` then renders per question.
    pub(crate) jinja: Option<crate::prompt::jinja::JinjaChatTemplate>,
    pub(crate) tokenizer: BPETokenizer,
    pub(crate) source: Arc<dyn TensorSource>,
    pub(crate) n_threads: usize,
    pub(crate) prefill_batch_size: usize,
    /// `--mmproj` path. Only read when an image was supplied.
    mmproj: Option<std::path::PathBuf>,
    /// `--image` path, when one was supplied.
    image: Option<std::path::PathBuf>,
    /// Rendered `(system, payload)` for the question being scored.
    /// `build_prompt` fills it, `forward_logits` consumes it.
    pending: Option<(String, String)>,
}

impl Qwen35JevScorer {
    pub(crate) fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
        mmproj_path: Option<&Path>,
        image_path: Option<&Path>,
        jinja: &crate::prompt::jinja::Options,
    ) -> Result<Self, String> {
        let jinja = jinja.resolve(&|k| source.metadata(k).cloned())?;
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        if image_path.is_some() && mmproj_path.is_none() {
            return Err("--jev --image requires --mmproj".into());
        }
        Ok(Self {
            jinja,
            tokenizer,
            source,
            n_threads,
            prefill_batch_size,
            mmproj: mmproj_path.map(Path::to_path_buf),
            image: image_path.map(Path::to_path_buf),
            pending: None,
        })
    }
}

/// What `build_prompt` decided to do with a turn.
enum PromptPlan {
    /// Render the prompt here; these are the token ids.
    Rendered(Vec<u32>),
    /// Defer to the multimodal forward, which owns the vision placeholders.
    /// `build_prompt` sets `pending` and returns no ids so `forward_logits`
    /// takes this branch.
    Multimodal,
}

/// Decide how to build the scored prompt.
///
/// The image check has to come first. `--jev --image --jinja` used to render
/// the text through the template and return those ids without setting
/// `pending`, so `forward_logits` never reached the multimodal path: the
/// question was scored while the image was silently ignored.
fn plan_prompt(
    has_image: bool,
    tokenizer: &dyn crate::core::tokenizer::Tokenizer,
    jinja: Option<&crate::prompt::jinja::JinjaChatTemplate>,
    system: &str,
    payload: &str,
) -> Result<PromptPlan, String> {
    if has_image {
        return Ok(PromptPlan::Multimodal);
    }
    // `--jinja` renders the model's own template. JEV opens the assistant turn
    // so the next token is the decision being scored, i.e.
    // `add_generation_prompt = true`. `thinking` is off so the scored position
    // does not move into a reasoning block.
    if let Some(template) = jinja {
        let ids = crate::prompt::jinja::render_text_conversation(
            tokenizer,
            template,
            Some(system),
            payload,
            false,
        )?;
        return Ok(PromptPlan::Rendered(ids));
    }
    Ok(PromptPlan::Multimodal)
}

impl JevScorer for Qwen35JevScorer {
    fn scorer_label(&self) -> &'static str {
        "Qwen3.5"
    }

    fn build_prompt(
        &mut self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let system = jev_system_prompt(q.mode);
        let payload = jev_payload_json(context, q)?;
        // With an image the prompt is not built here at all: `pending` is set
        // and the empty token list makes `forward_logits` route to the
        // multimodal logits helper, which owns the vision placeholders and
        // applies the template itself.
        match plan_prompt(
            self.image.is_some(),
            &self.tokenizer,
            self.jinja.as_ref(),
            system,
            &payload,
        )? {
            PromptPlan::Rendered(ids) => return Ok((labels, ids)),
            // With an image the prompt is not built here at all: `pending` is
            // set and the empty token list makes `forward_logits` route to the
            // multimodal logits helper, which owns the vision placeholders and
            // applies the template itself.
            PromptPlan::Multimodal => {
                if self.image.is_some() {
                    self.pending = Some((system.to_string(), payload));
                    return Ok((labels, Vec::new()));
                }
            }
        }
        let encoding = EncodeOptions {
            add_special: false,
            parse_special: true,
        };
        let mut token_ids = if self.image.is_some() {
            // Hand the multimodal helper the rendered system + payload so it
            // rebuilds the chat template and vision placeholders itself.
            // Tokenizing here too forked the template from /v1/jev/image and
            // produced a different (and wrong) answer for the same input.
            self.pending = Some((system.to_string(), payload));
            Vec::new()
        } else {
            let prompt_text = format!(
                "<|im_start|>system\n{system}<|im_end|>\n\
                 <|im_start|>user\n{payload}<|im_end|>\n\
                 <|im_start|>assistant\n"
            );
            self.tokenizer.encode(&prompt_text, encoding)
        };
        if let Some(bos) = self.tokenizer.bos_id() {
            token_ids.insert(0, bos);
        }
        Ok((labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        if let Some((system, payload)) = self.pending.take() {
            // Rendered prompts go through the shared multimodal helper so the
            // CLI and `/v1/jev/image` build an identical chat template and
            // vision injection. The hand-rolled template this used to build
            // answered the same image and question differently (and wrongly).
            let mmproj_path = self
                .mmproj
                .as_deref()
                .ok_or("internal error: visual scorer without an mmproj path")?;
            let image_path = self
                .image
                .as_deref()
                .ok_or("internal error: visual scorer without an image path")?;
            return crate::app::run_qwen35_family_multimodal_logits(
                self.source.as_ref(),
                mmproj_path,
                Some(image_path),
                None,
                None,
                &payload,
                self.n_threads,
                self.prefill_batch_size,
                8192,
                Some(&system),
                // The resolved template, so `--chat-template-file` survives.
                // Rebuilding `Options { file: None, .. }` here silently fell
                // back to the GGUF's own template.
                self.jinja.as_ref(),
            );
        }
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

#[cfg(test)]
mod tests {
    use super::{plan_prompt, PromptPlan};
    use crate::core::tokenizer::{MockTokenizer, Tokenizer};
    use crate::prompt::jinja::JinjaChatTemplate;

    fn marker_template() -> JinjaChatTemplate {
        JinjaChatTemplate::compile("MARK-{{ messages[0].content }}", "test").unwrap()
    }

    /// The regression: with an image *and* `--jinja`, `build_prompt` used to
    /// return the rendered text ids and leave `pending` unset, so
    /// `forward_logits` scored the text and the image was silently ignored.
    #[test]
    fn image_wins_over_the_jinja_text_path() {
        let tok = MockTokenizer::new();
        let plan = plan_prompt(true, &tok, Some(&marker_template()), "SYS", "PAY").unwrap();
        assert!(
            matches!(plan, PromptPlan::Multimodal),
            "an image must route to the multimodal forward even with --jinja"
        );
    }

    #[test]
    fn image_without_jinja_also_routes_to_multimodal() {
        let tok = MockTokenizer::new();
        let plan = plan_prompt(true, &tok, None, "SYS", "PAY").unwrap();
        assert!(matches!(plan, PromptPlan::Multimodal));
    }

    #[test]
    fn text_only_with_jinja_renders_the_template() {
        let tok = MockTokenizer::new();
        let plan = plan_prompt(false, &tok, Some(&marker_template()), "SYS", "PAY").unwrap();
        match plan {
            PromptPlan::Rendered(ids) => {
                assert_eq!(tok.decode(&ids, false), "MARK-SYS");
            }
            PromptPlan::Multimodal => panic!("text-only turn must render here"),
        }
    }
}
