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
    ) -> Result<Self, String> {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        if image_path.is_some() && mmproj_path.is_none() {
            return Err("--jev --image requires --mmproj".into());
        }
        Ok(Self {
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
