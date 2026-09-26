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
    image: Option<JevVision>,
}

struct JevVision {
    grid: VisionGrid,
    embeddings: Vec<f32>,
    image_token_id: u32,
    vision_start: u32,
    vision_end: u32,
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
        let image = if let Some(image_path) = image_path {
            let mmproj_path = mmproj_path.ok_or("--jev --image requires --mmproj")?;
            let mmproj =
                open_model_source(mmproj_path, ComponentRole::Mmproj).map_err(|error| {
                    format!("Failed to load mmproj {}: {error}", mmproj_path.display())
                })?;
            let (grid, embeddings) = encode_qwen35_image(mmproj.as_ref(), image_path, n_threads)?;
            let special = |name| {
                tokenizer
                    .special_token_id(name)
                    .ok_or_else(|| format!("Required vision token missing: {name}"))
            };
            Some(JevVision {
                grid,
                embeddings,
                image_token_id: special("image_pad")?,
                vision_start: special("vision_start")?,
                vision_end: special("vision_end")?,
            })
        } else {
            None
        };
        Ok(Self {
            tokenizer,
            source,
            n_threads,
            prefill_batch_size,
            image,
        })
    }

    fn forward_visual_logits(
        &self,
        token_ids: &[u32],
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        let image = self.image.as_ref().expect("visual scorer has image");
        let start = Instant::now();
        let mut model = Qwen35Model::from_source(self.source.as_ref())
            .map_err(|error| format!("Failed to parse Qwen3.5 model: {error}"))?;
        let max_context = model.config.n_ctx.min(8192);
        if token_ids.len() > max_context {
            return Err(format!(
                "Qwen3.5 visual JEV prompt has {} tokens, exceeds context {max_context}",
                token_ids.len()
            ));
        }
        let (positions, _) =
            build_qwen35_positions(token_ids, Some(image.image_token_id), &[image.grid])?;
        let tokens = token_ids
            .iter()
            .copied()
            .map(|token| i32::try_from(token).map_err(|_| format!("Token ID {token} exceeds i32")))
            .collect::<Result<Vec<_>, _>>()?;
        let embeddings = inject_vision_embeddings(
            &model,
            &tokens,
            Some(i32::try_from(image.image_token_id).map_err(|_| "Image token ID exceeds i32")?),
            &image.embeddings,
            image.grid.token_count(),
            model.config.n_embd,
        )?;
        let pool = Arc::new(ComputePool::new(self.n_threads));
        let capacity = (token_ids.len() + 1).min(max_context);
        let mut session = Qwen35Session::new_with_prefill_batch_size(
            &mut model,
            capacity,
            self.prefill_batch_size,
            pool,
        )?;
        let logits = session.step(&embeddings, token_ids.len(), &positions)?;
        Ok((logits, start.elapsed()))
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
        let encoding = EncodeOptions {
            add_special: false,
            parse_special: true,
        };
        let mut token_ids = if let Some(image) = &self.image {
            let mut ids = self.tokenizer.encode(
                &format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"),
                encoding,
            );
            ids.push(image.vision_start);
            ids.extend(std::iter::repeat_n(
                image.image_token_id,
                image.grid.token_count(),
            ));
            ids.push(image.vision_end);
            ids.extend(self.tokenizer.encode(
                &format!("{payload}<|im_end|>\n<|im_start|>assistant\n"),
                encoding,
            ));
            ids
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
        if self.image.is_some() {
            return self.forward_visual_logits(&token_ids);
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
