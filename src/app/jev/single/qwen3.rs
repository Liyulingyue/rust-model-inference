//! JEV single-mode scorer for qwen3.

use super::super::types::JevResult;
use super::build_jev_prompt;
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
use crate::core::tokenizer::BPETokenizer;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_jev_decision_qwen3(
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
    // Image questions go through the shared multimodal logits helper rather than
    // `Qwen3JevScorer`: it owns the two projector families (qwen3vl merger and
    // Qwen2.5-Omni) plus their deepstack wiring, and is the same code the
    // `/v1/jev/image` HTTP endpoint exercises.
    if image_path.is_some() {
        let mmproj_path = mmproj_path.ok_or("--jev --image requires --mmproj")?;
        return run_jev_decision_qwen3_visual(
            source,
            context,
            per_question,
            n_threads,
            prefill_batch_size,
            output_json,
            mmproj_path,
            image_path.expect("image_path is Some"),
        );
    }
    let mut scorer = Qwen3JevScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads", scorer.pool().n_threads());
    }
    run_jev_decision_core(context, per_question, output_json, &mut scorer)
}

/// Score each JEV question from one multimodal forward pass.
#[allow(clippy::too_many_arguments)]
fn run_jev_decision_qwen3_visual(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedQuestion],
    n_threads: usize,
    prefill_batch_size: usize,
    output_json: bool,
    mmproj_path: &Path,
    image_path: &Path,
) -> Result<Vec<JevResult>, String> {
    use super::run_jev_decision_core;
    use super::JevScorer;

    let mut scorer = Qwen3VisualJevScorer {
        source,
        pending: None,
        mmproj_path: mmproj_path.to_path_buf(),
        image_path: image_path.to_path_buf(),
        n_threads,
        prefill_batch_size,
    };
    if !output_json {
        eprintln!("compute pool: {} threads (Qwen3 visual)", n_threads);
    }
    run_jev_decision_core(context, per_question, output_json, &mut scorer)
}

/// JEV scorer backed by the shared qwen3 multimodal logits helper, so the two
/// projector families and their deepstack wiring stay in one place.
struct Qwen3VisualJevScorer {
    /// Rendered `(system, payload)` for the question being scored. `build_prompt`
    /// fills it and `forward_logits` consumes it; `run_jev_decision_core` calls
    /// the two back to back, so a single slot is enough.
    pending: Option<(String, String)>,
    mmproj_path: PathBuf,
    image_path: PathBuf,
    source: Arc<dyn TensorSource>,
    n_threads: usize,
    prefill_batch_size: usize,
}

impl JevScorer for Qwen3VisualJevScorer {
    fn scorer_label(&self) -> &'static str {
        "Qwen3"
    }

    fn build_prompt(
        &mut self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let system = jev_system_prompt(q.mode);
        let payload = jev_payload_json(context, q)?;
        self.pending = Some((system.to_string(), payload));
        // The token ids are rebuilt by the multimodal helper, which owns the
        // qwen3vl chat template, so nothing needs to be returned here.
        Ok((labels, Vec::new()))
    }

    fn forward_logits(
        &mut self,
        _token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        let (system, payload) = self
            .pending
            .take()
            .ok_or("visual JEV forward without a rendered prompt")?;
        crate::app::run_qwen3_family_multimodal_logits(
            self.source.as_ref(),
            self.source.clone(),
            &self.mmproj_path,
            Some(&self.image_path),
            None,
            None,
            &payload,
            self.n_threads,
            self.prefill_batch_size,
            Some(&system),
        )
    }

    fn tokenizer(&self) -> &BPETokenizer {
        unreachable!("the visual scorer never tokenizes the prompt itself")
    }
}

/// Qwen3 JEV scorer — owns the `Qwen3Model` (which embeds the
/// tokenizer + compute pool). Each `forward_logits` rebuilds a
/// fresh `Qwen3Session` so the KV cache stays ephemeral.
pub(crate) struct Qwen3JevScorer {
    pub(crate) model: crate::models::qwen3::Qwen3Model,
    pub(crate) max_ctx: usize,
    pub(crate) prefill_batch_size: usize,
}

impl Qwen3JevScorer {
    pub(crate) fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        let pool = Arc::new(ComputePool::new(n_threads));
        let model = crate::models::qwen3::Qwen3Model::from_source(
            source.clone(),
            Arc::new(tokenizer),
            pool,
        )?;
        let max_ctx = model.config().n_ctx;
        Ok(Self {
            model,
            max_ctx,
            prefill_batch_size,
        })
    }

    pub(crate) fn pool(&self) -> Arc<ComputePool> {
        self.model.pool()
    }
}

impl JevScorer for Qwen3JevScorer {
    fn scorer_label(&self) -> &'static str {
        "Qwen3"
    }

    fn build_prompt(
        &mut self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let (token_ids, _payload) = build_jev_prompt(self.model.tokenizer(), context, q, false)?;
        Ok((labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        let mut session = crate::models::qwen3::Qwen3Session::new_with_kv_state(
            &self.model,
            self.max_ctx,
            KvFormat::F16,
            crate::core::scratchpad::KvLifecycle::Ephemeral,
        )?;
        let positions: Vec<[usize; 4]> = (0..token_ids.len()).map(|i| [i, 0, 0, 0]).collect();
        let input = crate::models::qwen3::Qwen3Input {
            token_ids: &token_ids,
            positions: &positions,
            embeddings: None,
            deepstack_embeddings: None,
        };
        session
            .forward_logits(input, self.prefill_batch_size)
            .map_err(|e| format!("Qwen3 forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.model.tokenizer()
    }
}
