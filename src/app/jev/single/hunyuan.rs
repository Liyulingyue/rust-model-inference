//! JEV single-mode scorer for hunyuan.

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
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::prompt::{build_hunyuan_chat_prompt, HunyuanMessage};
use std::sync::Arc;

pub(crate) fn run_jev_decision_hunyuan(
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
    let mut scorer = HunyuanJevScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Hunyuan)", n_threads);
    }
    run_jev_decision_core(source, context, per_question, output_json, &mut scorer)
}

/// Hunyuan JEV scorer — wraps a `Qwen3Model` (Hunyuan reuses
/// Qwen3's trunk under the hood). The chat template is
/// `build_hunyuan_chat_prompt`; the forward rebuilds a fresh
/// `Qwen3Session` per question for ephemeral KV.
pub(crate) struct HunyuanJevScorer {
    pub(crate) model: crate::models::qwen3::Qwen3Model,
    pub(crate) max_ctx: usize,
    pub(crate) prefill_batch_size: usize,
}

impl HunyuanJevScorer {
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
}

impl JevScorer for HunyuanJevScorer {
    fn scorer_label(&self) -> &'static str {
        "Hunyuan"
    }

    fn build_prompt(
        &self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let system = jev_system_prompt(q.mode);
        let payload = jev_payload_json(context, q)?;
        let token_ids = build_hunyuan_chat_prompt(
            self.model.tokenizer(),
            &[
                HunyuanMessage {
                    role: "system",
                    content: system,
                },
                HunyuanMessage {
                    role: "user",
                    content: &payload,
                },
            ],
            true,
        )?;
        Ok((labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        let mut session = crate::models::qwen3::Qwen3Session::new_with_kv_state(
            &self.model,
            self.max_ctx.min(token_ids.len() + 1),
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
            .map_err(|e| format!("Hunyuan forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.model.tokenizer()
    }
}
