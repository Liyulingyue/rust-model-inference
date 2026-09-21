//! JEV grouped-mode scorer for hunyuan.

use super::super::single::hunyuan::HunyuanJevScorer;
use super::super::types::{JevGroupedQuestionInput, JevGroupedResult, PreparedGroupedQuestion};
use super::{JevGroupedScorer, allocate_group_labels, build_grouped_payload, build_grouped_system, run_jev_grouped_core};
use super::super::single::{JevScorer, verify_label_tokens_single};
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::prompt::{append_qwen_assistant_prefix, append_qwen_message_tokens, build_hunyuan_chat_prompt, HunyuanMessage};
use std::sync::Arc;
use std::time::{Duration, Instant};


pub(crate) fn run_jev_grouped_hunyuan(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevGroupedResult>, String> {
    let available_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = HunyuanJevGroupedScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Hunyuan)", n_threads);
    }
    run_jev_grouped_core(source, context, per_question, output_json, &mut scorer)
}

/// Hunyuan JEV grouped scorer — wraps [`hunyuan::HunyuanJevScorer`] (uses
/// Qwen3Model under the hood + Session API).
struct HunyuanJevGroupedScorer {
    inner: super::super::single::hunyuan::HunyuanJevScorer,
}

impl HunyuanJevGroupedScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: super::super::single::hunyuan::HunyuanJevScorer::new(source, n_threads, prefill_batch_size)?,
        })
    }
}

impl JevGroupedScorer for HunyuanJevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "Hunyuan"
    }

    fn build_grouped_prompt(
        &self,
        context: &str,
        q: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String> {
        let group_labels = allocate_group_labels(q);
        let system = build_grouped_system();
        let payload = build_grouped_payload(context, q)?;
        let token_ids = build_hunyuan_chat_prompt(
            self.inner.model.tokenizer(),
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
        Ok((group_labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        // super::super::single::hunyuan::HunyuanJevScorer::forward_logits runs the session for us.
        self.inner
            .forward_logits(token_ids)
            .map_err(|e| format!("Hunyuan grouped forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}

