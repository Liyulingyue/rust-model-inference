//! JEV single-mode scorer for lfm2moe.

use super::super::types::{JevMode, JevQuestionInput, JevResult};
use super::lfm2::Lfm2JevScorer;
use super::run_jev_decision_core;
use super::JevScorer;
use super::PreparedQuestion;
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::lfm2::trunk::forward::run_forward_logits_lfm2_with_batch;
use crate::models::lfm2moe::trunk::forward::run_forward_logits_lfm2moe_with_batch;
use crate::prompt::append_qwen_message_tokens;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) fn run_jev_decision_lfm2moe(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevResult>, String> {
    // LFM2-MoE chat format is identical to LFM2 / LFM2.5 — only the
    // forward module differs (`crate::models::lfm2moe`).
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = Lfm2MoeJevScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (LFM2-MoE)", n_threads);
    }
    run_jev_decision_core(source, context, per_question, output_json, &mut scorer)
}

/// LFM2-MoE JEV scorer — wraps the LFM2 chat template via the
/// `Lfm2JevScorer` payload builder, routes the forward through
/// `crate::models::lfm2moe::run_forward_logits_lfm2moe_with_batch`.
struct Lfm2MoeJevScorer {
    pub(crate) inner: Lfm2JevScorer,
}

impl Lfm2MoeJevScorer {
    pub(crate) fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: Lfm2JevScorer::new(source, n_threads, prefill_batch_size)?,
        })
    }
}

impl JevScorer for Lfm2MoeJevScorer {
    fn scorer_label(&self) -> &'static str {
        "LFM2-MoE"
    }

    fn build_prompt(
        &self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        self.inner.build_prompt(context, q)
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::lfm2moe::run_forward_logits_lfm2moe_with_batch(
            self.inner.source.as_ref(),
            &token_ids,
            self.inner.n_threads,
            KvFormat::F16,
            8192,
            self.inner.prefill_batch_size,
        )
        .map_err(|e| format!("LFM2-MoE forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}
