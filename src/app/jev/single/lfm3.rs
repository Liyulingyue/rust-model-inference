//! LFM3 decision prompts use the GGUF template with thinking disabled.
use super::super::types::JevResult;
use super::{
    jev_labels, jev_payload_json, jev_system_prompt, run_jev_decision_core,
    verify_label_tokens_single, JevScorer, PreparedQuestion,
};
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::BPETokenizer;
use crate::prompt::jinja::{JinjaChatTemplate, Options};
use std::sync::Arc;

pub(crate) struct Lfm3JevScorer {
    source: Arc<dyn TensorSource>,
    tokenizer: BPETokenizer,
    template: JinjaChatTemplate,
    threads: usize,
    batch: usize,
}

impl Lfm3JevScorer {
    pub(crate) fn new(
        source: Arc<dyn TensorSource>,
        threads: usize,
        batch: usize,
        jinja: &Options,
    ) -> Result<Self, String> {
        crate::core::prefill::checked_prefill_batch_size(Some(batch))?;
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())?;
        verify_label_tokens_single(&tokenizer)?;
        let template = Options {
            jinja: true,
            file: jinja.file.clone(),
        }
        .resolve(&|k| source.metadata(k).cloned())?
        .ok_or("LFM3 requires a chat template")?;
        Ok(Self {
            source,
            tokenizer,
            template,
            threads,
            batch,
        })
    }

    pub(crate) fn render(&self, system: &str, payload: &str) -> Result<Vec<u32>, String> {
        crate::prompt::jinja::render_text_conversation(
            &self.tokenizer,
            &self.template,
            Some(system),
            payload,
            false,
        )
    }
}

impl JevScorer for Lfm3JevScorer {
    fn scorer_label(&self) -> &'static str {
        "LFM3"
    }
    fn build_prompt(
        &mut self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        Ok((
            jev_labels(q),
            self.render(jev_system_prompt(q.mode), &jev_payload_json(context, q)?)?,
        ))
    }
    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::lfm3::run_forward_logits_lfm3_with_batch(
            self.source.as_ref(),
            &token_ids,
            self.threads,
            crate::core::scratchpad::KvFormat::F32,
            8192,
            self.batch,
        )
    }
    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}

pub(crate) fn run_jev_decision_lfm3(
    source: Arc<dyn TensorSource>,
    context: &str,
    questions: &[PreparedQuestion],
    threads: usize,
    batch: usize,
    output_json: bool,
    jinja: Options,
) -> Result<Vec<JevResult>, String> {
    let mut scorer = Lfm3JevScorer::new(source, threads, batch, &jinja)?;
    run_jev_decision_core(context, questions, output_json, &mut scorer)
}
