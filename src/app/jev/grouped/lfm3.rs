use super::super::single::{lfm3::Lfm3JevScorer, JevScorer};
use super::super::types::{JevGroupedResult, PreparedGroupedQuestion};
use super::{
    allocate_group_labels, build_grouped_payload, build_grouped_system, run_jev_grouped_core,
    JevGroupedScorer,
};
use crate::core::{tensor::TensorSource, tokenizer::BPETokenizer};
use std::sync::Arc;

impl JevGroupedScorer for Lfm3JevScorer {
    fn scorer_label(&self) -> &'static str {
        "LFM3"
    }
    fn build_grouped_prompt(
        &self,
        context: &str,
        q: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String> {
        Ok((
            allocate_group_labels(q),
            self.render(build_grouped_system(), &build_grouped_payload(context, q)?)?,
        ))
    }
    fn forward_logits(&mut self, ids: Vec<u32>) -> Result<(Vec<f32>, std::time::Duration), String> {
        JevScorer::forward_logits(self, ids)
    }
    fn tokenizer(&self) -> &BPETokenizer {
        JevScorer::tokenizer(self)
    }
}

pub(crate) fn run_jev_grouped_lfm3(
    source: Arc<dyn TensorSource>,
    context: &str,
    questions: &[PreparedGroupedQuestion],
    threads: usize,
    batch: usize,
    output_json: bool,
    jinja: crate::prompt::jinja::Options,
) -> Result<Vec<JevGroupedResult>, String> {
    let mut scorer = Lfm3JevScorer::new(source.clone(), threads, batch, &jinja)?;
    run_jev_grouped_core(source, context, questions, output_json, &mut scorer)
}
