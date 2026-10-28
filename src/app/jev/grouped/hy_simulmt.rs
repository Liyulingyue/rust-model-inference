use super::super::single::hy_simulmt::{build_hy_prompt, HyJevScorer};
use super::super::single::JevScorer;
use super::super::types::{JevGroupedResult, PreparedGroupedQuestion};
use super::{
    allocate_group_labels, build_grouped_payload, build_grouped_system, run_jev_grouped_core,
    JevGroupedScorer,
};
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::BPETokenizer;
use std::sync::Arc;

pub(crate) fn run_jev_grouped_hy_simulmt(
    source: Arc<dyn TensorSource>,
    context: &str,
    questions: &[PreparedGroupedQuestion],
    threads: usize,
    batch: usize,
    output_json: bool,
    jinja: crate::prompt::jinja::Options,
) -> Result<Vec<JevGroupedResult>, String> {
    let mut scorer = HyGroupedScorer(HyJevScorer::new(source.clone(), threads, batch, &jinja)?);
    run_jev_grouped_core(source, context, questions, output_json, &mut scorer)
}

struct HyGroupedScorer(HyJevScorer);

impl JevGroupedScorer for HyGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "Hy-SimulMT"
    }

    fn build_grouped_prompt(
        &self,
        context: &str,
        question: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String> {
        let ids = build_hy_prompt(
            self.0.tokenizer(),
            build_grouped_system(),
            &build_grouped_payload(context, question)?,
            self.0.jinja.as_ref(),
        )?;
        Ok((allocate_group_labels(question), ids))
    }

    fn forward_logits(&mut self, ids: Vec<u32>) -> Result<(Vec<f32>, std::time::Duration), String> {
        self.0.forward_logits(ids)
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.0.tokenizer()
    }
}
