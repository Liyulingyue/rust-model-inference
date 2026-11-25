use super::grouped::{
    allocate_group_labels, build_grouped_payload, build_grouped_system, JevGroupedScorer,
};
use super::single::{
    jev_labels, jev_payload_json, jev_system_prompt, verify_label_tokens_single, JevScorer,
    PreparedQuestion,
};
use super::types::PreparedGroupedQuestion;
use crate::app::text::RuntimeOptions;
use crate::compute::ComputePolicy;
use crate::core::scratchpad::KvFormat;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::models::aion::{prompt::load_template, runtime::AionTextRuntime};
use crate::prompt::jinja::{render_text_conversation, JinjaChatTemplate, Options};
use std::sync::Arc;
use std::time::Duration;

pub(crate) struct AionScorer {
    runtime: AionTextRuntime,
    template: JinjaChatTemplate,
}

impl AionScorer {
    pub(crate) fn new(
        source: Arc<dyn TensorSource>,
        threads: usize,
        batch: usize,
        max_context: usize,
        jinja: &Options,
    ) -> Result<Self, String> {
        let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
            source.metadata(key).cloned()
        })?);
        verify_label_tokens_single(&tokenizer)?;
        let template = load_template(source.as_ref(), jinja)?;
        let threads = crate::app::cli::resolve_thread_count(
            threads,
            std::thread::available_parallelism().map_or(1, usize::from),
        );
        let options =
            RuntimeOptions::from_model(source, Arc::new(ComputePool::new(threads)), tokenizer)
                .with_prefill_batch_size(batch)
                .with_max_context(max_context)
                .with_kv_format(KvFormat::F32)
                .with_compute(ComputePolicy::legacy());
        Ok(Self {
            runtime: AionTextRuntime::new(options)?,
            template,
        })
    }

    fn render(&self, system: &str, user: &str) -> Result<Vec<u32>, String> {
        render_text_conversation(
            self.runtime.tokenizer(),
            &self.template,
            Some(system),
            user,
            None,
        )
    }
}

impl JevScorer for AionScorer {
    fn scorer_label(&self) -> &'static str {
        "Aion"
    }

    fn build_prompt(
        &mut self,
        context: &str,
        question: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        Ok((
            jev_labels(question),
            self.render(
                jev_system_prompt(question.mode),
                &jev_payload_json(context, question)?,
            )?,
        ))
    }

    fn forward_logits(&mut self, ids: Vec<u32>) -> Result<(Vec<f32>, Duration), String> {
        self.runtime.forward_logits(&ids)
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.runtime.tokenizer()
    }
}

impl JevGroupedScorer for AionScorer {
    fn scorer_label(&self) -> &'static str {
        "Aion"
    }

    fn build_grouped_prompt(
        &self,
        context: &str,
        question: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String> {
        Ok((
            allocate_group_labels(question),
            self.render(
                build_grouped_system(),
                &build_grouped_payload(context, question)?,
            )?,
        ))
    }

    fn forward_logits(&mut self, ids: Vec<u32>) -> Result<(Vec<f32>, Duration), String> {
        self.runtime.forward_logits(&ids)
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.runtime.tokenizer()
    }
}
