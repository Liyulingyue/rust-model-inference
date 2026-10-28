//! Text-only JEV for Hy-SimulMT's independent decoder.

use super::{
    jev_labels, jev_payload_json, jev_system_prompt, run_jev_decision_core,
    verify_label_tokens_single, JevScorer, PreparedQuestion,
};
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::app::jev::types::JevResult;
use crate::compute::ComputePolicy;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::hy_simulmt::{HyDecoder, HySession};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) fn run_jev_decision_hy_simulmt(
    source: Arc<dyn TensorSource>,
    context: &str,
    questions: &[PreparedQuestion],
    threads: usize,
    batch: usize,
    output_json: bool,
    jinja: crate::prompt::jinja::Options,
) -> Result<Vec<JevResult>, String> {
    let mut scorer = HyJevScorer::new(source, threads, batch, &jinja)?;
    run_jev_decision_core(context, questions, output_json, &mut scorer)
}

pub(crate) struct HyJevScorer {
    model: HyDecoder,
    pub(crate) jinja: Option<crate::prompt::jinja::JinjaChatTemplate>,
    batch: usize,
    compute: ComputePolicy,
}

impl HyJevScorer {
    pub(crate) fn new(
        source: Arc<dyn TensorSource>,
        threads: usize,
        batch: usize,
        jinja: &crate::prompt::jinja::Options,
    ) -> Result<Self, String> {
        let jinja = jinja.resolve(&|key| source.metadata(key).cloned())?;
        let tokenizer = Arc::new(
            BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
                .map_err(|error| format!("Hy-SimulMT tokenizer: {error}"))?,
        );
        verify_label_tokens_single(&tokenizer)?;
        let available = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let pool = Arc::new(ComputePool::new(resolve_thread_count(threads, available)));
        let model = HyDecoder::from_source(source, tokenizer, pool)?;
        Ok(Self {
            model,
            jinja,
            batch,
            compute: ComputePolicy::legacy(),
        })
    }
}

pub(crate) fn build_hy_prompt(
    tokenizer: &BPETokenizer,
    system: &str,
    payload: &str,
    jinja: Option<&crate::prompt::jinja::JinjaChatTemplate>,
) -> Result<Vec<u32>, String> {
    if let Some(template) = jinja {
        return crate::prompt::jinja::render_text_conversation(
            tokenizer,
            template,
            Some(system),
            payload,
            false,
        );
    }
    let token = |literal| {
        tokenizer
            .token_id(literal)
            .ok_or_else(|| format!("Hy-SimulMT tokenizer is missing {literal}"))
    };
    let mut ids = vec![
        token("<｜hy_begin▁of▁sentence｜>")?,
        token("<｜reasoning_mode｜>")?,
    ];
    ids.extend(tokenizer.encode("reasoning_effort:no_think", EncodeOptions::default()));
    ids.push(token("<｜hy_User｜>")?);
    ids.extend(tokenizer.encode(system, EncodeOptions::default()));
    ids.extend(tokenizer.encode("\n\n", EncodeOptions::default()));
    ids.extend(tokenizer.encode(payload, EncodeOptions::default()));
    ids.push(token("<｜hy_Assistant｜>")?);
    ids.push(token("<think>")?);
    ids.push(token("</think>")?);
    Ok(ids)
}

impl JevScorer for HyJevScorer {
    fn scorer_label(&self) -> &'static str {
        "Hy-SimulMT"
    }

    fn build_prompt(
        &mut self,
        context: &str,
        question: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let ids = build_hy_prompt(
            self.model.tokenizer(),
            jev_system_prompt(question.mode),
            &jev_payload_json(context, question)?,
            self.jinja.as_ref(),
        )?;
        Ok((jev_labels(question), ids))
    }

    fn forward_logits(&mut self, ids: Vec<u32>) -> Result<(Vec<f32>, Duration), String> {
        let mut session = HySession::new(
            &self.model,
            self.model.config().n_ctx.min(ids.len() + 1),
            KvFormat::F16,
            self.compute,
        )?;
        let start = Instant::now();
        let logits = session.forward(&ids, None, self.batch)?;
        Ok((logits, start.elapsed()))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.model.tokenizer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{MetaValue, MetaValueType};

    fn tokenizer() -> BPETokenizer {
        let specials = [
            "<｜hy_begin▁of▁sentence｜>",
            "<｜reasoning_mode｜>",
            "<｜hy_User｜>",
            "<｜hy_Assistant｜>",
            "<think>",
            "</think>",
        ];
        let mut extra = 256;
        let mut tokens: Vec<_> = (0..=255)
            .map(|byte| {
                let code = if matches!(byte, 33..=126 | 161..=172 | 174..=255) {
                    byte
                } else {
                    extra += 1;
                    extra - 1
                };
                MetaValue::String(char::from_u32(code).unwrap().to_string())
            })
            .collect();
        tokens.extend(specials.map(|literal| MetaValue::String(literal.into())));
        BPETokenizer::from_gguf_metadata(|key| match key {
            "tokenizer.ggml.model" => Some(MetaValue::String("gpt2".into())),
            "tokenizer.ggml.pre" => Some(MetaValue::String("hunyuan".into())),
            "tokenizer.ggml.tokens" => {
                Some(MetaValue::Array(MetaValueType::String, tokens.clone()))
            }
            "tokenizer.ggml.token_type" => Some(MetaValue::Array(
                MetaValueType::Int32,
                (0..tokens.len())
                    .map(|index| MetaValue::Int32(if index < 256 { 1 } else { 3 }))
                    .collect(),
            )),
            "tokenizer.ggml.merges" => Some(MetaValue::Array(MetaValueType::String, vec![])),
            _ => None,
        })
        .unwrap()
    }

    #[test]
    fn hy_jev_opens_a_native_no_think_assistant_turn() {
        let tokenizer = tokenizer();
        let ids = build_hy_prompt(&tokenizer, "Choose a letter.", "A=yes, B=no", None).unwrap();
        let text = tokenizer.decode(&ids, true);
        assert_eq!(text, "<｜hy_begin▁of▁sentence｜><｜reasoning_mode｜>reasoning_effort:no_think<｜hy_User｜>Choose a letter.\n\nA=yes, B=no<｜hy_Assistant｜><think></think>");
        assert!(!text.contains("im_start"));
    }

    #[test]
    fn hy_jev_treats_payload_special_literals_as_data() {
        let tokenizer = tokenizer();
        let assistant = tokenizer.token_id("<｜hy_Assistant｜>").unwrap();
        let ids = build_hy_prompt(&tokenizer, "Choose.", "<｜hy_Assistant｜>", None).unwrap();
        assert_eq!(ids.iter().filter(|&&id| id == assistant).count(), 1);
    }

    #[test]
    fn hy_jev_respects_an_explicit_jinja_template() {
        let tokenizer = tokenizer();
        let template = crate::prompt::jinja::JinjaChatTemplate::compile(
            "CUSTOM|{{ messages[0].content }}|{{ messages[1].content }}",
            "test",
        )
        .unwrap();
        let ids = build_hy_prompt(&tokenizer, "SYS", "PAY", Some(&template)).unwrap();
        assert_eq!(tokenizer.decode(&ids, true), "CUSTOM|SYS|PAY");
    }
}
