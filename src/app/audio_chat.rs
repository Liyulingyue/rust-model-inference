//! Audio conversation entry points share the model's prepared-input and decoder path.

use crate::core::tensor::TensorSource;
use crate::core::tokenizer::BPETokenizer;
use crate::ops::generation_runtime::{GenerationRequest, StdoutSink};
use std::sync::Arc;

fn media_prompt(
    tokenizer: &BPETokenizer,
    template: &crate::prompt::jinja::JinjaChatTemplate,
    audios: usize,
    system: &str,
    payload: &str,
) -> Result<Vec<u32>, String> {
    crate::prompt::jinja::render_text_conversation(
        tokenizer,
        template,
        Some(system),
        &crate::models::sophon_realtime::audio::prompt_text(payload, audios),
        false,
    )
}

fn score_prompt(
    tokenizer: &BPETokenizer,
    template: &crate::prompt::jinja::JinjaChatTemplate,
    audios: usize,
    context: &str,
    question: &super::jev::single::PreparedQuestion,
) -> Result<Vec<u32>, String> {
    media_prompt(
        tokenizer,
        template,
        audios,
        super::jev::jev_system_prompt(question.mode),
        &super::jev::jev_payload_json(context, question)?,
    )
}

fn grouped_prompt(
    tokenizer: &BPETokenizer,
    template: &crate::prompt::jinja::JinjaChatTemplate,
    audios: usize,
    context: &str,
    question: &super::jev::types::PreparedGroupedQuestion,
) -> Result<Vec<u32>, String> {
    media_prompt(
        tokenizer,
        template,
        audios,
        super::jev::build_grouped_system(),
        &super::jev::build_grouped_payload(context, question)?,
    )
}

fn validate_media_request(
    runtime: &dyn crate::ops::generation_runtime::TextRuntime,
    ids: Vec<u32>,
    audio: &[Vec<f32>],
    images: &[Vec<u8>],
) -> Result<(), String> {
    if audio.is_empty() {
        runtime.validate_request(&ids, images, 1)
    } else {
        let mut request = GenerationRequest::new(ids, 1);
        request.images = images.to_vec();
        runtime.validate_audio_request(&request, audio)
    }
}

struct MediaScorer<'a> {
    runtime: &'a mut dyn crate::ops::generation_runtime::TextRuntime,
    tokenizer: &'a BPETokenizer,
    template: &'a crate::prompt::jinja::JinjaChatTemplate,
    audio: &'a [Vec<f32>],
    images: &'a [Vec<u8>],
}

impl MediaScorer<'_> {
    fn forward_logits(&mut self, ids: Vec<u32>) -> Result<(Vec<f32>, std::time::Duration), String> {
        let start = std::time::Instant::now();
        let logits = if self.audio.is_empty() {
            let mut request = GenerationRequest::new(ids, 1);
            request.images = self.images.to_vec();
            self.runtime.logits(&request)?
        } else {
            self.runtime.audio_logits(&ids, self.audio)?
        };
        if logits.len() < self.tokenizer.vocab_size() || logits.iter().any(|v| !v.is_finite()) {
            return Err("candidate scorer returned invalid vocabulary logits".into());
        }
        Ok((logits, start.elapsed()))
    }
}

impl super::jev::single::JevScorer for MediaScorer<'_> {
    fn scorer_label(&self) -> &'static str {
        "media"
    }
    fn tokenizer(&self) -> &BPETokenizer {
        self.tokenizer
    }
    fn build_prompt(
        &mut self,
        context: &str,
        q: &super::jev::single::PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let ids = score_prompt(self.tokenizer, self.template, self.audio.len(), context, q)?;
        validate_media_request(self.runtime, ids.clone(), self.audio, self.images)?;
        Ok((super::jev::single::jev_labels(q), ids))
    }
    fn forward_logits(&mut self, ids: Vec<u32>) -> Result<(Vec<f32>, std::time::Duration), String> {
        self.forward_logits(ids)
    }
}

impl super::jev::grouped::JevGroupedScorer for MediaScorer<'_> {
    fn scorer_label(&self) -> &'static str {
        "media"
    }
    fn tokenizer(&self) -> &BPETokenizer {
        self.tokenizer
    }
    fn build_grouped_prompt(
        &self,
        context: &str,
        q: &super::jev::types::PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String> {
        let ids = grouped_prompt(self.tokenizer, self.template, self.audio.len(), context, q)?;
        validate_media_request(self.runtime, ids.clone(), self.audio, self.images)?;
        Ok((super::jev::grouped::allocate_group_labels(q), ids))
    }
    fn forward_logits(&mut self, ids: Vec<u32>) -> Result<(Vec<f32>, std::time::Duration), String> {
        self.forward_logits(ids)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_score(
    runtime: &dyn crate::ops::generation_runtime::TextRuntime,
    tokenizer: &BPETokenizer,
    template: &crate::prompt::jinja::JinjaChatTemplate,
    audio: &[Vec<f32>],
    images: &[Vec<u8>],
    context: &str,
    questions: &[super::jev::JevQuestionInput],
    positive: Option<&str>,
) -> Result<(), String> {
    super::jev::single::verify_label_tokens_single(tokenizer)?;
    for question in super::jev::prepare_jev_questions(questions, positive)? {
        let ids = score_prompt(tokenizer, template, audio.len(), context, &question)?;
        validate_media_request(runtime, ids, audio, images)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn score(
    runtime: &mut dyn crate::ops::generation_runtime::TextRuntime,
    tokenizer: &BPETokenizer,
    template: &crate::prompt::jinja::JinjaChatTemplate,
    audio: &[Vec<f32>],
    images: &[Vec<u8>],
    context: &str,
    questions: &[super::jev::JevQuestionInput],
    positive: Option<&str>,
) -> Result<Vec<super::jev::JevResult>, String> {
    super::jev::single::verify_label_tokens_single(tokenizer)?;
    let prepared = super::jev::prepare_jev_questions(questions, positive)?;
    super::jev::single::run_jev_decision_core(
        context,
        &prepared,
        true,
        &mut MediaScorer {
            runtime,
            tokenizer,
            template,
            audio,
            images,
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_grouped_score(
    runtime: &dyn crate::ops::generation_runtime::TextRuntime,
    tokenizer: &BPETokenizer,
    template: &crate::prompt::jinja::JinjaChatTemplate,
    audio: &[Vec<f32>],
    images: &[Vec<u8>],
    context: &str,
    questions: &[super::jev::JevGroupedQuestionInput],
    mode: super::jev::JevMode,
) -> Result<(), String> {
    super::jev::single::verify_label_tokens_single(tokenizer)?;
    for question in super::jev::prepare_jev_grouped_questions(questions, mode)? {
        let ids = grouped_prompt(tokenizer, template, audio.len(), context, &question)?;
        validate_media_request(runtime, ids, audio, images)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn score_grouped(
    source: Arc<dyn TensorSource>,
    runtime: &mut dyn crate::ops::generation_runtime::TextRuntime,
    tokenizer: &BPETokenizer,
    template: &crate::prompt::jinja::JinjaChatTemplate,
    audio: &[Vec<f32>],
    images: &[Vec<u8>],
    context: &str,
    questions: &[super::jev::JevGroupedQuestionInput],
    mode: super::jev::JevMode,
) -> Result<Vec<super::jev::types::JevGroupedResult>, String> {
    super::jev::single::verify_label_tokens_single(tokenizer)?;
    let prepared = super::jev::prepare_jev_grouped_questions(questions, mode)?;
    super::jev::grouped::run_jev_grouped_core(
        source,
        context,
        &prepared,
        true,
        &mut MediaScorer {
            runtime,
            tokenizer,
            template,
            audio,
            images,
        },
    )
}

pub(crate) fn request(
    source: &dyn TensorSource,
    tokenizer: &BPETokenizer,
    template: &crate::prompt::jinja::Options,
    prompt: &str,
    audios: usize,
    max_tokens: usize,
    temperature: f32,
) -> Result<GenerationRequest, String> {
    if max_tokens == 0 || !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err("max_tokens must be positive and temperature must be finite in [0,2]".into());
    }
    let user = crate::models::sophon_realtime::audio::prompt_text(prompt, audios);
    let ids = crate::prompt::jinja::single_turn_tokens(
        tokenizer,
        template,
        &|key| source.metadata(key).cloned(),
        &user,
        false,
    )?
    .ok_or("audio conversation requires a chat template")?;
    let mut request = GenerationRequest::new(ids, max_tokens);
    request.sampling.temperature = temperature;
    Ok(request)
}

pub fn run_cli(
    source: Arc<dyn TensorSource>,
    options: &super::CliOptions,
    threads: usize,
) -> Result<(), String> {
    if options.video.is_some() || (options.image.is_some() && options.audio.is_some()) {
        return Err("combined audio/image/video generation is not wired yet".into());
    }
    let audio = options
        .audio
        .as_deref()
        .map(super::media::decode_audio)
        .transpose()?;
    let mmproj = options
        .mmproj
        .as_deref()
        .map(|component| {
            crate::format::ggufrs::open_model_source(
                component,
                crate::format::ggufrs::ComponentRole::Mmproj,
            )
            .map(Arc::from)
            .map_err(|e| e.to_string())
        })
        .transpose()?;
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
        source.metadata(key).cloned()
    })?);
    let template = crate::prompt::jinja::Options {
        jinja: true,
        file: options.chat_template_file.clone(),
    };
    let (max_tokens, temperature) = super::resolve_cli_generation_options(options);
    let mut request = request(
        source.as_ref(),
        &tokenizer,
        &template,
        options.prompt.as_deref().unwrap_or_default(),
        usize::from(audio.is_some()),
        max_tokens,
        temperature,
    )?;
    request.sampling.repetition_penalty = options.effective_repetition_penalty();
    if let Some(path) = &options.image {
        request
            .images
            .push(std::fs::read(path).map_err(|e| format!("image: {e}"))?);
    }
    let mut speech = if let Some(path) = options.tts_model.as_deref() {
        super::tts::validate_tag_speech_options(options)?;
        let codec = options
            .tts_mmproj
            .as_deref()
            .ok_or("--tts-model requires --tts-mmproj")?;
        let open = |path, role| {
            crate::format::ggufrs::open_model_source(path, role)
                .map(Arc::from)
                .map_err(|e| e.to_string())
        };
        Some(crate::models::sophon_realtime::speech::SpeechEngine::new(
            open(path, crate::format::ggufrs::ComponentRole::Llm)?,
            open(codec, crate::format::ggufrs::ComponentRole::Mmproj)?,
            threads,
            options.compute_policy(),
        )?)
    } else {
        None
    };
    if speech.is_some() && options.out.is_none() {
        return Err("--tts-model requires --out <wav>".into());
    }
    let pool = Arc::new(crate::core::thread_pool::ComputePool::new(threads));
    let runtime_options =
        super::text::RuntimeOptions::from_model(source.clone(), pool, tokenizer.clone())
            .with_compute(options.compute_policy())
            .with_threads(threads)
            .with_kv_format(options.kv_format)
            .with_prefill_batch_size(options.effective_prefill_batch_size()?)
            .with_max_context(options.effective_max_context())
            .with_mmproj(mmproj);
    let mut runtime =
        super::text::build_text_runtime(&super::arch_of(source.as_ref()), runtime_options)?;
    if options.jev {
        let template = template
            .resolve(&|key| source.metadata(key).cloned())?
            .ok_or("JEV requires a chat template")?;
        let audio = audio.as_ref().map(std::slice::from_ref).unwrap_or(&[]);
        match super::jev::build_jev_inputs(options)?.ok_or("missing JEV inputs")? {
            super::jev::JevInputs::Single {
                context,
                questions,
                positive,
            } => {
                let results = score(
                    runtime.as_mut(),
                    &tokenizer,
                    &template,
                    audio,
                    &request.images,
                    &context,
                    &questions,
                    positive.as_deref(),
                )?;
                for result in results {
                    if options.jev_output_json {
                        println!(
                            "{}",
                            serde_json::to_string(&result).map_err(|e| e.to_string())?
                        );
                    } else {
                        println!(
                            "{}: {} (confidence {:.6})",
                            result.question,
                            result.choice_label.unwrap_or('?'),
                            result.confidence
                        );
                    }
                }
            }
            super::jev::JevInputs::Grouped {
                context,
                questions,
                mode,
            } => {
                let results = score_grouped(
                    source.clone(),
                    runtime.as_mut(),
                    &tokenizer,
                    &template,
                    audio,
                    &request.images,
                    &context,
                    &questions,
                    mode,
                )?;
                for result in results {
                    if options.jev_output_json {
                        println!(
                            "{}",
                            serde_json::to_string(&result).map_err(|e| e.to_string())?
                        );
                    } else {
                        super::jev::grouped::print_grouped_result_text(&result);
                    }
                }
            }
        }
        return Ok(());
    }
    let generated = if let Some(audio) = audio {
        runtime.generate_audio(&request, &[audio], &mut StdoutSink)?
    } else {
        runtime.generate(&request, &mut StdoutSink)?
    };
    println!();
    if let Some(engine) = &mut speech {
        let result = engine.synthesize(
            &crate::models::sophon_realtime::speech::SpeechRequest {
                text: &generated.text,
                voice: options.voice.as_deref().unwrap_or("brand_zh"),
                max_tokens: max_tokens.saturating_mul(4).clamp(128, 1250),
                temperature,
                seed: options.seed.unwrap_or(0) as u64,
                stream: false,
            },
            |_| true,
        )?;
        let wav = crate::format::wav::encode_wav_pcm16_channels(&result.waveform, 24_000, 1)?;
        super::write_output_atomically(options.out.as_deref().unwrap(), &wav, false)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::generation_runtime::{GeneratedText, TextRuntime, TokenSink};

    #[test]
    fn grouped_audio_scoring_preserves_media_and_normalizes_each_group() {
        struct Source;
        impl TensorSource for Source {
            fn metadata(&self, _: &str) -> Option<&crate::MetaValue> {
                None
            }
            fn tensor_info(&self, _: &str) -> Option<&crate::core::tensor::TensorInfo> {
                None
            }
            fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
                None
            }
        }
        struct Audio<'a>(&'a BPETokenizer);
        impl TextRuntime for Audio<'_> {
            fn arch(&self) -> &str {
                "audio-test"
            }
            fn context_length(&self) -> usize {
                1024
            }
            fn generate(
                &mut self,
                _: &GenerationRequest,
                _: &mut dyn TokenSink,
            ) -> Result<GeneratedText, String> {
                unreachable!()
            }
            fn validate_audio_request(
                &self,
                request: &GenerationRequest,
                audio: &[Vec<f32>],
            ) -> Result<(), String> {
                assert_eq!(request.max_new_tokens, 1);
                assert!(request.images.is_empty());
                assert_eq!(audio, [vec![0.25, -0.25]]);
                let prompt = String::from_utf8(
                    request
                        .token_ids
                        .iter()
                        .flat_map(|&id| self.0.token_piece_bytes(id, true))
                        .collect(),
                )
                .unwrap();
                assert!(prompt.starts_with("PINNED|For each group, select the best option."));
                assert!(prompt.contains("<|audio|>\n{\"context\": \"ctx\""));
                Ok(())
            }
            fn audio_logits(&mut self, _: &[u32], audio: &[Vec<f32>]) -> Result<Vec<f32>, String> {
                assert_eq!(audio, [vec![0.25, -0.25]]);
                let mut logits = vec![0.0; self.0.vocab_size()];
                for (label, value) in [("A", 1.0), ("B", 2.0), ("C", 3.0), ("D", 1.0)] {
                    logits[self.0.token_id(label).unwrap() as usize] = value;
                }
                Ok(logits)
            }
        }
        let tokenizer = BPETokenizer::from_qwen3_embedded_merges().unwrap();
        let template = crate::prompt::jinja::JinjaChatTemplate::compile(
            "PINNED|{{ messages[0].content }}|{{ messages[1].content }}",
            "grouped-audio-test",
        )
        .unwrap();
        let questions = [super::super::jev::JevGroupedQuestionInput {
            text: "q".into(),
            groups: vec![
                super::super::jev::JevGroupInput {
                    label: "first".into(),
                    options: vec!["no".into(), "yes".into()],
                },
                super::super::jev::JevGroupInput {
                    label: "second".into(),
                    options: vec!["chosen".into(), "other".into()],
                },
            ],
        }];
        let result = score_grouped(
            Arc::new(Source),
            &mut Audio(&tokenizer),
            &tokenizer,
            &template,
            &[vec![0.25, -0.25]],
            &[],
            "ctx",
            &questions,
            super::super::jev::JevMode::MultiSelect,
        )
        .unwrap();
        assert_eq!(result[0].groups[0].choice_label, 'B');
        assert_eq!(result[0].groups[1].choice_label, 'C');
        assert!((result[0].groups[0].probabilities[1] - 0.7310586).abs() < 1e-6);
        assert!((result[0].groups[1].probabilities[0] - 0.8807971).abs() < 1e-6);
    }

    #[test]
    fn audio_scoring_preserves_payload_and_scores_candidate_logits() {
        struct Mock<'a>(&'a BPETokenizer, bool);
        impl TextRuntime for Mock<'_> {
            fn arch(&self) -> &str {
                "audio-mock"
            }
            fn context_length(&self) -> usize {
                1024
            }
            fn generate(
                &mut self,
                _: &GenerationRequest,
                _: &mut dyn TokenSink,
            ) -> Result<GeneratedText, String> {
                panic!("scoring must use logits")
            }
            fn validate_audio_request(
                &self,
                request: &GenerationRequest,
                audio: &[Vec<f32>],
            ) -> Result<(), String> {
                assert_eq!(request.max_new_tokens, 1);
                assert_eq!(audio.len(), 1);
                let bytes: Vec<u8> = request
                    .token_ids
                    .iter()
                    .flat_map(|&id| self.0.token_piece_bytes(id, true))
                    .collect();
                let text = String::from_utf8(bytes).unwrap();
                assert!(
                    text.contains("<|audio|>\n{\"context\": \"a\\\"b\""),
                    "{text}"
                );
                Ok(())
            }
            fn audio_logits(&mut self, _: &[u32], _: &[Vec<f32>]) -> Result<Vec<f32>, String> {
                let mut logits = vec![0.0; self.0.vocab_size()];
                logits[self.0.token_id("A").unwrap() as usize] = 1.0;
                logits[self.0.token_id("B").unwrap() as usize] =
                    if self.1 { f32::NAN } else { 2.0 };
                Ok(logits)
            }
            fn logits(&mut self, request: &GenerationRequest) -> Result<Vec<f32>, String> {
                assert_eq!(request.images, vec![vec![1, 2, 3]]);
                let bytes: Vec<u8> = request
                    .token_ids
                    .iter()
                    .flat_map(|&id| self.0.token_piece_bytes(id, true))
                    .collect();
                let text = String::from_utf8(bytes).unwrap();
                assert!(!text.contains("<|audio|>"));
                assert!(text.contains("{\"context\": \"a\\\"b\""));
                self.audio_logits(&request.token_ids, &[])
            }
        }
        let tokenizer = BPETokenizer::from_qwen3_embedded_merges().unwrap();
        let template = crate::prompt::jinja::JinjaChatTemplate::compile(
            "{{ messages[0].content }}\n{{ messages[1].content }}",
            "test",
        )
        .unwrap();
        let questions = [super::super::jev::JevQuestionInput {
            text: "q".into(),
            options: vec!["one".into(), "two".into()],
        }];
        let result = score(
            &mut Mock(&tokenizer, false),
            &tokenizer,
            &template,
            &[vec![0.0]],
            &[],
            "a\"b",
            &questions,
            None,
        )
        .unwrap();
        assert_eq!(result[0].choice_label, Some('B'));
        assert!((result[0].probabilities[1] - 0.7310586).abs() < 1e-6);
        let image_result = score(
            &mut Mock(&tokenizer, false),
            &tokenizer,
            &template,
            &[],
            &[vec![1, 2, 3]],
            "a\"b",
            &questions,
            None,
        )
        .unwrap();
        assert_eq!(image_result[0].probabilities, result[0].probabilities);
        assert!(score(
            &mut Mock(&tokenizer, true),
            &tokenizer,
            &template,
            &[vec![0.0]],
            &[],
            "a\"b",
            &questions,
            None
        )
        .unwrap_err()
        .contains("invalid vocabulary logits"));
    }
}
