use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::qwen3::{Qwen3Input, Qwen3Model, Qwen3Session};
use crate::models::qwen35::{build_qwen35_positions, Qwen35Model, Qwen35Session};
use crate::models::gemma4::{Gemma4Model, Gemma4Session};
use crate::models::lfm2moe::trunk::forward::run_forward_logits_lfm2moe_with_batch;
use crate::models::lfm2::trunk::forward::run_forward_logits_lfm2_with_batch;
use crate::models::lfm25::trunk::forward::run_forward_logits_lfm25_with_batch;
use crate::models::llama::trunk::forward::run_forward_logits_llama_with_batch;
use crate::models::nemotron_h::trunk::load_nemotron_tokenizer;
use crate::models::nemotron_h::trunk::run_forward_logits_nemotron_h;
use crate::models::spark::SparkSession;
use crate::prompt::{
    append_qwen_assistant_prefix, append_qwen_message_tokens, build_hunyuan_chat_prompt,
    HunyuanMessage,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct JevQuestionInput {
    pub text: String,
    pub options: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JevMode {
    Choice,
    Binary,
    Score,
    MultiSelect,
    BlockChoice,
}

#[derive(Clone, Debug)]
pub struct JevResult {
    pub mode: JevMode,
    pub question: String,
    pub labels: Vec<char>,
    pub descriptions: Vec<String>,
    pub values: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub choice_label: Option<char>,
    pub positive_label: Option<char>,
    pub probability_positive: Option<f32>,
    pub score: Option<f32>,
    pub confidence: f32,
    pub entropy: f32,
    pub margin: f32,
    pub prefill_ms: u128,
}

/// JEV-style single-forward-pass decision.
///
/// Protocol (paraphrased from `references/openjev/decisionmaking/prompts.py`):
///   1. For each question, build chat prompt with system instruction + JSON
///      user payload.
///   2. Run a single prefill per question, read last-position logits.
///   3. Look up logits for label tokens (A, B, C, …) which must each be a
///      single token in the tokenizer (otherwise error out).
///   4. Return softmax over those labels.
///
/// Per-question mode is auto-detected:
///   - any option contains `:` → score (text:value)
///   - K==2 and --jev-positive set → binary
///   - else → choice
///
/// Output format (stdout):
///   text mode (single question):
///     choice: B
///     probabilities:
///       A: 0.124
///       B: 0.683
///       C: 0.193
///   text mode (binary):
///     choice: A
///     probability: 0.683
///   text mode (score):
///     score: 3.42
///   json mode: one JSON object per question (newline-delimited).
///
/// Notes:
/// - This routine intentionally does **not** run autoregressive decode.
/// - Qwen3-0.6B-Instruct is recommended; base Qwen3-0.6B still runs but
///   the protocol's instruction-following is not guaranteed.
///
/// `Choice` / `Binary` / `Score` use a **global softmax** over all candidates
/// (mutual-exclusion normalization). For **multi-select** (independent
/// per-item binary decisions) and **block-choice** (per-block single-select
/// with blocks independent), see `run_jev_grouped_decision` which uses
/// **per-group softmax** — groups are normalized independently, avoiding
/// cross-group probability contamination. The two code paths are fully
/// isolated: this function and its helpers (`build_jev_prompt`,
/// `compute_jev_result`, `prepare_jev_questions`) are never called by the
/// grouped path, and vice versa.

pub fn run_jev_decision(
    source: Arc<dyn TensorSource>,
    context: &str,
    questions: &[JevQuestionInput],
    positive: Option<&str>,
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<(), String> {
    let prepared = prepare_jev_questions(questions, positive)?;

    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    eprintln!("JEV: arch = {:?}", arch);

    let t0 = Instant::now();
    let results = match &*arch {
        "qwen3" | "qwen3vl" => run_jev_decision_qwen3(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "qwen35" => run_jev_decision_qwen35(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "llama" | "k2-horizon" | "granite" | "nanbeige" | "qwen2_2" => run_jev_decision_llama(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "gemma4" => run_jev_decision_gemma4(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "lfm2" => run_jev_decision_lfm2(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "spark2_5" => run_jev_decision_spark(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "lfm25" => run_jev_decision_lfm25(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "lfm2moe" => run_jev_decision_lfm2moe(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "nemotron_h" => run_jev_decision_nemotron_h(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "hunyuan-dense" => run_jev_decision_hunyuan(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        other => {
            return Err(format!(
                "--jev is not yet supported for architecture {:?}; \
                 currently supported: qwen3 / qwen3vl / qwen35 / \
                 llama / k2-horizon / granite / nanbeige / qwen2_2 / \
                 gemma4 / lfm2 / lfm25 / spark2_5 / hunyuan-dense / nemotron_h",
                other
            ));
        }
    };

    if output_json {
        for r in &results {
            let line = serde_json::to_string(r).map_err(|e| format!("json encode: {e}"))?;
            println!("{}", line);
        }
    } else if results.len() == 1 {
        let r = &results[0];
        println!("\n--- JEV decision ---");
        match r.mode {
            JevMode::Choice => {
                println!("choice: {}", r.choice_label.unwrap_or('?'));
                println!("probabilities:");
                for (i, p) in r.probabilities.iter().enumerate() {
                    println!("  {}: {:.4}", r.labels[i], p);
                }
            }
            JevMode::Binary => {
                println!("choice: {}", r.choice_label.unwrap_or('?'));
                println!(
                    "probability ({}): {:.4}",
                    r.positive_label.unwrap_or('?'),
                    r.probability_positive.unwrap_or(0.0)
                );
            }
            JevMode::Score => {
                println!("score: {:.4}", r.score.unwrap_or(0.0));
                println!("breakdown:");
                for (i, p) in r.probabilities.iter().enumerate() {
                    println!(
                        "  {}: {:.4} × {} = {:.4}",
                        r.labels[i],
                        p,
                        r.values[i],
                        p * r.values[i]
                    );
                }
            }
            JevMode::MultiSelect | JevMode::BlockChoice => {
                return Err(
                    "Grouped modes (MultiSelect/BlockChoice) use run_jev_grouped_decision, not run_jev_decision"
                        .into(),
                );
            }
        }
        println!(
            "confidence: {:.4} | entropy: {:.4} | margin: {:.4}",
            r.confidence, r.entropy, r.margin
        );
        println!("prefill: {} ms", r.prefill_ms);
    } else {
        println!("\n--- JEV decisions ({} questions) ---", results.len());
        for (qi, r) in results.iter().enumerate() {
            println!("\nQ{}: {}", qi + 1, r.question);
            match r.mode {
                JevMode::Choice => {
                    println!("  choice: {}", r.choice_label.unwrap_or('?'));
                    for (i, p) in r.probabilities.iter().enumerate() {
                        println!("    {}: {:.4}", r.labels[i], p);
                    }
                }
                JevMode::Binary => {
                    println!("  choice: {}", r.choice_label.unwrap_or('?'));
                    println!(
                        "  P({}): {:.4}",
                        r.positive_label.unwrap_or('?'),
                        r.probability_positive.unwrap_or(0.0)
                    );
                }
                JevMode::Score => {
                    println!("  score: {:.4}", r.score.unwrap_or(0.0));
                }
                JevMode::MultiSelect | JevMode::BlockChoice => {
                    return Err(
                        "Grouped modes (MultiSelect/BlockChoice) use run_jev_grouped_decision, not run_jev_decision"
                            .into(),
                    );
                }
            }
            println!(
                "  confidence: {:.4} | entropy: {:.4} | margin: {:.4}",
                r.confidence, r.entropy, r.margin
            );
        }
    }

    let total_ms = t0.elapsed().as_millis();
    eprintln!("\nJEV total: {} ms ({} questions)", total_ms, results.len());
    Ok(())
}

fn prepare_jev_questions(
    questions: &[JevQuestionInput],
    positive: Option<&str>,
) -> Result<Vec<PreparedQuestion>, String> {
    if questions.is_empty() {
        return Err("--jev requires at least one --jev-question".into());
    }
    let mut per_question = Vec::with_capacity(questions.len());
    for q in questions {
        if q.options.len() < 2 {
            return Err(format!(
                "Question {:?} requires at least 2 --jev-option values",
                q.text
            ));
        }
        if q.options.len() > 26 {
            return Err(format!(
                "Question {:?} has {} options; JEV supports at most 26 (A..Z)",
                q.text,
                q.options.len()
            ));
        }
        let has_colon = q.options.iter().any(|opt| opt.contains(':'));
        let mode = if has_colon {
            JevMode::Score
        } else if q.options.len() == 2 && positive.is_some() {
            JevMode::Binary
        } else {
            JevMode::Choice
        };
        let mut descriptions = Vec::with_capacity(q.options.len());
        let mut values = Vec::with_capacity(q.options.len());
        for opt in &q.options {
            if mode == JevMode::Score {
                let (desc, val) = opt.rsplit_once(':').ok_or_else(|| {
                    format!("Score option \"{}\" must be \"description:value\"", opt)
                })?;
                let v: f32 = val
                    .parse()
                    .map_err(|e| format!("Score value {:?} is not a float: {}", val, e))?;
                descriptions.push(desc.to_string());
                values.push(v);
            } else {
                descriptions.push(opt.clone());
                values.push(0.0);
            }
        }
        let positive_label = if mode == JevMode::Binary {
            let p = positive.unwrap().to_string();
            let pch = p.chars().next().unwrap_or('A').to_ascii_uppercase();
            if !('A'..='Z').contains(&pch) {
                return Err(format!(
                    "--jev-positive must be a single letter A..Z, got {:?}",
                    p
                ));
            }
            Some(pch)
        } else {
            None
        };
        per_question.push(PreparedQuestion {
            mode,
            text: q.text.clone(),
            descriptions,
            values,
            positive_label,
        });
    }
    Ok(per_question)
}

fn run_jev_decision_qwen3(
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
    let mut scorer = Qwen3JevScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads", scorer.pool().n_threads());
    }
    run_jev_decision_core(source, context, per_question, output_json, &mut scorer)
}

/// Qwen3 JEV scorer — owns the `Qwen3Model` (which embeds the
/// tokenizer + compute pool). Each `forward_logits` rebuilds a
/// fresh `Qwen3Session` so the KV cache stays ephemeral.
struct Qwen3JevScorer {
    model: crate::models::qwen3::Qwen3Model,
    max_ctx: usize,
    prefill_batch_size: usize,
}

impl Qwen3JevScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        let pool = Arc::new(ComputePool::new(n_threads));
        let model =
            crate::models::qwen3::Qwen3Model::from_source(source.clone(), Arc::new(tokenizer), pool)?;
        let max_ctx = model.config().n_ctx;
        Ok(Self {
            model,
            max_ctx,
            prefill_batch_size,
        })
    }

    fn pool(&self) -> Arc<ComputePool> {
        self.model.pool()
    }
}

impl JevScorer for Qwen3JevScorer {
    fn scorer_label(&self) -> &'static str {
        "Qwen3"
    }

    fn build_prompt(
        &self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let (token_ids, _payload) =
            build_jev_prompt(self.model.tokenizer(), context, q, false)?;
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

fn run_jev_decision_qwen35(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevResult>, String> {
    // Qwen3.5's `Qwen35Model<'a>` borrows from the source. Holding it
    // inside the scorer struct would force the scorer itself to be
    // lifetime-parameterized, which complicates the `JevScorer`
    // trait object story. For Qwen3.5 we keep the per-question loop
    // inline here — the scorer pattern works well for the trunks
    // that own their model data (gemma4, llama-family) or wrap a
    // free function (lfm*/spark/nemotron-h/hunyuan).
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
    verify_label_tokens_single(&tokenizer)?;

    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let pool = Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());

    let mut model = crate::models::qwen35::Qwen35Model::from_source(source.as_ref())
        .map_err(|error| format!("Failed to parse Qwen3.5 model: {error}"))?;
    let n_ctx = model.config.n_ctx;

    let mut results: Vec<JevResult> = Vec::with_capacity(per_question.len());
    for q in per_question {
        let labels = jev_labels(q);
        let (token_ids, payload_str) = build_jev_prompt(&tokenizer, context, q, output_json)?;
        if !output_json {
            print_jev_question(q, &labels);
        }
        let _ = payload_str;

        let (positions, _next) = build_qwen35_positions(&token_ids, None, &[])
            .map_err(|e| format!("Failed to build Qwen3.5 positions: {e}"))?;

        // Recreate session per question (Ephemeral KV).
        let mut session = crate::models::qwen35::Qwen35Session::new_with_prefill_batch_size(
            &mut model,
            n_ctx.min(token_ids.len() + 1),
            prefill_batch_size,
            pool.clone(),
        )?;
        let t0 = Instant::now();
        let logits = session
            .forward_logits(&token_ids, &positions)
            .map_err(|e| format!("Qwen3.5 forward_logits failed: {e}"))?;
        let prefill_dur = t0.elapsed();
        let result = compute_jev_result(q, &tokenizer, &labels, &logits, prefill_dur.as_millis());
        results.push(result);
    }
    Ok(results)
}

fn run_jev_decision_llama(
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
    let mut scorer = LlamaJevScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Llama-family)", n_threads);
    }
    run_jev_decision_core(source, context, per_question, output_json, &mut scorer)
}

/// Llama-family JEV scorer — covers llama / k2-horizon / granite /
/// nanbeige / qwen2_2 / minicpm. The chat template varies per arch
/// but the forward path is uniform
/// (`run_forward_logits_llama_with_batch`).
struct LlamaJevScorer {
    tokenizer: BPETokenizer,
    source: Arc<dyn TensorSource>,
    arch: String,
    n_threads: usize,
    prefill_batch_size: usize,
}

impl LlamaJevScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        let arch = source
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val())
            .unwrap_or_default()
            .to_string();
        Ok(Self {
            tokenizer,
            source,
            arch,
            n_threads,
            prefill_batch_size,
        })
    }
}

impl JevScorer for LlamaJevScorer {
    fn scorer_label(&self) -> &'static str {
        "Llama-family"
    }

    fn build_prompt(
        &self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let system = jev_system_prompt(q.mode);
        let payload = jev_payload_json(context, q)?;
        // Per-arch chat template; mirrors `llama::trunk::forward::run_inference`.
        let prompt_text = if self.arch == "k2-horizon" {
            format!(
                "<|start_of_role|>system<|end_of_role|>{system}<|end_of_text|>\n\
                 <|start_of_role|>user<|end_of_role|>{payload}<|end_of_text|>\n\
                 <|start_of_role|>assistant<|end_of_role|>"
            )
        } else if self.arch == "granite" {
            format!(
                "<|start_of_role|>system<|end_of_role|>{system}<|end_of_text|>\n\
                 <|start_of_role|>user<|end_of_role|>{payload}<|end_of_text|>\n\
                 <|start_of_role|>assistant<|end_of_role|>"
            )
        } else if self.arch == "nanbeige" {
            format!("{system}\n\n{payload}\n\nAnswer:")
        } else {
            format!("system\n{system}\nuser\n{payload}\nassistant\n")
        };
        let add_special = self.arch == "nanbeige";
        let mut token_ids = self.tokenizer.encode(
            &prompt_text,
            EncodeOptions {
                add_special,
                parse_special: true,
            },
        );
        if !add_special {
            if let Some(bos) = self.tokenizer.bos_id() {
                token_ids.insert(0, bos);
            }
        }
        Ok((labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::llama::trunk::run_forward_logits_llama_with_batch(
            self.source.as_ref(),
            &token_ids,
            self.n_threads,
            KvFormat::F16,
            8192,
            self.prefill_batch_size,
        )
        .map_err(|e| format!("Llama forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}

fn run_jev_decision_gemma4(
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
    let mut scorer = Gemma4JevScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Gemma4)", n_threads);
    }
    run_jev_decision_core(source, context, per_question, output_json, &mut scorer)
}

/// Gemma4 JEV scorer — uses the session API. The session is
/// recreated per question to mimic the legacy ephemeral-KV
/// behaviour (each question is a fresh prefill).
struct Gemma4JevScorer {
    tokenizer: BPETokenizer,
    model: crate::models::gemma4::Gemma4Model,
    prefill_batch_size: usize,
}

impl Gemma4JevScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        _n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        let model = crate::models::gemma4::Gemma4Model::from_source(source.clone(), _n_threads)
            .map_err(|e| format!("Failed to load Gemma4 model: {e}"))?;
        Ok(Self {
            tokenizer,
            model,
            prefill_batch_size,
        })
    }
}

impl JevScorer for Gemma4JevScorer {
    fn scorer_label(&self) -> &'static str {
        "Gemma4"
    }

    fn build_prompt(
        &self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let system = jev_system_prompt(q.mode);
        let payload = jev_payload_json(context, q)?;
        let prompt_text = format!("{system}\n\n{payload}\n\n<turn|>\n<|turn>model\n");
        let bos = self
            .tokenizer
            .bos_id()
            .ok_or("Gemma4 tokenizer missing BOS")?;
        let mut ids = self.tokenizer.encode(
            &prompt_text,
            EncodeOptions {
                add_special: false,
                parse_special: true,
            },
        );
        if ids.first() != Some(&bos) {
            ids.insert(0, bos);
        }
        Ok((labels, ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        let mut session = crate::models::gemma4::Gemma4Session::new_with_prefill_batch_size(
            &self.model,
            KvFormat::F16,
            self.prefill_batch_size,
        )?;
        let t0 = Instant::now();
        let logits = session
            .forward_logits(&token_ids)
            .map_err(|e| format!("Gemma4 forward_logits failed: {e}"))?;
        Ok((logits, t0.elapsed()))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}

fn run_jev_decision_lfm2(
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
    let mut scorer = Lfm2JevScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (LFM2)", n_threads);
    }
    run_jev_decision_core(
        source,
        context,
        per_question,
        output_json,
        &mut scorer,
    )
}

/// LFM2 JEV scorer: uses the free-function prefill path
/// (`run_forward_logits_lfm2_with_batch`). The chat template is
/// `"system\n...\nuser\n{json}\nassistant\n"` which is the LFM2
/// convention shared with LFM2.5 / LFM2-MoE.
struct Lfm2JevScorer {
    tokenizer: BPETokenizer,
    source: Arc<dyn TensorSource>,
    n_threads: usize,
    prefill_batch_size: usize,
}

impl Lfm2JevScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        Ok(Self {
            tokenizer,
            source,
            n_threads,
            prefill_batch_size,
        })
    }
}

impl JevScorer for Lfm2JevScorer {
    fn scorer_label(&self) -> &'static str {
        "LFM2"
    }

    fn build_prompt(
        &self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let system = jev_system_prompt(q.mode);
        let payload = jev_payload_json(context, q)?;
        let mut token_ids = Vec::new();
        if let Some(bos) = self.tokenizer.bos_id() {
            token_ids.push(bos);
        }
        token_ids.extend(self.tokenizer.encode(
            &format!("system\n{system}\n"),
            EncodeOptions {
                add_special: false,
                parse_special: false,
            },
        ));
        token_ids.extend(self.tokenizer.encode(
            &format!("user\n{payload}\n"),
            EncodeOptions {
                add_special: false,
                parse_special: false,
            },
        ));
        token_ids.extend(self.tokenizer.encode(
            "assistant\n",
            EncodeOptions {
                add_special: false,
                parse_special: false,
            },
        ));
        Ok((labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::lfm2::run_forward_logits_lfm2_with_batch(
            self.source.as_ref(),
            &token_ids,
            self.n_threads,
            KvFormat::F16,
            8192,
            self.prefill_batch_size,
        )
        .map_err(|e| format!("LFM2 forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}

fn run_jev_decision_spark(
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
    let mut scorer = SparkJevScorer::new(source.clone(), n_threads)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Spark)", n_threads);
    }
    let _ = prefill_batch_size;
    run_jev_decision_core(source, context, per_question, output_json, &mut scorer)
}

/// Spark 2.5 JEV scorer — uses the session API
/// (`SparkSession::forward_logits`) which owns its compute pool
/// internally, so the scorer holds the session rather than the
/// pool + free function.
struct SparkJevScorer {
    tokenizer: BPETokenizer,
    session: crate::models::spark::SparkSession,
}

impl SparkJevScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
    ) -> Result<Self, String> {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        let session = crate::models::spark::SparkSession::new(
            source.as_ref(),
            Arc::new(ComputePool::new(n_threads)),
            8192,
        )?;
        Ok(Self { tokenizer, session })
    }
}

impl JevScorer for SparkJevScorer {
    fn scorer_label(&self) -> &'static str {
        "Spark"
    }

    fn build_prompt(
        &self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let sos = "<｜start▁of▁sentence｜>";
        let eos = "<｜end▁of▁sentence｜>";
        let system = jev_system_prompt(q.mode);
        let payload = jev_payload_json(context, q)?;
        let prompt_text = format!(
            "{sos}<|System|>\n{system}{eos}\
             {sos}<|User|>{payload}{eos}\
             {sos}<|Bot|></think>",
            sos = sos,
            eos = eos,
        );
        let mut token_ids = self.tokenizer.encode(
            &prompt_text,
            EncodeOptions {
                add_special: false,
                parse_special: true,
            },
        );
        if self.tokenizer.add_bos() {
            if let Some(bos) = self.tokenizer.bos_id() {
                token_ids.insert(0, bos);
            }
        }
        Ok((labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        let t0 = Instant::now();
        let logits = self
            .session
            .forward_logits(&token_ids)
            .map_err(|e| format!("Spark forward_logits failed: {e}"))?;
        Ok((logits, t0.elapsed()))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}

fn run_jev_decision_lfm25(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevResult>, String> {
    // LFM2.5 chat format is identical to LFM2 — see [`Lfm2JevScorer`].
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = Lfm25JevScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (LFM2.5)", n_threads);
    }
    run_jev_decision_core(source, context, per_question, output_json, &mut scorer)
}

/// LFM2.5 JEV scorer — chat template + free-function forward path,
/// identical to LFM2 but routed at the `crate::models::lfm25`
/// module instead of `lfm2`.
struct Lfm25JevScorer {
    inner: Lfm2JevScorer,
}

impl Lfm25JevScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: Lfm2JevScorer::new(source, n_threads, prefill_batch_size)?,
        })
    }
}

impl JevScorer for Lfm25JevScorer {
    fn scorer_label(&self) -> &'static str {
        "LFM2.5"
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
        crate::models::lfm25::run_forward_logits_lfm25_with_batch(
            self.inner.source.as_ref(),
            &token_ids,
            self.inner.n_threads,
            KvFormat::F16,
            8192,
            self.inner.prefill_batch_size,
        )
        .map_err(|e| format!("LFM2.5 forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}

fn run_jev_decision_lfm2moe(
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
    inner: Lfm2JevScorer,
}

impl Lfm2MoeJevScorer {
    fn new(
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

fn run_jev_decision_nemotron_h(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevResult>, String> {
    let _ = (n_threads_arg, prefill_batch_size);
    let mut scorer = NemotronHJevScorer::new(source.clone())?;
    let _ = output_json;
    run_jev_decision_core(source, context, per_question, false, &mut scorer)
}

/// Nemotron-H JEV scorer — base model (no chat template). Uses
/// the standalone `load_nemotron_tokenizer` + `run_forward_logits_nemotron_h`
/// free function (Nemotron-H has no Session API). The
/// `n_threads_arg` and `prefill_batch_size` parameters are
/// accepted for trait-compatibility but ignored at runtime —
/// Nemotron-H's per-step body still walks the per-token path.
struct NemotronHJevScorer {
    tokenizer: BPETokenizer,
    source: Arc<dyn TensorSource>,
}

impl NemotronHJevScorer {
    fn new(source: Arc<dyn TensorSource>) -> Result<Self, String> {
        let tokenizer = crate::models::nemotron_h::trunk::load_nemotron_tokenizer(source.as_ref())?;
        verify_label_tokens_single(&tokenizer)?;
        Ok(Self { tokenizer, source })
    }
}

impl JevScorer for NemotronHJevScorer {
    fn scorer_label(&self) -> &'static str {
        "Nemotron-H"
    }

    fn build_prompt(
        &self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String> {
        let labels = jev_labels(q);
        let system = jev_system_prompt(q.mode);
        let payload = jev_payload_json(context, q)?;
        let prompt_text = format!("{system}\n\n{payload}\n\nAnswer:");
        let token_ids = self.tokenizer.encode(
            &prompt_text,
            EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        );
        Ok((labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::nemotron_h::trunk::run_forward_logits_nemotron_h(
            self.source.clone(),
            &token_ids,
        )
        .map_err(|e| format!("Nemotron-H forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }
}

fn run_jev_decision_hunyuan(
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
struct HunyuanJevScorer {
    model: crate::models::qwen3::Qwen3Model,
    max_ctx: usize,
    prefill_batch_size: usize,
}

impl HunyuanJevScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        verify_label_tokens_single(&tokenizer)?;
        let pool = Arc::new(ComputePool::new(n_threads));
        let model =
            crate::models::qwen3::Qwen3Model::from_source(source.clone(), Arc::new(tokenizer), pool)?;
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

fn verify_label_tokens_single(tokenizer: &BPETokenizer) -> Result<(), String> {
    for label_char in b'A'..=b'Z' {
        let label_text = (label_char as char).to_string();
        let label_enc = tokenizer.encode(
            &label_text,
            EncodeOptions {
                add_special: false,
                parse_special: false,
            },
        );
        if label_enc.len() != 1 {
            return Err(format!(
                "Label \"{}\" tokenizes to {} tokens; the tokenizer must encode A..Z as single tokens",
                label_text, label_enc.len()
            ));
        }
    }
    Ok(())
}

/// Returns the system prompt for a JEV question based on its `mode`.
/// Shared across every per-arch `run_jev_decision_*` so a single edit
/// flows through all backends.
fn jev_system_prompt(mode: JevMode) -> &'static str {
    match mode {
        JevMode::Score => {
            "Score the situation using the supplied context and numeric candidates. \
                           Reply with only its letter label."
        }
        JevMode::Choice | JevMode::Binary => {
            "Answer the question using the supplied context and candidate answers. \
              Select the single best answer. Reply with only its letter label."
        }
        JevMode::MultiSelect | JevMode::BlockChoice => {
            // Grouped modes use the per-group scoring style — the
            // runner enumerates the candidate set per question so the
            // exact wording is per-arch (handled inside the grouped
            // payload builders below).
            "Answer the question using the supplied context and candidate answers. \
              Reply with the relevant labels."
        }
    }
}

/// Allocates the A..Z letter labels for `q`'s candidates. Returns
/// `vec![]` for an empty question so callers can `is_empty()` to
/// distinguish "no candidates" from a build error.
fn jev_labels(q: &PreparedQuestion) -> Vec<char> {
    (b'A'..=(b'A' + q.descriptions.len() as u8 - 1))
        .map(|b| b as char)
        .collect()
}

/// Renders the JEV JSON payload (context + question + candidates)
/// for embedding in the per-arch chat template. Used by every
/// per-arch `run_jev_decision_*` so the JSON shape stays in sync.
fn jev_payload_json(context: &str, q: &PreparedQuestion) -> Result<String, String> {
    let labels = jev_labels(q);
    let mut payload = String::from("{\"context\": ");
    payload
        .push_str(&serde_json::to_string(context).map_err(|e| format!("context json: {e}"))?);
    payload.push_str(", \"question\": ");
    payload
        .push_str(&serde_json::to_string(&q.text).map_err(|e| format!("question json: {e}"))?);
    payload.push_str(", \"candidates\": {");
    for (i, (label_char, desc)) in labels.iter().zip(q.descriptions.iter()).enumerate() {
        if i > 0 {
            payload.push(',');
        }
        payload.push('"');
        payload.push(*label_char);
        payload.push_str("\": ");
        payload
            .push_str(&serde_json::to_string(desc).map_err(|e| format!("desc json: {e}"))?);
    }
    payload.push_str("}}");
    Ok(payload)
}

fn build_jev_prompt(
    tokenizer: &BPETokenizer,
    context: &str,
    q: &PreparedQuestion,
    _output_json: bool,
) -> Result<(Vec<u32>, String), String> {
    let system = jev_system_prompt(q.mode);
    let labels = jev_labels(q);
    let payload = jev_payload_json(context, q)?;

    let mut token_ids = Vec::new();
    append_qwen_message_tokens(
        &mut token_ids,
        tokenizer,
        "system",
        &tokenizer.encode(
            system,
            EncodeOptions {
                add_special: false,
                parse_special: false,
            },
        ),
    )?;
    append_qwen_message_tokens(
        &mut token_ids,
        tokenizer,
        "user",
        &tokenizer.encode(
            &payload,
            EncodeOptions {
                add_special: false,
                parse_special: false,
            },
        ),
    )?;
    append_qwen_assistant_prefix(&mut token_ids, tokenizer, false)?;
    Ok((token_ids, payload))
}

fn print_jev_question(q: &PreparedQuestion, labels: &[char]) {
    println!(
        "\n--- JEV question ({} candidates) ---",
        q.descriptions.len()
    );
    println!("Q: {}", q.text);
    for (i, desc) in q.descriptions.iter().enumerate() {
        if q.mode == JevMode::Score {
            println!("  {}: {} = {}", labels[i], desc, q.values[i]);
        } else {
            println!("  {}: {}", labels[i], desc);
        }
    }
}

fn compute_jev_result(
    q: &PreparedQuestion,
    tokenizer: &BPETokenizer,
    labels: &[char],
    logits: &[f32],
    prefill_ms: u128,
) -> JevResult {
    let label_ids: Vec<u32> = labels
        .iter()
        .map(|c| {
            let s = c.to_string();
            tokenizer
                .encode(
                    &s,
                    EncodeOptions {
                        add_special: false,
                        parse_special: false,
                    },
                )
                .into_iter()
                .next()
                .unwrap_or(0)
        })
        .collect();
    let max_logit = label_ids
        .iter()
        .map(|&id| logits[id as usize])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = label_ids
        .iter()
        .map(|&id| (logits[id as usize] - max_logit).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    for v in exps.iter_mut() {
        *v /= sum;
    }

    let chosen_idx = exps
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let choice_label = Some(labels[chosen_idx]);
    let (prob_positive, score) = match q.mode {
        JevMode::Binary => {
            let pos_ch = q.positive_label.unwrap();
            let pos_idx = (pos_ch as u32 - 'A' as u32) as usize;
            let prob = exps[pos_idx];
            (Some(prob), None)
        }
        JevMode::Score => {
            let s: f32 = exps.iter().zip(q.values.iter()).map(|(p, v)| p * v).sum();
            (None, Some(s))
        }
        _ => (None, None),
    };

    let confidence = exps[chosen_idx];

    let entropy: f32 = -exps
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| p * p.ln())
        .sum::<f32>();

    let mut sorted = exps.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let margin = if sorted.len() >= 2 {
        sorted[0] - sorted[1]
    } else {
        sorted[0]
    };

    JevResult {
        mode: q.mode,
        question: q.text.clone(),
        labels: labels.to_vec(),
        descriptions: q.descriptions.clone(),
        values: q.values.clone(),
        probabilities: exps,
        choice_label,
        positive_label: q.positive_label,
        probability_positive: prob_positive,
        score,
        confidence,
        entropy,
        margin,
        prefill_ms,
    }
}

struct PreparedQuestion {
    mode: JevMode,
    text: String,
    descriptions: Vec<String>,
    values: Vec<f32>,
    positive_label: Option<char>,
}

/// Per-architecture JEV scorer.
///
/// Each implementation encapsulates:
/// - the per-arch chat template + tokenizer interaction,
/// - the prefill + LM-head forward (session-based for qwen3/qwen35/
///   gemma4/spark, free-function for llama/lfm2/lfm25/lfm2moe/
///   nemotron_h),
/// - tokenizer access for `compute_jev_result` to resolve label
///   tokens.
///
/// `run_jev_decision_core` (defined further down) drives the
/// per-question loop using only this trait surface, so adding a
/// new trunk is a single `JevScorer` impl + one dispatch table
/// entry instead of ~95 lines of copy-pasted boilerplate.
trait JevScorer {
    /// Display name used in `eprintln!("compute pool: {} threads ({})", n, label)`.
    fn scorer_label(&self) -> &'static str;

    /// Build the per-question chat-template token ids for `q`.
    /// Returns the A..Z labels that the prompt encoded + the
    /// token id sequence to feed into `forward_logits`. The
    /// `payload_str` field is the rendered JSON (used by
    /// `print_jev_question` for debug output).
    fn build_prompt(
        &self,
        context: &str,
        q: &PreparedQuestion,
    ) -> Result<(Vec<char>, Vec<u32>), String>;

    /// Run the prefill + LM-head forward for a single question,
    /// returning the final logits and the elapsed duration.
    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String>;

    /// Borrow the tokenizer for `compute_jev_result`.
    fn tokenizer(&self) -> &BPETokenizer;
}

/// Per-question dispatch shared by every `run_jev_decision_*`
/// function. Calls `scorer.build_prompt` for token ids + labels,
/// `scorer.forward_logits` for the prefill, and finally
/// `compute_jev_result` to score the candidate labels. The
/// per-arch code now only owns the scorer construction (see the
/// `run_jev_decision_<arch>` functions below), not the
/// per-question loop.
fn run_jev_decision_core<S: JevScorer>(
    _source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedQuestion],
    output_json: bool,
    scorer: &mut S,
) -> Result<Vec<JevResult>, String> {
    let mut results: Vec<JevResult> = Vec::with_capacity(per_question.len());
    for q in per_question {
        let (labels, token_ids) = scorer.build_prompt(context, q)?;
        if !output_json {
            print_jev_question(q, &labels);
        }
        let (logits, prefill_dur) = scorer.forward_logits(token_ids)?;
        let result = compute_jev_result(
            q,
            scorer.tokenizer(),
            &labels,
            &logits,
            prefill_dur.as_millis(),
        );
        results.push(result);
    }
    Ok(results)
}

/// Per-architecture JEV scorer for the **grouped** JEV modes
/// (`MultiSelect` + `BlockChoice`). Same idea as [`JevScorer`]
/// but the input is a `PreparedGroupedQuestion` (one question
/// with multiple independent groups) and the result is a
/// `JevGroupedResult` (per-group probabilities under per-group
/// softmax — see [`run_jev_grouped_decision`] for the protocol).
///
/// `run_jev_grouped_core` drives the per-question loop using only
/// this trait surface, so adding a new trunk is a single impl
/// + dispatch entry instead of ~40 lines of copy-pasted boilerplate.
trait JevGroupedScorer {
    /// Display name used in the dispatch table's error message
    /// (the `eprintln!("compute pool: {} threads (...)")` is owned
    /// by the per-arch wrapper, not the scorer).
    fn scorer_label(&self) -> &'static str;

    /// Build the per-question grouped chat-template token ids
    /// for `q`. Returns the per-group A..Z labels (one vec per
    /// group, in the same order as `q.groups`) and the token id
    /// sequence to feed into `forward_logits`.
    fn build_grouped_prompt(
        &self,
        context: &str,
        q: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String>;

    /// Run the prefill + LM-head forward for a single grouped
    /// question, returning the final logits and elapsed duration.
    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String>;

    /// Borrow the tokenizer for `compute_grouped_jev_result`.
    fn tokenizer(&self) -> &BPETokenizer;
}

/// Per-question dispatch shared by every `run_jev_grouped_<arch>`
/// function. Calls `scorer.build_grouped_prompt` for token ids +
/// per-group labels, `scorer.forward_logits` for the prefill,
/// and finally `compute_grouped_jev_result` to score each group.
fn run_jev_grouped_core<S: JevGroupedScorer>(
    _source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    output_json: bool,
    scorer: &mut S,
) -> Result<Vec<JevGroupedResult>, String> {
    let mut results = Vec::with_capacity(per_question.len());
    for q in per_question {
        let (group_labels, token_ids) = scorer.build_grouped_prompt(context, q)?;
        if !output_json {
            eprintln!("\n--- JEV grouped question ({} groups) ---", q.groups.len());
            eprintln!("Q: {}", q.text);
        }
        let (logits, prefill_dur) = scorer.forward_logits(token_ids)?;
        results.push(compute_grouped_jev_result(
            q,
            scorer.tokenizer(),
            &group_labels,
            &logits,
            prefill_dur.as_millis(),
        ));
    }
    Ok(results)
}

impl serde::Serialize for JevResult {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("JevResult", 10)?;
        st.serialize_field(
            "mode",
            match self.mode {
                JevMode::Choice => "choice",
                JevMode::Binary => "binary",
                JevMode::Score => "score",
                JevMode::MultiSelect => "multi_select",
                JevMode::BlockChoice => "block_choice",
            },
        )?;
        st.serialize_field("question", &self.question)?;
        let labels_str: Vec<String> = self.labels.iter().map(|c| c.to_string()).collect();
        st.serialize_field("labels", &labels_str)?;
        st.serialize_field("descriptions", &self.descriptions)?;
        if self.mode == JevMode::Score {
            st.serialize_field("values", &self.values)?;
        }
        let mut probs = serde_json::Map::new();
        for (i, p) in self.probabilities.iter().enumerate() {
            probs.insert(self.labels[i].to_string(), serde_json::json!(p));
        }
        st.serialize_field("probabilities", &probs)?;
        if self.mode == JevMode::Choice {
            st.serialize_field("choice", &self.choice_label.map(|c| c.to_string()))?;
        }
        if self.mode == JevMode::Binary {
            st.serialize_field("choice", &self.choice_label.map(|c| c.to_string()))?;
            st.serialize_field("positive", &self.positive_label.map(|c| c.to_string()))?;
            st.serialize_field("probability", &self.probability_positive)?;
        }
        if self.mode == JevMode::Score {
            st.serialize_field("score", &self.score)?;
        }
        st.serialize_field("confidence", &self.confidence)?;
        st.serialize_field("entropy", &self.entropy)?;
        st.serialize_field("margin", &self.margin)?;
        st.serialize_field("prefill_ms", &self.prefill_ms)?;
        st.end()
    }
}

// ---------------------------------------------------------------------------
// Grouped JEV: MultiSelect + BlockChoice
//
// Fully isolated from Choice/Binary/Score above. Uses per-group softmax
// (groups normalized independently) instead of global softmax. This avoids
// cross-group probability contamination: a high-scoring candidate in group 1
// does not suppress probabilities in group 2.
//
// - MultiSelect: each pair of options forms a binary group. Per-group softmax
//   gives independent yes/no probability per item.
// - BlockChoice: user explicitly defines blocks; each block is a group with
//   its own independent softmax.
//
// The forward pass is identical (single prefill, read last-token logits).
// Only the payload construction and post-processing differ.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct JevGroupedOption {
    pub description: String,
}

#[derive(Clone, Debug)]
pub struct JevGroupInput {
    pub label: String,
    pub options: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct JevGroupedQuestionInput {
    pub text: String,
    pub groups: Vec<JevGroupInput>,
}

struct PreparedGroup {
    label: String,
    descriptions: Vec<String>,
    values: Vec<f32>,
}

struct PreparedGroupedQuestion {
    mode: JevMode,
    text: String,
    groups: Vec<PreparedGroup>,
}

#[derive(Clone, Debug)]
pub struct JevGroupResult {
    pub label: String,
    pub labels: Vec<char>,
    pub descriptions: Vec<String>,
    pub values: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub choice_label: char,
    pub score: Option<f32>,
    pub confidence: f32,
    pub entropy: f32,
    pub margin: f32,
}

#[derive(Clone, Debug)]
pub struct JevGroupedResult {
    pub mode: JevMode,
    pub question: String,
    pub groups: Vec<JevGroupResult>,
    pub prefill_ms: u128,
}

pub fn prepare_jev_grouped_questions(
    questions: &[JevGroupedQuestionInput],
    mode: JevMode,
) -> Result<Vec<PreparedGroupedQuestion>, String> {
    if questions.is_empty() {
        return Err("--jev requires at least one --jev-question".into());
    }
    let mut per_question = Vec::with_capacity(questions.len());
    for q in questions {
        if q.groups.is_empty() {
            return Err(format!(
                "Question {:?} has no groups; use --jev-block to define at least one block",
                q.text
            ));
        }
        let mut total_options = 0usize;
        let mut prepared_groups = Vec::with_capacity(q.groups.len());
        for (gi, g) in q.groups.iter().enumerate() {
            if g.options.len() < 2 {
                return Err(format!(
                    "Question {:?} group {} ({:?}) needs at least 2 options, got {}",
                    q.text, gi + 1, g.label, g.options.len()
                ));
            }
            total_options += g.options.len();
            if total_options > 26 {
                return Err(format!(
                    "Question {:?} has {} total options across all groups; JEV supports at most 26 (A..Z)",
                    q.text, total_options
                ));
            }
            let has_colon = g.options.iter().any(|opt| opt.contains(':'));
            let mut descriptions = Vec::with_capacity(g.options.len());
            let mut values = Vec::with_capacity(g.options.len());
            for opt in &g.options {
                if has_colon {
                    let (desc, val_str) = opt.rsplit_once(':').ok_or_else(|| {
                        format!(
                            "Group {:?} option {:?} must be \"description:value\" (mixed with/without colon is not allowed)",
                            g.label, opt
                        )
                    })?;
                    let v: f32 = val_str
                        .parse()
                        .map_err(|e| format!("Score value {:?} is not a float: {}", val_str, e))?;
                    descriptions.push(desc.to_string());
                    values.push(v);
                } else {
                    descriptions.push(opt.clone());
                    values.push(0.0);
                }
            }
            prepared_groups.push(PreparedGroup {
                label: if g.label.is_empty() {
                    format!("group_{}", gi + 1)
                } else {
                    g.label.clone()
                },
                descriptions,
                values,
            });
        }
        per_question.push(PreparedGroupedQuestion {
            mode,
            text: q.text.clone(),
            groups: prepared_groups,
        });
    }
    Ok(per_question)
}

fn build_grouped_system() -> &'static str {
    "For each group, select the best option. Reply with only a letter label."
}

fn build_grouped_payload(context: &str, q: &PreparedGroupedQuestion) -> Result<String, String> {
    let mut payload = String::from("{\"context\": ");
    payload.push_str(&serde_json::to_string(context).map_err(|e| format!("context json: {e}"))?);
    payload.push_str(", \"question\": ");
    payload.push_str(&serde_json::to_string(&q.text).map_err(|e| format!("question json: {e}"))?);
    payload.push_str(", \"groups\": [");
    let mut label_char = b'A';
    for (gi, group) in q.groups.iter().enumerate() {
        if gi > 0 {
            payload.push_str(", ");
        }
        payload.push('{');
        for (oi, desc) in group.descriptions.iter().enumerate() {
            if oi > 0 {
                payload.push_str(", ");
            }
            payload.push('"');
            payload.push(label_char as char);
            payload.push_str("\": ");
            payload.push_str(&serde_json::to_string(desc).map_err(|e| format!("desc json: {e}"))?);
            label_char += 1;
        }
        payload.push('}');
    }
    payload.push_str("]}");
    Ok(payload)
}

fn allocate_group_labels(q: &PreparedGroupedQuestion) -> Vec<Vec<char>> {
    let mut all = Vec::with_capacity(q.groups.len());
    let mut next = b'A';
    for group in &q.groups {
        let count = group.descriptions.len();
        let labels: Vec<char> = (next..next + count as u8)
            .map(|b| b as char)
            .collect();
        next += count as u8;
        all.push(labels);
    }
    all
}

fn build_jev_token_ids_for_arch(
    arch: &str,
    tokenizer: &BPETokenizer,
    system: &str,
    payload: &str,
) -> Result<Vec<u32>, String> {
    match arch {
        "qwen3" | "qwen3vl" | "hunyuan-dense" => {
            let mut token_ids = Vec::new();
            append_qwen_message_tokens(
                &mut token_ids,
                tokenizer,
                "system",
                &tokenizer.encode(system, EncodeOptions { add_special: false, parse_special: false }),
            )?;
            append_qwen_message_tokens(
                &mut token_ids,
                tokenizer,
                "user",
                &tokenizer.encode(payload, EncodeOptions { add_special: false, parse_special: false }),
            )?;
            append_qwen_assistant_prefix(&mut token_ids, tokenizer, false)?;
            Ok(token_ids)
        }
        "qwen35" => {
            let mut token_ids = Vec::new();
            append_qwen_message_tokens(
                &mut token_ids,
                tokenizer,
                "system",
                &tokenizer.encode(system, EncodeOptions { add_special: false, parse_special: false }),
            )?;
            append_qwen_message_tokens(
                &mut token_ids,
                tokenizer,
                "user",
                &tokenizer.encode(payload, EncodeOptions { add_special: false, parse_special: false }),
            )?;
            append_qwen_assistant_prefix(&mut token_ids, tokenizer, false)?;
            Ok(token_ids)
        }
        "llama" | "k2-horizon" | "granite" | "nanbeige" | "qwen2_2" => {
            if arch == "k2-horizon" || arch == "granite" {
                let prompt = format!(
                    "<|start_of_role|>system<|end_of_role|>{system}<|end_of_text|>\n\
                     <|start_of_role|>user<|end_of_role|>{payload}<|end_of_text|>\n\
                     <|start_of_role|>assistant<|end_of_role|>"
                );
                let mut ids = tokenizer.encode(&prompt, EncodeOptions { add_special: false, parse_special: true });
                if let Some(bos) = tokenizer.bos_id() {
                    if ids.first() != Some(&bos) {
                        ids.insert(0, bos);
                    }
                }
                Ok(ids)
            } else if arch == "nanbeige" {
                let prompt = format!("{system}\n\n{payload}\n\nAnswer:");
                Ok(tokenizer.encode(&prompt, EncodeOptions { add_special: true, parse_special: true }))
            } else {
                let prompt = format!("system\n{system}\nuser\n{payload}\nassistant\n");
                let mut ids = tokenizer.encode(&prompt, EncodeOptions { add_special: false, parse_special: true });
                if let Some(bos) = tokenizer.bos_id() {
                    ids.insert(0, bos);
                }
                Ok(ids)
            }
        }
        "gemma4" => {
            let prompt = format!("{system}\n\n{payload}\n\n<turn|>\n<|turn>model\n");
            let bos = tokenizer.bos_id().ok_or("Gemma4 tokenizer missing BOS")?;
            let mut ids = tokenizer.encode(&prompt, EncodeOptions { add_special: false, parse_special: true });
            if ids.first() != Some(&bos) {
                ids.insert(0, bos);
            }
            Ok(ids)
        }
        "lfm2" | "lfm25" => {
            let mut token_ids = Vec::new();
            if let Some(bos) = tokenizer.bos_id() {
                token_ids.push(bos);
            }
            token_ids.extend(tokenizer.encode(
                &format!("system\n{system}\n"),
                EncodeOptions { add_special: false, parse_special: false },
            ));
            token_ids.extend(tokenizer.encode(
                &format!("user\n{payload}\n"),
                EncodeOptions { add_special: false, parse_special: false },
            ));
            token_ids.extend(tokenizer.encode(
                "assistant\n",
                EncodeOptions { add_special: false, parse_special: false },
            ));
            Ok(token_ids)
        }
        "spark2_5" => {
            let sos = "<｜start▁of▁sentence｜>";
            let eos = "<｜end▁of▁sentence｜>";
            let prompt = format!(
                "{sos}<|System|>\n{system}{eos}\
                 {sos}<|User|>{payload}{eos}\
                 {sos}<|Bot|>\n"
            );
            let mut token_ids = tokenizer.encode(&prompt, EncodeOptions { add_special: false, parse_special: true });
            if tokenizer.add_bos() {
                if let Some(bos) = tokenizer.bos_id() {
                    token_ids.insert(0, bos);
                }
            }
            Ok(token_ids)
        }
        "nemotron_h" => {
            let prompt = format!("{system}\n\n{payload}\n\nAnswer:");
            Ok(tokenizer.encode(&prompt, EncodeOptions { add_special: true, parse_special: true }))
        }
        other => Err(format!(
            "--jev grouped is not yet supported for architecture {:?}; \
             currently supported: qwen3 / qwen3vl / qwen35 / llama / k2-horizon / \
             granite / nanbeige / qwen2_2 / gemma4 / lfm2 / lfm25 / spark2_5 / \
             hunyuan-dense / nemotron_h",
            other
        )),
    }
}

fn compute_grouped_jev_result(
    q: &PreparedGroupedQuestion,
    tokenizer: &BPETokenizer,
    group_labels: &[Vec<char>],
    logits: &[f32],
    prefill_ms: u128,
) -> JevGroupedResult {
    let mut groups = Vec::with_capacity(q.groups.len());
    for (gi, group) in q.groups.iter().enumerate() {
        let labels = &group_labels[gi];
        let label_ids: Vec<u32> = labels
            .iter()
            .map(|c| {
                let s = c.to_string();
                tokenizer
                    .encode(&s, EncodeOptions { add_special: false, parse_special: false })
                    .into_iter()
                    .next()
                    .unwrap_or(0)
            })
            .collect();
        let group_logits: Vec<f32> = label_ids.iter().map(|&id| logits[id as usize]).collect();
        let max_logit = group_logits.iter().fold(f32::NEG_INFINITY, |a, &b| f32::max(a, b));
        let mut exps: Vec<f32> = group_logits.iter().map(|&z| (z - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for v in exps.iter_mut() {
            *v /= sum;
        }
        let chosen_idx = exps
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let confidence = exps[chosen_idx];
        let entropy: f32 = -exps.iter().filter(|p| **p > 0.0).map(|p| p * p.ln()).sum::<f32>();
        let mut sorted = exps.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let margin = if sorted.len() >= 2 {
            sorted[0] - sorted[1]
        } else {
            sorted[0]
        };
        let has_values = group.values.iter().any(|v| *v != 0.0);
        let score = if has_values {
            Some(exps.iter().zip(group.values.iter()).map(|(p, v)| p * v).sum())
        } else {
            None
        };
        groups.push(JevGroupResult {
            label: group.label.clone(),
            labels: labels.to_vec(),
            descriptions: group.descriptions.clone(),
            values: group.values.clone(),
            probabilities: exps,
            choice_label: labels[chosen_idx],
            score,
            confidence,
            entropy,
            margin,
        });
    }
    JevGroupedResult {
        mode: q.mode,
        question: q.text.clone(),
        groups,
        prefill_ms,
    }
}

pub fn run_jev_grouped_decision(
    source: Arc<dyn TensorSource>,
    context: &str,
    questions: &[JevGroupedQuestionInput],
    mode: JevMode,
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<(), String> {
    let prepared = prepare_jev_grouped_questions(questions, mode)?;
    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    eprintln!("JEV grouped: arch = {:?}, mode = {:?}", arch, mode);

    let t0 = Instant::now();
    let results = match &*arch {
        "qwen3" | "qwen3vl" => run_jev_grouped_qwen3(
            source.clone(), context, &prepared, n_threads_arg, prefill_batch_size, output_json,
        )?,
        "qwen35" => run_jev_grouped_qwen35(
            source.clone(), context, &prepared, n_threads_arg, prefill_batch_size, output_json,
        )?,
        "llama" | "k2-horizon" | "granite" | "nanbeige" | "qwen2_2" => run_jev_grouped_llama(
            source.clone(), context, &prepared, n_threads_arg, prefill_batch_size, output_json,
        )?,
        "gemma4" => run_jev_grouped_gemma4(
            source.clone(), context, &prepared, n_threads_arg, prefill_batch_size, output_json,
        )?,
        "lfm2" => run_jev_grouped_lfm2(
            source.clone(), context, &prepared, n_threads_arg, prefill_batch_size, output_json,
        )?,
        "lfm25" => run_jev_grouped_lfm25(
            source.clone(), context, &prepared, n_threads_arg, prefill_batch_size, output_json,
        )?,
        "spark2_5" => run_jev_grouped_spark(
            source.clone(), context, &prepared, n_threads_arg, prefill_batch_size, output_json,
        )?,
        "nemotron_h" => run_jev_grouped_nemotron_h(
            source.clone(), context, &prepared, n_threads_arg, prefill_batch_size, output_json,
        )?,
        "hunyuan-dense" => run_jev_grouped_hunyuan(
            source.clone(), context, &prepared, n_threads_arg, prefill_batch_size, output_json,
        )?,
        other => {
            return Err(format!(
                "--jev grouped is not yet supported for architecture {:?}; \
                 currently supported: qwen3 / qwen3vl / qwen35 / llama / k2-horizon / \
                 granite / nanbeige / qwen2_2 / gemma4 / lfm2 / lfm25 / spark2_5 / \
                 hunyuan-dense / nemotron_h",
                other
            ));
        }
    };

    if output_json {
        for r in &results {
            let line = serde_json::to_string(r).map_err(|e| format!("json encode: {e}"))?;
            println!("{}", line);
        }
    } else if results.len() == 1 {
        print_grouped_result_text(&results[0]);
    } else {
        println!("\n--- JEV grouped decisions ({} questions) ---", results.len());
        for r in &results {
            println!("\nQ: {}", r.question);
            print_grouped_result_text(r);
        }
    }

    let total_ms = t0.elapsed().as_millis();
    eprintln!("\nJEV grouped total: {} ms ({} questions)", total_ms, results.len());
    Ok(())
}

fn print_grouped_result_text(r: &JevGroupedResult) {
    println!("\n--- JEV decision ({:?}) ---", r.mode);
    for g in &r.groups {
        if let Some(score) = g.score {
            println!("  [{}] score: {:.4}", g.label, score);
            for (i, p) in g.probabilities.iter().enumerate() {
                println!(
                    "    {}: {:.4} × {} = {:.4} — {}",
                    g.labels[i], p, g.values[i], p * g.values[i], g.descriptions[i]
                );
            }
        } else {
            println!("  [{}] choice: {}", g.label, g.choice_label);
            for (i, p) in g.probabilities.iter().enumerate() {
                println!("    {}: {:.4} — {}", g.labels[i], p, g.descriptions[i]);
            }
        }
        println!(
            "    confidence: {:.4} | entropy: {:.4} | margin: {:.4}",
            g.confidence, g.entropy, g.margin
        );
    }
    println!("prefill: {} ms", r.prefill_ms);
}

impl serde::Serialize for JevGroupedResult {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("JevGroupedResult", 4)?;
        st.serialize_field(
            "mode",
            match self.mode {
                JevMode::MultiSelect => "multi_select",
                JevMode::BlockChoice => "block_choice",
                _ => "unknown",
            },
        )?;
        st.serialize_field("question", &self.question)?;
        let groups: Vec<serde_json::Value> = self
            .groups
            .iter()
            .map(|g| {
                let probs: serde_json::Map<String, serde_json::Value> = g
                    .labels
                    .iter()
                    .zip(g.probabilities.iter())
                    .map(|(l, p)| (l.to_string(), serde_json::json!(p)))
                    .collect();
                let mut obj = serde_json::json!({
                    "label": g.label,
                    "choice": g.choice_label.to_string(),
                    "probabilities": probs,
                    "confidence": g.confidence,
                    "entropy": g.entropy,
                    "margin": g.margin,
                });
                if let Some(score) = g.score {
                    obj.as_object_mut().unwrap().insert("score".to_string(), serde_json::json!(score));
                }
                obj
            })
            .collect();
        st.serialize_field("groups", &groups)?;
        st.serialize_field("prefill_ms", &self.prefill_ms)?;
        st.end()
    }
}

fn run_jev_grouped_qwen3(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevGroupedResult>, String> {
    let available_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = Qwen3JevGroupedScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads", scorer.pool().n_threads());
    }
    run_jev_grouped_core(source, context, per_question, output_json, &mut scorer)
}

/// Qwen3 JEV grouped scorer — wraps [`Qwen3JevScorer`] for the
/// Qwen3Model + Session API.
struct Qwen3JevGroupedScorer {
    inner: Qwen3JevScorer,
}

impl Qwen3JevGroupedScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: Qwen3JevScorer::new(source, n_threads, prefill_batch_size)?,
        })
    }

    fn pool(&self) -> Arc<ComputePool> {
        self.inner.pool()
    }
}

impl JevGroupedScorer for Qwen3JevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "Qwen3"
    }

    fn build_grouped_prompt(
        &self,
        context: &str,
        q: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String> {
        let group_labels = allocate_group_labels(q);
        let system = build_grouped_system();
        let payload = build_grouped_payload(context, q)?;
        let token_ids = build_jev_token_ids_for_arch(
            "qwen3",
            self.inner.model.tokenizer(),
            system,
            &payload,
        )?;
        Ok((group_labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        // Qwen3JevScorer::forward_logits already runs the session,
        // computes positions, and returns (logits, dur). Delegate
        // rather than re-deriving positions / input.
        self.inner
            .forward_logits(token_ids)
            .map_err(|e| format!("Qwen3 grouped forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}

fn run_jev_grouped_qwen35(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevGroupedResult>, String> {
    // Qwen3.5's `Qwen35Model<'a>` borrows from the source. Keeping
    // it inside a lifetime-parameterized scorer would complicate
    // the trait-object story (same constraint as
    // `run_jev_decision_qwen35`); for now the per-question loop
    // stays inline here.
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
    verify_label_tokens_single(&tokenizer)?;
    let available_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let pool = Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());
    let mut model = crate::models::qwen35::Qwen35Model::from_source(source.as_ref())
        .map_err(|error| format!("Failed to parse Qwen3.5 model: {error}"))?;
    let n_ctx = model.config.n_ctx;
    let mut results = Vec::with_capacity(per_question.len());
    for q in per_question {
        let group_labels = allocate_group_labels(q);
        let system = build_grouped_system();
        let payload = build_grouped_payload(context, q)?;
        let token_ids = build_jev_token_ids_for_arch("qwen35", &tokenizer, system, &payload)?;
        let (positions, _next) = build_qwen35_positions(&token_ids, None, &[])
            .map_err(|e| format!("Failed to build Qwen3.5 positions: {e}"))?;
        let mut session = crate::models::qwen35::Qwen35Session::new_with_prefill_batch_size(
            &mut model, n_ctx.min(token_ids.len() + 1), prefill_batch_size, pool.clone(),
        )?;
        let t0 = Instant::now();
        let logits = session.forward_logits(&token_ids, &positions)
            .map_err(|e| format!("Qwen3.5 forward_logits failed: {e}"))?;
        let prefill_dur = t0.elapsed();
        results.push(compute_grouped_jev_result(q, &tokenizer, &group_labels, &logits, prefill_dur.as_millis()));
    }
    Ok(results)
}

fn run_jev_grouped_llama(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevGroupedResult>, String> {
    let available_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = LlamaJevGroupedScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Llama-family)", n_threads);
    }
    run_jev_grouped_core(source, context, per_question, output_json, &mut scorer)
}

/// Llama-family JEV grouped scorer — wraps [`LlamaJevScorer`]
/// for the per-arch chat template + free-function forward path.
struct LlamaJevGroupedScorer {
    inner: LlamaJevScorer,
}

impl LlamaJevGroupedScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: LlamaJevScorer::new(source, n_threads, prefill_batch_size)?,
        })
    }
}

impl JevGroupedScorer for LlamaJevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "Llama-family"
    }

    fn build_grouped_prompt(
        &self,
        context: &str,
        q: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String> {
        let group_labels = allocate_group_labels(q);
        let system = build_grouped_system();
        let payload = build_grouped_payload(context, q)?;
        let token_ids = build_jev_token_ids_for_arch(
            &self.inner.arch,
            self.inner.tokenizer(),
            system,
            &payload,
        )?;
        Ok((group_labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::llama::trunk::run_forward_logits_llama_with_batch(
            self.inner.source.as_ref(),
            &token_ids,
            self.inner.n_threads,
            KvFormat::F16,
            8192,
            self.inner.prefill_batch_size,
        )
        .map_err(|e| format!("Llama forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}

fn run_jev_grouped_gemma4(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevGroupedResult>, String> {
    let available_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = Gemma4JevGroupedScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Gemma4)", n_threads);
    }
    run_jev_grouped_core(source, context, per_question, output_json, &mut scorer)
}

/// Gemma4 JEV grouped scorer — wraps [`Gemma4JevScorer`] for
/// the session API + tokenizer access.
struct Gemma4JevGroupedScorer {
    inner: Gemma4JevScorer,
}

impl Gemma4JevGroupedScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: Gemma4JevScorer::new(source, n_threads, prefill_batch_size)?,
        })
    }
}

impl JevGroupedScorer for Gemma4JevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "Gemma4"
    }

    fn build_grouped_prompt(
        &self,
        context: &str,
        q: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String> {
        let group_labels = allocate_group_labels(q);
        let system = build_grouped_system();
        let payload = build_grouped_payload(context, q)?;
        let token_ids = build_jev_token_ids_for_arch(
            "gemma4",
            self.inner.tokenizer(),
            system,
            &payload,
        )?;
        Ok((group_labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        let mut session = crate::models::gemma4::Gemma4Session::new_with_prefill_batch_size(
            &self.inner.model,
            KvFormat::F16,
            self.inner.prefill_batch_size,
        )?;
        let t0 = Instant::now();
        let logits = session
            .forward_logits(&token_ids)
            .map_err(|e| format!("Gemma4 forward_logits failed: {e}"))?;
        Ok((logits, t0.elapsed()))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}

fn run_jev_grouped_lfm2(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevGroupedResult>, String> {
    let available_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = Lfm2JevGroupedScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (LFM2)", n_threads);
    }
    run_jev_grouped_core(source, context, per_question, output_json, &mut scorer)
}

/// LFM2 JEV grouped scorer — uses the free-function prefill
/// path (`run_forward_logits_lfm2_with_batch`). Chat template
/// dispatched through [`build_jev_token_ids_for_arch`] (the
/// `"lfm2"` arm).
struct Lfm2JevGroupedScorer {
    inner: Lfm2JevScorer,
}

impl Lfm2JevGroupedScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: Lfm2JevScorer::new(source, n_threads, prefill_batch_size)?,
        })
    }
}

impl JevGroupedScorer for Lfm2JevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "LFM2"
    }

    fn build_grouped_prompt(
        &self,
        context: &str,
        q: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String> {
        let group_labels = allocate_group_labels(q);
        let system = build_grouped_system();
        let payload = build_grouped_payload(context, q)?;
        let token_ids = build_jev_token_ids_for_arch(
            "lfm2",
            self.inner.tokenizer(),
            system,
            &payload,
        )?;
        Ok((group_labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::lfm2::run_forward_logits_lfm2_with_batch(
            self.inner.source.as_ref(),
            &token_ids,
            self.inner.n_threads,
            KvFormat::F16,
            8192,
            self.inner.prefill_batch_size,
        )
        .map_err(|e| format!("LFM2 forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}

fn run_jev_grouped_lfm25(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevGroupedResult>, String> {
    let available_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = Lfm25JevGroupedScorer::new(source.clone(), n_threads, prefill_batch_size)?;
    if !output_json {
        eprintln!("compute pool: {} threads (LFM2.5)", n_threads);
    }
    run_jev_grouped_core(source, context, per_question, output_json, &mut scorer)
}

/// LFM2.5 JEV grouped scorer — wraps [`Lfm25JevScorer`] for
/// the chat template / tokenizer + free-function forward path.
struct Lfm25JevGroupedScorer {
    inner: Lfm25JevScorer,
}

impl Lfm25JevGroupedScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: Lfm25JevScorer::new(source, n_threads, prefill_batch_size)?,
        })
    }
}

impl JevGroupedScorer for Lfm25JevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "LFM2.5"
    }

    fn build_grouped_prompt(
        &self,
        context: &str,
        q: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String> {
        let group_labels = allocate_group_labels(q);
        let system = build_grouped_system();
        let payload = build_grouped_payload(context, q)?;
        let token_ids = build_jev_token_ids_for_arch(
            "lfm25",
            self.inner.tokenizer(),
            system,
            &payload,
        )?;
        Ok((group_labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::lfm25::run_forward_logits_lfm25_with_batch(
            self.inner.inner.source.as_ref(),
            &token_ids,
            self.inner.inner.n_threads,
            KvFormat::F16,
            8192,
            self.inner.inner.prefill_batch_size,
        )
        .map_err(|e| format!("LFM2.5 forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}

fn run_jev_grouped_spark(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    n_threads_arg: usize,
    _prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevGroupedResult>, String> {
    let available_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);
    let mut scorer = SparkJevGroupedScorer::new(source.clone(), n_threads)?;
    if !output_json {
        eprintln!("compute pool: {} threads (Spark)", n_threads);
    }
    run_jev_grouped_core(source, context, per_question, output_json, &mut scorer)
}

/// Spark JEV grouped scorer — wraps [`SparkJevScorer`] for the
/// session API + tokenizer access.
struct SparkJevGroupedScorer {
    inner: SparkJevScorer,
}

impl SparkJevGroupedScorer {
    fn new(source: Arc<dyn TensorSource>, n_threads: usize) -> Result<Self, String> {
        Ok(Self {
            inner: SparkJevScorer::new(source, n_threads)?,
        })
    }
}

impl JevGroupedScorer for SparkJevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "Spark"
    }

    fn build_grouped_prompt(
        &self,
        context: &str,
        q: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String> {
        let group_labels = allocate_group_labels(q);
        let system = build_grouped_system();
        let payload = build_grouped_payload(context, q)?;
        let token_ids = build_jev_token_ids_for_arch(
            "spark2_5",
            self.inner.tokenizer(),
            system,
            &payload,
        )?;
        Ok((group_labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        let t0 = Instant::now();
        let logits = self
            .inner
            .session
            .forward_logits(&token_ids)
            .map_err(|e| format!("Spark forward_logits failed: {e}"))?;
        Ok((logits, t0.elapsed()))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}

fn run_jev_grouped_nemotron_h(
    source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    _n_threads_arg: usize,
    _prefill_batch_size: usize,
    output_json: bool,
) -> Result<Vec<JevGroupedResult>, String> {
    let mut scorer = NemotronHJevGroupedScorer::new(source.clone())?;
    let _ = output_json;
    run_jev_grouped_core(source, context, per_question, false, &mut scorer)
}

/// Nemotron-H JEV grouped scorer — wraps [`NemotronHJevScorer`]
/// (base model, free-function forward path).
struct NemotronHJevGroupedScorer {
    inner: NemotronHJevScorer,
}

impl NemotronHJevGroupedScorer {
    fn new(source: Arc<dyn TensorSource>) -> Result<Self, String> {
        Ok(Self {
            inner: NemotronHJevScorer::new(source)?,
        })
    }
}

impl JevGroupedScorer for NemotronHJevGroupedScorer {
    fn scorer_label(&self) -> &'static str {
        "Nemotron-H"
    }

    fn build_grouped_prompt(
        &self,
        context: &str,
        q: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String> {
        let group_labels = allocate_group_labels(q);
        let system = build_grouped_system();
        let payload = build_grouped_payload(context, q)?;
        let token_ids = build_jev_token_ids_for_arch(
            "nemotron_h",
            self.inner.tokenizer(),
            system,
            &payload,
        )?;
        Ok((group_labels, token_ids))
    }

    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        crate::models::nemotron_h::trunk::run_forward_logits_nemotron_h(
            self.inner.source.clone(),
            &token_ids,
        )
        .map_err(|e| format!("Nemotron-H forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}

fn run_jev_grouped_hunyuan(
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

/// Hunyuan JEV grouped scorer — wraps [`HunyuanJevScorer`] (uses
/// Qwen3Model under the hood + Session API).
struct HunyuanJevGroupedScorer {
    inner: HunyuanJevScorer,
}

impl HunyuanJevGroupedScorer {
    fn new(
        source: Arc<dyn TensorSource>,
        n_threads: usize,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            inner: HunyuanJevScorer::new(source, n_threads, prefill_batch_size)?,
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
        // HunyuanJevScorer::forward_logits runs the session for us.
        self.inner
            .forward_logits(token_ids)
            .map_err(|e| format!("Hunyuan grouped forward_logits failed: {e}"))
    }

    fn tokenizer(&self) -> &BPETokenizer {
        self.inner.tokenizer()
    }
}

