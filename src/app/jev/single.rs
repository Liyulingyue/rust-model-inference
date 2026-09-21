use super::types::{
    JevGroupInput, JevGroupResult, JevGroupedOption, JevGroupedQuestionInput, JevGroupedResult,
    JevMode, JevQuestionInput, JevResult,
};
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::gemma4::{Gemma4Model, Gemma4Session};
use crate::models::lfm2::trunk::forward::run_forward_logits_lfm2_with_batch;
use crate::models::lfm25::trunk::forward::run_forward_logits_lfm25_with_batch;
use crate::models::lfm2moe::trunk::forward::run_forward_logits_lfm2moe_with_batch;
use crate::models::llama::trunk::forward::run_forward_logits_llama_with_batch;
use crate::models::qwen3::{Qwen3Input, Qwen3Model, Qwen3Session};
use crate::models::qwen35::{build_qwen35_positions, Qwen35Model, Qwen35Session};
use crate::models::spark::SparkSession;
use crate::prompt::{
    append_qwen_assistant_prefix, append_qwen_message_tokens, build_hunyuan_chat_prompt,
    HunyuanMessage,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
        "qwen3" | "qwen3vl" => qwen3::run_jev_decision_qwen3(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "qwen35" => qwen35::run_jev_decision_qwen35(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "llama" | "k2-horizon" | "granite" | "nanbeige" | "qwen2_2" => {
            llama::run_jev_decision_llama(
                source.clone(),
                context,
                &prepared,
                n_threads_arg,
                prefill_batch_size,
                output_json,
            )?
        }
        "gemma4" => gemma4::run_jev_decision_gemma4(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "lfm2" => lfm2::run_jev_decision_lfm2(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "spark2_5" => spark::run_jev_decision_spark(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "lfm25" => lfm25::run_jev_decision_lfm25(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "lfm2moe" => lfm2moe::run_jev_decision_lfm2moe(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "nemotron_h" => nemotron_h::run_jev_decision_nemotron_h(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            output_json,
        )?,
        "hunyuan-dense" => hunyuan::run_jev_decision_hunyuan(
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

pub(crate) mod gemma4;
pub(crate) mod hunyuan;
pub(crate) mod lfm2;
pub(crate) mod lfm25;
pub(crate) mod lfm2moe;
pub(crate) mod llama;
pub(crate) mod nemotron_h;
pub(crate) mod qwen3;
pub(crate) mod qwen35;
pub(crate) mod spark;

pub(crate) fn verify_label_tokens_single(tokenizer: &BPETokenizer) -> Result<(), String> {
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

pub(crate) fn jev_system_prompt(mode: JevMode) -> &'static str {
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
pub(crate) fn jev_labels(q: &PreparedQuestion) -> Vec<char> {
    (b'A'..=(b'A' + q.descriptions.len() as u8 - 1))
        .map(|b| b as char)
        .collect()
}

/// Renders the JEV JSON payload (context + question + candidates)
/// for embedding in the per-arch chat template. Used by every
/// per-arch `run_jev_decision_*` so the JSON shape stays in sync.
pub(crate) fn jev_payload_json(context: &str, q: &PreparedQuestion) -> Result<String, String> {
    let labels = jev_labels(q);
    let mut payload = String::from("{\"context\": ");
    payload.push_str(&serde_json::to_string(context).map_err(|e| format!("context json: {e}"))?);
    payload.push_str(", \"question\": ");
    payload.push_str(&serde_json::to_string(&q.text).map_err(|e| format!("question json: {e}"))?);
    payload.push_str(", \"candidates\": {");
    for (i, (label_char, desc)) in labels.iter().zip(q.descriptions.iter()).enumerate() {
        if i > 0 {
            payload.push(',');
        }
        payload.push('"');
        payload.push(*label_char);
        payload.push_str("\": ");
        payload.push_str(&serde_json::to_string(desc).map_err(|e| format!("desc json: {e}"))?);
    }
    payload.push_str("}}");
    Ok(payload)
}

pub(crate) fn build_jev_prompt(
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

pub(crate) fn print_jev_question(q: &PreparedQuestion, labels: &[char]) {
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

pub(crate) fn compute_jev_result(
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

pub(crate) struct PreparedQuestion {
    pub(crate) mode: JevMode,
    pub(crate) text: String,
    pub(crate) descriptions: Vec<String>,
    pub(crate) values: Vec<f32>,
    pub(crate) positive_label: Option<char>,
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
pub(crate) trait JevScorer {
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
pub(crate) fn run_jev_decision_core<S: JevScorer>(
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
