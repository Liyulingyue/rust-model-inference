//! GLiNER2.5-Decide as a JEV-family decision mode.
//!
//! GLiNER2 is not a generative scorer: one DeBERTa-v3 pass over
//! `schema prompt + [SEP_TEXT] + text` produces a logit per label from the
//! hidden states at the `[L]` markers. So it produces the same `JevResult`
//! shape the logit-based scorers emit — one entry per task, labels `A`, `B`,
//! … in declaration order — and the CLI/server output does not fork.
//!
//! Two differences from the token-logit scorers are load-bearing:
//!
//! - `values` are real classifier logits, not A/B/C token logits.
//! - `probabilities` are the *task's* probabilities: softmax for a
//!   single-label head, sigmoid for a multi-label one. A multi-label head
//!   therefore has no single `choice_label` winner, so `choice_label` reports
//!   the argmax (what the reference falls back to when nothing clears the
//!   threshold) and `margin` is still the top-two gap.

use std::sync::Arc;

use super::{JevQuestionInput, JevMode, JevResult};
use crate::app::cli::CliOptions;
use crate::core::sentencepiece::SentencePieceTokenizer;
use crate::core::tensor::TensorSource;
use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::models::gliner::prompt::Task;
use crate::models::gliner::GlinerModel;

/// Resolve the task mapping for a `--jev --gliner2-decide` invocation.
///
/// `--gliner2-schema` wins and is taken verbatim, so any `classify_text` call
/// from the reference README runs unchanged. Without it, each `--jev-question`
/// becomes one task: the question text is the head name and the `--jev-option`
/// list is its label set. That keeps the plain `--jev-option` shell usable and
/// makes "several decisions at once" fall out of the existing flags.
pub fn gliner2_schema(
    options: &CliOptions,
    questions: &[JevQuestionInput],
) -> Result<serde_json::Value, String> {
    if let Some(raw) = &options.gliner2_schema {
        return serde_json::from_str(raw)
            .map_err(|error| format!("--gliner2-schema is not valid JSON: {error}"));
    }
    schema_from_questions(questions)
}

/// One task per `--jev-question`: the question text is the head name and the
/// options are its label set. This is the HTTP shape too, so a caller uses the
/// same request body whichever front end it talks to.
pub fn schema_from_questions(
    questions: &[JevQuestionInput],
) -> Result<serde_json::Value, String> {
    let tasks: Vec<LabelSet> = questions
        .iter()
        .map(|question| LabelSet {
                name: question.text.clone(),
                labels: question.options.clone(),
                descriptions: None,
            multi_label: false,
            cls_threshold: None,
            prompt: None,
        })
        .collect();
    schema_from_label_sets(&tasks)
}

/// One task's label set, in the shape the HTTP handler reads off the request.
pub struct LabelSet {
    pub name: String,
    pub labels: Vec<String>,
    pub descriptions: Option<Vec<String>>,
    pub multi_label: bool,
    pub cls_threshold: Option<f64>,
    pub prompt: Option<String>,
}

pub fn schema_from_label_sets(tasks: &[LabelSet]) -> Result<serde_json::Value, String> {
    if tasks.is_empty() {
        return Err("expected at least one question with options".to_string());
    }
    let mut schema = serde_json::Map::new();
    for task in tasks {
        let name = task.name.trim();
        if name.is_empty() {
            return Err("each question must name its task".into());
        }
        if task.labels.is_empty() {
            return Err(format!("task {name:?} has no labels"));
        }
        let mut entry = serde_json::Map::new();
        match &task.descriptions {
            Some(descriptions) => {
                if descriptions.len() != task.labels.len() {
                    return Err(format!(
                        "task {name:?}: {} labels but {} descriptions",
                        task.labels.len(),
                        descriptions.len()
                    ));
                }
                let mut labelled = serde_json::Map::new();
                for (label, description) in task.labels.iter().zip(descriptions) {
                    labelled.insert(label.clone(), serde_json::Value::String(description.clone()));
                }
                entry.insert("labels".into(), serde_json::Value::Object(labelled));
            }
            None => {
                entry.insert(
                    "labels".into(),
                    serde_json::Value::Array(
                        task.labels
                            .iter()
                            .map(|label| serde_json::Value::String(label.clone()))
                            .collect(),
                    ),
                );
            }
        }
        if task.multi_label {
            entry.insert("multi_label".into(), serde_json::Value::Bool(true));
        }
        if let Some(threshold) = task.cls_threshold {
            entry.insert(
                "cls_threshold".into(),
                serde_json::Value::Number(serde_json::Number::from_f64(threshold).ok_or_else(
                    || format!("task {name:?}: cls_threshold is not a finite number"),
                )?),
            );
        }
        if let Some(prompt) = &task.prompt {
            entry.insert("prompt".into(), serde_json::Value::String(prompt.clone()));
        }
        schema.insert(name.to_string(), serde_json::Value::Object(entry));
    }
    Ok(serde_json::Value::Object(schema))
}

/// Parse a `classify_text`-shaped task mapping.
pub fn parse_schema(schema: &serde_json::Value) -> Result<Vec<Task>, String> {
    let object = schema
        .as_object()
        .ok_or("gliner2 schema must be a JSON object of {task: labels}")?;
    if object.is_empty() {
        return Err("gliner2 schema must declare at least one task".into());
    }
    object
        .iter()
        .map(|(name, value)| Task::from_json(name, value))
        .collect()
}

/// Labels are `A`, `B`, ... just like the logit-based JEV scorers use.
fn labels(count: usize) -> Vec<char> {
    (0..count).map(|index| (b'A' + index as u8) as char).collect()
}

/// Score with an already-loaded model. The server keeps it alive across
/// requests, so it uses this instead of [`run_gliner2_decision`].
pub fn run_gliner2_scoring(
    model: &GlinerModel<'_>,
    tasks: &[Task],
    context: &str,
    n_threads_arg: usize,
) -> Result<Vec<JevResult>, String> {
    let started = std::time::Instant::now();
    let results = model.classify_text(context, tasks, n_threads_arg)?;
    let prefill_ms = started.elapsed().as_millis();
    let out: Vec<JevResult> = results
        .iter()
        .map(|result| {
            let probabilities: Vec<f32> =
                result.scores.iter().map(|score| score.probability).collect();
            let values: Vec<f32> = result.scores.iter().map(|score| score.logit).collect();
            let labels = labels(result.scores.len());
            let descriptions: Vec<String> =
                result.scores.iter().map(|score| score.label.clone()).collect();
            let best = probabilities
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index)
                .unwrap_or(0);
            let choice_label = labels.get(best).copied();

            let confidence = probabilities[best]
                - probabilities.iter().sum::<f32>() / probabilities.len() as f32;
            let entropy = -probabilities
                .iter()
                .filter(|&&value| value > 0.0)
                .map(|&value| value * value.ln())
                .sum::<f32>();
            let mut sorted = probabilities.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let margin = sorted.first().copied().unwrap_or(0.0)
                - sorted.get(1).copied().unwrap_or(0.0);
            JevResult {
                mode: if result.multi_label { JevMode::MultiSelect } else { JevMode::Choice },
                question: result.task.clone(),
                labels,
                descriptions,
                values,
                probabilities,
                choice_label,
                positive_label: None,
                probability_positive: None,
                score: None,
                confidence,
                entropy,
                margin,
                prefill_ms,
                selected: result.selected.clone(),
            }
        })
        .collect();
    Ok(out)
}

/// Load the GGUF, then delegate to [`run_gliner2_scoring`]. CLI only.
pub fn run_gliner2_decision(
    source: Arc<dyn TensorSource>,
    tasks: &[Task],
    context: &str,
    n_threads_arg: usize,
    output_json: bool,
) -> Result<(), String> {
    let started = std::time::Instant::now();
    let model = GlinerModel::from_source(source.as_ref())?;
    eprintln!(
        "GLiNER2: deberta-v3 {}x{}x{}, {} task(s)",
        model.config().n_layer,
        model.config().n_embd,
        model.config().n_head,
        tasks.len()
    );
    let results = run_gliner2_scoring(&model, tasks, context, n_threads_arg)?;
    if output_json {
        for result in &results {
            let line = serde_json::to_string(result).map_err(|e| format!("json encode: {e}"))?;
            println!("{line}");
        }
        return Ok(());
    }
    for result in &results {
        println!("Q: {}", result.question);
        for (label, (name, (probability, value))) in result.labels.iter().zip(
            result
                .descriptions
                .iter()
                .zip(result.probabilities.iter().zip(result.values.iter())),
        ) {
            println!("  {label}. {name}  p={probability:.4}  score={value:.4}");
        }
        if let Some(choice) = result.choice_label {
            println!("  -> choice: {choice}");
        }
        println!("  ({:.0} ms)", started.elapsed().as_millis());
    }
    Ok(())
}

/// Open a GLiNER2 GGUF and build its tokenizer. Server startup only: the model
/// itself is a set of zero-copy views, so it is cheap to rebuild per request
/// once the tokenizer is cached.
pub fn load_gliner2_source(
    path: &std::path::Path,
) -> Result<(Box<dyn TensorSource>, SentencePieceTokenizer), String> {
    let source = open_model_source(path, ComponentRole::Llm)
        .map_err(|e| format!("open gliner2 model ({}): {e}", path.display()))?;
    let spm = crate::models::gliner::load_spm(source.as_ref())?;
    Ok((source, spm))
}
