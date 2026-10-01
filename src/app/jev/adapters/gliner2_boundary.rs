//! GLiNER2.5 BoundaryExtractor as a CLI mode.
//!
//! The Decide adapter in `gliner2.rs` produces one logit per label because the
//! Decide head is a classifier over marker embeddings. The boundary variant is
//! a *span extractor*: the same schema declares fields, and the model returns
//! `(start, end)` word spans per field with a confidence. So the output shape
//! differs and is not forced into `JevResult`.
//!
//! The flag surface is deliberately shared with Decide — `--jev --model
//! --jev-context --gliner2-schema` — so the only new things a caller has to
//! learn are the flag name and the schema shape.

use std::sync::Arc;

use crate::core::tensor::TensorSource;
use crate::models::gliner::prompt::{BoundaryTaskKind, Label, Task};
use crate::models::gliner_boundary::extract::{
    apply_abstention, boundary_overlap_policy, decode_spans, run_mixed_extraction,
    ClassificationResult, ExtractedSpan, Extraction,
};
use crate::models::gliner_boundary::BoundaryModel;

/// One extracted span, as emitted by `--jev-output-json`.
#[derive(Debug, serde::Serialize)]
struct SpanJson {
    field: String,
    score: f32,
    start: usize,
    end: usize,
    text: String,
}

/// One classification group, as emitted by `--jev-output-json`.
#[derive(Debug, serde::Serialize)]
struct ClassJson {
    task: String,
    activation: &'static str,
    labels: Vec<String>,
    probabilities: Vec<f32>,
    selected: Vec<String>,
    choice_label: Option<String>,
}

/// The whole JSON payload for `--jev-output json`.
#[derive(Debug, serde::Serialize)]
struct BoundaryJson {
    spans: Vec<SpanJson>,
    classifications: Vec<ClassJson>,
}

/// Parse the reference's extractive schema shape.
///
/// The reference's `_process_entities` reads the *keys* of `schema["entities"]`
/// as the field names and `schema["entity_descriptions"]` for their
/// descriptions, so a faithful request is:
///
/// ```json
/// {"entities": ["person", "organization"],
///  "entity_descriptions": {"person": "an individual human being"}}
/// ```
///
/// The dict form `{"entities": {"person": [...]}}` — which is what the
/// reference's own training data uses, where the values are gold spans — is
/// accepted too and its keys are taken, in insertion order. Field order is the
/// contract: it fixes the query order, so it is preserved rather than sorted.
///
/// The classification parser in `gliner2.rs` cannot be reused: it would read
/// `entity_descriptions` as a second task, silently producing twice as many
/// queries, half of them descriptions.
/// The groups a boundary schema declares, in the order the reference emits them.
///
/// `_transform_record` calls `_process_json_structures`, `_process_entities`,
/// `_process_relations`, then `_process_classifications`
/// (`processor.py:893-904`), so the groups are laid out in that order. Order is
/// part of the contract: it fixes which marker index each field ends up at.
pub fn parse_boundary_schema(
    schema: &serde_json::Value,
) -> Result<(Vec<Task>, Vec<BoundaryTaskKind>), String> {
    let object = schema
        .as_object()
        .ok_or("gliner2 boundary schema must be a JSON object")?;
    let mut tasks = Vec::new();
    let mut kinds = Vec::new();
    if let Some(classifications) = object.get("classifications") {
        let items = classifications
            .as_array()
            .ok_or("\"classifications\" must be a list of {task, labels}")?;
        for item in items {
            tasks.push(parse_classification_group(item)?);
            kinds.push(BoundaryTaskKind::Classification);
        }
    }
    if object.contains_key("entities") {
        tasks.push(parse_entities_group(object)?);
        kinds.push(BoundaryTaskKind::Entities);
    }
    if tasks.is_empty() {
        return Err("gliner2 boundary schema needs \"entities\" or \"classifications\"".into());
    }
    Ok((tasks, kinds))
}

fn parse_classification_group(item: &serde_json::Value) -> Result<Task, String> {
    let task_name = item
        .get("task")
        .and_then(|value| value.as_str())
        .ok_or("each classification needs a \"task\" name")?;
    let raw_labels = item
        .get("labels")
        .and_then(|value| value.as_array())
        .ok_or_else(|| format!("classification {task_name:?} needs a \"labels\" array"))?;
    if raw_labels.is_empty() {
        return Err(format!("classification {task_name:?} has no labels"));
    }
    let descriptions = item.get("label_descriptions").and_then(|v| v.as_object());
    let mut labels: Vec<Label> = Vec::with_capacity(raw_labels.len());
    for value in raw_labels {
        {
            let name = value
                .as_str()
                .ok_or_else(|| format!("classification {task_name:?}: labels must be strings"))?;
            let mut label = Label::new(name);
            // Inference uses `example_mode = "both"`, so descriptions are part of
            // the prompt text and shift every later marker index.
            label.description = descriptions
                .and_then(|map| map.get(name))
                .and_then(|value| value.as_str())
                .map(str::to_string);
            labels.push(label);
        }
    }
    let mut task = Task::new(task_name, labels);
    if let Some(prompt) = item.get("prompt").and_then(|value| value.as_str()) {
        task.prompt = Some(prompt.to_string());
    }
    task.multi_label = item
        .get("multi_label")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    if let Some(threshold) = item.get("cls_threshold").and_then(|value| value.as_f64()) {
        task.cls_threshold = threshold as f32;
    }
    Ok(task)
}

#[allow(clippy::needless_pass_by_value)]
fn parse_entities_group(
    object: &serde_json::Map<String, serde_json::Value>,
) -> Result<Task, String> {
    let entities = object
        .get("entities")
        .ok_or("gliner2 boundary schema needs an \"entities\" group")?;
    let fields: Vec<String> = match entities {
        serde_json::Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| "entities[] must hold strings".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?,
        serde_json::Value::Object(map) => map.keys().cloned().collect(),
        _ => return Err("\"entities\" must be an array of names or an object".into()),
    };
    if fields.is_empty() {
        return Err("\"entities\" declares no fields".into());
    }
    let descriptions = object.get("entity_descriptions");
    let labels = fields
        .iter()
        .map(|name| {
            let description = descriptions.and_then(|value| {
                value
                    .as_object()
                    .and_then(|map| map.get(name))
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
            });
            let mut label = Label::new(name.clone());
            label.description = description;
            label
        })
        .collect();
    Ok(Task::new("entities", labels))
}

/// Run one extraction with an already-loaded model, applying abstention.
///
/// The server keeps the model alive across requests and uses this instead of
/// [`run_gliner2_boundary`]. Abstention is applied here rather than left to the
/// caller, because a query whose `null_projection` clears the threshold is
/// emptied wholesale by the reference and a partial application would leak
/// spans it dropped.
pub fn extract(
    model: &BoundaryModel<'_>,
    text: &str,
    tasks: &[Task],
    kinds: &[BoundaryTaskKind],
    n_threads_arg: usize,
    threshold: Option<f32>,
) -> Result<Extraction, String> {
    let mut result = run_mixed_extraction(model, text, tasks, kinds, n_threads_arg)?;
    if !result.query_names.is_empty() {
        result.spans = decode_spans(
            &result.candidates,
            &result.words,
            &result.query_names,
            model.settings.pair_temperature,
            threshold.unwrap_or(0.5),
            Some(boundary_overlap_policy(model)?),
        );
        apply_abstention(
            &mut result.spans,
            &result.query_heads,
            model.settings.abstention_threshold,
        );
    }
    Ok(result)
}

fn print_classifications(results: &[ClassificationResult]) {
    for group in results {
        println!("{}:", group.task);
        for (label, probability) in group.labels.iter().zip(group.probabilities.iter()) {
            let mark = if group.selected.iter().any(|chosen| chosen == label) {
                '*'
            } else {
                ' '
            };
            println!("  {mark} {label}  p={probability:.4}");
        }
        println!("  ({})", group.activation);
    }
}

/// Load the boundary GGUF and run extraction once. CLI only.
pub fn run_gliner2_boundary(
    source: Arc<dyn TensorSource>,
    tasks: &[Task],
    kinds: &[BoundaryTaskKind],
    context: &str,
    n_threads_arg: usize,
    threshold: Option<f32>,
    output_json: bool,
) -> Result<(), String> {
    let started = std::time::Instant::now();
    if !crate::models::gliner_boundary::is_boundary_gguf(source.as_ref()) {
        return Err("--gliner2-boundary needs a gliner2 boundary variant GGUF \
             (gliner2.variant = \"boundary\")"
            .into());
    }
    // The GGUF's settings are a superset of what the loader needs, so read them
    // once and hand them to the model rather than re-reading per request.
    let model = BoundaryModel::from_source(source.as_ref())?;
    eprintln!(
        "GLiNER2 boundary: deberta-v3 {}x{}x{}, boundary_dim {}, pool {}, {} field(s)",
        model.config.n_layer,
        model.config.n_embd,
        model.config.n_head,
        model.settings.boundary_dim,
        model.settings.pool_size,
        tasks.iter().map(|task| task.labels.len()).sum::<usize>(),
    );
    let result = extract(&model, context, tasks, kinds, n_threads_arg, threshold)?;
    let spans = &result.spans;
    let elapsed = started.elapsed().as_millis();

    if output_json {
        let payload = BoundaryJson {
            spans: spans
                .iter()
                .map(|span: &ExtractedSpan| SpanJson {
                    field: span.field.clone(),
                    score: span.score,
                    start: span.start,
                    end: span.end,
                    text: span.text.clone(),
                })
                .collect(),
            classifications: result
                .classifications
                .iter()
                .map(|group| ClassJson {
                    task: group.task.clone(),
                    activation: group.activation,
                    labels: group.labels.clone(),
                    probabilities: group.probabilities.clone(),
                    selected: group.selected.clone(),
                    choice_label: group.choice_label.clone(),
                })
                .collect(),
        };
        println!(
            "{}",
            serde_json::to_string(&payload).map_err(|error| format!("json encode: {error}"))?
        );
        return Ok(());
    }
    if !result.query_names.is_empty() {
        if spans.is_empty() {
            println!("(no spans above the threshold)");
        }
        let mut current = String::new();
        for span in spans {
            if span.field != current {
                current = span.field.clone();
                println!("{current}:");
            }
            println!(
                "  [{}..{}] p={:.4} {}",
                span.start, span.end, span.score, span.text
            );
        }
    }
    print_classifications(&result.classifications);
    println!(
        "({elapsed} ms, {} span(s), {} classification(s))",
        spans.len(),
        result.classifications.len()
    );
    Ok(())
}
