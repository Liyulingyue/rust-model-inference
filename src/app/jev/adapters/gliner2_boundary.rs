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
use crate::models::gliner::prompt::{Label, Task, E_TOKEN};
use crate::models::gliner_boundary::extract::{extract_spans, ExtractedSpan};
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
pub fn parse_boundary_schema(schema: &serde_json::Value) -> Result<Vec<Task>, String> {
    let object = schema
        .as_object()
        .ok_or("gliner2 boundary schema must be a JSON object")?;
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
    Ok(vec![Task::new("entities", labels)])
}

/// Extract spans with an already-loaded model. The server keeps the model alive
/// across requests and uses this instead of [`run_gliner2_boundary`].
pub fn extract(
    model: &BoundaryModel<'_>,
    text: &str,
    tasks: &[Task],
    n_threads_arg: usize,
    threshold: Option<f32>,
) -> Result<Vec<ExtractedSpan>, String> {
    extract_spans(model, text, tasks, E_TOKEN, n_threads_arg, threshold)
}

/// Load the boundary GGUF and run extraction once. CLI only.
pub fn run_gliner2_boundary(
    source: Arc<dyn TensorSource>,
    tasks: &[Task],
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
    let spans = extract(&model, context, tasks, n_threads_arg, threshold)?;
    let elapsed = started.elapsed().as_millis();

    if output_json {
        let rows: Vec<SpanJson> = spans
            .iter()
            .map(|span| SpanJson {
                field: span.field.clone(),
                score: span.score,
                start: span.start,
                end: span.end,
                text: span.text.clone(),
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string(&rows).map_err(|error| format!("json encode: {error}"))?
        );
        return Ok(());
    }
    if spans.is_empty() {
        println!("(no spans above the threshold)");
    }
    let mut current = String::new();
    for span in &spans {
        if span.field != current {
            current = span.field.clone();
            println!("{current}:");
        }
        println!(
            "  [{}..{}] p={:.4} {}",
            span.start, span.end, span.score, span.text
        );
    }
    println!("({elapsed} ms, {} span(s))", spans.len());
    Ok(())
}
