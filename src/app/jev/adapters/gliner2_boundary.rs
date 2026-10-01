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

/// One relation edge, as emitted by `--jev-output json`.
#[derive(Debug, serde::Serialize)]
struct RelationJson {
    relation: String,
    score: f32,
    head: String,
    head_start: usize,
    head_end: usize,
    tail: String,
    tail_start: usize,
    tail_end: usize,
}

/// One record's field -> spans, as emitted by `--jev-output json`.
type RecordFieldsJson = std::collections::BTreeMap<String, Vec<RecordSpanJson>>;

/// One span inside a record field.
#[derive(Debug, serde::Serialize)]
struct RecordSpanJson {
    start: usize,
    end: usize,
    text: String,
}

/// One decoded record, as emitted by `--jev-output json`.
#[derive(Debug, serde::Serialize)]
struct RecordJson {
    task: String,
    mode: String,
    score: f32,
    fields: RecordFieldsJson,
}

/// The whole JSON payload for `--jev-output json`.
#[derive(Debug, serde::Serialize)]
struct BoundaryJson {
    spans: Vec<SpanJson>,
    relations: Vec<RelationJson>,
    records: Vec<RecordJson>,
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
    // The group order *is* the contract, in two ways at once: it fixes which
    // marker index each field lands on, and it fixes the extractive query ids,
    // which the relation head reads as head/tail role slots and the record head
    // as anchor/field slots. `_transform_record` emits json_structures, then
    // entities, then relations, then classifications (`processor.py:893-904`).
    // Parsing relations first silently handed the relation roles ids 0 and 1 and
    // pushed the entity queries after them.
    for task in parse_json_structure_groups(object)? {
        tasks.push(task);
        kinds.push(BoundaryTaskKind::JsonStructure);
    }
    if object.contains_key("entities") {
        tasks.push(parse_entities_group(object)?);
        kinds.push(BoundaryTaskKind::Entities);
    }
    for task in parse_relation_groups(object)? {
        tasks.push(task);
        kinds.push(BoundaryTaskKind::Relation);
    }
    if let Some(classifications) = object.get("classifications") {
        let items = classifications
            .as_array()
            .ok_or("\"classifications\" must be a list of {task, labels}")?;
        for item in items {
            tasks.push(parse_classification_group(item)?);
            kinds.push(BoundaryTaskKind::Classification);
        }
    }
    if tasks.is_empty() {
        return Err(
            "gliner2 boundary schema needs one of \"entities\", \"json_structures\", \
             \"relations\" or \"classifications\""
                .into(),
        );
    }
    Ok((tasks, kinds))
}

/// Parse the reference's `"relations"` group into `[R]` tasks.
///
/// The reference's `_process_relations` (`processor.py:1074-1120`) reads
/// `schema["relations"]` as a list of single-key objects whose value maps a
/// *role field name* to its gold span, and takes `list(value.keys())` as the
/// field list — the spans are training targets, not part of the prompt. The
/// group name is the key, and `relation_descriptions` supplies the prompt.
///
/// The field order is the contract: the first two fields become the head and
/// tail roles, so reordering them swaps the relation's direction.
fn parse_relation_groups(
    object: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<Task>, String> {
    let Some(relations) = object.get("relations") else {
        return Ok(Vec::new());
    };
    let items = relations
        .as_array()
        .ok_or("\"relations\" must be a list of {relation_name: {\"head\": ..., \"tail\": ...}}")?;
    let descriptions = object
        .get("relation_descriptions")
        .and_then(|v| v.as_object());
    let mut tasks = Vec::with_capacity(items.len());
    for item in items {
        let map = item
            .as_object()
            .ok_or("each relation must be an object of {relation_name: {field: span}}")?;
        if map.len() != 1 {
            return Err(format!(
                "each relation entry must name exactly one relation type, got {} keys",
                map.len()
            ));
        }
        let (name, roles) = map.iter().next().expect("len checked");
        let fields: Vec<String> = roles
            .as_object()
            .ok_or_else(|| format!("relation {name:?} must map role names to spans, got {roles}"))?
            .keys()
            .cloned()
            .collect();
        if fields.len() < 2 {
            return Err(format!(
                "relation {name:?} declares {} role(s); the reference reads the first two as \
                 head and tail, so it needs at least 2",
                fields.len()
            ));
        }
        let labels = fields
            .iter()
            .map(|field| {
                let mut label = Label::new(field);
                // Relation roles carry no description of their own: the
                // description belongs to the relation type and becomes the
                // group prompt, which `_schema_group_name` later splits off.
                label.description = None;
                label
            })
            .collect();
        let mut task = Task::new(name, labels);
        if let Some(description) = descriptions
            .and_then(|map| map.get(name))
            .and_then(|value| value.as_str())
        {
            task.prompt = Some(description.to_string());
        }
        tasks.push(task);
    }
    Ok(tasks)
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

/// Parse the reference's `"json_structures"` groups into `[C]` tasks.
///
/// `_process_json_structures` (`processor.py:921-1022`) reads a list of
/// single-key objects whose value is a list of *occurrences*, unions the field
/// names across them **in first-seen order**, and keeps only the keys — each
/// occurrence is a `{field: gold_span}` dict whose values are training targets. That order is the
/// contract and the reference keeps it deliberately: routing it through a `set`
/// made schema prompts depend on `PYTHONHASHSEED`, which would change both the
/// query order and the decoded values across otherwise identical processes.
/// Only the keys reach the prompt; the per-occurrence values are training
/// targets.
///
/// Unlike an entities group, `json_descriptions[parent]` is a **field →
/// description map**, not a single description string.
///
/// A group named in `record_metadata` with a `mode` is a *record* rather than a
/// legacy structure; that is decided later, by `compile_record_specs`, so the
/// task shape is identical either way.
fn parse_json_structure_groups(
    object: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<Task>, String> {
    let Some(structures) = object.get("json_structures") else {
        return Ok(Vec::new());
    };
    let items = structures
        .as_array()
        .ok_or("\"json_structures\" must be a list of {structure_name: [field, ...]}")?;
    let descriptions = object.get("json_descriptions").and_then(|v| v.as_object());

    // Union the fields per group, keeping declaration order. The reference uses
    // a dict keyed by parent, so two entries naming the same parent merge into
    // one group; a `Vec` in first-seen order does the same without relying on
    // hash order.
    let mut order: Vec<String> = Vec::new();
    let mut groups: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();
    for item in items {
        let map = item
            .as_object()
            .ok_or("each json_structures entry must be an object of {name: [fields]}")?;
        if map.is_empty() {
            return Err("each json_structures entry must name a structure".into());
        }
        for (name, occurrences) in map {
            // Each element of `json_structures[parent]` is one *occurrence* of the
            // structure. The reference iterates `for field_name in occ` where `occ`
            // is a `{field: gold_span}` dict (`processor.py:956`), so the field
            // names are that dict's **keys** — the values are training targets and
            // never reach the prompt. A bare string is accepted too, since that is
            // the same thing without the unused span values.
            // The reference tolerates a bare `{field: span}` dict here as a single
            // occurrence (`occ = {"name": ..., "employer": ...}`), so accept that
            // shorthand alongside the documented list form.
            let single: Vec<serde_json::Value> = match occurrences {
                serde_json::Value::Array(items) => items.clone(),
                serde_json::Value::Object(_) => vec![occurrences.clone()],
                _ => {
                    return Err(format!(
                        "json_structures[{name:?}] must be a list of occurrences"
                    ))
                }
            };
            let occurrences = &single;
            let entry = groups.entry(name.clone()).or_insert_with(|| {
                order.push(name.clone());
                Vec::new()
            });
            for occurrence in occurrences {
                let fields: Vec<String> = match occurrence {
                    serde_json::Value::Object(map) => map.keys().cloned().collect(),
                    serde_json::Value::String(field) => vec![field.clone()],
                    other => {
                        return Err(format!(
                            "json_structures[{name:?}] occurrence must be a {{field: span}} \
                             object or a field name, got {other}"
                        ))
                    }
                };
                for field in fields {
                    if !entry.contains(&field) {
                        entry.push(field);
                    }
                }
            }
        }
    }

    let mut tasks = Vec::with_capacity(order.len());
    for name in order {
        let fields = &groups[&name];
        if fields.is_empty() {
            // The reference skips an empty field set rather than emitting a
            // group with no `[C]` children, which would have no query at all.
            continue;
        }
        let group_descriptions = descriptions.and_then(|map| map.get(&name));
        let labels = fields
            .iter()
            .map(|field| {
                let mut label = Label::new(field.clone());
                label.description = group_descriptions
                    .and_then(|value| value.as_object())
                    .and_then(|map| map.get(field))
                    .and_then(|value| value.as_str())
                    .map(str::to_string);
                label
            })
            .collect();
        tasks.push(Task::new(name, labels));
    }
    Ok(tasks)
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
    record_metadata: Option<&serde_json::Value>,
) -> Result<Extraction, String> {
    let mut result = run_mixed_extraction(
        model,
        text,
        tasks,
        kinds,
        n_threads_arg,
        threshold,
        record_metadata,
    )?;
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

/// Caller-supplied knobs for one boundary extraction.
///
/// Grouped so the CLI and HTTP entry points do not have to thread three
/// separate `Option`s in a fixed order, which is exactly the kind of positional
/// contract that gets one argument silently transposed.
#[derive(Clone, Copy, Debug, Default)]
pub struct BoundaryDecodeOptions<'a> {
    /// Span and relation score threshold; `None` is the checkpoint default.
    pub threshold: Option<f32>,
    /// The schema's top-level `record_metadata`. `None` means every
    /// `json_structures` group keeps the legacy structure path.
    pub record_metadata: Option<&'a serde_json::Value>,
    pub output_json: bool,
}

/// Load the boundary GGUF and run extraction once. CLI only.
pub fn run_gliner2_boundary(
    source: Arc<dyn TensorSource>,
    tasks: &[Task],
    kinds: &[BoundaryTaskKind],
    context: &str,
    n_threads_arg: usize,
    decode: BoundaryDecodeOptions<'_>,
) -> Result<(), String> {
    let started = std::time::Instant::now();
    let BoundaryDecodeOptions {
        threshold,
        record_metadata,
        output_json,
    } = decode;
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
    let result = extract(
        &model,
        context,
        tasks,
        kinds,
        n_threads_arg,
        threshold,
        record_metadata,
    )?;
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
            relations: result
                .relations
                .iter()
                .map(|relation| RelationJson {
                    relation: relation.relation_type.clone(),
                    score: relation.score,
                    head: relation.head_text.clone(),
                    head_start: relation.head_start,
                    head_end: relation.head_end,
                    tail: relation.tail_text.clone(),
                    tail_start: relation.tail_start,
                    tail_end: relation.tail_end,
                })
                .collect(),
            records: result
                .records
                .iter()
                .map(|record| {
                    let mut fields: RecordFieldsJson = std::collections::BTreeMap::new();
                    for (query_id, spans) in &record.fields {
                        fields.insert(
                            query_id.to_string(),
                            spans
                                .iter()
                                .map(|(start, end)| RecordSpanJson {
                                    start: *start,
                                    end: *end,
                                    text: result.words[*start..*end].join(" "),
                                })
                                .collect(),
                        );
                    }
                    RecordJson {
                        task: record.task.clone(),
                        mode: record.mode.clone(),
                        score: record.score,
                        fields,
                    }
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
    if !result.relations.is_empty() {
        let mut current = String::new();
        for relation in &result.relations {
            if relation.relation_type != current {
                current = relation.relation_type.clone();
                println!("{current}:");
            }
            println!(
                "  [{}..{}] -> [{}..{}] p={:.4} {} -> {}",
                relation.head_start,
                relation.head_end,
                relation.tail_start,
                relation.tail_end,
                relation.score,
                relation.head_text,
                relation.tail_text
            );
        }
    }
    if !result.records.is_empty() {
        let mut current = String::new();
        for record in &result.records {
            if record.task != current {
                current = record.task.clone();
                println!("{current} ({} mode):", record.mode);
            }
            println!("  p={:.4}", record.score);
            for (query_id, bound) in &record.fields {
                let text = bound
                    .iter()
                    .map(|(start, end)| result.words[*start..*end].join(" "))
                    .collect::<Vec<_>>()
                    .join(" | ");
                println!("    field {query_id}: {text}");
            }
        }
    }
    print_classifications(&result.classifications);
    println!(
        "({elapsed} ms, {} span(s), {} relation(s), {} record(s), {} classification(s))",
        spans.len(),
        result.relations.len(),
        result.records.len(),
        result.classifications.len()
    );
    Ok(())
}
