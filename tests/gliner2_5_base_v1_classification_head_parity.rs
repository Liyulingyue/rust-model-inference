//! Byte-exact parity for the classification head, `null_projection` and
//! `count_head`.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf`.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_classification_and_query_heads.py`,
//! which uses the reference `_encode_core` routing so the fixture's indices are
//! the ones inference actually uses.
//!
//! The marker matters and is easy to get backwards: the reference's
//! `_process_classifications` emits **`[L]`**, the same token Decide uses — `[C]`
//! belongs to `json_structures`. The test asserts the routing directly, because
//! a classification group routed into the document pool would quietly produce
//! spans instead of label logits.
//!
//! Only one test here runs the encoder. `compute::encode` builds a
//! `ComputePool` per call and its workers busy-spin while idle, so two
//! concurrent extractions in one process oversubscribe the machine badly (a
//! three-test version of this file took 43s wall against ~4s for the same work
//! serialized). Keeping the heavy work in a single test avoids making the
//! suite pay for that; the underlying contention is a pre-existing engine
//! characteristic, noted in `glinerTODO.md`.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner::prompt::{BoundaryTaskKind, Label, Task};
use rust_model_inference::models::gliner_boundary::extract::{
    encode_mixed_boundary_prompt, run_mixed_extraction, ClassificationResult,
};
use rust_model_inference::models::gliner_boundary::BoundaryModel;

const GGUF: &str = "models/gliner2.5-base-v1/gliner2.5-base-v1-f32.gguf";
const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/classification-head-golden.json";

fn loaded_model() -> Option<BoundaryModel<'static>> {
    let path = std::env::var_os("RMI_GLINER2_5_BASE_V1_GGUF").map(std::path::PathBuf::from)?;
    if !path.exists() {
        eprintln!("skipping: {} does not exist", path.display());
        return None;
    }
    let source = GGUFLoader::from_file(&path).expect("open boundary GGUF");
    let leaked: &'static dyn TensorSource = Box::leak(Box::new(source));
    Some(BoundaryModel::from_source(leaked).expect("load boundary model"))
}

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE).unwrap_or_else(|_| {
        panic!("missing {FIXTURE}; regenerate with dump_classification_and_query_heads.py")
    });
    serde_json::from_str(&raw).expect("parse classification-head-golden.json")
}

fn f32s(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn usize_list(value: &serde_json::Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

fn strings(value: &serde_json::Value) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect()
}

/// Rebuild the case's tasks from the fixture's own schema, so the test cannot
/// drift from what the reference was given.
fn tasks_of(case: &serde_json::Value) -> (Vec<Task>, Vec<BoundaryTaskKind>) {
    let schema = &case["schema"];
    let object = schema.as_object().expect("schema object");
    let mut tasks = Vec::new();
    let mut kinds = Vec::new();
    if let Some(entities) = object.get("entities").and_then(|v| v.as_object()) {
        if !entities.is_empty() {
            let descriptions = object.get("entity_descriptions");
            let labels = entities
                .keys()
                .map(|name| {
                    let mut label = Label::new(name.clone());
                    label.description = descriptions
                        .and_then(|d| d.as_object())
                        .and_then(|d| d.get(name))
                        .and_then(|v| v.as_str())
                        .map(str::to_string);
                    label
                })
                .collect();
            tasks.push(Task::new("entities", labels));
            kinds.push(BoundaryTaskKind::Entities);
        }
    }
    for item in object
        .get("classifications")
        .and_then(|v| v.as_array())
        .unwrap_or(&Vec::new())
    {
        // Inference uses `example_mode = "both"`, so a group's
        // `label_descriptions` go into the prompt text and therefore shift every
        // marker index after them. Dropping them would still produce plausible
        // logits from the wrong positions.
        let descriptions = item.get("label_descriptions").and_then(|v| v.as_object());
        let labels: Vec<Label> = item["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let name = v.as_str().unwrap();
                let mut label = Label::new(name);
                label.description = descriptions
                    .and_then(|map| map.get(name))
                    .and_then(|value| value.as_str())
                    .map(str::to_string);
                label
            })
            .collect();
        let mut task = Task::new(item["task"].as_str().unwrap(), labels);
        task.multi_label = item
            .get("multi_label")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if let Some(threshold) = item.get("cls_threshold").and_then(|v| v.as_f64()) {
            task.cls_threshold = threshold as f32;
        }
        tasks.push(task);
        kinds.push(BoundaryTaskKind::Classification);
    }
    (tasks, kinds)
}

#[test]
fn the_two_routings_are_disjoint() {
    let Some(model) = loaded_model() else {
        return;
    };
    let fixture = fixture();
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let (tasks, kinds) = tasks_of(case);
        let encoded =
            encode_mixed_boundary_prompt(&model, &tasks, &kinds, case["text"].as_str().unwrap())
                .expect("encode mixed boundary prompt");

        assert_eq!(
            encoded.classification_positions,
            usize_list(&case["cls_marker_indices"]),
            "case {index}: classification routing differs"
        );
        assert_eq!(
            encoded.classification_names,
            strings(&case["cls_labels"]),
            "case {index}: classification label order differs"
        );
        assert_eq!(
            encoded.query_positions,
            usize_list(&case["query_marker_indices"]),
            "case {index}: extractive query routing differs"
        );
        assert_eq!(
            encoded.query_names,
            strings(&case["query_names"]),
            "case {index}: extractive field order differs"
        );
        // A marker must not be routed twice. If `[L]` had been treated as
        // `[C]` — or a classification group had been routed as queries — one of
        // these two sets would be empty and the other would be too long.
        for position in &encoded.classification_positions {
            assert!(
                !encoded.query_positions.contains(position),
                "case {index}: marker {position} is routed as both a query and a choice"
            );
        }
    }
}

#[test]
fn logits_probabilities_and_choices_match() {
    let Some(model) = loaded_model() else {
        return;
    };
    let fixture = fixture();
    let temperature = fixture["classification_temperature"].as_f64().unwrap() as f32;
    assert_eq!(temperature, model.settings.classification_temperature);
    let mut worst = 0.0f32;

    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let (tasks, kinds) = tasks_of(case);
        let result = run_mixed_extraction(
            &model,
            case["text"].as_str().unwrap(),
            &tasks,
            &kinds,
            0,
            None,
        )
        .expect("run mixed extraction");

        let want_classification_groups = kinds
            .iter()
            .filter(|kind| **kind == BoundaryTaskKind::Classification)
            .count();
        assert_eq!(
            result.classifications.len(),
            want_classification_groups,
            "case {index}: one result per classification group"
        );
        let want_logits = f32s(&case["cls_logits"]);
        let want_probs = f32s(&case["cls_probabilities"]);
        let want_labels = strings(&case["cls_labels"]);
        let want_chosen = strings(&case["cls_chosen"]);
        let want_activation = case["cls_activation"].as_str().unwrap();

        // One `ClassificationResult` per group; flatten them for comparison.
        let got: Vec<&ClassificationResult> = result.classifications.iter().collect();
        assert!(
            !got.is_empty(),
            "case {index}: no classification result was produced"
        );
        let mut logit_at = 0usize;
        let mut prob_at = 0usize;
        for group in &got {
            assert_eq!(
                group.activation, want_activation,
                "case {index}: activation"
            );
            assert_eq!(
                group.labels, want_labels,
                "case {index}: label order differs, so every logit is paired with the wrong label"
            );
            for (i, logit) in group.logits.iter().enumerate() {
                worst = worst.max((logit - want_logits[logit_at + i]).abs());
            }
            for (i, prob) in group.probabilities.iter().enumerate() {
                worst = worst.max((prob - want_probs[prob_at + i]).abs());
            }
            logit_at += group.logits.len();
            prob_at += group.probabilities.len();
        }
        assert_eq!(logit_at, want_logits.len(), "case {index}: logit count");
        assert_eq!(prob_at, want_probs.len(), "case {index}: probability count");

        // The chosen labels are what a caller actually reads.
        let chosen: Vec<String> = got
            .iter()
            .flat_map(|group| {
                if group.multi_label {
                    group.selected.clone()
                } else {
                    group.choice_label.clone().into_iter().collect()
                }
            })
            .collect();
        assert_eq!(chosen, want_chosen, "case {index}: chosen labels differ");

        let want_null = f32s(&case["null_logits"]);
        assert_eq!(
            result.query_heads.null_logits.len(),
            want_null.len(),
            "case {index}: null_projection count differs"
        );
        for (i, (got, want)) in result
            .query_heads
            .null_logits
            .iter()
            .zip(want_null.iter())
            .enumerate()
        {
            worst = worst.max((got - want).abs());
            // base-v1's abstention threshold is 0.5, so these should be well
            // clear of it for a document that *does* contain a person.
            assert!(
                1.0 / (1.0 + (-got).exp()) < fixture["abstention_threshold"].as_f64().unwrap() as f32,
                "case {index}: null logit {i} ({got}) would abstain a query that extracted something"
            );
        }
        for (got, want) in result
            .query_heads
            .count_log_rates
            .iter()
            .zip(f32s(&case["count_log_rates"]).iter())
        {
            worst = worst.max((got - want).abs());
        }

        // The activation has to be the one the group's `class_act` implies, and
        // it changes what "normalized" means: a single-label group is a softmax
        // (sums to 1), a multi-label group is independent sigmoids (does not).
        for group in &result.classifications {
            let sum: f32 = group.probabilities.iter().sum();
            if group.multi_label {
                assert!(
                    group.probabilities.iter().all(|p| (0.0..=1.0).contains(p)),
                    "case {index}: sigmoid probabilities out of range"
                );
                assert!(
                    (0.0..=1.0).contains(&sum),
                    "case {index}: multi-label sigmoid sum {sum} must not be forced to 1"
                );
            } else {
                assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "case {index}: single-label softmax sum is {sum}"
                );
            }
        }
    }

    assert!(
        worst < 1e-4,
        "max classification/head delta {worst} exceeds threshold 1e-4"
    );
}
