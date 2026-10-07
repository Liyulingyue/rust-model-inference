//! End-to-end parity for the relation head: real text + relation schema -> edges.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_relations_end_to_end.py`, which
//! runs the reference pipeline (SchemaTransformer → DeBERTa encoder →
//! BoundaryHead → pair generator → scorer → `_decode_relations`) on the same
//! text and schema.
//!
//! This is the only test that pins the two things the generator/scorer fixture
//! cannot see:
//!
//! * **query-id assignment.** A relation group's first two fields are its head
//!   and tail slots, and ids are assigned in schema-group order. The mixed case
//!   (two entity queries plus one relation) is the only place that arithmetic is
//!   observable — with a lone relation group, every implementation agrees the
//!   head is 0 and the tail is 1.
//! * **the relation type string.** `_schema_group_name` recovers the
//!   prompt-joined name and `_decode_relations` maps it back to the bare schema
//!   name. A description leaking into the output is invisible without a
//!   `relation_descriptions` entry.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf`.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner::prompt::BoundaryTaskKind;
use rust_model_inference::models::gliner_boundary::{
    run_mixed_extraction, BoundaryModel, Extraction, SchemaOptions,
};

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/relations-e2e-golden.json";
const TOLERANCE: f32 = 2.0e-4;

fn loaded_model() -> Option<(Box<dyn std::any::Any>, BoundaryModel<'static>)> {
    let path = match std::env::var_os("RMI_GLINER2_5_BASE_V1_GGUF") {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            eprintln!("skipping: set RMI_GLINER2_5_BASE_V1_GGUF to enable this test");
            return None;
        }
    };
    if !path.exists() {
        eprintln!("skipping: {} does not exist", path.display());
        return None;
    }
    let source = GGUFLoader::from_file(&path).expect("open boundary GGUF");
    let leaked: &'static dyn TensorSource = Box::leak(Box::new(source));
    let model = BoundaryModel::from_source(leaked).expect("load boundary model");
    Some((Box::new(()), model))
}

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE).unwrap_or_else(|_| {
        panic!("missing {FIXTURE}; regenerate with dump_relations_end_to_end.py")
    });
    serde_json::from_str(&raw).expect("parse relations-e2e-golden.json")
}

#[derive(Debug, serde::Deserialize)]
struct Edge {
    relation: String,
    score: f32,
    head: String,
    head_start: usize,
    head_end: usize,
    tail: String,
    tail_start: usize,
    tail_end: usize,
}

fn run_case(
    model: &BoundaryModel<'_>,
    schema: &serde_json::Value,
    text: &str,
    threshold: f32,
) -> Extraction {
    let (tasks, kinds) =
        rust_model_inference::app::parse_boundary_schema(schema).expect("parse schema");
    assert_eq!(tasks.len(), kinds.len());
    assert!(
        kinds.iter().any(|k| *k == BoundaryTaskKind::Relation),
        "the fixture cases all declare relation groups"
    );
    run_mixed_extraction(
        model,
        text,
        &tasks,
        &kinds,
        0,
        Some(threshold),
        SchemaOptions::default(),
    )
    .expect("extract")
}

#[test]
fn relation_end_to_end_matches_the_reference() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let data = fixture();
    let cases = data["cases"].as_array().expect("cases");
    assert!(!cases.is_empty());

    for case in cases {
        let text = case["text"].as_str().unwrap();
        let threshold = case["threshold"].as_f64().unwrap() as f32;
        // The mixed schema is the fourth case; the rest all use the plain
        // relation schema. The fixture records which by its `query_names`.
        let mixed = case["query_names"][0] == "person";
        let schema = if mixed {
            data["mixed_schema"].clone()
        } else {
            data["relation_schema"].clone()
        };

        let result = run_case(&model, &schema, text, threshold);

        // --- routing: the extractive query names, in order ---
        let want_names: Vec<&str> = case["query_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(
            result.query_names, want_names,
            "{text:?}: extractive query routing"
        );
        assert_eq!(
            result.query_names.len() as u64,
            case["query_count"].as_u64().unwrap(),
            "{text:?}: query count"
        );

        // --- the word list the offsets index into ---
        let want_words: Vec<String> = case["text_words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(result.words, want_words, "{text:?}: word list");

        // --- the edges ---
        let want: Vec<Edge> = serde_json::from_value(case["edges"].clone()).unwrap();
        assert_eq!(
            result.relations.len(),
            want.len(),
            "{text:?}: edge count (the reference proposed {} pair(s))",
            case["pair_count"].as_u64().unwrap()
        );
        for (index, (got, expected)) in result.relations.iter().zip(&want).enumerate() {
            assert_eq!(
                got.relation_type, expected.relation,
                "{text:?}[{index}] type"
            );
            // The bare name, never the description-joined form: a leak here is
            // the whole point of the `relation_descriptions` fixture entries.
            assert!(
                !got.relation_type.contains("who worked"),
                "{text:?}[{index}]: the description leaked into the type: {:?}",
                got.relation_type
            );
            assert!(
                (got.score - expected.score).abs() < TOLERANCE,
                "{text:?}[{index}]: score {} vs {} (delta {})",
                got.score,
                expected.score,
                (got.score - expected.score).abs()
            );
            assert_eq!(got.head_text, expected.head, "{text:?}[{index}] head text");
            assert_eq!(got.tail_text, expected.tail, "{text:?}[{index}] tail text");
            assert_eq!(
                got.head_start, expected.head_start,
                "{text:?}[{index}] head_start"
            );
            assert_eq!(
                got.head_end, expected.head_end,
                "{text:?}[{index}] head_end"
            );
            assert_eq!(
                got.tail_start, expected.tail_start,
                "{text:?}[{index}] tail_start"
            );
            assert_eq!(
                got.tail_end, expected.tail_end,
                "{text:?}[{index}] tail_end"
            );
            // Offsets must index the word list this same call produced.
            assert_eq!(
                result.words[got.head_start..got.head_end].join(" "),
                got.head_text,
                "{text:?}[{index}]: head offsets disagree with their own text"
            );
            assert_eq!(
                result.words[got.tail_start..got.tail_end].join(" "),
                got.tail_text,
                "{text:?}[{index}]: tail offsets disagree with their own text"
            );
            // The reference drops pairs below the threshold, and a head and tail
            // that are the same mention never reach the scorer.
            assert!(
                got.score >= threshold,
                "{text:?}[{index}]: edge below the requested threshold"
            );
            assert!(
                got.head_start != got.tail_start || got.head_end != got.tail_end,
                "{text:?}[{index}]: a same-span relation survived"
            );
        }
    }
}

#[test]
fn relation_end_to_end_is_semantically_right() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let data = fixture();
    let schema = data["relation_schema"].clone();

    // "Marie Curie worked with Pierre Curie in Paris." should yield the
    // symmetric `collaborated_with` and both `worked_in` edges. A port that got
    // the head/tail slots backwards would still produce the right *set* of
    // spans but the wrong pairing, so this checks the pairs, not the texts.
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["text"].as_str().unwrap().contains("Marie Curie"))
        .expect("the Marie Curie case");
    let result = run_case(
        &model,
        &schema,
        case["text"].as_str().unwrap(),
        case["threshold"].as_f64().unwrap() as f32,
    );
    let pairs: Vec<(&str, &str, &str)> = result
        .relations
        .iter()
        .map(|r| {
            (
                r.relation_type.as_str(),
                r.head_text.as_str(),
                r.tail_text.as_str(),
            )
        })
        .collect();
    assert!(
        pairs.contains(&("collaborated_with", "marie curie", "pierre curie")),
        "missing the collaborated_with edge, got {pairs:?}"
    );
    assert!(
        pairs.contains(&("worked_in", "marie curie", "paris")),
        "missing the worked_in edge, got {pairs:?}"
    );
    // And not the reverse: `collaborated_with` is directional.
    assert!(
        !pairs.contains(&("collaborated_with", "pierre curie", "marie curie")),
        "the relation direction is reversed, got {pairs:?}"
    );
}

#[test]
fn negative_text_produces_no_relations() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let data = fixture();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["text"].as_str().unwrap().starts_with("nothing here"))
        .expect("the negative case");
    let result = run_case(
        &model,
        &data["relation_schema"],
        case["text"].as_str().unwrap(),
        case["threshold"].as_f64().unwrap() as f32,
    );
    assert_eq!(
        case["pair_count"].as_u64().unwrap(),
        0,
        "the fixture should exercise the empty case"
    );
    assert!(
        result.relations.is_empty(),
        "invented relations: {:?}",
        result.relations
    );
}

#[test]
fn a_relation_group_needs_a_head_and_a_tail() {
    // A single role is a schema error, not a silently empty relation: the
    // reference reads `role_ids[:2]` and skips the group only when it has fewer
    // than two, which would hide a caller's typo.
    let one_role = serde_json::json!({"relations": [{"worked_in": {"head": "person"}}]});
    let error = rust_model_inference::app::parse_boundary_schema(&one_role)
        .expect_err("a one-role relation must be rejected");
    assert!(
        error.contains("at least 2"),
        "error should say what is missing, got {error}"
    );
}
