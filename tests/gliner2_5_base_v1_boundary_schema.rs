//! Schema parsing for the boundary families: `entities`, `json_structures`
//! (`[C]`), `relations` (`[R]`) and `classifications` (`[L]`).
//!
//! Pure — no GGUF, so this always runs. The part that matters most here is
//! **group order**, because it is load-bearing twice: it fixes which marker index
//! each field lands on, *and* it fixes the extractive query ids that the relation
//! and record heads read as head/tail and anchor/field slots. A parser that
//! reorders groups produces a schema that looks fine and decodes into the wrong
//! pairings.

use rust_model_inference::app::parse_boundary_schema;
use rust_model_inference::models::gliner::prompt::BoundaryTaskKind;

fn parse(
    schema: &str,
) -> (
    Vec<String>,
    Vec<(String, BoundaryTaskKind)>,
    Vec<Vec<String>>,
) {
    let (tasks, kinds) =
        parse_boundary_schema(&serde_json::from_str(schema).expect("schema is valid JSON"))
            .expect("schema should parse");
    let names = tasks.iter().map(|t| t.name.clone()).collect();
    let labelled = tasks
        .iter()
        .zip(&kinds)
        .map(|(task, kind)| (task.name.clone(), *kind))
        .collect();
    let fields = tasks
        .iter()
        .map(|task| task.labels.iter().map(|l| l.name.clone()).collect())
        .collect();
    (names, labelled, fields)
}

#[test]
fn groups_come_out_in_the_references_order() {
    // `_transform_record` emits json_structures, then entities, then relations,
    // then classifications (`processor.py:893-904`).
    let (names, labelled, _) = parse(
        r#"{
            "classifications": [{"task": "tone", "labels": ["a", "b"]}],
            "relations": [{"worked_in": {"head": "person", "tail": "location"}}],
            "entities": ["person", "location"],
            "json_structures": [{"event": ["who", "what"]}]
        }"#,
    );
    assert_eq!(
        names,
        vec!["event", "entities", "worked_in", "tone"],
        "group order must follow _transform_record"
    );
    assert_eq!(
        labelled.iter().map(|(_, kind)| *kind).collect::<Vec<_>>(),
        vec![
            BoundaryTaskKind::JsonStructure,
            BoundaryTaskKind::Entities,
            BoundaryTaskKind::Relation,
            BoundaryTaskKind::Classification,
        ]
    );
}

#[test]
fn entity_queries_take_the_lower_ids_when_a_relation_is_present() {
    // A relation group reads its head/tail roles off the extractive query ids, so
    // the entity fields must come first. The relation's own fields then occupy
    // the ids after them, which is what `relation_specs` in extract.rs relies on.
    let (_, _, fields) = parse(
        r#"{
            "entities": ["person", "location"],
            "relations": [{"worked_in": {"head": "person", "tail": "location"}}]
        }"#,
    );
    assert_eq!(
        fields,
        vec![
            vec!["person".to_string(), "location".to_string()],
            vec!["head".to_string(), "tail".to_string()],
        ],
        "entity queries must be routed before the relation roles"
    );
}

#[test]
fn json_structure_fields_union_in_first_seen_order() {
    // The reference unions the fields across occurrences keeping first-seen
    // order. Routing this through a set would make the schema prompt depend on
    // PYTHONHASHSEED, which changes the query order and the decoded values.
    let (_, _, fields) = parse(
        r#"{"json_structures": [
            {"person": ["name", "employer", "city"]},
            {"person": ["name", "born", "employer"]}
        ]}"#,
    );
    assert_eq!(
        fields,
        vec![vec![
            "name".to_string(),
            "employer".to_string(),
            "city".to_string(),
            "born".to_string(),
        ]],
        "the union must be first-seen order, not sorted and not last-wins"
    );
}

#[test]
fn two_entries_naming_the_same_structure_merge_into_one_group() {
    let (names, _, fields) =
        parse(r#"{"json_structures": [{"event": ["who"]}, {"event": ["what"]}]}"#);
    assert_eq!(
        names,
        vec!["event"],
        "one group per name, not one per entry"
    );
    assert_eq!(fields, vec![vec!["who".to_string(), "what".to_string()]]);
}

#[test]
fn json_descriptions_is_a_field_to_description_map() {
    // Unlike a relations group, whose description is a plain string on the
    // group, `json_descriptions[parent]` maps *field* to description. Reading it
    // as a string would silently drop every description and shift every later
    // marker index.
    let (tasks, _) = parse_boundary_schema(
        &serde_json::from_str(
            r#"{"json_structures": [{"person": ["name", "employer"]}],
                "json_descriptions": {"person": {"name": "the full name",
                                                "employer": "who they work for"}}}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        tasks[0].labels[0].description.as_deref(),
        Some("the full name")
    );
    assert_eq!(
        tasks[0].labels[1].description.as_deref(),
        Some("who they work for")
    );

    // And the relations side stays a single string on the group.
    let (tasks, _) = parse_boundary_schema(
        &serde_json::from_str(
            r#"{"relations": [{"worked_in": {"head": "person", "tail": "location"}}],
                "relation_descriptions": {"worked_in": "who worked in which place"}}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        tasks[0].prompt.as_deref(),
        Some("who worked in which place")
    );
    assert!(
        tasks[0].labels.iter().all(|l| l.description.is_none()),
        "relation roles carry no per-field description"
    );
}

#[test]
fn a_relation_needs_a_head_and_a_tail() {
    for schema in [
        r#"{"relations": [{"worked_in": {"head": "person"}}]}"#,
        r#"{"relations": [{"worked_in": {"tail": "location"}}]}"#,
        r#"{"relations": [{"worked_in": {}}]}"#,
    ] {
        let error = parse_boundary_schema(&serde_json::from_str(schema).unwrap())
            .expect_err("fewer than two roles must be rejected");
        assert!(
            error.contains("at least 2"),
            "{schema} should say what is missing, got {error}"
        );
    }
}

#[test]
fn an_empty_json_structure_group_is_skipped() {
    // The reference skips a group whose unioned field set is empty rather than
    // emitting one with no `[C]` children, which would have no query at all.
    let (names, _, _) =
        parse(r#"{"entities": ["person"], "json_structures": [{"empty": []}, {"real": ["a"]}]}"#);
    assert_eq!(
        names,
        vec!["real", "entities"],
        "the empty group contributes no task but does not displace the others"
    );
    // With nothing else in the schema there is no group left to run, which is an
    // error rather than a successful empty result.
    let error = parse_boundary_schema(
        &serde_json::from_str(r#"{"json_structures": [{"empty": []}]}"#).unwrap(),
    )
    .expect_err("an all-empty schema has no group to run");
    assert!(error.contains("json_structures"), "got {error}");
}

#[test]
fn a_schema_with_no_recognised_group_is_rejected() {
    let error = parse_boundary_schema(&serde_json::from_str(r#"{"unrelated": 1}"#).unwrap())
        .expect_err("an empty schema must be rejected");
    // The message should name every accepted key, so a caller who typo'd one is
    // told what it should have been.
    for key in [
        "entities",
        "json_structures",
        "relations",
        "classifications",
    ] {
        assert!(
            error.contains(key),
            "the error should mention {key}: {error}"
        );
    }
}

#[test]
fn a_record_annotated_structure_still_parses_as_a_task() {
    // Whether a `[C]` group is a record or a legacy structure is decided later,
    // by `compile_record_specs`. The task shape is identical either way, so the
    // parser must not branch on `record_metadata` — doing so would make the same
    // schema produce a different prompt depending on its annotation.
    let bare = parse(r#"{"json_structures": [{"person": ["name", "employer"]}]}"#);
    let annotated = parse(
        r#"{"json_structures": [{"person": ["name", "employer"]}],
            "record_metadata": {"person": {"mode": "natural", "anchor": "name"}}}"#,
    );
    assert_eq!(
        bare, annotated,
        "record_metadata must not change the prompt"
    );
}
