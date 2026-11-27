//! Parity for record-metadata normalization and `RecordSpec` compilation.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_record_specs.py`, which runs the
//! reference's `normalize_record_metadata` and `compile_record_specs` on the same
//! 16 cases. Both are pure, so this needs no GGUF and always runs.
//!
//! The validation half is what this is really for. Each rule rejects a schema
//! that would otherwise fail much later and much less legibly: a `natural` group
//! with no anchor, a `latent` group *with* one, a cardinality outside the enum,
//! an anchor naming a field the group does not declare. And one rule is the
//! opposite — a group with no `mode` is skipped rather than defaulted, which is
//! the silent path and therefore the one most worth pinning.

use std::collections::BTreeMap;

use rust_model_inference::models::gliner_boundary::record_spec::{
    compile_record_specs, default_cardinality, FieldCardinality, LayoutQuery,
};

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/record-specs-golden.json";

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_record_specs.py"));
    serde_json::from_str(&raw).expect("parse record-specs-golden.json")
}

fn field_dtypes(value: &serde_json::Value) -> BTreeMap<String, BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for (task, fields) in value.as_object().expect("field_dtypes object") {
        out.insert(
            task.clone(),
            fields
                .as_object()
                .expect("field_dtypes entry")
                .iter()
                .map(|(field, dtype)| (field.clone(), dtype.as_str().unwrap().to_string()))
                .collect(),
        );
    }
    out
}

/// Rebuild the layout the oracle used: one global `query_id` counter, walking
/// groups in order and roles in order, exactly like the reference's `QueryLayout`.
fn layout(groups: &serde_json::Value) -> Vec<LayoutQuery> {
    let mut queries = Vec::new();
    let mut query_id = 0usize;
    for (task_index, group) in groups.as_array().expect("groups").iter().enumerate() {
        let task_type = group["task_type"].as_str().unwrap().to_string();
        let task_name = group["name"].as_str().unwrap().to_string();
        for (role_index, role) in group["roles"].as_array().unwrap().iter().enumerate() {
            queries.push(LayoutQuery {
                query_id,
                task_index,
                task_type: task_type.clone(),
                task_name: task_name.clone(),
                role_index,
                role_name: role.as_str().unwrap().to_string(),
            });
            query_id += 1;
        }
    }
    queries
}

#[test]
fn record_specs_match_the_reference() {
    let data = fixture();
    let cases = data["cases"].as_array().expect("cases");
    assert!(!cases.is_empty());

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let queries = layout(&case["groups"]);
        let dtypes = field_dtypes(&case["field_dtypes"]);
        let metadata = &case["record_metadata"];

        let result = compile_record_specs(&queries, metadata, &dtypes);
        if let Some(expected_error) = case["error"].as_str() {
            let error = result.err().unwrap_or_else(|| {
                panic!("{name}: the reference rejected this with {expected_error:?}")
            });
            // Match on the substance, not the exact wording: the reference's
            // messages are Python-flavoured and the Rust ones will not read
            // identically, but they must reject for the same reason.
            let needle = error_key(expected_error);
            assert!(
                error.contains(needle),
                "{name}: expected an error about {needle:?}, got {error:?}"
            );
            assert_eq!(
                case["specs"].as_object().unwrap().len(),
                0,
                "{name}: a rejected schema compiles no specs"
            );
            continue;
        }

        let specs = result.unwrap_or_else(|error| panic!("{name}: {error}"));
        let want = case["specs"].as_object().expect("specs object");
        assert_eq!(specs.len(), want.len(), "{name}: spec count");

        for (key, want_spec) in want {
            let got = specs
                .get(&key.parse::<usize>().expect("numeric task index"))
                .unwrap_or_else(|| panic!("{name}: missing spec for task {key}"));
            assert_eq!(
                got.task_name,
                want_spec["task_name"].as_str().unwrap(),
                "{name} name"
            );
            assert_eq!(
                got.task_type,
                want_spec["task_type"].as_str().unwrap(),
                "{name} type"
            );
            assert_eq!(got.mode, want_spec["mode"].as_str().unwrap(), "{name} mode");
            assert_eq!(
                got.occurrence_policy,
                want_spec["occurrence_policy"].as_str().unwrap(),
                "{name} policy"
            );
            // `None` vs `Some(-1)`: the fixture writes JSON null for the absent
            // case, and `anchor_query_id` is genuinely optional.
            assert_eq!(
                got.anchor_query_id,
                want_spec["anchor_query_id"].as_u64().map(|v| v as usize),
                "{name} anchor_query_id"
            );
            let want_fields = want_spec["fields"].as_array().expect("fields");
            assert_eq!(got.fields.len(), want_fields.len(), "{name} field count");
            for (got_field, want_field) in got.fields.iter().zip(want_fields) {
                assert_eq!(
                    got_field.query_id as u64,
                    want_field["query_id"].as_u64().unwrap(),
                    "{name} field {} query_id",
                    got_field.name
                );
                assert_eq!(
                    got_field.name,
                    want_field["name"].as_str().unwrap(),
                    "{name} field name"
                );
                assert_eq!(
                    got_field.role_index as u64,
                    want_field["role_index"].as_u64().unwrap(),
                    "{name} field {} role_index",
                    got_field.name
                );
                assert_eq!(
                    got_field.cardinality.as_str(),
                    want_field["cardinality"].as_str().unwrap(),
                    "{name} field {} cardinality",
                    got_field.name
                );
                assert_eq!(
                    got_field.is_anchor,
                    want_field["is_anchor"].as_bool().unwrap(),
                    "{name} field {} is_anchor",
                    got_field.name
                );
                assert_eq!(
                    got_field.exclusive,
                    want_field["exclusive"].as_bool().unwrap(),
                    "{name} field {} exclusive",
                    got_field.name
                );
            }
        }
    }
}

/// Pull the discriminating phrase out of the reference's Python error so the
/// Rust side is checked for the same reason, not the same wording.
fn error_key(message: &str) -> &str {
    if message.contains("requires 'anchor'") {
        "requires 'anchor'"
    } else if message.contains("must not set 'anchor'") {
        "must not set 'anchor'"
    } else if message.contains("mode must be one of") {
        "mode must be one of"
    } else if message.contains("cardinality invalid") {
        "cardinality invalid"
    } else if message.contains("occurrence_policy must be one of") {
        "occurrence_policy must be one of"
    } else if message.contains("no matching field query") {
        "no matching field query"
    } else if message.contains("must be a mapping") {
        "must be a mapping"
    } else {
        message
    }
}

#[test]
fn cardinality_defaults_follow_the_reference_rule() {
    // The anchor is required_one regardless of dtype, `str` is an optional
    // scalar, and anything untyped falls through to a list.
    assert_eq!(
        default_cardinality(Some("str"), true),
        FieldCardinality::RequiredOne
    );
    assert_eq!(
        default_cardinality(None, true),
        FieldCardinality::RequiredOne
    );
    assert_eq!(
        default_cardinality(Some("str"), false),
        FieldCardinality::OptionalOne
    );
    assert_eq!(
        default_cardinality(None, false),
        FieldCardinality::ZeroOrMore
    );
    assert_eq!(
        default_cardinality(Some("int"), false),
        FieldCardinality::ZeroOrMore
    );

    // `is_scalar` and `allows_absent` are what split the decoder's two paths, so
    // they are checked against the enum's own definition rather than trusted.
    for (cardinality, scalar, absent) in [
        (FieldCardinality::OptionalOne, true, true),
        (FieldCardinality::RequiredOne, true, false),
        (FieldCardinality::ZeroOrMore, false, true),
        (FieldCardinality::OneOrMore, false, false),
    ] {
        assert_eq!(cardinality.is_scalar(), scalar, "{cardinality:?} is_scalar");
        assert_eq!(
            cardinality.allows_absent(),
            absent,
            "{cardinality:?} allows_absent"
        );
        assert_eq!(
            FieldCardinality::parse(cardinality.as_str()),
            Some(cardinality),
            "{cardinality:?} round-trips"
        );
    }
    assert_eq!(FieldCardinality::parse("maybe"), None);
}

#[test]
fn an_unannotated_group_compiles_nothing() {
    // The silent path. A group with a `record_metadata` entry but no `mode` is
    // skipped, so it keeps the legacy structure decode rather than becoming a
    // record. If this ever starts compiling a spec, every unannotated
    // json_structures group silently changes meaning.
    let queries = vec![LayoutQuery {
        query_id: 0,
        task_index: 0,
        task_type: "json_structures".into(),
        task_name: "person".into(),
        role_index: 0,
        role_name: "name".into(),
    }];
    let dtypes = BTreeMap::new();
    let specs = compile_record_specs(
        &queries,
        &serde_json::json!({"person": {"anchor": "name"}}),
        &dtypes,
    )
    .expect("an unannotated group is not an error");
    assert!(
        specs.is_empty(),
        "compiled {specs:?} from an unannotated group"
    );
}

#[test]
fn only_json_structures_groups_compile() {
    // `RECORD_TASK_TYPES = ("json_structures",)`. An entities group named in the
    // metadata must not become a record, even with a valid mode.
    let queries = vec![LayoutQuery {
        query_id: 0,
        task_index: 0,
        task_type: "entities".into(),
        task_name: "person".into(),
        role_index: 0,
        role_name: "name".into(),
    }];
    let dtypes = BTreeMap::new();
    let specs = compile_record_specs(
        &queries,
        &serde_json::json!({"person": {"mode": "latent"}}),
        &dtypes,
    )
    .expect("a non-record task type is skipped, not an error");
    assert!(specs.is_empty());
}

#[test]
fn the_fixture_keeps_its_rejection_cases() {
    // Every rule that rejects has to stay in the fixture, or this test quietly
    // stops checking it.
    let data = fixture();
    let names: Vec<&str> = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| case["name"].as_str().unwrap())
        .collect();
    for required in [
        "valid_natural",
        "valid_latent",
        "valid_anchorless",
        "no_metadata_is_a_legacy_no_op",
        "mode_absent_is_a_legacy_no_op",
        "defaults_anchor_dtype_and_fallback",
        "explicit_beats_dtype",
        "two_groups_keep_their_own_anchor",
        "error_natural_without_anchor",
        "error_latent_with_anchor",
        "error_unknown_mode",
        "error_bad_cardinality",
        "error_anchor_not_a_field",
        "error_bad_occurrence_policy",
        "error_metadata_not_a_mapping",
    ] {
        assert!(
            names.contains(&required),
            "the fixture lost its {required:?} case"
        );
    }
}
