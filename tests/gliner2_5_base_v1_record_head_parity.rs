//! Parity for the record head: `RecordHead::forward_group` and `decode_group`.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_record_head.py`, which runs the
//! reference's `RecordHead.forward_group` + `decode_group` on the same 10 cases
//! with the checkpoint's own `record_decoder.*` weights. Only the inputs are
//! synthetic, so no encoder forward pass is needed — but the GGUF is, because the
//! head is what is under test.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf`.

use std::collections::BTreeMap;

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner_boundary::record_head::{
    decode_group, RecordCandidates, RecordDecodeSettings, RecordHead,
};
use rust_model_inference::models::gliner_boundary::record_spec::{
    compile_record_specs, LayoutQuery,
};
use rust_model_inference::models::gliner_boundary::BoundaryModel;

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/record-head-golden.json";
const HIDDEN: usize = 768;
const TOLERANCE: f32 = 1.0e-3;

// The synthetic candidate states are rebuilt from the oracle's formula, so these
// two constants are part of the contract. See the long note beside them in
// `dump_record_head.py`: both values were measured, because getting them wrong
// makes every `latent_seed_head` logit fall below the 0.5 threshold and the whole
// latent path decodes zero records *while still passing*.
const CANDIDATE_STATE_SCALE: f32 = 1.0;
const CANDIDATE_STATE_OFFSET: f32 = 2.0;

/// Returns the leaked source alongside the model: `BoundaryModel` holds
/// zero-copy views, not the mapping, and the record head is loaded straight from
/// the tensors the same way the model's own heads are.
fn loaded_model() -> Option<(
    Box<dyn std::any::Any>,
    BoundaryModel<'static>,
    &'static dyn TensorSource,
)> {
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
    Some((Box::new(()), model, leaked))
}

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_record_head.py"));
    serde_json::from_str(&raw).expect("parse record-head-golden.json")
}

/// `(arange(seq_len * hidden) * 0.011).sin() - 1.0`, the shared synthetic
/// document. `documents()`, not `states()`, because the caller slices rows out of
/// it for the query states.
fn document(seq_len: usize) -> Vec<f32> {
    (0..seq_len * HIDDEN)
        .map(|i| ((i as f32) * 0.011).sin() - 1.0)
        .collect()
}

/// Flatten any nesting depth. `logits` / `valid` are stored per query (`[Q][C]`)
/// because that is how a case is written by hand; the tensors want them flat.
fn numbers(value: &serde_json::Value) -> Vec<f32> {
    let mut out = Vec::new();
    flatten(value, &mut out, &|v| {
        v.as_f64().unwrap_or_else(|| panic!("not a number: {v}")) as f32
    });
    out
}

/// `&dyn Fn` rather than `impl Fn`: passing the closure by value at each
/// recursion level instantiates one monomorphization per depth, which never
/// terminates for a self-recursive function.
fn flatten<T>(value: &serde_json::Value, out: &mut Vec<T>, f: &dyn Fn(&serde_json::Value) -> T) {
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                flatten(item, out, f);
            }
        }
        scalar => out.push(f(scalar)),
    }
}

fn booleans(value: &serde_json::Value) -> Vec<bool> {
    let mut out = Vec::new();
    flatten(value, &mut out, &|v| {
        v.as_bool().unwrap_or_else(|| panic!("not a bool: {v}"))
    });
    out
}

/// Rebuild `[C, hidden]` candidate states from the oracle's formula.
///
/// **Candidate-major**: `RecordCandidates::states` is query-independent, matching
/// `DocumentCandidateBatch::candidate_states`. The oracle's synthetic
/// `candidate_states` are `[1, Q, C, H]` because the reference's batch is, but
/// every query's copy of a slot is the same vector, so one `[C, H]` block
/// reproduces all of them.
fn candidate_states(case: &serde_json::Value, doc: &[f32]) -> Vec<f32> {
    let c_count = case["c_count"].as_u64().unwrap() as usize;
    let seq_len = case["seq_len"].as_u64().unwrap() as usize;
    let mut out = vec![0.0f32; c_count * HIDDEN];
    for c in 0..c_count {
        let mid = 1 + (c * 2) % seq_len.saturating_sub(2).max(1);
        let width = (mid + 2).min(seq_len) - mid;
        let covered = &doc[mid * HIDDEN..][..width * HIDDEN];
        let offset = CANDIDATE_STATE_OFFSET * (c % 5) as f32;
        // `span.mean(dim=0)`: the mean is over the span's *rows*, per dimension.
        // `covered` holds those rows flattened, so row `r`'s value for `dim` sits
        // at `r * HIDDEN + dim`.
        for dim in 0..HIDDEN {
            let total: f32 = (0..width).map(|row| covered[row * HIDDEN + dim]).sum();
            out[c * HIDDEN + dim] = total / width as f32 * CANDIDATE_STATE_SCALE + offset;
        }
    }
    out
}

fn flatten_indices(case: &serde_json::Value) -> Vec<usize> {
    let mut out = Vec::new();
    for query in case["candidates"].as_array().unwrap() {
        for span in query.as_array().unwrap() {
            out.push(span[0].as_u64().unwrap() as usize);
            out.push(span[1].as_u64().unwrap() as usize);
        }
    }
    out
}

fn layout(case: &serde_json::Value) -> Vec<LayoutQuery> {
    case["roles"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, role)| LayoutQuery {
            query_id: index,
            task_index: 0,
            task_type: "json_structures".into(),
            task_name: "record".into(),
            role_index: index,
            role_name: role.as_str().unwrap().to_string(),
        })
        .collect()
}

#[test]
fn record_head_and_decode_match_the_reference() {
    let Some((_anchor, model, source)) = loaded_model() else {
        return;
    };
    let data = fixture();
    let head = RecordHead::load(
        source,
        HIDDEN,
        model.settings.record_dim,
        model.settings.record_instance_queries,
    )
    .expect("load record_decoder");

    for case in data["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap();
        let q_count = case["q_count"].as_u64().unwrap() as usize;
        let c_count = case["c_count"].as_u64().unwrap() as usize;
        let seq_len = case["seq_len"].as_u64().unwrap() as usize;
        let doc = document(seq_len);

        let specs = compile_record_specs(&layout(case), &case["record_metadata"], &BTreeMap::new())
            .unwrap_or_else(|error| panic!("{name}: compile specs: {error}"));
        let spec = specs.values().next().expect("one spec");
        assert_eq!(spec.mode, case["mode"].as_str().unwrap(), "{name}: mode");
        assert_eq!(
            spec.anchor_query_id,
            case["anchor_query_id"].as_u64().map(|v| v as usize),
            "{name}: anchor_query_id"
        );

        let query_states = doc[..q_count * HIDDEN].to_vec();
        let group = head
            .forward_group(
                &spec,
                &query_states,
                &RecordCandidates {
                    indices: &flatten_indices(case),
                    pair_logits: &numbers(&case["logits"]),
                    valid_mask: &booleans(&case["valid"]),
                    states: &candidate_states(case, &doc),
                    q_count,
                    c_count,
                },
            )
            .unwrap_or_else(|error| panic!("{name}: forward_group: {error}"));

        // --- object logits ---
        let want_object = numbers(&case["object_logits"]);
        assert_eq!(
            group.object_logits.len(),
            want_object.len(),
            "{name}: instance count (mode {})",
            case["mode"]
        );
        let object_deltas: Vec<f32> = group
            .object_logits
            .iter()
            .zip(&want_object)
            .map(|(got, want)| (got - want).abs())
            .collect();
        let worst = object_deltas
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, delta)| (*delta, index))
            .unwrap_or((0.0, 0));
        assert!(
            worst.0 < TOLERANCE,
            "{name}: worst object logit delta {} at [{}] (first 6 got {:?} want {:?})",
            worst.0,
            worst.1,
            &group.object_logits[..6.min(group.object_logits.len())],
            &want_object[..6.min(want_object.len())]
        );

        // --- assignment logits, as a flat [fields * instances, width] block ---
        let want_shape: Vec<usize> = case["assign_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let fields = want_shape[0];
        let instances = want_shape[1];
        let width = want_shape[2];
        assert_eq!(
            group.assign_logits.len(),
            fields,
            "{name}: assign field count"
        );
        for (field, rows) in group.assign_logits.iter().enumerate() {
            assert_eq!(
                rows.len(),
                instances,
                "{name}: assign rows for field {field}"
            );
            for row in rows {
                assert_eq!(row.len(), width, "{name}: assign width for field {field}");
                // Column 0 is the null column, so width == candidates + 1
                // whenever the field has any.
                assert!(width >= 1, "{name}: assign needs the null column");
            }
        }
        let want_assign = numbers(&case["assign_logits"]);
        let mut cursor = 0;
        for rows in &group.assign_logits {
            for row in rows {
                for value in row {
                    assert!(
                        (value - want_assign[cursor]).abs() < TOLERANCE,
                        "{name}: assign logit {value} vs {}",
                        want_assign[cursor]
                    );
                    cursor += 1;
                }
            }
        }
        assert_eq!(cursor, want_assign.len(), "{name}: assign logit count");

        // --- instance spans ---
        let want_spans = case["instance_spans"].as_array().expect("instance_spans");
        assert_eq!(
            group.instance_spans.len(),
            want_spans.len(),
            "{name}: instance spans"
        );
        for (index, (got, want)) in group.instance_spans.iter().zip(want_spans).enumerate() {
            match (got, want.is_null()) {
                (Some((start, end)), false) => {
                    assert_eq!(
                        *start as u64,
                        want[0].as_u64().unwrap(),
                        "{name}[{index}]: instance span start"
                    );
                    assert_eq!(
                        *end as u64,
                        want[1].as_u64().unwrap(),
                        "{name}[{index}]: instance span end"
                    );
                }
                (None, true) => {}
                _ => panic!("{name}[{index}]: instance span presence differs: {got:?} vs {want}"),
            }
        }

        // --- decode ---
        let decode = &case["decode"];
        let settings = RecordDecodeSettings {
            anchor_threshold: decode["anchor_threshold"].as_f64().unwrap() as f32,
            object_threshold: decode["object_threshold"].as_f64().unwrap() as f32,
            field_threshold: decode["field_threshold"].as_f64().unwrap() as f32,
            temperature: decode["temperature"].as_f64().unwrap() as f32,
        };
        let records = decode_group(&group, settings)
            .unwrap_or_else(|error| panic!("{name}: decode_group: {error}"));
        let want_records = case["records"].as_array().expect("records");
        // Pair by field set. The only known divergence is *which index* carries
        // the row that lost the exclusive-field tie-break, and pairing by field
        // set makes that visible as an ordering note rather than a false
        // "these fields differ" failure. Every span and score below is still
        // compared exactly.
        // A queue per field set, not a single entry: several records legitimately
        // share a field set (two instances binding the same spans), and popping
        // keeps the multiplicity instead of silently comparing one of them twice.
        let mut want_by_signature: std::collections::BTreeMap<String, Vec<&serde_json::Value>> =
            std::collections::BTreeMap::new();
        for want in want_records {
            let keys: Vec<String> = want["fields"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect();
            want_by_signature
                .entry(format!("{keys:?}"))
                .or_default()
                .push(want);
        }
        for (index, got) in records.iter().enumerate() {
            let keys: Vec<String> = got.fields.keys().map(|k| k.to_string()).collect();
            // Front, not back: records sharing a field set must still pair up in
            // their original order, otherwise two identical shapes swap scores.
            let want = want_by_signature
                .get_mut(&format!("{keys:?}"))
                .and_then(|queue| (!queue.is_empty()).then(|| queue.remove(0)))
                .unwrap_or_else(|| {
                    panic!("{name}[{index}]: no reference record has fields {keys:?}")
                });
            assert!(
                (got.score - want["score"].as_f64().unwrap() as f32).abs() < TOLERANCE,
                "{name}[{index}]: record score"
            );
            match (got.anchor_span, want["anchor_span"].is_null()) {
                (Some((start, end)), false) => {
                    assert_eq!(start as u64, want["anchor_span"][0].as_u64().unwrap());
                    assert_eq!(end as u64, want["anchor_span"][1].as_u64().unwrap());
                }
                (None, true) => {}
                _ => panic!("{name}[{index}]: anchor_span presence differs"),
            }
            let want_fields = want["fields"].as_object().expect("record fields");
            assert_eq!(
                got.fields.len(),
                want_fields.len(),
                "{name}[{index}]: field count; got {:?} want {:?}",
                got.fields.keys().collect::<Vec<_>>(),
                want_fields.keys().collect::<Vec<_>>()
            );
            for (query_id, spans) in &got.fields {
                let key = query_id.to_string();
                let want_spans = want_fields
                    .get(&key)
                    .unwrap_or_else(|| panic!("{name}[{index}]: no reference field {key}"))
                    .as_array()
                    .expect("spans");
                assert_eq!(
                    spans.len(),
                    want_spans.len(),
                    "{name}[{index}]: field {key} span count"
                );
                for (span_index, (span, want_span)) in spans.iter().zip(want_spans).enumerate() {
                    assert_eq!(
                        span.0 as u64,
                        want_span[0].as_u64().unwrap(),
                        "{name}[{index}]: field {key} span {span_index} start"
                    );
                    assert_eq!(
                        span.1 as u64,
                        want_span[1].as_u64().unwrap(),
                        "{name}[{index}]: field {key} span {span_index} end"
                    );
                }
                // Half-open, non-empty, and inside the document.
                for (span_index, (start, end)) in spans.iter().enumerate() {
                    assert!(
                        start < end && *end <= seq_len,
                        "{name}[{index}]: field {key} span {span_index} out of range: \
                         {start}..{end} of {seq_len}"
                    );
                }
            }
        }
    }
}

#[test]
fn the_fixture_keeps_its_discriminating_cases() {
    // Two of these cases degenerated to "0 records" the first time round, which
    // left the whole latent and assignment path untested while still passing. A
    // fixture that silently stops exercising anything is worse than no fixture.
    let data = fixture();
    let cases = data["cases"].as_array().unwrap();
    let mut producing = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let instances = case["object_logits"].as_array().unwrap().len();
        let records = case["records"].as_array().unwrap().len();
        match name {
            "natural_anchor_absent_from_candidates" => {
                // The one case that *should* have no instances: every anchor
                // candidate is invalid, so there is nothing to form one from.
                assert_eq!(instances, 0, "{name}: no instances expected");
                assert_eq!(records, 0, "{name}: no records expected");
            }
            "anchorless_without_candidates_isolates_instance_embed" => {
                // 32 instances but no records is the *point*: with every candidate
                // invalid the attention context is empty, so the object logits
                // are `object_head(instance_embed)` alone. That isolates the
                // parameter from the attention path, and no candidate means no
                // field can bind, so no record survives.
                assert_eq!(instances, 32, "{name}: instance_queries instances");
                assert_eq!(records, 0, "{name}: nothing to bind without candidates");
            }
            _ => {
                assert!(
                    records > 0,
                    "{name} decoded no records, so it no longer exercises the decoder"
                );
                producing += 1;
            }
        }
    }
    assert!(
        producing >= 9,
        "only {producing} cases produce records; the fixture has lost coverage"
    );
}

#[test]
fn anchorless_uses_the_object_threshold_not_the_anchor_threshold() {
    // The case sets the two thresholds far apart (0.99 vs 0.1) precisely so a
    // decoder that reached for the wrong one selects a different set.
    let data = fixture();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "anchorless_uses_object_threshold")
        .expect("the anchorless case");
    assert_eq!(case["decode"]["anchor_threshold"].as_f64().unwrap(), 0.99);
    assert_eq!(case["decode"]["object_threshold"].as_f64().unwrap(), 0.1);
    // 32 instance queries is `record_instance_queries`; none of them is seeded
    // from a span, so no record may carry an anchor.
    assert_eq!(case["instance_spans"].as_array().unwrap().len(), 32);
    assert!(
        case["instance_spans"]
            .as_array()
            .unwrap()
            .iter()
            .all(|span| span.is_null()),
        "anchorless instances must not be seeded from spans"
    );
    for record in case["records"].as_array().unwrap() {
        assert!(
            record["anchor_span"].is_null(),
            "an anchorless record must have no anchor span"
        );
    }
}

#[test]
fn natural_mode_passes_the_pool_logits_through() {
    // In `natural` mode the object logits are the anchor field's `pair_logits`
    // verbatim. If `object_head` were applied instead, these would be different
    // numbers — this is the case that catches it.
    let data = fixture();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "natural_two_fields")
        .expect("the natural case");
    let anchor_logits = numbers(&case["logits"])[..3].to_vec();
    let object_logits = numbers(&case["object_logits"]);
    assert_eq!(
        object_logits, anchor_logits,
        "natural mode must forward the anchor candidates' pair_logits unchanged"
    );
}
