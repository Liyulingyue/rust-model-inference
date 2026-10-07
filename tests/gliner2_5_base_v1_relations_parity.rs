//! Byte-exact parity for the relation head — `TypedRelationPairGenerator` and
//! `SparseRelationScorer`.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_relations.py`, which runs the
//! reference pair generator and scorer on the same synthetic cases.
//!
//! Needs the GGUF (the scorer has weights); the pair generator alone is pure,
//! but it is checked together with the scorer so a relation that is *proposed*
//! and then *scored* wrong cannot hide.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf`.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner_boundary::relations::{
    generate_typed_relation_pairs, RelationCandidates, RelationProposalSettings, RelationTypeSpec,
};
use rust_model_inference::models::gliner_boundary::BoundaryModel;

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/relations-golden.json";
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
    let raw = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_relations.py"));
    serde_json::from_str(&raw).expect("parse relations-golden.json")
}

/// `(arange(seq_len * hidden) * 0.011).sin() - 1.0`, the shared synthetic
/// document every boundary oracle uses.
fn synthetic_states(seq_len: usize, hidden: usize) -> Vec<f32> {
    (0..seq_len * hidden)
        .map(|i| ((i as f32) * 0.011).sin() - 1.0)
        .collect()
}

/// Flatten any nesting depth. The fixture keeps `logits` and `valid` shaped
/// per query (`[Q][C]`) because that is how a case is written by hand; the
/// candidate tensors want them flat in `[Q, C]` order.
fn numbers(value: &serde_json::Value) -> Vec<f32> {
    let mut out = Vec::new();
    flatten(value, &mut out, &|v| {
        v.as_f64().unwrap_or_else(|| panic!("not a number: {v}")) as f32
    });
    out
}

fn booleans(value: &serde_json::Value) -> Vec<bool> {
    let mut out = Vec::new();
    flatten(value, &mut out, &|v| {
        v.as_bool().unwrap_or_else(|| panic!("not a bool: {v}"))
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

#[test]
fn relation_settings_match_the_checkpoint() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let settings = RelationProposalSettings::from_settings(&model.settings);
    // These are the values the fixture was generated with. If a converter stops
    // transcribing one, this fails here rather than as a silently wider or
    // narrower pair set.
    assert_eq!(settings.heads_per_relation, 32);
    assert_eq!(settings.tails_per_relation, 32);
    assert_eq!(settings.pair_cap, 64);
    // The reference's dataclass default is 0.0; base-v1's published value is
    // 0.2, and reading a default instead would admit far too many arguments.
    assert_eq!(settings.argument_threshold, 0.2);
    assert!(model.settings.relation_biaffine_content);
    assert!(model.settings.directional_relation_states);
    assert_eq!(model.settings.relation_query_dim(model.config.n_embd), 1536);
}

#[test]
fn typed_relation_pairs_match_the_reference() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let data = fixture();
    let settings = RelationProposalSettings::from_settings(&model.settings);
    let cases = data["cases"].as_array().expect("cases");
    assert!(!cases.is_empty(), "the fixture must exercise something");

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let q_count = case["q_count"].as_u64().unwrap() as usize;
        let c_count = case["c_count"].as_u64().unwrap() as usize;

        // `[Q, C, 2]` start/end pairs.
        let mut indices = Vec::with_capacity(q_count * c_count * 2);
        for query in case["candidates"].as_array().unwrap() {
            for span in query.as_array().unwrap() {
                indices.push(span[0].as_u64().unwrap() as usize);
                indices.push(span[1].as_u64().unwrap() as usize);
            }
        }
        assert_eq!(indices.len(), q_count * c_count * 2, "{name}: indices");

        let logits = numbers(&case["logits"]);
        assert_eq!(logits.len(), q_count * c_count, "{name}: logits");
        let valid = booleans(&case["valid"]);
        assert_eq!(valid.len(), q_count * c_count, "{name}: valid");

        let specs: Vec<RelationTypeSpec> = case["specs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|spec| {
                RelationTypeSpec::two_role(
                    spec[0].as_str().unwrap(),
                    spec[1].as_u64().unwrap() as usize,
                    spec[2].as_u64().unwrap() as usize,
                )
            })
            .collect();

        let candidates = RelationCandidates {
            indices: &indices,
            pair_logits: &logits,
            valid_mask: &valid,
            q_count,
            c_count,
        };
        let pairs = generate_typed_relation_pairs(&candidates, &specs, &settings);

        let expected = case["pairs"].as_array().expect("pairs");
        assert_eq!(
            pairs.len(),
            expected.len(),
            "{name}: pair count (the reference proposed {})",
            case["pair_count"]
        );

        for (index, (pair, want)) in pairs.iter().zip(expected).enumerate() {
            let relation_type = case["relation_types"][index].as_str().unwrap();
            assert_eq!(
                specs[pair.relation_index].relation_type, relation_type,
                "{name}[{index}]: relation type"
            );
            assert_eq!(
                pair.head_start,
                want[0].as_u64().unwrap() as usize,
                "{name}[{index}] head_start"
            );
            assert_eq!(
                pair.head_end,
                want[1].as_u64().unwrap() as usize,
                "{name}[{index}] head_end"
            );
            assert_eq!(
                pair.tail_start,
                want[2].as_u64().unwrap() as usize,
                "{name}[{index}] tail_start"
            );
            assert_eq!(
                pair.tail_end,
                want[3].as_u64().unwrap() as usize,
                "{name}[{index}] tail_end"
            );
            assert!(
                (pair.head_prob - want[4].as_f64().unwrap() as f32).abs() < TOLERANCE,
                "{name}[{index}] head_prob {} vs {}",
                pair.head_prob,
                want[4]
            );
            assert!(
                (pair.tail_prob - want[5].as_f64().unwrap() as f32).abs() < TOLERANCE,
                "{name}[{index}] tail_prob {} vs {}",
                pair.tail_prob,
                want[5]
            );
            // A head and tail that are the same mention must never survive.
            assert!(
                pair.head_start != pair.tail_start || pair.head_end != pair.tail_end,
                "{name}[{index}]: a same-span pair survived the generator"
            );
        }
    }
}

#[test]
fn relation_scorer_logits_match_the_reference() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let data = fixture();
    let hidden = model.config.n_embd;
    let settings = RelationProposalSettings::from_settings(&model.settings);
    let scorer = model
        .relation_scorer
        .as_ref()
        .expect("the checkpoint sets enable_relations, so the scorer is loaded");

    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let seq_len = case["seq_len"].as_u64().unwrap() as usize;
        let q_count = case["q_count"].as_u64().unwrap() as usize;
        let c_count = case["c_count"].as_u64().unwrap() as usize;

        let mut indices = Vec::with_capacity(q_count * c_count * 2);
        for query in case["candidates"].as_array().unwrap() {
            for span in query.as_array().unwrap() {
                indices.push(span[0].as_u64().unwrap() as usize);
                indices.push(span[1].as_u64().unwrap() as usize);
            }
        }
        let logits_in = numbers(&case["logits"]);
        let valid = booleans(&case["valid"]);
        let specs: Vec<RelationTypeSpec> = case["specs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|spec| {
                RelationTypeSpec::two_role(
                    spec[0].as_str().unwrap(),
                    spec[1].as_u64().unwrap() as usize,
                    spec[2].as_u64().unwrap() as usize,
                )
            })
            .collect();
        let candidates = RelationCandidates {
            indices: &indices,
            pair_logits: &logits_in,
            valid_mask: &valid,
            q_count,
            c_count,
        };
        let pairs = generate_typed_relation_pairs(&candidates, &specs, &settings);

        let states = synthetic_states(seq_len, hidden);
        // `_build_rel_specs`: the relation query state is the concatenation of
        // its own head and tail role states (directional_relation_states), not
        // their mean. `states[i]` is row i, so the head and tail states are two
        // disjoint halves of one 1536-wide vector.
        let role_width = model.settings.relation_query_dim(hidden);
        let mut relation_states = Vec::new();
        for spec in &specs {
            let head = spec.head_query_ids[0];
            let tail = spec.tail_query_ids[0];
            if model.settings.directional_relation_states {
                relation_states.extend_from_slice(&states[head * hidden..][..hidden]);
                relation_states.extend_from_slice(&states[tail * hidden..][..hidden]);
            } else {
                for dim in 0..hidden {
                    relation_states
                        .push((states[head * hidden + dim] + states[tail * hidden + dim]) / 2.0);
                }
            }
        }
        assert_eq!(relation_states.len(), specs.len() * role_width);

        let got = scorer.forward(&states, &relation_states, &pairs);
        let expected = case["pairs"].as_array().unwrap();
        assert_eq!(got.len(), expected.len(), "{name}: scored pair count");
        for (index, want) in expected.iter().enumerate() {
            let reference = want[6].as_f64().unwrap() as f32;
            assert!(
                (got[index] - reference).abs() < TOLERANCE,
                "{name}[{index}]: relation logit {} vs {reference} (delta {})",
                got[index],
                (got[index] - reference).abs()
            );
        }
    }
}
