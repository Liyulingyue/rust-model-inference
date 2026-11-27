//! Byte-exact parity for `_deduplicate_relation_edges` — the four-stage
//! relation-edge canonicalizer.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_relation_dedup.py`. The reference
//! method is a `@staticmethod` over a list of dicts, so the fixture is a truth
//! table and the test needs no GGUF.
//!
//! Relation proposals deliberately score a capped head x tail cross-product, so
//! the raw list holds every occurrence pairing and every contained partial
//! mention. Four stages remove four redundancies in order, each feeding the next:
//!
//! 1. **Per-side containment** folds a mention into the longest mention
//!    containing it. Head and tail canonicalize independently.
//! 2. **Exact `(h0, h1, t0, t1)` dedup** keeps the higher score. The key is
//!    offsets only, so the survivor's score comes from the winning edge while its
//!    *text* comes from whichever edge last occupied those coordinates — a split
//!    of provenance that is easy to miss and that the fixture pins.
//! 3. **Case/whitespace-folded semantic dedup** ranks by **distance first**, the
//!    opposite of stage 2, so a closer pair beats a higher-scoring one.
//! 4. **Strict token-subset dominance** drops an edge whose one side is a proper
//!    token subset of another's with the other side exactly equal.
//!
//! The result is sorted by `(head_start, tail_start, -score)`.

use rust_model_inference::models::gliner_boundary::relations::{
    deduplicate_relation_edges, ExtractedRelation,
};

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/relation-dedup-golden.json";

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_relation_dedup.py"));
    serde_json::from_str(&raw).expect("parse relation-dedup-golden.json")
}

/// `[head_text, head_start, head_end, tail_text, tail_start, tail_end, score]`.
fn edge(row: &serde_json::Value) -> ExtractedRelation {
    let row = row.as_array().expect("edge row");
    ExtractedRelation {
        relation_type: "works_at".to_string(),
        head_text: row[0].as_str().expect("head_text").to_string(),
        head_start: row[1].as_u64().expect("head_start") as usize,
        head_end: row[2].as_u64().expect("head_end") as usize,
        tail_text: row[3].as_str().expect("tail_text").to_string(),
        tail_start: row[4].as_u64().expect("tail_start") as usize,
        tail_end: row[5].as_u64().expect("tail_end") as usize,
        score: row[6].as_f64().expect("score") as f32,
    }
}

fn summarize(edge: &ExtractedRelation) -> (String, usize, usize, String, usize, usize, f32) {
    (
        edge.head_text.clone(),
        edge.head_start,
        edge.head_end,
        edge.tail_text.clone(),
        edge.tail_start,
        edge.tail_end,
        edge.score,
    )
}

#[test]
fn deduplication_matches_the_reference() {
    let fixture = fixture();
    for case in fixture["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let edges: Vec<ExtractedRelation> = case["edges"]
            .as_array()
            .expect("edges")
            .iter()
            .map(edge)
            .collect();
        let got: Vec<_> = deduplicate_relation_edges(&edges)
            .iter()
            .map(summarize)
            .collect();
        let want: Vec<_> = case["deduped"]
            .as_array()
            .expect("deduped")
            .iter()
            .map(edge)
            .map(|item| {
                (
                    item.head_text,
                    item.head_start,
                    item.head_end,
                    item.tail_text,
                    item.tail_start,
                    item.tail_end,
                    item.score,
                )
            })
            .collect();
        assert_eq!(got, want, "{name}: deduplicated edges differ");
    }
}

/// Fewer than two edges short-circuits, so a single edge and an empty list pass
/// through untouched.
#[test]
fn a_short_list_is_returned_unchanged() {
    assert!(deduplicate_relation_edges(&[]).is_empty());
    let one = vec![edge(&serde_json::json!([
        "Marie", 0, 5, "Paris", 22, 27, 0.9
    ]))];
    assert_eq!(deduplicate_relation_edges(&one), one);
}

/// The four stages, asserted directly so a fixture edit that weakened them fails
/// here rather than quietly making the parity test vacuous.
#[test]
fn each_stage_removes_its_own_redundancy() {
    let fixture = fixture();
    let case = |name: &str| {
        fixture["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["name"] == name)
            .unwrap_or_else(|| panic!("no case {name}"))
            .clone()
    };
    let run = |name: &str| {
        let case = case(name);
        let edges: Vec<ExtractedRelation> =
            case["edges"].as_array().unwrap().iter().map(edge).collect();
        deduplicate_relation_edges(&edges)
    };

    // Stage 1: a contained partial mention folds into the longer one.
    let contained = run("contained_head_folds_to_the_longer_mention");
    assert_eq!(contained.len(), 1);
    assert_eq!(contained[0].head_text, "Marie Curie");
    assert_eq!(
        contained[0].score, 0.8,
        "the longer mention's edge supplies the score"
    );

    // Stage 2: identical offsets, different text and score. The higher score
    // wins, but the text comes from the *last* edge at those coordinates.
    let exact = run("exact_offsets_collapse_ignoring_text");
    assert_eq!(exact.len(), 1);
    assert_eq!(exact[0].score, 0.8, "the higher-scoring edge wins");
    assert_eq!(
        exact[0].head_text, "M. Curie",
        "the text comes from the last edge at those offsets"
    );

    // Stage 3: folded text is the same, and distance — not score — decides.
    let semantic = run("semantic_text_folds_case_and_whitespace");
    assert_eq!(
        semantic.len(),
        1,
        "case and whitespace folding is the same entity"
    );
    assert_eq!(
        semantic[0].tail_start, 22,
        "the closer pair wins even though the other scores higher"
    );
    assert_eq!(semantic[0].score, 0.9);

    // Stage 4: a strict token subset with an equal opposite side is dominated.
    let dominated = run("token_subset_dominance");
    assert_eq!(dominated.len(), 1);
    assert_eq!(
        dominated[0].tail_text, "Paris France",
        "the longer mention survives"
    );

    // ...and on the head side too.
    let head = run("head_token_subset_dominance");
    assert_eq!(head.len(), 1);
    assert_eq!(head[0].head_text, "Paris France");

    // Stage 4's subset must be **strict**: two edges with the same token set on a
    // side both survive. A non-strict comparison would delete both, since each is
    // a subset of the other.
    let equal = run("identical_token_sets_both_survive");
    assert_eq!(
        equal.len(),
        2,
        "an equal token set is not a strict subset, so neither edge is dominated"
    );
}

/// The final order is `(head_start, tail_start, -score)`.
#[test]
fn the_result_is_sorted_by_head_then_tail_then_score() {
    let fixture = fixture();
    let case = fixture["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|case| case["name"] == "final_sort_is_by_head_then_tail_then_score")
        .expect("case")
        .clone();
    let edges: Vec<ExtractedRelation> =
        case["edges"].as_array().unwrap().iter().map(edge).collect();
    let got = deduplicate_relation_edges(&edges);
    for pair in got.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        assert!(
            (a.head_start, a.tail_start, -a.score) <= (b.head_start, b.tail_start, -b.score),
            "out of order: {a:?} then {b:?}"
        );
    }
}

/// Two edges at the same offsets and the same text but different scores collapse
/// to the higher one, and the dedup is on offsets so a differing surface does not
/// save them.
#[test]
fn identical_edges_collapse_to_the_higher_score() {
    let edges = vec![
        edge(&serde_json::json!(["Marie", 0, 5, "Paris", 22, 27, 0.3])),
        edge(&serde_json::json!(["Marie", 0, 5, "Paris", 22, 27, 0.9])),
    ];
    let got = deduplicate_relation_edges(&edges);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].score, 0.9);
}
