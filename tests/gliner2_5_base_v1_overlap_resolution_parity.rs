//! Byte-exact parity for `resolve_overlaps` — the span-conflict resolver.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_overlap_resolution.py`, which
//! calls the reference `resolve_overlaps` on hand-written cases. This is a pure
//! function, so unlike the model fixtures it needs no GGUF and always runs.
//!
//! The `disallow` policy (what base-v1's `overlap_policy = "flat"` normalizes
//! to) is weighted interval scheduling, not a greedy pass. The fixture includes a
//! case where greedy provably loses — three crossing spans where the
//! highest-scoring middle span blocks both outer ones, so the optimum is the two
//! outer spans — because a greedy implementation passes every other case.

use rust_model_inference::models::gliner_boundary::overlap::{
    normalize_overlap_policy, resolve_overlaps, OverlapPolicy, ScoredSpan,
};

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/overlap-resolution-golden.json";

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE).unwrap_or_else(|_| {
        panic!("missing {FIXTURE}; regenerate with dump_overlap_resolution.py")
    });
    serde_json::from_str(&raw).expect("parse overlap-resolution-golden.json")
}

fn span(value: &serde_json::Value) -> ScoredSpan {
    let row = value.as_array().expect("span triple");
    assert_eq!(row.len(), 3, "a span is (score, start, end)");
    ScoredSpan {
        score: row[0].as_f64().unwrap() as f32,
        start: row[1].as_u64().unwrap() as usize,
        end: row[2].as_u64().unwrap() as usize,
    }
}

fn canonical_name(policy: OverlapPolicy) -> &'static str {
    match policy {
        OverlapPolicy::Allow => "allow",
        OverlapPolicy::Nested => "nested",
        OverlapPolicy::Disallow => "disallow",
        OverlapPolicy::Longest => "longest",
    }
}

#[test]
fn matches_the_reference_on_every_case() {
    let fixture = fixture();
    for case in fixture["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let policy_name = case["policy"].as_str().expect("policy");
        let items: Vec<ScoredSpan> = case["items"]
            .as_array()
            .expect("items")
            .iter()
            .map(span)
            .collect();
        let want: Vec<ScoredSpan> = case["kept"]
            .as_array()
            .expect("kept")
            .iter()
            .map(span)
            .collect();

        // `default = "disallow"` matches the reference call, so a `None` policy
        // and an explicit `"flat"` both land on the architecture default.
        let policy = normalize_overlap_policy(Some(policy_name), "disallow")
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(
            canonical_name(policy),
            case["canonical"].as_str().unwrap(),
            "{name}: canonical policy name"
        );

        let kept = resolve_overlaps(&items, policy);
        let got: Vec<ScoredSpan> = kept.into_iter().map(|index| items[index]).collect();
        assert_eq!(got, want, "{name}: policy {policy_name}");
    }
}

#[test]
fn greedy_would_get_the_crossing_case_wrong() {
    // The case that separates weighted interval scheduling from "take the
    // highest score, repeat". Middle span scores highest and blocks both ends,
    // so taking it yields 0.5 while the true optimum is 0.55 + 0.45 = 1.0.
    let items = vec![
        ScoredSpan {
            score: 0.55,
            start: 0,
            end: 4,
        },
        ScoredSpan {
            score: 0.50,
            start: 3,
            end: 7,
        },
        ScoredSpan {
            score: 0.45,
            start: 6,
            end: 10,
        },
    ];
    let policy = OverlapPolicy::Disallow;
    let kept: Vec<ScoredSpan> = resolve_overlaps(&items, policy)
        .into_iter()
        .map(|index| items[index])
        .collect();
    assert_eq!(
        kept,
        vec![items[0], items[2]],
        "the two outer spans must both survive"
    );
    let total: f32 = kept.iter().map(|s| s.score).sum();
    assert!(
        (total - 1.0).abs() < 1e-6,
        "optimum total is {total}, not 1.0"
    );

    // And the same input under `allow` keeps all three, so the policy really is
    // what decides, not a filter inside the resolver.
    let all: Vec<ScoredSpan> = resolve_overlaps(&items, OverlapPolicy::Allow)
        .into_iter()
        .map(|index| items[index])
        .collect();
    assert_eq!(all.len(), 3);
}

#[test]
fn unknown_policies_are_rejected() {
    let fixture = fixture();
    for row in fixture["unknown_policies"]
        .as_array()
        .expect("unknown_policies")
    {
        let policy = row["policy"].as_str().expect("policy");
        let want = row["error"].as_str().expect("error");
        let got = normalize_overlap_policy(Some(policy), "disallow")
            .expect_err("the reference raises here");
        // Not string equality: Python's `repr` and Rust's `{:?}` quote the
        // offending name differently, and pinning that would make this a test of
        // the formatter. What matters is that it errors, names the bad value,
        // and lists what is supported.
        assert!(
            got.contains(policy.trim()) || got.contains(policy),
            "policy {policy:?}: the error should name it, got {got:?}"
        );
        for supported in ["allow", "nested", "flat", "longest"] {
            assert!(
                got.contains(supported),
                "policy {policy:?}: the error should list {supported:?}, got {got:?}"
            );
        }
        assert!(
            !want.is_empty(),
            "the reference records a message for {policy:?}"
        );
    }
}

#[test]
fn aliases_and_the_default_resolve_the_same_way() {
    let items = vec![
        ScoredSpan {
            score: 0.6,
            start: 0,
            end: 3,
        },
        ScoredSpan {
            score: 0.9,
            start: 2,
            end: 5,
        },
    ];
    let keep = |policy: OverlapPolicy| {
        resolve_overlaps(&items, policy)
            .into_iter()
            .map(|index| items[index])
            .collect::<Vec<_>>()
    };
    let expected = vec![items[1]];
    for alias in [
        "flat",
        "disallow",
        "no-overlap",
        "non_overlapping",
        "no_overlap",
    ] {
        let policy = normalize_overlap_policy(Some(alias), "disallow").unwrap();
        assert_eq!(keep(policy), expected, "alias {alias:?}");
    }
    // `None` resolves through the default, and the default itself is a legal
    // name — which is how the reference preserves an architecture default
    // instead of imposing one.
    assert_eq!(
        normalize_overlap_policy(None, "flat").unwrap(),
        OverlapPolicy::Disallow
    );
    assert_eq!(
        normalize_overlap_policy(None, "longest").unwrap(),
        OverlapPolicy::Longest
    );
    // A missing default is an error, not a silent "allow".
    assert!(normalize_overlap_policy(None, "nonsense").is_err());
}
