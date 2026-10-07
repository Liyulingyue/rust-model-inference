//! Byte-exact parity for `_record_local_choice_mentions`.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_record_choice_mentions.py`. The
//! reference calls this a pure function of `(text, choices, anchor_char_spans)`,
//! so the fixture is a truth table and the test needs no GGUF.
//!
//! Three rules the fixture pins, each a way a plausible port diverges:
//!
//! 1. **A choice is a whole word** — `(?<!\w)…(?!\w)`, case-insensitively — so
//!    `paris` is not reported inside `Parisian`.
//! 2. **Ownership is by the *preceding* anchor, not the nearest.** A mention
//!    between two anchors goes to the earlier one, and one before the first
//!    anchor binds to the first.
//! 3. **Values are semantic sets**: within a record a value is kept once, at its
//!    first source occurrence, in source order, and reported with the *declared*
//!    casing rather than the document's.
//!
//! This is the document-level half of record-mode `choices`. The other half —
//! falling back to the schema-prefix enum tokens when the document mentions no
//! choice — is a separate stage and is **not** covered here; see
//! `GLINER_ADAPT_PLAN.md` F-1.

use rust_model_inference::models::gliner_boundary::structure::record_local_choice_mentions;

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/record-choice-mentions-golden.json";

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE).unwrap_or_else(|_| {
        panic!("missing {FIXTURE}; regenerate with dump_record_choice_mentions.py")
    });
    serde_json::from_str(&raw).expect("parse record-choice-mentions-golden.json")
}

fn case<'a>(cases: &'a [serde_json::Value], name: &str) -> &'a serde_json::Value {
    cases
        .iter()
        .find(|case| case["name"] == name)
        .unwrap_or_else(|| panic!("no case {name}"))
}

#[test]
fn assignment_matches_the_reference() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(!cases.is_empty());

    for entry in cases {
        let name = entry["name"].as_str().expect("name");
        let text = entry["text"].as_str().expect("text");
        let choices: Vec<String> = entry["choices"]
            .as_array()
            .expect("choices")
            .iter()
            .map(|choice| choice.as_str().expect("choice").to_string())
            .collect();
        let anchors: Vec<Option<(usize, usize)>> = entry["anchors"]
            .as_array()
            .expect("anchors")
            .iter()
            .map(|anchor| {
                // An unresolved anchor is serialized as `[null]`; a resolved one
                // is `[start, end]`.
                let pair = anchor.as_array().expect("anchor is an array");
                if pair.len() == 1 && pair[0].is_null() {
                    return None;
                }
                Some((
                    pair[0].as_u64().expect("start") as usize,
                    pair[1].as_u64().expect("end") as usize,
                ))
            })
            .collect();

        let (has_literal, assigned) = record_local_choice_mentions(text, &choices, &anchors);
        assert_eq!(
            has_literal,
            entry["has_literal_choices"].as_bool().expect("has"),
            "{name}: has_literal_choices"
        );

        let want = entry["assigned"].as_object().expect("assigned");
        assert_eq!(
            assigned.len(),
            want.len(),
            "{name}: record count with assignments; port {assigned:?} vs reference {want:?}"
        );
        for (index, mentions) in &assigned {
            let expected = want
                .get(&index.to_string())
                .unwrap_or_else(|| panic!("{name}: reference has no record {index}"))
                .as_array()
                .expect("mentions");
            assert_eq!(
                mentions.len(),
                expected.len(),
                "{name}: record {index} mention count"
            );
            for (position, (choice, start, end)) in mentions.iter().enumerate() {
                let want = &expected[position];
                assert_eq!(
                    choice,
                    want[0].as_str().expect("choice"),
                    "{name}: record {index} mention {position} value"
                );
                assert_eq!(
                    (*start, *end),
                    (
                        want[1].as_u64().expect("start") as usize,
                        want[2].as_u64().expect("end") as usize
                    ),
                    "{name}: record {index} mention {position} span"
                );
            }
        }
    }
}

/// The three rules, asserted directly so a fixture edit that weakened them fails
/// here rather than quietly making the parity test vacuous.
#[test]
fn the_rules_hold_on_the_cases_that_target_them() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");

    // 1. Whole-word: "Parisian" must not report, and only the real "Paris" does.
    let boundary = case(cases, "word_boundary_excludes_substring");
    let (has, assigned) = record_local_choice_mentions(
        boundary["text"].as_str().unwrap(),
        &["paris".to_string()],
        &[Some((0, 14))],
    );
    assert!(has);
    assert_eq!(
        assigned[&0],
        vec![("paris".to_string(), 27, 32)],
        "only the standalone occurrence is a mention"
    );

    // 2. Preceding anchor, not nearest.
    let preceding = case(cases, "two_anchors_mention_between_belongs_to_preceding");
    let (_, assigned) = record_local_choice_mentions(
        preceding["text"].as_str().unwrap(),
        &["books".to_string()],
        &[(0, 6).into(), (19, 25).into()],
    );
    assert_eq!(
        assigned.keys().copied().collect::<Vec<_>>(),
        vec![0],
        "a mention between two anchors belongs to the preceding one"
    );

    // A mention before the first anchor binds to the first.
    let before = case(cases, "mention_before_first_anchor_binds_to_first");
    let (_, assigned) = record_local_choice_mentions(
        before["text"].as_str().unwrap(),
        &["paris".to_string()],
        &[(12, 17).into()],
    );
    assert_eq!(assigned[&0], vec![("paris".to_string(), 3, 8)]);

    // 3. Semantic set: one occurrence per value per record, first one kept.
    let dedup = case(cases, "repeated_value_in_one_record_kept_once");
    let (_, assigned) = record_local_choice_mentions(
        dedup["text"].as_str().unwrap(),
        &["paris".to_string()],
        &[(0, 5).into()],
    );
    assert_eq!(assigned[&0], vec![("paris".to_string(), 0, 5)]);

    // Declared casing is reported, not the document's.
    let casing = case(cases, "case_insensitive_but_reported_verbatim");
    let (_, assigned) = record_local_choice_mentions(
        casing["text"].as_str().unwrap(),
        &["Paris".to_string()],
        &[(0, 10).into()],
    );
    assert_eq!(
        assigned[&0],
        vec![("Paris".to_string(), 11, 16)],
        "the declared literal is reported even when the text differs in case"
    );
}

/// A record whose anchor could not be resolved owns nothing, and its presence
/// does not stop the others from being assigned.
#[test]
fn an_unresolved_anchor_owns_nothing() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    let case = case(cases, "anchor_none_is_skipped");
    let (has, assigned) = record_local_choice_mentions(
        case["text"].as_str().unwrap(),
        &["paris".to_string()],
        &[None, Some((13, 17))],
    );
    assert!(
        has,
        "a mention exists even though the first anchor is unresolved"
    );
    assert!(
        !assigned.contains_key(&0),
        "the record with no anchor owns nothing"
    );
    assert_eq!(assigned[&1], vec![("paris".to_string(), 0, 5)]);
}

/// No mention anywhere means the caller must fall back to the schema-prefix
/// enum tokens, which is exactly what the `has_literal_choices` flag signals.
#[test]
fn no_mention_signals_the_prefix_fallback() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    for name in [
        "no_mention_anywhere",
        "all_choices_absent",
        "empty_choices",
        "empty_text",
    ] {
        let case = case(cases, name);
        // Each case's own choices — `empty_choices` declares none, and a
        // hardcoded list here would assert a mention exists where there is none.
        let choices: Vec<String> = case["choices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().to_string())
            .collect();
        let (has, assigned) = record_local_choice_mentions(
            case["text"].as_str().unwrap(),
            &choices,
            &[(0, 5).into()],
        );
        assert!(!has, "{name}: nothing in the document mentions a choice");
        assert!(assigned.is_empty(), "{name}: nothing to assign");
    }
}

/// A choice containing regex metacharacters must be matched literally.
#[test]
fn a_choice_is_matched_literally() {
    let (has, assigned) = record_local_choice_mentions(
        "the a.b and axb both appear",
        &["a.b".to_string()],
        &[Some((0, 3))],
    );
    assert!(has);
    assert_eq!(
        assigned[&0],
        vec![("a.b".to_string(), 4, 7)],
        "`a.b` is a literal, so `axb` must not match"
    );
}
