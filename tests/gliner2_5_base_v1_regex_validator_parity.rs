//! Byte-exact parity for `RegexValidator` — the schema-level span filter.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_regex_validator.py`, which is a
//! truth table rather than a model run: `validate` is a pure function of
//! `(pattern, mode, exclude, flags, text)`. No GGUF, always runs.
//!
//! Two engine differences are measured rather than assumed, and both are
//! recorded in the fixture:
//!
//! - Python's `re.IGNORECASE` matches `İ` (U+0130) and `ı` (U+0131) against a
//!   pattern `i`. Rust's `(?i)` does not — Unicode simple case folding treats
//!   them as distinct letters, and Rust is the more correct of the two. This
//!   port does **not** paper over it: `dotted_i_diverges_from_python` asserts the
//!   disagreement, so the gap is a tracked boundary instead of a surprise.
//! - A user-written `\w` includes combining marks under the `regex` crate and
//!   does not under Python. Same treatment.
//!
//! Kelvin sign, long s, the Angstrom sign, `fullmatch` anchoring, and `.`
//! versus a newline all agree, and those are asserted for real parity.

use rust_model_inference::models::gliner_boundary::validator::{
    CompiledValidators, RegexValidator, ValidatorMode, IGNORECASE_DIVERGENCES,
};

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/regex-validator-golden.json";

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_regex_validator.py"));
    serde_json::from_str(&raw).expect("parse regex-validator-golden.json")
}

/// The exact `(case, text)` pairs where this port knowingly differs from
/// Python's `re`.
///
/// Listed explicitly rather than filtered by a character class, so a new
/// divergence shows up as a failure in this test instead of being silently
/// skipped. Each is asserted the other way round by a dedicated test.
const DIVERGENT: [(&str, &str); 3] = [
    // Turkish dotted capital I, against the bare letter.
    ("full_dotted_i", "\u{0130}"),
    // Turkish dotless i, against the bare letter.
    ("full_dotted_i", "\u{131}"),
    // A combining mark under a user-written `\w`.
    ("full_unicode_word", "a\u{301}"),
];

/// The fixture case name that covers a Turkish dotted/dotless i.
fn extras_case_name(extra: char) -> String {
    match extra {
        '\u{0130}' | '\u{0131}' => "full_dotted_i".to_string(),
        other => panic!("no fixture case covers U+{:04X}", other as u32),
    }
}

fn mode(name: &str) -> ValidatorMode {
    match name {
        "full" => ValidatorMode::Full,
        "partial" => ValidatorMode::Partial,
        other => panic!("unknown mode {other:?}"),
    }
}

fn validator_for(case: &serde_json::Value) -> RegexValidator {
    RegexValidator {
        pattern: case["pattern"].as_str().expect("pattern").to_string(),
        mode: mode(case["mode"].as_str().expect("mode")),
        exclude: case["exclude"].as_bool().expect("exclude"),
        ignore_case: case["ignore_case"].as_bool().expect("ignore_case"),
        dot_all: case["dot_all"].as_bool().expect("dot_all"),
    }
}

#[test]
fn validate_matches_the_reference_on_every_case() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(!cases.is_empty());

    let mut checked = 0usize;
    let mut skipped = 0usize;
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let validator = validator_for(case);
        for entry in case["results"].as_array().expect("results") {
            let text = entry[0].as_str().expect("text");
            let want = entry[1].as_bool().expect("verdict");
            let got = validator
                .validate(text)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            if DIVERGENT.contains(&(name, text)) {
                // Asserted the other way round by the two dedicated tests, which
                // fail if the engines ever start agreeing.
                skipped += 1;
                continue;
            }
            assert_eq!(
                got, want,
                "{name}: validate({text:?}) on pattern {:?} mode {:?} exclude {} ignore_case {}",
                validator.pattern, validator.mode, validator.exclude, validator.ignore_case
            );
            checked += 1;
        }
    }
    assert!(
        checked > 80,
        "only {checked} comparisons ran; the fixture lost coverage"
    );
    assert_eq!(
        skipped,
        DIVERGENT.len(),
        "a skipped divergence is no longer divergent, so close it"
    );
    // Every letter Python folds differently must appear here, or the list has
    // gone stale.
    for (letter, extras) in IGNORECASE_DIVERGENCES {
        for extra in extras {
            let name = extras_case_name(extra);
            assert!(
                DIVERGENT.contains(&(name.as_str(), extra.to_string().as_str())),
                "U+{:04X} is listed as divergent but is not in DIVERGENT",
                extra as u32
            );
        }
        let _ = letter;
    }
}

#[test]
fn fullmatch_anchors_absolutely() {
    // Both engines reject a trailing newline for `a$`, because `fullmatch`
    // requires the whole string and the newline is left over. `\A`/`\z` is what
    // this port uses, and it agrees.
    let validator = RegexValidator {
        pattern: "a$".to_string(),
        mode: ValidatorMode::Full,
        exclude: false,
        ignore_case: true,
        dot_all: false,
    };
    assert!(validator.validate("a").unwrap());
    assert!(!validator.validate("a\n").unwrap());
    assert!(!validator.validate("ab").unwrap());

    let absolute = RegexValidator {
        pattern: "\\Aa\\z".to_string(),
        ..validator.clone()
    };
    assert!(absolute.validate("a").unwrap());
    assert!(!absolute.validate("a\n").unwrap());
}

#[test]
fn mode_is_load_bearing() {
    // `mode` is the difference between a substring hit and a whole-string hit,
    // so a port that dropped it would report spans it should drop.
    let fixture = fixture();
    for case in fixture["mode_cases"].as_array().expect("mode_cases") {
        let name = case["name"].as_str().expect("name");
        let base = |mode| RegexValidator {
            pattern: case["pattern"].as_str().unwrap().to_string(),
            mode,
            exclude: false,
            ignore_case: true,
            dot_all: false,
        };
        let full = base(ValidatorMode::Full);
        let partial = base(ValidatorMode::Partial);
        assert_eq!(
            full.validate("").unwrap_or_default(),
            false,
            "{name}: a bare pattern must not match the empty string in full mode"
        );
        assert_eq!(
            partial.validate("").unwrap_or_default(),
            false,
            "{name}: a non-empty pattern cannot match the empty string"
        );
    }

    let partial = RegexValidator {
        pattern: "curie".to_string(),
        mode: ValidatorMode::Partial,
        exclude: false,
        ignore_case: true,
        dot_all: false,
    };
    assert!(partial.validate("marie curie").unwrap());
    // `full` on the same pattern rejects the sentence.
    let full = RegexValidator {
        mode: ValidatorMode::Full,
        ..partial.clone()
    };
    assert!(!full.validate("marie curie").unwrap());
    assert!(full.validate("curie").unwrap());
}

#[test]
fn exclude_inverts_the_verdict() {
    let base = RegexValidator {
        pattern: "\\d+".to_string(),
        mode: ValidatorMode::Full,
        exclude: false,
        ignore_case: true,
        dot_all: false,
    };
    let mut excluded = base.clone();
    excluded.exclude = true;

    for (text, plain) in [
        ("123", true),
        ("12a", false),
        ("a12", false),
        ("abc", false),
        ("", false),
    ] {
        assert_eq!(base.validate(text).unwrap(), plain, "full \\d+ on {text:?}");
        assert_eq!(
            excluded.validate(text).unwrap(),
            !plain,
            "exclude must invert the verdict for {text:?}"
        );
    }
}

#[test]
fn all_validators_must_accept_and_an_empty_list_admits_everything() {
    let list = vec![
        // A surface must start with a capital letter ...
        RegexValidator {
            pattern: "[A-Z][a-z]+".to_string(),
            mode: ValidatorMode::Full,
            exclude: false,
            // Case-sensitive, or `[A-Z]` would also admit the lower-case form
            // and the "needs a capital" assertion below would prove nothing.
            ignore_case: false,
            dot_all: false,
        },
        // ... and must not contain a digit.
        RegexValidator {
            pattern: "[0-9]".to_string(),
            mode: ValidatorMode::Partial,
            exclude: true,
            ignore_case: true,
            dot_all: false,
        },
    ];
    let compiled = CompiledValidators::compile(Some(&list)).expect("compile");
    assert!(!compiled.is_empty());
    // `all(...)`: one rejecting validator drops the span.
    assert!(compiled.accepts("Marie"));
    assert!(
        !compiled.accepts("Marie2"),
        "the excluding validator drops the span"
    );
    assert!(
        !compiled.accepts("marie"),
        "the first validator needs a capital"
    );

    // The reference's `if validators and not all(...)` short-circuits on an
    // empty list, so an absent list admits everything.
    assert!(CompiledValidators::compile(None)
        .expect("compile")
        .accepts("anything at all"));
    assert!(CompiledValidators::compile(Some(&[]))
        .expect("compile")
        .accepts("anything at all"));
}

#[test]
fn construction_errors_match_the_reference() {
    let fixture = fixture();
    let errors = fixture["construction_errors"].as_object().expect("errors");

    let bad_pattern = RegexValidator {
        pattern: "a(".to_string(),
        mode: ValidatorMode::Full,
        exclude: false,
        ignore_case: true,
        dot_all: false,
    };
    assert!(bad_pattern.compile().is_err());
    assert_eq!(
        errors["bad_pattern"].as_str().unwrap(),
        "Invalid regex: 'a('"
    );

    // The reference raises for a mode outside its two values at construction.
    // `ValidatorMode` is a closed enum here, so an unknown mode is rejected by
    // deserialization rather than by a runtime branch.
    let bad_mode = serde_json::from_str::<RegexValidator>(r#"{"pattern": "a", "mode": "prefix"}"#);
    assert!(bad_mode.is_err(), "an unknown mode must not deserialize");
    assert_eq!(
        errors["bad_mode"].as_str().unwrap(),
        "mode must be 'full' or 'partial', got 'prefix'"
    );
}

/// The known divergence, asserted rather than skipped.
///
/// If a future `regex` release folds the Turkish dotted i the way Python does,
/// this test fails and the divergence can be closed. If it stays divergent, the
/// test is the record.
#[test]
fn dotted_i_diverges_from_python() {
    let fixture = fixture();
    let case = fixture["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|case| case["name"] == "full_dotted_i")
        .expect("full_dotted_i case")
        .clone();
    let validator = validator_for(&case);
    for entry in case["results"].as_array().expect("results") {
        let text = entry[0].as_str().expect("text");
        let want = entry[1].as_bool().expect("verdict");
        let got = validator.validate(text).expect("validate");
        let is_dotless_or_dotted = text.starts_with('\u{0130}') || text.starts_with('\u{131}');
        if is_dotless_or_dotted {
            assert!(
                want && !got,
                "expected the documented divergence for {text:?}, but the engines now agree — \
                 this port can be closed against IGNORECASE_DIVERGENCES"
            );
        } else {
            assert_eq!(got, want, "non-Turkish input must agree for {text:?}");
        }
    }
}

/// A user-written `\w` is the second divergence: the `regex` crate's `\w`
/// includes combining marks and Python's does not.
#[test]
fn user_written_word_class_diverges_on_combining_marks() {
    let fixture = fixture();
    let case = fixture["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|case| case["name"] == "full_unicode_word")
        .expect("full_unicode_word case")
        .clone();
    let validator = validator_for(&case);
    let combined = "a\u{0301}";
    let entry = case["results"]
        .as_array()
        .expect("results")
        .iter()
        .find(|entry| entry[0].as_str() == Some(combined))
        .expect("combining-mark entry");
    assert!(
        !entry[1].as_bool().unwrap(),
        "Python rejects a combining mark under \\w"
    );
    assert!(
        validator.validate(combined).unwrap(),
        "the regex crate's \\w includes \\p{{M}}, which is the documented divergence"
    );
    // The class this port does spell out agrees with Python.
    let spelled_out = RegexValidator {
        pattern: "[\\p{L}\\p{N}_]+".to_string(),
        ..validator
    };
    assert!(!spelled_out.validate(combined).unwrap());
}

/// The decode paths must actually consult the table, keyed by the field name and
/// applied to the surface the output reports.
///
/// The truth table above pins what `validate` decides; this pins that
/// `decode_spans` and `decode_legacy_structures` call it at all, and with which
/// string. A validator that compiled correctly but was never consulted would
/// leave every other test in this file green.
mod wiring {
    use std::collections::BTreeMap;

    use rust_model_inference::models::gliner_boundary::extract::decode_spans;
    use rust_model_inference::models::gliner_boundary::overlap::OverlapPolicy;
    use rust_model_inference::models::gliner_boundary::spans::DocumentCandidateBatch;
    use rust_model_inference::models::gliner_boundary::structure::{
        decode_legacy_structures, LegacyStructureGroup,
    };
    use rust_model_inference::models::gliner_boundary::validator::{
        CompiledValidators, RegexValidator, ValidatorMode,
    };

    fn validator(pattern: &str) -> RegexValidator {
        RegexValidator {
            pattern: pattern.to_string(),
            mode: ValidatorMode::Full,
            exclude: false,
            ignore_case: true,
            dot_all: false,
        }
    }

    /// One query, one candidate spanning the whole word list.
    fn batch() -> DocumentCandidateBatch {
        DocumentCandidateBatch {
            indices: vec![0, 2],
            // A logit whose sigmoid is ~0.999, so a 0.5 threshold keeps it.
            pair_logits: vec![9.0],
            valid_mask: vec![true],
            candidate_states: Vec::new(),
            pool_candidate_features: Vec::new(),
            pool_size: 1,
        }
    }

    #[test]
    fn a_validator_drops_the_span_it_rejects() {
        let words = vec!["marie".to_string(), "curie".to_string()];
        let fields = vec!["person".to_string()];
        let unfiltered = decode_spans(
            &batch(),
            &words,
            &fields,
            1.0,
            &[0.5],
            0.5,
            Some(OverlapPolicy::Disallow),
            &BTreeMap::new(),
        );
        assert_eq!(unfiltered.len(), 1);
        assert_eq!(unfiltered[0].text, "marie curie");

        // A full-match validator for a single capitalised word rejects the
        // two-word surface.
        let mut table = BTreeMap::new();
        table.insert(
            "person".to_string(),
            CompiledValidators::compile(Some(&[validator("[A-Z][a-z]+")])).unwrap(),
        );
        let filtered = decode_spans(
            &batch(),
            &words,
            &fields,
            1.0,
            &[0.5],
            0.5,
            Some(OverlapPolicy::Disallow),
            &table,
        );
        assert!(filtered.is_empty(), "the validator must drop the span");
    }

    #[test]
    fn the_table_is_keyed_by_field_name() {
        let words = vec!["marie".to_string(), "curie".to_string()];
        let mut table = BTreeMap::new();
        // A validator configured for a *different* field must not apply.
        table.insert(
            "city".to_string(),
            CompiledValidators::compile(Some(&[validator("nothing-matches")])).unwrap(),
        );
        let kept = decode_spans(
            &batch(),
            &words,
            &["person".to_string()],
            1.0,
            &[0.5],
            0.5,
            Some(OverlapPolicy::Disallow),
            &table,
        );
        assert_eq!(kept.len(), 1, "another field's validator must not apply");
    }

    #[test]
    fn the_structure_path_filters_on_the_derived_surface() {
        let words = vec!["paris".to_string(), "london".to_string()];
        let scored = vec![vec![(0.9f32, 0usize, 2usize)]];
        let group = LegacyStructureGroup {
            name: "trip",
            field_names: vec!["destination"],
            query_ids: vec![0],
            scored: &scored,
            is_scalar: &[false],
            words: &words,
        };
        let unfiltered = decode_legacy_structures(
            std::slice::from_ref(&group),
            OverlapPolicy::Disallow,
            &BTreeMap::new(),
        );
        assert_eq!(unfiltered.len(), 1);
        match &unfiltered[0].fields[0].1 {
            rust_model_inference::models::gliner_boundary::structure::StructureField::List(
                spans,
            ) => assert_eq!(spans.len(), 1),
            other => panic!("expected a list field, got {other:?}"),
        }

        // `field_metadata` is keyed `<group>.<field>`, which is what
        // `score_structures` builds the table with.
        let mut table = BTreeMap::new();
        table.insert(
            "trip.destination".to_string(),
            CompiledValidators::compile(Some(&[validator("paris")])).unwrap(),
        );
        let filtered = decode_legacy_structures(
            std::slice::from_ref(&group),
            OverlapPolicy::Disallow,
            &table,
        );
        assert!(
            filtered.is_empty(),
            "a structure whose every field came back empty is dropped entirely"
        );
    }
}
