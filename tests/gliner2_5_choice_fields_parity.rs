//! Byte-exact parity for the `choices` prefix — literal-enum structure fields.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_choice_fields.py`. The prefix is
//! pure token-stream construction, so this needs no GGUF and always runs.
//!
//! Three properties are the whole difficulty, and the fixture pins each:
//!
//! 1. **The prefix lands on the text stream, after `[SEP_TEXT]`, not on the
//!    schema stream.** The reference does `text_tokens = prefix + text_tokens`
//!    (`processor.py:645`). A port that appended it to the schema prefix would
//!    shift every marker after it by a variable amount, and the `[C]` marker
//!    stride is load-bearing for all four task kinds.
//! 2. **It is word-routed**, so the encoder produces a text state per prefix
//!    token — that is what lets a choice be scored as a one-token span.
//! 3. **A choice is found in the prefix region only**, by lower-cased exact token
//!    match, so the same word appearing in the document is not a second choice.

use rust_model_inference::models::gliner::prompt::render_choice_prefix;

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/choice-fields-golden.json";

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_choice_fields.py"));
    serde_json::from_str(&raw).expect("parse choice-fields-golden.json")
}

fn strings(value: &serde_json::Value) -> Vec<String> {
    value
        .as_array()
        .expect("token array")
        .iter()
        .map(|token| token.as_str().expect("token").to_string())
        .collect()
}

#[test]
fn the_prefix_renders_exactly_as_the_reference_does() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(!cases.is_empty());

    for case in cases {
        let name = case["name"].as_str().expect("name");
        let got = render_choice_prefix(&case["schema"]);
        let want = strings(&case["prefix_tokens"]);
        assert_eq!(
            got, want,
            "{name}: choice prefix differs; the fixture records the reference's \
             `_build_classification_prefix` output verbatim"
        );
    }
}

#[test]
fn the_rendered_shape_is_the_documented_one() {
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

    // `( parent: field ( c1 | c2 ) )`
    assert_eq!(
        render_choice_prefix(&case("three_choices_two_fields")["schema"]),
        strings(&case("three_choices_two_fields")["prefix_tokens"])
    );
    let two = render_choice_prefix(&case("three_choices_two_fields")["schema"]);
    assert_eq!(two[0], "(");
    assert_eq!(two[1], "trip:");
    assert_eq!(two[2], "mood");
    assert_eq!(two[3], "(");
    // Choices are two apart, with `|` between them.
    assert_eq!(two[4], "happy");
    assert_eq!(two[5], "|");
    assert_eq!(two[6], "sad");
    assert_eq!(two[7], "|");
    assert_eq!(two[8], "neutral");
    assert_eq!(two[9], ")");
    // The second field follows a comma and repeats the parenthesised run.
    assert_eq!(two[10], ",");
    assert_eq!(two[11], "status");
    assert_eq!(two[12], "(");

    // A group with no choice field contributes nothing, so a plain schema
    // renders the empty prefix and every other path stays byte-identical.
    assert!(render_choice_prefix(&serde_json::json!({
        "entities": {"person": {}}
    }))
    .is_empty());
    assert!(render_choice_prefix(&serde_json::json!({
        "json_structures": [{"person": {"name": []}}]
    }))
    .is_empty());
    // A group with a choice field alongside a plain one takes only the choice
    // field's run.
    let mixed = render_choice_prefix(&case("choices_mixed_with_a_plain_field")["schema"]);
    assert!(!mixed.contains(&"destination".to_string()));
}

#[test]
fn choice_literals_reach_the_encoder_verbatim() {
    let fixture = fixture();
    let case = fixture["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|case| case["name"] == "uppercase_choice_values")
        .expect("case")
        .clone();
    let prefix = render_choice_prefix(&case["schema"]);
    // The reference does not lower-case the prefix; `_find_choice_idx` lower-cases
    // both sides at lookup instead. Lower-casing here would still be a
    // byte-different prompt, which the encoder sees.
    assert!(
        prefix.contains(&"Happy".to_string()),
        "the prefix must keep the declared casing: {prefix:?}"
    );
    assert!(prefix.contains(&"SAD".to_string()));
}

#[test]
fn a_choice_is_located_in_the_prefix_region_only() {
    let fixture = fixture();
    let case = fixture["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|case| case["name"] == "choice_values_also_appear_in_the_document")
        .expect("case")
        .clone();
    let prefix = render_choice_prefix(&case["schema"]);
    let prefix_len = prefix.len();

    // `paris` is both a declared choice and a word in the document. The lookup
    // searches `text_tokens[:len_prefix]`, so it finds the prefix occurrence at
    // index 4 and never the document one.
    let document_words = strings(&case["text_tokens"]);
    assert!(
        document_words.iter().any(|word| word == "paris"),
        "the fixture must actually contain the collision"
    );
    let found = prefix
        .iter()
        .position(|token| token.to_lowercase() == "paris");
    assert_eq!(
        found,
        Some(4),
        "the prefix occurrence is the one the reference finds"
    );
    assert!(
        found.unwrap() < prefix_len,
        "a choice index is a prefix index, not a document index"
    );

    // The fixture's own recorded lookup agrees.
    for entry in case["choice_lookup"].as_array().expect("choice_lookup") {
        for pair in entry[1].as_array().expect("choices") {
            let choice = pair[0].as_str().expect("choice");
            let index = pair[1].as_i64().expect("index");
            if index >= 0 {
                assert!(
                    (index as usize) < prefix_len,
                    "{choice:?} was found outside the prefix"
                );
            }
        }
    }
}

#[test]
fn a_schema_without_choices_leaves_the_text_stream_untouched() {
    // The regression this whole feature risks: a non-empty prefix on a schema
    // that declared no `choices` would shift every span index by a variable
    // amount while still producing plausible-looking output.
    for schema in [
        serde_json::json!({}),
        serde_json::json!({"entities": {"person": {}, "city": {}}}),
        serde_json::json!({"json_structures": [{"trip": {"traveller": []}}]}),
        serde_json::json!({"json_structures": [{"trip": {"mood": {"value": ""}}}]}),
        // `choices` present but empty is not a choice field either.
        serde_json::json!({"json_structures": [{"trip": {"mood": {"value": "", "choices": []}}}]}),
        // A `value` without `choices` is not a choice field.
        serde_json::json!({"json_structures": [{"trip": {"mood": {"value": "x"}}}]}),
    ] {
        assert!(
            render_choice_prefix(&schema).is_empty(),
            "{schema} declared no choices, so the prefix must be empty"
        );
    }
}

/// The prefix must land on the **text** stream, after `[SEP_TEXT]`.
///
/// The renderer test above cannot see this: it only compares the token list. The
/// failure mode is putting the prefix before `[SEP_TEXT]`, which produces the
/// same tokens in a different stream and would shift every span index by a
/// variable amount. A fake tokenizer makes the order observable without a GGUF —
/// each distinct token gets a distinct id, so `input_ids` *is* the stream.
mod stream_placement {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    use rust_model_inference::models::gliner::prompt::{
        build_mixed_boundary_prompt_with, render_choice_prefix, BoundaryTaskKind, Label, Task,
        SEP_TEXT,
    };

    /// One id per distinct token, shared between the prompt build and the
    /// read-back, so `input_ids` reads back as the token stream it encodes.
    type Ids = std::rc::Rc<std::cell::RefCell<HashMap<String, u32>>>;

    fn fake_tokenizer(ids: &Ids) -> impl FnMut(&str) -> Result<Vec<u32>, String> {
        let ids = ids.clone();
        move |token: &str| {
            let mut map = ids.borrow_mut();
            let next = map.len() as u32;
            Ok(vec![*map.entry(token.to_string()).or_insert(next)])
        }
    }

    fn name_of(id: u32, ids: &Ids) -> String {
        ids.borrow()
            .iter()
            .find(|(_, value)| **value == id)
            .map(|(token, _)| token.clone())
            .unwrap_or_else(|| format!("<unknown id {id}>"))
    }

    fn task() -> (Task, BoundaryTaskKind) {
        let mut trip = Task::new("trip", vec![Label::new("mood")]);
        trip.labels.push(Label::new("destination"));
        (trip, BoundaryTaskKind::JsonStructure)
    }

    fn schema() -> serde_json::Value {
        serde_json::json!({
            "json_structures": [{
                "trip": {"mood": {"value": "", "choices": ["happy", "sad"]}}
            }]
        })
    }

    #[test]
    fn the_prefix_sits_between_sep_text_and_the_document_words() {
        let prefix = render_choice_prefix(&schema());
        assert_eq!(
            prefix,
            vec!["(", "trip:", "mood", "(", "happy", "|", "sad", ")", ")"]
        );
        let (trip, kind) = task();
        let ids: Ids = Rc::new(RefCell::new(HashMap::new()));
        let encoded = build_mixed_boundary_prompt_with(
            std::slice::from_ref(&trip),
            &[kind],
            &prefix,
            "Marie Curie worked in Paris.",
            fake_tokenizer(&ids),
        )
        .expect("prompt");

        assert_eq!(encoded.text_prefix_len, prefix.len());

        let stream: Vec<String> = encoded
            .input_ids
            .iter()
            .map(|id| name_of(*id, &ids))
            .collect();

        let at = |needle: &str| {
            stream
                .iter()
                .position(|token| token == needle)
                .unwrap_or_else(|| panic!("{needle} missing from {stream:?}"))
        };
        let sep = at(SEP_TEXT);
        let first_choice = at("happy");
        let first_word = at("marie");
        let last_choice = at("sad");

        assert!(
            sep < first_choice,
            "[SEP_TEXT] must precede the choice prefix, got {stream:?}"
        );
        assert!(
            last_choice < first_word,
            "the whole prefix must precede the document words, got {stream:?}"
        );
        // The prefix is contiguous: nothing interleaves between its first and
        // last token, which is what makes `text_prefix_len` an offset.
        for expected in 0..prefix.len() {
            assert_eq!(
                stream[first_choice - 4 + expected],
                prefix[expected],
                "prefix token {expected} moved: {stream:?}"
            );
        }
    }

    #[test]
    fn a_schema_without_choices_produces_no_prefix_rows() {
        let (trip, kind) = task();
        let encoded = build_mixed_boundary_prompt_with(
            std::slice::from_ref(&trip),
            &[kind],
            &[],
            "Marie Curie worked in Paris.",
            fake_tokenizer(&Rc::new(RefCell::new(HashMap::new()))),
        )
        .expect("prompt");
        assert_eq!(
            encoded.text_prefix_len, 0,
            "an empty prefix must leave the text stream exactly as before"
        );
        // The word rows are 1:1 with the document words, which is the invariant
        // every span index depends on.
        assert_eq!(encoded.text_word_first_positions.len(), encoded.words.len());
    }
}
