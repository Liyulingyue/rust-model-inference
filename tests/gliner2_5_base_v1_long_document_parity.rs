//! Byte-exact parity for long-document chunking and chunk merging.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_long_document.py`. Both
//! reference helpers are pure, so this needs no GGUF and always runs.
//!
//! The merge is type-dependent and each branch is separate code in the
//! reference, so the fixture cases put a value **directly** under its key — the
//! shape that lets the matching branch fire. Nesting inside a list (as real
//! `json_structures` output does) sends every case down concatenate-and-dedupe
//! instead and pins none of them.
//!
//! What the branches pin:
//!
//! - a classification dict merges to the **max confidence**;
//! - a **bare string merges by majority vote**, ties to the **earliest** chunk;
//! - **span** items go through the overlap resolver and are re-sorted by
//!   `(start, end, text)`, and their `text` is re-derived from the document
//!   rather than carried from the chunk;
//! - **non-span** items collapse on a **confidence-insensitive** canonical key,
//!   keeping the higher-confidence survivor;
//! - stripping runs after merging, and a choice/enum field collapses to a bare
//!   string when confidence is off — a shape that does not exist off this path.

use rust_model_inference::models::gliner::prompt::WordSplitter;
use rust_model_inference::models::gliner_boundary::long_document::{
    merge_chunk_results, split_text_into_chunks, TextChunk,
};
use rust_model_inference::models::gliner_boundary::overlap::OverlapPolicy;

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/long-document-golden.json";

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_long_document.py"));
    serde_json::from_str(&raw).expect("parse long-document-golden.json")
}

fn expected_chunks(case: &serde_json::Value) -> Vec<TextChunk> {
    case["chunks"]
        .as_array()
        .expect("chunks")
        .iter()
        .map(|chunk| TextChunk {
            text: chunk["text"].as_str().expect("text").to_string(),
            start_char: chunk["start_char"].as_u64().expect("start") as usize,
            end_char: chunk["end_char"].as_u64().expect("end") as usize,
            start_word: chunk["start_word"].as_u64().expect("start_word") as usize,
            end_word: chunk["end_word"].as_u64().expect("end_word") as usize,
        })
        .collect()
}

#[test]
fn chunking_matches_the_reference() {
    let fixture = fixture();
    for case in fixture["chunking"].as_array().expect("chunking") {
        let name = case["name"].as_str().expect("name");
        let text = case["text"].as_str().expect("text");
        let size = case["chunk_size"].as_u64().expect("size") as usize;
        let overlap = case["chunk_overlap"].as_u64().expect("overlap") as usize;

        let got = split_text_into_chunks(text, size, overlap, WordSplitter::Whitespace)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let want = expected_chunks(case);
        assert_eq!(got.len(), want.len(), "{name}: chunk count");
        for (index, (got, want)) in got.iter().zip(&want).enumerate() {
            assert_eq!(got, want, "{name}: chunk {index}");
        }
    }
}

/// A document with no words still yields one chunk spanning the whole text —
/// zero chunks would make `merge_chunk_results`' length check fire.
#[test]
fn an_empty_document_still_yields_one_chunk() {
    for text in ["", "     "] {
        let chunks =
            split_text_into_chunks(text, 10, 2, WordSplitter::Whitespace).expect("chunking");
        assert_eq!(chunks.len(), 1, "empty document {text:?} must still chunk");
        assert_eq!(chunks[0].end_char, text.chars().count());
    }
}

/// The final chunk is not duplicated by a trailing empty window, and the window
/// advances by `chunk_size - chunk_overlap` rather than by `chunk_size`.
#[test]
fn windows_step_by_the_overlap_and_do_not_duplicate_the_tail() {
    let document = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9";
    let chunks =
        split_text_into_chunks(document, 4, 2, WordSplitter::Whitespace).expect("chunking");
    let words: Vec<(usize, usize)> = chunks
        .iter()
        .map(|chunk| (chunk.start_word, chunk.end_word))
        .collect();
    assert_eq!(words[0], (0, 4));
    // Step 2, so the second window starts at word 2 and overlaps by 2.
    assert_eq!(words[1], (2, 6));
    // The last window ends at the document end and the loop breaks there.
    assert_eq!(words.last().unwrap().1, 10);
    assert!(
        chunks.iter().all(|chunk| chunk.end_word <= 10),
        "no window runs past the end"
    );
}

/// A chunk's text is a slice of the original, so the document's casing survives —
/// lower-casing happens later, in the model.
#[test]
fn a_chunk_preserves_the_documents_casing() {
    let document = "Marie Curie WAS here";
    let chunks =
        split_text_into_chunks(document, 3, 1, WordSplitter::Whitespace).expect("chunking");
    assert!(chunks[0].text.contains("Marie"), "got {:?}", chunks[0].text);
    assert!(chunks.iter().any(|chunk| chunk.text.contains("WAS")));
}

fn merge_case(case: &serde_json::Value) -> (String, Vec<TextChunk>) {
    let document = case["_document"].as_str().expect("_document");
    let results = case["chunk_results"].as_array().expect("chunk_results");
    // The window the oracle used, recorded alongside the case, so the test
    // rebuilds the same chunks the reference merged over.
    let size = case["_chunk_size"].as_u64().expect("_chunk_size") as usize;
    let overlap = case["_chunk_overlap"].as_u64().expect("_chunk_overlap") as usize;
    let chunks = split_text_into_chunks(document, size, overlap, WordSplitter::Whitespace)
        .expect("chunking");
    assert!(
        chunks.len() >= results.len(),
        "{}: window yields {} chunks but the case has {} results",
        case["name"].as_str().unwrap_or("case"),
        chunks.len(),
        results.len()
    );
    (
        document.to_string(),
        chunks.into_iter().take(results.len()).collect(),
    )
}

#[test]
fn merging_matches_the_reference() {
    let fixture = fixture();
    for case in fixture["merging"].as_array().expect("merging") {
        let name = case["name"].as_str().expect("name");
        let (document, chunks) = merge_case(case);
        let results: Vec<serde_json::Value> = case["chunk_results"]
            .as_array()
            .expect("chunk_results")
            .clone();
        let scalar: Vec<String> = case["scalar_entity_labels"]
            .as_array()
            .expect("scalar")
            .iter()
            .map(|value| value.as_str().expect("label").to_string())
            .collect();
        let include_confidence = case["include_confidence"]
            .as_bool()
            .expect("include_confidence");
        let include_spans = case["include_spans"].as_bool().expect("include_spans");

        let got = merge_chunk_results(
            &document,
            &chunks,
            &results,
            include_confidence,
            include_spans,
            &scalar,
            OverlapPolicy::Disallow,
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(got, case["merged"], "{name}: merged result differs");
    }
}

/// The type-dependent branches, asserted directly so a fixture edit that weakened
/// them fails here rather than making the parity test vacuous.
#[test]
fn each_value_type_takes_its_own_merge_rule() {
    let fixture = fixture();
    let case = |name: &str| {
        fixture["merging"]
            .as_array()
            .expect("merging")
            .iter()
            .find(|case| case["name"] == name)
            .unwrap_or_else(|| panic!("no case {name}"))
            .clone()
    };

    // A classification dict takes the **max confidence**, not the most common.
    let classification = case("classification_takes_max_confidence");
    assert_eq!(classification["merged"]["mood"]["label"], "sad");
    assert_eq!(classification["merged"]["mood"]["confidence"], 0.9);

    // A bare string takes the **majority**.
    let majority = case("bare_strings_take_the_majority");
    assert_eq!(
        majority["merged"]["mood"], "happy",
        "two of three chunks said happy"
    );

    // A tie goes to the **earliest** chunk, not the highest score — bare strings
    // carry no score, so "earliest" is the whole rule.
    let tie = case("bare_string_tie_goes_to_the_earliest_chunk");
    assert_eq!(
        tie["merged"]["mood"], "happy",
        "the first chunk's value wins a tie"
    );
}

/// A span's `text` is re-derived from the document, not carried from the chunk —
/// the fixture supplies a deliberately wrong `text` for exactly this.
#[test]
fn a_span_surface_is_re_derived_from_the_document() {
    let fixture = fixture();
    let stripped = fixture["merging"]
        .as_array()
        .expect("merging")
        .iter()
        .find(|case| case["name"] == "strips_spans_when_not_requested")
        .expect("case")
        .clone();
    // The input's `text` was "ignored"; the merged value is the document's slice.
    let input_text = stripped["chunk_results"][0]["city"][0]["text"]
        .as_str()
        .unwrap();
    assert_eq!(input_text, "ignored");
    let merged = &stripped["merged"]["city"][0];
    assert_ne!(
        merged, "ignored",
        "the surface must be re-sliced from the document"
    );
}

/// With neither flag set, a span collapses to its bare text; with only confidence,
/// it keeps `text` + `confidence` and drops the offsets.
#[test]
fn stripping_follows_the_flags() {
    let fixture = fixture();
    let case = |name: &str| {
        fixture["merging"]
            .as_array()
            .expect("merging")
            .iter()
            .find(|case| case["name"] == name)
            .unwrap_or_else(|| panic!("no case {name}"))
            .clone()
    };
    // Neither flag: a bare string.
    assert!(case("strips_spans_when_not_requested")["merged"]["city"][0].is_string());
    // Confidence only: an object, no offsets.
    let only_confidence = case("keeps_only_confidence_when_spans_are_off");
    let entry = &only_confidence["merged"]["city"][0];
    assert!(entry.is_object());
    assert!(entry.get("confidence").is_some());
    assert!(entry.get("start").is_none(), "offsets are dropped");
    assert!(entry.get("end").is_none());
}

/// A choice/enum field (`text` + `confidence`, no offsets) collapses to the bare
/// string when confidence is off — a shape that does not exist off the long path.
#[test]
fn an_enum_field_collapses_to_a_bare_string() {
    let fixture = fixture();
    let case = fixture["merging"]
        .as_array()
        .expect("merging")
        .iter()
        .find(|case| case["name"] == "enum_choice_field_collapses_to_a_bare_string")
        .expect("case")
        .clone();
    assert_eq!(case["merged"]["mood"], "happy");
}

/// The overlap policy is `disallow` for a scalar label, so the highest-scoring
/// of two overlapping spans wins and the field collapses to it.
#[test]
fn a_scalar_label_collapses_to_the_single_best_span() {
    let fixture = fixture();
    let case = fixture["merging"]
        .as_array()
        .expect("merging")
        .iter()
        .find(|case| case["name"] == "scalar_entity_label_collapses_to_one_value")
        .expect("case")
        .clone();
    let merged = &case["merged"]["people"];
    assert!(
        merged.is_array() && merged.as_array().unwrap().len() == 1,
        "a non-list dtype collapses to one value, got {merged}"
    );
    assert_eq!(
        merged[0]["text"], "Marie Curie",
        "the higher-scoring span wins"
    );
}

/// The length check is a real error, not a silent zip.
#[test]
fn mismatched_chunk_lengths_are_rejected() {
    let document = "one two three four five six";
    let chunks =
        split_text_into_chunks(document, 3, 1, WordSplitter::Whitespace).expect("chunking");
    let too_few = vec![serde_json::json!({"city": []})];
    assert!(merge_chunk_results(
        document,
        &chunks[..1],
        &too_few,
        false,
        false,
        &[],
        OverlapPolicy::Disallow,
    )
    .is_ok());
    assert!(merge_chunk_results(
        document,
        &chunks,
        &too_few,
        false,
        false,
        &[],
        OverlapPolicy::Disallow,
    )
    .is_err());
}

/// The window parameters are validated the way the reference validates them, so a
/// `chunk_overlap >= chunk_size` cannot loop forever.
#[test]
fn window_parameters_are_validated() {
    assert!(split_text_into_chunks("a b", 0, 0, WordSplitter::Whitespace).is_err());
    assert!(split_text_into_chunks("a b", 4, 4, WordSplitter::Whitespace).is_err());
    assert!(split_text_into_chunks("a b", 4, 5, WordSplitter::Whitespace).is_err());
    assert!(split_text_into_chunks("a b", 4, 3, WordSplitter::Whitespace).is_ok());
}
