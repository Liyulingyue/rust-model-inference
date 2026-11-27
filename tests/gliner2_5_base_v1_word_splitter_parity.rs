//! Byte-exact parity for both word splitters.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_word_splitter.py`, which runs the
//! reference `WhitespaceTokenSplitter` and `CharLevelSplitter` over the same
//! texts. Pure text in, pure text out, so like `overlap_resolution_parity` it
//! needs no GGUF and always runs.
//!
//! The whitespace splitter is what every boundary checkpoint runs, so its cases
//! double as a regression guard on the existing port. The char splitter is the
//! new surface, and its cases are chosen so each of these fails if the port
//! gets it wrong:
//!
//! - **Code point offsets, not byte offsets.** `中华人民共和国` is seven code
//!   points and twenty-one bytes. The reference's second token is `(1, 2)`; the
//!   byte range for that same token is `3..6`. The `regex` crate reports bytes,
//!   so the conversion is explicit.
//! - **One code point per `\S` token, not one grapheme.** `👋🏽` is a base emoji
//!   plus a skin-tone modifier: two code points, two tokens, even though it
//!   renders as one glyph.
//! - **A combining mark is its own token, in *both* splitters.** `cafe` + U+0301
//!   is five code points and three graphemes, but a combining mark is not
//!   matched by `\w` either, so the whitespace splitter splits it too.
//! - **`lower=False` returns the same offsets with original casing**, which is
//!   what makes "match first, lower after" observable.

use rust_model_inference::models::gliner::prompt::{word_spans, WordSplitter};

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/word-splitter-golden.json";

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_word_splitter.py"));
    serde_json::from_str(&raw).expect("parse word-splitter-golden.json")
}

fn expect(text: &str, splitter: WordSplitter, lower: bool) -> Vec<(String, usize, usize)> {
    word_spans(text, splitter, lower)
        .into_iter()
        .map(|span| (span.token, span.start, span.end))
        .collect()
}

fn want(case: &serde_json::Value, key: &str) -> Vec<(String, usize, usize)> {
    case[key]
        .as_array()
        .unwrap_or_else(|| panic!("case has no {key}"))
        .iter()
        .map(|entry| {
            let triple = entry.as_array().expect("(token, start, end)");
            (
                triple[0].as_str().expect("token").to_string(),
                triple[1].as_u64().expect("start") as usize,
                triple[2].as_u64().expect("end") as usize,
            )
        })
        .collect()
}

fn assert_spans(got: &[(String, usize, usize)], want: &[(String, usize, usize)], context: &str) {
    assert_eq!(got.len(), want.len(), "{context}: token count");
    for (index, (actual, expected)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            actual.0, expected.0,
            "{context}: token {index} text (offsets {:?} vs {:?})",
            actual, expected
        );
        assert_eq!(
            (actual.1, actual.2),
            (expected.1, expected.2),
            "{context}: token {index} offsets for {:?}",
            expected.0
        );
    }
}

#[test]
fn whitespace_splitter_matches_the_reference() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(!cases.is_empty());
    for case in cases {
        let text = case["text"].as_str().expect("text");
        assert_spans(
            &expect(text, WordSplitter::Whitespace, true),
            &want(case, "whitespace"),
            &format!("whitespace splitter on {text:?}"),
        );
    }
}

#[test]
fn char_level_splitter_matches_the_reference() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(!cases.is_empty());
    for case in cases {
        let text = case["text"].as_str().expect("text");
        assert_spans(
            &expect(text, WordSplitter::CharLevel, true),
            &want(case, "char"),
            &format!("char splitter on {text:?}"),
        );
    }
}

/// `lower=False` must preserve casing without moving any offset, which is the
/// observable consequence of the reference matching the original text and
/// lower-casing only the token value.
#[test]
fn case_preserving_splitter_keeps_offsets() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    let preserved = fixture["whitespace_case_preserving"]
        .as_array()
        .expect("case-preserving");
    assert_eq!(cases.len(), preserved.len());

    for (case, expected) in cases.iter().zip(preserved) {
        let text = case["text"].as_str().expect("text");
        let lowered = expect(text, WordSplitter::Whitespace, true);
        let original = expect(text, WordSplitter::Whitespace, false);
        assert_eq!(
            original.len(),
            lowered.len(),
            "lowering must not change the token count for {text:?}"
        );
        for (index, (was, now)) in lowered.iter().zip(&original).enumerate() {
            assert_eq!(
                (was.1, was.2),
                (now.1, now.2),
                "lowering token {index} of {text:?} moved its offset"
            );
            assert_eq!(
                was.0.to_lowercase(),
                now.0.to_lowercase(),
                "token {index} of {text:?} differs beyond case"
            );
        }
        assert_spans(
            &original,
            &want(&serde_json::json!({ "x": expected }), "x"),
            &format!("case-preserving whitespace splitter on {text:?}"),
        );
    }
}

#[test]
fn splitter_names_resolve_like_the_reference() {
    assert_eq!(WordSplitter::default(), WordSplitter::Whitespace);
    assert_eq!(
        WordSplitter::from_name("whitespace").unwrap(),
        WordSplitter::Whitespace
    );
    assert_eq!(
        WordSplitter::from_name("char").unwrap(),
        WordSplitter::CharLevel
    );
    let error = WordSplitter::from_name("nope").unwrap_err();
    assert_eq!(
        error,
        "Unknown word_splitter \"nope\". Supported names: 'char', 'whitespace'."
    );
}

/// The two splitters must actually differ, and differ in the ways the doc
/// comment claims — otherwise the char splitter is untested surface.
#[test]
fn the_two_splitters_differ_where_the_feature_says_they_do() {
    // CJK: one `\w+` run versus one token per character.
    let cjk = "中华人民共和国";
    assert_eq!(word_spans(cjk, WordSplitter::Whitespace, true).len(), 1);
    assert_eq!(word_spans(cjk, WordSplitter::CharLevel, true).len(), 7);

    // Offsets are code points, so the CJK spans advance by one each. Under byte
    // offsets the second token would be (3, 6).
    let char_spans = word_spans(cjk, WordSplitter::CharLevel, true);
    assert_eq!(char_spans[1].start, 1);
    assert_eq!(char_spans[1].end, 2);
    assert!(
        cjk.len() > char_spans[6].end,
        "the text is longer in bytes than code points"
    );

    // A combining mark is its own token in *both* splitters, because `\w` does
    // not match it either. Grapheme clustering would report one token.
    let decomposed = "cafe\u{301}";
    for splitter in [WordSplitter::Whitespace, WordSplitter::CharLevel] {
        let spans = word_spans(decomposed, splitter, true);
        assert_eq!(
            spans.len(),
            2,
            "{splitter:?} must split the combining mark off"
        );
        assert_eq!(spans[1].start, 4);
        assert_eq!(spans[1].end, 5);
    }

    // A base emoji plus a skin-tone modifier is two code points and therefore
    // two tokens, though it renders as one glyph.
    let emoji = "hi \u{1F44B}\u{1F3FD} there";
    assert_eq!(
        word_spans(emoji, WordSplitter::CharLevel, true)[1].token,
        "\u{1F44B}"
    );
    assert_eq!(
        word_spans(emoji, WordSplitter::CharLevel, true)[2].token,
        "\u{1F3FD}"
    );

    // `.` is inside the char splitter's class, so a trailing period attaches to
    // the preceding word — which means a span covering the last word also
    // covers its punctuation. The whitespace splitter separates them.
    assert_eq!(
        word_spans("has period.", WordSplitter::CharLevel, true)
            .last()
            .unwrap()
            .token,
        "period."
    );
    assert_eq!(
        word_spans("has period.", WordSplitter::Whitespace, true)
            .last()
            .unwrap()
            .token,
        "."
    );

    // Non-ASCII digits are `\w` but not in the char splitter's ASCII class.
    let digits = "١٢٣٤";
    assert_eq!(word_spans(digits, WordSplitter::Whitespace, true).len(), 1);
    assert_eq!(word_spans(digits, WordSplitter::CharLevel, true).len(), 4);

    // Empty and whitespace-only input yields no tokens, not an empty token.
    for text in ["", "   "] {
        assert!(word_spans(text, WordSplitter::Whitespace, true).is_empty());
        assert!(word_spans(text, WordSplitter::CharLevel, true).is_empty());
    }
}
