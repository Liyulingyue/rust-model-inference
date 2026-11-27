//! ByteLevel BPE parity for `fastino/GLiNER2.5-Decide-1B`.
//!
//! Checked on its own, before the encoder, because the tokenizer is where this
//! checkpoint departs most from the other eleven: they are all SentencePiece
//! with a `▁` marker and a dummy prefix, and this one is a ByteLevel BPE with an
//! NFC normalizer and a GPT-2 pre-tokenization regex. Every case below was
//! produced by `tokenizers`' own `Tokenizer.encode` on the released
//! `tokenizer.json`, so the expectations are the reference's, not this
//! implementation's reading of a spec.
//!
//! The cases are chosen for the parts that are easy to get subtly wrong:
//!
//! - byte-to-unicode, so `é` appears as `Ã©` and a multi-byte character is
//!   several pieces
//! - contraction handling (`'t`) and punctuation runs (`'!!'` stays whole)
//! - runs of whitespace, which survive as their own chunk (`'   '`)
//! - digits split from letters, and a float split at the point (`3.14159`)
//!
//! Run with `RMI_GLINER2_DECIDE_1B_GGUF=.../GLiNER2.5-Decide-1B-f32.gguf`.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::models::gliner_ettin::bpe;
use serde_json::Value;

const GGUF: &str = "models/GLiNER2.5-Decide-1B/GLiNER2.5-Decide-1B-f32.gguf";
const FIXTURE: &str = "tests/fixtures/GLiNER2.5-Decide-1B/bpe-cases.json";

fn fixture() -> Value {
    let raw = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate from tokenizers"));
    serde_json::from_str(&raw).expect("parse bpe-cases.json")
}

#[test]
fn tokenizes_exactly_like_the_reference() {
    let path = std::path::Path::new(GGUF);
    if !path.exists() {
        panic!("missing {GGUF}; run tools/converter/gliner/convert_ettin.py first");
    }
    let source = GGUFLoader::from_file(path).expect("open Ettin GGUF");
    let tokenizer = bpe::from_gguf(&source).expect("build ByteLevel BPE from GGUF");

    let value = fixture();
    let cases = value["cases"].as_array().expect("cases");
    assert!(
        cases.len() >= 10,
        "fixture looks truncated: {} cases",
        cases.len()
    );

    for case in cases {
        let text = case["text"].as_str().expect("text");
        let want_tokens: Vec<String> = case["tokens"]
            .as_array()
            .expect("tokens")
            .iter()
            .map(|value| value.as_str().expect("token").to_string())
            .collect();
        let want_ids: Vec<u32> = case["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|value| value.as_u64().expect("id") as u32)
            .collect();

        // The ground truth carries `[CLS]` / `[SEP]` because the reference went
        // through the post-processor; the BPE itself does not add them, so they
        // are stripped before comparing and the encoding path adds them.
        let stripped: Vec<&String> = want_tokens
            .iter()
            .filter(|token| *token != "[CLS]" && *token != "[SEP]")
            .collect();
        let expected: Vec<String> = stripped.into_iter().cloned().collect();

        let got = tokenizer.tokenize(text);
        assert_eq!(
            got, expected,
            "tokens differ for {text:?}\\n  got  {got:?}\\n  want {expected:?}"
        );

        let got_ids = tokenizer
            .encode_with_specials(text)
            .unwrap_or_else(|error| panic!("encode {text:?}: {error}"));
        let expected_ids: Vec<u32> = want_ids
            .iter()
            .copied()
            .filter(|id| *id != 50281 && *id != 50282)
            .collect();
        assert_eq!(
            got_ids, expected_ids,
            "ids differ for {text:?}\\n  got  {got_ids:?}\\n  want {expected_ids:?}"
        );
    }
}

#[test]
fn specials_are_cut_out_of_the_surrounding_text() {
    let path = std::path::Path::new(GGUF);
    if !path.exists() {
        panic!("missing {GGUF}");
    }
    let source = GGUFLoader::from_file(path).expect("open Ettin GGUF");
    let tokenizer = bpe::from_gguf(&source).expect("build ByteLevel BPE");
    // GLiNER2 splices `[DESCRIPTION]` and friends *into* the prompt, so the
    // tokenizer has to cut them out wherever they appear. If it tokenized the
    // whole string as one chunk the surrounding words would merge differently.
    let description = *tokenizer
        .encode_with_specials("[DESCRIPTION]")
        .expect("encode a lone special")
        .first()
        .expect("one id");
    let spliced = tokenizer
        .encode_with_specials("alpha[DESCRIPTION]beta")
        .expect("encode around a special");
    let alpha = tokenizer.encode_with_specials("alpha").expect("alpha");
    let beta = tokenizer.encode_with_specials("beta").expect("beta");
    assert_eq!(
        spliced,
        vec![alpha[0], description, beta[0]],
        "the special must split the surrounding text rather than merge with it"
    );
    // The markers must stay distinct: two of them share a suffix and one
    // truncation of the name would map both onto the same row.
    let sep_struct = *tokenizer
        .encode_with_specials("[SEP_STRUCT]")
        .expect("encode another special")
        .first()
        .expect("one id");
    assert_ne!(
        sep_struct, description,
        "[SEP_STRUCT] and [DESCRIPTION] must not share an id"
    );
}
