//! End-to-end parity: real text + schema -> spans.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf`.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_extract_spans_end_to_end.py`,
//! which is the only oracle here that does not start from synthetic states. It
//! runs the reference `SchemaTransformer`, a `transformers` DeBERTa-v3-base
//! with the checkpoint's *fine-tuned* `encoder.*` weights, the reference
//! `BoundaryHead`, and `decode_candidates` — so this test covers the Rust
//! tokenizer, prompt builder, encoder, gather routing, pool, scorer and decode
//! all at once.
//!
//! The comparison is deliberately two-tier:
//!  - `input_ids` and the routing indices must match **exactly**. A prompt or
//!    routing difference moves every subsequent number, so it has to be caught
//!    as its own failure rather than as a logit delta.
//!  - the decoded spans must match **exactly** at the fixture's threshold, and
//!    the raw pair logits must agree numerically. Spans are thresholded, so two
//!    implementations can agree on every span while their logits still differ
//!    near the cut; the logit check is what catches that.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner::prompt::{Label, Task, E_TOKEN};
use rust_model_inference::models::gliner_boundary::{extract_spans, run_extraction, BoundaryModel};

const GGUF: &str = "models/gliner2.5-base-v1/gliner2.5-base-v1-f32.gguf";
const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/extract-spans-e2e-golden.json";

fn loaded_model() -> Option<BoundaryModel<'static>> {
    let path = std::env::var_os("RMI_GLINER2_5_BASE_V1_GGUF").map(std::path::PathBuf::from)?;
    if !path.exists() {
        eprintln!("skipping: {} does not exist", path.display());
        return None;
    }
    let source = GGUFLoader::from_file(&path).expect("open boundary GGUF");
    let leaked: &'static dyn TensorSource = Box::leak(Box::new(source));
    Some(BoundaryModel::from_source(leaked).expect("load boundary model"))
}

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE).unwrap_or_else(|_| {
        panic!("missing {FIXTURE}; regenerate with dump_extract_spans_end_to_end.py")
    });
    serde_json::from_str(&raw).expect("parse extract-spans-e2e-golden.json")
}

/// The schema, spelled the same way on both sides. Field order is the contract:
/// it fixes the query order, so the Rust `Vec` and the JSON object must agree.
fn tasks() -> Vec<Task> {
    let fields = [
        ("person", "an individual human being"),
        ("organization", "a company or institution"),
        ("location", "a city, country or other place"),
    ];
    vec![Task::new(
        "entities",
        fields
            .iter()
            .map(|(name, description)| {
                let mut label = Label::new(*name);
                label.description = Some((*description).to_string());
                label
            })
            .collect(),
    )]
}

#[test]
fn prompt_and_routing_match_the_reference() {
    let Some(model) = loaded_model() else {
        return;
    };
    let fixture = fixture();
    let tasks = tasks();
    let encoded = rust_model_inference::models::gliner_boundary::extract::encode_boundary_prompt(
        &model,
        &tasks,
        "Ada Lovelace worked with Charles Babbage in London.",
        E_TOKEN,
    )
    .expect("encode boundary prompt");

    let case = &fixture["cases"][0];
    let want_ids: Vec<u32> = case["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(
        encoded.input_ids, want_ids,
        "input_ids differ; every downstream number depends on them"
    );
    let want_words: Vec<String> = case["text_words"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(encoded.words, want_words, "word split differs");
    let want_positions: Vec<usize> = case["text_word_first_positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(
        encoded.text_word_first_positions, want_positions,
        "word routing differs"
    );
    // Text normalization is part of the contract: `_collate_batch` appends a
    // "." when the text does not already end in sentence punctuation, and that
    // extra word shifts every index after it.
    assert_eq!(
        encoded.input_ids.len(),
        want_ids.len(),
        "token count differs (text normalization?)"
    );
    let want_queries: Vec<usize> = case["query_marker_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(
        encoded.query_positions, want_queries,
        "query marker routing differs"
    );
    let want_names: Vec<String> = case["query_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        encoded.query_names, want_names,
        "field order differs, so queries are paired with the wrong fields"
    );
}

#[test]
fn logits_and_spans_match_the_reference() {
    let Some(model) = loaded_model() else {
        return;
    };
    let fixture = fixture();
    let tasks = tasks();
    let mut worst = 0.0f32;

    for (case_index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let text = case["text"].as_str().unwrap();
        let threshold = case["threshold"].as_f64().unwrap() as f32;

        let (batch, words) =
            run_extraction(&model, text, &tasks, E_TOKEN, 0).expect("run extraction");
        let want_logits: Vec<f32> = case["pair_logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(
            batch.pair_logits.len(),
            want_logits.len(),
            "case {case_index}: pair logit count differs"
        );
        for (i, (got, want)) in batch.pair_logits.iter().zip(want_logits.iter()).enumerate() {
            // Padded slots are MASK_LOGIT on both sides; comparing them is
            // harmless but tells us nothing, so skip.
            if !batch.valid_mask[i] {
                continue;
            }
            worst = worst.max((got - want).abs());
        }

        let got = extract_spans(&model, text, &tasks, E_TOKEN, 0, Some(threshold))
            .expect("extract spans");
        // `decode_candidates` drops the score, so the golden is (start, end)
        // pairs in the reference's sort order.
        let mut want: Vec<(String, usize, usize)> = Vec::new();
        for (query_index, spans) in case["spans"].as_array().unwrap().iter().enumerate() {
            let field = case["query_names"][query_index].as_str().unwrap();
            for span in spans.as_array().unwrap() {
                let pair = span.as_array().unwrap();
                assert_eq!(pair.len(), 2, "a decoded span is a (start, end) pair");
                want.push((
                    field.to_string(),
                    pair[0].as_u64().unwrap() as usize,
                    pair[1].as_u64().unwrap() as usize,
                ));
            }
        }
        assert_eq!(
            got.len(),
            want.len(),
            "case {case_index}: {text:?} produced {} spans, reference produced {}",
            got.len(),
            want.len()
        );
        for (g, (field, start, end)) in got.iter().zip(want.iter()) {
            assert_eq!(&g.field, field, "case {case_index}: field mismatch");
            assert_eq!(
                (g.start, g.end),
                (*start, *end),
                "case {case_index}: span mismatch for field {field}"
            );
            assert_eq!(
                g.text,
                words[g.start..g.end].join(" "),
                "case {case_index}: span text must be the spanned words"
            );
            // The reported score is the temperature-scaled sigmoid, not the raw
            // logit: a caller filtering on it must get the reference's ordering.
            let want_score = 1.0 / (1.0 + (-g.logit / model.settings.pair_temperature).exp());
            assert!(
                (g.score - want_score).abs() < 1e-6,
                "case {case_index}: reported score {} disagrees with sigmoid(logit) {want_score}",
                g.score
            );
            assert!(
                g.score >= threshold,
                "case {case_index}: a reported span is below the threshold"
            );
        }
    }

    assert!(
        worst < 1e-2,
        "max pair-logit delta {worst} exceeds threshold 1e-2"
    );
}

/// The fixture is only meaningful if the reference actually found spans, and
/// only if a negative case stays empty. A test that passes because everything is
/// below threshold would hide a completely broken pipeline.
#[test]
fn the_fixture_covers_both_polarities() {
    let fixture = fixture();
    let mut positive = 0usize;
    let mut negative = 0usize;
    for case in fixture["cases"].as_array().unwrap() {
        let total: usize = case["spans"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row.as_array().unwrap().len())
            .sum();
        if total > 0 {
            positive += 1;
        } else {
            negative += 1;
        }
    }
    assert!(positive > 0, "no case extracts anything");
    assert!(
        negative > 0,
        "no case is empty, so false positives are unchecked"
    );
}
