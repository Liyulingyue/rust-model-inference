//! Smoke integration test for `fastino/gliner2-multi-v1`.
//!
//! Run with `RMI_GLINER2_MULTI_V1_GGUF=/path/to/gliner2-multi-v1-f32.gguf`.
//!
//! This is the **first size variant of the SpanExtractor (pre-2.5) family**.
//! Decide and `gliner2-large-v1` are both DeBERTa-v3-large, so every dimension
//! they exercise was 1024 / 24 layers / 16 heads. Nothing in the Rust inference
//! path was allowed to assume that, and this test is what holds that line: the
//! dims arrive from GGUF metadata, so base-v1 needed no Rust change at all.
//!
//! The two things this pins that Decide / large-v1 could not:
//!
//! - The encoder dims are identical to gliner2-base-v1's (768 / 12 / 12 / 3072):
//!   mDeBERTa-v3's published config is field-for-field the same as
//!   deberta-v3-base apart from `vocab_size`. So the encoder half is a pure
//!   reweight, and the vocab half is where all the work was.
//! - `counting_layer: "count_lstm"` here (large-v1's variant), so `count_embed`
//!   is a projector rather than base-v1's 2-layer transformer. Again it never
//!   reaches the GGUF.
//! - The marker ids used to be a `const` array pinned at 128000..128010. For
//!   this model the block starts at 250101, so every `[P]`/`[L]`/`[E]` was
//!   encoded at an id that does not exist in a 250112-row table. The ids are now
//!   derived from the tokenizer's own piece count, which is what makes the same
//!   prompt builder work across vocabularies.
//!
//! Byte-exact parity lives in `tests/gliner2_base_v1_parity.rs`.

use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};
use rust_model_inference::models::gliner::prompt::{Label, Task};
use rust_model_inference::models::gliner::GlinerModel;

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var_os("RMI_GLINER2_MULTI_V1_GGUF").map(std::path::PathBuf::from)
}

fn load_model() -> Option<(Box<dyn std::any::Any>, GlinerModel<'static>)> {
    let path = gguf_path()?;
    if !path.exists() {
        eprintln!("skipping: {} does not exist", path.display());
        return None;
    }
    let source = open_model_source(&path, ComponentRole::Llm).expect("open gliner2 gguf");
    // The model holds a `&'a TensorSource`. Leak the source to 'static for
    // the test's lifetime; the OS reclaims the bytes on process exit.
    let leaked: &'static dyn rust_model_inference::core::tensor::TensorSource = Box::leak(source);
    let model = GlinerModel::from_source(leaked).expect("load gliner2 multi-v1");
    Some((Box::new(()), model))
}

fn case_intent() -> Task {
    Task::new(
        "intent",
        vec![
            Label {
                name: "refund".into(),
                description: None,
                examples: Vec::new(),
            },
            Label {
                name: "other".into(),
                description: None,
                examples: Vec::new(),
            },
        ],
    )
}

fn case_sentiment() -> Task {
    Task::new(
        "sentiment",
        ["positive", "negative", "mixed", "neutral"]
            .into_iter()
            .map(|name| Label {
                name: name.into(),
                description: None,
                examples: Vec::new(),
            })
            .collect(),
    )
}

fn case_aspects() -> Task {
    let mut task = Task::new(
        "aspects",
        vec![
            Label {
                name: "battery".into(),
                description: Some("power life".into()),
                examples: Vec::new(),
            },
            Label {
                name: "keyboard".into(),
                description: Some("typing feel".into()),
                examples: Vec::new(),
            },
            Label {
                name: "camera".into(),
                description: Some("image quality".into()),
                examples: Vec::new(),
            },
        ],
    );
    task.multi_label = true;
    task.cls_threshold = 0.4;
    task
}

#[test]
fn contract_reports_base_dims_from_metadata() {
    let Some((_anchor, model)) = load_model() else {
        return;
    };
    // The whole point: the config is not Decide's. 768 / 12 / 12 / 3072, read
    // out of GGUF metadata rather than compiled in.
    assert_eq!(model.config().n_embd, 768);
    assert_eq!(model.config().n_layer, 12);
    assert_eq!(model.config().n_head, 12);
    assert_eq!(model.config().n_ff, 3072);
    // The only dimension that is *not* the same as gliner2-base-v1: mDeBERTa-v3's
    // SPM has 250101 pieces, so the embedding table is 250112 rows. This is the
    // field the schema markers hang off, and getting it wrong is silent — the
    // prompt still tokenizes, every marker just lands on another row.
    assert_eq!(model.config().vocab_size, 250112);
    // `head_dim` is 64 for base *and* for large — 768/12 and 1024/16 coincide —
    // so it is not evidence of anything by itself. It is asserted because the
    // disentangled-attention scale divisor is `sqrt(head_dim * 3)` and this is
    // the first size where that 64 came from a division other than 1024/16; a
    // converter that paired base's hidden_size with large's head_count would
    // silently produce 42 here.
    assert_eq!(model.config().head_dim, 64);
    // `count_lstm_v2` is a different head shape than large-v1's `count_lstm`,
    // and none of it is packed: the converter drops `span_rep.*` /
    // `count_embed.*` / `count_pred.*` and refuses to drop anything else, so
    // the classifier is still the 2-layer intermediate=hidden*2 form with ReLU.
}

#[test]
fn refund_classifier_picks_refund() {
    let Some((_anchor, model)) = load_model() else {
        return;
    };
    let tasks = vec![case_intent()];
    let text = "Refund please";
    let encoded = model.encode_prompt(&tasks, text).expect("encode");
    let hidden = model.forward(&encoded.input_ids, 0).expect("forward");
    let results = model
        .score_prompt(&tasks, &encoded, &hidden)
        .expect("score");
    assert_eq!(results.len(), 1);
    let scores = &results[0].scores;
    assert_eq!(scores.len(), 2);
    assert!(
        scores[0].logit > scores[1].logit,
        "refund should outrank other"
    );
    assert_eq!(results[0].selected, vec!["refund".to_string()]);
}

#[test]
fn multitask_routes_sentiment_and_aspects() {
    let Some((_anchor, model)) = load_model() else {
        return;
    };
    let text = "Battery dies, but the keyboard is excellent";
    let tasks = vec![case_sentiment(), case_aspects()];
    let encoded = model.encode_prompt(&tasks, text).expect("encode");
    let hidden = model.forward(&encoded.input_ids, 0).expect("forward");
    let results = model
        .score_prompt(&tasks, &encoded, &hidden)
        .expect("score");
    assert_eq!(results.len(), 2);

    let sentiment = &results[0];
    assert_eq!(sentiment.task, "sentiment");
    let mass: f32 = sentiment.scores.iter().map(|s| s.probability).sum();
    assert!((mass - 1.0).abs() < 1e-4, "single-label head must softmax");

    let aspects = &results[1];
    assert_eq!(aspects.task, "aspects");
    assert!(aspects.multi_label);
    let by_label: std::collections::HashMap<&str, f32> = aspects
        .scores
        .iter()
        .map(|score| (score.label.as_str(), score.probability))
        .collect();
    assert!(by_label["battery"] > by_label["camera"]);
    assert!(by_label["keyboard"] > by_label["camera"]);
    assert!(aspects.selected.contains(&"battery".to_string()));
    assert!(aspects.selected.contains(&"keyboard".to_string()));
}

#[test]
fn example_conditioning_changes_selection() {
    let Some((_anchor, model)) = load_model() else {
        return;
    };
    let mut task = case_intent();
    task.prompt = Some("Choose the route".into());
    task.labels[0]
        .examples
        .push(("Refund now".into(), "refund".into()));
    let tasks = vec![task];
    let text = "I need a refund";
    let encoded = model.encode_prompt(&tasks, text).expect("encode");
    let hidden = model.forward(&encoded.input_ids, 0).expect("forward");
    let results = model
        .score_prompt(&tasks, &encoded, &hidden)
        .expect("score");
    assert_eq!(results[0].selected, vec!["refund".to_string()]);
}

#[test]
fn long_position_context_matches_the_reference() {
    let Some((_anchor, model)) = load_model() else {
        return;
    };
    // 60 repetitions of "great" — the same input `gliner2_base_v1.rs` asserts
    // resolves to "positive". It does not here, and that is not a bug: the
    // reference agrees with us. Scored through the same
    // `SchemaTransformer` + `DebertaV2Model` path, multi-v1 gives
    // positive -1.7515 / negative 1.5389, so it picks "negative", while
    // gliner2-base-v1 gives positive 10.6351 / negative -11.9870.
    //
    // Despite the inherited test name this is *not* a long-context case: the
    // prompt is 76 tokens here and 72 for both DeBERTa checkpoints, because
    // SentencePiece folds the repeated word into a few pieces. What it does
    // cover is the multi-label prompt layout and finite logits over a prompt
    // with several rows, and the label is pinned to the reference's answer so a
    // change in either direction is visible.
    let text = std::iter::repeat("great ").take(60).collect::<String>();
    let tasks = vec![Task::new(
        "intent",
        ["positive", "negative"]
            .into_iter()
            .map(|name| Label {
                name: name.into(),
                description: None,
                examples: Vec::new(),
            })
            .collect(),
    )];
    let encoded = model.encode_prompt(&tasks, &text).expect("encode");
    let hidden = model.forward(&encoded.input_ids, 0).expect("forward");
    let results = model
        .score_prompt(&tasks, &encoded, &hidden)
        .expect("score");
    assert_eq!(results[0].selected, vec!["negative".to_string()]);
    assert!(results[0].scores.iter().all(|s| s.logit.is_finite()));
}

#[test]
fn inputs_match_the_pinned_schema_marker_positions() {
    let Some((_anchor, model)) = load_model() else {
        return;
    };
    let tasks = vec![case_intent()];
    let text = "Refund please";
    let encoded = model.encode_prompt(&tasks, text).expect("encode");
    let markers = &encoded.markers[0].positions;
    // One [P] (system prompt row) + one [L] per label = 3 marker rows, the
    // same shape Decide and large-v1 pin: the prompt layout does not depend on
    // the encoder size.
    assert_eq!(
        markers.len(),
        3,
        "expected [P] + 2 [L] rows, got {:?}",
        markers
    );
    assert!(
        markers[0] < markers[1] && markers[1] < markers[2],
        "markers must be in declaration order: {:?}",
        markers
    );
}
