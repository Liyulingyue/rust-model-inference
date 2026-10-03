//! End-to-end byte parity for `fastino/GLiNER2.5-Decide-1B`.
//!
//! Mirrors `tools/oracle/gliner2/dump_ettin_golden.py`, which runs the
//! checkpoint through `transformers`' own `ModernBertModel` — the same object the
//! reference reaches via `AutoModel.from_pretrained`, so there is no GLiNER-side
//! encoder to mirror and the port is measured directly against it.
//!
//! This is the only checkpoint in the family whose encoder is not DeBERTa-v2,
//! and every one of those differences is a way for a plausible-looking forward
//! to be wrong while keeping the logits finite:
//!
//! - **LayerNorm without bias**, not RMSNorm
//! - **interleaved** `Wqkv` (`view(seq, 3, heads, head_dim)`), not three
//!   contiguous blocks
//! - **GeLU GLU** (`act(first) * second` after `chunk(2)`), not SwiGLU
//! - **hybrid attention**: layers 0, 3, 6 … global, the rest a 128-wide window
//! - `attn_norm` is `Identity` at layer 0, and the checkpoint stores no tensor
//!   for it
//!
//! Run with `RMI_GLINER2_DECIDE_1B_GGUF=.../GLiNER2.5-Decide-1B-f32.gguf`.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner::prompt::{Label, Task};
use rust_model_inference::models::gliner_ettin::{self as ettin, EttinConfig};
use serde_json::Value;

const GGUF: &str = "models/GLiNER2.5-Decide-1B/GLiNER2.5-Decide-1B-f32.gguf";
const FIXTURE: &str = "tests/fixtures/GLiNER2.5-Decide-1B/classify-golden.json";

/// Same tolerance as the other span parity tests. Decide measured 7.2e-6 and
/// large-v1 2.193e-5, so this leaves room for F32 accumulation order across a
/// 28-layer encoder while a structural error still moves logits by whole units.
const LOGIT_TOLERANCE: f32 = 1e-4;

fn fixture() -> Value {
    let raw = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_ettin_golden.py"));
    serde_json::from_str(&raw).expect("parse fixture")
}

fn tasks_from(value: &Value) -> Vec<Task> {
    value
        .as_array()
        .expect("tasks array")
        .iter()
        .map(|raw| {
            let mut task = Task::new(
                raw["name"].as_str().expect("task name"),
                raw["labels"]
                    .as_array()
                    .expect("labels")
                    .iter()
                    .map(|label| Label {
                        name: label["name"].as_str().expect("label name").to_string(),
                        description: None,
                        examples: Vec::new(),
                    })
                    .collect(),
            );
            task.multi_label = raw
                .get("multi_label")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            task.cls_threshold = raw
                .get("cls_threshold")
                .and_then(Value::as_f64)
                .unwrap_or(0.5) as f32;
            task
        })
        .collect()
}

fn ids_from(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .expect("input_ids")
        .iter()
        .map(|id| id.as_u64().expect("token id") as u32)
        .collect()
}

#[test]
fn matches_the_reference_stack() {
    let path = std::path::Path::new(GGUF);
    if !path.exists() {
        panic!("missing {GGUF}; run tools/converter/gliner/convert_ettin.py first");
    }
    // The encoder and the task head are two views of one GGUF. Both borrow the
    // source for the whole test, so it is leaked once up front rather than
    // moved into a leak later; the OS reclaims it on exit.
    let source = GGUFLoader::from_file(path).expect("open Ettin GGUF");
    let leaked: &'static dyn TensorSource = Box::leak(Box::new(source));
    let config = EttinConfig::from_source(leaked).expect("read Ettin config");
    let weights = ettin::load_weights(leaked, &config).expect("load Ettin weights");
    let model = rust_model_inference::models::gliner::GlinerModel::from_source(leaked)
        .expect("load the task head via the shared loader");

    // The dimensions are the whole point: this encoder is 1792 wide with 28
    // heads, and nothing in the DeBERTa path could produce it.
    assert_eq!(config.n_embd, 1792);
    assert_eq!(config.n_layer, 28);
    assert_eq!(config.n_head, 28);
    assert_eq!(config.head_dim, 64);
    assert_eq!(config.n_ff, 3840);
    assert_eq!(config.norm_eps, 1e-5);
    // Hybrid schedule, and layer 0 is the one without an `attn_norm` tensor.
    assert_eq!(config.local_window, Some(128));
    assert_eq!(config.global_every_n_layers, 3);
    assert_eq!(config.identity_attn_norm, vec![0]);
    assert!(config.is_local_layer(1) && !config.is_local_layer(3));
    assert_eq!(config.local_attention(1), Some((64, 64)));
    assert_eq!(config.local_attention(3), None);

    let value = fixture();
    let cases = value["cases"].as_array().expect("cases");
    assert!(
        cases.len() >= 6,
        "fixture looks truncated: {} cases",
        cases.len()
    );

    for case in cases {
        let text = case["text"].as_str().expect("text");
        let tasks = tasks_from(&case["tasks"]);
        let want_ids = ids_from(&case["input_ids"]);

        // The prompt has to be built before the encoder can run, and the ids are
        // what the tokenizer produced: `Ġ`-prefixed byte-level pieces, not
        // SentencePiece ones.
        let encoded = model.encode_prompt(&tasks, text).expect("encode prompt");
        assert_eq!(
            encoded.input_ids, want_ids,
            "input_ids differ for {text:?} — the ByteLevel BPE must match the \
             reference token by token"
        );

        let mask = vec![true; encoded.input_ids.len()];
        let hidden = ettin::encode(&weights, &config, &encoded.input_ids, &mask)
            .unwrap_or_else(|error| panic!("ettin encode {text:?}: {error}"));
        assert_eq!(hidden.len(), encoded.input_ids.len() * config.n_embd);
        assert!(
            hidden.iter().all(|value| value.is_finite()),
            "non-finite hidden state for {text:?}"
        );

        for (task_index, want_logits) in case["logits"]
            .as_array()
            .expect("logits")
            .iter()
            .enumerate()
        {
            let positions = &encoded.markers[task_index].positions;
            // Row 0 of each task's markers is `[P]`, which the reference scores
            // nowhere; the label rows follow.
            let want: Vec<f32> = want_logits
                .as_array()
                .expect("logits")
                .iter()
                .map(|value| value.as_f64().expect("logit") as f32)
                .collect();
            assert_eq!(
                positions.len() - 1,
                want.len(),
                "label count differs for {text:?}"
            );
            for (index, &position) in positions[1..].iter().enumerate() {
                let start = position * config.n_embd;
                let logit =
                    ettin::classify(&weights, &config, &hidden[start..start + config.n_embd])
                        .expect("classify");
                assert!(
                    (logit - want[index]).abs() < LOGIT_TOLERANCE,
                    "{text:?} task {task_index} label {index} ({}): \
                     logit {logit} vs {} (delta {})",
                    tasks[task_index].labels[index].name,
                    want[index],
                    (logit - want[index]).abs()
                );
            }
        }
    }
}
