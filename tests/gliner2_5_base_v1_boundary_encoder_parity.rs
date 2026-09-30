//! Byte-exact parity for `BoundaryEncoder.forward` against the GLiNER2
//! reference (`tools/oracle/gliner_boundary/dump_boundary_encoder.py`).
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=/path/to/gliner2.5-base-v1-f32.gguf`.
//! The reference uses ``target/gliner2-oracle/gliner2/models/boundary/
//! encoding.py::BoundaryEncoder.forward`` (eval mode) on the same fixed
//! `text_states` input as the Rust smoke, and saves the resulting
//! `states` + `mask` to
//! `tests/fixtures/gliner2.5-base-v1/boundary-encoder-golden.json`.
//!
//! The boundary state encoding is a closed-form projection (no top-K,
// no sampling), so byte-for-byte equality is the strongest possible check:
//! any wrong activation, wrong tensor index, wrong residual order, or
//! wrong attention mask produces a delta > 0 on the F32 bits.
//!
//! Regenerate with:
//!     PYTHONPATH=target/gliner2-oracle \
//!         models/.venv/bin/python3 \
//!         tools/oracle/gliner_boundary/dump_boundary_encoder.py

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner_boundary::{BoundaryEncoding, BoundaryModel};

const GGUF: &str = "models/gliner2.5-base-v1/gliner2.5-base-v1-f32.gguf";
const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/boundary-encoder-golden.json";

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var_os("RMI_GLINER2_5_BASE_V1_GGUF").map(std::path::PathBuf::from)
}

fn loaded_model() -> Option<(Box<dyn std::any::Any>, BoundaryModel<'static>)> {
    let path = match gguf_path() {
        Some(path) => path,
        None => {
            eprintln!("skipping: set RMI_GLINER2_5_BASE_V1_GGUF to enable this test");
            return None;
        }
    };
    if !path.exists() {
        eprintln!("skipping: {} does not exist", path.display());
        return None;
    }
    let source = GGUFLoader::from_file(&path).expect("open boundary GGUF");
    let leaked: &'static dyn TensorSource = Box::leak(Box::new(source));
    let model = BoundaryModel::from_source(leaked).expect("load boundary model");
    Some((Box::new(()), model))
}

trait BoundaryEncodingExt {
    fn boundary_len(&self) -> usize;
}

impl BoundaryEncodingExt for BoundaryEncoding {
    fn boundary_len(&self) -> usize {
        self.seq_len + 1
    }
}

#[test]
fn matches_the_reference_stack() {
    let Some((_anchor, model)) = loaded_model() else { return };
    let raw = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_boundary_encoder.py"));
    let fixture: serde_json::Value =
        serde_json::from_str(&raw).expect("parse boundary-encoder-golden.json");

    let hidden_size = model.config.n_embd;
    let seq_len = fixture["config"]["seq_len"].as_u64().expect("seq_len") as usize;
    let valid_tokens = fixture["config"]["valid_tokens"].as_u64().expect("valid_tokens") as usize;

    // Deterministic text_states mirroring the Python dump.
    let mut text_states: Vec<f32> = Vec::with_capacity(seq_len * hidden_size);
    for i in 0..(seq_len * hidden_size) {
        let v = (i as f32) * 0.011;
        text_states.push(v.sin() - 1.0);
    }
    let mut text_mask: Vec<Vec<bool>> = vec![vec![false; seq_len]];
    for slot in text_mask[0].iter_mut().take(valid_tokens) {
        *slot = true;
    }

    let encoding = model.boundary.forward(&text_states, &text_mask);

    // Mask equality first (cheap gate; if wrong, the projection is wrong).
    let want_mask: Vec<bool> = fixture["mask"]
        .as_array()
        .expect("mask")
        .iter()
        .map(|v| v.as_bool().expect("mask element"))
        .collect();
    assert_eq!(want_mask, &encoding.mask[..want_mask.len()],
        "boundary mask differs from the reference ({} vs {})",
        want_mask.iter().filter(|m| **m).count(),
        encoding.mask.iter().filter(|m| **m).count(),
    );

    // States equality.
    let want_states: Vec<f32> = fixture["states"]
        .as_array()
        .expect("states")
        .iter()
        .map(|v| v.as_f64().expect("state element") as f32)
        .collect();
    assert_eq!(want_states.len(), encoding.states.len(),
        "states length {} != expected {}",
        encoding.states.len(), want_states.len());
    let mut max_delta: f32 = 0.0f32;
    let mut first_diff: Option<(usize, f32, f32)> = None;
    for (i, (got, want)) in encoding.states.iter().zip(want_states.iter()).enumerate() {
        let delta = (got - want).abs();
        if delta > max_delta {
            max_delta = delta;
            if first_diff.is_none() {
                first_diff = Some((i, *got, *want));
            }
        }
    }
    assert!(
        max_delta < 1e-4,
        "max state delta {max_delta} exceeds threshold 1e-4 (first diff at index {first_diff:?})",
    );
}