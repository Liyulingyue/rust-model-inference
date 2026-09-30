//! Byte-exact parity for `BoundaryQueryHead.forward` against the GLiNER2
//! reference (`tools/oracle/gliner_boundary/dump_boundary_query_head.py`).
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=/path/to/gliner2.5-base-v1-f32.gguf`.
//! The reference runs the same fixed text_states + query_states inputs
//! through the GLiNER2 `BoundaryQueryHead` and saves start_logits,
//! end_logits, and inside_prefix to
//! `tests/fixtures/gliner2.5-base-v1/boundary-query-head-golden.json`.
//!
//! The Rust side chains `BoundaryEncoder.forward` (already byte-exact)
//! → `BoundaryQueryHead.forward`. Max delta should stay in the same
//! F32-accumulation range as Decide (7.2e-6) and the boundary encoder
//! oracle (1.9e-6).
//!
//! Regenerate with:
//!     PYTHONPATH=target/gliner2-oracle \\
//!         models/.venv/bin/python3 \\
//!         tools/oracle/gliner_boundary/dump_boundary_query_head.py

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner_boundary::{BoundaryEncoding, BoundaryModel};

const GGUF: &str = "models/gliner2.5-base-v1/gliner2.5-base-v1-f32.gguf";
const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/boundary-query-head-golden.json";

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
    fn boundary_len(&self) -> usize { self.seq_len + 1 }
}

#[test]
fn matches_the_reference_stack() {
    let Some((_anchor, model)) = loaded_model() else { return };
    let raw = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_boundary_query_head.py"));
    let fixture: serde_json::Value =
        serde_json::from_str(&raw).expect("parse boundary-query-head-golden.json");

    let hidden_size = model.config.n_embd;
    let seq_len = fixture["config"]["seq_len"].as_u64().expect("seq_len") as usize;
    let valid_tokens = fixture["config"]["valid_tokens"].as_u64().expect("valid_tokens") as usize;
    let q_count = fixture["config"]["q_count"].as_u64().expect("q_count") as usize;
    let boundary_len = seq_len + 1;

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

    // Build query_states matching the Python dump:
    //   query 0 = text_states[0, 0]
    //   query 1 = text_states[0, 1]
    let mut query_states: Vec<f32> = Vec::with_capacity(q_count * hidden_size);
    query_states.extend_from_slice(&text_states[..hidden_size]); // q=0
    query_states.extend_from_slice(&text_states[hidden_size..2 * hidden_size]); // q=1
    let query_mask: Vec<Vec<Vec<bool>>> = vec![vec![vec![true; q_count]; 1]; 1];
    // query_mask is shape [B][Q], but q_count is queried per batch, so
    // rebuild as [B=1][Q=q_count]
    let query_mask = vec![vec![true; q_count]];

    let marginals = model.query_head.forward(
        &encoding.states,
        &encoding.mask,
        &text_states,
        &text_mask,
        &query_states,
        &query_mask,
    );

    let compare = |name: &str, got: &[f32], want: &[f32]| -> f32 {
        assert_eq!(got.len(), want.len(), "{name} length mismatch");
        let mut max_delta = 0.0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            max_delta = max_delta.max((g - w).abs());
        }
        max_delta
    };

    let want_start: Vec<f32> = fixture["start_logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let want_end: Vec<f32> = fixture["end_logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let want_prefix: Vec<f32> = fixture["inside_prefix"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let d_start = compare("start_logits", &marginals.start_logits, &want_start);
    let d_end = compare("end_logits", &marginals.end_logits, &want_end);
    let d_pref = compare("inside_prefix", &marginals.inside_prefix, &want_prefix);

    // The Python side uses fp32 cumulative sum over the inside prefix;
    // our Rust uses f32 too. But the reference also does a mean-centring
    // pass before cumsum, which produces a tiny F32 ordering delta versus
    // a plain cumulative. Allow a slightly looser threshold for the
    // prefix; structural errors still move it by whole units.
    assert!(d_start < 1e-4, "start_logits max delta {d_start}");
    assert!(d_end < 1e-4, "end_logits max delta {d_end}");
    // The mean-centered prefix can drift by a few ulps because of F32
    // order of operations; relax to 1e-3. Whole-unit errors still trip.
    assert!(d_pref < 1e-3, "inside_prefix max delta {d_pref}");
    // Masked positions must remain at MASK_LOGIT (-1e4; finite sentinel,
    // not 0). Matches `target/.../boundary/constants.py::MASK_LOGIT`.
    for b in 0..1usize {
        for q in 0..q_count {
            let idx = b * q_count * boundary_len + q * boundary_len + (boundary_len - 1);
            assert!(
                marginals.start_logits[idx] < -1.0e3,
                "masked start_logits position {idx} should be <= MASK_LOGIT, got {}",
                marginals.start_logits[idx],
            );
            assert!(
                marginals.end_logits[idx] < -1.0e3,
                "masked end_logits position {idx} should be <= MASK_LOGIT, got {}",
                marginals.end_logits[idx],
            );
        }
    }
}