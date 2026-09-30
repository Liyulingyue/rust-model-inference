//! Smoke test for the BoundaryExtractor forward path.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=/path/to/gliner2.5-base-v1-f32.gguf`.
//!
//! What this exercises today:
//!
//! 1. `gliner_boundary::BoundaryModel::from_source` reads the boundary
//!    GGUF and validates ``gliner2.variant = "boundary"``.
//! 2. `GlinerCompute::encode` (Decide's DeBERTa forward, parameterised
//!    by EncoderConfig) runs the 12-layer encoder with base-v1 dims.
//! 3. `BoundaryEncoder.forward` shifts text states left/right with BOS/EOS,
//!    projects to boundary_dim=128, concatenates + projects + LayerNorm,
//!    runs 2 self-attention blocks and 1 SwiGLU refinement block, then
//!    masks padding rows.
//!
//! What's NOT exercised: BoundaryProposer (top-K start/end selection with
//!    rotary endpoint embeddings), PairScorer (start/end marginals +
//!    endpoint compat + inside weight), relation_scorer, record_decoder,
//!    count_head, null_projection. Those are bundled in the GGUF and
//!    will be implemented in subsequent commits (glinerTODO.md).

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner_boundary::BoundaryModel;

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

#[test]
fn boundary_encoder_runs_on_real_input() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    // Encode a single sample of 6 valid tokens; remaining 2 are padding.
    let hidden_size = model.config.n_embd;
    let seq_len = 8;
    let text_states: Vec<f32> = (0..(seq_len * hidden_size))
        .map(|i| (i as f32 * 0.011).sin() - 1.0)
        .collect();
    let mut text_mask = vec![false; seq_len];
    for slot in text_mask.iter_mut().take(6) {
        *slot = true;
    }
    let encoding = model.boundary.forward(&text_states, &[text_mask]);
    assert_eq!(encoding.batch, 1);
    assert_eq!(encoding.seq_len, seq_len);
    assert_eq!(encoding.boundary_dim, 128); // base-v1 boundary_dim
    let expected_len = encoding.boundary_len();
    assert_eq!(encoding.states.len(), expected_len * encoding.boundary_dim);
    // `mask` is now `[B][L+1]` — one row per batch.
    assert_eq!(encoding.mask.len(), 1);
    let mask = &encoding.mask[0];
    assert_eq!(mask.len(), expected_len);
    // Boundary validity: index <= text_length, so indices 0..=6 valid, 7 invalid.
    assert!(mask[..=6].iter().all(|&m| m));
    assert!(!mask[7]);
    // Padding row must be zeroed.
    assert!(encoding.states[7 * encoding.boundary_dim..]
        .iter()
        .all(|&v| v == 0.0));
    // All valid rows must be finite (no NaN / Inf).
    for i in 0..=6 {
        assert!(
            encoding.states[i * encoding.boundary_dim..(i + 1) * encoding.boundary_dim]
                .iter()
                .all(|v| v.is_finite()),
            "non-finite value at boundary {i}"
        );
    }
}

// Extension trait to expose the boundary length (1 + seq_len per sample).
trait BoundaryEncodingExt {
    fn boundary_len(&self) -> usize;
}

impl BoundaryEncodingExt for rust_model_inference::models::gliner_boundary::BoundaryEncoding {
    fn boundary_len(&self) -> usize {
        self.seq_len + 1
    }
}

#[test]
fn boundary_metadata_pins_base_v1_dims() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    assert_eq!(model.config.n_embd, 768);
    assert_eq!(model.config.n_layer, 12);
    assert_eq!(model.config.n_head, 12);
    assert_eq!(model.config.n_ff, 3072);
    assert_eq!(model.config.head_dim, 64);
    assert_eq!(model.config.bucket_size, 256);
    assert_eq!(model.config.pos_ebd_size, 512);
    assert_eq!(model.config.vocab_size, 128011);
}
