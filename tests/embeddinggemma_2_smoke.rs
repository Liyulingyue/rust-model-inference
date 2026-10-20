//! End-to-end smoke test for EmbeddingGemma-2 text forward path.
//! Loads the GGUF, runs `forward` on a few synthetic token IDs, checks
//! the output is the right shape (768-dim) and free of NaN/Inf.
//! Run with:
//!   RMI_EMBEDDINGGEMMA_2=/path/to/embeddinggemma-2-F16.gguf \
//!   cargo test --profile release-fast --test embeddinggemma_2_smoke

use std::sync::Arc;

#[test]
fn text_forward_returns_768_dim_finite_embedding() {
    let Some(path) = std::env::var_os("RMI_EMBEDDINGGEMMA_2") else {
        return;
    };
    let loader = rust_model_inference::GGUFLoader::from_file(path).expect("load GGUF");
    let source: Arc<dyn rust_model_inference::TensorSource> = Arc::new(loader);

    // Build the model.
    let model =
        rust_model_inference::models::embeddinggemma2::EmbeddingGemmaTextModel::from_source(
            source.as_ref(),
        )
        .expect("EmbeddingGemmaTextModel::from_source");

    // Sanity: config matches the converter contract.
    let cfg = &model.config;
    assert_eq!(cfg.n_embd, 512);
    assert_eq!(cfg.n_layer, 24);
    assert_eq!(cfg.n_head, 4);
    assert_eq!(cfg.n_ff, 2048);
    assert_eq!(cfg.n_vocab, 262_144);
    assert_eq!(cfg.head_dims.len(), 24);
    for (i, &d) in cfg.head_dims.iter().enumerate() {
        if [5, 11, 17, 23].contains(&i) {
            assert_eq!(d, 512);
        } else {
            assert_eq!(d, 256);
        }
    }

    // Run a few tokens through.
    let tokens = vec![1u32, 5, 9, 47, 91];
    let emb = model.forward(&tokens).expect("forward");

    assert_eq!(
        emb.len(),
        rust_model_inference::models::embeddinggemma2::text::EMBEDDING_DIM
    );
    assert_eq!(emb.len(), 768);

    for (i, &v) in emb.iter().enumerate() {
        assert!(v.is_finite(), "emb[{i}] = {v} (non-finite)");
    }

    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        norm > 1e-6,
        "embedding has near-zero norm ({norm}) — check forward math"
    );

    // Run with a different token sequence; embeddings should differ.
    let tokens2 = vec![100u32, 200, 300];
    let emb2 = model.forward(&tokens2).expect("forward");
    let diff: f32 = emb
        .iter()
        .zip(emb2.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    assert!(
        diff > 1e-4,
        "different inputs produced identical embeddings ({diff})"
    );
}
