//! End-to-end smoke test for EmbeddingGemma-2 audio forward path.
//! Run with:
//!   RMI_EMBEDDINGGEMMA_2=/path/to/embeddinggemma-2-F16.gguf \
//!   cargo test --profile release-fast --test embeddinggemma_2_audio_smoke

use std::sync::Arc;

#[test]
fn audio_forward_returns_correct_shape_finite_values() {
    let Some(path) = std::env::var_os("RMI_EMBEDDINGGEMMA_2") else {
        return;
    };
    let loader = rust_model_inference::GGUFLoader::from_file(path).expect("load GGUF");
    let source: Arc<dyn rust_model_inference::TensorSource> = Arc::new(loader);

    let model =
        rust_model_inference::models::embeddinggemma2::EmbeddingGemmaAudioModel::from_source(
            &source,
        )
        .expect("EmbeddingGemmaAudioModel::from_source");

    let cfg = &model.config;
    assert_eq!(cfg.depth, 12);
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.ffn_hidden_size, 4096);
    assert_eq!(cfg.projection_dim, 1536);

    // Generate 1 second of 16 kHz sine-wave audio (the embedding path
    // doesn't care about content, only that values are finite).
    let sample_count = 16000;
    // 440 Hz sine — produces clear energy at k=11 (440 Hz bin).
    let pcm_440: Vec<i16> = (0..sample_count)
        .map(|i| ((i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 16000.0).sin() * 16000.0) as i16)
        .collect();
    let tokens = model.encode(&pcm_440).expect("audio encode 440Hz");
    // Different signal — 1000 Hz sine, should produce different embeddings.
    let pcm_1000: Vec<i16> = (0..sample_count)
        .map(|i| {
            ((i as f32 * 2.0 * std::f32::consts::PI * 1000.0 / 16000.0).sin() * 16000.0) as i16
        })
        .collect();
    let tokens_alt = model.encode(&pcm_1000).expect("audio encode 1000Hz");
    // Verify different inputs produce different embeddings.
    let diff: f32 = tokens
        .iter()
        .zip(tokens_alt.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 0.0,
        "two different signals produced identical embeddings (diff={diff})"
    );
    // Without center padding: n_frames = (16000 - 400) / 160 = 97 hops + offset 0 = 98 frames.
    // n_hidden = n_frames / 4 = 24 (integer division).
    let n_frames = 98 / 4;
    assert_eq!(tokens.len(), n_frames * cfg.projection_dim);
    // Verify all outputs are finite.
    let n_finite = tokens.iter().filter(|x| x.is_finite()).count();
    assert_eq!(n_finite, tokens.len(), "non-finite values in audio output");
}
