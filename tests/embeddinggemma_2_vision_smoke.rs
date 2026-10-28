//! End-to-end smoke test for EmbeddingGemma-2 vision forward path.
//! Run with:
//!   RMI_EMBEDDINGGEMMA_2=/path/to/embeddinggemma-2-F16.gguf \
//!   cargo test --profile release-fast --test embeddinggemma_2_vision_smoke

use std::sync::Arc;

#[test]
fn vision_forward_returns_correct_shape_finite_values() {
    let Some(path) = std::env::var_os("RMI_EMBEDDINGGEMMA_2") else {
        return;
    };
    let loader = rust_model_inference::GGUFLoader::from_file(path).expect("load GGUF");
    let source: Arc<dyn rust_model_inference::TensorSource> = Arc::new(loader);

    let model =
        rust_model_inference::models::embeddinggemma2::EmbeddingGemmaVisionModel::from_source(
            &source,
        )
        .expect("EmbeddingGemmaVisionModel::from_source");

    let cfg = &model.config;
    assert_eq!(cfg.depth, 16);
    assert_eq!(cfg.embd, 768);
    assert_eq!(cfg.head_count, 12);
    assert_eq!(cfg.head_dim, 64);
    assert_eq!(cfg.patch_size, 16);
    assert_eq!(cfg.spatial_merge_size, 3);
    assert_eq!(cfg.projection_dim, 1536);

    // Build a synthetic 48x48 RGB image. After patch_size=16, we get
    // 3×3 = 9 patches. After merge=3, we get 1×1 = 1 merged token.
    let image_h = 48;
    let image_w = 48;
    let pixels: Vec<f32> = (0..image_h * image_w * 3)
        .map(|i| (i as f32 * 0.001).sin() * 0.5 + 0.5)
        .collect();
    let tower = model
        .encode_tower(&pixels, image_h, image_w)
        .expect("vision tower encode");
    assert_eq!(tower.len(), 768);
    let tokens = model
        .encode(&pixels, image_h, image_w)
        .expect("vision encode");
    // A 3×3 patch block pools to one 768-wide tower row, then projects to
    // the 512-wide text embedding space.
    assert_eq!(tokens.len(), 512);
    let n_finite = tokens.iter().filter(|v| v.is_finite()).count();
    assert_eq!(n_finite, tokens.len(), "non-finite values in vision output");
    // Different image: solid red instead of noisy gradient.
    let pixels_alt: Vec<f32> = (0..image_h * image_w * 3)
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect();
    let tokens_alt = model
        .encode(&pixels_alt, image_h, image_w)
        .expect("vision encode alt");
    let diff: f32 = tokens
        .iter()
        .zip(tokens_alt.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 0.0,
        "two different images produced identical embeddings (diff={diff})"
    );
}
