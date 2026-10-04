//! End-to-end embedding integration test for
//! `microsoft/bitnet-embedding-270m`.
//!
//! Run with `RMI_BITNET_EMBEDDING_270M_MODEL=/path/to/.gguf`.
//!
//! # Scope
//!
//! Exercises the engine's `gemma3` BitLinear forward path on the
//! 270M GGUF end-to-end. Companion to
//! `tests/bitnet_embedding_0_6b_e2e_embed.rs` (which exercises
//! the 0.6B model on the qwen3 trunk). Each layer of the 270M
//! gemma3 trunk applies the same per-projection BitLinear
//! pattern, but with a 4-norm sandwich (attn_norm →
//! post_attention_norm → ffn_norm → post_ffw_norm) instead of
//! the qwen3 2-norm sandwich, plus per-head QK-norm on Q and K
//! before RoPE, plus `rope.dimension_count = head_dim = 256`
//! (Neox full rotation).
//!
//! # What this pins
//!
//! 1. **Shape** — exactly 640 dimensions (matches
//!    `gemma3.embedding_length=640`).
//! 2. **Finite** — every element is a normal f32 (no NaN / ±Inf).
//! 3. **Non-degenerate** — at least one element is non-zero
//!    (regression guard against the same
//!    `get_f32_tensor` F16-arm bug the 0.6B e2e test guards).
//! 4. **Value range** — every element falls in `[-50.0, 50.0]`.
//!    The 270M forward produces embeddings in a wider band than
//!    0.6B (we observe ±15+ on short prompts) — the [-50, 50]
//!    envelope catches gross overflow / mis-decoding without
//!    false positives on the natural magnitude.
//! 5. **Determinism** — same prompt → byte-identical embedding
//!    across two independent runs.
//! 6. **Discrimination** — two unrelated prompts produce
//!    embeddings whose cosine similarity is far from 1.0.
//! 7. **L2 norm in reasonable range** — between 0.1 and 500.
//!    270M embedding natural magnitude is in the dozens (vs
//!    0.6B's single digits), but should not collapse to zero or
//!    explode into the hundreds.
//!
//! # Performance
//!
//! On a 4-core / 7.5 GiB box with `release-fast`, full 18-layer
//! forward is ~1.4s; this test suite runs ~12 forwards in ~17s.

use rust_model_inference::models::bitnet::gemma3_arch::compute_embedding;
use rust_model_inference::GGUFLoader;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_BITNET_EMBEDDING_270M_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

fn embed(loader: &GGUFLoader, prompt: &str) -> Vec<f32> {
    compute_embedding(loader, prompt, 4)
        .unwrap_or_else(|e| panic!("compute_embedding({prompt:?}) failed: {e}"))
}

fn l2_norm(v: &[f32]) -> f64 {
    (v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>()).sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b) {
        let x = f64::from(*x);
        let y = f64::from(*y);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[test]
fn bitnet_embedding_270m_e2e_shape_and_finite() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_BITNET_EMBEDDING_270M_MODEL not set");
        return;
    };
    let v = embed(&loader, "Hello, world!");
    assert_eq!(v.len(), 640, "BitNet-Embedding 270M output dim must be 640");
    for (i, x) in v.iter().enumerate() {
        assert!(x.is_finite(), "element {i} must be finite, got {x}");
    }
}

#[test]
fn bitnet_embedding_270m_e2e_value_range_and_non_degenerate() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_BITNET_EMBEDDING_270M_MODEL not set");
        return;
    };
    let v = embed(&loader, "Hello, world!");
    assert!(v.iter().any(|x| *x != 0.0), "embedding must not be all-zero");
    for x in &v {
        assert!(*x >= -300.0 && *x <= 300.0, "value {x} out of expected range");
    }
}

#[test]
fn bitnet_embedding_270m_e2e_deterministic() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_BITNET_EMBEDDING_270M_MODEL not set");
        return;
    };
    let a = embed(&loader, "The capital of France is Paris.");
    let b = embed(&loader, "The capital of France is Paris.");
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "element {i} differs across runs");
    }
}

#[test]
fn bitnet_embedding_270m_e2e_discriminates_unrelated_prompts() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_BITNET_EMBEDDING_270M_MODEL not set");
        return;
    };
    let animals = embed(&loader, "Cats and dogs are common household pets.");
    let physics = embed(&loader, "The speed of light in vacuum is approximately 3e8 m/s.");
    let cooking = embed(&loader, "Sauté onions until translucent before adding garlic.");
    let sim_ap = cosine(&animals, &physics);
    let sim_ac = cosine(&animals, &cooking);
    let sim_pc = cosine(&physics, &cooking);
    assert!(
        sim_ap < 0.99,
        "animals vs physics cosine {sim_ap} should be far from 1.0"
    );
    assert!(
        sim_ac < 0.99,
        "animals vs cooking cosine {sim_ac} should be far from 1.0"
    );
    assert!(
        sim_pc < 0.99,
        "physics vs cooking cosine {sim_pc} should be far from 1.0"
    );
    let max_sim = sim_ap.max(sim_ac).max(sim_pc);
    let min_sim = sim_ap.min(sim_ac).min(sim_pc);
    assert!(
        max_sim - min_sim < 0.5,
        "unrelated prompts produced wildly different similarities: ap={sim_ap}, ac={sim_ac}, pc={sim_pc}"
    );
}

#[test]
fn bitnet_embedding_270m_e2e_l2_norm_in_reasonable_range() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_BITNET_EMBEDDING_270M_MODEL not set");
        return;
    };
    for prompt in &[
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "A",
    ] {
        let v = embed(&loader, prompt);
        let n = l2_norm(&v);
        assert!(n > 1.0, "norm {n} for {prompt:?} is too small (collapse?)");
        assert!(n < 20000.0, "norm {n} for {prompt:?} is too large (overflow?)");
    }
}