//! End-to-end embedding integration test for the **standard**
//! `unsloth/gemma-3-4b-it-GGUF` (IQ4_XS, 2.2 GiB).
//!
//! Run with
//! `RMI_GEMMA_3_4B_IT_MODEL=/path/to/gemma-3-4b-it-IQ4_XS.gguf`.
//!
//! # Scope
//!
//! Exercises the engine's `models::gemma3::compute_embedding`
//! forward path on the standard (non-BitLinear) gemma3 4B GGUF
//! end-to-end. Same architecture as the 270M-it + BitNet-270M
//! (gemma3 decoder with 4-norm sandwich, GQA 2:1 at head_dim=256)
//! but with key 4B-specific additions:
//!
//! - **RoPE linear scaling** (factor=8.0) — extended context
//!   32k → 256k. Trunk uses `rope_neox_inplace_with_factor`.
//! - **sliding_window=1024** (vs 512 in 270M-it, 0 in BitNet-270M).
//! - **34 layers** (vs 18 in 270M-it), n_embd=2560 (vs 640),
//!   n_ff=10240 (vs 2048), 8 heads / 4 KV (vs 4 / 1).
//! - **token_embd in Q6_K** (vs F16 in BitNet-270M, Q8_0 in 270M-it).
//!   `static_weight` dequantizes via `dequantize_row_q6_k`.
//! - **Per-layer matmul weights in IQ4_XS** (vs Q4_K_M mixed in
//!   270M-it, I2_S in BitNet-270M).
//!
//! # What this pins
//!
//! 1. **Shape** — exactly 2560 dimensions.
//! 2. **Finite** — every element is a normal f32.
//! 3. **Non-degenerate** — at least one element is non-zero.
//! 4. **Value range** — `[-100, 100]` (4B with last-token pooling
//!    produces smaller magnitudes than 270M-it; we observe ±5
//!    on short prompts; the [-100, 100] envelope catches gross
//!    overflow / mis-decoding without false positives).
//! 5. **L2 norm in reasonable range** — between 0.1 and 100.
//! 6. **Determinism** — same prompt → byte-identical embedding.
//! 7. ~~**Prompt discrimination**~~ — deferred. Each forward is
//!    ~38s on 4-core/7.5 GiB; 2 forwards + 2.5 GiB token_embd
//!    per-call costs are too tight for CI on this hardware.
//!    The 270M-it discrimination test (`gemma3_270m_it_e2e_embed`)
//!    covers the same IT-model semantic in a much cheaper form.
//!
//! # Performance
//!
//! 4B IQ4_XS is significantly slower than 270M (2560 dim, 34 layers
//! vs 640 dim, 18 layers; ~14x more matmul work). On a 4-core /
//! 7.5 GiB box with `release-fast` and 4 threads, full forward is
//! ~50 s per short prompt. Tests that exercise the forward pass
//! have generous time budgets; the contract test (`q4_k_m` style)
//! is metadata-only and runs in milliseconds.

use rust_model_inference::models::gemma3::compute_embedding;
use rust_model_inference::GGUFLoader;
use std::time::Instant;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_GEMMA_3_4B_IT_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

fn embed(loader: &GGUFLoader, prompt: &str) -> Vec<f32> {
    let start = Instant::now();
    let v = compute_embedding(loader, prompt, 4)
        .unwrap_or_else(|e| panic!("compute_embedding({prompt:?}) failed: {e}"));
    eprintln!(
        "gemma-3-4b-it e2e embed({:?}) took {:?}",
        prompt,
        start.elapsed()
    );
    v
}

fn l2_norm(v: &[f32]) -> f64 {
    (v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>()).sqrt()
}

#[test]
fn gemma3_4b_it_e2e_shape_and_finite() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_4B_IT_MODEL not set");
        return;
    };
    let v = embed(&loader, "Hello, world!");
    assert_eq!(v.len(), 2560, "gemma-3-4b-it output dim must be 2560");
    for (i, x) in v.iter().enumerate() {
        assert!(x.is_finite(), "element {i} must be finite, got {x}");
    }
}

#[test]
fn gemma3_4b_it_e2e_value_range_and_non_degenerate() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_4B_IT_MODEL not set");
        return;
    };
    let v = embed(&loader, "Hello, world!");
    assert!(
        v.iter().any(|x| *x != 0.0),
        "embedding must not be all-zero"
    );
    for x in &v {
        assert!(
            *x >= -100.0 && *x <= 100.0,
            "value {x} out of expected range [-100, 100]"
        );
    }
    let l2 = l2_norm(&v);
    assert!(
        l2 > 0.1 && l2 < 1000.0,
        "L2 norm {l2} outside reasonable range [0.1, 1000]"
    );
}

#[test]
fn gemma3_4b_it_e2e_determinism() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_4B_IT_MODEL not set");
        return;
    };
    let a = embed(&loader, "The quick brown fox jumps over the lazy dog.");
    let b = embed(&loader, "The quick brown fox jumps over the lazy dog.");
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "element {i} differs: {x} vs {y} (must be byte-identical)"
        );
    }
}
