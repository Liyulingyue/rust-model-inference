//! End-to-end embedding integration test for the **standard**
//! `unsloth/gemma-3-270m-it-GGUF` (Q4_K_M, 242 MiB).
//!
//! Run with
//! `RMI_GEMMA_3_270M_IT_MODEL=/path/to/gemma-3-270m-it-Q4_K_M.gguf`.
//!
//! # Scope
//!
//! Exercises the engine's `models::gemma3::compute_embedding`
//! forward path on the standard (non-BitLinear) gemma3 270M GGUF
//! end-to-end. Same architecture as the BitNet 270M (gemma3
//! decoder with 4-norm sandwich, 640-dim hidden, 4:1 GQA at
//! head_dim=256, 18 layers) but:
//!
//! - **Standard Q4_K / Q5_0 / Q6K / Q8_0 mixed-quant matmul weights**
//!   instead of I2_S BitLinear.
//! - **`sliding_window = 512`** attention (standard gemma3 hybrid
//!   local/global; BitNet-270M has no sliding window).
//! - **SPM tokenizer** with the same `model = "llama"` shape as
//!   BitNet-270M but EOS=106 (`<end_of_turn>` marker) instead
//!   of EOS=1.
//! - **Q8_0-quantized token embedding** (`token_embd.weight`)
//!   instead of F16 — `static_weight` dequantizes Q8_0 to F32 on
//!   load (the F16 path is preserved for BitNet 270M).
//!
//! # What this pins
//!
//! 1. **Shape** — exactly 640 dimensions (matches
//!    `gemma3.embedding_length=640`).
//! 2. **Finite** — every element is a normal f32 (no NaN / ±Inf).
//! 3. **Non-degenerate** — at least one element is non-zero
//!    (regression guard against mis-decode of the Q8_0 token
//!    embedding or the mixed-quant matmul weights).
//! 4. **Value range** — every element falls in `[-300.0, 300.0]`.
//!    Standard gemma3 forward produces embeddings in a slightly
//!    wider band than BitNet-270M (IT models with last-token
//!    pooling can have outliers; we observe ±90 on short prompts).
//!    The [-300, 300] envelope catches gross overflow / mis-decoding
//!    without false positives on the natural magnitude.
//! 5. **Determinism** — same prompt → byte-identical embedding
//!    across two independent runs.
//! 6. **Discrimination** — two unrelated prompts produce
//!    embeddings whose cosine similarity is far from 1.0.
//! 7. **L2 norm in reasonable range** — between 0.1 and 1000.
//!
//! # Performance
//!
//! On a 4-core / 7.5 GiB box with `release-fast`, full 18-layer
//! forward is ~1.0s for a short prompt.

use rust_model_inference::models::gemma3::compute_embedding;
use rust_model_inference::GGUFLoader;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_GEMMA_3_270M_IT_MODEL")?;
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
fn gemma3_270m_it_e2e_shape_and_finite() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_270M_IT_MODEL not set");
        return;
    };
    let v = embed(&loader, "Hello, world!");
    assert_eq!(v.len(), 640, "gemma-3-270m-it output dim must be 640");
    for (i, x) in v.iter().enumerate() {
        assert!(x.is_finite(), "element {i} must be finite, got {x}");
    }
}

#[test]
fn gemma3_270m_it_e2e_value_range_and_non_degenerate() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_270M_IT_MODEL not set");
        return;
    };
    let v = embed(&loader, "Hello, world!");
    assert!(v.iter().any(|x| *x != 0.0), "embedding must not be all-zero");
    for x in &v {
        assert!(
            *x >= -300.0 && *x <= 300.0,
            "value {x} out of expected range"
        );
    }
    let l2 = l2_norm(&v);
    assert!(
        l2 > 0.1 && l2 < 1000.0,
        "L2 norm {l2} outside reasonable range [0.1, 1000]"
    );
}

#[test]
fn gemma3_270m_it_e2e_determinism() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_270M_IT_MODEL not set");
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

#[test]
fn gemma3_270m_it_e2e_prompt_discrimination() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_270M_IT_MODEL not set");
        return;
    };
    // Two long, topic-distinct prompts. With last-token pooling
    // and only 18 layers, the final-token representation is
    // dominated by the last few tokens — so we deliberately
    // embed multiple sentences per prompt and accept a more
    // permissive 0.99 threshold (IT models are not trained
    // embeddings and the discrimination is weaker than the
    // dedicated BitNet-270M Embedding model).
    let cooking = embed(
        &loader,
        "To bake sourdough bread, mix flour, water, salt, and a \
         sourdough starter. Let the dough rest overnight, then \
         shape the loaves and bake at 230 degrees Celsius for \
         thirty minutes with steam.",
    );
    let software = embed(
        &loader,
        "Rust is a systems programming language focused on safety, \
         speed, and concurrency. It achieves memory safety without \
         garbage collection through its borrow checker and \
         ownership system, which enforces a strict set of rules \
         at compile time.",
    );
    let sim = cosine(&cooking, &software);
    assert!(
        sim < 0.99,
        "two topic-distinct prompts should not be >0.99 similar \
         (got {sim}); IT-model last-token pooling is inherently \
         weak at discrimination, so we only catch obvious \
         uniformity here"
    );
}

#[test]
fn gemma3_270m_it_e2e_longer_prompt_finite() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_270M_IT_MODEL not set");
        return;
    };
    // Multi-token prompt: exercises that the standard matmul path
    // (per-token Q8_0 quantize + SIMD matmul) produces stable output
    // across token counts. 32 tokens is comfortably below the
    // 512-token sliding-window so the sliding-window mask is
    // active but not yet constraining.
    let prompt = "The history of natural language processing began \
                  in the 1950s with rule-based systems, evolved through \
                  statistical methods, and now uses deep learning.";
    let v = embed(&loader, prompt);
    assert_eq!(v.len(), 640);
    assert!(v.iter().all(|x| x.is_finite()));
    assert!(v.iter().any(|x| *x != 0.0));
}