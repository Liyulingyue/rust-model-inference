//! End-to-end embedding integration test for the **standard**
//! `unsloth/gemma-3-1b-it-GGUF` (Q4_K_M, 769 MiB).
//!
//! Run with
//! `RMI_GEMMA_3_1B_IT_MODEL=/path/to/gemma-3-1b-it-Q4_K_M.gguf`.
//!
//! # Scope
//!
//! Exercises the engine's `models::gemma3::compute_embedding`
//! forward path on the standard (non-BitLinear) gemma3 1B GGUF
//! end-to-end. Same architecture as 270M-it (no RoPE scaling,
//! sliding_window=512) but 4B+-size dims:
//!
//! - 26 layers (vs 18 in 270M-it, 34 in 4B-it)
//! - n_embd=1152 (vs 640 in 270M-it, 2560 in 4B-it)
//! - n_ff=6912 (vs 2048 in 270M-it, 10240 in 4B-it)
//! - Mixed Q4_K / Q5_0 / Q6_K / Q8_0 matmul weights (vs Q4_K_M-only
//!   270M-it, IQ4_XS-only 4B-it)
//!
//! # What this pins
//!
//! 1. **Shape** — exactly 1152 dimensions.
//! 2. **Finite** — every element is a normal f32.
//! 3. **Non-degenerate** — at least one element is non-zero.
//! 4. **Value range** — `[-100, 100]` (1B + last-token pooling
//!    produces smaller magnitudes than 270M-it / 4B-it; ±10
//!    observed on short prompts).
//! 5. **L2 norm in reasonable range** — between 0.1 and 1000.
//! 6. **Determinism** — same prompt → byte-identical embedding.
//! 7. **Prompt discrimination** — cooking vs software cosine
//!    sim < 0.99 (IT-model + last-token pooling is weak;
//!    matches 270M-it test threshold).
//!
//! # Performance
//!
//! On a 4-core / 7.5 GiB box with `release-fast` and 4 threads,
//! full forward is ~3.4 s per short prompt — much faster than 4B
//! (~38 s) thanks to smaller dims (1152 vs 2560) and fewer
//! layers (26 vs 34). Comfortable for a full e2e suite in CI.

use rust_model_inference::models::gemma3::compute_embedding;
use rust_model_inference::GGUFLoader;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_GEMMA_3_1B_IT_MODEL")?;
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
fn gemma3_1b_it_e2e_shape_and_finite() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_1B_IT_MODEL not set");
        return;
    };
    let v = embed(&loader, "Hello, world!");
    assert_eq!(v.len(), 1152, "gemma-3-1b-it output dim must be 1152");
    for (i, x) in v.iter().enumerate() {
        assert!(x.is_finite(), "element {i} must be finite, got {x}");
    }
}

#[test]
fn gemma3_1b_it_e2e_value_range_and_non_degenerate() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_1B_IT_MODEL not set");
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
fn gemma3_1b_it_e2e_determinism() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_1B_IT_MODEL not set");
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
fn gemma3_1b_it_e2e_prompt_discrimination() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_1B_IT_MODEL not set");
        return;
    };
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
        "two topic-distinct prompts should not be >0.99 similar (got {sim})"
    );
}

#[test]
fn gemma3_1b_it_e2e_longer_prompt_finite() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_1B_IT_MODEL not set");
        return;
    };
    // 32 tokens — well within sliding_window=512 so the mask is
    // active but not constraining. Exercises that the per-token
    // Q8_0 quantize + mixed-quant matmul path is stable across
    // token counts.
    let prompt = "The history of natural language processing began \
                  in the 1950s with rule-based systems, evolved through \
                  statistical methods, and now uses deep learning.";
    let v = embed(&loader, prompt);
    assert_eq!(v.len(), 1152);
    assert!(v.iter().all(|x| x.is_finite()));
    assert!(v.iter().any(|x| *x != 0.0));
}
