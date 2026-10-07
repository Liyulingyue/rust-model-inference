//! Cross-format check: loading the same model as Q8_0 instead of F32 lands
//! within a small tolerance of the F32 logits.
//!
//! The runtime's `Weight::from_quantized` dispatches on the per-tensor
//! `ggml_type`, so a GGUF that carries F32 vocab + F32 norms + Q8_0
//! weights loads unchanged. This test exists to prove the boundary wire
//! path holds across that mix — the byte-exactness is F32-only by design,
//! but the quantization adapter that produces the Q8_0 GGUF needs a
//! non-trivial runtime witness that no kernel misbehaves on it.
//!
//! The encoder is quantized too (117 of 334 tensors on base-v1, 783 MB →
//! 490 MB): `gliner::compute::decode_row` accepts the block-quantized
//! embedding types for the relative-position table and the word embedding,
//! and the per-layer projections already went through `Weight`. Only
//! `token_embd` and the 1-D norms stay F32 — see
//! `tools/converter/utils/quantize_gguf.py` for why.
//!
//! `tools/converter/utils/quantize_gguf.py` produces the Q8_0 file when run
//! against the F32 GGUF. The test reads both, runs the same text/schema
//! through `run_extraction`, and compares the pair logits.
//!
//! **Compare sorted, not by index.** `DocumentCandidatePool` is a ranked
//! top-K list, so a slot index is a rank rather than a stable identity: two
//! candidates whose scores sit within quantization noise of each other can
//! come back in the opposite order, and the same slot index then names a
//! different span. Comparing by index on this model reported a max delta of
//! 1.28 (first text) and 2.90 (second text) and read exactly like a stride
//! bug — adjacent slots appeared to trade values — while the sorted
//! comparison puts the same runs at 0.064 and 0.070, i.e. ordinary Q8_0
//! noise. A separate `tests/q8_0_row_identity.rs` pins the kernel's row
//! mapping directly so this class of misreading does not recur.
//!
//! Padded pool slots carry `MASK_LOGIT`, a large negative sentinel rather
//! than a score, so they are filtered out before the comparison.
//!
//! Run with both:
//!   `RMI_GLINER2_5_BASE_V1_GGUF=…f32.gguf  RMI_GLINER2_5_BASE_V1_Q8_0_GGUF=…q8_0.gguf`
//!
//! Skips silently when either path is missing: the F32-only path is the
//! byte-exact cover, this one is the cross-format guard.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner::prompt::{Label, Task, E_TOKEN};
use rust_model_inference::models::gliner_boundary::{run_extraction, BoundaryModel};

const TEXT: &str = "The product shipped late but the support team was great.";
const TEXT_2: &str = "Tesla delivered 500,000 cars last year. BMW should be in second place.";
const SCHEMA: &str = r#"{"entities": ["product"]}"#;
/// Median pair-logit tolerance per format, as a fraction.
///
/// Q8_0 carries 8 bits per weight and lands at 1.1–2.5% across 28 layers of
/// DeBERTa plus the boundary heads. The 4-bit k-quants carry half that and
/// land at 4.5–13% depending on the input, so the bound scales with the bit
/// width rather than being one number for all of them. These are regression
/// guards, not accuracy claims: anything above the bound means the decode
/// path changed, not that the format misbehaves.
fn median_tolerance_for(quantized: &std::path::Path) -> f32 {
    // The format comes from the `general.file_type` the converter stamped
    // into the file, not from the file name.
    let Ok(source) = GGUFLoader::from_file(quantized) else {
        return 5.0e-2;
    };
    let Some(text) = source.metadata("general.file_type") else {
        return 5.0e-2;
    };
    let Some(text) = text.to_string_val() else {
        return 5.0e-2;
    };
    match text {
        "q4_k" | "Q4_K" => 2.0e-1,
        "q6_k" | "Q6_K" => 1.0e-1,
        _ => 5.0e-2,
    }
}
/// Padded pool slots carry `MASK_LOGIT`, a large negative sentinel rather
/// than a score. Anything at or below it is padding, not a live candidate.
const MASK_LOGIT_FLOOR: f32 = -1.0e4;

fn tasks() -> Vec<Task> {
    vec![Task {
        name: "product".into(),
        prompt: None,
        labels: vec![Label {
            name: "product".into(),
            description: None,
            examples: Vec::new(),
        }],
        activation: None,
        multi_label: false,
        cls_threshold: 0.5,
        temperature: 1.0,
    }]
}

fn loaded(model_path: &std::path::Path) -> BoundaryModel<'static> {
    let source = GGUFLoader::from_file(model_path).expect("open boundary GGUF");
    let leaked: &'static dyn TensorSource = Box::leak(Box::new(source));
    BoundaryModel::from_source(leaked).expect("load boundary model")
}

fn collect_logits(model: &BoundaryModel<'_>, text: &str, _schema: &str) -> Vec<f32> {
    // `run_extraction` is the span-only entry: `{"entities": ["product"]}` with a
    // single label produces one query × one slot pair-logit vector of fixed
    // shape, which is enough to expose quantization noise along the entire
    // 28-layer DeBERTa forward.
    let (batch, _words) =
        run_extraction(model, text, &tasks(), E_TOKEN, 0).expect("run extraction");
    batch.pair_logits.clone()
}

fn assert_cross_format(f32_pair: &[f32], q8_pair: &[f32], label: &str, tolerance: f32) {
    assert_eq!(
        f32_pair.len(),
        q8_pair.len(),
        "pair-logit vectors differ in length"
    );
    // `DocumentCandidatePool` is a *ranked* top-K list, so a slot index is
    // a rank, not a stable identity: two candidates whose scores are within
    // quantization noise of each other can come back in the opposite order,
    // and the same slot index then names a different span. Comparing by
    // index therefore measures ranking jitter, not the forward pass. Sort
    // both sides and compare the multisets instead — that is the question
    // this file is actually asking.
    let sorted_multiset = |values: &[f32]| -> Vec<f32> {
        let mut kept: Vec<f32> = values
            .iter()
            .copied()
            .filter(|v| *v > MASK_LOGIT_FLOOR)
            .collect();
        kept.sort_by(|a, b| a.total_cmp(b));
        kept
    };
    let f32_sorted = sorted_multiset(f32_pair);
    let q8_sorted = sorted_multiset(q8_pair);
    assert_eq!(
        f32_sorted.len(),
        q8_sorted.len(),
        "{label}: live-slot count differs"
    );
    let abs_deltas: Vec<f32> = f32_sorted
        .iter()
        .zip(q8_sorted.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();
    let mut sorted = abs_deltas.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    assert!(
        median <= tolerance,
        "{label}: median |delta| is {median}, tolerance is {tolerance}; \
         the quantized decode path is drifting on the bulk of slots"
    );
    let mut indexed: Vec<(usize, f32, f32, f32)> = f32_sorted
        .iter()
        .zip(q8_sorted.iter())
        .enumerate()
        .map(|(i, (a, b))| (i, *a, *b, (a - b).abs()))
        .collect();
    indexed.sort_by(|x, y| y.3.partial_cmp(&x.3).unwrap());
    eprintln!(
        "{label}: {} live slots, median |delta| {median}, worst 5: {:?}",
        f32_sorted.len(),
        &indexed[..5.min(indexed.len())]
    );
}

#[test]
fn boundary_q8_0_logits_track_f32() {
    let f32_path = std::env::var_os("RMI_GLINER2_5_BASE_V1_GGUF").map(std::path::PathBuf::from);
    let q8_path = std::env::var_os("RMI_GLINER2_5_BASE_V1_Q8_0_GGUF").map(std::path::PathBuf::from);
    let (Some(f32_path), Some(q8_path)) = (f32_path, q8_path) else {
        eprintln!("skipping: set RMI_GLINER2_5_BASE_V1_GGUF and RMI_GLINER2_5_BASE_V1_Q8_0_GGUF");
        return;
    };
    if !f32_path.exists() || !q8_path.exists() {
        eprintln!(
            "skipping: {} or {} missing",
            f32_path.display(),
            q8_path.display()
        );
        return;
    }

    let f32_model = loaded(&f32_path);
    let q8_model = loaded(&q8_path);
    let tolerance = median_tolerance_for(&q8_path);

    let f32_pair = collect_logits(&f32_model, TEXT, SCHEMA);
    let q8_pair = collect_logits(&q8_model, TEXT, SCHEMA);
    assert_cross_format(&f32_pair, &q8_pair, "first text", tolerance);

    // A second text shows the gap does not blow up across inputs; the
    // quantize-and-replay risk is that a particular short input happens to
    // land on zero everywhere, which would mask a real regression.
    let f32_pair_b = collect_logits(&f32_model, TEXT_2, SCHEMA);
    let q8_pair_b = collect_logits(&q8_model, TEXT_2, SCHEMA);
    assert_cross_format(&f32_pair_b, &q8_pair_b, "second text", tolerance);
}
