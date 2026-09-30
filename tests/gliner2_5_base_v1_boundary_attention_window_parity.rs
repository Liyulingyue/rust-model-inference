//! Byte-exact parity for the `BoundaryAttentionBlock` local-window mask.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf` for
//! the settings assertion; the mask comparison itself is pure and always runs.
//!
//! `boundary_attention_window` is 128 for base-v1, so `|i - j| <= window` only
//! excludes keys once a document exceeds `2 * 128 + 1 = 257` boundary
//! positions. The end-to-end fixtures in this directory top out at 24 tokens,
//! which means a port that dropped the window entirely would still be
//! byte-exact against all of them. This fixture covers `n = 8` (band inactive)
//! and `n = 273` (band active) and pins every mask entry.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner_boundary::forward::attention_allowed;
use rust_model_inference::models::gliner_boundary::BoundaryModel;

const GGUF: &str = "models/gliner2.5-base-v1/gliner2.5-base-v1-f32.gguf";
const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/boundary-attention-window-golden.json";

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE).unwrap_or_else(|_| {
        panic!("missing {FIXTURE}; regenerate with dump_boundary_attention_window.py")
    });
    serde_json::from_str(&raw).expect("parse boundary-attention-window-golden.json")
}

#[test]
fn matches_the_reference_mask() {
    let fixture = fixture();
    let window = fixture["window"].as_u64().expect("window") as usize;
    assert_eq!(window, 128, "base-v1's boundary_attention_window");
    let shapes = fixture["shapes"].as_array().expect("shapes");
    assert!(
        shapes
            .iter()
            .any(|s| s["n"].as_u64().unwrap() as usize > 2 * window + 1),
        "the fixture must include a shape long enough for the window to bind"
    );

    for shape in shapes {
        let n = shape["n"].as_u64().expect("n") as usize;
        // The reference fixture assumes an all-valid boundary mask; the mask
        // half of the rule is checked separately below.
        let mask_row = vec![true; n];
        let allowed: Vec<Vec<usize>> = shape["allowed"]
            .as_array()
            .expect("allowed")
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect()
            })
            .collect();
        assert_eq!(allowed.len(), n, "n = {n}: one row per query position");
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    attention_allowed(&mask_row, i, j, window),
                    allowed[i].contains(&j),
                    "n = {n} window = {window}: mask[{i}][{j}] disagrees with the reference"
                );
            }
        }
    }
}

#[test]
fn the_mask_half_of_the_rule_still_applies() {
    // `allowed = mask[j] & band | diagonal`, so an invalid key stays invalid
    // for a non-diagonal position even inside the band.
    let window = 4usize;
    let mask_row = vec![true, true, false, true, true, true, true];
    for j in 0..mask_row.len() {
        for i in 0..mask_row.len() {
            let expected = (mask_row[j] && i.abs_diff(j) <= window) || i == j;
            assert_eq!(
                attention_allowed(&mask_row, i, j, window),
                expected,
                "masked key {j} from query row {i}"
            );
        }
    }
    // A padding query row still has exactly one legal key, which is what keeps
    // its softmax finite.
    let padding_row = vec![false, false, false, false, false];
    let legal: Vec<usize> = (0..padding_row.len())
        .filter(|j| attention_allowed(&padding_row, 4, *j, window))
        .collect();
    assert_eq!(legal, vec![4], "padding row must self-attend");
}

#[test]
fn gguf_carries_the_window() {
    let Some(path) = std::env::var_os("RMI_GLINER2_5_BASE_V1_GGUF").map(std::path::PathBuf::from)
    else {
        eprintln!("skipping: set RMI_GLINER2_5_BASE_V1_GGUF to enable this test");
        return;
    };
    if !path.exists() {
        eprintln!("skipping: {} does not exist", path.display());
        return;
    }
    let source = GGUFLoader::from_file(&path).expect("open boundary GGUF");
    let leaked: &'static dyn TensorSource = Box::leak(Box::new(source));
    let model = BoundaryModel::from_source(leaked).expect("load boundary model");
    assert_eq!(model.settings.boundary_attention_window, 128);
    assert!(
        !model.boundary.attention_blocks.is_empty(),
        "base-v1 has 2 boundary attention blocks"
    );
    for (index, block) in model.boundary.attention_blocks.iter().enumerate() {
        assert_eq!(block.window, 128, "attention block {index} window");
    }
}
