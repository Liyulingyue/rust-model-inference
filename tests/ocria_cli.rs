//! The OCRia entry's own flags, and one real page when a packed model is given.
//!
//! `--ocr-options` is checked in two places on purpose: the parser rejects bad
//! JSON and unknown fields before a model is loaded, and the model rejects a
//! value the pipeline cannot run with once it has its own defaults as a base.
//! The first two tests need no model; the last one runs only when `OCRIA_GGUF`
//! names a packed file.

use std::process::Command;

fn run(args: &[&str]) -> (bool, String) {
    let result = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .args(args)
        .output()
        .unwrap();
    (
        result.status.success(),
        String::from_utf8_lossy(&result.stderr).into_owned(),
    )
}

#[test]
fn malformed_ocr_options_are_rejected_before_the_model_loads() {
    for (value, needle) in [
        ("{", "Invalid --ocr-options"),
        ("{\"threshold\":}", "Invalid --ocr-options"),
        ("{\"nonsense\":1}", "unknown field"),
    ] {
        let (ok, error) = run(&["--model", "/nonexistent.gguf", "--ocr-options", value]);
        assert!(!ok, "{value} should be rejected");
        assert!(error.contains(needle), "{value}: {error}");
    }
}

#[test]
fn ocr_options_require_an_ocria_model() {
    let (ok, error) = run(&[
        "--model",
        "/nonexistent.gguf",
        "--ocr-options",
        "{\"long_side\":800}",
    ]);
    assert!(!ok);
    assert!(
        error.contains("--ocr-options requires an OCRia model"),
        "{error}"
    );
}

/// The whole path on a real page: `OCRIA_GGUF` plus `OCRIA_PAGE`.
#[test]
fn a_page_is_read_with_the_options_it_was_given() {
    let (Ok(model), Ok(page)) = (std::env::var("OCRIA_GGUF"), std::env::var("OCRIA_PAGE")) else {
        eprintln!("skipped: set OCRIA_GGUF and OCRIA_PAGE");
        return;
    };
    let out = std::env::temp_dir().join("ocria_cli_options.json");
    let _ = std::fs::remove_file(&out);
    let (ok, error) = run(&[
        "--model",
        &model,
        "--image",
        &page,
        "--out",
        out.to_str().unwrap(),
        "--ocr-options",
        "{\"long_side\":800,\"faint_threshold\":0.0}",
        "--threads",
        "4",
        "--compute",
        "cpu",
    ]);
    assert!(ok, "{error}");
    let text = std::fs::read_to_string(&out).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(parsed["lines"].as_array().unwrap().len() > 0);
    // The faint pass is off in these options, so no line may be flagged.
    for line in parsed["lines"].as_array().unwrap() {
        assert_eq!(line["faint"], serde_json::json!(false));
    }
    let _ = std::fs::remove_file(&out);
}
