use std::path::Path;
use std::process::Command;

/// Requires the Qwen3.5-0.8B GGUF + mmproj (roughly 1 GB on disk), so it is
/// `#[ignore]`d: run it explicitly with
/// `cargo test --test qwen35_visual_jev -- --ignored --nocapture`.
///
/// Drives the CLI end to end and asserts on the scorer's own invariants:
/// a valid single choice, a normalized distribution over the labels, and a
/// JSON shape the caller can parse. The semantic answer is deliberately not
/// pinned — it depends on the checkpoint, and pinning it just makes the test
/// fail on every model swap.
#[ignore = "requires models/qwen3.5-0.8B GGUF and mmproj (~1 GB)"]
fn image_jev_scores_the_binary_choice() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = root.join("models/qwen3.5-0.8B/Qwen3.5-0.8B-UD-Q8_K_XL.gguf");
    let mmproj = root.join("models/qwen3.5-0.8B/mmproj-F16.gguf");
    for path in [&model, &mmproj] {
        assert!(
            path.exists(),
            "{} is required by this test",
            path.display()
        );
    }
    // Several images are tried in order until one exists, so the test does not
    // silently pass on a checkout that only has a subset of the assets.
    let image = [
        "models/YuE2-Vae/assets/logo.png",
        "models/apple.png",
    ]
    .iter()
    .map(|path| root.join(path))
    .find(|path| path.exists())
    .expect("no image fixture available (tried models/YuE2-Vae/assets/logo.png, models/apple.png)");

    let mut command = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"));
    command
        .args(["--model", model.to_str().unwrap()])
        .args(["--mmproj", mmproj.to_str().unwrap()])
        .args(["--image", image.to_str().unwrap()])
        .args(["--jev", "--jev-context", "天空乌云密布，能听到远处雷声"])
        .args(["--jev-question", "现在在下雨吗？"])
        .args(["--jev-option", "是的", "--jev-option", "没有"])
        .args(["--jev-positive", "A", "--threads", "4", "--max-tokens", "1"]);
    let output = command.output().expect("run visual JEV");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    assert!(
        stdout.contains("--- JEV decision ---"),
        "missing decision block\n{stdout}\n{stderr}"
    );
    assert!(
        stdout.contains("probability (A):"),
        "missing binary probability line\n{stdout}\n{stderr}"
    );
    assert!(
        stderr.contains("Vision tokens:"),
        "vision encoder never ran\n{stdout}\n{stderr}"
    );

    let output = command
        .args(["--jev-output", "json"])
        .output()
        .expect("run visual JEV with JSON output");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    let result: serde_json::Value =
        serde_json::from_str(&stdout).expect("JEV stdout must contain only JSON");

    // A binary scorer must pick exactly one of the two labels it was given.
    let choice = result["choice_label"]
        .as_str()
        .unwrap_or_else(|| panic!("no choice_label in {stdout}"));
    assert!(
        choice == "A" || choice == "B",
        "choice_label {choice} is not one of A/B\n{stdout}"
    );

    // Softmax correctness: every label is present, in range, and the
    // distribution is normalized. This catches a scorer that scores only
    // some labels, or that leaks an unnormalized distribution.
    let labels = result["labels"].as_array().expect("labels array");
    let probabilities = result["probabilities"].as_object().expect("probabilities");
    assert_eq!(labels.len(), 2, "expected A and B, got {stdout}");
    let total: f64 = probabilities.values().map(|v| v.as_f64().unwrap()).sum();
    assert!(
        (total - 1.0).abs() < 1e-3,
        "probabilities sum to {total}, expected 1.0\n{stdout}"
    );
    for (label, probability) in probabilities {
        let probability = probability.as_f64().unwrap();
        assert!(
            (0.0..=1.0).contains(&probability),
            "probability for {label} is {probability}\n{stdout}"
        );
    }

    // The scored probability must be the one reported for the chosen label.
    let chosen_probability = probabilities[choice].as_f64().unwrap();
    let reported = result["probability_positive"].as_f64().unwrap();
    assert!(
        (chosen_probability - reported).abs() < 1e-6,
        "reported probability {reported} != chosen label probability {chosen_probability}\n{stdout}"
    );
}

/// Guards against the CLI `--jev --image` gate and the HTTP `/v1/jev/image`
/// dispatcher drifting apart: they must accept exactly the same architectures,
/// and every accepted arch must have a matching HTTP dispatch arm.
#[test]
fn cli_and_http_jev_image_accept_the_same_architectures() {
    let expected: &[&str] = &["qwen3", "qwen35", "qwen3vl", "qwen3vlmoe"];
    let mut accepted: Vec<&str> = expected
        .iter()
        .copied()
        .filter(|arch| {
            rust_model_inference::app::image_supported_arch(arch)
        })
        .collect();
    accepted.sort_unstable();
    assert_eq!(accepted, {
        let mut sorted = expected.to_vec();
        sorted.sort_unstable();
        sorted
    });

    // Every accepted arch must be dispatched by the HTTP handler too.
    let api = include_str!("../src/app/server/api.rs");
    for arch in accepted {
        assert!(
            api.contains(arch),
            "HTTP /v1/jev/image has no dispatch arm for {arch}"
        );
    }
}
