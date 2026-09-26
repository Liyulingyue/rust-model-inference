use std::path::Path;
use std::process::Command;

#[test]
fn image_jev_scores_the_binary_choice() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = root.join("models/qwen3.5-0.8B/Qwen3.5-0.8B-UD-Q8_K_XL.gguf");
    let mmproj = root.join("models/qwen3.5-0.8B/mmproj-F16.gguf");
    let image = root.join("models/YuE2-Vae/assets/logo.png");
    if !model.exists() || !mmproj.exists() || !image.exists() {
        return;
    }

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
        "{stdout}\n{stderr}"
    );
    assert!(stdout.contains("probability (A):"), "{stdout}\n{stderr}");
    assert!(stderr.contains("Vision tokens:"), "{stdout}\n{stderr}");

    let output = command
        .args(["--jev-output", "json"])
        .output()
        .expect("run visual JEV with JSON output");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    let result: serde_json::Value =
        serde_json::from_str(&stdout).expect("JEV stdout must contain only JSON");
    let probability = result["probability"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&probability));
}
