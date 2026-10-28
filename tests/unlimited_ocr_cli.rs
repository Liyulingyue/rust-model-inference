use std::process::Command;

#[test]
fn unlimited_ocr_directory_reaches_its_own_image_validation() {
    let root = std::env::temp_dir().join(format!("rmi-uocr-cli-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("config.json"),
        r#"{"model_type":"unlimited-ocr"}"#,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .args(["--model", root.to_str().unwrap(), "--max-tokens", "1"])
        .output()
        .unwrap();
    std::fs::remove_dir_all(root).unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("Unlimited-OCR") && stderr.contains("--image"),
        "{stderr}"
    );
}

#[test]
fn unlimited_ocr_rejects_other_task_modes_before_loading_weights() {
    let root = std::env::temp_dir().join(format!("rmi-uocr-modes-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("config.json"),
        r#"{"model_type":"unlimited-ocr"}"#,
    )
    .unwrap();
    for flags in [
        &["--dreamx"][..],
        &["--embedding"][..],
        &["--temp", "0.5"][..],
        &["--top-p", "0.9"][..],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
            .args(["--model", root.to_str().unwrap(), "--image", "missing.png"])
            .args(flags)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success());
        assert!(
            stderr.contains("Unlimited-OCR")
                && (stderr.contains("CPU OCR") || stderr.contains("greedy")),
            "{stderr}"
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}
