use std::process::Command;

#[test]
fn unlimited_ocr_auto_and_http_reach_native_config_validation() {
    let root = std::env::temp_dir().join(format!("rmi-uocr-dispatch-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("config.json"),
        r#"{"model_type":"unlimited-ocr"}"#,
    )
    .unwrap();
    for flags in [
        &["--compute", "auto", "--image", "missing.png"][..],
        &["--gpu", "--image", "missing.png"][..],
        &["--serve"][..],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
            .args(["--model", root.to_str().unwrap()])
            .args(flags)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success());
        assert!(
            stderr.contains("Unsupported --compute auto for Unlimited-OCR")
                || stderr.contains("Unsupported --gpu for Unlimited-OCR")
                || stderr.contains("Unlimited-OCR"),
            "{flags:?}: {stderr}"
        );
    }
    let output = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .args(["--model", root.to_str().unwrap()])
        .args([
            "--jev",
            "--jev-context",
            "hello",
            "--jev-question",
            "Q",
            "--jev-option",
            "yes",
            "--jev-option",
            "no",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(stderr.contains("Unlimited-OCR supports OCR mode only; --jev is unsupported"));
    std::fs::remove_dir_all(root).unwrap();
}

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
                && (stderr.contains("OCR mode only")
                    || stderr.contains("greedy")
                    || stderr.contains("--jev")),
            "{stderr}"
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}
