use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn native_d3_validates_requests_without_requiring_a_vela_adapter() {
    let root = std::env::temp_dir().join(format!("d3-turbo-cli-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("decision_config.json"),
        r#"{"format_id":"d3-code-readout-v1"}"#,
    )
    .unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .args([
            "--model",
            root.to_str().unwrap(),
            "--system-one-request",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(br#"{"state":"hello","questions":{}}"#)
        .unwrap();
    let output = child.wait_with_output().unwrap();
    std::fs::remove_dir_all(root).unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("questions must be a nonempty"), "{error}");
}
