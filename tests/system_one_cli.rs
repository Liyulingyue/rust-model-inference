use rust_model_inference::app::{parse_cli_options, validate_cli_options};
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn system_one_request_uses_shared_cli_options() {
    for request in ["request.json", "-"] {
        let args = [
            "rmi",
            "--model",
            "vela",
            "--base-model",
            "d3",
            "--system-one-request",
            request,
            "--threads",
            "2",
            "--compute",
            "cpu",
        ]
        .map(String::from);
        let options = parse_cli_options(&args).unwrap();
        assert_eq!(options.model.to_str(), Some("vela"));
        assert_eq!(
            options.base_model.as_deref(),
            Some(std::path::Path::new("d3"))
        );
        assert_eq!(
            options.system_one_request.as_deref(),
            Some(std::path::Path::new(request))
        );
        assert_eq!(options.threads, 2);
        assert!(
            options.audio.is_none(),
            "System One JSON is not audio input"
        );
        validate_cli_options(&options).unwrap();
    }
}

#[test]
fn system_one_flag_as_prompt_text_does_not_select_the_mode() {
    let args = [
        "rmi",
        "--model",
        "text.gguf",
        "--prompt",
        "--system-one-request",
    ]
    .map(String::from);
    let options = parse_cli_options(&args).unwrap();
    assert_eq!(options.prompt.as_deref(), Some("--system-one-request"));
    assert!(options.system_one_request.is_none());
    validate_cli_options(&options).unwrap();
}

#[test]
fn system_one_options_require_their_mode_and_values() {
    for flags in [
        vec!["rmi", "--system-one-request"],
        vec!["rmi", "--system-one-request", "request.json"],
        vec!["rmi", "--model", "vela", "--base-model"],
        vec!["rmi", "--model", "vela", "--base-model", "d3"],
        vec!["rmi", "--system-one-request", "--model", "vela"],
        vec!["rmi", "--system-one-request", "request.json", "--model"],
        vec!["rmi", "--system-one-request", "request.json", "--threads"],
        vec![
            "rmi",
            "--system-one-request",
            "request.json",
            "--threads",
            "bad",
        ],
    ] {
        assert!(
            parse_cli_options(&flags.iter().map(|s| s.to_string()).collect::<Vec<_>>())
                .and_then(|options| validate_cli_options(&options))
                .is_err(),
            "{flags:?}"
        );
    }
}

#[test]
fn system_one_conflicts_fail_before_model_loading() {
    for flags in [
        vec!["--serve"],
        vec!["--jev"],
        vec!["--rerank"],
        vec!["--embedding"],
        vec!["--tts"],
        vec!["--yue2"],
        vec!["--dreamx"],
        vec!["--astronex"],
        vec!["--qwen-image-turbo"],
        vec!["--timestral-request", "series.json"],
        vec!["--laya-request", "laya.json"],
        vec!["--wevisone-request", "vision.json"],
        vec!["--prompt", "hello"],
        vec!["--image", "missing.png"],
        vec!["--kind", "turbo"],
        vec!["--output", "result.json"],
        vec!["--max-tokens", "1"],
        vec!["--seed", "1"],
        vec!["--unknown"],
        vec!["--gpu"],
        vec!["--compute", "auto"],
        vec!["--compute", "vulkan"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
            .args([
                "--model",
                "missing.gguf",
                "--system-one-request",
                "missing.json",
            ])
            .args(&flags)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{flags:?}");
        assert!(
            stderr.contains("--system-one-request"),
            "{flags:?}: {stderr}"
        );
        assert!(!stderr.contains("Failed to load"), "{flags:?}: {stderr}");
        assert!(output.stdout.is_empty(), "{flags:?}");
    }
}

#[test]
fn system_one_reads_json_from_files_and_stdin_before_loading_weights() {
    let path = std::env::temp_dir().join(format!("rmi-system-one-{}.json", std::process::id()));
    let bytes = br#"{"questions":[]}"#;
    std::fs::write(&path, bytes).unwrap();
    for request in [path.to_str().unwrap(), "-"] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"));
        command
            .args(["--model", "missing.gguf", "--system-one-request", request])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().unwrap();
        if request == "-" {
            child.stdin.take().unwrap().write_all(bytes).unwrap();
        }
        let output = child.wait_with_output().unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("request must contain a questions object"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty());
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn system_one_is_rejected_by_the_dedicated_server() {
    let output = Command::new(env!("CARGO_BIN_EXE_rust-model-server"))
        .args([
            "--model",
            "missing.gguf",
            "--system-one-request",
            "missing.json",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("--system-one-request is not supported by rust-model-server"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
