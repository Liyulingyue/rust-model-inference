use rust_model_inference::app::{parse_cli_options, validate_cli_options};
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn decision_request_is_json_input_and_accepts_compute_policies() {
    for compute in ["cpu", "auto", "vulkan"] {
        let args = [
            "rmi",
            "--model",
            "thx02.gguf",
            "--decision-request",
            "-",
            "--decision-thinker",
            "thinker.gguf",
            "--compute",
            compute,
        ]
        .map(String::from);
        let options = parse_cli_options(&args).unwrap();
        assert!(
            options.audio.is_none(),
            "decision JSON must not become audio input"
        );
        validate_cli_options(&options).unwrap();
    }
}

#[test]
fn decision_conflicts_fail_before_loading_any_assets() {
    for flags in [
        vec!["--serve"],
        vec!["--jev"],
        vec!["--rerank"],
        vec!["--embedding"],
        vec!["--tts"],
        vec!["--astronex"],
        vec!["--dreamx"],
        vec!["--yue2"],
        vec!["--prompt", "hello"],
        vec!["--image", "missing.png"],
        vec!["--mmproj", "missing.gguf"],
        vec!["--kind", "turbo"],
        vec!["--timestral-request", "missing.json"],
        vec!["--system-one-request", "missing.json"],
        vec!["--max-tokens", "1"],
        vec!["--temp", "0"],
        vec!["--kv-cache", "f32"],
        vec!["--jinja"],
        vec!["--unknown"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
            .args([
                "--model",
                "missing.gguf",
                "--decision-request",
                "missing.json",
            ])
            .args(&flags)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{flags:?}");
        assert!(stderr.contains("--decision-request"), "{flags:?}: {stderr}");
        assert!(!stderr.contains("Failed to load"), "{flags:?}: {stderr}");
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn decision_reads_and_validates_file_and_stdin_before_loading() {
    let path = std::env::temp_dir().join(format!("rmi-decision-{}.json", std::process::id()));
    let bytes = br#"{"state":"hello","questions":[]}"#;
    std::fs::write(&path, bytes).unwrap();
    for request in [path.to_str().unwrap(), "-"] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
            .args(["--model", "missing.gguf", "--decision-request", request])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        if request == "-" {
            child.stdin.take().unwrap().write_all(bytes).unwrap();
        }
        let output = child.wait_with_output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success());
        assert!(stderr.contains("questions"), "{stderr}");
        assert!(!stderr.contains("missing.gguf"), "{stderr}");
        assert!(output.stdout.is_empty());
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn decision_rejects_streaming_before_loading_assets() {
    for stream in ["true", "null", "0", "\"false\""] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
            .args(["--model", "missing.gguf", "--decision-request", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        write!(
            child.stdin.take().unwrap(),
            "{{\"state\":\"hello\",\"questions\":{{\"q\":{{\"type\":\"noul\"}}}},\"stream\":{stream}}}"
        )
        .unwrap();
        let output = child.wait_with_output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success());
        assert!(
            stderr.contains("do not support streaming"),
            "{stream}: {stderr}"
        );
        assert!(!stderr.contains("missing.gguf"), "{stream}: {stderr}");
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn decision_flags_require_values_and_task_context() {
    for flags in [
        vec!["rmi", "--decision-request"],
        vec!["rmi", "--decision-request", "request.json"],
        vec!["rmi", "--model", "x.gguf", "--decision-thinker"],
        vec![
            "rmi",
            "--model",
            "x.gguf",
            "--decision-thinker",
            "thinker.gguf",
        ],
        vec![
            "rmi",
            "--model",
            "x.gguf",
            "--decision-request",
            "-",
            "--threads",
            "bad",
        ],
    ] {
        assert!(
            parse_cli_options(&flags.iter().map(|x| x.to_string()).collect::<Vec<_>>())
                .and_then(|options| validate_cli_options(&options))
                .is_err(),
            "{flags:?}"
        );
    }
}

#[test]
fn dedicated_server_rejects_a_cli_decision_request() {
    let output = Command::new(env!("CARGO_BIN_EXE_rust-model-server"))
        .args([
            "--model",
            "missing.gguf",
            "--decision-request",
            "missing.json",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--decision-request is a CLI"));
}

fn arch_fixture(name: &str, arch: &str) -> std::path::PathBuf {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(1u64.to_le_bytes());
    let key = "general.architecture";
    bytes.extend((key.len() as u64).to_le_bytes());
    bytes.extend(key.bytes());
    bytes.extend(8u32.to_le_bytes());
    bytes.extend((arch.len() as u64).to_le_bytes());
    bytes.extend(arch.bytes());
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    let path =
        std::env::temp_dir().join(format!("rmi-decision-{name}-{}.gguf", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn decision_server_selects_explicit_arch_and_rejects_generation_flags() {
    let model = arch_fixture("server", "thx02");
    let output = Command::new(env!("CARGO_BIN_EXE_rust-model-server"))
        .args(["--model", model.to_str().unwrap(), "--prompt", "ignored"])
        .output()
        .unwrap();
    std::fs::remove_file(model).unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("THX-02 decisions cannot be combined with --prompt"),
        "{stderr}"
    );
}

#[test]
fn decision_jev_selects_explicit_arch_before_generation_or_projector_loading() {
    let model = arch_fixture("jev", "thx02");
    for flag in ["--max-tokens", "--temp", "--mmproj"] {
        let value = if flag == "--mmproj" {
            "missing.gguf"
        } else {
            "1"
        };
        let output = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
            .args([
                "--model",
                model.to_str().unwrap(),
                "--jev",
                "--jev-context",
                "state",
                "--jev-question",
                "pick",
                "--jev-option",
                "a",
                "--jev-option",
                "b",
                flag,
                value,
            ])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!("THX-02 decisions cannot be combined with {flag}")),
            "{stderr}"
        );
    }
    std::fs::remove_file(model).unwrap();
}

#[test]
fn decision_cli_image_errors_precede_model_and_component_loading() {
    let directory =
        std::env::temp_dir().join(format!("rmi-decision-images-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let invalid = directory.join("invalid.png");
    std::fs::write(&invalid, b"not an image").unwrap();
    let request = directory.join("request.json");
    let model = arch_fixture("image-validation", "thx02");
    for source in [
        directory.join("missing.png").to_string_lossy().into_owned(),
        invalid.to_string_lossy().into_owned(),
        "https://example.com/image.png".into(),
    ] {
        std::fs::write(&request, serde_json::to_vec(&serde_json::json!({
            "state":"x","questions":{"q":{"type":"choice","criteria":{"a":"A","b":"B"}}},"images":[source]
        })).unwrap()).unwrap();
        for arguments in [
            vec![
                "--model",
                "missing.gguf",
                "--decision-request",
                request.to_str().unwrap(),
            ],
            vec![
                "--model",
                model.to_str().unwrap(),
                "--jev",
                "--jev-context",
                "x",
                "--jev-question",
                "pick",
                "--jev-option",
                "a",
                "--jev-option",
                "b",
                "--image",
                &source,
            ],
        ] {
            let output = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
                .args(arguments)
                .output()
                .unwrap();
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!output.status.success());
            assert!(stderr.contains("image"), "{stderr}");
            assert!(!stderr.contains("decision model missing.gguf"), "{stderr}");
            assert!(!stderr.contains("Missing THX-02"), "{stderr}");
            assert!(output.stdout.is_empty());
        }
    }
    std::fs::remove_file(model).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}
