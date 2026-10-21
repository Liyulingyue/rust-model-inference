use rust_model_inference::app::{parse_cli_options, validate_cli_options};

#[test]
fn timestral_request_is_a_forecast_mode() {
    let args = [
        "rmi",
        "--model",
        "timestral.gguf",
        "--timestral-request",
        "series.json",
    ]
    .map(String::from);
    let options = parse_cli_options(&args).unwrap();
    assert_eq!(
        options.timestral_request.as_ref().unwrap().to_str(),
        Some("series.json")
    );
    assert!(
        options.audio.is_none(),
        "forecast JSON must not become audio input"
    );
    validate_cli_options(&options).unwrap();
    for flag in [
        "--jev",
        "--tts",
        "--embedding",
        "--serve",
        "--dreamx",
        "--yue2",
    ] {
        let conflicting = args
            .iter()
            .cloned()
            .chain([flag.into()])
            .collect::<Vec<_>>();
        assert!(
            parse_cli_options(&conflicting)
                .and_then(|options| validate_cli_options(&options))
                .is_err(),
            "{flag}"
        );
    }
    for args in [
        vec!["rmi", "--timestral-request"],
        vec!["rmi", "--timestral-request", "series.json"],
    ] {
        assert!(
            parse_cli_options(&args.into_iter().map(String::from).collect::<Vec<_>>())
                .and_then(|options| validate_cli_options(&options))
                .is_err()
        );
    }
}

#[test]
fn forecast_conflicts_fail_before_other_task_dispatch() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .args([
            "--model",
            "missing.gguf",
            "--timestral-request",
            "missing.json",
            "--kind",
            "turbo",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("forecasting mode"));
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .args([
            "--serve",
            "--model",
            "missing.gguf",
            "--timestral-request",
            "missing.json",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--serve is unsupported"));
}

#[test]
fn forecast_accepts_gpu_and_explicit_compute_policies() {
    for flags in [
        vec!["--gpu"],
        vec!["--compute", "auto"],
        vec!["--compute", "vulkan"],
        vec!["--compute", "cpu"],
    ] {
        let mut args = vec![
            "rmi",
            "--model",
            "timestral.gguf",
            "--timestral-request",
            "series.json",
        ];
        args.extend(flags);
        let options =
            parse_cli_options(&args.into_iter().map(String::from).collect::<Vec<_>>()).unwrap();
        validate_cli_options(&options).unwrap();
        options.validate_compute_mode().unwrap();
    }
}
