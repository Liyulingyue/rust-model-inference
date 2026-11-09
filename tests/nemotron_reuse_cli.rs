use rust_model_inference::app::{parse_cli_options, validate_cli_options};

fn validate(args: &[&str]) -> Result<(), String> {
    let options = parse_cli_options(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())?;
    validate_cli_options(&options)
}

#[test]
fn enhancement_is_an_explicit_audio_task() {
    let args = [
        "rmi",
        "--enhance",
        "--model",
        "reuse",
        "--audio",
        "in.wav",
        "--out",
        "out.wav",
    ];
    assert!(validate(&args).is_ok());
    for extra in [
        "--jev",
        "--embedding",
        "--tts",
        "--dreamx",
        "--yue2",
        "--rerank",
    ] {
        let mut mixed = args.to_vec();
        mixed.push(extra);
        assert!(validate(&mixed).is_err(), "{extra}");
    }
    assert!(validate(&["rmi", "--enhance", "--model", "reuse"]).is_err());
    for extra in [
        vec!["--instruction", "test"],
        vec!["--ref-text", "test"],
        vec!["--srt"],
        vec!["--chunk", "3"],
        vec!["--jev-multi"],
        vec!["--jev-context", "test"],
        vec!["--tts-model", "other"],
        vec!["--system-one-request", "request.json"],
        vec!["--adapter", "adapter"],
        vec!["--kv-cache", "f16"],
        vec!["--jev-output", "text"],
        vec!["--repetition-penalty", "1"],
        vec!["--unknown-future-flag"],
        vec!["--threads", "invalid"],
    ] {
        let mut mixed = args.to_vec();
        mixed.extend(extra.iter().copied());
        assert!(validate(&mixed).is_err(), "{extra:?}");
    }
}

#[test]
fn enhancement_allowlist_handles_order_values_and_server_inputs() {
    let base = [
        "rmi", "--model", "reuse", "--audio", "in.wav", "--out", "out.wav",
    ];
    let mut args = base.to_vec();
    args.extend([
        "--threads",
        "2",
        "--compute",
        "auto",
        "--overwrite",
        "--enhance",
    ]);
    assert!(validate(&args).is_ok());
    for extra in [
        vec!["--system-one-request", "request.json"],
        vec!["--serve"],
    ] {
        let mut mixed = args.clone();
        mixed.extend(extra);
        assert!(validate(&mixed).is_err());
    }
    assert!(validate(&["rmi", "--serve", "--model", "reuse", "--enhance"]).is_ok());
    assert!(validate(&[
        "rmi",
        "--enhance",
        "--model",
        "reuse",
        "--compute",
        "vulkan"
    ])
    .is_err());
    let parsed = parse_cli_options(
        &["rmi", "--model", "text.gguf", "--prompt", "--enhance"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert!(!parsed.enhance);
    assert_eq!(parsed.prompt.as_deref(), Some("--enhance"));
}
