use rust_model_inference::app::{parse_cli_options, validate_cli_options};

fn options(args: &[&str]) -> rust_model_inference::app::CliOptions {
    parse_cli_options(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap()
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
    assert!(validate_cli_options(&options(&args)).is_ok());
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
        assert!(validate_cli_options(&options(&mixed)).is_err(), "{extra}");
    }
    assert!(validate_cli_options(&options(&["rmi", "--enhance", "--model", "reuse"])).is_err());
    for extra in [
        vec!["--instruction", "test"],
        vec!["--ref-text", "test"],
        vec!["--srt"],
        vec!["--chunk", "3"],
        vec!["--jev-multi"],
        vec!["--jev-context", "test"],
        vec!["--tts-model", "other"],
    ] {
        let mut mixed = args.to_vec();
        mixed.extend(extra.iter().copied());
        assert!(validate_cli_options(&options(&mixed)).is_err(), "{extra:?}");
    }
}
