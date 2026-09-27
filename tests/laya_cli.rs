use rust_model_inference::app::{parse_cli_options, validate_cli_options};

#[test]
fn laya_request_is_an_explicit_decision_mode() {
    let args = [
        "rmi",
        "--model",
        "model.gguf",
        "--laya-request",
        "request.json",
    ]
    .map(String::from);
    let options = parse_cli_options(&args).expect("Laya request flag must be recognized");
    validate_cli_options(&options).unwrap();
    let conflicting = args.into_iter().chain(["--jev".into()]).collect::<Vec<_>>();
    assert!(validate_cli_options(&parse_cli_options(&conflicting).unwrap()).is_err());
}
