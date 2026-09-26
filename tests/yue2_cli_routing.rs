use std::process::Command;

#[test]
fn yue2_cli_rejects_chat_template_before_loading_models() {
    let output = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .args([
            "--yue2",
            "--model",
            "missing.gguf",
            "--vae",
            "missing-vae.gguf",
            "--prompt",
            "jazz",
            "--lyrics",
            "hello",
            "--out",
            "song.wav",
            "--chat-template",
            "chatml",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("--chat-template"));
}

#[test]
fn yue2_server_mode_rejects_before_loading_models() {
    let output = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .args([
            "--serve",
            "--yue2",
            "--model",
            "missing.gguf",
            "--vae",
            "missing-vae.gguf",
            "--prompt",
            "jazz",
            "--lyrics",
            "hello",
            "--out",
            "song.wav",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("--yue2"));
}
