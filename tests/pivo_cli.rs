use std::process::Command;

fn reject(args: &[&str], message: &str) {
    let result = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .args(args)
        .output()
        .unwrap();
    assert!(!result.status.success());
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(error.contains(message), "{error}");
}

#[test]
fn speech_controls_require_tts() {
    reject(&["--voice", "Pi"], "--voice/--speed require --tts");
    reject(&["--speed", "1.5"], "--voice/--speed require --tts");
}

#[test]
fn invalid_speech_speed_is_rejected() {
    for speed in ["NaN", "0", "2.1", "fast"] {
        reject(&["--tts", "--speed", speed], "--speed");
    }
}
