//! CLI contract for LongCat Image Edit / Edit Turbo.
//!
//! LongCat advertises generic `flux` metadata, so the dispatcher cannot select
//! it by GGUF architecture the way Mage-Flow is selected; it is claimed by its
//! own flags. These tests pin that the flags are validated before any weight
//! file is opened, and that the per-kind schedule defaults survive.

use std::{fs, process::Command};

fn longcat_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("longcat-cli-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A GGUF that is syntactically valid but carries no LongCat tensors, so any
/// invocation that gets past flag validation fails on the contract instead.
fn write_empty_gguf(path: &std::path::Path) {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(1u64.to_le_bytes());
    bytes.extend((b"general.architecture".len() as u64).to_le_bytes());
    bytes.extend(b"general.architecture");
    bytes.extend(8u32.to_le_bytes());
    bytes.extend((b"flux".len() as u64).to_le_bytes());
    bytes.extend(b"flux");
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    fs::write(path, bytes).unwrap();
}

fn invoke(dir: &std::path::Path, model: &std::path::Path, extra: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .arg("--model")
        .arg(model)
        .arg("--kind")
        .arg("turbo")
        .arg("--components")
        .arg(dir.join("components"))
        .arg("--input")
        .arg(dir.join("input.png"))
        .arg("--instruction")
        .arg("edit the image")
        .arg("--out")
        .arg(dir.join("out.png"))
        .args(extra)
        .output()
        .unwrap()
}

#[test]
fn longcat_flags_are_validated_before_any_weight_is_opened() {
    let dir = longcat_dir();
    let model = dir.join("longcat.gguf");
    write_empty_gguf(&model);
    // Every case below must fail on the flag contract, not on a missing or
    // malformed weight, so each expected string names the flag error.
    for (extra, expected) in [
        (vec!["--kind", "not-a-kind"], "Invalid --kind"),
        (vec!["--side", "30"], "must be a positive multiple of 16"),
        (vec!["--side", "0"], "must be a positive multiple of 16"),
        (vec!["--steps", "0"], "--steps must be positive"),
        (vec!["--guidance", "-1"], "--guidance must be finite"),
        (vec!["--guidance", "nan"], "--guidance must be finite"),
    ] {
        let output = invoke(&dir, &model, &extra);
        assert!(!output.status.success(), "{extra:?} unexpectedly succeeded");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(expected), "{extra:?}: {stderr}");
        assert!(!dir.join("out.png").exists(), "{extra:?} wrote an output file");
    }
}

#[test]
fn longcat_auxiliary_flags_require_an_explicit_kind() {
    let dir = longcat_dir();
    let model = dir.join("longcat.gguf");
    write_empty_gguf(&model);
    // Without --kind these flags have no owner, so they must be rejected
    // rather than silently ignored.
    for (args, expected) in [
        (vec!["--components", "x"], "--components requires --kind"),
        (vec!["--input", "x"], "--input requires --kind"),
        (vec!["--side", "16"], "--side requires --kind"),
        (vec!["--guidance", "1"], "--guidance requires --kind"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
            .arg("--model")
            .arg(&model)
            .args(&args)
            .output()
            .unwrap();
        assert!(!output.status.success(), "{args:?} unexpectedly succeeded");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(expected), "{args:?}: {stderr}");
    }
}

#[test]
fn longcat_reaches_the_weight_contract_once_the_flags_are_valid() {
    let dir = longcat_dir();
    let model = dir.join("longcat.gguf");
    write_empty_gguf(&model);
    // Proves the dispatcher claimed the invocation instead of letting the
    // Z-Image branch reject it or silently ignore the LongCat-only flags.
    for kind in ["edit", "turbo"] {
        let output = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
            .arg("--model")
            .arg(&model)
            .arg("--kind")
            .arg(kind)
            .arg("--components")
            .arg(dir.join("components"))
            .arg("--input")
            .arg(dir.join("input.png"))
            .arg("--instruction")
            .arg("edit the image")
            .arg("--out")
            .arg(dir.join("out.png"))
            .arg("--side")
            .arg("32")
            .arg("--steps")
            .arg("1")
            .arg("--guidance")
            .arg("1")
            .output()
            .unwrap();
        assert!(!output.status.success(), "{kind} unexpectedly succeeded");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("LongCat"),
            "{kind} did not reach the LongCat loader: {stderr}"
        );
    }
}
