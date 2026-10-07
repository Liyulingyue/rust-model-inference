use std::{fs, process::Command};

#[test]
fn mage_is_selected_by_architecture_before_z_image_or_text_loading() {
    let dir = std::env::temp_dir().join(format!("mage-cli-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let model = dir.join("mage.gguf");
    let out = dir.join("out.png");
    let write_model = |variant: &str| {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(2u64.to_le_bytes());
        for (key, value) in [
            ("general.architecture", "mage_flow"),
            ("mage_flow.variant", variant),
        ] {
            bytes.extend((key.len() as u64).to_le_bytes());
            bytes.extend(key.as_bytes());
            bytes.extend(8u32.to_le_bytes());
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.as_bytes());
        }
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
        fs::write(&model, bytes).unwrap();
    };
    write_model("edit-turbo");
    let invoke = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
            .arg("--model")
            .arg(&model)
            .arg("--out")
            .arg(&out)
            .args(extra)
            .output()
            .unwrap()
    };
    for (extra, expected) in [
        (vec![], "--text-encoder"),
        (
            vec![
                "--text-encoder",
                "missing.gguf",
                "--vae",
                "missing-vae.gguf",
                "--prompt",
                "blue",
            ],
            "reference",
        ),
        (
            vec![
                "--text-encoder",
                "missing.gguf",
                "--vae",
                "missing-vae.gguf",
                "--prompt",
                "blue",
                "--image",
                "input.png",
                "--mmproj",
                "missing-vision.gguf",
                "--cfg",
                "1",
                "--width",
                "16",
                "--height",
                "16",
            ],
            "Missing Mage-Flow tensor: img_in.weight",
        ),
        (vec!["--embedding"], "--embedding"),
    ] {
        let output = invoke(&extra);
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains(expected), "{extra:?}: {error}");
        assert!(!out.exists());
    }
    for variant in ["base", "flow", "turbo", "edit-base", "edit", "edit-turbo"] {
        write_model(variant);
        let mut args = vec![
            "--text-encoder",
            "missing.gguf",
            "--vae",
            "missing-vae.gguf",
            "--prompt",
            "blue",
        ];
        if variant.starts_with("edit") {
            args.extend([
                "--reference",
                "input.png",
                "--mmproj",
                "missing-vision.gguf",
            ]);
        }
        let result = invoke(&args);
        assert!(!result.status.success());
        assert!(
            String::from_utf8_lossy(&result.stderr)
                .contains("Missing Mage-Flow tensor: img_in.weight"),
            "variant {variant}"
        );
        assert!(!out.exists());
    }
    for extra in [
        vec!["--cfg", "NaN"],
        vec!["--cfg", "0"],
        vec!["--steps", "0"],
        vec!["--width", "17"],
        vec!["--height", "0"],
        vec!["--seed", "-1"],
        vec!["--threads", "257"],
        vec![
            "--reference",
            "second.png",
            "--reference",
            "third.png",
            "--reference",
            "fourth.png",
        ],
    ] {
        let mut args = vec![
            "--text-encoder",
            "missing.gguf",
            "--vae",
            "missing-vae.gguf",
            "--prompt",
            "blue",
            "--image",
            "input.png",
            "--mmproj",
            "missing-vision.gguf",
        ];
        args.extend(extra);
        let result = invoke(&args);
        assert!(!result.status.success());
        assert!(!String::from_utf8_lossy(&result.stderr).contains("Missing Mage-Flow tensor"));
        assert!(!out.exists());
    }
    fs::write(&out, b"original").unwrap();
    let result = invoke(&[
        "--text-encoder",
        "missing.gguf",
        "--vae",
        "missing-vae.gguf",
        "--prompt",
        "blue",
    ]);
    assert!(String::from_utf8_lossy(&result.stderr).contains("Output already exists"));
    assert_eq!(fs::read(&out).unwrap(), b"original");
    fs::remove_dir_all(dir).unwrap();
}
