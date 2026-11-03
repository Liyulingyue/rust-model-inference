use rust_model_inference::models::arktts::{codec::Codec, Frame, Model, Sampling};
use rust_model_inference::{open_model_source, ComponentRole};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "requires the pinned real checkpoint and tools/oracle/arktts/official.py fixtures"]
fn real_checkpoint_matches_official() {
    let root = PathBuf::from(
        std::env::var("RMI_ARKTTS_MODEL_DIR").unwrap_or("models/Audio8-TTS-Realistic-Air".into()),
    );
    let oracle = PathBuf::from(
        std::env::var("RMI_ARKTTS_ORACLE_DIR")
            .unwrap_or(root.join("oracle").to_string_lossy().into_owned()),
    );
    for extra in [
        vec![],
        vec![
            "--jev",
            "--jev-context",
            "hello",
            "--jev-question",
            "good?",
            "--jev-option",
            "yes",
        ],
    ] {
        let result = std::process::Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
            .args([
                "--model",
                root.join("Audio8-TTS-Realistic-Air-BF16.gguf")
                    .to_str()
                    .unwrap(),
                "--prompt",
                "hello",
            ])
            .args(extra)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("requires --tts"));
    }
    let source = open_model_source(
        &root.join("Audio8-TTS-Realistic-Air-BF16.gguf"),
        ComponentRole::Llm,
    )
    .unwrap();
    let audio = open_model_source(
        &root.join("Audio8-TTS-Realistic-Air-codec-F32.gguf"),
        ComponentRole::Mmproj,
    )
    .unwrap();
    let model = Model::from_source(source.as_ref(), 4).unwrap();
    let codec = Codec::from_source(audio.as_ref()).unwrap();
    let data: Value =
        serde_json::from_slice(&std::fs::read(oracle.join("cases.json")).unwrap()).unwrap();
    let max_frames = data["max_frames"].as_u64().unwrap_or(4) as usize;
    for case in data["cases"].as_array().unwrap() {
        let reference_codes: Option<Vec<Frame>> =
            serde_json::from_value(case["reference_codes"].clone()).unwrap();
        let reference = case["reference_text"]
            .as_str()
            .zip(reference_codes.as_deref());
        let text = case["text"].as_str().unwrap();
        let prompt: Vec<[u32; 11]> = serde_json::from_value(case["prompt"].clone()).unwrap();
        assert_eq!(
            model.prompt(text, reference).unwrap(),
            prompt,
            "prompt {}",
            case["name"]
        );
        let expected: Vec<Frame> = serde_json::from_value(case["frames"].clone()).unwrap();
        let frames = model
            .generate(
                text,
                reference,
                max_frames,
                Sampling {
                    temperature: 0.0,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(frames, expected, "frames {}", case["name"]);
        println!(
            "{}: {} prompt columns / {} greedy frames aligned",
            case["name"],
            prompt.len(),
            frames.len()
        );
    }
    let fixed: Vec<Frame> = serde_json::from_value(data["fixed_codes"].clone()).unwrap();
    let actual = codec.decode(&fixed).unwrap();
    let expected = read_f32(&oracle.join("waveform.f32"));
    assert_eq!(actual.len(), expected.len());
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rmse = (actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / actual.len() as f64)
        .sqrt();
    println!(
        "waveform: samples={} max_abs={max_abs:e} RMSE={rmse:e}",
        actual.len()
    );
    assert!(
        max_abs < 2e-4 && rmse < 2e-5,
        "waveform max_abs={max_abs} RMSE={rmse}"
    );
    let expected: Vec<Frame> = serde_json::from_value(data["reference_codes"].clone()).unwrap();
    let actual = codec
        .encode(&read_f32(&oracle.join("reference.f32")))
        .unwrap();
    assert_eq!(actual, expected, "reference encoder codes");
    assert!(codec.encode(&[]).is_err());
    assert!(codec.encode(&[f32::NAN]).is_err());
    assert!(model.generate("", None, 4, Sampling::default()).is_err());
    assert!(model
        .generate("hello", None, 2048, Sampling::default())
        .is_err());
    assert!(model
        .prompt("hello", Some(("reference", &[[4096; 10]])))
        .is_err());
}
