use rust_model_inference::compute::ComputePolicy;
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
    let main_path = std::env::var_os("RMI_ARKTTS_MAIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("Audio8-TTS-Realistic-Air-BF16.gguf"));
    let codec_path = std::env::var_os("RMI_ARKTTS_CODEC")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("Audio8-TTS-Realistic-Air-codec-F32.gguf"));
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
            .args(["--model", main_path.to_str().unwrap(), "--prompt", "hello"])
            .args(extra)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("requires --tts"));
    }
    let compute = match std::env::var("RMI_ARKTTS_COMPUTE").as_deref() {
        Ok("vulkan") => ComputePolicy::Vulkan,
        Ok("auto") => ComputePolicy::Auto,
        Ok("cpu") | Err(_) => ComputePolicy::Cpu,
        other => panic!("invalid compute {other:?}"),
    };
    let source = open_model_source(&main_path, ComponentRole::Llm).unwrap();
    let audio = open_model_source(&codec_path, ComponentRole::Mmproj).unwrap();
    let model = Model::from_source_with_compute(source.as_ref(), 4, compute).unwrap();
    let codec = Codec::from_source_with_compute(audio.as_ref(), 4, compute).unwrap();
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
        #[cfg(feature = "parity-trace")]
        let replay = std::env::var_os("RMI_ARKTTS_REPLAY").is_some();
        #[cfg(not(feature = "parity-trace"))]
        let replay = false;
        let began = std::time::Instant::now();
        let frames = if replay {
            #[cfg(feature = "parity-trace")]
            {
                model.replay(text, reference, &expected).unwrap()
            }
            #[cfg(not(feature = "parity-trace"))]
            {
                unreachable!()
            }
        } else {
            model
                .generate(
                    text,
                    reference,
                    max_frames,
                    Sampling {
                        temperature: 0.0,
                        ..Default::default()
                    },
                )
                .unwrap()
        };
        println!(
            "performance phase=generate case={} replay={replay} frames={} seconds={:.9}",
            case["name"].as_str().unwrap(),
            frames.len(),
            began.elapsed().as_secs_f64()
        );
        if !replay && !data["exact_generated_codes"].as_bool().unwrap_or(true) {
            let expected: Vec<Frame> =
                serde_json::from_value(case["greedy_frames"].clone()).unwrap();
            assert!(!frames.is_empty() && frames.len() <= max_frames);
            assert!(frames.iter().flatten().all(|&id| id < 4096));
            println!(
                "greedy_comparison {}",
                serde_json::json!({"case": case["name"], "actual": frames, "expected": expected})
            );
        } else {
            assert_eq!(frames, expected, "frames {}", case["name"]);
        }
        println!(
            "{}: {} prompt columns / {} generated frames checked",
            case["name"],
            prompt.len(),
            frames.len()
        );
    }
    let fixed: Vec<Frame> = serde_json::from_value(data["fixed_codes"].clone()).unwrap();
    let began = std::time::Instant::now();
    let actual = codec.decode(&fixed).unwrap();
    println!(
        "performance phase=codec_decode frames={} seconds={:.9}",
        fixed.len(),
        began.elapsed().as_secs_f64()
    );
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
    let waveform_max = data["waveform_max_abs"].as_f64().unwrap_or(2e-4);
    let waveform_rmse = data["waveform_rmse"].as_f64().unwrap_or(2e-5);
    assert!(
        f64::from(max_abs) < waveform_max && rmse < waveform_rmse,
        "waveform max_abs={max_abs} RMSE={rmse}"
    );
    let expected: Vec<Frame> = serde_json::from_value(data["reference_codes"].clone()).unwrap();
    let reference = read_f32(&oracle.join("reference.f32"));
    let began = std::time::Instant::now();
    let actual = codec.encode(&reference).unwrap();
    println!(
        "performance phase=codec_encode samples={} seconds={:.9}",
        reference.len(),
        began.elapsed().as_secs_f64()
    );
    if data["exact_reference_codes"].as_bool().unwrap_or(true) {
        assert_eq!(actual, expected, "reference encoder codes");
    } else {
        assert_eq!(actual.len(), expected.len());
        println!(
            "reference encoder IDs: {}/{} equal",
            actual
                .iter()
                .flatten()
                .zip(expected.iter().flatten())
                .filter(|(a, b)| a == b)
                .count(),
            expected.len() * 10
        );
    }
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
