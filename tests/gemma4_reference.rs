use rust_model_inference::core::scratchpad::KvFormat;
use rust_model_inference::models::gemma4::audio::Gemma4AudioModel;
use rust_model_inference::models::gemma4::multimodal::{run_multimodal, Gemma4Request};
use rust_model_inference::models::gemma4::vision::Gemma4VisionModel;
use rust_model_inference::{GGMLType, GGUFLoader, MetaValue};
use std::path::{Path, PathBuf};
use std::process::Command;

const GEMMA4_MODEL_NAME: &str = "gemma-4-E2B-it-Q8_0.gguf";
const GEMMA4_MMPROJ_NAME: &str = "mmproj-F16.gguf";

fn gemma4_model_path() -> PathBuf {
    std::env::var_os("RMI_GEMMA4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-e2b").join(GEMMA4_MODEL_NAME))
}

fn gemma4_mmproj_path() -> PathBuf {
    std::env::var_os("RMI_GEMMA4_MMPROJ")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-e2b").join(GEMMA4_MMPROJ_NAME))
}

fn require_gemma4_gguf(
    path: &Path,
    expected_name: &str,
    architecture: &str,
    tensor: &str,
    ggml_type: GGMLType,
) {
    assert_eq!(
        path.file_name().and_then(|name| name.to_str()),
        Some(expected_name)
    );
    let loader = GGUFLoader::from_file(path).unwrap();
    assert!(
        matches!(loader.metadata("general.architecture"), Some(MetaValue::String(value)) if value == architecture)
    );
    assert_eq!(
        loader.tensor_info(tensor).map(|info| info.ggml_type),
        Some(ggml_type),
        "invalid {tensor}"
    );
}

#[test]
fn gemma4_preflight_rejects_arbitrary_files() {
    assert_ne!(
        Path::new("/etc/passwd")
            .file_name()
            .and_then(|name| name.to_str()),
        Some(GEMMA4_MODEL_NAME)
    );
    assert_ne!(
        Path::new("/etc/passwd")
            .file_name()
            .and_then(|name| name.to_str()),
        Some(GEMMA4_MMPROJ_NAME)
    );
}

fn ensure_gemma4_oracle() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("LLAMA_GEMMA4_TRACE_BIN").map(PathBuf::from) {
        if path.is_file() {
            return Ok(path);
        }
    }

    let llama_dir = std::env::var_os("LLAMA_CPP_DIR").ok_or_else(|| {
        "LLAMA_CPP_DIR is required when LLAMA_GEMMA4_TRACE_BIN is not a file".to_owned()
    })?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tools/gemma4/build_oracle.sh");
    let output = Command::new("bash")
        .arg(&script)
        .arg(llama_dir)
        .output()
        .map_err(|error| format!("failed to run {}: {error}", script.display()))?;
    if !output.status.success() {
        return Err(format!(
            "Gemma4 oracle build failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<_> = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    if lines.len() != 1 {
        return Err(format!(
            "Gemma4 oracle builder must print exactly one path, got {} non-empty lines",
            lines.len()
        ));
    }
    let path = PathBuf::from(lines[0]);
    if !path.is_file() {
        return Err(format!(
            "Gemma4 oracle builder printed a non-file: {}",
            path.display()
        ));
    }
    Ok(path)
}

#[test]
#[ignore = "requires Gemma4 GGUFs and pinned llama.cpp trace binary"]
fn gemma4_matches_pinned_cpu_oracle() {
    require_gemma4_gguf(
        &gemma4_model_path(),
        GEMMA4_MODEL_NAME,
        "gemma4",
        "token_embd.weight",
        GGMLType::Q8_0,
    );
    require_gemma4_gguf(
        &gemma4_mmproj_path(),
        GEMMA4_MMPROJ_NAME,
        "clip",
        "v.patch_embd.weight",
        GGMLType::F16,
    );
    assert!(ensure_gemma4_oracle().unwrap().is_file());
}

#[test]
#[ignore = "requires the Gemma4 model"]
fn gemma4_text_smoke() {
    let model = gemma4_model_path();
    require_gemma4_gguf(
        &model,
        GEMMA4_MODEL_NAME,
        "gemma4",
        "token_embd.weight",
        GGMLType::Q8_0,
    );
    run_multimodal(Gemma4Request {
        model: &model,
        mmproj: None,
        image: None,
        audio: None,
        prompt: "hello",
        max_tokens: 1,
        threads: 4,
        kv_format: KvFormat::F32,
    })
    .unwrap();
}

#[test]
#[ignore = "requires the Gemma4 mmproj"]
fn gemma4_image_smoke() {
    let mmproj = gemma4_mmproj_path();
    require_gemma4_gguf(
        &mmproj,
        GEMMA4_MMPROJ_NAME,
        "clip",
        "v.patch_embd.weight",
        GGMLType::F16,
    );
    let loader = GGUFLoader::from_file(&mmproj).unwrap();
    let model = Gemma4VisionModel::from_source(&loader, 4).unwrap();
    let fixture =
        std::env::temp_dir().join(format!("rmi-gemma4-image-smoke-{}.png", std::process::id()));
    image::RgbImage::from_fn(64, 48, |x, y| {
        image::Rgb([
            (x.wrapping_mul(3).wrapping_add(y)) as u8,
            (y.wrapping_mul(5).wrapping_add(x)) as u8,
            (x ^ y) as u8,
        ])
    })
    .save(&fixture)
    .unwrap();
    let projected = model.encode_path(&fixture).unwrap();
    let _ = std::fs::remove_file(&fixture);
    assert!(!projected.is_empty());
    assert_eq!(projected.len() % 1536, 0);
    assert!(projected.iter().all(|value| value.is_finite()));
    #[cfg(feature = "parity-trace")]
    if let Some(trace) = std::env::var_os("RMI_PARITY_TRACE") {
        let records: Vec<serde_json::Value> = std::fs::read_to_string(trace)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let shape = |name: &str| {
            records
                .iter()
                .find(|record| record["name"] == name)
                .unwrap_or_else(|| panic!("missing {name} trace checkpoint"))["shape"]
                .clone()
        };
        assert_eq!(
            shape("gemma4.vision.preprocessed"),
            serde_json::json!([384, 288, 3, 1])
        );
        assert_eq!(
            shape("gemma4.vision.projected"),
            serde_json::json!([1536, projected.len() / 1536, 1, 1])
        );
    }
}

#[test]
#[ignore = "requires the Gemma4 mmproj"]
fn gemma4_audio_smoke() {
    let mmproj = gemma4_mmproj_path();
    require_gemma4_gguf(
        &mmproj,
        GEMMA4_MMPROJ_NAME,
        "clip",
        "a.pre_encode.out.weight",
        GGMLType::F16,
    );
    let loader = GGUFLoader::from_file(&mmproj).unwrap();
    let model = Gemma4AudioModel::from_source(&loader, 4).unwrap();
    let fixture =
        std::env::temp_dir().join(format!("rmi-gemma4-audio-smoke-{}.wav", std::process::id()));
    let samples: Vec<i16> = (0..320)
        .map(|index| ((index as f32 * 0.03125).sin() * 16_384.0) as i16)
        .collect();
    let data_len = u32::try_from(samples.len() * 2).unwrap();
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&16_000u32.to_le_bytes());
    wav.extend_from_slice(&32_000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    std::fs::write(&fixture, wav).unwrap();
    let projected = model.encode_wav_path(&fixture).unwrap();
    let _ = std::fs::remove_file(&fixture);
    assert!(!projected.is_empty());
    assert_eq!(projected.len() % 1536, 0);
    assert!(projected.iter().all(|value| value.is_finite()));
    #[cfg(feature = "parity-trace")]
    if let Some(trace) = std::env::var_os("RMI_PARITY_TRACE") {
        let records: Vec<serde_json::Value> = std::fs::read_to_string(trace)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let shape = |name: &str| {
            records
                .iter()
                .find(|record| record["name"] == name)
                .unwrap_or_else(|| panic!("missing {name} trace checkpoint"))["shape"]
                .clone()
        };
        assert_eq!(shape("gemma4.audio.mel"), serde_json::json!([128, 1]));
        assert_eq!(
            shape("gemma4.audio.projected"),
            serde_json::json!([1536, projected.len() / 1536, 1, 1])
        );
    }
}

#[test]
#[ignore = "requires the Gemma4 model and mmproj"]
fn gemma4_image_audio_smoke() {
    let model = gemma4_model_path();
    let mmproj = gemma4_mmproj_path();
    require_gemma4_gguf(
        &model,
        GEMMA4_MODEL_NAME,
        "gemma4",
        "token_embd.weight",
        GGMLType::Q8_0,
    );
    require_gemma4_gguf(
        &mmproj,
        GEMMA4_MMPROJ_NAME,
        "clip",
        "v.patch_embd.weight",
        GGMLType::F16,
    );

    let suffix = std::process::id();
    let image_path = std::env::temp_dir().join(format!("rmi-gemma4-turn-{suffix}.png"));
    image::RgbImage::from_fn(64, 48, |x, y| {
        image::Rgb([(x * 3 + y) as u8, (y * 5 + x) as u8, (x ^ y) as u8])
    })
    .save(&image_path)
    .unwrap();

    let audio_path = std::env::temp_dir().join(format!("rmi-gemma4-turn-{suffix}.wav"));
    let samples: Vec<i16> = (0..320)
        .map(|index| ((index as f32 * 0.03125).sin() * 16_384.0) as i16)
        .collect();
    let data_len = u32::try_from(samples.len() * 2).unwrap();
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&16_000u32.to_le_bytes());
    wav.extend_from_slice(&32_000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    std::fs::write(&audio_path, wav).unwrap();

    let result = run_multimodal(Gemma4Request {
        model: &model,
        mmproj: Some(&mmproj),
        image: Some(&image_path),
        audio: Some(&audio_path),
        prompt: "describe",
        max_tokens: 1,
        threads: 4,
        kv_format: KvFormat::F32,
    });
    let _ = std::fs::remove_file(image_path);
    let _ = std::fs::remove_file(audio_path);
    result.unwrap();
}
