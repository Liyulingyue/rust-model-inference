#[cfg(target_arch = "aarch64")]
use rust_model_inference::models::qwen3::tts::codec::conv::{
    conv_transpose1d_causal, ConvTranspose1dState,
};
use rust_model_inference::models::qwen3::tts::predictor_top_k;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
fn zero_temperature_uses_deterministic_predictor_top_k() {
    assert_eq!(predictor_top_k(0.0), 1);
    assert_eq!(predictor_top_k(-1.0), 1);
    assert_eq!(predictor_top_k(0.9), 50);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn causal_transpose_conv_uses_ggml_neon_dot_order() {
    let input = [
        0xc1c63800, 0x3fca6000, 0xc1d18800, 0x41fa7e00, 0x41c3a400, 0x4175e800, 0xc105c000,
        0x411a3400, 0xc08ab800, 0xc184fa00, 0xbf7dc000, 0x41711c00, 0xc0ea9800, 0xc1ba6a00,
        0xc1ae4a00, 0x41ce1400,
    ]
    .map(f32::from_bits);
    let weight0 = [
        0xc1b15a00, 0xc0fc8000, 0x41a2d000, 0xc13cdc00, 0xc0579000, 0xc1874c00, 0xc1a77000,
        0xc1a75a00, 0xc0d8d000, 0xc14fe000, 0x411f0000, 0x41d34a00, 0xc1f25a00, 0x41b01000,
        0xc1e24000, 0xc1139800,
    ]
    .map(f32::from_bits);
    let weight1 = [
        0x3dd20000, 0xc0eac800, 0x41c4b800, 0x3fdda000, 0x414d3400, 0x41927c00, 0x4184d200,
        0x41b17000, 0x3ebf8000, 0x4029d000, 0x40b41000, 0xc1699c00, 0xc199a000, 0xc1c52a00,
        0x41940200, 0xc1ada600,
    ]
    .map(f32::from_bits);
    let mut kernel = Vec::with_capacity(32);
    for (&left, &right) in weight0.iter().zip(&weight1) {
        kernel.extend([left, right]);
    }

    let output = conv_transpose1d_causal(
        &kernel,
        None,
        &input,
        16,
        1,
        1,
        2,
        2,
        &mut ConvTranspose1dState::default(),
    )
    .unwrap();

    assert_eq!(
        output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [0xc1580e40, 0xc3e233d4],
    );
}

#[test]
fn oracle_builder_treats_input_checkout_as_read_only() {
    let directory =
        std::env::temp_dir().join(format!("rmi-qwen3-tts-oracle-input-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    assert!(Command::new("git")
        .args(["init", "-q"])
        .arg(&directory)
        .status()
        .unwrap()
        .success());
    assert!(Command::new("git")
        .args(["-C", directory.to_str().unwrap(), "remote", "add", "origin"])
        .arg(directory.join("missing-origin"))
        .status()
        .unwrap()
        .success());
    std::fs::write(directory.join("dirty"), b"untracked").unwrap();

    let before = Command::new("git")
        .args([
            "-C",
            directory.to_str().unwrap(),
            "status",
            "--porcelain=v1",
        ])
        .output()
        .unwrap();

    let output = Command::new("bash")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tools/tts/build_qwen3_tts_oracle.sh"
        ))
        .arg(&directory)
        .output()
        .unwrap();
    let after = Command::new("git")
        .args([
            "-C",
            directory.to_str().unwrap(),
            "status",
            "--porcelain=v1",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert_eq!(before.stdout, after.stdout);
    assert!(!String::from_utf8_lossy(&output.stderr).contains("clean checkout"));
    let _ = std::fs::remove_dir_all(&directory);
}

fn run(command: &mut Command) {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn records(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn last_record<'a>(records: &'a [Value], name: &str) -> &'a Value {
    records
        .iter()
        .rev()
        .find(|record| record["name"] == name)
        .unwrap_or_else(|| panic!("missing parity checkpoint {name}"))
}

fn shape(record: &Value) -> Vec<usize> {
    record["shape"]
        .as_array()
        .expect("checkpoint must contain a shape")
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect()
}

fn token_ids(record: &Value) -> Vec<u32> {
    record["token_ids"]
        .as_array()
        .expect("Rust integer checkpoint must contain token_ids")
        .iter()
        .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
        .collect()
}

fn read_i32(path: &Path) -> Vec<u32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| {
            u32::try_from(i32::from_le_bytes(bytes.try_into().unwrap()))
                .expect("oracle token IDs must be nonnegative")
        })
        .collect()
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len() % 4, 0, "{}", path.display());
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn oracle_sidecar(directory: &Path, record: &Value) -> PathBuf {
    directory.join(record["path"].as_str().unwrap())
}

fn rust_sidecar(trace: &Path, name: &str) -> PathBuf {
    PathBuf::from(format!("{}.{}.f32", trace.display(), name))
}

fn compare_f32_checkpoint(
    name: &str,
    rust_trace: &Path,
    rust_records: &[Value],
    oracle_directory: &Path,
    oracle_records: &[Value],
    tolerance: f32,
) {
    let rust_record = last_record(rust_records, name);
    let oracle_record = last_record(oracle_records, name);
    let rust_shape = shape(rust_record);
    let oracle_shape = shape(oracle_record);
    assert_eq!(rust_shape.len(), oracle_shape.len(), "{name}");
    assert_eq!(&rust_shape[1..], &oracle_shape[1..], "{name}");
    assert!(rust_shape[0] <= oracle_shape[0], "{name}");

    let rust = read_f32(&rust_sidecar(rust_trace, name));
    let oracle = read_f32(&oracle_sidecar(oracle_directory, oracle_record));
    assert_eq!(rust.len(), rust_shape.iter().product::<usize>(), "{name}");
    assert_eq!(
        oracle.len(),
        oracle_shape.iter().product::<usize>(),
        "{name}"
    );
    assert!(
        rust.iter().chain(&oracle).all(|value| value.is_finite()),
        "{name}"
    );

    let oracle = &oracle[..rust.len()];
    let max_abs = rust
        .iter()
        .zip(oracle)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    if let Some((index, (&rust, &oracle))) = rust
        .iter()
        .zip(oracle)
        .enumerate()
        .find(|(_, (left, right))| (*left - *right).abs() > tolerance)
    {
        panic!(
            "{name}[{index}] Rust={rust} ({:08x}) oracle={oracle} ({:08x}), max_abs={max_abs}",
            rust.to_bits(),
            oracle.to_bits(),
        );
    }
}

fn pcm16(path: &Path) -> Vec<i16> {
    let bytes = std::fs::read(path).unwrap();
    assert!(bytes.len() >= 44, "{}", path.display());
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 1);
    assert_eq!(
        u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
        24_000
    );
    assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 16);
    let data_len = u32::from_le_bytes(bytes[40..44].try_into().unwrap()) as usize;
    assert!(data_len > 0 && data_len % 2 == 0);
    assert_eq!(bytes.len(), 44 + data_len);
    bytes[44..]
        .chunks_exact(2)
        .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn parity_fixture(fixture: usize, prompt: &str) -> (Vec<u32>, Vec<i16>) {
    let model = std::env::var("QWEN3_TTS_MODEL").unwrap();
    let mmproj = std::env::var("QWEN3_TTS_MMPROJ").unwrap();
    let reference = std::env::var("QWEN3_TTS_REF_WAV").unwrap();
    let oracle = std::env::var("QWEN3_TTS_ORACLE_BIN").unwrap();
    let trace_root = PathBuf::from(std::env::var("QWEN3_TTS_ORACLE_TRACE").unwrap());
    let directory = trace_root.join(format!("rmi-qwen3-tts-{}-{fixture}", std::process::id()));
    let oracle_trace = directory.join("oracle");
    let rust_trace = directory.join("rust.jsonl");
    let oracle_wav = directory.join("oracle.wav");
    let rust_wav = directory.join("rust.wav");
    std::fs::create_dir_all(&oracle_trace).unwrap();

    run(Command::new(oracle)
        .env("QWEN3_TTS_ORACLE_TRACE", &oracle_trace)
        .args([
            "-m",
            &model,
            "-mm",
            &mmproj,
            "--tts-speaker-file",
            &reference,
            "--tts-lang",
            "en",
            "-p",
            prompt,
            "-n",
            "4",
            "--temp",
            "0",
            "--top-k",
            "1",
            "--top-p",
            "1.0",
            "-o",
            oracle_wav.to_str().unwrap(),
        ]));
    run(Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .env("RMI_PARITY_TRACE", &rust_trace)
        .env(
            "RMI_PARITY_FILTER",
            "tts.prompt_ids,tts.prompt_embeddings,tts.speaker_embedding,tts.frame_codes,tts.rvq_hidden,tts.wav_pre_conv,tts.wav_tfm,tts.pcm",
        )
        .args([
            "--model",
            &model,
            "--mmproj",
            &mmproj,
            "--tts",
            "--prompt",
            prompt,
            "--language",
            "en",
            "--ref-audio",
            &reference,
            "--out",
            rust_wav.to_str().unwrap(),
            "--max-tokens",
            "4",
            "--temp",
            "0",
            "--threads",
            "1",
        ]));

    let oracle_records = records(&oracle_trace.join("trace.jsonl"));
    let rust_records = records(&rust_trace);
    let oracle_prompt = last_record(&oracle_records, "tts.prompt_ids");
    assert_eq!(
        token_ids(last_record(&rust_records, "tts.prompt_ids")),
        read_i32(&oracle_sidecar(&oracle_trace, oracle_prompt)),
    );
    let rust_codes: Vec<u32> = rust_records
        .iter()
        .filter(|record| record["name"] == "tts.frame_codes")
        .flat_map(token_ids)
        .collect();
    let oracle_codes_record = last_record(&oracle_records, "tts.frame_codes");
    assert_eq!(
        rust_codes,
        read_i32(&oracle_sidecar(&oracle_trace, oracle_codes_record)),
    );
    assert_eq!(rust_codes.len(), 4 * 16);

    for (name, tolerance) in [
        ("tts.speaker_embedding", 1e-5),
        ("tts.prompt_embeddings", 1e-5),
        ("tts.rvq_hidden", 2e-4),
        ("tts.wav_pre_conv", 2e-4),
        ("tts.wav_tfm", 2e-4),
        ("tts.pcm", 2e-4),
    ] {
        compare_f32_checkpoint(
            name,
            &rust_trace,
            &rust_records,
            &oracle_trace,
            &oracle_records,
            tolerance,
        );
    }

    let rust_pcm = pcm16(&rust_wav);
    assert_eq!(rust_pcm, pcm16(&oracle_wav));
    let _ = std::fs::remove_dir_all(&directory);
    (rust_codes, rust_pcm)
}

#[test]
#[ignore = "requires Qwen3-TTS models, reference WAV, and pinned llama.cpp oracle"]
fn qwen3_tts_matches_pinned_oracle_and_distinguishes_prompts() {
    let (first_codes, first_pcm) = parity_fixture(0, "Hello from Qwen");
    let (second_codes, second_pcm) = parity_fixture(1, "A different sentence");
    assert_ne!(
        first_codes.iter().step_by(16).collect::<Vec<_>>(),
        second_codes.iter().step_by(16).collect::<Vec<_>>()
    );
    assert_ne!(first_pcm, second_pcm);
}
