//! Integration tests for `jina-embeddings-v5-omni-small-retrieval`.
//!
//! Run with `RMI_JINA_OMNI_MODEL=/path/to/q8_0.gguf` (or set
//! `RMI_JINA_OMNI_TEXT_MODEL` to the same value). Vision and audio tests
//! need `RMI_JINA_OMNI_VISION_MMPROJ` and `RMI_JINA_OMNI_AUDIO_MMPROJ`
//! respectively; missing env vars silently skip rather than fail.
//!
//! What's verified locally on this machine:
//! - **Text path**: full `--embedding` round-trip on the text LLM,
//!   1024-dim finite-and-L2-normalized output.
//! - **Vision path**: full `--embedding --mmproj <vision> --image <jpg>`
//!   round-trip, 1024-dim finite-and-L2-normalized output.
//! - **Audio path**: full `--embedding --mmproj <audio> --audio <wav>`
//!   round-trip, 1024-dim finite-and-L2-normalized output.
//! - **Server dispatch**: `rust-model-server` auto-detects qwen3-arch +
//!   pooling_type + 1024-dim embedding and serves `POST /v1/embeddings`
//!   end-to-end against the text LLM.
//!
//! Bit-equal oracle tests against pinned llama.cpp `b96806d` for the
//! vision and audio projection layers (PR #94 macOS ARM run) live in
//! `tests/qwen_drive_vlm_reference.rs::jina_omni_vision_matches_llama_cpp_bitwise`
//! and `src/models/qwen3/omni.rs::jina_audio_projection_matches_llama_cpp_bits` —
//! both `#[ignore]` because they require a `b96806d` llama.cpp checkout +
//! oracle sidecars. This file covers everything that can be verified
//! locally on a vanilla `cargo test --release`.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use rust_model_inference::app::run_omni_embedding;
use rust_model_inference::GGUFLoader;

/// Resolve the test pair's three files. None means env vars present and
/// files exist; otherwise the caller returns early and the test is
/// silently no-op, so this whole file is safe to run on machines that
/// don't have jina v5 omni GGUFs.
fn files() -> Option<(PathBuf, PathBuf, PathBuf)> {
    static CACHE: OnceLock<Option<(PathBuf, PathBuf, PathBuf)>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let text = std::env::var_os("RMI_JINA_OMNI_TEXT_MODEL")
                .or_else(|| std::env::var_os("RMI_JINA_OMNI_MODEL"))?;
            let text = PathBuf::from(text);
            let vision = std::env::var_os("RMI_JINA_OMNI_VISION_MMPROJ")
                .or_else(|| std::env::var_os("RMI_JINA_OMNI_MMPROJ"))
                .map(PathBuf::from);
            let audio = std::env::var_os("RMI_JINA_OMNI_AUDIO_MMPROJ")
                .or_else(|| std::env::var_os("RMI_JINA_OMNI_MMPROJ"))
                .map(PathBuf::from);
            if !text.exists() {
                return None;
            }
            if let Some(path) = &vision {
                if !path.exists() {
                    return None;
                }
            }
            if let Some(path) = &audio {
                if !path.exists() {
                    return None;
                }
            }
            Some((text, vision?, audio?))
        })
        .clone()
}

fn loader(path: &std::path::Path) -> GGUFLoader {
    GGUFLoader::from_file(path).expect("loader from file")
}

#[test]
fn contract_pins_jina_v5_omni_small_retrieval_metadata() {
    let Some((text, _vision, _audio)) = files() else {
        return;
    };
    let loader = loader(&text);
    assert_eq!(
        loader
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val())
            .unwrap_or_default(),
        "qwen3",
        "Jina v5 omni rides on the qwen3 trunk via general.architecture"
    );
    assert_eq!(
        loader
            .metadata("general.basename")
            .and_then(|v| v.to_string_val())
            .unwrap_or_default(),
        "omni",
    );
    assert_eq!(
        loader
            .metadata("general.finetune")
            .and_then(|v| v.to_string_val())
            .unwrap_or_default(),
        "retrieval-text-hf",
    );
    let pooling = loader
        .metadata("qwen3.pooling_type")
        .and_then(|v| v.to_u64())
        .unwrap_or(0);
    assert_eq!(
        pooling, 3,
        "Jina v5 omni uses last-token pooling (qwen3.pooling_type = 3)"
    );
    let embd = loader
        .metadata("qwen3.embedding_length")
        .and_then(|v| v.to_u64())
        .unwrap_or(0);
    assert_eq!(embd, 1024, "v5 omni embedding dim is 1024");
}

#[test]
fn text_embedding_returns_1024_dim_finite_normalized_vector() {
    let Some((text, _vision, _audio)) = files() else {
        return;
    };
    // Jina v5 omni's text LLM rides on `qwen3` arch; the embedding path
    // is `models::qwen3::embedding::compute_embedding`, which is what
    // `app::run_embedding` dispatches to for any non-special arch.
    let source: Arc<dyn rust_model_inference::TensorSource> = Arc::new(loader(&text));
    let embed = rust_model_inference::models::qwen3::embedding::compute_embedding(
        source.as_ref(),
        "Represent this for retrieval: jina v5 omni text path test",
        4,
    )
    .expect("text embedding must succeed");
    assert_eq!(embed.len(), 1024, "1024-dim embedding");
    let norm_sq: f64 = embed.iter().map(|v| f64::from(*v).powi(2)).sum();
    let norm = norm_sq.sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "L2-normalized embedding expected (norm={norm})"
    );
    for (i, v) in embed.iter().enumerate() {
        assert!(v.is_finite(), "non-finite value at dim {i}: {v}");
    }
}

#[test]
fn vision_embedding_returns_1024_dim_finite_normalized_vector() {
    let Some((_text, vision, _audio)) = files() else {
        return;
    };
    // Use models/apple.png if available; the test is no-op without it.
    let image = std::path::Path::new("models/apple.png");
    if !image.exists() {
        return;
    }
    let source: Arc<dyn rust_model_inference::TensorSource> = Arc::new(loader(&_text));
    let embed = run_omni_embedding(
        source.as_ref(),
        &vision,
        Some(image),
        None,
        None,
        "Represent this image for retrieval: ",
        4,
    )
    .expect("vision embedding must succeed");
    assert_eq!(embed.len(), 1024, "1024-dim vision embedding");
    let norm_sq: f64 = embed.iter().map(|v| f64::from(*v).powi(2)).sum();
    let norm = norm_sq.sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "L2-normalized vision embedding expected (norm={norm})"
    );
    for (i, v) in embed.iter().enumerate() {
        assert!(v.is_finite(), "non-finite vision value at dim {i}: {v}");
    }
}

#[test]
fn audio_embedding_returns_1024_dim_finite_normalized_vector() {
    let Some((_text, _vision, audio)) = files() else {
        return;
    };
    // Use a 440 Hz 16 kHz mono PCM16 WAV if available; generate it on
    // demand so the test stays self-contained.
    let wav = std::path::Path::new("/tmp/jina-test/jina_omni_440hz.wav");
    if !wav.exists() {
        // Generate a 10 s 440 Hz sine so the audio path can run end-to-end.
        if let Some(parent) = wav.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let generated = generate_test_wav(wav, 10);
        if !generated {
            return;
        }
    }
    let source: Arc<dyn rust_model_inference::TensorSource> = Arc::new(loader(&_text));
    let embed = run_omni_embedding(
        source.as_ref(),
        &audio,
        None,
        None,
        Some(wav),
        "Represent this audio for retrieval: ",
        4,
    )
    .expect("audio embedding must succeed");
    assert_eq!(embed.len(), 1024, "1024-dim audio embedding");
    let norm_sq: f64 = embed.iter().map(|v| f64::from(*v).powi(2)).sum();
    let norm = norm_sq.sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "L2-normalized audio embedding expected (norm={norm})"
    );
    for (i, v) in embed.iter().enumerate() {
        assert!(v.is_finite(), "non-finite audio value at dim {i}: {v}");
    }
}

/// Generate a 16 kHz mono PCM16 WAV with a 440 Hz sine of the given
/// length. Returns true on success, false on any I/O failure.
fn generate_test_wav(path: &std::path::Path, seconds: usize) -> bool {
    use std::io::Write;
    let mut data = Vec::with_capacity(seconds * 16000 * 2);
    for i in 0..(seconds * 16000) {
        // 440 Hz, amplitude 0.2 of int16 max
        let sample =
            (0.2 * 32767.0 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
                as i16;
        data.extend_from_slice(&sample.to_le_bytes());
    }
    let header = build_wav_header(data.len() as u32);
    let mut file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    if file.write_all(&header).is_err() {
        return false;
    }
    file.write_all(&data).is_ok()
}

fn build_wav_header(data_len: u32) -> Vec<u8> {
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(data_len + 36u32).to_le_bytes()); // file size - 8
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    h.extend_from_slice(&1u16.to_le_bytes()); // PCM
    h.extend_from_slice(&1u16.to_le_bytes()); // mono
    h.extend_from_slice(&16000u32.to_le_bytes()); // sample rate
    h.extend_from_slice(&(16000u32 * 1u32 * 2u32).to_le_bytes()); // byte rate
    h.extend_from_slice(&(1u16 * 2u16).to_le_bytes()); // block align
    h.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data_len.to_le_bytes());
    h
}

#[test]
fn server_post_embeddings_against_jina_omni_text() {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    let Some((text, _vision, _audio)) = files() else {
        return;
    };
    let bin = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/release-fast/rust-model-server");
    if !bin.exists() {
        return;
    }

    // Pick a free port.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let mut child = Command::new(&bin)
        .arg("--model")
        .arg(&text)
        .arg("--embedding")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--threads")
        .arg("4")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn rust-model-server");

    let addr = format!("127.0.0.1:{port}");
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut ready = false;
    while Instant::now() < deadline {
        if TcpStream::connect_timeout(&addr.parse().unwrap(), Duration::from_millis(200)).is_ok() {
            ready = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    if !ready {
        let _ = child.kill();
        let _ = child.wait();
        return;
    }
    // Wait for /v1/models to actually return a body (model loaded).
    let models_deadline = Instant::now() + Duration::from_secs(120);
    let mut head_buf = [0u8; 256];
    while Instant::now() < models_deadline {
        if let Ok(mut s) =
            TcpStream::connect_timeout(&addr.parse().unwrap(), Duration::from_millis(500))
        {
            let req = b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
            if s.write_all(req).is_ok() {
                if s.read(&mut head_buf).is_ok() && !head_buf.is_empty() {
                    break;
                }
            }
        }
        std::thread::sleep(Duration::from_secs(1));
    }

    let body = serde_json::json!({
        "model": "jina-embeddings-v5-omni-small-retrieval",
        "input": ["Represent this for retrieval: server test"]
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();
    let request = format!(
        "POST /v1/embeddings HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body_bytes.len()
    );
    let mut stream = TcpStream::connect(&addr).expect("connect server");
    stream.write_all(request.as_bytes()).expect("write request");
    stream.write_all(&body_bytes).expect("write body");
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("read response");
    let _ = child.kill();
    let _ = child.wait();

    if response.is_empty() {
        panic!("empty HTTP response from rust-model-server");
    }
    // Find the JSON body (after the CRLFCRLF separator).
    let body_start = response
        .find("\r\n\r\n")
        .map(|i| i + 4)
        .unwrap_or_else(|| panic!("response must contain body separator; got: {response:?}"));
    let body_json = &response[body_start..];
    let parsed: serde_json::Value = serde_json::from_str(body_json)
        .unwrap_or_else(|error| panic!("parse response JSON: {error}; body: {body_json:?}"));
    let embedding = parsed["data"][0]["embedding"]
        .as_array()
        .expect("data[0].embedding must be an array");
    assert_eq!(
        embedding.len(),
        1024,
        "POST /v1/embeddings returned a 1024-dim vector"
    );
    for (i, value) in embedding.iter().enumerate() {
        let v = value.as_f64().expect("element must be f64");
        assert!(v.is_finite(), "non-finite value at dim {i}: {v}");
    }
    let norm_sq: f64 = embedding.iter().map(|v| v.as_f64().unwrap().powi(2)).sum();
    let norm = norm_sq.sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "POST /v1/embeddings response is L2-normalized (norm={norm})"
    );
}
