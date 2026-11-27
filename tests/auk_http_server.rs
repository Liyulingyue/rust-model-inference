//! Integration test for the AuK HTTP routes.
//!
//! Spawns a real `rust-model-server` with the AuK components, posts speech
//! requests to `/v1/audio/speech` and `/v1/audio/generations`, and checks the
//! answers: the WAV container, the agreement between the two routes and the
//! rejection of an unsupported response format.
//!
//! Run with:
//!   RMI_AUK_DIT=models/AuK-Base-and-Flash-GGUF/auk-base-f16.gguf \
//!   RMI_AUK_VAE=models/AuK-Base-and-Flash-GGUF/auk-vae-f32.gguf \
//!   RMI_AUK_TEXT=models/AuK-Base-and-Flash-GGUF/qwen2.5-omni-3b-q8_0.gguf \
//!   cargo test --features vulkan --profile release-fast --test auk_http_server -- --ignored

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// The AuK components the server loads.
fn component(name: &str, fallback: &str) -> Option<PathBuf> {
    let path = std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(fallback));
    path.is_file().then_some(path)
}

fn pick_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

struct ServerGuard {
    child: Child,
    addr: SocketAddr,
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_server(dit: &Path, vae: &Path, text: &Path) -> ServerGuard {
    let bin = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/release-fast/rust-model-server");
    let port = pick_free_port();
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let mut cmd = Command::new(&bin);
    cmd.arg("--model")
        .arg(dit)
        .arg("--vae")
        .arg(vae)
        .arg("--text-encoder")
        .arg(text)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--threads")
        .arg("4")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let child = cmd.spawn().expect("spawn rust-model-server");
    let deadline = Instant::now() + Duration::from_secs(300);
    let models_url = format!("http://{addr}/v1/models");
    while Instant::now() < deadline {
        if std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok() {
            std::thread::sleep(Duration::from_millis(500));
            if Command::new("curl")
                .args(["-s", "-m", "5", &models_url])
                .output()
                .map(|output| !output.stdout.is_empty())
                .unwrap_or(false)
            {
                return ServerGuard { child, addr };
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    panic!("rust-model-server never became reachable on {addr}");
}

/// Post a JSON body and return the status code, the content type and the body.
fn post_json(addr: SocketAddr, path: &str, body: &str) -> (u16, String, Vec<u8>) {
    let url = format!("http://{addr}{path}");
    let output = Command::new("curl")
        .args([
            "-s",
            "-m",
            "900",
            "-o",
            "/tmp/auk_http_body.bin",
            "-w",
            "%{http_code} %{content_type}",
        ])
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("-d")
        .arg(body)
        .arg(&url)
        .output()
        .expect("run curl");
    let head = String::from_utf8_lossy(&output.stdout).to_string();
    let mut parts = head.split_whitespace();
    let status: u16 = parts.next().unwrap_or("0").parse().unwrap_or(0);
    let content_type = parts.next().unwrap_or("").to_string();
    let body = std::fs::read("/tmp/auk_http_body.bin").unwrap_or_default();
    (status, content_type, body)
}

#[test]
#[ignore = "needs the AuK weights and renders real audio"]
fn the_auk_routes_answer_with_wav() {
    let Some(dit) = component(
        "RMI_AUK_DIT",
        "models/AuK-Base-and-Flash-GGUF/auk-base-f16.gguf",
    ) else {
        eprintln!("skipping: set RMI_AUK_DIT to the AuK DiT GGUF");
        return;
    };
    let Some(vae) = component(
        "RMI_AUK_VAE",
        "models/AuK-Base-and-Flash-GGUF/auk-vae-f32.gguf",
    ) else {
        eprintln!("skipping: set RMI_AUK_VAE to the AuK VAE GGUF");
        return;
    };
    let Some(text) = component(
        "RMI_AUK_TEXT",
        "models/AuK-Base-and-Flash-GGUF/qwen2.5-omni-3b-q8_0.gguf",
    ) else {
        eprintln!("skipping: set RMI_AUK_TEXT to the Qwen2.5-Omni GGUF");
        return;
    };

    let server = spawn_server(&dit, &vae, &text);
    let speech_body = r#"{"model":"auk","input":"The next train leaves in ten minutes.","steps":4,"duration":1,"seed":0}"#;
    let generations_body =
        r#"{"prompt":"The next train leaves in ten minutes.","steps":4,"duration":1,"seed":0}"#;

    let (status, content_type, speech) = post_json(server.addr, "/v1/audio/speech", speech_body);
    assert_eq!(status, 200, "the speech route answered {status}");
    assert_eq!(content_type, "audio/wav");
    assert_eq!(&speech[0..4], b"RIFF");
    assert_eq!(&speech[8..12], b"WAVE");
    assert!(speech.len() > 44, "the WAV carries no samples");

    let (status, content_type, generations) =
        post_json(server.addr, "/v1/audio/generations", generations_body);
    assert_eq!(status, 200, "the generations route answered {status}");
    assert_eq!(content_type, "audio/wav");
    assert_eq!(
        speech, generations,
        "the two routes must render the same audio"
    );

    let (status, _, error) = post_json(
        server.addr,
        "/v1/audio/generations",
        r#"{"prompt":"hi","response_format":"mp3"}"#,
    );
    assert_eq!(status, 400);
    assert!(
        String::from_utf8_lossy(&error).contains("response_format"),
        "the error names the bad field: {}",
        String::from_utf8_lossy(&error)
    );
}
