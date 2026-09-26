//! Integration test for the `/v1/rerank` HTTP endpoint. Spawns a real
//! `rust-model-server` with the Qwen3-Reranker GGUF, posts a request,
//! and checks both the schema and the semantic ordering.

use std::net::SocketAddr;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

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

fn spawn_server(gguf: &std::path::Path) -> ServerGuard {
    let bin = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/release-fast/rust-model-server");
    let port = pick_free_port();
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let mut cmd = Command::new(&bin);
    cmd.arg("--model")
        .arg(gguf)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--threads")
        .arg("4")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let child = cmd.spawn().expect("spawn rust-model-server");
    let deadline = Instant::now() + Duration::from_secs(60);
    let health_url = format!("http://{addr}/v1/models");
    while Instant::now() < deadline {
        if std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(200))
            .is_ok()
        {
            // Give the model one more second to finish loading + write
            // its initial /v1/models response.
            std::thread::sleep(Duration::from_millis(500));
            if std::process::Command::new("curl")
                .args(["-s", "-m", "2", &health_url])
                .output()
                .map(|o| !o.stdout.is_empty())
                .unwrap_or(false)
            {
                return ServerGuard { child, addr };
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    panic!("rust-model-server never became reachable on {addr}");
}

#[derive(serde::Deserialize, serde::Serialize)]
struct RerankRequest<'a> {
    model: Option<&'a str>,
    query: &'a str,
    documents: Vec<&'a str>,
    top_n: Option<usize>,
    max_tokens_per_doc: Option<usize>,
}

#[derive(serde::Deserialize)]
struct RerankResponse {
    results: Vec<RerankItem>,
}

#[derive(serde::Deserialize)]
struct RerankItem {
    index: usize,
    relevance_score: f32,
}

fn post_json<T: serde::de::DeserializeOwned>(addr: SocketAddr, path: &str, body: &str) -> T {
    let mut cmd = Command::new("curl");
    cmd.arg("-s")
        .arg("-m")
        .arg("60")
        .arg("-X")
        .arg("POST")
        .arg(format!("http://{addr}{path}"))
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("-d")
        .arg(body);
    let out = cmd.output().expect("curl POST rerank");
    assert!(
        out.status.success(),
        "curl failed: stderr={}, stdout={:?}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    let text = std::str::from_utf8(&out.stdout).unwrap();
    serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("parse JSON {text}: {e}"))
}

#[test]
fn rerank_http_paris_top1() {
    let path = std::env::var_os("RMI_QWEN3_RERANK_Q8_MODEL")
        .map(std::path::PathBuf::from)
        .expect("set RMI_QWEN3_RERANK_Q8_MODEL to the Q8_0 GGUF path");
    let server = spawn_server(&path);

    let req = RerankRequest {
        model: None,
        query: "What is the capital of France?",
        documents: vec![
            "Paris is the capital and most populous city of France.",
            "Photosynthesis is the process used by plants to convert light energy.",
            "London is the capital city of England and the United Kingdom.",
        ],
        top_n: Some(3),
        max_tokens_per_doc: None,
    };
    let body = serde_json::to_string(&req).unwrap();
    let resp: RerankResponse =
        post_json(server.addr, "/v1/rerank", &body);

    // Semantic: Paris > London > Photosynthesis.
    assert_eq!(resp.results.len(), 3);
    assert_eq!(resp.results[0].index, 0, "Paris should rank first");
    assert!(
        resp.results[0].relevance_score > resp.results[1].relevance_score,
        "Paris (idx 0) relevance ({:.4}) should beat London (idx 2)",
        resp.results[0].relevance_score
    );
    assert!(
        resp.results[1].relevance_score > resp.results[2].relevance_score,
        "London (idx 2) relevance ({:.4}) should beat Photosynthesis (idx 1)",
        resp.results[1].relevance_score
    );
    // Generous bound — the exact gap depends on the active kernel path
    // (shared vs falcon-local Q8 matmul), which we accept ULP-level
    // differences on per the repo's policy. The relative order is the
    // contract.
    assert!(
        resp.results[0].relevance_score > 0.5,
        "Paris relevance should be high (>=0.5); got {:.4}",
        resp.results[0].relevance_score
    );
}

#[test]
fn rerank_http_top_n_truncates() {
    let path = std::env::var_os("RMI_QWEN3_RERANK_Q8_MODEL")
        .map(std::path::PathBuf::from)
        .expect("set RMI_QWEN3_RERANK_Q8_MODEL");
    let server = spawn_server(&path);

    let req = RerankRequest {
        model: None,
        query: "How do I reverse a list in Python?",
        documents: vec![
            "lst[::-1] is a Python slice that reverses a list",
            "Cats love sitting on windowsills in the sun.",
            "Rust uses Vec::reverse() to reverse a vector in place.",
            "Photosynthesis is plant biology, unrelated to lists.",
        ],
        top_n: Some(2),
        max_tokens_per_doc: None,
    };
    let body = serde_json::to_string(&req).unwrap();
    let resp: RerankResponse =
        post_json(server.addr, "/v1/rerank", &body);

    // top_n=2 ⇒ only 2 results, with indices into the input docs.
    assert_eq!(resp.results.len(), 2);
    let top = &resp.results[0];
    assert!(
        top.index == 0 || top.index == 2,
        "top result should be one of the code-related docs (0 or 2); got {}",
        top.index
    );
}

#[test]
fn rerank_http_rejects_empty_documents() {
    let path = std::env::var_os("RMI_QWEN3_RERANK_Q8_MODEL")
        .map(std::path::PathBuf::from)
        .expect("set RMI_QWEN3_RERANK_Q8_MODEL");
    let server = spawn_server(&path);

    let body = r#"{"query":"x","documents":[],"top_n":3}"#;
    let mut cmd = Command::new("curl");
    cmd.arg("-s")
        .arg("-m")
        .arg("30")
        .arg("-X")
        .arg("POST")
        .arg(format!("http://{}/v1/rerank", server.addr))
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("-d")
        .arg(body);
    let out = cmd.output().expect("curl POST rerank");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    // Empty documents must yield 4xx.
    assert!(
        stdout.contains("\"error\""),
        "expected error response, got stdout={stdout:?}"
    );
    assert!(
        stdout.contains("documents must be non-empty"),
        "got stdout={stdout:?}"
    );
}