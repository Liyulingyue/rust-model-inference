//! Integration test for the `/v1/forecast` HTTP endpoint.
//!
//! Spawns a real `rust-model-server` with the Timestral checkpoint, posts
//! a forecast request, and checks the schema, the shape of the answer and
//! that the server agrees with the CLI on the same input.

use std::net::SocketAddr;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// The checkpoint the server loads.
///
/// The model ships with the repository, so the path is known; an
/// environment variable overrides it for a differently named file.
fn checkpoint() -> std::path::PathBuf {
    std::env::var_os("RMI_TIMESTRAL_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("models/Timestral-209M/Timestral-209M-F32.gguf")
        })
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
    let deadline = Instant::now() + Duration::from_secs(120);
    let health_url = format!("http://{addr}/v1/models");
    while Instant::now() < deadline {
        if std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok() {
            std::thread::sleep(Duration::from_millis(500));
            if Command::new("curl")
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

/// Post a JSON body and return the status code and the parsed body.
fn post(addr: SocketAddr, path: &str, body: &str) -> (u16, serde_json::Value) {
    let out = Command::new("curl")
        .arg("-s")
        .arg("-m")
        .arg("120")
        .arg("-w")
        .arg("\n%{http_code}")
        .arg("-X")
        .arg("POST")
        .arg(format!("http://{addr}{path}"))
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("-d")
        .arg(body)
        .output()
        .expect("curl POST");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let (payload, code) = text.rsplit_once('\n').expect("curl wrote a status line");
    (
        code.trim().parse().expect("status code"),
        serde_json::from_str(payload).unwrap_or_else(|e| panic!("parse {payload}: {e}")),
    )
}

fn get_status(addr: SocketAddr, path: &str) -> u16 {
    let out = Command::new("curl")
        .arg("-s")
        .arg("-m")
        .arg("10")
        .arg("-o")
        .arg("/dev/null")
        .arg("-w")
        .arg("%{http_code}")
        .arg(format!("http://{addr}{path}"))
        .output()
        .expect("curl GET");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .expect("status code")
}

const CONTEXT: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

fn univariate_request() -> String {
    serde_json::json!({
        "context": CONTEXT,
        "prediction_length": 4,
    })
    .to_string()
}

#[test]
fn forecast_endpoint_returns_the_quantile_grid() {
    let path = checkpoint();
    assert!(
        path.exists(),
        "the Timestral checkpoint is missing: {}",
        path.display()
    );
    let server = spawn_server(&path);
    let (status, body) = post(server.addr, "/v1/forecast", &univariate_request());
    assert_eq!(status, 200, "body={body}");

    let levels = body["quantile_levels"].as_array().expect("quantile_levels");
    assert_eq!(levels.len(), 21, "the model has 21 quantile levels");
    assert!(
        levels.windows(2).all(|w| w[0].as_f64() < w[1].as_f64()),
        "the quantile levels have to be sorted"
    );

    // `quantiles` is [target variate][horizon step][quantile].
    let quantiles = body["quantiles"].as_array().expect("quantiles");
    assert_eq!(quantiles.len(), 1, "a univariate request has one target");
    let horizon = quantiles[0].as_array().expect("horizon");
    assert_eq!(horizon.len(), 4, "prediction_length steps");
    for step in horizon {
        let values = step.as_array().expect("a step");
        assert_eq!(values.len(), 21, "one value per quantile level");
        // The quantile function is monotone in the level.
        assert!(
            values
                .windows(2)
                .all(|w| { w[0].as_f64().unwrap() <= w[1].as_f64().unwrap() }),
            "the quantiles of one step have to be sorted: {values:?}"
        );
    }
}

#[test]
fn forecast_endpoint_agrees_with_the_cli() {
    let path = checkpoint();
    assert!(path.exists(), "{}", path.display());
    let server = spawn_server(&path);
    let request = univariate_request();
    let (_, body) = post(server.addr, "/v1/forecast", &request);
    assert_eq!(body["quantile_levels"].as_array().unwrap().len(), 21);

    // The same request through the CLI has to produce the same numbers.
    let request_path = std::env::temp_dir().join("rmi-timestral-http-request.json");
    std::fs::write(&request_path, &request).expect("write the request");
    let bin = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/release-fast/rust-model-inference");
    let out = Command::new(&bin)
        .arg("--model")
        .arg(&path)
        .arg("--timestral-request")
        .arg(&request_path)
        .arg("--compute")
        .arg("cpu")
        .arg("--threads")
        .arg("4")
        .output()
        .expect("run the CLI");
    assert!(
        out.status.success(),
        "the CLI failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let cli: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("the CLI prints one JSON object");
    assert_eq!(
        cli["quantiles"], body["quantiles"],
        "the CLI and the endpoint have to agree"
    );
    assert_eq!(cli["quantile_levels"], body["quantile_levels"]);
    let _ = std::fs::remove_file(&request_path);
}

#[test]
fn forecast_endpoint_rejects_a_bad_request() {
    let path = checkpoint();
    assert!(path.exists(), "{}", path.display());
    let server = spawn_server(&path);

    // An empty context is a request the model itself refuses.
    let (status, body) = post(
        server.addr,
        "/v1/forecast",
        r#"{"context":[],"prediction_length":4}"#,
    );
    assert_eq!(status, 400, "body={body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .contains("context"),
        "the error has to say what is wrong: {body}"
    );

    // A body that is not a forecast request at all.
    let (status, _) = post(server.addr, "/v1/forecast", "{not json");
    assert_eq!(status, 400, "a malformed body is a bad request");
}

#[test]
fn the_forecast_backend_serves_nothing_else() {
    let path = checkpoint();
    assert!(path.exists(), "{}", path.display());
    let server = spawn_server(&path);
    // The backend decides which routes exist. A chat request has to be
    // absent, not an error.
    assert_eq!(get_status(server.addr, "/v1/chat/completions"), 404);
    assert_eq!(get_status(server.addr, "/v1/embeddings"), 404);
    assert_eq!(get_status(server.addr, "/health"), 200);
}
