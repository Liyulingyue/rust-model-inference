//! End-to-end HTTP test for `/v1/jev/boundary`, the GLiNER2.5
//! BoundaryExtractor route. Spawns a real server against the base-v1 GGUF and
//! posts a real schema, so this covers the wiring the library-level oracles
//! cannot: route dispatch, the raw `schema` body, the JSON response shape, and
//! the error paths.
//!
//! Set `RMI_GLINER2_5_BASE_V1_GGUF` to the F32 GGUF.

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
        .join("target/release-fast/rust-model-inference");
    let port = pick_free_port();
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let mut cmd = Command::new(&bin);
    // `--jev` is what selects the JEV backend family; `--gliner2-boundary` picks
    // the boundary head within it. `--serve` is the server flag (not
    // `--server`).
    cmd.arg("--model")
        .arg(gguf)
        .arg("--gliner2-boundary")
        .arg("--jev")
        .arg("--serve")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--threads")
        .arg("4")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let child = cmd.spawn().expect("spawn rust-model-inference --serve");
    let deadline = Instant::now() + Duration::from_secs(90);
    while Instant::now() < deadline {
        if std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok() {
            // Give the model a moment to finish loading before the first call.
            std::thread::sleep(Duration::from_millis(500));
            if std::process::Command::new("curl")
                .args(["-s", "-m", "2", &format!("http://{addr}/v1/models")])
                .output()
                .map(|o| !o.stdout.is_empty())
                .unwrap_or(false)
            {
                return ServerGuard { child, addr };
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    panic!("boundary server never became reachable on {addr}");
}

fn post(addr: SocketAddr, path: &str, body: &str) -> (u16, serde_json::Value) {
    let out = Command::new("curl")
        .arg("-s")
        .arg("-m")
        .arg("120")
        // The body is a fixture string, so `-w` output parsing is safe here.
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

#[derive(Debug, serde::Deserialize)]
struct Span {
    field: String,
    score: f32,
    start: usize,
    end: usize,
    text: String,
}

#[derive(serde::Deserialize)]
struct Classification {
    task: String,
    activation: String,
    labels: Vec<String>,
    probabilities: Vec<f32>,
    selected: Vec<String>,
    choice_label: Option<String>,
}

#[derive(serde::Deserialize)]
struct Head {
    field: String,
    null_logit: Option<f32>,
}

#[derive(serde::Deserialize)]
struct Response {
    mode: String,
    overlap_policy: String,
    spans: Vec<Span>,
    classifications: Vec<Classification>,
    query_heads: Vec<Head>,
}

const TEXT: &str = "Ada Lovelace worked in London. She collaborated with Charles Babbage.";

/// A single-extractive-group body: `{"context": ..., "schema": {"entities": [...]}}`.
fn entity_body(fields: &[&str]) -> String {
    serde_json::json!({
        "context": TEXT,
        "schema": {"entities": fields},
    })
    .to_string()
}

#[test]
fn boundary_http_extracts_spans_and_classifies() {
    let path = std::env::var_os("RMI_GLINER2_5_BASE_V1_GGUF")
        .map(std::path::PathBuf::from)
        .expect("set RMI_GLINER2_5_BASE_V1_GGUF to the F32 boundary GGUF");
    let server = spawn_server(&path);

    let body = entity_body(&["person", "location"]);
    let (code, value) = post(server.addr, "/v1/jev/boundary", &body);
    assert_eq!(code, 200, "body was {value}");
    let resp: Response = serde_json::from_value(value).expect("response shape");

    assert_eq!(resp.mode, "boundary");
    // `overlap_policy = "flat"` in the checkpoint normalizes to the canonical
    // name, and echoing it back makes a mis-transcribed setting visible
    // instead of silently doubling up spans.
    assert_eq!(resp.overlap_policy, "disallow");
    assert!(
        resp.classifications.is_empty(),
        "no classification group sent"
    );

    // Semantic check, not a golden: two people and one place in that text.
    let mut got: Vec<(&str, &str)> = resp
        .spans
        .iter()
        .map(|span| (span.field.as_str(), span.text.as_str()))
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            ("location", "london"),
            ("person", "ada lovelace"),
            ("person", "charles babbage"),
        ],
        "unexpected spans: {got:?}"
    );
    // Spans must be scored, and every score must clear the default threshold.
    assert!(
        resp.spans.iter().all(|span| span.score > 0.5),
        "a span below the default threshold leaked through"
    );
    // One query head per field, in schema order.
    let head_fields: Vec<&str> = resp.query_heads.iter().map(|h| h.field.as_str()).collect();
    assert_eq!(head_fields, vec!["person", "location"]);
    assert!(resp.query_heads.iter().all(|h| h.null_logit.is_some()));

    // Offsets index the reference's word list, not `split_whitespace`: that
    // list splits trailing punctuation into its own word ("London." becomes
    // "london" + ".") and the inference collator appends a final "." when the
    // text lacks one. Reconstructing it here would duplicate the tokenizer
    // oracle's job and test the wrong layer, so instead check the properties
    // the route owes: half-open non-empty ranges, per-field order and
    // disjointness, and text that really came from the context.
    for span in &resp.spans {
        assert!(
            span.end > span.start,
            "half-open range must be non-empty: {span:?}"
        );
        // Compare against a punctuation-stripped context: the reference's word
        // list splits "Babbage." into "babbage" + ".", so the raw context keeps
        // the period glued to the final word while the span text does not.
        let haystack = format!(
            " {} ",
            TEXT.to_lowercase()
                .split_whitespace()
                .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()))
                .filter(|word| !word.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        );
        let needle = format!(" {} ", span.text);
        assert!(
            haystack.contains(&needle),
            "span text {needle:?} does not occur in the context: {haystack:?}"
        );
    }
    for field in ["person", "location"] {
        let mut spans: Vec<&Span> = resp.spans.iter().filter(|s| s.field == field).collect();
        spans.sort_by_key(|span| (span.start, span.end));
        for pair in spans.windows(2) {
            assert!(
                pair[0].end <= pair[1].start,
                "{field} spans overlap or are unsorted: {:?} then {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    // A mixed schema runs both heads in one pass and keeps field order.
    let mixed = r#"{"context": "The product shipped late but the support team was great.",
        "schema": {"classifications": [{"task": "overall sentiment",
                        "labels": ["positive","neutral","negative"]}],
                   "entities": ["product"]}}"#
        .to_string();
    let (code, value) = post(server.addr, "/v1/jev/boundary", &mixed);
    assert_eq!(code, 200, "body was {value}");
    let resp: Response = serde_json::from_value(value).expect("mixed response shape");
    assert_eq!(resp.classifications.len(), 1);
    let group = &resp.classifications[0];
    assert_eq!(group.task, "overall sentiment");
    assert_eq!(group.activation, "softmax");
    // The label vocabulary comes back in schema order, and `probabilities` is
    // indexed by it — so this pins the two to each other, not just the sum.
    assert_eq!(group.labels, ["positive", "neutral", "negative"]);
    assert_eq!(group.probabilities.len(), group.labels.len());
    assert!((group.probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    // Single-label group: the winner is reported in `choice_label`, and
    // `selected` stays empty.
    assert!(
        group.selected.is_empty(),
        "single-label group leaves selected empty"
    );
    assert_eq!(group.choice_label.as_deref(), Some("positive"));
    // Only the extractive group contributes a query head.
    assert_eq!(resp.query_heads.len(), 1);
    assert_eq!(resp.query_heads[0].field, "product");
}

#[test]
fn boundary_http_threshold_override_reaches_the_decoder() {
    let path = std::env::var_os("RMI_GLINER2_5_BASE_V1_GGUF")
        .map(std::path::PathBuf::from)
        .expect("set RMI_GLINER2_5_BASE_V1_GGUF to the F32 boundary GGUF");
    let server = spawn_server(&path);

    // At the default threshold this model is confident enough that no two
    // spans of one field overlap, so the resolver is a no-op and the route
    // would look correct even if `threshold` were ignored. At 0.02 candidates
    // multiply, and `disallow` then has to collapse them.
    let body = r#"{"context": "Apple Inc. announced the iPhone in Cupertino.",
        "schema": {"entities": ["company"]}, "threshold": 0.02}"#
        .to_string();
    let (code, value) = post(server.addr, "/v1/jev/boundary", &body);
    assert_eq!(code, 200, "body was {value}");
    let resp: Response = serde_json::from_value(value).expect("response shape");
    assert!(!resp.spans.is_empty(), "low threshold should admit spans");
    assert!(
        resp.spans.iter().all(|span| span.score >= 0.02),
        "a span below the requested threshold came back"
    );
    // `disallow` means non-overlapping, so the surviving spans must not
    // intersect. Checked per field, since overlap is only ever resolved within
    // one field's query.
    for (i, a) in resp.spans.iter().enumerate() {
        for b in &resp.spans[i + 1..] {
            if a.field != b.field {
                continue;
            }
            assert!(
                a.end <= b.start || b.end <= a.start,
                "overlapping spans survived disallow: {a:?} and {b:?}"
            );
        }
    }
}

#[test]
fn boundary_http_rejects_bad_requests() {
    let path = std::env::var_os("RMI_GLINER2_5_BASE_V1_GGUF")
        .map(std::path::PathBuf::from)
        .expect("set RMI_GLINER2_5_BASE_V1_GGUF to the F32 boundary GGUF");
    let server = spawn_server(&path);

    // A schema with neither group is a 400, not an empty 200: the reference
    // raises rather than returning nothing.
    let (code, value) = post(
        server.addr,
        "/v1/jev/boundary",
        r#"{"context": "x", "schema": {"unrelated": 1}}"#,
    );
    assert_eq!(code, 400, "body was {value}");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("entities"),
        "error should name the missing keys, got {value}"
    );

    // Blank context is a 400 rather than a model call on whitespace.
    let (code, _) = post(
        server.addr,
        "/v1/jev/boundary",
        r#"{"context": "   ", "schema": {"entities": ["person"]}}"#,
    );
    assert_eq!(code, 400);

    // Malformed JSON.
    let (code, _) = post(server.addr, "/v1/jev/boundary", "{not json");
    assert_eq!(code, 400);

    // The JEV score route is deliberately not mounted on this backend: the
    // boundary head returns spans, which `JevResult` cannot hold, so a
    // JEV-shaped body must 404 rather than half-work.
    let out = Command::new("curl")
        .arg("-s")
        .arg("-m")
        .arg("20")
        .arg("-o")
        .arg("/dev/null")
        .arg("-w")
        .arg("%{http_code}")
        .arg("-X")
        .arg("POST")
        .arg(format!("http://{}/v1/jev/score", server.addr))
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("-d")
        .arg(r#"{"context": "x", "questions": [{"text": "q", "options": ["a"]}]}"#)
        .output()
        .expect("curl");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "404",
        "/v1/jev/score should not exist on the boundary backend"
    );
}
