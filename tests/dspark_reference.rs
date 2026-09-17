//! Real GGUF checks; these fail closed when artifacts or the pinned Oracle are absent.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name} before running this ignored test"))
}

fn hash(path: &str, expected: &str) {
    let mut file = File::open(path).unwrap();
    let mut hasher = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    assert_eq!(
        format!("{:x}", hasher.finalize()),
        expected,
        "artifact SHA256: {path}"
    );
}

struct Oracle {
    process: Child,
    url: String,
    log: PathBuf,
}
impl Drop for Oracle {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

fn post(url: &str, body: Value) -> Value {
    let mut curl = Command::new("curl")
        .args([
            "--fail-with-body",
            "--silent",
            "--show-error",
            "--max-time",
            "60",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
            url,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    curl.stdin
        .take()
        .unwrap()
        .write_all(body.to_string().as_bytes())
        .unwrap();
    let output = curl.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{} {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

impl Oracle {
    fn start(target: &str, draft: &str, label: &str) -> Self {
        let binary = required("RMI_DSPARK_ORACLE");
        let version = Command::new(&binary).arg("--version").output().unwrap();
        let version = format!(
            "{}{}",
            String::from_utf8_lossy(&version.stdout),
            String::from_utf8_lossy(&version.stderr)
        );
        assert!(version.contains("84075273c"), "wrong Oracle: {version}");
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port().to_string();
        drop(listener);
        let log =
            std::env::temp_dir().join(format!("rmi-dspark-{label}-{}.log", std::process::id()));
        let output = File::create(&log).unwrap();
        let process = Command::new(binary)
            .args([
                "--model",
                target,
                "--spec-draft-model",
                draft,
                "--spec-type",
                "draft-dspark",
                "--spec-draft-n-max",
                "7",
                "--spec-draft-p-min",
                "0",
                "--threads",
                "1",
                "--threads-batch",
                "1",
                "--spec-draft-threads",
                "1",
                "--ctx-size",
                "512",
                "--cache-type-k",
                "f32",
                "--cache-type-v",
                "f32",
                "--spec-draft-type-k",
                "f32",
                "--spec-draft-type-v",
                "f32",
                "--no-warmup",
                "--flash-attn",
                "off",
                "--host",
                "127.0.0.1",
                "--port",
                &port,
            ])
            .env("RUST_DSPARK_TRACE_IDS", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::from(output.try_clone().unwrap()))
            .stderr(Stdio::from(output))
            .spawn()
            .unwrap();
        let mut oracle = Self {
            process,
            url: format!("http://127.0.0.1:{port}"),
            log,
        };
        let started = Instant::now();
        loop {
            assert!(
                oracle.process.try_wait().unwrap().is_none(),
                "Oracle stopped: {}",
                std::fs::read_to_string(&oracle.log).unwrap()
            );
            if Command::new("curl")
                .args([
                    "--fail",
                    "--silent",
                    "--max-time",
                    "1",
                    &format!("{}/health", oracle.url),
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .unwrap()
                .success()
            {
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(30),
                "Oracle startup timeout"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
        oracle
    }
}

fn ids(stderr: &str, marker: &str) -> Value {
    let line = stderr
        .lines()
        .find(|line| line.contains(marker))
        .expect("missing token trace");
    let value = line.split_once(marker).unwrap().1.trim();
    serde_json::from_str(
        value
            .strip_prefix("n=")
            .map_or(value, |_| value.split_once("ids=").unwrap().1),
    )
    .unwrap()
}

fn blocks(log: &str) -> Vec<Value> {
    log.lines()
        .filter_map(|line| line.split_once("[DSPARK_BLOCK] "))
        .map(|(_, value)| {
            let value: Value = serde_json::from_str(value).unwrap();
            json!({"draft_ids": value["draft_ids"], "accepted": value["accepted"]})
        })
        .collect()
}

fn rust(target: &str, draft: Option<&str>, prompt: &str, count: &str) -> String {
    let binary = std::env::var("RMI_DSPARK_RUST")
        .unwrap_or_else(|_| env!("CARGO_BIN_EXE_rust-model-inference").into());
    let mut command = Command::new(binary);
    command.args([
        "--model",
        target,
        "--prompt",
        prompt,
        "--max-tokens",
        count,
        "--temp",
        "0",
        "--threads",
        "1",
        "--kv-cache",
        "f32",
        "--prefill-batch-size",
        "1",
    ]);
    if let Some(draft) = draft {
        command.args([
            "--draft-model",
            draft,
            "--spec-draft-n-max",
            "7",
            "--spec-draft-conf-min",
            "0",
        ]);
    }
    let output = command.env("RUST_DSPARK_TRACE_IDS", "1").output().unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(output.status.success(), "Rust CLI failed: {stderr}");
    stderr
}

fn check(label: &str, target_hash: &str, draft_hash: &str, compare_oracle: bool) {
    let target = required(&format!("RMI_{label}_DSPARK_TARGET"));
    let draft = required(&format!("RMI_{label}_DSPARK_DRAFT"));
    hash(&target, target_hash);
    hash(&draft, draft_hash);
    let oracle = compare_oracle.then(|| Oracle::start(&target, &draft, label));
    let prompts = if compare_oracle {
        vec![("Hello", "16")]
    } else {
        vec![("Hello", "16"), ("用一句话说明为什么天空是蓝色的。", "12")]
    };
    let mut accepted_total = 0;
    for (prompt, count) in prompts {
        let baseline = rust(&target, None, prompt, count);
        let speculative = rust(&target, Some(&draft), prompt, count);
        let prompt_ids = ids(&baseline, "[RUST_TOKENS] ");
        assert_eq!(ids(&speculative, "[RUST_TOKENS] "), prompt_ids);
        let generated = ids(&speculative, "[RUST_GENERATED_IDS] ");
        assert_eq!(
            generated,
            ids(&baseline, "[RUST_GENERATED_IDS] "),
            "DSpark changed target tokens: {prompt}"
        );
        let ours = blocks(&speculative);
        assert!(!ours.is_empty(), "DSpark silently unused");
        accepted_total += ours
            .iter()
            .map(|block| block["accepted"].as_u64().unwrap())
            .sum::<u64>();
        eprintln!("{label}: prompt={prompt:?}, generated={generated}, blocks={ours:?}");
        let Some(oracle) = oracle.as_ref() else {
            continue;
        };
        let mut raw_prompt =
            format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        if label == "QWEN3" {
            raw_prompt.push_str("<think>\n\n</think>\n\n");
        }
        let tokenized = post(
            &format!("{}/tokenize", oracle.url),
            json!({
                "content": raw_prompt, "add_special": true, "parse_special": true,
            }),
        );
        assert_eq!(
            prompt_ids, tokenized["tokens"],
            "Oracle prompt tokenization: {prompt}"
        );
        let start_blocks = blocks(&std::fs::read_to_string(&oracle.log).unwrap()).len();
        let response = post(
            &format!("{}/completion", oracle.url),
            json!({
                "prompt": prompt_ids, "n_predict": count.parse::<usize>().unwrap(), "temperature": 0,
                "return_tokens": true, "seed": 1, "cache_prompt": false,
            }),
        );
        let mut reference_ids = response["tokens"].clone();
        let eos = if label == "QWEN3" { 151645 } else { 7 };
        if reference_ids.as_array().unwrap().last() == Some(&json!(eos)) {
            reference_ids.as_array_mut().unwrap().pop(); // Rust callbacks omit the terminal EOS.
        }
        assert_eq!(generated, reference_ids, "Oracle target tokens: {prompt}");
        let reference = blocks(&std::fs::read_to_string(&oracle.log).unwrap());
        assert_eq!(
            ours,
            reference[start_blocks..],
            "Oracle draft/acceptance blocks: {prompt}"
        );
        eprintln!("{label}: prompt={prompt:?}, generated={generated}, blocks={ours:?}");
    }
    assert!(accepted_total > 0, "no accepted draft tokens");
}

#[test]
#[ignore = "requires fixed real GGUF files and an instrumented pinned llama-server"]
fn qwen3_matches_oracle() {
    check(
        "QWEN3",
        "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
        "f81a1877d6db00d1f8476d365d4c252e94ab27fa57bb481dd9bf0079fd276c97",
        true,
    );
}

#[test]
#[ignore = "requires fixed real GGUF files and an instrumented pinned llama-server"]
fn lfm25_matches_oracle() {
    check(
        "LFM25",
        "b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5",
        "5cf9bb2947638dd74a47b486b817f407831c0da420aeebb6973fb66c25af51e4",
        true,
    );
}

#[test]
#[ignore = "requires fixed real GGUF files"]
fn qwen3_preserves_target_tokens() {
    check(
        "QWEN3",
        "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
        "f81a1877d6db00d1f8476d365d4c252e94ab27fa57bb481dd9bf0079fd276c97",
        false,
    );
}

#[test]
#[ignore = "requires fixed real GGUF files"]
fn lfm25_preserves_target_tokens() {
    check(
        "LFM25",
        "b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5",
        "5cf9bb2947638dd74a47b486b817f407831c0da420aeebb6973fb66c25af51e4",
        false,
    );
}
