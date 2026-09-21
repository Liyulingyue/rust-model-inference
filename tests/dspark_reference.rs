//! Real GGUF checks; these fail closed when artifacts or the pinned Oracle are absent.
//! Run bitwise Oracle checks with `--features parity-trace,scalar-parity`.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
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
    trace: PathBuf,
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
        let trace = std::env::temp_dir().join(format!("rmi-dspark-{label}-{port}.jsonl"));
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
            .env("RMI_DSPARK_PARITY_TRACE", &trace)
            .stdin(Stdio::null())
            .stdout(Stdio::from(output.try_clone().unwrap()))
            .stderr(Stdio::from(output))
            .spawn()
            .unwrap();
        let mut oracle = Self {
            process,
            url: format!("http://127.0.0.1:{port}"),
            log,
            trace,
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

#[derive(Debug)]
struct Checkpoint {
    block: usize,
    name: String,
    row: Option<usize>,
    shape: Vec<usize>,
    words: Vec<u32>,
}

fn checkpoint(
    value: &Value,
    block: usize,
    name: &str,
    row: Option<usize>,
    oracle: bool,
) -> Checkpoint {
    let raw_shape = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| usize::try_from(value.as_u64().unwrap()).unwrap())
        .collect::<Vec<_>>();
    let shape = if oracle && matches!(name, "dspark.result_norm" | "dspark.result_output") {
        vec![raw_shape[1], raw_shape[0]]
    } else if oracle {
        vec![raw_shape[0]]
    } else {
        raw_shape
    };
    let binary = value
        .get(if oracle { "binary" } else { "binary_path" })
        .and_then(Value::as_str)
        .unwrap();
    let bytes = std::fs::read(binary).unwrap();
    assert_eq!(bytes.len() % 4, 0, "partial F32 word in {binary}");
    let words = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(shape.iter().product::<usize>(), words.len(), "{name} shape");
    Checkpoint {
        block,
        name: name.into(),
        row,
        shape,
        words,
    }
}

fn rust_checkpoints(path: &Path) -> Vec<Checkpoint> {
    let mut checkpoints = Vec::new();
    let mut block = None;
    let mut next_block = 0;
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let value: Value = serde_json::from_str(line).unwrap();
        let name = value["name"].as_str().unwrap();
        if name == "dspark.result_norm" {
            block = Some(next_block);
            next_block += 1;
        }
        if matches!(
            name,
            "dspark.result_norm"
                | "dspark.result_output"
                | "dspark.markov_bias"
                | "dspark.markov_logits"
                | "dspark.confidence"
        ) {
            checkpoints.push(checkpoint(
                &value,
                block.expect("DSpark row trace before result_norm"),
                name,
                value
                    .get("layer")
                    .and_then(Value::as_u64)
                    .map(|row| row as usize),
                false,
            ));
        }
    }
    checkpoints
}

fn oracle_checkpoints(path: &Path) -> Vec<Checkpoint> {
    let mut checkpoints = Vec::new();
    let mut result_norm = None;
    let mut result_output = None;
    let mut block = None;
    let mut next_block = 0;
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let value: Value = serde_json::from_str(line).unwrap();
        let name = value["name"].as_str().unwrap();
        if name == "result_norm" {
            result_norm = Some(value);
            result_output = None;
            block = None;
            continue;
        }
        if name == "result_output" {
            result_output = Some(value);
            continue;
        }
        let Some((name, row)) = [
            ("dspark_markov_bias-", "dspark.markov_bias"),
            ("dspark_markov_logits-", "dspark.markov_logits"),
            ("dspark_confidence-", "dspark.confidence"),
        ]
        .into_iter()
        .find_map(|(prefix, canonical)| {
            name.strip_prefix(prefix)
                .and_then(|row| row.parse().ok())
                .map(|row| (canonical, row))
        }) else {
            continue;
        };
        let active_block = match block {
            Some(block) => block,
            None => {
                let active_block = next_block;
                next_block += 1;
                checkpoints.push(checkpoint(
                    result_norm
                        .as_ref()
                        .expect("Oracle DSpark trace missing result_norm"),
                    active_block,
                    "dspark.result_norm",
                    None,
                    true,
                ));
                checkpoints.push(checkpoint(
                    result_output
                        .as_ref()
                        .expect("Oracle DSpark trace missing result_output"),
                    active_block,
                    "dspark.result_output",
                    None,
                    true,
                ));
                block = Some(active_block);
                active_block
            }
        };
        checkpoints.push(checkpoint(&value, active_block, name, Some(row), true));
    }
    checkpoints
}

fn assert_checkpoint_parity(rust: &Path, oracle: &Path, label: &str) {
    let rust = rust_checkpoints(rust);
    let oracle = oracle_checkpoints(oracle);
    let rust_blocks = rust.iter().map(|record| record.block).max().unwrap() + 1;
    let oracle_blocks = oracle.iter().map(|record| record.block).max().unwrap() + 1;
    let rows = |records: &[Checkpoint], block| {
        records
            .iter()
            .filter(|record| record.block == block && record.name == "dspark.markov_bias")
            .count()
    };
    let first_rows = rows(&rust, 0);
    let first_oracle = (0..oracle_blocks)
        .find(|&block| rows(&oracle, block) >= first_rows)
        .expect("Oracle DSpark trace has no matching draft block");
    assert_eq!(
        oracle_blocks - first_oracle,
        rust_blocks,
        "{label} logical checkpoint block count"
    );

    for rust_block in 0..rust_blocks {
        let oracle_block = first_oracle + rust_block;
        let rust_rows = rows(&rust, rust_block);
        assert!(
            rows(&oracle, oracle_block) >= rust_rows,
            "{label} Oracle draft block is shorter than Rust block {rust_block}"
        );
        for rust in rust.iter().filter(|record| record.block == rust_block) {
            let oracle = oracle
                .iter()
                .find(|record| {
                    record.block == oracle_block
                        && record.name == rust.name
                        && record.row == rust.row
                })
                .unwrap_or_else(|| {
                    panic!(
                        "{label} missing Oracle checkpoint {} row {:?} in block {rust_block}",
                        rust.name, rust.row
                    )
                });
            if matches!(
                rust.name.as_str(),
                "dspark.result_norm" | "dspark.result_output"
            ) {
                assert_eq!(
                    &oracle.shape[1..],
                    &rust.shape[1..],
                    "{label} checkpoint width"
                );
                assert!(
                    oracle.shape[0] >= rust.shape[0],
                    "{label} Oracle checkpoint rows"
                );
            } else {
                assert_eq!(oracle.shape, rust.shape, "{label} checkpoint shape");
            }
            let oracle_words = &oracle.words[..rust.words.len()];
            if rust.words != oracle_words {
                let index = rust
                    .words
                    .iter()
                    .zip(oracle_words)
                    .position(|(rust, oracle)| rust != oracle)
                    .unwrap();
                panic!(
                    "{label} block {} {} row {:?} index {index}: Rust=0x{:08x} Oracle=0x{:08x}",
                    rust.block, rust.name, rust.row, rust.words[index], oracle.words[index]
                );
            }
        }
    }
}

fn rust(
    target: &str,
    draft: Option<&str>,
    prompt: &str,
    count: &str,
    trace: Option<&Path>,
) -> String {
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
    if target.contains("LFM2.5") || target.contains("lfm") {
        command.arg("--thinking");
    }
    command.env("RUST_DSPARK_TRACE_IDS", "1");
    if let Some(trace) = trace {
        command.env("RMI_PARITY_TRACE", trace).env(
            "RMI_PARITY_FILTER",
            "dspark.result_norm,dspark.result_output,dspark.markov_bias,dspark.markov_logits,dspark.confidence",
        );
    }
    let output = command.output().unwrap();
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
        let rust_trace = std::env::temp_dir().join(format!(
            "rmi-dspark-rust-{label}-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&rust_trace);
        let baseline = rust(&target, None, prompt, count, None);
        let speculative = rust(
            &target,
            Some(&draft),
            prompt,
            count,
            oracle.as_ref().map(|_| rust_trace.as_path()),
        );
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
        assert_checkpoint_parity(&rust_trace, &oracle.trace, label);
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
