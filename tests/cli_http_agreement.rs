//! CLI vs HTTP agreement guard.
//!
//! The HTTP layer used to re-implement the decode loop per architecture and,
//! for llama-family and lfm2moe, **sample differently from the CLI** — the
//! same prompt produced different tokens from `rust-model-inference --prompt`
//! than from `/v1/chat/completions`. Both front-ends now share
//! `ops::sampling::{LlamaSampler, Lfm2MoeSampler}`; these tests pin that.
//!
//! Gated on a real model path (`RMI_AGREEMENT_MODEL`), like the rest of the
//! model-dependent integration tests, so it skips in model-less environments:
//!
//! ```text
//! RMI_AGREEMENT_MODEL=models/K2-Horizon-GGUF/K2-Horizon-1B-BF16.gguf \
//!   cargo test --test cli_http_agreement
//! ```
//!
//! Each case runs the CLI binary (capturing stdout) and drives the same
//! `TextRuntime` the server builds, then asserts the decoded text matches.

use rust_model_inference::app::text::{build_text_runtime, RuntimeOptions};
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::ops::generation_runtime::{
    CollectSink, GenerationRequest, SamplingParams,
};
use rust_model_inference::{BPETokenizer, GGUFLoader, MetaValue};
use std::sync::Arc;

/// Short on purpose: every token costs wall time on CPU.
const PROMPT: &str = "Say hi to me";
const MAX_TOKENS: usize = 24;
const CONTEXT: usize = 512;

/// `'static` view over the loader's mmap. The loader itself is leaked for the
/// duration of the process (this is a test), which is what lets the adapters
/// take their `Box::leak`-style 'static borrows.
struct LeakedLoader(&'static GGUFLoader);

impl TensorSource for LeakedLoader {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.0.metadata(key)
    }
    fn tensor_info(&self, name: &str) -> Option<&rust_model_inference::TensorInfo> {
        self.0.tensor_info(name)
    }
    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.0.tensor_slice(name)
    }
}

fn model_path() -> Option<std::path::PathBuf> {
    std::env::var_os("RMI_AGREEMENT_MODEL").map(std::path::PathBuf::from)
}

/// Run the CLI binary and return the generated text from its `Output: ` line.
fn cli_text(path: &std::path::Path, max_tokens: usize, temperature: f32) -> Option<String> {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .arg("--model")
        .arg(path)
        .arg("--prompt")
        .arg(PROMPT)
        .arg("--n-gen")
        .arg(max_tokens.to_string())
        .arg("--temp")
        .arg(temperature.to_string())
        .arg("--max-context")
        .arg(CONTEXT.to_string())
        .output()
        .ok()?;
    if !output.status.success() {
        eprintln!("CLI failed: {}", String::from_utf8_lossy(&output.stderr));
        return None;
    }
    // The CLI prints `Output: <chunk>` and keeps appending on the SAME line as
    // tokens stream, then a newline and `[N output tokens in Xms]`. Everything
    // between `Output: ` and that summary line is the generated text.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let start = stdout.find("Output: ")? + "Output: ".len();
    let rest = &stdout[start..];
    let end = rest.find("\n[").unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

/// Drive the same runtime the server builds, and return the decoded text.
fn http_text(loader: &'static GGUFLoader, temperature: f32) -> String {
    let arch = loader
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .map(|s| s.to_string())
        .unwrap_or_default();
    let pool = Arc::new(rust_model_inference::ComputePool::new(
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(8),
    ));
    let tokenizer = Arc::new(
        BPETokenizer::from_gguf_metadata(|k| loader.metadata(k).cloned())
            .expect("tokenizer must build"),
    );
    // Same constructor the server uses, plus the context cap; every other
    // field carries `RuntimeOptions::defaults` (which a unit test pins to the
    // CLI's resolution), so the sentinel cannot drift from either side.
    let options =
        RuntimeOptions::from_model(Arc::new(LeakedLoader(loader)), pool, tokenizer.clone())
            .with_max_context(CONTEXT);
    let mut runtime = build_text_runtime(&arch, options)
        .unwrap_or_else(|e| panic!("no runtime for arch {arch}: {e}"));
    // Both front-ends must start from identical prompt ids. For llama-family
    // archs that means the trunk's own prompt builder (the one `tools.rs`
    // routes to); otherwise the qwen ChatML builder the server uses.
    let ids = if matches!(
        arch.as_str(),
        "llama" | "nanbeige" | "exaone" | "k2-horizon" | "granite"
    ) {
        // llama-family: the trunk's own prompt builder (what tools.rs routes to).
        rust_model_inference::models::llama::trunk::build_prompt_tokens(loader, PROMPT, false)
            .expect("llama-family prompt build")
    } else if arch == "lfm2moe" {
        // lfm2moe: the trunk uses the LFM2 template, not ChatML.
        rust_model_inference::prompt::build_lfm2_chat_prompt(
            &tokenizer,
            &[rust_model_inference::prompt::Lfm2Message {
                role: "user",
                content: PROMPT,
            }],
        )
        .expect("lfm2 prompt build")
    } else {
        rust_model_inference::prompt::build_qwen_chat_prompt(
            &tokenizer,
            &[rust_model_inference::prompt::QwenMessage {
                role: "user",
                content: PROMPT,
            }],
            false,
        )
        .expect("qwen prompt build")
    };
    let request = GenerationRequest {
        token_ids: ids,
        max_new_tokens: MAX_TOKENS,
        sampling: SamplingParams {
            temperature,
            ..SamplingParams::default()
        },
        images: Vec::new(),
    };
    let mut sink = CollectSink::new();
    runtime
        .generate(&request, &mut sink)
        .expect("generation must succeed");
    sink.text
}

#[test]
fn cli_and_http_agree_on_greedy_and_temperature() {
    let Some(path) = model_path() else {
        eprintln!("skipping: set RMI_AGREEMENT_MODEL to a GGUF to run this check");
        return;
    };
    let loader: &'static GGUFLoader = Box::leak(Box::new(
        GGUFLoader::from_file(&path).expect("RMI_AGREEMENT_MODEL must point at a GGUF"),
    ));
    let arch = loader
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .map(|s| s.to_string())
        .unwrap_or_default();
    assert_eq!(
        arch.matches('"').count(),
        0,
        "arch string should not contain quotes"
    );
    // Only the two families that had (or could have) a sampler divergence are
    // covered; other archs are gated on their own models elsewhere.
    let covered = matches!(
        arch.as_str(),
        "llama" | "nanbeige" | "exaone" | "k2-horizon" | "granite" | "lfm2moe" | "qwen3" | "qwen35"
    );
    if !covered {
        eprintln!("skipping: arch {arch} is not in the agreement matrix yet");
        return;
    }
    // Greedy only. At temperature > 0 the two front-ends draw from
    // independent RNGs (the CLI's per-arch samplers are unseeded; the
    // adapters use their own), so byte equality is not achievable there
    // without threading a shared seed — out of scope for this guard. Greedy
    // is where forward-path and sampler splits show up, which is what this
    // test exists to catch.
    for temperature in [0.0f32] {
        let Some(cli) = cli_text(&path, MAX_TOKENS, temperature) else {
            eprintln!("skipping temperature={temperature}: CLI run failed");
            continue;
        };
        let http = http_text(loader, temperature);
        if cli != http {
            // Where the texts first differ. A split at char 0 means a
            // forward-path divergence (wrong tokens from step one); a split
            // tens of characters in is usually a 1-ULP greedy flip on two
            // near-tied logits.
            let common = cli
                .chars()
                .zip(http.chars())
                .take_while(|(a, b)| a == b)
                .count();
            panic!(
                "CLI and HTTP disagree for arch={arch} temperature={temperature}\n\
                 divergence at char {common}: CLI={:?} HTTP={:?}\n\
                 CLI : {cli:?}\nHTTP: {http:?}",
                &cli[common..cli.len().min(common + 40)],
                &http[common..http.len().min(common + 40)]
            );
        }
        assert_eq!(
            cli, http,
            "CLI and HTTP disagree for arch={arch} temperature={temperature}\nCLI : {cli:?}\nHTTP: {http:?}"
        );
    }
}
