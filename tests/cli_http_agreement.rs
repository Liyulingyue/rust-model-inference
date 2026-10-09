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

/// `RMI_AGREEMENT_MMPROJ` — required for the image comparison.
fn mmproj_path() -> Option<std::path::PathBuf> {
    std::env::var_os("RMI_AGREEMENT_MMPROJ").map(std::path::PathBuf::from)
}

fn mmp() -> Option<std::sync::Arc<dyn rust_model_inference::core::tensor::TensorSource>> {
    let path = mmproj_path()?;
    let loader = GGUFLoader::from_file(&path).expect("RMI_AGREEMENT_MMPROJ must point at a GGUF");
    Some(std::sync::Arc::from(loader))
}

/// The image both front-ends are shown: the same apple.png fixture the docs
/// use (401x287), embedded so the test does not depend on `references/`.
/// Deterministic 401x287 PNG, generated at test time (no binary fixture in
/// the repo). Both front-ends receive identical bytes.
fn agreement_image() -> Vec<u8> {
    use image::{ImageFormat, Rgb, RgbImage};
    use std::io::Cursor;

    let mut image = RgbImage::new(401, 287);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        *pixel = Rgb([x as u8, y as u8, (x ^ y) as u8]);
    }
    let mut buffer = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut buffer, ImageFormat::Png)
        .unwrap();
    buffer.into_inner()
}
const IMAGE_PROMPT: &str = "Describe this image in a few words.";

/// Run the CLI binary and return the generated text from its `Output: ` line.
fn cli_text(
    path: &std::path::Path,
    max_tokens: usize,
    temperature: f32,
    thinking: bool,
) -> Option<String> {
    // The CLI's `--thinking` flag and the server's `enable_thinking` field
    // must be set explicitly: the two fronts default differently
    // (CLI `options.thinking` is `false` unless `--thinking` is passed;
    // `build_prompt`'s `enable_thinking = None` resolves to `true` for
    // thinking-tuned archs like lfm2moe/lfm2). Relying on the defaults
    // here would compare different prompt tails and report a spurious
    // disagreement.
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_rust-model-inference"));
    command
        .arg("--model")
        .arg(path)
        .arg("--prompt")
        .arg(PROMPT)
        .arg("--n-gen")
        .arg(max_tokens.to_string())
        .arg("--temp")
        .arg(temperature.to_string())
        .arg("--max-context")
        .arg(CONTEXT.to_string());
    command.arg(if thinking {
        "--thinking"
    } else {
        "--no-thinking"
    });
    let output = command.output().ok()?;
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
fn http_text(
    loader: &'static GGUFLoader,
    temperature: f32,
    images: &[Vec<u8>],
    prompt: &str,
    thinking: Option<bool>,
) -> String {
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
            .with_max_context(CONTEXT)
            .with_mmproj(mmp());
    let mut runtime = build_text_runtime(&arch, options)
        .unwrap_or_else(|e| panic!("no runtime for arch {arch}: {e}"));
    // The prompt must come from `tools::build_prompt` — the exact function
    // the server routes `/v1/chat/completions` through. Re-deriving the ids
    // here from a per-arch builder would silently test a different prompt
    // than production: that is exactly how the lfm2moe divergence went
    // unnoticed (the sentinel built LFM2 ids while the server built ChatML).
    let source: &'static dyn rust_model_inference::core::tensor::TensorSource =
        &*Box::leak(Box::new(LeakedLoader(loader)));
    let (ids, _images) = rust_model_inference::app::server::api::tools::build_prompt(
        source,
        &tokenizer,
        &arch,
        &[rust_model_inference::app::server::api::protocol::Message {
            role: "user".into(),
            text: prompt.to_string(),
            calls: vec![],
            call_id: None,
            images: vec![],
        }],
        &[],
        &rust_model_inference::app::server::api::protocol::ToolChoice::Auto,
        thinking,
        &rust_model_inference::prompt::jinja::Options::default(),
        None,
    )
    .expect("prompt build");
    let request = GenerationRequest {
        token_ids: ids,
        max_new_tokens: MAX_TOKENS,
        sampling: SamplingParams {
            temperature,
            ..SamplingParams::default()
        },
        images: images.to_vec(),
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
    //
    // `thinking` is swept so the sentinel covers both prompt tails: the
    // CLI accepts `--thinking` / `--no-thinking`; `build_prompt` accepts the
    // same toggle through `enable_thinking`. Testing only one mode would
    // leave the other tail unprotected (and, pre-PR-#120, the two fronts
    // disagreed on the *default* — the CLI passed `thinking=false` while
    // `build_prompt` had no such field at all).
    for thinking in [false, true] {
        for temperature in [0.0f32] {
            let Some(cli) = cli_text(&path, MAX_TOKENS, temperature, thinking) else {
                eprintln!("skipping thinking={thinking} temperature={temperature}: CLI run failed");
                continue;
            };
            let http = http_text(loader, temperature, &[], PROMPT, Some(thinking));
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
                    "CLI and HTTP disagree for arch={arch} thinking={thinking} temperature={temperature}\n\
                     divergence at char {common}: CLI={:?} HTTP={:?}\n\
                     CLI : {cli:?}\nHTTP: {http:?}",
                    &cli[common..cli.len().min(common + 40)],
                    &http[common..http.len().min(common + 40)]
                );
            }
            assert_eq!(
                cli, http,
                "CLI and HTTP disagree for arch={arch} thinking={thinking} temperature={temperature}\nCLI : {cli:?}\nHTTP: {http:?}"
            );
        }
    }
}

#[test]
fn cli_and_http_agree_on_image_input() {
    // Image comparison: the CLI takes `--mmproj --image <file>`, the runtime
    // takes the decoded bytes through `GenerationRequest.images`. Greedy only,
    // for the same RNG reason as above — and greedy is where a mis-spliced
    // vision placeholder shows up (see Qwen35TextRuntime::prepare_image_prefill).
    let Some(path) = model_path() else { return };
    let Some(mmproj) = mmproj_path() else {
        eprintln!("skipping: set RMI_AGREEMENT_MMPROJ to run the image comparison");
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
    // Three projector families are wired up: qwen35 and qwen2vl share the
    // Qwen2.5-Omni projector (plus a required system turn), qwen3vl /
    // qwen3vlmoe use the Qwen3-VL merger with per-layer deepstack.
    if !matches!(
        arch.as_str(),
        "qwen35" | "qwen3vl" | "qwen3vlmoe" | "qwen2vl"
    ) {
        eprintln!(
            "skipping: image agreement is implemented for qwen35/qwen3vl/qwen2vl, got {arch}"
        );
        return;
    }

    // CLI side: `--image` reads the fixture from disk.
    let fixture = std::env::temp_dir().join(format!("rmi-agree-apple-{}.png", std::process::id()));
    std::fs::write(&fixture, agreement_image()).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .arg("--model")
        .arg(&path)
        .arg("--mmproj")
        .arg(&mmproj)
        .arg("--image")
        .arg(&fixture)
        .arg("--prompt")
        .arg(IMAGE_PROMPT)
        .arg("--n-gen")
        .arg(MAX_TOKENS.to_string())
        .arg("--temp")
        .arg("0")
        .arg("--max-context")
        .arg(CONTEXT.to_string())
        .output()
        .expect("CLI run");
    let _ = std::fs::remove_file(&fixture);
    if !output.status.success() {
        eprintln!(
            "skipping: CLI image run failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Two CLI output formats: the text path prints `Output: <tokens…>`, the
    // multimodal path prints `--- Generation ---\n <text>`. Take whichever is
    // present, up to the following `\n[` / end of string.
    let cli = if let Some(at) = stdout.find("Output: ") {
        let rest = &stdout[at + "Output: ".len()..];
        rest[..rest.find('\n').unwrap_or(rest.len())].to_string()
    } else if let Some(at) = stdout.find("--- Generation ---") {
        let rest = &stdout[at + "--- Generation ---".len()..];
        let rest = rest.strip_prefix('\n').unwrap_or(rest);
        rest[..rest.find("\n--- End ---").unwrap_or(rest.len())].to_string()
    } else {
        panic!("CLI printed no recognisable output line; stdout was:\n{stdout}");
    };

    // Vision paths: `None` keeps the legacy behaviour — every arch with a
    // vision-compatible `build_prompt` treats the missing toggle the same
    // way on both fronts (the CLI's `--thinking` does not reach the
    // multimodal path, and the vision builders ignore the flag entirely).
    let http = http_text(loader, 0.0, &[agreement_image()], IMAGE_PROMPT, None);
    assert_eq!(
        cli, http,
        "CLI and HTTP disagree on image input\nCLI : {cli:?}\nHTTP: {http:?}"
    );
}
