//! The `lfm2` and `lfm25` trunks must agree.
//!
//! `src/app/text/generation.rs` picks between them by asking whether
//! `general.basename` contains `"2.5"`. That key is a *provenance* field, not an
//! architecture field: Liquid's QAD pipeline writes it, llama.cpp's
//! `convert_hf_to_gguf.py` does not. So the very same LFM2.5 checkpoint takes
//! the `lfm25` trunk when it was quantized by QAD and the `lfm2` trunk when it
//! was quantized by anything else — same weights, different arithmetic, and a
//! different built-in chat template on top (the CLI prints "N tokens" on one
//! path and "N items" on the other).
//!
//! llama.cpp has no `lfm25` architecture at all: `llama-arch.cpp` defines only
//! `LLM_ARCH_LFM2` and `LLM_ARCH_LFM2MOE`, and both dense generations share the
//! single `src/models/lfm2.cpp`. There is nothing for the router to switch on.
//!
//! This test pins the two trunks to each other on a real GGUF so the routing
//! heuristic cannot silently split the arithmetic. It is skipped unless
//! `RMI_LFM2_DENSE_MODEL` points at a real file.

use std::path::PathBuf;
use std::sync::Arc;

use rust_model_inference::core::scratchpad::{KvCache, KvFormat};
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::lfm2::Lfm2Session;
use rust_model_inference::models::lfm25::Lfm25Session;

const DEFAULT_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/LFM2.5-230M-GGUF/LFM2.5-230M-Q4_K_M.gguf"
);

const PROMPT: &str = "What is 2+2? Answer briefly.";

/// `general.architecture = "lfm2"`, 14 layers, n_embd 1024, rope base 1e6,
/// Q4_K_M. From the pinned scalar oracle (llama.cpp 749f688), prompt
/// `What is 2+2? Answer briefly.` and `-n 24 --temp 0 -c 512`, run with
/// `RMI_PARITY_TRACE` pointing at a manifest file:
///
///   LFM2.5-230M-Q4_K_M.gguf
///     prompt_ids    [1, 3493, 856, 730, 527, 520, 527, 540, 42275, 21923, 523]
///     generated_ids [509, 527, 520, 527, 30322, 730, 529, 523, 7, 2, 1, 2, ...]
///
/// F16 and Q8_0 of the same checkpoint produce the identical prefix, so the
/// prefix is a property of the model rather than of the quantization.
/// Token 8 is EOS (`tokenizer.ggml.eos_token_id = 7`).
const ORACLE_PROMPT_IDS: &[u32] = &[1, 3493, 856, 730, 527, 520, 527, 540, 42275, 21923, 523];
const ORACLE_GENERATED_IDS: &[u32] = &[509, 527, 520, 527, 30322, 730, 529, 523, 7];

/// Oracle ids for a different checkpoint, as comma-separated lists. Lets the
/// same binary sweep every dense GGUF in `models/` after the oracle has been
/// run against each one — see `tools/oracle/lfm2moe/README.md` for the oracle
/// invocation.
fn env_ids(key: &str) -> Option<Vec<u32>> {
    std::env::var(key).ok().map(|raw| {
        raw.split(',')
            .filter(|part| !part.trim().is_empty())
            .map(|part| part.trim().parse().expect("id list must be integers"))
            .collect()
    })
}

/// The expected prompt / greedy ids for the model under test: the pinned
/// 230M baseline unless the environment overrides it.
fn expected_prompt_ids() -> Vec<u32> {
    env_ids("RMI_ORACLE_PROMPT_IDS").unwrap_or_else(|| ORACLE_PROMPT_IDS.to_vec())
}

fn expected_generated_ids() -> Vec<u32> {
    env_ids("RMI_ORACLE_GENERATED_IDS").unwrap_or_else(|| ORACLE_GENERATED_IDS.to_vec())
}

fn model_path() -> Option<PathBuf> {
    let path = std::env::var("RMI_LFM2_DENSE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_MODEL));
    if path.exists() {
        Some(path)
    } else {
        eprintln!("skipping: {} not present", path.display());
        None
    }
}

fn prompt_ids(source: &Arc<dyn TensorSource>) -> Vec<u32> {
    let tokenizer =
        rust_model_inference::core::tokenizer::BPETokenizer::from_gguf_metadata(|key| {
            source.metadata(key).cloned()
        })
        .expect("tokenizer");
    let mut ids = tokenizer.encode(
        PROMPT,
        rust_model_inference::core::tokenizer::EncodeOptions {
            // The `lfm2` pre-tokenizer does not prepend BOS on `add_special`;
            // the oracle harness adds `bos_token_id` by hand, so mirror that.
            add_special: false,
            parse_special: true,
        },
    );
    ids.insert(0, tokenizer.bos_id().expect("bos_token_id"));
    assert_eq!(
        ids,
        expected_prompt_ids(),
        "tokenizer drifted from the oracle's prompt; the greedy comparison below \
         would be meaningless"
    );
    ids
}

fn argmax(logits: &[f32]) -> usize {
    top_two(logits).0
}

/// The winning token plus its margin over the runner-up.
///
/// A divergence where this margin is at the f32 epsilon is a coin flip on
/// rounding noise, not an arithmetic bug — greedy decoding has no defined
/// behaviour there. A divergence with a wide margin means a real difference.
fn top_two(logits: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut second = usize::MAX;
    for (index, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            second = best;
            best = index;
        } else if second == usize::MAX || *value > logits[second] {
            second = index;
        }
    }
    let margin = if second == usize::MAX {
        f32::INFINITY
    } else {
        logits[best] - logits[second]
    };
    (best, margin)
}

#[test]
fn dense_trunks_produce_identical_logits() {
    let Some(path) = model_path() else { return };
    let loader =
        rust_model_inference::core::loader::GGUFLoader::from_file(&path).expect("open GGUF");
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let ids = prompt_ids(&source);
    eprintln!("prompt ids: {:?}", ids);

    let mut legacy =
        Lfm2Session::from_source(source.as_ref(), 1, KvFormat::F32, 512).expect("lfm2 session");
    let mut current =
        Lfm25Session::from_source(source.as_ref(), 1, KvFormat::F32, 512).expect("lfm25 session");

    assert_eq!(
        legacy.config.n_layer, current.config.n_layer,
        "the two configs disagree on layer count; they read the same metadata keys"
    );
    assert_eq!(
        legacy.config.n_embd, current.config.n_embd,
        "the two configs disagree on embedding width"
    );

    // One thread on both sides: the oracle harness pins n_threads = 1 and some
    // kernels split their reduction order per thread.
    let legacy_logits = legacy
        .forward_logits_chunked(&ids, 1)
        .expect("lfm2 forward");
    let current_logits = current
        .forward_logits_chunked(&ids, 1)
        .expect("lfm25 forward");

    assert_eq!(
        legacy_logits.len(),
        current_logits.len(),
        "vocabulary size differs between the two trunks"
    );
    assert_eq!(
        legacy_logits.len(),
        legacy.vocab,
        "the trunk should project one logit per vocabulary entry"
    );

    let legacy_top = argmax(&legacy_logits);
    let current_top = argmax(&current_logits);
    eprintln!(
        "lfm2  argmax = {legacy_top} (logit {})",
        legacy_logits[legacy_top]
    );
    eprintln!(
        "lfm25 argmax = {current_top} (logit {})",
        current_logits[current_top]
    );

    let mut mismatches = 0usize;
    let mut first_bad: Option<(usize, f32, f32)> = None;
    for (index, (lhs, rhs)) in legacy_logits.iter().zip(current_logits.iter()).enumerate() {
        if lhs.to_bits() != rhs.to_bits() {
            mismatches += 1;
            if first_bad.is_none() {
                first_bad = Some((index, *lhs, *rhs));
            }
        }
    }

    if mismatches != 0 {
        let (index, lhs, rhs) = first_bad.unwrap();
        panic!(
            "dense trunks disagree on {mismatches}/{} logits; first at index {index}: \
             lfm2 = {lhs:?} (0x{:08x}), lfm25 = {rhs:?} (0x{:08x})",
            legacy_logits.len(),
            lhs.to_bits(),
            rhs.to_bits(),
        );
    }
}

/// The session must honour the KV format it was constructed with.
///
/// `forward_logits_chunked` used to hard-code `KvFormat::F16` and the struct
/// did not even store the requested format, so a session built with `F32`
/// silently computed in F16. That is invisible in ordinary use -- greedy output
/// still looks fine -- but it manufactures phantom divergences against the
/// oracle, which pins `cache_type_k = cache_type_v = GGML_TYPE_F32`.
#[test]
fn session_honours_its_kv_format() {
    let Some(path) = model_path() else { return };
    let loader =
        rust_model_inference::core::loader::GGUFLoader::from_file(&path).expect("open GGUF");
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let ids = prompt_ids(&source);

    for (format, expect_f32_cache) in [(KvFormat::F16, false), (KvFormat::F32, true)] {
        let mut session =
            Lfm2Session::from_source(source.as_ref(), 1, format, 512).expect("lfm2 session");
        assert_eq!(
            session.kv_format, format,
            "the session must remember the format the caller asked for"
        );
        assert_eq!(
            matches!(session.kv_cache, KvCache::F32(_)),
            expect_f32_cache,
            "{format:?}: the session's own cache does not match the requested format"
        );
        let logits = session
            .forward_logits_chunked(&ids, 1)
            .expect("lfm2 forward");
        assert_eq!(logits.len(), session.vocab);
    }

    let f16 = Lfm2Session::from_source(source.as_ref(), 1, KvFormat::F16, 512)
        .expect("lfm2 session")
        .forward_logits_chunked(&ids, 1)
        .expect("lfm2 forward");
    let f32 = Lfm2Session::from_source(source.as_ref(), 1, KvFormat::F32, 512)
        .expect("lfm2 session")
        .forward_logits_chunked(&ids, 1)
        .expect("lfm2 forward");
    let differing = f16
        .iter()
        .zip(f32.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        differing > 0,
        "F16 and F32 KV produced bit-identical logits, so the format never \
         reached the forward at all"
    );
    eprintln!("F16 vs F32 KV: {differing}/{} logits differ", f16.len());
}

/// Drive the free-function path directly so the KV format can be chosen.
///
/// `Lfm2Session::forward_logits_chunked` allocates the KV cache inside the
/// free function, so it needs the format handed to it explicitly rather than
/// inferred. This predates the `kv_format` field and is what the oracle
/// comparison has always wanted.
fn greedy_logits(source: &Arc<dyn TensorSource>, ids: &[u32]) -> Vec<f32> {
    rust_model_inference::models::lfm2::run_forward_logits_lfm2_with_batch(
        source.as_ref(),
        ids,
        1,
        KvFormat::F32,
        512,
        1,
    )
    .expect("lfm2 forward")
    .0
}

/// Greedy continuation must reproduce the oracle's token ids.
///
/// Re-prefills the whole prefix on each step instead of stepping a live
/// session, so this covers the *prefill* arithmetic end to end. The CLI's
/// incremental decode path (where the LFM2-MoE shortconv state bug lived) is
/// not exercised here.
///
/// # Coverage
///
/// Driving `run_forward_logits_lfm2_with_batch` directly with `KvFormat::F32`
/// (see `greedy_logits`) and `RMI_SCALAR=1` on both sides, every dense LFM2
/// GGUF in `models/` reproduces the oracle's greedy ids:
///
/// ```text
/// LFM2.5-230M           QAD-Q4_0 Q4_K_M Q5_K_M Q6_K Q8_0 F16 BF16   match
/// LFM2.5-1.2B-Instruct  QAD-Q4_0 Q4_K_M Q6_K Q8_0                     match
/// LFM2.5-2.6B           QAD-Q4_0 Q4_K_M Q6_K Q8_0                     match
/// ```
///
/// and on `LFM2.5-2.6B-Q4_K_M` the per-layer trace agrees bitwise across all
/// 35 steps: 75 240 checkpoints, zero mismatches, `result_output` included.
///
/// An earlier run of this test reported four "divergences". All four were
/// harness bugs rather than model bugs, and each had to be found the hard way:
///
/// 1. The session hard-coded `KvFormat::F16`, so the comparison ran in F16 KV
///    against an F32 oracle. `session_honours_its_kv_format` now pins it.
/// 2. The sweep chose the test binary with `sorted(glob)[-1]`. Cargo keeps one
///    binary per feature combination and the hashes sort arbitrarily, so this
///    silently re-ran a stale build -- whose output happened to differ.
/// 3. The compare script treated "the other side has no record for this key"
///    as agreement. `attn_scores` / `attn_probs` live in the F32 branch of the
///    attention loop and were simply absent, which read as "matches" and sent
///    the hunt after a phantom bug in the attention output assembly.
///
/// The test stays `#[ignore]`d so the default suite needs no real GGUF, but it
/// is the harness that produced the coverage table above.
///
/// Pass `RMI_LFM2_DENSE_MODEL`, `RMI_ORACLE_PROMPT_IDS` and
/// `RMI_ORACLE_GENERATED_IDS` to point it at a different checkpoint.
#[test]
#[ignore = "requires a real GGUF plus oracle ids from tools/oracle/lfm2moe/"]
fn greedy_matches_the_pinned_oracle() {
    let Some(path) = model_path() else { return };
    let loader =
        rust_model_inference::core::loader::GGUFLoader::from_file(&path).expect("open GGUF");
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let mut ids = prompt_ids(&source);

    let expected = expected_generated_ids();
    let mut generated = Vec::new();
    for (step, expected_id) in expected.iter().enumerate() {
        let logits = greedy_logits(&source, &ids);
        let (next, margin) = top_two(&logits);
        if next as u32 != *expected_id {
            let expected_logit = logits[*expected_id as usize];
            let scale = logits[next].abs().max(expected_logit.abs()).max(1.0);
            eprintln!(
                "greedy token {step}: rust {next} (logit {}) vs oracle {expected_id} (logit {expected_logit})",
                logits[next],
            );
            eprintln!(
                "  top-2 margin = {margin:e}  (relative {relative:e})",
                relative = margin / scale
            );
            eprintln!("  context so far: {generated:?}");
            panic!(
                "greedy token {step} diverged from the oracle with a top-2 margin of \
                 {margin:e} ({relative:e} relative) after {step} matching tokens; a margin at \
                 f32 epsilon is rounding noise, anything wider is a real arithmetic difference",
                relative = margin / scale,
            );
        }
        generated.push(next as u32);
        ids.push(next as u32);
    }
    assert_eq!(generated, expected);
    eprintln!("greedy prefix matches the oracle: {generated:?}");
}
