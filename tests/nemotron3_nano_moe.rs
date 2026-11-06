//! Nemotron-3-Nano 30B-A3B (`nemotron_h_moe`) parity trace against the
//! upstream llama.cpp oracle.
//!
//! The oracle is `tools/oracle/nemotron3_nano/oracle.cpp`, built by
//! `tools/oracle/nemotron3_nano/build_oracle.sh`. It replays the upstream
//! graph and dumps the named tensors listed in its header. This test drives
//! the Rust trunk on the **same raw prompt text**, tokenised the same way
//! (`add_special = true`, `parse_special = true`), so the two traces join on
//! `(name, layer, step)` and can be diffed bit for bit.
//!
//! Why a test rather than the CLI: the CLI wraps the prompt in the
//! architecture's default chat template (ChatML for `nemotron_h_moe`), while
//! the oracle feeds the raw prompt. That is a real behavioural difference
//! between the two entry points, but it is not what this test is checking --
//! the point here is the trunk's arithmetic. Feeding explicit token ids is
//! what the Dory parity test does too (`tests/dory_q5_k_m.rs`), and it keeps
//! the comparison about the model rather than about prompt formatting.
//!
//! Run:
//! ```sh
//! RMI_NEMOTRON3_MODEL=$PWD/models/Nemotron-3-Nano-30B-A3B-GGUF/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf \
//! RMI_PARITY_TRACE=/tmp/opencode/n3_30b_trace/rust.jsonl \
//! RMI_NEMOTRON3_PROMPT_FILE=/tmp/opencode/n3_30b_trace/prompt.txt \
//! RMI_NEMOTRON3_N_PREDICT=4 \
//! cargo test --locked --profile release-fast --features parity-trace \
//!   --test nemotron3_nano_moe -- --ignored --nocapture
//! ```

use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
use rust_model_inference::models::nemotron_h::trunk::{NemotronModel, NemotronScratch};
use rust_model_inference::{GGUFLoader, TensorSource};
use std::sync::Arc;

fn source() -> Option<Arc<dyn TensorSource>> {
    let path = std::env::var_os("RMI_NEMOTRON3_MODEL")?;
    Some(Arc::new(GGUFLoader::from_file(path).expect("open 30B GGUF")))
}

fn reference_prompt() -> String {
    match std::env::var_os("RMI_NEMOTRON3_PROMPT_FILE") {
        Some(path) => std::fs::read_to_string(path).expect("read raw prompt"),
        None => std::env::var("RMI_NEMOTRON3_PROMPT").unwrap_or_else(|_| "Hello".into()),
    }
}

/// The 30B's contract, read off the real GGUF header. Asserted here because
/// every checkpoint shape in the trace depends on it: if a different
/// quantization or a different checkpoint is pointed at, the trace join would
/// silently compare the wrong things.
#[test]
fn nemotron3_nano_30b_contract() {
    let Some(source) = source() else {
        return;
    };
    let config = NemotronModel::from_source(source, 1)
        .expect("load 30B")
        .config;
    assert_eq!(config.architecture, "nemotron_h_moe");
    assert_eq!(config.n_embd, 2688);
    assert_eq!(config.n_layer, 52);
    assert_eq!(config.n_head, 32);
    assert_eq!(config.n_head_kv, 2);
    assert_eq!(config.vocab_size, 131_072);
    let moe = config.moe.expect("30B is a MoE");
    assert_eq!(moe.n_expert, 128);
    assert_eq!(moe.n_expert_used, 6);
    assert_eq!(moe.n_ff_exp, 1856);
    assert_eq!(moe.n_ff_shexp, 3712);
    assert!(moe.expert_weights_norm);
    // 23 MoE layers, 23 Mamba-2 layers, 6 attention layers.
    assert_eq!(config.ffn_layers.len(), 23);
    assert_eq!(config.attention_layers.len(), 6);
}

/// Emit the Rust side of the parity trace.
///
/// `RMI_PARITY_TRACE` must be set, otherwise the checkpoints are no-ops and
/// the compare script has nothing to join -- which would look like a pass.
#[test]
#[ignore = "requires the real 30B GGUF; see tools/oracle/nemotron3_nano/README.md"]
fn nemotron3_nano_30b_reference_trace() {
    let Some(source) = source() else {
        return;
    };
    assert!(
        std::env::var_os("RMI_PARITY_TRACE").is_some(),
        "set RMI_PARITY_TRACE to the manifest path"
    );
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
        .expect("30B tokenizer");
    let prompt = tokenizer.encode(
        &reference_prompt(),
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert!(!prompt.is_empty());
    rust_model_inference::parity_trace::token_ids("prompt_ids", &prompt).unwrap();

    let count: usize = std::env::var("RMI_NEMOTRON3_N_PREDICT")
        .unwrap_or_else(|_| "4".into())
        .parse()
        .unwrap();
    let threads: usize = std::env::var("RMI_NEMOTRON3_THREADS")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    // The oracle runs one token per decode step; the Rust side must match or
    // the `step` axis of the join would be offset.
    let batch: usize = std::env::var("RMI_NEMOTRON3_BATCH")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    assert!((1..=32).contains(&batch));

    let model = NemotronModel::from_source(source, threads).expect("load 30B");
    let mut scratch = NemotronScratch::new(&model.config, prompt.len() + count);
    #[cfg(feature = "parity-trace")]
    let batches: Vec<u32> = prompt
        .chunks(batch)
        .map(|chunk| chunk.len() as u32)
        .chain(std::iter::repeat_n(1, count))
        .collect();
    #[cfg(feature = "parity-trace")]
    rust_model_inference::parity_trace::token_ids("prefill_batches", &batches).unwrap();

    let mut logits = Vec::new();
    for chunk in prompt.chunks(batch) {
        logits = model.prefill(chunk, &mut scratch).expect("prefill");
    }
    let mut generated = Vec::new();
    for _ in 0..count {
        assert!(
            logits.iter().all(|value| value.is_finite()),
            "logits must stay finite; a non-finite value means the trunk diverged"
        );
        let mut best = 0usize;
        for (index, value) in logits.iter().enumerate().skip(1) {
            if *value > logits[best] {
                best = index;
            }
        }
        generated.push(best as u32);
        logits = model.prefill(&[best as u32], &mut scratch).expect("decode");
    }
    rust_model_inference::parity_trace::token_ids("generated_ids", &generated).unwrap();
    println!("prompt_ids={prompt:?}\ngenerated_ids={generated:?}");
    println!("output={:?}", tokenizer.decode(&generated, true));
}

/// The same trace, but for the 4B (`nemotron_h`), which is the control.
///
/// The 30B and the 4B run the *same* Mamba-2 trunk code; they differ only
/// in the SSM weight formats (30B: `ssm_in` is Q5_0 and `ssm_out` is Q4_K;
/// 4B: both are Q8_0) and in the SSM dimensions (`inner_size` 4096 vs 7680,
/// `dt_rank` 64 vs 96). So the 4B isolates the two possible causes of a
/// divergence in the SSM branch:
///
///   * if the 4B's SSM branch matches bit for bit, the trunk arithmetic is
///     right and the 30B's divergence comes from a quantized-weight path
///     (Q5_0 or Q4_K);
///   * if the 4B diverges too, the divergence is in the shared Mamba-2
///     arithmetic and the 30B's weight formats are irrelevant.
///
/// Run:
/// ```sh
/// RMI_NEMOTRON_Q8_MODEL=$PWD/models/Nemotron-3-Nano-4B-GGUF/Nemotron-3-Nano-4B-Q8_0.gguf \
/// RMI_PARITY_TRACE=/tmp/opencode/n3_4b_trace/rust.jsonl \
/// RMI_NEMOTRON3_PROMPT_FILE=/tmp/opencode/n3_4b_trace/prompt.txt \
/// RMI_NEMOTRON3_N_PREDICT=4 \
/// cargo test --locked --profile release-fast --features parity-trace \
///   --test nemotron3_nano_moe -- --ignored --nocapture
/// ```
#[test]
#[ignore = "requires the real 4B GGUF; see tools/oracle/nemotron3_nano/README.md"]
fn nemotron3_nano_4b_reference_trace() {
    let Some(path) = std::env::var_os("RMI_NEMOTRON_Q8_MODEL") else {
        return;
    };
    assert!(
        std::env::var_os("RMI_PARITY_TRACE").is_some(),
        "set RMI_PARITY_TRACE to the manifest path"
    );
    let source: Arc<dyn TensorSource> = Arc::new(GGUFLoader::from_file(path).expect("open 4B GGUF"));
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
        .expect("4B tokenizer");
    let prompt = tokenizer.encode(
        &reference_prompt(),
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert!(!prompt.is_empty());
    rust_model_inference::parity_trace::token_ids("prompt_ids", &prompt).unwrap();

    let count: usize = std::env::var("RMI_NEMOTRON3_N_PREDICT")
        .unwrap_or_else(|_| "4".into())
        .parse()
        .unwrap();
    let threads: usize = std::env::var("RMI_NEMOTRON3_THREADS")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    let batch: usize = std::env::var("RMI_NEMOTRON3_BATCH")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    assert!((1..=32).contains(&batch));

    let model = NemotronModel::from_source(source, threads).expect("load 4B");
    let mut scratch = NemotronScratch::new(&model.config, prompt.len() + count);
    #[cfg(feature = "parity-trace")]
    let batches: Vec<u32> = prompt
        .chunks(batch)
        .map(|chunk| chunk.len() as u32)
        .chain(std::iter::repeat_n(1, count))
        .collect();
    #[cfg(feature = "parity-trace")]
    rust_model_inference::parity_trace::token_ids("prefill_batches", &batches).unwrap();

    let mut logits = Vec::new();
    for chunk in prompt.chunks(batch) {
        logits = model.prefill(chunk, &mut scratch).expect("prefill");
    }
    let mut generated = Vec::new();
    for _ in 0..count {
        assert!(
            logits.iter().all(|value| value.is_finite()),
            "logits must stay finite; a non-finite value means the trunk diverged"
        );
        let mut best = 0usize;
        for (index, value) in logits.iter().enumerate().skip(1) {
            if *value > logits[best] {
                best = index;
            }
        }
        generated.push(best as u32);
        logits = model.prefill(&[best as u32], &mut scratch).expect("decode");
    }
    rust_model_inference::parity_trace::token_ids("generated_ids", &generated).unwrap();
    println!("prompt_ids={prompt:?}\ngenerated_ids={generated:?}");
}
