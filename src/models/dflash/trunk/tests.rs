//! DFlash / DSpark draft tests.
//!
//! The real-file tests are guarded by environment variables, following the
//! repo's oracle-test convention (`RMI_QWEN35_MODEL` and friends):
//!
//! | variable | meaning |
//! |---|---|
//! | `RMI_DFLASH_SIDECAR` | path to a `dflash` sidecar GGUF |
//! | `RMI_DFLASH_TARGET` | path to the target model GGUF (its `token_embd` is shared) |

use super::session::{DflashSession, DsparkDraft};
use super::weights::load_weights;
use crate::core::loader::GGUFLoader;
use crate::core::thread_pool::ComputePool;

fn sidecar_path() -> String {
    std::env::var("RMI_DFLASH_SIDECAR").expect("RMI_DFLASH_SIDECAR must be set")
}

fn target_path() -> String {
    std::env::var("RMI_DFLASH_TARGET").expect("RMI_DFLASH_TARGET must be set")
}

#[test]
#[ignore = "requires RMI_DFLASH_SIDECAR and RMI_DFLASH_TARGET"]
fn sidecar_loads_with_expected_shapes() {
    let loader = GGUFLoader::from_file(sidecar_path()).expect("sidecar GGUF must parse");
    let weights = load_weights(&loader).expect("sidecar weights must load");

    let config = &weights.config;
    // Qwen3.5-2B-DSpark, as exported by rasyosef's SpecForge pipeline.
    assert_eq!(config.n_layer, 5);
    assert_eq!(config.n_embd, 2048);
    assert_eq!(config.n_ff, 6144);
    assert_eq!(config.n_head, 16);
    assert_eq!(config.n_head_kv, 8);
    assert_eq!(config.n_embd_head_k, 128);
    assert_eq!(config.block_size, 8);
    assert_eq!(config.target_layers, vec![1, 6, 11, 16, 21]);
    assert!(config.sample_from_anchor);
    assert!(config.has_confidence_head);
    assert!(config.is_sliding());
    assert_eq!(config.sliding_window, 2048);
    assert_eq!(config.mask_token, 248077);
    // The floating-point hyperparameters: read as floats, not as the raw
    // bits of their truncated integers.
    assert!(
        (config.norm_eps - 1e-6).abs() < 1e-9,
        "norm_eps = {}",
        config.norm_eps
    );
    assert_eq!(config.rope_freq_base, 10_000_000.0);
    assert_eq!(config.rope_sections, [64, 0, 0, 0]);

    // The encoder fuses five layers of the target's 2048-wide hidden states.
    assert_eq!(weights.n_embd_enc, 5 * 2048);

    // The DSpark heads.
    assert!(weights.is_dspark());
    assert_eq!(config.markov_rank, 256);
    assert_eq!(config.n_vocab, 248320);
    assert_eq!(config.n_vocab_draft, 50000);
    let markov_w1 = weights.markov_w1.as_ref().expect("markov_w1");
    assert_eq!(markov_w1.rank, 256);
    assert_eq!(markov_w1.rows, 248320);
    assert!(weights.markov_w2.is_some());
    assert!(weights.conf_proj.is_some());
    let d2t = weights.d2t.as_ref().expect("d2t map");
    assert_eq!(d2t.len(), 50000);
}

#[test]
#[ignore = "requires RMI_DFLASH_SIDECAR and RMI_DFLASH_TARGET"]
fn draft_block_runs_after_an_injection() {
    let sidecar = GGUFLoader::from_file(sidecar_path()).expect("sidecar GGUF must parse");
    let target = GGUFLoader::from_file(target_path()).expect("target GGUF must parse");

    let weights = load_weights(&sidecar).expect("sidecar weights must load");
    let token_embd = super::weights::load_weight(&target, "token_embd.weight")
        .expect("the target must carry token_embd.weight");
    assert_eq!(token_embd.n_out, weights.config.n_vocab);

    let pool = ComputePool::new(4);
    let capacity = 512usize;
    // The scratchpad is sized for the larger of the injection chunk and the
    // block; here the injection (12 rows) dominates the block (8).
    let mut session = DflashSession::new(weights, token_embd, 4, capacity, 16);

    // A fake encoder injection: deterministic pseudo-random values stand in
    // for the target's hidden states. All-zero rows would be a degenerate
    // input (every rms-norm row zero); parity against llama.cpp is the oracle
    // tests' job.
    let rows = 12usize;
    let n_embd_enc = session.n_embd_enc();
    let mut lcg = 0x5eedu32;
    let mut hidden = vec![0.0f32; rows * n_embd_enc];
    for value in hidden.iter_mut() {
        lcg = lcg.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        *value = ((lcg >> 8) as f32 / 8_388_608.0 - 0.5) * 0.04;
    }
    let positions: Vec<usize> = (0..rows).collect();
    session
        .inject_target_features(&hidden, &positions, &pool)
        .expect("encoder pass must run");

    let anchor = 1234u32;
    let draft: DsparkDraft = session
        .draft_block(anchor, rows - 1, 8, 0.0, &pool)
        .expect("decoder pass must run");
    assert!(!draft.tokens.is_empty());
    assert_eq!(draft.tokens.len(), draft.confidences.len());
    for (id, confidence) in draft.tokens.iter().zip(draft.confidences.iter()) {
        assert!(*id < session.n_vocab() as u32, "token id out of range");
        assert!(
            (0.0..=1.0).contains(confidence),
            "confidence {confidence} out of range"
        );
    }
    // The block's own positions were written, on top of the injected ones.
    assert!(session.kv_n_filled() >= rows - 1 + 8);
}
