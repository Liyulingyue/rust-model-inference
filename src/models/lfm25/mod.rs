//! LFM2.5 dense support — a thin alias over the `lfm2` trunk.
//!
//! # Why this is not its own implementation
//!
//! This used to be a near-copy of [`crate::models::lfm2`]: same config keys,
//! same graph, and a `config.rs` that differed only in the struct name. The two
//! were then routed by `src/app/text/generation.rs` on whether
//! `general.basename` contains `"2.5"` — a *provenance* field, not an
//! architecture one. Liquid's QAD pipeline writes it; llama.cpp's
//! `convert_hf_to_gguf.py` does not. So the same LFM2.5 checkpoint computed via
//! one trunk or the other depending on which quantizer produced the file.
//!
//! llama.cpp never had that choice to make. `llama-arch.cpp` defines only
//! `LLM_ARCH_LFM2` and `LLM_ARCH_LFM2MOE`, every dense generation goes through
//! the single `src/models/lfm2.cpp`, and there is no 2.5 branch anywhere.
//! Measured across the 15 dense LFM2 GGUF files in `models/`, only the QAD
//! tiers carry `general.basename` — the other eight tiers were all silently
//! taking the `lfm2` path.
//!
//! Today both trunks happen to agree bit for bit (see
//! `tests/lfm2_dense_trunk_agreement.rs::dense_trunks_produce_identical_logits`,
//! zero differences across 65 536 logits), so this was fragile rather than
//! wrong. But a second copy of a transformer forward pass is exactly the kind
//! of thing that rots silently the first time one side is edited, and the
//! router had nothing holding the two together except an optional metadata key.
//!
//! So the duplicate is gone: these names are aliases, not a second
//! implementation, and the two can no longer drift.

pub mod trunk;

pub use trunk::{
    get_f32_tensor, load_layers, run_forward_logits_lfm25_with_batch, run_inference,
    Lfm25Config, Lfm25LayerWeights, Lfm25Session,
};
