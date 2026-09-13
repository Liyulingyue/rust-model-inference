//! Nemotron-3 Nano (Hybrid Mamba-Transformer).
//!
//! The first cut only implements the attention branch + FFN; the Mamba2
//! SSM branch weights are loaded but contribute zero until a parity test
//! against the pinned llama.cpp commit exists. See
//! `docs/REFERENCE_IMPLEMENTATIONS.md`.

pub mod trunk;
