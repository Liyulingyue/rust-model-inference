//! Nemotron-3 Nano trunk
//!
//! Hybrid Mamba-Transformer architecture: every layer runs both an
//! attention branch (qwen3-style RMSNorm + RoPE + GQA) and a Mamba2 SSM
//! branch; their outputs are summed. Per the upstream reference, the SSM
//! output is a residual add that contributes a fraction of the channel
//! width, not a full hidden-state replace — so the residual semantics
//! match Qwen3's. (See `docs/REFERENCE_IMPLEMENTATIONS.md` for the
//! pinned llama.cpp commit once a corresponding model loader lands.)

pub mod config;
pub mod forward;
pub mod weights;

pub use config::NemotronConfig;
pub use forward::run_inference;
pub use weights::NemotronLayerWeights;
