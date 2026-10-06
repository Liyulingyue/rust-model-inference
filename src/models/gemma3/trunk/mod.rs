//! Gemma3 decoder-only trunk (BitNet b1.58 W1.58A8 variant).
//!
//! Used by `microsoft/bitnet-embedding-270m` (`general.architecture
//! = "gemma3"`, `general.file_type = 40`). Per the oracle
//! `microsoft/BitNet/src/llama-bitnet-b1.58.cpp` plus the GGUF
//! tensor inventory of `bitnet-embeddings-270m-bf16-i2_s.gguf`,
//! this trunk differs from the qwen3 trunk in three concrete ways:
//!
//! 1. **4-norm sandwich per layer**: `attn_norm` -> attn ->
//!    `post_attention_norm` -> `ffn_norm` -> ffn -> `post_ffw_norm`.
//!    Qwen3 only has pre-norm (`attn_norm`, `ffn_norm`).
//! 2. **QK-norm** on q and k: `attn_q_norm`, `attn_k_norm` (per-head
//!    RMSNorm applied **before** RoPE). Qwen3 has the same.
//! 3. **Different projection shapes**: `n_embd=640`, `n_head=4`,
//!    `n_head_kv=1`, `head_dim=256`, `n_ff=2048` (vs Qwen3-0.6B's
//!    1024/16/8/128/3072).
//!
//! Everything else — RoPE, causal attention, GQA repeat,
//! `silu(gate) * up` FFN, BitLinear W1.58A8 forward — is shared
//! with the qwen3 trunk's pattern.
//!
//! # Pooling convention
//!
//! `gemma3.pooling_type = 1` with the BitNet flag means last-token
//! pooling. `compute_embedding` enforces this (the engine never
//! routes `arch=gemma3` through the qwen3-Embedding mean-pool path).

pub mod config;
pub mod forward;
pub mod weights;

pub use config::{Gemma3Config, Gemma3Rope};
pub use forward::{run_shared_inference, text_encode};
pub use weights::{
    get_f32_tensor, load_layers, load_layers_static, static_weight, BitLinearSlot,
    BitLinearWeights, Gemma3LayerWeights, Gemma3Model, Weight,
};
