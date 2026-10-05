//! Gemma3-architecture BitNet b1.58 decoder trunk.
//!
//! Used by `microsoft/bitnet-embedding-270m` (`general.architecture
//! = "gemma3"`, `general.file_type = 40`). Concrete architectural
//! differences vs the Qwen3-arch BitNet trunk
//! ([`crate::models::bitnet::qwen3_arch`]):
//!
//! 1. **4-norm sandwich per layer**: `attn_norm` -> attn ->
//!    `post_attention_norm` -> `ffn_norm` -> ffn -> `post_ffw_norm`.
//!    Qwen3 only has pre-norm.
//! 2. **QK-norm** on q and k (per-head, before RoPE). Both
//!    BitNet trunks have this.
//! 3. **Different projection shapes**: `n_embd=640`, `n_head=4`,
//!    `n_head_kv=1`, `head_dim=256`, `n_ff=2048` (vs Qwen3-0.6B's
//!    1024/16/8/128/3072).
//!
//! # Pooling convention
//!
//! `gemma3.pooling_type = 1` with the BitNet flag means last-token
//! pooling. `compute_embedding` enforces this.

pub mod config;
pub mod embedding;
pub mod forward;
pub mod weights;

pub use config::{build_config, Gemma3Config, Gemma3Rope};
pub use embedding::{compute_embedding, run_embedding, run_embedding_tokens};
pub use forward::{run_shared_inference, text_encode};
pub use weights::{
    get_f32_tensor, load_layers, load_layers_static, static_weight, Gemma3LayerWeights, Gemma3Model,
};
