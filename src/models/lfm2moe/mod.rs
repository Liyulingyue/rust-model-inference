//! LFM2-8B-A1B (LFM2 MoE hybrid) architecture support.
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2:
//! the trunk lives in `trunk/`; `lfm2moe` has no sibling modules.
//!
//! The architecture is `lfm2` with a Mixture-of-Experts FFN: the first
//! `leading_dense_block_count` blocks use a dense SwiGLU FFN, the rest route
//! each token through `n_expert_used` of `n_expert` experts (sigmoid gating
//! with a selection bias, weights renormalized over the selected experts).

pub mod trunk;

pub use trunk::{get_f32_tensor, load_layers, run_inference, Lfm2MoeConfig, Lfm2MoeLayerWeights};
