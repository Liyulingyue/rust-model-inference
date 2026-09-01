//! LFM2.5 (Liquid Foundation Model 2.5) hybrid architecture support.
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2:
//! the trunk lives in `trunk/`; `lfm25` has no sibling modules.

pub mod trunk;

pub use trunk::{get_f32_tensor, load_layers, run_inference, Lfm25Config, Lfm25LayerWeights};
