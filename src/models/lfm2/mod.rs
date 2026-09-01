//! LFM2 (Liquid Foundation Model 2) hybrid architecture support.
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2:
//! the trunk lives in `trunk/`; `lfm2` has no sibling modules.

pub mod trunk;
pub mod vision;

pub use trunk::{get_f32_tensor, load_layers, run_inference, Lfm2Config, Lfm2LayerWeights};
