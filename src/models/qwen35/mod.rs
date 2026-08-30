//! Qwen3.5 (hybrid Mamba SSM + dense attention) inference.
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2:
//! - `trunk/` contains the pure LLM decoder (`Qwen35Model` + `Qwen35Session`).
//! - `vision/` is a sibling — vision encoder for VL inputs.
//!
//! ## Public surface
//!
//! Re-exports [`Qwen35Config`], [`Qwen35Session`], [`Qwen35Model`],
//! [`Qwen35LayerWeights`], [`Qwen35Scratchpad`], [`build_qwen35_positions`]
//! at `models::qwen35::*`.

pub mod trunk;
pub mod vision;

pub use crate::models::qwen35::vision::VisionGrid;
pub use trunk::{
    build_qwen35_positions, Qwen35Config, Qwen35LayerWeights, Qwen35Model, Qwen35Scratchpad,
    Qwen35Session,
};