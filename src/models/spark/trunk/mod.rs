//! Qwen 2.5 model family: Spark 2.5 (Xunfei Spark 2.5)
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2,
//! `trunk/` holds the pure transformer decoder. Spark 2.5 has no sibling
//! modules (no ASR/TTS/vision).

pub mod config;
pub mod forward;
pub mod weights;

pub use config::SparkConfig;
pub use forward::{run_inference, SparkModel, SparkSession};
pub use weights::{load_layers, SparkLayerWeights};
