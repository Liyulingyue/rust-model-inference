//! Microsoft Aion Instruct: GQA with per-layer sliding or YaRN attention.
mod model;
pub(crate) mod prompt;
pub(crate) mod runtime;

pub use model::{AionModel, AionSession};
