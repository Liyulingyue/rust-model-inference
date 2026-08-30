pub mod asr;
mod contract;
pub mod trunk;
pub mod vision;

pub use asr::{Gemma4AudioConfig, Gemma4AudioFeatures, Gemma4AudioModel};
pub use trunk::{Gemma4Config, Gemma4InputRow, Gemma4Model, Gemma4Session};
pub use vision::{Gemma4VisionConfig, Gemma4VisionModel};

// Compatibility for callers that used the pre-organization module name.
pub use asr as audio;

#[cfg(test)]
use contract::gemma4_token_table_digest;
#[cfg(test)]
mod tests;
