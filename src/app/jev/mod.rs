//! JEV (Joint Embedding Verification) decision scoring.
//!
//! Two modes are supported:
//!
//! - **Single** (`run_jev_decision`): one question → softmax over
//!   candidates A..Z. Uses [`JevMode::Choice` | `Binary` | `Score`].
//! - **Grouped** (`run_jev_grouped_decision`): one question with
//!   multiple independent groups → per-group softmax. Uses
//!   [`JevMode::MultiSelect` | `BlockChoice`].
//!
//! Submodules:
//! - [`types`]: shared data types ([`JevMode`], [`JevResult`],
//!   [`JevGroupedResult`], etc.).
//! - [`single`]: single-mode scoring (entry point + 9 per-arch
//!   `JevScorer` implementations).
//! - [`grouped`]: grouped-mode scoring (entry point + 8 per-arch
//!   `JevGroupedScorer` implementations).

pub(crate) mod grouped;
pub(crate) mod single;
pub(crate) mod types;

pub use grouped::run_jev_grouped_decision;
pub use single::run_jev_decision;
pub use types::{
    JevGroupInput, JevGroupResult, JevGroupedOption, JevGroupedQuestionInput, JevGroupedResult,
    JevMode, JevQuestionInput, JevResult,
};