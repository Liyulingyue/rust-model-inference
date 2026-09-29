//! Xing4.0 (xing4_0) — XingChen-AGI / TeleChat4 sparse MoE model with
//! MLA attention and hyper-connection residuals.
//!
//! Module layout follows `MODEL_ORGANIZATION.md` §2: the trunk lives
//! under `trunk/`, with `config` / `weights` / `forward` split out.

pub mod trunk;

pub use trunk::config::Xing4Config;
