//! Mellum 2.x text-generation entry points.
//!
//! Re-exports the trunk so callers use `crate::models::mellum::{...}`.

pub mod trunk;

pub use trunk::run::build_mellum_prompt;
pub use trunk::{
    MellumConfig, MellumLayerWeights, MellumModel, MellumRope, MellumRunner, MellumScratch,
    MellumTextRuntime, YarnRope, ARCH,
};
