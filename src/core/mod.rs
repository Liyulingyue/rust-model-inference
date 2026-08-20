//! Core domain types: tensor abstraction, GGUF loader, generic model graph.
//!
//! Phase 3 split from the original `src/model.rs` (1395 lines). See
//! `docs/REFACTOR_PLAN.md` for the overall refactor plan.
//!
//! Module layout:
//! - [`tensor`]: GGMLType, MetaValue*, TensorInfo, TensorSource trait
//! - [`loader`]: ByteReader, GGUFLoader, model_config_from_source
//! - [`model`]: QuantizedLinear, ModelGraph (generic Layer container)
//!
//! Dependency rules (Phase 3):
//! - `tensor` depends on nothing else in `core`.
//! - `loader` depends on `tensor` (uses MetaValueType, TensorInfo).
//! - `model` depends on `tensor` (TensorSource) and on `crate::traits`
//!   (Layer, ModelConfig) — Phase 5 will move traits into `core/traits.rs`.

pub mod loader;
pub mod model;
pub mod tensor;
