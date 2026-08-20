//! File-format domain modules.
//!
//! - [`ggufrs`]: `.ggufrs` multi-component container format (read/write/validate).
//! - [`load_plan`]: heterogeneous-device load planning (NUMA / tensor split).
//!
//! Phase 5B split from the root-level `src/ggufrs.rs` (4344 lines) and
//! `src/load_plan.rs` (1226 lines). See `docs/REFACTOR_PLAN.md`.
//!
//! Dependency rules (Phase 5B):
//! - `format/*` depends on `core/*` (GGUF value types, tensor source, model
//!   config derivation).
//! - `format/*` must NOT depend on `models/*` or `app/*`.

pub mod ggufrs;
pub mod load_plan;
