//! Core domain types: tensor abstraction, GGUF loader, generic model graph.
//!
//! Phase 3 split from the original `src/model.rs` (1395 lines). See
//! `docs/REFACTOR_PLAN.md` for the overall refactor plan.
//!
//! Module layout:
//! - [`tensor`]: GGMLType, MetaValue*, TensorInfo, TensorSource trait
//! - [`loader`]: ByteReader, GGUFLoader, model_config_from_source
//! - [`model`]: QuantizedLinear, ModelGraph (generic Layer container)
//! - [`tokenizer`]: BPETokenizer, EncodeOptions, StreamingDecoder
//! - [`memory`]: BlockAllocator, MemoryArena, PagedKVBlock, KVCacheView
//! - [`thread_pool`]: ComputePool
//! - [`scratchpad`]: ExecutionScratchpad, KvCache (F16/F32)
//! - [`traits`]: Layer trait, ExecContext, ModelConfig
//!
//! Dependency rules (Phase 5A):
//! - `tensor` / `memory` / `thread_pool` / `scratchpad` are foundational
//!   (depend on nothing else in `core`).
//! - `traits` depends on `memory` (uses `KVCacheView`).
//! - `loader` depends on `tensor` (uses MetaValueType, TensorInfo).
//! - `model` depends on `tensor` and `traits`.
//! - `tokenizer` depends on `tensor` (uses MetaValue for vocab loading).

pub mod loader;
pub mod memory;
pub mod model;
pub mod scratchpad;
pub mod tensor;
pub mod thread_pool;
pub mod tokenizer;
pub mod traits;
