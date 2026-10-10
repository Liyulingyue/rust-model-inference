//! Occamy-1.0 (`qwen35moe`) — a Qwen3.5 hybrid trunk whose FFN is replaced by
//! a 256-expert sparse MoE.
//!
//! The attention and SSM tensors are byte-identical in shape to `qwen35`, so
//! the hybrid trunk runs unchanged; only the FFN differs, and that rides the
//! `Option<&[Edge0MoeWeights]>` extension point the trunk already carries.
//! What differs from Edge0 is the shared expert: Occamy gates it with its own
//! sigmoid before adding it to the routed output, following the Qwen3-Next
//! reference that `references/llama.cpp/src/models/qwen35moe.cpp` implements.

pub mod forward;
pub mod run;
pub mod weights;

pub use weights::OccamyMoeWeights;
