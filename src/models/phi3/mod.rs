//! Phi-3 / Phi-4 architecture glue.
//!
//! Phi-4 ships attention and FFN as *fused* tensors that the llama trunk
//! can't read directly. [`Phi3Source`] wraps the GGUF source and exposes
//! per-projection tensors the llama trunk already knows how to consume.
//!
//! See [`source`] for the byte-level split rules.
pub mod source;

pub use source::Phi3Source;
