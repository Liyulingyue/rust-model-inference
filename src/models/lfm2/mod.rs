//! LFM2 (Liquid Foundation Model 2) hybrid architecture support.
//!
//! Each layer is either an attention layer (with Q/K norm, RoPE, and KV
//! cache) or a recurrent layer using a short convolution over a
//! persistent per-channel state. FFN tensors exist on every layer.

pub mod base;
pub mod skeleton;

pub use base::{run_inference, KvCacheFmt};
pub use skeleton::{
    get_f32_tensor, load_layers, Lfm2Config, Lfm2LayerWeights,
};