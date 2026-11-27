//! Microsoft Mage-Flow NR-MMDiT transformer.
//!
//! The ModelScope checkpoints all share this contract.  This module consumes
//! the lossless BF16 GGUF emitted by `tools/converter/mage_flow` and exposes a
//! deterministic DiT forward for parity harnesses.  Text encoding and Mage-VAE
//! are intentionally separate components; the raw transformer path does not
//! pretend to generate pixels without those inputs.

pub mod dit;
pub mod text;
pub mod vae;

pub const IN_CHANNELS: usize = 128;
pub const OUT_CHANNELS: usize = 128;
pub const CONTEXT_DIM: usize = 2560;
pub const HIDDEN: usize = 3072;
pub const HEADS: usize = 24;
pub const HEAD_DIM: usize = 128;
pub const LAYERS: usize = 12;
pub const FFN: usize = 12_288;
pub const AXES_DIM: [usize; 3] = [16, 56, 56];
