//! MiniCPM-V 4.7 multimodal projector (`clip.projector_type = "minicpmv4_7"`).
//!
//! The language tower is an ordinary `qwen35` GGUF; everything specific to
//! MiniCPM-V lives in the `mmproj` (a `clip` GGUF holding a SigLIP-style
//! vision tower, the `vit_merger` and the projector MLP).

pub mod config;
pub mod slicing;
pub mod vision;
pub mod weights;

pub use config::{MiniCpmV47Config, PROJECTOR_TYPE};
pub use slicing::{get_slice_instructions, ImageSize, SliceBox, SliceInstructions};
pub use vision::{
    append_suffix, encode_tile, suffix_plan, MiniCpmV47Scratch, SUFFIX_LAST, SUFFIX_OV,
    SUFFIX_OV_SLICE, SUFFIX_ROW_END, SUFFIX_SLICE,
};
pub use weights::{MiniCpmV47Weights, ViTLayer};
