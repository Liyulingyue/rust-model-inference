//! Matmul kernel abstraction.
//!
//! Module layout:
//! - `Kernel` trait — per-dtype matmul dispatch interface (`trait.rs`)
//! - `QuantizedTensor<'a>` — borrowed weight enum, zero-copy from mmap
//!   (`quantized_tensor.rs`)
//! - `QTensorOwned` — owned weight enum for fuse / batch / weight-side
//!   transforms (`qtensor_owned.rs`)
//! - per-dtype SIMD kernels: `f16`, `f32`, `q4_0`, `q4_1`, `q4_k`, `q5_k`,
//!   `q6_k`, `q8_0`
//!
//! Hot path: `Kernel::forward_prequantized` → per-dtype SIMD kernel.
//! Fuse path (FFN gate+up, attention QKV): `QTensorOwned::fuse_vstack`.
//!
//! Design rule (R1): a single weight enum (`QuantizedTensor<'a>`) is used
//! on the hot path; `QTensorOwned` exists only for cases that need to own
//! weight bytes. Future sharded / distributed inference should add a
//! `Sharded` variant to `QuantizedTensor<'a>` rather than introduce a new
//! top-level type.

pub mod f16;
pub mod f32;
pub mod q4_0;
pub mod q4_1;
pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;

mod trait_;
mod quantized_tensor;
mod qtensor_owned;



// Re-exports for convenient access from `ops::kernel::*`.
pub use trait_::Kernel;
pub use quantized_tensor::{F16Weight, QuantizedTensor};
pub use qtensor_owned::QTensorOwned;
