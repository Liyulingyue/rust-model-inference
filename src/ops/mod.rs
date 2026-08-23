#[cfg(target_arch = "x86_64")]
use std::sync::atomic::{AtomicBool, Ordering};

pub mod activation;
pub mod attention;
pub mod dot;
pub mod embedding;
pub mod float;
pub mod kernel;
pub mod math;
pub mod matmul;
pub mod norm;
pub mod quant;
pub mod rope;
pub mod sampling;
pub mod ssm;
pub use activation::*;
pub use attention::*;
pub use dot::*;
pub use embedding::*;
pub use float::*;
pub use matmul::*;
pub use norm::*;
pub use quant::q8_0::*;
pub use quant::{quantize_row_q8_k, quantize_row_q8_k_into, vec_dot_q4k_q8k, vec_dot_q4k_q8k_scalar, BlockQ8K, BLOCK_Q4K_SIZE, BLOCK_Q6K_SIZE, QK_K};
pub use rope::*;
pub use sampling::*;
pub use ssm::*;



#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}
