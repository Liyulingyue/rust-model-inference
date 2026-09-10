//! Shared mathematical primitives used by model kernels.

pub mod exp;
pub mod sigmoid;
pub mod tanh;

pub use exp::{exp_approx_inplace, exp_inplace};
pub use sigmoid::sigmoid_inplace;
pub use tanh::{tanh_approx_inplace, tanh_inplace};
