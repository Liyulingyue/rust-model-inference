//! Shared mathematical primitives used by model kernels.

pub mod exp;
pub mod sigmoid;
pub mod tanh;
mod torch;

pub use exp::{exp_approx_inplace, exp_inplace};
pub use sigmoid::sigmoid_inplace;
pub use tanh::{tanh_approx_inplace, tanh_inplace};
pub(crate) use torch::{torch28_erf, torch28_exp, torch28_expm1, torch28_sigmoid, torch28_tanh};
