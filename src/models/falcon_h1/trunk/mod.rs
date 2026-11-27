//! Falcon-H1 trunk
//!
//! Parallel hybrid: every layer runs a GQA attention branch (RoPE NORM,
//! no QK norm) AND a Mamba2 SSM branch on the same normed input, sums
//! both into the residual, then applies a gated (SwiGLU) FFN. Reference:
//! llama.cpp `src/models/falcon-h1.cpp` @ `171e8846b4af9766c354064cb776cb34a50f053f`.

pub mod config;
pub mod forward;
pub mod weights;

pub use config::FalconH1Config;
pub use forward::{run_inference, FalconH1Model, FalconH1Scratch};
pub use weights::FalconH1LayerWeights;
