//! Edge0 text model built on the shared hybrid attention/SSM trunk.

pub mod forward;
pub mod weights;

pub use weights::Edge0Model;
pub type Edge0Session<'a, 'm> =
    crate::models::qwen35::trunk::session::HybridSession<'a, 'm, Edge0Model<'m>>;
