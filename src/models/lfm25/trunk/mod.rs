//! LFM2.5 trunk — aliases onto the `lfm2` trunk.
//!
//! See [`crate::models::lfm25`] for why these are aliases rather than a second
//! implementation. Nothing here computes anything; the types and functions are
//! the `lfm2` ones under their historical `Lfm25*` names, so existing call
//! sites (JEV single/grouped scorers included) keep working while there is only
//! ever one forward pass to reason about.

pub use crate::models::lfm2::trunk::{
    config::Lfm2Config as Lfm25Config,
    forward::run_forward_logits_lfm2_with_batch as run_forward_logits_lfm25_with_batch,
    forward::run_inference, session::Lfm2Session as Lfm25Session, weights::get_f32_tensor,
    weights::load_layers, weights::Lfm2LayerWeights as Lfm25LayerWeights,
};
