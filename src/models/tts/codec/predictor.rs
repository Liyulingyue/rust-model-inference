//! Code predictor stub. Full implementation in progress.
use crate::core::tensor::TensorSource;

pub struct CodePredictor;

impl CodePredictor {
    pub fn from_source(_source: &dyn TensorSource) -> Result<Self, String> {
        Ok(Self)
    }
}