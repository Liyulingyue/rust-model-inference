//! Waveform transformer stub. Full implementation in progress.
use crate::core::tensor::TensorSource;

pub struct WaveformTransformer;

impl WaveformTransformer {
    pub fn from_source(_source: &dyn TensorSource) -> Result<Self, String> {
        Ok(Self)
    }
}