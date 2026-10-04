//! AuK VAE wrapper: BigVGANFlowVAE with 64-dim latent, 480× downsample,
//! 24 kHz output. The reference implementation lives at
//! `references/audio.cpp/src/community_models/auk/vae.cpp`.
//!
//! This module provides just enough scaffolding for the lib to compile and
//! for the contract tests to find the tensor inventory; the actual decode
//! pass is deferred to a follow-up commit once the DiT forward is wired.

use std::sync::Arc;

use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;

use super::AukAudio;

pub(crate) struct BigVGANFlowVae {
    source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
}

impl BigVGANFlowVae {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        Ok(Self { source, pool })
    }

    pub(crate) fn decode(
        &self,
        _latent: &[f32],
        _sample_rate: u32,
    ) -> Result<AukAudio, String> {
        Err("AuK VAE decode not implemented yet".into())
    }
}

/// F32-vector loader for norm weights / biases (parallels dit.rs).
pub(crate) fn load_f32_vector(
    source: &dyn TensorSource,
    name: &str,
    len: usize,
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [len as u64] {
        return Err(format!("Invalid {name} dimensions"));
    }
    if !matches!(info.ggml_type, GGMLType::F32) {
        return Err(format!(
            "Invalid {name} type {:?}: expected F32",
            info.ggml_type
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let mut values = vec![0.0_f32; len];
    for (dst, raw) in values.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = f32::from_le_bytes(raw.try_into().unwrap());
    }
    Ok(values)
}