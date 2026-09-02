//! dots.tts model configuration, derived from the mmproj GGUF metadata.

use crate::core::tensor::{MetaValue, TensorSource};

#[derive(Debug, Clone)]
pub struct DotsTtsConfig {
    /// Number of latent frames per patch (config.patch_size).
    pub patch_size: usize,
    /// AudioVAE latent width (128).
    pub latent_dim: usize,
    /// Latent frames per second of audio (48 kHz / hop 1920 = 25).
    pub hop_size: usize,
    /// Output sample rate (48000).
    pub sample_rate: usize,
    /// Flow-matching hidden width (DiT hidden_size, 1024).
    pub fm_hidden_size: usize,
    /// LLM hidden width (1536).
    pub llm_hidden_size: usize,
    /// Speaker x-vector width (512).
    pub xvec_dim: usize,
    /// Patch encoder transformer depth.
    pub patch_encoder_layers: usize,
    /// DiT block depth.
    pub dit_layers: usize,
    /// DiT attention heads.
    pub dit_heads: usize,
}

impl DotsTtsConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let arch = source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            .unwrap_or_default();
        if arch != "dotstts" {
            return Err(format!("Unsupported dots.tts architecture: {arch}"));
        }
        let u = |key: &str| -> Result<usize, String> {
            source
                .metadata(key)
                .and_then(MetaValue::to_u64)
                .map(|value| value as usize)
                .ok_or_else(|| format!("Missing metadata: {key}"))
        };
        Ok(Self {
            patch_size: u("dotstts.patch_size")?,
            latent_dim: u("dotstts.latent_dim")?,
            hop_size: u("dotstts.hop_size")?,
            sample_rate: u("dotstts.sample_rate")?,
            fm_hidden_size: u("dotstts.fm_hidden_size")?,
            llm_hidden_size: u("dotstts.llm_hidden_size")?,
            xvec_dim: u("dotstts.xvec_dim")?,
            patch_encoder_layers: 24,
            dit_layers: 18,
            dit_heads: 16,
        })
    }

    /// Audio samples per patch (4 patches × 1920 frames = 7680 samples).
    pub fn samples_per_patch(&self) -> usize {
        self.patch_size * self.hop_size
    }

    /// Total FM sequence rows per audio patch: 1 hidden row + patch_size latent rows.
    pub fn unit_len(&self) -> usize {
        1 + self.patch_size
    }
}