use crate::core::tensor::{MetaValue, TensorSource, MetaValueType};

use super::protocol::{CONTEXT, PROTOCOL_VERSION, VOCAB_SIZE};

#[derive(Debug, Clone, PartialEq)]
pub struct YuE2Config {
    pub hidden: usize,
    pub layers: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub context: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
    pub latent_channels: usize,
    pub timestep_shift: f32,
}

impl YuE2Config {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        super::require_string(source, "general.architecture", "yue2")?;
        super::require_string(source, "yue2.protocol_version", PROTOCOL_VERSION)?;
        for (key, expected) in [
            ("yue2.context_length", CONTEXT as u64),
            ("yue2.embedding_length", 2048),
            ("yue2.block_count", 28),
            ("yue2.attention.head_count", 16),
            ("yue2.attention.head_count_kv", 8),
            ("yue2.attention.head_dim", 128),
            ("yue2.feed_forward_length", 6144),
            ("yue2.vocab_size", VOCAB_SIZE as u64),
            ("yue2.latent_channels", 64),
            ("yue2.tensor_count", 628),
        ] {
            super::require_u64(source, key, expected)?;
        }
        super::require_f64(source, "yue2.rms_norm_eps", 0.000001)?;
        super::require_f64(source, "yue2.rope.freq_base", 1_000_000.0)?;
        super::require_f64(source, "yue2.timestep_shift", 1.0)?;
        Ok(Self {
            hidden: 2048,
            layers: 28,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            ffn: 6144,
            vocab: VOCAB_SIZE,
            context: CONTEXT,
            rms_eps: 0.000001,
            rope_base: 1_000_000.0,
            latent_channels: 64,
            timestep_shift: 1.0,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct YuE2VaeConfig {
    pub strides: [usize; 6],
    pub latent_channels: usize,
    pub output_channels: usize,
    pub sample_rate: usize,
    pub ratio: usize,
    pub core: usize,
    pub halo: usize,
}

impl YuE2VaeConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        super::require_string(source, "general.architecture", "yue2_vae")?;
        super::require_string(source, "yue2_vae.release_variant", "standard")?;
        for (key, expected) in [
            ("yue2_vae.latent_channels", 64),
            ("yue2_vae.output_channels", 2),
            ("yue2_vae.sample_rate", 48_000),
            ("yue2_vae.downsampling_ratio", 1920),
            ("yue2_vae.decode_core_frames", 1024),
            ("yue2_vae.decode_halo_frames", 16),
            ("yue2_vae.tensor_count", 217),
        ] {
            super::require_u64(source, key, expected)?;
        }
        let key = "yue2_vae.strides";
        match source.metadata(key) {
            Some(MetaValue::Array(crate::core::tensor::MetaValueType::Uint32, values))
                if values
                    .iter()
                    .map(MetaValue::to_u64)
                    .collect::<Option<Vec<_>>>()
                    == Some(vec![2, 2, 4, 4, 5, 6]) => {}
            value => {
                return Err(format!(
                    "Invalid {key}: expected [2, 2, 4, 4, 5, 6], got {value:?}"
                ))
            }
        }
        Ok(Self {
            strides: [2, 2, 4, 4, 5, 6],
            latent_channels: 64,
            output_channels: 2,
            sample_rate: 48_000,
            ratio: 1920,
            core: 1024,
            halo: 16,
        })
    }
}
