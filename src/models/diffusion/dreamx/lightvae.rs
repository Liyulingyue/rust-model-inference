use std::sync::Arc;

use image::RgbImage;

use super::video_vae::{LightVaeDecoderCore, VideoLatent};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;

pub const SCHEME3_DIMS: [usize; 5] = [256, 256, 256, 128, 64];

pub fn pixel_frames(latent_frames: usize) -> Result<usize, String> {
    latent_frames
        .checked_sub(1)
        .and_then(|frames| frames.checked_mul(4))
        .and_then(|frames| frames.checked_add(1))
        .ok_or_else(|| "DreamX LightVAE requires at least one latent frame".into())
}

pub struct LightVae {
    decoder: LightVaeDecoderCore,
}

impl LightVae {
    pub fn load(source: Arc<dyn TensorSource>, pool: Arc<ComputePool>) -> Result<Self, String> {
        Ok(Self {
            decoder: LightVaeDecoderCore::load(source, pool, SCHEME3_DIMS)?,
        })
    }

    pub fn decode_frames(&self, latent: &VideoLatent) -> Result<Vec<RgbImage>, String> {
        let expected = pixel_frames(latent.shape()[1])?;
        let frames = self.decoder.decode_frames(latent)?;
        if frames.len() != expected {
            return Err(format!(
                "DreamX LightVAE produced {} frames, expected {expected}",
                frames.len()
            ));
        }
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheme3_uses_released_channels_and_wan_frame_expansion() {
        assert_eq!(SCHEME3_DIMS, [256, 256, 256, 128, 64]);
        assert_eq!(pixel_frames(1).unwrap(), 1);
        assert_eq!(pixel_frames(3).unwrap(), 9);
        assert!(pixel_frames(0).is_err());
    }
}
