//! Map the supplied BF16 Flux VAE decoder to the existing Flux decoder graph.

use crate::core::tensor::{GGMLType, MetaValue, TensorInfo, TensorSource};
use crate::format::safetensors::SafetensorSource;
use std::collections::HashMap;
use std::path::Path;

pub struct LongCatVaeSource {
    weights: SafetensorSource,
    aliases: HashMap<String, (String, TensorInfo)>,
}

impl LongCatVaeSource {
    pub fn open(component_root: &Path) -> Result<Self, String> {
        let weights = SafetensorSource::open(&[
            component_root.join("vae/diffusion_pytorch_model.safetensors")
        ])?;
        let mut source = Self {
            weights,
            aliases: HashMap::new(),
        };
        source.conv("decoder.conv_in", "decoder.conv_in", 16, 512, 3)?;
        source.residual(
            "decoder.mid.block_1",
            "decoder.mid_block.resnets.0",
            512,
            512,
        )?;
        source.attention("decoder.mid.attn_1", "decoder.mid_block.attentions.0", 512)?;
        source.residual(
            "decoder.mid.block_2",
            "decoder.mid_block.resnets.1",
            512,
            512,
        )?;
        for (stage, hf, input, output) in [
            (3, 0, 512, 512),
            (2, 1, 512, 512),
            (1, 2, 512, 256),
            (0, 3, 256, 128),
        ] {
            for block in 0..3 {
                source.residual(
                    &format!("decoder.up.{stage}.block.{block}"),
                    &format!("decoder.up_blocks.{hf}.resnets.{block}"),
                    if block == 0 { input } else { output },
                    output,
                )?;
            }
            if stage != 0 {
                source.conv(
                    &format!("decoder.up.{stage}.upsample.conv"),
                    &format!("decoder.up_blocks.{hf}.upsamplers.0.conv"),
                    output,
                    output,
                    3,
                )?;
            }
        }
        source.norm("decoder.norm_out", "decoder.conv_norm_out", 128)?;
        source.conv("decoder.conv_out", "decoder.conv_out", 128, 3, 3)?;
        source.conv("encoder.conv_in", "encoder.conv_in", 3, 128, 3)?;
        for (stage, input, output) in [(0, 128, 128), (1, 128, 256), (2, 256, 512), (3, 512, 512)] {
            for block in 0..2 {
                source.residual(
                    &format!("encoder.down.{stage}.block.{block}"),
                    &format!("encoder.down_blocks.{stage}.resnets.{block}"),
                    if block == 0 { input } else { output },
                    output,
                )?;
            }
            if stage != 3 {
                source.conv(
                    &format!("encoder.down.{stage}.downsample.conv"),
                    &format!("encoder.down_blocks.{stage}.downsamplers.0.conv"),
                    output,
                    output,
                    3,
                )?;
            }
        }
        source.residual(
            "encoder.mid.block_1",
            "encoder.mid_block.resnets.0",
            512,
            512,
        )?;
        source.attention("encoder.mid.attn_1", "encoder.mid_block.attentions.0", 512)?;
        source.residual(
            "encoder.mid.block_2",
            "encoder.mid_block.resnets.1",
            512,
            512,
        )?;
        source.norm("encoder.norm_out", "encoder.conv_norm_out", 512)?;
        source.conv("encoder.conv_out", "encoder.conv_out", 512, 32, 3)?;
        Ok(source)
    }

    fn alias(
        &mut self,
        target: &str,
        original: &str,
        shape: &[u64],
        target_shape: &[u64],
    ) -> Result<(), String> {
        let info = self
            .weights
            .tensor_info(original)
            .ok_or_else(|| format!("LongCat VAE missing {original}"))?;
        if info.dims != shape || info.ggml_type != GGMLType::BF16 {
            return Err(format!("LongCat VAE invalid {original}: {:?}", info.dims));
        }
        let mut info = info.clone();
        info.name = target.into();
        info.dims = target_shape.into();
        if info.checked_nbytes() != self.weights.tensor_info(original).unwrap().checked_nbytes() {
            return Err(format!("LongCat VAE alias size mismatch: {target}"));
        }
        self.aliases.insert(target.into(), (original.into(), info));
        Ok(())
    }

    fn norm(&mut self, target: &str, original: &str, channels: u64) -> Result<(), String> {
        for field in ["weight", "bias"] {
            self.alias(
                &format!("{target}.{field}"),
                &format!("{original}.{field}"),
                &[channels],
                &[channels],
            )?;
        }
        Ok(())
    }

    fn conv(
        &mut self,
        target: &str,
        original: &str,
        input: u64,
        output: u64,
        kernel: u64,
    ) -> Result<(), String> {
        self.alias(
            &format!("{target}.weight"),
            &format!("{original}.weight"),
            &[kernel, kernel, input, output],
            &[kernel, kernel, input, output],
        )?;
        self.alias(
            &format!("{target}.bias"),
            &format!("{original}.bias"),
            &[output],
            &[output],
        )
    }

    fn linear_as_conv(
        &mut self,
        target: &str,
        original: &str,
        channels: u64,
    ) -> Result<(), String> {
        self.alias(
            &format!("{target}.weight"),
            &format!("{original}.weight"),
            &[channels, channels],
            &[1, 1, channels, channels],
        )?;
        self.alias(
            &format!("{target}.bias"),
            &format!("{original}.bias"),
            &[channels],
            &[channels],
        )
    }

    fn residual(
        &mut self,
        target: &str,
        original: &str,
        input: u64,
        output: u64,
    ) -> Result<(), String> {
        self.norm(
            &format!("{target}.norm1"),
            &format!("{original}.norm1"),
            input,
        )?;
        self.conv(
            &format!("{target}.conv1"),
            &format!("{original}.conv1"),
            input,
            output,
            3,
        )?;
        self.norm(
            &format!("{target}.norm2"),
            &format!("{original}.norm2"),
            output,
        )?;
        self.conv(
            &format!("{target}.conv2"),
            &format!("{original}.conv2"),
            output,
            output,
            3,
        )?;
        if input != output {
            self.conv(
                &format!("{target}.nin_shortcut"),
                &format!("{original}.conv_shortcut"),
                input,
                output,
                1,
            )?;
        }
        Ok(())
    }

    fn attention(&mut self, target: &str, original: &str, channels: u64) -> Result<(), String> {
        self.norm(
            &format!("{target}.norm"),
            &format!("{original}.group_norm"),
            channels,
        )?;
        for (target_part, original_part) in [
            ("q", "to_q"),
            ("k", "to_k"),
            ("v", "to_v"),
            ("proj_out", "to_out.0"),
        ] {
            self.linear_as_conv(
                &format!("{target}.{target_part}"),
                &format!("{original}.{original_part}"),
                channels,
            )?;
        }
        Ok(())
    }
}

impl TensorSource for LongCatVaeSource {
    fn metadata(&self, _key: &str) -> Option<&MetaValue> {
        None
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.aliases.get(name).map(|(_, info)| info)
    }

    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.weights.tensor_slice(&self.aliases.get(name)?.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::thread_pool::ComputePool;
    use crate::models::diffusion::z_image::vae::FluxVae;
    use std::sync::Arc;

    #[test]
    #[ignore = "requires RMI_LONGCAT_COMPONENT_ROOT and local LongCat weights"]
    fn loads_real_vae_and_round_trips_small_image() {
        let root = std::path::PathBuf::from(std::env::var("RMI_LONGCAT_COMPONENT_ROOT").unwrap());
        let source: Arc<dyn TensorSource> = Arc::new(LongCatVaeSource::open(&root).unwrap());
        let vae = FluxVae::load_longcat(source, Arc::new(ComputePool::new(1))).unwrap();
        let input = vec![128u8; 16 * 16 * 3];
        let latent = vae.encode_rgb(&input, 16, 42).unwrap();
        assert_eq!(latent.len(), 16 * 2 * 2);
        assert!(latent.iter().all(|value| value.is_finite()));
        let rgb = vae.decode_rgb(&latent, 2).unwrap();
        assert_eq!(
            (rgb.width, rgb.height, rgb.bytes.len()),
            (16, 16, 16 * 16 * 3)
        );
    }
}
