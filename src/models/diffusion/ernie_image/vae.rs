//! Flux2 VAE input mapping pinned to stable-diffusion.cpp 3f8527a.
use super::super::z_image::{vae::FluxVae, ZImageRgb};
use crate::core::{tensor::TensorSource, thread_pool::ComputePool};
use std::sync::Arc;

pub(super) struct ErnieImageVae(FluxVae);
impl ErnieImageVae {
    pub(super) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        Ok(Self(FluxVae::load_flux2(source, pool)?))
    }
    pub(super) fn decode_rgb(&self, latent: &[f32], side: usize) -> Result<ZImageRgb, String> {
        let mapped = unpack_latent(latent, side)?;
        self.0.decode_mapped_rgb(&mapped, side * 2)
    }
}

// Flux2 normalizes each of the 128 packed channels before pixel shuffle.
fn unpack_latent(latent: &[f32], side: usize) -> Result<Vec<f32>, String> {
    let spatial = side.checked_mul(side).ok_or("Flux2 latent size overflow")?;
    let len = spatial
        .checked_mul(128)
        .ok_or("Flux2 latent size overflow")?;
    let output_side = side.checked_mul(2).ok_or("Flux2 latent size overflow")?;
    if side == 0 || latent.len() != len || latent.iter().any(|x| !x.is_finite()) {
        return Err("Invalid Flux2 packed latent".into());
    }
    let mut output = vec![0.; len];
    for channel in 0..128 {
        let out_channel = channel / 4;
        let dy = (channel % 4) / 2;
        let dx = channel % 2;
        for y in 0..side {
            for x in 0..side {
                let input = latent[channel * spatial + y * side + x];
                let dst = (out_channel * output_side + 2 * y + dy) * output_side + 2 * x + dx;
                output[dst] = input * STD[channel] + MEAN[channel];
            }
        }
    }
    Ok(output)
}

const MEAN: [f32; 128] = [
    -0.0676, -0.0715, -0.0753, -0.0745, 0.0223, 0.0180, 0.0142, 0.0184, -0.0001, -0.0063, -0.0002,
    -0.0031, -0.0272, -0.0281, -0.0276, -0.0290, -0.0769, -0.0672, -0.0902, -0.0892, 0.0168,
    0.0152, 0.0079, 0.0086, 0.0083, 0.0015, 0.0003, -0.0043, -0.0439, -0.0419, -0.0438, -0.0431,
    -0.0102, -0.0132, -0.0066, -0.0048, -0.0311, -0.0306, -0.0279, -0.0180, 0.0030, 0.0015, 0.0126,
    0.0145, 0.0347, 0.0338, 0.0337, 0.0283, 0.0020, 0.0047, 0.0047, 0.0050, 0.0123, 0.0081, 0.0081,
    0.0146, 0.0681, 0.0679, 0.0767, 0.0732, -0.0462, -0.0474, -0.0392, -0.0511, -0.0528, -0.0477,
    -0.0470, -0.0517, -0.0317, -0.0316, -0.0345, -0.0283, 0.0510, 0.0445, 0.0578, 0.0458, -0.0412,
    -0.0458, -0.0487, -0.0467, -0.0088, -0.0106, -0.0088, -0.0046, -0.0376, -0.0432, -0.0436,
    -0.0499, 0.0118, 0.0166, 0.0203, 0.0279, 0.0113, 0.0129, 0.0016, 0.0072, -0.0118, -0.0018,
    -0.0141, -0.0054, -0.0091, -0.0138, -0.0145, -0.0187, 0.0323, 0.0305, 0.0259, 0.0300, 0.0540,
    0.0614, 0.0495, 0.0590, -0.0511, -0.0603, -0.0478, -0.0524, -0.0227, -0.0274, -0.0154, -0.0255,
    -0.0572, -0.0565, -0.0518, -0.0496, 0.0116, 0.0054, 0.0163, 0.0104,
];

const STD: [f32; 128] = [
    1.8029, 1.7786, 1.7868, 1.7837, 1.7717, 1.7590, 1.7610, 1.7479, 1.7336, 1.7373, 1.7340, 1.7343,
    1.8626, 1.8527, 1.8629, 1.8589, 1.7593, 1.7526, 1.7556, 1.7583, 1.7363, 1.7400, 1.7355, 1.7394,
    1.7342, 1.7246, 1.7392, 1.7304, 1.7551, 1.7513, 1.7559, 1.7488, 1.8449, 1.8454, 1.8550, 1.8535,
    1.8240, 1.7813, 1.7854, 1.7945, 1.8047, 1.7876, 1.7695, 1.7676, 1.7782, 1.7667, 1.7925, 1.7848,
    1.7579, 1.7407, 1.7483, 1.7368, 1.7961, 1.7998, 1.7920, 1.7925, 1.7780, 1.7747, 1.7727, 1.7749,
    1.7526, 1.7447, 1.7657, 1.7495, 1.7775, 1.7720, 1.7813, 1.7813, 1.8162, 1.8013, 1.8023, 1.8033,
    1.7527, 1.7331, 1.7563, 1.7482, 1.7610, 1.7507, 1.7681, 1.7613, 1.7665, 1.7545, 1.7828, 1.7726,
    1.7896, 1.7999, 1.7864, 1.7760, 1.7613, 1.7625, 1.7560, 1.7577, 1.7783, 1.7671, 1.7810, 1.7799,
    1.7201, 1.7068, 1.7265, 1.7091, 1.7793, 1.7578, 1.7502, 1.7455, 1.7587, 1.7500, 1.7525, 1.7362,
    1.7616, 1.7572, 1.7444, 1.7430, 1.7509, 1.7610, 1.7634, 1.7612, 1.7254, 1.7135, 1.7321, 1.7226,
    1.7664, 1.7624, 1.7718, 1.7664, 1.7457, 1.7441, 1.7569, 1.7530,
];

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires real Flux2 GGUF and pinned Oracle fixtures"]
    fn oracle_vae_fixture() {
        let model = std::env::var("RMI_ERNIE_VAE_GGUF").unwrap();
        let fixtures = std::env::var("RMI_ERNIE_ORACLE_FIXTURES").unwrap();
        let bytes = std::fs::read(format!("{fixtures}/rmi.ernie.sample.f32")).unwrap();
        let latent: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let source = crate::format::ggufrs::open_model_source(
            std::path::Path::new(&model),
            crate::format::ggufrs::ComponentRole::Llm,
        )
        .unwrap();
        let vae = ErnieImageVae::load(Arc::from(source), Arc::new(ComputePool::new(1))).unwrap();
        let rgb = vae.decode_rgb(&latent, 4).unwrap();
        assert_eq!((rgb.width, rgb.height), (64, 64));
        let expected = std::fs::read(format!("{fixtures}/rmi.ernie.vae.rgb_channels.f32")).unwrap();
        let channels: Vec<f32> = expected
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(channels.len(), 64 * 64 * 3);
        let mut pixels = Vec::with_capacity(channels.len());
        for position in 0..64 * 64 {
            for channel in 0..3 {
                pixels.push(super::super::super::z_image::vae::to_rgb_byte(
                    channels[channel * 64 * 64 + position],
                ));
            }
        }
        assert_eq!(
            rgb.bytes, pixels,
            "RGB bytes from the pinned Oracle F32 channels"
        );
    }

    #[test]
    fn flux2_unpack_pins_channel_and_spatial_order() {
        let latent: Vec<f32> = (0..128 * 4).map(|i| i as f32).collect();
        let output = unpack_latent(&latent, 2).unwrap();
        for c in 0..128 {
            for y in 0..2 {
                for x in 0..2 {
                    let dst = ((c / 4) * 4 + 2 * y + c % 4 / 2) * 4 + 2 * x + c % 2;
                    assert_eq!(
                        output[dst].to_bits(),
                        (latent[c * 4 + y * 2 + x] * STD[c] + MEAN[c]).to_bits()
                    );
                }
            }
        }
        assert!(unpack_latent(&latent[..127], 1).is_err());
    }
}
