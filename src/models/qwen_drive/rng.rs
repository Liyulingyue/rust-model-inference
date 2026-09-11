use crate::models::diffusion::z_image::dit::TorchMt19937;

pub struct TorchNormalRng;

impl TorchNormalRng {
    pub fn normal_f32(seed: u64, count: usize) -> Vec<f32> {
        let mut values = vec![0.0; count];
        TorchMt19937::new(seed).fill_normal(&mut values);
        values
    }
}
