pub mod config;
pub mod kernels;
pub mod text;

pub use config::DreamXConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentUpsampleKind {
    Bilinear,
    Flash,
    Causal2d,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefinerDecoderKind {
    Wan,
    LightVae,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DreamXRefinerOptions {
    pub kv_len: usize,
    pub latent_upsample: LatentUpsampleKind,
    pub decoder: RefinerDecoderKind,
}

impl Default for DreamXRefinerOptions {
    fn default() -> Self {
        Self {
            kv_len: 9,
            latent_upsample: LatentUpsampleKind::Bilinear,
            decoder: RefinerDecoderKind::Wan,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DreamXOptions {
    pub duration_seconds: f32,
    pub fps: usize,
    pub steps: usize,
    pub seed: i64,
    pub target_spatial_tokens: usize,
    pub refine: bool,
    pub refiner: DreamXRefinerOptions,
}

impl Default for DreamXOptions {
    fn default() -> Self {
        Self {
            duration_seconds: 5.0,
            fps: 24,
            steps: 50,
            seed: 0,
            target_spatial_tokens: 880,
            refine: true,
            refiner: DreamXRefinerOptions::default(),
        }
    }
}
