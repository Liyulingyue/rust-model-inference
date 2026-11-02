//! Idol image-to-3D Gaussian splats, using the released safetensors components.
mod background;
mod common;
mod conditioning;
mod decoder;
mod denoiser;
mod preprocess;
#[cfg(test)]
mod tests;

use crate::format::safetensors::SafetensorSource;
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub seed: u64,
    pub steps: usize,
    pub cfg_strength: f32,
    pub rescale_t: f32,
    pub num_gaussians: usize,
    pub mode: String,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            seed: 42,
            steps: 50,
            cfg_strength: 3.,
            rescale_t: 3.,
            num_gaussians: 262144,
            mode: "synthetic".into(),
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<(), String> {
        if self.steps == 0 || self.steps > 1000 {
            return Err("Idol steps must be within 1..=1000".into());
        }
        if !self.cfg_strength.is_finite() || !(0.0..=100.0).contains(&self.cfg_strength) {
            return Err("Idol CFG strength must be finite and within 0..=100".into());
        }
        if !self.rescale_t.is_finite() || self.rescale_t <= 0. || self.rescale_t > 100. {
            return Err("Idol time rescale must be finite and within (0,100]".into());
        }
        if !(32768..=1048576).contains(&self.num_gaussians) || self.num_gaussians % 32 != 0 {
            return Err(
                "Idol num_gaussians must be a multiple of 32 within 32768..=1048576".into(),
            );
        }
        if self.mode != "synthetic" && self.mode != "real" {
            return Err("Idol mode must be synthetic or real".into());
        }
        Ok(())
    }
}
#[derive(Debug, Serialize)]
pub struct Camera {
    pub yaw: f64,
    pub pitch: f64,
    pub radius: f64,
    pub fov: f64,
}
pub struct Output {
    pub gaussians: Vec<[f32; 17]>,
    pub camera: Camera,
}

pub fn matches_directory(root: &Path) -> bool {
    root.is_dir()
        && std::fs::read(root.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .is_some_and(|c| c["denoiser"]["name"] == "LatentSeqMMFlowModel")
}
fn validate_config(cfg: &Value) -> Result<(), String> {
    let expected = json!({"denoiser":{"name":"LatentSeqMMFlowModel","args":{"q_token_length":8192,"in_channels":16,"cam_channels":5,"model_channels":1024,"cond_channels":1280,"cond2_channels":128,"num_blocks":24,"num_refiner_blocks":2,"num_heads":16,"mlp_ratio":4}},"decoder":{"name":"OctreeProbabilityFixedlenDecoder","args":{"model_channels":1024,"cond_channels":16,"num_blocks":4,"num_heads":16,"mlp_ratio":4}},"decoder_gs":{"name":"GaussianFixedlenDecoder","args":{"in_channels":3,"model_channels":1024,"cond_channels":16,"num_blocks":16,"num_heads":16,"mlp_ratio":4,"representation_config":{"num_gaussians":32,"lr":{"_xyz":1.0,"_features_dc":1.0,"_opacity":1.0,"_scaling":1.0,"_rotation":0.1},"perturbe_size":1.5,"3d_filter_kernel_size":0.0009,"scaling_bias":0.004,"opacity_bias":0.1}}},"image_size":1024,"max_voxel_level":8,"image_encoders":{"dinov3":"facebook/dinov3-vith16plus-pretrain-lvd1689m","flux2_vae":"black-forest-labs/FLUX.2-dev"}});
    if cfg != &expected {
        return Err("Unsupported Idol configuration: expected the released speridlabs/idol 1024px/8192-token contract".into());
    }
    Ok(())
}

fn component_config(path: &Path, expected: Value) -> Result<(), String> {
    let config: Value = serde_json::from_slice(
        &std::fs::read(path).map_err(|e| format!("Read {}: {e}", path.display()))?,
    )
    .map_err(|e| format!("Parse {}: {e}", path.display()))?;
    for (key, value) in expected.as_object().unwrap() {
        if config.get(key) != Some(value) {
            return Err(format!(
                "Unsupported Idol component {}: {key} must be {value}",
                path.display()
            ));
        }
    }
    Ok(())
}

pub struct Model {
    denoiser: denoiser::Denoiser,
    decoder: decoder::Decoder,
    conditioning: conditioning::Conditioning,
    background: background::Background,
}
impl Model {
    /// `components` holds `dinov3-vith16plus/` and `FLUX.2-dev/` from the pinned ModelScope manifest.
    pub fn load(root: &Path, components: &Path) -> Result<Self, String> {
        let cfg: Value = serde_json::from_slice(
            &std::fs::read(root.join("config.json"))
                .map_err(|e| format!("Read Idol config: {e}"))?,
        )
        .map_err(|e| format!("Parse Idol config: {e}"))?;
        validate_config(&cfg)?;
        component_config(
            &components.join("dinov3-vith16plus/config.json"),
            json!({"model_type":"dinov3_vit","hidden_size":1280,"intermediate_size":5120,"num_hidden_layers":32,"num_attention_heads":20,"num_register_tokens":4,"patch_size":16,"num_channels":3,"hidden_act":"silu","use_gated_mlp":true,"key_bias":false,"query_bias":true,"value_bias":true,"proj_bias":true,"mlp_bias":true,"layer_norm_eps":1e-5,"rope_theta":100.0,"pos_embed_rescale":2.0,"pos_embed_shift":null,"pos_embed_jitter":null}),
        )?;
        component_config(
            &components.join("FLUX.2-dev/vae/config.json"),
            json!({"_class_name":"AutoencoderKLFlux2","act_fn":"silu","in_channels":3,"latent_channels":32,"block_out_channels":[128,256,512,512],"down_block_types":["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],"layers_per_block":2,"norm_num_groups":32,"mid_block_add_attention":true,"use_quant_conv":true,"patch_size":[2,2],"batch_norm_eps":0.0001}),
        )?;
        let source = |path: std::path::PathBuf| SafetensorSource::open(&[path]);
        let denoiser = source(root.join("denoiser.safetensors"))?;
        let octree = source(root.join("decoder.safetensors"))?;
        let gaussian = source(root.join("decoder_gs.safetensors"))?;
        let dino = source(components.join("dinov3-vith16plus/model.safetensors"))?;
        let vae = source(components.join("FLUX.2-dev/vae/diffusion_pytorch_model.safetensors"))?;
        let background = source(root.join("birefnet.safetensors"))?;
        Ok(Self {
            denoiser: denoiser::Denoiser::load(&denoiser)?,
            decoder: decoder::Decoder::load(&octree, &gaussian)?,
            conditioning: conditioning::Conditioning::load(&dino, &vae)?,
            background: background::Background::load(&background)?,
        })
    }
    pub fn generate(
        &self,
        image: image::DynamicImage,
        options: &Options,
    ) -> Result<Output, String> {
        options.validate()?;
        let _scope = crate::compute::ComputePolicy::Cpu.enter_legacy_scope();
        let prepared = preprocess::image(
            image,
            1024,
            if options.mode == "real" { 1.7 } else { 1.2 },
            |rgb| {
                let mask = self.background.alpha(
                    rgb.as_raw(),
                    rgb.width() as usize,
                    rgb.height() as usize,
                )?;
                image::GrayImage::from_raw(rgb.width(), rgb.height(), mask)
                    .ok_or_else(|| "Invalid Idol background mask storage".into())
            },
        )?;
        let mut rng = crate::models::diffusion::z_image::dit::TorchMt19937::new(options.seed);
        let mut vae_noise = vec![0.; 32 * 128 * 128];
        rng.fill_normal(&mut vae_noise);
        let (feature1, feature2) = self.conditioning.encode(prepared.as_raw(), &vae_noise)?;
        common::trace("idol.feature1", &[1, 4101, 1280], &feature1);
        common::trace("idol.feature2", &[1, 4101, 128], &feature2);
        let mut latent = vec![0.; 8192 * 16];
        rng.fill_normal(&mut latent);
        let mut camera = vec![0.; 5];
        rng.fill_normal(&mut camera);
        self.denoise(&mut latent, &mut camera, &feature1, &feature2, options)?;
        let mut random = StdRng::seed_from_u64(options.seed);
        let anchors =
            self.decoder
                .sample_anchors(&latent, options.num_gaussians / 32, 8, &mut random)?;
        let features = self.decoder.gaussian_features(&anchors, &latent)?;
        let gaussians = self.decoder.representation(&anchors, &features)?;
        Ok(Output {
            gaussians,
            camera: decode_camera(&camera)?,
        })
    }
    fn denoise(
        &self,
        latent: &mut [f32],
        camera: &mut [f32],
        feature1: &[f32],
        feature2: &[f32],
        options: &Options,
    ) -> Result<(), String> {
        let negative1 = vec![0.; feature1.len()];
        let negative2 = vec![0.; feature2.len()];
        for step in 0..options.steps {
            let t = sigma(step, options.steps, options.rescale_t);
            let next = sigma(step + 1, options.steps, options.rescale_t);
            let (mut vl, mut vc) =
                self.denoiser
                    .forward(latent, camera, (t * 1000.) as f32, feature1, feature2)?;
            if options.cfg_strength > 0. {
                let (nl, nc) = self.denoiser.forward(
                    latent,
                    camera,
                    (t * 1000.) as f32,
                    &negative1,
                    &negative2,
                )?;
                for (v, n) in vl.iter_mut().zip(nl) {
                    *v = (1. + options.cfg_strength) * *v - options.cfg_strength * n;
                }
                for (v, n) in vc.iter_mut().zip(nc) {
                    *v = (1. + options.cfg_strength) * *v - options.cfg_strength * n;
                }
            }
            let dt = (t - next) as f32;
            for (x, v) in latent.iter_mut().zip(vl) {
                *x -= v * dt;
            }
            for (x, v) in camera.iter_mut().zip(vc) {
                *x -= v * dt;
            }
            common::finite(latent, "sampled latent")?;
            common::finite(camera, "sampled camera")?;
            common::trace(
                &format!("idol.sample.{step}.latent"),
                &[1, 8192, 16],
                latent,
            );
            common::trace(&format!("idol.sample.{step}.camera"), &[1, 1, 5], camera);
            eprintln!("[idol] sample {}/{}", step + 1, options.steps);
        }
        Ok(())
    }
}
fn sigma(index: usize, steps: usize, shift: f32) -> f64 {
    let t = 1. - index as f64 / steps as f64;
    let shifted = shift as f64 * t;
    shifted / (shifted + (1. - t))
}
fn decode_camera(x: &[f32]) -> Result<Camera, String> {
    if x.len() != 5 {
        return Err("Idol camera requires five values".into());
    }
    common::finite(x, "camera")?;
    let length = crate::ops::dot_f32(&x[..3], &x[..3], 3).sqrt();
    let (dx, dy, dz) = if length > 1e-6 {
        (
            (x[0] / length) as f64,
            (x[1] / length) as f64,
            (x[2] / length) as f64,
        )
    } else {
        (0., 0., 1.)
    };
    let radius = if x[3] < 1e-3 { 1000. } else { 1. / x[3] as f64 };
    Ok(Camera {
        yaw: dx.atan2(dy),
        pitch: dz.clamp(-1., 1.).asin(),
        radius,
        fov: (2. * (x[4] as f64 / radius).atan()).to_degrees(),
    })
}
#[cfg(test)]
mod unit_tests {
    use super::*;
    #[test]
    fn validates_options_and_camera() {
        assert!(Options::default().validate().is_ok());
        assert!(Options {
            num_gaussians: 32769,
            ..Options::default()
        }
        .validate()
        .is_err());
        assert!(Options {
            steps: 0,
            ..Options::default()
        }
        .validate()
        .is_err());
        let c = decode_camera(&[0.; 5]).unwrap();
        assert_eq!(c.radius, 1000.);
        assert_eq!(c.pitch, std::f64::consts::FRAC_PI_2);
        assert_eq!(sigma(50, 50, 3.), 0.);
    }
    #[test]
    fn rejects_other_model_configuration() {
        assert!(validate_config(&json!({"denoiser":{"name":"LatentSeqMMFlowModel"}})).is_err());
    }
    #[test]
    fn tiny_time_shifts_keep_exact_finite_endpoints() {
        for shift in [f32::MIN_POSITIVE, 1e-20, 1e-12, 3., 100.] {
            assert_eq!(sigma(0, 50, shift), 1.);
            assert_eq!(sigma(50, 50, shift), 0.);
            assert!((0.0..=1.0).contains(&sigma(1, 50, shift)));
        }
    }
    #[test]
    fn component_contract_rejects_same_width_with_wrong_position_encoding() {
        let path = std::env::temp_dir().join(format!("idol-config-{}.json", std::process::id()));
        std::fs::write(&path, br#"{"hidden_size":1280,"rope_theta":10000.0}"#).unwrap();
        let result = component_config(&path, json!({"hidden_size":1280,"rope_theta":100.0}));
        std::fs::remove_file(&path).unwrap();
        assert!(result.unwrap_err().contains("rope_theta"));
    }
}
