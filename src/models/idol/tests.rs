#![cfg(feature = "parity-trace")]
//! Real-weight diagnostic fixtures are generated on the remote test host; never checked into Git.
use super::{
    background::Background, common, conditioning::Conditioning, decoder::Decoder,
    denoiser::Denoiser, preprocess,
};
use crate::format::safetensors::SafetensorSource;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn source(root: &Path, relative: &str) -> Result<SafetensorSource, String> {
    SafetensorSource::open(&[root.join(relative)])
}
fn bytes(values: Vec<f32>) -> Result<Vec<u8>, String> {
    if values
        .iter()
        .any(|&v| !v.is_finite() || v < 0. || v > 255. || v.fract() != 0.)
    {
        return Err("Idol pixel fixtures must be integer bytes".into());
    }
    Ok(values.into_iter().map(|v| v as u8).collect())
}

#[test]
#[ignore = "requires pinned remote Idol/DINO/VAE weights and generated oracle fixtures"]
#[cfg(feature = "parity-trace")]
fn same_input_oracle_probe() -> Result<(), String> {
    let compute = match std::env::var("IDOL_COMPUTE").as_deref().unwrap_or("cpu") {
        "cpu" => crate::compute::ComputePolicy::Cpu,
        "auto" => crate::compute::ComputePolicy::Auto,
        "vulkan" => crate::compute::ComputePolicy::Vulkan,
        _ => return Err("IDOL_COMPUTE must be cpu, auto or vulkan".into()),
    };
    let _scope = compute.configure_cli()?;
    #[cfg(feature = "vulkan")]
    let device = if compute == crate::compute::ComputePolicy::Cpu {
        None
    } else {
        crate::ops::get_vulkan_context()
    };
    #[cfg(feature = "vulkan")]
    let submissions_before = device.map_or(0, |context| context.submission_count());
    let model_root = PathBuf::from(
        std::env::var("IDOL_ROOT").map_err(|_| "Set IDOL_ROOT to the parent models directory")?,
    );
    let fixtures = PathBuf::from(
        std::env::var("IDOL_FIXTURES")
            .map_err(|_| "Set IDOL_FIXTURES to generated artifacts/idol")?,
    )
    .canonicalize()
    .map_err(|e| e.to_string())?;
    let metadata: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixtures.join("fixtures.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if metadata["oracle_commit"] != "4702a7f00850e34b6e83a2170cc123c3e9570744" {
        return Err("Unpinned Idol fixture oracle".into());
    }
    for (name, key) in [
        ("inputs.safetensors", "inputs_sha256"),
        ("oracle.jsonl", "trace_sha256"),
    ] {
        let data = std::fs::read(fixtures.join(name)).map_err(|e| e.to_string())?;
        let actual = format!("{:x}", Sha256::digest(data));
        if metadata[key].as_str() != Some(&actual) {
            return Err(format!("Idol fixture hash mismatch: {name}"));
        }
    }
    let components = metadata["components"]
        .as_array()
        .ok_or("Missing fixture components")?;
    let contains = |name: &str| components.iter().any(|value| value == name);
    let oracle = fixtures.join("oracle.jsonl");
    let checkpoints: Vec<serde_json::Value> = std::fs::read_to_string(&oracle)
        .map_err(|e| e.to_string())?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    let filter = checkpoints
        .iter()
        .map(|value| {
            value["name"]
                .as_str()
                .ok_or("Missing oracle checkpoint name")
        })
        .collect::<Result<Vec<_>, _>>()?
        .join(",");
    let native = fixtures.join(if compute == crate::compute::ComputePolicy::Cpu {
        "native.jsonl"
    } else {
        "native-gpu.jsonl"
    });
    if native.exists() {
        std::fs::remove_file(&native).map_err(|e| e.to_string())?;
    }
    // This ignored probe runs alone; environment controls the repository's existing parity writer.
    std::env::set_var("RMI_PARITY_TRACE", &native);
    std::env::set_var("RMI_PARITY_FILTER", filter);
    let inputs = source(&fixtures, "inputs.safetensors")?;
    let rgba = bytes(common::tensor(&inputs, "rgba", &[25, 37, 4])?)?;
    let rgba = image::RgbaImage::from_raw(37, 25, rgba).ok_or("Invalid RGBA fixture")?;
    let prepared = preprocess::image(image::DynamicImage::ImageRgba8(rgba), 64, 1.2, |_| {
        Err("RGBA fixture must bypass background removal".into())
    })?;
    let prepared: Vec<_> = prepared.as_raw().iter().map(|&v| v as f32).collect();
    common::trace("idol.preprocess", &[64, 64, 3], &prepared);
    let latent = common::tensor(&inputs, "latent", &[1, 32, 16])?;
    let points = common::tensor(&inputs, "points", &[1, 12, 3])?;
    if contains("denoiser") {
        let denoiser = Denoiser::load(&source(&model_root, "idol/denoiser.safetensors")?)?;
        let camera = common::tensor(&inputs, "camera", &[1, 1, 5])?;
        let feature1 = common::tensor(&inputs, "feature1", &[1, 9, 1280])?;
        let feature2 = common::tensor(&inputs, "feature2", &[1, 9, 128])?;
        let _ = denoiser.forward_fixture(&latent, &camera, 500., &feature1, &feature2)?;
    }
    if contains("decoder") {
        let decoder = Decoder::load(
            &source(&model_root, "idol/decoder.safetensors")?,
            &source(&model_root, "idol/decoder_gs.safetensors")?,
        )?;
        let _ = decoder.octree_logits(&points, 16., &latent)?;
        let _ = decoder.gaussian_features(&points, &latent)?;
        // Isolate the representation transform from neural error by injecting the oracle's features.
        let features = common::tensor(&inputs, "representation_features", &[1, 12, 480])?;
        let ply = decoder.representation(&points, &features)?;
        let values: Vec<_> = ply.iter().flat_map(|row| row.iter().copied()).collect();
        common::trace("idol.ply", &[384, 17], &values);
    }
    if contains("conditioning") {
        let model = Conditioning::load(
            &source(&model_root, "dinov3-vith16plus/model.safetensors")?,
            &source(
                &model_root,
                "FLUX.2-dev/vae/diffusion_pytorch_model.safetensors",
            )?,
        )?;
        let pixels = bytes(common::tensor(&inputs, "pixels", &[64, 64, 3])?)?;
        let noise = common::tensor(&inputs, "vae_noise", &[1, 32, 8, 8])?;
        let _ = model.dino_tokens(&pixels, 64)?;
        let _ = model.vae_tokens(&pixels, 64, &noise)?;
    }
    if contains("background") {
        let model = Background::load(&source(&model_root, "idol/birefnet.safetensors")?)?;
        let image = common::tensor(&inputs, "background_normalized", &[1, 64, 64, 3])?;
        let _ = model.forward_fixture(&image, 64)?;
    }
    super::gpu::check_failure()?;
    #[cfg(feature = "vulkan")]
    if let Some(context) = device {
        let submitted = context.submission_count() - submissions_before;
        eprintln!(
            "Idol oracle device submissions={submitted} {:?}",
            context.compute_stats()
        );
        if submitted == 0 && compute == crate::compute::ComputePolicy::Vulkan {
            return Err("Strict Vulkan oracle did not submit model work".into());
        }
    }
    let python = std::env::var_os("IDOL_PYTHON").unwrap_or_else(|| "python3".into());
    let status = std::process::Command::new(python)
        .arg(
            std::env::var_os("IDOL_COMPARE")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/oracle/idol/compare.py")
                }),
        )
        .arg(&oracle)
        .arg(&native)
        .arg("--report")
        .arg(
            fixtures.join(if compute == crate::compute::ComputePolicy::Cpu {
                "comparison.json"
            } else {
                "comparison-gpu.json"
            }),
        )
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(
            "Idol numerical oracle comparison failed; inspect artifacts/idol/comparison.json"
                .into(),
        );
    }
    Ok(())
}
