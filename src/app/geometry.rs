//! Image-to-3D CLI orchestration and PLY serialization.
use super::CliOptions;
use crate::models::idol::{Model, Options};
use std::path::{Path, PathBuf};

pub fn selected(o: &CliOptions) -> bool {
    o.image_to_3d || crate::models::idol::matches_directory(&o.model)
}
pub fn validate_options(o: &CliOptions) -> Result<(), String> {
    if o.model.as_os_str().is_empty() {
        return Err("--image-to-3d requires --model <Idol directory>".into());
    }
    if o.compute_policy() == crate::compute::ComputePolicy::Vulkan {
        return Err(
            "Idol has no calibrated Vulkan path; use --compute cpu or auto (CPU fallback)".into(),
        );
    }
    o.validate_compute_mode()?;
    if o.jev
        || o.embedding
        || o.rerank
        || o.tts
        || o.yue2
        || o.dreamx
        || o.astronex
        || o.prompt.is_some()
        || o.audio.is_some()
        || o.video.is_some()
        || o.mmproj.is_some()
        || o.vae.is_some()
        || o.text_encoder.is_some()
        || o.longcat_kind.is_some()
        || o.planner.is_some()
        || o.perception.is_some()
        || o.laya_request.is_some()
        || o.timestral_request.is_some()
        || o.wevisone_request.is_some()
        || o.wevisone_decoder.is_some()
        || o.clm_head.is_some()
        || o.gliner2_decide
        || o.gliner2_boundary
        || o.noise.is_some()
        || !o.references.is_empty()
        || o.width.is_some()
        || o.height.is_some()
        || o.temperature.is_some()
        || o.cfg_scale.is_some()
        || o.max_tokens.is_some()
        || o.max_context.is_some()
        || o.prefill_batch_size.is_some()
        || o.repetition_penalty.is_some()
        || o.top_k.is_some()
        || o.top_p.is_some()
        || o.negative_prompt.is_some()
        || o.chat_template.is_some()
        || o.chat_template_file.is_some()
        || o.jinja
        || o.thinking
        || o.dump_logits
        || o.bench
        || o.profile
        || o.dry_run
        || o.kv_format_explicit
        || o.scenes.is_some()
        || o.image_root.is_some()
        || o.frames.is_some()
        || o.planning_mode.is_some()
        || o.num_samples.is_some()
        || o.num_steps.is_some()
        || o.output.is_some()
        || o.gliner2_schema.is_some()
        || o.ref_audio.is_some()
        || o.ref_text.is_some()
        || o.lyrics.is_some()
        || o.language.is_some()
        || o.astronex_frames.is_some()
        || o.resolution.is_some()
        || o.input.is_some()
        || o.side.is_some()
        || o.guidance.is_some()
        || o.duration_seconds.is_some()
        || o.fps.is_some()
        || o.target_spatial_tokens.is_some()
        || o.refine.is_some()
        || o.refiner_kv_len.is_some()
        || o.latent_upsample.is_some()
        || o.refiner_decoder.is_some()
        || o.edit
        || o.source_audio.is_some()
        || o.source_text.is_some()
        || o.target_text.is_some()
        || o.instruction.is_some()
        || o.use_xvector_supplied
        || o.tts_model.is_some()
        || o.tts_mmproj.is_some()
        || o.qwen_omni_bf16.is_some()
        || o.jev_context.is_some()
        || !o.jev_questions.is_empty()
        || o.jev_positive.is_some()
        || o.jev_output_json
        || o.jev_multi
        || !o.jev_blocks.is_empty()
        || o.chunk_seconds.is_some()
        || o.srt
        || o.vad.is_some()
        || o.qwen_context_file.is_some()
        || o.qwen_latent_file.is_some()
        || o.qwen_latent_width.is_some()
        || o.qwen_latent_height.is_some()
        || o.qwen_timestep.is_some()
        || o.qwen_image_turbo
        || o.rerank_query.is_some()
        || !o.rerank_documents.is_empty()
        || o.rerank_documents_file.is_some()
        || o.rerank_instruction.is_some()
        || o.rerank_max_tokens.is_some()
    {
        return Err("Idol image-to-3D cannot be combined with other task modes or generation flags; use --steps, --cfg, --seed, --geometry-mode and --num-gaussians".into());
    }
    if o.serve
        && (o.image.is_some()
            || o.out.is_some()
            || o.overwrite
            || o.steps.is_some()
            || o.seed.is_some()
            || o.cfg.is_some()
            || o.geometry_mode.is_some()
            || o.num_gaussians.is_some())
    {
        return Err("Idol server generation options belong in each /v1/3d/generations request, not server startup flags".into());
    }
    if !o.serve && (o.image.is_none() || o.out.is_none()) {
        return Err("Idol requires --image <input> and --out <object.ply>".into());
    }
    options(o)?.validate()
}
pub fn options(o: &CliOptions) -> Result<Options, String> {
    let seed = o.seed.unwrap_or(42);
    if seed < 0 {
        return Err("Idol --seed must be nonnegative".into());
    }
    Ok(Options {
        seed: seed as u64,
        steps: o.steps.unwrap_or(50),
        cfg_strength: o.cfg.unwrap_or(3.),
        num_gaussians: o.num_gaussians.unwrap_or(262144),
        mode: o
            .geometry_mode
            .clone()
            .unwrap_or_else(|| "synthetic".into()),
        ..Options::default()
    })
}
pub fn components(o: &CliOptions) -> PathBuf {
    o.components
        .clone()
        .unwrap_or_else(|| o.model.parent().unwrap_or(Path::new(".")).to_path_buf())
}
pub fn load(o: &CliOptions) -> Result<Model, String> {
    validate_options(o)?;
    if o.compute_policy() == crate::compute::ComputePolicy::Auto {
        eprintln!("compute requested=Auto backend=Cpu arch=idol reason=no_calibrated_device_path");
    }
    Model::load(&o.model, &components(o))
}
pub fn ply_bytes(rows: &[[f32; 17]]) -> Result<Vec<u8>, String> {
    if rows.is_empty() || rows.iter().flatten().any(|v| !v.is_finite()) {
        return Err("Cannot serialize empty or non-finite 3D Gaussians".into());
    }
    let mut header = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n",
        rows.len()
    );
    for name in [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0",
        "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        header.push_str(&format!("property float {name}\n"));
    }
    header.push_str("end_header\n");
    let mut out = header.into_bytes();
    out.reserve(rows.len() * 17 * 4);
    for row in rows {
        for value in row {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(out)
}
pub fn run_cli(o: &CliOptions) -> Result<(), String> {
    validate_options(o)?;
    let out = o.out.as_deref().unwrap();
    if out.exists() && !o.overwrite {
        return Err(format!(
            "Output {} exists; use --overwrite to replace it",
            out.display()
        ));
    }
    let image = super::media::decode_image(o.image.as_deref().unwrap())?;
    let model = load(o)?;
    let output = model.generate(image, &options(o)?)?;
    let bytes = ply_bytes(&output.gaussians)?;
    super::write_output_atomically(out, &bytes, o.overwrite)?;
    eprintln!(
        "Idol wrote {} Gaussians to {}; camera={}",
        output.gaussians.len(),
        out.display(),
        serde_json::to_string(&output.camera).map_err(|e| e.to_string())?
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn flags_and_strict_gpu_are_rejected_before_weights() {
        let mut o = CliOptions {
            image_to_3d: true,
            model: "idol".into(),
            image: Some("a.png".into()),
            out: Some("a.ply".into()),
            ..CliOptions::default()
        };
        assert!(validate_options(&o).is_ok());
        o.jev = true;
        assert!(validate_options(&o).is_err());
        o.jev = false;
        o.compute = Some(crate::compute::ComputePolicy::Vulkan);
        assert!(validate_options(&o).unwrap_err().contains("Vulkan"));
    }
    #[test]
    fn ply_format_has_exact_binary_payload() {
        let bytes = ply_bytes(&[[0.; 17]]).unwrap();
        let marker = b"end_header\n";
        let at = bytes
            .windows(marker.len())
            .position(|v| v == marker)
            .unwrap()
            + marker.len();
        assert_eq!(bytes.len() - at, 68);
        assert!(ply_bytes(&[[f32::NAN; 17]]).is_err());
    }
    #[test]
    fn ignored_text_and_server_generation_flags_are_rejected() {
        let mut o = CliOptions {
            image_to_3d: true,
            model: "idol".into(),
            serve: true,
            ..CliOptions::default()
        };
        assert!(validate_options(&o).is_ok());
        o.steps = Some(1);
        assert!(validate_options(&o).unwrap_err().contains("request"));
        o.steps = None;
        o.max_tokens = Some(8);
        assert!(validate_options(&o).is_err());
        o.max_tokens = None;
        o.image = Some("ignored.png".into());
        assert!(validate_options(&o).is_err());
    }
}
