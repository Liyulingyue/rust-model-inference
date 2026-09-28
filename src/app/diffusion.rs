use crate::app::cli::ZImageCliOptions;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::models::diffusion::longcat::{LongCatKind, LongCatTransformer};
use crate::models::diffusion::longcat_pipeline::{
    denoise as denoise_longcat, pack_latents, unpack_latents,
};
use crate::models::diffusion::longcat_text::{
    encode_prompt as encode_longcat_prompt, load_tokenizer as load_longcat_tokenizer,
    LongCatTextSource,
};
use crate::models::diffusion::longcat_vae::LongCatVaeSource;
use crate::models::diffusion::pig;
use crate::models::diffusion::qwen_image_2_1::{
    config_from_source, prepare_dit_inputs, validate_dit, QwenImage21Dit,
};
use crate::models::diffusion::z_image::dit::TorchMt19937;
use crate::models::diffusion::z_image::vae::FluxVae;
use crate::models::diffusion::z_image::{ZImageOptions, ZImagePipeline, ZImageRgb};
use crate::models::qwen3::Qwen3Model;
use crate::models::qwen35::vision::{qwen_smart_resize, VisionEncoder, VisionScratchpad};
use image::ImageEncoder;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

fn trace_longcat(name: &str, values: &[f32], shape: &[usize]) -> Result<(), String> {
    let Some(dir) = std::env::var_os("RMI_LONGCAT_TRACE_DIR") else {
        return Ok(());
    };
    let path = PathBuf::from(dir).join(name);
    let mut data = Vec::with_capacity(values.len() * 4);
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path.with_extension("f32"), data).map_err(|error| error.to_string())?;
    std::fs::write(
        path.with_extension("shape"),
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(" ")
            + "\n",
    )
    .map_err(|error| error.to_string())
}

pub fn run_longcat_image_edit(
    model_path: &Path,
    component_root: &Path,
    input_path: &Path,
    output_path: &Path,
    instruction: &str,
    kind: LongCatKind,
    side: usize,
    steps: usize,
    guidance: f32,
    seed: u64,
    threads: usize,
) -> Result<(), String> {
    if instruction.is_empty()
        || side == 0
        || side % 16 != 0
        || steps == 0
        || !guidance.is_finite()
        || threads == 0
    {
        return Err("Invalid LongCat image edit options".into());
    }
    let source = open_model_source(model_path, ComponentRole::Llm)
        .map_err(|error| format!("Open LongCat Transformer: {error}"))?;
    let transformer = LongCatTransformer::load(source.as_ref(), kind)?;
    let vision_pool = Arc::new(ComputePool::new(threads));
    let input = crate::app::media::decode_image(input_path)?;
    if input.width() != input.height() {
        return Err("LongCat image edit currently requires a square input image".into());
    }
    let rgb = input.to_rgb8();
    let resized = crate::models::gemma4::vision::resize_bicubic_pillow(
        rgb.as_raw(),
        rgb.width() as usize,
        rgb.height() as usize,
        side,
        side,
    )?;
    let side_u32 = u32::try_from(side).map_err(|_| "LongCat image side exceeds u32")?;
    let resized_image = image::RgbImage::from_raw(side_u32, side_u32, resized.clone())
        .ok_or("Invalid resized LongCat RGB image")?;
    if std::env::var_os("RMI_LONGCAT_TRACE_DIR").is_some() {
        let mut planar = Vec::with_capacity(resized.len());
        for channel in 0..3 {
            for pixel in 0..side * side {
                planar.push(resized[pixel * 3 + channel] as f32 / 255.0);
            }
        }
        trace_longcat("ref_image", &planar, &[side, side, 3, 1])?;
    }
    let text_source: Arc<dyn TensorSource> = Arc::new(LongCatTextSource::open(component_root)?);
    let mut vision = VisionEncoder::from_source(text_source.as_ref())?;
    // LongCat's reference-image preset uses 384^2..560^2 pixels for the VLM.
    vision.config.image_min_pixels = 384 * 384;
    vision.config.image_max_pixels = 560 * 560;
    let grid = qwen_smart_resize(side, side, &vision.config)?;
    // sd.cpp's clip_preprocess uses nearest interpolation on F32 RGB values.
    let mut pixels = Vec::with_capacity(grid.image_width() * grid.image_height() * 3);
    for y in 0..grid.image_height() {
        for x in 0..grid.image_width() {
            let source =
                ((y * side / grid.image_height()) * side + x * side / grid.image_width()) * 3;
            for channel in 0..3 {
                pixels.push(
                    (resized_image.as_raw()[source + channel] as f32 / 255.0
                        - vision.config.image_mean[channel])
                        / vision.config.image_std[channel],
                );
            }
        }
    }
    let mut vision_scratch = VisionScratchpad::new(&vision.config);
    vision.encode_image(
        &pixels,
        grid.image_width(),
        grid.image_height(),
        &mut vision_scratch,
        &vision_pool,
    )?;
    drop(vision_pool);
    let visual = std::mem::take(&mut vision_scratch.projected);
    trace_longcat("visual", &visual, &[3584, visual.len() / 3584])?;
    eprintln!("[longcat] vision encoded: {} tokens", grid.token_count());
    drop(vision_scratch);
    drop(vision);
    let tokenizer = Arc::new(load_longcat_tokenizer(component_root)?);
    let text = Qwen3Model::from_source(
        text_source,
        Arc::clone(&tokenizer),
        Arc::new(ComputePool::new(1)),
    )?;
    let positive = encode_longcat_prompt(&text, &tokenizer, instruction, &visual)?;
    trace_longcat("cond", &positive, &[3584, positive.len() / 3584])?;
    let negative = if guidance > 1.0 {
        Some(encode_longcat_prompt(&text, &tokenizer, "", &visual)?)
    } else {
        None
    };
    if let Some(negative) = &negative {
        trace_longcat("uncond", negative, &[3584, negative.len() / 3584])?;
    }
    eprintln!("[longcat] text encoded: {} tokens", positive.len() / 3584);
    drop(text);
    let vae_source: Arc<dyn TensorSource> = Arc::new(LongCatVaeSource::open(component_root)?);
    let vae = FluxVae::load_longcat(vae_source, Arc::new(ComputePool::new(threads)))?;
    let reference_latent = vae.encode_rgb(&resized, side, seed)?;
    trace_longcat(
        "ref_latent",
        &reference_latent,
        &[side / 8, side / 8, 16, 1],
    )?;
    let reference = pack_latents(&reference_latent, side / 8, side / 8)?;
    eprintln!("[longcat] reference image encoded");
    let target_len = (side / 8)
        .checked_mul(side / 8)
        .and_then(|value| value.checked_mul(16))
        .ok_or("LongCat noise size overflow")?;
    let mut target = vec![0.0; target_len];
    TorchMt19937::new(seed).fill_normal(&mut target);
    trace_longcat("noise", &target, &[side / 8, side / 8, 16, 1])?;
    let target = pack_latents(&target, side / 8, side / 8)?;
    let latent = denoise_longcat(
        &transformer,
        target,
        &reference,
        &positive,
        negative.as_deref(),
        side / 16,
        side / 16,
        steps,
        guidance,
    )?;
    eprintln!("[longcat] denoising complete");
    let latent = unpack_latents(&latent, side / 8, side / 8)?;
    trace_longcat("denoised", &latent, &[side / 8, side / 8, 16, 1])?;
    write_png_atomically(output_path, &vae.decode_rgb(&latent, side / 8)?)
}

pub fn run_pig_image(
    source: std::sync::Arc<dyn TensorSource>,
    vae_source: Option<std::sync::Arc<dyn TensorSource>>,
    text_encoder_source: Option<std::sync::Arc<dyn TensorSource>>,
    prompt: &str,
    steps: usize,
    resolution: usize,
    n_threads: usize,
) -> Result<(), String> {
    let started = Instant::now();

    let pool = Arc::new(ComputePool::new(n_threads.max(1)));
    let model = pig::PigModel::from_source(source.clone(), pool)?;

    println!(
        "Model: pig (Z-Image) | layers={} | loaded in {}ms",
        model.config().n_layer,
        started.elapsed().as_millis()
    );

    let vae = if let Some(vs) = vae_source {
        match pig::PigVAE::from_source(vs.as_ref()) {
            Ok(v) => {
                println!("VAE loaded successfully");
                Some(v)
            }
            Err(e) => {
                println!("Failed to load VAE: {}", e);
                None
            }
        }
    } else {
        None
    };

    println!("Generating image for prompt: {}", prompt);

    let mut session = pig::PigSession::new(&model, resolution)?;
    if let Some(ref v) = vae {
        session.set_vae(v);
    }

    let text_context = if let Some(ref te_source) = text_encoder_source {
        let te_pool = Arc::new(ComputePool::new(n_threads.max(1)));
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| te_source.metadata(k).cloned())
            .map_err(|e| format!("Failed to load text encoder tokenizer: {}", e))?;
        let text_model =
            Qwen3Model::from_source(Arc::clone(te_source), Arc::new(tokenizer), te_pool)
                .map_err(|e| format!("Failed to load text encoder model: {}", e))?;

        let full_prompt = format!(
            "<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n",
            prompt
        );
        let token_ids = text_model
            .tokenizer()
            .encode(&full_prompt, Default::default());
        let n_tokens = token_ids.len();
        let positions: Vec<[usize; 4]> = (0..n_tokens).map(|i| [i, 0, 0, 0]).collect();

        println!("Encoding text: {} tokens", n_tokens);
        let text_embeddings = text_model
            .text_encode(
                &token_ids.iter().map(|&t| t as u32).collect::<Vec<u32>>(),
                &positions,
            )
            .map_err(|e| format!("Text encoding failed: {}", e))?;
        println!("Text encoding done: {} dimensions", text_embeddings.len());

        text_embeddings
    } else {
        println!("WARNING: No text encoder provided; using zero context");
        let cap_dim = 2560;
        let context_len = 256;
        vec![0.0f32; cap_dim * context_len]
    };

    match session.generate_image(&text_context, steps) {
        Ok(pixels) => {
            println!(
                "Generated {} bytes image in {}ms",
                pixels.len(),
                started.elapsed().as_millis()
            );

            let img_side = (pixels.len() / 4) as u32;
            let img = image::RgbaImage::from_raw(img_side, img_side, pixels)
                .ok_or("Failed to create image from pixels")?;
            img.save("output.png")
                .map_err(|e| format!("Failed to save PNG: {}", e))?;
            println!("Image saved to output.png");
        }
        Err(e) => {
            return Err(format!("Image generation failed: {}", e));
        }
    }

    Ok(())
}

pub fn run_z_image_cli(
    diffusion: Arc<dyn TensorSource>,
    text: Arc<dyn TensorSource>,
    vae: Arc<dyn TensorSource>,
    prompt: &str,
    options: ZImageCliOptions,
    n_threads: usize,
) -> Result<(), String> {
    let started = Instant::now();
    let ZImageCliOptions {
        steps,
        resolution,
        seed,
        out,
    } = options;
    let pipeline = ZImagePipeline::load(diffusion, text, vae, n_threads)?;
    println!(
        "Z-Image components loaded in {}ms",
        started.elapsed().as_millis()
    );
    let rgb = pipeline.generate_rgb(
        prompt,
        &ZImageOptions {
            steps,
            resolution,
            seed,
        },
    )?;
    write_png_atomically(&out, &rgb)?;
    println!(
        "Z-Image PNG saved to {} in {}ms",
        out.display(),
        started.elapsed().as_millis()
    );
    Ok(())
}

pub struct QwenImage21Request {
    pub latent: Option<Vec<f32>>,
    pub context: Option<Vec<f32>>,
    pub latent_width: usize,
    pub latent_height: usize,
    pub timestep: f32,
    pub out: PathBuf,
}

/// Reads a file of raw little-endian f32 values for a diffusion model input.
pub fn read_f32_file(path: &Path) -> Result<Vec<f32>, String> {
    let resolved = path
        .canonicalize()
        .map_err(|error| format!("Resolve {}: {error}", path.display()))?;
    if !resolved.is_file() {
        return Err(format!("{} is not a file", resolved.display()));
    }
    let bytes = std::fs::read(&resolved)
        .map_err(|error| format!("Read {}: {error}", resolved.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{} must hold whole f32 values ({} bytes)",
            resolved.display(),
            bytes.len()
        ));
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    if values.iter().all(|value| value.is_finite()) {
        Ok(values)
    } else {
        Err(format!("{} contains NaN or infinity", resolved.display()))
    }
}

pub fn run_qwen_image_2_1(
    source: Arc<dyn TensorSource>,
    request: QwenImage21Request,
    n_threads: usize,
) -> Result<(), String> {
    validate_dit(source.as_ref())?;
    let config = config_from_source(source.as_ref())?;
    let mut dit = QwenImage21Dit::load(Arc::clone(&source), n_threads)?;
    let synthetic_input = request.latent.is_none() || request.context.is_none();
    let (latent, context, context_len) = prepare_dit_inputs(
        &config,
        request.latent,
        request.context,
        request.latent_width,
        request.latent_height,
        request.timestep,
    )?;
    if synthetic_input {
        eprintln!(
            "Qwen-Image-2.1: using deterministic synthetic input for missing files; this is DiT-only and does not generate an image"
        );
    }
    let velocity = dit.forward(
        &latent,
        request.latent_width,
        request.latent_height,
        &context,
        context_len,
        request.timestep,
    )?;
    if !velocity.iter().all(|value| value.is_finite()) {
        return Err("Qwen-Image-2.1 produced NaN or infinity".into());
    }
    let mut bytes = Vec::with_capacity(velocity.len() * 4);
    for value in &velocity {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    if let Some(parent) = request
        .out
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Create output directory {}: {error}", parent.display()))?;
    }
    write_sibling_temp_then_rename(&request.out, &bytes)?;
    println!(
        "Qwen-Image-2.1 velocity written to {} ({} values, context {context_len}, timestep {})",
        request.out.display(),
        velocity.len(),
        request.timestep,
    );
    Ok(())
}

pub fn write_png_atomically(path: &Path, rgb: &ZImageRgb) -> Result<(), String> {
    let expected = usize::try_from(rgb.width)
        .ok()
        .and_then(|width| {
            usize::try_from(rgb.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or("Z-Image RGB size overflow")?;
    if rgb.bytes.len() != expected {
        return Err(format!(
            "Invalid Z-Image RGB length: expected {expected}, got {}",
            rgb.bytes.len()
        ));
    }
    let mut encoded = Vec::new();
    image::codecs::png::PngEncoder::new(&mut encoded)
        .write_image(
            &rgb.bytes,
            rgb.width,
            rgb.height,
            image::ColorType::Rgb8.into(),
        )
        .map_err(|error| format!("Encode Z-Image PNG: {error}"))?;
    write_sibling_temp_then_rename(path, &encoded)
}

fn write_sibling_temp_then_rename(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let file_name = path
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or("Output path requires a file name")?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    for counter in 0..256u16 {
        let mut temp_name = OsString::from(".");
        temp_name.push(file_name);
        temp_name.push(format!(".tmp-{}-{counter}", std::process::id()));
        let temp_path: PathBuf = parent.join(temp_name);
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!(
                    "Create temporary output {}: {error}",
                    temp_path.display()
                ));
            }
        };
        let result = (|| -> std::io::Result<()> {
            file.write_all(bytes)?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&temp_path, path)
        })();
        if let Err(error) = result {
            let _ = std::fs::remove_file(&temp_path);
            return Err(format!("Publish output to {}: {error}", path.display()));
        }
        return Ok(());
    }
    Err("Could not create a unique temporary output".into())
}

// ======================= DreamX =======================

use crate::app::cli::DreamXCliOptions;
use crate::models::diffusion::dreamx::{
    ensure_memory, gib, resize_to_token_budget, DreamXEstimate, DreamXPipeline, DreamXRequest,
};

pub fn run_dreamx_cli(
    main: Arc<dyn TensorSource>,
    mmproj: Arc<dyn TensorSource>,
    options: DreamXCliOptions,
    n_threads: usize,
) -> Result<(), String> {
    let started = Instant::now();
    let input = image::open(&options.image)
        .map_err(|error| {
            format!(
                "Load DreamX input image {}: {error}",
                options.image.display()
            )
        })?
        .into_rgb8();
    let image = resize_to_token_budget(&input, options.options.target_spatial_tokens)?;
    let request = DreamXRequest {
        image,
        prompt: options.prompt,
        negative_prompt: options.negative_prompt.unwrap_or_default(),
        output: options.out,
        options: options.options,
        overwrite: options.overwrite,
        allow_memory_overcommit: options.allow_memory_overcommit,
    };
    let pipeline = DreamXPipeline::load(main, mmproj, n_threads)?;
    let estimate = pipeline.estimate(&request)?;
    print_estimate(&estimate, request.options.refine);
    ensure_memory(&estimate, request.allow_memory_overcommit)?;
    if options.dry_run {
        println!("DreamX dry-run complete; no tensors were loaded");
        return Ok(());
    }

    let artifacts = pipeline.generate(&request)?;
    println!(
        "DreamX base video: {}\nDreamX audio: {}\nDreamX base mux: {}",
        artifacts.base_video.display(),
        artifacts.audio.display(),
        artifacts.base_muxed.display(),
    );
    if let (Some(video), Some(muxed)) = (artifacts.refined_video, artifacts.refined_muxed) {
        println!(
            "DreamX refined video: {}\nDreamX refined mux: {}",
            video.display(),
            muxed.display(),
        );
    }
    println!(
        "DreamX generation completed in {}ms",
        started.elapsed().as_millis()
    );
    Ok(())
}

fn print_estimate(estimate: &DreamXEstimate, refine: bool) {
    println!(
        "DreamX input: {}x{}, {} spatial tokens, {} output frames",
        estimate.width, estimate.height, estimate.spatial_tokens, estimate.output_frames
    );
    println!(
        "DreamX text stage: {:.2} GiB weights, {:.2} GiB scratch",
        gib(estimate.text_weight_bytes),
        gib(estimate.text_scratch_bytes),
    );
    println!(
        "DreamX first-frame/base VAE stage: {:.2} GiB weights",
        gib(estimate.video_vae_weight_bytes),
    );
    println!(
        "DreamX Creator stage: {:.2} GiB weights, {:.2} GiB scratch",
        gib(estimate.creator_weight_bytes),
        gib(estimate.creator_scratch_bytes),
    );
    println!(
        "DreamX audio stage: {:.2} GiB weights, {} Hz output",
        gib(estimate.audio_vae_weight_bytes),
        estimate.audio_sample_rate,
    );
    if refine {
        println!(
            "DreamX refiner stage: {:.2} GiB weights, {:.2} GiB scratch, {:.2} GiB KV",
            gib(estimate.refiner_weight_bytes),
            gib(estimate.refiner_scratch_bytes),
            gib(estimate.refiner_kv_bytes),
        );
    } else {
        println!("DreamX refiner stage: disabled");
    }
    match estimate.physical_memory_bytes {
        Some(physical) => println!(
            "DreamX estimated peak: {:.2} GiB / {:.2} GiB physical memory",
            gib(estimate.peak_bytes),
            gib(physical),
        ),
        None => println!(
            "DreamX estimated peak: {:.2} GiB; physical memory unavailable",
            gib(estimate.peak_bytes),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::ZImageCliOptions;
    use crate::core::tensor::{GGMLType, MetaValue, TensorInfo};
    use crate::models::diffusion::z_image::ZImageRgb;
    use std::path::{Path, PathBuf};

    struct TextSignatureSource {
        info: TensorInfo,
    }

    impl TextSignatureSource {
        fn new() -> Self {
            Self {
                info: TensorInfo {
                    name: "model.embed_tokens.weight".into(),
                    dims: vec![2560, 151936],
                    ggml_type: GGMLType::Q8_0,
                    offset: 0,
                },
            }
        }
    }

    impl TensorSource for TextSignatureSource {
        fn metadata(&self, _key: &str) -> Option<&MetaValue> {
            None
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            (name == self.info.name).then_some(&self.info)
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    struct MustNotBeRead;

    impl TensorSource for MustNotBeRead {
        fn metadata(&self, _key: &str) -> Option<&MetaValue> {
            panic!("later Z-Image component was read")
        }

        fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
            panic!("later Z-Image component was read")
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            panic!("later Z-Image component was read")
        }
    }

    fn test_temp_dir(line: u32) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "rust-model-inference-z-image-{}-{line}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        path
    }

    fn valid_rgb() -> ZImageRgb {
        ZImageRgb {
            width: 2,
            height: 1,
            bytes: vec![255, 0, 0, 0, 255, 0],
        }
    }

    #[test]
    fn text_signature_cannot_be_dispatched_as_a_dit() {
        let options = ZImageCliOptions {
            steps: 1,
            resolution: 16,
            seed: 7,
            out: "not-reached.png".into(),
        };
        let result = run_z_image_cli(
            Arc::new(TextSignatureSource::new()),
            Arc::new(MustNotBeRead),
            Arc::new(MustNotBeRead),
            "fox",
            options,
            1,
        );
        assert!(result.unwrap_err().contains("cap_embedder.0.weight"));
    }

    #[test]
    fn failed_png_encoding_preserves_the_existing_output() {
        let dir = test_temp_dir(line!());
        let output = dir.join("image.png");
        std::fs::write(&output, b"old").unwrap();
        let invalid = ZImageRgb {
            width: 2,
            height: 2,
            bytes: vec![0; 11],
        };

        assert!(write_png_atomically(&output, &invalid).is_err());
        assert_eq!(std::fs::read(&output).unwrap(), b"old");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn successful_publication_is_a_decodable_png_at_the_requested_size() {
        let dir = test_temp_dir(line!());
        let output = dir.join("image.png");

        write_png_atomically(&output, &valid_rgb()).unwrap();

        let decoded = image::open(&output).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 1));
        assert_eq!(decoded.into_raw(), vec![255, 0, 0, 0, 255, 0]);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn publication_rejects_invalid_paths_and_rgb_overflow() {
        let dir = test_temp_dir(line!());
        let missing_parent = dir.join("missing").join("image.png");
        let overflow = ZImageRgb {
            width: u32::MAX,
            height: u32::MAX,
            bytes: Vec::new(),
        };

        assert!(write_png_atomically(Path::new(""), &valid_rgb()).is_err());
        assert!(write_png_atomically(&missing_parent, &valid_rgb()).is_err());
        assert!(write_png_atomically(&dir.join("overflow.png"), &overflow)
            .unwrap_err()
            .contains("overflow"));
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failed_rename_removes_only_its_sibling_temp() {
        let dir = test_temp_dir(line!());
        let output = dir.join("image.png");
        std::fs::create_dir(&output).unwrap();

        assert!(write_png_atomically(&output, &valid_rgb()).is_err());
        assert!(output.is_dir());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
