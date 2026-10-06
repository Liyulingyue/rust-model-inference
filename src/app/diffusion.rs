use crate::app::cli::ZImageCliOptions;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::models::diffusion::pig;
use crate::models::diffusion::qwen_image_2_1::{
    config_from_source, prepare_dit_inputs, validate_dit, QwenImage21Dit,
};
use crate::models::diffusion::z_image::{ZImageOptions, ZImagePipeline, ZImageRgb};
use crate::models::qwen3::Qwen3Model;
use image::ImageEncoder;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

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

pub fn run_mage_flow_cli(
    source: Arc<dyn TensorSource>,
    options: &crate::app::CliOptions,
    n_threads: usize,
) -> Result<(), String> {
    use crate::core::loader::GGUFLoader;
    use crate::format::ggufrs::{open_model_source, ComponentRole};
    use crate::models::diffusion::mage_flow::{
        dit::MageFlowDit,
        text::{encode_prompt, load_text, load_vision, ReferenceFeatures},
        vae::MageVae,
    };
    use crate::models::gemma4::vision::resize_bicubic_pillow;
    use crate::models::qwen3::vision::{qwen_smart_resize, VisionScratchpad};

    for (present, flag) in [
        (options.embedding, "--embedding"),
        (options.rerank, "--rerank"),
        (options.jev, "--jev"),
        (options.tts, "--tts"),
        (options.yue2, "--yue2"),
        (options.dreamx, "--dreamx"),
        (options.audio.is_some(), "--audio"),
        (options.video.is_some(), "--video"),
        (options.dump_logits, "--dump-logits"),
        (options.bench, "--bench"),
        (options.profile, "--profile"),
        (options.thinking, "--thinking"),
        (options.language.is_some(), "--language"),
        (options.lyrics.is_some(), "--lyrics"),
        (options.max_tokens.is_some(), "--max-tokens"),
        (options.temperature.is_some(), "--temp/--temperature"),
        (
            options.top_k.is_some() || options.top_p.is_some(),
            "--top-k/--top-p",
        ),
        (
            options.edit,
            "--edit (use an edit variant and --image/--reference)",
        ),
        (options.cfg_scale.is_some(), "--cfg-scale (use --cfg)"),
        (options.gpu, "--gpu"),
        (options.overwrite, "--overwrite"),
        (
            options.ref_audio.is_some() || options.ref_text.is_some(),
            "--ref-audio/--ref-text",
        ),
        (
            options.source_audio.is_some()
                || options.source_text.is_some()
                || options.target_text.is_some()
                || options.instruction.is_some()
                || options.use_xvector_supplied,
            "--source-audio/--source-text/--target-text/--instruction/--use-xvector",
        ),
        (
            options.planner.is_some() || options.perception.is_some(),
            "--planner/--perception",
        ),
        (options.laya_request.is_some(), "--laya-request"),
    ] {
        if present {
            return Err(format!("Mage-Flow cannot be used with {flag}"));
        }
    }
    let text_path = options
        .text_encoder
        .as_deref()
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or("Mage-Flow requires --text-encoder")?;
    let vae_path = options
        .vae
        .as_deref()
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or("Mage-Flow requires --vae")?;
    let prompt = options
        .prompt
        .as_deref()
        .filter(|p| !p.trim().is_empty())
        .ok_or("Mage-Flow requires a non-empty --prompt")?;
    let out = options
        .out
        .as_deref()
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or("Mage-Flow requires --out")?;
    if out.exists() {
        return Err(format!("Output already exists: {}", out.display()));
    }
    let variant = source
        .metadata("mage_flow.variant")
        .and_then(crate::core::tensor::MetaValue::to_string_val)
        .ok_or("Missing Mage variant")?;
    let refs: Vec<_> = options.image.iter().chain(&options.references).collect();
    if refs.len() > 3 || refs.iter().any(|p| p.as_os_str().is_empty()) {
        return Err("Mage supports at most three non-empty reference paths".into());
    }
    if variant.starts_with("edit") != !refs.is_empty() {
        return Err("Mage edit variants require --image/--reference; generation variants require text-only input".into());
    }
    if !refs.is_empty()
        && options
            .mmproj
            .as_deref()
            .is_none_or(|p| p.as_os_str().is_empty())
    {
        return Err("Mage edit conditioning requires --mmproj".into());
    }
    if refs.is_empty() && options.mmproj.is_some() {
        return Err("Mage --mmproj requires reference images".into());
    }
    let h = options.height.or(options.resolution).unwrap_or(1024);
    let w = options.width.or(options.resolution).unwrap_or(1024);
    if h == 0 || w == 0 || h > 2048 || w > 2048 || h % 16 != 0 || w % 16 != 0 {
        return Err("Pixel sides must be multiples of 16 within 16..=2048".into());
    }
    let cfg = options
        .cfg
        .unwrap_or(if variant.ends_with("turbo") { 1.0 } else { 5.0 });
    let steps = options.steps.unwrap_or(if variant.ends_with("turbo") {
        4
    } else if variant.ends_with("base") {
        30
    } else {
        20
    });
    if !cfg.is_finite() || cfg < 1.0 || steps == 0 || steps > 1000 {
        return Err("Mage requires finite --cfg >= 1 and --steps in 1..=1000".into());
    }
    if options.seed.is_some_and(|s| s < 0) || n_threads == 0 || n_threads > 256 {
        return Err("Mage requires a non-negative --seed and --threads in 1..=256".into());
    }
    crate::app::init_rayon_global_pool(n_threads);
    let pool = Arc::new(ComputePool::new(n_threads));
    let dit = MageFlowDit::load(source.as_ref(), pool.clone())?;
    let vae_source = open_model_source(vae_path, ComponentRole::Llm).map_err(|e| e.to_string())?;
    let vae = MageVae::load(vae_source.as_ref(), pool.clone())?;
    let text = load_text(text_path, pool.clone())?;
    let mut features = Vec::new();
    if let Some(path) = options.mmproj.as_deref() {
        let vision = GGUFLoader::from_file(path)?;
        let encoder = load_vision(&vision, pool.clone())?;
        for p in &refs {
            let image = image::open(p).map_err(|e| e.to_string())?.to_rgb8();
            let (w, h) = image.dimensions();
            let (mut w, mut h) = (w as usize, h as usize);
            let mut pixels = image.into_raw();
            if w.max(h) > 384 {
                let ratio = 384.0 / w.max(h) as f64;
                let new_w = (w as f64 * ratio).round_ties_even().max(1.0) as usize;
                let new_h = (h as f64 * ratio).round_ties_even().max(1.0) as usize;
                pixels = resize_bicubic_pillow(&pixels, w, h, new_w, new_h)?;
                (w, h) = (new_w, new_h);
            }
            let mut cfg = encoder.config.clone();
            cfg.image_min_pixels = 65536;
            cfg.image_max_pixels = 16777216;
            let grid = qwen_smart_resize(w, h, &cfg)?;
            let (new_h, new_w) = (grid.image_height(), grid.image_width());
            let pixels = resize_bicubic_pillow(&pixels, w, h, new_w, new_h)?;
            let pixels: Vec<_> = pixels
                .iter()
                .map(|&v| (v as f32 / 255.0 - 0.5) / 0.5)
                .collect();
            if new_h > 512 || new_w > 512 || pixels.iter().any(|v| !v.is_finite()) {
                return Err("Invalid Mage reference pixels".into());
            }
            let mut scratch = VisionScratchpad::new(&encoder.config);
            encoder.encode_image(&pixels, new_w, new_h, &mut scratch)?;
            features.push(ReferenceFeatures {
                embeddings: scratch.projected,
                deepstack: scratch.deepstack,
            });
        }
    }
    let context = encode_prompt(&text, prompt, &features)?;
    let negative = if cfg > 1.0 {
        Some(encode_prompt(
            &text,
            options
                .negative_prompt
                .as_deref()
                .filter(|s| !s.is_empty())
                .unwrap_or(" "),
            &features,
        )?)
    } else {
        None
    };
    let (lh, lw) = (h / 16, w / 16);
    let n = lh * lw;
    let mut packed = if let Some(path) = options.noise.as_deref() {
        read_f32_file(path)?
    } else {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(options.seed.unwrap_or(42) as u64);
        (0..n * 128)
            .map(|_| {
                let a: f32 = rng.gen_range(f32::MIN_POSITIVE..1.0);
                let b: f32 = rng.gen_range(0.0..1.0);
                (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
            })
            .collect()
    };
    if packed.len() != n * 128 {
        return Err(
            "Fixed noise must contain H/16*W/16*128 F32 values in token-major order".into(),
        );
    }
    let mut shapes = vec![[lh, lw]];
    for p in &refs {
        let image = image::open(p).map_err(|e| e.to_string())?.to_rgb8();
        let rgb = resize_bicubic_pillow(
            image.as_raw(),
            image.width() as usize,
            image.height() as usize,
            w,
            h,
        )?;
        let mut pixels = vec![0.0; 3 * h * w];
        for (i, pixel) in rgb.chunks_exact(3).enumerate() {
            for c in 0..3 {
                pixels[c * h * w + i] = (pixel[c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
        let latent = vae.encode(&pixels, h, w)?;
        packed.extend((0..n * 128).map(|i| latent[(i % 128) * n + i / 128]));
        shapes.push([lh, lw]);
    }
    let latent = dit.sample(&packed, &shapes, &context, negative.as_deref(), steps, cfg)?;
    let chw: Vec<_> = (0..n * 128)
        .map(|i| latent[(i % n) * 128 + i / n])
        .collect();
    let pixels = vae.decode(&chw, lh, lw)?;
    let mut bytes = vec![0; 3 * h * w];
    for (i, pixel) in bytes.chunks_exact_mut(3).enumerate() {
        for c in 0..3 {
            pixel[c] =
                (127.5 * (pixels[c * h * w + i].clamp(-1.0, 1.0) + 1.0)).clamp(0.0, 255.0) as u8;
        }
    }
    write_png_atomically(
        out,
        &ZImageRgb {
            width: w as u32,
            height: h as u32,
            bytes,
        },
        false,
    )?;
    println!(
        "Mage-Flow {variant}: saved {w}x{h} image to {}",
        out.display()
    );
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
    write_png_atomically(&out, &rgb, true)?;
    println!(
        "Z-Image PNG saved to {} in {}ms",
        out.display(),
        started.elapsed().as_millis()
    );
    Ok(())
}

pub fn run_auk_cli(
    diffusion: Arc<dyn TensorSource>,
    vae: Arc<dyn TensorSource>,
    text: Option<Arc<dyn TensorSource>>,
    prompt: &str,
    steps: usize,
    sample_rate: usize,
    seed: i64,
    out: std::path::PathBuf,
    n_threads: usize,
) -> Result<(), String> {
    let started = Instant::now();
    let pipeline =
        crate::models::diffusion::auk::AukPipeline::load(diffusion, vae, text, n_threads)?;
    println!(
        "AuK components loaded in {}ms",
        started.elapsed().as_millis()
    );
    let duration_sec = 1usize;
    let audio = pipeline.generate_audio(
        prompt,
        &crate::models::diffusion::auk::AukOptions {
            steps,
            sample_rate: sample_rate as u32,
            duration_sec,
            seed,
            guidance_scale: 2.0,
            instruct: None,
        },
    )?;
    write_wav(&out, &audio)?;
    println!(
        "AuK audio written to {} ({} samples @ {} Hz) in {}ms",
        out.display(),
        audio.samples.len(),
        audio.sample_rate,
        started.elapsed().as_millis(),
    );
    Ok(())
}

fn write_wav(
    path: &std::path::Path,
    audio: &crate::models::diffusion::auk::AukAudio,
) -> Result<(), String> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)
        .map_err(|e| format!("Failed to create {}: {e}", path.display()))?;
    let sample_rate = audio.sample_rate as u32;
    let channels = audio.channels as u16;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * channels as u32 * bits_per_sample as u32 / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_len = (audio.samples.len() * 2) as u32;
    let fmt_chunk_size: u32 = 16;
    let riff_size: u32 = 4 + (8 + fmt_chunk_size) + (8 + data_len);
    file.write_all(b"RIFF").map_err(|e| e.to_string())?;
    file.write_all(&riff_size.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(b"WAVE").map_err(|e| e.to_string())?;
    file.write_all(b"fmt ").map_err(|e| e.to_string())?;
    file.write_all(&fmt_chunk_size.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&1u16.to_le_bytes())
        .map_err(|e| e.to_string())?; // PCM
    file.write_all(&channels.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&sample_rate.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&byte_rate.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&block_align.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&bits_per_sample.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(b"data").map_err(|e| e.to_string())?;
    file.write_all(&data_len.to_le_bytes())
        .map_err(|e| e.to_string())?;
    for sample in &audio.samples {
        let scaled = (*sample * 32_768.0).clamp(-32_768.0, 32_767.0) as i16;
        file.write_all(&scaled.to_le_bytes())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn run_ernie_image_cli(
    diffusion: Arc<dyn TensorSource>,
    text: Arc<dyn TensorSource>,
    vae: Arc<dyn TensorSource>,
    prompt: &str,
    steps: usize,
    resolution: usize,
    seed: i64,
    out: std::path::PathBuf,
    n_threads: usize,
) -> Result<(), String> {
    use crate::models::diffusion::ernie_image::{
        ErnieImageOptions, ErnieImagePipeline, ErnieImageRgb,
    };
    let started = Instant::now();
    let pipeline = ErnieImagePipeline::load(diffusion, text, vae, n_threads)?;
    println!(
        "ERNIE-Image components loaded in {}ms",
        started.elapsed().as_millis()
    );
    let rgb = pipeline.generate_rgb(
        prompt,
        &ErnieImageOptions {
            steps,
            resolution,
            seed,
        },
    )?;
    let z_rgb = ZImageRgb {
        width: rgb.width,
        height: rgb.height,
        bytes: rgb.bytes,
    };
    write_png_atomically(&out, &z_rgb, true)?;
    let _ = ErnieImageRgb {
        width: 0,
        height: 0,
        bytes: Vec::new(),
    };
    println!(
        "ERNIE-Image PNG saved to {} in {}ms",
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
    write_output_atomically(&request.out, &bytes, true)?;
    println!(
        "Qwen-Image-2.1 velocity written to {} ({} values, context {context_len}, timestep {})",
        request.out.display(),
        velocity.len(),
        request.timestep,
    );
    Ok(())
}

pub fn write_png_atomically(path: &Path, rgb: &ZImageRgb, overwrite: bool) -> Result<(), String> {
    let expected = usize::try_from(rgb.width)
        .ok()
        .and_then(|width| {
            usize::try_from(rgb.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or("RGB size overflow")?;
    if rgb.bytes.len() != expected {
        return Err(format!(
            "Invalid RGB length: expected {expected}, got {}",
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
        .map_err(|error| format!("Encode PNG: {error}"))?;
    write_output_atomically(path, &encoded, overwrite)
}

pub fn write_output_atomically(path: &Path, bytes: &[u8], overwrite: bool) -> Result<(), String> {
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
            if overwrite {
                std::fs::rename(&temp_path, path)
            } else {
                std::fs::hard_link(&temp_path, path)?;
                let _ = std::fs::remove_file(&temp_path);
                Ok(())
            }
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

        assert!(write_png_atomically(&output, &invalid, true).is_err());
        assert_eq!(std::fs::read(&output).unwrap(), b"old");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn successful_publication_is_a_decodable_png_at_the_requested_size() {
        let dir = test_temp_dir(line!());
        let output = dir.join("image.png");

        write_png_atomically(&output, &valid_rgb(), true).unwrap();

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

        assert!(write_png_atomically(Path::new(""), &valid_rgb(), true).is_err());
        assert!(write_png_atomically(&missing_parent, &valid_rgb(), true).is_err());
        assert!(
            write_png_atomically(&dir.join("overflow.png"), &overflow, true)
                .unwrap_err()
                .contains("overflow")
        );
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failed_rename_removes_only_its_sibling_temp() {
        let dir = test_temp_dir(line!());
        let output = dir.join("image.png");
        std::fs::create_dir(&output).unwrap();

        assert!(write_png_atomically(&output, &valid_rgb(), true).is_err());
        assert!(output.is_dir());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn exclusive_publication_keeps_an_existing_output_and_cleans_up() {
        let dir = test_temp_dir(line!());
        let output = dir.join("image.png");
        write_png_atomically(&output, &valid_rgb(), false).unwrap();
        let original = std::fs::read(&output).unwrap();
        assert!(write_output_atomically(&output, b"replacement", false).is_err());
        assert_eq!(std::fs::read(&output).unwrap(), original);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
