use rust_model_inference::app::read_f32_file;
use rust_model_inference::core::{tensor::GGMLType, tokenizer::BPETokenizer};
use rust_model_inference::models::gemma4::vision::resize_bicubic_pillow;
use rust_model_inference::models::{
    diffusion::mage_flow::{
        dit::MageFlowDit,
        text::{encode_prompt, ReferenceFeatures},
        vae::MageVae,
    },
    qwen3::{
        vision::{qwen_smart_resize, VisionEncoder, VisionScratchpad},
        Qwen3Model,
    },
};
use rust_model_inference::{ComputePool, GGUFLoader};
use std::{
    collections::HashMap,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

fn arguments() -> Result<(String, HashMap<String, String>, Vec<String>), String> {
    let mut args = std::env::args().skip(1);
    let mode = args
        .next()
        .ok_or("Expected dit, sample, vae-encode, vae-decode, vision, text, generate or edit")?;
    if mode == "--help" {
        println!("mage-flow <dit|sample|vae-encode|vae-decode|vision|text|generate|edit> --model FILE --output FILE\nRaw modes: --input F32 --context F32 --shapes HxW,HxW --sigma 0.5 --height N --width N\nSample: --steps N --cfg N [--negative-context F32]\nGenerate/edit: --vae FILE --text-encoder FILE --prompt TEXT [--vision FILE --reference IMAGE (repeat up to 3)]\n[--input fixed-noise.f32 --negative-prompt TEXT --steps N --cfg N --seed N --threads N --deepstack F32]");
        std::process::exit(0);
    }
    let mut values = HashMap::new();
    let mut refs = Vec::new();
    while let Some(key) = args.next() {
        if !matches!(
            key.as_str(),
            "--model"
                | "--input"
                | "--context"
                | "--negative-context"
                | "--shapes"
                | "--sigma"
                | "--height"
                | "--width"
                | "--threads"
                | "--output"
                | "--vae"
                | "--text-encoder"
                | "--prompt"
                | "--negative-prompt"
                | "--steps"
                | "--cfg"
                | "--seed"
                | "--vision"
                | "--reference"
                | "--reference-count"
                | "--deepstack"
        ) {
            return Err(format!("Unknown option {key}"));
        }
        let value = args
            .next()
            .ok_or_else(|| format!("Missing value for {key}"))?;
        if key == "--reference" {
            refs.push(value);
            continue;
        }
        if value.starts_with("--") || values.insert(key.clone(), value).is_some() {
            return Err(format!("Invalid or duplicate option {key}"));
        }
    }
    if refs.len() > 3 {
        return Err("Mage supports at most three references".into());
    }
    Ok((mode, values, refs))
}
fn required<'a>(args: &'a HashMap<String, String>, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("Missing {key}"))
}
fn number<T: std::str::FromStr>(
    args: &HashMap<String, String>,
    key: &str,
    default: &str,
) -> Result<T, String> {
    args.get(key)
        .map_or(default, String::as_str)
        .parse()
        .map_err(|_| format!("Invalid {key}"))
}
fn shapes(value: &str) -> Result<Vec<[usize; 2]>, String> {
    value
        .split(',')
        .map(|s| {
            let (h, w) = s.split_once('x').ok_or("Expected HxW shapes")?;
            Ok([
                h.parse().map_err(|_| "Invalid latent height")?,
                w.parse().map_err(|_| "Invalid latent width")?,
            ])
        })
        .collect()
}
fn publish(path: &Path, write: impl FnOnce(&Path) -> Result<(), String>) -> Result<(), String> {
    let temporary = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name()
            .ok_or("Invalid output path")?
            .to_string_lossy(),
        std::process::id()
    ));
    let result = (|| {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| e.to_string())?;
        drop(file);
        write(&temporary)?;
        std::fs::hard_link(&temporary, path).map_err(|e| e.to_string())
    })();
    let _ = std::fs::remove_file(&temporary);
    result
}
fn write_f32(path: &Path, values: &[f32]) -> Result<(), String> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err("Model output contains NaN or infinity".into());
    }
    publish(path, |temp| {
        let mut f = std::fs::File::create(temp).map_err(|e| e.to_string())?;
        for v in values {
            f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        f.sync_all().map_err(|e| e.to_string())
    })
}
fn text_model(path: &str, pool: Arc<ComputePool>) -> Result<Qwen3Model, String> {
    let source = Arc::new(GGUFLoader::from_file(path)?);
    if source
        .tensors()
        .iter()
        .any(|t| !matches!(t.ggml_type, GGMLType::BF16 | GGMLType::F32))
    {
        return Err("Mage text conditioning requires lossless BF16/F32 tensors".into());
    }
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|k| {
        source.metadata(k).cloned()
    })?);
    Qwen3Model::from_source(source, tokenizer, pool)
}
fn vision_model<'a>(
    source: &'a GGUFLoader,
    pool: Arc<ComputePool>,
) -> Result<VisionEncoder<'a>, String> {
    if source
        .tensors()
        .iter()
        .any(|t| t.ggml_type != GGMLType::F32)
    {
        return Err("Mage vision requires lossless F32 mmproj tensors".into());
    }
    let mut model = VisionEncoder::from_source(source, pool)?;
    if model.config.n_embd != 1024
        || model.config.n_layer != 24
        || model.config.projection_dim != 2560
    {
        return Err("Expected released Qwen3-VL 4B vision model".into());
    }
    model.precompute_lossless();
    Ok(model)
}
fn reference_features(
    refs: &[String],
    source: Option<&GGUFLoader>,
    pool: Arc<ComputePool>,
) -> Result<Vec<ReferenceFeatures>, String> {
    if refs.is_empty() {
        return Ok(Vec::new());
    }
    let encoder = vision_model(source.ok_or("Edit conditioning requires --vision")?, pool)?;
    refs.iter()
        .map(|p| {
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
            Ok(ReferenceFeatures {
                embeddings: scratch.projected,
                deepstack: scratch.deepstack,
            })
        })
        .collect()
}
fn run() -> Result<(), String> {
    let (mode, args, refs) = arguments()?;
    let output = PathBuf::from(required(&args, "--output")?);
    if output.exists() {
        return Err(format!("Output already exists: {}", output.display()));
    }
    let threads: usize = number(&args, "--threads", "8")?;
    if threads == 0 || threads > 256 {
        return Err("Threads must be in 1..=256".into());
    }
    let pool = Arc::new(ComputePool::new(threads));
    let default_side = match mode.as_str() {
        "generate" | "edit" => "1024",
        "vae-decode" => "1",
        _ => "16",
    };
    let h: usize = number(&args, "--height", default_side)?;
    let w: usize = number(&args, "--width", default_side)?;
    if matches!(mode.as_str(), "generate" | "edit") {
        if h == 0 || w == 0 || h > 2048 || w > 2048 || h % 16 != 0 || w % 16 != 0 {
            return Err("Pixel sides must be multiples of 16 within 16..=2048".into());
        }
        if (mode == "edit") != !refs.is_empty() {
            return Err("Use generate for text-only input and edit with reference images".into());
        }
        let model = GGUFLoader::from_file(required(&args, "--model")?)?;
        let variant = model
            .metadata("mage_flow.variant")
            .and_then(|v| v.to_string_val())
            .ok_or("Missing Mage variant")?;
        if variant.starts_with("edit") != (mode == "edit") {
            return Err("Selected Mage variant does not match generate/edit mode".into());
        }
        let dit = MageFlowDit::load(&model, pool.clone())?;
        let vae_source = GGUFLoader::from_file(required(&args, "--vae")?)?;
        let vae = MageVae::load(&vae_source, pool.clone())?;
        let vision = args
            .get("--vision")
            .map(|p| GGUFLoader::from_file(p))
            .transpose()?;
        let text = text_model(required(&args, "--text-encoder")?, pool.clone())?;
        let features = reference_features(&refs, vision.as_ref(), pool.clone())?;
        let context = encode_prompt(&text, required(&args, "--prompt")?, &features)?;
        let cfg: f32 = number(
            &args,
            "--cfg",
            if variant.ends_with("turbo") { "1" } else { "5" },
        )?;
        let negative = if cfg > 1.0 {
            Some(encode_prompt(
                &text,
                args.get("--negative-prompt")
                    .filter(|s| !s.is_empty())
                    .map_or(" ", String::as_str),
                &features,
            )?)
        } else {
            None
        };
        let steps: usize = number(
            &args,
            "--steps",
            if variant.ends_with("turbo") {
                "4"
            } else if variant.ends_with("base") {
                "30"
            } else {
                "20"
            },
        )?;
        let lh = h / 16;
        let lw = w / 16;
        let n = lh * lw;
        let mut packed = if let Some(path) = args.get("--input") {
            read_f32_file(Path::new(path))?
        } else {
            use rand::{Rng, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(number(&args, "--seed", "42")?);
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
        let mut image = image::RgbImage::new(w as u32, h as u32);
        for (i, pixel) in image.pixels_mut().enumerate() {
            for c in 0..3 {
                pixel[c] = (127.5 * (pixels[c * h * w + i].clamp(-1.0, 1.0) + 1.0))
                    .clamp(0.0, 255.0) as u8;
            }
        }
        publish(&output, |p| {
            image
                .save_with_format(p, image::ImageFormat::Png)
                .map_err(|e| e.to_string())
        })?;
        println!("Saved {w}x{h} image to {}", output.display());
        return Ok(());
    }
    let values = if mode == "text" {
        let model = text_model(required(&args, "--model")?, pool.clone())?;
        let vision = args
            .get("--vision")
            .map(|p| GGUFLoader::from_file(p))
            .transpose()?;
        let features = if let Some(path) = args.get("--input") {
            let count: usize = number(&args, "--reference-count", "1")?;
            if !refs.is_empty() || !(1..=3).contains(&count) {
                return Err(
                    "Cached text references require 1..=3 features and no --reference images"
                        .into(),
                );
            }
            let embeddings = read_f32_file(Path::new(path))?;
            let deepstack = read_f32_file(Path::new(required(&args, "--deepstack")?))?;
            (0..count)
                .map(|_| ReferenceFeatures {
                    embeddings: embeddings.clone(),
                    deepstack: deepstack.clone(),
                })
                .collect()
        } else {
            reference_features(&refs, vision.as_ref(), pool)?
        };
        encode_prompt(&model, required(&args, "--prompt")?, &features)?
    } else {
        let model = GGUFLoader::from_file(required(&args, "--model")?)?;
        let input = read_f32_file(Path::new(required(&args, "--input")?))?;
        match mode.as_str() {
            "dit" | "sample" => {
                let context = read_f32_file(Path::new(required(&args, "--context")?))?;
                let dit = MageFlowDit::load(&model, pool)?;
                let shapes = shapes(required(&args, "--shapes")?)?;
                if mode == "sample" {
                    let negative = args
                        .get("--negative-context")
                        .map(|p| read_f32_file(Path::new(p)))
                        .transpose()?;
                    dit.sample(
                        &input,
                        &shapes,
                        &context,
                        negative.as_deref(),
                        number(&args, "--steps", "4")?,
                        number(&args, "--cfg", "1")?,
                    )?
                } else {
                    dit.forward(&input, &shapes, &context, number(&args, "--sigma", "0.5")?)?
                }
            }
            "vae-encode" => MageVae::load(&model, pool)?.encode(&input, h, w)?,
            "vae-decode" => MageVae::load(&model, pool)?.decode(&input, h, w)?,
            "vision" => {
                let encoder = vision_model(&model, pool)?;
                let mut scratch = VisionScratchpad::new(&encoder.config);
                encoder.encode_image(&input, w, h, &mut scratch)?;
                if let Some(path) = args.get("--deepstack") {
                    write_f32(Path::new(path), &scratch.deepstack)?;
                }
                scratch.projected
            }
            _ => return Err(format!("Unknown mode {mode}")),
        }
    };
    write_f32(&output, &values)?;
    println!("Saved {} F32 values to {}", values.len(), output.display());
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("Mage-Flow: {error}");
        std::process::exit(1);
    }
}
