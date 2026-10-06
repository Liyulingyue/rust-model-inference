//! Raw component fixtures for same-host Mage-Flow scalar Oracle comparison.
//! Generation and editing use the main rust-model-inference CLI.
use rust_model_inference::app::{read_f32_file, write_output_atomically};
use rust_model_inference::models::{
    diffusion::mage_flow::{
        dit::MageFlowDit,
        text::{encode_prompt, load_text, load_vision, ReferenceFeatures},
        vae::MageVae,
    },
    qwen3::vision::VisionScratchpad,
};
use rust_model_inference::{ComputePool, GGUFLoader};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

fn arguments() -> Result<(String, HashMap<String, String>), String> {
    let mut args = std::env::args().skip(1);
    let mode = args
        .next()
        .ok_or("Expected dit, sample, vae-encode, vae-decode, vision or text")?;
    if mode == "--help" {
        println!("mage_flow_trace <dit|sample|vae-encode|vae-decode|vision|text> --model FILE --output FILE\n--input F32 --context F32 --shapes HxW,HxW --sigma N --height N --width N\n--steps N --cfg N --negative-context F32 --prompt TEXT --reference-count N --deepstack F32 --threads N");
        std::process::exit(0);
    }
    let mut values = HashMap::new();
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
                | "--prompt"
                | "--steps"
                | "--cfg"
                | "--reference-count"
                | "--deepstack"
        ) {
            return Err(format!("Unknown option {key}"));
        }
        let value = args
            .next()
            .ok_or_else(|| format!("Missing value for {key}"))?;
        if value.starts_with("--") || values.insert(key.clone(), value).is_some() {
            return Err(format!("Invalid or duplicate option {key}"));
        }
    }
    Ok((mode, values))
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
fn write_f32(path: &Path, values: &[f32]) -> Result<(), String> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err("Model output contains NaN or infinity".into());
    }
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_output_atomically(path, &bytes, false)
}
fn run() -> Result<(), String> {
    let (mode, args) = arguments()?;
    let output = PathBuf::from(required(&args, "--output")?);
    if output.exists() {
        return Err(format!("Output already exists: {}", output.display()));
    }
    let threads: usize = number(&args, "--threads", "8")?;
    if threads == 0 || threads > 256 {
        return Err("Threads must be in 1..=256".into());
    }
    let pool = Arc::new(ComputePool::new(threads));
    let default_side = if mode == "vae-decode" { "1" } else { "16" };
    let h: usize = number(&args, "--height", default_side)?;
    let w: usize = number(&args, "--width", default_side)?;
    let values = if mode == "text" {
        let model = load_text(Path::new(required(&args, "--model")?), pool.clone())?;
        let features = if let Some(path) = args.get("--input") {
            let count: usize = number(&args, "--reference-count", "1")?;
            if !(1..=3).contains(&count) {
                return Err("Cached text references require 1..=3 features".into());
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
            Vec::new()
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
                let encoder = load_vision(&model, pool)?;
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
