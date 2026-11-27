//! Probe the Mage-Flow vision tower: load an F32 mmproj, encode a reference
//! image and report what comes out.
//!
//! ```text
//! cargo run --release --features vulkan --example mage_vision_probe -- \
//!     models/mage-flow/mmproj-mage-flow-F32.gguf out/mage_verify_cpu.png
//! ```

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::models::diffusion::mage_flow::text::load_vision;
use rust_model_inference::models::qwen3::vision::{qwen_smart_resize, VisionScratchpad};
use std::sync::Arc;

fn main() -> std::process::ExitCode {
    let mut args = std::env::args().skip(1);
    let (Some(mmproj), Some(image_path)) = (args.next(), args.next()) else {
        eprintln!("usage: mage_vision_probe <mmproj.gguf> <image.png>");
        return std::process::ExitCode::FAILURE;
    };
    let loader = match GGUFLoader::from_file(&mmproj) {
        Ok(loader) => loader,
        Err(error) => {
            eprintln!("cannot load {mmproj}: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let pool = Arc::new(rust_model_inference::core::thread_pool::ComputePool::new(4));
    let encoder = match load_vision(&loader, pool) {
        Ok(encoder) => encoder,
        Err(error) => {
            eprintln!("cannot build the vision tower: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let image = match image::open(&image_path) {
        Ok(image) => image.to_rgb8(),
        Err(error) => {
            eprintln!("cannot open {image_path}: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let (width, height) = image.dimensions();
    let (mut width, mut height) = (width as usize, height as usize);
    let mut pixels = image.into_raw();
    if width.max(height) > 384 {
        let ratio = 384.0 / width.max(height) as f64;
        let new_width = (width as f64 * ratio).round_ties_even().max(1.0) as usize;
        let new_height = (height as f64 * ratio).round_ties_even().max(1.0) as usize;
        pixels = resize_bicubic(&pixels, width, height, new_width, new_height);
        (width, height) = (new_width, new_height);
    }
    let mut config = encoder.config.clone();
    config.image_min_pixels = 65536;
    config.image_max_pixels = 16777216;
    let grid = match qwen_smart_resize(width, height, &config) {
        Ok(grid) => grid,
        Err(error) => {
            eprintln!("cannot size the reference: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let (new_height, new_width) = (grid.image_height(), grid.image_width());
    let pixels = resize_bicubic(&pixels, width, height, new_width, new_height);
    let pixels: Vec<f32> = pixels
        .iter()
        .map(|value| (*value as f32 / 255.0 - 0.5) / 0.5)
        .collect();
    let mut scratch = VisionScratchpad::new(&encoder.config);
    match encoder.encode_image(&pixels, new_width, new_height, &mut scratch) {
        Ok(grid) => {
            let projected = &scratch.projected;
            let finite = projected.iter().filter(|value| value.is_finite()).count();
            let sum: f64 = projected.iter().map(|value| f64::from(*value)).sum();
            let peak = projected
                .iter()
                .fold(0.0f32, |peak, value| peak.max(value.abs()));
            println!(
                "grid {}x{} -> tokens {} projected {} finite {}/{} sum {:.4} peak {:.4}",
                new_width,
                new_height,
                projected.len() / 2560,
                projected.len(),
                finite,
                projected.len(),
                sum,
                peak
            );
            println!("deepstack 元素 {}", scratch.deepstack.len());
            if finite != projected.len() || projected.is_empty() {
                return std::process::ExitCode::FAILURE;
            }
        }
        Err(error) => {
            eprintln!("cannot encode the reference: {error}");
            return std::process::ExitCode::FAILURE;
        }
    }
    std::process::ExitCode::SUCCESS
}

fn resize_bicubic(
    pixels: &[u8],
    width: usize,
    height: usize,
    new_width: usize,
    new_height: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; new_width * new_height * 3];
    for y in 0..new_height {
        let source_y = (y as f64 * height as f64 / new_height as f64).min(height as f64 - 1.0);
        for x in 0..new_width {
            let source_x = (x as f64 * width as f64 / new_width as f64).min(width as f64 - 1.0);
            let x0 = source_x.floor() as usize;
            let y0 = source_y.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);
            for c in 0..3 {
                let p00 = pixels[(y0 * width + x0) * 3 + c] as f32;
                let p01 = pixels[(y0 * width + x1) * 3 + c] as f32;
                let p10 = pixels[(y1 * width + x0) * 3 + c] as f32;
                let p11 = pixels[(y1 * width + x1) * 3 + c] as f32;
                let fx = source_x - x0 as f64;
                let fy = source_y - y0 as f64;
                let top = p00 + (p01 - p00) * fx as f32;
                let bottom = p10 + (p11 - p10) * fx as f32;
                let value = top + (bottom - top) * fy as f32;
                out[(y * new_width + x) * 3 + c] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}
