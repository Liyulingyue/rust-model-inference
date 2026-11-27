//! Split the Mage-Flow conditioning path into its two halves and check each
//! one on its own: the Qwen3-VL prompt encoding and the denoiser step that
//! consumes it.
//!
//! The GPU path of the model is not reproducible, and the two halves use
//! different kernels, so it matters which one drifts.
//!
//! ```text
//! cargo run --release --features vulkan --example mage_conditioning_probe -- \
//!     models/mage-flow/qwen3vl-text-BF16.gguf \
//!     models/mage-flow/mage-flow-base-dit-BF16.gguf
//! ```

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::thread_pool::ComputePool;
use rust_model_inference::models::diffusion::mage_flow::dit::MageFlowDit;
use rust_model_inference::models::diffusion::mage_flow::text::{encode_prompt, load_text};
use std::sync::Arc;

fn difference(a: &[f32], b: &[f32]) -> (usize, f64) {
    if a.len() != b.len() {
        return (usize::MAX, f64::INFINITY);
    }
    let mut differing = 0usize;
    let mut worst = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.to_bits() != y.to_bits() {
            differing += 1;
        }
        worst = worst.max(f64::from((x - y).abs()));
    }
    (differing, worst)
}

fn main() -> std::process::ExitCode {
    let mut args = std::env::args().skip(1);
    let (Some(text_path), Some(dit_path)) = (args.next(), args.next()) else {
        eprintln!("usage: mage_conditioning_probe <text.gguf> <dit.gguf>");
        return std::process::ExitCode::FAILURE;
    };
    let prompt = "A blue cat sitting by the window, soft morning light, photorealistic";
    let pool = Arc::new(ComputePool::new(16));
    let text = match load_text(std::path::Path::new(&text_path), pool.clone()) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("cannot load the text tower: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let loader = match GGUFLoader::from_file(&dit_path) {
        Ok(loader) => loader,
        Err(error) => {
            eprintln!("cannot load {dit_path}: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let dit = match MageFlowDit::load(&loader, pool.clone()) {
        Ok(dit) => dit,
        Err(error) => {
            eprintln!("cannot load the denoiser: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };

    println!("== 文本编码（Qwen3-VL prefill）==");
    let first = match encode_prompt(&text, prompt, &[]) {
        Ok(context) => context,
        Err(error) => {
            eprintln!("cannot encode: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    for round in 1..4 {
        let again = match encode_prompt(&text, prompt, &[]) {
            Ok(context) => context,
            Err(error) => {
                eprintln!("cannot encode: {error}");
                return std::process::ExitCode::FAILURE;
            }
        };
        let (differing, worst) = difference(&first, &again);
        println!(
            "  第 {round} 次: 长度 {} 不同元素={differing} 最大绝对差={worst:.6e}",
            first.len()
        );
    }

    println!("== 去噪器一步（固定条件）==");
    let shapes = [[16usize, 16usize]];
    let image = vec![0.25f32; 16 * 16 * 128];
    let reference = match dit.forward(&image, &shapes, &first, 0.5) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("cannot run the denoiser: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    for round in 1..4 {
        let again = match dit.forward(&image, &shapes, &first, 0.5) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("cannot run the denoiser: {error}");
                return std::process::ExitCode::FAILURE;
            }
        };
        let (differing, worst) = difference(&reference, &again);
        println!(
            "  第 {round} 次: 长度 {} 不同元素={differing} 最大绝对差={worst:.6e}",
            reference.len()
        );
    }
    std::process::ExitCode::SUCCESS
}
