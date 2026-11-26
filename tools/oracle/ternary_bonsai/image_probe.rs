// Use the same non-thinking prompt and 64x64 red PNG as holo4/27b/image.cpp.
use rust_model_inference::{
    app::run_qwen35_family_multimodal_logits, compute::ComputePolicy, GGUFLoader,
};
use std::{env, fs, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 6 {
        return Err("MODEL MMPROJ cpu|auto|vulkan RED.png OUTPUT.f32".into());
    }
    let source = GGUFLoader::from_file(&args[1])?;
    let policy = match args[3].as_str() {
        "cpu" => ComputePolicy::Cpu,
        "auto" => ComputePolicy::Auto,
        "vulkan" => ComputePolicy::Vulkan,
        _ => return Err("unknown compute policy".into()),
    };
    let _scope = policy.enter_legacy_scope();
    let gpu = if policy != ComputePolicy::Cpu {
        rust_model_inference::ops::enable_gpu();
        rust_model_inference::ops::get_vulkan_context()
    } else {
        None
    };
    let before = gpu.map_or(0, |ctx| ctx.submission_count());
    let (logits, _) = run_qwen35_family_multimodal_logits(
        &source,
        Path::new(&args[2]),
        Some(Path::new(&args[4])),
        None,
        None,
        "What color is the image? Answer with one word.",
        8,
        16,
        128,
        None,
        None,
    )?;
    let submissions = gpu.map_or(0, |ctx| ctx.submission_count()) - before;
    println!("submissions {submissions}");
    if policy == ComputePolicy::Vulkan && submissions == 0 {
        return Err("strict Vulkan submitted no work".into());
    }
    fs::write(&args[5], bytemuck::cast_slice(&logits))?;
    Ok(())
}
