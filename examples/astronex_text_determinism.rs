#![cfg(feature = "vulkan")]
//! Run-to-run determinism probe for the Astronex text encoder.
//!
//! The Astronex GPU path produced two different sampled latents from the same
//! seed, and the online attention and the BF16 projections are each
//! bit-deterministic in isolation, so the text encoder is the next candidate:
//! its context embedding feeds every denoising step.
//!
//! Requires the converted components GGUF:
//! `models/Astronex-World/mmproj-Astronex-World.gguf`.
//!
//! On the GB10 box the short-prompt encodings differ between two calls while
//! the long ones usually match; a single-threaded `ComputePool` removes the
//! difference, and `RUST_GPU_MIN_OUTPUTS` moves the whole encoder to the CPU
//! path, which is reproducible. See `docs/develop/ASTRONEX_WORLD.md`.
use rust_model_inference::core::thread_pool::ComputePool;
use rust_model_inference::models::diffusion::dreamx::text::DreamXTextEncoder;
use rust_model_inference::ops::float::enable_gpu;
use rust_model_inference::ops::get_vulkan_context;
use rust_model_inference::GGUFLoader;
use std::sync::Arc;

fn main() {
    enable_gpu();
    let Some(context) = get_vulkan_context() else {
        println!("没有可用的 Vulkan 上下文");
        return;
    };
    println!("设备: {}", context.device_name());
    let loader = GGUFLoader::from_file("models/Astronex-World/mmproj-Astronex-World.gguf")
        .expect("mmproj 载入失败");
    let source: Arc<dyn rust_model_inference::TensorSource> = Arc::new(loader);
    // A single worker makes the encoder reproducible on the GB10 box, so the
    // default here keeps the production thread count to expose the race.
    let pool = Arc::new(ComputePool::new(16));
    let encoder = DreamXTextEncoder::load_astronex(source, pool).expect("文本编码器载入失败");
    let prompt = "A small wooden boat glides along a calm river, surrounded by green mountains.";
    let negative = "static, blurry, low quality";
    let check = |tag: &str, a: &[f32], b: &[f32]| {
        let same = a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        println!("{tag}: 逐位相同={same} 最大绝对差={max_diff:.6e}");
        if !same && a.len() % 4096 == 0 {
            let dim = 4096usize;
            let mut worst_row = 0usize;
            let mut worst = 0.0f32;
            for row in 0..a.len() / dim {
                for col in 0..dim {
                    let d = (a[row * dim + col] - b[row * dim + col]).abs();
                    if d > worst {
                        worst = d;
                        worst_row = row;
                    }
                }
            }
            println!("  最大差出现在第 {worst_row} 行");
        }
    };
    // 1) 同一 prompt 同时作为正负样本
    let a = encoder.encode(prompt, prompt).expect("encode A");
    let b = encoder.encode(prompt, prompt).expect("encode B");
    check(
        "同 prompt 编码 A 正样本 vs B 正样本",
        &a.positive,
        &b.positive,
    );
    check(
        "同 prompt 编码 A 负样本 vs B 负样本",
        &a.negative,
        &b.negative,
    );
    // 2) 两次调用之间插入等待，检验是否为竞态
    std::thread::sleep(std::time::Duration::from_millis(1500));
    let c = encoder.encode(prompt, negative).expect("encode C");
    std::thread::sleep(std::time::Duration::from_millis(1500));
    let d = encoder.encode(prompt, negative).expect("encode D");
    check("等待后 C 正样本 vs D 正样本", &c.positive, &d.positive);
    check("等待后 C 负样本 vs D 负样本", &c.negative, &d.negative);
    // 3) 只编码短 prompt
    let e = encoder.encode(negative, negative).expect("encode E");
    let f = encoder.encode(negative, negative).expect("encode F");
    check("短 prompt E 正样本 vs F 正样本", &e.positive, &f.positive);
    check("短 prompt E 负样本 vs F 负样本", &e.negative, &f.negative);
}
