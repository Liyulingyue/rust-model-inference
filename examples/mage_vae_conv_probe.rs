#![cfg(feature = "vulkan")]
//! MageVAE convolution correctness probe.
//!
//! MageVAE's 2-D convolutions are routed through conv3d_with_options, so on
//! the GPU they run the conv3d_tiled shader that #174 added. That shader does
//! not exist on the pre-#174 main, where the same convolution ran on the CPU.
//!
//! Run twice and compare the outputs:
//!
//! ```sh
//! cargo run --release --features vulkan --example mage_vae_conv_probe
//! RMI_SCALAR=1 cargo run --release --features vulkan --example mage_vae_conv_probe
//! ```
use rust_model_inference::core::thread_pool::ComputePool;
use rust_model_inference::models::diffusion::dreamx::kernels::{conv3d_with_options, Conv3dSpec};
use rust_model_inference::ops::float::enable_gpu;
use rust_model_inference::ops::get_vulkan_context;
use rust_model_inference::GGUFLoader;

fn main() {
    enable_gpu();
    let scalar = std::env::var_os("RMI_SCALAR").is_some();
    println!("模式: {}", if scalar { "CPU 标量" } else { "GPU" });
    if !scalar {
        match get_vulkan_context() {
            Some(context) => println!("设备: {}", context.device_name()),
            None => println!("没有 Vulkan 上下文"),
        }
    }
    let Ok(loader) = GGUFLoader::from_file("models/mage-flow/mage-vae-BF16.gguf") else {
        println!("缺少 models/mage-flow/mage-vae-BF16.gguf");
        return;
    };
    let pool = ComputePool::new(8);
    for (in_ch, out_ch, h, w, kh, kw, pad) in [
        (128usize, 384usize, 64usize, 64usize, 3usize, 3usize, 1usize),
        (128, 384, 32, 32, 3, 3, 1),
        (384, 384, 16, 16, 3, 3, 1),
        (128, 384, 64, 64, 3, 3, 0),
    ] {
        let mut state: u64 = 0x243F6A8885A308D3;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 1024.0 - 8.0
        };
        let input: Vec<f32> = (0..in_ch * h * w).map(|_| next() / 8.0).collect();
        let weight: Vec<f32> = (0..out_ch * in_ch * kh * kw)
            .map(|_| next() / 8.0)
            .collect();
        let bias: Vec<f32> = (0..out_ch).map(|_| next() / 8.0).collect();
        let (out, shape) = conv3d_with_options(
            &pool,
            &input,
            [in_ch, 1, h, w],
            &weight,
            [out_ch, in_ch, 1, kh, kw],
            Some(&bias),
            Conv3dSpec {
                stride: [1, 1, 1],
                padding: [0, pad, pad],
                dilation: [1; 3],
                groups: 1,
            },
        )
        .expect("conv3d");
        let mut sum = 0.0f64;
        let mut absmax = 0.0f32;
        for value in &out {
            sum += *value as f64;
            absmax = absmax.max(value.abs());
        }
        println!(
            "conv in={in_ch} out={out_ch} {h}x{w} k={kh}x{kw} pad={pad} -> shape={shape:?} \
             n={} mean={:.6e} absmax={:.6e}",
            out.len(),
            sum / out.len() as f64,
            absmax
        );
        // Dump the whole tensor so two runs can be compared bit for bit.
        let path = format!("out/mage_vae_conv_{in_ch}_{out_ch}_{h}_{kh}_{pad}.bin");
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).expect("write dump");
    }
}
