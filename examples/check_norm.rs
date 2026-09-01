use rust_model_inference::{GGUFLoader, TensorSource};

fn main() {
    let loader = GGUFLoader::from_file(
        "/home/liyulingyue/Codes/rust-model-inference/models/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q8_0.gguf",
    )
    .unwrap();

    println!("\n==== blk.0.attn_norm.weight ====");
    let name = "blk.0.attn_norm.weight";
    let bytes = loader.tensor_slice(name).unwrap();
    let mut vals = vec![0.0f32; 2048];
    for i in 0..2048 {
        let off = i * 4;
        vals[i] = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
    }
    println!("first 8: {:?}", &vals[..8]);
    println!(
        "sum: {:.6}, mean: {:.6}, norm: {:.6}",
        vals.iter().sum::<f32>(),
        vals.iter().sum::<f32>() / 2048.0,
        vals.iter().map(|v| v * v).sum::<f32>().sqrt()
    );

    println!("\n==== blk.0.ffn_norm.weight ====");
    let name = "blk.0.ffn_norm.weight";
    let bytes = loader.tensor_slice(name).unwrap();
    let mut vals = vec![0.0f32; 2048];
    for i in 0..2048 {
        let off = i * 4;
        vals[i] = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
    }
    println!("first 8: {:?}", &vals[..8]);
    println!(
        "sum: {:.6}, mean: {:.6}, norm: {:.6}",
        vals.iter().sum::<f32>(),
        vals.iter().sum::<f32>() / 2048.0,
        vals.iter().map(|v| v * v).sum::<f32>().sqrt()
    );

    // Compare with what RMSNorm would produce
    let x = [
        -0.0070474744f32,
        0.0016092658,
        -0.0006659031,
        -0.0010543466,
        -0.00033295155,
        0.0018312335,
        0.0009433627,
        0.0039954185,
    ];
    let w = &vals[..8];
    let eps = 1e-5;
    let n = 2048;
    let sum_sq: f64 = x.iter().map(|&v| f64::from(v * v)).sum::<f64>() * (n as f64 / 8.0);
    let mean_sq = sum_sq as f32 / n as f32;
    let scale = 1.0f32 / (mean_sq + eps).sqrt();
    let normed: Vec<f32> = x
        .iter()
        .zip(w.iter())
        .map(|(xi, wi)| xi * scale * wi)
        .collect();
    println!(
        "\nFor BOS embedding first 8 vals, after rms_norm with blk.0.attn_norm: {:?}",
        normed
    );
}
