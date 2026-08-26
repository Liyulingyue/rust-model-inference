use rust_model_inference::{GGUFLoader, TensorSource};

fn main() {
    let loader = GGUFLoader::from_file(
        "/home/liyulingyue/Codes/rust-model-inference/models/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q8_0.gguf",
    )
    .unwrap();

    let name = "blk.0.shortconv.in_proj.weight";
    let info = loader.tensor_info(name).unwrap();
    println!("{}: dims={:?}, ggml_type={:?}", name, info.dims, info.ggml_type);

    let bytes = loader.tensor_slice(name).unwrap();
    let n_embd = 2048usize;
    let blocks_per_row = n_embd / 32;
    let bytes_per_row = blocks_per_row * 34;

    // First 3 rows of weight (b, c, x channels for index 0), first 8 values each
    for &ch in &[0usize, 2048usize, 4096usize] {
        let row_off = ch * bytes_per_row;
        let mut row = vec![0.0f32; n_embd];
        for b in 0..blocks_per_row {
            let off = row_off + b * 34;
            let scale_bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            let scale = half::f16::from_bits(scale_bits).to_f32();
            for j in 0..32 {
                let q = bytes[off + 2 + j] as i8 as f32;
                row[b * 32 + j] = scale * q;
            }
        }
        println!(
            "\nRow {} (channel {}) first 8 values: {:?}",
            ch / n_embd,
            ch,
            &row[..8]
        );
    }

    // Now check the conv kernel
    println!("\n\n==== shortconv.conv.weight ====");
    let name = "blk.0.shortconv.conv.weight";
    let info = loader.tensor_info(name).unwrap();
    println!("{}: dims={:?}", name, info.dims);
    let bytes = loader.tensor_slice(name).unwrap();
    let l_cache = info.dims[0] as usize;
    let n_embd = info.dims[1] as usize;
    println!(
        "size: {} bytes, expected for F32: {}",
        bytes.len(),
        l_cache * n_embd * 4
    );
    // Print first row of kernel (kernel[0])
    let mut k0 = vec![0.0f32; n_embd];
    for j in 0..n_embd {
        let off = j * 4;
        k0[j] = f32::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ]);
    }
    println!("kernel[0] first 8: {:?}", &k0[..8]);
    println!("kernel[0] sum: {}, norm: {}", k0.iter().sum::<f32>(), k0.iter().map(|v| v*v).sum::<f32>().sqrt());

    // Row 1
    let mut k1 = vec![0.0f32; n_embd];
    for j in 0..n_embd {
        let off = (n_embd + j) * 4;
        k1[j] = f32::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ]);
    }
    println!("kernel[1] first 8: {:?}", &k1[..8]);

    // Row 2
    let mut k2 = vec![0.0f32; n_embd];
    for j in 0..n_embd {
        let off = (2 * n_embd + j) * 4;
        k2[j] = f32::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ]);
    }
    println!("kernel[2] first 8: {:?}", &k2[..8]);
}
// Check attn_norm weight
println!("\n\n==== blk.0.attn_norm.weight ====");
let name = "blk.0.attn_norm.weight";
let bytes = loader.tensor_slice(name).unwrap();
let mut vals = vec![0.0f32; 2048];
for i in 0..2048 {
    let off = i * 4;
    vals[i] = f32::from_le_bytes([
        bytes[off],
        bytes[off + 1],
        bytes[off + 2],
        bytes[off + 3],
    ]);
}
println!("first 8: {:?}", &vals[..8]);
println!(
    "sum: {:.6}, mean: {:.6}, norm: {:.6}",
    vals.iter().sum::<f32>(),
    vals.iter().sum::<f32>() / 2048.0,
    vals.iter().map(|v| v * v).sum::<f32>().sqrt()
);
