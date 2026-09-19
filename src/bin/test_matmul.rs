use rust_model_inference::GGUFLoader;
use std::io::Write;

fn main() {
    let path = std::env::args().nth(1).expect("gguf path");
    let loader = GGUFLoader::from_file(&path).expect("load gguf");

    let w_data = loader.tensor_slice("v.patch_embd.weight").unwrap();
    let w: Vec<f32> = w_data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    println!("w shape: {} (expect 26542080 = 6912*3840)", w.len());
    println!("w first 8: {:?}", &w[..8]);

    let b_data = loader.tensor_slice("v.patch_embd.bias").unwrap();
    let b: Vec<f32> = b_data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    println!("b len: {}", b.len());

    // patches[0] = patch_norm_1_bias (6912 elements)
    let b1_data = loader.tensor_slice("v.patch_norm.1.bias").unwrap();
    let b1: Vec<f32> = b1_data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    println!("b1 len: {}", b1.len());
    println!("b1 first 8: {:?}", &b1[..8]);

    // Compute output[0, 0] in pure scalar
    let in_dim = 6912;
    let out_dim = 3840;

    let mut sum: f32 = 0.0;
    for k in 0..in_dim {
        sum += w[k] * b1[k];
    }
    sum += b[0];
    println!("\nscalar sum_k w[k] * b1[k] + b[0] = {}", sum);

    // Also try AVX2+AVX-512 manually (use has_avx2_fma path)
    let val = dot_f32(&b1, &w[0..in_dim], in_dim);
    println!("dot_f32 (via crate::ops::dot_f32) = {} (Rust val)", val);
    println!("expected: ~-321.96");

    // Print the first 8 actual Rust matmul outputs for token 0
    let mut out = vec![0.0f32; 8];
    for j in 0..8 {
        let val = dot_f32(&b1, &w[j * in_dim..(j + 1) * in_dim], in_dim);
        out[j] = val;
    }
    println!("Rust dot_f32 first 8 (input = b1): {:?}", out);
}

fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut s = 0.0f32;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}