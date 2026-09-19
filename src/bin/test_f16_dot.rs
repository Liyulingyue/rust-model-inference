use rust_model_inference::ops::{dot_f16_f16_bytes, f16_to_f32, f32_slice_to_f16};

fn main() {
    let n = 3840;
    // Pick positive F16 values that fit in [0.5, 2.0) — realistic RMSNorm output range
    // F16 with biased exponent 0x3c..=0x3e (no normal) represent 0.5..2.0
    let x: Vec<u16> = (0..n).map(|i| 0x3c00 + (i as u16 % 0x300)).collect();
    let w: Vec<u16> = (0..n).map(|i| 0x3c00 + ((i as u32 * 7) as u16 % 0x300)).collect();

    // 1) Reproduce ggml scalar fallback: f16_to_f32 -> f32 * f32 -> double accumulate
    let scalar_sum: f64 = x.iter().zip(w.iter())
        .map(|(a, b)| f64::from(f16_to_f32(*a) * f16_to_f32(*b)))
        .sum();

    // 2) Use my f16_matmul's dot_f16_f16_bytes — this is what the model actually uses
    let mut activation = vec![0u16; n];
    f32_slice_to_f16(&x.iter().map(|&v| f16_to_f32(v)).collect::<Vec<_>>(), &mut activation);
    // Reuse x as the "weight" bytes (need packed u8)
    let mut weight_bytes = vec![0u8; n * 2];
    for (i, w) in w.iter().enumerate() {
        weight_bytes[i * 2..i * 2 + 2].copy_from_slice(&w.to_le_bytes());
    }
    let dot_result = dot_f16_f16_bytes(&activation, &weight_bytes, n);

    println!("scalar_sum (ggml-style): {scalar_sum}");
    println!("dot_f16_f16_bytes: {dot_result}");
    println!("diff: {}", scalar_sum - dot_result as f64);
    let as_f32 = scalar_sum as f32;
    println!("scalar_sum as f32: {as_f32}, diff from dot_result: {}", as_f32 - dot_result);
}