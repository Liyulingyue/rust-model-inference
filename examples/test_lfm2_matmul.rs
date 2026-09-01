// Compare Q8Kernel matmul with F32 reference matmul
use rust_model_inference::ops::kernel::q8_0::scalar::matmul_q8_0_quantized_scalar_range;
use rust_model_inference::ops::quantize_q8_0_into;

fn main() {
    let loader = rust_model_inference::GGUFLoader::from_file(
        "/home/liyulingyue/Codes/rust-model-inference/models/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q8_0.gguf"
    ).unwrap();

    // Load BOS embedding
    let emb_tensor = loader
        .tensors()
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .unwrap();
    let emb_bytes = loader.tensor_slice("token_embd.weight").unwrap();
    let emb = dequantize_q8_row(emb_bytes, 1, 2048);

    // Load attn_norm
    let norm_bytes = loader.tensor_slice("blk.0.attn_norm.weight").unwrap();
    let attn_norm = dequantize_f32(norm_bytes);

    // RMSNorm
    let sum_sq: f64 = emb.iter().map(|&x| (x as f64).powi(2)).sum();
    let mean_sq = (sum_sq / 2048.0) as f32;
    let scale = 1.0 / (mean_sq + 1e-5).sqrt();
    let normed: Vec<f32> = emb
        .iter()
        .zip(attn_norm.iter())
        .map(|(&x, &w)| x * scale * w)
        .collect();
    println!("normed first 8: {:?}", &normed[..8]);

    // Load in_proj weight (Q8_0)
    let weight = loader
        .tensor_slice("blk.0.shortconv.in_proj.weight")
        .unwrap();
    let n_in = 2048usize;
    let n_out = 6144usize;

    // Compute F32 reference: dequantize weight, do F32 matmul
    let weight_f32 = dequantize_q8_full(weight, n_out, n_in);
    let mut out_f32 = vec![0f32; n_out];
    for r in 0..n_out {
        let mut sum = 0.0f32;
        for k in 0..n_in {
            sum += weight_f32[r * n_in + k] * normed[k];
        }
        out_f32[r] = sum;
    }
    let b_f32 = &out_f32[..2048];
    let x_f32 = &out_f32[4096..];

    // Quantize normed to Q8_0
    let mut input_q8 = vec![0u8; 2048];
    let mut input_scales = vec![0f32; 64];
    quantize_q8_0_into(&normed, 2048, &mut input_q8, &mut input_scales);

    // Run our Q8 matmul
    let mut out_q8 = vec![0f32; n_out];
    matmul_q8_0_quantized_scalar_range(
        &weight,
        &input_q8,
        &input_scales,
        &mut out_q8,
        n_in,
        0,
        n_out,
    );

    // Manually compute out[1] and out[4097] using F32 reference
    let blocks_per_row = 64;
    let bytes_per_row = 2176;

    let row1_off = 1 * bytes_per_row;
    let mut out1_f32 = 0.0f32;
    for b in 0..blocks_per_row {
        let off = row1_off + b * 34;
        let scale =
            half::f16::from_bits(u16::from_le_bytes([weight[off], weight[off + 1]])).to_f32();
        for j in 0..32 {
            let q = weight[off + 2 + j] as i8 as f32;
            out1_f32 += scale * q * normed[b * 32 + j];
        }
    }
    println!("\nManual out[1] F32 (recomputed) = {:.6}", out1_f32);
    println!("out_f32[1] from matmul       = {:.6}", out_f32[1]);
    println!("out_q8[1] from Q8 matmul     = {:.6}", out_q8[1]);

    let row4097_off = 4097 * bytes_per_row;
    let mut out4097_f32 = 0.0f32;
    for b in 0..blocks_per_row {
        let off = row4097_off + b * 34;
        let scale =
            half::f16::from_bits(u16::from_le_bytes([weight[off], weight[off + 1]])).to_f32();
        for j in 0..32 {
            let q = weight[off + 2 + j] as i8 as f32;
            out4097_f32 += scale * q * normed[b * 32 + j];
        }
    }
    println!("\nManual out[4097] F32 (recomputed) = {}", out4097_f32);
    println!("out_f32[4097] (x[1]) = {}", out_f32[4097]);
    println!("out_q8[4097] (x[1] Q8) = {}", out_q8[4097]);

    // Also check: in our code, x[1] = out_q8[4097]?
    println!("\nCheck: if bx[0,1] (Q8) = b[1] * x[1]");
    println!("  b[1] (Q8) = {}", out_q8[1]);
    println!("  x[1] (Q8) = {}", out_q8[4097]);
    println!("  b[1] * x[1] = {}", out_q8[1] * out_q8[4097]);
    println!("Llama.cpp bx[0,1] = -1.39759779");
    println!("\nCheck: if bx[0,1] (F32) = b[1] * x[1]");
    println!("  b[1] (F32) = {}", out_f32[1]);
    println!("  x[1] (F32) = {}", out_f32[4097]);
    println!("  b[1] * x[1] = {}", out_f32[1] * out_f32[4097]);

    // Also manually compute out[4098] (x[2] for BOS)
    let row4098_off = 4098 * bytes_per_row;
    let mut out4098_f32 = 0.0f32;
    let mut debug_scales = vec![];
    let mut debug_qs = vec![];
    for b in 0..blocks_per_row {
        let off = row4098_off + b * 34;
        let scale =
            half::f16::from_bits(u16::from_le_bytes([weight[off], weight[off + 1]])).to_f32();
        debug_scales.push(scale);
        let mut qsum = 0i32;
        let mut qs_arr = Vec::new();
        for j in 0..32 {
            let q = weight[off + 2 + j] as i8;
            qs_arr.push(q);
            qsum += (q as i32) * (normed[b * 32 + j] as i32);
            out4098_f32 += scale * (q as f32) * normed[b * 32 + j];
        }
        debug_qs.push(qs_arr);
    }
    // Final summary
    println!("\n=== SUMMARY ===");
    println!("Llama.cpp (transposed) row 0 (BOS):");
    println!("  bx[0,0] = -0.00142849");
    println!("  bx[0,1] = -1.39759779");
    println!("  bx[0,2] = -0.00581596");
    println!("  bx[0,2045] = -0.00581596");
    println!("  bx[0,2046] = -0.00043886");
    println!("  bx[0,2047] = -0.00581596");

    println!("\nMy Rust (F32 matmul):");
    println!("  bx[0,0] = {}", b_f32[0]);
    println!("  bx[0,1] = {} (vs llama -1.3976)", b_f32[1]);
    println!("  bx[0,2] = {} (vs llama -0.0058)", b_f32[2]);
    println!("  bx[0,2045] = {} (vs llama -0.0058)", b_f32[2045]);
    println!("  bx[0,2046] = {} (vs llama -0.00044)", b_f32[2046]);
    println!("  bx[0,2047] = {} (vs llama -0.0058)", b_f32[2047]);

    println!("\nMy Q8 matmul:");
    println!("  bx[0,0] = {}", out_q8[0] * out_q8[4096 + 0]);
    println!(
        "  bx[0,1] = {} (vs llama -1.3976)",
        out_q8[1] * out_q8[4097]
    );
    println!(
        "  bx[0,2] = {} (vs llama -0.0058)",
        out_q8[2] * out_q8[4098]
    );
    println!("  bx[0,2045] = {}", out_q8[2045] * out_q8[4096 + 2045]);
    println!("  bx[0,2046] = {}", out_q8[2046] * out_q8[4096 + 2046]);
    println!("  bx[0,2047] = {}", out_q8[2047] * out_q8[4096 + 2047]);

    println!(
        "\nMy bcx[0, 0..3] = [{}, {}, {}]",
        out_q8[0], out_q8[1], out_q8[2]
    );
    println!("Llama.cpp bcx[0, 0..3] = [0.0940, -0.4133, 0.0687]");

    // Now look at the difference more carefully. Llama.cpp bx[0, 1] is 108x larger than mine.
    // My b[0,1] (Q8) = -0.4133 (matches llama b).
    // So llama.cpp x[0, 1] = -1.3976 / -0.4133 = 3.38.
    // But my x[0,1] (Q8) = 0.0276 (matmul output).
    // 3.38 vs 0.0276 = 122x difference.

    // Verify: my x[0,1] from the F32 matmul
    println!("\nMy x[1] (F32 matmul): {}", out_f32[4097]);
    println!("Llama.cpp inferred x[1] = 3.38");

    // This is suspicious. Let me check the in_proj matmul output for the FULL x chunk.
    println!("\nMy x[0..8] (F32): {:?}", &out_f32[4096..4104]);
    println!("Llama.cpp shows bcx (transposed) row 0 first 3 + bx = b*x");

    // If my x is correct, then bx = b*x should be ~ -0.0114 at position 1, NOT -1.3976.
    // So llama.cpp's bx[0, 1] = -1.3976 must NOT be the actual b[1] * x[1].

    // Possible explanation: the dump might be displaying x (or something else) instead of bx.
    // OR my Q8 matmul is correct but the value -1.3976 is from a DIFFERENT computation.

    println!(
        "\nDebug: weight[4098] scales (first 5): {:?}",
        &debug_scales[..5]
    );
    println!("Debug: weight[4098] qs first block: {:?}", &debug_qs[0]);

    // Run our Q8 matmul
    let mut out_q8 = vec![0f32; n_out];
    matmul_q8_0_quantized_scalar_range(
        &weight,
        &input_q8,
        &input_scales,
        &mut out_q8,
        n_in,
        0,
        n_out,
    );

    // Compare
    let mut max_diff = 0f32;
    let mut max_idx = 0;
    for i in 0..n_out {
        let d = (out_f32[i] - out_q8[i]).abs();
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
    }
    println!("\nQ8 vs F32: max_diff={} at index {}", max_diff, max_idx);
    println!(
        "out_f32[max_idx]={}, out_q8[max_idx]={}",
        out_f32[max_idx], out_q8[max_idx]
    );

    // Print first 5 values
    println!("\nFirst 8 values:");
    for i in 0..8 {
        println!(
            "  i={}: F32={:.6}, Q8={:.6}, diff={:.6}",
            i,
            out_f32[i],
            out_q8[i],
            out_f32[i] - out_q8[i]
        );
    }

    // Print b[0..3], b[2045..2047] for BOS
    println!("\nb[0..3] (channel 0..3 of b chunk):");
    for i in 0..3 {
        println!("  i={}: F32={:.6}, Q8={:.6}", i, out_f32[i], out_q8[i]);
    }
    println!("\nb[2045..2047]:");
    for i in 2045..2047 {
        println!("  i={}: F32={:.6}, Q8={:.6}", i, out_f32[i], out_q8[i]);
    }

    // Now compute bx = b * x (the gating)
    let b = &out_q8[..2048];
    let c = &out_q8[2048..4096];
    let x = &out_q8[4096..];

    // Print x[0..8] for verification
    println!("\nx[0..8] for BOS (Q8):");
    for i in 0..8 {
        println!("  x[{}] = {:.6}", i, x[i]);
    }
    println!("x[2045..2047] for BOS (Q8):");
    for i in 2045..2047 {
        println!("  x[{}] = {:.6}", i, x[i]);
    }

    let bx: Vec<f32> = b.iter().zip(x.iter()).map(|(bi, xi)| bi * xi).collect();

    println!("\nbx first 8:");
    for i in 0..8 {
        println!("  bx[{}] = {:.6}", i, bx[i]);
    }
    println!("bx[2045..2048]:");
    for i in 2045..2048 {
        println!("  bx[{}] = {:.6}", i, bx[i]);
    }

    // Now compute conv_out = bx * kernel[2] (state=zeros)
    let conv_bytes = loader.tensor_slice("blk.0.shortconv.conv.weight").unwrap();
    let kernel = dequantize_f32(conv_bytes);
    let conv_out: Vec<f32> = bx
        .iter()
        .zip(kernel[2..].chunks_exact(3).next().unwrap().iter())
        .map(|(bxi, ki)| bxi * ki)
        .take(2048)
        .collect();

    // Actually kernel has shape (3, 2048). kernel[2] is row 2.
    let mut kernel_row2 = vec![0f32; 2048];
    for j in 0..2048 {
        kernel_row2[j] = kernel[2 * 2048 + j];
    }
    let conv_out: Vec<f32> = bx
        .iter()
        .zip(kernel_row2.iter())
        .map(|(bxi, ki)| bxi * ki)
        .collect();

    println!("\nconv_out (state=zeros, token=BOS) first 8:");
    for i in 0..8 {
        println!("  conv_out[{}] = {:.6}", i, conv_out[i]);
    }

    // Now compute y = c * conv_out
    let y: Vec<f32> = c
        .iter()
        .zip(conv_out.iter())
        .map(|(ci, oi)| ci * oi)
        .collect();
    println!("\ny first 8:");
    for i in 0..8 {
        println!("  y[{}] = {:.6}", i, y[i]);
    }
}

fn dequantize_q8_row(bytes: &[u8], row: usize, n_in: usize) -> Vec<f32> {
    let blocks_per_row = n_in / 32;
    let bytes_per_row = blocks_per_row * 34;
    let row_data = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
    let mut out = vec![0f32; n_in];
    for b in 0..blocks_per_row {
        let block = &row_data[b * 34..(b + 1) * 34];
        let scale = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        for j in 0..32 {
            out[b * 32 + j] = scale * (block[2 + j] as i8) as f32;
        }
    }
    out
}

fn dequantize_q8_full(bytes: &[u8], n_out: usize, n_in: usize) -> Vec<f32> {
    let blocks_per_row = n_in / 32;
    let bytes_per_row = blocks_per_row * 34;
    let mut out = vec![0f32; n_out * n_in];
    for r in 0..n_out {
        let row_data = &bytes[r * bytes_per_row..(r + 1) * bytes_per_row];
        for b in 0..blocks_per_row {
            let block = &row_data[b * 34..(b + 1) * 34];
            let scale = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            for j in 0..32 {
                out[r * n_in + b * 32 + j] = scale * (block[2 + j] as i8) as f32;
            }
        }
    }
    out
}

fn dequantize_f32(bytes: &[u8]) -> Vec<f32> {
    let mut out = vec![0f32; bytes.len() / 4];
    for i in 0..out.len() {
        out[i] = f32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
    }
    out
}
