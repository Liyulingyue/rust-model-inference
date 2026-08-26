use rust_model_inference::{GGUFLoader, GGMLType, TensorSource};

fn main() {
    let path = std::env::args().nth(1).expect("Usage: dump_emb <model.gguf> [token_id]");
    let token_id: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let loader = GGUFLoader::from_file(&path).expect("open");
    for t in loader.tensors() {
        if t.name == "token_embd.weight" {
            println!("Found token_embd.weight: dims={:?} type={:?}", t.dims, t.ggml_type);
            let n_embd = t.dims[0] as usize;
            let n_vocab = t.dims[1] as usize;
            println!("n_embd={} n_vocab={}", n_embd, n_vocab);

            // Read embedding directly via dequantize_row
            let weight = loader.tensor_slice("token_embd.weight").expect("slice");
            let mut vals = vec![0f32; n_embd];
            rust_model_inference::ops::embedding_lookup(weight, token_id, n_embd, t.ggml_type, &mut vals);
            println!("First 16 embedding values for token {} (via embedding_lookup):", token_id);
            for (i, v) in vals.iter().take(16).enumerate() {
                print!("{:.5} ", v);
                if (i+1) % 8 == 0 { println!(); }
            }
            println!();

            // Manual dump for comparison
            if let GGMLType::Q8_0 = t.ggml_type {
                let blocks_per_token = n_embd / 32;
                let token_off = token_id as usize * blocks_per_token * 34;
                println!("Reading block 0 at offset {} of weight len", token_off);
                for b in 0..4 {
                    let off = token_off + b * 34;
                    let scale_bytes = [weight[off], weight[off+1]];
                    let scale_u16_le = u16::from_le_bytes(scale_bytes);
                    let scale_u16_be = u16::from_be_bytes(scale_bytes);
                    let scale_le = half::f16::from_bits(scale_u16_le).to_f32();
                    let scale_be = half::f16::from_bits(scale_u16_be).to_f32();
                    println!("  block {} off={} bytes={:#04x},{:#04x}", b, off, scale_bytes[0], scale_bytes[1]);
                    println!("    scale_le={:#06x} -> {:.5}", scale_u16_le, scale_le);
                    println!("    scale_be={:#06x} -> {:.5}", scale_u16_be, scale_be);
                    print!("    qs:");
                    for j in 0..8 {
                        print!(" {:3}", weight[off + 2 + j] as i8);
                    }
                    println!();
                }
            }
            return;
        }
    }
    println!("token_embd.weight not found");
}