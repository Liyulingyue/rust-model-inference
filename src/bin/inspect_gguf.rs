use rust_model_inference::GGUFLoader;

fn main() {
    let path = std::env::args().nth(1).expect("gguf path");
    let target = std::env::args().nth(2);
    let loader = GGUFLoader::from_file(&path).expect("load gguf");
    for t in loader.tensors() {
        if let Some(tgt) = &target {
            if t.name == tgt.as_str() {
                use std::io::Write;
                let mut f = std::fs::File::create("dump_tensor.bin").unwrap();
                let data = loader.tensor_slice(&t.name).unwrap();
                f.write_all(data).unwrap();
                println!("wrote dump_tensor.bin ({} bytes)", data.len());
                if t.ggml_type == rust_model_inference::GGMLType::F32 {
                    let floats: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    println!("abs max: {}", floats.iter().map(|x| x.abs()).fold(0.0f32, f32::max));
                    println!("abs mean: {}", floats.iter().map(|x| x.abs()).sum::<f32>() / floats.len() as f32);
                    println!("first 16: {:?}", &floats[..16]);
                }
                return;
            }
        }
    }
    println!("not found");
}