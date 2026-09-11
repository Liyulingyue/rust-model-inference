use rust_model_inference::GGUFLoader;

fn main() {
    let path = std::env::args().nth(1).expect("path");
    let loader = GGUFLoader::from_file(&path).expect("open");
    for t in loader.tensors() {
        let name = &t.name;
        if name.starts_with("blk.0.") || name.starts_with("blk.5.") || name.starts_with("blk.10.") {
            if name.contains("attn_k.weight") || name.contains("attn_q.weight") {
                let dims: Vec<u64> = t.dims.iter().map(|d| *d).collect();
                println!("{} dims={:?} type={:?}", name, dims, t.ggml_type);
            }
        }
    }
}