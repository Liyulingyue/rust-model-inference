use rust_model_inference::GGUFLoader;

fn main() {
    let path = std::env::args().nth(1).expect("path");
    let loader = GGUFLoader::from_file(&path).expect("open");
    for (k, v) in loader.metadata_entries() {
        println!("{k} = {v:?}");
    }
    println!("=== TENSORS ===");
    for t in loader.tensors() {
        let dims: Vec<u64> = t.dims.iter().map(|d| *d).collect();
        println!("{} dims={:?} type={:?}", t.name, dims, t.ggml_type);
    }
}
