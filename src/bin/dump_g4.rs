use rust_model_inference::GGUFLoader;

fn main() {
    let path = std::env::args().nth(1).expect("path");
    let loader = GGUFLoader::from_file(&path).expect("open");
    let target: Vec<usize> = std::env::args()
        .nth(2)
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![0, 1, 5, 6, 11]);
    println!("=== TENSORS for layers {:?} ===", target);
    for t in loader.tensors() {
        let name = &t.name;
        for &layer in &target {
            let prefix = format!("blk.{layer}.");
            if let Some(rest) = name.strip_prefix(&prefix) {
                let dims: Vec<u64> = t.dims.iter().map(|d| *d).collect();
                println!("blk.{layer}.{} dims={:?} type={:?}", rest, dims, t.ggml_type);
                break;
            }
        }
    }
    println!("=== ALL NON-BLK TENSORS ===");
    for t in loader.tensors() {
        let name = &t.name;
        if !name.starts_with("blk.") {
            let dims: Vec<u64> = t.dims.iter().map(|d| *d).collect();
            println!("{} dims={:?} type={:?}", name, dims, t.ggml_type);
        }
    }
}
