use rust_model_inference::GGUFLoader;

fn dump(path: &str) {
    println!("=== {path} ===");
    let loader = GGUFLoader::from_file(path).expect("open");
    println!("metadata keys:");
    for (k, v) in loader.metadata_entries() {
        let headline = format!("  {k} = {v:?}");
        if headline.len() > 220 {
            println!("  {k} = [long value, {} bytes]", headline.len());
        } else {
            println!("{headline}");
        }
    }
    println!("tensors ({}):", loader.tensors().len());
    for t in loader.tensors() {
        let dims: Vec<u64> = t.dims.iter().map(|d| *d).collect();
        println!("  {:<60} dims={:?} type={:?}", t.name, dims, t.ggml_type);
    }
    println!();
}

fn main() {
    for p in std::env::args().skip(1) {
        dump(&p);
    }
}
