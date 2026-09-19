use rust_model_inference::GGUFLoader;
use std::io::Write;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let mmproj_path = std::env::args().nth(1).expect("mmproj path");
    let image_path = std::env::args().nth(2).expect("image path");
    let out_path = std::env::args().nth(3).expect("output bin path");

    let loader = GGUFLoader::from_file(&mmproj_path).map_err(|e| format!("{e:?}"))?;
    let source: Box<dyn rust_model_inference::TensorSource> = Box::new(loader);
    let model = rust_model_inference::models::gemma4::Gemma4UvVisionModel::from_source(
        source.as_ref(),
        16,
    )?;
    let emb = model.encode_path(&PathBuf::from(&image_path))?;
    let n_embd = model.config.embd;
    let n_tokens = emb.len() / n_embd;
    eprintln!(
        "[dump-uv-emb] n_tokens={}, n_embd={}, total_floats={}",
        n_tokens,
        n_embd,
        emb.len()
    );
    eprintln!("[dump-uv-emb] token 0 first 8: {:?}", &emb[..8]);
    let mut f = std::fs::File::create(&out_path).map_err(|e| e.to_string())?;
    let mut hdr = [0u8; 8];
    hdr[..4].copy_from_slice(&(n_tokens as i32).to_le_bytes());
    hdr[4..].copy_from_slice(&(n_embd as i32).to_le_bytes());
    f.write_all(&hdr).map_err(|e| e.to_string())?;
    for v in &emb {
        f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}