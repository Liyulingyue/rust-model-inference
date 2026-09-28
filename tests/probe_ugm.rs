use rust_model_inference::core::tokenizer::{EncodeOptions, Tokenizer};
use rust_model_inference::core::ugm::UgmTokenizer;

fn loader() -> Option<rust_model_inference::GGUFLoader> {
    let path = std::env::var_os("RMI_NOMIC_EMBED_TEXT_V2_MOE_MODEL")?;
    Some(rust_model_inference::GGUFLoader::from_file(path).unwrap())
}

#[test]
fn probe_ugm_tokenize() {
    let Some(g) = loader() else { return };
    let t = UgmTokenizer::from_gguf_metadata(|k| g.metadata(k).cloned()).expect("ugm");
    eprintln!("vocab={} unk_id={}", t.vocab_size(), t.unk_id());

    // What does normalize() produce?
    let norm = t.normalize("What is the capital of France?".as_bytes());
    eprintln!("normalized = {:?}", String::from_utf8_lossy(&norm));

    // What does the trie contain for some keys?
    for kw in &["▁What", "▁is", "▁the", "▁capital", "What", "is", "the"] {
        let mut node = &t.matcher;
        let bytes = kw.as_bytes();
        let mut depth = 0;
        for &b in bytes {
            let c = b as char;
            match node.traverse(c) {
                Some(n) => {
                    node = n;
                    depth += 1;
                }
                None => {
                    eprintln!("  trie lookup '{:?}' failed at depth {}", kw, depth);
                    break;
                }
            }
        }
        if node.value.is_some() {
            eprintln!(
                "  trie lookup {:?} -> id={:?} (depth={})",
                kw, node.value, depth
            );
        }
    }

    let ids = t.encode_with_options("What is the capital of France?", false, false);
    eprintln!(
        "ids(no_specials,len={}): {:?}",
        ids.len(),
        &ids[..ids.len().min(40)]
    );
}
