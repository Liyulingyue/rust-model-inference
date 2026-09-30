//! Smoke integration test for `fastino/gliner2.5-base-v1` GGUF.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=/path/to/gliner2.5-base-v1-f32.gguf`.
//!
//! BoundaryExtractor is a multi-task model (boundary + relation + record +
//! count + abstention heads) that lives in
//! `target/gliner2-oracle/gliner2/models/boundary/` — 8149 lines of
//! reference Python. Only the encoder side is wired into Rust today; this
//! test only verifies that the bundled GGUF:
//!
//! 1. Carries the expected encoder dims (DeBERTa-v3-base: 12 layers,
//!    768 hidden, 12 heads, 3072 ff, 256 pos buckets).
//! 2. Carries every bundled head tensor (boundary_head / relation_scorer
//!    / record_decoder) with their original safetensors names so a future
//!    Rust BoundaryExtractor forward can pick them up directly.
//! 3. Records the boundary variant in metadata (`gliner2.variant =
//!    "boundary"`, `classifier.last_layer_index = 3`) so the Rust loader
//!    can distinguish boundary from Decide at load time.
//!
//! Real byte-exact parity (boundary detection + relation scoring + record
//! decoding) is out of scope until a Rust BoundaryExtractor forward is
//! written. See `glinerTODO.md`.

use rust_model_inference::core::loader::GGUFLoader;

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var_os("RMI_GLINER2_5_BASE_V1_GGUF").map(std::path::PathBuf::from)
}

fn loader() -> Option<GGUFLoader> {
    let path = gguf_path()?;
    if !path.exists() {
        eprintln!("skipping: {} does not exist", path.display());
        return None;
    }
    Some(GGUFLoader::from_file(path).expect("open boundary GGUF"))
}

#[test]
fn metadata_pins_boundary_variant() {
    let Some(loader) = loader() else { return };
    let arch = loader
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(arch, "gliner2");
    let variant = loader
        .metadata("gliner2.variant")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(variant, "boundary");
    let last_layer = loader
        .metadata("gliner2.classifier.last_layer_index")
        .and_then(|v| v.to_u64())
        .unwrap_or_default();
    assert_eq!(last_layer, 3, "boundary classifier lives at index 3, not 2");
}

#[test]
fn encoder_dimensions_match_deberta_v3_base() {
    let Some(loader) = loader() else { return };
    assert_eq!(
        loader.metadata("gliner2.embedding_length").and_then(|v| v.to_u64()),
        Some(768),
        "base-v1 uses DeBERTa-v3-base (768 hidden)"
    );
    assert_eq!(
        loader.metadata("gliner2.block_count").and_then(|v| v.to_u64()),
        Some(12),
        "base-v1 is 12 layers"
    );
    assert_eq!(
        loader.metadata("gliner2.attention.head_count").and_then(|v| v.to_u64()),
        Some(12),
        "base-v1 is 12 heads"
    );
    assert_eq!(
        loader.metadata("gliner2.feed_forward_length").and_then(|v| v.to_u64()),
        Some(3072),
        "base-v1 FF is 3072 (4× hidden)"
    );
    assert_eq!(
        loader.metadata("gliner2.relative_attention.bucket_size").and_then(|v| v.to_u64()),
        Some(256),
        "DeBERTa position buckets = 256 (so rel_embeddings row count = 512)"
    );
}

#[test]
fn encoder_tensor_shapes_match_deberta_v3_base() {
    let Some(loader) = loader() else { return };
    // Spot-check a few encoder tensor shapes. The full list of 202 encoder
    // tensors is pinned by the converter; if any name drops or remaps,
    // the 6/6 decide parity test still passes (those are GGUF tensor
    // names), so this test catches the boundary-specific remap regression.
    let checks: &[(&str, &[u64])] = &[
        ("token_embd.weight", &[128011u64, 768]),
        ("rel_embeddings.weight", &[512u64, 768]),
        ("tok_norm.weight", &[768u64]),
        ("blk.0.attn_q.weight", &[768u64, 768]),
        ("blk.0.ffn_up.weight", &[3072u64, 768]),
        ("blk.11.attn_output.weight", &[768u64, 768]),
        ("classifier.0.weight", &[1536u64, 768]),
        ("classifier.3.weight", &[1u64, 1536]),
    ];
    for (name, expected_shape) in checks {
        let info = loader
            .tensor_info(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"));
        assert_eq!(
            info.dims, *expected_shape,
            "{name}: shape {:?} != expected {expected_shape:?}",
            info.dims
        );
        assert_eq!(info.ggml_type, rust_model_inference::core::tensor::GGMLType::F32);
    }
}

#[test]
fn bundled_heads_are_present_with_original_names() {
    let Some(loader) = loader() else { return };
    let bundled = loader
        .metadata("gliner2.boundary.bundled_tensor_count")
        .and_then(|v| v.to_u64())
        .unwrap_or_default();
    assert!(
        bundled >= 132,
        "expected at least 132 bundled head tensors (boundary_head 102 + relation_scorer 12 + record_decoder 18); got {bundled}"
    );

    // Spot-check the four buckets: bos/eos states from boundary_encoder,
    // one relation_scorer tensor, one record_decoder tensor, and the
    // count_head scalar projection.
    for (name, expected_shape) in [
        ("boundary_head.boundary_encoder.bos_state", vec![768u64]),
        ("boundary_head.boundary_encoder.eos_state", vec![768u64]),
        ("boundary_head.count_head.weight", vec![1u64, 768]),
        ("boundary_head.null_projection.weight", vec![1u64, 768]),
        ("relation_scorer.head_content_projection.weight", vec![768u64, 768]),
        ("record_decoder.field_proj.weight", vec![128u64, 768]),
    ] {
        let info = loader
            .tensor_info(name)
            .unwrap_or_else(|| panic!("missing bundled tensor {name}"));
        assert_eq!(
            info.dims, expected_shape,
            "{name}: shape {:?} != expected {expected_shape:?}",
            info.dims
        );
    }
}

#[test]
fn tokenizer_is_sentencepiece_with_unigram_vocab() {
    let Some(loader) = loader() else { return };
    assert_eq!(
        loader.metadata("tokenizer.ggml.model").and_then(|v| v.to_string_val()).as_deref(),
        Some("hf-json"),
        "boundary family uses the HF fast tokenizer (Unigram SPM wrapped in HF JSON)"
    );
    let vocab = loader
        .metadata("tokenizer.ggml.vocab_size")
        .and_then(|v| v.to_u64())
        .unwrap_or_default();
    assert_eq!(vocab, 128011);
    // `tokenizer_json` should be present as a GGUF string metadata blob;
    // the bytes contain the SPM vocab + added-token map.
    assert!(
        loader.metadata("gliner2.tokenizer_json").is_some(),
        "missing gliner2.tokenizer_json metadata"
    );
    let sha = loader
        .metadata("gliner2.tokenizer_sha256")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(sha.len(), 64, "tokenizer_sha256 should be 64 hex chars");
}