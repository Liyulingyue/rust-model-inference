//! Integration tests for the DeepCybo/PhysBrain1.5-2B GGUF (converted from
//! `Qwen3-VL-2B-Instruct` backbone fine-tuned with ActionPiece + visual-state
//! tokens).
//!
//! Run with:
//!   RMI_PHYSBRAIN_1_5_2B_TEXT=/path/to/physbrain-1.5-2b-Q8_0.gguf \
//!   RMI_PHYSBRAIN_1_5_2B_MMPROJ=/path/to/mmproj-physbrain-1.5-2b-F16.gguf \
//!   cargo test --profile release-fast --test physbrain_1_5_2b
//!
//! PhysBrain rides the `qwen3vl` text path and the `qwen3vl_merger` mmproj
//! path — both already supported by the engine (the `Qwen3-VL-2B 配置`
//! whitelist). The only PhysBrain-specific surface area is the tokenizer
//! vocab extension to 168576 tokens (vs 151936 base Qwen3) and the
//! deepstack visual indexes `[5, 11, 17]` in the mmproj.
//!
//! The test pins loader + tokenizer + tensor contract; no end-to-end image
//! Oracle (the contract surface is enough to detect a bad converter run).

use rust_model_inference::GGUFLoader;
use rust_model_inference::TensorInfo;

fn text_loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_PHYSBRAIN_1_5_2B_TEXT")?;
    GGUFLoader::from_file(path).ok()
}

fn mmproj_loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_PHYSBRAIN_1_5_2B_MMPROJ")?;
    GGUFLoader::from_file(path).ok()
}

fn pick(loader: &GGUFLoader, key: &str) -> usize {
    loader
        .metadata(key)
        .and_then(|v| v.to_u64())
        .map(|v| v as usize)
        .unwrap_or(0)
}

fn pick_str<'a>(loader: &'a GGUFLoader, key: &str) -> Option<&'a str> {
    loader.metadata(key).and_then(|v| match v {
        rust_model_inference::core::tensor::MetaValue::String(s) => Some(s.as_str()),
        _ => None,
    })
}

fn shape<'a>(loader: &'a GGUFLoader, name: &str) -> Option<&'a TensorInfo> {
    loader.tensor_info(name)
}

#[test]
fn text_arch_and_dims_match_qwen3vl_2b_whitelist() {
    let Some(loader) = text_loader() else { return };
    assert_eq!(
        pick_str(&loader, "general.architecture"),
        Some("qwen3vl"),
        "PhysBrain text GGUF must declare general.architecture=qwen3vl"
    );
    assert_eq!(pick(&loader, "qwen3vl.embedding_length"), 2048);
    assert_eq!(pick(&loader, "qwen3vl.block_count"), 28);
    assert_eq!(pick(&loader, "qwen3vl.attention.head_count"), 16);
    assert_eq!(pick(&loader, "qwen3vl.attention.head_count_kv"), 8);
    assert_eq!(pick(&loader, "qwen3vl.feed_forward_length"), 6144);
    assert_eq!(pick(&loader, "qwen3vl.context_length"), 262_144);
    assert_eq!(pick(&loader, "qwen3vl.vocab_size"), 168_576);
    let rope_base: f64 = loader
        .metadata("qwen3vl.rope.freq_base")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    assert!(
        (rope_base - 5_000_000.0).abs() < 1.0,
        "rope.freq_base={rope_base}"
    );
    let eps: f64 = loader
        .metadata("qwen3vl.attention.layer_norm_rms_epsilon")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    assert!((eps - 1e-6).abs() < 1e-9, "norm_eps={eps}");
}

#[test]
fn text_rope_sections_match_qwen3vl_mrope() {
    let Some(loader) = text_loader() else { return };
    let sections = loader
        .metadata("qwen3vl.rope.dimension_sections")
        .and_then(|v| v.to_arr())
        .expect("qwen3vl.rope.dimension_sections");
    let section_vec: Vec<i64> = sections
        .iter()
        .filter_map(|v| v.to_u64().map(|x| x as i64))
        .collect();
    assert_eq!(section_vec, vec![24, 20, 20, 0]);
}

#[test]
fn text_image_token_ids_match_physbrain_specials() {
    let Some(loader) = text_loader() else { return };
    assert_eq!(pick(&loader, "qwen3vl.image_token_id"), 151_655);
    assert_eq!(pick(&loader, "qwen3vl.vision_start_token_id"), 151_652);
    assert_eq!(pick(&loader, "qwen3vl.vision_end_token_id"), 151_653);
}

#[test]
fn text_tied_embedding_omits_output_weight() {
    // PhysBrain sets tie_word_embeddings=true in HF config; the engine
    // constructs `output` from `token_embd.weight` (see
    // `src/models/qwen3/trunk/weights.rs`). Verifying absence of
    // `output.weight` is the contract surface.
    let Some(loader) = text_loader() else { return };
    assert!(
        shape(&loader, "output.weight").is_none(),
        "PhysBrain uses tied embeddings; output.weight must be absent"
    );
    let embed = shape(&loader, "token_embd.weight").expect("token_embd.weight");
    // GGUF dims are reversed; this is [n_embd, vocab_size] in GGUF order.
    assert_eq!(embed.dims, vec![2048, 168_576]);
}

#[test]
fn text_layer_inventory_is_complete() {
    let Some(loader) = text_loader() else { return };
    let n_layer = 28;
    let expected_names = [
        "attn_norm",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "attn_q_norm",
        "attn_k_norm",
        "ffn_norm",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ];
    for i in 0..n_layer {
        for name in expected_names {
            assert!(
                shape(&loader, &format!("blk.{i}.{name}.weight")).is_some(),
                "missing blk.{i}.{name}.weight",
            );
        }
    }
    assert!(shape(&loader, "output_norm.weight").is_some());
}

#[test]
fn text_tokenizer_vocab_is_extended_to_168576() {
    let Some(loader) = text_loader() else { return };
    let n_tokens = loader
        .metadata("tokenizer.ggml.tokens")
        .and_then(|v| v.to_arr())
        .map(|arr| arr.len())
        .unwrap_or(0);
    assert_eq!(
        n_tokens, 168_576,
        "PhysBrain vocab extends the base Qwen3 151643 + 16933 added tokens",
    );
    // ActionPiece + visual-state tokens must be CONTROL type (3).
    let types = loader
        .metadata("tokenizer.ggml.token_type")
        .and_then(|v| v.to_arr())
        .expect("tokenizer.ggml.token_type");
    let type_vec: Vec<i64> = types
        .iter()
        .filter_map(|v| v.to_u64().map(|x| x as i64))
        .collect();
    assert_eq!(type_vec.len(), 168_576);
    let control_count = type_vec.iter().filter(|&&t| t == 3).count();
    // The HF `added_tokens.json` has 16926 entries (151643-168568), but
    // any beyond vocab_size are dropped by the engine. 16926 is the
    // upper bound on CONTROL count.
    assert!(
        control_count >= 168_568 - 151_643 + 1,
        "expected at least 16926 CONTROL tokens, got {control_count}",
    );
}

#[test]
fn mmproj_arch_is_clip_with_qwen3vl_merger() {
    let Some(loader) = mmproj_loader() else { return };
    assert_eq!(pick_str(&loader, "general.architecture"), Some("clip"));
    assert_eq!(
        pick_str(&loader, "clip.projector_type"),
        Some("qwen3vl_merger"),
    );
    // clip.has_vision_encoder is stored as a GGUF bool; MetaValue exposes
    // it as `Bool(_)`, and pick() (which uses to_u64) returns 0 for bool.
    // Read the bool directly via `Bool(true)`.
    let has_ve = loader
        .metadata("clip.has_vision_encoder")
        .is_some_and(|v| matches!(
            v,
            rust_model_inference::core::tensor::MetaValue::Bool(true)
        ));
    assert!(has_ve, "clip.has_vision_encoder must be Bool(true)");
}

#[test]
fn mmproj_vision_dims_match_qwen3_vl_2b() {
    let Some(loader) = mmproj_loader() else { return };
    assert_eq!(pick(&loader, "clip.vision.embedding_length"), 1024);
    assert_eq!(pick(&loader, "clip.vision.feed_forward_length"), 4096);
    assert_eq!(pick(&loader, "clip.vision.block_count"), 24);
    assert_eq!(pick(&loader, "clip.vision.attention.head_count"), 16);
    assert_eq!(pick(&loader, "clip.vision.image_size"), 768);
    assert_eq!(pick(&loader, "clip.vision.patch_size"), 16);
    assert_eq!(pick(&loader, "clip.vision.spatial_merge_size"), 2);
    assert_eq!(pick(&loader, "clip.vision.projection_dim"), 2048);
}

#[test]
fn mmproj_deepstack_indexes_match_physbrain_5_11_17() {
    let Some(loader) = mmproj_loader() else { return };
    let arr = loader
        .metadata("clip.vision.is_deepstack_layers")
        .and_then(|v| v.to_arr())
        .expect("clip.vision.is_deepstack_layers");
    let bools: Vec<i64> = arr
        .iter()
        .filter_map(|v| match v {
            rust_model_inference::core::tensor::MetaValue::Bool(b) => Some(if *b { 1 } else { 0 }),
            rust_model_inference::core::tensor::MetaValue::Uint64(u) => Some(*u as i64),
            _ => None,
        })
        .collect();
    assert_eq!(bools.len(), 24, "24 ViT blocks");
    assert_eq!(bools[5], 1, "block 5 is a deepstack layer");
    assert_eq!(bools[11], 1, "block 11 is a deepstack layer");
    assert_eq!(bools[17], 1, "block 17 is a deepstack layer");
}

#[test]
fn mmproj_deepstack_tensors_exist_for_each_index() {
    let Some(loader) = mmproj_loader() else { return };
    for i in [5usize, 11, 17] {
        for kind in ["norm", "fc1", "fc2"] {
            assert!(
                shape(&loader, &format!("v.deepstack.{i}.{kind}.weight")).is_some(),
                "missing v.deepstack.{i}.{kind}.weight",
            );
            assert!(
                shape(&loader, &format!("v.deepstack.{i}.{kind}.bias")).is_some(),
                "missing v.deepstack.{i}.{kind}.bias",
            );
        }
    }
}

#[test]
fn mmproj_conv3d_split_into_two_conv2d_weights() {
    // patch_embed.proj is shape (1024, 3, 2, 16, 16) in HF — the converter
    // splits along the temporal axis into `v.patch_embd.weight` and
    // `.weight.1`. Both must be shape (16, 16, 3, 1024) after the
    // (kH, kW, in, out) GGUF reorder.
    let Some(loader) = mmproj_loader() else { return };
    // gguf_dims reverses torch dims into GGUF storage order; the reader
    // returns the storage order verbatim, so torch (1024, 3, 16, 16) shows
    // up here as (16, 16, 3, 1024).
    let w0 = shape(&loader, "v.patch_embd.weight").expect("v.patch_embd.weight");
    let w1 = shape(&loader, "v.patch_embd.weight.1").expect("v.patch_embd.weight.1");
    assert_eq!(w0.dims, vec![16, 16, 3, 1024]);
    assert_eq!(w1.dims, vec![16, 16, 3, 1024]);
    let bias = shape(&loader, "v.patch_embd.bias").expect("v.patch_embd.bias");
    assert_eq!(bias.dims, vec![1024]);
}

#[test]
fn mmproj_merger_tensors_match_qwen3vl_merger_contract() {
    let Some(loader) = mmproj_loader() else { return };
    let mm0 = shape(&loader, "mm.0.weight").expect("mm.0.weight");
    let mm2 = shape(&loader, "mm.2.weight").expect("mm.2.weight");
    // mm.0: [merged_embd, merged_embd] = [n_embd*merge*merge, n_embd*merge*merge] = [4096, 4096]
    // mm.2: [merged_embd, projection_dim] = [4096, 2048]
    assert_eq!(mm0.dims, vec![4096, 4096]);
    assert_eq!(mm2.dims, vec![4096, 2048]);
    let post_ln = shape(&loader, "v.post_ln.weight").expect("v.post_ln.weight");
    // post_ln operates per-token on n_embd=1024; weight must be 1024 long.
    assert_eq!(post_ln.dims, vec![1024]);
}

