//! Integration tests for `google/embeddinggemma-2` (multimodal encoder:
//! text + vision + audio). The GGUF is produced by
//! `tools/converter/embeddinggemma2/convert_embeddinggemma2.py`.
//!
//! Run with:
//!   RMI_EMBEDDINGGEMMA_2=/path/to/embeddinggemma-2-F16.gguf \
//!   cargo test --profile release-fast --test embeddinggemma_2
//!
//! This test pins the converter's metadata contract + tensor inventory.
//! End-to-end text/forward/multimodal Oracle is out of scope for this
//! test — the engine does not yet implement the `gemma4-embedding` arch
//! forward path (see TODO.md `EmbeddingGemma2 forward path`).

use rust_model_inference::GGUFLoader;
use rust_model_inference::TensorInfo;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_EMBEDDINGGEMMA_2")?;
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
fn arch_is_gemma4_embedding() {
    let Some(loader) = loader() else { return };
    assert_eq!(
        pick_str(&loader, "general.architecture"),
        Some("gemma4-embedding")
    );
}

#[test]
fn text_backbone_dims_match_embeddinggemma_2() {
    let Some(loader) = loader() else { return };
    assert_eq!(pick(&loader, "gemma4-embedding.embedding_length"), 512);
    assert_eq!(pick(&loader, "gemma4-embedding.block_count"), 24);
    assert_eq!(pick(&loader, "gemma4-embedding.attention.head_count"), 4);
    assert_eq!(pick(&loader, "gemma4-embedding.feed_forward_length"), 2048);
    assert_eq!(pick(&loader, "gemma4-embedding.context_length"), 262_144);
    assert_eq!(pick(&loader, "gemma4-embedding.vocab_size"), 262_144);
    assert_eq!(pick(&loader, "gemma4-embedding.embedding_dim"), 768);
    assert_eq!(pick(&loader, "gemma4-embedding.sliding_window"), 512);
}

#[test]
fn text_per_layer_config_dispatch_indices_5_11_17_23() {
    let Some(loader) = loader() else { return };
    let layers = loader
        .metadata("gemma4-embedding.full_attention_layers")
        .and_then(|v| v.to_arr())
        .expect("gemma4-embedding.full_attention_layers");
    let idx: Vec<i64> = layers
        .iter()
        .filter_map(|v| v.to_u64().map(|x| x as i64))
        .collect();
    assert_eq!(idx, vec![5, 11, 17, 23]);

    // GGUF string arrays round-trip as `String` MetaValues, not Uint —
    // parse them via `String(_)` matching.
    let parse_int_array = |key: &str| -> Vec<i64> {
        loader
            .metadata(key)
            .and_then(|v| v.to_arr())
            .unwrap_or_else(|| panic!("{key}"))
            .iter()
            .filter_map(|v| match v {
                rust_model_inference::core::tensor::MetaValue::String(s) => s.parse().ok(),
                _ => None,
            })
            .collect()
    };
    let hd = parse_int_array("gemma4-embedding.attention.head_dim_per_layer");
    assert_eq!(hd.len(), 24);
    for (i, &d) in hd.iter().enumerate() {
        if [5, 11, 17, 23].contains(&i) {
            assert_eq!(
                d, 512,
                "layer {i} should have head_dim=512 (full attention)"
            );
        } else {
            assert_eq!(
                d, 256,
                "layer {i} should have head_dim=256 (sliding attention)"
            );
        }
    }
    let kh = parse_int_array("gemma4-embedding.attention.head_count_kv_per_layer");
    for (i, &k) in kh.iter().enumerate() {
        if [5, 11, 17, 23].contains(&i) {
            assert_eq!(k, 1, "layer {i} should have kv_heads=1 (full attention)");
        } else {
            assert_eq!(k, 2, "layer {i} should have kv_heads=2 (sliding attention)");
        }
    }
}

#[test]
fn text_rope_theta_dispatches_per_layer_type() {
    let Some(loader) = loader() else { return };
    let full: f64 = loader
        .metadata("gemma4-embedding.rope.freq_base_full")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    let sliding: f64 = loader
        .metadata("gemma4-embedding.rope.freq_base_sliding")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    assert!((full - 1_000_000.0).abs() < 1.0, "rope_full={full}");
    assert!((sliding - 10_000.0).abs() < 1.0, "rope_sliding={sliding}");
}

#[test]
fn text_special_token_ids_match_embeddinggemma_2() {
    let Some(loader) = loader() else { return };
    assert_eq!(pick(&loader, "gemma4-embedding.boi_token_id"), 255_999);
    assert_eq!(pick(&loader, "gemma4-embedding.eoi_token_id"), 258_882);
    assert_eq!(pick(&loader, "gemma4-embedding.boa_token_id"), 256_000);
    assert_eq!(pick(&loader, "gemma4-embedding.eoa_token_id"), 258_883);
    assert_eq!(pick(&loader, "gemma4-embedding.image_token_id"), 258_880);
    assert_eq!(pick(&loader, "gemma4-embedding.audio_token_id"), 258_881);
    assert_eq!(pick(&loader, "gemma4-embedding.video_token_id"), 258_884);
}

#[test]
fn text_tokenizer_vocab_is_262144() {
    let Some(loader) = loader() else { return };
    let n_tokens = loader
        .metadata("tokenizer.ggml.tokens")
        .and_then(|v| v.to_arr())
        .map(|arr| arr.len())
        .unwrap_or(0);
    assert_eq!(n_tokens, 262_144);
}

#[test]
fn text_per_layer_block_inventory_complete() {
    let Some(loader) = loader() else { return };
    // Every block must carry: 4 norms + layer_scalar + 3 PLE tensors + 6 attn + 3 ffn.
    let expected = [
        "attn_norm",
        "post_attention_norm",
        "pre_ffn_norm",
        "post_ffn_norm",
        "layer_scalar",
        "ple_input_gate",
        "ple_projection",
        "ple_post_norm",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "attn_q_norm",
        "attn_k_norm",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ];
    for i in 0..24 {
        for name in expected {
            // `layer_scalar` is a 1-value tensor without `.weight` suffix.
            let suffix = if name == "layer_scalar" {
                ""
            } else {
                ".weight"
            };
            let key = format!("blk.{i}.{name}{suffix}");
            assert!(shape(&loader, &key).is_some(), "missing {key}");
        }
    }
    // Shared per-layer vocab projection (vocab-projection) lives at top-level.
    assert!(shape(&loader, "per_layer_model_proj.weight").is_some());
    assert!(shape(&loader, "per_layer_proj_norm.weight").is_some());
    // Text embedding projection to embedding_dim=768.
    assert!(shape(&loader, "text_embproj.weight").is_some());
    // Final RMSNorm.
    assert!(shape(&loader, "output_norm.weight").is_some());
}

#[test]
fn text_sliding_vs_full_attention_q_shape_differs() {
    let Some(loader) = loader() else { return };
    // Sliding (layer 0, head_dim=256, kv_heads=2):
    //   q_proj: [n_head * head_dim, n_embd] = [4*256, 512] = [1024, 512]
    //   k_proj: [n_kv * head_dim, n_embd]  = [2*256, 512] = [512, 512]
    // Full (layer 5, head_dim=512, kv_heads=1):
    //   q_proj: [4*512, 512] = [2048, 512]
    //   k_proj: [1*512, 512] = [512, 512]
    let q0 = shape(&loader, "blk.0.attn_q.weight").expect("blk.0.attn_q.weight");
    let k0 = shape(&loader, "blk.0.attn_k.weight").expect("blk.0.attn_k.weight");
    // gguf_dims reverses dims into GGUF storage order.
    assert_eq!(q0.dims, vec![512, 1024], "sliding q_proj");
    assert_eq!(k0.dims, vec![512, 512], "sliding k_proj");
    let q5 = shape(&loader, "blk.5.attn_q.weight").expect("blk.5.attn_q.weight");
    let k5 = shape(&loader, "blk.5.attn_k.weight").expect("blk.5.attn_k.weight");
    assert_eq!(q5.dims, vec![512, 2048], "full q_proj");
    assert_eq!(k5.dims, vec![512, 512], "full k_proj");
}

#[test]
fn vision_dims_match_gemma4_vision() {
    let Some(loader) = loader() else { return };
    assert_eq!(pick(&loader, "gemma4-embedding.vision.depth"), 16);
    assert_eq!(
        pick(&loader, "gemma4-embedding.vision.embedding_length"),
        768
    );
    assert_eq!(
        pick(&loader, "gemma4-embedding.vision.feed_forward_length"),
        3072
    );
    assert_eq!(
        pick(&loader, "gemma4-embedding.vision.attention.head_count"),
        12
    );
    assert_eq!(pick(&loader, "gemma4-embedding.vision.patch_size"), 16);
    assert_eq!(
        pick(&loader, "gemma4-embedding.vision.spatial_merge_size"),
        3
    );
    assert_eq!(
        pick(&loader, "gemma4-embedding.vision.projection_dim"),
        1536
    );
    assert_eq!(
        pick(&loader, "gemma4-embedding.vision.position_embedding_size"),
        10_240
    );
    assert_eq!(
        pick(&loader, "gemma4-embedding.vision.default_output_length"),
        280
    );
}

#[test]
fn vision_axial_rope_type_marker() {
    let Some(loader) = loader() else { return };
    assert_eq!(
        pick_str(&loader, "gemma4-embedding.vision.rope_type"),
        Some("axial"),
    );
}

#[test]
fn vision_block_inventory_complete() {
    let Some(loader) = loader() else { return };
    let expected = [
        "ln1",
        "pre_ffn_norm",
        "post_attention_norm",
        "post_ffn_norm",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "attn_q_norm",
        "attn_k_norm",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ];
    for i in 0..16 {
        for name in expected {
            assert!(
                shape(&loader, &format!("vision.blk.{i}.{name}.weight")).is_some(),
                "missing vision.blk.{i}.{name}.weight",
            );
        }
    }
    assert!(shape(&loader, "vision.patch_input_proj.weight").is_some());
    let pos =
        shape(&loader, "vision.position_embedding_table").expect("vision.position_embedding_table");
    // Position embedding: (hidden_size, position_embedding_size, 2)
    assert_eq!(pos.dims, vec![768, 10_240, 2]);
}

#[test]
fn audio_dims_match_gemma4_audio() {
    let Some(loader) = loader() else { return };
    assert_eq!(pick(&loader, "gemma4-embedding.audio.depth"), 12);
    assert_eq!(pick(&loader, "gemma4-embedding.audio.hidden_size"), 1024);
    assert_eq!(
        pick(&loader, "gemma4-embedding.audio.ffn_hidden_size"),
        4096
    );
    assert_eq!(pick(&loader, "gemma4-embedding.audio.projection_dim"), 1536);
}

#[test]
fn audio_block_inventory_is_complete() {
    let Some(loader) = loader() else { return };
    // Each Conformer layer carries: lconv1d.{conv_norm,depthwise_conv1d,
    // linear_start,linear_end}.linear.weight + pre_layer_norm + norm_out +
    // norm_pre_attn + norm_post_attn + self_attn.{q,k,v,post}.linear.weight
    // + self_attn.per_dim_scale + self_attn.relative_k_proj.weight +
    // feed_forward1/2.{pre_layer_norm,ffw_layer_1,ffw_layer_2,post_layer_norm}.linear.weight
    for i in 0..12 {
        for suffix in [
            "lconv1d.conv_norm.weight",
            "lconv1d.depthwise_conv1d.weight",
            "lconv1d.linear_end.linear.weight",
            "lconv1d.linear_start.linear.weight",
            "lconv1d.pre_layer_norm.weight",
            "norm_out.weight",
            "norm_pre_attn.weight",
            "norm_post_attn.weight",
            "self_attn.q_proj.linear.weight",
            "self_attn.k_proj.linear.weight",
            "self_attn.v_proj.linear.weight",
            "self_attn.post.linear.weight",
            // `per_dim_scale` is a (128,) vector stored without `.weight`.
            "self_attn.per_dim_scale",
            "self_attn.relative_k_proj.weight",
            "feed_forward1.ffw_layer_1.linear.weight",
            "feed_forward1.ffw_layer_2.linear.weight",
            "feed_forward1.pre_layer_norm.weight",
            "feed_forward1.post_layer_norm.weight",
            "feed_forward2.ffw_layer_1.linear.weight",
            "feed_forward2.ffw_layer_2.linear.weight",
            "feed_forward2.pre_layer_norm.weight",
            "feed_forward2.post_layer_norm.weight",
        ] {
            let key = format!("audio.layers.{i}.{suffix}");
            assert!(shape(&loader, &key).is_some(), "missing {key}");
        }
    }
    // Conformer subsampling convolution (input projection).
    assert!(shape(
        &loader,
        "audio.subsample_conv_projection.input_proj_linear.weight"
    )
    .is_some());
    assert!(shape(
        &loader,
        "audio.subsample_conv_projection.layer0.conv.weight"
    )
    .is_some());
    assert!(shape(
        &loader,
        "audio.subsample_conv_projection.layer0.norm.weight"
    )
    .is_some());
    // Audio output projection to projection_dim=1536.
    assert!(shape(&loader, "audio.output_proj.weight").is_some());
    assert!(shape(&loader, "audio.output_proj.bias").is_some());
}

#[test]
fn multimodal_projectors_present() {
    let Some(loader) = loader() else { return };
    // Image → text soft-token projector (vision 768 → text 512).
    let ve = shape(&loader, "vision_embed.weight").expect("vision_embed.weight");
    assert_eq!(ve.dims, vec![768, 512]);
    // Audio → text soft-token projector (audio 1536 → text 512).
    let ae = shape(&loader, "audio_embed.weight").expect("audio_embed.weight");
    assert_eq!(ae.dims, vec![1536, 512]);
}

#[test]
fn calibration_stats_passthrough() {
    // EmbeddingGemma2 ships TensorRT-LLM FP8 calibration stats on the audio
    // tower (input_max / input_min / output_max / output_min). The
    // converter emits them verbatim under `audio.layers.{i}.{...}_max/min`
    //`.weight` (engine currently ignores them but the contract surface
    // must round-trip).
    let Some(loader) = loader() else { return };
    for i in [0usize, 5, 11] {
        for stat in [
            "feed_forward1.ffw_layer_1.input_max",
            "feed_forward1.ffw_layer_1.input_min",
            "feed_forward1.ffw_layer_1.output_max",
            "feed_forward1.ffw_layer_1.output_min",
        ] {
            assert!(
                shape(&loader, &format!("audio.layers.{i}.{stat}")).is_some(),
                "missing audio.layers.{i}.{stat}",
            );
        }
    }
}
