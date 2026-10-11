//! Integration tests for the `Shanghai_AI_Laboratory/Intern-Decision-0.8B`
//! GGUF (Qwen3.5 hybrid 0.8B multimodal fine-tuned for decision-making).
//!
//! Run with:
//!   RMI_INTERN_DECISION_0_8B_TEXT=/path/to/intern-decision-0.8b-Q8_0.gguf \
//!   RMI_INTERN_DECISION_0_8B_MMPROJ=/path/to/mmproj-intern-decision-0.8b-F16.gguf \
//!   cargo test --profile release-fast --test intern_decision_0_8b
//!
//! Intern-Decision rides the existing `qwen35` text path (hybrid Mamba+attention
//! trunk, same as Qwen3.5-0.8B) and the `qwen3vl_merger` mmproj path
//! (12-layer ViT @ hidden 768, no deepstack). Zero new model code: just the
//! converter `tools/converter/intern_decision/convert_intern_decision.py`.

use rust_model_inference::GGUFLoader;
use rust_model_inference::TensorInfo;

fn text_loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_INTERN_DECISION_0_8B_TEXT")?;
    GGUFLoader::from_file(path).ok()
}

fn mmproj_loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_INTERN_DECISION_0_8B_MMPROJ")?;
    GGUFLoader::from_file(path).ok()
}

fn pick(loader: &GGUFLoader, key: &str) -> usize {
    loader.metadata(key).and_then(|v| v.to_u64()).map(|v| v as usize).unwrap_or(0)
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
fn text_arch_and_dims_match_qwen35_0_8b() {
    let Some(loader) = text_loader() else { return };
    assert_eq!(pick_str(&loader, "general.architecture"), Some("qwen35"));
    assert_eq!(pick(&loader, "qwen35.embedding_length"), 1024);
    assert_eq!(pick(&loader, "qwen35.block_count"), 24);
    assert_eq!(pick(&loader, "qwen35.attention.head_count"), 8);
    assert_eq!(pick(&loader, "qwen35.attention.head_count_kv"), 2);
    assert_eq!(pick(&loader, "qwen35.attention.key_length"), 256);
    assert_eq!(pick(&loader, "qwen35.feed_forward_length"), 3584);
    assert_eq!(pick(&loader, "qwen35.context_length"), 262_144);
    assert_eq!(pick(&loader, "qwen35.vocab_size"), 248_320);
    // SSM constants match Qwen3.5-0.8B.
    assert_eq!(pick(&loader, "qwen35.ssm.inner_size"), 2048);
    assert_eq!(pick(&loader, "qwen35.ssm.state_size"), 128);
    assert_eq!(pick(&loader, "qwen35.ssm.time_step_rank"), 16);
    assert_eq!(pick(&loader, "qwen35.ssm.group_count"), 16);
    assert_eq!(pick(&loader, "qwen35.ssm.conv_kernel"), 4);
    assert_eq!(pick(&loader, "qwen35.full_attention_interval"), 4);
}

#[test]
fn text_rope_sections_match_qwen35_mrope() {
    let Some(loader) = text_loader() else { return };
    let sections = loader
        .metadata("qwen35.rope.dimension_sections")
        .and_then(|v| v.to_arr())
        .expect("qwen35.rope.dimension_sections");
    let section_vec: Vec<i64> = sections
        .iter()
        .filter_map(|v| v.to_u64().map(|x| x as i64))
        .collect();
    assert_eq!(section_vec, vec![11, 11, 10, 0]);
}

#[test]
fn text_image_token_ids_match_intern_decision_specials() {
    let Some(loader) = text_loader() else { return };
    assert_eq!(pick(&loader, "qwen35.image_token_id"), 248_056);
    assert_eq!(pick(&loader, "qwen35.vision_start_token_id"), 248_053);
    assert_eq!(pick(&loader, "qwen35.vision_end_token_id"), 248_054);
}

#[test]
fn text_tied_embedding_omits_output_weight() {
    let Some(loader) = text_loader() else { return };
    assert!(
        shape(&loader, "output.weight").is_none(),
        "Intern-Decision uses tied embeddings; output.weight must be absent"
    );
    let embed = shape(&loader, "token_embd.weight").expect("token_embd.weight");
    assert_eq!(embed.dims, vec![1024, 248_320]);
}

#[test]
fn text_hybrid_layer_dispatch_full_vs_linear() {
    // qwen35 layer_types pattern: every 4th layer is full attention, the
    // rest are linear (Mamba SSM). Indices 3, 7, 11, 15, 19, 23 are full.
    let Some(loader) = text_loader() else { return };

    // Linear-attention (SSM) layers must carry ssm_* tensors and lack
    // attn_{q,k,v,output}.
    for i in [0usize, 1, 2, 4, 5, 6] {
        for name in ["attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta",
                     "ssm_conv1d", "ssm_norm", "ssm_out", "ssm_a", "ssm_dt.bias"] {
            assert!(
                shape(&loader, &format!("blk.{i}.{name}.weight")).is_some()
                    || shape(&loader, &format!("blk.{i}.{name}")).is_some()
                    || name == "ssm_a" || name == "ssm_dt.bias",
                "linear block {i} missing {name}",
            );
        }
        for name in ["attn_q", "attn_k", "attn_v", "attn_output",
                     "attn_q_norm", "attn_k_norm"] {
            assert!(
                shape(&loader, &format!("blk.{i}.{name}.weight")).is_none(),
                "linear block {i} should NOT carry {name}",
            );
        }
    }

    // Full-attention layers must carry q/k/v/o + q/k_norm but lack ssm_*.
    for i in [3usize, 7, 11, 15, 19, 23] {
        for name in ["attn_q", "attn_k", "attn_v", "attn_output",
                     "attn_q_norm", "attn_k_norm"] {
            assert!(
                shape(&loader, &format!("blk.{i}.{name}.weight")).is_some(),
                "full block {i} missing {name}.weight",
            );
        }
        for name in ["attn_qkv", "ssm_norm", "ssm_a"] {
            assert!(
                shape(&loader, &format!("blk.{i}.{name}.weight")).is_none()
                    && shape(&loader, &format!("blk.{i}.{name}")).is_none(),
                "full block {i} should NOT carry {name}",
            );
        }
    }
}

#[test]
fn text_ffn_present_for_every_block() {
    let Some(loader) = text_loader() else { return };
    for i in 0..24 {
        for name in ["attn_norm", "post_attention_norm",
                     "ffn_gate", "ffn_up", "ffn_down"] {
            assert!(
                shape(&loader, &format!("blk.{i}.{name}.weight")).is_some(),
                "missing blk.{i}.{name}.weight",
            );
        }
    }
    assert!(shape(&loader, "output_norm.weight").is_some());
}

#[test]
fn mmproj_arch_is_clip_with_qwen3vl_merger() {
    let Some(loader) = mmproj_loader() else { return };
    assert_eq!(pick_str(&loader, "general.architecture"), Some("clip"));
    assert_eq!(
        pick_str(&loader, "clip.projector_type"),
        Some("qwen3vl_merger"),
    );
    let has_ve = loader
        .metadata("clip.has_vision_encoder")
        .is_some_and(|v| matches!(
            v,
            rust_model_inference::core::tensor::MetaValue::Bool(true)
        ));
    assert!(has_ve, "clip.has_vision_encoder must be Bool(true)");
}

#[test]
fn mmproj_vision_dims_match_intern_decision() {
    let Some(loader) = mmproj_loader() else { return };
    assert_eq!(pick(&loader, "clip.vision.embedding_length"), 768);
    assert_eq!(pick(&loader, "clip.vision.feed_forward_length"), 3072);
    assert_eq!(pick(&loader, "clip.vision.block_count"), 12);
    assert_eq!(pick(&loader, "clip.vision.attention.head_count"), 12);
    assert_eq!(pick(&loader, "clip.vision.image_size"), 768);
    assert_eq!(pick(&loader, "clip.vision.patch_size"), 16);
    assert_eq!(pick(&loader, "clip.vision.spatial_merge_size"), 2);
    assert_eq!(pick(&loader, "clip.vision.projection_dim"), 1024);
}

#[test]
fn mmproj_deepstack_disabled_for_every_layer() {
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
    assert_eq!(bools.len(), 12);
    assert!(bools.iter().all(|&b| b == 0), "no deepstack layers in Intern-Decision");
}

#[test]
fn mmproj_conv3d_split_and_merger_shape() {
    let Some(loader) = mmproj_loader() else { return };
    let w0 = shape(&loader, "v.patch_embd.weight").expect("v.patch_embd.weight");
    let w1 = shape(&loader, "v.patch_embd.weight.1").expect("v.patch_embd.weight.1");
    // Intern-Decision vision out_channels=768 (vs 1024 for PhysBrain).
    assert_eq!(w0.dims, vec![16, 16, 3, 768]);
    assert_eq!(w1.dims, vec![16, 16, 3, 768]);
    let mm0 = shape(&loader, "mm.0.weight").expect("mm.0.weight");
    let mm2 = shape(&loader, "mm.2.weight").expect("mm.2.weight");
    // mm.0: [merged_embd, merged_embd] = [n_embd*merge^2, n_embd*merge^2] = [3072, 3072]
    // mm.2: [merged_embd, projection_dim] = [3072, 1024]
    assert_eq!(mm0.dims, vec![3072, 3072]);
    assert_eq!(mm2.dims, vec![3072, 1024]);
    let post_ln = shape(&loader, "v.post_ln.weight").expect("v.post_ln.weight");
    // post_ln operates per-token on n_embd=768.
    assert_eq!(post_ln.dims, vec![768]);
}
