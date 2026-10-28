use rust_model_inference::core::loader::{model_config_from_source, qwen3_arch_knobs};
use rust_model_inference::core::tensor::{MetaValue, TensorInfo, TensorSource};
use std::collections::HashMap;

struct Source(HashMap<String, MetaValue>);

impl TensorSource for Source {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.0.get(key)
    }
    fn tensor_info(&self, _: &str) -> Option<&TensorInfo> {
        None
    }
    fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
        None
    }
}

fn source() -> Source {
    let mut metadata = HashMap::from([
        (
            "general.architecture".into(),
            MetaValue::String("hy_v3".into()),
        ),
        ("hy_v3.expert_weights_norm".into(), MetaValue::Bool(true)),
        (
            "hy_v3.expert_weights_scale".into(),
            MetaValue::Float32(2.826),
        ),
        (
            "hy_v3.rope.freq_base".into(),
            MetaValue::Float32(11_158_840.0),
        ),
        (
            "hy_v3.attention.layer_norm_rms_epsilon".into(),
            MetaValue::Float32(1e-5),
        ),
    ]);
    for (key, value) in [
        ("embedding_length", 2048),
        ("block_count", 48),
        ("attention.head_count", 32),
        ("attention.head_count_kv", 4),
        ("attention.key_length", 128),
        ("attention.value_length", 128),
        ("feed_forward_length", 6912),
        ("context_length", 262144),
        ("vocab_size", 120832),
        ("expert_count", 128),
        ("expert_used_count", 8),
        ("expert_feed_forward_length", 768),
        ("expert_shared_feed_forward_length", 768),
        ("expert_gating_func", 2),
    ] {
        metadata.insert(format!("hy_v3.{key}"), MetaValue::Uint32(value));
    }
    Source(metadata)
}

#[test]
fn hy_v3_has_explicit_dispatch_and_128_wide_heads() {
    let source = source();
    let config = model_config_from_source(&source).unwrap();
    assert_eq!(
        (config.n_embd, config.n_head, config.n_embd_head),
        (2048, 32, 128)
    );
    let knobs = qwen3_arch_knobs(&source).unwrap();
    assert_eq!(knobs.arch, "hy_v3");
    assert!(knobs.has_qk_norm);
    assert!(!knobs.has_qkv_bias);
    let moe = knobs.moe.unwrap();
    assert_eq!((moe.expert_count, moe.expert_used_count), (128, 8));
    assert_eq!((moe.expert_ffn, moe.shared_expert_ffn), (768, 768));
}

#[test]
fn hy_v3_rejects_other_router_contracts() {
    for (key, value) in [
        ("expert_gating_func", MetaValue::Uint32(1)),
        ("expert_weights_norm", MetaValue::Bool(false)),
        ("expert_weights_scale", MetaValue::Float32(f32::NAN)),
        ("expert_count", MetaValue::Uint32(0)),
        ("expert_used_count", MetaValue::Uint32(129)),
    ] {
        let mut source = source();
        source.0.insert(format!("hy_v3.{key}"), value);
        assert!(qwen3_arch_knobs(&source).is_err(), "accepted {key}");
    }
}
