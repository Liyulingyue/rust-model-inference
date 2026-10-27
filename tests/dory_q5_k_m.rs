use rust_model_inference::app::cli::KvFormat;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
use rust_model_inference::models::dory::{DoryConfig, DoryModel};
use rust_model_inference::GGUFLoader;
use std::path::PathBuf;
use std::sync::Arc;

fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("RMI_DORY_Q5_MODEL")?);
    path.exists().then_some(path)
}

fn source() -> Option<Arc<dyn TensorSource>> {
    let path = model_path()?;
    Some(Arc::new(
        GGUFLoader::from_file(path).expect("open Dory GGUF"),
    ))
}

#[test]
fn dory_q5_contract_and_tokenizer() {
    let Some(source) = source() else { return };
    let config = DoryConfig::from_source(source.as_ref()).expect("Dory config");
    assert_eq!(config.n_embd, 2560);
    assert_eq!(config.n_layer, 72);
    assert_eq!(config.n_input_layers, 16);
    assert_eq!(config.n_recurrent_layers, 40);
    assert_eq!(config.n_output_layers, 16);
    assert_eq!(config.n_loops, 4);
    assert_eq!(config.n_head, 16);
    assert_eq!(config.n_head_kv, 4);
    assert_eq!(config.head_dim, 256);
    assert_eq!(config.n_ff, 7168);
    assert_eq!(config.n_ctx, 262_144);
    assert_eq!(config.vocab_size, 131_072);
    assert_eq!(config.rope_freq_base, 10_000_000.0);
    assert_eq!(config.rope_freq_base_2, 10_000.0);
    assert_eq!(config.sliding_window, 1024);
    assert_eq!(config.recurrent_kv_cache_mode, "per_loop");
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
        .expect("Dory GPT-2 tokenizer");
    assert_eq!(tokenizer.bos_id(), Some(1));
    assert_eq!(tokenizer.eos_id(), Some(11));
    assert_eq!(
        tokenizer.encode(
            "Hello",
            EncodeOptions {
                add_special: false,
                parse_special: true,
            }
        ),
        vec![22177]
    );
}

#[test]
fn dory_q5_single_token_logits_are_finite() {
    let Some(source) = source() else { return };
    let model = DoryModel::from_source(source, 1).expect("load Dory model");
    let mut session = model.new_session(2, KvFormat::F32);
    let logits = model.prefill(&[1], &mut session).expect("Dory forward");
    assert_eq!(logits.len(), 131_072);
    assert!(logits.iter().all(|value| value.is_finite()));
}

fn reference_prompt() -> String {
    match std::env::var_os("RMI_DORY_PROMPT_FILE") {
        Some(path) => std::fs::read_to_string(path).expect("read raw prompt"),
        None => std::env::var("RMI_DORY_PROMPT").unwrap_or_else(|_| "Hello".into()),
    }
}

#[test]
#[ignore = "requires the real Q5_K_M GGUF; see tools/oracle/dory/README.md"]
fn dory_q5_reference_trace() {
    let source = source().expect("set RMI_DORY_Q5_MODEL to the real GGUF");
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap();
    let prompt = tokenizer.encode(
        &reference_prompt(),
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert!(!prompt.is_empty());
    let count: usize = std::env::var("RMI_DORY_N_PREDICT")
        .unwrap_or_else(|_| "8".into())
        .parse()
        .unwrap();
    let threads = std::env::var("RMI_DORY_THREADS")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    let kv = match std::env::var("RMI_DORY_KV").as_deref().unwrap_or("f32") {
        "f32" => KvFormat::F32,
        "f16" => KvFormat::F16,
        other => panic!("unknown KV format {other}"),
    };
    let mut model = DoryModel::from_source(source, threads).unwrap();
    if let Ok(window) = std::env::var("RMI_DORY_TEST_SWA") {
        model.config.sliding_window = window.parse().unwrap();
        assert!(model.config.sliding_window > 0);
    }
    if let Ok(mode) = std::env::var("RMI_DORY_CACHE_MODE") {
        assert!(mode == "per_loop" || mode == "last_loop");
        model.config.recurrent_kv_cache_mode = mode;
    }
    let batch: usize = std::env::var("RMI_DORY_BATCH")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    assert!((1..=32).contains(&batch));
    let policy = if std::env::var("RMI_DORY_COMPUTE").as_deref() == Ok("vulkan") {
        rust_model_inference::compute::ComputePolicy::Vulkan
    } else {
        rust_model_inference::compute::ComputePolicy::Cpu
    };
    let mut session = model
        .new_session_with_compute(prompt.len() + count, kv, policy)
        .unwrap();
    #[cfg(feature = "parity-trace")]
    rust_model_inference::parity_trace::token_ids("prompt_ids", &prompt).unwrap();
    let mut logits = Vec::new();
    #[cfg(feature = "parity-trace")]
    let batches: Vec<u32> = prompt
        .chunks(batch)
        .map(|c| c.len() as u32)
        .chain(std::iter::repeat_n(1, count))
        .collect();
    #[cfg(feature = "parity-trace")]
    rust_model_inference::parity_trace::token_ids("prefill_batches", &batches).unwrap();
    for (i, chunk) in prompt.chunks(batch).enumerate() {
        logits = model.prefill(chunk, &mut session).unwrap();
        eprintln!(
            "prefill {}/{}",
            ((i + 1) * batch).min(prompt.len()),
            prompt.len()
        );
    }
    let mut generated = Vec::new();
    for _ in 0..count {
        assert!(logits.iter().all(|x| x.is_finite()));
        let mut best = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        generated.push(best as u32);
        logits = model.prefill(&[best as u32], &mut session).unwrap();
    }
    #[cfg(feature = "parity-trace")]
    rust_model_inference::parity_trace::token_ids("generated_ids", &generated).unwrap();
    println!("prompt_ids={prompt:?}\ngenerated_ids={generated:?}");
    println!("output={:?}", tokenizer.decode(&generated, true));
}

#[test]
#[ignore = "requires the real Q5_K_M GGUF"]
fn dory_q5_prefill_chunking_and_reset() {
    rust_model_inference::ops::enable_gpu();
    let model = DoryModel::from_source(source().expect("set RMI_DORY_Q5_MODEL"), 4).unwrap();
    let tokens = [22177, 1897, 1429];
    for kv in [KvFormat::F32, KvFormat::F16] {
        let mut batched = model.new_session(tokens.len(), kv);
        let expected = model.prefill(&tokens, &mut batched).unwrap();
        let mut chunked = model.new_session(tokens.len(), kv);
        model.prefill(&tokens[..2], &mut chunked).unwrap();
        let actual = model.prefill(&tokens[2..], &mut chunked).unwrap();
        let assert_close = |actual: &[f32]| {
            assert_eq!(expected.len(), actual.len());
            assert!(expected.iter().zip(actual).all(|(a, b)| a.is_finite()
                && b.is_finite()
                && (a - b).abs() <= 1e-6 * a.abs().max(1.0)));
        };
        assert_close(&actual);
        assert!(model.prefill(&[], &mut chunked).is_err());
        assert!(model.prefill(&[1], &mut chunked).is_err());
        chunked.reset(&model, kv).unwrap();
        assert_close(&model.prefill(&tokens, &mut chunked).unwrap());
    }
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires the real Q5_K_M GGUF and a hardware Vulkan device"]
fn dory_q5_vulkan_executes_and_matches_cpu() {
    use rust_model_inference::compute::ComputePolicy;
    assert!(std::env::var_os("RMI_PARITY_TRACE").is_none());
    assert!(std::env::var_os("RMI_SCALAR").is_none());
    rust_model_inference::ops::enable_gpu();
    let model = DoryModel::from_source(source().expect("set RMI_DORY_Q5_MODEL"), 4).unwrap();
    for kv in [KvFormat::F32, KvFormat::F16] {
        let mut gpu = model
            .new_session_with_compute(4, kv, ComputePolicy::Vulkan)
            .unwrap();
        let mut cpu = model.new_session(4, kv);
        let context = rust_model_inference::ops::get_vulkan_context().unwrap();
        let before = context.compute_stats();
        for token in [22177, 1897, 1429] {
            let actual = model.prefill(&[token], &mut gpu).unwrap();
            let expected = model.prefill(&[token], &mut cpu).unwrap();
            assert_numerically_aligned(&actual, &expected);
        }
        assert!(context.compute_stats().since(before).submissions > 0);
        assert!(model.prefill(&[u32::MAX], &mut gpu).is_err());
        gpu.reset(&model, kv).unwrap();
        cpu.reset(&model, kv).unwrap();
        assert_numerically_aligned(
            &model.prefill(&[22177, 1897, 1429], &mut gpu).unwrap(),
            &model.prefill(&[22177, 1897, 1429], &mut cpu).unwrap(),
        );
    }
}

#[cfg(feature = "vulkan")]
fn assert_numerically_aligned(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let (mut diff, mut reference, mut max_abs, mut scale) = (0.0f64, 0.0f64, 0.0f64, 1.0f64);
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        let delta = f64::from(a) - f64::from(b);
        diff += delta * delta;
        reference += f64::from(b) * f64::from(b);
        max_abs = max_abs.max(delta.abs());
        scale = scale.max(f64::from(b).abs());
    }
    let l2 = (diff / reference.max(f64::MIN_POSITIVE)).sqrt();
    eprintln!("GPU relative_l2={l2} scaled_max_abs={}", max_abs / scale);
    assert!(l2 <= 0.01);
    assert!(max_abs / scale <= 0.01);
    let best = |v: &[f32]| (1..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
    assert_eq!(best(actual), best(expected));
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires the real GGUF and hardware Vulkan"]
fn dory_batched_modes_and_packed_kv() {
    use rust_model_inference::compute::ComputePolicy;
    rust_model_inference::ops::enable_gpu();
    let mut model = DoryModel::from_source(source().expect("set RMI_DORY_Q5_MODEL"), 4).unwrap();
    model.config.sliding_window = 2;
    let tokens: Vec<u32> = (0..35).map(|i| [22177, 1897, 1429][i % 3]).collect();
    for mode in ["per_loop", "last_loop"] {
        model.config.recurrent_kv_cache_mode = mode.into();
        let mut cache_bytes = Vec::new();
        for kv in [KvFormat::F32, KvFormat::F16] {
            let mut gpu = model
                .new_session_with_compute(tokens.len(), kv, ComputePolicy::Vulkan)
                .unwrap();
            let mut cpu = model.new_session(tokens.len(), kv);
            cache_bytes.push(gpu.kv_cache_bytes());
            let mut expected_last = Vec::new();
            for chunk in tokens.chunks(32) {
                let actual = model.prefill(chunk, &mut gpu).unwrap();
                let expected = model.prefill(chunk, &mut cpu).unwrap();
                assert_numerically_aligned(&actual, &expected);
                expected_last = expected;
            }
            gpu.reset(&model, kv).unwrap();
            assert_numerically_aligned(&model.prefill(&tokens, &mut gpu).unwrap(), &expected_last);
            gpu.reset(&model, kv).unwrap();
            cpu.reset(&model, kv).unwrap();
            assert_numerically_aligned(
                &model.prefill(&tokens[..3], &mut gpu).unwrap(),
                &model.prefill(&tokens[..3], &mut cpu).unwrap(),
            );
        }
        assert_eq!(cache_bytes[0], cache_bytes[1] * 2);
    }
    let mut gpu = model
        .new_session_with_compute(2, KvFormat::F32, ComputePolicy::Vulkan)
        .unwrap();
    let full_bytes = gpu.kv_cache_bytes();
    gpu.reset(&model, KvFormat::F16).unwrap();
    assert_eq!(gpu.kv_cache_bytes() * 2, full_bytes);
    let mut cpu = model.new_session(2, KvFormat::F16);
    assert_numerically_aligned(
        &model.prefill(&tokens[..2], &mut gpu).unwrap(),
        &model.prefill(&tokens[..2], &mut cpu).unwrap(),
    );
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires the real GGUF and hardware Vulkan"]
fn dory_reset_rebuilds_changed_config() {
    use rust_model_inference::compute::ComputePolicy;
    rust_model_inference::ops::enable_gpu();
    let mut model = DoryModel::from_source(source().expect("set RMI_DORY_Q5_MODEL"), 4).unwrap();
    model.config.recurrent_kv_cache_mode = "last_loop".into();
    let mut gpu = model
        .new_session_with_compute(2, KvFormat::F16, ComputePolicy::Vulkan)
        .unwrap();
    let mut cpu = model.new_session(2, KvFormat::F16);
    let last_bytes = gpu.kv_cache_bytes();
    for mode in ["per_loop", "last_loop"] {
        model.config.recurrent_kv_cache_mode = mode.into();
        gpu.reset(&model, KvFormat::F16).unwrap();
        if mode == "per_loop" {
            assert!(gpu.kv_cache_bytes() > last_bytes);
        } else {
            assert_eq!(gpu.kv_cache_bytes(), last_bytes);
        }
        assert!(model
            .prefill(&[22177], &mut cpu)
            .unwrap_err()
            .contains("configuration changed"));
        cpu.reset(&model, KvFormat::F16).unwrap();
        assert_numerically_aligned(
            &model.prefill(&[22177, 1897], &mut gpu).unwrap(),
            &model.prefill(&[22177, 1897], &mut cpu).unwrap(),
        );
    }
}
