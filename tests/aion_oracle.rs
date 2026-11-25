//! Opt-in real checkpoint verification against tools/oracle/aion/oracle.py.
use rust_model_inference::compute::ComputePolicy;
use rust_model_inference::core::{
    scratchpad::KvFormat,
    thread_pool::ComputePool,
    tokenizer::{BPETokenizer, EncodeOptions},
};
use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};
use rust_model_inference::models::aion::{AionModel, AionSession};
use rust_model_inference::prompt::jinja::JinjaChatTemplate;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn top_token(logits: &[f32]) -> u32 {
    // Reverse iteration makes an exact tie select the lower token ID, as PyTorch does.
    logits
        .iter()
        .enumerate()
        .rev()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

#[test]
#[ignore = "requires the Aion checkpoint and generated PyTorch logits"]
fn aion_logits_match_pinned_pytorch() {
    let path = PathBuf::from(std::env::var_os("AION_GGUF").expect("AION_GGUF"));
    let oracle = PathBuf::from(std::env::var_os("AION_ORACLE").expect("AION_ORACLE"));
    let policy = match std::env::var("AION_COMPUTE").as_deref() {
        Ok("auto") => ComputePolicy::Auto,
        Ok("vulkan") => ComputePolicy::Vulkan,
        Ok("cpu") | Err(_) => ComputePolicy::Cpu,
        Ok(other) => panic!("unknown compute policy {other}"),
    };
    let kv = match std::env::var("AION_KV").as_deref() {
        Ok("f16") => KvFormat::F16,
        Ok("f32") | Err(_) => KvFormat::F32,
        Ok(other) => panic!("unknown KV format {other}"),
    };
    let threads = 8;
    let batch = std::env::var("AION_BATCH").map_or(64, |n| n.parse().unwrap());
    let source = open_model_source(&path, ComponentRole::Llm).unwrap();
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap();
    let template = JinjaChatTemplate::compile_huggingface(
        source
            .metadata("tokenizer.chat_template")
            .unwrap()
            .to_string_val()
            .unwrap(),
        "Aion checkpoint",
    )
    .unwrap();
    let model = AionModel::load(source.as_ref(), Arc::new(ComputePool::new(threads))).unwrap();
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(oracle.join("cases.json")).unwrap()).unwrap();
    let precision = path.file_stem().unwrap().to_str().unwrap();
    let storage = source.tensor_info("token_embd.weight").unwrap().ggml_type;
    let (block_elements, _) = storage.type_traits();
    let quantized = block_elements > 1;
    assert!(
        !manifest["rounded_activations"].as_bool().unwrap_or(false),
        "the production oracle requires independently unpacked weights and F32 activations"
    );
    // Retain the pre-established budgets for the same-weight F32-activation reference.
    let (max_abs, max_rmse, min_cosine) = if quantized {
        (0.80, 0.12, 0.999)
    } else if kv == KvFormat::F16 {
        (0.10, 0.015, 0.99999)
    } else {
        (0.050, 0.008, 0.999999)
    };
    let mut all_results = Vec::new();
    let mut violations = Vec::new();
    let cases = manifest["cases"].as_array().unwrap();
    assert!(!cases.is_empty(), "empty reference suite");
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let rendered = template
            .render_context(&json!({
                "messages":case["messages"], "tools":case["tools"], "current_date":manifest["date"],
                "add_generation_prompt":true,
            }))
            .unwrap();
        assert_eq!(rendered, case["prompt"].as_str().unwrap(), "{name} prompt");
        let ids = tokenizer.encode(
            &rendered,
            EncodeOptions {
                parse_special: true,
                add_special: false,
            },
        );
        let expected_ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        assert_eq!(ids, expected_ids, "{name} tokenizer");
        let expected_greedy: Vec<u32> = serde_json::from_value(case["greedy"].clone()).unwrap();
        let forced: Vec<u32> = serde_json::from_value(case["forced"].clone()).unwrap();
        assert!(!forced.is_empty(), "{name}: missing prefill");
        assert_eq!(forced.len(), expected_greedy.len(), "{name}: decode steps");
        let data = std::fs::read(oracle.join(format!("{name}.f32"))).unwrap();
        let vocab = tokenizer.vocab_size();
        assert_eq!(data.len(), forced.len() * vocab * 4);
        let started = Instant::now();
        let mut session =
            AionSession::new(&model, policy, kv, ids.len() + forced.len(), batch).unwrap();
        let setup_seconds = started.elapsed().as_secs_f64();
        let mut input = ids.clone();
        let mut steps = Vec::new();
        let mut greedy = Vec::new();
        for (step, (&next, &oracle_top)) in forced.iter().zip(&expected_greedy).enumerate() {
            let start = Instant::now();
            let logits = session.forward(&input).unwrap();
            let seconds = start.elapsed().as_secs_f64();
            assert_eq!(logits.len(), vocab, "{name} step {step}: vocabulary size");
            let expected: Vec<_> = data[step * vocab * 4..(step + 1) * vocab * 4]
                .chunks_exact(4)
                .map(|row| f32::from_le_bytes(row.try_into().unwrap()))
                .collect();
            assert_eq!(
                top_token(&expected),
                oracle_top,
                "{name} step {step}: reference top-1"
            );
            let mut max = 0.0f64;
            let (mut squared, mut dot, mut actual_norm, mut expected_norm) = (0.0, 0.0, 0.0, 0.0);
            for (&a, &b) in logits.iter().zip(&expected) {
                assert!(a.is_finite() && b.is_finite());
                let (a, b) = (f64::from(a), f64::from(b));
                max = max.max((a - b).abs());
                squared += (a - b) * (a - b);
                dot += a * b;
                actual_norm += a * a;
                expected_norm += b * b;
            }
            let rmse = (squared / vocab as f64).sqrt();
            let cosine = dot / (actual_norm * expected_norm).sqrt();
            let token = top_token(logits);
            greedy.push(token);
            let margin = expected[oracle_top as usize] - expected[token as usize];
            steps.push(json!({"max_abs":max,"rmse":rmse,"cosine":cosine,
                             "margin":margin,"seconds":seconds}));
            println!("{precision} {policy:?} {kv:?} {name} step={step} token={token}/{oracle_top} max_abs={max:.7} rmse={rmse:.7} cosine={cosine:.10} margin={margin:.5} secs={seconds:.5}");
            if !(max <= max_abs && rmse <= max_rmse && cosine >= min_cosine) {
                violations.push(format!("{name} step {step}: outside numeric budget"));
            }
            if (quantized && margin > 0.20) || (!quantized && token != oracle_top) {
                violations.push(format!(
                    "{name} step {step}: greedy token {token}/{oracle_top}, margin {margin}"
                ));
            }
            assert_eq!(session.position(), ids.len() + step);
            input = vec![next];
        }
        all_results.push(
            json!({"name":name,"input_tokens":ids.len(),"setup_seconds":setup_seconds,
                                "greedy":greedy,"oracle_greedy":expected_greedy,"steps":steps}),
        );
    }
    if let Ok(path) = std::env::var("AION_RESULT") {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&json!({
                "format":precision,"compute":format!("{policy:?}"),"kv":format!("{kv:?}"),
                "batch":batch,"threads":threads,"rounded_activations":false,
            "violations":violations,"cases":all_results,
            }))
            .unwrap(),
        )
        .unwrap();
    }
    assert!(violations.is_empty(), "{}", violations.join("\n"));
}
