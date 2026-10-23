use rust_model_inference::models::qwen35::Qwen35Config;
#[cfg(feature = "parity-trace")]
use rust_model_inference::models::qwen35::{Qwen35Model, Qwen35Session};
use rust_model_inference::{BPETokenizer, EncodeOptions, GGMLType, GGUFLoader};
use std::path::PathBuf;
use std::process::Command;
#[cfg(feature = "parity-trace")]
use std::sync::Arc;

fn path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .unwrap_or_else(|| panic!("{name} is required"))
        .into()
}

#[test]
#[ignore = "requires RMI_SAGE_MODEL and RMI_SAGE_TOKENIZE"]
fn sage_contract_and_tokenizer() {
    let model = path("RMI_SAGE_MODEL");
    let source = GGUFLoader::from_file(&model).unwrap();
    let config = Qwen35Config::from_source(&source).unwrap();
    assert_eq!((config.n_embd, config.vocab_size), (2048, 248320));
    assert_eq!(
        (config.n_layer, config.n_layer_impl(), config.n_nextn),
        (41, 40, 1)
    );
    assert_eq!(
        (config.n_head, config.n_head_kv, config.key_length),
        (16, 2, 256)
    );
    assert_eq!((config.key_dim(), config.value_dim()), (2048, 4096));
    let moe = config.moe.unwrap();
    assert_eq!((moe.n_expert, moe.n_expert_used), (256, 8));
    assert_eq!((moe.n_ff_exp, moe.n_ff_shexp), (512, 512));
    for (name, shape) in [
        ("token_embd.weight", vec![2048, 248320]),
        ("output.weight", vec![2048, 248320]),
        ("blk.0.attn_qkv.weight", vec![2048, 8192]),
        ("blk.3.attn_q.weight", vec![2048, 8192]),
        ("blk.39.ffn_down_exps.weight", vec![512, 2048, 256]),
        ("blk.40.nextn.eh_proj.weight", vec![4096, 2048]),
    ] {
        let tensor = source.tensor_info(name).unwrap();
        assert_eq!(tensor.dims, shape, "{name}");
        assert_eq!(tensor.ggml_type, GGMLType::Q8_0, "{name}");
    }
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned()).unwrap();
    assert_eq!(tokenizer.bos_id(), Some(248044));
    assert_eq!(tokenizer.eos_id(), Some(248046));
    for text in [
        "Hello",
        "你好",
        " hello   world\n\t1234567890",
        "e\u{301} résumé 日本語 🧪",
        "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n",
    ] {
        let output = Command::new(path("RMI_SAGE_TOKENIZE"))
            .args(["-m"])
            .arg(&model)
            .args(["--ids", "--no-bos", "--no-escape", "-p", text])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<u32> = serde_json::from_slice(&output.stdout).unwrap();
        let actual = tokenizer.encode(
            text,
            EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        );
        assert_eq!(actual, expected, "{text:?}");
        assert_eq!(tokenizer.decode(&actual, true), text);
    }
    let template =
        rust_model_inference::prompt::jinja::template_from_source(&|k| source.metadata(k).cloned())
            .unwrap()
            .unwrap();
    let rendered = template
        .render(
            &[rust_model_inference::prompt::jinja::ChatMessage::text(
                "user", "你好",
            )],
            true,
            false,
            &rust_model_inference::prompt::jinja::SpecialTokens::default(),
        )
        .unwrap();
    assert_eq!(
        rendered,
        "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[cfg(feature = "parity-trace")]
#[test]
#[ignore = "helper for tools/oracle/sage_a15/compare.py; requires the full model"]
fn sage_dump_trace() {
    let source = GGUFLoader::from_file(path("RMI_SAGE_MODEL")).unwrap();
    let mut model = Qwen35Model::from_source(&source).unwrap();
    assert_eq!(
        model.layers.len(),
        40,
        "MTP must not execute in the main decoder"
    );
    let prompt = std::env::var("RMI_SAGE_PROMPT").unwrap();
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned()).unwrap();
    let ids = tokenizer.encode(
        &prompt,
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert!(!ids.is_empty());
    let backend = std::env::var("RMI_SAGE_BACKEND").unwrap_or_else(|_| "cpu".into());
    let policy = match backend.as_str() {
        "cpu" => rust_model_inference::compute::ComputePolicy::Cpu,
        "vulkan" => rust_model_inference::compute::ComputePolicy::Vulkan,
        _ => panic!("unsupported RMI_SAGE_BACKEND {backend}"),
    };
    let mut session = Qwen35Session::new_with_compute(
        &mut model,
        ids.len() + 4,
        1,
        Arc::new(rust_model_inference::ComputePool::new(8)),
        policy,
    )
    .unwrap();
    println!("Sage trace backend: {backend}");
    let mut processed = Vec::new();
    let mut generated = Vec::new();
    let mut logits = Vec::new();
    let mut decode = |id, position| {
        let embedding = session.embed_token(id).unwrap();
        rust_model_inference::parity_trace::checkpoint_at(
            "model.input_embed",
            None,
            Some(position),
            &[2048],
            &embedding,
        )
        .unwrap();
        let logits = session.step_with_tokens(&[id], &[[position; 4]]).unwrap();
        assert_eq!(logits.len(), 248320);
        assert!(logits.iter().all(|value| value.is_finite()));
        logits
    };
    for &id in &ids {
        logits = decode(id, processed.len());
        processed.push(id);
    }
    for step in 0..4 {
        let next = (0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then_with(|| b.cmp(&a)))
            .unwrap() as u32;
        generated.push(next);
        if step < 3 {
            logits = decode(next, processed.len());
            processed.push(next);
        }
    }
    std::fs::write(
        path("RMI_SAGE_TOKENS"),
        serde_json::to_vec(&serde_json::json!({"processed": processed, "generated": generated}))
            .unwrap(),
    )
    .unwrap();
}

#[cfg(all(feature = "vulkan", feature = "parity-trace"))]
fn assert_sage_logits(label: &str, step: usize, cpu: &[f32], gpu: &[f32]) {
    assert_eq!(gpu.len(), cpu.len());
    assert!(cpu.iter().chain(gpu).all(|v| v.is_finite()));
    let norm = |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
    let (a, b) = (norm(cpu), norm(gpu));
    let delta = cpu
        .iter()
        .zip(gpu)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum::<f64>();
    let relative_l2 = (delta / a).sqrt();
    let cosine = cpu
        .iter()
        .zip(gpu)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum::<f64>()
        / (a * b).sqrt();
    let max_abs = cpu
        .iter()
        .zip(gpu)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!("Sage {label} step={step}: relative_l2={relative_l2:.8e} cosine={cosine:.10} max_abs={max_abs:.8e}");
    assert!(relative_l2 <= 0.01 && cosine >= 0.99995 && max_abs <= 0.5);
}

#[cfg(all(feature = "vulkan", feature = "parity-trace"))]
#[test]
#[ignore = "requires a physical Vulkan GPU and RMI_SAGE_MODEL; run without RMI_PARITY_TRACE"]
fn sage_vulkan_batched_prefill_and_decode() {
    use rust_model_inference::compute::ComputePolicy;
    assert!(
        std::env::var_os("RMI_PARITY_TRACE").is_none(),
        "trace mode forces single-token chunks"
    );
    let source = GGUFLoader::from_file(path("RMI_SAGE_MODEL")).unwrap();
    let mut model = Qwen35Model::from_source(&source).unwrap();
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned()).unwrap();
    let prompt = "<|im_start|>user\n请用三句话向初学者解释混合专家（MoE）模型：它如何选择专家、为什么能减少计算，以及一个主要代价。不要列标题。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let ids = tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    let mut run = |policy, batch, reference: Option<&[u32]>, expected: Option<&[Vec<f32>]>| {
        let mut session = Qwen35Session::new_with_compute(
            &mut model,
            ids.len() + 64,
            batch,
            Arc::new(rust_model_inference::ComputePool::new(8)),
            policy,
        )
        .unwrap();
        let positions = (0..ids.len()).map(|p| [p; 4]).collect::<Vec<_>>();
        let mut logits = session.step_with_tokens(&ids, &positions).unwrap();
        let mut generated = Vec::new();
        let mut outputs = Vec::new();
        for step in 0..64 {
            if let Some(expected) = expected {
                assert_sage_logits(
                    &format!("{policy:?}/batch={batch}"),
                    step,
                    &expected[step],
                    &logits,
                );
            }
            let next = (0..logits.len())
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then_with(|| b.cmp(&a)))
                .unwrap() as u32;
            generated.push(next);
            outputs.push(logits);
            if step < 63 {
                logits = session
                    .step_with_tokens(
                        &[reference.map_or(next, |ids| ids[step])],
                        &[[ids.len() + step; 4]],
                    )
                    .unwrap();
            } else {
                break;
            }
        }
        (generated, outputs)
    };
    let (cpu_ids, cpu) = run(ComputePolicy::Cpu, 1, None, None);
    let (batched_ids, batched) = run(ComputePolicy::Cpu, 64, Some(&cpu_ids), Some(&cpu));
    assert_eq!(batched_ids, cpu_ids);
    assert_eq!(batched, cpu);
    let (gpu_ids, gpu) = run(ComputePolicy::Vulkan, 64, Some(&cpu_ids), Some(&cpu));
    println!("CPU greedy={cpu_ids:?}; GPU teacher-forced greedy={gpu_ids:?}");
    assert_eq!(gpu_ids, cpu_ids);
    println!("Sage Vulkan/CPU batch=64 vs CPU batch=1, 64 decode logits checked");
}
