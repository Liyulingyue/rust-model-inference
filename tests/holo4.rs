//! Real Holo4-35B-A3B artifacts and pinned CPU reference; see tools/oracle/holo4.
use rust_model_inference::{
    core::{
        thread_pool::ComputePool,
        tokenizer::{BPETokenizer, EncodeOptions},
    },
    models::qwen35::{vision, Qwen35Model},
    GGUFLoader,
};
use std::{
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
    sync::Arc,
};

fn artifact(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("{name} is required")))
}

#[cfg(feature = "vulkan")]
fn relative_l2(expected: &[f32], actual: &[f32]) -> f64 {
    assert_eq!(expected.len(), actual.len());
    assert!(expected.iter().chain(actual).all(|v| v.is_finite()));
    (expected
        .iter()
        .zip(actual)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        / expected.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>())
    .sqrt()
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires real Holo4 weights and a Vulkan device"]
fn holo4_vulkan_batched_prefill_and_reset() {
    use rust_model_inference::{compute::ComputePolicy, models::qwen35::Qwen35Session};
    let source = GGUFLoader::from_file(artifact("RMI_HOLO4_MODEL")).unwrap();
    let mut model = Qwen35Model::from_source(&source).unwrap();
    let pool = Arc::new(ComputePool::new(8));
    let mut session =
        Qwen35Session::new_with_compute(&mut model, 32, 4, pool, ComputePolicy::Vulkan).unwrap();
    let tokens = [1, 2, 3, 4, 5, 6, 7];
    let positions: Vec<_> = (0..tokens.len()).map(|p| [p; 4]).collect();
    let mut sequential = Vec::new();
    for (&token, &position) in tokens.iter().zip(&positions) {
        sequential = session.step_with_tokens(&[token], &[position]).unwrap();
    }
    let continued = session.step_with_tokens(&[8], &[[7; 4]]).unwrap();
    session.reset();
    let batched = session.step_with_tokens(&tokens, &positions).unwrap();
    let batched_continued = session.step_with_tokens(&[8], &[[7; 4]]).unwrap();
    let errors = [
        relative_l2(&sequential, &batched),
        relative_l2(&continued, &batched_continued),
    ];
    session.reset();
    let repeated = session.step_with_tokens(&tokens, &positions).unwrap();
    assert_eq!(
        batched, repeated,
        "reset must reproduce the same GPU computation"
    );
    eprintln!("HOLO4_VULKAN_BATCHED relative_l2={errors:?}");
    assert!(
        errors.iter().all(|&e| e < 0.02),
        "batched/continuation: {errors:?}"
    );
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires real Holo4 weights and a Vulkan device"]
fn holo4_vulkan_text() {
    use rust_model_inference::{compute::ComputePolicy, models::qwen35::Qwen35Session};
    let source = GGUFLoader::from_file(artifact("RMI_HOLO4_MODEL")).unwrap();
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap();
    let prompt = std::env::var("RMI_HOLO4_GPU_PROMPT").unwrap_or_else(|_| "Hello".into());
    let tokens = tokenizer.encode(
        &prompt,
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    let mut cpu_model = Qwen35Model::from_source(&source).unwrap();
    let mut gpu_model = Qwen35Model::from_source(&source).unwrap();
    let pool = Arc::new(ComputePool::new(8));
    let mut cpu =
        Qwen35Session::new_with_compute(&mut cpu_model, 512, 1, pool.clone(), ComputePolicy::Cpu)
            .unwrap();
    let mut gpu =
        Qwen35Session::new_with_compute(&mut gpu_model, 512, 1, pool, ComputePolicy::Vulkan)
            .unwrap();
    let mut history = tokens.clone();
    let mut errors = Vec::new();
    let mut greedy = Vec::new();
    let mut gpu_greedy = Vec::new();
    for step in 0..tokens.len() + 7 {
        let input = if step < tokens.len() {
            tokens[step]
        } else {
            *history.last().unwrap()
        };
        let a = cpu.step_with_tokens(&[input], &[[step; 4]]).unwrap();
        let b = gpu.step_with_tokens(&[input], &[[step; 4]]).unwrap();
        let error = relative_l2(&a, &b);
        errors.push(error);
        if step + 1 >= tokens.len() {
            let argmax = |values: &[f32]| {
                values
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as u32
            };
            greedy.push(argmax(&a));
            gpu_greedy.push(argmax(&b));
            history.push(argmax(&a));
        }
    }
    let report = serde_json::json!({"prompt": prompt, "prompt_ids": tokens, "relative_l2": errors, "cpu_greedy": greedy, "gpu_argmax_on_cpu_history": gpu_greedy});
    eprintln!("HOLO4_VULKAN {report}");
    if let Some(path) = std::env::var_os("RMI_HOLO4_GPU_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
    assert!(errors.iter().all(|&e| e < 0.02), "relative_l2={errors:?}");
    assert_eq!(greedy, gpu_greedy);
}

#[test]
#[ignore = "requires the real Holo4 GGUF and pinned llama-tokenize"]
fn holo4_contract_and_tokenizer() {
    let path = artifact("RMI_HOLO4_MODEL");
    let source = GGUFLoader::from_file(&path).unwrap();
    assert_eq!(
        source
            .metadata("general.architecture")
            .unwrap()
            .to_string_val(),
        Some("qwen35moe")
    );
    let model = Qwen35Model::from_source(&source).unwrap();
    let cfg = &model.config;
    assert_eq!(
        (cfg.n_layer_impl(), cfg.n_embd, cfg.vocab_size),
        (40, 2048, 248320)
    );
    assert_eq!((cfg.n_head, cfg.n_head_kv, cfg.key_length), (16, 2, 256));
    assert_eq!(cfg.is_recurrent.iter().filter(|&&v| v).count(), 30);
    assert_eq!(cfg.rope_dimension_sections, [11, 11, 10, 0]);
    let moe = cfg.moe.unwrap();
    assert_eq!(
        (
            moe.n_expert,
            moe.n_expert_used,
            moe.n_ff_exp,
            moe.n_ff_shexp
        ),
        (256, 8, 512, 512)
    );
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap();
    for prompt in [
        "",
        "Hello world",
        " 你好，Rust! 👋\n\n",
        "\t a  b\r\n",
        "<|im_start|>user\nHi<|im_end|>\n",
        "<tool_call>hello</tool_call> <think>想一想</think>",
        "<|vision_start|><|image_pad|><|vision_end|>",
    ] {
        let mut child = Command::new(artifact("RMI_LLAMA_TOKENIZE"))
            .arg("-m")
            .arg(&path)
            .args(["--stdin", "--ids", "--no-escape"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(prompt.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<u32> = serde_json::from_slice(&output.stdout).unwrap();
        eprintln!(
            "HOLO4_TOKENIZER {}",
            serde_json::json!({"input": prompt, "ids": expected})
        );
        assert_eq!(
            tokenizer.encode(
                prompt,
                EncodeOptions {
                    add_special: true,
                    parse_special: true
                }
            ),
            expected,
            "{prompt:?}"
        );
    }
}

#[test]
#[ignore = "requires real Holo4 mmproj and pinned llama.cpp projection dump"]
fn holo4_vision_projection() {
    let source = GGUFLoader::from_file(artifact("RMI_HOLO4_MMPROJ")).unwrap();
    let pool = Arc::new(ComputePool::new(8));
    let encoder = vision::VisionEncoder::from_source(&source, pool.clone()).unwrap();
    assert_eq!(
        (
            encoder.config.n_layer,
            encoder.config.n_embd,
            encoder.config.projection_dim
        ),
        (27, 1152, 2048)
    );
    let side = 128;
    // llama-mtmd-debug's --image blue supplies normalized RGB floats directly.
    let pixels: Vec<f32> = (0..side * side).flat_map(|_| [0.0, 0.0, 1.0]).collect();
    let mut scratch = vision::VisionScratchpad::new(&encoder.config);
    let grid = encoder
        .encode_image(&pixels, side, side, &mut scratch, &pool)
        .unwrap();
    assert_eq!(grid.token_count(), 16);
    let bytes = std::fs::read(artifact("RMI_HOLO4_VISION_REFERENCE")).unwrap();
    assert_eq!(bytes.len(), scratch.projected.len() * 4);
    let expected: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!(expected
        .iter()
        .chain(&scratch.projected)
        .all(|v| v.is_finite()));
    let norm: f64 = expected.iter().map(|&v| f64::from(v).powi(2)).sum();
    let diff: f64 = scratch
        .projected
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let relative_l2 = (diff / norm).sqrt();
    eprintln!(
        "Holo4 vision: {} values, relative L2={relative_l2:.8}",
        expected.len()
    );
    assert!(relative_l2 < 0.02, "vision relative L2={relative_l2}");
}

#[cfg(feature = "parity-trace")]
#[test]
#[ignore = "requires real Holo4 GGUF and a sequential Oracle log"]
fn holo4_teacher_forced_text() {
    use rust_model_inference::{models::qwen35::Qwen35Session, parity_trace};
    let log = std::fs::read_to_string(artifact("RMI_HOLO4_ORACLE_LOG")).unwrap();
    let prompts: Vec<Vec<u32>> = log
        .lines()
        .filter_map(|line| line.strip_prefix("RMI_PROMPT "))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(prompts.len(), 1);
    let prompt = &prompts[0];
    let generated: Vec<u32> = log
        .lines()
        .filter_map(|line| line.strip_prefix("RMI_GREEDY "))
        .map(|line| line.parse().unwrap())
        .collect();
    assert!(!prompt.is_empty() && generated.len() >= 2);
    let history: Vec<u32> = prompt
        .iter()
        .chain(&generated[..generated.len() - 1])
        .copied()
        .collect();
    assert!(history.len() <= 512);
    let source = GGUFLoader::from_file(artifact("RMI_HOLO4_MODEL")).unwrap();
    let mut model = Qwen35Model::from_source(&source).unwrap();
    let hidden = model.config.n_embd;
    assert!(history
        .iter()
        .all(|&id| (id as usize) < model.config.vocab_size));
    let pool = Arc::new(ComputePool::new(8));
    let policy = match std::env::var("RMI_HOLO4_COMPUTE").as_deref() {
        Ok("vulkan") => rust_model_inference::compute::ComputePolicy::Vulkan,
        Ok("cpu") | Err(_) => rust_model_inference::compute::ComputePolicy::Cpu,
        Ok(value) => panic!("unsupported RMI_HOLO4_COMPUTE {value}"),
    };
    let mut session = Qwen35Session::new_with_compute(&mut model, 512, 1, pool, policy).unwrap();
    parity_trace::token_ids("qwen35.prompt_ids", prompt).unwrap();
    parity_trace::token_ids("teacher_forced_ids", &history).unwrap();
    let embeddings = session.embed_tokens(prompt).unwrap();
    parity_trace::checkpoint(
        "qwen35.embedding",
        None,
        &[prompt.len(), hidden],
        &embeddings,
    )
    .unwrap();
    for (position, &token) in history.iter().enumerate() {
        let logits = session
            .step_with_tokens(&[token], &[[position; 4]])
            .unwrap();
        assert!(logits.iter().all(|value| value.is_finite()));
    }
}
