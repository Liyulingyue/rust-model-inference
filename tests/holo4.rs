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
    let mut session = Qwen35Session::new_with_prefill_batch_size(&mut model, 512, 1, pool).unwrap();
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
