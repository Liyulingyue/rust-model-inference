//! Contract checks for the LFM3-1.7B GGUF shipped in `models/`.
//!
//! The file is intentionally skippable so normal CI does not require a
//! 1.8-GB model download. Set `RMI_LFM3_MODEL` to compare another copy.

use std::path::PathBuf;

use rust_model_inference::core::loader::{model_config_from_source, GGUFLoader};
use rust_model_inference::core::scratchpad::KvFormat;
use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
use rust_model_inference::models::lfm3::{Lfm3Config, Lfm3Session};
use rust_model_inference::prompt::jinja::{self, Options};

const DEFAULT_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf"
);

fn model_path() -> Option<PathBuf> {
    let path = std::env::var("RMI_LFM3_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_MODEL));
    if path.exists() {
        Some(path)
    } else {
        eprintln!("skipping: {} not present", path.display());
        None
    }
}

#[test]
fn lfm3_gguf_contract_and_tokenizer() {
    let Some(path) = model_path() else { return };
    let source = GGUFLoader::from_file(&path).expect("open LFM3 GGUF");
    assert_eq!(
        source
            .metadata("general.architecture")
            .and_then(|value| value.to_string_val()),
        Some("lfm3")
    );

    let config = Lfm3Config::from_source(&source).expect("parse LFM3 metadata");
    assert_eq!(
        (config.n_embd, config.n_layer, config.n_head, config.n_ff),
        (2048, 24, 16, 7680)
    );
    assert_eq!(config.n_embd_head_k, 128);
    assert_eq!(config.n_embd_head_v, 128);
    assert_eq!(config.d_conv, 6);
    assert_eq!(config.sliding_window, 256);
    assert_eq!(config.rope_freq_base, 1_000_000.0);
    assert_eq!(config.rope_freq_base_swa, 10_000.0);
    assert_eq!(
        config.n_head_kv_per_layer,
        vec![0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0]
    );
    assert_eq!(
        config.sliding_window_pattern,
        vec![
            false, false, false, false, true, false, false, false, true, false, false, false, true,
            false, false, false, true, false, false, false, true, false, false, false,
        ]
    );
    assert!(source.tensor_info("output_norm.weight").is_some());
    assert!(source.tensor_info("output.weight").is_none());
    for layer in 0..config.n_layer {
        assert!(source
            .tensor_info(&format!("blk.{layer}.post_attention_norm.weight"))
            .is_some());
        assert!(source
            .tensor_info(&format!("blk.{layer}.post_ffw_norm.weight"))
            .is_some());
        if config.n_head_kv_per_layer[layer] > 0 {
            assert!(source
                .tensor_info(&format!("blk.{layer}.attn_gate.weight"))
                .is_some());
        }
    }

    let generic = model_config_from_source(&source).expect("generic model config accepts lfm3");
    assert_eq!(generic.n_head_kv, 4);

    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
        .expect("load LFM3 tokenizer");
    for (text, expected) in [
        ("", vec![]),
        ("Hello, LFM3!", vec![35808, 20, 402, 37115, 27, 9]),
        (" \t\n  hello\n\n", vec![229, 1970, 229, 63532, 8]),
        (
            "你好，LFM3!\n\t café  🦀",
            vec![
                11253, 5899, 697, 52, 37115, 27, 2159, 206, 34998, 229, 23304, 108, 231,
            ],
        ),
        (
            "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>",
            vec![
                124894, 124899, 5922, 207, 41619, 124900, 207, 124899, 63514, 207, 124901,
            ],
        ),
    ] {
        let ids = tokenizer.encode(
            text,
            EncodeOptions {
                add_special: false,
                parse_special: true,
            },
        );
        assert_eq!(ids, expected, "tokenizer fixture: {text:?}");
        assert_eq!(
            tokenizer.encode(
                text,
                EncodeOptions {
                    add_special: true,
                    parse_special: true
                }
            ),
            ids
        );
    }
    assert_eq!(tokenizer.bos_id(), Some(124894));
    assert_eq!(tokenizer.eos_id(), Some(124900));
}

#[test]
fn lfm3_forward_smoke() {
    if std::env::var_os("RMI_LFM3_FORWARD_SMOKE").is_none() {
        eprintln!("skipping: set RMI_LFM3_FORWARD_SMOKE=1 for the real-weight forward smoke");
        return;
    }
    let Some(path) = model_path() else { return };
    let source = GGUFLoader::from_file(&path).expect("open LFM3 GGUF");
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
        .expect("load LFM3 tokenizer");
    let ids = jinja::single_turn_tokens(
        &tokenizer,
        &Options {
            jinja: true,
            file: None,
        },
        &|key| source.metadata(key).cloned(),
        "Reply with exactly one short sentence.",
        true,
    )
    .expect("render LFM3 chat template")
    .expect("LFM3 ships a chat template");
    assert!(!ids.is_empty());
    assert_eq!(
        ids,
        vec![
            124894, 124899, 5922, 207, 57922, 415, 8043, 734, 2789, 12683, 22, 124900, 207, 124899,
            63514, 207, 124901,
        ],
        "LFM3 GGUF Jinja prompt must match the llama.cpp oracle",
    );
    let mut session =
        Lfm3Session::from_source(&source, 1, KvFormat::F32, 512).expect("load LFM3 session");
    let logits = session
        .forward_logits_chunked(&ids, 1)
        .expect("run LFM3 forward");
    assert_eq!(logits.len(), session.vocab);
    assert!(logits.iter().all(|value| value.is_finite()));
    let top = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(id, _)| id)
        .unwrap();
    assert_eq!(
        top, 597,
        "LFM3 greedy next token must match llama.cpp (The)"
    );
}

/// Build tools/oracle/lfm3/oracle.cpp and set RMI_LFM3_ORACLE to enable.
#[test]
fn lfm3_oracle_alignment() {
    let Some(oracle) = std::env::var_os("RMI_LFM3_ORACLE") else {
        return;
    };
    let path = model_path().expect("alignment requires RMI_LFM3_MODEL");
    let source = GGUFLoader::from_file(&path).unwrap();
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap();
    let root = std::env::var_os("RMI_LFM3_ALIGN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("rmi-lfm3-alignment-{}", std::process::id()))
        });
    std::fs::create_dir_all(&root).unwrap();
    let mut cases = vec![
        ("hello", "Hello".to_owned()),
        ("chat", "<|startoftext|><|im_start|>user\nReply with exactly one short sentence.<|im_end|>\n<|im_start|>assistant\n<think>".to_owned()),
        ("unicode", "你好，LFM3!\n\t café  🦀".to_owned()),
    ];
    if std::env::var_os("RMI_LFM3_LONG_CONTEXT").is_some() {
        cases = vec![("window", "The cat sat on the mat. A dog ran. ".repeat(24))];
    }
    if let Ok(case) = std::env::var("RMI_LFM3_CASE") {
        cases.retain(|(name, _)| *name == case);
    }
    assert!(!cases.is_empty());
    let checkpoints = std::env::var_os("RMI_LFM3_CHECKPOINTS").is_some();
    for (name, prompt) in cases {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("prompt.txt");
        std::fs::write(&file, &prompt).unwrap();
        let mut command = std::process::Command::new(&oracle);
        command.arg(&path).arg(&file).arg(&dir).arg("8");
        if !checkpoints {
            command.env("LFM3_NO_TRACE", "1");
        }
        let status = command
            .stdout(std::fs::File::create(dir.join("oracle.log")).unwrap())
            .stderr(std::fs::File::create(dir.join("oracle.err")).unwrap())
            .status()
            .unwrap();
        assert!(
            status.success(),
            "{name}: oracle failed; see {}",
            dir.display()
        );
        let streams = std::fs::read_to_string(dir.join("tokens.txt")).unwrap();
        let mut lines = streams.lines();
        let parse = |line: &str| {
            line.split_whitespace()
                .map(|value| value.parse::<u32>().unwrap())
                .collect::<Vec<_>>()
        };
        let reference_ids = parse(lines.next().unwrap());
        let generated = parse(lines.next().unwrap());
        let ids = tokenizer.encode(
            &prompt,
            EncodeOptions {
                add_special: false,
                parse_special: true,
            },
        );
        assert_eq!(ids, reference_ids, "{name}: tokenizer differs");
        if name == "window" {
            assert!(ids.len() > 256);
        }
        if checkpoints {
            assert!(
                cfg!(feature = "parity-trace"),
                "checkpoints require --features parity-trace"
            );
            let trace = dir.join("rust.jsonl");
            if trace.exists() {
                std::fs::remove_file(&trace).unwrap();
            }
            std::env::set_var("RMI_PARITY_TRACE", trace);
        }
        let mut session =
            Lfm3Session::from_source(&source, 1, KvFormat::F32, ids.len() + 8).unwrap();
        let mut logits = session.forward_logits_chunked(&ids, 7).unwrap();
        let mut max_error = 0.0f32;
        let mut max_rmse = 0.0f64;
        for (step, &next) in generated.iter().enumerate() {
            let bytes = std::fs::read(dir.join(format!("logits-{step}.f32"))).unwrap();
            let reference: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            assert_eq!(logits.len(), reference.len());
            let error = logits
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let rmse = (logits
                .iter()
                .zip(&reference)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum::<f64>()
                / logits.len() as f64)
                .sqrt();
            assert!(logits.iter().all(|v| v.is_finite()));
            max_error = max_error.max(error);
            max_rmse = max_rmse.max(rmse);
            // Empirical Q8_0/SIMD regression bounds for this pinned CPU Oracle.
            assert!(
                error <= 1.25 && rmse <= 0.20,
                "{name}/{step}: max={error}, rmse={rmse}"
            );
            let top = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            assert_eq!(
                top, next,
                "{name}/{step}: greedy differs, max={error}, rmse={rmse}"
            );
            if step + 1 < generated.len() {
                logits = session.forward_logits_chunked(&[next], 1).unwrap();
            }
        }
        eprintln!("{name}: {} prompt IDs exact; {generated:?}; max_abs={max_error:.8}, max_rmse={max_rmse:.8}; {}", ids.len(), dir.display());
    }
}
