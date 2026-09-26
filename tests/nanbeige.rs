use rust_model_inference::core::loader::model_config_from_source;
use rust_model_inference::core::tensor::{MetaValue, MetaValueType, TensorInfo, TensorSource};
use rust_model_inference::core::tokenizer::{EncodeOptions, SPMTokenizer};
use std::collections::HashMap;

#[derive(Default)]
struct MapTensorSource {
    metadata: HashMap<String, MetaValue>,
}
impl TensorSource for MapTensorSource {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }
    fn tensor_info(&self, _: &str) -> Option<&TensorInfo> {
        None
    }
    fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
        None
    }
}
#[test]
fn nanbeige_uses_explicit_head_dimension() {
    let mut source = MapTensorSource::default();
    source.metadata.insert(
        "general.architecture".into(),
        MetaValue::String("nanbeige".into()),
    );
    for (key, value) in [
        ("embedding_length", 3072),
        ("block_count", 22),
        ("attention.head_count", 48),
        ("attention.head_count_kv", 8),
        ("attention.key_length", 128),
        ("attention.value_length", 128),
        ("feed_forward_length", 10752),
        ("context_length", 262144),
        ("vocab_size", 166144),
    ] {
        source
            .metadata
            .insert(format!("nanbeige.{key}"), MetaValue::Uint32(value));
    }
    source.metadata.insert(
        "nanbeige.attention.layer_norm_rms_epsilon".into(),
        MetaValue::Float32(1e-5),
    );
    assert_eq!(model_config_from_source(&source).unwrap().n_embd_head, 128);
    source
        .metadata
        .insert("nanbeige.attention.key_length".into(), MetaValue::Uint32(0));
    assert!(model_config_from_source(&source).is_err());
}

#[test]
fn spm_merges_highest_score_then_leftmost_pair() {
    let metadata = HashMap::from([
        ("tokenizer.ggml.model", MetaValue::String("llama".into())),
        ("tokenizer.ggml.add_space_prefix", MetaValue::Bool(false)),
        ("tokenizer.ggml.add_bos_token", MetaValue::Bool(false)),
        (
            "tokenizer.ggml.tokens",
            MetaValue::Array(
                MetaValueType::String,
                ["<unk>", "<s>", "</s>", "a", "b", "c", "ab", "bc"]
                    .map(|s| MetaValue::String(s.into()))
                    .to_vec(),
            ),
        ),
        (
            "tokenizer.ggml.scores",
            MetaValue::Array(
                MetaValueType::Float32,
                [-1000., -1000., -1000., 0., 0., 0., 10., 1.]
                    .map(MetaValue::Float32)
                    .to_vec(),
            ),
        ),
    ]);
    let tokenizer = SPMTokenizer::from_gguf_metadata(|k| metadata.get(k).cloned()).unwrap();
    let options = EncodeOptions {
        add_special: false,
        parse_special: false,
    };
    assert_eq!(tokenizer.encode("abc", options), [6, 5]);
    let mut equal = metadata;
    if let MetaValue::Array(_, scores) = equal.get_mut("tokenizer.ggml.scores").unwrap() {
        scores[7] = MetaValue::Float32(10.);
    }
    let tokenizer = SPMTokenizer::from_gguf_metadata(|k| equal.get(k).cloned()).unwrap();
    assert_eq!(tokenizer.encode("abc", options), [6, 5]);
}

#[test]
#[ignore = "requires RMI_NANBEIGE_MODEL"]
fn nanbeige_has_shared_weights_and_independent_loop_kv() {
    use rust_model_inference::app::cli::KvFormat;
    use rust_model_inference::models::llama::trunk::LlamaSession;
    let loader =
        rust_model_inference::GGUFLoader::from_file(&std::env::var("RMI_NANBEIGE_MODEL").unwrap())
            .unwrap();
    for kv_format in [KvFormat::F32, KvFormat::F16] {
        let mut session = LlamaSession::from_source(&loader, 1, kv_format, 32).unwrap();
        assert_eq!(session.config.n_layer, 44);
        assert_eq!(session.weights.layers.len(), 22);
        let tokens = [166100, 52338, 152372, 1716, 152956];
        let expected = session.forward_logits_per_token(&tokens).unwrap();
        let mut batched =
            LlamaSession::from_source_with_max_rows(&loader, 4, kv_format, 32, 2).unwrap();
        let actual = batched.forward_logits_chunked(&tokens, 2).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{kv_format:?} logit {index}"
            );
        }
        match &session.kv_cache {
            rust_model_inference::core::scratchpad::KvCache::F32(cache) => {
                let stride = 32 * 1024;
                assert_ne!(&cache.k[..2048], &cache.k[22 * stride..22 * stride + 2048]);
            }
            rust_model_inference::core::scratchpad::KvCache::F16(cache) => {
                let stride = 32 * 1024;
                assert_ne!(&cache.k[..2048], &cache.k[22 * stride..22 * stride + 2048]);
            }
        }
    }
}

#[test]
#[ignore = "requires RMI_NANBEIGE_MODEL"]
fn nanbeige_tokenizer_matches_pinned_oracle() {
    let loader =
        rust_model_inference::GGUFLoader::from_file(&std::env::var("RMI_NANBEIGE_MODEL").unwrap())
            .unwrap();
    let tokenizer = SPMTokenizer::from_gguf_metadata(|k| loader.metadata(k).cloned()).unwrap();
    // llama.cpp b96806d96061049a5b574269b049bf6241d63d46, add_special/parse_special=true.
    for (text, ids) in [
        ("", vec![166100]),
        ("Hello", vec![166100, 23877]),
        ("你好，世界！", vec![166100, 52338, 152372, 1716, 152956]),
        ("a  b\t\nc", vec![166100, 261, 259, 152364, 12, 13, 152355]),
        (
            "é e\u{301} 日本語 😀",
            vec![166100, 5354, 90391, 35632, 156612, 152343, 158621],
        ),
        (
            "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n",
            vec![
                166100, 166100, 2028, 13, 22779, 166101, 152343, 13, 166100, 13886, 13, 166103,
                152343, 13,
            ],
        ),
    ] {
        assert_eq!(
            tokenizer.encode(
                text,
                EncodeOptions {
                    add_special: true,
                    parse_special: true
                }
            ),
            ids,
            "{text:?}"
        );
    }
    assert_eq!(tokenizer.bos_id(), Some(166100));
    assert_eq!(tokenizer.eos_id(), Some(166101));
}

struct ChangedSource<'a> {
    inner: &'a rust_model_inference::GGUFLoader,
    key: &'a str,
    value: MetaValue,
    tensor: Option<TensorInfo>,
}
impl TensorSource for ChangedSource<'_> {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        if key == self.key {
            Some(&self.value)
        } else {
            self.inner.metadata(key)
        }
    }
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor
            .as_ref()
            .filter(|t| t.name == name)
            .or_else(|| self.inner.tensor_info(name))
    }
    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.inner.tensor_slice(name)
    }
}

#[test]
#[ignore = "requires RMI_NANBEIGE_MODEL"]
fn nanbeige_rejects_invalid_metadata_and_projection_shape() {
    use rust_model_inference::app::cli::KvFormat;
    use rust_model_inference::models::llama::trunk::LlamaSession;
    let loader =
        rust_model_inference::GGUFLoader::from_file(&std::env::var("RMI_NANBEIGE_MODEL").unwrap())
            .unwrap();
    for (key, value) in [
        ("nanbeige.num_loops", MetaValue::Uint32(0)),
        ("nanbeige.num_loops", MetaValue::Uint32(u32::MAX)),
        ("nanbeige.skip_loop_final_norm", MetaValue::Uint32(0)),
        ("nanbeige.attention.value_length", MetaValue::Uint32(64)),
    ] {
        let source = ChangedSource {
            inner: &loader,
            key,
            value,
            tensor: None,
        };
        assert!(
            LlamaSession::from_source(&source, 1, KvFormat::F32, 32).is_err(),
            "{key}"
        );
    }
    let mut tensor = loader.tensor_info("blk.0.attn_q.weight").unwrap().clone();
    tensor.dims[1] = 3072;
    let source = ChangedSource {
        inner: &loader,
        key: "",
        value: MetaValue::Bool(false),
        tensor: Some(tensor),
    };
    let error = LlamaSession::from_source(&source, 1, KvFormat::F32, 32)
        .err()
        .unwrap();
    assert!(error.contains("blk.0.attn_q.weight shape"), "{error}");
}

#[test]
#[cfg(feature = "parity-trace")]
#[ignore = "requires RMI_NANBEIGE_MODEL, RMI_NANBEIGE_ORACLE_TRACE with matching CPU/KV mode"]
fn nanbeige_matches_oracle_bit_for_bit() {
    use rust_model_inference::app::cli::KvFormat;
    use rust_model_inference::core::tokenizer::load_tokenizer;
    let kv_format = if std::env::var_os("RMI_NANBEIGE_F16").is_some() {
        KvFormat::F16
    } else {
        KvFormat::F32
    };
    let loader =
        rust_model_inference::GGUFLoader::from_file(&std::env::var("RMI_NANBEIGE_MODEL").unwrap())
            .unwrap();
    let oracle_path = std::env::var("RMI_NANBEIGE_ORACLE_TRACE").unwrap();
    let read = |path: &std::path::Path| -> Vec<serde_json::Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect()
    };
    let oracle = read(std::path::Path::new(&oracle_path));
    let prompt = std::env::var("RMI_NANBEIGE_PROMPT").unwrap_or_else(|_| "Hello".into());
    let tokenizer = load_tokenizer(|k| loader.metadata(k).cloned()).unwrap();
    let tokens = tokenizer.encode(
        &prompt,
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert_eq!(
        serde_json::json!(tokens),
        oracle[0]["token_ids"],
        "tokenizer IDs"
    );
    let max_tokens = oracle.last().unwrap()["token_ids"]
        .as_array()
        .unwrap()
        .len();
    let trace = std::env::temp_dir().join(format!("nanbeige-rust-{}.jsonl", std::process::id()));
    std::env::set_var("RMI_PARITY_TRACE", &trace);
    rust_model_inference::models::llama::run_inference_tokens(
        &loader, tokens, max_tokens, 0.0, 1, false, false, kv_format, 512, 1.0,
    )
    .unwrap();
    let rust = read(&trace);
    assert_eq!(
        rust.len(),
        oracle.len(),
        "trace count; retained {}",
        trace.display()
    );
    for (index, (actual, expected)) in rust.iter().zip(&oracle).enumerate() {
        if actual.get("token_ids").is_some() {
            assert_eq!(actual["name"], expected["name"]);
            assert_eq!(actual["token_ids"], expected["token_ids"]);
            continue;
        }
        for key in [
            "name",
            "layer",
            "step",
            "shape",
            "len",
            "occurrence",
            "token_ids",
        ] {
            assert_eq!(
                actual[key],
                expected[key],
                "record {index}, field {key}; retained {}",
                trace.display()
            );
        }
        if let Some(path) = actual["binary_path"].as_str() {
            let got = std::fs::read(path).unwrap();
            let want = std::fs::read(expected["binary_path"].as_str().unwrap()).unwrap();
            assert_eq!(got.len(), want.len());
            for (lane, (a, b)) in got.chunks_exact(4).zip(want.chunks_exact(4)).enumerate() {
                assert_eq!(a, b, "first divergence: record={index} name={} layer={} step={} lane={lane}; retained {}", actual["name"], actual["layer"], actual["step"], trace.display());
            }
        }
    }
    for record in &rust {
        if let Some(path) = record["binary_path"].as_str() {
            std::fs::remove_file(path).unwrap();
        }
    }
    let mut session = rust_model_inference::models::llama::trunk::LlamaSession::from_source(
        &loader, 1, kv_format, 512,
    )
    .unwrap();
    let mut token = None;
    for record in &oracle {
        if record["name"] == "input_token" {
            token = Some(record["token_ids"][0].as_u64().unwrap() as u32);
        }
        if record["name"] == "result_output" {
            let actual = session
                .forward_logits_per_token(&[token.take().unwrap()])
                .unwrap();
            let expected = std::fs::read(record["binary_path"].as_str().unwrap()).unwrap();
            for (lane, (got, want)) in actual.iter().zip(expected.chunks_exact(4)).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    u32::from_le_bytes(want.try_into().unwrap()),
                    "session logits step={} lane={lane}",
                    record["step"]
                );
            }
        }
    }
    std::fs::remove_file(trace).unwrap();
}
