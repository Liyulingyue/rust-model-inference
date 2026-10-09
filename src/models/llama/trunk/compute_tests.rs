use super::session::LlamaSession;
use crate::app::cli::KvFormat;
use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
use std::collections::HashMap;

struct DenseFixture {
    metadata: HashMap<String, MetaValue>,
    tensors: HashMap<String, (TensorInfo, Vec<u8>)>,
}
impl TensorSource for DenseFixture {
    fn metadata(&self, name: &str) -> Option<&MetaValue> {
        self.metadata.get(name)
    }
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name).map(|x| &x.0)
    }
    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(|x| x.1.as_slice())
    }
}
fn fixture() -> DenseFixture {
    fixture_with_scale(0.0005)
}

fn fixture_with_scale(scale: f32) -> DenseFixture {
    let mut source = DenseFixture {
        metadata: HashMap::new(),
        tensors: HashMap::new(),
    };
    source.metadata.insert(
        "tokenizer.ggml.add_bos_token".into(),
        MetaValue::Bool(false),
    );
    for (name, value) in [
        ("general.architecture", "llama"),
        ("tokenizer.ggml.model", "llama"),
    ] {
        source
            .metadata
            .insert(name.into(), MetaValue::String(value.into()));
    }
    source.metadata.insert(
        "tokenizer.ggml.tokens".into(),
        MetaValue::Array(
            MetaValueType::String,
            (0..8).map(|i| MetaValue::String(format!("t{i}"))).collect(),
        ),
    );
    source.metadata.insert(
        "tokenizer.ggml.scores".into(),
        MetaValue::Array(
            MetaValueType::Float32,
            (0..8).map(|_| MetaValue::Float32(0.0)).collect(),
        ),
    );
    for (name, value) in [
        ("embedding_length", 256),
        ("block_count", 2),
        ("attention.head_count", 4),
        ("attention.head_count_kv", 2),
        ("feed_forward_length", 256),
        ("context_length", 96),
    ] {
        source
            .metadata
            .insert(format!("llama.{name}"), MetaValue::Uint32(value));
    }
    source.metadata.insert(
        "llama.attention.layer_norm_rms_epsilon".into(),
        MetaValue::Float32(1e-5),
    );
    let mut add = |name: String, n_in: usize, n_out: usize, norm: bool, seed: usize| {
        let bytes = if norm {
            bytemuck::cast_slice(&vec![1.0f32; n_in]).to_vec()
        } else {
            (0..n_in * n_out)
                .flat_map(|i| {
                    crate::ops::f32_to_f16((((i * 17 + seed * 29) % 101) as f32 - 50.0) * scale)
                        .to_le_bytes()
                })
                .collect()
        };
        source.tensors.insert(
            name.clone(),
            (
                TensorInfo {
                    name,
                    dims: if norm {
                        vec![n_in as u64]
                    } else {
                        vec![n_in as u64, n_out as u64]
                    },
                    ggml_type: if norm { GGMLType::F32 } else { GGMLType::F16 },
                    offset: 0,
                },
                bytes,
            ),
        );
    };
    add("token_embd.weight".into(), 256, 8, false, 7);
    add("output.weight".into(), 256, 8, false, 9);
    add("output_norm.weight".into(), 256, 1, true, 0);
    for layer in 0..2 {
        for (i, (name, ni, no, norm)) in [
            ("attn_norm", 256, 1, true),
            ("ffn_norm", 256, 1, true),
            ("attn_q", 256, 256, false),
            ("attn_k", 256, 128, false),
            ("attn_v", 256, 128, false),
            ("attn_output", 256, 256, false),
            ("ffn_gate", 256, 256, false),
            ("ffn_up", 256, 256, false),
            ("ffn_down", 256, 256, false),
        ]
        .into_iter()
        .enumerate()
        {
            add(
                format!("blk.{layer}.{name}.weight"),
                ni,
                no,
                norm,
                i + layer * 11,
            );
        }
    }
    source
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 2e-3 + 2e-3 * b.abs(),
            "index {i}: {a} vs {b}"
        );
    }
}
#[test]
fn llama_dense_prefill_preserves_gate_and_up_roles() {
    let source = fixture_with_scale(0.003);
    let mut one =
        LlamaSession::from_source_with_max_rows(&source, 1, KvFormat::F32, 96, 1).unwrap();
    let mut batch =
        LlamaSession::from_source_with_max_rows(&source, 1, KvFormat::F32, 96, 3).unwrap();
    let reference = one.forward_logits_chunked(&[1, 2, 3], 1).unwrap();
    let actual = batch.forward_logits_chunked(&[1, 2, 3], 3).unwrap();
    close(&actual, &reference);
}

#[test]
fn llama_dense_recipe_preserves_original_cpu_bits() {
    let source = fixture();
    for rows in [1, 3] {
        let mut legacy =
            LlamaSession::from_source_with_max_rows(&source, 1, KvFormat::F16, 96, rows).unwrap();
        let mut shared =
            LlamaSession::from_source_with_max_rows(&source, 1, KvFormat::F16, 96, rows).unwrap();
        let tokens = &[1, 2, 3][..rows];
        if rows == 1 {
            legacy.forward_one_token(1).unwrap();
        } else {
            legacy
                .forward_chunk_batched_real(tokens, rows, 0, true)
                .unwrap();
        }
        assert!(
            legacy.scratch.logits.iter().all(|v| v.is_finite()),
            "legacy CPU produced non-finite logits at rows {rows}"
        );
        let actual = shared.forward_logits_chunked(tokens, rows).unwrap();
        assert!(
            actual
                .iter()
                .zip(&legacy.scratch.logits)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "rows {rows} changed CPU logits"
        );
        match (&legacy.kv_cache, &shared.kv_cache) {
            (
                crate::core::scratchpad::KvCache::F16(a),
                crate::core::scratchpad::KvCache::F16(b),
            ) => {
                assert!(a.k == b.k);
                assert!(a.v == b.v);
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires a Vulkan device"]
fn llama_dense_device_prefill_decode_reset_and_failure() {
    use crate::compute::ComputePolicy;
    let source = fixture();
    for batch in [1, 3, 64] {
        let mut cpu = LlamaSession::from_source_with_compute(
            &source,
            1,
            KvFormat::F16,
            96,
            batch,
            ComputePolicy::Cpu,
        )
        .unwrap();
        let mut gpu = LlamaSession::from_source_with_compute(
            &source,
            1,
            KvFormat::F16,
            96,
            batch,
            ComputePolicy::Vulkan,
        )
        .unwrap();
        let tokens: Vec<_> = (0..65).map(|i| i % 8).collect();
        // Keep 32 decode positions within the fixture's context.
        let tokens = &tokens[..if batch == 64 { 63 } else { 5 }];
        for repeat in 0..2 {
            let mut reference = cpu.forward_logits_chunked(tokens, batch).unwrap();
            let mut actual = gpu.forward_logits_chunked(tokens, batch).unwrap();
            close(&actual, &reference);
            for step in 0..32 {
                let best = |values: &[f32]| {
                    values
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0 as u32
                };
                let token = best(&reference);
                assert_eq!(
                    best(&actual),
                    token,
                    "batch={batch} repeat={repeat} step={step} cpu={reference:?} gpu={actual:?}"
                );
                reference = cpu.forward_logits_chunked(&[token], batch).unwrap();
                actual = gpu.forward_logits_chunked(&[token], batch).unwrap();
                close(&actual, &reference);
            }
            assert_eq!(cpu.seq_len, gpu.seq_len);
            assert!(gpu.gpu.is_some(), "batch {batch} repeat {repeat} fell back");
            if let (
                crate::core::scratchpad::KvCache::F16(a),
                crate::core::scratchpad::KvCache::F16(b),
            ) = (&cpu.kv_cache, &gpu.kv_cache)
            {
                let a: Vec<_> =
                    a.k.iter()
                        .chain(&a.v)
                        .map(|&x| crate::ops::f16_to_f32(x))
                        .collect();
                let b: Vec<_> =
                    b.k.iter()
                        .chain(&b.v)
                        .map(|&x| crate::ops::f16_to_f32(x))
                        .collect();
                close(&b, &a);
            }
            cpu.reset();
            gpu.reset();
        }
        gpu.forward_logits_chunked(&[1], 1).unwrap();
        gpu.gpu.as_mut().unwrap().fail_after_row = Some(0);
        assert!(gpu.forward_logits_chunked(&[2], 1).is_err());
        assert_eq!(gpu.seq_len, 1);
        assert!(gpu.forward_logits_chunked(&[2], 1).is_err());
    }
}

#[test]
fn llama_dense_rejects_unsupported_semantics_before_device_setup() {
    for (name, value) in [
        ("llama.attention.sliding_window", MetaValue::Uint32(4)),
        ("llama.rope.dimension_count", MetaValue::Uint32(32)),
        ("llama.rope.scaling.attn_factor", MetaValue::Float32(2.0)),
        ("llama.embedding_scale", MetaValue::Float32(2.0)),
        ("llama.residual_scale", MetaValue::Float32(2.0)),
        ("llama.logit_scale", MetaValue::Float32(2.0)),
        ("llama.attn_logit_softcapping", MetaValue::Float32(30.0)),
    ] {
        let mut source = fixture();
        source.metadata.insert(name.into(), value);
        let cpu = LlamaSession::from_source_with_compute(
            &source,
            1,
            KvFormat::F16,
            96,
            3,
            crate::compute::ComputePolicy::Cpu,
        )
        .unwrap();
        assert!(super::dense::eligible(&cpu).is_err(), "{name}");
        #[cfg(feature = "vulkan")]
        {
            let _scope = crate::compute::ComputePolicy::Cpu.cpu_scope();
            let error = LlamaSession::from_source_with_compute(
                &source,
                1,
                KvFormat::F16,
                96,
                3,
                crate::compute::ComputePolicy::Vulkan,
            )
            .err()
            .unwrap();
            assert!(error.contains("standard Llama"), "{name}: {error}");
        }
    }
}

#[test]
fn llama_dense_incremental_calls_preserve_cpu_state_and_check_bounds() {
    let source = fixture();
    let mut chunk =
        LlamaSession::from_source_with_max_rows(&source, 1, KvFormat::F16, 96, 3).unwrap();
    let mut token =
        LlamaSession::from_source_with_max_rows(&source, 1, KvFormat::F16, 96, 1).unwrap();
    chunk.forward_logits_chunked(&[1, 2, 3], 3).unwrap();
    token.forward_logits_chunked(&[1, 2, 3], 1).unwrap();
    close(
        &chunk.forward_logits_chunked(&[4, 5], 3).unwrap(),
        &token.forward_logits_chunked(&[4, 5], 1).unwrap(),
    );
    assert_eq!(chunk.seq_len, 5);
    assert!(chunk.forward_logits_chunked(&[99], 1).is_err());
    assert_eq!(chunk.seq_len, 5);
    chunk.reset();
    assert_eq!(chunk.seq_len, 0);
}

#[cfg(feature = "vulkan")]
fn mixed_fixture() -> DenseFixture {
    let mut source = fixture();
    for (name, (info, bytes)) in &mut source.tensors {
        let format = if name.contains("attn_q.weight") {
            GGMLType::Q8_0
        } else if name.contains("attn_k.weight") {
            GGMLType::Q4_0
        } else if name.contains("ffn_gate.weight")
            || name.contains("ffn_up.weight")
            || name.contains("attn_v.weight")
        {
            GGMLType::Q4K
        } else {
            continue;
        };
        let (block, size) = format.type_traits();
        let count = info.dims.iter().product::<u64>() as usize / block;
        bytes.clear();
        for row in 0..count {
            let mut packed = vec![0u8; size];
            packed[..2].copy_from_slice(&crate::ops::f32_to_f16(0.0005).to_le_bytes());
            match format {
                GGMLType::Q8_0 => {
                    for (i, v) in packed[2..].iter_mut().enumerate() {
                        *v = (((i + row) % 16) as i8 - 8) as u8;
                    }
                }
                GGMLType::Q4_0 => {
                    for (i, v) in packed[2..].iter_mut().enumerate() {
                        *v = ((i + row) % 16) as u8 | ((((i * 3 + row) % 16) as u8) << 4);
                    }
                }
                GGMLType::Q4K => {
                    packed[2..4].copy_from_slice(&crate::ops::f32_to_f16(0.0005).to_le_bytes());
                    packed[4..16].fill(1);
                    for (i, v) in packed[16..].iter_mut().enumerate() {
                        *v = ((i + row) % 16) as u8 | ((((i * 3 + row) % 16) as u8) << 4);
                    }
                }
                _ => unreachable!(),
            }
            bytes.extend(packed);
        }
        info.ggml_type = format;
    }
    source
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires a Vulkan device"]
fn llama_dense_device_mixed_quantization() {
    use crate::compute::ComputePolicy;
    let source = mixed_fixture();
    for batch in [1, 3, 64] {
        let mut cpu = LlamaSession::from_source_with_compute(
            &source,
            2,
            KvFormat::F16,
            96,
            batch,
            ComputePolicy::Cpu,
        )
        .unwrap();
        let mut gpu = LlamaSession::from_source_with_compute(
            &source,
            2,
            KvFormat::F16,
            96,
            batch,
            ComputePolicy::Vulkan,
        )
        .unwrap();
        let tokens: Vec<_> = (0..63).map(|i| i % 8).collect();
        let mut a = cpu.forward_logits_chunked(&tokens, batch).unwrap();
        let mut b = gpu.forward_logits_chunked(&tokens, batch).unwrap();
        for _ in 0..32 {
            close(&b, &a);
            let best = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as u32
            };
            let token = best(&a);
            assert_eq!(best(&b), token);
            a = cpu.forward_logits_chunked(&[token], 1).unwrap();
            b = gpu.forward_logits_chunked(&[token], 1).unwrap();
        }
        assert!(gpu.gpu.is_some());
    }
}
