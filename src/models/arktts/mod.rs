//! Audio8-TTS-Realistic-Air (official ArkTTS DualAR and 44.1 kHz codec).

pub mod codec;
mod transformer;

use crate::compute::linear::{LinearBinding, LinearExecutor, LinearMode};
use crate::compute::ComputePolicy;
use crate::core::tensor::{
    load_f32_tensor, require_tensor_type, GGMLType, MetaValue, TensorSource,
};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{QuantizedTensor, Weight};
use rand::{Rng, SeedableRng};
use serde_json::Value;
use std::sync::Arc;
use transformer::{Cache, Shape, Transformer};

pub const SAMPLE_RATE: u32 = 44_100;
pub const SAMPLES_PER_FRAME: usize = 2048;
const WIDTH: usize = 896;
const VOCAB: usize = 155776;
const SEMANTIC: u32 = 151678;
const EOS: u32 = 151645;
const CONTEXT: usize = 2048;
pub type Frame = [u32; 10];

struct Matrix<'a> {
    weight: Weight<'a>,
    bias: Vec<f32>,
}

impl<'a> Matrix<'a> {
    fn load(
        s: &'a dyn TensorSource,
        p: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Self, String> {
        let name = format!("{p}.weight");
        let kind = require_tensor_type(
            s,
            &name,
            &[input as u64, output as u64],
            &[
                GGMLType::F32,
                GGMLType::F16,
                GGMLType::BF16,
                GGMLType::Q8_0,
                GGMLType::Q4_0,
                GGMLType::Q4_1,
                GGMLType::Q4K,
                GGMLType::Q5K,
                GGMLType::Q6K,
            ],
        )?;
        if !input.is_multiple_of(kind.type_traits().0) {
            return Err(format!(
                "ArkTTS matrix {name} has an unaligned {kind:?} row"
            ));
        }
        let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
            s.tensor_slice(&name).unwrap(),
            kind,
            input,
            output,
        ));
        weight.n_in = input;
        weight.n_out = output;
        Ok(Self {
            weight,
            bias: if bias {
                vector(s, &format!("{p}.bias"), output)?
            } else {
                Vec::new()
            },
        })
    }

    fn forward(
        &self,
        input: &[f32],
        linear: &mut LinearExecutor<'_, '_>,
    ) -> Result<Vec<f32>, String> {
        if input.is_empty() || !input.len().is_multiple_of(self.weight.n_in) {
            return Err("invalid ArkTTS projection input".into());
        }
        let mut output = vec![0.0; input.len() / self.weight.n_in * self.weight.n_out];
        let id = linear.id_for(&self.weight).map_err(|e| e.to_string())?;
        for (x, y) in input
            .chunks(self.weight.n_in * linear.max_rows())
            .zip(output.chunks_mut(self.weight.n_out * linear.max_rows()))
        {
            linear
                .run(id, x, x.len() / self.weight.n_in, y)
                .map_err(|e| e.to_string())?;
        }
        if !self.bias.is_empty() {
            for row in output.chunks_exact_mut(self.bias.len()) {
                for (v, bias) in row.iter_mut().zip(&self.bias) {
                    *v += bias;
                }
            }
        }
        Ok(output)
    }

    fn embed(&self, token: u32) -> Vec<f32> {
        let mut row = vec![0.0; self.weight.n_in];
        self.weight.embedding_lookup(token, &mut row);
        row
    }
}

fn vector(s: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    require_tensor_type(s, name, &[len as u64], &[GGMLType::F32, GGMLType::BF16])?;
    let values = load_f32_tensor(s, name, &[len as u64])?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(format!("non-finite ArkTTS tensor: {name}"));
    }
    Ok(values)
}

pub fn validate_compute(policy: ComputePolicy) -> Result<(), String> {
    if policy == ComputePolicy::Vulkan && crate::ops::scalar_mode() {
        return Err("ArkTTS strict Vulkan execution is incompatible with scalar mode".into());
    }
    policy.check_build()
}

fn linear<'m, 'w>(
    matrices: Vec<&'m Matrix<'w>>,
    compute: ComputePolicy,
    threads: usize,
) -> Result<LinearExecutor<'m, 'w>, String> {
    // Bound staging even for the 155776-row tied vocabulary head.
    LinearExecutor::new(
        compute,
        matrices
            .into_iter()
            .map(|m| LinearBinding {
                weight: &m.weight,
                mode: LinearMode::Prepared,
            })
            .collect(),
        32,
        Arc::new(ComputePool::new(threads)),
    )
    .map_err(|e| e.to_string())
}

fn config(s: &dyn TensorSource, arch: &str) -> Result<Value, String> {
    if s.metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        != Some(arch)
    {
        return Err(format!("expected {arch} GGUF component"));
    }
    let key = format!("{arch}.config_json");
    let raw = s
        .metadata(&key)
        .and_then(MetaValue::to_string_val)
        .ok_or_else(|| format!("missing {key}"))?;
    let config: Value = serde_json::from_str(raw).map_err(|e| format!("invalid {key}: {e}"))?;
    // Pin every computation-affecting field; near-matching Qwen/DualAR variants are not interchangeable.
    let contract: Value = serde_json::from_str(include_str!("contract.json")).unwrap();
    for (field, expected) in contract.as_object().unwrap() {
        if config.get(field) != Some(expected) {
            return Err(format!("unsupported ArkTTS config {field}"));
        }
    }
    Ok(config)
}

fn bundle(s: &dyn TensorSource) -> Result<String, String> {
    s.metadata("arktts.bundle_id")
        .and_then(MetaValue::to_string_val)
        .filter(|id| id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit()))
        .map(str::to_owned)
        .ok_or_else(|| {
            "missing or invalid ArkTTS bundle identity; re-export both components together".into()
        })
}

pub fn validate_pair(main: &dyn TensorSource, codec: &dyn TensorSource) -> Result<(), String> {
    config(main, "arktts")?;
    config(codec, "arktts_codec")?;
    if bundle(main)? != bundle(codec)? {
        return Err("ArkTTS main and codec belong to different checkpoint bundles".into());
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
            seed: 42,
        }
    }
}

impl Sampling {
    pub fn validate(self) -> Result<(), String> {
        if !self.temperature.is_finite()
            || self.temperature < 0.0
            || self.top_k == 0
            || !self.top_p.is_finite()
            || self.top_p <= 0.0
            || self.top_p > 1.0
        {
            return Err(
                "ArkTTS requires finite temperature >= 0, top_k > 0 and 0 < top_p <= 1".into(),
            );
        }
        Ok(())
    }
}

pub struct Model<'a> {
    bundle: String,
    tokenizer: tokenizers::Tokenizer,
    embeddings: Matrix<'a>,
    code_embeddings: Matrix<'a>,
    fast_embeddings: Matrix<'a>,
    fast_output: Matrix<'a>,
    slow: Transformer<'a>,
    fast: Transformer<'a>,
    pool: rayon::ThreadPool,
    compute: ComputePolicy,
    threads: usize,
}

impl<'a> Model<'a> {
    pub fn from_source(s: &'a dyn TensorSource, threads: usize) -> Result<Self, String> {
        Self::from_source_with_compute(s, threads, ComputePolicy::legacy())
    }

    pub fn from_source_with_compute(
        s: &'a dyn TensorSource,
        threads: usize,
        compute: ComputePolicy,
    ) -> Result<Self, String> {
        validate_compute(compute)?;
        config(s, "arktts")?;
        if threads == 0 {
            return Err("ArkTTS threads must be positive".into());
        }
        let tokenizer = s
            .metadata("arktts.tokenizer_json")
            .and_then(MetaValue::to_string_val)
            .ok_or("missing ArkTTS tokenizer")?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer.as_bytes())
            .map_err(|e| format!("invalid ArkTTS tokenizer: {e}"))?;
        for (token, id) in [
            ("<|im_start|>", 151644),
            ("<|im_end|>", EOS),
            ("<|voice|>", 151673),
            ("<|semantic:0|>", SEMANTIC),
        ] {
            if tokenizer.token_to_id(token) != Some(id) {
                return Err(format!("ArkTTS tokenizer special token mismatch: {token}"));
            }
        }
        let shape = Shape {
            width: WIDTH,
            heads: 14,
            kv: 2,
            intermediate: 4864,
            epsilon: 1e-6,
            rope_base: 1_000_000.0,
            window: None,
        };
        Ok(Self {
            compute,
            threads,
            bundle: bundle(s)?,
            tokenizer,
            embeddings: Matrix::load(s, "embeddings", WIDTH, VOCAB, false)?,
            code_embeddings: Matrix::load(s, "codebook_embeddings", WIDTH, 40960, false)?,
            fast_embeddings: Matrix::load(s, "fast_embeddings", WIDTH, 4096, false)?,
            fast_output: Matrix::load(s, "fast_output", WIDTH, 4096, false)?,
            slow: Transformer::load(s, "layers", "norm", 24, shape, true, false)?,
            fast: Transformer::load(s, "fast_layers", "fast_norm", 4, shape, false, false)?,
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| e.to_string())?,
        })
    }

    fn encode_parts(&self, parts: &[&str]) -> Result<Vec<u32>, String> {
        let mut ids = Vec::new();
        for part in parts {
            ids.extend(
                self.tokenizer
                    .encode(*part, false)
                    .map_err(|e| e.to_string())?
                    .get_ids(),
            );
        }
        Ok(ids)
    }

    pub fn prompt(
        &self,
        text: &str,
        reference: Option<(&str, &[Frame])>,
    ) -> Result<Vec<[u32; 11]>, String> {
        let text = clean_text(text);
        if text.is_empty() {
            return Err("ArkTTS text must not be empty".into());
        }
        let mut columns = Vec::new();
        let suffix = [
            "<|im_end|>\n",
            "<|im_start|>user\n",
            &text,
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        ];
        if let Some((transcript, codes)) = reference {
            let transcript = clean_text(transcript);
            if transcript.is_empty()
                || codes.is_empty()
                || codes.iter().flatten().any(|&v| v >= 4096)
            {
                return Err(
                    "ArkTTS reference needs a transcript and nonempty codes in 0..4096".into(),
                );
            }
            let speaker_pattern = regex::Regex::new(r"<\|speaker:\d+\|>").unwrap();
            let transcript = if speaker_pattern.is_match(&transcript) {
                transcript
            } else {
                format!("<|speaker:0|>{transcript}")
            };
            let ids = self.encode_parts(&[
                "<|im_start|>system\n",
                "convert the provided text to speech reference to the following:\n\nText:\n",
                &transcript,
                "\n\nSpeech:\n",
            ])?;
            columns.extend(ids.into_iter().map(text_column));
            for frame in codes {
                let mut column = [0; 11];
                column[0] = SEMANTIC + frame[0];
                column[1..].copy_from_slice(frame);
                columns.push(column);
            }
            columns.extend(self.encode_parts(&suffix)?.into_iter().map(text_column));
        } else {
            let mut parts = vec![
                "<|im_start|>system\n",
                "convert the provided text to speech",
            ];
            parts.extend_from_slice(&suffix);
            columns.extend(self.encode_parts(&parts)?.into_iter().map(text_column));
        }
        if columns.len() >= CONTEXT {
            return Err(
                "ArkTTS prompt leaves no generation space in the 2048-position context".into(),
            );
        }
        Ok(columns)
    }

    fn embed(&self, columns: &[[u32; 11]]) -> Result<Vec<f32>, String> {
        let mut values = Vec::with_capacity(columns.len() * WIDTH);
        for column in columns {
            if column[0] as usize >= VOCAB || column[1..].iter().any(|&v| v >= 4096) {
                return Err("invalid ArkTTS packed token".into());
            }
            let mut hidden = self.embeddings.embed(column[0]);
            if (SEMANTIC..SEMANTIC + 4096).contains(&column[0]) {
                let mut sum = vec![0.0; WIDTH];
                for (index, &id) in column[1..].iter().enumerate() {
                    let row = self.code_embeddings.embed(id + index as u32 * 4096);
                    for (v, add) in sum.iter_mut().zip(row) {
                        *v += add;
                    }
                }
                for (v, add) in hidden.iter_mut().zip(sum) {
                    *v += add;
                }
            }
            values.extend(hidden);
        }
        Ok(values)
    }

    pub fn generate(
        &self,
        text: &str,
        reference: Option<(&str, &[Frame])>,
        max_frames: usize,
        sampling: Sampling,
    ) -> Result<Vec<Frame>, String> {
        self.generate_impl(text, reference, max_frames, sampling, None)
    }

    /// Compare lossy kernels on the same history without token divergence hiding the first error.
    #[cfg(feature = "parity-trace")]
    pub fn replay(
        &self,
        text: &str,
        reference: Option<(&str, &[Frame])>,
        frames: &[Frame],
    ) -> Result<Vec<Frame>, String> {
        if frames.iter().flatten().any(|&id| id >= 4096) {
            return Err("invalid ArkTTS replay frame".into());
        }
        self.generate_impl(
            text,
            reference,
            frames.len(),
            Sampling {
                temperature: 0.0,
                ..Default::default()
            },
            Some(frames),
        )
    }

    fn generate_impl(
        &self,
        text: &str,
        reference: Option<(&str, &[Frame])>,
        max_frames: usize,
        sampling: Sampling,
        history: Option<&[Frame]>,
    ) -> Result<Vec<Frame>, String> {
        sampling.validate()?;
        let prompt = self.prompt(text, reference)?;
        if max_frames == 0 || max_frames > CONTEXT - prompt.len() {
            return Err("ArkTTS prompt plus max frames must fit the 2048-position context".into());
        }
        tokens(
            "arktts.prompt_ids",
            &prompt.iter().map(|c| c[0]).collect::<Vec<_>>(),
        )?;
        self.pool.install(|| {
            let _scope = self.compute.cpu_scope();
            let mut matrices = vec![&self.embeddings, &self.fast_output];
            matrices.extend(self.slow.matrices());
            matrices.extend(self.fast.matrices());
            let mut linear = linear(matrices, self.compute, self.threads)?;
            let mut rng = rand::rngs::StdRng::seed_from_u64(sampling.seed);
            let mut cache = Cache::default();
            let mut hidden = self
                .slow
                .forward(self.embed(&prompt)?, &mut cache, &mut linear)?;
            let mut frames = Vec::new();
            let mut previous: Option<[u32; 10]> = None;
            for step in 0..max_frames {
                let normalized = self.slow.normalize(&hidden[hidden.len() - WIDTH..]);
                let logits = self.embeddings.forward(&normalized, &mut linear)?;
                trace("arktts.slow.logits", None, Some(step), &[VOCAB], &logits)?;
                let candidates: Vec<(u32, f32)> = std::iter::once((EOS, logits[EOS as usize]))
                    .chain((0..4096).map(|id| (SEMANTIC + id, logits[(SEMANTIC + id) as usize])))
                    .collect();
                let mut semantic = draw(&candidates, sampling, &mut rng)?;
                if let Some(history) = history {
                    semantic = SEMANTIC + history[step][0];
                }
                if sampling.temperature > 0.0 {
                    let high = draw(
                        &candidates,
                        Sampling {
                            temperature: 1.0,
                            top_p: 0.9,
                            ..sampling
                        },
                        &mut rng,
                    )?;
                    if previous.is_some_and(|p| p.contains(&semantic)) && semantic != EOS {
                        semantic = high;
                    }
                }
                if semantic == EOS {
                    break;
                }
                let mut frame = [0; 10];
                frame[0] = semantic - SEMANTIC;
                let mut fast_cache = Cache::default();
                self.fast
                    .forward(normalized, &mut fast_cache, &mut linear)?;
                for codebook in 1..10 {
                    let fast_hidden = self.fast.forward(
                        self.fast_embeddings.embed(frame[codebook - 1]),
                        &mut fast_cache,
                        &mut linear,
                    )?;
                    let logits = self
                        .fast_output
                        .forward(&self.fast.normalize(&fast_hidden), &mut linear)?;
                    trace(
                        "arktts.fast.logits",
                        Some(codebook),
                        Some(step),
                        &[4096],
                        &logits,
                    )?;
                    frame[codebook] = draw(
                        &logits
                            .into_iter()
                            .enumerate()
                            .map(|(i, v)| (i as u32, v))
                            .collect::<Vec<_>>(),
                        sampling,
                        &mut rng,
                    )?;
                    if let Some(history) = history {
                        frame[codebook] = history[step][codebook];
                    }
                }
                tokens("arktts.frame", &frame)?;
                frames.push(frame);
                if let Some(previous) = &mut previous {
                    previous.rotate_left(1);
                    previous[9] = semantic;
                } else {
                    previous = Some([0; 10]);
                }
                if step + 1 < max_frames {
                    let mut column = [0; 11];
                    column[0] = semantic;
                    column[1..].copy_from_slice(&frame);
                    hidden = self
                        .slow
                        .forward(self.embed(&[column])?, &mut cache, &mut linear)?;
                }
            }
            Ok(frames)
        })
    }

    pub fn synthesize(
        &self,
        codec: &codec::Codec<'_>,
        text: &str,
        reference: Option<(&str, &[f32])>,
        max_frames: usize,
        sampling: Sampling,
    ) -> Result<Vec<f32>, String> {
        if self.bundle != codec.bundle {
            return Err("ArkTTS main and codec belong to different checkpoint bundles".into());
        }
        self.pool.install(|| {
            let codes = reference
                .map(|(_, audio)| codec.encode(audio))
                .transpose()?;
            let reference =
                reference.map(|(transcript, _)| (transcript, codes.as_deref().unwrap()));
            let frames = self.generate(text, reference, max_frames, sampling)?;
            if frames.is_empty() {
                return Err("ArkTTS produced no audio frames".into());
            }
            codec.decode(&frames)
        })
    }
}

fn clean_text(text: &str) -> String {
    text.split(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}
fn text_column(id: u32) -> [u32; 11] {
    let mut column = [0; 11];
    column[0] = id;
    column
}

fn draw(candidates: &[(u32, f32)], sampling: Sampling, rng: &mut impl Rng) -> Result<u32, String> {
    if candidates.is_empty() || candidates.iter().any(|(_, v)| !v.is_finite()) {
        return Err("ArkTTS logits must be finite and nonempty".into());
    }
    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if sampling.temperature == 0.0 {
        return Ok(sorted[0].0);
    }
    let maximum = sorted[0].1;
    let denominator: f32 = sorted.iter().map(|(_, v)| (v - maximum).exp()).sum();
    let mut cumulative = 0.0;
    let mut best = (sorted[0].0, f32::NEG_INFINITY);
    for (position, &(id, logit)) in sorted.iter().enumerate() {
        cumulative += (logit - maximum).exp() / denominator;
        // Official legacy filtering excludes the candidate that first crosses top_p.
        if position > 0 && (position >= sampling.top_k || cumulative > sampling.top_p) {
            break;
        }
        let u = rng.gen::<f32>().max(f32::MIN_POSITIVE);
        let score = (logit - maximum) / sampling.temperature.max(1e-5) - (-u.ln()).ln();
        if score > best.1 {
            best = (id, score);
        }
    }
    Ok(best.0)
}

fn trace(
    name: &str,
    layer: Option<usize>,
    step: Option<usize>,
    shape: &[usize],
    values: &[f32],
) -> Result<(), String> {
    #[cfg(feature = "parity-trace")]
    if std::env::var_os("RMI_PARITY_TRACE").is_some() {
        crate::parity_trace::checkpoint_at(name, layer, step, shape, values)
            .map_err(|e| e.to_string())?;
    }
    let _ = (name, layer, step, shape, values);
    Ok(())
}
fn tokens(name: &str, ids: &[u32]) -> Result<(), String> {
    #[cfg(feature = "parity-trace")]
    if std::env::var_os("RMI_PARITY_TRACE").is_some() {
        crate::parity_trace::token_ids(name, ids).map_err(|e| e.to_string())?;
    }
    let _ = (name, ids);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::TensorInfo;
    use std::collections::HashMap;

    struct Source {
        metadata: HashMap<String, MetaValue>,
        info: TensorInfo,
        bytes: Vec<u8>,
    }
    impl Source {
        fn new(arch: &str) -> Self {
            Self {
                metadata: [
                    (
                        "general.architecture".into(),
                        MetaValue::String(arch.into()),
                    ),
                    (
                        format!("{arch}.config_json"),
                        MetaValue::String(include_str!("contract.json").into()),
                    ),
                    ("arktts.bundle_id".into(), MetaValue::String("a".repeat(64))),
                ]
                .into_iter()
                .collect(),
                info: TensorInfo {
                    name: "projection.weight".into(),
                    dims: vec![2, 2],
                    ggml_type: GGMLType::F32,
                    offset: 0,
                },
                bytes: [1.0f32, 2.0, 3.0, 4.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            }
        }
    }
    impl TensorSource for Source {
        fn metadata(&self, name: &str) -> Option<&MetaValue> {
            self.metadata.get(name)
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            (name == self.info.name).then_some(&self.info)
        }
        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            (name == self.info.name).then_some(&self.bytes)
        }
    }

    #[test]
    fn contract_rejects_nearby_architectures_and_mixed_checkpoint_bundles() {
        let source = Source::new("qwen2");
        assert!(Model::from_source(&source, 1).is_err());
        let main = Source::new("arktts");
        let mut codec = Source::new("arktts_codec");
        validate_pair(&main, &codec).unwrap();
        codec
            .metadata
            .insert("arktts.bundle_id".into(), MetaValue::String("b".repeat(64)));
        assert!(validate_pair(&main, &codec)
            .unwrap_err()
            .contains("different checkpoint"));
        codec.metadata.remove("arktts.bundle_id");
        assert!(validate_pair(&main, &codec).is_err());
        let mut config: Value = serde_json::from_str(include_str!("contract.json")).unwrap();
        config["attention_qk_norm"] = Value::Bool(true);
        codec.metadata.insert(
            "arktts_codec.config_json".into(),
            MetaValue::String(config.to_string()),
        );
        assert!(super::config(&codec, "arktts_codec")
            .unwrap_err()
            .contains("attention_qk_norm"));
    }

    #[test]
    fn projections_validate_dtype_shape_and_storage_before_running_native_kernels() {
        let mut source = Source::new("arktts");
        {
            let matrix = Matrix::load(&source, "projection", 2, 2, false).unwrap();
            let mut executor = linear(vec![&matrix], ComputePolicy::Cpu, 2).unwrap();
            assert_eq!(
                matrix.forward(&[1.0, 2.0], &mut executor).unwrap(),
                vec![5.0, 11.0]
            );
            assert_eq!(
                matrix
                    .forward(&[1.0, 2.0].repeat(33), &mut executor)
                    .unwrap(),
                [5.0, 11.0].repeat(33)
            );
            assert!(matrix.forward(&[1.0], &mut executor).is_err());
            assert!(matrix.forward(&[f32::NAN, 2.0], &mut executor).is_err());
        }
        source.info.dims = vec![4, 1];
        assert!(Matrix::load(&source, "projection", 2, 2, false).is_err());
        source.info.dims = vec![2, 2];
        source.bytes.pop();
        assert!(Matrix::load(&source, "projection", 2, 2, false).is_err());
        source.bytes.push(0);
        source.info.ggml_type = GGMLType::I32;
        assert!(Matrix::load(&source, "projection", 2, 2, false).is_err());
    }
    #[test]
    fn projections_accept_shared_float_and_block_formats() {
        for kind in [
            GGMLType::F32,
            GGMLType::F16,
            GGMLType::BF16,
            GGMLType::Q8_0,
            GGMLType::Q4_0,
            GGMLType::Q4_1,
            GGMLType::Q4K,
            GGMLType::Q5K,
            GGMLType::Q6K,
        ] {
            let mut source = Source::new("arktts");
            let (block, bytes) = kind.type_traits();
            source.info.dims = vec![256, 2];
            source.info.ggml_type = kind;
            source.bytes = vec![0; 256 / block * 2 * bytes];
            let matrix = Matrix::load(&source, "projection", 256, 2, false)
                .unwrap_or_else(|e| panic!("{kind:?}: {e}"));
            assert_eq!(matrix.embed(1), vec![0.0; 256]);
            drop(matrix);
            source.bytes.pop();
            assert!(Matrix::load(&source, "projection", 256, 2, false).is_err());
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires a real Vulkan device"]
    fn projections_submit_to_vulkan_and_match_cpu() {
        let source = Source::new("arktts");
        let matrix = Matrix::load(&source, "projection", 2, 2, false).unwrap();
        let mut gpu = linear(vec![&matrix], ComputePolicy::Vulkan, 2).unwrap();
        let mut cpu = linear(vec![&matrix], ComputePolicy::Cpu, 2).unwrap();
        let context = gpu.vulkan_context().unwrap();
        let before = context.submission_count();
        let input = [0.1, -0.2].repeat(33);
        let expected = matrix.forward(&input, &mut cpu).unwrap();
        let actual = matrix.forward(&input, &mut gpu).unwrap();
        assert!(context.submission_count() > before);
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    #[ignore = "requires real GGUFs and independent same-input projection fixtures"]
    fn same_input_projections_match_independent_oracle() {
        use std::path::Path;
        let path = std::env::var("RMI_ARKTTS_PROJECTIONS").unwrap();
        let cases: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let compute = match std::env::var("RMI_ARKTTS_COMPUTE").as_deref() {
            Ok("vulkan") => ComputePolicy::Vulkan,
            Ok("auto") => ComputePolicy::Auto,
            _ => ComputePolicy::Cpu,
        };
        for case in cases.as_array().unwrap() {
            let source = crate::open_model_source(
                Path::new(case["gguf"].as_str().unwrap()),
                crate::ComponentRole::Llm,
            )
            .unwrap();
            let name = case["tensor"]
                .as_str()
                .unwrap()
                .strip_suffix(".weight")
                .unwrap();
            let input: Vec<f32> = serde_json::from_value(case["input"].clone()).unwrap();
            let expected: Vec<f32> = serde_json::from_value(case["output"].clone()).unwrap();
            let n_in = case["n_in"].as_u64().unwrap() as usize;
            let n_out = expected.len() / (input.len() / n_in);
            let matrix = Matrix::load(source.as_ref(), name, n_in, n_out, false).unwrap();
            let embedding: Vec<f32> = serde_json::from_value(case["embedding"].clone()).unwrap();
            assert!(matrix
                .embed(1)
                .iter()
                .zip(&embedding)
                .all(|(a, b)| (a - b).abs() < 1e-6));
            let mut linear = linear(vec![&matrix], compute, 4).unwrap();
            let actual = matrix.forward(&input, &mut linear).unwrap();
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>();
            let norm = expected.iter().map(|b| f64::from(*b).powi(2)).sum::<f64>();
            let relative = (error / norm.max(1e-20)).sqrt();
            println!("{name} relative_rmse={relative:e}");
            assert!(
                relative < case["relative_limit"].as_f64().unwrap(),
                "{name}: {relative}"
            );
        }
    }

    #[test]
    fn sampling_filters_before_temperature_and_keeps_the_best_candidate() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let p = Sampling {
            temperature: 1.0,
            top_p: 0.7,
            ..Default::default()
        };
        for _ in 0..20 {
            assert_eq!(draw(&[(7, 2.0), (8, 1.0)], p, &mut rng).unwrap(), 7);
        }
        assert_eq!(
            draw(
                &[(9, 1.0), (3, 1.0)],
                Sampling {
                    temperature: 0.0,
                    ..p
                },
                &mut rng
            )
            .unwrap(),
            3
        );
        assert!(draw(&[(0, f32::NAN)], p, &mut rng).is_err());
        assert!(Sampling { top_k: 0, ..p }.validate().is_err());
        assert_eq!(
            validate_compute(ComputePolicy::Vulkan).is_ok(),
            cfg!(feature = "vulkan")
        );
        let _scalar = crate::ops::ScalarDispatchScope::new(true);
        assert!(validate_compute(ComputePolicy::Vulkan).is_err());
    }
    #[test]
    fn prompt_whitespace_matches_python_clean_text() {
        assert_eq!(
            clean_text(" \n你好\t世界\u{3000}hello\u{a0}\u{1c} world\u{1f}  "),
            "你好 世界 hello world"
        );
    }
}
