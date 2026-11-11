//! Portable Qwen3-ASR encoder and the release's stack-2 Ultravox projector.

use crate::core::tensor::{load_f32_tensor, require_tensor_type, GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::audio::frontend::compute_log_mel_trimmed;
use crate::ops::audio::transformer::{
    apply_gelu_erf, full_attention_into, layer_norm_rows, reserved_f32, LayerNormWeights,
};
use crate::ops::kernel::{PreparedRows, QuantizedTensor, Weight};

const HIDDEN: usize = 896;
const MEL: usize = 128;
const OUTPUT: usize = 5120;
const CHUNK: usize = 3000;

pub fn token_count(samples: usize) -> Result<usize, String> {
    if samples == 0 {
        return Err("audio samples must be nonempty".into());
    }
    let frames = samples
        .max(320)
        .checked_add(2559)
        .ok_or("audio length overflow")?
        / 2560
        * 16;
    (frames / CHUNK)
        .checked_mul(CHUNK.div_ceil(16))
        .and_then(|n| n.checked_add((frames % CHUNK).div_ceil(16)))
        .ok_or_else(|| "audio token count overflow".into())
}

pub fn prompt_text(prompt: &str, audios: usize) -> String {
    if audios == 0 || prompt.contains("<|audio|>") {
        prompt.to_owned()
    } else {
        format!("{}\n{prompt}", "<|audio|>".repeat(audios))
    }
}

pub(crate) fn expand_prompt(
    tokenizer: &crate::core::tokenizer::BPETokenizer,
    ids: &[u32],
    lengths: &[usize],
    max_new_tokens: usize,
    capacity: usize,
) -> Result<Vec<u32>, String> {
    let placeholder = tokenizer
        .token_id("<|audio|>")
        .ok_or("missing Sophon <|audio|> token")?;
    if lengths.is_empty() || lengths.contains(&0) {
        return Err("at least one nonempty audio is required".into());
    }
    let extra = lengths
        .iter()
        .try_fold(0usize, |n, length| n.checked_add(length - 1))
        .ok_or("audio prompt length overflow")?;
    if ids
        .len()
        .checked_add(extra)
        .and_then(|n| n.checked_add(max_new_tokens))
        .is_none_or(|n| n > capacity)
    {
        return Err(format!(
            "audio prompt plus output budget exceeds context {capacity}"
        ));
    }
    crate::prompt::jinja::expand_vision_placeholders(ids, placeholder, lengths)
        .map_err(|e| format!("Sophon audio placeholder mismatch: {e}"))
}

pub(crate) fn inject_embeddings(
    model: &crate::models::qwen3::Qwen3Model,
    ids: &[u32],
    audio: &[AudioEmbeddings],
) -> Result<Vec<f32>, String> {
    let placeholder = model
        .tokenizer()
        .token_id("<|audio|>")
        .ok_or("missing Sophon audio token")?;
    let count = ids.iter().filter(|&&id| id == placeholder).count();
    let expected = audio.iter().try_fold(0usize, |n, item| {
        if item.tokens == 0
            || item.tokens.checked_mul(OUTPUT) != Some(item.values.len())
            || item.values.iter().any(|v| !v.is_finite())
        {
            return Err("invalid Sophon audio embeddings".to_string());
        }
        n.checked_add(item.tokens)
            .ok_or_else(|| "audio length overflow".into())
    })?;
    if count != expected || model.config().n_embd != OUTPUT {
        return Err("Sophon audio embedding/placeholder shape mismatch".into());
    }
    let mut embeddings = model.embed_tokens(ids)?;
    let mut rows = audio
        .iter()
        .flat_map(|item| item.values.chunks_exact(OUTPUT));
    for (row, &id) in embeddings.chunks_exact_mut(OUTPUT).zip(ids) {
        if id == placeholder {
            row.copy_from_slice(rows.next().ok_or("missing audio embedding row")?);
        }
    }
    Ok(embeddings)
}

fn cnn_rows(frames: usize) -> usize {
    frames / 100 * 13 + (frames % 100).div_ceil(8)
}

fn stack_frames(hidden: &[f32], width: usize) -> Result<Vec<f32>, String> {
    if width == 0
        || hidden.is_empty()
        || hidden.len() % width != 0
        || hidden.iter().any(|v| !v.is_finite())
    {
        return Err("invalid audio frame stack".into());
    }
    let mut stacked = hidden.to_vec();
    if hidden.len() / width % 2 != 0 {
        stacked.resize(hidden.len() + width, 0.0);
    }
    Ok(stacked)
}

fn weight<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    dims: &[u64],
) -> Result<Weight<'a>, String> {
    let kind = require_tensor_type(
        source,
        name,
        dims,
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
    let input = dims[..dims.len() - 1].iter().product::<u64>() as usize;
    let output = dims[dims.len() - 1] as usize;
    let data = source
        .tensor_slice(name)
        .ok_or_else(|| format!("missing {name}"))?;
    if kind == GGMLType::F32 {
        Weight::from_f32_bytes(data, input, output)
            .ok_or_else(|| format!("invalid F32 audio matrix {name}"))
    } else {
        Ok(Weight::from_quantized(QuantizedTensor::from_bytes(
            data, kind, input, output,
        )))
    }
}

struct Linear<'a> {
    weight: Weight<'a>,
    bias: Option<Vec<f32>>,
    #[cfg(feature = "parity-trace")]
    trace_name: Option<String>,
}

impl<'a> Linear<'a> {
    fn load(
        source: &'a dyn TensorSource,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Self, String> {
        Ok(Self {
            #[cfg(feature = "parity-trace")]
            trace_name: Some(name.to_owned()),
            weight: weight(
                source,
                &format!("{name}.weight"),
                &[input as u64, output as u64],
            )?,
            bias: if bias {
                Some(load_f32_tensor(
                    source,
                    &format!("{name}.bias"),
                    &[output as u64],
                )?)
            } else {
                None
            },
        })
    }
    fn forward(&self, input: &[f32], pool: &ComputePool) -> Result<Vec<f32>, String> {
        let weight = &self.weight;
        if input.is_empty()
            || input.len() % weight.n_in != 0
            || input.iter().any(|v| !v.is_finite())
        {
            return Err("invalid audio projection input".into());
        }
        let rows = input.len() / weight.n_in;
        #[cfg(feature = "parity-trace")]
        self.trace("input", &[rows, weight.n_in], input);
        let mut prepared = PreparedRows::new(rows, weight.n_in);
        prepared.prepare(
            input,
            rows,
            weight.n_in,
            weight.needs_q8_0_activation(),
            weight.uses_q8_k(),
        )?;
        let mut output = reserved_f32("audio projection", rows * weight.n_out)?;
        prepared.matmul(weight, input, &mut output, pool)?;
        if let Some(bias) = &self.bias {
            for row in output.chunks_exact_mut(weight.n_out) {
                for (value, bias) in row.iter_mut().zip(bias) {
                    *value += bias;
                }
            }
        }
        if output.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite audio projection".into());
        }
        #[cfg(feature = "parity-trace")]
        self.trace("output", &[rows, weight.n_out], &output);
        Ok(output)
    }

    #[cfg(feature = "parity-trace")]
    fn trace(&self, stage: &str, shape: &[usize], values: &[f32]) {
        if let Some(name) = &self.trace_name {
            if std::env::var("RMI_SOPHON_AUDIO_LINEAR_TRACE")
                .is_ok_and(|filter| filter == "all" || name.contains(&filter))
            {
                trace(&format!("linear.{name}.{stage}"), shape, values);
            }
        }
    }
}

struct Conv<'a> {
    linear: Linear<'a>,
    channels: usize,
}

impl<'a> Conv<'a> {
    fn load(source: &'a dyn TensorSource, index: usize, channels: usize) -> Result<Self, String> {
        let prefix = format!("audio_tower.conv2d{index}");
        require_tensor_type(
            source,
            &format!("{prefix}.weight"),
            &[3, 3, channels as u64, 480],
            &[GGMLType::F32],
        )?;
        Ok(Self {
            channels,
            linear: Linear {
                #[cfg(feature = "parity-trace")]
                trace_name: None,
                weight: weight(
                    source,
                    &format!("{prefix}.weight"),
                    &[3, 3, channels as u64, 480],
                )?,
                bias: Some(load_f32_tensor(source, &format!("{prefix}.bias"), &[480])?),
            },
        })
    }
    fn forward(
        &self,
        input: &[f32],
        height: usize,
        width: usize,
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        if input.len() != self.channels * height * width {
            return Err("invalid audio convolution shape".into());
        }
        let (oh, ow) = (height.div_ceil(2), width.div_ceil(2));
        let mut output = reserved_f32("audio convolution", 480 * oh * ow)?;
        // Bounded im2col rows use the shared SIMD/thread-pool matrix kernels.
        for start in (0..oh * ow).step_by(64) {
            let count = (oh * ow - start).min(64);
            let mut patches = reserved_f32("audio convolution patches", count * self.channels * 9)?;
            for row in 0..count {
                let (y, x) = ((start + row) / ow, (start + row) % ow);
                for channel in 0..self.channels {
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let (iy, ix) =
                                ((2 * y + ky).checked_sub(1), (2 * x + kx).checked_sub(1));
                            if let (Some(iy), Some(ix)) = (iy, ix) {
                                if iy < height && ix < width {
                                    patches[row * self.channels * 9 + channel * 9 + ky * 3 + kx] =
                                        input[(channel * height + iy) * width + ix];
                                }
                            }
                        }
                    }
                }
            }
            let mut values = self.linear.forward(&patches, pool)?;
            apply_gelu_erf(&mut values)?;
            for row in 0..count {
                for channel in 0..480 {
                    output[channel * oh * ow + start + row] = values[row * 480 + channel];
                }
            }
        }
        Ok(output)
    }
}

struct Block<'a> {
    ln1: LayerNormWeights,
    ln2: LayerNormWeights,
    q: Linear<'a>,
    k: Linear<'a>,
    v: Linear<'a>,
    out: Linear<'a>,
    up: Linear<'a>,
    down: Linear<'a>,
}

pub struct AudioEncoder<'a> {
    conv: [Conv<'a>; 3],
    conv_out: Linear<'a>,
    blocks: Vec<Block<'a>>,
    post: LayerNormWeights,
    pre: Vec<f32>,
    mid: Vec<f32>,
    first: Linear<'a>,
    second: Linear<'a>,
}

pub struct AudioEmbeddings {
    pub values: Vec<f32>,
    pub tokens: usize,
}

pub(crate) struct PreparedAudio {
    pub token_ids: Vec<u32>,
    pub positions: Vec<[usize; 4]>,
    pub embeddings: Vec<f32>,
}

pub(crate) struct AudioDecoder<'a> {
    encoder: AudioEncoder<'a>,
    decoder: &'a crate::models::qwen3::Qwen3Model,
    capacity: usize,
}

impl<'a> AudioDecoder<'a> {
    pub(crate) fn new(
        thinker: &dyn TensorSource,
        decoder: &'a crate::models::qwen3::Qwen3Model,
        component: Option<&'a dyn TensorSource>,
        capacity: usize,
        compute: crate::compute::ComputePolicy,
    ) -> Result<Self, String> {
        if !super::is_thinker(thinker) {
            return Err("this text runtime does not support audio input".into());
        }
        if compute == crate::compute::ComputePolicy::Vulkan {
            return Err("Sophon strict Vulkan audio awaits shared block-attention and frontend device kernels".into());
        }
        let encoder = AudioEncoder::from_source(
            component.ok_or("audio input requires --mmproj with the Sophon audio component")?,
        )?;
        Ok(Self {
            encoder,
            decoder,
            capacity,
        })
    }

    pub(crate) fn validate(
        &self,
        ids: &[u32],
        audio: &[Vec<f32>],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, String> {
        let lengths = audio
            .iter()
            .map(|samples| {
                if samples.iter().any(|v| !v.is_finite()) {
                    return Err("audio samples must be finite".to_string());
                }
                token_count(samples.len())
            })
            .collect::<Result<Vec<_>, _>>()?;
        expand_prompt(
            self.decoder.tokenizer(),
            ids,
            &lengths,
            max_new_tokens,
            self.capacity,
        )
    }

    pub(crate) fn prepare(
        &self,
        ids: &[u32],
        audio: &[Vec<f32>],
        max_new_tokens: usize,
    ) -> Result<PreparedAudio, String> {
        let token_ids = self.validate(ids, audio, max_new_tokens)?;
        let encoded = audio
            .iter()
            .map(|samples| self.encoder.encode(samples, &self.decoder.pool()))
            .collect::<Result<Vec<_>, _>>()?;
        let embeddings = inject_embeddings(self.decoder, &token_ids, &encoded)?;
        let positions = crate::models::qwen3::trunk::qwen_text_positions(token_ids.len());
        Ok(PreparedAudio {
            token_ids,
            positions,
            embeddings,
        })
    }

    pub(crate) fn logits(
        &self,
        ids: &[u32],
        audio: &[Vec<f32>],
        kv: crate::core::scratchpad::KvFormat,
        batch: usize,
        compute: crate::compute::ComputePolicy,
    ) -> Result<Vec<f32>, String> {
        let input = self.prepare(ids, audio, 1)?;
        let mut session = crate::models::qwen3::Qwen3Session::new_with_compute(
            self.decoder,
            self.capacity,
            kv,
            crate::KvLifecycle::Ephemeral,
            compute,
        )?;
        session
            .forward_logits(
                crate::models::qwen3::Qwen3Input {
                    token_ids: &input.token_ids,
                    positions: &input.positions,
                    embeddings: Some(&input.embeddings),
                    deepstack_embeddings: None,
                },
                batch,
            )
            .map(|(logits, _)| logits)
    }
}

impl<'a> AudioEncoder<'a> {
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        super::require_string(source, "general.architecture", "sophon_realtime_audio")?;
        super::require_string(source, "sophon_realtime.component", "audio")?;
        super::require_string(
            source,
            "sophon_realtime.source_revision",
            super::SOURCE_REVISION,
        )?;
        let mut blocks = Vec::with_capacity(18);
        for i in 0..18 {
            let prefix = format!("audio_tower.layers.{i}");
            let linear = |name: &str, input, output| {
                Linear::load(source, &format!("{prefix}.{name}"), input, output, true)
            };
            blocks.push(Block {
                ln1: LayerNormWeights::load(
                    source,
                    &format!("{prefix}.self_attn_layer_norm"),
                    &[896],
                )?,
                ln2: LayerNormWeights::load(source, &format!("{prefix}.final_layer_norm"), &[896])?,
                q: linear("self_attn.q_proj", 896, 896)?,
                k: linear("self_attn.k_proj", 896, 896)?,
                v: linear("self_attn.v_proj", 896, 896)?,
                out: linear("self_attn.out_proj", 896, 896)?,
                up: linear("fc1", 896, 3584)?,
                down: linear("fc2", 3584, 896)?,
            });
        }
        Ok(Self {
            conv: [
                Conv::load(source, 1, 1)?,
                Conv::load(source, 2, 480)?,
                Conv::load(source, 3, 480)?,
            ],
            conv_out: Linear::load(source, "audio_tower.conv_out", 7680, 896, false)?,
            blocks,
            post: LayerNormWeights::load(source, "audio_tower.ln_post", &[896])?,
            pre: load_f32_tensor(source, "multi_modal_projector.ln_pre.weight", &[1792])?,
            mid: load_f32_tensor(source, "multi_modal_projector.ln_mid.weight", &[2048])?,
            first: Linear::load(source, "multi_modal_projector.linear_1", 1792, 4096, false)?,
            second: Linear::load(source, "multi_modal_projector.linear_2", 2048, 5120, false)?,
        })
    }

    pub fn encode(&self, samples: &[f32], pool: &ComputePool) -> Result<AudioEmbeddings, String> {
        let expected_tokens = token_count(samples.len())?;
        if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
            return Err("audio samples must be nonempty and finite".into());
        }
        if crate::compute::ComputePolicy::legacy() == crate::compute::ComputePolicy::Vulkan {
            return Err("Sophon strict Vulkan audio awaits shared block-attention and frontend device kernels".into());
        }
        let length = samples
            .len()
            .max(320)
            .checked_add(2559)
            .ok_or("audio length overflow")?
            / 2560
            * 2560;
        let mut aligned = samples.to_vec();
        aligned.resize(length, 0.0);
        let mel =
            compute_log_mel_trimmed(&aligned).map_err(|e| format!("Sophon frontend: {e:?}"))?;
        let mut values = Vec::new();
        for start in (0..mel.frames).step_by(CHUNK) {
            let frames = (mel.frames - start).min(CHUNK);
            let mut features = reserved_f32("audio mel chunk", MEL * frames)?;
            for bin in 0..MEL {
                features[bin * frames..(bin + 1) * frames].copy_from_slice(
                    &mel.normalized[bin * mel.frames + start..bin * mel.frames + start + frames],
                );
            }
            trace("mel", &[MEL, frames], &features);
            let hidden = self.encode_mel(&features, frames, pool)?;
            trace("encoder", &[hidden.len() / HIDDEN, HIDDEN], &hidden);
            let stacked = stack_frames(&hidden, HIDDEN)?;
            let normed = rms_rows(&stacked, &self.pre)?;
            let first = self.first.forward(&normed, pool)?;
            let mut activated = reserved_f32("audio SwiGLU", first.len() / 2)?;
            for (input, output) in first
                .chunks_exact(4096)
                .zip(activated.chunks_exact_mut(2048))
            {
                for i in 0..2048 {
                    output[i] = input[i] * crate::ops::silu(input[i + 2048]);
                }
            }
            let projected = self
                .second
                .forward(&rms_rows(&activated, &self.mid)?, pool)?;
            // The released processor uses ceil(mel/16), not the encoder's full 6.5 tokens/s.
            let keep = frames.div_ceil(16) * OUTPUT;
            if keep > projected.len() {
                return Err("Sophon audio token length exceeds projector rows".into());
            }
            values.extend_from_slice(&projected[..keep]);
        }
        trace("projected", &[values.len() / OUTPUT, OUTPUT], &values);
        if values.len() / OUTPUT != expected_tokens {
            return Err("Sophon frontend/projector token count mismatch".into());
        }
        Ok(AudioEmbeddings {
            tokens: values.len() / OUTPUT,
            values,
        })
    }

    fn encode_mel(
        &self,
        features: &[f32],
        frames: usize,
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        let mut hidden = Vec::with_capacity(cnn_rows(frames) * HIDDEN);
        for start in (0..frames).step_by(100) {
            let valid = (frames - start).min(100);
            let mut input = reserved_f32("audio CNN chunk", MEL * 100)?;
            for bin in 0..MEL {
                input[bin * 100..bin * 100 + valid]
                    .copy_from_slice(&features[bin * frames + start..bin * frames + start + valid]);
            }
            let first = self.conv[0].forward(&input, 128, 100, pool)?;
            let second = self.conv[1].forward(&first, 64, 50, pool)?;
            let third = self.conv[2].forward(&second, 32, 25, pool)?;
            let mut rows = reserved_f32("audio CNN transpose", 13 * 7680)?;
            for t in 0..13 {
                for channel in 0..480 {
                    for frequency in 0..16 {
                        rows[t * 7680 + channel * 16 + frequency] =
                            third[(channel * 16 + frequency) * 13 + t];
                    }
                }
            }
            let mut projected = self.conv_out.forward(&rows, pool)?;
            for t in 0..13 {
                for d in 0..448 {
                    let increment = 10000f64.ln() / 447.0;
                    let inverse = (-(increment as f32) * (d as f32)).exp();
                    let angle = t as f32 * inverse;
                    projected[t * HIDDEN + d] += angle.sin();
                    projected[t * HIDDEN + 448 + d] += angle.cos();
                }
            }
            hidden.extend_from_slice(&projected[..valid.div_ceil(8) * HIDDEN]);
        }
        let rows = hidden.len() / HIDDEN;
        let mut normed = Vec::new();
        for (index, block) in self.blocks.iter().enumerate() {
            layer_norm_rows(&hidden, rows, &block.ln1, 1e-5, &mut normed)?;
            let q = block.q.forward(&normed, pool)?;
            let k = block.k.forward(&normed, pool)?;
            let v = block.v.forward(&normed, pool)?;
            let mut attention = reserved_f32("audio attention", hidden.len())?;
            for start in (0..rows).step_by(104) {
                let end = (start + 104).min(rows);
                let range = start * HIDDEN..end * HIDDEN;
                let mut result = Vec::new();
                let mut scores = Vec::new();
                full_attention_into(
                    &q[range.clone()],
                    &k[range.clone()],
                    &v[range.clone()],
                    end - start,
                    14,
                    64,
                    &mut scores,
                    &mut result,
                )?;
                attention[range].copy_from_slice(&result);
            }
            add(&mut hidden, &block.out.forward(&attention, pool)?);
            layer_norm_rows(&hidden, rows, &block.ln2, 1e-5, &mut normed)?;
            let mut up = block.up.forward(&normed, pool)?;
            apply_gelu_erf(&mut up)?;
            add(&mut hidden, &block.down.forward(&up, pool)?);
            trace(&format!("block.{index}"), &[rows, HIDDEN], &hidden);
        }
        layer_norm_rows(&hidden, rows, &self.post, 1e-5, &mut normed)?;
        Ok(normed)
    }
}

fn add(hidden: &mut [f32], residual: &[f32]) {
    for (value, residual) in hidden.iter_mut().zip(residual) {
        *value += residual;
    }
}

fn rms_rows(input: &[f32], weight: &[f32]) -> Result<Vec<f32>, String> {
    if weight.is_empty() || input.len() % weight.len() != 0 || weight.iter().any(|v| !v.is_finite())
    {
        return Err("invalid audio RMSNorm".into());
    }
    let mut output = reserved_f32("audio RMSNorm", input.len())?;
    for (input, output) in input
        .chunks_exact(weight.len())
        .zip(output.chunks_exact_mut(weight.len()))
    {
        crate::ops::rms_norm(input, weight, output, 1e-6);
    }
    Ok(output)
}

fn trace(name: &str, shape: &[usize], values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(
        &format!("sophon.audio.{name}"),
        None,
        shape,
        values,
    ));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, shape, values);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_placeholders_preserve_order_and_check_expanded_context() {
        use crate::core::tensor::{MetaValue, MetaValueType};
        let tokenizer = crate::core::tokenizer::BPETokenizer::from_gguf_metadata(|key| match key {
            "tokenizer.ggml.model" => Some(MetaValue::String("gpt2".into())),
            "tokenizer.ggml.pre" => Some(MetaValue::String("qwen2".into())),
            "tokenizer.ggml.tokens" => Some(MetaValue::Array(
                MetaValueType::String,
                ["a", "<|audio|>", "b"]
                    .into_iter()
                    .map(|s| MetaValue::String(s.into()))
                    .collect(),
            )),
            "tokenizer.ggml.merges" => Some(MetaValue::Array(MetaValueType::String, vec![])),
            "tokenizer.ggml.token_type" => Some(MetaValue::Array(
                MetaValueType::Int32,
                [1, 3, 1].map(MetaValue::Int32).to_vec(),
            )),
            _ => None,
        })
        .unwrap();
        assert_eq!(
            expand_prompt(&tokenizer, &[0, 1, 2, 1], &[2, 1], 2, 7).unwrap(),
            [0, 1, 1, 2, 1]
        );
        for (ids, lengths, max_tokens, capacity) in [
            (vec![0, 1, 2, 1], vec![2, 1], 2, 6),
            (vec![0, 1, 2, 1], vec![2], 2, 100),
            (vec![0, 1, 2], vec![0], 2, 100),
            (vec![0, 1, 2], vec![], 2, 100),
            (vec![1, 1], vec![usize::MAX, usize::MAX], 2, usize::MAX),
        ] {
            assert!(expand_prompt(&tokenizer, &ids, &lengths, max_tokens, capacity).is_err());
        }
    }

    #[test]
    fn token_budget_accounts_for_waveform_alignment_and_processor_chunks() {
        for (samples, tokens) in [
            (1, 1),
            (2560, 1),
            (2561, 2),
            (478720, 187),
            (480000, 189),
            (960000, 376),
        ] {
            assert_eq!(token_count(samples).unwrap(), tokens, "samples={samples}");
        }
        assert!(token_count(0).is_err());
        assert!(token_count(usize::MAX).is_err());
        assert_eq!(prompt_text("describe", 2), "<|audio|><|audio|>\ndescribe");
        assert_eq!(prompt_text("text only", 0), "text only");
        assert_eq!(prompt_text("Here: <|audio|>.", 1), "Here: <|audio|>.");
    }

    #[test]
    #[ignore = "requires the remote pinned audio GGUF and mono 16kHz PCM16 WAV"]
    fn real_audio_checkpoints() {
        let model = std::env::var("RMI_SOPHON_AUDIO").expect("set RMI_SOPHON_AUDIO");
        let wav = std::env::var("RMI_SOPHON_WAV").expect("set RMI_SOPHON_WAV");
        let source = crate::core::loader::GGUFLoader::from_file(&model).unwrap();
        let audio = crate::ops::audio::frontend::decode_pcm16_wav_any(&std::fs::read(wav).unwrap())
            .unwrap();
        assert_eq!((audio.sample_rate, audio.channels), (16_000, 1));
        let _scope = crate::compute::ComputePolicy::Cpu.enter_legacy_scope();
        let encoder = AudioEncoder::from_source(&source).unwrap();
        let started = std::time::Instant::now();
        let result = encoder
            .encode(&audio.samples, &ComputePool::new(4))
            .unwrap();
        assert!(result.tokens > 0);
        assert_eq!(result.values.len(), result.tokens * OUTPUT);
        assert!(result.values.iter().all(|v| v.is_finite()));
        eprintln!(
            "audio tokens={} seconds={:.6}",
            result.tokens,
            started.elapsed().as_secs_f64()
        );
    }

    #[test]
    fn cnn_lengths_preserve_hundred_frame_boundaries() {
        for (frames, rows) in [
            (0, 0),
            (1, 1),
            (8, 1),
            (9, 2),
            (99, 13),
            (100, 13),
            (101, 14),
            (800, 104),
            (3000, 390),
        ] {
            assert_eq!(cnn_rows(frames), rows, "frames={frames}");
        }
    }

    #[test]
    fn stack_keeps_adjacent_rows_and_zero_pads_only_the_last_row() {
        assert_eq!(
            stack_frames(&[1., 2., 3., 4., 5., 6.], 2).unwrap(),
            [1., 2., 3., 4., 5., 6., 0., 0.]
        );
        assert!(stack_frames(&[1.], 2).is_err());
        assert!(stack_frames(&[], 0).is_err());
    }
}
