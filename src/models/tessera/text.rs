//! The frozen OpenAI CLIP ViT-B/32 text tower used to condition Tessera.
//! Every one of its 24 rows (including EOS padding) conditions the pixel stream.

#[cfg(feature = "vulkan")]
mod device;

use std::sync::Arc;

use tokenizers::{Tokenizer, TruncationParams};

use crate::{
    compute::{linear::LinearExecutor, ComputePolicy},
    core::{
        tensor::{load_weight_any, GGMLType, TensorSource},
        thread_pool::ComputePool,
    },
    ops::{
        kernel::Weight,
        nn::{attention_on_cpu, layer_norm_rows, load_float_values, AttentionSpec, Linear},
        vec_add_into,
    },
};

const CONTEXT: usize = 24;
const WIDTH: usize = 512;
const HEADS: usize = 8;
const FF: usize = 2048;
const VOCAB: usize = 49408;
const MAX_POS: usize = 77;
const BOS: u32 = 49406;
const EOS: u32 = 49407;
const FORMATS: &[GGMLType] = &[
    GGMLType::F32,
    GGMLType::F16,
    GGMLType::BF16,
    GGMLType::Q8_0,
    GGMLType::Q4_0,
    GGMLType::Q4_1,
    GGMLType::Q4K,
    GGMLType::Q5K,
    GGMLType::Q6K,
];

fn matrix<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    cols: usize,
    rows: usize,
) -> Result<Weight<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("missing CLIP tensor {name}"))?;
    if info.dims != [cols as u64, rows as u64] {
        return Err(format!(
            "invalid CLIP tensor {name}: expected [{cols}, {rows}], got {:?}",
            info.dims
        ));
    }
    load_weight_any(source, name, cols, rows, Some(FORMATS))
}

struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Norm {
    fn load(source: &dyn TensorSource, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            weight: load_float_values(source, &format!("{prefix}.weight"), &[WIDTH as u64])?,
            bias: load_float_values(source, &format!("{prefix}.bias"), &[WIDTH as u64])?,
        })
    }

    fn apply(&self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        layer_norm_rows(hidden, CONTEXT, Some(&self.weight), Some(&self.bias), 1e-5)
    }
}

struct Block<'a> {
    norm1: Norm,
    norm2: Norm,
    q: Linear<'a>,
    k: Linear<'a>,
    v: Linear<'a>,
    out: Linear<'a>,
    fc1: Linear<'a>,
    fc2: Linear<'a>,
}

impl<'a> Block<'a> {
    fn load(source: &'a dyn TensorSource, index: usize) -> Result<Self, String> {
        let layer = format!("text_model.encoder.layers.{index}");
        let linear = |key: &str, cols, rows| {
            let prefix = format!("{layer}.{key}");
            let weight = matrix(source, &format!("{prefix}.weight"), cols, rows)?;
            let bias = load_float_values(source, &format!("{prefix}.bias"), &[rows as u64])?;
            Linear::from_weight(weight, Some(bias))
        };
        Ok(Self {
            norm1: Norm::load(source, &format!("{layer}.layer_norm1"))?,
            norm2: Norm::load(source, &format!("{layer}.layer_norm2"))?,
            q: linear("self_attn.q_proj", WIDTH, WIDTH)?,
            k: linear("self_attn.k_proj", WIDTH, WIDTH)?,
            v: linear("self_attn.v_proj", WIDTH, WIDTH)?,
            out: linear("self_attn.out_proj", WIDTH, WIDTH)?,
            fc1: linear("mlp.fc1", WIDTH, FF)?,
            fc2: linear("mlp.fc2", FF, WIDTH)?,
        })
    }

    fn forward(
        &self,
        hidden: &mut [f32],
        executor: &mut LinearExecutor<'_, 'a>,
    ) -> Result<(), String> {
        let normed = self.norm1.apply(hidden)?;
        let [q, k, v] =
            Linear::forward_group([&self.q, &self.k, &self.v], executor, &normed, CONTEXT)?;
        let attended = attention_on_cpu(
            &q,
            &k,
            &v,
            AttentionSpec {
                query_tokens: CONTEXT,
                key_tokens: CONTEXT,
                query_heads: HEADS,
                key_value_heads: HEADS,
                head_dim: WIDTH / HEADS,
                causal: true,
                scale: ((WIDTH / HEADS) as f32).sqrt().recip(),
            },
        )?;
        let [projected] = Linear::forward_group([&self.out], executor, &attended, CONTEXT)?;
        vec_add_into(&projected, hidden);
        let normed = self.norm2.apply(hidden)?;
        let [mut ff] = Linear::forward_group([&self.fc1], executor, &normed, CONTEXT)?;
        for x in &mut ff {
            *x /= 1.0 + (-1.702 * *x).exp();
        }
        let [projected] = Linear::forward_group([&self.fc2], executor, &ff, CONTEXT)?;
        vec_add_into(&projected, hidden);
        Ok(())
    }
}

pub(super) struct TextEncoder<'a> {
    tokenizer: Tokenizer,
    #[cfg(feature = "vulkan")]
    device: Option<device::Device>,
    tokens: Weight<'a>,
    positions: Weight<'a>,
    blocks: Vec<Block<'a>>,
    norm: Norm,
}

impl<'a> TextEncoder<'a> {
    pub(super) fn load(source: &'a dyn TensorSource) -> Result<Self, String> {
        let json = source
            .metadata("tessera.tokenizer_json")
            .and_then(|value| value.to_string_val())
            .ok_or("missing Tessera CLIP tokenizer")?;
        let mut tokenizer = Tokenizer::from_bytes(json.as_bytes())
            .map_err(|error| format!("CLIP tokenizer: {error}"))?;
        if tokenizer.get_vocab_size(true) != VOCAB
            || tokenizer.token_to_id("<|startoftext|>") != Some(BOS)
            || tokenizer.token_to_id("<|endoftext|>") != Some(EOS)
        {
            return Err("invalid Tessera CLIP vocabulary or special tokens".into());
        }
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: CONTEXT,
                ..Default::default()
            }))
            .map_err(|error| format!("CLIP truncation: {error}"))?;
        Ok(Self {
            tokenizer,
            #[cfg(feature = "vulkan")]
            device: None,
            tokens: matrix(
                source,
                "text_model.embeddings.token_embedding.weight",
                WIDTH,
                VOCAB,
            )?,
            positions: matrix(
                source,
                "text_model.embeddings.position_embedding.weight",
                WIDTH,
                MAX_POS,
            )?,
            blocks: (0..12)
                .map(|i| Block::load(source, i))
                .collect::<Result<_, _>>()?,
            norm: Norm::load(source, "text_model.final_layer_norm")?,
        })
    }

    pub(super) fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, String> {
        let encoded = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|error| format!("CLIP encoding: {error}"))?;
        let mut ids = encoded.get_ids().to_vec();
        if ids.len() > CONTEXT
            || ids.first() != Some(&BOS)
            || ids.last() != Some(&EOS)
            || ids.iter().any(|&id| id as usize >= VOCAB)
        {
            return Err("invalid Tessera CLIP token sequence".into());
        }
        ids.resize(CONTEXT, EOS);
        Ok(ids)
    }

    pub(super) fn encode(
        &mut self,
        prompt: &str,
        policy: ComputePolicy,
        pool: Arc<ComputePool>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let ids = self.tokenize(prompt)?;
        let eos = ids
            .iter()
            .position(|&id| id == EOS)
            .ok_or("missing CLIP EOS")?;
        #[cfg(feature = "vulkan")]
        if policy != ComputePolicy::Cpu {
            // Auto retries this encode on CPU only if setting up or executing the
            // device graph failed. Strict Vulkan propagates the original error.
            let device_output = (|| -> Result<_, String> {
                let ctx = if crate::ops::scalar_mode() {
                    policy.context_with(|| Err("CLIP scalar mode requires CPU".into()))
                } else {
                    policy.context()
                }
                .map_err(|e| e.to_string())?;
                let Some(context) = ctx else { return Ok(None) };
                let device = match &self.device {
                    Some(device) => device,
                    None => {
                        let device =
                            device::Device::new(self, context).map_err(|e| e.to_string())?;
                        self.device.insert(device)
                    }
                };
                device
                    .encode(&ids, &self.tokens)
                    .map(Some)
                    .map_err(|e| e.to_string())
            })();
            match device_output {
                Ok(Some(states)) => {
                    let pooled = states[eos * WIDTH..(eos + 1) * WIDTH].to_vec();
                    return Ok((states, pooled));
                }
                Err(error) if policy == ComputePolicy::Vulkan => return Err(error),
                Err(error) => {
                    log::info!("Tessera CLIP falling back to CPU: {error}");
                    self.device = None;
                }
                Ok(None) => {}
            }
        }
        policy.check_build()?;
        let mut hidden = vec![0.; CONTEXT * WIDTH];
        let mut pos_embedding = vec![0.; WIDTH];
        for (pos, (&token, row)) in ids.iter().zip(hidden.chunks_exact_mut(WIDTH)).enumerate() {
            self.tokens.embedding_lookup(token, row);
            self.positions
                .embedding_lookup(pos as u32, &mut pos_embedding);
            vec_add_into(&pos_embedding, row);
        }
        let bindings = self
            .blocks
            .iter()
            .flat_map(|b| [&b.q, &b.k, &b.v, &b.out, &b.fc1, &b.fc2])
            .map(Linear::binding)
            .collect();
        let mut executor = LinearExecutor::new(ComputePolicy::Cpu, bindings, CONTEXT, pool)
            .map_err(|e| e.to_string())?;
        for block in &self.blocks {
            block.forward(&mut hidden, &mut executor)?;
        }
        let states = self.norm.apply(&hidden)?;
        let pooled = states[eos * WIDTH..(eos + 1) * WIDTH].to_vec();
        Ok((states, pooled))
    }
}

#[cfg(test)]
mod tests;
