use super::{DSparkConfig, TargetShape};
use crate::core::scratchpad::{KvCache, KvFormat, KvLifecycle};
use crate::core::tensor::{load_f32_tensor, MetaValue, TensorInfo, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::models::qwen3::trunk::{
    load_layers_static, Qwen3Config, Qwen3Model, Qwen3Rope, Qwen3Session,
};
use crate::ops::kernel::{QuantizedTensor, Weight};
use crate::ops::{f32_slice_to_f16, rms_norm, rms_norm_inplace, rope_neox_inplace};
use std::sync::Arc;

pub struct SharedHead {
    source: Arc<dyn TensorSource>,
    tokenizer: Arc<BPETokenizer>,
    target: TargetShape,
    context: usize,
}

impl SharedHead {
    pub fn new(
        source: Arc<dyn TensorSource>,
        tokenizer: Arc<BPETokenizer>,
        target: TargetShape,
        context: usize,
    ) -> Result<Self, String> {
        if target.hidden == 0 || target.vocab == 0 || target.layers == 0 || context == 0 {
            return Err("Invalid shared target head dimensions".into());
        }
        if tokenizer.vocab_size() != target.vocab {
            return Err(format!(
                "Target tokenizer vocabulary {} does not match model vocabulary {}",
                tokenizer.vocab_size(),
                target.vocab
            ));
        }
        require_tensor(
            &*source,
            "token_embd.weight",
            &[target.hidden, target.vocab],
        )?;
        if source.tensor_info("output.weight").is_some() {
            require_tensor(&*source, "output.weight", &[target.hidden, target.vocab])?;
        }
        Ok(Self {
            source,
            tokenizer,
            target,
            context,
        })
    }
}

struct CombinedSource {
    draft: Arc<dyn TensorSource>,
    target: Arc<dyn TensorSource>,
}

impl TensorSource for CombinedSource {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.draft.metadata(key)
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        if matches!(name, "token_embd.weight" | "output.weight") {
            self.target.tensor_info(name)
        } else {
            self.draft.tensor_info(name)
        }
    }

    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        if matches!(name, "token_embd.weight" | "output.weight") {
            self.target.tensor_slice(name)
        } else {
            self.draft.tensor_slice(name)
        }
    }
}

pub struct DSparkModel {
    _source: Arc<dyn TensorSource>,
    pub config: DSparkConfig,
    backbone: Qwen3Model,
    encoder: Weight<'static>,
    encoder_norm: Vec<f32>,
    markov_w1: Weight<'static>,
    markov_w2: Weight<'static>,
    confidence: Weight<'static>,
    confidence_bias: f32,
    mask_token_id: u32,
}

impl DSparkModel {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        head: SharedHead,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let config = DSparkConfig::from_source(&*source, head.target)?;
        validate_backbone(&*source, &config, head.target.hidden)?;
        let mask_token_id = source
            .metadata("tokenizer.ggml.mask_token_id")
            .and_then(metadata_u32)
            .filter(|&id| id < config.vocab as u32)
            .ok_or("Missing or invalid metadata: tokenizer.ggml.mask_token_id")?;

        let combined: Arc<dyn TensorSource> = Arc::new(CombinedSource {
            draft: source,
            target: head.source,
        });
        let qwen_config = Qwen3Config {
            architecture: "dflash".into(),
            n_embd: config.hidden,
            n_layer: config.layers,
            n_head: config.heads,
            n_head_kv: config.kv_heads,
            n_embd_head_k: config.head_dim,
            n_embd_head_v: config.head_dim,
            n_ff: config.ffn,
            vocab: config.vocab,
            n_ctx: head.context,
            eps: config.eps,
            freq_base: config.rope_base,
            has_qk_norm: true,
            has_qkv_bias: false,
            n_deepstack_layers: 0,
            moe: None,
            rope: Qwen3Rope::Neox,
        };
        let query = config
            .heads
            .checked_mul(config.head_dim)
            .ok_or("DSpark query width overflow")?;
        let kv = config
            .kv_heads
            .checked_mul(config.head_dim)
            .ok_or("DSpark KV width overflow")?;
        let layers = load_layers_static(
            Arc::clone(&combined),
            config.layers,
            config.hidden,
            query,
            kv,
            config.ffn,
            config.head_dim,
            true,
            false,
            None,
        )?;
        let output_norm = load_f32_tensor(
            &*combined,
            "output_norm.weight",
            &[u64::try_from(config.hidden).map_err(|_| "DSpark hidden width overflow")?],
        )?;
        let token_embedding =
            load_weight(&combined, "token_embd.weight", config.hidden, config.vocab)?;
        let output = if combined.tensor_info("output.weight").is_some() {
            load_weight(&combined, "output.weight", config.hidden, config.vocab)?
        } else {
            load_weight(&combined, "token_embd.weight", config.hidden, config.vocab)?
        };
        let backbone = Qwen3Model {
            source: Arc::clone(&combined),
            tokenizer: head.tokenizer,
            pool,
            config: qwen_config,
            layers,
            output_norm,
            token_embedding,
            output,
        };
        let encoder_input = config
            .target_layers
            .len()
            .checked_mul(head.target.hidden)
            .ok_or("DSpark encoder input overflow")?;
        let encoder = load_weight(&combined, "fc.weight", encoder_input, config.hidden)?;
        let encoder_norm = load_f32_tensor(
            &*combined,
            "enc.output_norm.weight",
            &[config.hidden as u64],
        )?;
        let markov_w1 = load_weight(
            &combined,
            "markov_w1.weight",
            config.markov_rank,
            config.vocab,
        )?;
        let markov_w2 = load_weight(
            &combined,
            "markov_w2.weight",
            config.markov_rank,
            config.vocab,
        )?;
        let confidence_width = config
            .hidden
            .checked_add(config.markov_rank)
            .ok_or("DSpark confidence width overflow")?;
        let confidence = load_weight(&combined, "conf_proj.weight", confidence_width, 1)?;
        let confidence_bias = if combined.tensor_info("conf_proj.bias").is_some() {
            load_f32_tensor(&*combined, "conf_proj.bias", &[1])?[0]
        } else {
            0.0
        };

        Ok(Self {
            _source: combined,
            config,
            backbone,
            encoder,
            encoder_norm,
            markov_w1,
            markov_w2,
            confidence,
            confidence_bias,
            mask_token_id,
        })
    }
}

pub struct DSparkSession<'model> {
    model: &'model DSparkModel,
    target_hidden: usize,
    session: Qwen3Session<'model>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DraftBlock {
    pub token_ids: Vec<u32>,
    pub confidence: Vec<f32>,
}

impl<'model> DSparkSession<'model> {
    pub fn new(model: &'model DSparkModel, capacity: usize) -> Result<Self, String> {
        let mut session = Qwen3Session::new_with_kv_state(
            &model.backbone,
            capacity,
            KvFormat::F16,
            KvLifecycle::Ephemeral,
        )?;
        #[cfg(feature = "vulkan")]
        {
            session.gpu = None;
            session.full_model_gpu_failed = true;
        }
        let target_hidden = model.encoder.n_in / model.config.target_layers.len();
        Ok(Self {
            model,
            target_hidden,
            session,
        })
    }

    pub fn position(&self) -> usize {
        self.session.kv_state.seq_len
    }

    pub fn target_layers(&self) -> &[usize] {
        &self.model.config.target_layers
    }

    pub fn block_size(&self) -> usize {
        self.model.config.block_size
    }

    pub fn inject(&mut self, position: usize, features: &[f32]) -> Result<(), String> {
        if position > self.session.kv_state.seq_len || position >= self.session.capacity {
            return Err(format!("Invalid DSpark injection position: {position}"));
        }
        let expected = self
            .model
            .config
            .target_layers
            .len()
            .checked_mul(self.target_hidden)
            .ok_or("DSpark feature width overflow")?;
        if features.len() != expected || features.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "Invalid DSpark feature width: expected {expected}, got {}",
                features.len()
            ));
        }
        let hidden = self.model.config.hidden;
        let mut fused = self.model.encoder.matmul(features);
        let mut normalized = vec![0.0; hidden];
        rms_norm(
            &mut fused,
            &self.model.encoder_norm,
            &mut normalized,
            self.model.config.eps,
        );
        let kv_width = self
            .model
            .config
            .kv_heads
            .checked_mul(self.model.config.head_dim)
            .ok_or("DSpark KV width overflow")?;
        let cache_len = self
            .model
            .config
            .layers
            .checked_mul(self.session.capacity)
            .and_then(|len| len.checked_mul(kv_width))
            .ok_or("DSpark cache length overflow")?;
        for (layer, weights) in self.model.backbone.layers.iter().enumerate() {
            let mut key = weights.wk.matmul(&normalized);
            let value = weights.wv.matmul(&normalized);
            let key_norm = weights.k_norm.as_deref().ok_or("Missing DSpark key norm")?;
            for head in key.chunks_exact_mut(self.model.config.head_dim) {
                rms_norm_inplace(head, key_norm, self.model.config.eps);
                rope_neox_inplace(
                    head,
                    position,
                    self.model.config.head_dim,
                    self.model.config.rope_base,
                );
            }
            let offset = (layer * self.session.capacity + position) * kv_width;
            match &mut self.session.kv_state.cache {
                KvCache::F16(cache) => {
                    debug_assert_eq!(cache.k.len(), cache_len);
                    f32_slice_to_f16(&key, &mut cache.k[offset..offset + kv_width]);
                    f32_slice_to_f16(&value, &mut cache.v[offset..offset + kv_width]);
                }
                KvCache::F32(cache) => {
                    debug_assert_eq!(cache.k.len(), cache_len);
                    cache.k[offset..offset + kv_width].copy_from_slice(&key);
                    cache.v[offset..offset + kv_width].copy_from_slice(&value);
                }
            }
        }
        self.session.kv_state.seq_len = position
            .checked_add(1)
            .ok_or("DSpark injection position overflow")?;
        self.session.kv_state.update_access();
        Ok(())
    }

    pub fn draft(
        &mut self,
        last_token: u32,
        n: usize,
        confidence_min: f32,
    ) -> Result<DraftBlock, String> {
        if n == 0 || n > self.model.config.block_size {
            return Err(format!(
                "DSpark draft size must be within 1..={}",
                self.model.config.block_size
            ));
        }
        if last_token as usize >= self.model.config.vocab
            || !confidence_min.is_finite()
            || !(0.0..=1.0).contains(&confidence_min)
        {
            return Err("Invalid DSpark draft arguments".into());
        }
        let base = self.session.kv_state.seq_len;
        let end = base
            .checked_add(n)
            .ok_or("DSpark draft position overflow")?;
        let mut tokens = vec![self.model.mask_token_id; n];
        tokens[0] = last_token;
        let positions = (base..end)
            .map(|position| [position; 4])
            .collect::<Vec<_>>();
        let capture = self.session.forward_non_causal_block(&tokens, &positions)?;
        let hidden = self.model.config.hidden;
        let vocab = self.model.config.vocab;
        let hidden_len = n
            .checked_mul(hidden)
            .ok_or("DSpark hidden capture length overflow")?;
        let logits_len = n
            .checked_mul(vocab)
            .ok_or("DSpark logits capture length overflow")?;
        if capture.hidden.len() != hidden_len || capture.logits.len() != logits_len {
            return Err("DSpark backbone returned invalid output shapes".into());
        }

        let rank = self.model.config.markov_rank;
        let mut previous = last_token;
        let mut token_ids = Vec::with_capacity(n);
        let mut confidence = Vec::with_capacity(n);
        let mut markov = vec![0.0; rank];
        let confidence_width = hidden
            .checked_add(rank)
            .ok_or("DSpark confidence width overflow")?;
        let mut confidence_input = vec![0.0; confidence_width];
        for row in 0..n {
            self.model.markov_w1.embedding_lookup(previous, &mut markov);
            let bias = self.model.markov_w2.matmul(&markov);
            let logits = &capture.logits[row * vocab..(row + 1) * vocab];
            let token = logits
                .iter()
                .zip(&bias)
                .enumerate()
                .max_by(|(_, (left, left_bias)), (_, (right, right_bias))| {
                    (*left + *left_bias).total_cmp(&(*right + *right_bias))
                })
                .map(|(token, _)| token as u32)
                .ok_or("DSpark produced no logits")?;
            confidence_input[..hidden]
                .copy_from_slice(&capture.hidden[row * hidden..(row + 1) * hidden]);
            confidence_input[hidden..].copy_from_slice(&markov);
            let score =
                self.model.confidence.matmul(&confidence_input)[0] + self.model.confidence_bias;
            let probability = 1.0 / (1.0 + (-score).exp());
            if !probability.is_finite() {
                return Err("DSpark produced non-finite confidence".into());
            }
            if probability < confidence_min {
                break;
            }
            token_ids.push(token);
            confidence.push(probability);
            previous = token;
        }
        Ok(DraftBlock {
            token_ids,
            confidence,
        })
    }
}

fn metadata_u32(value: &MetaValue) -> Option<u32> {
    match value {
        MetaValue::Uint8(value) => Some((*value).into()),
        MetaValue::Uint16(value) => Some((*value).into()),
        MetaValue::Uint32(value) => Some(*value),
        MetaValue::Uint64(value) => u32::try_from(*value).ok(),
        MetaValue::Int8(value) => u32::try_from(*value).ok(),
        MetaValue::Int16(value) => u32::try_from(*value).ok(),
        MetaValue::Int32(value) => u32::try_from(*value).ok(),
        MetaValue::Int64(value) => u32::try_from(*value).ok(),
        _ => None,
    }
}

fn load_weight(
    source: &Arc<dyn TensorSource>,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Weight<'static>, String> {
    require_tensor(&**source, name, &[n_in, n_out])?;
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let bytes: &'static [u8] = unsafe { std::mem::transmute(bytes) };
    let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        n_in,
        n_out,
    ));
    weight.n_in = n_in;
    weight.n_out = n_out;
    Ok(weight)
}

fn require_tensor(source: &dyn TensorSource, name: &str, dims: &[usize]) -> Result<(), String> {
    let expected = dims
        .iter()
        .map(|&value| u64::try_from(value))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| format!("Tensor {name} shape overflow"))?;
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != expected {
        return Err(format!(
            "Invalid tensor {name} shape {:?}; expected {expected:?}",
            info.dims
        ));
    }
    let expected_bytes = info
        .checked_nbytes()
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?;
    let actual_bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?
        .len();
    if actual_bytes != expected_bytes {
        return Err(format!(
            "Invalid tensor data length for {name}: {actual_bytes}; expected {expected_bytes}"
        ));
    }
    Ok(())
}

fn validate_backbone(
    source: &dyn TensorSource,
    config: &DSparkConfig,
    target_hidden: usize,
) -> Result<(), String> {
    let encoder_input = config
        .target_layers
        .len()
        .checked_mul(target_hidden)
        .ok_or("DSpark encoder input overflow")?;
    let confidence_width = config
        .hidden
        .checked_add(config.markov_rank)
        .ok_or("DSpark confidence width overflow")?;
    require_tensor(source, "fc.weight", &[encoder_input, config.hidden])?;
    require_tensor(source, "enc.output_norm.weight", &[config.hidden])?;
    require_tensor(source, "output_norm.weight", &[config.hidden])?;
    require_tensor(
        source,
        "markov_w1.weight",
        &[config.markov_rank, config.vocab],
    )?;
    require_tensor(
        source,
        "markov_w2.weight",
        &[config.markov_rank, config.vocab],
    )?;
    require_tensor(source, "conf_proj.weight", &[confidence_width, 1])?;
    if source.tensor_info("conf_proj.bias").is_some() {
        require_tensor(source, "conf_proj.bias", &[1])?;
    }
    let query = config
        .heads
        .checked_mul(config.head_dim)
        .ok_or("DSpark query width overflow")?;
    let kv = config
        .kv_heads
        .checked_mul(config.head_dim)
        .ok_or("DSpark KV width overflow")?;
    for layer in 0..config.layers {
        for (name, dims) in [
            (format!("blk.{layer}.attn_norm.weight"), vec![config.hidden]),
            (
                format!("blk.{layer}.attn_q_norm.weight"),
                vec![config.head_dim],
            ),
            (
                format!("blk.{layer}.attn_k_norm.weight"),
                vec![config.head_dim],
            ),
            (format!("blk.{layer}.ffn_norm.weight"), vec![config.hidden]),
            (
                format!("blk.{layer}.attn_q.weight"),
                vec![config.hidden, query],
            ),
            (
                format!("blk.{layer}.attn_k.weight"),
                vec![config.hidden, kv],
            ),
            (
                format!("blk.{layer}.attn_v.weight"),
                vec![config.hidden, kv],
            ),
            (
                format!("blk.{layer}.attn_output.weight"),
                vec![query, config.hidden],
            ),
            (
                format!("blk.{layer}.ffn_gate.weight"),
                vec![config.hidden, config.ffn],
            ),
            (
                format!("blk.{layer}.ffn_up.weight"),
                vec![config.hidden, config.ffn],
            ),
            (
                format!("blk.{layer}.ffn_down.weight"),
                vec![config.ffn, config.hidden],
            ),
        ] {
            require_tensor(source, &name, &dims)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{DSparkModel, DSparkSession, SharedHead};
    use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
    use crate::core::thread_pool::ComputePool;
    use crate::core::tokenizer::BPETokenizer;
    use crate::models::dspark::TargetShape;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[derive(Default)]
    struct FixtureSource {
        metadata: HashMap<String, MetaValue>,
        tensors: HashMap<String, TensorInfo>,
        data: HashMap<String, Vec<u8>>,
    }

    impl FixtureSource {
        fn metadata(mut self, key: &str, value: MetaValue) -> Self {
            self.metadata.insert(key.into(), value);
            self
        }

        fn tensor(mut self, name: &str, dims: &[u64], values: &[f32]) -> Self {
            assert_eq!(dims.iter().product::<u64>() as usize, values.len());
            self.tensors.insert(
                name.into(),
                TensorInfo {
                    name: name.into(),
                    dims: dims.to_vec(),
                    ggml_type: GGMLType::F32,
                    offset: 0,
                },
            );
            self.data.insert(
                name.into(),
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            );
            self
        }
    }

    impl TensorSource for FixtureSource {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.metadata.get(key)
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.tensors.get(name)
        }

        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.data.get(name).map(Vec::as_slice)
        }
    }

    fn tokenizer() -> Arc<BPETokenizer> {
        let metadata = HashMap::from([
            (
                "tokenizer.ggml.model".to_string(),
                MetaValue::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.pre".to_string(),
                MetaValue::String("qwen2".into()),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                MetaValue::Array(
                    MetaValueType::String,
                    (0..4)
                        .map(|token| MetaValue::String(format!("t{token}")))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                MetaValue::Array(MetaValueType::Uint32, vec![MetaValue::Uint32(1); 4]),
            ),
            (
                "tokenizer.ggml.merges".to_string(),
                MetaValue::Array(MetaValueType::String, vec![]),
            ),
        ]);
        Arc::new(BPETokenizer::from_gguf_metadata(|key| metadata.get(key).cloned()).unwrap())
    }

    fn target_source() -> Arc<dyn TensorSource> {
        Arc::new(
            FixtureSource::default()
                .tensor("token_embd.weight", &[2, 4], &[0.0; 8])
                .tensor("output.weight", &[2, 4], &[0.0; 8]),
        )
    }

    fn draft_source() -> Arc<dyn TensorSource> {
        let zero_2x2 = [0.0; 4];
        let markov_w1 = [
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 1.0, 0.0, 0.0, // token 1
            0.0, 0.0, 1.0, 0.0, // token 2
            0.0, 0.0, 0.0, 1.0, // token 3
        ];
        let markov_w2 = [
            0.0, 0.0, 0.0, 3.0, // output token 0 follows token 3
            3.0, 0.0, 0.0, 0.0, // output token 1 follows token 0
            0.0, 3.0, 0.0, 0.0, // output token 2 follows token 1
            0.0, 0.0, 3.0, 0.0, // output token 3 follows token 2
        ];
        Arc::new(
            FixtureSource::default()
                .metadata("general.architecture", MetaValue::String("dflash".into()))
                .metadata("dflash.block_size", MetaValue::Uint32(3))
                .metadata(
                    "dflash.target_layers",
                    MetaValue::Array(MetaValueType::Uint32, vec![MetaValue::Uint32(1)]),
                )
                .metadata("dflash.embedding_length", MetaValue::Uint32(2))
                .metadata("dflash.block_count", MetaValue::Uint32(1))
                .metadata("dflash.attention.head_count", MetaValue::Uint32(1))
                .metadata("dflash.attention.head_count_kv", MetaValue::Uint32(1))
                .metadata("dflash.rope.dimension_count", MetaValue::Uint32(2))
                .metadata("dflash.feed_forward_length", MetaValue::Uint32(2))
                .metadata(
                    "dflash.attention.layer_norm_rms_epsilon",
                    MetaValue::Float32(1e-6),
                )
                .metadata("dflash.rope.freq_base", MetaValue::Float32(10_000.0))
                .metadata("tokenizer.ggml.mask_token_id", MetaValue::Uint32(0))
                .tensor("fc.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0])
                .tensor("enc.output_norm.weight", &[2], &[1.0, 1.0])
                .tensor("output_norm.weight", &[2], &[1.0, 1.0])
                .tensor("blk.0.attn_norm.weight", &[2], &[1.0, 1.0])
                .tensor("blk.0.attn_q_norm.weight", &[2], &[1.0, 1.0])
                .tensor("blk.0.attn_k_norm.weight", &[2], &[1.0, 1.0])
                .tensor("blk.0.ffn_norm.weight", &[2], &[1.0, 1.0])
                .tensor("blk.0.attn_q.weight", &[2, 2], &zero_2x2)
                .tensor("blk.0.attn_k.weight", &[2, 2], &zero_2x2)
                .tensor("blk.0.attn_v.weight", &[2, 2], &zero_2x2)
                .tensor("blk.0.attn_output.weight", &[2, 2], &zero_2x2)
                .tensor("blk.0.ffn_gate.weight", &[2, 2], &zero_2x2)
                .tensor("blk.0.ffn_up.weight", &[2, 2], &zero_2x2)
                .tensor("blk.0.ffn_down.weight", &[2, 2], &zero_2x2)
                .tensor("markov_w1.weight", &[4, 4], &markov_w1)
                .tensor("markov_w2.weight", &[4, 4], &markov_w2)
                .tensor(
                    "conf_proj.weight",
                    &[6, 1],
                    &[0.0, 0.0, 2.0, 2.0, -2.0, 2.0],
                ),
        )
    }

    fn fixture_session() -> DSparkSession<'static> {
        let head = SharedHead::new(
            target_source(),
            tokenizer(),
            TargetShape {
                hidden: 2,
                vocab: 4,
                layers: 2,
            },
            16,
        )
        .unwrap();
        let model = Box::leak(Box::new(
            DSparkModel::from_source(draft_source(), head, Arc::new(ComputePool::new(1))).unwrap(),
        ));
        DSparkSession::new(model, 16).unwrap()
    }

    #[test]
    fn markov_head_chains_on_previous_prediction() {
        let block = fixture_session().draft(1, 3, 0.0).unwrap();
        assert_eq!(block.token_ids, vec![2, 3, 0]);
        assert_eq!(block.confidence.len(), 3);
    }

    #[test]
    fn confidence_threshold_truncates_at_first_low_position() {
        let block = fixture_session().draft(1, 3, 0.5).unwrap();
        assert_eq!(block.token_ids, vec![2]);
        assert_eq!(block.confidence.len(), 1);
    }
}
