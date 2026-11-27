use super::{json_metadata, Linear, Norm};
use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::qwen3::embedding::l2_normalize_embedding;
use crate::ops::{dot_f32, gelu_erf_inplace, softmax_inplace};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Deserialize)]
pub(super) struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "epsilon")]
    pub layer_norm_eps: f32,
    #[serde(default = "activation")]
    pub hidden_act: String,
    #[serde(default)]
    pub image_size: usize,
    #[serde(default)]
    pub patch_size: usize,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub vocab_size: usize,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}
fn epsilon() -> f32 {
    1e-5
}
fn activation() -> String {
    "quick_gelu".into()
}

fn pooled_token(ids: &[u32], eos: Option<u32>) -> Result<usize, String> {
    let index = if eos == Some(2) {
        ids.iter()
            .max()
            .and_then(|largest| ids.iter().position(|id| id == largest))
    } else {
        ids.iter().position(|id| *id == eos.unwrap_or(49407))
    };
    index.ok_or_else(|| "CLIP input has no EOS token".into())
}

pub(super) struct Block<'a> {
    norm1: Norm,
    norm2: Norm,
    q: Linear<'a>,
    k: Linear<'a>,
    v: Linear<'a>,
    output: Linear<'a>,
    up: Linear<'a>,
    down: Linear<'a>,
}
impl<'a> Block<'a> {
    fn load(source: &'a dyn TensorSource, prefix: &str, cfg: &Config) -> Result<Self, String> {
        let h = cfg.hidden_size;
        let ff = cfg.intermediate_size;
        Ok(Self {
            norm1: Norm::load(
                source,
                &format!("{prefix}.layer_norm1"),
                h,
                cfg.layer_norm_eps,
            )?,
            norm2: Norm::load(
                source,
                &format!("{prefix}.layer_norm2"),
                h,
                cfg.layer_norm_eps,
            )?,
            q: Linear::load(source, &format!("{prefix}.self_attn.q_proj"), h, h, true)?,
            k: Linear::load(source, &format!("{prefix}.self_attn.k_proj"), h, h, true)?,
            v: Linear::load(source, &format!("{prefix}.self_attn.v_proj"), h, h, true)?,
            output: Linear::load(source, &format!("{prefix}.self_attn.out_proj"), h, h, true)?,
            up: Linear::load(source, &format!("{prefix}.mlp.fc1"), h, ff, true)?,
            down: Linear::load(source, &format!("{prefix}.mlp.fc2"), ff, h, true)?,
        })
    }
    fn forward(
        &self,
        hidden: &mut [f32],
        cfg: &Config,
        causal: bool,
        pool: &ComputePool,
    ) -> Result<(), String> {
        let n = hidden.len() / cfg.hidden_size;
        let h = cfg.hidden_size;
        let normed = self.norm1.apply(hidden);
        let mut q = self.q.apply(&normed, pool)?;
        let k = self.k.apply(&normed, pool)?;
        let v = self.v.apply(&normed, pool)?;
        let head_dim = h / cfg.num_attention_heads;
        // HF CLIP scales Q before the QK product.
        let scale = 1.0 / (head_dim as f32).sqrt();
        for value in &mut q {
            *value *= scale;
        }
        let attended = attention(&q, &k, &v, n, h, cfg.num_attention_heads, causal);
        let projected = self.output.apply(&attended, pool)?;
        for (x, y) in hidden.iter_mut().zip(projected) {
            *x += y;
        }
        let normed = self.norm2.apply(hidden);
        let mut ff = self.up.apply(&normed, pool)?;
        match cfg.hidden_act.as_str() {
            "quick_gelu" => {
                for value in &mut ff {
                    *value *= 1.0 / (1.0 + (-1.702 * *value).exp());
                }
            }
            "gelu" => gelu_erf_inplace(&mut ff),
            other => return Err(format!("unsupported CLIP activation {other:?}")),
        }
        let projected = self.down.apply(&ff, pool)?;
        for (x, y) in hidden.iter_mut().zip(projected) {
            *x += y;
        }
        Ok(())
    }
}

pub(super) fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    h: usize,
    heads: usize,
    causal: bool,
) -> Vec<f32> {
    let d = h / heads;
    let mut result = vec![0.0; n * h];
    let mut scores = vec![0.0; n];
    for t in 0..n {
        for head in 0..heads {
            let range = t * h + head * d..t * h + (head + 1) * d;
            for s in 0..n {
                scores[s] = if causal && s > t {
                    f32::NEG_INFINITY
                } else {
                    dot_f32(
                        &q[range.clone()],
                        &k[s * h + head * d..s * h + (head + 1) * d],
                        d,
                    )
                };
            }
            softmax_inplace(&mut scores);
            for (s, p) in scores.iter().enumerate() {
                for i in 0..d {
                    result[t * h + head * d + i] += *p * v[s * h + head * d + i];
                }
            }
        }
    }
    result
}

pub(super) struct Vision<'a> {
    pub cfg: Config,
    patch: Linear<'a>,
    class: Vec<f32>,
    position: Vec<f32>,
    pre_norm: Norm,
    post_norm: Norm,
    blocks: Vec<Block<'a>>,
}
impl<'a> Vision<'a> {
    pub fn load(source: &'a dyn TensorSource, prefix: &str, cfg: Config) -> Result<Self, String> {
        validate(&cfg)?;
        if cfg.image_size == 0
            || cfg.patch_size == 0
            || !cfg.image_size.is_multiple_of(cfg.patch_size)
        {
            return Err("invalid CLIP image/patch size".into());
        }
        let h = cfg.hidden_size;
        let p = cfg.patch_size;
        let n = (cfg.image_size / p).pow(2) + 1;
        let weight_name = format!("{prefix}.embeddings.patch_embedding.weight");
        let patch = Linear::load_shape(
            source,
            &weight_name,
            &[p as u64, p as u64, 3, h as u64],
            p * p * 3,
            h,
            None,
        )?;
        let class = load_f32_tensor(
            source,
            &format!("{prefix}.embeddings.class_embedding"),
            &[h as u64],
        )?;
        let position = load_f32_tensor(
            source,
            &format!("{prefix}.embeddings.position_embedding.weight"),
            &[h as u64, n as u64],
        )?;
        let pre_norm = Norm::load(
            source,
            &format!("{prefix}.pre_layrnorm"),
            h,
            cfg.layer_norm_eps,
        )?;
        let post_norm = Norm::load(
            source,
            &format!("{prefix}.post_layernorm"),
            h,
            cfg.layer_norm_eps,
        )?;
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(source, &format!("{prefix}.encoder.layers.{i}"), &cfg))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            cfg,
            patch,
            class,
            position,
            pre_norm,
            post_norm,
            blocks,
        })
    }
    pub fn encode(
        &self,
        pixels: &[f32],
        stop: usize,
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        let side = self.cfg.image_size;
        let p = self.cfg.patch_size;
        let grid = side / p;
        let h = self.cfg.hidden_size;
        if pixels.len() != side * side * 3 || stop > self.blocks.len() {
            return Err("invalid CLIP image shape or feature layer".into());
        }
        let mut patches = Vec::with_capacity(grid * grid * p * p * 3);
        for y in 0..grid {
            for x in 0..grid {
                for c in 0..3 {
                    for py in 0..p {
                        for px in 0..p {
                            patches.push(pixels[((y * p + py) * side + x * p + px) * 3 + c]);
                        }
                    }
                }
            }
        }
        let encoded = self.patch.apply(&patches, pool)?;
        let mut hidden = self.class.clone();
        hidden.extend(encoded);
        for (x, pos) in hidden.iter_mut().zip(&self.position) {
            *x += *pos;
        }
        hidden = self.pre_norm.apply(&hidden);
        for block in &self.blocks[..stop] {
            block.forward(&mut hidden, &self.cfg, false, pool)?;
        }
        if hidden.len() != (grid * grid + 1) * h {
            return Err("invalid CLIP feature shape".into());
        }
        Ok(hidden)
    }
    pub fn pooled(&self, pixels: &[f32], pool: &ComputePool) -> Result<Vec<f32>, String> {
        let hidden = self.encode(pixels, self.blocks.len(), pool)?;
        Ok(self.post_norm.apply(&hidden[..self.cfg.hidden_size]))
    }
}
fn validate(cfg: &Config) -> Result<(), String> {
    if cfg.hidden_size == 0
        || cfg.intermediate_size == 0
        || cfg.num_hidden_layers == 0
        || cfg.num_attention_heads == 0
        || !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads)
        || !cfg.layer_norm_eps.is_finite()
        || cfg.layer_norm_eps <= 0.0
    {
        return Err("invalid CLIP transformer dimensions".into());
    }
    if !["quick_gelu", "gelu"].contains(&cfg.hidden_act.as_str()) {
        return Err(format!("unsupported CLIP activation {:?}", cfg.hidden_act));
    }
    Ok(())
}

#[derive(Deserialize)]
struct ModelConfig {
    text_config: Config,
    vision_config: Config,
    projection_dim: usize,
}

pub(super) fn compute(
    source: &dyn TensorSource,
    prompt: &str,
    image: Option<&Path>,
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    let config: ModelConfig = json_metadata(source, "bge.config")?;
    let mut embedding = vec![0.0; config.projection_dim];
    if !prompt.is_empty() || image.is_none() {
        let cfg = &config.text_config;
        validate(cfg)?;
        let raw = source
            .metadata("bge.tokenizer_json")
            .and_then(|v| v.to_string_val())
            .ok_or("missing BGE CLIP tokenizer")?;
        let mut tokenizer = tokenizers::Tokenizer::from_bytes(raw).map_err(|e| e.to_string())?;
        if cfg.max_position_embeddings < 2 {
            return Err("invalid CLIP text context".into());
        }
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: cfg.max_position_embeddings,
                ..Default::default()
            }))
            .map_err(|e| e.to_string())?;
        let encoding = tokenizer.encode(prompt, true).map_err(|e| e.to_string())?;
        let ids = encoding.get_ids();
        let h = cfg.hidden_size;
        if ids.is_empty()
            || ids.len() > cfg.max_position_embeddings
            || ids.iter().any(|i| *i as usize >= cfg.vocab_size)
        {
            return Err("invalid CLIP token IDs".into());
        }
        let info = source
            .tensor_info("text_model.embeddings.token_embedding.weight")
            .ok_or("missing CLIP token embedding")?;
        if info.dims != [h as u64, cfg.vocab_size as u64] {
            return Err("invalid CLIP token embedding shape".into());
        }
        let bytes = source
            .tensor_slice("text_model.embeddings.token_embedding.weight")
            .ok_or("missing CLIP token embedding data")?;
        let position = load_f32_tensor(
            source,
            "text_model.embeddings.position_embedding.weight",
            &[h as u64, cfg.max_position_embeddings as u64],
        )?;
        let mut hidden = vec![0.0; ids.len() * h];
        for (t, id) in ids.iter().enumerate() {
            let row = &mut hidden[t * h..(t + 1) * h];
            crate::ops::embedding_lookup(bytes, *id, h, info.ggml_type, row);
            for (x, pos) in row.iter_mut().zip(&position[t * h..(t + 1) * h]) {
                *x += *pos;
            }
        }
        for i in 0..cfg.num_hidden_layers {
            Block::load(source, &format!("text_model.encoder.layers.{i}"), cfg)?.forward(
                &mut hidden,
                cfg,
                true,
                pool,
            )?;
        }
        let normalized = Norm::load(source, "text_model.final_layer_norm", h, cfg.layer_norm_eps)?
            .apply(&hidden);
        let eos = pooled_token(ids, cfg.eos_token_id)?;
        embedding = Linear::load(source, "text_projection", h, config.projection_dim, false)?
            .apply(&normalized[eos * h..(eos + 1) * h], pool)?;
    }
    if let Some(path) = image {
        let vision = Vision::load(source, "vision_model", config.vision_config)?;
        let pixels = super::preprocess(source, path, vision.cfg.image_size, false)?;
        let pooled = vision.pooled(&pixels, pool)?;
        let projected = Linear::load(
            source,
            "visual_projection",
            vision.cfg.hidden_size,
            config.projection_dim,
            false,
        )?
        .apply(&pooled, pool)?;
        for (x, v) in embedding.iter_mut().zip(projected) {
            *x += v;
        }
    }
    super::trace("embedding.pooled", &embedding)?;
    l2_normalize_embedding(&mut embedding)?;
    super::trace("embedding.final", &embedding)?;
    Ok(embedding)
}

#[cfg(test)]
mod tests {
    use super::pooled_token;

    #[test]
    fn clip_pooling_uses_first_eos_or_first_legacy_maximum() {
        assert_eq!(pooled_token(&[1, 49407, 9, 49407], None).unwrap(), 1);
        assert_eq!(pooled_token(&[1, 7, 9, 7], Some(7)).unwrap(), 1);
        assert_eq!(pooled_token(&[1, 9, 7, 9], Some(2)).unwrap(), 1);
        assert!(pooled_token(&[1, 7], None).is_err());
        assert!(pooled_token(&[], Some(2)).is_err());
    }
}
