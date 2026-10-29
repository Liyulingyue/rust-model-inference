//! Frozen d3 Qwen3.5 backbone and task-gated Vela adapters.

mod vision;
pub(crate) use vision::{D3ImageInput, D3Images};

use super::tensor;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::format::safetensors::SafetensorSource;
use crate::ops::{
    dot_f32, rms_norm, silu, softmax_inplace, ssm_matvec, ssm_state_decay, vec_mad_f32,
};
use half::bf16;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy)]
enum Mask {
    Full,
    Causal,
    Prefix(usize),
}

struct Config {
    hidden: usize,
    intermediate: usize,
    layers: Vec<bool>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv: usize,
    eps: f32,
    vocab: usize,
    context: usize,
    rope_dim: usize,
    rope_base: f32,
    rope_sections: [usize; 3],
    eos: Option<u32>,
    tied: bool,
}

impl Config {
    fn parse(value: &Value) -> Result<Self, String> {
        if value["model_type"] != "qwen3_5" {
            return Err("Vela d3 requires an explicit qwen3_5 backbone".into());
        }
        let text = &value["text_config"];
        let n = |name: &str| -> Result<usize, String> {
            text[name]
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .filter(|&v| v > 0)
                .ok_or_else(|| format!("d3 text_config.{name} must be a positive integer"))
        };
        let hidden = n("hidden_size")?;
        let heads = n("num_attention_heads")?;
        let kv_heads = n("num_key_value_heads")?;
        let head_dim = n("head_dim")?;
        let key_heads = n("linear_num_key_heads")?;
        let value_heads = n("linear_num_value_heads")?;
        if heads % kv_heads != 0 || value_heads % key_heads != 0 {
            return Err("d3 attention head counts do not form complete groups".into());
        }
        let layers = text["layer_types"]
            .as_array()
            .ok_or("d3 has no layer_types")?
            .iter()
            .map(|v| match v.as_str() {
                Some("linear_attention") => Ok(false),
                Some("full_attention") => Ok(true),
                _ => Err("d3 has an unknown layer type".to_owned()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if layers.len() != n("num_hidden_layers")? {
            return Err("d3 layer_types length differs from num_hidden_layers".into());
        }
        if text.get("hidden_act").is_some_and(|v| v != "silu")
            || text.get("attn_output_gate").is_some_and(|v| v != true)
        {
            return Err("d3 needs SiLU MLP and attention output gates".into());
        }
        let rope = &text["rope_parameters"];
        if rope["rope_type"] != "default" {
            return Err("d3 supports the checkpoint's default partial mRoPE only".into());
        }
        let factor = rope["partial_rotary_factor"]
            .as_f64()
            .or_else(|| text["partial_rotary_factor"].as_f64())
            .unwrap_or(1.0);
        let rope_dim = (head_dim as f64 * factor) as usize;
        let rope_base = rope["rope_theta"].as_f64().ok_or("d3 has no rope_theta")? as f32;
        let eps = text["rms_norm_eps"]
            .as_f64()
            .ok_or("d3 has no rms_norm_eps")? as f32;
        if !factor.is_finite()
            || factor <= 0.
            || factor > 1.
            || rope_dim == 0
            || rope_dim % 2 != 0
            || !rope_base.is_finite()
            || rope_base <= 0.
            || !eps.is_finite()
            || eps <= 0.
        {
            return Err("d3 has invalid RoPE or normalization parameters".into());
        }
        let mut rope_sections = [0; 3];
        if let Some(sections) = rope["mrope_section"].as_array() {
            if sections.len() != 3 {
                return Err("d3 mrope_section must contain three axes".into());
            }
            for (dst, v) in rope_sections.iter_mut().zip(sections) {
                *dst = v
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or("invalid mrope_section")?;
            }
            if rope_sections
                .iter()
                .try_fold(0usize, |sum, &v| sum.checked_add(v))
                != Some(rope_dim / 2)
            {
                return Err("d3 mrope_section does not cover the rotary dimensions".into());
            }
        }
        let cfg = Self {
            hidden,
            intermediate: n("intermediate_size")?,
            layers,
            heads,
            kv_heads,
            head_dim,
            key_heads,
            value_heads,
            key_dim: n("linear_key_head_dim")?,
            value_dim: n("linear_value_head_dim")?,
            conv: n("linear_conv_kernel_dim")?,
            eps,
            vocab: n("vocab_size")?,
            context: n("max_position_embeddings")?,
            rope_dim,
            rope_base,
            rope_sections,
            eos: text["eos_token_id"]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok()),
            tied: text["tie_word_embeddings"]
                .as_bool()
                .or_else(|| value["tie_word_embeddings"].as_bool())
                .unwrap_or(false),
        };
        for dims in [
            [cfg.heads, cfg.head_dim],
            [cfg.key_heads, cfg.key_dim],
            [cfg.value_heads, cfg.value_dim],
            [cfg.hidden, cfg.intermediate],
        ] {
            dims[0]
                .checked_mul(dims[1])
                .ok_or("d3 dimensions overflow")?;
        }
        cfg.heads
            .checked_mul(cfg.head_dim)
            .and_then(|n| n.checked_mul(2))
            .ok_or("d3 query dimensions overflow")?;
        cfg.kv_heads
            .checked_mul(cfg.head_dim)
            .ok_or("d3 key dimensions overflow")?;
        let conv_width = cfg
            .key_heads
            .checked_mul(cfg.key_dim)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| {
                cfg.value_heads
                    .checked_mul(cfg.value_dim)
                    .and_then(|v| n.checked_add(v))
            })
            .ok_or("d3 convolution dimensions overflow")?;
        conv_width
            .checked_mul(cfg.conv)
            .ok_or("d3 convolution kernel dimensions overflow")?;
        cfg.value_heads
            .checked_mul(cfg.key_dim)
            .and_then(|n| n.checked_mul(cfg.value_dim))
            .ok_or("d3 recurrent state dimensions overflow")?;
        Ok(cfg)
    }
}

struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    mixer: Mixer,
}

enum Mixer {
    Full {
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    },
    Delta {
        conv: Vec<f32>,
        a_log: Vec<f32>,
        dt_bias: Vec<f32>,
        norm: Vec<f32>,
    },
}

struct Adapter {
    source: SafetensorSource,
    modules: HashSet<String>,
    scale: f32,
}

/// Mmaps the frozen checkpoint. Each request selects one adapter without mutating the model.
pub(crate) struct D3Encoder {
    source: SafetensorSource,
    readout: SafetensorSource,
    adapters: HashMap<String, Adapter>,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    config: Config,
    pool: Arc<ComputePool>,
    bf16: bool,
    prefix_bidir: bool,
    gated_noncausal: bool,
    temperature: f32,
    max_length: usize,
    generation_head: Option<SafetensorSource>,
    vision_config: Value,
    image_token: Option<u32>,
}

fn json_file(path: &Path) -> Result<Value, String> {
    serde_json::from_reader(File::open(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn require_shape(source: &dyn TensorSource, name: &str, expected: &[usize]) -> Result<(), String> {
    let actual = tensor::shape(source, name)?;
    if actual != expected {
        return Err(format!(
            "d3 {name} has shape {actual:?}, expected {expected:?}"
        ));
    }
    Ok(())
}

fn norm_weight(
    source: &dyn TensorSource,
    name: &str,
    width: usize,
    zero_centered: bool,
) -> Result<Vec<f32>, String> {
    require_shape(source, name, &[width])?;
    let mut weight = tensor::values(source, name)?;
    if !weight.iter().all(|v| v.is_finite()) {
        return Err(format!("d3 {name} contains nonfinite values"));
    }
    if zero_centered {
        for v in &mut weight {
            *v += 1.;
        }
    }
    Ok(weight)
}

fn safetensors_header(path: &Path) -> Result<Value, String> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut len = [0; 8];
    file.read_exact(&mut len)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let len = usize::try_from(u64::from_le_bytes(len)).map_err(|_| "adapter header overflow")?;
    if len > 100 * 1024 * 1024 {
        return Err("adapter safetensors header too large".into());
    }
    let mut bytes = vec![0; len];
    file.read_exact(&mut bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

impl Adapter {
    fn load(path: &Path, base: &dyn TensorSource, declaration: &Value) -> Result<Self, String> {
        let header = safetensors_header(path)?;
        let table = header
            .as_object()
            .ok_or("adapter header must be an object")?;
        let meta = &header["__metadata__"];
        let rank = meta["r"]
            .as_str()
            .ok_or("adapter metadata has no r")?
            .parse::<usize>()
            .map_err(|_| "invalid adapter rank")?;
        let alpha = meta["alpha"]
            .as_str()
            .ok_or("adapter metadata has no alpha")?
            .parse::<usize>()
            .map_err(|_| "invalid adapter alpha")?;
        if rank == 0 || alpha == 0 {
            return Err("invalid adapter scaling".into());
        }
        let source = SafetensorSource::open(&[path])?;
        let mut modules = HashSet::new();
        for name in table.keys().filter(|n| n.as_str() != "__metadata__") {
            let module = name
                .strip_suffix(".A")
                .or_else(|| name.strip_suffix(".B"))
                .ok_or_else(|| format!("invalid adapter tensor {name}"))?;
            modules.insert(module.to_owned());
        }
        if source.len() != 2 * modules.len() || modules.is_empty() {
            return Err(format!(
                "{} has unpaired or no LoRA tensors",
                path.display()
            ));
        }
        for module in &modules {
            let base_name = format!("language_model.{module}.weight");
            let base_shape = tensor::shape(base, &base_name)?;
            if base_shape.len() != 2 {
                return Err(format!("adapter target {module} is not Linear"));
            }
            require_shape(&source, &format!("{module}.A"), &[rank, base_shape[1]])?;
            require_shape(&source, &format!("{module}.B"), &[base_shape[0], rank])?;
        }
        for (field, actual) in [("r", rank), ("alpha", alpha), ("modules", modules.len())] {
            if declaration[field].as_f64() != Some(actual as f64) {
                return Err(format!(
                    "{} adapter {field} declaration {} does not match checkpoint value {actual}",
                    path.display(),
                    declaration[field]
                ));
            }
        }
        Ok(Self {
            source,
            modules,
            scale: alpha as f32 / rank as f32,
        })
    }
}

impl D3Encoder {
    pub(crate) fn load(
        root: &Path,
        base_root: &Path,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let model_config = json_file(&base_root.join("config.json"))?;
        let config = Config::parse(&model_config)?;
        let decision = json_file(&base_root.join("decision_config.json"))?;
        if decision["format_version"] != 1
            || decision["pooling"] != "last"
            || decision["readout_dtype"] != "float32"
        {
            return Err("d3 requires code-readout v1, last pooling, and FP32 readout".into());
        }
        let prefix_bidir = match decision["attention_mode"].as_str() {
            Some("noncausal_full_attention") => false,
            Some("prefix_bidir") => true,
            other => return Err(format!("unsupported d3 decision attention_mode {other:?}")),
        };
        let temperature = decision["temperature"].as_f64().unwrap_or(1.) as f32;
        if !temperature.is_finite() || temperature <= 0. {
            return Err("invalid d3 temperature".into());
        }
        let max_length = decision["max_length"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
            .unwrap_or(config.context)
            .min(config.context);
        let source = tensor::open_weights(base_root)?;
        require_shape(
            &source,
            "language_model.embed_tokens.weight",
            &[config.vocab, config.hidden],
        )?;
        let bf16 = match source
            .tensor_info("language_model.embed_tokens.weight")
            .unwrap()
            .ggml_type
        {
            GGMLType::BF16 => true,
            GGMLType::F32 => false,
            other => return Err(format!("unsupported d3 activation dtype {other:?}")),
        };
        let mut layers = Vec::with_capacity(config.layers.len());
        let matrix = |name: &str, output: usize, input: usize| {
            require_shape(&source, name, &[output, input])
        };
        for (i, &full) in config.layers.iter().enumerate() {
            let p = format!("language_model.layers.{i}");
            let input_norm = norm_weight(
                &source,
                &format!("{p}.input_layernorm.weight"),
                config.hidden,
                true,
            )?;
            let post_norm = norm_weight(
                &source,
                &format!("{p}.post_attention_layernorm.weight"),
                config.hidden,
                true,
            )?;
            matrix(
                &format!("{p}.mlp.gate_proj.weight"),
                config.intermediate,
                config.hidden,
            )?;
            matrix(
                &format!("{p}.mlp.up_proj.weight"),
                config.intermediate,
                config.hidden,
            )?;
            matrix(
                &format!("{p}.mlp.down_proj.weight"),
                config.hidden,
                config.intermediate,
            )?;
            let mixer = if full {
                let qw = config.heads * config.head_dim;
                let kw = config.kv_heads * config.head_dim;
                matrix(
                    &format!("{p}.self_attn.q_proj.weight"),
                    2 * qw,
                    config.hidden,
                )?;
                matrix(&format!("{p}.self_attn.k_proj.weight"), kw, config.hidden)?;
                matrix(&format!("{p}.self_attn.v_proj.weight"), kw, config.hidden)?;
                matrix(&format!("{p}.self_attn.o_proj.weight"), config.hidden, qw)?;
                Mixer::Full {
                    q_norm: norm_weight(
                        &source,
                        &format!("{p}.self_attn.q_norm.weight"),
                        config.head_dim,
                        true,
                    )?,
                    k_norm: norm_weight(
                        &source,
                        &format!("{p}.self_attn.k_norm.weight"),
                        config.head_dim,
                        true,
                    )?,
                }
            } else {
                let kw = config.key_heads * config.key_dim;
                let vw = config.value_heads * config.value_dim;
                let cw = 2 * kw + vw;
                matrix(
                    &format!("{p}.linear_attn.in_proj_qkv.weight"),
                    cw,
                    config.hidden,
                )?;
                matrix(
                    &format!("{p}.linear_attn.in_proj_z.weight"),
                    vw,
                    config.hidden,
                )?;
                for projection in ["in_proj_a", "in_proj_b"] {
                    matrix(
                        &format!("{p}.linear_attn.{projection}.weight"),
                        config.value_heads,
                        config.hidden,
                    )?;
                }
                matrix(
                    &format!("{p}.linear_attn.out_proj.weight"),
                    config.hidden,
                    vw,
                )?;
                require_shape(
                    &source,
                    &format!("{p}.linear_attn.conv1d.weight"),
                    &[cw, 1, config.conv],
                )?;
                Mixer::Delta {
                    conv: tensor::values(&source, &format!("{p}.linear_attn.conv1d.weight"))?,
                    a_log: norm_weight(
                        &source,
                        &format!("{p}.linear_attn.A_log"),
                        config.value_heads,
                        false,
                    )?,
                    dt_bias: norm_weight(
                        &source,
                        &format!("{p}.linear_attn.dt_bias"),
                        config.value_heads,
                        false,
                    )?,
                    norm: norm_weight(
                        &source,
                        &format!("{p}.linear_attn.norm.weight"),
                        config.value_dim,
                        false,
                    )?,
                }
            };
            layers.push(Layer {
                input_norm,
                post_norm,
                mixer,
            });
        }
        let final_norm = norm_weight(&source, "language_model.norm.weight", config.hidden, true)?;
        let readout = SafetensorSource::open(&[base_root.join("readout.safetensors")])?;
        require_shape(&readout, "weight", &[255, config.hidden])?;
        let vela = json_file(&root.join("vela3_config.json"))?;
        let gated_noncausal = match vela["decisions"]["attention"].as_str() {
            None | Some("prefix_bidir") => false,
            Some("noncausal") => true,
            Some(other) => return Err(format!("unsupported Vela decision attention {other}")),
        };
        let mut adapters = HashMap::new();
        for (name, info) in vela["adapters"]
            .as_object()
            .ok_or("Vela config has no adapters")?
        {
            let file = info["file"].as_str().ok_or("Vela adapter has no file")?;
            let path = Path::new(file);
            if path.components().count() != 1 {
                return Err("Vela adapter filename must be local".into());
            }
            adapters.insert(
                name.clone(),
                Adapter::load(&root.join(file), &source, info)?,
            );
        }
        let generation_head = if vela["definitions"]["writer"] == "grafted_head" {
            let file = vela["definitions"]["lm_head_file"]
                .as_str()
                .ok_or("missing grafted lm_head_file")?;
            if Path::new(file).components().count() != 1 {
                return Err("grafted head filename must be local".into());
            }
            let path = root.join(file);
            if path.is_file() {
                let head = SafetensorSource::open(&[path])?;
                require_shape(&head, "lm_head.weight", &[config.vocab, config.hidden])?;
                Some(head)
            } else {
                None
            }
        } else {
            None
        };
        Ok(Self {
            source,
            readout,
            adapters,
            layers,
            final_norm,
            config,
            pool,
            bf16,
            prefix_bidir,
            gated_noncausal,
            temperature,
            max_length,
            generation_head,
            vision_config: model_config["vision_config"].clone(),
            image_token: model_config["image_token_id"]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok()),
        })
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.config.hidden
    }

    pub(crate) fn uses_bf16(&self) -> bool {
        self.bf16
    }

    pub(crate) fn prepare_images(&self, images: &[Value]) -> Result<D3Images, String> {
        vision::prepare(
            &self.source,
            &self.vision_config,
            Arc::clone(&self.pool),
            images,
        )
    }

    /// `ids` contains one unexpanded image placeholder per image, in request order.
    pub(crate) fn image_input(
        &self,
        ids: &[u32],
        images: &D3Images,
    ) -> Result<D3ImageInput, String> {
        vision::expand(
            self,
            ids,
            images,
            self.image_token.ok_or("d3 config has no image_token_id")?,
        )
    }

    pub(crate) fn hidden_image(
        &self,
        input: &D3ImageInput,
        adapter: Option<&str>,
    ) -> Result<Vec<f32>, String> {
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::token_ids(
            "vela3.d3.tokens",
            &input.ids,
        ));
        self.hidden_embeddings(&input.embeddings, &input.positions, adapter, None)
    }

    /// Official Vela image decisions use the frozen base with adapters off.
    pub(crate) fn decide_image_logits(
        &self,
        input: &D3ImageInput,
        option_count: usize,
        state_tokens: Option<usize>,
    ) -> Result<Vec<f32>, String> {
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::token_ids(
            "vela3.d3.tokens",
            &input.ids,
        ));
        let state = if self.prefix_bidir {
            Some(
                state_tokens
                    .ok_or("prefix_bidir image decision needs its expanded state token count")?,
            )
        } else {
            None
        };
        let hidden = self.hidden_embeddings(&input.embeddings, &input.positions, None, state)?;
        self.readout_logits(&hidden, option_count)
    }

    fn cast(&self, x: &mut [f32]) {
        if self.bf16 {
            for value in x {
                *value = bf16::from_f32(*value).to_f32();
            }
        }
    }

    fn rounded(&self, value: f32) -> f32 {
        if self.bf16 {
            bf16::from_f32(value).to_f32()
        } else {
            value
        }
    }

    fn norm(&self, x: &[f32], weight: &[f32]) -> Vec<f32> {
        let mut out = vec![0.; x.len()];
        for (input, output) in x
            .chunks_exact(weight.len())
            .zip(out.chunks_exact_mut(weight.len()))
        {
            rms_norm(input, weight, output, self.config.eps);
        }
        self.cast(&mut out);
        out
    }

    fn project(
        &self,
        module: &str,
        x: &[f32],
        rows: usize,
        adapter: Option<&Adapter>,
    ) -> Result<Vec<f32>, String> {
        let mut y = tensor::linear(
            &self.source,
            &format!("language_model.{module}.weight"),
            x,
            rows,
            &self.pool,
        )?;
        self.cast(&mut y);
        if let Some(a) = adapter.filter(|a| a.modules.contains(module)) {
            let low = tensor::linear(&a.source, &format!("{module}.A"), x, rows, &self.pool)?;
            let delta = tensor::linear(&a.source, &format!("{module}.B"), &low, rows, &self.pool)?;
            for (y, delta) in y.iter_mut().zip(delta) {
                *y = self.rounded(*y + self.rounded(delta * a.scale));
            }
        }
        Ok(y)
    }

    /// Final-normalized states of one unpadded span prompt, including future-token full attention.
    pub(crate) fn hidden(&self, ids: &[u32], adapter: Option<&str>) -> Result<Vec<f32>, String> {
        self.hidden_mode(ids, adapter, Mask::Full)
    }

    fn hidden_mode(
        &self,
        ids: &[u32],
        adapter: Option<&str>,
        mask: Mask,
    ) -> Result<Vec<f32>, String> {
        self.validate_length(ids.len())?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::token_ids("vela3.d3.tokens", ids));
        let x = tensor::embed(&self.source, "language_model.embed_tokens.weight", ids)?;
        let positions: Vec<_> = (0..ids.len()).map(|i| [i; 3]).collect();
        self.forward(x, &positions, adapter, mask)
    }

    fn validate_length(&self, length: usize) -> Result<(), String> {
        if length == 0 || length > self.max_length {
            return Err(format!(
                "d3 prompt length {length} is outside 1..={}",
                self.max_length
            ));
        }
        length
            .checked_mul(self.config.hidden)
            .ok_or("d3 prompt size overflow")?;
        let c = &self.config;
        for width in [
            2 * c.heads * c.head_dim,
            c.intermediate,
            2 * c.key_heads * c.key_dim + c.value_heads * c.value_dim,
        ] {
            length
                .checked_mul(width)
                .ok_or("d3 activation size overflow")?;
        }
        Ok(())
    }

    /// Projected image tokens can replace the matching placeholder embeddings before this call.
    pub(crate) fn hidden_embeddings(
        &self,
        embeddings: &[f32],
        positions: &[[usize; 3]],
        adapter: Option<&str>,
        state_tokens: Option<usize>,
    ) -> Result<Vec<f32>, String> {
        self.validate_length(positions.len())?;
        if embeddings.len() != positions.len() * self.config.hidden {
            return Err("d3 embedding/position shape mismatch".into());
        }
        #[cfg(feature = "parity-trace")]
        trace(
            "vela3.d3.positions",
            None,
            &[positions.len(), 3],
            &positions
                .iter()
                .flatten()
                .map(|&p| p as f32)
                .collect::<Vec<_>>(),
        );
        let mask = state_tokens.map_or(Mask::Full, Mask::Prefix);
        self.forward(embeddings.to_vec(), positions, adapter, mask)
    }

    fn forward(
        &self,
        mut x: Vec<f32>,
        positions: &[[usize; 3]],
        adapter: Option<&str>,
        mask: Mask,
    ) -> Result<Vec<f32>, String> {
        let adapter = adapter
            .map(|name| {
                self.adapters
                    .get(name)
                    .ok_or_else(|| format!("unknown Vela adapter {name}"))
            })
            .transpose()?;
        let rows = positions.len();
        trace("vela3.d3.embedding", None, &[rows, self.config.hidden], &x);
        for (index, layer) in self.layers.iter().enumerate() {
            let p = format!("layers.{index}");
            let input = self.norm(&x, &layer.input_norm);
            trace(
                "vela3.d3.input_norm",
                Some(index),
                &[rows, self.config.hidden],
                &input,
            );
            let delta = match &layer.mixer {
                Mixer::Full { q_norm, k_norm } => {
                    self.full_attention(&p, &input, positions, q_norm, k_norm, adapter, mask)?
                }
                Mixer::Delta {
                    conv,
                    a_log,
                    dt_bias,
                    norm,
                } => self.delta_attention(&p, &input, rows, conv, a_log, dt_bias, norm, adapter)?,
            };
            for (value, delta) in x.iter_mut().zip(delta) {
                *value = self.rounded(*value + delta);
            }
            trace(
                "vela3.d3.attention_residual",
                Some(index),
                &[rows, self.config.hidden],
                &x,
            );
            let input = self.norm(&x, &layer.post_norm);
            let mut gate = self.project(&format!("{p}.mlp.gate_proj"), &input, rows, adapter)?;
            let up = self.project(&format!("{p}.mlp.up_proj"), &input, rows, adapter)?;
            for (gate, up) in gate.iter_mut().zip(up) {
                *gate = self.rounded(self.rounded(silu(*gate)) * up);
            }
            let delta = self.project(&format!("{p}.mlp.down_proj"), &gate, rows, adapter)?;
            for (value, delta) in x.iter_mut().zip(delta) {
                *value = self.rounded(*value + delta);
            }
            trace(
                "vela3.d3.layer",
                Some(index),
                &[rows, self.config.hidden],
                &x,
            );
        }
        let hidden = self.norm(&x, &self.final_norm);
        trace(
            "vela3.d3.hidden",
            None,
            &[rows, self.config.hidden],
            &hidden,
        );
        if !hidden.iter().all(|v| v.is_finite()) {
            return Err("d3 produced nonfinite hidden states".into());
        }
        Ok(hidden)
    }

    fn rotary(&self, x: &mut [f32], positions: &[[usize; 3]], heads: usize) {
        let c = &self.config;
        let pairs = c.rope_dim / 2;
        let mut cos = vec![0.; pairs];
        let mut sin = vec![0.; pairs];
        for (row, position) in positions.iter().enumerate() {
            for i in 0..pairs {
                let axis = if i % 3 == 1 && i < 3 * c.rope_sections[1] {
                    1
                } else if i % 3 == 2 && i < 3 * c.rope_sections[2] {
                    2
                } else {
                    0
                };
                let angle =
                    position[axis] as f32 / c.rope_base.powf((2 * i) as f32 / c.rope_dim as f32);
                cos[i] = self.rounded(angle.cos());
                sin[i] = self.rounded(angle.sin());
            }
            for head in 0..heads {
                let start = (row * heads + head) * c.head_dim;
                let vector = &mut x[start..start + c.rope_dim];
                if self.bf16 {
                    crate::ops::rope_neox_inplace_with_table(vector, c.rope_dim, &cos, &sin);
                } else {
                    for i in 0..pairs {
                        let (a, b) = (vector[i], vector[i + pairs]);
                        vector[i] = a * cos[i] - b * sin[i];
                        vector[i + pairs] = a * sin[i] + b * cos[i];
                    }
                }
            }
        }
    }

    fn full_attention(
        &self,
        p: &str,
        x: &[f32],
        positions: &[[usize; 3]],
        q_norm: &[f32],
        k_norm: &[f32],
        adapter: Option<&Adapter>,
        mask: Mask,
    ) -> Result<Vec<f32>, String> {
        let c = &self.config;
        let rows = positions.len();
        let projected = self.project(&format!("{p}.self_attn.q_proj"), x, rows, adapter)?;
        let mut q = vec![0.; rows * c.heads * c.head_dim];
        let mut gate = vec![0.; q.len()];
        for (i, head) in projected.chunks_exact(2 * c.head_dim).enumerate() {
            q[i * c.head_dim..(i + 1) * c.head_dim].copy_from_slice(&head[..c.head_dim]);
            gate[i * c.head_dim..(i + 1) * c.head_dim].copy_from_slice(&head[c.head_dim..]);
        }
        q = self.norm(&q, q_norm);
        let mut k = self.norm(
            &self.project(&format!("{p}.self_attn.k_proj"), x, rows, adapter)?,
            k_norm,
        );
        let v = self.project(&format!("{p}.self_attn.v_proj"), x, rows, adapter)?;
        self.rotary(&mut q, positions, c.heads);
        self.rotary(&mut k, positions, c.kv_heads);
        let mut out = attention(&q, &k, &v, rows, c.heads, c.kv_heads, c.head_dim, mask)?;
        self.cast(&mut out);
        for (out, gate) in out.iter_mut().zip(gate) {
            *out = self.rounded(*out * self.rounded(1. / (1. + (-gate).exp())));
        }
        self.project(&format!("{p}.self_attn.o_proj"), &out, rows, adapter)
    }

    #[allow(clippy::too_many_arguments)]
    fn delta_attention(
        &self,
        p: &str,
        x: &[f32],
        rows: usize,
        conv: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        norm: &[f32],
        adapter: Option<&Adapter>,
    ) -> Result<Vec<f32>, String> {
        let c = &self.config;
        let kw = c.key_heads * c.key_dim;
        let vw = c.value_heads * c.value_dim;
        let cw = 2 * kw + vw;
        let mixed = self.project(&format!("{p}.linear_attn.in_proj_qkv"), x, rows, adapter)?;
        let z = self.project(&format!("{p}.linear_attn.in_proj_z"), x, rows, adapter)?;
        let a = self.project(&format!("{p}.linear_attn.in_proj_a"), x, rows, adapter)?;
        let mut beta = self.project(&format!("{p}.linear_attn.in_proj_b"), x, rows, adapter)?;
        let mut decay = vec![0.; rows * c.value_heads];
        for (i, (&a, beta)) in a.iter().zip(beta.iter_mut()).enumerate() {
            let h = i % c.value_heads;
            let biased = a + dt_bias[h];
            let softplus = if biased > 20. {
                biased
            } else {
                biased.exp().ln_1p()
            };
            decay[i] = -a_log[h].exp() * softplus;
            *beta = self.rounded(1. / (1. + (-*beta).exp()));
        }
        let mut mixed_conv = vec![0.; mixed.len()];
        for row in 0..rows {
            for channel in 0..cw {
                let mut sum = 0.;
                for j in 0..c.conv.min(row + 1) {
                    sum +=
                        mixed[(row - j) * cw + channel] * conv[channel * c.conv + c.conv - 1 - j];
                }
                mixed_conv[row * cw + channel] = self.rounded(silu(self.rounded(sum)));
            }
        }
        let mut q = Vec::with_capacity(rows * kw);
        let mut k = Vec::with_capacity(rows * kw);
        let mut v = Vec::with_capacity(rows * vw);
        for row in mixed_conv.chunks_exact(cw) {
            q.extend_from_slice(&row[..kw]);
            k.extend_from_slice(&row[kw..2 * kw]);
            v.extend_from_slice(&row[2 * kw..]);
        }
        let mut core = delta_scan(
            &q,
            &k,
            &v,
            &decay,
            &beta,
            rows,
            c.key_heads,
            c.value_heads,
            c.key_dim,
            c.value_dim,
        )?;
        self.cast(&mut core);
        let mut out = vec![0.; core.len()];
        for ((core, z), out) in core
            .chunks_exact(c.value_dim)
            .zip(z.chunks_exact(c.value_dim))
            .zip(out.chunks_exact_mut(c.value_dim))
        {
            let inv = (dot_f32(core, core, core.len()) / c.value_dim as f32 + c.eps)
                .sqrt()
                .recip();
            for i in 0..c.value_dim {
                let normalized = self.rounded(norm[i] * self.rounded(core[i] * inv));
                out[i] = self.rounded(normalized * silu(z[i]));
            }
        }
        self.project(&format!("{p}.linear_attn.out_proj"), &out, rows, adapter)
    }

    /// FP32 last-token code logits, already divided by the checkpoint temperature.
    #[cfg(test)]
    pub(crate) fn decide_logits(
        &self,
        ids: &[u32],
        adapter: Option<&str>,
        option_count: usize,
    ) -> Result<Vec<f32>, String> {
        self.decide_logits_with_state(ids, adapter, option_count, None)
    }

    pub(crate) fn decide_logits_with_state(
        &self,
        ids: &[u32],
        adapter: Option<&str>,
        option_count: usize,
        state_tokens: Option<usize>,
    ) -> Result<Vec<f32>, String> {
        if !(1..=255).contains(&option_count) {
            return Err("d3 decision requires 1..=255 options".into());
        }
        // Turbo first answers with the base prefix mask, then replaces gated router rows using Vela's mask.
        let mask = if self.prefix_bidir && !(adapter.is_some() && self.gated_noncausal) {
            Mask::Prefix(state_tokens.ok_or("prefix_bidir decision needs its state token count")?)
        } else {
            Mask::Full
        };
        let hidden = self.hidden_mode(ids, adapter, mask)?;
        self.readout_logits(&hidden, option_count)
    }

    fn readout_logits(&self, hidden: &[f32], option_count: usize) -> Result<Vec<f32>, String> {
        if !(1..=255).contains(&option_count) {
            return Err("d3 decision requires 1..=255 options".into());
        }
        let last = &hidden[hidden.len() - self.config.hidden..];
        let mut logits = tensor::linear(&self.readout, "weight", last, 1, &self.pool)?;
        logits.truncate(option_count);
        for value in &mut logits {
            *value /= self.temperature;
        }
        if !logits.iter().all(|v| v.is_finite()) {
            return Err("d3 produced nonfinite decision logits".into());
        }
        trace("vela3.d3.logits", None, &[option_count], &logits);
        Ok(logits)
    }

    pub(crate) fn generation_logits(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let hidden = self.hidden_mode(ids, None, Mask::Causal)?;
        let last = &hidden[hidden.len() - self.config.hidden..];
        let mut logits = if let Some(head) = &self.generation_head {
            tensor::linear(head, "lm_head.weight", last, 1, &self.pool)
        } else if self.config.tied {
            tensor::linear(
                &self.source,
                "language_model.embed_tokens.weight",
                last,
                1,
                &self.pool,
            )
        } else {
            Err("d3 definition generation requires tied embeddings or the configured grafted LM head".into())
        }?;
        self.cast(&mut logits);
        if !logits.iter().all(|v| v.is_finite()) {
            return Err("nonfinite d3 generation logits".into());
        }
        trace(
            "vela3.d3.generation_logits",
            None,
            &[self.config.vocab],
            &logits,
        );
        Ok(logits)
    }

    pub(crate) fn generate_definition_ids(
        &self,
        ids: &[u32],
        max_new: usize,
    ) -> Result<Vec<u32>, String> {
        if max_new > 40 {
            return Err("Vela definition generation is limited to 40 new tokens".into());
        }
        let mut prompt = ids.to_vec();
        let mut output = Vec::new();
        for _ in 0..max_new {
            let logits = self.generation_logits(&prompt)?;
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .ok_or("empty definition logits")?
                .0 as u32;
            if Some(next) == self.config.eos {
                break;
            }
            output.push(next);
            prompt.push(next);
        }
        Ok(output)
    }
}

#[allow(clippy::too_many_arguments)]
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    mask: Mask,
) -> Result<Vec<f32>, String> {
    if heads == 0
        || kv_heads == 0
        || dim == 0
        || heads % kv_heads != 0
        || q.len() != rows * heads * dim
        || k.len() != rows * kv_heads * dim
        || k.len() != v.len()
    {
        return Err("invalid d3 attention shapes".into());
    }
    if let Mask::Prefix(end) = mask {
        if end == 0 || end >= rows {
            return Err("invalid d3 state block boundary".into());
        }
    }
    let mut out = vec![0.; q.len()];
    let scale = (dim as f32).sqrt().recip();
    let mut scores = vec![0.; rows];
    for row in 0..rows {
        let count = match mask {
            Mask::Causal => row + 1,
            Mask::Prefix(end) if row < end => end,
            _ => rows,
        };
        for head in 0..heads {
            let q_start = (row * heads + head) * dim;
            let kv_head = head / (heads / kv_heads);
            for (key, score) in scores[..count].iter_mut().enumerate() {
                let k_start = (key * kv_heads + kv_head) * dim;
                *score =
                    dot_f32(&q[q_start..q_start + dim], &k[k_start..k_start + dim], dim) * scale;
            }
            softmax_inplace(&mut scores[..count]);
            for (key, &prob) in scores[..count].iter().enumerate() {
                let v_start = (key * kv_heads + kv_head) * dim;
                vec_mad_f32(
                    &mut out[q_start..q_start + dim],
                    &v[v_start..v_start + dim],
                    prob,
                );
            }
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn delta_scan(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: &[f32],
    beta: &[f32],
    rows: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Result<Vec<f32>, String> {
    if key_heads == 0
        || value_heads == 0
        || value_heads % key_heads != 0
        || key_dim == 0
        || value_dim == 0
        || q.len() != rows * key_heads * key_dim
        || k.len() != q.len()
        || v.len() != rows * value_heads * value_dim
        || decay.len() != rows * value_heads
        || beta.len() != decay.len()
    {
        return Err("invalid d3 Gated DeltaNet shapes".into());
    }
    let mut out = vec![0.; v.len()];
    let mut state = vec![0.; value_heads * key_dim * value_dim];
    let mut query = vec![0.; key_dim];
    let mut key = vec![0.; key_dim];
    let mut delta = vec![0.; value_dim];
    let query_scale = (key_dim as f32).sqrt().recip();
    for row in 0..rows {
        for head in 0..value_heads {
            let kh = head / (value_heads / key_heads);
            let qk_start = (row * key_heads + kh) * key_dim;
            let qs = &q[qk_start..qk_start + key_dim];
            let ks = &k[qk_start..qk_start + key_dim];
            let qi = (dot_f32(qs, qs, key_dim) + 1e-6).sqrt().recip() * query_scale;
            let ki = (dot_f32(ks, ks, key_dim) + 1e-6).sqrt().recip();
            for i in 0..key_dim {
                query[i] = qs[i] * qi;
                key[i] = ks[i] * ki;
            }
            let st = &mut state[head * value_dim * key_dim..(head + 1) * value_dim * key_dim];
            ssm_state_decay(st, decay[row * value_heads + head].exp());
            ssm_matvec(st, &key, key_dim, value_dim, &mut delta);
            let start = (row * value_heads + head) * value_dim;
            for d in 0..value_dim {
                delta[d] = (v[start + d] - delta[d]) * beta[row * value_heads + head];
                vec_mad_f32(&mut st[d * key_dim..(d + 1) * key_dim], &key, delta[d]);
            }
            ssm_matvec(
                st,
                &query,
                key_dim,
                value_dim,
                &mut out[start..start + value_dim],
            );
        }
    }
    Ok(out)
}

fn trace(name: &str, layer: Option<usize>, shape: &[usize], values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(name, layer, shape, values));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, layer, shape, values);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write_weights(path: &Path, specs: &[(String, Vec<usize>)], adapter: bool) {
        let mut table = serde_json::Map::new();
        if adapter {
            table.insert("__metadata__".into(), json!({"r":"2","alpha":"3"}));
        }
        let mut bytes = Vec::new();
        for (name, shape) in specs {
            let start = bytes.len();
            let salt = name.bytes().map(usize::from).sum::<usize>();
            for i in 0..shape.iter().product::<usize>() {
                let value = (((i + salt) % 17) as f32 - 8.) * 0.025;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            table.insert(
                name.clone(),
                json!({"dtype":"F32","shape":shape,"data_offsets":[start,bytes.len()]}),
            );
        }
        let header = serde_json::to_vec(&table).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(bytes);
        std::fs::write(path, file).unwrap();
    }

    pub(super) struct Tiny(pub(super) std::path::PathBuf);
    impl Drop for Tiny {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    pub(super) fn tiny_fixture() -> Tiny {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "vela3-d3-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        std::fs::create_dir(&root).unwrap();
        let config = json!({"model_type":"qwen3_5", "text_config": {
            "hidden_size":4,"intermediate_size":8,"num_hidden_layers":2,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,
            "linear_num_key_heads":1,"linear_num_value_heads":2,
            "linear_key_head_dim":2,"linear_value_head_dim":2,
            "linear_conv_kernel_dim":4,"rms_norm_eps":1e-6,"vocab_size":8,
            "max_position_embeddings":32,"layer_types":["linear_attention","full_attention"],
            "tie_word_embeddings":true,"eos_token_id":7,
            "rope_parameters":{"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":1.,"mrope_section":[1,0,0]}
        }});
        std::fs::write(root.join("config.json"), config.to_string()).unwrap();
        std::fs::write(root.join("decision_config.json"), json!({"format_version":1,
            "pooling":"last","readout_dtype":"float32","attention_mode":"noncausal_full_attention","temperature":1.}).to_string()).unwrap();
        std::fs::write(
            root.join("vela3_config.json"),
            json!({"adapters":{"span":{"file":"span_lora.safetensors","r":2,"alpha":3,"modules":3}}}).to_string(),
        )
        .unwrap();
        let mut specs = vec![
            ("language_model.embed_tokens.weight".into(), vec![8, 4]),
            ("language_model.norm.weight".into(), vec![4]),
        ];
        for i in 0..2 {
            let p = format!("language_model.layers.{i}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                specs.push((format!("{p}.{name}.weight"), vec![4]));
            }
            for name in ["gate_proj", "up_proj"] {
                specs.push((format!("{p}.mlp.{name}.weight"), vec![8, 4]));
            }
            specs.push((format!("{p}.mlp.down_proj.weight"), vec![4, 8]));
            if i == 0 {
                for (name, dims) in [
                    ("A_log", vec![2]),
                    ("dt_bias", vec![2]),
                    ("conv1d.weight", vec![8, 1, 4]),
                    ("in_proj_qkv.weight", vec![8, 4]),
                    ("in_proj_z.weight", vec![4, 4]),
                    ("in_proj_a.weight", vec![2, 4]),
                    ("in_proj_b.weight", vec![2, 4]),
                    ("norm.weight", vec![2]),
                    ("out_proj.weight", vec![4, 4]),
                ] {
                    specs.push((format!("{p}.linear_attn.{name}"), dims));
                }
            } else {
                for (name, dims) in [
                    ("q_proj", vec![8, 4]),
                    ("k_proj", vec![2, 4]),
                    ("v_proj", vec![2, 4]),
                    ("o_proj", vec![4, 4]),
                    ("q_norm", vec![2]),
                    ("k_norm", vec![2]),
                ] {
                    specs.push((format!("{p}.self_attn.{name}.weight"), dims));
                }
            }
        }
        write_weights(&root.join("model.safetensors"), &specs, false);
        write_weights(
            &root.join("readout.safetensors"),
            &[("weight".into(), vec![255, 4])],
            false,
        );
        let mut adapters = Vec::new();
        for (module, output, input) in [
            ("layers.0.linear_attn.in_proj_qkv", 8, 4),
            ("layers.1.self_attn.q_proj", 8, 4),
            ("layers.1.mlp.down_proj", 4, 8),
        ] {
            adapters.push((format!("{module}.A"), vec![2, input]));
            adapters.push((format!("{module}.B"), vec![output, 2]));
        }
        write_weights(&root.join("span_lora.safetensors"), &adapters, true);
        Tiny(root)
    }

    #[test]
    fn tiny_graph_and_gated_lora_match_transformers_5_19() {
        let fixture = tiny_fixture();
        let encoder =
            D3Encoder::load(&fixture.0, &fixture.0, Arc::new(ComputePool::new(1))).unwrap();
        let expected = [
            -1.70971167,
            -0.83400577,
            -0.65063226,
            -0.51642776,
            -1.51642442,
            -1.30944121,
            -0.53001869,
            0.12353250,
            1.18386185,
            0.94179869,
            0.75706154,
            1.28466070,
        ];
        let actual = encoder.hidden(&[1, 2, 3], None).unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-6, "{actual} vs {expected}");
        }
        let logits = encoder.decide_logits(&[1, 2, 3], Some("span"), 3).unwrap();
        for (actual, expected) in logits.iter().zip([-0.46231505, -0.04273150, 0.37685204]) {
            assert!((actual - expected).abs() < 2e-6, "{actual} vs {expected}");
        }
        assert_eq!(
            actual,
            encoder.hidden(&[1, 2, 3], None).unwrap(),
            "an active adapter must not persist into the next request"
        );
        let generation = encoder.generation_logits(&[1, 2, 3]).unwrap();
        for (actual, expected) in generation.iter().zip([
            0.67867041,
            -0.67572898,
            -0.25899071,
            0.15774760,
            0.57448590,
            -0.27677226,
            -0.36317527,
            0.05356303,
        ]) {
            assert!((actual - expected).abs() < 2e-6, "{actual} vs {expected}");
        }
        // Four independent official causal forwards, including the growing prompt at each step.
        assert_eq!(
            encoder.generate_definition_ids(&[1, 2, 3], 4).unwrap(),
            [0, 0, 0, 0]
        );
        assert!(encoder.hidden(&[8], None).is_err());
        assert!(encoder.hidden(&[], None).is_err());
        assert!(encoder.hidden(&[1], Some("unknown")).is_err());
        assert!(encoder.decide_logits(&[1], None, 256).is_err());
        assert!(encoder.generate_definition_ids(&[1], 41).is_err());
    }

    #[test]
    fn adapter_manifest_must_match_rank_alpha_and_complete_module_pairs() {
        let fixture = tiny_fixture();
        let path = fixture.0.join("vela3_config.json");
        let valid = json_file(&path).unwrap();
        for (field, wrong) in [("r", 3), ("alpha", 4), ("modules", 4)] {
            let mut invalid = valid.clone();
            invalid["adapters"]["span"][field] = json!(wrong);
            std::fs::write(&path, invalid.to_string()).unwrap();
            let error = D3Encoder::load(&fixture.0, &fixture.0, Arc::new(ComputePool::new(1)))
                .err()
                .unwrap_or_else(|| panic!("accepted wrong adapter manifest {field}"));
            assert!(error.contains(field), "{error}");
            invalid["adapters"]["span"]
                .as_object_mut()
                .unwrap()
                .remove(field);
            std::fs::write(&path, invalid.to_string()).unwrap();
            assert!(
                D3Encoder::load(&fixture.0, &fixture.0, Arc::new(ComputePool::new(1))).is_err()
            );
        }
        std::fs::write(&path, valid.to_string()).unwrap();
        assert!(D3Encoder::load(&fixture.0, &fixture.0, Arc::new(ComputePool::new(1))).is_ok());
    }

    #[test]
    fn rejects_unpaired_lora_and_requires_prefix_boundary() {
        let fixture = tiny_fixture();
        let path = fixture.0.join("decision_config.json");
        let mut config = json_file(&path).unwrap();
        config["attention_mode"] = json!("prefix_bidir");
        std::fs::write(&path, config.to_string()).unwrap();
        let model = D3Encoder::load(&fixture.0, &fixture.0, Arc::new(ComputePool::new(1))).unwrap();
        assert!(model.decide_logits(&[1, 2, 3], None, 3).is_err());
        assert!(model
            .decide_logits_with_state(&[1, 2, 3], None, 3, Some(2))
            .is_ok());
        // Span prompts keep full noncausal attention even on a prefix_bidir decision checkpoint.
        assert!(model.hidden(&[1, 2, 3], Some("span")).is_ok());
        let path = fixture.0.join("vela3_config.json");
        let mut vela = json_file(&path).unwrap();
        vela["decisions"] = json!({"attention": "noncausal"});
        std::fs::write(path, vela.to_string()).unwrap();
        let gated = D3Encoder::load(&fixture.0, &fixture.0, Arc::new(ComputePool::new(1))).unwrap();
        assert!(gated.decide_logits(&[1, 2, 3], None, 3).is_err());
        let full = gated.hidden(&[1, 2, 3], Some("span")).unwrap();
        assert_eq!(
            gated.decide_logits(&[1, 2, 3], Some("span"), 3).unwrap(),
            gated.readout_logits(&full, 3).unwrap()
        );
        write_weights(
            &fixture.0.join("span_lora.safetensors"),
            &[("layers.1.self_attn.q_proj.A".into(), vec![2, 4])],
            true,
        );
        assert!(D3Encoder::load(&fixture.0, &fixture.0, Arc::new(ComputePool::new(1))).is_err());
    }

    #[test]
    #[ignore = "requires local d3/Vela weights and explicit parity trace input"]
    fn trace_local_d3_checkpoint() {
        let base = std::path::PathBuf::from(std::env::var("RMI_VELA3_D3_BASE").unwrap());
        let root = std::path::PathBuf::from(std::env::var("RMI_VELA3_D3_ROOT").unwrap());
        let ids: Vec<u32> =
            serde_json::from_str(&std::env::var("RMI_VELA3_D3_IDS").unwrap()).unwrap();
        let mut model = D3Encoder::load(&root, &base, Arc::new(ComputePool::new(1))).unwrap();
        // A diagnostic FP32-activation pass isolates structure from BF16 rounding on real weights.
        if std::env::var_os("RMI_VELA3_D3_F32").is_some() {
            model.bf16 = false;
        }
        let adapter = std::env::var("RMI_VELA3_D3_ADAPTER").ok();
        let state = std::env::var("RMI_VELA3_D3_STATE")
            .ok()
            .map(|v| v.parse().unwrap());
        if let Ok(images) = std::env::var("RMI_VELA3_D3_IMAGES") {
            let images: Vec<Value> = serde_json::from_str(&images).unwrap();
            let images = model.prepare_images(&images).unwrap();
            let input = model.image_input(&ids, &images).unwrap();
            let state = state.map(|s| input.original_to_expanded[s]);
            if let Some(adapter) = adapter {
                let hidden = model.hidden_image(&input, Some(&adapter)).unwrap();
                model.readout_logits(&hidden, 3).unwrap();
            } else {
                model.decide_image_logits(&input, 3, state).unwrap();
            }
        } else if std::env::var_os("RMI_VELA3_D3_CAUSAL").is_some() {
            if let Ok(steps) = std::env::var("RMI_VELA3_D3_STEPS") {
                let ids = model
                    .generate_definition_ids(&ids, steps.parse().unwrap())
                    .unwrap();
                #[cfg(feature = "parity-trace")]
                crate::parity_trace::report(crate::parity_trace::token_ids(
                    "vela3.d3.generated_tokens",
                    &ids,
                ));
            } else {
                model.generation_logits(&ids).unwrap();
            }
        } else {
            model
                .decide_logits_with_state(&ids, adapter.as_deref(), 3, state)
                .unwrap();
        }
    }

    #[test]
    fn full_attention_reads_future_and_prefix_state_does_not() {
        let q = [1., 0., 1., 0.];
        let k = [1., 0., 0., 1.];
        let v = [2., 0., 0., 4.];
        let full = attention(&q, &k, &v, 2, 1, 1, 2, Mask::Full).unwrap();
        let causal = attention(&q, &k, &v, 2, 1, 1, 2, Mask::Causal).unwrap();
        let prefix = attention(&q, &k, &v, 2, 1, 1, 2, Mask::Prefix(1)).unwrap();
        assert!(full[1] > 1., "the first query must see the future value");
        assert_eq!(&causal[..2], &[2., 0.]);
        assert_eq!(&prefix[..2], &[2., 0.]);
        assert_eq!(&prefix[2..], &full[2..]);
        assert!(attention(&q, &k, &v, 2, 1, 1, 2, Mask::Prefix(2)).is_err());
    }

    #[test]
    fn delta_recurrence_matches_independent_torch_reference_and_is_causal() {
        // torch_recurrent_gated_delta_rule, l2-normalized q/k, no cache.
        let q = [0.1, 0.2, 0.4, -0.2, -0.5, 0.3];
        let k = [0.2, -0.3, 0.1, 0.4, -0.2, 0.6];
        let v = [1., 2., -2., 0.5, 3., -1.];
        let decay = [-0.1, -0.2, -0.3];
        let beta = [0.4, 0.6, 0.8];
        let out = delta_scan(&q, &k, &v, &decay, &beta, 3, 1, 1, 2, 2).unwrap();
        let first = delta_scan(
            &q[..2],
            &k[..2],
            &v[..2],
            &decay[..1],
            &beta[..1],
            1,
            1,
            1,
            2,
            2,
        )
        .unwrap();
        assert_eq!(&out[..2], first);
        let expected = [
            -0.14032736,
            -0.28065473,
            0.36485538,
            0.31555188,
            1.35776615,
            -0.60393286,
        ];
        for (&actual, expected) in out.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-6, "{actual} vs {expected}");
        }
    }

    #[test]
    fn config_rejects_unknown_architecture_and_invalid_head_groups() {
        let mut c = json!({"model_type":"qwen3_5", "text_config": {
            "hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,
            "linear_num_key_heads":1,"linear_num_value_heads":2,
            "linear_key_head_dim":2,"linear_value_head_dim":2,
            "linear_conv_kernel_dim":4,"rms_norm_eps":1e-6,"vocab_size":8,
            "max_position_embeddings":32,"layer_types":["full_attention"],
            "rope_parameters":{"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":1.}
        }});
        assert!(Config::parse(&c).is_ok());
        c["model_type"] = json!("qwen3");
        assert!(Config::parse(&c).is_err());
        c["model_type"] = json!("qwen3_5");
        c["text_config"]["num_key_value_heads"] = json!(3);
        assert!(Config::parse(&c).is_err());
    }
}
