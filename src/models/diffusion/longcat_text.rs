//! The supplied Qwen2.5-VL-7B safetensors as a Qwen2VL text weight source.

use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
use crate::core::tokenizer::BPETokenizer;
use crate::core::tokenizer::EncodeOptions;
use crate::format::safetensors::SafetensorSource;
use crate::models::qwen3::Qwen3Model;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

const HIDDEN: u64 = 3584;
const FFN: u64 = 18944;
const KV: u64 = 512;
const VOCAB: u64 = 152064;
const LAYERS: usize = 28;
const VISION: u64 = 1280;
const VISION_FF: u64 = 3420;
const VISION_LAYERS: usize = 32;
const IMAGE_TOKEN: u32 = 151655;
const PAD_TOKEN: u32 = 151643;
const PREFIX: &str = "<|im_start|>system\nAs an image editing expert, first analyze the content and attributes of the input image(s). Then, based on the user's editing instructions, clearly and precisely determine how to modify the given image(s), ensuring that only the specified parts are altered and all other aspects remain consistent with the original(s).<|im_end|>\n<|im_start|>user\n<|vision_start|>";
const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";

struct PromptInput {
    ids: Vec<u32>,
    mask: Vec<bool>,
}

fn tokenize(tokenizer: &BPETokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(
        text,
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    )
}

fn tokenize_instruction(tokenizer: &BPETokenizer, prompt: &str) -> Vec<u32> {
    let chars: Vec<char> = prompt.chars().collect();
    let mut ids = Vec::new();
    let mut plain = String::new();
    let mut index = 0;
    while index < chars.len() {
        let close = match chars[index] {
            '\'' if !(index > 0
                && index + 1 < chars.len()
                && chars[index - 1].is_ascii_alphabetic()
                && chars[index + 1].is_ascii_alphabetic()) =>
            {
                Some('\'')
            }
            '"' => Some('"'),
            '‘' => Some('’'),
            '“' => Some('”'),
            _ => None,
        };
        if let Some(close) = close {
            if let Some(end) = chars[index + 1..]
                .iter()
                .position(|&ch| ch == close)
                .map(|offset| index + 1 + offset)
            {
                ids.extend(tokenize(tokenizer, &plain));
                plain.clear();
                for ch in &chars[index..=end] {
                    ids.extend(tokenize(tokenizer, &ch.to_string()));
                }
                index = end + 1;
                continue;
            }
        }
        plain.push(chars[index]);
        index += 1;
    }
    ids.extend(tokenize(tokenizer, &plain));
    ids
}

fn prepare_prompt(
    tokenizer: &BPETokenizer,
    prompt: &str,
    vision_tokens: usize,
) -> Result<PromptInput, String> {
    if vision_tokens == 0 {
        return Err("LongCat prompt needs image tokens".into());
    }
    let mut ids = tokenize(tokenizer, PREFIX);
    ids.extend(std::iter::repeat_n(IMAGE_TOKEN, vision_tokens));
    ids.extend(tokenize(tokenizer, "<|vision_end|>"));
    let mut instruction = tokenize_instruction(tokenizer, prompt);
    ids.append(&mut instruction);
    ids.extend(tokenize(tokenizer, SUFFIX));
    let used = ids.len();
    ids.resize(used.max(579), PAD_TOKEN);
    let mut mask = vec![true; used];
    mask.resize(ids.len(), false);
    Ok(PromptInput { ids, mask })
}

pub fn encode_prompt(
    model: &Qwen3Model,
    tokenizer: &BPETokenizer,
    prompt: &str,
    vision_embeddings: &[f32],
) -> Result<Vec<f32>, String> {
    if vision_embeddings.len() % HIDDEN as usize != 0 {
        return Err("Invalid LongCat vision embedding width".into());
    }
    let input = prepare_prompt(tokenizer, prompt, vision_embeddings.len() / HIDDEN as usize)?;
    let mut embeddings = model.embed_tokens(&input.ids)?;
    let mut image_row = 0;
    for (token, row) in input
        .ids
        .iter()
        .zip(embeddings.chunks_exact_mut(HIDDEN as usize))
    {
        if *token == IMAGE_TOKEN {
            let start = image_row * HIDDEN as usize;
            row.copy_from_slice(&vision_embeddings[start..start + HIDDEN as usize]);
            image_row += 1;
        }
    }
    let positions: Vec<_> = (0..input.ids.len()).map(|i| [i, i, i, 0]).collect();
    let hidden = model.text_encode_embeddings(embeddings, &positions, &input.mask)?;
    let begin = 67 * HIDDEN as usize;
    if hidden.len() <= begin {
        return Err("LongCat prompt is shorter than its 67-token template".into());
    }
    Ok(hidden[begin..].to_vec())
}

pub struct LongCatTextSource {
    weights: SafetensorSource,
    aliases: HashMap<String, (String, TensorInfo)>,
    extra: HashMap<String, (TensorInfo, Vec<u8>)>,
    metadata: HashMap<String, MetaValue>,
}

impl LongCatTextSource {
    pub fn open(component_root: &Path) -> Result<Self, String> {
        let encoder = component_root.join("text_encoder");
        let config: Value = serde_json::from_slice(
            &std::fs::read(encoder.join("config.json"))
                .map_err(|e| format!("Read LongCat encoder config: {e}"))?,
        )
        .map_err(|e| format!("Parse LongCat encoder config: {e}"))?;
        for (key, expected) in [
            ("hidden_size", HIDDEN),
            ("intermediate_size", FFN),
            ("vocab_size", VOCAB),
            ("num_hidden_layers", LAYERS as u64),
            ("num_attention_heads", 28),
            ("num_key_value_heads", 4),
        ] {
            if config.get(key).and_then(Value::as_u64) != Some(expected) {
                return Err(format!("Unsupported LongCat encoder {key}"));
            }
        }
        if config["architectures"][0] != "Qwen2_5_VLForConditionalGeneration"
            || config["rope_scaling"]["mrope_section"] != serde_json::json!([16, 24, 24])
            || config["vision_config"]["hidden_size"] != VISION
            || config["vision_config"]["intermediate_size"] != VISION_FF
            || config["vision_config"]["depth"] != VISION_LAYERS
            || config["vision_config"]["num_heads"] != 16
            || config["vision_config"]["patch_size"] != 14
            || config["vision_config"]["temporal_patch_size"] != 2
            || config["vision_config"]["spatial_merge_size"] != 2
        {
            return Err("Unsupported LongCat encoder architecture, vision or RoPE".into());
        }
        let files: Vec<_> = (1..=5)
            .map(|part| encoder.join(format!("model-{part:05}-of-00005.safetensors")))
            .collect();
        let weights = SafetensorSource::open(&files)?;
        let mut source = Self {
            weights,
            aliases: HashMap::new(),
            extra: HashMap::new(),
            metadata: HashMap::from([
                (
                    "general.architecture".into(),
                    MetaValue::String("qwen2vl".into()),
                ),
                ("qwen2vl.embedding_length".into(), MetaValue::Uint64(HIDDEN)),
                (
                    "qwen2vl.block_count".into(),
                    MetaValue::Uint32(LAYERS as u32),
                ),
                ("qwen2vl.attention.head_count".into(), MetaValue::Uint32(28)),
                (
                    "qwen2vl.attention.head_count_kv".into(),
                    MetaValue::Uint32(4),
                ),
                ("qwen2vl.feed_forward_length".into(), MetaValue::Uint64(FFN)),
                (
                    "qwen2vl.context_length".into(),
                    MetaValue::Uint64(config["max_position_embeddings"].as_u64().unwrap_or(128000)),
                ),
                ("qwen2vl.vocab_size".into(), MetaValue::Uint64(VOCAB)),
                (
                    "qwen2vl.rope.freq_base".into(),
                    MetaValue::Float32(1_000_000.0),
                ),
                (
                    "qwen2vl.attention.layer_norm_rms_epsilon".into(),
                    MetaValue::Float32(1e-6),
                ),
                (
                    "qwen2vl.rope.dimension_sections".into(),
                    MetaValue::Array(
                        MetaValueType::Int32,
                        [16, 24, 24, 0].into_iter().map(MetaValue::Int32).collect(),
                    ),
                ),
                (
                    "clip.vision.projection_dim".into(),
                    MetaValue::Uint64(HIDDEN),
                ),
                ("clip.vision.image_size".into(), MetaValue::Uint32(1024)),
                ("clip.vision.patch_size".into(), MetaValue::Uint32(14)),
                (
                    "clip.vision.embedding_length".into(),
                    MetaValue::Uint64(VISION),
                ),
                (
                    "clip.vision.feed_forward_length".into(),
                    MetaValue::Uint64(VISION_FF),
                ),
                (
                    "clip.vision.block_count".into(),
                    MetaValue::Uint32(VISION_LAYERS as u32),
                ),
                (
                    "clip.vision.attention.head_count".into(),
                    MetaValue::Uint32(16),
                ),
                (
                    "clip.vision.attention.layer_norm_epsilon".into(),
                    MetaValue::Float32(1e-6),
                ),
                (
                    "clip.vision.spatial_merge_size".into(),
                    MetaValue::Uint32(2),
                ),
                (
                    "clip.vision.image_min_pixels".into(),
                    MetaValue::Uint32(3136),
                ),
                (
                    "clip.vision.image_max_pixels".into(),
                    MetaValue::Uint32(12845056),
                ),
                ("clip.vision.n_wa_pattern".into(), MetaValue::Uint32(8)),
                ("clip.use_silu".into(), MetaValue::Bool(true)),
            ]),
        };
        source.alias(
            "token_embd.weight",
            "model.embed_tokens.weight",
            &[HIDDEN, VOCAB],
        )?;
        source.alias("output_norm.weight", "model.norm.weight", &[HIDDEN])?;
        for layer in 0..LAYERS {
            let gguf = format!("blk.{layer}");
            let hf = format!("model.layers.{layer}");
            for (target, original, dims) in [
                ("attn_norm.weight", "input_layernorm.weight", vec![HIDDEN]),
                (
                    "ffn_norm.weight",
                    "post_attention_layernorm.weight",
                    vec![HIDDEN],
                ),
                (
                    "attn_q.weight",
                    "self_attn.q_proj.weight",
                    vec![HIDDEN, HIDDEN],
                ),
                ("attn_k.weight", "self_attn.k_proj.weight", vec![HIDDEN, KV]),
                ("attn_v.weight", "self_attn.v_proj.weight", vec![HIDDEN, KV]),
                ("attn_q.bias", "self_attn.q_proj.bias", vec![HIDDEN]),
                ("attn_k.bias", "self_attn.k_proj.bias", vec![KV]),
                ("attn_v.bias", "self_attn.v_proj.bias", vec![KV]),
                (
                    "attn_output.weight",
                    "self_attn.o_proj.weight",
                    vec![HIDDEN, HIDDEN],
                ),
                ("ffn_gate.weight", "mlp.gate_proj.weight", vec![HIDDEN, FFN]),
                ("ffn_up.weight", "mlp.up_proj.weight", vec![HIDDEN, FFN]),
                ("ffn_down.weight", "mlp.down_proj.weight", vec![FFN, HIDDEN]),
            ] {
                source.alias(
                    &format!("{gguf}.{target}"),
                    &format!("{hf}.{original}"),
                    &dims,
                )?;
            }
        }
        source.split_temporal_patch()?;
        source.alias("v.post_ln.weight", "visual.merger.ln_q.weight", &[VISION])?;
        source.alias(
            "mm.0.weight",
            "visual.merger.mlp.0.weight",
            &[VISION * 4, VISION * 4],
        )?;
        source.alias("mm.0.bias", "visual.merger.mlp.0.bias", &[VISION * 4])?;
        source.alias(
            "mm.2.weight",
            "visual.merger.mlp.2.weight",
            &[VISION * 4, HIDDEN],
        )?;
        source.alias("mm.2.bias", "visual.merger.mlp.2.bias", &[HIDDEN])?;
        for layer in 0..VISION_LAYERS {
            let gguf = format!("v.blk.{layer}");
            let hf = format!("visual.blocks.{layer}");
            for (target, original, dims) in [
                ("ln1.weight", "norm1.weight", vec![VISION]),
                ("ln2.weight", "norm2.weight", vec![VISION]),
                (
                    "attn_qkv.weight",
                    "attn.qkv.weight",
                    vec![VISION, VISION * 3],
                ),
                ("attn_qkv.bias", "attn.qkv.bias", vec![VISION * 3]),
                ("attn_out.weight", "attn.proj.weight", vec![VISION, VISION]),
                ("attn_out.bias", "attn.proj.bias", vec![VISION]),
                (
                    "ffn_up.weight",
                    "mlp.up_proj.weight",
                    vec![VISION, VISION_FF],
                ),
                ("ffn_up.bias", "mlp.up_proj.bias", vec![VISION_FF]),
                (
                    "ffn_gate.weight",
                    "mlp.gate_proj.weight",
                    vec![VISION, VISION_FF],
                ),
                ("ffn_gate.bias", "mlp.gate_proj.bias", vec![VISION_FF]),
                (
                    "ffn_down.weight",
                    "mlp.down_proj.weight",
                    vec![VISION_FF, VISION],
                ),
                ("ffn_down.bias", "mlp.down_proj.bias", vec![VISION]),
            ] {
                source.alias(
                    &format!("{gguf}.{target}"),
                    &format!("{hf}.{original}"),
                    &dims,
                )?;
            }
        }
        Ok(source)
    }

    fn split_temporal_patch(&mut self) -> Result<(), String> {
        let name = "visual.patch_embed.proj.weight";
        let info = self
            .weights
            .tensor_info(name)
            .ok_or("LongCat vision patch weight missing")?;
        if info.ggml_type != GGMLType::BF16 || info.dims != [14, 14, 2, 3, VISION] {
            return Err(format!("LongCat vision patch shape: {:?}", info.dims));
        }
        let raw = self
            .weights
            .tensor_slice(name)
            .ok_or("LongCat vision patch data missing")?;
        let part_len = VISION as usize * 3 * 14 * 14 * 2;
        for temporal in 0..2 {
            let mut bytes = vec![0u8; part_len];
            for output in 0..VISION as usize {
                for channel in 0..3 {
                    let src = ((output * 3 + channel) * 2 + temporal) * 14 * 14 * 2;
                    let dst = (output * 3 + channel) * 14 * 14 * 2;
                    bytes[dst..dst + 14 * 14 * 2].copy_from_slice(&raw[src..src + 14 * 14 * 2]);
                }
            }
            let alias = if temporal == 0 {
                "v.patch_embd.weight"
            } else {
                "v.patch_embd.weight.1"
            };
            self.extra.insert(
                alias.into(),
                (
                    TensorInfo {
                        name: alias.into(),
                        dims: vec![14, 14, 3, VISION],
                        ggml_type: GGMLType::BF16,
                        offset: 0,
                    },
                    bytes,
                ),
            );
        }
        Ok(())
    }

    fn alias(&mut self, target: &str, original: &str, dims: &[u64]) -> Result<(), String> {
        let info = self
            .weights
            .tensor_info(original)
            .ok_or_else(|| format!("LongCat encoder missing {original}"))?;
        if info.dims != dims || info.ggml_type != GGMLType::BF16 {
            return Err(format!(
                "LongCat encoder invalid {original}: {:?}",
                info.dims
            ));
        }
        let mut info = info.clone();
        info.name = target.into();
        self.aliases.insert(target.into(), (original.into(), info));
        Ok(())
    }
}

impl TensorSource for LongCatTextSource {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.aliases
            .get(name)
            .map(|(_, info)| info)
            .or_else(|| self.extra.get(name).map(|(info, _)| info))
    }

    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        if let Some((original, _)) = self.aliases.get(name) {
            self.weights.tensor_slice(original)
        } else {
            self.extra.get(name).map(|(_, bytes)| bytes.as_slice())
        }
    }
}

pub fn load_tokenizer(component_root: &Path) -> Result<BPETokenizer, String> {
    let tokenizer_path = component_root.join("tokenizer/tokenizer.json");
    let json: Value = serde_json::from_slice(
        &std::fs::read(&tokenizer_path)
            .map_err(|e| format!("Read {}: {e}", tokenizer_path.display()))?,
    )
    .map_err(|e| format!("Parse {}: {e}", tokenizer_path.display()))?;
    let vocab = json["model"]["vocab"]
        .as_object()
        .ok_or("LongCat tokenizer has no vocabulary")?;
    let mut tokens = (0..VOCAB)
        .map(|id| format!("<|reserved_{id}|>"))
        .collect::<Vec<_>>();
    let mut types = vec![5u32; VOCAB as usize];
    for (token, id) in vocab {
        let id = id.as_u64().ok_or("Invalid LongCat token ID")? as usize;
        if id >= tokens.len() {
            return Err("LongCat token ID exceeds model vocabulary".into());
        }
        tokens[id] = token.clone();
        types[id] = 1;
    }
    for entry in json["added_tokens"]
        .as_array()
        .ok_or("LongCat tokenizer has no added tokens")?
    {
        let id = entry["id"]
            .as_u64()
            .ok_or("Invalid LongCat added token ID")? as usize;
        if id >= tokens.len() {
            return Err("LongCat added token ID exceeds model vocabulary".into());
        }
        tokens[id] = entry["content"]
            .as_str()
            .ok_or("Invalid LongCat added token")?
            .into();
        types[id] = if entry["special"] == true { 3 } else { 4 };
    }
    let merges = json["model"]["merges"]
        .as_array()
        .ok_or("LongCat tokenizer has no BPE merges")?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or("Invalid LongCat BPE merge")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let metadata = HashMap::from([
        ("tokenizer.ggml.model", MetaValue::String("gpt2".into())),
        ("tokenizer.ggml.pre", MetaValue::String("qwen2".into())),
        (
            "tokenizer.ggml.tokens",
            MetaValue::Array(
                MetaValueType::String,
                tokens.into_iter().map(MetaValue::String).collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type",
            MetaValue::Array(
                MetaValueType::Uint32,
                types.into_iter().map(MetaValue::Uint32).collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges",
            MetaValue::Array(
                MetaValueType::String,
                merges.into_iter().map(MetaValue::String).collect(),
            ),
        ),
        ("tokenizer.ggml.eos_token_id", MetaValue::Uint32(151645)),
        ("tokenizer.ggml.add_bos_token", MetaValue::Bool(false)),
        ("tokenizer.ggml.add_eos_token", MetaValue::Bool(false)),
    ]);
    BPETokenizer::from_gguf_metadata(|key| metadata.get(key).cloned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::thread_pool::ComputePool;
    use crate::core::tokenizer::EncodeOptions;
    use crate::models::qwen3::Qwen3Model;
    use crate::models::qwen35::vision::VisionEncoder;
    use std::sync::Arc;

    #[test]
    #[ignore = "requires RMI_LONGCAT_COMPONENT_ROOT and local LongCat weights"]
    fn opens_encoder_and_matches_hf_tokenizer() {
        let root = std::path::PathBuf::from(std::env::var("RMI_LONGCAT_COMPONENT_ROOT").unwrap());
        let tokenizer = load_tokenizer(&root).unwrap();
        let hf = tokenizers::Tokenizer::from_file(root.join("tokenizer/tokenizer.json")).unwrap();
        for text in [
            "Change the cat to a dog.",
            "把猫改成狗，保留背景。",
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>",
        ] {
            let ours = tokenizer.encode(
                text,
                EncodeOptions {
                    add_special: false,
                    parse_special: true,
                },
            );
            let expected = hf.encode(text, false).unwrap().get_ids().to_vec();
            assert_eq!(ours, expected, "{text}");
        }
        let input = prepare_prompt(&tokenizer, "Change the cat to a dog.", 4).unwrap();
        let prefix = format!("{PREFIX}{}<|vision_end|>", "<|image_pad|>".repeat(4));
        let mut expected = hf.encode(prefix, false).unwrap().get_ids().to_vec();
        expected.extend(
            hf.encode("Change the cat to a dog.", false)
                .unwrap()
                .get_ids(),
        );
        expected.extend(hf.encode(SUFFIX, false).unwrap().get_ids());
        let used = expected.len();
        expected.resize(579, PAD_TOKEN);
        assert_eq!(input.ids, expected);
        assert_eq!(
            input.mask.iter().filter(|&&valid| !valid).count(),
            579 - used
        );
        if let Some(dir) = std::env::var_os("RMI_LONGCAT_TRACE_DIR") {
            let input = prepare_prompt(&tokenizer, "Change the blue area to green.", 196).unwrap();
            let ids = input
                .ids
                .iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<_>>();
            let mask = input
                .mask
                .iter()
                .flat_map(|&valid| (valid as u32 as f32).to_le_bytes())
                .collect::<Vec<_>>();
            std::fs::write(std::path::PathBuf::from(&dir).join("prompt_0.u32"), ids).unwrap();
            std::fs::write(std::path::PathBuf::from(&dir).join("prompt_0.mask"), mask).unwrap();
        }
        let quoted = "Please write 'Hello' on the sign.";
        let mut oracle = hf
            .encode("Please write ", false)
            .unwrap()
            .get_ids()
            .to_vec();
        for ch in "'Hello'".chars() {
            oracle.extend(hf.encode(ch.to_string(), false).unwrap().get_ids());
        }
        oracle.extend(hf.encode(" on the sign.", false).unwrap().get_ids());
        assert_eq!(tokenize_instruction(&tokenizer, quoted), oracle);
        let source: Arc<dyn TensorSource> = Arc::new(LongCatTextSource::open(&root).unwrap());
        assert_eq!(
            source.tensor_info("blk.27.ffn_down.weight").unwrap().dims,
            [FFN, HIDDEN]
        );
        let model = Qwen3Model::from_source(
            Arc::clone(&source),
            Arc::new(tokenizer),
            Arc::new(ComputePool::new(1)),
        )
        .unwrap();
        assert_eq!(model.config().n_embd, HIDDEN as usize);
        assert_eq!(model.config().n_layer, LAYERS);
        let vision = VisionEncoder::from_source(source.as_ref()).unwrap();
        assert_eq!(vision.config.n_layer, VISION_LAYERS);
    }
}
