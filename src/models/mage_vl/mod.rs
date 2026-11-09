//! Mage-VL-Cowork: the released Mage-ViT and an explicitly registered Qwen3 trunk.
use crate::core::tensor::{GGMLType, TensorSource};
use crate::models::qwen3::{qwen_text_positions, Qwen3Model};
use crate::ops::kernel::{QuantizedTensor, Weight};
use crate::ops::nn::{attention_on_cpu, layer_norm_rows, load_float_values, AttentionSpec, Linear};
use crate::ComputePool;
use std::sync::Arc;

pub const ARCH: &str = "mage_vl";
pub const REVISION: &str = "287c5fdc8c83e43c4d207d21bc51024a52fdff38";
const HIDDEN: usize = 1024;
const PATCH: usize = 16;
pub const WIDTH: usize = 2560;
pub const IMAGE: u32 = 151655;
const TEXT_TYPES: &[GGMLType] = &[
    GGMLType::F32,
    GGMLType::F16,
    GGMLType::BF16,
    GGMLType::Q8_0,
    GGMLType::Q4_0,
    GGMLType::Q4K,
    GGMLType::Q5K,
    GGMLType::Q6K,
];

pub fn prompt_tokens(
    tokenizer: &crate::BPETokenizer,
    template: &crate::prompt::jinja::JinjaChatTemplate,
    messages: &serde_json::Value,
    tools: &serde_json::Value,
    thinking: bool,
) -> Result<Vec<u32>, String> {
    let text = template.render_context(&serde_json::json!({"messages":messages,"tools":tools,
        "enable_thinking":thinking,"add_generation_prompt":true,"add_vision_id":false}))?;
    Ok(tokenizer.encode(
        &text,
        crate::core::tokenizer::EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    ))
}

pub fn validate_components(
    text: &dyn TensorSource,
    vision: &dyn TensorSource,
) -> Result<(), String> {
    let revision = |s: &dyn TensorSource| {
        s.metadata("mage_vl.source_revision")
            .and_then(|v| v.to_string_val())
            .map(str::to_owned)
    };
    if text
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        != Some(ARCH)
        || revision(text).as_deref() != Some(REVISION)
        || revision(text) != revision(vision)
        || vision
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val())
            != Some("clip")
        || vision
            .metadata("clip.projector_type")
            .and_then(|v| v.to_string_val())
            != Some(ARCH)
    {
        return Err(
            "Mage-VL-Cowork requires matching mage_vl text and Mage-ViT mmproj components".into(),
        );
    }
    Ok(())
}

fn require_tensor(
    s: &dyn TensorSource,
    name: &str,
    dims: &[u64],
    types: &[GGMLType],
) -> Result<(), String> {
    let info = s
        .tensor_info(name)
        .ok_or_else(|| format!("Mage-VL missing tensor {name}"))?;
    let raw = s
        .tensor_slice(name)
        .ok_or_else(|| format!("Mage-VL missing data {name}"))?;
    if info.dims != dims
        || !types.contains(&info.ggml_type)
        || info.checked_nbytes() != Some(raw.len() as u64)
    {
        return Err(format!("invalid Mage-VL tensor {name}: shape/type/storage"));
    }
    Ok(())
}

pub(crate) fn validate_text_tensors(s: &dyn TensorSource) -> Result<(), String> {
    for name in ["token_embd.weight", "output.weight"] {
        require_tensor(s, name, &[2560, 151936], TEXT_TYPES)?;
    }
    require_tensor(s, "output_norm.weight", &[2560], &[GGMLType::F32])?;
    for i in 0..36 {
        for (suffix, dims) in [
            ("attn_norm", vec![2560]),
            ("ffn_norm", vec![2560]),
            ("attn_q_norm", vec![128]),
            ("attn_k_norm", vec![128]),
            ("attn_q", vec![2560, 4096]),
            ("attn_k", vec![2560, 1024]),
            ("attn_v", vec![2560, 1024]),
            ("attn_output", vec![4096, 2560]),
            ("ffn_gate", vec![2560, 9728]),
            ("ffn_up", vec![2560, 9728]),
            ("ffn_down", vec![9728, 2560]),
        ] {
            let types = if dims.len() == 1 {
                &[GGMLType::F32][..]
            } else {
                TEXT_TYPES
            };
            require_tensor(s, &format!("blk.{i}.{suffix}.weight"), &dims, types)?;
        }
    }
    Ok(())
}

fn linear<'a>(
    s: &'a dyn TensorSource,
    name: &str,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<Linear<'a>, String> {
    require_tensor(
        s,
        &format!("{name}.weight"),
        &[input as u64, output as u64],
        &[GGMLType::F32],
    )?;
    if bias {
        require_tensor(
            s,
            &format!("{name}.bias"),
            &[output as u64],
            &[GGMLType::F32],
        )?;
    }
    Linear::from_source(
        s,
        &format!("{name}.weight"),
        bias.then(|| format!("{name}.bias")).as_deref(),
        input,
        output,
    )
}

struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}
impl Norm {
    fn load(s: &dyn TensorSource, name: &str) -> Result<Self, String> {
        for suffix in ["weight", "bias"] {
            require_tensor(
                s,
                &format!("{name}.{suffix}"),
                &[HIDDEN as u64],
                &[GGMLType::F32],
            )?;
        }
        Ok(Self {
            weight: load_float_values(s, &format!("{name}.weight"), &[HIDDEN as u64])?,
            bias: load_float_values(s, &format!("{name}.bias"), &[HIDDEN as u64])?,
        })
    }
    fn apply(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        layer_norm_rows(
            input,
            input.len() / HIDDEN,
            Some(&self.weight),
            Some(&self.bias),
            1e-6,
        )
    }
}

struct Block<'a> {
    n1: Norm,
    n2: Norm,
    qkv: Linear<'a>,
    out: Linear<'a>,
    up: Linear<'a>,
    down: Linear<'a>,
}

pub struct Vision<'a> {
    patch: Linear<'a>,
    pre: Norm,
    blocks: Vec<Block<'a>>,
    norm: Norm,
    fc1: Linear<'a>,
    fc2: Linear<'a>,
}
impl<'a> Vision<'a> {
    pub fn load(s: &'a dyn TensorSource) -> Result<Self, String> {
        if s.metadata("clip.projector_type")
            .and_then(|v| v.to_string_val())
            != Some(ARCH)
            || s.metadata("mage_vl.source_revision")
                .and_then(|v| v.to_string_val())
                != Some(REVISION)
        {
            return Err("expected Mage-ViT mmproj (clip.projector_type=mage_vl)".into());
        }
        let name = "visual.embeddings.patch_embedding.weight";
        let info = s
            .tensor_info(name)
            .ok_or_else(|| format!("Missing {name}"))?;
        let raw = s
            .tensor_slice(name)
            .ok_or_else(|| format!("Missing {name} data"))?;
        if info.dims != [16, 16, 3, 1024]
            || info.ggml_type != GGMLType::F32
            || info.checked_nbytes() != Some(raw.len() as u64)
        {
            return Err("invalid Mage-ViT patch embedding shape/type/storage".into());
        }
        let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
            raw,
            info.ggml_type,
            768,
            HIDDEN,
        ));
        weight.n_in = 768;
        weight.n_out = HIDDEN;
        let mut blocks = Vec::new();
        for i in 0..24 {
            let p = format!("visual.encoder.layers.{i}");
            blocks.push(Block {
                n1: Norm::load(s, &format!("{p}.layer_norm1"))?,
                n2: Norm::load(s, &format!("{p}.layer_norm2"))?,
                qkv: linear(s, &format!("{p}.self_attn.qkv"), HIDDEN, 3 * HIDDEN, true)?,
                out: linear(s, &format!("{p}.self_attn.proj"), HIDDEN, HIDDEN, true)?,
                up: linear(s, &format!("{p}.mlp.fc1"), HIDDEN, 4096, true)?,
                down: linear(s, &format!("{p}.mlp.fc2"), 4096, HIDDEN, true)?,
            });
        }
        Ok(Self {
            patch: Linear::from_weight(weight, None)?,
            pre: Norm::load(s, "visual.layernorm_pre")?,
            blocks,
            norm: Norm::load(s, "visual.merger.ln_q")?,
            fc1: linear(s, "visual.merger.mlp.0", 4096, 4096, true)?,
            fc2: linear(s, "visual.merger.mlp.2", 4096, WIDTH, true)?,
        })
    }

    /// Patches are CHW inside each 2x2 block, exactly as Qwen2VLImageProcessor.
    pub fn encode(
        &self,
        patches: &[f32],
        h: usize,
        w: usize,
        pool: &Arc<ComputePool>,
    ) -> Result<Vec<f32>, String> {
        if h == 0
            || w == 0
            || h % 2 != 0
            || w % 2 != 0
            || h.checked_mul(w).and_then(|n| n.checked_mul(768)) != Some(patches.len())
            || patches.iter().any(|x| !x.is_finite())
        {
            return Err("invalid Mage-ViT image patch grid".into());
        }
        let n = h * w;
        let policy = crate::compute::ComputePolicy::legacy();
        let mut executor = if policy == crate::compute::ComputePolicy::Cpu {
            None
        } else {
            let bindings = [&self.patch, &self.fc1, &self.fc2]
                .into_iter()
                .chain(
                    self.blocks
                        .iter()
                        .flat_map(|b| [&b.qkv, &b.out, &b.up, &b.down]),
                )
                .map(Linear::binding)
                .collect();
            Some(
                crate::compute::linear::LinearExecutor::new(policy, bindings, n, pool.clone())
                    .map_err(|e| e.to_string())?,
            )
        };
        // shortcut: offload projections; keep Mage's per-lane RoPE and F32 attention on CPU until device kernels exist.
        let mut project = |linear: &Linear<'a>, input: &[f32], rows| {
            if let Some(executor) = &mut executor {
                Linear::forward_group([linear], executor, input, rows).map(|[out]| out)
            } else {
                linear.forward(pool, input, rows)
            }
        };
        let mut x = project(&self.patch, patches, n)?;
        trace("patch", &x)?;
        x = self.pre.apply(&x)?;
        trace("pre", &x)?;
        for (i, b) in self.blocks.iter().enumerate() {
            let qkv = project(&b.qkv, &b.n1.apply(&x)?, n)?;
            let (mut q, mut k, mut v) = (
                Vec::with_capacity(n * HIDDEN),
                Vec::with_capacity(n * HIDDEN),
                Vec::with_capacity(n * HIDDEN),
            );
            for row in qkv.chunks_exact(HIDDEN * 3) {
                q.extend_from_slice(&row[..HIDDEN]);
                k.extend_from_slice(&row[HIDDEN..2 * HIDDEN]);
                v.extend_from_slice(&row[2 * HIDDEN..]);
            }
            apply_rope(&mut q, h, w);
            apply_rope(&mut k, h, w);
            let attn = attention_on_cpu(
                &q,
                &k,
                &v,
                AttentionSpec {
                    query_tokens: n,
                    key_tokens: n,
                    query_heads: 16,
                    key_value_heads: 16,
                    head_dim: 64,
                    causal: false,
                    scale: 0.125,
                },
            )?;
            let out = project(&b.out, &attn, n)?;
            crate::ops::vec_add_into(&out, &mut x);
            let mut ff = project(&b.up, &b.n2.apply(&x)?, n)?;
            crate::ops::gelu_erf_inplace(&mut ff);
            crate::ops::vec_add_into(&project(&b.down, &ff, n)?, &mut x);
            trace(&format!("vit{i}"), &x)?;
        }
        let mut merged = project(&self.fc1, &self.norm.apply(&x)?, n / 4)?;
        crate::ops::gelu_erf_inplace(&mut merged);
        let projected = project(&self.fc2, &merged, n / 4)?;
        trace("projected", &projected)?;
        Ok(projected)
    }
}

fn apply_rope(values: &mut [f32], h: usize, w: usize) {
    for (t, token) in values.chunks_exact_mut(HIDDEN).enumerate() {
        let block = t / 4;
        let y = block / (w / 2) * 2 + t % 4 / 2;
        let x = block % (w / 2) * 2 + t % 2;
        debug_assert!(y < h);
        for head in token.chunks_exact_mut(64) {
            for pair in 0..32 {
                let (a, b) = (head[2 * pair], head[2 * pair + 1]);
                // The release concatenates [freqs,freqs] but rotates adjacent pairs.
                for lane in 0..2 {
                    let dim = (2 * pair + lane) % 32;
                    let (position, index, size) = if dim < 8 {
                        (0, dim, 8)
                    } else if dim < 20 {
                        (y, dim - 8, 12)
                    } else {
                        (x, dim - 20, 12)
                    };
                    let phase = position as f32 / 10000.0f32.powf(index as f32 / size as f32);
                    let (cos, sin) = crate::ops::rope_sin_cos(phase);
                    head[2 * pair + lane] = if lane == 0 {
                        a * cos - b * sin
                    } else {
                        b * cos + a * sin
                    };
                }
            }
        }
    }
}

pub fn resize(width: usize, height: usize) -> Result<(usize, usize), String> {
    if width == 0 || height == 0 || width.max(height) as f64 / width.min(height) as f64 > 200.0 {
        return Err("Mage-ViT requires nonzero image dimensions and aspect ratio <=200".into());
    }
    let (mut w, mut h) = (
        ((width as f64 / 32.0).round_ties_even() as usize * 32).max(32),
        ((height as f64 / 32.0).round_ties_even() as usize * 32).max(32),
    );
    let pixels = width as f64 * height as f64;
    if w as f64 * h as f64 > 4_000_000.0 {
        let scale = (pixels / 4_000_000.0).sqrt();
        w = (width as f64 / scale / 32.0).floor() as usize * 32;
        h = (height as f64 / scale / 32.0).floor() as usize * 32;
    } else if w * h < 3136 {
        let scale = (3136.0 / pixels).sqrt();
        w = (width as f64 * scale / 32.0).ceil() as usize * 32;
        h = (height as f64 * scale / 32.0).ceil() as usize * 32;
    }
    if w == 0 || h == 0 {
        return Err("Mage-ViT resized image has zero size".into());
    }
    Ok((w, h))
}

pub fn image_patches(image: &image::DynamicImage) -> Result<(Vec<f32>, usize, usize), String> {
    let rgb = image.to_rgb8();
    let (w, h) = resize(rgb.width() as usize, rgb.height() as usize)?;
    // shortcut: reuse Pillow bicubic; the release's Fast tensor resize can differ by 1–2 uint8 levels.
    let pixels = crate::models::gemma4::vision::resize_bicubic_pillow(
        rgb.as_raw(),
        rgb.width() as usize,
        rgb.height() as usize,
        w,
        h,
    )?;
    let (gh, gw) = (h / PATCH, w / PATCH);
    let (mean, std): ([f32; 3], [f32; 3]) = (
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711],
    );
    let mut patches = Vec::with_capacity(w * h * 3);
    for by in (0..gh).step_by(2) {
        for bx in (0..gw).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    for c in 0..3 {
                        for py in 0..PATCH {
                            for px in 0..PATCH {
                                let value =
                                    pixels[(((by + dy) * PATCH + py) * w + (bx + dx) * PATCH + px)
                                        * 3
                                        + c] as f32;
                                patches.push((value - mean[c] * 255.0) / (std[c] * 255.0));
                            }
                        }
                    }
                }
            }
        }
    }
    trace("pixels", &patches)?;
    Ok((patches, gh, gw))
}

pub fn encode_image(
    s: &dyn TensorSource,
    image: &image::DynamicImage,
    pool: &Arc<ComputePool>,
) -> Result<Vec<f32>, String> {
    let (patches, h, w) = image_patches(image)?;
    Vision::load(s)?.encode(&patches, h, w, pool)
}

pub struct Prepared {
    pub tokens: Vec<u32>,
    pub embeddings: Vec<f32>,
    pub positions: Vec<[usize; 4]>,
}

pub fn check_image_budget(
    ids: &[u32],
    images: &[image::DynamicImage],
    capacity: usize,
) -> Result<usize, String> {
    if ids.iter().filter(|&&id| id == IMAGE).count() != images.len() || images.is_empty() {
        return Err("Mage-VL image placeholders must match attached images".into());
    }
    let mut rows = 0usize;
    for image in images {
        let (w, h) = resize(image.width() as usize, image.height() as usize)?;
        rows = rows
            .checked_add(w * h / 1024)
            .ok_or("Mage-VL image token overflow")?;
    }
    if rows > 2048 {
        return Err(
            "Mage-VL images exceed the 2048 visual token budget; downscale the images".into(),
        );
    }
    let total = ids
        .len()
        .checked_sub(images.len())
        .and_then(|n| n.checked_add(rows))
        .ok_or("Mage-VL prompt length overflow")?;
    if total > capacity {
        return Err("Mage-VL image prompt exceeds model context / --max-context".into());
    }
    Ok(total)
}

/// Expand each template image placeholder in place, preserving multi-turn image order.
pub fn prepare(
    model: &Qwen3Model,
    vision: &dyn TensorSource,
    ids: &[u32],
    images: &[image::DynamicImage],
    pool: &Arc<ComputePool>,
    capacity: usize,
) -> Result<Prepared, String> {
    let total = check_image_budget(ids, images, capacity.min(model.config().n_ctx))?;
    let mut tokens = Vec::with_capacity(total);
    let mut media = Vec::new();
    let mut next = images.iter();
    for &id in ids {
        if id == IMAGE {
            let encoded = encode_image(vision, next.next().unwrap(), pool)?;
            tokens.extend(std::iter::repeat_n(IMAGE, encoded.len() / WIDTH));
            media.extend(encoded);
        } else {
            tokens.push(id);
        }
    }
    debug_assert_eq!(tokens.len(), total);
    let mut embeddings = model.embed_tokens(&tokens)?;
    crate::models::qwen3::vision::media::inject_qwen_media_embeddings(
        &tokens,
        IMAGE,
        &mut embeddings,
        &media,
        &[],
        WIDTH,
    )?;
    let positions = qwen_text_positions(tokens.len());
    Ok(Prepared {
        tokens,
        embeddings,
        positions,
    })
}

/// Shared image scoring pipeline for CLI and HTTP JEV.
#[allow(clippy::too_many_arguments)]
pub fn image_logits(
    source: std::sync::Arc<dyn TensorSource>,
    vision: &dyn TensorSource,
    image: image::DynamicImage,
    prompt: &str,
    system: Option<&str>,
    threads: usize,
    prefill_batch_size: usize,
    max_context: usize,
) -> Result<(Vec<f32>, std::time::Duration), String> {
    use crate::core::scratchpad::{KvFormat, KvLifecycle};
    use crate::models::qwen3::{Qwen3Input, Qwen3Session};
    use std::sync::Arc;

    validate_components(source.as_ref(), vision)?;
    let tokenizer = Arc::new(crate::BPETokenizer::from_gguf_metadata(|k| {
        source.metadata(k).cloned()
    })?);
    let template = crate::prompt::jinja::template_from_source(&|k| source.metadata(k).cloned())
        .ok_or("missing Mage-VL template")??;
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(serde_json::json!({"role":"system","content":system}));
    }
    messages.push(serde_json::json!({"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}));
    let ids = prompt_tokens(
        &tokenizer,
        &template,
        &serde_json::json!(messages),
        &serde_json::json!([]),
        false,
    )?;
    let pool = Arc::new(ComputePool::new(threads));
    let model = Qwen3Model::from_source(source, tokenizer, Arc::clone(&pool))?;
    let prepared = prepare(&model, vision, &ids, &[image], &pool, max_context)?;
    let mut session = Qwen3Session::new_with_kv_state(
        &model,
        prepared.tokens.len(),
        KvFormat::F32,
        KvLifecycle::Ephemeral,
    )?;
    session.forward_logits(
        Qwen3Input {
            token_ids: &prepared.tokens,
            positions: &prepared.positions,
            embeddings: Some(&prepared.embeddings),
            deepstack_embeddings: None,
        },
        prefill_batch_size,
    )
}

fn trace(name: &str, values: &[f32]) -> Result<(), String> {
    if let Some(dir) = std::env::var_os("RMI_MAGE_TRACE_DIR") {
        let dir = std::path::Path::new(&dir);
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("{name}.f32")), bytes).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{MetaValue, TensorInfo};
    struct Source(std::collections::HashMap<String, MetaValue>);
    impl TensorSource for Source {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.0.get(key)
        }
        fn tensor_info(&self, _: &str) -> Option<&TensorInfo> {
            None
        }
        fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
            None
        }
    }
    #[test]
    fn compressed_matrices_require_aligned_complete_storage() {
        struct Matrix(TensorInfo, Vec<u8>);
        impl TensorSource for Matrix {
            fn metadata(&self, _: &str) -> Option<&MetaValue> {
                None
            }
            fn tensor_info(&self, _: &str) -> Option<&TensorInfo> {
                Some(&self.0)
            }
            fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
                Some(&self.1)
            }
        }
        for &ggml_type in TEXT_TYPES {
            let info = TensorInfo {
                name: "matrix".into(),
                dims: vec![256, 2],
                ggml_type,
                offset: 0,
            };
            let mut source = Matrix(
                info.clone(),
                vec![0; info.checked_nbytes().unwrap() as usize],
            );
            require_tensor(&source, "matrix", &[256, 2], TEXT_TYPES).unwrap();
            source.1.pop();
            assert!(require_tensor(&source, "matrix", &[256, 2], TEXT_TYPES).is_err());
            source.0.dims[0] = 255;
            assert!(require_tensor(&source, "matrix", &[256, 2], TEXT_TYPES).is_err());
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires hardware Vulkan; run alone with RUST_GPU_MIN_OUTPUTS=0 and --test-threads=1"]
    fn mage_vision_projections_submit_and_preserve_cpu_rope_attention() {
        use crate::compute::ComputePolicy;
        assert_eq!(crate::ops::kernel::vulkan::min_gpu_matmul_outputs(), 0);
        let linear = |n_in, n_out| {
            let mut values = vec![0.0f32; n_in * n_out];
            for row in 0..n_out {
                values[row * n_in + row % n_in] = 0.25;
                values[row * n_in + (row * 13 + 7) % n_in] += 0.125;
            }
            Linear::from_weight(
                Weight::from_quantized(QuantizedTensor::F32 {
                    data: values,
                    n_in,
                    n_out,
                }),
                Some((0..n_out).map(|i| (i % 7) as f32 * 0.001).collect()),
            )
            .unwrap()
        };
        let norm = || Norm {
            weight: vec![1.0; HIDDEN],
            bias: vec![0.0; HIDDEN],
        };
        let vision = Vision {
            patch: linear(768, HIDDEN),
            pre: norm(),
            norm: norm(),
            blocks: vec![Block {
                n1: norm(),
                n2: norm(),
                qkv: linear(HIDDEN, HIDDEN * 3),
                out: linear(HIDDEN, HIDDEN),
                up: linear(HIDDEN, 4096),
                down: linear(4096, HIDDEN),
            }],
            fc1: linear(4096, 4096),
            fc2: linear(4096, WIDTH),
        };
        let pool = Arc::new(ComputePool::new(2));
        let patches: Vec<_> = (0..16 * 768)
            .map(|i| (i % 179) as f32 / 89.0 - 1.0)
            .collect();
        let started = std::time::Instant::now();
        let expected = {
            let _scope = ComputePolicy::Cpu.enter_legacy_scope();
            vision.encode(&patches, 4, 4, &pool).unwrap()
        };
        eprintln!("Mage synthetic vision Cpu elapsed={:?}", started.elapsed());
        let context = ComputePolicy::Vulkan.context().unwrap().unwrap();
        for policy in [ComputePolicy::Vulkan, ComputePolicy::Auto] {
            let _scope = policy.enter_legacy_scope();
            let before = context.compute_stats();
            let started = std::time::Instant::now();
            let actual = vision.encode(&patches, 4, 4, &pool).unwrap();
            let submissions = context.compute_stats().submissions - before.submissions;
            assert!(
                submissions >= 7,
                "all seven synthetic vision projections must execute"
            );
            let max_abs = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs < 0.002, "max_abs={max_abs}");
            eprintln!("Mage synthetic vision {policy:?} submissions={submissions} max_abs={max_abs} elapsed={:?}", started.elapsed());
        }
        let before = context.compute_stats();
        let _scope = ComputePolicy::Cpu.enter_legacy_scope();
        assert_eq!(vision.encode(&patches, 4, 4, &pool).unwrap(), expected);
        assert_eq!(context.compute_stats().submissions, before.submissions);
    }
    #[test]
    fn mage_dispatch_requires_its_own_metadata_and_ordinary_text_rope() {
        let mut meta = std::collections::HashMap::new();
        meta.insert(
            "general.architecture".into(),
            MetaValue::String(ARCH.into()),
        );
        meta.insert(
            "mage_vl.source_revision".into(),
            MetaValue::String(REVISION.into()),
        );
        for (key, value) in [
            ("embedding_length", 2560),
            ("block_count", 36),
            ("attention.head_count", 32),
            ("attention.head_count_kv", 8),
            ("attention.key_length", 128),
            ("attention.value_length", 128),
            ("feed_forward_length", 9728),
            ("context_length", 262144),
            ("vocab_size", 151936),
        ] {
            meta.insert(format!("mage_vl.{key}"), MetaValue::Uint32(value));
        }
        meta.insert(
            "mage_vl.rope.freq_base".into(),
            MetaValue::Float32(5_000_000.),
        );
        meta.insert(
            "mage_vl.attention.layer_norm_rms_epsilon".into(),
            MetaValue::Float32(1e-6),
        );
        let mut source = Source(meta);
        let cfg = crate::models::qwen3::Qwen3Config::from_source(&source).unwrap();
        assert_eq!(cfg.architecture, ARCH);
        assert_eq!(cfg.n_embd_head_k, 128);
        assert_eq!(cfg.rope, crate::models::qwen3::Qwen3Rope::Neox);
        assert!(cfg.has_qk_norm);
        source
            .0
            .insert("mage_vl.block_count".into(), MetaValue::Uint32(24));
        assert!(crate::models::qwen3::Qwen3Config::from_source(&source).is_err());
        source.0.remove("mage_vl.embedding_length");
        source
            .0
            .insert("qwen3.embedding_length".into(), MetaValue::Uint32(2560));
        assert!(crate::models::qwen3::Qwen3Config::from_source(&source)
            .unwrap_err()
            .contains("mage_vl.embedding_length"));
    }
    #[test]
    fn mage_rejects_missing_or_mismatched_projectors_and_allows_partial_vulkan() {
        let source = Source(std::collections::HashMap::new());
        assert!(validate_components(&source, &source).is_err());
        assert!(Vision::load(&source).is_err());
        assert!(validate_text_tensors(&source)
            .unwrap_err()
            .contains("token_embd.weight"));
        assert!(crate::compute::ComputePolicy::Vulkan
            .validate_text_arch(ARCH)
            .is_ok());
        assert!(!crate::compute::ComputePolicy::has_complete_device_path(
            ARCH
        ));
        assert!(crate::compute::ComputePolicy::Auto
            .validate_text_arch(ARCH)
            .is_ok());
        let text = Source(std::collections::HashMap::from([
            (
                "general.architecture".into(),
                MetaValue::String(ARCH.into()),
            ),
            (
                "mage_vl.source_revision".into(),
                MetaValue::String(REVISION.into()),
            ),
        ]));
        let mut vision = Source(std::collections::HashMap::from([
            (
                "general.architecture".into(),
                MetaValue::String("clip".into()),
            ),
            ("clip.projector_type".into(), MetaValue::String(ARCH.into())),
            (
                "mage_vl.source_revision".into(),
                MetaValue::String(REVISION.into()),
            ),
        ]));
        assert!(validate_components(&text, &vision).is_ok());
        vision.0.insert(
            "mage_vl.source_revision".into(),
            MetaValue::String("wrong".into()),
        );
        assert!(validate_components(&text, &vision).is_err());
    }
    #[test]
    fn mage_rope_rotates_adjacent_pairs_with_distinct_axes() {
        let mut v = vec![1.; 4 * HIDDEN];
        apply_rope(&mut v, 2, 2);
        assert_eq!(&v[..64], &[1.; 64]);
        let (sin, cos) = 1.0f32.sin_cos();
        assert!((v[HIDDEN + 20] - (cos - sin)).abs() < 1e-6);
        assert_eq!(v[HIDDEN + 8], 1.);
        assert!((v[2 * HIDDEN + 8] - (cos - sin)).abs() < 1e-6);
        assert_eq!(v[2 * HIDDEN + 20], 1.);
        let phase = 10000.0f32.powf(-1.0 / 12.0);
        let (sin, cos) = phase.sin_cos();
        assert!((v[2 * HIDDEN + 9] - (cos + sin)).abs() < 1e-6);
    }
    #[test]
    fn mage_resize_uses_python_rounding_and_rejects_bad_geometry() {
        assert_eq!(resize(144, 80).unwrap(), (128, 64));
        assert!(resize(0, 32).is_err());
        assert!(resize(10000, 1).is_err());
        assert_eq!(resize(64, 64).unwrap(), (64, 64));
    }

    #[test]
    fn mage_image_budget_is_checked_before_encoding() {
        let image = image::DynamicImage::new_rgb8(64, 64);
        assert_eq!(
            check_image_budget(&[7, IMAGE, 8], std::slice::from_ref(&image), 6).unwrap(),
            6
        );
        assert!(check_image_budget(&[7, IMAGE, 8], std::slice::from_ref(&image), 5).is_err());
        assert!(check_image_budget(&[7], &[image], 100).is_err());
        assert!(
            check_image_budget(&[IMAGE], &[image::DynamicImage::new_rgb8(1600, 1600)], 8192)
                .unwrap_err()
                .contains("2048")
        );
    }

    #[test]
    fn mage_tensor_contract_rejects_bad_shape_type_and_storage() {
        struct Tensor {
            info: TensorInfo,
            bytes: Vec<u8>,
        }
        impl TensorSource for Tensor {
            fn metadata(&self, _: &str) -> Option<&MetaValue> {
                None
            }
            fn tensor_info(&self, _: &str) -> Option<&TensorInfo> {
                Some(&self.info)
            }
            fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
                Some(&self.bytes)
            }
        }
        let mut tensor = Tensor {
            info: TensorInfo {
                name: "test".into(),
                dims: vec![1],
                ggml_type: GGMLType::F32,
                offset: 0,
            },
            bytes: vec![0; 4],
        };
        assert!(require_tensor(&tensor, "test", &[1], &[GGMLType::F32]).is_ok());
        assert!(require_tensor(&tensor, "test", &[2], &[GGMLType::F32]).is_err());
        tensor.info.ggml_type = GGMLType::F16;
        assert!(require_tensor(&tensor, "test", &[1], &[GGMLType::F32]).is_err());
        tensor.info.ggml_type = GGMLType::F32;
        tensor.bytes.pop();
        assert!(require_tensor(&tensor, "test", &[1], &[GGMLType::F32]).is_err());
    }
}
