//! Native UltraVBERT image layout, SigLIP encoder, connector and task readouts.

use super::tensor::{self, embed, layer_norm, linear, rms_norm, shape, values};
use crate::core::tensor::{MetaValue, TensorInfo, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::format::safetensors::SafetensorSource;
use base64::Engine;
use image::RgbImage;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::Path;

type Trace = Vec<(String, Vec<f32>)>;

fn record(trace: &mut Option<&mut Trace>, name: &str, x: &[f32]) {
    if let Some(trace) = trace {
        trace.push((name.into(), x.to_vec()));
    }
}

fn dimension(config: &Value, name: &str) -> Result<usize, String> {
    config[name]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| format!("missing or invalid UltraVBERT {name}"))
}

fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(
        &std::fs::read(path).map_err(|e| format!("Read {}: {e}", path.display()))?,
    )
    .map_err(|e| format!("Parse {}: {e}", path.display()))
}

pub(super) fn load_image(value: &Value) -> Result<RgbImage, String> {
    let name = value
        .as_str()
        .ok_or("images must contain paths, URLs or image data URLs")?;
    let bytes = if name.starts_with("data:image") {
        let (header, data) = name.split_once(',').ok_or("invalid image data URL")?;
        if !header.ends_with(";base64") {
            return Err("image data URL must use base64".into());
        }
        base64::engine::general_purpose::STANDARD
            .decode(data)
            .map_err(|e| format!("Decode image base64: {e}"))?
    } else if name.starts_with("http://") || name.starts_with("https://") {
        reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| format!("Build image HTTP client: {e}"))?
            .get(name)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .and_then(reqwest::blocking::Response::bytes)
            .map_err(|e| format!("Fetch image: {e}"))?
            .to_vec()
    } else {
        std::fs::read(name).map_err(|e| format!("Read image {name}: {e}"))?
    };
    image::load_from_memory(&bytes)
        .map(|im| im.to_rgb8())
        .map_err(|e| format!("Decode image: {e}"))
}

pub(super) fn resize(image: &RgbImage, width: usize, height: usize) -> Result<RgbImage, String> {
    if width == 0 || height == 0 || image.width() == 0 || image.height() == 0 {
        return Err("invalid resized UltraVBERT image".into());
    }
    let mut current = image.clone();
    for horizontal in [true, false] {
        let input = if horizontal {
            current.width()
        } else {
            current.height()
        } as usize;
        let output = if horizontal { width } else { height };
        if input == output {
            continue;
        }
        let (bounds, weights, precision) = bicubic_weights(input, output);
        let (target_w, target_h) = if horizontal {
            (width, current.height() as usize)
        } else {
            (current.width() as usize, height)
        };
        let mut resized = RgbImage::new(target_w as u32, target_h as u32);
        for y in 0..target_h {
            for x in 0..target_w {
                for c in 0..3 {
                    let position = if horizontal { x } else { y };
                    let mut sum = 1i32 << (precision - 1);
                    for (i, &weight) in weights[position].iter().enumerate() {
                        let pixel = if horizontal {
                            current.get_pixel((bounds[position] + i) as u32, y as u32)
                        } else {
                            current.get_pixel(x as u32, (bounds[position] + i) as u32)
                        };
                        sum += pixel[c] as i32 * weight;
                    }
                    resized.get_pixel_mut(x as u32, y as u32)[c] =
                        (sum >> precision).clamp(0, 255) as u8;
                }
            }
        }
        current = resized;
    }
    Ok(current)
}

fn bicubic_weights(input: usize, output: usize) -> (Vec<usize>, Vec<Vec<i32>>, u32) {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.);
    let support = 2. * filter_scale;
    let mut bounds = Vec::with_capacity(output);
    let mut floating = Vec::with_capacity(output);
    let mut maximum = 0f64;
    for position in 0..output {
        let center = (position as f64 + 0.5) * scale;
        let start = ((center - support + 0.5) as isize).max(0) as usize;
        let end = ((center + support + 0.5) as usize).min(input);
        let mut weights = Vec::with_capacity(end - start);
        for i in start..end {
            let x = ((i as f64 - center + 0.5) / filter_scale).abs();
            weights.push(if x < 1. {
                ((1.5 * x - 2.5) * x) * x + 1.
            } else if x < 2. {
                (((-0.5 * x + 2.5) * x - 4.) * x) + 2.
            } else {
                0.
            });
        }
        let total: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= total;
            maximum = maximum.max(*w);
        }
        bounds.push(start);
        floating.push(weights);
    }
    // PyTorch's uint8 antialias kernel chooses int16 coefficient precision; Pillow fixes it at 22 bits.
    let mut precision = 0;
    while precision < 22 && ((0.5 + maximum * (1u32 << (precision + 1)) as f64) as i32) < 1 << 15 {
        precision += 1;
    }
    let weights = floating
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|w| {
                    let scaled = w * (1u32 << precision) as f64;
                    (scaled + if scaled < 0. { -0.5 } else { 0.5 }) as i32
                })
                .collect()
        })
        .collect();
    (bounds, weights, precision)
}

fn normalize(image: &RgbImage) -> Vec<f32> {
    let plane = image.width() as usize * image.height() as usize;
    let mut pixels = vec![0.; plane * 3];
    for (i, p) in image.pixels().enumerate() {
        for c in 0..3 {
            pixels[c * plane + i] = (p[c] as f32 * (1.0 / 255.0) - 0.5) / 0.5;
        }
    }
    if pixels.iter().all(|&p| p == 0.) {
        pixels[0] = 1e-3;
    }
    pixels
}

fn preprocess(image: &RgbImage, edge: usize) -> Result<(Vec<Vec<f32>>, (usize, usize)), String> {
    let layout = image_layout(image.width() as usize, image.height() as usize, edge)?;
    // transformers 5.19 + torchvision < 0.27 uses antialiased bicubic for the package's Lanczos setting.
    let image = resize(image, layout.width, layout.height)?;
    let mut frames = Vec::new();
    if layout.rows == 0 {
        frames.push(normalize(&resize(&image, 512, 512)?));
    } else {
        let image = resize(&image, layout.cols * 512, layout.rows * 512)?;
        for row in 0..layout.rows {
            for col in 0..layout.cols {
                frames.push(normalize(
                    &image::imageops::crop_imm(
                        &image,
                        (col * 512) as u32,
                        (row * 512) as u32,
                        512,
                        512,
                    )
                    .to_image(),
                ));
            }
        }
        frames.push(normalize(&resize(&image, 512, 512)?));
    }
    Ok((frames, (layout.rows, layout.cols)))
}

fn prefix_string(grid: (usize, usize), image_seq_len: usize) -> String {
    let token_run = "<image>".repeat(image_seq_len);
    let (rows, cols) = grid;
    let mut out = String::new();
    for row in 1..=rows {
        for col in 1..=cols {
            out.push_str(&format!(
                "<fake_token_around_image><row_{row}_col_{col}>{token_run}"
            ));
        }
        out.push('\n');
    }
    if rows > 0 {
        out.push('\n');
    }
    out.push_str(&format!(
        "<fake_token_around_image><global-img>{token_run}<fake_token_around_image>"
    ));
    out
}

struct Vocab {
    tokenizer: tokenizers::Tokenizer,
    question: u32,
    sep: u32,
    image: u32,
}

impl Vocab {
    fn load(root: &Path) -> Result<Self, String> {
        let mut tokenizer = tokenizers::Tokenizer::from_file(root.join("tokenizer.json"))
            .map_err(|e| format!("Load UltraVBERT tokenizer: {e}"))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| format!("Configure UltraVBERT tokenizer: {e}"))?;
        let id = |token: &str| {
            tokenizer
                .token_to_id(token)
                .ok_or_else(|| format!("UltraVBERT tokenizer missing {token}"))
        };
        let (question, sep, image) = (id("?")?, id("<|sep|>")?, id("<image>")?);
        Ok(Self {
            tokenizer,
            question,
            sep,
            image,
        })
    }
    fn raw(&self, text: &str, special: bool) -> Result<tokenizers::Encoding, String> {
        self.tokenizer
            .encode(text, special)
            .map_err(|e| format!("UltraVBERT tokenization: {e}"))
    }
    fn ids(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(self
            .raw(text, false)?
            .get_ids()
            .iter()
            .map(|&id| if id >= 49152 { self.question } else { id })
            .collect())
    }
}

#[derive(Debug)]
struct Example {
    ids: Vec<u32>,
    frames: Vec<Vec<f32>>,
    selections: Vec<(usize, usize)>,
    offsets: Vec<(usize, usize)>,
    grid: (usize, usize),
    names: Vec<String>,
    text: String,
    threshold: f64,
    task: &'static str,
    question_type: String,
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn as_text(v: &Value) -> String {
    v.as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| v.to_string())
}

fn py_string(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        _ => as_text(v),
    }
}

fn criteria_names(v: &Value) -> Vec<String> {
    match v {
        Value::Object(o) => o.keys().cloned().collect(),
        Value::Array(a) => a.iter().map(py_string).collect(),
        _ => Vec::new(),
    }
}

fn flatten_fields(value: &Value, prefix: &str, out: &mut Vec<String>) {
    let entries: Vec<_> = match value {
        Value::Object(o) => o.iter().map(|(k, v)| (k.clone(), v)).collect(),
        Value::Array(a) => a
            .iter()
            .enumerate()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
        _ => return,
    };
    for (k, v) in entries {
        if v.is_object() || v.is_array() {
            flatten_fields(v, &format!("{prefix}{k}."), out);
        } else if !v.is_null() {
            out.push(format!("{prefix}{k}: {}", py_string(v)));
        }
    }
}

fn prepare(
    vocab: &Vocab,
    task_config: &Value,
    state: &Value,
    question: &Value,
    images: &[RgbImage],
) -> Result<Example, String> {
    let qtype = question["type"].as_str().unwrap_or("");
    if !["choice", "noul", "span", "box"].contains(&qtype) {
        return Err(format!(
            "unsupported question type {qtype:?} (choice, noul, span, box)"
        ));
    }
    let crit = question
        .get("criteria")
        .filter(|v| truthy(v))
        .or_else(|| question.get("labels").filter(|v| truthy(v)));
    let ocr = qtype == "span"
        && (question["task"] == "ocr_check"
            || crit.map(criteria_names).as_deref() == Some(&["ocr_error".into()]));
    let task = if ocr {
        "ocr"
    } else if qtype == "choice" || qtype == "noul" {
        "decision"
    } else {
        qtype
    };
    let task: &'static str = match task {
        "ocr" => "ocr",
        "decision" => "decision",
        "span" => "span",
        _ => "box",
    };
    let cfg = &task_config["tasks"][task];
    let edge = cfg["edge"].as_u64().unwrap_or(1024) as usize;
    if task == "decision" {
        let max = cfg["max_images"].as_u64().unwrap_or(3) as usize;
        if images.len() > max {
            return Err(format!(
                "this model reads at most {max} images per request (got {})",
                images.len()
            ));
        }
    } else if images.is_empty() {
        return Err(if ocr {
            "ocr_check questions need the scanned document as an image"
        } else if task == "span" {
            "span questions need an image"
        } else {
            "box questions need an image"
        }
        .into());
    }
    let mut e = Example {
        ids: vec![0],
        frames: Vec::new(),
        selections: Vec::new(),
        offsets: Vec::new(),
        grid: (0, 0),
        names: Vec::new(),
        text: String::new(),
        threshold: 0.,
        task,
        question_type: qtype.into(),
    };
    if !images.is_empty() {
        let mut pre = String::new();
        for image in images
            .iter()
            .take(if task == "decision" { images.len() } else { 1 })
        {
            let (frames, grid) = preprocess(image, edge)?;
            e.frames.extend(frames);
            e.grid = grid;
            pre.push_str(&prefix_string(grid, 64));
        }
        e.ids = vocab.raw(&pre, true)?.get_ids().to_vec();
    }
    if task == "decision" {
        let options: Vec<String>;
        if qtype == "noul" {
            e.names = vec!["no".into(), "yes".into()];
            options = vec!["No / false".into(), "Yes / true".into()];
        } else if let Some(o) = question["criteria"].as_object().filter(|o| !o.is_empty()) {
            e.names = o.keys().cloned().collect();
            options = o
                .iter()
                .map(|(k, v)| {
                    if truthy(v) {
                        format!("{k}: {}", py_string(v))
                    } else {
                        k.clone()
                    }
                })
                .collect();
        } else if let Some(a) = question["criteria"].as_array().filter(|a| !a.is_empty()) {
            e.names = a.iter().map(py_string).collect();
            options = e.names.clone();
        } else {
            return Err("choice questions need criteria: {option: description or null} or a list of options".into());
        }
        let state = if state.is_string() || truthy(state) {
            as_text(state)
        } else {
            String::new()
        };
        if !state.trim().is_empty() {
            let ids = vocab.ids(&format!("Input: {}", state.trim()))?;
            let max = cfg["max_state_tokens"].as_u64().unwrap_or(32768) as usize;
            if ids.len() > max {
                return Err(format!(
                    "input longer than the maximum model length (state {} > {max} tokens)",
                    ids.len()
                ));
            }
            e.ids.extend(ids);
        }
        let instructions = question
            .get("instructions")
            .filter(|v| truthy(v))
            .map(as_text)
            .unwrap_or_default();
        e.ids
            .extend(vocab.ids(&format!(" Question: {}", instructions.trim()))?);
        for option in options {
            e.ids.push(vocab.sep);
            let start = e.ids.len();
            e.ids.extend(vocab.ids(&format!(" {option}"))?);
            e.selections.push((start, e.ids.len()));
        }
    } else if task == "box" {
        let instructions = question
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");
        e.selections = e
            .ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == vocab.image).then_some((i, i + 1)))
            .collect();
        e.ids
            .extend(vocab.ids(&format!(" Locate: {}", instructions.trim()))?);
    } else {
        let context;
        if ocr {
            if let Some(text) = state.as_str() {
                e.text = text.into();
            } else if let Some(text) = state
                .get("ocr")
                .filter(|v| truthy(v))
                .or_else(|| state.get("answer"))
                .and_then(Value::as_str)
            {
                e.text = text.into();
            } else if state.is_object() {
                let mut fields = Vec::new();
                flatten_fields(state, "", &mut fields);
                e.text = fields.join("\n");
            } else {
                return Err("the state needs the OCR text".into());
            }
            context =
                "User request: Check this OCR extraction against the scanned document.".into();
            e.names = vec!["ocr_error".into()];
        } else {
            e.names = crit
                .map(criteria_names)
                .unwrap_or_else(|| vec!["unsupported".into()]);
            if e.names != ["unsupported"] {
                return Err("this preview answers one span label set: {\"unsupported\"} (image-grounded hallucination)".into());
            }
            let user;
            if let Some(text) = state.as_str() {
                user = String::new();
                e.text = text.into();
            } else if state.is_object() {
                user = state
                    .get("user")
                    .filter(|v| truthy(v))
                    .or_else(|| state.get("context"))
                    .filter(|v| truthy(v))
                    .map(py_string)
                    .unwrap_or_default();
                e.text = state
                    .get("answer")
                    .filter(|v| truthy(v))
                    .map(py_string)
                    .unwrap_or_default();
            } else {
                return Err("the state needs an answer to tag".into());
            }
            if e.text.is_empty() {
                return Err("the state needs an answer to tag".into());
            }
            context = format!("User request: {user}");
        }
        let threshold = question.get("threshold").unwrap_or(&cfg["threshold"]);
        e.threshold = super::text::threshold_value(threshold)?;
        e.ids
            .extend(vocab.ids(&format!("{}\nAnswer:", context.trim()))?);
        let encoded = vocab.raw(&e.text, false)?;
        let offsets = &encoded.get_offsets()[..encoded.len().min(768)];
        let base = e.ids.len();
        for (start, end) in super::text::split_words(&e.text) {
            if let Some(i) = offsets
                .iter()
                .position(|&(a, b)| b > a && b > start && a < end)
            {
                e.offsets.push((start, end));
                e.selections.push((base + i, base + i + 1));
            }
        }
        e.ids.extend(encoded.get_ids().iter().take(768).map(|&id| {
            if id >= 49152 {
                vocab.question
            } else {
                id
            }
        }));
    }
    e.ids.push(vocab.sep);
    if e.ids.len() > 32768 {
        return Err(format!(
            "input longer than the maximum model length ({} > 32768 tokens)",
            e.ids.len()
        ));
    }
    if e.selections.iter().any(|&(a, b)| a >= b || b > e.ids.len()) {
        return Err("option has no tokens".into());
    }
    Ok(e)
}

struct EncSource<'a>(&'a dyn TensorSource);
impl TensorSource for EncSource<'_> {
    fn metadata(&self, k: &str) -> Option<&MetaValue> {
        self.0.metadata(k)
    }
    fn tensor_info(&self, k: &str) -> Option<&TensorInfo> {
        self.0.tensor_info(k.strip_prefix("enc.").unwrap_or(k))
    }
    fn tensor_slice(&self, k: &str) -> Option<&[u8]> {
        self.0.tensor_slice(k.strip_prefix("enc.").unwrap_or(k))
    }
}

fn normalized(
    source: &dyn TensorSource,
    prefix: &str,
    x: &[f32],
    eps: f32,
) -> Result<Vec<f32>, String> {
    layer_norm(
        x,
        &values(source, &format!("{prefix}.weight"))?,
        &values(source, &format!("{prefix}.bias"))?,
        eps,
    )
}

fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tokens: usize,
    hidden: usize,
    heads: usize,
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    if heads == 0
        || hidden % heads != 0
        || q.len() != tokens * hidden
        || k.len() != q.len()
        || v.len() != q.len()
    {
        return Err("invalid SigLIP attention shape".into());
    }
    let dim = hidden / heads;
    let scale = (dim as f32).sqrt().recip();
    let mut out = vec![0.; q.len()];
    let pointer = out.as_mut_ptr() as usize;
    pool.compute(|ith, nth| {
        let mut probabilities = vec![0.; tokens];
        for row in (ith..tokens).step_by(nth) {
            // Each worker owns complete, disjoint output rows.
            let out = unsafe {
                std::slice::from_raw_parts_mut((pointer as *mut f32).add(row * hidden), hidden)
            };
            for head in 0..heads {
                let start = row * hidden + head * dim;
                let query = &q[start..start + dim];
                for token in 0..tokens {
                    let start = token * hidden + head * dim;
                    probabilities[token] =
                        crate::ops::dot_f32(query, &k[start..start + dim], dim) * scale;
                }
                let max = probabilities
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.;
                for p in &mut probabilities {
                    *p = (*p - max).exp();
                    sum += *p;
                }
                let dst = &mut out[head * dim..(head + 1) * dim];
                for token in 0..tokens {
                    let start = token * hidden + head * dim;
                    crate::ops::vec_mad_f32(
                        dst,
                        &v[start..start + dim],
                        probabilities[token] / sum,
                    );
                }
            }
        }
    });
    Ok(out)
}

fn encode_vision(
    source: &dyn TensorSource,
    config: &Value,
    pixels: &[f32],
    pool: &ComputePool,
    mut trace: Option<&mut Trace>,
) -> Result<Vec<f32>, String> {
    let cfg = &config["vision_config"];
    let (size, patch, hidden, heads) = (
        dimension(cfg, "image_size")?,
        dimension(cfg, "patch_size")?,
        dimension(cfg, "hidden_size")?,
        dimension(cfg, "num_attention_heads")?,
    );
    let layers = cfg["num_hidden_layers"]
        .as_u64()
        .ok_or("missing SigLIP layers")? as usize;
    let channels = dimension(cfg, "num_channels")?;
    if size % patch != 0 || channels != 3 || pixels.len() != channels * size * size {
        return Err("invalid SigLIP pixel shape".into());
    }
    let prefix = if source
        .tensor_info("enc.vision_model.embeddings.patch_embedding.weight")
        .is_some()
    {
        "enc.vision_model."
    } else {
        "enc.vision_model.vision_model."
    };
    let weight_name = format!("{prefix}embeddings.patch_embedding.weight");
    if shape(source, &weight_name)? != [hidden, channels, patch, patch] {
        return Err("invalid SigLIP patch weight shape".into());
    }
    let (weight, bias, positions) = (
        values(source, &weight_name)?,
        values(source, &format!("{prefix}embeddings.patch_embedding.bias"))?,
        values(
            source,
            &format!("{prefix}embeddings.position_embedding.weight"),
        )?,
    );
    let side = size / patch;
    let tokens = side * side;
    let patch_dim = channels * patch * patch;
    if bias.len() != hidden || positions.len() != tokens * hidden {
        return Err("invalid SigLIP patch bias or positions".into());
    }
    let mut h = vec![0.; tokens * hidden];
    let pointer = h.as_mut_ptr() as usize;
    pool.compute(|ith, nth| {
        let mut input = vec![0.; patch_dim];
        for token in (ith..tokens).step_by(nth) {
            for c in 0..channels {
                for y in 0..patch {
                    let start =
                        c * size * size + (token / side * patch + y) * size + token % side * patch;
                    let dst = (c * patch + y) * patch;
                    input[dst..dst + patch].copy_from_slice(&pixels[start..start + patch]);
                }
            }
            let row = unsafe {
                std::slice::from_raw_parts_mut((pointer as *mut f32).add(token * hidden), hidden)
            };
            for i in 0..hidden {
                row[i] = crate::ops::dot_f32(
                    &input,
                    &weight[i * patch_dim..(i + 1) * patch_dim],
                    patch_dim,
                ) + bias[i]
                    + positions[token * hidden + i];
            }
        }
    });
    record(&mut trace, "patch_embeddings", &h);
    let eps = cfg["layer_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
    for layer in 0..layers {
        let p = format!("{prefix}encoder.layers.{layer}");
        let norm = normalized(source, &format!("{p}.layer_norm1"), &h, eps)?;
        let q = linear(
            source,
            &format!("{p}.self_attn.q_proj.weight"),
            &norm,
            tokens,
            pool,
        )?;
        let k = linear(
            source,
            &format!("{p}.self_attn.k_proj.weight"),
            &norm,
            tokens,
            pool,
        )?;
        let v = linear(
            source,
            &format!("{p}.self_attn.v_proj.weight"),
            &norm,
            tokens,
            pool,
        )?;
        let a = attention(&q, &k, &v, tokens, hidden, heads, pool)?;
        let a = linear(
            source,
            &format!("{p}.self_attn.out_proj.weight"),
            &a,
            tokens,
            pool,
        )?;
        if a.len() != h.len() {
            return Err("invalid SigLIP output projection".into());
        }
        for (h, a) in h.iter_mut().zip(a) {
            *h += a;
        }
        let norm = normalized(source, &format!("{p}.layer_norm2"), &h, eps)?;
        let mut up = linear(source, &format!("{p}.mlp.fc1.weight"), &norm, tokens, pool)?;
        match cfg["hidden_act"].as_str().unwrap_or("gelu_pytorch_tanh") {
            "gelu_pytorch_tanh" | "gelu_new" => {
                for x in &mut up {
                    *x = crate::ops::gelu(*x);
                }
            }
            "gelu" => {
                for x in &mut up {
                    *x = crate::ops::gelu_erf(*x);
                }
            }
            other => return Err(format!("unsupported SigLIP activation {other}")),
        }
        let down = linear(source, &format!("{p}.mlp.fc2.weight"), &up, tokens, pool)?;
        if down.len() != h.len() {
            return Err("invalid SigLIP MLP shape".into());
        }
        for (h, x) in h.iter_mut().zip(down) {
            *h += x;
        }
        record(&mut trace, &format!("vision_layer_{layer}"), &h);
    }
    h = normalized(source, &format!("{prefix}post_layernorm"), &h, eps)?;
    record(&mut trace, "vision_hidden", &h);
    let factor = dimension(config, "pixel_shuffle_factor")?;
    let shuffled = pixel_shuffle(&h, side, hidden, factor)?;
    record(&mut trace, "pixel_shuffle", &shuffled);
    let projected = linear(
        source,
        "enc.connector.modality_projection.weight",
        &shuffled,
        tokens / (factor * factor),
        pool,
    )?;
    let projected = if config["connector_norm"].as_bool().unwrap_or(true) {
        rms_norm(
            &projected,
            &values(source, "enc.connector.norm.weight")?,
            config["text_config"]["rms_norm_eps"]
                .as_f64()
                .unwrap_or(1e-6) as f32,
        )?
    } else {
        projected
    };
    record(&mut trace, "image_embeddings", &projected);
    Ok(projected)
}

struct Task {
    base: SafetensorSource,
    adapter: SafetensorSource,
    scale: f32,
}

impl Task {
    fn load(root: &Path, config: &Value, name: &str) -> Result<Self, String> {
        let cfg = &config["tasks"][name];
        let base = match cfg["backbone_file"].as_str() {
            Some(file) => SafetensorSource::open(&[root.join(file)])?,
            None => tensor::open_weights(root)?,
        };
        let file = cfg["file"]
            .as_str()
            .ok_or("missing UltraVBERT adapter file")?;
        let adapter = SafetensorSource::open(&[root.join(file)])?;
        let rank = cfg["lora_rank"].as_f64().unwrap_or(0.);
        let scale = if rank > 0. {
            cfg["lora_alpha"].as_f64().unwrap_or(0.) as f32 / rank as f32
        } else {
            0.
        };
        Ok(Self {
            base,
            adapter,
            scale,
        })
    }
    fn forward(
        &self,
        config: &Value,
        e: &Example,
        image_id: u32,
        pool: &ComputePool,
        mut trace: Option<&mut Trace>,
    ) -> Result<Vec<f32>, String> {
        let base = EncSource(&self.base);
        let source = tensor::OverlaySource::new(&base, &self.adapter);
        let hidden = dimension(&config["text_config"], "hidden_size")?;
        let mut embeddings = embed(&source, "enc.text_model.embed_tokens.weight", &e.ids)?;
        let mut image_values = Vec::new();
        for pixels in &e.frames {
            image_values.extend(encode_vision(
                &source,
                config,
                pixels,
                pool,
                trace.as_deref_mut(),
            )?);
        }
        let positions: Vec<_> = e
            .ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == image_id).then_some(i))
            .collect();
        if image_values.len() != positions.len() * hidden {
            return Err("UltraVBERT image-token count does not match connector embeddings".into());
        }
        for (i, &position) in positions.iter().enumerate() {
            embeddings[position * hidden..(position + 1) * hidden]
                .copy_from_slice(&image_values[i * hidden..(i + 1) * hidden]);
        }
        record(&mut trace, "merged_embeddings", &embeddings);
        let h = super::ultrabert::encode_adapted(
            &base,
            &self.adapter,
            self.scale,
            &config["text_config"],
            "enc.text_model.",
            &e.ids,
            Some(&embeddings),
            pool,
        )?;
        record(&mut trace, "text_hidden", &h);
        let mut selected = vec![0.; e.selections.len() * hidden];
        for (row, &(a, b)) in e.selections.iter().enumerate() {
            let selected = &mut selected[row * hidden..(row + 1) * hidden];
            for token in a..b {
                for col in 0..hidden {
                    selected[col] += h[token * hidden + col];
                }
            }
            for value in selected {
                *value /= (b - a) as f32;
            }
        }
        record(&mut trace, "selected_hidden", &selected);
        let rows = e.selections.len();
        if rows == 0 {
            return Ok(Vec::new());
        }
        let mut h = linear(&self.adapter, "head.net.0.weight", &selected, rows, pool)?;
        for x in &mut h {
            *x = crate::ops::gelu_erf(*x);
        }
        h = normalized(&self.adapter, "head.net.2", &h, 1e-5)?;
        let logits = linear(&self.adapter, "head.net.3.weight", &h, rows, pool)?;
        if logits.len() != rows {
            return Err("invalid UltraVBERT task-head shape".into());
        }
        record(&mut trace, "logits", &logits);
        Ok(logits)
    }
}

fn answer(e: &Example, logits: &[f32], task_config: &Value) -> Result<Value, String> {
    if logits.len() != e.selections.len() || logits.iter().any(|v| !v.is_finite()) {
        return Err("invalid UltraVBERT task logits".into());
    }
    if e.task == "decision" {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut p: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
        let total: f32 = p.iter().sum();
        for x in &mut p {
            *x /= total;
        }
        if e.question_type == "noul" {
            return Ok(json!({"type":"noul","noul":p[1]}));
        }
        let mut probabilities = Map::new();
        for (name, &p0) in e.names.iter().zip(&p) {
            probabilities.insert(name.clone(), json!(p0));
        }
        let mut best = e.names.first().ok_or("choice needs an option")?;
        for name in e.names.iter().skip(1) {
            if probabilities[name].as_f64().unwrap() > probabilities[best].as_f64().unwrap() {
                best = name;
            }
        }
        return Ok(json!({"type":"choice","choice":best,"probabilities":probabilities}));
    }
    let p: Vec<_> = logits.iter().map(|x| 1. / (1. + (-x).exp())).collect();
    if e.task == "box" {
        let (h, height, width) = heat(&p, e.grid)?;
        let (b, point, confidence) = region(
            &h,
            height,
            width,
            task_config["tasks"]["box"]["frac"].as_f64().unwrap_or(0.7) as f32,
        )?;
        return Ok(
            json!({"type":"box","box":{"top":b[0],"bottom":b[1],"left":b[2],"right":b[3]},"point":{"y":point[0],"x":point[1]},"confidence":confidence}),
        );
    }
    // Outside [0, 1], clamping preserves all comparisons without overflowing the F32 decoder.
    let threshold = e.threshold.clamp(-1., 2.) as f32;
    let spans = super::text::decode_spans(&e.text, &e.offsets, &p, &e.names, threshold)?;
    Ok(
        json!({"type":"span","target":if e.task=="ocr" {"ocr"} else {"answer"},"spans":spans,"head":if e.task=="ocr" {"ocr_check"} else {"image_halu"},"threshold":e.threshold}),
    )
}

pub(crate) fn predict(root: &Path, request: &Value, pool: &ComputePool) -> Result<Value, String> {
    predict_traced(root, request, pool, None)
}

fn predict_traced(
    root: &Path,
    request: &Value,
    pool: &ComputePool,
    mut trace: Option<&mut Trace>,
) -> Result<Value, String> {
    let config = read_json(&root.join("config.json"))?;
    let task_config = read_json(&root.join("vela3v_config.json"))?;
    if config["model_type"] != "ultravbert" {
        return Err("Vela3 vision requires the UltraVBERT architecture".into());
    }
    let vocab = Vocab::load(root)?;
    let questions = request["questions"]
        .as_object()
        .ok_or("questions must be an object")?;
    let images = match request.get("images") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a.iter().map(load_image).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("images must be an array".into()),
    };
    let state = &request["state"];
    let mut tasks = HashMap::new();
    let mut answers = Map::new();
    let mut tokens = 0usize;
    for (key, question) in questions {
        let e = match prepare(&vocab, &task_config, state, question, &images) {
            Ok(e) => e,
            Err(message) => {
                answers.insert(key.clone(),json!({"type":question.get("type").unwrap_or(&Value::Null),"error":"invalid_question","message":message}));
                continue;
            }
        };
        if !tasks.contains_key(e.task) {
            tasks.insert(e.task, Task::load(root, &task_config, e.task)?);
        }
        let logits = tasks[e.task].forward(&config, &e, vocab.image, pool, trace.as_deref_mut())?;
        answers.insert(key.clone(), answer(&e, &logits, &task_config)?);
        tokens += e.ids.len();
    }
    Ok(
        json!({"model":format!("vllm-sr/{}",task_config["name"].as_str().unwrap_or("vela3-fast-vision")),"answers":answers,"usage":{"input_tokens":tokens,"output_tokens":0}}),
    )
}

#[derive(Debug, PartialEq)]
struct ImageLayout {
    width: usize,
    height: usize,
    rows: usize,
    cols: usize,
}

fn image_layout(width: usize, height: usize, edge: usize) -> Result<ImageLayout, String> {
    if width == 0 || height == 0 || edge == 0 || edge > 3072 {
        return Err("invalid UltraVBERT image dimensions or edge".into());
    }
    let aspect = width as f64 / height as f64;
    let (width, height) = if width >= height {
        let h = (edge as f64 / aspect) as usize;
        (edge, h.max(1) + usize::from(h % 2 != 0))
    } else {
        let w = (edge as f64 * aspect) as usize;
        (w.max(1) + usize::from(w % 2 != 0), edge)
    };
    let (rows, cols) = if edge <= 512 {
        (0, 0)
    } else if width >= height {
        let cols = width.div_ceil(512);
        let rows = (((cols * 512) as f64 / (width as f64 / height as f64)) as usize)
            .div_ceil(512)
            .max(1);
        (rows, cols)
    } else {
        let rows = height.div_ceil(512);
        let cols = (((rows * 512) as f64 * (width as f64 / height as f64)) as usize)
            .div_ceil(512)
            .max(1);
        (rows, cols)
    };
    Ok(ImageLayout {
        width,
        height,
        rows,
        cols,
    })
}

fn pixel_shuffle(x: &[f32], side: usize, hidden: usize, factor: usize) -> Result<Vec<f32>, String> {
    if factor == 0
        || side == 0
        || hidden == 0
        || side % factor != 0
        || x.len() != side * side * hidden
    {
        return Err("invalid UltraVBERT pixel-shuffle shape".into());
    }
    let mut output = Vec::with_capacity(x.len());
    for y in (0..side).step_by(factor) {
        for x0 in (0..side).step_by(factor) {
            for dy in 0..factor {
                let start = ((y + dy) * side + x0) * hidden;
                output.extend_from_slice(&x[start..start + factor * hidden]);
            }
        }
    }
    Ok(output)
}

fn heat(p: &[f32], grid: (usize, usize)) -> Result<(Vec<f32>, usize, usize), String> {
    let (rows, cols) = grid;
    if rows == 0 && cols == 0 && p.len() == 64 {
        return Ok((p.to_vec(), 8, 8));
    }
    if rows == 0 || cols == 0 || p.len() != (rows * cols + 1) * 64 {
        return Err("invalid UltraVBERT box heat shape".into());
    }
    let (height, width) = (rows * 8, cols * 8);
    let mut out = vec![0.; height * width];
    for y in 0..height {
        for x in 0..width {
            let local = ((y / 8) * cols + x / 8) * 64 + (y % 8) * 8 + x % 8;
            let global = rows * cols * 64 + (y * 8 / height) * 8 + x * 8 / width;
            out[y * width + x] = 0.5 * (p[local] + p[global]);
        }
    }
    Ok((out, height, width))
}

fn region(
    h: &[f32],
    height: usize,
    width: usize,
    frac: f32,
) -> Result<([f32; 4], [f32; 2], f32), String> {
    if width == 0
        || height == 0
        || h.len() != width * height
        || h.iter().any(|v| !v.is_finite())
        || !frac.is_finite()
        || !(0. ..=1.).contains(&frac)
    {
        return Err("invalid UltraVBERT region heat or fraction".into());
    }
    let mut peak = 0;
    for i in 1..h.len() {
        if h[i] > h[peak] {
            peak = i;
        }
    }
    let cutoff = h[peak] * frac;
    let mut pending = vec![peak];
    let mut visited = vec![false; h.len()];
    visited[peak] = true;
    let (mut top, mut bottom, mut left, mut right) =
        (peak / width, peak / width, peak % width, peak % width);
    while let Some(i) = pending.pop() {
        let (y, x) = (i / width, i % width);
        top = top.min(y);
        bottom = bottom.max(y);
        left = left.min(x);
        right = right.max(x);
        for next in [
            y.checked_sub(1).map(|yy| yy * width + x),
            (y + 1 < height).then_some((y + 1) * width + x),
            x.checked_sub(1).map(|xx| y * width + xx),
            (x + 1 < width).then_some(y * width + x + 1),
        ]
        .into_iter()
        .flatten()
        {
            if !visited[next] && h[next] >= cutoff {
                visited[next] = true;
                pending.push(next);
            }
        }
    }
    Ok((
        [
            top as f32 / height as f32,
            (bottom + 1) as f32 / height as f32,
            left as f32 / width as f32,
            (right + 1) as f32 / width as f32,
        ],
        [
            (peak / width) as f32 / height as f32 + 0.5 / height as f32,
            (peak % width) as f32 / width as f32 + 0.5 / width as f32,
        ],
        h[peak],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vocab {
        let tokenizer=tokenizers::Tokenizer::from_bytes(br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"?":1,"<|sep|>":2,"<image>":3},"unk_token":"[UNK]"}}"#).unwrap();
        Vocab {
            tokenizer,
            question: 1,
            sep: 2,
            image: 3,
        }
    }

    #[test]
    fn questions_refuse_missing_criteria_and_excess_images_without_truncating() {
        let v = vocab();
        let cfg = json!({"tasks":{"decision":{"max_images":3,"max_state_tokens":2,"edge":1024}}});
        assert!(prepare(
            &v,
            &cfg,
            &json!(""),
            &json!({"type":"choice","criteria":[]}),
            &[]
        )
        .unwrap_err()
        .contains("criteria"));
        let images = vec![RgbImage::new(1, 1); 4];
        assert!(
            prepare(&v, &cfg, &Value::Null, &json!({"type":"noul"}), &images)
                .unwrap_err()
                .contains("at most 3 images")
        );
        assert!(
            prepare(&v, &cfg, &json!("a b c"), &json!({"type":"noul"}), &[])
                .unwrap_err()
                .contains("state")
        );
        assert!(
            prepare(&v, &cfg, &json!("answer"), &json!({"type":"span"}), &[])
                .unwrap_err()
                .contains("need an image")
        );
    }

    #[test]
    fn contract_thresholds_accept_finite_python_numeric_inputs() {
        let v = vocab();
        let cfg = json!({"tasks":{"span":{"edge":512,"threshold":0.05}}});
        let images = [RgbImage::new(1, 1)];
        for (threshold, expected, empty) in [
            (json!(" -1 "), -1., false),
            (json!(2), 2., true),
            (json!("0.125"), 0.125, true),
            (json!("0.1"), 0.1, false),
            (json!(true), 1., true),
            (json!(1e300), 1e300, true),
            (json!("-1e300"), -1e300, false),
        ] {
            let e = prepare(
                &v,
                &cfg,
                &json!("answer"),
                &json!({"type":"span","threshold":threshold}),
                &images,
            )
            .unwrap();
            assert_eq!(e.threshold, expected);
            let output = answer(&e, &vec![-2.; e.selections.len()], &cfg).unwrap();
            assert_eq!(output["threshold"].as_f64().unwrap(), expected);
            assert_eq!(output["spans"].as_array().unwrap().is_empty(), empty);
        }
        for threshold in [json!("NaN"), json!("inf"), json!("-inf"), Value::Null] {
            assert!(prepare(
                &v,
                &cfg,
                &json!("answer"),
                &json!({"type":"span","threshold":threshold}),
                &images
            )
            .is_err());
        }
    }

    #[test]
    fn contract_duplicate_choices_select_from_final_probabilities() {
        let v = vocab();
        let cfg = json!({"tasks":{"decision":{}}});
        let e = prepare(
            &v,
            &cfg,
            &Value::Null,
            &json!({"type":"choice","criteria":["a","b","a"]}),
            &[],
        )
        .unwrap();
        let output = answer(&e, &[4., 2., 0.], &cfg).unwrap();
        assert_eq!(output["choice"], "b");
        assert!(
            output["probabilities"]["b"].as_f64().unwrap()
                > output["probabilities"]["a"].as_f64().unwrap()
        );
        let e = prepare(
            &v,
            &cfg,
            &Value::Null,
            &json!({"type":"choice","criteria":["z","a","z"]}),
            &[],
        )
        .unwrap();
        assert_eq!(answer(&e, &[8., 0., 0.], &cfg).unwrap()["choice"], "z");
    }

    #[test]
    fn layouts_match_official_two_stage_resize_and_tiling() {
        assert_eq!(
            image_layout(1024, 1024, 1024).unwrap(),
            ImageLayout {
                width: 1024,
                height: 1024,
                rows: 2,
                cols: 2
            }
        );
        assert_eq!(
            image_layout(1000, 200, 1024).unwrap(),
            ImageLayout {
                width: 1024,
                height: 204,
                rows: 1,
                cols: 2
            }
        );
        assert_eq!(
            image_layout(200, 1000, 512).unwrap(),
            ImageLayout {
                width: 102,
                height: 512,
                rows: 0,
                cols: 0
            }
        );
        assert!(image_layout(0, 100, 1024).is_err());
    }

    #[test]
    fn connector_shuffle_keeps_vertical_then_horizontal_patch_order() {
        let x: Vec<_> = (0..16).map(|n| n as f32).collect();
        assert_eq!(
            pixel_shuffle(&x, 4, 1, 2).unwrap(),
            [0., 1., 4., 5., 2., 3., 6., 7., 8., 9., 12., 13., 10., 11., 14., 15.]
        );
        assert!(pixel_shuffle(&x, 4, 1, 3).is_err());
    }

    #[test]
    fn heat_combines_local_and_global_and_region_uses_four_neighbors() {
        let mut p = vec![0.2; 64];
        p.extend(vec![0.4; 64]);
        p.extend(vec![0.6; 64]);
        let (h, height, width) = heat(&p, (1, 2)).unwrap();
        assert_eq!((height, width), (8, 16));
        assert!((h[0] - 0.4).abs() < 1e-6);
        assert!((h[15] - 0.5).abs() < 1e-6);
        let (b, pt, confidence) = region(&[0.9, 0., 0., 0.8], 2, 2, 0.7).unwrap();
        assert_eq!(b, [0., 0.5, 0., 0.5]);
        assert_eq!(pt, [0.25, 0.25]);
        assert_eq!(confidence, 0.9);
    }

    fn f32_file(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len() % 4, 0);
        bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect()
    }

    fn compare(
        label: &str,
        actual: &[f32],
        expected: &[f32],
        absolute: f32,
        peak_relative: f32,
        rms_relative: f32,
    ) -> bool {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        assert!(
            actual.iter().chain(expected).all(|v| v.is_finite()),
            "{label} finite values"
        );
        let max = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        let rmse = (actual
            .iter()
            .zip(expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / actual.len().max(1) as f64)
            .sqrt();
        let peak = expected.iter().map(|v| v.abs()).fold(0., f32::max);
        let rms = (expected.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>()
            / expected.len().max(1) as f64)
            .sqrt();
        eprintln!(
            "{label}: values={} max_abs={max:.9} rmse={rmse:.9} ref_peak={peak:.9} ref_rms={rms:.9} normalized_rmse={:.9}",
            actual.len(), rmse / rms.max(1e-30)
        );
        max <= absolute + peak_relative * peak
            && rmse <= f64::from(absolute) + f64::from(rms_relative) * rms
    }

    fn compare_output(path: &str, actual: &Value, expected: &Value) {
        match expected {
            Value::Object(e) => {
                let a = actual
                    .as_object()
                    .unwrap_or_else(|| panic!("{path}: expected object"));
                assert_eq!(
                    a.keys().collect::<Vec<_>>(),
                    e.keys().collect::<Vec<_>>(),
                    "{path} keys"
                );
                for (key, value) in e {
                    compare_output(&format!("{path}.{key}"), &a[key], value);
                }
            }
            Value::Array(e) => {
                let a = actual
                    .as_array()
                    .unwrap_or_else(|| panic!("{path}: expected array"));
                assert_eq!(a.len(), e.len(), "{path} length");
                for (i, (a, e)) in a.iter().zip(e).enumerate() {
                    compare_output(&format!("{path}[{i}]"), a, e);
                }
            }
            Value::Number(e) if e.is_f64() => {
                let a = actual
                    .as_f64()
                    .unwrap_or_else(|| panic!("{path}: expected number"));
                let e = e.as_f64().unwrap();
                assert!(
                    a.is_finite() && (a - e).abs() <= 5e-4 + 5e-4 * e.abs(),
                    "{path}: {a} != {e}"
                );
            }
            _ => assert_eq!(actual, expected, "{path}"),
        }
    }

    #[test]
    #[ignore = "requires RMI_VELA3_VISION_FIXTURE generated by tools/oracle/vela3/vision_oracle.py"]
    fn official_fixture() {
        let fixture = std::path::PathBuf::from(std::env::var("RMI_VELA3_VISION_FIXTURE").unwrap());
        let manifest = read_json(&fixture.join("fixture.json")).unwrap();
        let source_root = Path::new(manifest["source_root"].as_str().unwrap());
        let vocab = Vocab::load(source_root).unwrap();
        let image = image::open(fixture.join("image.png")).unwrap().to_rgb8();
        for layout in manifest["layouts"].as_array().unwrap() {
            let edge = layout["edge"].as_u64().unwrap() as usize;
            let (frames, grid) = preprocess(&image, edge).unwrap();
            assert_eq!(json!([grid.0, grid.1]), layout["grid"]);
            let ids = vocab
                .raw(&prefix_string(grid, 64), true)
                .unwrap()
                .get_ids()
                .to_vec();
            assert_eq!(json!(ids), layout["input_ids"]);
            let expected = f32_file(&fixture.join(layout["pixels"]["file"].as_str().unwrap()));
            assert!(compare(
                &format!("preprocess_{edge}"),
                &frames.into_iter().flatten().collect::<Vec<_>>(),
                &expected,
                1e-6,
                0.,
                0.,
            ));
        }
        let task_cfg = read_json(&source_root.join("vela3v_config.json")).unwrap();
        for case in manifest["assembly"].as_array().into_iter().flatten() {
            let count = case["images"].as_u64().unwrap() as usize;
            let e = prepare(
                &vocab,
                &task_cfg,
                &case["state"],
                &case["question"],
                &vec![image.clone(); count],
            )
            .unwrap();
            assert_eq!(e.task, case["task"].as_str().unwrap());
            assert_eq!(json!(e.ids), case["input_ids"], "{} input assembly", e.task);
            assert_eq!(
                json!(e.selections),
                case["selections"],
                "{} readout positions",
                e.task
            );
            let character_offsets: Vec<_> = e
                .offsets
                .iter()
                .map(|&(a, b)| (e.text[..a].chars().count(), e.text[..b].chars().count()))
                .collect();
            assert_eq!(
                json!(character_offsets),
                case["offsets"],
                "{} Unicode offsets",
                e.task
            );
        }
        let name = manifest["task"].as_str().unwrap();
        let task_name = match name {
            "span" => "span",
            "box" => "box",
            "ocr" => "ocr",
            _ => "decision",
        };
        let e = Example {
            ids: manifest["input_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect(),
            frames: manifest["pixels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| f32_file(&fixture.join(p["file"].as_str().unwrap())))
                .collect(),
            selections: manifest["selections"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| {
                    (
                        v[0].as_u64().unwrap() as usize,
                        v[1].as_u64().unwrap() as usize,
                    )
                })
                .collect(),
            offsets: Vec::new(),
            grid: (0, 0),
            names: Vec::new(),
            text: String::new(),
            threshold: 0.,
            task: task_name,
            question_type: "choice".into(),
        };
        let mut trace = Vec::new();
        let pool = ComputePool::new(4);
        if let Some(request) = manifest.get("request") {
            let output = predict_traced(source_root, request, &pool, Some(&mut trace)).unwrap();
            eprintln!("public_system_one: {output}");
            compare_output("system_one", &output, &manifest["expected"]);
        } else {
            let task = Task {
                base: SafetensorSource::open(&[manifest["backbone"].as_str().unwrap()]).unwrap(),
                adapter: SafetensorSource::open(&[manifest["adapter"].as_str().unwrap()]).unwrap(),
                scale: manifest["lora_scale"].as_f64().unwrap() as f32,
            };
            task.forward(
                &manifest["config"],
                &e,
                manifest["config"]["image_token_id"].as_u64().unwrap() as u32,
                &pool,
                Some(&mut trace),
            )
            .unwrap();
        }
        let expected = manifest["checkpoints"].as_array().unwrap();
        assert_eq!(trace.len(), expected.len());
        let real = dimension(&manifest["config"]["text_config"], "hidden_size").unwrap() > 8;
        let mut matches = true;
        for ((name, actual), expected) in trace.iter().zip(expected) {
            assert!(expected["name"].as_str().unwrap().ends_with(name));
            // Real F32 reductions amplify at image/sentinel positions; task readouts keep a tighter bound.
            let (absolute, peak_relative, rms_relative) = match (real, name.as_str()) {
                (false, _) => (2e-5, 0., 0.),
                (true, "text_hidden") => (2e-5, 3e-3, 2e-3),
                (true, "selected_hidden" | "logits") => (5e-4, 5e-4, 5e-4),
                (true, _) => (2e-5, 1e-4, 1e-4),
            };
            matches &= compare(
                expected["name"].as_str().unwrap(),
                actual,
                &f32_file(&fixture.join(expected["file"].as_str().unwrap())),
                absolute,
                peak_relative,
                rms_relative,
            );
        }
        assert!(
            matches,
            "official vision checkpoints exceed numerical tolerance"
        );
    }
}
