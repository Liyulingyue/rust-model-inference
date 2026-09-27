pub mod request;

use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use crate::ops::gelu_erf;
use serde_json::{json, Value};
use std::collections::HashMap;

const D: usize = 768;
const H: usize = 12;
const HD: usize = 64;
const F: usize = 1152;
const LAYERS: usize = 22;

pub struct LayaModel {
    weights: HashMap<String, Vec<f32>>,
    tokenizer: tokenizers::Tokenizer,
    max_len: usize,
    head_max_len: usize,
    temperatures: [f32; 3],
}

impl LayaModel {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let arch = source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            .unwrap_or_default();
        if arch != "laya" {
            return Err(format!("Expected laya architecture, got {arch:?}"));
        }
        let encoder = source
            .metadata("laya.encoder_config")
            .and_then(MetaValue::to_string_val)
            .ok_or("Missing laya.encoder_config")?;
        let cfg: Value = serde_json::from_str(encoder).map_err(|e| e.to_string())?;
        if cfg["hidden_size"] != D
            || cfg["num_hidden_layers"] != LAYERS
            || cfg["num_attention_heads"] != H
            || cfg["intermediate_size"] != F
        {
            return Err("Unsupported Laya encoder dimensions".into());
        }
        let agent: Value = serde_json::from_str(
            source
                .metadata("laya.agent_config")
                .and_then(MetaValue::to_string_val)
                .ok_or("Missing laya.agent_config")?,
        )
        .map_err(|e| e.to_string())?;
        let tokenizer_json = source
            .metadata("laya.tokenizer_json")
            .and_then(MetaValue::to_string_val)
            .ok_or("Missing laya.tokenizer_json")?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_json.as_bytes())
            .map_err(|e| e.to_string())?;
        let mut weights = HashMap::new();
        let names = tensor_names();
        for (name, dims) in names {
            let info = source
                .tensor_info(&name)
                .ok_or_else(|| format!("Missing tensor {name}"))?;
            if info.dims != dims || info.ggml_type != GGMLType::F32 {
                return Err(format!("Invalid tensor shape {name}"));
            }
            let elements = dims.iter().product::<u64>() as usize;
            let bytes = source
                .tensor_slice(&name)
                .ok_or_else(|| format!("Missing tensor data {name}"))?;
            if bytes.len() != elements * 4 {
                return Err(format!("Invalid tensor bytes {name}"));
            }
            weights.insert(
                name,
                bytes
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect(),
            );
        }
        let temp = agent["temperature"]
            .as_array()
            .ok_or("Missing temperatures")?;
        let temperatures = [
            temp[0].as_f64().unwrap_or(1.) as f32,
            temp[1].as_f64().unwrap_or(1.) as f32,
            temp[2].as_f64().unwrap_or(1.) as f32,
        ];
        Ok(Self {
            weights,
            tokenizer,
            max_len: agent["max_len"].as_u64().unwrap_or(1024) as usize,
            head_max_len: agent["head_max_len"].as_u64().unwrap_or(256) as usize,
            temperatures,
        })
    }
    pub fn max_len(&self) -> usize {
        self.max_len
    }
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }
    pub fn predict(&self, request: &request::Request) -> Result<Value, String> {
        let questions =
            request::prepare(&self.tokenizer, request, self.max_len, self.head_max_len)?;
        let mut answers = serde_json::Map::new();
        for q in questions {
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("RMI_PARITY_TRACE").is_some() {
                crate::parity_trace::token_ids("laya.input_ids", &q.ids)
                    .map_err(|e| e.to_string())?;
                crate::parity_trace::usize_values("laya.markers", &[q.markers.len()], &q.markers)
                    .map_err(|e| e.to_string())?;
            }
            let (logits, act) = self.forward(&q)?;
            let scale = self.temperatures[q.qtype].max(0.5);
            let probs = softmax(&logits.iter().map(|v| *v / scale).collect::<Vec<_>>());
            let action = softmax(&act);
            let answer_confidence = probs.iter().copied().fold(0., f32::max);
            let confidence = if probs.len() < 2 {
                1.
            } else {
                let entropy = -probs
                    .iter()
                    .map(|p| p.max(1e-12) * p.max(1e-12).ln())
                    .sum::<f32>();
                (1. - entropy / (probs.len() as f32).ln()).clamp(0., 1.)
            };
            let mut result = serde_json::Map::new();
            result.insert("type".into(), json!(q.kind));
            match q.kind.as_str() {
                "choice" => {
                    let index = argmax(&probs);
                    result.insert("choice".into(), q.labels[index].clone());
                    let probabilities = q
                        .labels
                        .iter()
                        .zip(&probs)
                        .map(|(label, probability)| {
                            (
                                label.as_str().unwrap_or_default().to_owned(),
                                json!(round4(*probability)),
                            )
                        })
                        .collect::<serde_json::Map<_, _>>();
                    result.insert("probabilities".into(), Value::Object(probabilities));
                    result.insert("confidence".into(), json!(round4(confidence)));
                }
                "score" => {
                    let score = probs
                        .iter()
                        .enumerate()
                        .map(|(i, p)| i as f32 * p)
                        .sum::<f32>();
                    result.insert("score".into(), json!(round4(score)));
                    let probabilities = probs
                        .iter()
                        .enumerate()
                        .map(|(i, probability)| (i.to_string(), json!(round4(*probability))))
                        .collect::<serde_json::Map<_, _>>();
                    let legend = q
                        .criteria
                        .as_array()
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                        .enumerate()
                        .map(|(i, criterion)| (i.to_string(), criterion))
                        .collect::<serde_json::Map<_, _>>();
                    result.insert("legend".into(), Value::Object(legend));
                    result.insert("probabilities".into(), Value::Object(probabilities));
                    result.insert("confidence".into(), json!(round4(confidence)));
                }
                _ => {
                    result.insert("noul".into(), json!(round4(probs[1])));
                    result.insert(
                        "confidence".into(),
                        json!(round4(probs[1].max(1. - probs[1]))),
                    );
                }
            }
            result.insert("answer_confidence".into(), json!(round4(answer_confidence)));
            result.insert(
                "action".into(),
                json!({"act_probability": round4(action[0])}),
            );
            answers.insert(q.id, Value::Object(result));
        }
        Ok(json!({"answers": answers}))
    }
    fn forward(&self, q: &request::PreparedQuestion) -> Result<(Vec<f32>, Vec<f32>), String> {
        let n = q.ids.len();
        let emb = self.w("encoder.embeddings.tok_embeddings.weight")?;
        let mut x = vec![0.; n * D];
        for (i, &id) in q.ids.iter().enumerate() {
            x[i * D..(i + 1) * D].copy_from_slice(&emb[id as usize * D..(id as usize + 1) * D]);
        }
        trace("laya.embedding", &[n, D], &x)?;
        layer_norm(
            &mut x,
            self.w("encoder.embeddings.norm.weight")?,
            None,
            1e-5,
        );
        trace("laya.embedding_norm", &[n, D], &x)?;
        for l in 0..LAYERS {
            let p = format!("encoder.layers.{l}.");
            let input = x.clone();
            let norm = if l == 0 {
                input.clone()
            } else {
                norm_copy(&input, self.w(&(p.clone() + "attn_norm.weight"))?, None)
            };
            let attn = self.attention(&norm, l, n);
            for i in 0..x.len() {
                x[i] = input[i] + attn[i];
            }
            let norm = norm_copy(&x, self.w(&(p.clone() + "mlp_norm.weight"))?, None);
            if l == 0 {
                trace("laya.encoder.0.mlp_norm", &[n, D], &norm)?;
            }
            let wi = self.w(&(p.clone() + "mlp.Wi.weight"))?;
            let wo = self.w(&(p + "mlp.Wo.weight"))?;
            let mut y = vec![0.; n * 2 * F];
            matmul(&norm, wi, n, D, 2 * F, &mut y);
            if l == 0 {
                trace("laya.encoder.0.mlp_in", &[n, 2 * F], &y)?;
            }
            let mut gated = vec![0.; n * F];
            for r in 0..n {
                for i in 0..F {
                    let v = y[r * 2 * F + i];
                    gated[r * F + i] = gelu_erf(v) * y[r * 2 * F + F + i];
                }
            }
            let mut out = vec![0.; n * D];
            matmul(&gated, wo, n, F, D, &mut out);
            if l == 0 {
                trace("laya.encoder.0.mlp_out", &[n, D], &out)?;
            }
            for i in 0..x.len() {
                x[i] += out[i];
            }
            trace(&format!("laya.encoder.{l}"), &[n, D], &x)?;
        }
        layer_norm(&mut x, self.w("encoder.final_norm.weight")?, None, 1e-5);
        trace("laya.encoder_norm", &[n, D], &x)?;
        let mut h = x.clone();
        for i in 0..h.len() {
            h[i] += self.w("type_emb.weight")?[q.qtype * D + i % D];
        }
        trace("laya.typed_hidden", &[n, D], &h)?;
        for l in 0..2 {
            let p = format!("head.layers.{l}.");
            let head_input = h.clone();
            let norm = norm_copy(
                &head_input,
                self.w(&(p.clone() + "norm1.weight"))?,
                Some(self.w(&(p.clone() + "norm1.bias"))?),
            );
            let mut qkv = vec![0.; n * 3 * D];
            matmul(
                &norm,
                self.w(&(p.clone() + "self_attn.in_proj_weight"))?,
                n,
                D,
                3 * D,
                &mut qkv,
            );
            add_bias(
                &mut qkv,
                self.w(&(p.clone() + "self_attn.in_proj_bias"))?,
                3 * D,
            );
            let a = attention_dense(&qkv, n);
            let mut attn_out = vec![0.; n * D];
            matmul(
                &a,
                self.w(&(p.clone() + "self_attn.out_proj.weight"))?,
                n,
                D,
                D,
                &mut attn_out,
            );
            add_bias(
                &mut attn_out,
                self.w(&(p.clone() + "self_attn.out_proj.bias"))?,
                D,
            );
            h = head_input;
            for i in 0..h.len() {
                h[i] += attn_out[i];
            }
            let norm = norm_copy(
                &h,
                self.w(&(p.clone() + "norm2.weight"))?,
                Some(self.w(&(p.clone() + "norm2.bias"))?),
            );
            let mut y = vec![0.; n * 4 * D];
            matmul(
                &norm,
                self.w(&(p.clone() + "linear1.weight"))?,
                n,
                D,
                4 * D,
                &mut y,
            );
            add_bias(&mut y, self.w(&(p.clone() + "linear1.bias"))?, 4 * D);
            for v in &mut y {
                *v = v.max(0.);
            }
            let mut out = vec![0.; n * D];
            matmul(&y, self.w(&(p + "linear2.weight"))?, n, 4 * D, D, &mut out);
            add_bias(
                &mut out,
                self.w(&format!("head.layers.{l}.linear2.bias"))?,
                D,
            );
            for i in 0..h.len() {
                h[i] += out[i];
            }
            trace(&format!("laya.head.{l}"), &[n, D], &h)?;
        }
        let mut markers = Vec::new();
        for &pos in &q.markers {
            markers.push(h[pos * D..(pos + 1) * D].to_vec());
        }
        let sc0 = self.w("scorer.0.weight")?;
        for row in &mut markers {
            layer_norm(row, sc0, Some(self.w("scorer.0.bias")?), 1e-5);
            let mut y = vec![0.; D];
            matmul(row, self.w("scorer.1.weight")?, 1, D, D, &mut y);
            add_bias(&mut y, self.w("scorer.1.bias")?, D);
            for v in &mut y {
                *v = gelu_erf(*v);
            }
            *row = y;
        }
        let mut logits = vec![0.; q.markers.len()];
        for (i, row) in markers.iter().enumerate() {
            logits[i] = dot(row, self.w("scorer.3.weight")?) + self.w("scorer.3.bias")?[0];
        }
        let pooled = &h[..D];
        let p = softmax(&logits);
        let top1 = p.iter().copied().fold(0., f32::max);
        let mut sorted = p.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let top2 = sorted.get(1).copied().unwrap_or(0.);
        let ent = -(p
            .iter()
            .map(|x| x.max(1e-9) * x.max(1e-9).ln())
            .sum::<f32>())
            / (q.markers.len().max(2) as f32).ln();
        let mut af = pooled.to_vec();
        af.extend([top1, top1 - top2, ent, q.markers.len().max(2) as f32 / 255.]);
        let mut act = vec![0.; 256];
        matmul(&af, self.w("act_head.0.weight")?, 1, D + 4, 256, &mut act);
        add_bias(&mut act, self.w("act_head.0.bias")?, 256);
        for v in &mut act {
            *v = gelu_erf(*v);
        }
        let mut out = vec![0.; 2];
        matmul(&act, self.w("act_head.2.weight")?, 1, 256, 2, &mut out);
        add_bias(&mut out, self.w("act_head.2.bias")?, 2);
        trace("laya.logits", &[logits.len()], &logits)?;
        trace("laya.act_logits", &[out.len()], &out)?;
        Ok((logits, out))
    }
    fn w(&self, name: &str) -> Result<&[f32], String> {
        self.weights
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("Missing tensor {name}"))
    }
    fn attention(&self, x: &[f32], layer: usize, n: usize) -> Vec<f32> {
        let p = format!("encoder.layers.{layer}.attn.");
        let mut qkv = vec![0.; n * 3 * D];
        matmul(
            x,
            self.w(&(p + "Wqkv.weight")).unwrap(),
            n,
            D,
            3 * D,
            &mut qkv,
        );
        if layer == 0 {
            trace("laya.encoder.0.qkv", &[n, 3 * D], &qkv).unwrap();
        }
        rope_qk(&mut qkv, n);
        if layer == 0 {
            trace("laya.encoder.0.rope", &[n, 3 * D], &qkv).unwrap();
        }
        let a = attention_rows(&qkv, n, layer % 3 == 0);
        if layer == 0 {
            trace("laya.encoder.0.context", &[n, D], &a).unwrap();
        }
        let mut out = vec![0.; n * D];
        matmul(
            &a,
            self.w(&format!("encoder.layers.{layer}.attn.Wo.weight"))
                .unwrap(),
            n,
            D,
            D,
            &mut out,
        );
        if layer == 0 {
            trace("laya.encoder.0.attn_out", &[n, D], &out).unwrap();
        }
        out
    }
}

fn trace(name: &str, shape: &[usize], values: &[f32]) -> Result<(), String> {
    #[cfg(feature = "parity-trace")]
    if std::env::var_os("RMI_PARITY_TRACE").is_some() {
        crate::parity_trace::checkpoint(name, None, shape, values).map_err(|e| e.to_string())?;
    }
    let _ = (name, shape, values);
    Ok(())
}

fn tensor_names() -> Vec<(String, Vec<u64>)> {
    let mut v = vec![
        ("temperature".into(), vec![3]),
        ("type_emb.weight".into(), vec![D as u64, 3]),
        (
            "encoder.embeddings.tok_embeddings.weight".into(),
            vec![D as u64, 256000],
        ),
        ("encoder.embeddings.norm.weight".into(), vec![D as u64]),
        ("encoder.final_norm.weight".into(), vec![D as u64]),
    ];
    for l in 0..LAYERS {
        let p = format!("encoder.layers.{l}.");
        v.extend([
            (
                format!("{p}attn.Wqkv.weight"),
                vec![D as u64, (3 * D) as u64],
            ),
            (format!("{p}attn.Wo.weight"), vec![D as u64, D as u64]),
            (format!("{p}mlp_norm.weight"), vec![D as u64]),
            (format!("{p}mlp.Wi.weight"), vec![D as u64, (2 * F) as u64]),
            (format!("{p}mlp.Wo.weight"), vec![F as u64, D as u64]),
        ]);
        if l > 0 {
            v.push((format!("{p}attn_norm.weight"), vec![D as u64]));
        }
    }
    for l in 0..2 {
        let p = format!("head.layers.{l}.");
        for (n, o, i) in [
            ("self_attn.in_proj", 3 * D, D),
            ("self_attn.out_proj", D, D),
            ("linear1", 4 * D, D),
            ("linear2", D, 4 * D),
        ] {
            let sep = if n == "self_attn.in_proj" { "_" } else { "." };
            v.push((format!("{p}{n}{sep}weight"), vec![i as u64, o as u64]));
            v.push((format!("{p}{n}{sep}bias"), vec![o as u64]));
        }
        for norm in ["norm1", "norm2"] {
            v.push((format!("{p}{norm}.weight"), vec![D as u64]));
            v.push((format!("{p}{norm}.bias"), vec![D as u64]));
        }
    }
    v.extend([
        ("scorer.0.weight".into(), vec![D as u64]),
        ("scorer.0.bias".into(), vec![D as u64]),
        ("scorer.1.weight".into(), vec![D as u64, D as u64]),
        ("scorer.1.bias".into(), vec![D as u64]),
        ("scorer.3.weight".into(), vec![D as u64, 1]),
        ("scorer.3.bias".into(), vec![1]),
        ("act_head.0.weight".into(), vec![(D + 4) as u64, 256]),
        ("act_head.0.bias".into(), vec![256]),
        ("act_head.2.weight".into(), vec![256, 2]),
        ("act_head.2.bias".into(), vec![2]),
    ]);
    v
}

fn matmul(x: &[f32], w: &[f32], rows: usize, inp: usize, out: usize, y: &mut [f32]) {
    assert_eq!(
        w.len(),
        inp * out,
        "matmul weight len {} expected {}",
        w.len(),
        inp * out
    );
    for r in 0..rows {
        for o in 0..out {
            y[r * out + o] = dot(&x[r * inp..(r + 1) * inp], &w[o * inp..(o + 1) * inp]);
        }
    }
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn add_bias(x: &mut [f32], b: &[f32], n: usize) {
    for row in x.chunks_exact_mut(n) {
        for (i, v) in row.iter_mut().enumerate() {
            *v += b[i];
        }
    }
}
fn layer_norm(x: &mut [f32], w: &[f32], b: Option<&[f32]>, eps: f32) {
    for row in x.chunks_exact_mut(D) {
        let m = row.iter().sum::<f32>() / D as f32;
        let v = row.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / D as f32;
        let scale = 1. / (v + eps).sqrt();
        for (i, z) in row.iter_mut().enumerate() {
            *z = (*z - m) * scale * w[i] + b.map(|b| b[i]).unwrap_or(0.);
        }
    }
}

fn norm_copy(x: &[f32], w: &[f32], b: Option<&[f32]>) -> Vec<f32> {
    let mut y = x.to_vec();
    layer_norm(&mut y, w, b, 1e-5);
    y
}
fn attention_rows(qkv: &[f32], n: usize, full: bool) -> Vec<f32> {
    let mut out = vec![0.; n * D];
    for i in 0..n {
        for h in 0..H {
            let mut score = vec![f32::NEG_INFINITY; n];
            for j in 0..n {
                if full || ((i as isize - j as isize).unsigned_abs() <= 64) {
                    score[j] = dot(
                        &qkv[(i * 3 * D + h * HD)..(i * 3 * D + h * HD + HD)],
                        &qkv[(j * 3 * D + D + h * HD)..(j * 3 * D + D + h * HD + HD)],
                    ) / 8.;
                }
            }
            let p = softmax(&score);
            for j in 0..n {
                for d in 0..HD {
                    out[i * D + h * HD + d] += p[j] * qkv[j * 3 * D + 2 * D + h * HD + d];
                }
            }
        }
    }
    out
}
fn attention_dense(qkv: &[f32], n: usize) -> Vec<f32> {
    attention_rows(qkv, n, true)
}
fn rope_qk(qkv: &mut [f32], n: usize) {
    for pos in 0..n {
        for h in 0..H {
            for i in 0..(HD / 2) {
                let inv_freq = 1. / 160000f32.powf((2 * i) as f32 / HD as f32);
                let theta = (pos as f32) * inv_freq;
                let (s, c) = (scalar_sin(theta), scalar_cos(theta));
                for base in [0, D] {
                    let a = pos * 3 * D + base + h * HD + i;
                    let b = a + HD / 2;
                    let x = qkv[a];
                    let y = qkv[b];
                    qkv[a] = x * c - y * s;
                    qkv[b] = y * c + x * s;
                }
            }
        }
    }
}
// Keep LLVM from combining the two libm calls into sincos: on macOS it
// rounds differently from the independent scalar sinf/cosf reference.
#[inline(never)]
fn scalar_sin(x: f32) -> f32 {
    x.sin()
}
#[inline(never)]
fn scalar_cos(x: f32) -> f32 {
    x.cos()
}
fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut y = x.iter().map(|v| (*v - m).exp()).collect::<Vec<_>>();
    let s = y.iter().sum::<f32>();
    for v in &mut y {
        *v /= s.max(1e-12);
    }
    y
}
fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}
fn round4(v: f32) -> f64 {
    ((v as f64) * 10000.).round() / 10000.
}
