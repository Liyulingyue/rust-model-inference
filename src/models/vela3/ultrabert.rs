//! Bidirectional UltraBERT encoder with per-layer local or global attention.
use super::tensor::{embed, linear, linear_lora, rms_norm, values, OverlaySource};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::ops::{dot_f32, softmax_inplace};
use serde_json::Value;

pub(crate) fn encode(
    source: &dyn TensorSource,
    config: &Value,
    prefix: &str,
    ids: &[u32],
    embeddings: Option<&[f32]>,
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    encode_impl(source, None, config, prefix, ids, embeddings, pool)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_adapted(
    base: &dyn TensorSource,
    adapter: &dyn TensorSource,
    scale: f32,
    config: &Value,
    prefix: &str,
    ids: &[u32],
    embeddings: Option<&[f32]>,
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    encode_impl(
        base,
        Some((adapter, scale)),
        config,
        prefix,
        ids,
        embeddings,
        pool,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_impl(
    base: &dyn TensorSource,
    adapter: Option<(&dyn TensorSource, f32)>,
    config: &Value,
    prefix: &str,
    ids: &[u32],
    embeddings: Option<&[f32]>,
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    let hidden = integer(config, "hidden_size")?;
    let layers = config["num_hidden_layers"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or("missing num_hidden_layers")?;
    let heads = integer(config, "num_attention_heads")?;
    let kv_heads = integer(config, "num_key_value_heads").unwrap_or(heads);
    let dim = integer(config, "head_dim").unwrap_or(hidden / heads);
    if dim == 0 || !dim.is_multiple_of(2) || !heads.is_multiple_of(kv_heads) {
        return Err("invalid UltraBERT attention dimensions".into());
    }
    let n = ids.len();
    if n == 0 {
        return Err("UltraBERT needs at least one token".into());
    }
    let size = n
        .checked_mul(hidden)
        .ok_or("UltraBERT embedding size overflow")?;
    let eps = config["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
    let overlay = adapter.map(|(delta, _)| OverlaySource::new(base, delta));
    let source: &dyn TensorSource = overlay
        .as_ref()
        .map(|s| s as &dyn TensorSource)
        .unwrap_or(base);
    let mut x = if let Some(e) = embeddings {
        if e.len() != size || e.iter().any(|v| !v.is_finite()) {
            return Err("UltraBERT supplied embedding shape or values invalid".into());
        }
        e.to_vec()
    } else {
        let e = embed(source, &format!("{prefix}embed_tokens.weight"), ids)?;
        if e.len() != size {
            return Err("UltraBERT embedding width mismatch".into());
        }
        e
    };
    let project = |name: &str, input: &[f32]| match adapter {
        Some((delta, scale)) => linear_lora(base, delta, scale, name, input, n, pool),
        None => linear(source, name, input, n, pool),
    };
    let norm = |name: &str, input: &[f32]| {
        let w = values(source, name)?;
        if w.len() != hidden {
            return Err(format!("UltraBERT normalization width mismatch: {name}"));
        }
        rms_norm(input, &w, eps)
    };
    let mut rotary = std::collections::HashMap::new();
    for layer in 0..layers {
        let name = format!("{prefix}layers.{layer}");
        let h = norm(&format!("{name}.input_layernorm.weight"), &x)?;
        let mut q = project(&format!("{name}.self_attn.q_proj.weight"), &h)?;
        let mut k = project(&format!("{name}.self_attn.k_proj.weight"), &h)?;
        let v = project(&format!("{name}.self_attn.v_proj.weight"), &h)?;
        if n.checked_mul(heads).and_then(|x| x.checked_mul(dim)) != Some(q.len())
            || n.checked_mul(kv_heads).and_then(|x| x.checked_mul(dim)) != Some(k.len())
            || v.len() != k.len()
        {
            return Err("UltraBERT q/k/v projection shape mismatch".into());
        }
        let layer_type = config["layer_types"]
            .as_array()
            .and_then(|a| a.get(layer))
            .and_then(Value::as_str);
        let theta = config["rope_theta"]
            .as_array()
            .and_then(|a| a.get(layer))
            .and_then(Value::as_f64)
            .or_else(|| config["rope_theta"].as_f64())
            .or_else(|| {
                layer_type.and_then(|t| config["rope_parameters"][t]["rope_theta"].as_f64())
            })
            .ok_or("missing UltraBERT rope_theta")? as f32;
        if !theta.is_finite() || theta <= 0. {
            return Err("invalid UltraBERT rope_theta".into());
        }
        let (cos, sin) = rotary.entry(theta.to_bits()).or_insert_with(|| {
            let mut cos = Vec::with_capacity(n * dim / 2);
            let mut sin = Vec::with_capacity(n * dim / 2);
            for pos in 0..n {
                for i in 0..dim / 2 {
                    let inverse = 1. / theta.powf((2 * i) as f32 / dim as f32);
                    let phase = pos as f32 * inverse;
                    cos.push(phase.cos());
                    sin.push(phase.sin());
                }
            }
            (cos, sin)
        });
        rope(&mut q, n, heads, dim, cos, sin);
        rope(&mut k, n, kv_heads, dim, cos, sin);
        let window = if let Some(window) = config["attention_windows"]
            .as_array()
            .and_then(|a| a.get(layer))
        {
            if window.is_null() {
                None
            } else {
                Some(
                    window
                        .as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or("invalid UltraBERT attention window")?,
                )
            }
        } else {
            match layer_type {
                Some("full_attention") => None,
                Some("sliding_attention") => Some(integer(config, "sliding_window")?),
                _ => return Err("missing or unsupported UltraBERT attention window".into()),
            }
        };
        let att = attention(&q, &k, &v, n, heads, kv_heads, dim, window, pool)?;
        let projected = project(&format!("{name}.self_attn.o_proj.weight"), &att)?;
        if projected.len() != x.len() {
            return Err("UltraBERT output projection width mismatch".into());
        }
        for (a, b) in x.iter_mut().zip(projected) {
            *a += b;
        }
        let h = norm(&format!("{name}.post_attention_layernorm.weight"), &x)?;
        let mut gate = project(&format!("{name}.mlp.gate_proj.weight"), &h)?;
        let up = project(&format!("{name}.mlp.up_proj.weight"), &h)?;
        if gate.len() != up.len() {
            return Err("UltraBERT gate/up width mismatch".into());
        }
        match config["hidden_act"].as_str().unwrap_or("silu") {
            "silu" => {
                for (g, u) in gate.iter_mut().zip(up) {
                    *g = (*g / (1. + (-*g).exp())) * u;
                }
            }
            "gelu" => {
                for (g, u) in gate.iter_mut().zip(up) {
                    *g = crate::ops::gelu_erf(*g) * u;
                }
            }
            other => return Err(format!("unsupported UltraBERT activation: {other}")),
        }
        let mlp = project(&format!("{name}.mlp.down_proj.weight"), &gate)?;
        if mlp.len() != x.len() {
            return Err("UltraBERT MLP output width mismatch".into());
        }
        for (a, b) in x.iter_mut().zip(mlp) {
            *a += b;
        }
    }
    norm(&format!("{prefix}norm.weight"), &x)
}
fn integer(config: &Value, name: &str) -> Result<usize, String> {
    config[name]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| format!("invalid UltraBERT {name}"))
}
fn rope(x: &mut [f32], tokens: usize, heads: usize, dim: usize, cos: &[f32], sin: &[f32]) {
    for pos in 0..tokens {
        for head in 0..heads {
            let row = &mut x[(pos * heads + head) * dim..(pos * heads + head + 1) * dim];
            for i in 0..dim / 2 {
                let (a, b) = (row[i], row[i + dim / 2]);
                let (c, s) = (cos[pos * dim / 2 + i], sin[pos * dim / 2 + i]);
                row[i] = a * c - b * s;
                row[i + dim / 2] = b * c + a * s;
            }
        }
    }
}
#[allow(clippy::too_many_arguments)]
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tokens: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    window: Option<usize>,
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    let width = heads.checked_mul(dim).ok_or("attention width overflow")?;
    let mut out = vec![
        0.;
        tokens
            .checked_mul(width)
            .ok_or("attention output overflow")?
    ];
    let address = out.as_mut_ptr() as usize;
    pool.compute(|ith, nth| {
        let start = tokens * ith / nth;
        let end = tokens * (ith + 1) / nth;
        // Each worker exclusively owns a complete token band.
        let output = unsafe {
            std::slice::from_raw_parts_mut(
                (address as *mut f32).add(start * width),
                (end - start) * width,
            )
        };
        let mut scores = Vec::new();
        for pos in start..end {
            let lo = window.map(|w| pos.saturating_sub(w)).unwrap_or(0);
            let hi = window
                .map(|w| pos.saturating_add(w).saturating_add(1).min(tokens))
                .unwrap_or(tokens);
            scores.resize(hi - lo, 0.);
            for head in 0..heads {
                let kv = head / (heads / kv_heads);
                let query = &q[(pos * heads + head) * dim..(pos * heads + head + 1) * dim];
                for (index, other) in (lo..hi).enumerate() {
                    let key = &k[(other * kv_heads + kv) * dim..(other * kv_heads + kv + 1) * dim];
                    scores[index] = dot_f32(query, key, dim) / (dim as f32).sqrt();
                }
                softmax_inplace(&mut scores);
                let row = &mut output[((pos - start) * heads + head) * dim
                    ..((pos - start) * heads + head + 1) * dim];
                for (&p, other) in scores.iter().zip(lo..hi) {
                    let value =
                        &v[(other * kv_heads + kv) * dim..(other * kv_heads + kv + 1) * dim];
                    for (x, &v) in row.iter_mut().zip(value) {
                        *x += p * v;
                    }
                }
            }
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::vela3::tensor::tests::Source;
    #[test]
    fn supplied_embeddings_and_final_norm_use_the_selected_prefix() {
        let mut s = Source::default();
        s.add("text_model.norm.weight", &[2], &[1., 2.]);
        let c = serde_json::json!({"hidden_size":2,"num_hidden_layers":0,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,"rms_norm_eps":1e-6,"rope_theta":[],"attention_windows":[]});
        let out = encode(
            &s,
            &c,
            "text_model.",
            &[2, 3],
            Some(&[3., 4., 0., 2.]),
            &ComputePool::new(1),
        )
        .unwrap();
        for (a, b) in out.iter().zip([0.8485281, 2.2627416, 0., 2.8284264]) {
            assert!((a - b).abs() < 2e-6);
        }
        assert!(encode(
            &s,
            &c,
            "text_model.",
            &[2, 3],
            Some(&[1.]),
            &ComputePool::new(1)
        )
        .is_err());
    }
}

#[cfg(test)]
mod attention_reference {
    use super::*;
    use crate::models::vela3::tensor::tests::Source;
    #[test]
    fn local_bidirectional_rope_attention_matches_torch_reference() {
        let mut s = Source::default();
        s.add(
            "encoder.embed_tokens.weight",
            &[4, 2],
            &[1., 0., 0., 2., 2., 3., -1., 2.],
        );
        s.add("encoder.norm.weight", &[2], &[1., 1.]);
        s.add("encoder.layers.0.input_layernorm.weight", &[2], &[1., 1.]);
        s.add(
            "encoder.layers.0.post_attention_layernorm.weight",
            &[2],
            &[1., 1.],
        );
        for n in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ] {
            s.add(
                &format!("encoder.layers.0.{n}.weight"),
                &[2, 2],
                &[1., 0., 0., 1.],
            );
        }
        for n in ["gate_proj", "up_proj", "down_proj"] {
            s.add(
                &format!("encoder.layers.0.mlp.{n}.weight"),
                &[2, 2],
                &[0.; 4],
            );
        }
        let c = serde_json::json!({"hidden_size":2,"num_hidden_layers":1,"intermediate_size":2,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,"rms_norm_eps":1e-6,"rope_theta":[10000.0],"attention_windows":[1],"hidden_act":"silu"});
        let got = encode(
            &s,
            &c,
            "encoder.",
            &[0, 1, 2, 3],
            None,
            &ComputePool::new(2),
        )
        .unwrap();
        let want = [
            1.4129653,
            0.059396885,
            0.17616318,
            1.4031986,
            0.67972475,
            1.2401508,
            -0.5832067,
            1.2883593,
        ];
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() < 3e-6, "{a} vs {b}");
        }
        let mut modern = c.clone();
        modern.as_object_mut().unwrap().remove("rope_theta");
        modern.as_object_mut().unwrap().remove("attention_windows");
        modern["layer_types"] = serde_json::json!(["sliding_attention"]);
        modern["sliding_window"] = serde_json::json!(1);
        modern["rope_parameters"] =
            serde_json::json!({"sliding_attention":{"rope_theta":10000.,"rope_type":"default"}});
        let modern_out = encode(
            &s,
            &modern,
            "encoder.",
            &[0, 1, 2, 3],
            None,
            &ComputePool::new(1),
        )
        .unwrap();
        assert_eq!(modern_out, got);
    }
}
