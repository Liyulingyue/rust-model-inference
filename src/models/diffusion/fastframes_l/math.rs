use super::weights::Graph;
use crate::compute::linear::LinearExecutor;
use crate::ops::nn::{attention_online, layer_norm_rows, AttentionSpec, Linear};

pub(super) fn linear<'a>(
    g: &'a Graph<'a>,
    ex: &mut LinearExecutor<'a, 'a>,
    name: &str,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let w = &g.linear[name];
    if !input.len().is_multiple_of(w.n_in()) {
        return Err(format!("invalid {name} input width"));
    }
    let mut output = Vec::new();
    for chunk in input.chunks(ex.max_rows() * w.n_in()) {
        let [result] = Linear::forward_group([w], ex, chunk, chunk.len() / w.n_in())?;
        output.extend(result);
    }
    Ok(output)
}

pub(super) fn norm(input: &[f32], width: usize, eps: f32) -> Result<Vec<f32>, String> {
    layer_norm_rows(
        input,
        input.len() / width,
        Some(&vec![1.; width]),
        None,
        eps,
    )
}

pub(super) fn affine_norm<'a>(
    g: &'a Graph<'a>,
    name: &str,
    input: &[f32],
    eps: f32,
) -> Result<Vec<f32>, String> {
    let weight = &g.floats[&format!("{name}.weight")];
    layer_norm_rows(
        input,
        input.len() / weight.len(),
        Some(weight),
        Some(&g.floats[&format!("{name}.bias")]),
        eps,
    )
}

pub(super) fn modulate(input: &mut [f32], shift: &[f32], scale: &[f32]) {
    for row in input.chunks_exact_mut(shift.len()) {
        for ((x, s), c) in row.iter_mut().zip(shift).zip(scale) {
            *x = *x * (1. + c) + s;
        }
    }
}

pub(super) fn residual(input: &mut [f32], update: &[f32], gate: Option<&[f32]>) {
    for (i, (x, y)) in input.iter_mut().zip(update).enumerate() {
        *x += y * gate.map_or(1., |g| g[i % g.len()]);
    }
}

pub(super) fn sinusoidal(t: f32, dim: usize) -> Vec<f32> {
    let angles = (0..dim / 2)
        .map(|i| t as f64 * 10000_f64.powf(-(i as f64) / (dim / 2) as f64))
        .collect::<Vec<_>>();
    angles
        .iter()
        .map(|x| x.cos() as f32)
        .chain(angles.iter().map(|x| x.sin() as f32))
        .collect()
}

pub(super) fn rope(input: &mut [f32], heads: usize, frame: usize, height: usize, width: usize) {
    let half = input.len() / (height * width * heads * 2);
    let axes = [half - 2 * (half / 3), half / 3, half / 3];
    for (token, row) in input.chunks_exact_mut(heads * half * 2).enumerate() {
        for head in row.chunks_exact_mut(half * 2) {
            let mut start = 0;
            for (pairs, position) in axes.into_iter().zip([frame, token / width, token % width]) {
                for i in 0..pairs {
                    let angle = position as f64 * 10000_f64.powf(-(i as f64) / (pairs as f64));
                    let (s, c) = (angle.sin() as f32, angle.cos() as f32);
                    let j = 2 * (start + i);
                    let (a, b) = (head[j], head[j + 1]);
                    head[j] = a * c - b * s;
                    head[j + 1] = a * s + b * c;
                }
                start += pairs;
            }
        }
    }
}

pub(super) fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    head: usize,
) -> Result<Vec<f32>, String> {
    attention_online(
        q,
        k,
        v,
        AttentionSpec {
            query_tokens: q.len() / (heads * head),
            key_tokens: k.len() / (heads * head),
            query_heads: heads,
            key_value_heads: heads,
            head_dim: head,
            causal: false,
            scale: 1. / (head as f32).sqrt(),
        },
    )
}

pub(super) fn trace(name: &str, values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(
        name,
        None,
        &[values.len()],
        values,
    ));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, values);
}
