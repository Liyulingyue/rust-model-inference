use super::{math::*, patchify, unpatchify, weights::Graph};
use crate::compute::linear::LinearExecutor;
use crate::ops::nn::{gelu_approx_inplace, rms_norm_rows, silu_inplace, Linear};

pub(super) struct Cache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}
pub(super) struct Time {
    pub blocks: Vec<f32>,
    pub head: Vec<f32>,
}
pub(super) fn time<'a>(
    g: &'a Graph<'a>,
    ex: &mut LinearExecutor<'a, 'a>,
    t: f32,
) -> Result<Time, String> {
    let mut e = linear(g, ex, "dit.time_embedding.0", &sinusoidal(t, 256))?;
    silu_inplace(&mut e);
    let e = linear(g, ex, "dit.time_embedding.2", &e)?;
    let mut h = e.clone();
    silu_inplace(&mut h);
    Ok(Time {
        blocks: linear(g, ex, "dit.time_projection.1", &h)?,
        head: e,
    })
}

pub(super) fn text_cache<'a>(
    g: &'a Graph<'a>,
    ex: &mut LinearExecutor<'a, 'a>,
    embeds: &[f32],
) -> Result<Vec<Cache>, String> {
    let dim = g.spec.dims.dim;
    let mut x = linear(g, ex, "dit.text_embedding.0", embeds)?;
    gelu_approx_inplace(&mut x);
    x = linear(g, ex, "dit.text_embedding.2", &x)?;
    let mut cache = Vec::new();
    for layer in 0..g.spec.dims.layers {
        let p = format!("dit.blocks.{layer}.cross_attn");
        let [k, v] = Linear::forward_group(
            [&g.linear[&format!("{p}.k")], &g.linear[&format!("{p}.v")]],
            ex,
            &x,
            x.len() / dim,
        )?;
        cache.push(Cache {
            k: rms_norm_rows(
                &k,
                k.len() / dim,
                &g.floats[&format!("{p}.norm_k.weight")],
                1e-6,
            )?,
            v,
        });
    }
    Ok(cache)
}

pub(super) fn embed<'a>(
    g: &'a Graph<'a>,
    ex: &mut LinearExecutor<'a, 'a>,
    latent: &[f32],
    h: usize,
    w: usize,
) -> Result<Vec<f32>, String> {
    linear(g, ex, "dit.patch_embedding", &patchify(latent, h, w))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn predict<'a>(
    g: &'a Graph<'a>,
    ex: &mut LinearExecutor<'a, 'a>,
    latent: &[f32],
    h: usize,
    w: usize,
    frame: usize,
    t: &Time,
    text: &[Cache],
    cache: &mut [Cache],
    policy: crate::compute::ComputePolicy,
    pool: &std::sync::Arc<crate::core::thread_pool::ComputePool>,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    // The shared runtime retains uploads; bounded windows fit F32 weights on a 48 GiB device.
    if ex.uses_vulkan() {
        *ex = g.executor(policy, (h * w / 4).max(512), pool.clone())?;
    }
    let mut x = embed(g, ex, latent, h, w)?;
    let tokens = h * w / 4;
    let dim = g.spec.dims.dim;
    let (heads, head_dim) = (g.spec.dims.heads, g.spec.head_dim());
    trace("fastframes.patch", &x);
    for layer in 0..g.spec.dims.layers {
        if layer > 0 && layer % 8 == 0 && ex.uses_vulkan() {
            *ex = g.executor(policy, tokens.max(512), pool.clone())?;
        }
        let p = format!("dit.blocks.{layer}");
        let e = g.floats[&format!("{p}.modulation")]
            .iter()
            .zip(&t.blocks)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        let e = e.chunks_exact(dim).collect::<Vec<_>>();
        let mut z = norm(&x, dim, 1e-6)?;
        modulate(&mut z, e[0], e[1]);
        let sa = format!("{p}.self_attn");
        let [q, k, v] = Linear::forward_group(
            [
                &g.linear[&format!("{sa}.q")],
                &g.linear[&format!("{sa}.k")],
                &g.linear[&format!("{sa}.v")],
            ],
            ex,
            &z,
            tokens,
        )?;
        let mut q = rms_norm_rows(&q, tokens, &g.floats[&format!("{sa}.norm_q.weight")], 1e-6)?;
        let mut k = rms_norm_rows(&k, tokens, &g.floats[&format!("{sa}.norm_k.weight")], 1e-6)?;
        rope(&mut q, heads, frame, h / 2, w / 2);
        rope(&mut k, heads, frame, h / 2, w / 2);
        let start = frame * tokens * dim;
        let end = start + tokens * dim;
        cache[layer].k[start..end].copy_from_slice(&k);
        cache[layer].v[start..end].copy_from_slice(&v);
        let a = attention(
            &q,
            &cache[layer].k[..end],
            &cache[layer].v[..end],
            heads,
            head_dim,
        )?;
        let z = linear(g, ex, &format!("{sa}.o"), &a)?;
        residual(&mut x, &z, Some(e[2]));
        let ca = format!("{p}.cross_attn");
        let z = affine_norm(g, &format!("{p}.norm3"), &x, 1e-6)?;
        let q = linear(g, ex, &format!("{ca}.q"), &z)?;
        let q = rms_norm_rows(&q, tokens, &g.floats[&format!("{ca}.norm_q.weight")], 1e-6)?;
        let a = attention(&q, &text[layer].k, &text[layer].v, heads, head_dim)?;
        let z = linear(g, ex, &format!("{ca}.o"), &a)?;
        residual(&mut x, &z, None);
        let mut z = norm(&x, dim, 1e-6)?;
        modulate(&mut z, e[3], e[4]);
        let mut z = linear(g, ex, &format!("{p}.ffn.0"), &z)?;
        gelu_approx_inplace(&mut z);
        let z = linear(g, ex, &format!("{p}.ffn.2"), &z)?;
        residual(&mut x, &z, Some(e[5]));
        trace(&format!("fastframes.dit.{layer}"), &x);
    }
    let e = &g.floats["dit.head.modulation"];
    let shift = e[..dim]
        .iter()
        .zip(&t.head)
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    let scale = e[dim..]
        .iter()
        .zip(&t.head)
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    let mut z = norm(&x, dim, 1e-6)?;
    modulate(&mut z, &shift, &scale);
    let flow = unpatchify(&linear(g, ex, "dit.head.head", &z)?, h, w);
    trace("fastframes.flow", &flow);
    Ok((flow, x))
}
