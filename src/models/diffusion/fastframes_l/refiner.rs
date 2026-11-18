use super::{
    dit::{Cache, DIM},
    math::*,
    weights::Graph,
};
use crate::compute::linear::LinearExecutor;
use crate::ops::nn::{gelu_approx_inplace, silu_inplace};

pub(super) fn refine<'a>(
    g: &'a Graph<'a>,
    ex: &mut LinearExecutor<'a, 'a>,
    stem: &[f32],
    frame: usize,
    cache: &mut [Cache],
) -> Result<(), String> {
    const D: usize = 768;
    let tokens = stem.len() / DIM;
    let mut e = linear(g, ex, "refiner.t_embed.mlp.0", &sinusoidal(833., 256))?;
    silu_inplace(&mut e);
    let e = linear(g, ex, "refiner.t_embed.mlp.2", &e)?;
    let mut se = e.clone();
    silu_inplace(&mut se);
    let mut x = linear(g, ex, "refiner.stem_in", stem)?;
    for layer in 0..4 {
        let p = format!("refiner.stem_blocks.{layer}");
        let m = linear(g, ex, &format!("{p}.mod.lin"), &se)?;
        let m = m.chunks_exact(D).collect::<Vec<_>>();
        let mut z = norm(&x, D, 1e-5)?;
        modulate(&mut z, m[0], m[1]);
        let qkv = linear(g, ex, &format!("{p}.qkv"), &z)?;
        let mut q = Vec::with_capacity(tokens * D);
        let mut k = q.clone();
        let mut v = q.clone();
        for row in qkv.chunks_exact(3 * D) {
            q.extend_from_slice(&row[..D]);
            k.extend_from_slice(&row[D..2 * D]);
            v.extend_from_slice(&row[2 * D..]);
        }
        let a = attention(&q, &k, &v, 8, 96)?;
        let z = linear(g, ex, &format!("{p}.proj"), &a)?;
        residual(&mut x, &z, Some(m[2]));
        let mut z = norm(&x, D, 1e-5)?;
        modulate(&mut z, m[3], m[4]);
        let mut z = linear(g, ex, &format!("{p}.mlp.0"), &z)?;
        gelu_approx_inplace(&mut z);
        let z = linear(g, ex, &format!("{p}.mlp.2"), &z)?;
        residual(&mut x, &z, Some(m[5]));
    }
    x = affine_norm(g, "refiner.stem_norm", &x, 1e-5)?;
    let phi = linear(g, ex, "refiner.heads_grouped.phi_proj.768", &x)?;
    trace("fastframes.refiner.phi", &phi);
    let start = frame * tokens * DIM;
    let end = start + tokens * DIM;
    for (layer, cache) in cache.iter_mut().enumerate() {
        let p = format!("refiner.heads.{layer}");
        let pk = linear(g, ex, &format!("{p}.proj_k"), &cache.k[start..end])?;
        let pv = linear(g, ex, &format!("{p}.proj_v"), &cache.v[start..end])?;
        let mut joined = Vec::with_capacity(tokens * 3 * D);
        for i in 0..tokens {
            for data in [&pk, &pv, &phi] {
                joined.extend_from_slice(&data[i * D..(i + 1) * D]);
            }
        }
        let mut h = linear(g, ex, &format!("{p}.fuse"), &joined)?;
        let m = linear(g, ex, &format!("{p}.mod"), &se)?;
        h = norm(&h, D, 1e-5)?;
        modulate(&mut h, &m[..D], &m[D..]);
        let mut h = linear(g, ex, &format!("{p}.m0"), &h)?;
        gelu_approx_inplace(&mut h);
        let h = linear(g, ex, &format!("{p}.m2"), &h)?;
        let dk = linear(g, ex, &format!("{p}.dK"), &h)?;
        let dv = linear(g, ex, &format!("{p}.dV"), &h)?;
        repair_keys(&mut cache.k[start..end], &dk, 128);
        let scale = &g.floats[&format!("{p}.v_logscale")];
        for (i, (v, d)) in cache.v[start..end].iter_mut().zip(&dv).enumerate() {
            *v = (*v + d) * scale[i % DIM].exp();
        }
        trace(
            &format!("fastframes.refiner.{layer}.k"),
            &cache.k[start..end],
        );
        trace(
            &format!("fastframes.refiner.{layer}.v"),
            &cache.v[start..end],
        );
    }
    Ok(())
}

pub(super) fn repair_keys(keys: &mut [f32], delta: &[f32], head_dim: usize) {
    for (k, d) in keys
        .chunks_exact_mut(head_dim)
        .zip(delta.chunks_exact(head_dim))
    {
        let old = crate::ops::dot_f32(k, k, head_dim).sqrt();
        for (k, d) in k.iter_mut().zip(d) {
            *k += d;
        }
        let scale = old / (crate::ops::dot_f32(k, k, head_dim).sqrt() + 1e-6);
        for k in k {
            *k *= scale;
        }
    }
}
