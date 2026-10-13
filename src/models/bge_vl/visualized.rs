use super::{clip, json_metadata, Linear, Norm};
use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::silu_inplace;
use std::path::Path;

pub(super) fn compute(
    source: &dyn TensorSource,
    prompt: &str,
    image: Option<&Path>,
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    let Some(image) = image else {
        return crate::models::bert_family::compute_embedding(source, prompt, pool.n_threads());
    };
    let cfg: clip::Config = json_metadata(source, "bge.config")?;
    let h = cfg.hidden_size;
    let p = cfg.patch_size;
    let side = cfg.image_size;
    if h == 0
        || cfg.intermediate_size == 0
        || cfg.num_hidden_layers == 0
        || !cfg.layer_norm_eps.is_finite()
        || cfg.layer_norm_eps <= 0.0
        || side == 0
        || p == 0
        || !side.is_multiple_of(p)
        || cfg.num_attention_heads == 0
        || !h.is_multiple_of(cfg.num_attention_heads)
        || !(h / cfg.num_attention_heads).is_multiple_of(2)
    {
        return Err("invalid EVA dimensions".into());
    }
    let n = (side / p).pow(2) + 1;
    let head_dim = h / cfg.num_attention_heads;
    let prefix = "model_visual.visual";
    let pixels = super::preprocess(source, image, side, false)?;
    let mut patches = Vec::with_capacity((n - 1) * p * p * 3);
    for y in 0..side / p {
        for x in 0..side / p {
            for c in 0..3 {
                for py in 0..p {
                    for px in 0..p {
                        patches.push(pixels[((y * p + py) * side + x * p + px) * 3 + c]);
                    }
                }
            }
        }
    }
    let patch = Linear::load_shape(
        source,
        &format!("{prefix}.patch_embed.proj.weight"),
        &[p as u64, p as u64, 3, h as u64],
        p * p * 3,
        h,
        Some(format!("{prefix}.patch_embed.proj.bias")),
    )?;
    let mut hidden = load_f32_tensor(source, &format!("{prefix}.cls_token"), &[h as u64, 1, 1])?;
    hidden.extend(patch.apply(&patches, pool)?);
    let positions = load_f32_tensor(
        source,
        &format!("{prefix}.pos_embed"),
        &[h as u64, n as u64, 1],
    )?;
    for (x, pos) in hidden.iter_mut().zip(positions) {
        *x += pos;
    }
    for i in 0..cfg.num_hidden_layers {
        let b = format!("{prefix}.blocks.{i}");
        let normed =
            Norm::load(source, &format!("{b}.norm1"), h, cfg.layer_norm_eps)?.apply(&hidden);
        let mut q = Linear::load_shape(
            source,
            &format!("{b}.attn.q_proj.weight"),
            &[h as u64, h as u64],
            h,
            h,
            Some(format!("{b}.attn.q_bias")),
        )?
        .apply(&normed, pool)?;
        let mut k =
            Linear::load(source, &format!("{b}.attn.k_proj"), h, h, false)?.apply(&normed, pool)?;
        let v = Linear::load_shape(
            source,
            &format!("{b}.attn.v_proj.weight"),
            &[h as u64, h as u64],
            h,
            h,
            Some(format!("{b}.attn.v_bias")),
        )?
        .apply(&normed, pool)?;
        let cos = load_f32_tensor(
            source,
            &format!("{b}.attn.rope.freqs_cos"),
            &[head_dim as u64, (n - 1) as u64],
        )?;
        let sin = load_f32_tensor(
            source,
            &format!("{b}.attn.rope.freqs_sin"),
            &[head_dim as u64, (n - 1) as u64],
        )?;
        rotate(&mut q, h, head_dim, &cos, &sin);
        rotate(&mut k, h, head_dim, &cos, &sin);
        let scale = 1.0 / (head_dim as f32).sqrt();
        for x in &mut q {
            *x *= scale;
        }
        let attended = clip::attention(&q, &k, &v, n, h, cfg.num_attention_heads, false);
        let attended = Norm::load(
            source,
            &format!("{b}.attn.inner_attn_ln"),
            h,
            cfg.layer_norm_eps,
        )?
        .apply(&attended);
        let projected =
            Linear::load(source, &format!("{b}.attn.proj"), h, h, true)?.apply(&attended, pool)?;
        for (x, v) in hidden.iter_mut().zip(projected) {
            *x += v;
        }
        let normed =
            Norm::load(source, &format!("{b}.norm2"), h, cfg.layer_norm_eps)?.apply(&hidden);
        let ff = cfg.intermediate_size;
        let mut gate =
            Linear::load(source, &format!("{b}.mlp.w1"), h, ff, true)?.apply(&normed, pool)?;
        let up = Linear::load(source, &format!("{b}.mlp.w2"), h, ff, true)?.apply(&normed, pool)?;
        silu_inplace(&mut gate);
        for (g, u) in gate.iter_mut().zip(up) {
            *g *= u;
        }
        let gate =
            Norm::load(source, &format!("{b}.mlp.ffn_ln"), ff, cfg.layer_norm_eps)?.apply(&gate);
        let projected =
            Linear::load(source, &format!("{b}.mlp.w3"), ff, h, true)?.apply(&gate, pool)?;
        for (x, v) in hidden.iter_mut().zip(projected) {
            *x += v;
        }
    }
    let projected = Linear::load(source, "visual_proj", h, h, true)?.apply(&hidden[h..], pool)?;
    let ids = crate::models::bert_family::encode_embedding_input(source, prompt)?;
    crate::models::bert_family::run_embedding_tokens_with_image(
        source,
        &ids,
        &projected,
        pool.n_threads(),
    )
}

fn rotate(values: &mut [f32], width: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    for (t, row) in values.chunks_exact_mut(width).enumerate().skip(1) {
        for head in row.chunks_exact_mut(head_dim) {
            let offset = (t - 1) * head_dim;
            for i in (0..head_dim).step_by(2) {
                let a = head[i];
                let b = head[i + 1];
                head[i] = a * cos[offset + i] - b * sin[offset + i];
                head[i + 1] = b * cos[offset + i + 1] + a * sin[offset + i + 1];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::rotate;
    #[test]
    fn eva_rope_preserves_cls_and_rotates_each_head_using_saved_tables() {
        let mut v = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        rotate(&mut v, 4, 2, &[0., 0.], &[1., 1.]);
        assert_eq!(v, [1., 2., 3., 4., -6., 5., -8., 7.]);
    }
}
