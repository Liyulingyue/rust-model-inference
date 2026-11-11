use super::{trace, HIDDEN, TEXT_WIDTH};
use crate::compute::{
    linear::{LinearBinding, LinearExecutor, LinearMode},
    ComputePolicy,
};
use crate::core::{
    tensor::{GGMLType, TensorSource},
    thread_pool::ComputePool,
};
use crate::ops::{
    kernel::Weight,
    nn::{attention_scalar, layer_norm_rows, load_float_values, rms_norm_rows, AttentionSpec},
};
use std::{collections::HashMap, sync::Arc};

// Amortize synchronous device submissions at the default 1024px resolution.
const ROWS: usize = 256;
const TYPES: &[GGMLType] = &[
    GGMLType::F32,
    GGMLType::F16,
    GGMLType::BF16,
    GGMLType::Q8_0,
    GGMLType::Q4_0,
    GGMLType::Q4_1,
    GGMLType::Q4K,
    GGMLType::Q5K,
    GGMLType::Q6K,
];

pub struct PjitDit {
    matrices: Vec<(String, Weight<'static>)>,
    vectors: HashMap<String, Vec<f32>>,
    _source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
}
impl PjitDit {
    pub fn load(source: Arc<dyn TensorSource>, threads: usize) -> Result<Self, String> {
        if threads == 0 {
            return Err("P-JiT threads must be positive".into());
        }
        if super::architecture(source.as_ref()) != "pjit"
            || source.metadata("pjit.contract").and_then(|v| v.to_u64()) != Some(1)
        {
            return Err("P-JiT requires architecture pjit and contract 1".into());
        }
        let mut matrices = Vec::new();
        let mut vectors = HashMap::new();
        for (name, dims) in inventory() {
            crate::core::tensor::require_tensor_meta(
                source.as_ref(),
                &name,
                &dims,
                if dims.len() == 2 {
                    TYPES
                } else {
                    &[GGMLType::F32, GGMLType::F16, GGMLType::BF16]
                },
            )?;
            crate::core::tensor::require_tensor_data(source.as_ref(), &name)?;
            if dims.len() == 2 {
                matrices.push((
                    name.clone(),
                    crate::core::loader::load_static_weight(
                        source.as_ref(),
                        &name,
                        dims[0] as usize,
                        dims[1] as usize,
                    ),
                ));
            } else {
                vectors.insert(
                    name.clone(),
                    load_float_values(source.as_ref(), &name, &dims)?,
                );
            }
        }
        Ok(Self {
            matrices,
            vectors,
            _source: source,
            pool: Arc::new(ComputePool::new(threads)),
        })
    }
    pub fn session(&self) -> Result<PjitSession<'_>, String> {
        let executor = LinearExecutor::new(
            ComputePolicy::legacy(),
            self.matrices
                .iter()
                .map(|(_, w)| LinearBinding {
                    weight: w,
                    mode: LinearMode::Prepared,
                })
                .collect(),
            ROWS,
            self.pool.clone(),
        )
        .map_err(|e| e.to_string())?;
        let ids = self
            .matrices
            .iter()
            .enumerate()
            .map(|(i, (n, _))| (n.as_str(), i))
            .collect();
        Ok(PjitSession {
            dit: self,
            executor,
            ids,
        })
    }
}
pub struct PjitSession<'a> {
    dit: &'a PjitDit,
    executor: LinearExecutor<'a, 'static>,
    ids: HashMap<&'a str, usize>,
}
impl PjitSession<'_> {
    fn linear(&mut self, name: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let index = *self
            .ids
            .get(&format!("{name}.weight").as_str())
            .ok_or_else(|| format!("Missing P-JiT projection {name}"))?;
        let w = &self.dit.matrices[index].1;
        let id = self.executor.id(index).map_err(|e| e.to_string())?;
        if x.is_empty() || x.len() % w.n_in != 0 {
            return Err(format!("Invalid {name} input"));
        }
        trace(
            &format!("pjit.linear.{name}.input"),
            &[x.len() / w.n_in, w.n_in],
            x,
        )?;
        let mut y = vec![0.; x.len() / w.n_in * w.n_out];
        for (x, y) in x.chunks(ROWS * w.n_in).zip(y.chunks_mut(ROWS * w.n_out)) {
            self.executor
                .run(id, x, x.len() / w.n_in, y)
                .map_err(|e| e.to_string())?;
        }
        if let Some(b) = self.dit.vectors.get(&format!("{name}.bias")) {
            for row in y.chunks_exact_mut(w.n_out) {
                crate::ops::vec_add_into(b, row);
            }
        }
        trace(
            &format!("pjit.linear.{name}.output"),
            &[y.len() / w.n_out, w.n_out],
            &y,
        )?;
        Ok(y)
    }
    fn norm(&self, x: &[f32], name: &str) -> Result<Vec<f32>, String> {
        rms_norm_rows(
            x,
            x.len() / HIDDEN,
            &self.dit.vectors[&format!("{name}.weight")],
            1e-6,
        )
    }
    fn block(
        &mut self,
        p: &str,
        x: &[f32],
        condition: &[f32],
        grid: (usize, usize, usize),
    ) -> Result<Vec<f32>, String> {
        let rows = x.len() / HIDDEN;
        let mods = self.linear(&format!("{p}.adaln_up"), condition)?;
        let mut normalized = self.norm(x, &format!("{p}.norm1"))?;
        for row in normalized.chunks_exact_mut(HIDDEN) {
            for i in 0..HIDDEN {
                row[i] *= 1. + mods[i];
            }
        }
        let mut q = self.linear(&format!("{p}.attn.q"), &normalized)?;
        let mut k = self.linear(&format!("{p}.attn.k"), &normalized)?;
        let v = self.linear(&format!("{p}.attn.v"), &normalized)?;
        q = self.norm(&q, &format!("{p}.attn.norm_q"))?;
        k = self.norm(&k, &format!("{p}.attn.norm_k"))?;
        rope(&mut q, grid);
        rope(&mut k, grid);
        let attention = if ComputePolicy::legacy() == ComputePolicy::Cpu {
            cpu_attention
        } else {
            device_attention
        };
        let mut a = attention(
            &q,
            &k,
            &v,
            AttentionSpec {
                query_tokens: rows,
                key_tokens: rows,
                query_heads: 23,
                key_value_heads: 23,
                head_dim: 128,
                causal: false,
                scale: 1. / 128f32.sqrt(),
            },
            &self.dit.pool,
        )?;
        let gates = self.linear(&format!("{p}.attn.gate_proj"), &normalized)?;
        for (head, gate) in a.chunks_exact_mut(128).zip(gates) {
            let gate = 1. / (1. + (-gate).exp());
            for value in head {
                *value *= gate;
            }
        }
        a = self.linear(&format!("{p}.attn.o"), &a)?;
        a = self.norm(&a, &format!("{p}.post_norm1"))?;
        let mut result = x.to_vec();
        for row in 0..rows {
            for i in 0..HIDDEN {
                let j = row * HIDDEN + i;
                result[j] += mods[HIDDEN + i].tanh() * a[j];
            }
        }
        normalized = self.norm(&result, &format!("{p}.norm2"))?;
        for row in normalized.chunks_exact_mut(HIDDEN) {
            for i in 0..HIDDEN {
                row[i] *= 1. + mods[2 * HIDDEN + i];
            }
        }
        let ff = self.linear(&format!("{p}.ffn.w13"), &normalized)?;
        let width = ff.len() / rows / 2;
        let mut activated = vec![0.; rows * width];
        for (out, row) in activated
            .chunks_exact_mut(width)
            .zip(ff.chunks_exact(2 * width))
        {
            out.copy_from_slice(&row[..width]);
            crate::ops::silu_inplace(out);
            for (v, u) in out.iter_mut().zip(&row[width..]) {
                *v *= u;
            }
        }
        let ff = self.linear(&format!("{p}.ffn.w2"), &activated)?;
        let ff = self.norm(&ff, &format!("{p}.post_norm2"))?;
        for row in 0..rows {
            for i in 0..HIDDEN {
                let j = row * HIDDEN + i;
                result[j] += mods[3 * HIDDEN + i].tanh() * ff[j];
            }
        }
        super::finite(&result, p)?;
        trace(p, &[rows, HIDDEN], &result)?;
        Ok(result)
    }
    pub fn predict(
        &mut self,
        image: &[f32],
        width: usize,
        height: usize,
        text: &[f32],
        timestep: usize,
    ) -> Result<Vec<f32>, String> {
        super::validate_shape(width, height)?;
        if image.len() != 3 * width * height
            || text.is_empty()
            || text.len() % TEXT_WIDTH != 0
            || text.len() / TEXT_WIDTH > 256
            || timestep > 1000
        {
            return Err("Invalid P-JiT prediction input".into());
        }
        super::finite(image, "image")?;
        super::finite(text, "text")?;
        let grid = (height.div_ceil(32), width.div_ceil(32));
        let image_rows = grid.0 * grid.1;
        let mut image = self.linear("patch_embed.proj1", &super::patchify(image, width, height))?;
        image = self.linear("patch_embed.proj2", &image)?;
        trace("pjit.patch", &[image_rows, HIDDEN], &image)?;
        let mut time = vec![0.; 256];
        for i in 0..128 {
            let angle = timestep as f64 * 10000f64.powf(-(i as f64) / 128.);
            time[i] = angle.cos() as f32;
            time[i + 128] = angle.sin() as f32;
        }
        time = self.linear("time_embed.0", &time)?;
        crate::ops::silu_inplace(&mut time);
        time = self.linear("time_embed.2", &time)?;
        let mut activated = time.clone();
        crate::ops::silu_inplace(&mut activated);
        let ci = self.linear("image_adaln.1", &activated)?;
        let ct = self.linear("text_adaln.1", &activated)?;
        let mut text = self.linear("text_proj.0", text)?;
        crate::ops::gelu_approx_inplace(&mut text);
        text = self.linear("text_proj.2", &text)?;
        // Invalid padded caption rows are omitted: all attention is over valid prefixes.
        for i in 0..2 {
            image = self.block(
                &format!("image_refiner.{i}"),
                &image,
                &ci,
                (grid.0, grid.1, image_rows),
            )?;
        }
        for i in 0..2 {
            text = self.block(&format!("text_refiner.{i}"), &text, &ct, (0, 0, 0))?;
        }
        image.extend(text);
        for i in 0..22 {
            image = self.block(
                &format!("blocks.{i}"),
                &image,
                &ci,
                (grid.0, grid.1, image_rows),
            )?;
        }
        let mut head = layer_norm_rows(
            &image[..image_rows * HIDDEN],
            image_rows,
            Some(&self.dit.vectors["head.norm.weight"]),
            Some(&self.dit.vectors["head.norm.bias"]),
            1e-5,
        )?;
        let modulation = &self.dit.vectors["head.modulation"];
        for row in head.chunks_exact_mut(HIDDEN) {
            for i in 0..HIDDEN {
                row[i] = row[i] * (1. + time[i] + modulation[HIDDEN + i]) + time[i] + modulation[i];
            }
        }
        let patches = self.linear("head.proj", &head)?;
        let output = super::unpatchify(&patches, width, height);
        super::finite(&output, "prediction")?;
        trace("pjit.prediction", &[3, height, width], &output)?;
        Ok(output)
    }
}
fn device_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    spec: AttentionSpec,
    _pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    attention_scalar(q, k, v, spec)
}
pub(super) fn cpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    spec: AttentionSpec,
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    let outputs = std::sync::Mutex::new(Vec::new());
    pool.compute(|ith, nth| {
        for head in (ith..spec.query_heads).step_by(nth) {
            let gather = |values: &[f32], rows: usize| {
                (0..rows)
                    .flat_map(|row| {
                        let start = (row * spec.query_heads + head) * spec.head_dim;
                        values[start..start + spec.head_dim].iter().copied()
                    })
                    .collect::<Vec<_>>()
            };
            let result = crate::ops::nn::attention_on_cpu(
                &gather(q, spec.query_tokens),
                &gather(k, spec.key_tokens),
                &gather(v, spec.key_tokens),
                AttentionSpec {
                    query_heads: 1,
                    key_value_heads: 1,
                    ..spec
                },
            );
            outputs.lock().unwrap().push((head, result));
        }
    });
    let mut output = vec![0.; q.len()];
    for (head, values) in outputs.into_inner().map_err(|e| e.to_string())? {
        for (row, values) in values?.chunks_exact(spec.head_dim).enumerate() {
            let start = (row * spec.query_heads + head) * spec.head_dim;
            output[start..start + spec.head_dim].copy_from_slice(values);
        }
    }
    Ok(output)
}
pub(super) fn rope(x: &mut [f32], grid: (usize, usize, usize)) {
    // Spatial coordinates share the same phases across all tokens and 23 heads.
    let phases: Vec<Vec<_>> = (0..grid.0.max(grid.1))
        .map(|position| {
            (0..21)
                .map(|pair| {
                    let angle = position as f64 * 0.5 / 10000f64.powf(pair as f64 / 21.);
                    let (s, c) = angle.sin_cos();
                    (s as f32, c as f32)
                })
                .collect()
        })
        .collect();
    for (token, row) in x.chunks_exact_mut(HIDDEN).enumerate() {
        if token >= grid.2 {
            continue;
        }
        for head in row.chunks_exact_mut(128) {
            for (start, pairs, pos) in [
                (0, 22, 0),
                (44, 21, token / grid.1),
                (86, 21, token % grid.1),
            ] {
                for pair in 0..pairs {
                    let (s, c) = if start == 0 {
                        (0., 1.)
                    } else {
                        phases[pos][pair]
                    };
                    let i = start + pair * 2;
                    let (a, b) = (head[i], head[i + 1]);
                    head[i] = a * c - b * s;
                    head[i + 1] = a * s + b * c;
                }
            }
        }
    }
}
fn inventory() -> Vec<(String, Vec<u64>)> {
    let mut items = Vec::new();
    let mut linear = |name: String, ni, no, bias| {
        items.push((format!("{name}.weight"), vec![ni, no]));
        if bias {
            items.push((format!("{name}.bias"), vec![no]));
        }
    };
    linear("patch_embed.proj1".into(), 3072, 1024, false);
    linear("patch_embed.proj2".into(), 1024, 2944, true);
    linear("time_embed.0".into(), 256, 2944, true);
    linear("time_embed.2".into(), 2944, 2944, true);
    for p in ["image_adaln.1", "text_adaln.1"] {
        linear(p.into(), 2944, 256, true);
    }
    linear("text_proj.0".into(), 7680, 2944, false);
    linear("text_proj.2".into(), 2944, 2944, false);
    for (stack, count, ff) in [
        ("image_refiner", 2, 2816),
        ("text_refiner", 2, 2816),
        ("blocks", 22, 5376),
    ] {
        for i in 0..count {
            let p = format!("{stack}.{i}");
            for proj in ["q", "k", "v", "o"] {
                linear(format!("{p}.attn.{proj}"), 2944, 2944, true);
            }
            linear(format!("{p}.attn.gate_proj"), 2944, 23, true);
            linear(format!("{p}.ffn.w13"), 2944, 2 * ff, false);
            linear(format!("{p}.ffn.w2"), ff, 2944, false);
            linear(format!("{p}.adaln_up"), 256, 4 * 2944, true);
        }
    }
    linear("head.proj".into(), 2944, 3072, true);
    for (stack, count) in [("image_refiner", 2), ("text_refiner", 2), ("blocks", 22)] {
        for i in 0..count {
            for norm in [
                "norm1",
                "norm2",
                "post_norm1",
                "post_norm2",
                "attn.norm_q",
                "attn.norm_k",
            ] {
                items.push((format!("{stack}.{i}.{norm}.weight"), vec![2944]));
            }
        }
    }
    for name in ["head.norm.weight", "head.norm.bias"] {
        items.push((name.into(), vec![2944]));
    }
    items.push(("head.modulation".into(), vec![5888]));
    items
}
