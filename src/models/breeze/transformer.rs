//! Native BF16 T5Gemma2/Qwen3/Breeze-depth operations. Weight storage is unchanged.
use super::{bf, trace};
use crate::core::tensor::{load_f32_tensor, GGMLType, TensorSource};
use crate::models::dots::weights::load_weight;
use crate::ops::kernel::Weight;
use rayon::prelude::*;

pub(super) fn matrix<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    input: usize,
    output: usize,
) -> Result<Weight<'a>, String> {
    if source.tensor_info(name).map(|t| t.ggml_type) != Some(GGMLType::BF16) {
        return Err(format!("{name}: Breeze requires original BF16 weights"));
    }
    load_weight(source, name, &[input as u64, output as u64])
}

pub(super) fn linear(weight: &Weight<'_>, input: &[f32]) -> Vec<f32> {
    let ni = weight.n_in;
    let no = weight.n_out;
    assert_eq!(input.len() % ni, 0);
    let bytes = weight.kernel.bf16_bytes().expect("validated BF16 matrix");
    let mut out = vec![0.0; input.len() / ni * no];
    for (x, y) in input.chunks_exact(ni).zip(out.chunks_exact_mut(no)) {
        y.par_chunks_mut(64).enumerate().for_each(|(chunk, y)| {
            let start = chunk * 64 * ni * 2;
            let end = start + y.len() * ni * 2;
            for (out, row) in y.iter_mut().zip(bytes[start..end].chunks_exact(ni * 2)) {
                *out = super::bf16_math::dot_bf16(row, x);
            }
        });
    }
    out
}

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Kind {
    Text,
    Backbone,
    Depth,
}
impl Kind {
    fn dimensions(self) -> (usize, usize, usize, usize, usize, usize, f32) {
        match self {
            Self::Text => (1152, 26, 4, 1, 256, 6912, 1e-6),
            Self::Backbone => (2048, 28, 16, 8, 128, 6144, 1e-6),
            Self::Depth => (1024, 12, 8, 2, 128, 8192, 1e-5),
        }
    }
    fn prefix(self) -> &'static str {
        match self {
            Self::Text => "text_encoder",
            Self::Backbone => "backbone_model",
            Self::Depth => "depth_decoder.model",
        }
    }
    fn trace(self) -> &'static str {
        match self {
            Self::Text => "breeze.text",
            Self::Backbone => "breeze.backbone",
            Self::Depth => "breeze.depth",
        }
    }
}

struct Layer<'a> {
    index: usize,
    input_norm: Vec<f32>,
    ff_norm: Vec<f32>,
    attn_post: Option<Vec<f32>>,
    ff_post: Option<Vec<f32>>,
    q: Weight<'a>,
    k: Weight<'a>,
    v: Weight<'a>,
    o: Weight<'a>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    gate: Weight<'a>,
    up: Weight<'a>,
    down: Weight<'a>,
}

#[derive(Default)]
pub(super) struct Cache {
    layers: Vec<(Vec<f32>, Vec<f32>)>,
    pub len: usize,
}

pub(super) struct Transformer<'a> {
    kind: Kind,
    layers: Vec<Layer<'a>>,
    norm: Vec<f32>,
}
impl<'a> Transformer<'a> {
    pub fn load(source: &'a dyn TensorSource, kind: Kind) -> Result<Self, String> {
        let (hidden, count, heads, kv, hd, ff, _) = kind.dimensions();
        let prefix = kind.prefix();
        let mut layers = Vec::with_capacity(count);
        for index in 0..count {
            let p = format!("{prefix}.layers.{index}");
            let w = |name: &str, i, o| matrix(source, &format!("{p}.{name}.weight"), i, o);
            let norm =
                |name: &str, d| load_f32_tensor(source, &format!("{p}.{name}.weight"), &[d as u64]);
            let text = kind == Kind::Text;
            layers.push(Layer {
                index,
                input_norm: norm(
                    if text {
                        "pre_self_attn_layernorm"
                    } else {
                        "input_layernorm"
                    },
                    hidden,
                )?,
                ff_norm: norm(
                    if text {
                        "pre_feedforward_layernorm"
                    } else {
                        "post_attention_layernorm"
                    },
                    hidden,
                )?,
                attn_post: if text {
                    Some(norm("post_self_attn_layernorm", hidden)?)
                } else {
                    None
                },
                ff_post: if text {
                    Some(norm("post_feedforward_layernorm", hidden)?)
                } else {
                    None
                },
                q: w("self_attn.q_proj", hidden, heads * hd)?,
                k: w("self_attn.k_proj", hidden, kv * hd)?,
                v: w("self_attn.v_proj", hidden, kv * hd)?,
                o: w("self_attn.o_proj", heads * hd, hidden)?,
                q_norm: if kind != Kind::Depth {
                    Some(norm("self_attn.q_norm", hd)?)
                } else {
                    None
                },
                k_norm: if kind != Kind::Depth {
                    Some(norm("self_attn.k_norm", hd)?)
                } else {
                    None
                },
                gate: w("mlp.gate_proj", hidden, ff)?,
                up: w("mlp.up_proj", hidden, ff)?,
                down: w("mlp.down_proj", ff, hidden)?,
            });
        }
        Ok(Self {
            kind,
            layers,
            norm: load_f32_tensor(source, &format!("{prefix}.norm.weight"), &[hidden as u64])?,
        })
    }

    /// Text batches are right padded and independent; causal sessions use one row.
    pub fn forward(
        &self,
        mut x: Vec<f32>,
        lengths: &[usize],
        mut cache: Option<&mut Cache>,
        step: Option<usize>,
    ) -> Result<Vec<f32>, String> {
        let (hidden, _, heads, kv, hd, _, eps) = self.kind.dimensions();
        let batch = lengths.len();
        if batch == 0 || x.len() % (batch * hidden) != 0 {
            return Err("Invalid Breeze hidden-state shape".into());
        }
        let n = x.len() / batch / hidden;
        if n == 0 || lengths.iter().any(|&l| l == 0 || l > n) {
            return Err("Invalid Breeze sequence length".into());
        }
        let text = self.kind == Kind::Text;
        let start = cache.as_ref().map_or(0, |c| c.len);
        if let Some(c) = cache.as_mut() {
            if c.layers.is_empty() {
                c.layers.resize_with(self.layers.len(), Default::default);
            }
        }
        let shape = if batch == 1 {
            vec![n, hidden]
        } else {
            vec![batch, n, hidden]
        };
        let fine_prefix = match self.kind {
            Kind::Text => "RMI_BREEZE_FINE_TEXT",
            Kind::Backbone => "RMI_BREEZE_FINE_BACKBONE",
            Kind::Depth => "RMI_BREEZE_FINE_DEPTH",
        };
        let fine_enabled = std::env::var_os(fine_prefix).is_some();
        let fine_layer = std::env::var(format!("{fine_prefix}_LAYER"))
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        for (cache_index, layer) in self.layers.iter().enumerate() {
            let index = layer.index;
            let fine = |name: &str, dims: &[usize], values: &[f32]| -> Result<(), String> {
                if fine_enabled && index == fine_layer {
                    trace(
                        &format!("{}.layer.{index}.{name}", self.kind.trace()),
                        Some(index),
                        step,
                        dims,
                        values,
                    )?;
                }
                Ok(())
            };
            let normed = rms(&x, &layer.input_norm, eps, text);
            fine("pre_self_attn", &shape, &normed)?;
            let mut q = linear(&layer.q, &normed);
            let mut k = linear(&layer.k, &normed);
            let v = linear(&layer.v, &normed);
            let dims = |width| {
                if batch == 1 {
                    vec![n, width]
                } else {
                    vec![batch, n, width]
                }
            };
            fine("q_proj", &dims(heads * hd), &q)?;
            if text {
                fine("k_proj", &dims(kv * hd), &k)?;
                fine("v_proj", &dims(kv * hd), &v)?;
            }
            if let Some(w) = &layer.q_norm {
                q = rms(&q, w, eps, text);
                if fine_enabled && index == fine_layer {
                    fine(
                        "q_norm",
                        &[batch, heads, n, hd],
                        &head_major(&q, batch, n, heads, hd),
                    )?;
                }
            }
            if !text {
                fine("k_proj", &dims(kv * hd), &k)?;
            }
            if let Some(w) = &layer.k_norm {
                k = rms(&k, w, eps, text);
                if fine_enabled && index == fine_layer {
                    fine(
                        "k_norm",
                        &[batch, kv, n, hd],
                        &head_major(&k, batch, n, kv, hd),
                    )?;
                }
            }
            if !text {
                fine("v_proj", &dims(kv * hd), &v)?;
            }
            let local = text && (index + 1) % 6 != 0;
            let (theta, factor) = match self.kind {
                Kind::Text if local => (10000.0, 1.0),
                Kind::Text => (1_000_000.0, 8.0),
                Kind::Backbone => (1_000_000.0, 1.0),
                Kind::Depth => (500_000.0, 1.0),
            };
            let freq = inv_freq(hd, theta, factor, self.kind == Kind::Depth);
            for (b, &len) in lengths.iter().enumerate() {
                for t in 0..n {
                    let pos = if text && t >= len { 0 } else { start + t };
                    rope(
                        &mut q[(b * n + t) * heads * hd..(b * n + t + 1) * heads * hd],
                        hd,
                        pos,
                        &freq,
                    );
                    rope(
                        &mut k[(b * n + t) * kv * hd..(b * n + t + 1) * kv * hd],
                        hd,
                        pos,
                        &freq,
                    );
                }
            }
            if !text {
                fine("q_rope", &dims(heads * hd), &q)?;
                fine("k_rope", &dims(kv * hd), &k)?;
            }
            let (keys, values, nk) = if let Some(c) = cache.as_mut() {
                let (keys, values) = &mut c.layers[cache_index];
                keys.extend_from_slice(&k);
                values.extend_from_slice(&v);
                (&keys[..], &values[..], start + n)
            } else {
                (&k[..], &v[..], n)
            };
            let mut attended = vec![0.0; batch * n * heads * hd];
            attended
                .par_chunks_mut(heads * hd)
                .enumerate()
                .for_each(|(row, output)| {
                    let b = row / n;
                    let t = row % n;
                    for h in 0..heads {
                        let kh = h / (heads / kv);
                        let query = &q[(row * heads + h) * hd..(row * heads + h + 1) * hd];
                        let mut scores = vec![f32::NEG_INFINITY; nk];
                        for s in 0..nk {
                            let allowed = if text {
                                s < lengths[b]
                                    && (!local || (s <= t && t - s < 256) || (s > t && s - t < 257))
                            } else {
                                s <= start + t
                            };
                            if allowed {
                                let off = ((b * nk + s) * kv + kh) * hd;
                                scores[s] =
                                    bf(bf(crate::ops::dot_f32(query, &keys[off..off + hd], hd))
                                        * (hd as f32).sqrt().recip());
                            }
                        }
                        softmax(&mut scores);
                        for p in &mut scores {
                            *p = bf(*p);
                        }
                        for d in 0..hd {
                            let mut sums = [0.0f32; 4];
                            for s in 0..nk {
                                let lane = if n * nk * hd >= 400 && s < nk / 4 * 4 {
                                    s % 4
                                } else {
                                    0
                                };
                                sums[lane] += scores[s] * values[((b * nk + s) * kv + kh) * hd + d];
                            }
                            output[h * hd + d] = bf(((sums[0] + sums[1]) + sums[2]) + sums[3]);
                        }
                    }
                });
            if fine_enabled && index == fine_layer && !text {
                let mut raw = Vec::with_capacity(batch * heads * n * nk);
                let mut probabilities = Vec::with_capacity(raw.capacity());
                for b in 0..batch {
                    for h in 0..heads {
                        let kh = h / (heads / kv);
                        for t in 0..n {
                            let query = &q[((b * n + t) * heads + h) * hd
                                ..((b * n + t) * heads + h + 1) * hd];
                            let mut scores = (0..nk)
                                .map(|s| {
                                    let off = ((b * nk + s) * kv + kh) * hd;
                                    bf(bf(crate::ops::dot_f32(query, &keys[off..off + hd], hd))
                                        * (hd as f32).sqrt().recip())
                                })
                                .collect::<Vec<_>>();
                            for (s, score) in scores.iter_mut().enumerate() {
                                if s > start + t {
                                    *score = f32::from_bits(0xff7f0000);
                                }
                            }
                            raw.extend_from_slice(&scores);
                            softmax(&mut scores);
                            probabilities.extend(scores.into_iter().map(bf));
                        }
                    }
                }
                let score_shape = if batch == 1 {
                    vec![heads, n, nk]
                } else {
                    vec![batch, heads, n, nk]
                };
                fine("attn_scores", &score_shape, &raw)?;
                fine("probabilities", &score_shape, &probabilities)?;
                fine("attended", &dims(heads * hd), &attended)?;
            }
            let mut a = linear(&layer.o, &attended);
            fine("attn_output", &shape, &a)?;
            if let Some(w) = &layer.attn_post {
                a = rms(&a, w, eps, true);
                fine("post_self_attn", &shape, &a)?;
            }
            residual(&mut x, &a);
            let normed = rms(&x, &layer.ff_norm, eps, text);
            fine("pre_ff", &shape, &normed)?;
            let mut gate = linear(&layer.gate, &normed);
            fine("gate", &dims(layer.gate.n_out), &gate)?;
            for g in &mut gate {
                let activated = if text {
                    gelu_tanh(*g)
                } else {
                    *g / (1.0 + (-*g).exp())
                };
                *g = bf(activated);
            }
            fine("activation", &dims(layer.gate.n_out), &gate)?;
            let up = linear(&layer.up, &normed);
            fine("up", &dims(layer.up.n_out), &up)?;
            for (g, u) in gate.iter_mut().zip(up) {
                *g = bf(*g * u);
            }
            fine("ff_input", &dims(layer.gate.n_out), &gate)?;
            let mut f = linear(&layer.down, &gate);
            fine("ff", &shape, &f)?;
            if let Some(w) = &layer.ff_post {
                f = rms(&f, w, eps, true);
                fine("post_ff", &shape, &f)?;
            }
            residual(&mut x, &f);
            trace(
                &format!("{}.layer.{index}", self.kind.trace()),
                Some(index),
                step,
                &shape,
                &x,
            )?;
        }
        x = rms(&x, &self.norm, eps, text);
        trace(
            &format!("{}.norm", self.kind.trace()),
            None,
            step,
            &shape,
            &x,
        )?;
        if let Some(c) = cache {
            c.len += n;
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "{} produced non-finite hidden states",
                self.kind.trace()
            ));
        }
        Ok(x)
    }
}

fn gelu_tanh(x: f32) -> f32 {
    let cube = (x * x) * x;
    let inner = 0.797_884_560_802_865_4_f32 * (x + 0.044715_f32 * cube);
    (0.5 * x) * (1.0 + crate::models::dots::speaker::exp::torch28_tanh(inner))
}

fn residual(x: &mut [f32], add: &[f32]) {
    for (x, a) in x.iter_mut().zip(add) {
        *x = bf(*x + a);
    }
}

pub(super) fn rms(x: &[f32], w: &[f32], eps: f32, gemma: bool) -> Vec<f32> {
    let mut out = x.to_vec();
    for row in out.chunks_exact_mut(w.len()) {
        let variance = sum_squares(row) / w.len() as f32;
        let scale = (variance + eps).sqrt().recip();
        for (v, w) in row.iter_mut().zip(w) {
            *v = if gemma {
                bf((*v * scale) * (1.0 + w))
            } else {
                bf(bf(*v * scale) * w)
            };
        }
    }
    out
}

// PyTorch 2.9 CPU cascade reduction: four independent 4-lane vectors,
// with 16-vector partial sums. Keep additions separate from the square.
fn sum_squares(x: &[f32]) -> f32 {
    let groups = x.len() / 16;
    let mut levels = [[[0.0f32; 4]; 4]; 4];
    let mut i = 0;
    while i + 16 <= groups {
        for _ in 0..16 {
            for k in 0..4 {
                for lane in 0..4 {
                    let v = x[i * 16 + k * 4 + lane];
                    levels[0][k][lane] += v * v;
                }
            }
            i += 1;
        }
        for level in 1..4 {
            for k in 0..4 {
                for lane in 0..4 {
                    levels[level][k][lane] += levels[level - 1][k][lane];
                    levels[level - 1][k][lane] = 0.0;
                }
            }
            if i & (15 << (4 * level)) != 0 {
                break;
            }
        }
    }
    while i < groups {
        for k in 0..4 {
            for lane in 0..4 {
                let v = x[i * 16 + k * 4 + lane];
                levels[0][k][lane] += v * v;
            }
        }
        i += 1;
    }
    for level in 1..4 {
        for k in 0..4 {
            for lane in 0..4 {
                levels[0][k][lane] += levels[level][k][lane];
            }
        }
    }
    for j in groups * 4..x.len() / 4 {
        for lane in 0..4 {
            let v = x[j * 4 + lane];
            levels[0][0][lane] += v * v;
        }
    }
    for k in 1..4 {
        for lane in 0..4 {
            levels[0][0][lane] += levels[0][k][lane];
        }
    }
    let mut sum = 0.0;
    for &v in &x[x.len() / 4 * 4..] {
        sum += v * v;
    }
    for v in levels[0][0] {
        sum += v;
    }
    sum
}

fn inv_freq(dim: usize, theta: f32, linear_factor: f32, llama3: bool) -> Vec<f32> {
    (0..dim / 2)
        .map(|i| {
            let freq = (1.0 / theta.powf((2 * i) as f32 / dim as f32)) / linear_factor;
            if !llama3 {
                return freq;
            }
            let wavelength = 2.0 * std::f32::consts::PI / freq;
            if wavelength > 8192.0 {
                freq / 32.0
            } else if wavelength < 2048.0 {
                freq
            } else {
                let smooth = (16.0 / wavelength - 0.001953125) / (0.0078125 - 0.001953125);
                (1.0 - smooth) * freq / 32.0 + smooth * freq
            }
        })
        .collect()
}

fn rope(x: &mut [f32], hd: usize, pos: usize, freq: &[f32]) {
    for head in x.chunks_exact_mut(hd) {
        for i in 0..hd / 2 {
            let angle = pos as f32 * freq[i];
            let c = bf(angle.cos());
            let s = bf(angle.sin());
            let a = head[i];
            let b = head[i + hd / 2];
            head[i] = bf(bf(a * c) + bf(-b * s));
            head[i + hd / 2] = bf(bf(b * c) + bf(a * s));
        }
    }
}

pub(super) fn softmax(scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Torch's F32 CPU softmax reduces four SIMD lanes, then multiplies by the reciprocal.
    let mut sums = [0.0f32; 4];
    for (index, p) in scores.iter_mut().enumerate() {
        *p = crate::models::dots::speaker::exp::torch28_exp(*p - max);
        sums[index % 4] += *p;
    }
    let total = if scores.len() < 4 {
        (sums[0] + sums[1]) + sums[2]
    } else {
        (sums[0] + sums[2]) + (sums[1] + sums[3])
    };
    let inverse = total.recip();
    for p in scores {
        *p *= inverse;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires BREEZE_GGUF and BREEZE_ORACLE_TRACE; replays one causal layer with its real KV history"]
    fn causal_layer_replay() {
        use crate::format::ggufrs::{open_model_source, ComponentRole};
        let path = std::env::var("BREEZE_GGUF").unwrap();
        let source = open_model_source(std::path::Path::new(&path), ComponentRole::Llm).unwrap();
        let kind = match std::env::var("BREEZE_REPLAY_KIND").as_deref() {
            Ok("backbone") => Kind::Backbone,
            _ => Kind::Depth,
        };
        let index: usize = std::env::var("BREEZE_REPLAY_LAYER")
            .unwrap()
            .parse()
            .unwrap();
        let last_step: usize = std::env::var("BREEZE_REPLAY_STEP")
            .unwrap()
            .parse()
            .unwrap();
        let records =
            std::fs::read_to_string(std::env::var("BREEZE_ORACLE_TRACE").unwrap()).unwrap();
        let input_name = if index == 0 {
            format!("{}.input", kind.trace())
        } else {
            format!("{}.layer.{}", kind.trace(), index - 1)
        };
        let mut model = Transformer::load(source.as_ref(), kind).unwrap();
        let layer = model.layers.remove(index);
        model.layers = vec![layer];
        let mut cache = Cache::default();
        let mut count = 0;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for line in records.lines() {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            if record["name"] != input_name {
                continue;
            }
            let step = record["step"].as_u64().unwrap() as usize;
            if step > last_step {
                break;
            }
            if kind == Kind::Depth && step % 15 == 0 {
                cache = Cache::default();
            }
            let bytes = std::fs::read(record["binary_path"].as_str().unwrap()).unwrap();
            let input = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let n = input.len() / kind.dimensions().0;
            let output = pool
                .install(|| model.forward(input, &[n], Some(&mut cache), Some(step)))
                .unwrap();
            assert_eq!(output.len(), n * kind.dimensions().0);
            count += 1;
            if count == last_step + 1 {
                break;
            }
        }
        assert_eq!(count, last_step + 1);
    }

    #[test]
    #[ignore = "requires BREEZE_GGUF and RMI_PARITY_TRACE; traces the first text layer"]
    fn text_layer_replay() {
        use crate::format::ggufrs::{open_model_source, ComponentRole};
        let path = std::env::var("BREEZE_GGUF").unwrap();
        let source = open_model_source(std::path::Path::new(&path), ComponentRole::Llm).unwrap();
        let embeddings = matrix(
            source.as_ref(),
            "text_encoder.embed_tokens.weight",
            1152,
            262158,
        )
        .unwrap();
        let mut model = Transformer::load(source.as_ref(), Kind::Text).unwrap();
        model.layers.truncate(
            std::env::var("BREEZE_TEXT_LAYERS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1),
        );
        let ids = [2, 262146, 144626, 236924];
        let mut input = vec![0.0; 4 * 1152];
        for (&id, row) in ids.iter().zip(input.chunks_exact_mut(1152)) {
            embeddings.embedding_lookup(id, row);
            for v in row {
                *v = bf(*v * bf((1152f32).sqrt()));
            }
        }
        super::super::tokens("breeze.prompt_ids", &ids).unwrap();
        trace("breeze.text.embedding", None, None, &[4, 1152], &input).unwrap();
        model.forward(input, &[4], None, None).unwrap();
    }

    #[test]
    fn mean_square_matches_pinned_cpu_reduction_bits() {
        for (n, bits) in [
            (128, 0x498d2b35),
            (256, 0x4a219352),
            (1024, 0x4b1d5330),
            (1152, 0x4b344524),
            (2048, 0x4b9c8a7e),
        ] {
            let x = (0..n)
                .map(|i| {
                    let u = (i as u32).wrapping_mul(1664525).wrapping_add(1013904223);
                    let b = (u & 0x807f) | ((((u >> 16) % 20) + 116) << 7);
                    f32::from_bits(b << 16)
                })
                .collect::<Vec<_>>();
            assert_eq!(sum_squares(&x).to_bits(), bits, "width={n}");
        }
    }

    #[test]
    fn softmax_keeps_the_real_thirteen_key_rounding_boundary() {
        let mut scores = [
            0.84375,
            -0.010986328125,
            -0.72265625,
            -0.3203125,
            -1.9140625,
            -2.625,
            -2.75,
            -2.9375,
            -3.109375,
            -3.375,
            -3.625,
            -3.734375,
            -3.625,
        ];
        softmax(&mut scores);
        assert_eq!(
            scores.map(f32::to_bits),
            [
                0x3eed36c4, 0x3e49d1c4, 0x3dc61d95, 0x3e141fa5, 0x3cf0bebd, 0x3c6c8001, 0x3c50b5e5,
                0x3c2d06f1, 0x3c11b415, 0x3bdf6ded, 0x3bae01ca, 0x3b9bfa96, 0x3bae01ca
            ]
        );
    }

    #[test]
    fn softmax_keeps_the_nine_key_instruction_rounding_boundary() {
        let mut scores = [
            3.953125,
            0.1962890625,
            1.5703125,
            2.515625,
            0.90234375,
            -0.16796875,
            2.6875,
            f32::from_bits(0xff7f0000),
            f32::from_bits(0xff7f0000),
        ];
        softmax(&mut scores);
        assert_eq!(
            scores.map(f32::to_bits),
            [
                0x3f16b243, 0x3c6145f2, 0x3d5e8699, 0x3e0f2c8f, 0x3ce43324, 0x3c1c8001, 0x3e2a05fe,
                0, 0
            ]
        );
    }

    #[test]
    fn bf16_norm_and_rope_round_at_operator_boundaries() {
        assert_eq!(
            rms(&[3., 4.], &[1., 1.], 0., false),
            vec![bf(3.0 / (12.5f32).sqrt()), bf(4.0 / (12.5f32).sqrt())]
        );
        let mut x = [1., 2., 3., 4.];
        let original = x;
        rope(&mut x, 4, 0, &[1., 0.1]);
        assert_eq!(x, original);
        let mut scores = [-1000., 0., 0.];
        softmax(&mut scores);
        assert_eq!(scores, [0., 0.5, 0.5]);
    }
}

fn head_major(x: &[f32], batch: usize, n: usize, heads: usize, hd: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for b in 0..batch {
        for h in 0..heads {
            for t in 0..n {
                out.extend_from_slice(
                    &x[((b * n + t) * heads + h) * hd..((b * n + t) * heads + h + 1) * hd],
                );
            }
        }
    }
    out
}
