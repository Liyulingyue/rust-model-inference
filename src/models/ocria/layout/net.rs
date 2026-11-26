//! The LayoutNet network: a 6-layer transformer over text units with a CNN
//! image branch, producing node, pairwise and layout-class logits.
//!
//! The graph is `tablenet.onnx`; the release's `tablenet.bin` carries the same
//! 110 tensors. In order: an input MLP over the 66 non-id feature columns, a
//! **pre-norm** encoder (ReLU feed-forward, no final norm), a 2-way node head,
//! five bilinear pair heads plus a linear term over pairwise geometry, a
//! 13-class block head, and an image branch (a four-conv patch CNN, per-unit
//! 2x6 grid samples added to the unit embeddings, and 32x32 patch tokens that
//! the top two encoder layers cross-attend to).
//!
//! Every shape in the graph is checked once in [`TableNet::new`], so [`forward`]
//! cannot see a mismatched tensor. It still returns neutral logits rather than
//! panicking if the caller hands it a wrong-length input slice or an operator
//! rejects a shape: a page that decodes to nothing is a safe answer, a panic on
//! the inference path is not.
//!
//! [`forward`]: TableNet::forward

use super::features::{ImgInput, FEAT_DIM};
use super::{Tensor, Weights};
use crate::core::thread_pool::ComputePool;
use crate::ops::activation::{relu_inplace, vec_add_into};
use crate::ops::dot_f32;
use crate::ops::nn::conv2d;
use crate::ops::norm::layer_norm;
use crate::ops::softmax_inplace;

/// Side of the square canvas the image branch reads (`train_tablenet.IMG_SIDE`).
const IMG_SIDE: usize = 1024;
/// Units are sampled on an `IMG_GH x IMG_GW` grid inside their box.
const IMG_GH: usize = 2;
const IMG_GW: usize = 6;
/// Feature columns the encoder reads: all but the block id and the leaf id.
const N_IDS: usize = 2;
/// Encoder width, attention heads, layers.
const D: usize = 256;
const HEADS: usize = 4;
const LAYERS: usize = 6;
/// The pair heads' projection width, and how many heads the layout model has:
/// `same_table`, `row`, `col`, `cell` and `same_block`.
const PAIR_D: usize = 48;
const PAIR_HEADS: usize = 5;
const N_CLS: usize = 13;
/// Image-branch channels and cross-attention layers.
const IMG_C: usize = 64;
const XATTN: usize = 2;
/// Every `LayerNormalization` in the graph carries this epsilon.
const LN_EPS: f32 = 1e-5;
/// Pair-geometry inputs: 13 stacked terms plus the two rule features.
const GEO_DIM: usize = 13 + 2;
/// Per-unit image sample vector: one value per channel and grid cell.
const UNIT_SAMPLES: usize = IMG_C * IMG_GH * IMG_GW;

/// The image branch's four convolutions: contract name, stride, padding, input
/// channels, output channels, kernel size. BatchNorm is already folded into the
/// kernels and biases the converter reads, so there is no fold here.
const CNN: [(&str, usize, usize, usize, usize, usize); 4] = [
    ("img.cnn.0", 8, 0, 1, 48, 8),
    ("img.cnn.3.0", 1, 1, 48, 48, 3),
    ("img.cnn.4.0", 2, 1, 48, 64, 3),
    ("img.cnn.5.0", 1, 1, 64, 64, 3),
];

/// `[n, 2]` node logits, `[n_pair, n, n]` pair logits, `[n, 13]` class logits.
pub(crate) struct Logits {
    pub n: usize,
    pub node: Vec<f32>,
    pub pair: Vec<f32>,
    pub cls: Vec<f32>,
    pub n_cls: usize,
    pub n_pair: usize,
}

impl Logits {
    #[inline]
    pub(crate) fn node(&self, i: usize, k: usize) -> f32 {
        self.node[i * 2 + k]
    }

    #[inline]
    pub(crate) fn pair(&self, k: usize, i: usize, j: usize) -> f32 {
        self.pair[(k * self.n + i) * self.n + j]
    }
}

/// A row-major `[n_out, n_in]` weight matrix with its bias: `y = x W^T + b`.
///
/// The converter stores every matrix in the graph's `[out, in]` orientation, so
/// this is a direct read of the contract tensor — with one exception noted at
/// [`TableNet::new`], the cross-attention `in_proj_weight`.
struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    n_out: usize,
    n_in: usize,
}

impl Linear {
    fn load(weights: &Weights, prefix: &str, n_out: usize, n_in: usize) -> Result<Self, String> {
        let w = Tensor::get(weights, &format!("{prefix}.weight"))?;
        let b = Tensor::get(weights, &format!("{prefix}.bias"))?;
        if w.shape != [n_out, n_in] {
            return Err(format!(
                "Invalid LayoutNet tensor {prefix}.weight: shape {:?}, expected [{n_out}, {n_in}]",
                w.shape
            ));
        }
        if b.shape != [n_out] {
            return Err(format!(
                "Invalid LayoutNet tensor {prefix}.bias: shape {:?}, expected [{n_out}]",
                b.shape
            ));
        }
        Ok(Self {
            w: w.data.clone(),
            b: b.data.clone(),
            n_out,
            n_in,
        })
    }

    /// `y[n, n_out] = x[n, n_in] W^T + b`, one `dot_f32` per output element.
    fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut y = vec![0f32; n * self.n_out];
        for (r, row) in y.chunks_exact_mut(self.n_out).enumerate() {
            let xr = &x[r * self.n_in..(r + 1) * self.n_in];
            for (o, out) in row.iter_mut().enumerate() {
                *out =
                    self.b[o] + dot_f32(xr, &self.w[o * self.n_in..(o + 1) * self.n_in], self.n_in);
            }
        }
        y
    }
}

/// A LayerNorm over the last axis, applied to every row of an `[n, D]` block.
struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn load(weights: &Weights, prefix: &str) -> Result<Self, String> {
        let g = Tensor::get(weights, &format!("{prefix}.weight"))?;
        let b = Tensor::get(weights, &format!("{prefix}.bias"))?;
        if g.shape != [D] || b.shape != [D] {
            return Err(format!(
                "Invalid LayoutNet tensor {prefix}: shapes {:?} / {:?}, expected [{D}]",
                g.shape, b.shape
            ));
        }
        Ok(Self {
            g: g.data.clone(),
            b: b.data.clone(),
        })
    }

    fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut y = vec![0f32; n * D];
        for r in 0..n {
            layer_norm(
                &x[r * D..(r + 1) * D],
                &self.g,
                &self.b,
                LN_EPS,
                &mut y[r * D..(r + 1) * D],
            );
        }
        y
    }
}

/// One encoder layer: pre-norm self-attention then a pre-norm ReLU feed-forward.
struct Layer {
    in_proj: Linear,
    out_proj: Linear,
    lin1: Linear,
    lin2: Linear,
    norm1: Norm,
    norm2: Norm,
}

impl Layer {
    fn load(weights: &Weights, index: usize) -> Result<Self, String> {
        let p = format!("enc.layers.{index}");
        Ok(Self {
            in_proj: Self::packed(weights, &format!("{p}.self_attn.in_proj"))?,
            out_proj: Linear::load(weights, &format!("{p}.self_attn.out_proj"), D, D)?,
            lin1: Linear::load(weights, &format!("{p}.linear1"), 2 * D, D)?,
            lin2: Linear::load(weights, &format!("{p}.linear2"), D, 2 * D)?,
            norm1: Norm::load(weights, &format!("{p}.norm1"))?,
            norm2: Norm::load(weights, &format!("{p}.norm2"))?,
        })
    }

    /// The q|k|v projection, whose contract name ends in `_weight`/`_bias`
    /// rather than `.weight`/`.bias` (the reference module spells it that way).
    fn packed(weights: &Weights, prefix: &str) -> Result<Linear, String> {
        let w = Tensor::get(weights, &format!("{prefix}_weight"))?;
        let b = Tensor::get(weights, &format!("{prefix}_bias"))?;
        if w.shape != [3 * D, D] {
            return Err(format!(
                "Invalid LayoutNet tensor {prefix}_weight: shape {:?}, expected [{}, {D}]",
                w.shape,
                3 * D
            ));
        }
        if b.shape != [3 * D] {
            return Err(format!(
                "Invalid LayoutNet tensor {prefix}_bias: shape {:?}, expected [{}]",
                b.shape,
                3 * D
            ));
        }
        Ok(Linear {
            w: w.data.clone(),
            b: b.data.clone(),
            n_out: 3 * D,
            n_in: D,
        })
    }
}

/// A 2-D convolution with its bias, followed by ReLU.
struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    shape: [usize; 4],
    stride: [usize; 2],
    padding: [usize; 2],
}

impl Conv {
    fn load(
        weights: &Weights,
        name: &str,
        shape: [usize; 4],
        stride: usize,
        padding: usize,
    ) -> Result<Self, String> {
        let w = Tensor::get(weights, &format!("{name}.weight"))?;
        let b = Tensor::get(weights, &format!("{name}.bias"))?;
        if w.shape != shape {
            return Err(format!(
                "Invalid LayoutNet tensor {name}.weight: shape {:?}, expected {shape:?}",
                w.shape
            ));
        }
        if b.shape != [shape[0]] {
            return Err(format!(
                "Invalid LayoutNet tensor {name}.bias: shape {:?}, expected [{}]",
                b.shape, shape[0]
            ));
        }
        Ok(Self {
            w: w.data.clone(),
            b: b.data.clone(),
            shape,
            stride: [stride, stride],
            padding: [padding, padding],
        })
    }

    /// `x [cin, h, w] -> [cout, ho, wo]`, then ReLU: every convolution in the
    /// branch is followed by one.
    fn forward(
        &self,
        pool: &ComputePool,
        x: &[f32],
        h: usize,
        w: usize,
    ) -> Result<(Vec<f32>, usize, usize), String> {
        let (mut y, [_, ho, wo]) = conv2d(
            pool,
            x,
            [self.shape[1], h, w],
            &self.w,
            self.shape,
            Some(&self.b),
            self.stride,
            self.padding,
            [1, 1],
            1,
        )?;
        relu_inplace(&mut y);
        Ok((y, ho, wo))
    }
}

/// The image branch: the patch CNN, the per-unit sampler and the patch tokens.
struct ImgBranch {
    convs: Vec<Conv>,
    /// `img.unit.0` (768 -> 256), `img.unit.2` (256 -> 256).
    unit0: Linear,
    unit2: Linear,
    /// `img.tok` (64 -> 256).
    tok: Linear,
    /// One `in_proj` (3D x D) and `out_proj` (D x D) per cross-attention layer.
    attn: Vec<(Linear, Linear)>,
    /// One pre-norm per cross-attention layer, applied to the units.
    norms: Vec<Norm>,
}

/// What the image branch hands the encoder: the per-unit features to add to the
/// embedding, and (with cross-attention) the patch tokens and unit positions.
struct ImageOut {
    unit: Vec<f32>,
    tokens: Vec<f32>,
    upos: Vec<f32>,
}

/// The loaded model.
pub(crate) struct TableNet {
    inp0: Linear,
    inp2: Linear,
    layers: Vec<Layer>,
    node: Linear,
    qk: Linear,
    pair_geo: Linear,
    cls: Linear,
    img: ImgBranch,
    /// Convolutions go through the shared operator, which wants a pool. One
    /// thread: the pool's worker dispatch is single-caller, so a shared
    /// multi-threaded pool would be a data race if pages ran in parallel; a
    /// single-threaded pool runs the chunks inline and is safe from any thread.
    pool: ComputePool,
}

impl TableNet {
    /// Loads the model, checking every tensor against the graph's shape.
    pub(crate) fn new(weights: &Weights) -> Result<Self, String> {
        let inp0 = Linear::load(weights, "inp.0", D, FEAT_DIM - N_IDS)?;
        let inp2 = Linear::load(weights, "inp.2", D, D)?;
        let node = Linear::load(weights, "node", 2, D)?;
        let qk = Linear::load(weights, "qk", 2 * PAIR_HEADS * PAIR_D, D)?;
        let pair_geo = Linear::load(weights, "pair_geo", PAIR_HEADS, GEO_DIM)?;
        let cls = Linear::load(weights, "cls", N_CLS, D)?;
        let mut layers = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            layers.push(Layer::load(weights, index)?);
        }
        let mut convs = Vec::with_capacity(CNN.len());
        for (name, stride, padding, cin, cout, k) in CNN {
            convs.push(Conv::load(
                weights,
                name,
                [cout, cin, k, k],
                stride,
                padding,
            )?);
        }
        let mut attn = Vec::with_capacity(XATTN);
        let mut norms = Vec::with_capacity(XATTN);
        for index in 0..XATTN {
            // The one tensor the converter keeps in the graph's own `[out, in]`
            // orientation instead of transposing: `img.attn.*.in_proj_weight`.
            let w = Tensor::get(weights, &format!("img.attn.{index}.in_proj_weight"))?;
            let b = Tensor::get(weights, &format!("img.attn.{index}.in_proj_bias"))?;
            if w.shape != [3 * D, D] {
                return Err(format!(
                    "Invalid LayoutNet tensor img.attn.{index}.in_proj_weight: shape {:?}, \
                     expected [{}, {D}]",
                    w.shape,
                    3 * D
                ));
            }
            if b.shape != [3 * D] {
                return Err(format!(
                    "Invalid LayoutNet tensor img.attn.{index}.in_proj_bias: shape {:?}, \
                     expected [{}]",
                    b.shape,
                    3 * D
                ));
            }
            let in_proj = Linear {
                w: w.data.clone(),
                b: b.data.clone(),
                n_out: 3 * D,
                n_in: D,
            };
            attn.push((
                in_proj,
                Linear::load(weights, &format!("img.attn.{index}.out_proj"), D, D)?,
            ));
            norms.push(Norm::load(weights, &format!("img.norm.{index}"))?);
        }
        let img = ImgBranch {
            convs,
            unit0: Linear::load(weights, "img.unit.0", D, UNIT_SAMPLES)?,
            unit2: Linear::load(weights, "img.unit.2", D, D)?,
            tok: Linear::load(weights, "img.tok", D, IMG_C)?,
            attn,
            norms,
        };
        Ok(Self {
            inp0,
            inp2,
            layers,
            node,
            qk,
            pair_geo,
            cls,
            img,
            pool: ComputePool::new(1),
        })
    }

    /// `feats` is `[n, FEAT_DIM]`, `pairs` is `[n, n, 2]` rule pair features,
    /// `img` the page raster (`None`: the branch adds nothing, as for a page
    /// that has no render).
    pub(crate) fn forward(
        &self,
        feats: &[f32],
        pairs: &[f32],
        img: Option<&ImgInput>,
        n: usize,
    ) -> Logits {
        self.run(feats, pairs, img, n)
            .unwrap_or_else(|| Self::neutral(n))
    }

    /// Logits that decode to nothing: no table nodes, no relations and no class
    /// head, which is what the reference returns for a page it cannot run.
    fn neutral(n: usize) -> Logits {
        Logits {
            n,
            node: vec![-1.0; n * 2],
            pair: vec![-1.0; PAIR_HEADS * n * n],
            cls: Vec::new(),
            n_cls: 0,
            n_pair: PAIR_HEADS,
        }
    }

    fn run(
        &self,
        feats: &[f32],
        pairs: &[f32],
        img: Option<&ImgInput>,
        n: usize,
    ) -> Option<Logits> {
        self.run_recording(feats, pairs, img, n, |_, _| {})
    }

    /// The forward pass, calling `record` at the stage boundaries a fixture
    /// comparison wants. `run` passes a no-op, so the production path pays
    /// nothing for it.
    fn run_recording(
        &self,
        feats: &[f32],
        pairs: &[f32],
        img: Option<&ImgInput>,
        n: usize,
        mut record: impl FnMut(&'static str, &[f32]),
    ) -> Option<Logits> {
        if feats.len() != n * FEAT_DIM || pairs.len() != n * n * 2 {
            return None;
        }
        if n == 0 {
            return Some(Self::neutral(0));
        }
        // The input MLP reads every column but the two ids.
        let mut x0 = vec![0f32; n * (FEAT_DIM - N_IDS)];
        for r in 0..n {
            let row = &feats[r * FEAT_DIM..(r + 1) * FEAT_DIM];
            x0[r * (FEAT_DIM - N_IDS)..(r + 1) * (FEAT_DIM - N_IDS)]
                .copy_from_slice(&row[..FEAT_DIM - N_IDS]);
        }
        let mut x = self.inp0.forward(&x0, n);
        relu_inplace(&mut x);
        let mut x = self.inp2.forward(&x, n);

        // A malformed raster or box list degrades to the text branch alone
        // rather than failing the page.
        let img_out = img.and_then(|im| {
            if im.canvas.len() != IMG_SIDE * IMG_SIDE || im.boxes.len() < n {
                return None;
            }
            self.image_features(im, n)
        });
        if let Some(io) = &img_out {
            record("unit_mlp", &io.unit);
            vec_add_into(&io.unit, &mut x);
        }
        record("embed", &x);

        let nl = self.layers.len();
        let xa = self.img.attn.len();
        for (li, layer) in self.layers.iter().enumerate() {
            let h = layer.norm1.forward(&x, n);
            let a = self.attention(layer, &h, n);
            vec_add_into(&a, &mut x);
            let h = layer.norm2.forward(&x, n);
            let mut f = layer.lin1.forward(&h, n);
            relu_inplace(&mut f);
            let f = layer.lin2.forward(&f, n);
            vec_add_into(&f, &mut x);
            // The encoder layer's own output, before the cross-attention below.
            if li == 0 {
                record("enc_layer0", &x);
            }
            if li + 1 == nl {
                record("enc_layer5", &x);
            }
            // The top `xa` layers cross-attend to the image tokens.
            if let Some(io) = &img_out {
                if li + xa >= nl {
                    self.cross_attention(li + xa - nl, &mut x, n, io);
                }
            }
        }
        record("hidden_final", &x);

        let node = self.node.forward(&x, n);
        let pair = self.pair_logits(&x, feats, pairs, n);
        let cls = self.cls.forward(&x, n);
        Some(Logits {
            n,
            node,
            pair,
            cls,
            n_cls: N_CLS,
            n_pair: PAIR_HEADS,
        })
    }

    /// The stage tensors a fixture comparison needs, in order: `unit_mlp` (when
    /// the page has a raster), `embed`, `enc_layer0`, `enc_layer5` (the last
    /// encoder layer, before any cross-attention) and `hidden_final` (after
    /// both cross-attentions). Test-only: it shares [`TableNet::run_recording`]
    /// with `forward`, so it cannot drift from the real path.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn debug_stages(
        &self,
        feats: &[f32],
        pairs: &[f32],
        img: Option<&ImgInput>,
        n: usize,
    ) -> Vec<(&'static str, Vec<f32>)> {
        let mut stages = Vec::new();
        let _ = self.run_recording(feats, pairs, img, n, |name, values| {
            stages.push((name, values.to_vec()));
        });
        stages
    }

    /// Pre-norm self-attention, returning the `out_proj` output `[n, D]`.
    fn attention(&self, layer: &Layer, h: &[f32], n: usize) -> Vec<f32> {
        let hd = D / HEADS;
        let qkv = layer.in_proj.forward(h, n); // [n, 3D]: q | k | v
                                               // The graph divides the scores by sqrt(hd) = 8, which is a power of two,
                                               // so the reciprocal multiply below is bit-identical to the division.
        let scale = 1.0 / (hd as f32).sqrt();
        let mut ctx = vec![0f32; n * D];
        let mut q = vec![0f32; n * hd];
        let mut k = vec![0f32; n * hd];
        // v transposed to [hd, n]: one contiguous row per output channel, which
        // is what the context matmul's dot products want.
        let mut v = vec![0f32; hd * n];
        let mut scores = vec![0f32; n * n];
        for head in 0..HEADS {
            let (lo, hi) = (head * hd, (head + 1) * hd);
            for i in 0..n {
                let row = &qkv[i * 3 * D..(i + 1) * 3 * D];
                q[i * hd..(i + 1) * hd].copy_from_slice(&row[lo..hi]);
                k[i * hd..(i + 1) * hd].copy_from_slice(&row[D + lo..D + hi]);
                for c in 0..hd {
                    v[c * n + i] = row[2 * D + lo + c];
                }
            }
            for i in 0..n {
                for j in 0..n {
                    scores[i * n + j] =
                        dot_f32(&q[i * hd..(i + 1) * hd], &k[j * hd..(j + 1) * hd], hd) * scale;
                }
            }
            for row in scores.chunks_exact_mut(n) {
                softmax_inplace(row);
            }
            for i in 0..n {
                let row = &scores[i * n..(i + 1) * n];
                for c in 0..hd {
                    ctx[i * D + lo + c] = dot_f32(row, &v[c * n..(c + 1) * n], n);
                }
            }
        }
        layer.out_proj.forward(&ctx, n)
    }

    /// The units cross-attend to the image tokens, added to `x` in place.
    fn cross_attention(&self, index: usize, x: &mut [f32], n: usize, io: &ImageOut) {
        let hd = D / HEADS;
        let nt = io.tokens.len() / D;
        // Query: the pre-normed units plus their box centre's pos2d.
        let mut q = self.img.norms[index].forward(x, n);
        vec_add_into(&io.upos, &mut q);
        let in_proj = &self.img.attn[index].0;
        let out_proj = &self.img.attn[index].1;
        // Both projections are computed whole, so the q|k|v split is the same
        // row block for the query and the tokens.
        let qkv_q = in_proj.forward(&q, n);
        let qkv_t = in_proj.forward(&io.tokens, nt);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut ctx = vec![0f32; n * D];
        let mut sc = vec![0f32; n * nt];
        let mut qh = vec![0f32; n * hd];
        let mut kh = vec![0f32; nt * hd];
        let mut vh = vec![0f32; hd * nt];
        for head in 0..HEADS {
            let (lo, hi) = (head * hd, (head + 1) * hd);
            for i in 0..n {
                qh[i * hd..(i + 1) * hd].copy_from_slice(&qkv_q[i * 3 * D + lo..i * 3 * D + hi]);
            }
            for j in 0..nt {
                let row = &qkv_t[j * 3 * D..(j + 1) * 3 * D];
                kh[j * hd..(j + 1) * hd].copy_from_slice(&row[D + lo..D + hi]);
                for c in 0..hd {
                    vh[c * nt + j] = row[2 * D + lo + c];
                }
            }
            for i in 0..n {
                for j in 0..nt {
                    sc[i * nt + j] =
                        dot_f32(&qh[i * hd..(i + 1) * hd], &kh[j * hd..(j + 1) * hd], hd) * scale;
                }
            }
            for row in sc.chunks_exact_mut(nt) {
                softmax_inplace(row);
            }
            for i in 0..n {
                let row = &sc[i * nt..(i + 1) * nt];
                for c in 0..hd {
                    ctx[i * D + lo + c] = dot_f32(row, &vh[c * nt..(c + 1) * nt], nt);
                }
            }
        }
        let o = out_proj.forward(&ctx, n);
        vec_add_into(&o, x);
    }

    /// The five pair heads `[5, n, n]`: a symmetrised bilinear term plus a
    /// linear term over the pairwise geometry.
    fn pair_logits(&self, x: &[f32], feats: &[f32], pairs: &[f32], n: usize) -> Vec<f32> {
        let row = 2 * PAIR_HEADS * PAIR_D;
        let qk = self.qk.forward(x, n); // [n, 2 * P * pd]: (q | k) x head x pd
        let scale = 1.0 / (PAIR_D as f32).sqrt();
        let mut pair = vec![0f32; PAIR_HEADS * n * n];
        let mut q = vec![0f32; n * PAIR_D];
        let mut k = vec![0f32; n * PAIR_D];
        let mut g = [0f32; GEO_DIM];
        for head in 0..PAIR_HEADS {
            let (lo, hi) = (head * PAIR_D, (head + 1) * PAIR_D);
            for i in 0..n {
                let qrow = &qk[i * row..(i + 1) * row];
                q[i * PAIR_D..(i + 1) * PAIR_D].copy_from_slice(&qrow[lo..hi]);
                k[i * PAIR_D..(i + 1) * PAIR_D]
                    .copy_from_slice(&qrow[PAIR_HEADS * PAIR_D + lo..PAIR_HEADS * PAIR_D + hi]);
            }
            let block = &mut pair[head * n * n..(head + 1) * n * n];
            for i in 0..n {
                for j in 0..n {
                    block[i * n + j] = dot_f32(
                        &q[i * PAIR_D..(i + 1) * PAIR_D],
                        &k[j * PAIR_D..(j + 1) * PAIR_D],
                        PAIR_D,
                    ) * scale;
                }
            }
            // The heads are symmetric: the graph averages the score matrix with
            // its transpose before adding the geometry term.
            for i in 0..n {
                for j in i + 1..n {
                    let mean = (block[i * n + j] + block[j * n + i]) / 2.0;
                    block[i * n + j] = mean;
                    block[j * n + i] = mean;
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                Self::pair_geometry(feats, pairs, n, i, j, &mut g);
                for head in 0..PAIR_HEADS {
                    let w = &self.pair_geo.w[head * GEO_DIM..(head + 1) * GEO_DIM];
                    pair[(head * n + i) * n + j] += dot_f32(&g, w, GEO_DIM) + self.pair_geo.b[head];
                }
            }
        }
        pair
    }

    /// The pairwise geometry features of units `i` and `j` into `g`.
    ///
    /// Column order is the graph's `Concat_21`: eight terms (row gap, column
    /// gap, overlap, vertical overlap, centre deltas, same block, same leaf),
    /// five more from `stack_feats`, then the two rule features.
    fn pair_geometry(
        f: &[f32],
        pairf: &[f32],
        n: usize,
        i: usize,
        j: usize,
        g: &mut [f32; GEO_DIM],
    ) {
        let fi = &f[i * FEAT_DIM..(i + 1) * FEAT_DIM];
        let fj = &f[j * FEAT_DIM..(j + 1) * FEAT_DIM];
        let (cxi, cyi, x0i, x1i, y0i, y1i) = (fi[0], fi[1], fi[4], fi[5], fi[6], fi[7]);
        let (cxj, cyj, x0j, x1j, y0j, y1j) = (fj[0], fj[1], fj[4], fj[5], fj[6], fj[7]);
        let hhi = (y1i - y0i).max(1e-4);
        let hhj = (y1j - y0j).max(1e-4);
        let dy = (cyi - cyj).abs() / hhi;
        let dx = (cxi - cxj).abs() * 10.0;
        let xov = (x1i.min(x1j) - x0i.max(x0j)) * 20.0;
        let yov = (y1i.min(y1j) - y0i.max(y0j)) / hhi;
        let (bi, bj, li, lj) = (
            fi[FEAT_DIM - 2],
            fj[FEAT_DIM - 2],
            fi[FEAT_DIM - 1],
            fj[FEAT_DIM - 1],
        );
        let same_block = if bi == bj && bi >= 0.0 { 1.0 } else { 0.0 };
        let same_leaf = if li == lj && li >= 0.0 { 1.0 } else { 0.0 };
        g[0] = dy.min(20.0) / 20.0;
        g[1] = dx.min(20.0) / 20.0;
        g[2] = xov.clamp(-5.0, 5.0) / 5.0;
        g[3] = yov.clamp(-5.0, 2.0) / 5.0;
        g[4] = (cxi - cxj) * 5.0;
        g[5] = (cyi - cyj) * 5.0;
        g[6] = same_block;
        g[7] = same_leaf;
        let hm = hhi.min(hhj);
        let vgap = (y0j - y1i).max(y0i - y1j) / hm;
        let wi = (x1i - x0i).max(1e-4);
        let wj = (x1j - x0j).max(1e-4);
        let xov_frac = (x1i.min(x1j) - x0i.max(x0j)) / wi.min(wj);
        let la = (x0i - x0j).abs() / hm;
        let ra = (x1i - x1j).abs() / hm;
        let ca = ((x0i + x1i) - (x0j + x1j)).abs() / 2.0 / hm;
        g[8] = vgap.clamp(-3.0, 5.0) / 5.0;
        g[9] = xov_frac.clamp(-2.0, 1.0);
        g[10] = la.min(10.0) / 10.0;
        g[11] = ra.min(10.0) / 10.0;
        g[12] = ca.min(10.0) / 10.0;
        g[13] = pairf[(i * n + j) * 2];
        g[14] = pairf[(i * n + j) * 2 + 1];
    }

    /// The image branch on one page: the CNN over the canvas, the per-unit grid
    /// samples through the unit MLP, and the patch tokens the top encoder layers
    /// cross-attend to.
    fn image_features(&self, im: &ImgInput, n: usize) -> Option<ImageOut> {
        let mut map = im.canvas.clone();
        let (mut h, mut w) = (IMG_SIDE, IMG_SIDE);
        for conv in &self.img.convs {
            let (y, ho, wo) = conv.forward(&self.pool, &map, h, w).ok()?;
            map = y;
            h = ho;
            w = wo;
        }
        let plane = h * w;
        // `bilinear` maps a fraction of the *full* canvas stride-16 grid, which
        // is 64 cells a side, so the map's own size only bounds the reads.
        let full = IMG_SIDE / 16;

        // Per unit, one bilinear sample per channel and grid cell, laid out
        // channel-major exactly as the graph's Reshape/Transpose chain leaves
        // the MLP input: `ch * 12 + gh * 6 + gw`.
        let mut s = vec![0f32; n * UNIT_SAMPLES];
        for (i, b) in im.boxes.iter().take(n).enumerate() {
            for gh in 0..IMG_GH {
                let fy = b[1] + (b[3] - b[1]) * (gh as f32 + 0.5) / IMG_GH as f32;
                for gw in 0..IMG_GW {
                    let fx = b[0] + (b[2] - b[0]) * (gw as f32 + 0.5) / IMG_GW as f32;
                    for ch in 0..IMG_C {
                        s[i * UNIT_SAMPLES + ch * IMG_GH * IMG_GW + gh * IMG_GW + gw] =
                            bilinear(&map[ch * plane..(ch + 1) * plane], h, w, full, fx, fy);
                    }
                }
            }
        }
        let mut unit = self.img.unit0.forward(&s, n);
        relu_inplace(&mut unit);
        let unit = self.img.unit2.forward(&unit, n);

        // Patch tokens: the 2x2 average pool of the stride-16 map, one per
        // stride-32 cell, kept where the cell centre is inside the page. The
        // graph builds all 1024 and masks the rest with -1e9 before the softmax,
        // which is the same as not building them at all.
        let (gth, gtw) = (full / 2, full / 2);
        let mut cells = Vec::with_capacity(gth * gtw);
        for ty in 0..gth {
            for tx in 0..gtw {
                let px = (tx as f32 + 0.5) / gtw as f32;
                let py = (ty as f32 + 0.5) / gth as f32;
                if px <= im.valid.0 && py <= im.valid.1 && 2 * ty + 1 < h && 2 * tx + 1 < w {
                    cells.push((ty, tx, px, py));
                }
            }
        }
        if cells.is_empty() {
            return None;
        }
        let nt = cells.len();
        let mut pooled = vec![0f32; nt * IMG_C];
        for (k, &(ty, tx, _, _)) in cells.iter().enumerate() {
            for ch in 0..IMG_C {
                let plane_ch = &map[ch * plane..(ch + 1) * plane];
                pooled[k * IMG_C + ch] = average_pool_2x2(plane_ch, w, ty, tx);
            }
        }
        let mut tokens = self.img.tok.forward(&pooled, nt);
        let mut pe = [0f32; D];
        for (k, &(_, _, px, py)) in cells.iter().enumerate() {
            pos2d(px, py, &mut pe);
            vec_add_into(&pe, &mut tokens[k * D..(k + 1) * D]);
        }
        let mut upos = vec![0f32; n * D];
        for (i, b) in im.boxes.iter().take(n).enumerate() {
            pos2d(
                (b[0] + b[2]) / 2.0,
                (b[1] + b[3]) / 2.0,
                &mut upos[i * D..(i + 1) * D],
            );
        }
        Some(ImageOut { unit, tokens, upos })
    }
}

/// The 2-D sin-cos position encoding of a point in `[0, 1]^2`, `dim` values.
///
/// Local because the repository's only sin-cos helper is `rope::mrope`'s, which
/// is rope-specific; this is the graph's own frequency table (`k = dim/4 = 64`,
/// `step = -ln(1000)/(k-1)`, `fr = exp(i*step) * pi * 64`) and nothing else in
/// the tree computes it.
fn pos2d(x: f32, y: f32, out: &mut [f32]) {
    let k = out.len() / 4;
    let step = -(1000f32).ln() / (k.max(2) - 1) as f32;
    for i in 0..k {
        let fr = (i as f32 * step).exp() * std::f32::consts::PI * 64.0;
        out[i] = (x * fr).sin();
        out[k + i] = (x * fr).cos();
        out[2 * k + i] = (y * fr).sin();
        out[3 * k + i] = (y * fr).cos();
    }
}

/// A bilinear `grid_sample` read of one channel plane at a fractional position
/// of the canvas: `align_corners=0`, zeros outside.
///
/// Local because the repository's `grid_sample` (`qwen_drive::perception::ops`)
/// is a 4-D tensor operator that rounds every intermediate through bf16 and
/// takes a `[-1, 1]` grid; this reads one plane at a `[0, 1]` fraction, which is
/// the convention this graph's grid is built in, and must stay in f32.
#[inline]
fn bilinear(map: &[f32], h: usize, w: usize, full: usize, fx: f32, fy: f32) -> f32 {
    let x = fx * full as f32 - 0.5;
    let y = fy * full as f32 - 0.5;
    let (x0, y0) = (x.floor(), y.floor());
    let (ax, ay) = (x - x0, y - y0);
    let at = |yy: f32, xx: f32| -> f32 {
        if yy < 0.0 || xx < 0.0 || yy >= h as f32 || xx >= w as f32 {
            0.0
        } else {
            map[yy as usize * w + xx as usize]
        }
    };
    at(y0, x0) * (1.0 - ax) * (1.0 - ay)
        + at(y0, x0 + 1.0) * ax * (1.0 - ay)
        + at(y0 + 1.0, x0) * (1.0 - ax) * ay
        + at(y0 + 1.0, x0 + 1.0) * ax * ay
}

/// The mean of the 2x2 cell at `(ty, tx)` of one channel plane.
///
/// Local because no shared operator averages a 2x2 window: the two precedents
/// (`gemma4::vision`, `qwen3::omni`) are private to their models, and this is
/// four reads and three adds.
#[inline]
fn average_pool_2x2(map: &[f32], w: usize, ty: usize, tx: usize) -> f32 {
    let (y, x) = (2 * ty, 2 * tx);
    (map[y * w + x] + map[y * w + x + 1] + map[(y + 1) * w + x] + map[(y + 1) * w + x + 1]) / 4.0
}
