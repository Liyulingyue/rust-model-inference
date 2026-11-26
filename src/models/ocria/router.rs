//! The script router: group probabilities per line crop and the upside-down
//! vote.
//!
//! `router.onnx` reads a fixed 32x512 strip per line: a stem, seven
//! depthwise-separable blocks (the strided ones carry no residual), a squeeze
//! of the height axis, a mean and a maximum over the width, their concatenation
//! and two `Gemm`s that give the group and the flip logits. The page-level
//! policy on top -- the width-weighted share, the dominant group, the 0.9
//! override with a 4:1 aspect test and the flip vote -- is LiteParse's
//! `LineRouter.route`.

use super::img::Img;
use super::rec::prep;
use super::{Tensor, Weights};
use crate::core::thread_pool::ComputePool;
use crate::ops::activation::{relu_inplace, vec_add_into};
use crate::ops::dot_f32;
use crate::ops::nn::conv2d;
use crate::ops::softmax_inplace;
use std::sync::Arc;

/// The router's input height: the recognizers' height.
const H: usize = 32;
/// The router runs a fixed 512-pixel-wide strip: a shorter line is tiled across
/// it, a longer one cut to it.
const ROUTER_W: usize = 512;
/// Lines per router run.
const ROUTER_BATCH: usize = 64;
/// Script groups the `group` output indexes, in contract order.
const GROUPS: usize = 4;

/// The contract name of a router tensor.
///
/// The packed model names a graph's tensors after the ONNX node that consumes
/// them: `/m/body/body.0/dw/Conv` becomes `body.body.0.dw`, so the block's
/// depthwise weight is `router.body.body.0.dw.weight`.
fn key(path: &str) -> String {
    format!("router.{path}")
}

/// One convolution of the graph, with the weights the packed model stores.
struct Conv {
    weight: Vec<f32>,
    shape: [usize; 4],
    bias: Option<Vec<f32>>,
    stride: [usize; 2],
    padding: [usize; 2],
    groups: usize,
}

impl Conv {
    /// Loads a convolution. `shape` is the ONNX `[out, in, kh, kw]` kernel; the
    /// converter stores it flattened to `[out, in * kh * kw]` (or, for a
    /// quantized export, keeps the 4-D spelling), and row-major ONNX order
    /// makes the two the same values.
    fn load(
        weights: &Weights,
        path: &str,
        shape: [usize; 4],
        bias: bool,
        stride: [usize; 2],
        padding: [usize; 2],
        groups: usize,
    ) -> Result<Self, String> {
        let name = key(&format!("{path}.weight"));
        let tensor = Tensor::get(weights, &name)?;
        let [output_channels, channels, kernel_height, kernel_width] = shape;
        let kernel = channels * kernel_height * kernel_width;
        if tensor.shape != [output_channels, kernel] && tensor.shape != shape {
            return Err(format!(
                "Invalid OCRia tensor {name}: shape {:?}, expected [{output_channels}, {kernel}]",
                tensor.shape
            ));
        }
        if tensor.data.len() != output_channels * kernel {
            return Err(format!("Invalid OCRia tensor length: {name}"));
        }
        let bias = if bias {
            let name = key(&format!("{path}.bias"));
            let tensor = Tensor::get(weights, &name)?;
            if tensor.shape != [output_channels] {
                return Err(format!(
                    "Invalid OCRia tensor {name}: shape {:?}, expected [{output_channels}]",
                    tensor.shape
                ));
            }
            Some(tensor.data.clone())
        } else {
            None
        };
        Ok(Self {
            weight: tensor.data.clone(),
            shape,
            bias,
            stride,
            padding,
            groups,
        })
    }

    fn forward(
        &self,
        pool: &ComputePool,
        x: &[f32],
        shape: [usize; 3],
    ) -> Result<(Vec<f32>, [usize; 3]), String> {
        conv2d(
            pool,
            x,
            shape,
            &self.weight,
            self.shape,
            self.bias.as_deref(),
            self.stride,
            self.padding,
            [1, 1],
            self.groups,
        )
    }
}

/// A depthwise 3x3 + pointwise 1x1 block: the graph's `dw`/`pw` pair with a
/// `Relu` after each convolution, and, on the blocks that do not stride, the
/// residual `Add` around the pointwise one.
struct Block {
    dw: Conv,
    pw: Conv,
    residual: bool,
}

impl Block {
    fn forward(
        &self,
        pool: &ComputePool,
        x: &[f32],
        shape: [usize; 3],
    ) -> Result<(Vec<f32>, [usize; 3]), String> {
        let (mut values, next) = self.dw.forward(pool, x, shape)?;
        relu_inplace(&mut values);
        let (mut values, next) = self.pw.forward(pool, &values, next)?;
        if self.residual {
            if values.len() != x.len() {
                return Err(format!(
                    "OCRia router residual shape mismatch: {:?} vs {:?}",
                    next, shape
                ));
            }
            vec_add_into(x, &mut values);
        }
        relu_inplace(&mut values);
        Ok((values, next))
    }
}

/// The seven body blocks: input channels, output channels, depthwise stride.
/// The blocks that stride carry no residual; the others do.
const BODY: [(usize, usize, [usize; 2]); 7] = [
    (16, 32, [2, 2]),
    (32, 32, [1, 1]),
    (32, 64, [2, 2]),
    (64, 64, [1, 1]),
    (64, 96, [2, 2]),
    (96, 96, [1, 1]),
    (96, 96, [2, 1]),
];

pub(crate) struct Router {
    pool: Arc<ComputePool>,
    stem: Conv,
    body: Vec<Block>,
    /// The two `Gemm` classifiers, `transB=1`, so `[out, in]` row-major.
    group: (Vec<f32>, Vec<f32>),
    flip: (Vec<f32>, Vec<f32>),
}

impl Router {
    pub fn new(weights: &Weights, pool: &std::sync::Arc<ComputePool>) -> Result<Self, String> {
        let stem = Conv::load(
            weights,
            "stem.stem.0",
            [16, 1, 3, 3],
            true,
            [2, 2],
            [1, 1],
            1,
        )?;
        let mut body = Vec::with_capacity(BODY.len());
        for (index, (channels, out, stride)) in BODY.iter().enumerate() {
            let prefix = format!("body.body.{index}");
            body.push(Block {
                dw: Conv::load(
                    weights,
                    &format!("{prefix}.dw"),
                    [*channels, 1, 3, 3],
                    true,
                    *stride,
                    [1, 1],
                    *channels,
                )?,
                pw: Conv::load(
                    weights,
                    &format!("{prefix}.pw"),
                    [*out, *channels, 1, 1],
                    true,
                    [1, 1],
                    [0, 0],
                    1,
                )?,
                residual: *stride == [1, 1],
            });
        }
        // 96 channels of means then 96 of maxima feed the two classifiers
        let input = 2 * BODY[BODY.len() - 1].1;
        let classifier = |path: &str, out: usize| -> Result<(Vec<f32>, Vec<f32>), String> {
            let weight = Tensor::get(weights, &key(&format!("{path}.weight")))?;
            if weight.shape != [out, input] {
                return Err(format!(
                    "Invalid OCRia tensor {path}.weight: shape {:?}, expected [{out}, {input}]",
                    weight.shape
                ));
            }
            let bias = Tensor::get(weights, &key(&format!("{path}.bias")))?;
            if bias.shape != [out] {
                return Err(format!(
                    "Invalid OCRia tensor {path}.bias: shape {:?}, expected [{out}]",
                    bias.shape
                ));
            }
            Ok((weight.data.clone(), bias.data.clone()))
        };
        Ok(Self {
            pool: pool.clone(),
            stem,
            body,
            group: classifier("group", GROUPS)?,
            flip: classifier("flip", 2)?,
        })
    }

    /// The raw `group` `[count, 4]` and `flip` `[count, 2]` logits of the graph,
    /// before the softmaxes; this is the entry point the parity fixtures
    /// compare against ONNX Runtime.
    pub(crate) fn forward_logits(
        &self,
        x: &[f32],
        count: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let plane = H * ROUTER_W;
        if x.len() != count * plane {
            return Err(format!(
                "Invalid OCRia router input: {} values for {count} strips",
                x.len()
            ));
        }
        let pool = self.pool.as_ref();
        let mut groups = Vec::with_capacity(count * GROUPS);
        let mut flips = Vec::with_capacity(count * 2);
        for sample in 0..count {
            let (mut values, mut shape) = self.stem.forward(
                pool,
                &x[sample * plane..(sample + 1) * plane],
                [1, H, ROUTER_W],
            )?;
            relu_inplace(&mut values);
            for block in &self.body {
                let (next, next_shape) = block.forward(pool, &values, shape)?;
                values = next;
                shape = next_shape;
            }
            let [channels, height, width] = shape;
            let expected = BODY[BODY.len() - 1].1;
            if height != 1 || channels != expected {
                return Err(format!(
                    "The OCRia router body produced {channels}x{height}x{width}, \
                     expected {expected}x1x{width}"
                ));
            }
            // The graph squeezes the height axis, then reduces the width axis
            // twice: mean and max, concatenated as 96 then 96. The mean is
            // accumulated in F32 because ONNX Runtime's ReduceMean does, and 32
            // values give a wider accumulator nothing to buy.
            let mut features = Vec::with_capacity(2 * channels);
            for c in 0..channels {
                let row = &values[c * width..(c + 1) * width];
                features.push(row.iter().sum::<f32>() / width as f32);
            }
            for c in 0..channels {
                let row = &values[c * width..(c + 1) * width];
                features.push(row.iter().copied().fold(f32::NEG_INFINITY, f32::max));
            }
            groups.extend(classify(&self.group.0, &self.group.1, &features));
            flips.extend(classify(&self.flip.0, &self.flip.1, &features));
        }
        Ok((groups, flips))
    }

    /// Group probabilities per crop and P(upside down).
    fn probs(&self, crops: &[Img]) -> Result<(Vec<[f32; GROUPS]>, Vec<f32>), String> {
        let mut groups = Vec::with_capacity(crops.len());
        let mut flips = Vec::with_capacity(crops.len());
        for part in crops.chunks(ROUTER_BATCH) {
            let mut x = Vec::with_capacity(part.len() * H * ROUTER_W);
            for crop in part {
                x.extend(Self::strip(crop)?);
            }
            let (group, flip) = self.forward_logits(&x, part.len())?;
            for j in 0..part.len() {
                let mut probabilities = group[j * GROUPS..(j + 1) * GROUPS].to_vec();
                softmax_inplace(&mut probabilities);
                groups.push([
                    probabilities[0],
                    probabilities[1],
                    probabilities[2],
                    probabilities[3],
                ]);
                let mut probabilities = flip[j * 2..(j + 1) * 2].to_vec();
                softmax_inplace(&mut probabilities);
                flips.push(probabilities[1]);
            }
        }
        Ok((groups, flips))
    }

    /// One router strip: the line at height 32, tiled across the strip or cut
    /// to it, scaled to `[-1, 1]`.
    fn strip(crop: &Img) -> Result<Vec<f32>, String> {
        let gray = prep(crop, 8, usize::MAX);
        if gray.h != H || gray.w == 0 {
            return Err(format!(
                "The OCRia router needs a {}x32 line, got {}x{}",
                ROUTER_W, gray.w, gray.h
            ));
        }
        let mut x = vec![0f32; H * ROUTER_W];
        for y in 0..H {
            for column in 0..ROUTER_W {
                let value = gray.data[y * gray.w + column % gray.w];
                x[y * ROUTER_W + column] = value as f32 / 127.5 - 1.0;
            }
        }
        Ok(x)
    }

    /// Page-level routing: (group index per crop, turn the page?, each group's
    /// share of the line width). Every line takes the page's dominant group
    /// unless it is long enough to judge and confidently another; Latin unless
    /// another group holds a clear majority of the line width.
    pub fn route(&self, crops: &[Img]) -> Result<(Vec<usize>, bool, [f64; 4]), String> {
        const OVERRIDE: f32 = 0.9;
        const MIN_ASPECT: f64 = 4.0;
        const FLIP_MIN_LINES: usize = 3;
        const FLIP_VOTE: f64 = 0.7;
        const OTHER_SHARE: f64 = 0.65;
        if crops.is_empty() {
            return Ok((Vec::new(), false, [0.0; 4]));
        }
        let (probabilities, flips) = self.probs(crops)?;
        let widths: Vec<f64> = crops.iter().map(|c| c.w as f64).collect();
        let total: f64 = widths.iter().sum();
        if total <= 0.0 {
            return Err("The OCRia router needs crops with a width".into());
        }
        let mut share = [0f64; 4];
        for (p, &width) in probabilities.iter().zip(&widths) {
            for (k, slot) in share.iter_mut().enumerate() {
                *slot += p[k] as f64 * width;
            }
        }
        for slot in share.iter_mut() {
            *slot /= total;
        }
        let mut dom = argmax(&share);
        if dom != 0 && share[dom] <= OTHER_SHARE {
            dom = 0;
        }
        let groups = probabilities
            .iter()
            .zip(crops)
            .map(|(p, crop)| {
                let own = argmax(&p.map(|v| v as f64));
                let peak = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if own != dom && peak > OVERRIDE && crop.w as f64 >= MIN_ASPECT * crop.h as f64 {
                    own
                } else {
                    dom
                }
            })
            .collect();
        let vote: f64 = flips.iter().zip(&widths).map(|(&f, &w)| f as f64 * w).sum();
        let flip = crops.len() >= FLIP_MIN_LINES && vote > FLIP_VOTE * total;
        Ok((groups, flip, share))
    }
}

/// One `Gemm` classifier: `weight` is `[out, in]` row-major (`transB=1`), so
/// each row is a dot product with the features, plus the row's bias.
fn classify(weight: &[f32], bias: &[f32], features: &[f32]) -> Vec<f32> {
    weight
        .chunks_exact(features.len())
        .zip(bias)
        .map(|(row, &bias)| dot_f32(row, features, features.len()) + bias)
        .collect()
}

/// `np.argmax`: the first maximum. The share is accumulated in `f64`, and no
/// `f64` argmax lives in `crate::ops`.
fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}
