//! The line detector: a segmentation map from `det.onnx`, then connected
//! components expanded to line quads.
//!
//! The graph mixes the RGB input down to luma, runs a stem and four
//! depthwise-separable stages (`s4`/`s8`/`s16`/`s32`: a 3x3 depthwise
//! convolution, a 1x1 pointwise one, a residual add from the second block of a
//! stage on), projects every stage to 32 channels, adds the resulting pyramid
//! bottom-up through nearest-neighbour 2x `Resize` nodes, smooths each level,
//! concatenates them at the `s4` resolution, fuses, and upsamples once more to
//! the stem's resolution where a two-conv head ends in `Sigmoid`. `prob` is
//! therefore a probability map at half the resized page's resolution.
//!
//! Everything below the graph -- the threshold, the connected components, the
//! shrink expansion, the collinear merge, the sort order and the faint
//! short-token pass -- follows LiteParse's detector, the runtime the model
//! ships for.

use super::img::{box_points, components4, min_area_rect, resize_area, Img};
use super::{Quad, Tensor, Weights};
use crate::core::thread_pool::ComputePool;
use crate::ops::activation::{relu_inplace, vec_add_into};
use crate::ops::nn::conv2d;
use std::sync::Arc;

/// The detector's training targets shrink each text kernel to this ratio of its
/// line, so the box around a component has to grow back by the same amount.
const SHRINK: f32 = 0.3;

/// A component whose mean probability is under this is a speck, not a line.
const MIN_SCORE: f32 = 0.6;

/// The faint short-token pass: regions of the probability map above `thresh`
/// that touch no detected line become candidates, strongest first, at most
/// `max` per page. A lone narrow glyph in a table cell ("1", "l") looks like a
/// rule fragment, which the detector learned to suppress, and peaks around
/// 0.13-0.52 against a 0.55 threshold; the caller reads the candidates and
/// keeps the confident readings.
///
/// The model ships with `(0.1, 24)`; the pipeline's `Options` carries that pair
/// and hands it to [`Detector::new`].
#[derive(Clone, Copy, Debug)]
struct Faint {
    thresh: f32,
    max: usize,
}

/// The contract name of a detector tensor.
///
/// The packed model names a graph's tensors after the ONNX node that consumes
/// them: `/m/s4/s4.0/dw/Conv` becomes `s4.s4.0.dw`, so the block's depthwise
/// weight is `det.s4.s4.0.dw.weight`.
fn key(path: &str) -> String {
    format!("det.{path}")
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
    /// Loads a convolution from the packed model.
    ///
    /// `shape` is the kernel's ONNX shape, `[out, in, kh, kw]`. The converter
    /// stores every kernel flattened to `[out, in * kh * kw]` so that a Q8_0
    /// row is a whole kernel; row-major ONNX order is preserved, so the same
    /// values are the 4-D kernel the convolution operator wants. The 4-D
    /// spelling is accepted too, because the packed contract is the authority.
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
            if tensor.shape != [shape[0]] {
                return Err(format!(
                    "Invalid OCRia tensor {name}: shape {:?}, expected [{}]",
                    tensor.shape, shape[0]
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

    /// The convolution of `x` (`[channels, height, width]`), through the
    /// repository's operator so the pool decides the threading.
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
/// `Relu` after each convolution, and, from the second block of a stage on, the
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
                    "OCRia detector residual shape mismatch: {:?} vs {:?}",
                    next, shape
                ));
            }
            vec_add_into(x, &mut values);
        }
        relu_inplace(&mut values);
        Ok((values, next))
    }
}

/// A stage of the backbone: its first block strides, the rest are residual.
struct Stage {
    blocks: Vec<Block>,
}

/// The four backbone stages: name, block count, input channels, output channels.
const STAGES: [(&str, usize, usize, usize); 4] = [
    ("s4", 2, 16, 24),
    ("s8", 3, 24, 48),
    ("s16", 3, 48, 96),
    ("s32", 2, 96, 128),
];

/// Channels every level of the pyramid is projected to.
const LATERAL: usize = 32;

/// One detection pass's smoothed level, kept at 16 channels.
const SMOOTH: usize = 16;

pub(crate) struct Detector {
    pool: Arc<ComputePool>,
    long_side: usize,
    thresh: f32,
    merge: f32,
    stem: Conv,
    stages: [Stage; 4],
    /// `lat.0` .. `lat.3`, the 1x1 lateral projections of `s4` .. `s32`.
    lat: [Conv; 4],
    /// `smooth.0` .. `smooth.3`, one per pyramid level.
    smooth: [Block; 4],
    fuse: Block,
    squeeze: Conv,
    lat2: Conv,
    head: Block,
    head_out: Conv,
    /// The faint short-token pass, or `None` when it is off.
    faint: Option<Faint>,
}

impl Detector {
    /// `faint` turns the faint short-token pass on, with its probability
    /// threshold and its candidate budget; the model ships with
    /// `Some((0.1, 24))`.
    pub fn new(
        weights: &Weights,
        pool: &std::sync::Arc<ComputePool>,
        long_side: usize,
        thresh: f32,
        merge: f32,
        faint: Option<(f32, usize)>,
    ) -> Result<Self, String> {
        if long_side == 0 {
            return Err("The OCRia detector long side must be non-zero".into());
        }
        if let Some((thresh, max)) = faint {
            if !(thresh > 0.0 && thresh <= 1.0) {
                return Err(format!(
                    "The OCRia faint threshold must be in (0, 1], got {thresh}"
                ));
            }
            if max == 0 {
                return Err("The OCRia faint budget must be non-zero".into());
            }
        }
        let stem = Conv::load(
            weights,
            "stem.stem.0",
            [16, 1, 3, 3],
            true,
            [2, 2],
            [1, 1],
            1,
        )?;
        let mut stages = Vec::with_capacity(4);
        for (name, blocks, channels, out) in STAGES {
            let mut stage = Vec::with_capacity(blocks);
            for index in 0..blocks {
                let (dw_in, stride) = if index == 0 {
                    (channels, [2, 2])
                } else {
                    (out, [1, 1])
                };
                let prefix = format!("{name}.{name}.{index}");
                stage.push(Block {
                    dw: Conv::load(
                        weights,
                        &format!("{prefix}.dw"),
                        [dw_in, 1, 3, 3],
                        true,
                        stride,
                        [1, 1],
                        dw_in,
                    )?,
                    pw: Conv::load(
                        weights,
                        &format!("{prefix}.pw"),
                        [out, dw_in, 1, 1],
                        true,
                        [1, 1],
                        [0, 0],
                        1,
                    )?,
                    residual: index > 0,
                });
            }
            stages.push(Stage { blocks: stage });
        }
        let stages: [Stage; 4] = stages
            .try_into()
            .map_err(|_| "The OCRia detector needs four stages".to_string())?;
        let mut lat = Vec::with_capacity(4);
        for (index, (_, _, _, channels)) in STAGES.iter().enumerate() {
            lat.push(Conv::load(
                weights,
                &format!("lat.{index}"),
                [LATERAL, *channels, 1, 1],
                false,
                [1, 1],
                [0, 0],
                1,
            )?);
        }
        let lat: [Conv; 4] = lat
            .try_into()
            .map_err(|_| "The OCRia detector needs four laterals".to_string())?;
        // the smoothing blocks all take 32 channels and emit 16
        let mut smooth = Vec::with_capacity(4);
        for index in 0..4 {
            let prefix = format!("smooth.{index}.smooth.{index}.0.smooth.{index}.0");
            smooth.push(Block {
                dw: Conv::load(
                    weights,
                    &format!("{prefix}.0"),
                    [LATERAL, 1, 3, 3],
                    true,
                    [1, 1],
                    [1, 1],
                    LATERAL,
                )?,
                pw: Conv::load(
                    weights,
                    &format!("{prefix}.3"),
                    [SMOOTH, LATERAL, 1, 1],
                    true,
                    [1, 1],
                    [0, 0],
                    1,
                )?,
                residual: false,
            });
        }
        let smooth: [Block; 4] = smooth
            .try_into()
            .map_err(|_| "The OCRia detector needs four smoothing blocks".to_string())?;
        let fuse = Block {
            dw: Conv::load(
                weights,
                "fuse.fuse.0.fuse.0.0",
                [4 * SMOOTH, 1, 3, 3],
                true,
                [1, 1],
                [1, 1],
                4 * SMOOTH,
            )?,
            pw: Conv::load(
                weights,
                "fuse.fuse.0.fuse.0.3",
                [2 * SMOOTH, 4 * SMOOTH, 1, 1],
                true,
                [1, 1],
                [0, 0],
                1,
            )?,
            residual: false,
        };
        let squeeze = Conv::load(
            weights,
            "squeeze",
            [12, 2 * SMOOTH, 1, 1],
            false,
            [1, 1],
            [0, 0],
            1,
        )?;
        let lat2 = Conv::load(weights, "lat2", [12, 16, 1, 1], false, [1, 1], [0, 0], 1)?;
        let head = Block {
            dw: Conv::load(
                weights,
                "head.head.0.head.0.0",
                [12, 1, 3, 3],
                true,
                [1, 1],
                [1, 1],
                12,
            )?,
            pw: Conv::load(
                weights,
                "head.head.0.head.0.3",
                [SMOOTH, 12, 1, 1],
                true,
                [1, 1],
                [0, 0],
                1,
            )?,
            residual: false,
        };
        let head_out = Conv::load(
            weights,
            "head.head.3",
            [1, SMOOTH, 1, 1],
            true,
            [1, 1],
            [0, 0],
            1,
        )?;
        Ok(Self {
            pool: pool.clone(),
            long_side,
            thresh,
            merge,
            stem,
            stages,
            lat,
            smooth,
            fuse,
            squeeze,
            lat2,
            head,
            head_out,
            faint: faint.map(|(thresh, max)| Faint { thresh, max }),
        })
    }

    /// Line quads (TL, TR, BR, BL along the line) in `img` pixels.
    pub fn detect(&self, img: &Img) -> Result<Vec<Quad>, String> {
        Ok(self.detect_with_faint(img)?.0)
    }

    /// [`Detector::detect`] plus the faint short-token candidates of the pass.
    pub fn detect_with_faint(&self, img: &Img) -> Result<(Vec<Quad>, Vec<Quad>), String> {
        let (boxes, _, faint) = self.detect_at(img, self.long_side)?;
        Ok((boxes, faint))
    }

    /// The raw `prob` map of the graph for one normalised input, with its own
    /// height and width. `x` is `[3, height, width]`, channel first, scaled to
    /// `[-1, 1]` the way the pipeline does it; this is the entry point the
    /// parity fixtures compare against ONNX Runtime.
    pub(crate) fn forward_prob(
        &self,
        x: &[f32],
        height: usize,
        width: usize,
    ) -> Result<(Vec<f32>, usize, usize), String> {
        if height % 32 != 0 || width % 32 != 0 {
            return Err(format!(
                "The OCRia detector input must be a multiple of 32 pixels, got {height}x{width}"
            ));
        }
        if x.len() != 3 * height * width {
            return Err(format!(
                "Invalid OCRia detector input: {} values for {height}x{width} RGB",
                x.len()
            ));
        }
        let pool = self.pool.as_ref();
        let luma = luma_mix(x, height, width);
        let (mut values, mut shape) = self.stem.forward(pool, &luma, [1, height, width])?;
        relu_inplace(&mut values);
        // the stem's own output feeds the last lateral of the head
        let stem_values = (values.clone(), shape);
        // the four backbone levels, coarsest last
        let mut levels: Vec<(Vec<f32>, [usize; 3])> = Vec::with_capacity(4);
        for stage in &self.stages {
            for block in &stage.blocks {
                let (next, next_shape) = block.forward(pool, &values, shape)?;
                values = next;
                shape = next_shape;
            }
            levels.push((values.clone(), shape));
        }
        // lateral projections, coarsest first
        let (mut pyramid, mut pyramid_shape) =
            self.lat[3].forward(pool, &levels[3].0, levels[3].1)?;
        let mut pyramids: Vec<(Vec<f32>, [usize; 3])> = vec![(Vec::new(), [0; 3]); 4];
        pyramids[3] = (pyramid.clone(), pyramid_shape);
        for index in (0..3).rev() {
            let (up, up_shape) = upsample2(&pyramid, pyramid_shape);
            let (lateral, lateral_shape) =
                self.lat[index].forward(pool, &levels[index].0, levels[index].1)?;
            if lateral_shape != up_shape {
                return Err(format!(
                    "OCRia detector pyramid mismatch at level {index}: {lateral_shape:?} vs {up_shape:?}"
                ));
            }
            let mut sum = up;
            vec_add_into(&lateral, &mut sum);
            pyramid = sum;
            pyramid_shape = lateral_shape;
            pyramids[index] = (pyramid.clone(), pyramid_shape);
        }
        // one smoothed branch per level, all brought to the s4 resolution
        let mut branches: Vec<(Vec<f32>, [usize; 3])> = Vec::with_capacity(4);
        for index in 0..4 {
            let (values, shape) =
                self.smooth[index].forward(pool, &pyramids[index].0, pyramids[index].1)?;
            let mut values = values;
            let mut shape = shape;
            for _ in 0..index {
                let (up, up_shape) = upsample2(&values, shape);
                values = up;
                shape = up_shape;
            }
            branches.push((values, shape));
        }
        let (first, shape) = branches[0].clone();
        let channels = 4 * SMOOTH;
        let mut concat = Vec::with_capacity(channels * shape[1] * shape[2]);
        concat.extend_from_slice(&first);
        for (values, branch_shape) in &branches[1..] {
            if *branch_shape != shape {
                return Err(format!(
                    "OCRia detector branch mismatch: {branch_shape:?} vs {shape:?}"
                ));
            }
            concat.extend_from_slice(values);
        }
        let (values, shape) = self
            .fuse
            .forward(pool, &concat, [channels, shape[1], shape[2]])?;
        let (values, shape) = self.squeeze.forward(pool, &values, shape)?;
        let (mut values, shape) = upsample2(&values, shape);
        let (lateral, lateral_shape) = self.lat2.forward(pool, &stem_values.0, stem_values.1)?;
        if lateral_shape != shape {
            return Err(format!(
                "OCRia detector head mismatch: {lateral_shape:?} vs {shape:?}"
            ));
        }
        vec_add_into(&lateral, &mut values);
        let (values, shape) = self.head.forward(pool, &values, shape)?;
        let (mut values, shape) = self.head_out.forward(pool, &values, shape)?;
        for value in values.iter_mut() {
            *value = sigmoid(*value);
        }
        Ok((values, shape[1], shape[2]))
    }

    /// One detection pass with the page's long side resized to `long_side`
    /// (never enlarged); also returns the resize scale and the faint
    /// candidates.
    fn detect_at(
        &self,
        img: &Img,
        long_side: usize,
    ) -> Result<(Vec<Quad>, f64, Vec<Quad>), String> {
        if img.w == 0 || img.h == 0 {
            return Err(format!(
                "The OCRia detector needs a non-empty page, got {}x{}",
                img.w, img.h
            ));
        }
        let (big_h, big_w) = (img.h as f64, img.w as f64);
        let s = (long_side as f64 / big_h.max(big_w)).min(1.0);
        let h = 32usize.max(((big_h * s / 32.0).round_ties_even() as usize) * 32);
        let w = 32usize.max(((big_w * s / 32.0).round_ties_even() as usize) * 32);
        let (sx, sy) = (w as f64 / big_w, h as f64 / big_h);
        let small = resize_area(img, w, h);
        let mut x = vec![0f32; 3 * h * w];
        for c in 0..3 {
            for i in 0..h * w {
                let value = if small.ch == 3 {
                    small.data[i * 3 + c]
                } else {
                    small.data[i]
                };
                x[c * h * w + i] = value as f32 / 127.5 - 1.0;
            }
        }
        let (prob, ph, pw) = self.forward_prob(&x, h, w)?;
        let stride = (h / ph) as f32;
        let mask: Vec<bool> = prob.iter().map(|&p| p > self.thresh).collect();
        let mut boxes = Vec::new();
        for comp in components4(&mask, pw, ph) {
            if comp.pixels.len() < 2 {
                continue;
            }
            let mean = comp
                .pixels
                .iter()
                .map(|&(x, y)| prob[y * pw + x] as f64)
                .sum::<f64>()
                / comp.pixels.len() as f64;
            if (mean as f32) < MIN_SCORE {
                continue;
            }
            // the hull of the component's pixel cells: each row's leftmost and
            // rightmost cell
            let rows = comp.y1 - comp.y0 + 1;
            let mut span = vec![(usize::MAX, 0usize); rows];
            for &(x, y) in &comp.pixels {
                let r = &mut span[y - comp.y0];
                r.0 = r.0.min(x);
                r.1 = r.1.max(x);
            }
            let mut pts = Vec::with_capacity(rows * 4);
            for (i, &(x0, x1)) in span.iter().enumerate() {
                if x0 == usize::MAX {
                    continue;
                }
                let y = (comp.y0 + i) as f32;
                for (px, py) in [
                    (x0 as f32, y),
                    (x0 as f32, y + 1.0),
                    (x1 as f32 + 1.0, y),
                    (x1 as f32 + 1.0, y + 1.0),
                ] {
                    pts.push((px * stride, py * stride));
                }
            }
            let mut r = min_area_rect(&pts);
            // line height is the side nearer vertical, not the shorter side: a
            // single digit's kernel is taller than it is wide
            let w_is_horizontal = ((r.angle + 90.0).rem_euclid(180.0) - 90.0).abs() < 45.0;
            let kernel_h = if w_is_horizontal { r.h } else { r.w };
            let line_h = kernel_h / (1.0 - 2.0 * SHRINK);
            let d = SHRINK * line_h;
            r.w += 2.0 * d;
            r.h += 2.0 * d;
            let mut quad = box_points(&r);
            for p in quad.iter_mut() {
                p[0] = ((p[0] as f64 / sx) as f32).clamp(0.0, img.w as f32);
                p[1] = ((p[1] as f64 / sy) as f32).clamp(0.0, img.h as f32);
            }
            boxes.push(order_quad(&quad));
        }
        if self.merge > 0.0 {
            boxes = merge_collinear(&boxes, self.merge);
        }
        // (row band of 8 px, then left edge); a stable sort, as Python's
        boxes.sort_by(|a, b| {
            let ka = ((min_y(a) as f64 / 8.0).round_ties_even(), min_x(a));
            let kb = ((min_y(b) as f64 / 8.0).round_ties_even(), min_x(b));
            ka.0.total_cmp(&kb.0).then(ka.1.total_cmp(&kb.1))
        });
        let to_img = |x: usize, y: usize| {
            (
                ((x as f32 * stride) as f64 / sx) as f32,
                ((y as f32 * stride) as f64 / sy) as f32,
            )
        };
        let faint = match self.faint {
            Some(faint) => faint_boxes(&prob, &mask, pw, ph, to_img, &boxes, img, faint),
            None => Vec::new(),
        };
        Ok((boxes, s, faint))
    }
}

/// The graph's luma mix: `Mul` by `[0.299, 0.587, 0.114]` then `ReduceSum`
/// over the channel axis, kept as one plane.
fn luma_mix(x: &[f32], height: usize, width: usize) -> Vec<f32> {
    let plane = height * width;
    let mut out = vec![0f32; plane];
    for (i, value) in out.iter_mut().enumerate() {
        *value = 0.299 * x[i] + 0.587 * x[plane + i] + 0.114 * x[2 * plane + i];
    }
    out
}

/// The graph's `Resize` nodes: `mode=nearest`,
/// `coordinate_transformation_mode=asymmetric`, `nearest_mode=floor` and
/// `scales=[1, 1, 2, 2]`, so each source cell becomes a 2x2 block.
fn upsample2(x: &[f32], [channels, height, width]: [usize; 3]) -> (Vec<f32>, [usize; 3]) {
    let (height2, width2) = (height * 2, width * 2);
    let mut out = vec![0f32; channels * height2 * width2];
    for c in 0..channels {
        for y in 0..height {
            let row = &x[(c * height + y) * width..][..width];
            for dy in 0..2 {
                let start = (c * height2 + y * 2 + dy) * width2;
                let target = &mut out[start..start + width2];
                for (x, value) in row.iter().enumerate() {
                    target[2 * x] = *value;
                    target[2 * x + 1] = *value;
                }
            }
        }
    }
    (out, [channels, height2, width2])
}

/// The graph's closing `Sigmoid`. There is no shared elementwise sigmoid in
/// `crate::ops`, and this one call site does not justify one.
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// Candidate boxes for the faint short tokens, in `img` pixels. `to_img` maps a
/// probability-map cell corner to image pixels; `boxes` are the detected lines.
fn faint_boxes(
    prob: &[f32],
    mask: &[bool],
    pw: usize,
    ph: usize,
    to_img: impl Fn(usize, usize) -> (f32, f32),
    boxes: &[Quad],
    img: &Img,
    faint: Faint,
) -> Vec<Quad> {
    // a candidate keeps the side margin a detected line has around its text
    const PAD: f32 = 0.35;
    let bbox = |q: &Quad| {
        let xs = q.iter().map(|p| p[0]);
        let ys = q.iter().map(|p| p[1]);
        (
            xs.clone().fold(f32::INFINITY, f32::min),
            ys.clone().fold(f32::INFINITY, f32::min),
            xs.fold(f32::NEG_INFINITY, f32::max),
            ys.fold(f32::NEG_INFINITY, f32::max),
        )
    };
    let lines: Vec<(f32, f32, f32, f32)> = boxes.iter().map(bbox).collect();
    // rows: lines that run level, so their box is the row's band
    let rows: Vec<(f32, f32, f32, f32)> = boxes
        .iter()
        .zip(&lines)
        .filter(|(q, l)| (q[1][1] - q[0][1]).abs() <= 0.1 * (l.2 - l.0))
        .map(|(_, &l)| l)
        .collect();
    if rows.is_empty() {
        return Vec::new();
    }
    let weak: Vec<bool> = prob.iter().map(|&p| p > faint.thresh).collect();
    let mut cands: Vec<(f32, Quad)> = Vec::new();
    for comp in components4(&weak, pw, ph) {
        if comp.pixels.iter().any(|&(x, y)| mask[y * pw + x]) {
            continue;
        }
        let (x0, y0) = to_img(comp.x0, comp.y0);
        let (x1, y1) = to_img(comp.x1 + 1, comp.y1 + 1);
        let cy = (y0 + y1) / 2.0;
        // the row: a level line whose middle half holds the region's centre
        let Some(&(_, ry0, _, ry1)) = rows
            .iter()
            .filter(|r| {
                let h = r.3 - r.1;
                cy >= r.1 + 0.25 * h && cy <= r.3 - 0.25 * h
            })
            .min_by(|a, b| {
                (a.1 + a.3 - 2.0 * cy)
                    .abs()
                    .total_cmp(&(b.1 + b.3 - 2.0 * cy).abs())
            })
        else {
            continue;
        };
        let h = ry1 - ry0;
        // a short token's core is lower than its row's line box and at most a
        // few glyphs wide
        if y1 - y0 > 0.8 * h || x1 - x0 > 1.5 * h {
            continue;
        }
        let (bx0, bx1) = ((x0 - PAD * h).max(0.0), (x1 + PAD * h).min(img.w as f32));
        let (by0, by1) = (ry0.max(0.0), ry1.min(img.h as f32));
        let area = (bx1 - bx0) * (by1 - by0);
        // not part of a detected line (a gap between words, an accent, a
        // descender)
        let covered = lines.iter().any(|l| {
            let ix = (bx1.min(l.2) - bx0.max(l.0)).max(0.0);
            let iy = (by1.min(l.3) - by0.max(l.1)).max(0.0);
            ix * iy > 0.2 * area
        });
        if covered {
            continue;
        }
        let peak = comp
            .pixels
            .iter()
            .map(|&(x, y)| prob[y * pw + x])
            .fold(0f32, f32::max);
        cands.push((peak, [[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]]));
    }
    cands.sort_by(|a, b| b.0.total_cmp(&a.0));
    cands.truncate(faint.max);
    cands.into_iter().map(|(_, q)| q).collect()
}

fn min_y(q: &Quad) -> f32 {
    q.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min)
}

fn min_x(q: &Quad) -> f32 {
    q.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min)
}

fn norm(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// Corners as [top-left, top-right, bottom-right, bottom-left] along the
/// line's long axis.
fn order_quad(q: &Quad) -> Quad {
    let c = [
        (q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4.0,
        (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4.0,
    ];
    let e1 = [q[1][0] - q[0][0], q[1][1] - q[0][1]];
    let e2 = [q[2][0] - q[1][0], q[2][1] - q[1][1]];
    let mut u = if norm(e1) >= norm(e2) { e1 } else { e2 };
    let n = norm(u) + 1e-6;
    u = [u[0] / n, u[1] / n];
    if u[0] < 0.0 {
        u = [-u[0], -u[1]];
    }
    let v = [-u[1], u[0]];
    let a: Vec<f32> = q
        .iter()
        .map(|p| (p[0] - c[0]) * u[0] + (p[1] - c[1]) * u[1])
        .collect();
    let b: Vec<f32> = q
        .iter()
        .map(|p| (p[0] - c[0]) * v[0] + (p[1] - c[1]) * v[1])
        .collect();
    // np.argmin / argmax: the first extreme index
    let pick = |f: &dyn Fn(usize) -> f32, max: bool| -> usize {
        let mut best = 0;
        for i in 1..4 {
            if (max && f(i) > f(best)) || (!max && f(i) < f(best)) {
                best = i;
            }
        }
        best
    };
    let tl = pick(&|i| a[i] + b[i], false);
    let tr = pick(&|i| a[i] - b[i], true);
    let br = pick(&|i| a[i] + b[i], true);
    let bl = pick(&|i| a[i] - b[i], false);
    [q[tl], q[tr], q[br], q[bl]]
}

/// Join near-horizontal line quads that continue one another: each box links to
/// its nearest qualifying right neighbour when that neighbour's nearest left
/// neighbour is the same box, and each chain becomes the rectangle spanning it
/// along its longest member's axis.
fn merge_collinear(boxes: &[Quad], gap: f32) -> Vec<Quad> {
    const MAX_TILT: f32 = 0.35;
    const MAX_PERP: f32 = 0.35;
    const MIN_H_RATIO: f32 = 0.6;
    const MAX_DANGLE: f32 = 0.07;
    let n = boxes.len();
    if n < 2 {
        return boxes.to_vec();
    }
    let c: Vec<[f32; 2]> = boxes
        .iter()
        .map(|q| {
            [
                (q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4.0,
                (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4.0,
            ]
        })
        .collect();
    let mut u: Vec<[f32; 2]> = Vec::with_capacity(n);
    let mut len: Vec<f32> = Vec::with_capacity(n);
    let mut hgt: Vec<f32> = Vec::with_capacity(n);
    for q in boxes {
        let d = [q[1][0] - q[0][0], q[1][1] - q[0][1]];
        let l = norm(d).max(1e-3);
        u.push([d[0] / l, d[1] / l]);
        len.push(l);
        hgt.push(norm([q[3][0] - q[0][0], q[3][1] - q[0][1]]).max(1e-3));
    }
    let v: Vec<[f32; 2]> = u.iter().map(|u| [-u[1], u[0]]).collect();
    let flat: Vec<bool> = (0..n)
        .map(|i| u[i][1].abs() <= MAX_TILT && len[i] >= 0.5 * hgt[i])
        .collect();
    let cos_min = 1.0 - MAX_DANGLE * MAX_DANGLE / 2.0;
    let mut right = vec![-1isize; n];
    let mut left_best = vec![(f32::INFINITY, -1isize); n];
    for a in 0..n {
        let mut best = (f32::INFINITY, -1isize);
        if flat[a] {
            for b in 0..n {
                if a == b || !flat[b] {
                    continue;
                }
                let d = [c[b][0] - c[a][0], c[b][1] - c[a][1]];
                let along = d[0] * u[a][0] + d[1] * u[a][1];
                let perp = (d[0] * v[a][0] + d[1] * v[a][1]).abs();
                let hm = (hgt[a] + hgt[b]) / 2.0;
                let g = along - (len[a] + len[b]) / 2.0;
                let ok = along > 0.0
                    && hgt[a].min(hgt[b]) / hgt[a].max(hgt[b]) >= MIN_H_RATIO
                    && (u[a][0] * u[b][0] + u[a][1] * u[b][1]).abs() >= cos_min
                    && perp <= MAX_PERP * hm
                    && g <= gap * hm
                    && g >= -0.5 * len[a].min(len[b]);
                if ok {
                    if g < best.0 {
                        best = (g, b as isize);
                    }
                    // column-wise argmin for b: the first a reaching the minimum
                    if g < left_best[b].0 {
                        left_best[b] = (g, a as isize);
                    }
                }
            }
        }
        right[a] = best.1;
    }
    let mut nxt = vec![-1isize; n];
    for a in 0..n {
        let b = right[a];
        if b >= 0 && left_best[b as usize].1 == a as isize {
            nxt[a] = b;
        }
    }
    let mut has_prev = vec![false; n];
    for &b in &nxt {
        if b >= 0 {
            has_prev[b as usize] = true;
        }
    }
    let mut out = Vec::new();
    let mut seen = vec![false; n];
    for a in 0..n {
        if has_prev[a] || seen[a] {
            continue;
        }
        let mut chain = vec![a];
        seen[a] = true;
        while let Some(&last) = chain.last() {
            let nx = nxt[last];
            if nx < 0 || seen[nx as usize] {
                break;
            }
            chain.push(nx as usize);
            seen[nx as usize] = true;
        }
        if chain.len() == 1 {
            out.push(boxes[a]);
            continue;
        }
        let pts: Vec<[f32; 2]> = chain.iter().flat_map(|&i| boxes[i]).collect();
        // the chain's direction is its longest member's (the first on ties)
        let mut k = chain[0];
        for &i in &chain[1..] {
            if len[i] > len[k] {
                k = i;
            }
        }
        let np = pts.len() as f32;
        let ca = [
            pts.iter().map(|p| p[0]).sum::<f32>() / np,
            pts.iter().map(|p| p[1]).sum::<f32>() / np,
        ];
        let (ua, va) = (u[k], v[k]);
        let (mut a0, mut a1, mut b0, mut b1) = (
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
        );
        for p in &pts {
            let (dx, dy) = (p[0] - ca[0], p[1] - ca[1]);
            let pa = dx * ua[0] + dy * ua[1];
            let pb = dx * va[0] + dy * va[1];
            a0 = a0.min(pa);
            a1 = a1.max(pa);
            b0 = b0.min(pb);
            b1 = b1.max(pb);
        }
        let at = |s: f32, t: f32| [ca[0] + ua[0] * s + va[0] * t, ca[1] + ua[1] * s + va[1] * t];
        out.push(order_quad(&[
            at(a0, b0),
            at(a1, b0),
            at(a1, b1),
            at(a0, b1),
        ]));
    }
    for i in 0..n {
        if !seen[i] {
            out.push(boxes[i]);
        }
    }
    out
}
