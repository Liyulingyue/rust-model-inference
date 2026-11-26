//! The four CTC recognizers and the per-line decoding.
//!
//! `rec_<group>.onnx` reads a 32-pixel-high line: a stem convolution halves
//! both axes, nine depthwise-separable blocks bring the height to 1 and the
//! width down by four or eight, three `mix` blocks (a 9-tap 1-D depthwise
//! convolution with a residual, then two 1x1 convolutions with a residual
//! around them) mix the timesteps, and a head convolution emits the per-timestep
//! class logits. The graph itself computes the `ArgMax` over the classes and
//! `Exp(ReduceMax - ReduceLogSumExp)`, the winning class's softmax probability.
//!
//! The CTC collapse is the runtime's, not the graph's: a timestep contributes
//! when its class is not the blank (0) and differs from the previous timestep,
//! the reading is the mean of the confidences it kept, and the RTL recognizer's
//! visual-order reading is reordered to logical order.
//!
//! # Precision
//!
//! `rec_latin.onnx` is ONNX-Runtime-quantized (QDQ): its weights *and its
//! activations* are snapped to int8/uint8 grids inside the graph, including the
//! head logits (step 0.3328, zero point 181). The converter dequantizes the
//! weights and this module computes everything in F32; the activation
//! quantization is deliberately not reproduced, which is the accepted precision
//! loss for this model.
//!
//! Two consequences are easy to get wrong. The quantized graph contains **no
//! `Relu` nodes at all**: the quantizer folded each one into the
//! `QuantizeLinear` that follows it (with `zero_point = 0` the quantizer clamps
//! negatives, which is the `Relu`), so its node list reads as if the network had
//! no activations. The F32 path must apply the same 25 `Relu`s as
//! `rec_rtl`/`rec_world` -- after the stem, after every depthwise and every
//! pointwise convolution and after every residual add, and after each `mix`
//! depthwise convolution and `fc1` -- or the activations run away. And `ids`
//! match ONNX Runtime exactly on the measured random-line fixtures, but the
//! quantized graph snaps its logits to a 0.3328-wide grid, so where two classes
//! fall inside one bin the F32 path can pick the other: the committed text-like
//! fixtures record a small set of such timesteps. `conf` differs by up to 0.21
//! on a single timestep and ~2e-2 on average per timestep; a line's confidence
//! is the mean over the characters the collapse *kept*, so that noise averages
//! down to roughly 3e-3 per line, which is why the pipeline's thresholds behave
//! as they do against the reference. The other three graphs are plain F32 and
//! match to ~1e-5.

use super::img::{resize_area, resize_linear, Img};
use super::{Tensor, Weights};
use crate::core::thread_pool::ComputePool;
use crate::ops::activation::{relu_inplace, vec_add_into};
use crate::ops::argmax::argmax_f32;
use crate::ops::nn::{conv1d, conv2d};
use crate::ops::softmax_inplace;
use std::sync::Arc;

/// The script groups, in the order the router's output indexes them.
pub(crate) const GROUPS: [&str; 4] = ["latin", "cjk", "rtl", "world"];

/// Recognizer input height.
const H: usize = 32;
/// The widest line the recognizers read, in input pixels.
const MAX_W: usize = 1280;
/// Batched lines are padded to a multiple of this width, so a batch's activations
/// line up and two runs of the same line agree.
const BUCKET_W: usize = 128;
/// Lines per recognizer run.
const BATCH: usize = 32;
/// The channel count the `mix` blocks and the head work at.
const MIXED: usize = 192;

/// The recognizer's preprocessing: the crop as grey at height [`H`], width
/// scaled with the aspect ratio and clamped to `[min_w, max_w]`, resampled with
/// the area filter when shrinking and the bilinear one when enlarging.
pub(super) fn prep(crop: &Img, min_w: usize, max_w: usize) -> Img {
    let gray = crop.to_gray();
    // The upper bound is the reference's `MAX_W`: a line longer than the
    // clamp is read at reduced resolution, not truncated, so the clamp is
    // deliberately not an error.
    let width = ((gray.w as f64 * H as f64 / gray.h.max(1) as f64).round_ties_even() as usize)
        .clamp(min_w, max_w);
    if gray.h > H {
        resize_area(&gray, width, H)
    } else {
        resize_linear(&gray, width, H)
    }
}

/// One 2-D convolution of the backbone.
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
        prefix: &str,
        path: &str,
        shape: [usize; 4],
        bias: bool,
        stride: [usize; 2],
        padding: [usize; 2],
        groups: usize,
    ) -> Result<Self, String> {
        let name = format!("{prefix}{path}.weight");
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
        Ok(Self {
            weight: tensor.data.clone(),
            shape,
            bias: load_bias(
                weights,
                &format!("{prefix}{path}.bias"),
                bias,
                output_channels,
            )?,
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

/// A depthwise 3x3 + pointwise 1x1 block of the backbone.
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
                    "OCRia recognizer residual shape mismatch: {:?} vs {:?}",
                    next, shape
                ));
            }
            vec_add_into(x, &mut values);
        }
        relu_inplace(&mut values);
        Ok((values, next))
    }
}

/// One 1-D convolution of the `mix` blocks and the head, over the timestep
/// axis.
struct Conv1d {
    weight: Vec<f32>,
    shape: [usize; 3],
    bias: Option<Vec<f32>>,
    groups: usize,
}

impl Conv1d {
    /// Loads a 1-D convolution whose output width the contract alone decides.
    ///
    /// `input` is the number of input channels, `kernel` the window width. The
    /// weight is stored `[out, (input / groups) * kernel]` or in its ONNX 3-D
    /// spelling; the output width is the stored row count.
    fn load(
        weights: &Weights,
        prefix: &str,
        path: &str,
        input: usize,
        kernel: usize,
        bias: bool,
        groups: usize,
    ) -> Result<Self, String> {
        let name = format!("{prefix}{path}.weight");
        let tensor = Tensor::get(weights, &name)?;
        let channels = input / groups;
        let fan_in = channels * kernel;
        let output = match tensor.shape.as_slice() {
            [output, fan] if *fan == fan_in => *output,
            [output, channels, window] if *channels == input / groups && *window == kernel => {
                *output
            }
            _ => {
                return Err(format!(
                    "Invalid OCRia tensor {name}: shape {:?}, expected [out, {fan_in}]",
                    tensor.shape
                ))
            }
        };
        if tensor.data.len() != output * fan_in {
            return Err(format!("Invalid OCRia tensor length: {name}"));
        }
        Ok(Self {
            weight: tensor.data.clone(),
            shape: [output, channels, kernel],
            bias: load_bias(weights, &format!("{prefix}{path}.bias"), bias, output)?,
            groups,
        })
    }

    fn forward(
        &self,
        pool: &ComputePool,
        x: &[f32],
        shape: [usize; 2],
        padding: usize,
    ) -> Result<(Vec<f32>, [usize; 2]), String> {
        conv1d(
            pool,
            x,
            shape,
            &self.weight,
            self.shape,
            self.bias.as_deref(),
            1,
            padding,
            1,
            self.groups,
        )
    }
}

/// A `mix` block: a 9-tap depthwise convolution with a residual, then `fc1` and
/// `fc2` 1x1 convolutions with a residual around them.
struct Mix {
    dw: Conv1d,
    fc1: Conv1d,
    fc2: Conv1d,
}

impl Mix {
    fn forward(&self, pool: &ComputePool, x: &[f32], width: usize) -> Result<Vec<f32>, String> {
        let (mut sum, _) = self.dw.forward(pool, x, [MIXED, width], 4)?;
        relu_inplace(&mut sum);
        // `Add`: the depthwise output plus the block's input
        vec_add_into(x, &mut sum);
        let (mut values, _) = self.fc1.forward(pool, &sum, [MIXED, width], 0)?;
        relu_inplace(&mut values);
        let (mut values, _) = self
            .fc2
            .forward(pool, &values, [self.fc1.shape[0], width], 0)?;
        // `Add_1`: the second convolution plus the first block's sum
        vec_add_into(&sum, &mut values);
        Ok(values)
    }
}

/// The nine backbone blocks: input channels, output channels, depthwise stride.
/// The blocks that stride carry no residual; the others do.
const BODY: [(usize, usize, [usize; 2]); 9] = [
    (32, 64, [2, 1]),
    (64, 64, [1, 1]),
    (64, 128, [2, 2]),
    (128, 128, [1, 1]),
    (128, 128, [1, 1]),
    (128, 192, [2, 1]),
    (192, 192, [1, 1]),
    (192, 192, [1, 1]),
    (192, 192, [2, 2]),
];

/// The last block's width stride, the one thing that differs between the four
/// recognizers: `rec_cjk` halves the width there as well, so it reads `W / 8`
/// timesteps, while the other three only halve the height and read `W / 4`. The
/// packed contract fixes these four group names (`GROUPS`), so the name is the
/// key; the timestep stride itself is still probed from the graph.
fn last_stride(group: &str) -> [usize; 2] {
    if group == "cjk" {
        [2, 2]
    } else {
        [2, 1]
    }
}

pub(crate) struct Recognizer {
    pool: Arc<ComputePool>,
    chars: Vec<String>,
    /// Exported in visual order: RTL readings are reordered to logical order.
    visual_rtl: bool,
    /// Input pixels per output timestep, from the probe in [`Recognizer::new`].
    stride: usize,
    /// Classes the head emits: the blank plus one per charset entry.
    classes: usize,
    stem: Conv,
    body: Vec<Block>,
    mix: Vec<Mix>,
    /// One head convolution, or `rec_cjk`'s two.
    head: Vec<Conv1d>,
}

impl Recognizer {
    pub fn new(
        weights: &Weights,
        pool: &std::sync::Arc<ComputePool>,
        group: &str,
        chars: Vec<String>,
        visual_rtl: bool,
    ) -> Result<Self, String> {
        let prefix = format!("rec.{group}.");
        let classes = chars.len() + 1;
        let stem = Conv::load(
            weights,
            &prefix,
            "stem.stem.0",
            [32, 1, 3, 3],
            true,
            [2, 2],
            [1, 1],
            1,
        )?;
        let mut body = Vec::with_capacity(BODY.len());
        for (index, (channels, out, stride)) in BODY.iter().enumerate() {
            let stride = if index + 1 == BODY.len() {
                last_stride(group)
            } else {
                *stride
            };
            let path = format!("body.body.{index}");
            body.push(Block {
                dw: Conv::load(
                    weights,
                    &prefix,
                    &format!("{path}.dw"),
                    [*channels, 1, 3, 3],
                    true,
                    stride,
                    [1, 1],
                    *channels,
                )?,
                pw: Conv::load(
                    weights,
                    &prefix,
                    &format!("{path}.pw"),
                    [*out, *channels, 1, 1],
                    true,
                    [1, 1],
                    [0, 0],
                    1,
                )?,
                residual: stride == [1, 1],
            });
        }
        let mut mix = Vec::with_capacity(3);
        for index in 0..3 {
            let path = format!("mix.mix.{index}");
            let dw = Conv1d::load(
                weights,
                &prefix,
                &format!("{path}.dw"),
                MIXED,
                9,
                true,
                MIXED,
            )?;
            let fc1 = Conv1d::load(weights, &prefix, &format!("{path}.fc1"), MIXED, 1, true, 1)?;
            let fc2 = Conv1d::load(
                weights,
                &prefix,
                &format!("{path}.fc2"),
                fc1.shape[0],
                1,
                true,
                1,
            )?;
            mix.push(Mix { dw, fc1, fc2 });
        }
        // `rec_cjk` reads its logits through a 96-wide bottleneck convolution
        // and then a second one; there is no `Relu` between them, and the other
        // three groups use a single convolution.
        let two_heads = weights.contains_key(&format!("{prefix}head.head.1.weight"));
        let mut head = Vec::with_capacity(if two_heads { 2 } else { 1 });
        if two_heads {
            head.push(Conv1d::load(
                weights,
                &prefix,
                "head.head.0",
                MIXED,
                1,
                true,
                1,
            )?);
            head.push(Conv1d::load(
                weights,
                &prefix,
                "head.head.1",
                head[0].shape[0],
                1,
                true,
                1,
            )?);
        } else {
            head.push(Conv1d::load(weights, &prefix, "head", MIXED, 1, true, 1)?);
        }
        let output = head[head.len() - 1].shape[0];
        if output != classes {
            return Err(format!(
                "The OCRia {group} recognizer head emits {output} classes but its charset has \
                 {} entries",
                classes - 1
            ));
        }
        let mut recognizer = Self {
            pool: pool.clone(),
            chars,
            visual_rtl,
            stride: 0,
            classes,
            stem,
            body,
            mix,
            head,
        };
        // the timestep stride from one probe run: input width / output steps
        let probe = vec![0f32; H * 64];
        let (logits, steps) = recognizer.forward_logits(&probe, 1, 64)?;
        if steps == 0 || 64 % steps != 0 {
            return Err(format!(
                "The OCRia {group} recognizer turned 64 input pixels into {steps} timesteps"
            ));
        }
        recognizer.stride = 64 / steps;
        if logits.len() != classes * steps {
            return Err(format!(
                "The OCRia {group} recognizer emitted {} logits for {steps} timesteps, \
                 expected {}",
                logits.len(),
                classes * steps
            ));
        }
        Ok(recognizer)
    }

    /// The raw class logits `[count, classes, steps]` of the graph, before the
    /// `ArgMax` and the softmax-max; this is the entry point the parity fixtures
    /// compare against ONNX Runtime.
    ///
    /// `steps` is whatever the graph produces: every strided convolution rounds
    /// its output size the way ONNX does, so a width that is not a whole number
    /// of timesteps simply ends on the last one. [`Recognizer::read`] always
    /// passes a padded multiple of [`BUCKET_W`], where the two agree exactly.
    pub(crate) fn forward_logits(
        &self,
        x: &[f32],
        count: usize,
        width: usize,
    ) -> Result<(Vec<f32>, usize), String> {
        if width == 0 {
            return Err("The OCRia recognizer input must be at least one pixel wide".into());
        }
        if x.len() != count * H * width {
            return Err(format!(
                "Invalid OCRia recognizer input: {} values for {count} lines {width} wide",
                x.len()
            ));
        }
        let pool = self.pool.as_ref();
        let mut logits: Vec<f32> = Vec::new();
        let mut steps = 0;
        for sample in 0..count {
            let input = &x[sample * H * width..(sample + 1) * H * width];
            let (mut values, mut shape) = self.stem.forward(pool, input, [1, H, width])?;
            relu_inplace(&mut values);
            for block in &self.body {
                let (next, next_shape) = block.forward(pool, &values, shape)?;
                values = next;
                shape = next_shape;
            }
            let [channels, height, timesteps] = shape;
            if height != 1 || channels != MIXED {
                return Err(format!(
                    "The OCRia recognizer backbone produced {channels}x{height}x{timesteps}, \
                     expected {MIXED}x1x{timesteps}"
                ));
            }
            // the graph squeezes the height axis: the values are already a
            // `[channels, timesteps]` plane
            let mut sequence = values;
            for block in &self.mix {
                sequence = block.forward(pool, &sequence, timesteps)?;
            }
            for head in &self.head {
                let (next, _) = head.forward(pool, &sequence, [head.shape[1], timesteps], 0)?;
                sequence = next;
            }
            if sequence.len() != self.classes * timesteps {
                return Err(format!(
                    "The OCRia recognizer head produced {} values, expected {}",
                    sequence.len(),
                    self.classes * timesteps
                ));
            }
            steps = timesteps;
            logits.extend_from_slice(&sequence);
        }
        Ok((logits, steps))
    }

    /// (text, confidence) per crop.
    pub fn read(&self, crops: &[&Img]) -> Result<Vec<(String, f32)>, String> {
        for crop in crops {
            if crop.w == 0 || crop.h == 0 {
                return Err(format!(
                    "The OCRia recognizer needs a non-empty crop, got {}x{}",
                    crop.w, crop.h
                ));
            }
        }
        let lines: Vec<Img> = crops
            .iter()
            .map(|c| prep(c, self.stride * 2, MAX_W))
            .collect();
        let mut order: Vec<usize> = (0..lines.len()).collect();
        order.sort_by_key(|&i| lines[i].w);
        let mut out = vec![(String::new(), 0f32); lines.len()];
        for batch in order.chunks(BATCH) {
            let maxw = batch.iter().map(|&i| lines[i].w).max().unwrap_or(1);
            let width = maxw.div_ceil(BUCKET_W) * BUCKET_W;
            self.read_run(&lines, batch, width, &mut out)?;
        }
        Ok(out)
    }

    /// Read the lines `batch` in one run, padded to width `width`, into `out`.
    fn read_run(
        &self,
        lines: &[Img],
        batch: &[usize],
        width: usize,
        out: &mut [(String, f32)],
    ) -> Result<(), String> {
        // the padded columns read as white, exactly as the reference fills them
        let mut x = vec![1f32; batch.len() * H * width];
        for (j, &i) in batch.iter().enumerate() {
            let line = &lines[i];
            for y in 0..H {
                for column in 0..line.w {
                    x[(j * H + y) * width + column] =
                        (line.data[y * line.w + column] as f64 / 127.5 - 1.0) as f32;
                }
            }
        }
        let (logits, steps) = self.forward_logits(&x, batch.len(), width)?;
        for (j, &i) in batch.iter().enumerate() {
            // a line is read out to its own width, never to the padded one
            let timesteps = lines[i].w / self.stride;
            if timesteps > steps {
                return Err(format!(
                    "The OCRia recognizer read {steps} timesteps of a {} px line, which needs \
                     {timesteps}",
                    lines[i].w
                ));
            }
            let logits = &logits[j * self.classes * steps..(j + 1) * self.classes * steps];
            let (mut text, mut kept, mut previous) = (String::new(), Vec::new(), 0u32);
            let mut column = vec![0f32; self.classes];
            let mut probabilities = vec![0f32; self.classes];
            for k in 0..timesteps {
                // the graph's ArgMax over the class axis, first index on ties
                for (c, value) in column.iter_mut().enumerate() {
                    *value = logits[c * steps + k];
                }
                let class = argmax_f32(&column) as u32;
                if class != 0 && class != previous {
                    if let Some(chars) = self.chars.get((class - 1) as usize) {
                        text.push_str(chars);
                    }
                    // the graph's Exp(ReduceMax - ReduceLogSumExp): the winning
                    // class's softmax probability
                    probabilities.copy_from_slice(&column);
                    softmax_inplace(&mut probabilities);
                    kept.push(probabilities[class as usize]);
                }
                previous = class;
            }
            if self.visual_rtl && text.chars().any(is_rtl_char) {
                text = visual_to_logical(&text);
            }
            let mean = if kept.is_empty() {
                0.0
            } else {
                (kept.iter().map(|&v| v as f64).sum::<f64>() / kept.len() as f64) as f32
            };
            out[i] = (text, mean);
        }
        Ok(())
    }
}

/// One bias vector of the packed model, or nothing when the graph has none.
fn load_bias(
    weights: &Weights,
    name: &str,
    wanted: bool,
    output_channels: usize,
) -> Result<Option<Vec<f32>>, String> {
    if !wanted {
        return Ok(None);
    }
    let tensor = Tensor::get(weights, name)?;
    if tensor.shape != [output_channels] {
        return Err(format!(
            "Invalid OCRia tensor {name}: shape {:?}, expected [{output_channels}]",
            tensor.shape
        ));
    }
    Ok(Some(tensor.data.clone()))
}

fn is_rtl_char(ch: char) -> bool {
    ('\u{0590}'..='\u{08ff}').contains(&ch) || ('\u{fb1d}'..='\u{feff}').contains(&ch)
}

/// `python-bidi`'s `get_display(s, base_dir="R")` (itself unicode-bidi):
/// reorder a visual-order reading into logical order with an RTL paragraph
/// level.
fn visual_to_logical(s: &str) -> String {
    use unicode_bidi::{BidiInfo, Level};
    let info = BidiInfo::new(s, Some(Level::rtl()));
    let mut out = String::with_capacity(s.len());
    for para in &info.paragraphs {
        out.push_str(&info.reorder_line(para, para.range.clone()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The recognizer emits an RTL line in *visual* order (left to right as it
    /// appears on the page) and the pipeline turns it back into logical order.
    /// A pure Hebrew run therefore comes back reversed, and a line that mixes a
    /// number keeps the number's own digits in order while the run around it
    /// reverses -- which is exactly what `python-bidi`'s `get_display(s,
    /// base_dir="R")` does, and what the reference runtime calls.
    #[test]
    fn visual_order_rtl_text_is_reordered_to_logical_order() {
        // "שלום" read off the page left to right is "םולש".
        assert_eq!(visual_to_logical("םולש"), "שלום");
        // Arabic: "مرحبا" appears as "ابحرم".
        assert_eq!(visual_to_logical("ابحرم"), "مرحبا");
        // A number inside the run keeps its digits left to right.
        assert_eq!(visual_to_logical("2024 םולש"), "שלום 2024");
        // Latin text is untouched: the paragraph level only decides the
        // direction of the runs that are already RTL.
        assert_eq!(visual_to_logical("Invoice 42"), "Invoice 42");
        assert_eq!(visual_to_logical(""), "");
    }

    /// `is_rtl_char` is the gate the reorder runs behind, so it has to cover
    /// the Hebrew/Arabic blocks and their presentation forms and nothing else.
    #[test]
    fn rtl_characters_are_the_hebrew_and_arabic_blocks() {
        for ch in ['ש', 'ם', 'م', 'ا', '\u{0590}', '\u{08ff}', '\u{fb1d}'] {
            assert!(is_rtl_char(ch), "{ch:?}");
        }
        for ch in ['A', 'z', '0', '中', 'Ω', '\u{058f}', '\u{0900}'] {
            assert!(!is_rtl_char(ch), "{ch:?}");
        }
    }
}
