//! OCRia-nano-1: line detector, script router and four CTC recognizers.
//!
//! The upstream release (`llamaindex/OCRia-nano-1`) ships ONNX graphs for a
//! document-OCR stack: `det.onnx` finds text lines, `router.onnx` picks a
//! recognizer per line crop, and `rec_{latin,cjk,rtl,world}.onnx` read the
//! crops. The converter in `tools/converter/ocria/` packs the weights of all
//! six graphs into one GGUF; this module runs them with the repository's own
//! convolution and matmul operators.
//!
//! The pipeline around the graphs -- resize, connected components, the
//! collinear merge, the column-gutter split, the sideways-page retry, the
//! script rescue and the junk filter -- follows LiteParse, the runtime the
//! models ship for. `tools/oracle/ocria/` records the pinned revision and the
//! parity fixtures.
mod det;
mod img;
mod page;
mod rec;
mod router;
#[cfg(test)]
mod tests;

use crate::compute::{ComputePolicy, UsedBackend};
use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// The `general.architecture` value the OCRia converter writes.
pub const ARCHITECTURE: &str = "ocria";
const CONTRACT: &str = include_str!("contract.json");

/// A line quad: [top-left, top-right, bottom-right, bottom-left] along the text.
pub(crate) type Quad = [[f32; 2]; 4];

/// The weight storage an OCRia GGUF carries.
///
/// The recognizers and the detector always compute in F32. The quantized
/// export changes how the two-dimensional weight matrices are stored, not the
/// arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    F32,
    Q8_0,
}

impl WeightFormat {
    /// The `ocria.format` metadata value that names this storage.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "ocria-nano-1-f32-v1",
            Self::Q8_0 => "ocria-nano-1-q8_0-v1",
        }
    }

    fn from_metadata(value: Option<&str>) -> Result<Self, String> {
        match value {
            Some("ocria-nano-1-f32-v1") => Ok(Self::F32),
            Some("ocria-nano-1-q8_0-v1") => Ok(Self::Q8_0),
            other => Err(format!(
                "Unsupported OCRia format {other:?}; expected ocria-nano-1-f32-v1 or \
                 ocria-nano-1-q8_0-v1"
            )),
        }
    }
}

/// One tensor of the packed model.
pub(crate) struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Every tensor of a packed model, keyed by its contract name.
pub(crate) type Weights = HashMap<String, Tensor>;

impl Tensor {
    /// The tensor's values, or a contract error naming it.
    pub(crate) fn get<'a>(weights: &'a Weights, name: &str) -> Result<&'a Tensor, String> {
        weights
            .get(name)
            .ok_or_else(|| format!("Missing OCRia tensor: {name}"))
    }
}

/// Whether a path holds an OCRia model.
///
/// The app layer asks this before it decides which OCR pipeline to run, so it
/// lives next to the contract it checks against.
pub fn matches_model(path: &Path) -> bool {
    path.is_file()
        && crate::format::ggufrs::open_model_source(path, crate::format::ggufrs::ComponentRole::Llm)
            .ok()
            .is_some_and(|source| {
                source
                    .metadata("general.architecture")
                    .and_then(|value| value.to_string_val())
                    .as_deref()
                    == Some(ARCHITECTURE)
            })
}

/// The pipeline switches and thresholds a model runs with.
///
/// The defaults are the benchmarked configuration LiteParse runs: the
/// detector's long side, its `det.meta.json` threshold and merge distance, the
/// column-gutter split, the sideways-page retry, the script rescue and the
/// junk filter all on.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Long side the page is resized to before detection, in pixels.
    pub det_long_side: usize,
    /// Detector probability threshold.
    pub det_thresh: f32,
    /// Collinear box merge distance, in line heights (0 = off).
    pub det_merge: f32,
    /// Cut boxes that run across a column gutter.
    pub gutter: bool,
    /// Retry pages that look sideways a quarter turn round.
    pub orient: bool,
    /// Re-read low-confidence "Latin" pages with the other recognizers.
    pub rescue: bool,
    /// Drop readings with no letter or digit, or 1-2 unsure characters.
    pub junk: bool,
    /// Read faint short tokens next to detected lines, kept at this
    /// confidence (0 = off).
    pub faint_thresh: f32,
    /// Most faint candidates per page.
    pub faint_max: usize,
}

/// A faint candidate is kept only at this confidence. Rule fragments and
/// specks read as "|", "l" or a letter pair at 0.5-0.7.
const FAINT_CONF: f32 = 0.9;

impl Default for Options {
    fn default() -> Self {
        Options {
            det_long_side: 1600,
            det_thresh: 0.5,
            det_merge: 0.6,
            gutter: true,
            orient: true,
            rescue: true,
            junk: true,
            faint_thresh: 0.1,
            faint_max: 24,
        }
    }
}

/// One recognised text line.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Line {
    pub text: String,
    /// `[x1, y1, x2, y2]` in page pixels, truncated towards zero.
    pub bbox: [f32; 4],
    /// Mean per-character CTC confidence.
    pub confidence: f32,
    /// The reading quad in page pixels, `[tl, tr, br, bl]`.
    pub polygon: Quad,
    /// Whether the line came from the faint short-token pass.
    pub faint: bool,
}

/// What a page read produces.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Page {
    /// Page size in pixels, as the request gave it.
    pub width: u32,
    pub height: u32,
    /// Lines in reading order.
    pub lines: Vec<Line>,
}

#[derive(Debug)]
pub enum OcrError {
    /// The request or the model cannot be used as asked.
    Invalid(String),
    /// The arithmetic produced a non-finite value.
    Numerical(String),
}

impl std::fmt::Display for OcrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OcrError::Invalid(message) | OcrError::Numerical(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for OcrError {}

pub struct Model {
    det: det::Detector,
    router: router::Router,
    /// One recognizer per script group, in [`rec::GROUPS`] order.
    recs: [rec::Recognizer; 4],
    options: Options,
    /// How the GGUF stored the weights. The arithmetic is F32 either way.
    pub format: WeightFormat,
    /// The policy the model was loaded with.
    policy: ComputePolicy,
}

/// One string metadata value, if the file has it.
fn metadata_string(source: &dyn TensorSource, key: &str) -> Option<String> {
    source
        .metadata(key)
        .and_then(MetaValue::to_string_val)
        .map(str::to_string)
}

/// Reads one contract tensor and returns its F32 values.
///
/// A quantized export is dequantized once here, so the compute path stays F32
/// whatever the file stores. Biases and one-dimensional vectors stay F32 in
/// the file: they are small, precision sensitive, and the Q8_0 block size does
/// not divide them.
fn load_tensor(source: &dyn TensorSource, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
    let dims: Vec<u64> = shape.iter().rev().map(|&n| n as u64).collect();
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing OCRia tensor: {name}"))?;
    if info.dims != dims {
        return Err(format!(
            "Invalid OCRia tensor {name}: expected shape {dims:?}"
        ));
    }
    let count = shape.iter().product::<usize>();
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing OCRia data: {name}"))?;
    let data = match info.ggml_type {
        GGMLType::F32 => {
            if bytes.len() != count * 4 {
                return Err(format!("Invalid OCRia tensor length: {name}"));
            }
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        GGMLType::Q8_0 => {
            if shape.len() < 2 {
                return Err(format!(
                    "OCRia tensor {name} is one-dimensional; biases must stay F32"
                ));
            }
            // GGML quantizes a row at a time and a Q8_0 block holds 32 values,
            // so the row length -- the fastest varying dimension -- has to be a
            // whole number of blocks. The converter applies the same rule in
            // `quantizable` in `tools/converter/ocria/convert_ocria.py`.
            let row = shape[shape.len() - 1];
            if row % 32 != 0 {
                return Err(format!(
                    "OCRia tensor {name} has a row length of {row}, which is not a multiple \
                     of the Q8_0 block size 32"
                ));
            }
            if bytes.len() != count / 32 * crate::ops::quant::BLOCK_Q80_SIZE {
                return Err(format!("Invalid OCRia tensor length: {name}"));
            }
            crate::ops::quant::dequant_q80_weight(bytes, count, 1)
        }
        other => {
            return Err(format!(
                "Unsupported OCRia tensor type for {name}: {other:?}; expected F32 or Q8_0"
            ))
        }
    };
    if data.iter().any(|value| !value.is_finite()) {
        return Err(format!("Non-finite OCRia tensor: {name}"));
    }
    Ok(data)
}

/// The `ocria.config` metadata, checked against [`CONTRACT`].
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    det: DetConfig,
    rec: RecConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct DetConfig {
    thresh: f32,
    merge: f32,
    long_side: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecConfig {
    groups: Vec<String>,
    visual_rtl: Vec<bool>,
}

impl Model {
    /// Loads a packed model from `source`.
    ///
    /// `threads` sizes the convolution pool; `compute` records the requested
    /// policy. OCRia has no device path: the graphs are depthwise-separable
    /// convolutions the repository's Vulkan kernels do not cover, and the
    /// upstream release is a CPU pipeline, so `Auto` and `Cpu` both run on the
    /// host and `Vulkan` is refused rather than silently downgraded.
    pub fn from_source(
        source: &dyn TensorSource,
        compute: ComputePolicy,
        threads: usize,
    ) -> Result<Self, String> {
        let meta = |key: &str| metadata_string(source, key);
        if meta("general.architecture").as_deref() != Some(ARCHITECTURE) {
            return Err("Expected the ocria architecture".into());
        }
        if compute == ComputePolicy::Vulkan {
            return Err(
                "OCRia has no Vulkan path: the detector and recognizers are depthwise-separable \
                 convolutions the repository's device kernels do not cover, and the upstream \
                 release is a CPU pipeline. Use --compute cpu or auto"
                    .into(),
            );
        }
        let format = WeightFormat::from_metadata(meta("ocria.format").as_deref())?;
        let contract: serde_json::Value = serde_json::from_str(CONTRACT).unwrap();
        let config_json = meta("ocria.config").ok_or("Missing ocria.config")?;
        let config: Config =
            serde_json::from_str(&config_json).map_err(|e| format!("Invalid OCRia config: {e}"))?;
        let expected: Config = serde_json::from_value(contract["config"].clone()).unwrap();
        if config != expected {
            return Err("Unsupported OCRia configuration".into());
        }
        let mut weights = Weights::new();
        for (name, shape) in contract["tensors"].as_object().unwrap() {
            let shape: Vec<usize> = serde_json::from_value(shape.clone()).unwrap();
            let data = load_tensor(source, name, &shape)?;
            weights.insert(name.clone(), Tensor { shape, data });
        }
        let charset = |group: &str| -> Result<Vec<String>, String> {
            let key = format!("ocria.rec.{group}.charset");
            let raw = meta(&key).ok_or_else(|| format!("Missing {key}"))?;
            serde_json::from_str(&raw).map_err(|e| format!("Invalid {key}: {e}"))
        };
        let pool = Arc::new(ComputePool::new(threads.max(1)));
        let options = Options {
            det_long_side: expected.det.long_side,
            det_thresh: expected.det.thresh,
            det_merge: expected.det.merge,
            ..Options::default()
        };
        let faint =
            (options.faint_thresh > 0.0).then_some((options.faint_thresh, options.faint_max));
        let det = det::Detector::new(
            &weights,
            &pool,
            options.det_long_side,
            options.det_thresh,
            options.det_merge,
            faint,
        )?;
        let router = router::Router::new(&weights, &pool)?;
        let mut recs = Vec::with_capacity(rec::GROUPS.len());
        for (index, group) in expected.rec.groups.iter().enumerate() {
            recs.push(rec::Recognizer::new(
                &weights,
                &pool,
                group,
                charset(group)?,
                expected.rec.visual_rtl[index],
            )?);
        }
        let recs: [rec::Recognizer; 4] = recs
            .try_into()
            .map_err(|_| "The OCRia contract needs exactly four recognizers".to_string())?;
        Ok(Self {
            det,
            router,
            recs,
            options,
            format,
            policy: compute,
        })
    }

    /// The options this model was loaded with.
    pub fn options(&self) -> Options {
        self.options
    }

    /// OCR one page image: RGB (or grey) bytes, row-major.
    pub fn run(&self, data: &[u8], width: u32, height: u32) -> Result<Page, OcrError> {
        let (w, h) = (width as usize, height as usize);
        if w == 0 || h == 0 {
            return Err(OcrError::Invalid("The page is empty".into()));
        }
        if data.len() != w * h && data.len() != w * h * 3 {
            return Err(OcrError::Invalid(format!(
                "Expected {} or {} image bytes, got {}",
                w * h,
                w * h * 3,
                data.len()
            )));
        }
        let page = self.run_page(data, w, h).map_err(OcrError::Numerical)?;
        self.policy.trace("ocria", UsedBackend::Cpu, w * h);
        Ok(Page {
            width,
            height,
            lines: page,
        })
    }

    fn run_page(&self, data: &[u8], width: usize, height: usize) -> Result<Vec<Line>, String> {
        let mut img0 = img::Img {
            w: width,
            h: height,
            ch: if data.len() == width * height { 1 } else { 3 },
            data: data.to_vec(),
        };
        if img0.ch == 1 {
            // the upstream server decodes every request as RGB
            img0 = img0.to_rgb();
        }
        let (mut boxes, mut faint) = self.det.detect_with_faint(&img0)?;
        let mut sideways = false;
        let turned;
        let mut img = &img0;
        if self.options.orient && page::looks_sideways(&boxes) {
            // maybe a page scanned on its side: detect again a quarter turn
            // counter-clockwise and keep that when it finds far more lines
            turned = img0.rot90(1);
            let tboxes = self.det.detect(&turned)?;
            if page::long_lines(&tboxes) > 2 * page::long_lines(&boxes) + 5 {
                img = &turned;
                boxes = tboxes;
                faint.clear();
                sideways = true;
            }
        }
        if self.options.gutter {
            boxes = page::split_gutters(img, &boxes);
        }
        let (mut reads, flip, dom) = if boxes.is_empty() {
            (Vec::new(), false, 0)
        } else {
            self.read(img, &boxes)?
        };
        let n_lines = boxes.len();
        if !faint.is_empty() && !flip {
            // with the page's recognizer: a lone glyph is too short for the router
            let crops: Vec<img::Img> = faint.iter().map(|b| page::crop(img, b)).collect();
            let refs: Vec<&img::Img> = crops.iter().collect();
            let res = self.rec_for(dom).read(&refs)?;
            for (quad, (text, conf)) in faint.into_iter().zip(res) {
                let text = page::clean_latin(&text);
                if conf >= FAINT_CONF && page::is_short_token(&text) {
                    boxes.push(quad);
                    reads.push((text, conf));
                }
            }
        }
        let mut out = Vec::new();
        for (k, (quad, (text, conf))) in boxes.iter().zip(reads).enumerate() {
            if text.trim().is_empty() || (self.options.junk && page::is_junk(&text, conf)) {
                continue;
            }
            let pts = if self.options.orient {
                page::reading_frame(quad, flip, sideways, img0.w)
            } else {
                *quad
            };
            let (mut x1, mut y1, mut x2, mut y2) = (
                f32::INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            );
            for p in &pts {
                x1 = x1.min(p[0]);
                y1 = y1.min(p[1]);
                x2 = x2.max(p[0]);
                y2 = y2.max(p[1]);
            }
            out.push(Line {
                text,
                // the upstream server sends int() of each extent
                bbox: [x1.trunc(), y1.trunc(), x2.trunc(), y2.trunc()],
                confidence: conf,
                polygon: pts,
                faint: k >= n_lines,
            });
        }
        Ok(out)
    }

    fn rec_for(&self, group: usize) -> &rec::Recognizer {
        &self.recs[group]
    }

    /// Readings per box, whether the page was read upside down, and the page's
    /// script group.
    fn read(
        &self,
        img: &img::Img,
        boxes: &[Quad],
    ) -> Result<(Vec<(String, f32)>, bool, usize), String> {
        let mut crops: Vec<img::Img> = boxes.iter().map(|b| page::crop(img, b)).collect();
        let (mut groups, flip, share) = self.router.route(&crops)?;
        if flip {
            crops = crops.iter().map(|c| c.rot90(2)).collect();
        }
        let n = crops.len();
        let mut out = vec![(String::new(), 0f32); n];
        let subset = |idx: &[usize]| -> Vec<&img::Img> { idx.iter().map(|&i| &crops[i]).collect() };
        for gi in 0..rec::GROUPS.len() {
            let idx: Vec<usize> = (0..n).filter(|&i| groups[i] == gi).collect();
            if idx.is_empty() {
                continue;
            }
            for (&i, res) in idx.iter().zip(self.rec_for(gi).read(&subset(&idx))?) {
                out[i] = res;
            }
        }
        let mut dom = page::majority(&groups);
        // a routed reading this confident is a real reading, not the 0.5-0.7
        // junk of the wrong recognizer
        const KEEP_CONF: f32 = 0.75;
        if self.options.rescue && dom == 0 {
            // a whole page of another script sent to the Latin recognizer reads
            // as confident-looking junk at 0.5-0.7; the right recognizer then
            // scores 0.83-0.90
            const RESCUE_CONF: f64 = 0.75;
            const RESCUE_MARGIN: f64 = 0.15;
            // only scripts the router saw a fair share of are tried
            const RESCUE_SHARE: f64 = 0.2;
            let lat: Vec<usize> = (0..n).filter(|&i| groups[i] == dom).collect();
            if !lat.is_empty() && share[1..].iter().any(|&s| s >= RESCUE_SHARE) {
                let lat_conf = page::mean(lat.iter().map(|&i| out[i].1));
                if lat_conf < RESCUE_CONF {
                    let sample: Vec<usize> = lat.iter().copied().take(40).collect();
                    let (mut best, mut best_conf) = (None, lat_conf + RESCUE_MARGIN);
                    for gi in (1..rec::GROUPS.len()).filter(|&gi| share[gi] >= RESCUE_SHARE) {
                        let c = page::mean(
                            self.rec_for(gi)
                                .read(&subset(&sample))?
                                .into_iter()
                                .map(|r| r.1),
                        );
                        if c >= best_conf {
                            best = Some(gi);
                            best_conf = c;
                        }
                    }
                    if let Some(gi) = best {
                        for (&i, res) in lat.iter().zip(self.rec_for(gi).read(&subset(&lat))?) {
                            // per line: a bilingual page keeps its confident Latin lines
                            if !(out[i].1 >= KEEP_CONF && out[i].1 >= res.1) {
                                out[i] = res;
                                groups[i] = gi;
                            }
                        }
                        dom = gi;
                    }
                }
            }
            // per-line rescue when the page stays Latin
            if dom == 0 {
                const LINE_MARGIN: f32 = 0.1;
                const LINE_SHARE: f64 = 0.15;
                const LINE_CONF: f32 = 0.8;
                for gi in (1..rec::GROUPS.len()).filter(|&gi| share[gi] >= LINE_SHARE) {
                    let weak: Vec<usize> = (0..n).filter(|&i| groups[i] == 0).collect();
                    if weak.is_empty() {
                        break;
                    }
                    for (&i, res) in weak.iter().zip(self.rec_for(gi).read(&subset(&weak))?) {
                        if res.1 >= out[i].1 + LINE_MARGIN
                            && res.1 >= LINE_CONF
                            && page::script_chars(gi, &res.0) >= 3
                            && page::script_chars(gi, &res.0) > page::latin_letters(&res.0)
                        {
                            out[i] = res;
                            groups[i] = gi;
                        }
                    }
                }
            }
        }
        // lines routed away from the page's script are read with the page's
        // recognizer too; the off-script reading must win clearly
        let odd: Vec<usize> = (0..n).filter(|&i| groups[i] != dom).collect();
        if !odd.is_empty() {
            for (&i, res) in odd.iter().zip(self.rec_for(dom).read(&subset(&odd))?) {
                let routed_ok = out[i].1 >= KEEP_CONF
                    && page::script_chars(groups[i], &out[i].0) >= 2
                    && page::script_chars(groups[i], &out[i].0) > page::latin_letters(&out[i].0);
                let replace = !routed_ok
                    && (out[i].1 < res.1 + 0.1 || page::script_chars(groups[i], &out[i].0) < 3);
                if replace {
                    out[i] = res;
                }
            }
        }
        for o in out.iter_mut() {
            o.0 = page::clean_latin(&o.0);
        }
        Ok((out, flip, dom))
    }
}
