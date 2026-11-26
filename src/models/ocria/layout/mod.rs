//! LayoutNet: the release's table and layout-block model.
//!
//! `tablenet.onnx` is a 6-layer transformer over the page's text units with a
//! small CNN image branch. It takes 68 features per unit, pairwise geometry and
//! a page raster, and returns per-unit node logits, pairwise relation logits and
//! 13 layout classes; `decode` turns those into tables (rows, columns, cells,
//! spans) and layout blocks.
//!
//! This is layer 2 of the OCRia adaptation. The OCR stack feeds it: units come
//! from the lines `super::Model::run` reads, so the model needs no PDF text
//! layer. The reference runtime reaches the same place through its projection
//! pass; `tools/oracle/ocria/layout-plan.md` records the ablation showing the
//! table model reproduces the reference's table without those signals.
pub(crate) mod decode;
pub(crate) mod features;
pub(crate) mod net;
pub(crate) mod units;

use super::{Tensor, Weights};
use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use std::collections::HashMap;

/// The `general.architecture` value the layout converter writes.
pub(crate) const ARCHITECTURE: &str = "ocria-layout";
const CONTRACT: &str = include_str!("contract.json");

/// One text unit: a line of the page, in page points, top-left origin.
#[derive(Clone, Debug)]
pub(crate) struct Unit {
    /// Reading-frame box.
    pub bx: [f32; 4],
    /// Rendered-page box.
    pub pbox: [f32; 4],
    pub text: String,
    /// Line height without the recognizer's padding.
    pub h: f32,
    pub bold: Option<bool>,
    /// Whether the unit came from the faint short-token pass.
    pub faint: bool,
}

/// A decoded table: its page box and the cell grid.
#[derive(Clone, Debug)]
pub(crate) struct Table {
    pub bbox: [f64; 4],
    pub grid: decode::Grid,
}

/// One layout block: a run of units with one class and a page box.
#[derive(Clone, Debug)]
pub(crate) struct Block {
    pub bbox: [f64; 4],
    pub class: usize,
    pub units: Vec<usize>,
}

/// What one page decodes to.
#[derive(Clone, Debug, Default)]
pub(crate) struct Layout {
    pub tables: Vec<Table>,
    pub blocks: Vec<Block>,
    /// Per-unit layout class index (0 when the model did not run).
    pub classes: Vec<usize>,
}

/// The loaded table model.
pub(crate) struct LayoutNet {
    net: net::TableNet,
}

impl LayoutNet {
    /// Loads the model from its own packed GGUF.
    ///
    /// The file carries the model's config and the 110 tensors; the shapes are
    /// checked against `contract.json`, which the converter writes from the same
    /// graph, so a converter/loader drift fails at load rather than at inference.
    pub(crate) fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let meta = |key: &str| {
            source
                .metadata(key)
                .and_then(MetaValue::to_string_val)
                .map(str::to_string)
        };
        let contract: serde_json::Value = serde_json::from_str(CONTRACT).unwrap();
        let config = meta("ocria-layout.config").ok_or("Missing ocria-layout.config")?;
        let config: serde_json::Value = serde_json::from_str(&config)
            .map_err(|e| format!("Invalid ocria-layout.config: {e}"))?;
        if config != contract["config"] {
            return Err("Unsupported LayoutNet configuration".into());
        }
        let mut weights: Weights = HashMap::new();
        for (name, shape) in contract["tensors"].as_object().unwrap() {
            let shape: Vec<usize> = serde_json::from_value(shape.clone()).unwrap();
            let data = load_tensor(source, name, &shape)?;
            weights.insert(name.clone(), Tensor { shape, data });
        }
        Ok(Self {
            net: net::TableNet::new(&weights)?,
        })
    }

    /// The network, for the parity test in `models::ocria::tests`.
    #[cfg(test)]
    pub(crate) fn net(&self) -> &net::TableNet {
        &self.net
    }

    /// Runs the model on one page's units and decodes the result.
    ///
    /// `rules` and `image` are optional: the model was trained with both, and
    /// the page raster is what the image branch reads. A page with fewer than
    /// two units, or fewer than 20 characters over all of them, is not a
    /// layout page and decodes to nothing (the reference's own guard).
    pub(crate) fn run(
        &self,
        units: &[Unit],
        rules: Option<&features::Rules>,
        image: Option<&features::PageImage>,
    ) -> Layout {
        let n = units.len();
        if n < 2 || n > MAX_UNITS {
            return Layout::default();
        }
        if units.iter().map(|u| u.text.chars().count()).sum::<usize>() < 20 {
            return Layout::default();
        }
        let funits: Vec<features::FUnit> = units
            .iter()
            .map(|u| features::FUnit {
                bx: u.bx,
                pbox: u.pbox,
                text: u.text.clone(),
                h: u.h,
                bold: u.bold,
                proj: None,
            })
            .collect();
        let wh = page_size(units);
        let feats = features::unit_features(&funits, wh, rules);
        let pairs = features::pair_features(&funits, wh, rules);
        let img = image.map(|image| image.input(&units.iter().map(|u| u.pbox).collect::<Vec<_>>()));
        let logits = self.net.forward(&feats, &pairs, img.as_ref(), n);
        decode::decode(
            &units_to_dunits(units),
            &logits,
            &decode::DecodeOptions::default(),
        )
    }
}

/// The class head's labels, in the order the release's model emits them
/// (`build_grounded.py` `BLOCK_CLASSES`).
pub(crate) const BLOCK_CLASSES: [&str; 13] = [
    "Text",
    "Heading",
    "Title",
    "PageHeader",
    "PageFooter",
    "List",
    "Table",
    "Caption",
    "Footnote",
    "Formula",
    "Code",
    "FigureText",
    "KeyValue",
];

/// Pages with more units than this skip the model: attention and decoding grow
/// with `N^2`, and such pages are dense prose or indexes.
pub(crate) const MAX_UNITS: usize = 1500;

/// The page size the features normalise against: the union of the unit boxes.
fn page_size(units: &[Unit]) -> (f32, f32) {
    let mut w = 0.0f32;
    let mut h = 0.0f32;
    for unit in units {
        w = w.max(unit.pbox[2]);
        h = h.max(unit.pbox[3]);
    }
    (w.max(1.0), h.max(1.0))
}

fn units_to_dunits(units: &[Unit]) -> Vec<decode::DUnit> {
    units
        .iter()
        .map(|u| decode::DUnit {
            bx: [
                u.pbox[0] as f64,
                u.pbox[1] as f64,
                u.pbox[2] as f64,
                u.pbox[3] as f64,
            ],
            text: u.text.clone(),
            h: u.h as f64,
            minor: false,
        })
        .collect()
}

/// Reads one contract tensor as F32, dequantizing a Q8_0 row set.
fn load_tensor(source: &dyn TensorSource, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
    let dims: Vec<u64> = shape.iter().rev().map(|&n| n as u64).collect();
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing LayoutNet tensor: {name}"))?;
    if info.dims != dims {
        return Err(format!(
            "Invalid LayoutNet tensor {name}: expected shape {dims:?}"
        ));
    }
    let count = shape.iter().product::<usize>();
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing LayoutNet data: {name}"))?;
    let data = match info.ggml_type {
        GGMLType::F32 => {
            if bytes.len() != count * 4 {
                return Err(format!("Invalid LayoutNet tensor length: {name}"));
            }
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        GGMLType::Q8_0 => {
            let row = shape[shape.len() - 1];
            if shape.len() < 2 || row % 32 != 0 {
                return Err(format!(
                    "LayoutNet tensor {name} has a row length of {row}, which Q8_0 cannot store"
                ));
            }
            if bytes.len() != count / 32 * crate::ops::quant::BLOCK_Q80_SIZE {
                return Err(format!("Invalid LayoutNet tensor length: {name}"));
            }
            crate::ops::quant::dequant_q80_weight(bytes, count, 1)
        }
        other => {
            return Err(format!(
                "Unsupported LayoutNet tensor type for {name}: {other:?}; expected F32 or Q8_0"
            ))
        }
    };
    if data.iter().any(|value| !value.is_finite()) {
        return Err(format!("Non-finite LayoutNet tensor: {name}"));
    }
    Ok(data)
}
