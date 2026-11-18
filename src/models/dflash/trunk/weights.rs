//! The weights of a DFlash / DSpark draft model.
//!
//! A draft file is a small transformer plus, for DSpark, the two extra heads:
//!
//! ```text
//! fc            [n_target_layers * n_embd_target, n_embd]   the encoder's fusion
//! enc.output_norm [n_embd]                                  the encoder's norm
//! blk.N.*                                                    the draft's blocks
//! output_norm  [n_embd]                                     the decoder's final norm
//! output       [n_embd, n_vocab_draft]                      the decoder's head
//! markov_w1    [rank, n_vocab]                              the Markov head's table
//! markov_w2    [rank, n_vocab_draft]                        the Markov head's projection
//! conf_proj    [n_embd + rank, 1]                          the confidence head
//! d2t          [n_vocab_draft]                              draft row -> target token id
//! ```
//!
//! `token_embd` is **not** in the file: the draft embeds its anchor and its
//! mask tokens with the *target's* embedding table (llama.cpp shares it
//! through `ctx_other`).

use super::config::DflashConfig;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::ops::kernel::{QuantizedTensor, Weight};

/// One draft block: a plain (non-gated) attention layer plus a SwiGLU FFN.
///
/// The tensor names are llama.cpp's (`src/models/dflash.cpp`):
/// `attn_norm`, `attn_q/k/v`, `attn_q_norm`, `attn_k_norm`, `attn_output`,
/// `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down`.
pub struct DflashLayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    /// The norm in front of the FFN. Upstream calls it `ffn_norm`.
    pub ffn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub wq: Weight<'a>,
    pub wk: Weight<'a>,
    pub wv: Weight<'a>,
    pub wo: Weight<'a>,
    pub w_gate: Weight<'a>,
    pub w_up: Weight<'a>,
    pub w_down: Weight<'a>,
}

/// The Markov head's lookup table, kept as raw bytes.
///
/// It is *gathered* by row (`markov_w1[token_id]`), not multiplied, so it
/// does not go through the matmul path.
pub struct MarkovTable<'a> {
    pub bytes: &'a [u8],
    pub ggml_type: GGMLType,
    /// The rank: the length of one row.
    pub rank: usize,
    /// How many rows the table has.
    pub rows: usize,
}

impl<'a> MarkovTable<'a> {
    /// Dequantize one row.
    ///
    /// The formats a DSpark file ships the table in are the ones the draft's
    /// other weights use, so this mirrors the GGUF block layouts:
    /// Q8_0 is `[f16 scale][32 int8]` per block, BF16 and F16 are two bytes
    /// per element, F32 four.
    pub fn row(&self, index: usize) -> Result<Vec<f32>, String> {
        if index >= self.rows {
            return Err(format!(
                "DFlash Markov table row {index} is out of range ({} rows)",
                self.rows
            ));
        }
        let mut out = vec![0.0f32; self.rank];
        match self.ggml_type {
            GGMLType::F32 => {
                let offset = index * self.rank * 4;
                let bytes = self
                    .bytes
                    .get(offset..offset + self.rank * 4)
                    .ok_or_else(|| "DFlash Markov table is truncated".to_string())?;
                for (index, value) in out.iter_mut().enumerate() {
                    *value = f32::from_le_bytes([
                        bytes[index * 4],
                        bytes[index * 4 + 1],
                        bytes[index * 4 + 2],
                        bytes[index * 4 + 3],
                    ]);
                }
            }
            GGMLType::BF16 | GGMLType::F16 => {
                let offset = index * self.rank * 2;
                let bytes = self
                    .bytes
                    .get(offset..offset + self.rank * 2)
                    .ok_or_else(|| "DFlash Markov table is truncated".to_string())?;
                for (index, value) in out.iter_mut().enumerate() {
                    let bits = u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]]);
                    *value = if self.ggml_type == GGMLType::BF16 {
                        crate::ops::float::bf16_to_f32(bits)
                    } else {
                        crate::ops::float::f16_to_f32(bits)
                    };
                }
            }
            GGMLType::Q8_0 => {
                let blocks = self.rank.div_ceil(32);
                let row_bytes = blocks * 34;
                let offset = index * row_bytes;
                let bytes = self
                    .bytes
                    .get(offset..offset + row_bytes)
                    .ok_or_else(|| "DFlash Markov table is truncated".to_string())?;
                for block in 0..blocks {
                    let base = block * 34;
                    let scale = crate::ops::float::f16_to_f32(u16::from_le_bytes([
                        bytes[base],
                        bytes[base + 1],
                    ]));
                    for element in 0..32 {
                        let at = block * 32 + element;
                        if at >= self.rank {
                            break;
                        }
                        out[at] = scale * (bytes[base + 2 + element] as i8) as f32;
                    }
                }
            }
            other => {
                return Err(format!(
                    "DFlash Markov table uses the unsupported quant type {other:?}"
                ));
            }
        }
        Ok(out)
    }
}

/// The whole draft model.
pub struct DflashWeights<'a> {
    pub config: DflashConfig,
    /// The encoder's fusion projection: the target's hidden states, gathered
    /// from several layers, into the draft's own space.
    pub fc: Weight<'a>,
    /// The encoder's input width: `target_layers.len()` times the target's
    /// hidden size. Read off `fc`'s shape, since the draft file carries no
    /// copy of the target's hyperparameters.
    pub n_embd_enc: usize,
    /// The encoder's norm, applied after `fc`.
    pub enc_output_norm: Vec<f32>,
    pub layers: Vec<DflashLayerWeights<'a>>,
    /// The decoder's final norm.
    pub output_norm: Vec<f32>,
    /// The decoder's head. `None` means the target's own output weight is
    /// used (a draft that shares the target's vocabulary).
    pub output: Option<Weight<'a>>,
    /// The Markov head's lookup table. `None` makes this a plain DFlash
    /// draft rather than a DSpark one.
    pub markov_w1: Option<MarkovTable<'a>>,
    /// The Markov head's projection: the rank-sized state to the draft's
    /// vocabulary.
    pub markov_w2: Option<Weight<'a>>,
    /// The confidence head: the draft's hidden state and the Markov state to
    /// one acceptance probability.
    pub conf_proj: Option<Weight<'a>>,
    pub conf_proj_bias: Vec<f32>,
    /// The draft row to target token id map, for a draft with a reduced
    /// vocabulary.
    pub d2t: Option<Vec<i64>>,
}

impl<'a> DflashWeights<'a> {
    /// Whether this draft carries the DSpark Markov head.
    pub fn is_dspark(&self) -> bool {
        self.markov_w1.is_some() && self.markov_w2.is_some()
    }
}

/// Load a quantized weight, borrowing the GGUF bytes.
pub(crate) fn load_weight<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
) -> Option<Weight<'a>> {
    let info = source.tensor_info(name)?;
    let data = source.tensor_slice(name)?;
    let n_cols = info.dims.first().copied().unwrap_or(0) as usize;
    let n_rows = if info.dims.len() >= 2 {
        info.dims[1] as usize
    } else {
        1
    };
    match info.ggml_type {
        GGMLType::F32 => Weight::from_f32_bytes(data, n_cols, n_rows),
        GGMLType::F16
        | GGMLType::BF16
        | GGMLType::Q8_0
        | GGMLType::Q4_0
        | GGMLType::Q4_1
        | GGMLType::Q4K
        | GGMLType::Q5K
        | GGMLType::Q6K => {
            let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
                data,
                info.ggml_type,
                n_cols,
                n_rows,
            ));
            weight.n_in = n_cols;
            weight.n_out = n_rows;
            if info.ggml_type == GGMLType::BF16 {
                weight.kernel =
                    Box::new(crate::ops::kernel::bf16::BF16Kernel::with_bf16_input(data));
            }
            Some(weight)
        }
        other => {
            eprintln!(
                "WARNING: unsupported quant type {other:?} for tensor {name} (DFlash draft)"
            );
            None
        }
    }
}

/// Load a norm or bias into an owned `Vec<f32>`.
fn load_vec<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    expected: usize,
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("DFlash draft is missing {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("DFlash draft is missing {name}"))?;
    let count = info.dims.first().copied().unwrap_or(0) as usize;
    if count != expected {
        return Err(format!(
            "{name}: expected {expected} values, got {count}"
        ));
    }
    match info.ggml_type {
        GGMLType::F32 => {
            let mut out = Vec::with_capacity(count);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        GGMLType::BF16 | GGMLType::F16 => {
            let mut out = Vec::with_capacity(count);
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(if info.ggml_type == GGMLType::BF16 {
                    crate::ops::float::bf16_to_f32(bits)
                } else {
                    crate::ops::float::f16_to_f32(bits)
                });
            }
            Ok(out)
        }
        other => Err(format!("{name}: unsupported quant type {other:?} for a norm")),
    }
}

/// Load a draft model.
pub fn load_weights<'a, S: TensorSource + ?Sized>(
    source: &'a S,
) -> Result<DflashWeights<'a>, String> {
    let config = DflashConfig::from_source(source)?;

    let need = |name: &str| -> Result<Weight<'a>, String> {
        load_weight(source, name).ok_or_else(|| format!("DFlash draft is missing {name}"))
    };
    let need_vec = |name: &str, expected: usize| -> Result<Vec<f32>, String> {
        load_vec(source, name, expected)
    };

    let n_embd = config.n_embd;
    let _n_embd_q = config.n_head * config.n_embd_head_k;
    let _n_embd_gqa = config.n_head_kv * config.n_embd_head_k;

    // The encoder fuses the target's hidden states from several layers, so
    // its input width is that many times the target's hidden size. The
    // target's hidden size is not in the draft file's metadata — read the
    // width off fc's shape and check it divides evenly.
    let fc = need("fc.weight")?;
    let n_embd_enc = fc.n_in;
    if fc.n_out != n_embd {
        return Err(format!(
            "DFlash draft fc.weight is [{n_embd_enc}, {}], expected [.., {n_embd}]",
            fc.n_out
        ));
    }
    if n_embd_enc == 0 || n_embd_enc % config.target_layers.len() != 0 {
        return Err(format!(
            "DFlash draft fc input width {n_embd_enc} does not divide into {} target layers",
            config.target_layers.len()
        ));
    }
    let enc_output_norm = need_vec("enc.output_norm.weight", n_embd)?;
    let output_norm = need_vec("output_norm.weight", n_embd)?;

    let mut layers = Vec::with_capacity(config.n_layer);
    for layer in 0..config.n_layer {
        let prefix = format!("blk.{layer}.");
        layers.push(DflashLayerWeights {
            attn_norm: need_vec(&format!("{prefix}attn_norm.weight"), n_embd)?,
            ffn_norm: need_vec(&format!("{prefix}ffn_norm.weight"), n_embd)?,
            q_norm: need_vec(&format!("{prefix}attn_q_norm.weight"), config.n_embd_head_k)?,
            k_norm: need_vec(&format!("{prefix}attn_k_norm.weight"), config.n_embd_head_k)?,
            wq: need(&format!("{prefix}attn_q.weight"))?,
            wk: need(&format!("{prefix}attn_k.weight"))?,
            wv: need(&format!("{prefix}attn_v.weight"))?,
            wo: need(&format!("{prefix}attn_output.weight"))?,
            w_gate: need(&format!("{prefix}ffn_gate.weight"))?,
            w_up: need(&format!("{prefix}ffn_up.weight"))?,
            w_down: need(&format!("{prefix}ffn_down.weight"))?,
        });
    }

    // The head projects onto the draft's own vocabulary, which may be a
    // reduced one; without it the target's head is shared.
    let output = load_weight(source, "output.weight");
    let n_vocab_draft = config.n_vocab_draft.max(
        output
            .as_ref()
            .map(|weight| weight.n_out)
            .unwrap_or(0),
    );

    let markov_w1 = source.tensor_info("markov_w1.weight").map(|info| MarkovTable {
        bytes: source
            .tensor_slice("markov_w1.weight")
            .unwrap_or(&[]),
        ggml_type: info.ggml_type,
        rank: info.dims.first().copied().unwrap_or(0) as usize,
        rows: info.dims.get(1).copied().unwrap_or(0) as usize,
    });
    let markov_w2 = load_weight(source, "markov_w2.weight");

    let conf_proj = load_weight(source, "conf_proj.weight");
    let conf_proj_bias = match source.tensor_info("conf_proj.bias") {
        Some(_) => load_vec(source, "conf_proj.bias", 1)?,
        None => Vec::new(),
    };

    // The draft row to target token id map.
    let d2t = match source.tensor_info("d2t") {
        Some(info) => {
            let bytes = source
                .tensor_slice("d2t")
                .ok_or_else(|| "DFlash draft is missing the d2t map".to_string())?;
            let count = info.dims.first().copied().unwrap_or(0) as usize;
            let mut out = Vec::with_capacity(count);
            for chunk in bytes.chunks_exact(8) {
                out.push(i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                    chunk[7],
                ]));
            }
            Some(out)
        }
        None => None,
    };

    if let Some(map) = &d2t {
        if map.len() != n_vocab_draft {
            return Err(format!(
                "DFlash draft d2t has {} entries but the head projects onto {n_vocab_draft}",
                map.len()
            ));
        }
    }

    Ok(DflashWeights {
        config,
        fc,
        n_embd_enc,
        enc_output_norm,
        layers,
        output_norm,
        output,
        markov_w1,
        markov_w2,
        conf_proj,
        conf_proj_bias,
        d2t,
    })
}
