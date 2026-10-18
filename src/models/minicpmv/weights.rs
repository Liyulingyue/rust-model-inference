//! Weight binding for the `minicpmv4_7` projector.
//!
//! Tensor names follow `tools/mtmd/clip.cpp` (`case PROJECTOR_TYPE_MINICPMV4_7`)
//! at the pinned oracle `69f201a2051ea9b9e9b50c3cc56afd9e2ae64414`:
//! the vision tower is `v.*`, the vit_merger is `v.vit_merger.*`, the final
//! projector is `mm.*`, and the per-tile separators are the rows of
//! `v.tok_embd_sep`.

use super::config::MiniCpmV47Config;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::ops::kernel::QuantizedTensor;
use crate::ops::kernel::Weight;

pub type Linear = Weight<'static>;

/// Decode a F32/F16/BF16 parameter slice into row-major `[n_out][n_in]`.
pub fn decode_param(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let data = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let expected = n_in.checked_mul(n_out).ok_or("element count overflow")?;
    let got = info.checked_n_elements().unwrap_or(0) as usize;
    if got != expected {
        return Err(format!(
            "{name}: expected {n_out}x{n_in} = {expected} elements, got {got}"
        ));
    }
    decode_matrix(data, info.ggml_type, n_in, n_out)
}

/// Decode a parameter slice into row-major `[n_out][n_in]` f32.
///
/// F32 / F16 / BF16 are decoded inline, byte-for-byte as before. Everything
/// else is delegated to [`crate::ops::quant::dequantize_tensor`] rather than
/// refused: this used to reject every quantized type, so a Q8_0 / Q4_K / Q6_K
/// mmproj was unloadable even though the shared decoder covers exactly those
/// three. The projector's weights stay eagerly materialised as `Vec<f32>`
/// because its forward is pinned to an oracle trace; the lazy `Weight` /
/// `QuantizedTensor` path would change that, so only the decode is widened
/// here.
pub fn decode_matrix(
    data: &[u8],
    ty: GGMLType,
    n_in: usize,
    n_out: usize,
) -> Result<Vec<f32>, String> {
    let elements = n_in.checked_mul(n_out).ok_or("element count overflow")?;
    match ty {
        GGMLType::F32 => {
            if data.len() != elements * 4 {
                return Err(format!(
                    "F32 tensor needs {} bytes, got {}",
                    elements * 4,
                    data.len()
                ));
            }
            Ok(data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        GGMLType::F16 => {
            if data.len() != elements * 2 {
                return Err(format!(
                    "F16 tensor needs {} bytes, got {}",
                    elements * 2,
                    data.len()
                ));
            }
            Ok(data
                .chunks_exact(2)
                .map(|c| crate::ops::float::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        GGMLType::BF16 => {
            if data.len() != elements * 2 {
                return Err(format!(
                    "BF16 tensor needs {} bytes, got {}",
                    elements * 2,
                    data.len()
                ));
            }
            Ok(data
                .chunks_exact(2)
                .map(|c| crate::core::tensor::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        // Quantized projectors go through the shared whole-tensor decoder.
        // `n_cols` is `n_in`: the tensor is stored `[n_out][n_in]`.
        other => crate::ops::quant::dequantize_tensor(data, other, n_in, n_out)
            .map_err(|e| format!("unsupported ggml type {other:?} for a projector parameter: {e}")),
    }
}

#[derive(Debug, Clone, Default)]
pub struct ViTLayer {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub q_w: Vec<f32>,
    pub q_b: Vec<f32>,
    pub k_w: Vec<f32>,
    pub k_b: Vec<f32>,
    pub v_w: Vec<f32>,
    pub v_b: Vec<f32>,
    pub o_w: Vec<f32>,
    pub o_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub ff_up_w: Vec<f32>,
    pub ff_up_b: Vec<f32>,
    pub ff_down_w: Vec<f32>,
    pub ff_down_b: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct MergerWeights {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub q_w: Vec<f32>,
    pub q_b: Vec<f32>,
    pub k_w: Vec<f32>,
    pub k_b: Vec<f32>,
    pub v_w: Vec<f32>,
    pub v_b: Vec<f32>,
    pub o_w: Vec<f32>,
    pub o_b: Vec<f32>,
    pub ds_ln_w: Vec<f32>,
    pub ds_ln_b: Vec<f32>,
    pub ds_up_w: Vec<f32>,
    pub ds_up_b: Vec<f32>,
    pub ds_down_w: Vec<f32>,
    pub ds_down_b: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct ProjectorWeights {
    pub input_norm_w: Vec<f32>,
    pub input_norm_b: Vec<f32>,
    pub up_w: Vec<f32>,
    pub up_b: Vec<f32>,
    pub down_w: Vec<f32>,
    pub down_b: Vec<f32>,
}

pub struct MiniCpmV47Weights {
    pub config: MiniCpmV47Config,
    pub patch_embd_w: Vec<f32>,
    pub patch_embd_b: Vec<f32>,
    /// `[n_positions, n_embd]`, row major — ggml's `get_rows` indexes the
    /// first dimension, so this is indexed by patch position.
    pub position_embd: Vec<f32>,
    pub n_positions: usize,
    pub layers: Vec<ViTLayer>,
    pub post_ln_w: Vec<f32>,
    pub post_ln_b: Vec<f32>,
    pub merger: MergerWeights,
    pub projector: ProjectorWeights,
    /// `[n_rows][n_embd_text]` — the `</image>`, `<slice>`, `</slice>` and
    /// newline separators appended after each tile.
    pub tok_embd_sep: Vec<f32>,
    pub n_sep_rows: usize,
}

impl MiniCpmV47Weights {
    pub fn load(source: &dyn TensorSource) -> Result<Self, String> {
        let config = MiniCpmV47Config::from_source(source)?;
        let d = config.n_embd;
        let h = config.n_head;

        let patch_embd_w = decode_param(
            source,
            "v.patch_embd.weight",
            3 * config.patch_size * config.patch_size,
            d,
        )?;
        let patch_embd_b = decode_param(source, "v.patch_embd.bias", 1, d)?;

        let pos_info = source
            .tensor_info("v.position_embd.weight")
            .ok_or("Missing tensor: v.position_embd.weight")?;
        // GGUF lists ggml dimensions, so `dims[0]` is the contiguous width and
        // `dims[1]` the number of rows `ggml_get_rows` indexes.
        let pos_dims: Vec<usize> = pos_info.dims.iter().map(|&v| v as usize).collect();
        if pos_dims.len() != 2 || pos_dims[0] != d {
            return Err(format!(
                "v.position_embd.weight: expected [{d}, n_pos], got {pos_dims:?}"
            ));
        }
        let n_positions = pos_dims[1];
        let position_embd = decode_param(source, "v.position_embd.weight", d, n_positions)?;

        let mut layers = Vec::with_capacity(config.n_layer);
        for il in 0..config.n_layer {
            let p = format!("v.blk.{il}");
            layers.push(ViTLayer {
                ln1_w: decode_param(source, &format!("{p}.ln1.weight"), 1, d)?,
                ln1_b: decode_param(source, &format!("{p}.ln1.bias"), 1, d)?,
                q_w: decode_param(source, &format!("{p}.attn_q.weight"), d, d)?,
                q_b: decode_param(source, &format!("{p}.attn_q.bias"), 1, d)?,
                k_w: decode_param(source, &format!("{p}.attn_k.weight"), d, d)?,
                k_b: decode_param(source, &format!("{p}.attn_k.bias"), 1, d)?,
                v_w: decode_param(source, &format!("{p}.attn_v.weight"), d, d)?,
                v_b: decode_param(source, &format!("{p}.attn_v.bias"), 1, d)?,
                o_w: decode_param(source, &format!("{p}.attn_out.weight"), d, d)?,
                o_b: decode_param(source, &format!("{p}.attn_out.bias"), 1, d)?,
                ln2_w: decode_param(source, &format!("{p}.ln2.weight"), 1, d)?,
                ln2_b: decode_param(source, &format!("{p}.ln2.bias"), 1, d)?,
                ff_up_w: decode_param(source, &format!("{p}.ffn_up.weight"), d, config.n_ff)?,
                ff_up_b: decode_param(source, &format!("{p}.ffn_up.bias"), 1, config.n_ff)?,
                ff_down_w: decode_param(source, &format!("{p}.ffn_down.weight"), config.n_ff, d)?,
                ff_down_b: decode_param(source, &format!("{p}.ffn_down.bias"), 1, d)?,
            });
        }

        let post_ln_w = decode_param(source, "v.post_ln.weight", 1, d)?;
        let post_ln_b = decode_param(source, "v.post_ln.bias", 1, d)?;

        // vit_merger: attention over 4x4 windows, then a 2x2 downsample MLP
        // whose input width is n_embd * 4.
        let merged = d * 4;
        // `ds_ffn_up` widens the concatenated 4x-merged rows; read its output
        // width off the GGUF rather than hardcoding it.
        let ds_up_dims = tensor_dims(source, "v.vit_merger.ds_ffn_up.weight")?;
        if ds_up_dims.len() != 2 || ds_up_dims[0] != merged {
            return Err(format!(
                "v.vit_merger.ds_ffn_up.weight: expected [{merged}, *], got {ds_up_dims:?}"
            ));
        }
        let ds_hidden = ds_up_dims[1];
        let merger = MergerWeights {
            ln1_w: decode_param(source, "v.vit_merger.ln1.weight", 1, d)?,
            ln1_b: decode_param(source, "v.vit_merger.ln1.bias", 1, d)?,
            q_w: decode_param(source, "v.vit_merger.attn_q.weight", d, d)?,
            q_b: decode_param(source, "v.vit_merger.attn_q.bias", 1, d)?,
            k_w: decode_param(source, "v.vit_merger.attn_k.weight", d, d)?,
            k_b: decode_param(source, "v.vit_merger.attn_k.bias", 1, d)?,
            v_w: decode_param(source, "v.vit_merger.attn_v.weight", d, d)?,
            v_b: decode_param(source, "v.vit_merger.attn_v.bias", 1, d)?,
            o_w: decode_param(source, "v.vit_merger.attn_out.weight", d, d)?,
            o_b: decode_param(source, "v.vit_merger.attn_out.bias", 1, d)?,
            ds_ln_w: decode_param(source, "v.vit_merger.ds_ln.weight", 1, merged)?,
            ds_ln_b: decode_param(source, "v.vit_merger.ds_ln.bias", 1, merged)?,
            ds_up_w: decode_param(source, "v.vit_merger.ds_ffn_up.weight", merged, ds_hidden)?,
            ds_up_b: decode_param(source, "v.vit_merger.ds_ffn_up.bias", 1, ds_hidden)?,
            ds_down_w: decode_param(source, "v.vit_merger.ds_ffn_down.weight", ds_hidden, d)?,
            ds_down_b: decode_param(source, "v.vit_merger.ds_ffn_down.bias", 1, d)?,
        };

        let up_info = source
            .tensor_info("mm.up.weight")
            .ok_or("Missing tensor: mm.up.weight")?;
        let up_dims: Vec<usize> = up_info.dims.iter().map(|&v| v as usize).collect();
        if up_dims.len() != 2 || up_dims[0] != merged {
            return Err(format!(
                "mm.up.weight: expected [*, {merged}], got {up_dims:?}"
            ));
        }
        let projector = ProjectorWeights {
            input_norm_w: decode_param(source, "mm.input_norm.weight", 1, merged)?,
            input_norm_b: decode_param(source, "mm.input_norm.bias", 1, merged)?,
            up_w: decode_param(source, "mm.up.weight", merged, up_dims[1])?,
            up_b: decode_param(source, "mm.up.bias", 1, up_dims[1])?,
            down_w: decode_param(source, "mm.down.weight", merged, config.n_embd_text)?,
            down_b: decode_param(source, "mm.down.bias", 1, config.n_embd_text)?,
        };

        let sep_info = source
            .tensor_info("v.tok_embd_sep")
            .ok_or("Missing tensor: v.tok_embd_sep")?;
        let sep_dims: Vec<usize> = sep_info.dims.iter().map(|&v| v as usize).collect();
        if sep_dims.len() != 2 || sep_dims[0] != config.n_embd_text {
            return Err(format!(
                "v.tok_embd_sep: expected [{}, *], got {sep_dims:?}",
                config.n_embd_text
            ));
        }
        let n_sep_rows = sep_dims[1];
        let tok_embd_sep = decode_param(source, "v.tok_embd_sep", config.n_embd_text, n_sep_rows)?;

        Ok(Self {
            config,
            patch_embd_w,
            patch_embd_b,
            position_embd,
            n_positions,
            layers,
            post_ln_w,
            post_ln_b,
            merger,
            projector,
            tok_embd_sep,
            n_sep_rows,
        })
    }
}

/// Build a [`Weight`] from an already-decoded row-major `[n_out][n_in]` matrix.
///
/// `QuantizedTensor::F32` owns its data and the F32 kernel copies it again, so
/// the result carries no borrow of the temporary.
pub fn linear_from_rows(rows: Vec<f32>, n_in: usize, n_out: usize) -> Linear {
    debug_assert_eq!(rows.len(), n_in * n_out);
    Weight::from_quantized(QuantizedTensor::F32 {
        data: rows,
        n_in,
        n_out,
    })
}

fn tensor_dims(source: &dyn TensorSource, name: &str) -> Result<Vec<usize>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    Ok(info.dims.iter().map(|&v| v as usize).collect())
}

#[cfg(test)]
mod decode_matrix_tests {
    use super::decode_matrix;
    use crate::core::tensor::GGMLType;

    /// A Q8_0 block is 32 values: an F16 scale then 32 i8.
    const QK8_0: usize = 32;

    fn q8_0_block(scale: f32, values: &[i8; QK8_0]) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + QK8_0);
        out.extend_from_slice(&crate::ops::float::f32_to_f16(scale).to_le_bytes());
        out.extend(values.iter().map(|&q| q as u8));
        out
    }

    /// The regression: `decode_matrix` used to refuse every quantized type with
    /// "unsupported ggml type ... for a projector parameter", so a Q8_0 / Q4_K /
    /// Q6_K mmproj would not load even though `ops::quant::dequantize_tensor`
    /// covers exactly those three. It now delegates.
    #[test]
    fn quantized_projector_parameters_decode() {
        let mut data = q8_0_block(0.5, &[1; QK8_0]);
        data.extend(q8_0_block(0.25, &[2; QK8_0]));

        let got = decode_matrix(&data, GGMLType::Q8_0, QK8_0, 2).expect("Q8_0 projector row");
        assert_eq!(got.len(), 2 * QK8_0, "row-major [n_out][n_in]");
        for &v in &got[..QK8_0] {
            assert_eq!(v, 0.5, "row 0 = d * q");
        }
        for &v in &got[QK8_0..] {
            assert_eq!(v, 0.5, "row 1 = 0.25 * 2");
        }
    }

    /// The oracle pins the projector in F32, so widening the decode must not
    /// disturb the three types it already handled.
    #[test]
    fn f16_and_f32_projector_parameters_are_unchanged() {
        let f32_values: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let mut f32_bytes = Vec::new();
        for v in &f32_values {
            f32_bytes.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(
            decode_matrix(&f32_bytes, GGMLType::F32, 4, 2).unwrap(),
            f32_values
        );

        let mut f16_bytes = Vec::new();
        for v in &f32_values {
            f16_bytes.extend_from_slice(&crate::ops::float::f32_to_f16(*v).to_le_bytes());
        }
        let decoded = decode_matrix(&f16_bytes, GGMLType::F16, 4, 2).unwrap();
        for (got, want) in decoded.iter().zip(&f32_values) {
            assert_eq!(
                *got,
                crate::ops::float::f16_to_f32(crate::ops::float::f32_to_f16(*want))
            );
        }
    }

    /// A type the shared decoder does not cover still reports cleanly instead of
    /// panicking, and the message names the type.
    #[test]
    fn unsupported_type_still_errors_with_the_type_named() {
        let err = decode_matrix(&[0u8; 4], GGMLType::I32, 1, 1).unwrap_err();
        assert!(
            err.contains("I32"),
            "the error must name the offending type, got: {err}"
        );
    }
}
