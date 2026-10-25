//! Native F32 inference for the published Timestral-209M checkpoint.
mod forecast;
pub(crate) mod graph;
pub(crate) mod network;

use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use crate::ops::kernel::{QuantizedTensor, Weight};
pub use forecast::{Forecast, ForecastRequest, Series};
use rayon::prelude::*;
use std::collections::HashMap;

const CONFIG_JSON: &str = include_str!("config.json");
pub(crate) const D: usize = 768;
pub(crate) const HEADS: usize = 12;
pub(crate) const HD: usize = D / HEADS;
pub(crate) const F: usize = 2048;
pub(crate) const PS: usize = 32;
pub(crate) const Q: usize = 21;
pub(crate) const MEDIAN: usize = 10;
pub(crate) const LEVELS: [f32; Q] = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
    0.85, 0.9, 0.95, 0.99,
];

pub struct TimestralModel<'a> {
    /// Two-dimensional matmul weights.
    ///
    /// The tensors are block-quantized in some checkpoints, so the rows stay
    /// in their stored layout and the kernel picks the dot product. F32
    /// checkpoints keep the plain row-major slice.
    matrices: HashMap<String, Matrix<'a>>,
    /// One-dimensional tensors (norms, biases, the forecast query).
    ///
    /// These are read once per model load, so expanding them to F32 here
    /// costs nothing and keeps the norms exact.
    vectors: HashMap<String, Vec<f32>>,
    compute: crate::compute::ComputePolicy,
    #[cfg(feature = "vulkan")]
    gpu: std::sync::Mutex<Option<crate::vulkan::timestral::Transformer>>,
}

/// A matmul weight.
///
/// The F32 form borrows the mapped file, so a full-precision checkpoint
/// costs no extra memory. A block-quantized checkpoint keeps its rows in
/// the stored layout and hands them to the matching kernel.
pub(crate) enum Matrix<'a> {
    F32(&'a [f32]),
    Quantized(Weight<'a>),
}

impl<'a> Matrix<'a> {
    /// The stored format.
    pub(crate) fn ggml_type(&self) -> GGMLType {
        match self {
            Matrix::F32(_) => GGMLType::F32,
            Matrix::Quantized(weight) => weight.ggml_type,
        }
    }

    /// The F32 rows, when the weight is stored as F32.
    pub(crate) fn f32_slice(&self) -> Option<&'a [f32]> {
        match self {
            Matrix::F32(rows) => Some(rows),
            Matrix::Quantized(_) => None,
        }
    }

    /// The stored bytes, for a device upload.
    pub(crate) fn weight_bytes(&self) -> Option<&[u8]> {
        match self {
            Matrix::F32(_) => None,
            Matrix::Quantized(weight) => weight.kernel.weight_bytes(),
        }
    }

    /// The quantizing kernel, when the weight is block-quantized.
    fn kernel(&self) -> Option<&dyn crate::ops::kernel::Kernel> {
        match self {
            Matrix::F32(_) => None,
            Matrix::Quantized(weight) => Some(weight.kernel.as_ref()),
        }
    }
}

/// One tensor the checkpoint must provide.
enum Contract {
    /// A `[n_in, n_out]` matmul weight.
    Matrix { n_in: usize, n_out: usize },
    /// A one-dimensional tensor of `len` elements.
    Vector { len: usize },
}

impl Contract {
    /// The number of elements the tensor holds.
    fn elements(&self) -> u64 {
        match self {
            Contract::Matrix { n_in, n_out } => (*n_in as u64) * (*n_out as u64),
            Contract::Vector { len } => *len as u64,
        }
    }
}

/// Every tensor the published checkpoint provides, with the shape the
/// loader requires.
///
/// The shapes are the same for every quantization format: GGUF stores a
/// quantized tensor with its element count, not its block count, so only
/// the block alignment of `n_in` changes with the format.
fn tensor_contracts() -> Vec<(String, Contract)> {
    let mut shapes = vec![
        (
            "representation.forecast_query".into(),
            Contract::Vector { len: D },
        ),
        (
            "transformer.out_norm.scale".into(),
            Contract::Vector { len: D },
        ),
    ];
    for (prefix, inp, out, bias) in [
        ("representation.patch_proj", 2 * PS, D, true),
        ("head.projection", D, PS * Q, false),
    ] {
        for (name, n_in, n_out) in [
            ("hidden_layer", inp, 4 * D),
            ("output_layer", 4 * D, out),
            ("residual_layer", inp, out),
        ] {
            shapes.push((
                format!("{prefix}.{name}.weight"),
                Contract::Matrix { n_in, n_out },
            ));
            if bias {
                shapes.push((
                    format!("{prefix}.{name}.bias"),
                    Contract::Vector { len: n_out },
                ));
            }
        }
    }
    for i in 0..24 {
        let p = format!("transformer.layers.{i}");
        shapes.push((format!("{p}.mlp_norm.scale"), Contract::Vector { len: D }));
        shapes.push((
            format!("{p}.mlp.w12.weight"),
            Contract::Matrix {
                n_in: D,
                n_out: 2 * F,
            },
        ));
        shapes.push((
            format!("{p}.mlp.w3.weight"),
            Contract::Matrix { n_in: F, n_out: D },
        ));
        for j in 0..if i % 4 == 3 { 2 } else { 1 } {
            shapes.push((
                format!("{p}.attn_norms.{j}.scale"),
                Contract::Vector { len: D },
            ));
            for (name, n_out) in [("wQKV", 3 * D), ("wO", D), ("_gate", D)] {
                shapes.push((
                    format!("{p}.attns.{j}.{name}.weight"),
                    Contract::Matrix { n_in: D, n_out },
                ));
            }
        }
    }
    shapes
}

/// The activation layout a block-quantized weight expects.
///
/// The activation is quantized once for the whole row block, so the
/// per-row work is only the dot product. Which format is needed follows
/// from the weight: the Q8_0 family reads Q8_0 activations, the K-quant
/// family reads Q8_K activations, and the float formats read the F32
/// activation directly.
struct QuantizedActivation {
    q8: Vec<u8>,
    scales: Vec<f32>,
    q8k: Vec<crate::ops::quant::BlockQ8K>,
    n_in: usize,
    blocks: usize,
    q8k_blocks: usize,
}

impl QuantizedActivation {
    /// Quantize `rows` rows of `n_in` values for a weight of `format`.
    fn prepare(format: GGMLType, input: &[f32], rows: usize, n_in: usize) -> Self {
        let blocks = n_in.div_ceil(32);
        let q8k_blocks = n_in / crate::ops::quant::QK_K;
        let mut activation = Self {
            q8: Vec::new(),
            scales: Vec::new(),
            q8k: Vec::new(),
            n_in,
            blocks,
            q8k_blocks,
        };
        match format {
            GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 => {}
            GGMLType::Q4_0 | GGMLType::Q4_1 | GGMLType::Q8_0 | GGMLType::Q1_0 => {
                activation.q8.resize(rows * n_in, 0);
                activation.scales.resize(rows * blocks, 0.0);
                for row in 0..rows {
                    crate::ops::quantize_q8_0_into(
                        &input[row * n_in..(row + 1) * n_in],
                        n_in,
                        &mut activation.q8[row * n_in..(row + 1) * n_in],
                        &mut activation.scales[row * blocks..(row + 1) * blocks],
                    );
                }
            }
            GGMLType::Q2K
            | GGMLType::Q3K
            | GGMLType::Q4K
            | GGMLType::Q5K
            | GGMLType::Q6K
            | GGMLType::IQ1_S
            | GGMLType::IQ1_M
            | GGMLType::IQ2_XXS
            | GGMLType::IQ2_XS
            | GGMLType::IQ2_S
            | GGMLType::IQ3_XXS
            | GGMLType::IQ3_S
            | GGMLType::IQ4_NL
            | GGMLType::IQ4_XS => {
                activation.q8k.resize(
                    rows * q8k_blocks,
                    crate::ops::quant::BlockQ8K {
                        d: 0.0,
                        qs: [0; crate::ops::quant::QK_K],
                        bsums: [0; crate::ops::quant::QK_K / 16],
                    },
                );
                for row in 0..rows {
                    crate::ops::quantize_row_q8_k_into(
                        &input[row * n_in..(row + 1) * n_in],
                        &mut activation.q8k[row * q8k_blocks..(row + 1) * q8k_blocks],
                    );
                }
            }
            other => panic!("prepared matmul does not support {other:?}"),
        }
        activation
    }

    /// The Q8_0 bytes of one row, empty when the weight does not want them.
    fn q8_row(&self, row: usize) -> &[u8] {
        if self.q8.is_empty() {
            &[]
        } else {
            &self.q8[row * self.n_in..(row + 1) * self.n_in]
        }
    }

    /// The Q8_0 scales of one row.
    fn scales_row(&self, row: usize) -> &[f32] {
        if self.scales.is_empty() {
            &[]
        } else {
            &self.scales[row * self.blocks..(row + 1) * self.blocks]
        }
    }

    /// The Q8_K blocks of one row, absent when the weight does not want them.
    fn q8k_row(&self, row: usize) -> Option<&[crate::ops::quant::BlockQ8K]> {
        if self.q8k.is_empty() {
            None
        } else {
            Some(&self.q8k[row * self.q8k_blocks..(row + 1) * self.q8k_blocks])
        }
    }
}

/// Matrix formats the Timestral runtime can execute.
const MATRIX_TYPES: &[GGMLType] = &[
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

/// Load a matmul weight, keeping a block-quantized tensor in its stored
/// layout.
fn load_matrix<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Matrix<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor {name}"))?;
    if info.dims != [n_in as u64, n_out as u64] {
        return Err(format!(
            "Invalid tensor {name}: shape {:?}; expected [{n_in}, {n_out}]",
            info.dims
        ));
    }
    if !MATRIX_TYPES.contains(&info.ggml_type) {
        return Err(format!(
            "Unsupported Timestral matrix format {name}: {:?}",
            info.ggml_type
        ));
    }
    let (block, block_bytes) = info.ggml_type.type_traits();
    if !n_in.is_multiple_of(block) {
        return Err(format!(
            "Tensor {name}: width {n_in} is not a multiple of the {block}-value block"
        ));
    }
    let expected = n_in
        .checked_div(block)
        .and_then(|blocks| blocks.checked_mul(n_out))
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| format!("Tensor size overflow {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data {name}: {} bytes; expected {expected}",
            bytes.len()
        ));
    }
    if info.ggml_type == GGMLType::F32 {
        // The mapped file already holds the rows; borrow them.
        let values: &[f32] = bytemuck::try_cast_slice(bytes)
            .map_err(|error| format!("Invalid tensor {name}: {error}"))?;
        return Ok(Matrix::F32(values));
    }
    Ok(Matrix::Quantized(Weight::from_quantized(
        QuantizedTensor::from_bytes(bytes, info.ggml_type, n_in, n_out),
    )))
}

/// Load a one-dimensional tensor, expanding it to F32 once.
fn load_vector(source: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    crate::core::tensor::load_f32_tensor(source, name, &[len as u64])
        .map_err(|error| format!("{name}: {error}"))
}

impl<'a> TimestralModel<'a> {
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        Self::from_source_with_compute(source, crate::compute::ComputePolicy::legacy())
    }

    pub fn from_source_with_compute(
        source: &'a dyn TensorSource,
        compute: crate::compute::ComputePolicy,
    ) -> Result<Self, String> {
        compute.check_build()?;
        let arch = source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val);
        if arch != Some("timestral") {
            return Err("Expected timestral architecture".into());
        }
        let raw = source
            .metadata("timestral.config")
            .and_then(MetaValue::to_string_val)
            .ok_or("Missing timestral.config")?;
        let config: serde_json::Value = serde_json::from_str(raw).map_err(|e| e.to_string())?;
        if config != serde_json::from_str::<serde_json::Value>(CONFIG_JSON).unwrap() {
            return Err("Only the published Timestral-209M configuration is supported".into());
        }
        let mut matrices = HashMap::new();
        let mut vectors = HashMap::new();
        for (name, contract) in tensor_contracts() {
            match contract {
                Contract::Matrix { n_in, n_out } => {
                    matrices.insert(name.clone(), load_matrix(source, &name, n_in, n_out)?);
                }
                Contract::Vector { len } => {
                    vectors.insert(name.clone(), load_vector(source, &name, len)?);
                }
            }
        }
        Ok(Self {
            matrices,
            vectors,
            compute,
            #[cfg(feature = "vulkan")]
            gpu: std::sync::Mutex::new(None),
        })
    }

    /// Look up a one-dimensional tensor.
    pub(crate) fn weight(&self, name: &str) -> &[f32] {
        self.vectors
            .get(name)
            .map(|values| values.as_slice())
            .unwrap_or_else(|| panic!("missing Timestral tensor {name}"))
    }

    /// The weight formats the checkpoint holds.
    ///
    /// The device path reads this before it starts, so a checkpoint it
    /// cannot run is reported at load time instead of on the first
    /// request.
    pub(crate) fn matrix_types(&self) -> Vec<GGMLType> {
        let mut types: Vec<GGMLType> = self.matrices.values().map(Matrix::ggml_type).collect();
        types.sort_by_key(|value| format!("{value:?}"));
        types.dedup();
        types
    }

    /// Look up a matmul weight.
    pub(crate) fn matrix(&self, name: &str) -> &Matrix<'a> {
        self.matrices
            .get(name)
            .unwrap_or_else(|| panic!("missing Timestral matrix {name}"))
    }

    /// `weight @ input` for a fully qualified weight name.
    pub(crate) fn linear_named(
        &self,
        name: &str,
        input: &[f32],
        n_in: usize,
        n_out: usize,
    ) -> Vec<f32> {
        let w = self.matrix(name);
        // `name` is the matrix tensor; its bias is the sibling tensor.
        let bias = self.vectors.get(&format!(
            "{}.bias",
            name.strip_suffix(".weight").unwrap_or(name)
        ));
        let n_rows = input.len() / n_in;
        let mut output = vec![0.; n_rows * n_out];
        self.matmul_rows(w, input, &mut output, n_in, n_out);
        if let Some(bias) = bias {
            // The bias belongs to every output row.
            for values in output.chunks_mut(n_out) {
                for (value, bias) in values.iter_mut().zip(bias.iter().copied()) {
                    *value += bias;
                }
            }
        }
        output
    }

    fn linear(&self, prefix: &str, input: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
        self.linear_named(&format!("{prefix}.weight"), input, n_in, n_out)
    }

    /// `weight @ input` for one row block.
    ///
    /// F32 weights keep the SIMD row selector. A block-quantized weight
    /// needs the activation in the format its kernel expects, so the
    /// activation is quantized once for the whole block and the rows are
    /// then evaluated in parallel against it.
    fn matmul_rows(
        &self,
        weight: &Matrix<'_>,
        input: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) {
        if let Some(rows) = weight.f32_slice() {
            // Rows are independent, so the rayon pool splits them; every
            // worker keeps the whole weight matrix hot.
            output
                .par_chunks_mut(n_out)
                .zip(input.par_chunks(n_in))
                .for_each(|(out, inp)| {
                    crate::ops::kernel::f32::forward_f32_rows_dispatch(
                        rows, inp, out, n_in, n_out, 0, 1,
                    );
                });
            return;
        }
        let rows = input.len() / n_in;
        let kernel = weight
            .kernel()
            .expect("a non-F32 matrix always carries a kernel");
        let activation = QuantizedActivation::prepare(weight.ggml_type(), input, rows, n_in);
        output
            .par_chunks_mut(n_out)
            .zip(input.par_chunks(n_in))
            .enumerate()
            .for_each(|(row, (out, inp))| {
                kernel.forward_prepared(
                    inp,
                    activation.q8_row(row),
                    activation.scales_row(row),
                    activation.q8k_row(row),
                    out,
                    n_in,
                    n_out,
                    0,
                    1,
                );
            });
    }

    fn norm(&self, prefix: &str, input: &[f32]) -> Vec<f32> {
        let w = &self.vectors[&format!("{prefix}.scale")];
        let mut out = vec![0.; input.len()];
        out.par_chunks_mut(D)
            .zip(input.par_chunks(D))
            .for_each(|(out, inp)| {
                crate::ops::rms_norm(inp, w, out, 1e-6);
            });
        out
    }

    fn residual_block(&self, prefix: &str, input: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
        let mut hidden = self.linear(&format!("{prefix}.hidden_layer"), input, n_in, 4 * D);
        crate::ops::silu_inplace(&mut hidden);
        let mut out = self.linear(&format!("{prefix}.output_layer"), &hidden, 4 * D, n_out);
        let residual = self.linear(&format!("{prefix}.residual_layer"), input, n_in, n_out);
        add(&mut out, &residual);
        out
    }
}

fn add(out: &mut [f32], rhs: &[f32]) {
    for (a, b) in out.iter_mut().zip(rhs) {
        *a += b;
    }
}

pub(crate) fn trace(name: &str, shape: &[usize], values: &[f32]) -> Result<(), String> {
    #[cfg(feature = "parity-trace")]
    if std::env::var_os("RMI_PARITY_TRACE").is_some() {
        crate::parity_trace::checkpoint(&format!("timestral.{name}"), None, shape, values)
            .map_err(|e| e.to_string())?;
    }
    let _ = (name, shape, values);
    Ok(())
}

#[cfg(test)]
mod tests;
