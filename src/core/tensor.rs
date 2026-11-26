//! Tensor primitives: GGMLType enum, GGUF metadata value types, TensorInfo,
//! and the [`TensorSource`] trait that abstracts byte-level access to tensor
//! data regardless of physical storage (mmap, `.ggufrs`, in-memory, etc.).
//!
//! This module is the foundation of the `core` layer. It deliberately knows
//! nothing about GGUF file parsing (that lives in [`crate::core::loader`]) or
//! model architecture (that lives in `crate::models`).

/// GGUF magic bytes (`"GGUF"`) used to validate file headers.
pub(crate) const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Default GGUF tensor alignment when `general.alignment` is absent.
pub(crate) const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// Q8_K quantization block (256 elements per block).
///
/// This is a pure data structure — the quantization/dequantization operations
/// live in `ops::quant`. It is defined here in `core` because `scratchpad`
/// (a core-layer type) needs to pre-allocate Q8_K buffers for K-quant kernels.
#[derive(Clone)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; 256],
    pub bsums: [i16; 16],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    /// BitNet 1.58-bit ternary weight quantization (`QK_I2_S = 128`
    /// in the upstream `bitnet.cpp` reference). 128-element blocks,
    /// 2-bit signed symmetric ternary {-1, 0, +1} (no in-block scale —
    /// per-row scale is recovered from the `*_norm_in` RMSNorm that
    /// precedes the projection in the BitLinear forward). 32 bytes
    /// per block = 128 × 2 bits / 8. Currently recognized by the GGUF
    /// loader only for tensor listing / shape inspection — no dequant
    /// kernel is implemented in this repo, so any model that ships
    /// i2_s weights (Microsoft's `bitnet-embedding-0.6b` /
    /// `bitnet-embedding-270m` BitNet 1.58 GGUF conversions)
    /// cannot forward through the existing engine trunk yet.
    I2_S = 36,
    Q1_0 = 41,
    // Fork-specific NariQ v1 types. Blocks span 16 rows; a row is not independently coded.
    Trl1 = 44,
    Trl2 = 45,
    Trl3 = 46,
    Trl4 = 47,
    Trl5 = 48,
    Trl6 = 49,
}

impl GGMLType {
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::IQ2_XXS),
            17 => Some(Self::IQ2_XS),
            18 => Some(Self::IQ3_XXS),
            19 => Some(Self::IQ1_S),
            20 => Some(Self::IQ4_NL),
            21 => Some(Self::IQ3_S),
            22 => Some(Self::IQ2_S),
            23 => Some(Self::IQ4_XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1_M),
            30 => Some(Self::BF16),
            36 => Some(Self::I2_S),
            41 => Some(Self::Q1_0),
            44 => Some(Self::Trl1),
            45 => Some(Self::Trl2),
            46 => Some(Self::Trl3),
            47 => Some(Self::Trl4),
            48 => Some(Self::Trl5),
            49 => Some(Self::Trl6),
            _ => None,
        }
    }

    pub(crate) fn trellis_bits(self) -> Option<usize> {
        match self {
            Self::Trl1 | Self::Trl2 | Self::Trl3 | Self::Trl4 | Self::Trl5 | Self::Trl6 => {
                Some((self as usize) - Self::Trl1 as usize + 1)
            }
            _ => None,
        }
    }

    pub fn type_traits(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
            Self::IQ2_XXS => (256, 66),
            Self::IQ2_XS => (256, 74),
            Self::IQ2_S => (256, 82),
            Self::IQ3_XXS => (256, 98),
            Self::IQ3_S => (256, 110),
            Self::IQ1_S => (256, 50),
            Self::IQ1_M => (256, 56),
            Self::IQ4_NL => (32, 18),
            Self::IQ4_XS => (256, 136),
            Self::I8 => (1, 1),
            Self::I16 => (1, 2),
            Self::I32 => (1, 4),
            Self::I64 => (1, 8),
            Self::F64 => (1, 8),
            Self::BF16 => (1, 2),
            Self::I2_S => (128, 32),
            Self::Q1_0 => (128, 18),
            Self::Trl1 => (16, 2),
            Self::Trl2 => (16, 4),
            Self::Trl3 => (16, 6),
            Self::Trl4 => (16, 8),
            Self::Trl5 => (16, 10),
            Self::Trl6 => (16, 12),
        }
    }

    pub fn nbytes(self, n_elements: usize) -> usize {
        let (block_size, type_size) = self.type_traits();
        let n_blocks = (n_elements + block_size - 1) / block_size;
        n_blocks * type_size
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(i32)]
pub enum MetaValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl MetaValueType {
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Array(MetaValueType, Vec<MetaValue>),
}

impl MetaValue {
    pub fn to_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(v) => Some(*v as u64),
            Self::Int8(v) => Some(*v as u64),
            Self::Uint16(v) => Some(*v as u64),
            Self::Int16(v) => Some(*v as u64),
            Self::Uint32(v) => Some(*v as u64),
            Self::Int32(v) => Some(*v as u64),
            Self::Uint64(v) => Some(*v),
            Self::Int64(v) => Some(*v as u64),
            Self::Float32(v) => Some(*v as u64),
            Self::Float64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn to_f64(&self) -> Option<f64> {
        match self {
            Self::Float32(v) => Some(*v as f64),
            Self::Float64(v) => Some(*v),
            Self::Uint32(v) => Some(*v as f64),
            Self::Int32(v) => Some(*v as f64),
            Self::Uint64(v) => Some(*v as f64),
            Self::Int64(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn to_string_val(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn to_arr(&self) -> Option<&Vec<MetaValue>> {
        match self {
            Self::Array(_, v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GGMLType,
    pub offset: u64,
}

impl TensorInfo {
    pub fn checked_n_elements(&self) -> Option<u64> {
        let (first, rest) = self.dims.split_first()?;
        rest.iter()
            .try_fold(*first, |count, dimension| count.checked_mul(*dimension))
    }

    pub fn checked_nbytes(&self) -> Option<u64> {
        let row_elements = *self.dims.first()?;
        let rows = self.dims[1..]
            .iter()
            .try_fold(1u64, |count, dimension| count.checked_mul(*dimension))?;
        let (block_elements, block_bytes) = self.ggml_type.type_traits();
        let block_elements = block_elements as u64;
        if row_elements % block_elements != 0 {
            return None;
        }
        row_elements
            .checked_div(block_elements)?
            .checked_mul(block_bytes as u64)?
            .checked_mul(rows)
    }

    pub fn n_elements(&self) -> usize {
        usize::try_from(self.checked_n_elements().expect("validated tensor shape"))
            .expect("validated tensor element count fits usize")
    }

    pub fn nbytes(&self) -> usize {
        usize::try_from(self.checked_nbytes().expect("validated tensor byte size"))
            .expect("validated tensor byte size fits usize")
    }
}

/// Abstraction over a tensor byte store. Implementors may back onto mmap,
/// an in-memory map, a remote service, or any other source of bytes.
///
/// The trait is object-safe (no associated types, no generic methods) so
/// that callers can store `Box<dyn TensorSource>` in heterogeneous
/// containers. Per-architecture metadata queries should layer on top via
/// free functions (see [`crate::core::loader::model_config_from_source`]).
pub trait TensorSource: Send + Sync {
    fn metadata(&self, key: &str) -> Option<&MetaValue>;
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo>;
    fn tensor_slice(&self, name: &str) -> Option<&[u8]>;

    fn model_config(&self) -> Result<crate::core::traits::ModelConfig, String> {
        crate::core::loader::model_config_from_source(self)
    }
}

/// Check a tensor's metadata: it exists, has the expected shape, and its type
/// is one of the given GGML types.
///
/// This is the half of [`require_tensor_type`] that only reads the header, so
/// a loader that validates its whole contract before it touches any payload
/// can use it. The type list is the caller's: which formats a model can
/// execute is a property of the model, not of the format. Returns the
/// tensor's type so the caller can dispatch on it.
///
/// Models should call this instead of reimplementing the check. See
/// `docs/MODEL_ORGANIZATION.md` §6 for the rationale.
pub fn require_tensor_meta<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    dims: &[u64],
    types: &[GGMLType],
) -> Result<GGMLType, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims || !types.contains(&info.ggml_type) {
        return Err(if types.len() == 1 {
            format!(
                "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} {:?}",
                info.dims, info.ggml_type, dims, types[0]
            )
        } else {
            format!(
                "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} one of {types:?}",
                info.dims, info.ggml_type, dims
            )
        });
    }
    Ok(info.ggml_type)
}

/// Check that a tensor's payload is exactly as long as its type's block
/// layout implies.
///
/// The payload length is the part that is easy to get wrong: a block format
/// stores a fixed number of bytes per block, so the length follows
/// `GGMLType::type_traits` and not the element count. A model that accepts a
/// quantized format has to measure it that way — counting two bytes per
/// element, the way a BF16-only loader does, would accept a payload the
/// decoder reads past the end of.
pub fn require_tensor_data<S: TensorSource + ?Sized>(source: &S, name: &str) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?,
    )
    .map_err(|_| format!("Tensor byte size does not fit usize: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    Ok(())
}

/// Check a tensor's metadata and its payload, and return its type.
///
/// This is [`require_tensor_meta`] followed by [`require_tensor_data`], which
/// is what a loader wants when it validates and reads in one step.
pub fn require_tensor_type<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    dims: &[u64],
    types: &[GGMLType],
) -> Result<GGMLType, String> {
    let ggml_type = require_tensor_meta(source, name, dims, types)?;
    require_tensor_data(source, name)?;
    Ok(ggml_type)
}

/// Borrow a tensor's raw bytes after checking its shape and its type.
///
/// The shape, the storage type and the payload length all have to match
/// before the bytes are read: a loader that skips any of them ends up
/// decoding a payload it does not own. `dims` must be a non-empty shape
/// without a zero, because a zero-length axis has no bytes to read.
///
/// This is the canonical helper for "validate a tensor and borrow it in one
/// step". Models should call this instead of reimplementing the check. See
/// `docs/MODEL_ORGANIZATION.md` §6 for the rationale.
pub fn typed_tensor_bytes<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    dims: &[u64],
    kind: GGMLType,
) -> Result<&'a [u8], String> {
    if dims.is_empty() || dims.contains(&0) {
        return Err(format!(
            "Invalid tensor {name}: expected a non-zero shape, got {dims:?}"
        ));
    }
    require_tensor_type(source, name, dims, &[kind])?;
    Ok(source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?)
}

/// Borrow a tensor's raw bytes, checking its shape and that it is F32.
///
/// Some loaders keep the bytes and decode them later — a vision tower that
/// hands the same bias to several matmuls, for instance — so
/// [`load_f32_tensor`] cannot be used: it decodes eagerly and does not
/// accept BF16. Such a loader has to know the storage type up front,
/// because it decodes four bytes per element.
pub fn f32_tensor_bytes<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    dims: &[u64],
) -> Result<&'a [u8], String> {
    typed_tensor_bytes(source, name, dims, GGMLType::F32)
}

/// Borrow an optional tensor's raw bytes, rejecting storage that is not F32.
///
/// This is the optional counterpart of [`f32_tensor_bytes`]: a missing
/// tensor is `Ok(None)`, because the caller decides whether the tensor is
/// optional. A present tensor of the wrong type is an error, because
/// decoding it as F32 would read it at half length and let the caller walk
/// off the end of the buffer.
///
/// See `docs/MODEL_ORGANIZATION.md` §6 for the rationale on keeping this
/// next to the other tensor helpers instead of in each model.
pub fn optional_f32_tensor_bytes<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
) -> Result<Option<&'a [u8]>, String> {
    let Some(info) = source.tensor_info(name) else {
        return Ok(None);
    };
    if info.ggml_type != GGMLType::F32 {
        return Err(format!(
            "{name} must be F32, found {:?}: this tensor decodes four bytes per element",
            info.ggml_type
        ));
    }
    Ok(source.tensor_slice(name))
}

/// Load a 1-D tensor as `Vec<f32>`, accepting both F32 and BF16 storage.
///
/// This is the canonical helper for "norm weights" which ship as either F32
/// or BF16 across GGUF model variants. Models should call this instead of
/// reimplementing the byte-to-f32 conversion. See
/// `docs/MODEL_ORGANIZATION.md` §6 for the rationale.
pub fn load_f32_tensor<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    dims: &[u64],
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims || !DECODABLE_TO_F32.contains(&info.ggml_type) {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} in {:?}",
            info.dims, info.ggml_type, dims, DECODABLE_TO_F32
        ));
    }
    decode_tensor_to_f32(source, name, &info)
}

/// Load a tensor as `Vec<f32>` by element count instead of by shape.
///
/// The same tensor ships with different shapes across model variants: the
/// Flux2 VAE that ERNIE-Image uses keeps its attention projections as
/// two-dimensional matrices while the Flux1 one keeps them flat. The payload
/// and the element order are the same in both, so the element count is the
/// contract that matters, and the shape only selects the decoder for the
/// block-quantized layouts.
pub fn load_f32_tensor_flat<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    elements: usize,
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if !DECODABLE_TO_F32.contains(&info.ggml_type) {
        return Err(format!(
            "Invalid tensor {name}: type {:?}; expected one of {:?}",
            info.ggml_type, DECODABLE_TO_F32
        ));
    }
    let count = info
        .dims
        .iter()
        .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| format!("Invalid tensor dims for {name}"))?;
    if count != elements as u64 {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} holds {count} elements; expected {elements}",
            info.dims
        ));
    }
    decode_tensor_to_f32(source, name, &info)
}

/// Decode a validated tensor's payload into F32.
fn decode_tensor_to_f32<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    info: &crate::core::tensor::TensorInfo,
) -> Result<Vec<f32>, String> {
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?,
    )
    .map_err(|_| format!("Tensor byte size does not fit usize: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    if matches!(info.ggml_type, GGMLType::F32 | GGMLType::BF16) {
        return Ok(match info.ggml_type {
            GGMLType::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            _ => bytes
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        });
    }
    // Block-quantized layouts are expanded once, here. These are the small
    // tables a loader reads directly — the record head's `instance_embed`,
    // the norms and biases — which run once per model load rather than per
    // token, so the cost is amortized away. The big projections never come
    // through this path; they stay quantized and are consumed by
    // `QuantizedTensor::from_bytes`.
    let n_cols = *info
        .dims
        .first()
        .ok_or_else(|| format!("{name}: no dims"))? as usize;
    let n_rows = info.dims[1..]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
        .ok_or_else(|| format!("{name}: dims overflow"))?;
    crate::ops::quant::dequantize_tensor(bytes, info.ggml_type, n_cols, n_rows)
        .map_err(|e| format!("{name}: {e}"))
}

/// Load an optional tensor's values as F32.
///
/// The format comes from the tensor's own metadata, not from its byte
/// length. F16 and BF16 are both two bytes per element, and they decode to
/// different values, so a length cannot tell them apart — a checkpoint that
/// stores a tensor in BF16 has to be read as BF16, and the only place that
/// knows which it is, is the header.
///
/// A missing tensor is `Ok(None)`: the caller decides whether the tensor is
/// optional. A present tensor in a format that cannot be decoded is an
/// error, because reading it as something else would produce numbers that
/// look fine and are wrong.
pub fn optional_f32_values<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
) -> Result<Option<Vec<f32>>, String> {
    let Some(info) = source.tensor_info(name) else {
        return Ok(None);
    };
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?,
    )
    .map_err(|_| format!("Tensor byte size does not fit usize: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    Ok(Some(match info.ggml_type {
        GGMLType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        GGMLType::BF16 => bytes
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        GGMLType::F16 => bytes
            .chunks_exact(2)
            .map(|c| crate::ops::float::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        other => return Err(format!("{name} must be F32, BF16 or F16, found {other:?}")),
    }))
}

/// Load a matrix as F32, using the format the tensor declares.
///
/// This is the matrix counterpart of [`optional_f32_values`]. It also
/// accepts the block-quantized layout, because the vision towers ship
/// Q8_0 matrices as well.
///
/// The reason this exists is the same one: the format has to come from the
/// header. Guessing it from the byte length cannot tell BF16 from F16.
pub fn load_f32_matrix<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    dims: &[u64],
    n_in: usize,
    n_out: usize,
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor info: {name}"))?;
    let flattened = [n_in as u64, n_out as u64];
    if info.dims != dims && !(info.ggml_type == GGMLType::Q8_0 && info.dims == flattened) {
        return Err(format!(
            "Invalid tensor {name}: shape {:?}; expected {dims:?}",
            info.dims
        ));
    }
    if !matches!(
        info.ggml_type,
        GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 | GGMLType::Q8_0
    ) || (info.ggml_type == GGMLType::Q8_0 && n_in % 32 != 0)
    {
        return Err(format!("Unsupported matrix {name}: {:?}", info.ggml_type));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    decode_f32_matrix(info.ggml_type, bytes, n_in, n_out)
        .map_err(|error| format!("Invalid tensor {name}: {error}"))
}

/// Decode borrowed matrix bytes into F32, using the format the header declares.
///
/// This is the decode half of [`load_f32_matrix`], for callers that already
/// hold the bytes — the vision towers borrow them straight from the mapped
/// file, so they never get the name back — but still need the format that
/// the header declared.
///
/// The format cannot be guessed from the byte length: F16 and BF16 are both
/// two bytes per element, and a BF16 `1.0` read as F16 comes back as `1.875`.
/// The block-quantized layout is accepted too, because the vision towers
/// ship Q8_0 matrices as well.
pub fn decode_f32_matrix(
    ggml_type: GGMLType,
    bytes: &[u8],
    n_in: usize,
    n_out: usize,
) -> Result<Vec<f32>, String> {
    let elements = n_in
        .checked_mul(n_out)
        .ok_or_else(|| "Tensor element count overflow".to_string())?;
    let expected = match ggml_type {
        GGMLType::F32 => elements.checked_mul(4),
        GGMLType::F16 | GGMLType::BF16 => elements.checked_mul(2),
        GGMLType::Q8_0 => {
            if n_in % 32 != 0 {
                return Err(format!(
                    "Q8_0 matrix needs a multiple of 32 columns, got {n_in}"
                ));
            }
            Some(elements / 32 * 34)
        }
        other => return Err(format!("Unsupported matrix format {other:?}")),
    }
    .ok_or_else(|| "Tensor byte size overflow".to_string())?;
    if bytes.len() != expected {
        return Err(format!(
            "byte length {} differs from expected {expected}",
            bytes.len()
        ));
    }
    Ok(match ggml_type {
        GGMLType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        GGMLType::F16 => bytes
            .chunks_exact(2)
            .map(|c| crate::ops::float::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        GGMLType::BF16 => bytes
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        _ => crate::ops::quant::dequant_q80_weight(bytes, n_in, n_out),
    })
}

/// Load a matrix weight, using the format the tensor declares.
///
/// This is the one entry point for turning a GGUF tensor into a
/// [`Weight`]: the format comes from the header, the accepted formats from
/// `allowed`, and the shape from the caller.
///
/// `allowed` is the caller's own list of formats its model can execute.
/// `None` means "every format a [`QuantizedTensor`] can hold" — which is
/// every format [`QuantizedTensor::supports`] accepts, and nothing else. A
/// GGML file may still declare a format no kernel can hold (the integer and
/// double types, `Q8_1`, `Q8_K`); those are reported as an error, never
/// handed to a kernel.
///
/// The shape is passed in rather than derived from the tensor's dimensions
/// because the repository has two conventions that are both correct: GGUF's
/// own (the first dimension is the row length) and the dots vocoder's
/// (everything but the last dimension is the input width, which is how its
/// convolution weights are stored). Forcing one of them on the other would
/// silently reshape those weights.
///
/// The byte length is checked against the declared format, so a truncated or
/// mismatched tensor is an error rather than a kernel reading past its data.
///
/// [`Weight`]: crate::ops::kernel::Weight
/// [`QuantizedTensor`]: crate::ops::kernel::QuantizedTensor
/// [`QuantizedTensor::supports`]: crate::ops::kernel::QuantizedTensor::supports
pub fn load_weight_any<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    n_in: usize,
    n_out: usize,
    allowed: Option<&[GGMLType]>,
) -> Result<crate::ops::kernel::Weight<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if let Some(allowed) = allowed {
        if !allowed.contains(&info.ggml_type) {
            return Err(format!(
                "Unsupported matrix {name}: {:?}, expected one of {allowed:?}",
                info.ggml_type
            ));
        }
    }
    // A format no kernel can hold is reported here. Without this check the
    // type reaches `QuantizedTensor::from_bytes`, which panics on it — a
    // panic on the path that reads a user's model file.
    if !crate::ops::kernel::QuantizedTensor::supports(info.ggml_type) {
        return Err(format!(
            "Unsupported matrix {name}: {:?} has no weight kernel",
            info.ggml_type
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    // The block-quantized formats lay their blocks along the row, so the
    // row length has to be a whole number of blocks.
    let (block_elements, block_bytes) = info.ggml_type.type_traits();
    if n_in % block_elements != 0 {
        return Err(format!(
            "Unsupported matrix {name}: {:?} needs a row of a multiple of {block_elements} elements",
            info.ggml_type
        ));
    }
    let expected = n_in
        .checked_mul(n_out)
        .and_then(|elements| elements.checked_div(block_elements))
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| format!("Tensor byte size overflow: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    // F32 keeps the borrowed bytes and the matrix shape. The block-quantized
    // formats keep their bytes too, but their tensor does not carry the
    // shape, so it is set below.
    let mut weight =
        if info.ggml_type == GGMLType::F32 {
            crate::ops::kernel::Weight::from_f32_bytes(bytes, n_in, n_out)
                .ok_or_else(|| format!("Invalid F32 matrix {name}"))?
        } else {
            crate::ops::kernel::Weight::from_quantized(
                crate::ops::kernel::QuantizedTensor::from_bytes(bytes, info.ggml_type, n_in, n_out),
            )
        };
    weight.n_in = n_in;
    weight.n_out = n_out;
    Ok(weight)
}

/// Tensor types `load_f32_tensor` will expand to F32 on load.
const DECODABLE_TO_F32: &[GGMLType] = &[
    GGMLType::F32,
    GGMLType::BF16,
    GGMLType::Q8_0,
    GGMLType::Q4K,
    GGMLType::Q6K,
];

/// Bit-level BF16 → F32 conversion (zero-extend the 16 high bits).
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_nbytes() {
        assert_eq!(GGMLType::F32.nbytes(256), 1024);
        assert_eq!(GGMLType::F16.nbytes(256), 512);
        assert_eq!(GGMLType::BF16.nbytes(256), 512);
        assert_eq!(GGMLType::Q4K.nbytes(256), 144);
        assert_eq!(GGMLType::Q4K.nbytes(512), 288);
        assert_eq!(GGMLType::Q4_0.nbytes(32), 18);
        assert_eq!(GGMLType::Q8_0.nbytes(32), 34);
    }

    #[test]
    fn k_quant_block_sizes_match_ggml() {
        assert_eq!(GGMLType::Q2K.nbytes(256), 84);
        assert_eq!(GGMLType::Q3K.nbytes(256), 110);
    }

    #[test]
    fn tensor_size_overflow_is_rejected() {
        let info = TensorInfo {
            name: "bad".into(),
            dims: vec![u64::MAX, 2],
            ggml_type: GGMLType::F32,
            offset: 0,
        };
        assert_eq!(info.checked_n_elements(), None);
        assert_eq!(info.checked_nbytes(), None);
    }

    /// A fake source for the shared tensor check.
    struct Source {
        info: TensorInfo,
        bytes: Vec<u8>,
    }
    impl TensorSource for Source {
        fn metadata(&self, _key: &str) -> Option<&MetaValue> {
            None
        }
        fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
            Some(&self.info)
        }
        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            Some(&self.bytes)
        }
    }

    fn source(ggml_type: GGMLType, dims: &[u64]) -> Source {
        let info = TensorInfo {
            name: "weight".into(),
            dims: dims.to_vec(),
            ggml_type,
            offset: 0,
        };
        let bytes = vec![0u8; info.checked_nbytes().unwrap_or(0) as usize];
        Source { info, bytes }
    }

    /// `Weight` is not `Debug`, so `unwrap_err` is not available.
    fn expect_error<T>(result: Result<T, String>) -> String {
        match result {
            Ok(_) => panic!("expected an error"),
            Err(error) => error,
        }
    }

    /// A source that hands out one tensor with a declared format and a
    /// payload the test chooses.
    struct Matrix {
        info: TensorInfo,
        bytes: Vec<u8>,
    }

    impl TensorSource for Matrix {
        fn metadata(&self, _key: &str) -> Option<&MetaValue> {
            None
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            (name == self.info.name).then_some(&self.info)
        }

        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            (name == self.info.name).then_some(self.bytes.as_slice())
        }
    }

    fn matrix(ggml_type: GGMLType, dims: &[u64], bytes: Vec<u8>) -> Matrix {
        Matrix {
            info: TensorInfo {
                name: "weight".into(),
                dims: dims.to_vec(),
                ggml_type,
                offset: 0,
            },
            bytes,
        }
    }

    /// The weight loader is the one place that knows how to turn a GGUF
    /// tensor into a `Weight`, so the format has to come from the header:
    /// BF16 and F16 are both two bytes per element and cannot be told apart
    /// by length.
    #[test]
    fn load_weight_any_follows_the_declared_format() {
        let one = 1.0f32.to_le_bytes();
        let source = matrix(GGMLType::BF16, &[1, 1], vec![one[2], one[3]]);
        let weight = load_weight_any(&source, "weight", 1, 1, None).unwrap();
        assert_eq!(weight.ggml_type, GGMLType::BF16);
        assert_eq!(weight.n_in, 1);
        assert_eq!(weight.n_out, 1);
        // The BF16 kernel is the one that was selected, so the payload stays
        // BF16: read as F16 it would be 1.875 instead of 1.0.
        assert_eq!(weight.kernel.bf16_bytes().unwrap(), &[one[2], one[3]]);

        let source = matrix(GGMLType::F16, &[1, 1], vec![one[2], one[3]]);
        let weight = load_weight_any(&source, "weight", 1, 1, None).unwrap();
        assert_eq!(weight.ggml_type, GGMLType::F16);
        assert!(weight.kernel.bf16_bytes().is_none());
        assert_eq!(weight.kernel.weight_bytes().unwrap(), &[one[2], one[3]]);
    }

    /// F32 weights keep the borrowed bytes: the loader must not copy them.
    #[test]
    fn load_weight_any_borrows_the_f32_payload() {
        let bytes = [1.0f32, -2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let source = matrix(GGMLType::F32, &[2, 1], bytes);
        let weight = load_weight_any(&source, "weight", 2, 1, None).unwrap();
        assert_eq!(weight.n_in, 2);
        assert_eq!(weight.n_out, 1);
        assert_eq!(weight.kernel.f32_slice().unwrap(), [1.0, -2.0]);
        assert_eq!(
            weight.kernel.weight_bytes().unwrap().as_ptr(),
            source.bytes.as_ptr()
        );
    }

    #[test]
    fn load_weight_any_rejects_a_format_outside_the_whitelist() {
        let source = matrix(GGMLType::BF16, &[1, 1], vec![0u8; 2]);
        let error = expect_error(load_weight_any(
            &source,
            "weight",
            1,
            1,
            Some(&[GGMLType::F32]),
        ));
        assert!(error.contains("Unsupported matrix weight"), "{error}");
    }

    #[test]
    fn load_weight_any_rejects_a_payload_that_does_not_match_the_format() {
        let source = matrix(GGMLType::BF16, &[2, 1], vec![0u8; 2]);
        let error = expect_error(load_weight_any(&source, "weight", 2, 1, None));
        assert!(error.contains("expected 4"), "{error}");
    }

    /// A block-quantized row has to be a whole number of blocks.
    #[test]
    fn load_weight_any_rejects_a_row_that_is_not_block_aligned() {
        let source = matrix(GGMLType::Q8_0, &[33, 1], vec![0u8; 34 * 2]);
        let error = expect_error(load_weight_any(&source, "weight", 33, 1, None));
        assert!(error.contains("multiple of 32"), "{error}");
    }

    /// The format filter and the kernel builder have to agree: `supports`
    /// must be true exactly for the formats `from_bytes` accepts. A loader
    /// that filters on `supports` and then calls `from_bytes` panics on
    /// every type the two lists disagree about.
    #[test]
    fn the_supported_formats_are_exactly_the_ones_the_kernel_builder_accepts() {
        for value in 0..=64 {
            let Some(ggml_type) = GGMLType::from_i32(value) else {
                continue;
            };
            let data = vec![0u8; 64 * 4];
            let supported = crate::ops::kernel::QuantizedTensor::supports(ggml_type);
            let built = std::panic::catch_unwind(|| {
                crate::ops::kernel::QuantizedTensor::from_bytes(&data, ggml_type, 8, 8)
            });
            assert_eq!(
                supported,
                built.is_ok(),
                "{ggml_type:?}: supports() says {supported}, but the kernel builder {}",
                if built.is_ok() {
                    "accepts it"
                } else {
                    "panics on it"
                }
            );
        }
    }

    /// A GGUF file may declare a format no kernel can hold — the integer and
    /// double types, `Q8_1`, `Q8_K`. The loader has to report that, not hand
    /// it to the kernel builder, which panics on it. A panic on the path
    /// that reads a user's model file is never the right answer.
    #[test]
    fn load_weight_any_reports_a_format_that_has_no_kernel() {
        for ggml_type in [
            GGMLType::I8,
            GGMLType::I16,
            GGMLType::I32,
            GGMLType::I64,
            GGMLType::F64,
            GGMLType::Q8_1,
            GGMLType::Q8K,
        ] {
            let (block_elements, block_bytes) = ggml_type.type_traits();
            let n_in = block_elements;
            let n_out = 4;
            let bytes = vec![0u8; n_in * n_out / block_elements * block_bytes];
            let source = matrix(ggml_type, &[n_in as u64, n_out as u64], bytes);

            // `None` means "every format a kernel can hold" — and this one
            // has no kernel.
            let error = expect_error(load_weight_any(&source, "weight", n_in, n_out, None));
            assert!(error.contains("no weight kernel"), "{ggml_type:?}: {error}");

            // Naming it in the whitelist does not make it loadable either.
            let error = expect_error(load_weight_any(
                &source,
                "weight",
                n_in,
                n_out,
                Some(&[ggml_type]),
            ));
            assert!(error.contains("no weight kernel"), "{ggml_type:?}: {error}");
        }
    }

    #[test]
    fn typed_tensor_bytes_returns_the_payload_after_checking_it() {
        let source = source(GGMLType::F32, &[4]);
        assert_eq!(
            typed_tensor_bytes(&source, "weight", &[4], GGMLType::F32).map(|b| b.len()),
            Ok(16)
        );
    }

    /// The format has to come from the header. BF16 and F16 are both two
    /// bytes per element, so a byte length cannot tell them apart — and the
    /// two readings differ: `1.0` in BF16 read as F16 comes back as `1.875`.
    #[test]
    fn decode_f32_matrix_follows_the_declared_format() {
        let bf16_one = 1.0f32.to_le_bytes();
        let bf16_bytes = [bf16_one[2], bf16_one[3]];
        assert_eq!(
            decode_f32_matrix(GGMLType::BF16, &bf16_bytes, 1, 1).unwrap(),
            vec![1.0]
        );
        // The same bytes read as F16 are a different number, which is exactly
        // why the length cannot be used to pick the format.
        assert_eq!(
            decode_f32_matrix(GGMLType::F16, &bf16_bytes, 1, 1).unwrap(),
            vec![1.875]
        );

        let f32_bytes = 1.0f32.to_le_bytes();
        assert_eq!(
            decode_f32_matrix(GGMLType::F32, &f32_bytes, 1, 1).unwrap(),
            vec![1.0]
        );
    }

    #[test]
    fn decode_f32_matrix_rejects_a_format_it_cannot_expand() {
        let error = decode_f32_matrix(GGMLType::Q4_0, &[0u8; 18], 1, 1).unwrap_err();
        assert!(error.contains("Unsupported matrix format"), "{error}");
    }

    #[test]
    fn decode_f32_matrix_rejects_a_length_that_does_not_match_the_format() {
        // Two BF16 values are four bytes; six bytes cannot be read as them.
        let error = decode_f32_matrix(GGMLType::BF16, &[0u8; 6], 2, 1).unwrap_err();
        assert!(error.contains("byte length 6"), "{error}");
    }

    #[test]
    fn optional_f32_values_follows_the_declared_format() {
        let bf16_one = 1.0f32.to_le_bytes();
        let mut source = source(GGMLType::BF16, &[1]);
        source.bytes = vec![bf16_one[2], bf16_one[3]];
        assert_eq!(
            optional_f32_values(&source, "weight").unwrap(),
            Some(vec![1.0])
        );
    }

    #[test]
    fn typed_tensor_bytes_rejects_a_shape_with_a_zero_axis() {
        let source = source(GGMLType::F32, &[0]);
        let error = typed_tensor_bytes(&source, "weight", &[0], GGMLType::F32).unwrap_err();
        assert!(error.contains("non-zero shape"), "{error}");
    }

    #[test]
    fn f32_tensor_bytes_rejects_storage_that_is_not_f32() {
        let source = source(GGMLType::BF16, &[4]);
        let error = f32_tensor_bytes(&source, "weight", &[4]).unwrap_err();
        assert!(error.contains("Invalid tensor weight"), "{error}");
    }

    /// The optional variant is the one a vision loader uses for a bias that
    /// may be absent, so a missing tensor is not an error there.
    #[test]
    fn optional_f32_tensor_bytes_reports_a_missing_tensor_as_none() {
        struct Empty;
        impl TensorSource for Empty {
            fn metadata(&self, _key: &str) -> Option<&MetaValue> {
                None
            }
            fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
                None
            }
            fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
                None
            }
        }
        assert_eq!(optional_f32_tensor_bytes(&Empty, "weight").unwrap(), None);
    }

    #[test]
    fn optional_f32_tensor_bytes_rejects_storage_that_is_not_f32() {
        let source = source(GGMLType::BF16, &[4]);
        let error = optional_f32_tensor_bytes(&source, "weight").unwrap_err();
        assert!(error.contains("must be F32"), "{error}");
    }

    #[test]
    fn require_tensor_type_accepts_the_listed_type() {
        let source = source(GGMLType::Q8_0, &[64, 2]);
        assert_eq!(
            require_tensor_type(&source, "weight", &[64, 2], &[GGMLType::Q8_0]),
            Ok(GGMLType::Q8_0)
        );
    }

    #[test]
    fn require_tensor_type_rejects_a_type_outside_the_list() {
        let source = source(GGMLType::BF16, &[64, 2]);
        let error = require_tensor_type(
            &source,
            "weight",
            &[64, 2],
            &[GGMLType::Q8_0, GGMLType::Q6K],
        )
        .unwrap_err();
        assert!(error.contains("Invalid tensor weight"), "{error}");
        assert!(error.contains("one of"), "{error}");
    }

    #[test]
    fn require_tensor_type_rejects_a_wrong_shape() {
        let source = source(GGMLType::BF16, &[64, 2]);
        let error =
            require_tensor_type(&source, "weight", &[32, 2], &[GGMLType::BF16]).unwrap_err();
        assert!(error.contains("Invalid tensor weight"), "{error}");
    }

    /// The byte length follows the type's block layout, not the element
    /// count: a Q8_0 row of 64 values is 68 bytes, not 128.
    #[test]
    fn require_tensor_type_measures_a_block_payload() {
        let mut source = source(GGMLType::Q8_0, &[64, 2]);
        source.bytes = vec![0u8; 68 * 2];
        assert!(require_tensor_type(&source, "weight", &[64, 2], &[GGMLType::Q8_0]).is_ok());
        source.bytes = vec![0u8; 128 * 2];
        let error =
            require_tensor_type(&source, "weight", &[64, 2], &[GGMLType::Q8_0]).unwrap_err();
        assert!(error.contains("Invalid tensor data length"), "{error}");
    }
}
