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
    I8 = 16,
    I16 = 17,
    I32 = 18,
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
            16 => Some(Self::I8),
            17 => Some(Self::I16),
            18 => Some(Self::I32),
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
            Self::I8 => (1, 1),
            Self::I16 => (1, 2),
            Self::I32 => (1, 4),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_nbytes() {
        assert_eq!(GGMLType::F32.nbytes(256), 1024);
        assert_eq!(GGMLType::F16.nbytes(256), 512);
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
}
