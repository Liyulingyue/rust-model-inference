//! GGUF file loader: byte-level GGUF parser, mmap-backed [`GGUFLoader`], and
//! free-function [`model_config_from_source`] that derives a
//! [`crate::core::traits::ModelConfig`] from GGUF metadata.
//!
//! Depends on [`crate::core::tensor`] for the value types it produces.

use std::fs::File;

use memmap2::Mmap;

use crate::core::tensor::{
    GGMLType, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, MetaValue, MetaValueType, TensorInfo,
    TensorSource,
};
use crate::core::traits::ModelConfig;

pub(crate) struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_len(&mut self, context: &str) -> Result<usize, String> {
        usize::try_from(self.read_u64()?)
            .map_err(|_| format!("{context} length does not fit usize"))
    }

    pub(crate) fn read_exact(&mut self, len: usize, context: &str) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| format!("{context} range overflow"))?;
        let value = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| format!("EOF reading {context} of length {len}"))?;
        self.pos = end;
        Ok(value)
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn ensure_count(
        &self,
        count: usize,
        minimum_item_bytes: usize,
        context: &str,
    ) -> Result<(), String> {
        if count > self.remaining() / minimum_item_bytes {
            return Err(format!("{context} count exceeds remaining bytes"));
        }
        Ok(())
    }

    fn try_vec<T>(count: usize, context: &str) -> Result<Vec<T>, String> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| format!("failed to allocate {context}"))?;
        Ok(values)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        if self.remaining() < 1 {
            return Err("EOF reading u8".into());
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> Result<i8, String> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        if self.remaining() < 2 {
            return Err("EOF reading u16".into());
        }
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_i16(&mut self) -> Result<i16, String> {
        Ok(self.read_u16()? as i16)
    }

    pub(crate) fn read_u32(&mut self) -> Result<u32, String> {
        if self.remaining() < 4 {
            return Err("EOF reading u32".into());
        }
        let v = u32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    pub(crate) fn read_i32(&mut self) -> Result<i32, String> {
        Ok(self.read_u32()? as i32)
    }

    pub(crate) fn read_u64(&mut self) -> Result<u64, String> {
        if self.remaining() < 8 {
            return Err("EOF reading u64".into());
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.data[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i64(&mut self) -> Result<i64, String> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    pub(crate) fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_len("string")?;
        let bytes = self.read_exact(len, "string")?;
        String::from_utf8(bytes.to_vec()).map_err(|error| format!("Invalid UTF-8 string: {error}"))
    }

    pub(crate) fn read_meta_value(&mut self, vtype: MetaValueType) -> Result<MetaValue, String> {
        match vtype {
            MetaValueType::Uint8 => Ok(MetaValue::Uint8(self.read_u8()?)),
            MetaValueType::Int8 => Ok(MetaValue::Int8(self.read_i8()?)),
            MetaValueType::Uint16 => Ok(MetaValue::Uint16(self.read_u16()?)),
            MetaValueType::Int16 => Ok(MetaValue::Int16(self.read_i16()?)),
            MetaValueType::Uint32 => Ok(MetaValue::Uint32(self.read_u32()?)),
            MetaValueType::Int32 => Ok(MetaValue::Int32(self.read_i32()?)),
            MetaValueType::Float32 => Ok(MetaValue::Float32(self.read_f32()?)),
            MetaValueType::Bool => match self.read_u8()? {
                0 => Ok(MetaValue::Bool(false)),
                1 => Ok(MetaValue::Bool(true)),
                value => Err(format!("Invalid bool value: {value}")),
            },
            MetaValueType::String => Ok(MetaValue::String(self.read_string()?)),
            MetaValueType::Uint64 => Ok(MetaValue::Uint64(self.read_u64()?)),
            MetaValueType::Int64 => Ok(MetaValue::Int64(self.read_i64()?)),
            MetaValueType::Float64 => Ok(MetaValue::Float64(self.read_f64()?)),
            MetaValueType::Array => {
                let elem_type_i32 = self.read_i32()?;
                let elem_type = MetaValueType::from_i32(elem_type_i32)
                    .ok_or_else(|| format!("Unknown meta value type: {}", elem_type_i32))?;
                let n = self.read_len("array")?;
                let minimum_item_bytes = match elem_type {
                    MetaValueType::Uint8 | MetaValueType::Int8 | MetaValueType::Bool => 1,
                    MetaValueType::Uint16 | MetaValueType::Int16 => 2,
                    MetaValueType::Uint32 | MetaValueType::Int32 | MetaValueType::Float32 => 4,
                    MetaValueType::Uint64 | MetaValueType::Int64 | MetaValueType::Float64 => 8,
                    MetaValueType::String => 8,
                    MetaValueType::Array => {
                        return Err("Nested metadata arrays are not supported".into())
                    }
                };
                self.ensure_count(n, minimum_item_bytes, "array")?;
                let mut vals = Self::try_vec(n, "array values")?;
                for _ in 0..n {
                    vals.push(self.read_meta_value(elem_type)?);
                }
                Ok(MetaValue::Array(elem_type, vals))
            }
        }
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }
}

pub struct GGUFLoader {
    mmap: Mmap,
    pub version: u32,
    pub alignment: u64,
    data_offset: usize,
    metadata: Vec<(String, MetaValue)>,
    tensors: Vec<TensorInfo>,
}

impl std::fmt::Debug for GGUFLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GGUFLoader")
            .field("version", &self.version)
            .field("alignment", &self.alignment)
            .field("data_offset", &self.data_offset)
            .field("n_metadata", &self.metadata.len())
            .field("n_tensors", &self.tensors.len())
            .finish()
    }
}

impl GGUFLoader {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open GGUF file: {}", e))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("Failed to mmap: {}", e))?;
        Self::from_mmap(mmap)
    }

    pub fn from_mmap(mmap: Mmap) -> Result<Self, String> {
        let mut reader = ByteReader::new(&mmap);

        let magic = reader.read_u32()?;
        if &magic.to_le_bytes() != GGUF_MAGIC {
            return Err(format!(
                "Invalid GGUF magic: {:?} (expected {:?})",
                &magic.to_le_bytes(),
                GGUF_MAGIC
            ));
        }

        let version = reader.read_u32()?;
        if version < 2 || version > 3 {
            return Err(format!("Unsupported GGUF version: {}", version));
        }

        let n_tensors = usize::try_from(reader.read_u64()?)
            .map_err(|_| "tensor count does not fit usize".to_string())?;
        let n_kv = usize::try_from(reader.read_u64()?)
            .map_err(|_| "metadata count does not fit usize".to_string())?;

        reader.ensure_count(n_kv, 13, "metadata")?;
        let mut metadata = ByteReader::try_vec(n_kv, "metadata entries")?;
        for _ in 0..n_kv {
            let key = reader.read_string()?;
            let vtype_i32 = reader.read_i32()?;
            let vtype = MetaValueType::from_i32(vtype_i32)
                .ok_or_else(|| format!("Unknown meta value type: {}", vtype_i32))?;
            let value = reader.read_meta_value(vtype)?;
            metadata.push((key, value));
        }

        reader.ensure_count(n_tensors, 24, "tensor")?;
        let mut tensors = ByteReader::try_vec(n_tensors, "tensor entries")?;
        for _ in 0..n_tensors {
            let name = reader.read_string()?;
            let n_dims = usize::try_from(reader.read_u32()?)
                .map_err(|_| "tensor dimension count does not fit usize".to_string())?;
            reader.ensure_count(n_dims, 8, "tensor dimension")?;
            let mut dims = ByteReader::try_vec(n_dims, "tensor dimensions")?;
            for _ in 0..n_dims {
                dims.push(reader.read_u64()?);
            }
            let type_i32 = reader.read_i32()?;
            let ggml_type = GGMLType::from_i32(type_i32)
                .ok_or_else(|| format!("Unknown GGML type: {}", type_i32))?;
            let offset = reader.read_u64()?;
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        let alignment = metadata
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| v.to_u64())
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
        if alignment == 0 {
            return Err("GGUF alignment must be nonzero".into());
        }
        let alignment = usize::try_from(alignment)
            .map_err(|_| "GGUF alignment does not fit usize".to_string())?;

        let data_offset = reader.pos();
        let remainder = data_offset % alignment;
        let padded_data_offset = if remainder == 0 {
            data_offset
        } else {
            data_offset
                .checked_add(alignment - remainder)
                .ok_or_else(|| "GGUF data offset overflow".to_string())?
        };

        for tensor in &tensors {
            let offset = usize::try_from(tensor.offset)
                .map_err(|_| format!("tensor {} offset does not fit usize", tensor.name))?;
            let nbytes = usize::try_from(
                tensor
                    .checked_nbytes()
                    .ok_or_else(|| format!("invalid tensor shape: {}", tensor.name))?,
            )
            .map_err(|_| format!("tensor {} byte size does not fit usize", tensor.name))?;
            let end = padded_data_offset
                .checked_add(offset)
                .and_then(|start| start.checked_add(nbytes))
                .ok_or_else(|| format!("tensor {} range overflow", tensor.name))?;
            if end > mmap.len() {
                return Err(format!("tensor {} exceeds GGUF data", tensor.name));
            }
        }

        Ok(Self {
            mmap,
            version,
            alignment: alignment as u64,
            data_offset: padded_data_offset,
            metadata,
            tensors,
        })
    }

    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    pub fn n_kv(&self) -> usize {
        self.metadata.len()
    }

    pub fn data_offset(&self) -> usize {
        self.data_offset
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn metadata_entries(&self) -> &[(String, MetaValue)] {
        &self.metadata
    }

    pub fn metadata(&self, key: &str) -> Option<&MetaValue> {
        for (k, v) in &self.metadata {
            if k == key {
                return Some(v);
            }
        }
        None
    }

    pub fn metadata_keys(&self) -> impl Iterator<Item = &str> {
        self.metadata.iter().map(|(k, _)| k.as_str())
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        for t in &self.tensors {
            if t.name == name {
                return Some(t);
            }
        }
        None
    }

    pub fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        let tensor = self.tensor_info(name)?;
        let abs_offset = self
            .data_offset
            .checked_add(usize::try_from(tensor.offset).ok()?)?;
        let nbytes = usize::try_from(tensor.checked_nbytes()?).ok()?;
        let end = abs_offset.checked_add(nbytes)?;
        self.mmap.get(abs_offset..end)
    }

    pub fn model_config(&self) -> Result<ModelConfig, String> {
        model_config_from_source(self)
    }
}

pub fn model_config_from_source<S: TensorSource + ?Sized>(
    source: &S,
) -> Result<ModelConfig, String> {
    let arch = source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .unwrap_or_default();
    let prefix: &str = if arch == "hunyuan-dense" {
        "hunyuan-dense"
    } else {
        &arch
    };
    if !matches!(prefix, "qwen2" | "qwen3" | "qwen3vl" | "qwen35" | "qwen3tts" | "llama" | "hunyuan-dense" | "pig" | "lfm2") {
        return Err(format!("Unsupported architecture: {arch}"));
    }

    // Pig (Z-Image) uses default config since it doesn't have standard LLM metadata
    if arch == "pig" {
        return Ok(ModelConfig {
            n_embd: 1024,
            n_layer: 30,
            n_head: 16,
            n_head_kv: 16,
            n_embd_head: 64,
            n_ff: 4096,
            n_ctx: 512,
            vocab_size: 0,
            rope_freq_base: 0.0,
            norm_eps: 1e-5,
        });
    }
    let get_u64 = |key: &str| -> Result<u64, String> {
        source
            .metadata(key)
            .and_then(MetaValue::to_u64)
            .ok_or_else(|| format!("Missing metadata: {key}"))
    };
    let get_f64 = |key: &str| -> Result<f64, String> {
        source
            .metadata(key)
            .and_then(MetaValue::to_f64)
            .ok_or_else(|| format!("Missing metadata: {key}"))
    };
    let get_f64_opt = |key: &str, default: f64| -> Result<f64, String> {
        Ok(source
            .metadata(key)
            .and_then(MetaValue::to_f64)
            .unwrap_or(default))
    };
    let n_embd = usize::try_from(get_u64(&format!("{prefix}.embedding_length"))?)
        .map_err(|_| format!("{prefix}.embedding_length does not fit usize"))?;
    let n_head = usize::try_from(get_u64(&format!("{prefix}.attention.head_count"))?)
        .map_err(|_| format!("{prefix}.attention.head_count does not fit usize"))?;
    if n_head == 0 || n_embd % n_head != 0 {
        return Err(format!(
            "Invalid {prefix} head shape: embedding_length={n_embd}, head_count={n_head}"
        ));
    }
    let as_usize = |key: String| -> Result<usize, String> {
        usize::try_from(get_u64(&key)?).map_err(|_| format!("{key} does not fit usize"))
    };

    Ok(ModelConfig {
        n_embd,
        n_layer: as_usize(format!("{prefix}.block_count"))?,
        n_head,
        n_head_kv: if arch == "lfm2" {
            // LFM2 stores head_count_kv as a per-layer array; the attention
            // layers (kv=8) are the ones that matter for ModelConfig defaults.
            // lfm2 skeleton reads its own head_count_kv_array for per-layer
            // dispatch.
            8
        } else {
            as_usize(format!("{prefix}.attention.head_count_kv"))?
        },
        n_embd_head: n_embd / n_head,
        n_ff: as_usize(format!("{prefix}.feed_forward_length"))?,
        n_ctx: as_usize(format!("{prefix}.context_length"))?,
        vocab_size: match get_u64(&format!("{prefix}.vocab_size")) {
            Ok(value) => usize::try_from(value)
                .map_err(|_| format!("{prefix}.vocab_size does not fit usize"))?,
            Err(_) => source
                .metadata("tokenizer.ggml.tokens")
                .and_then(MetaValue::to_arr)
                .map(Vec::len)
                .unwrap_or(0),
        },
        rope_freq_base: get_f64_opt(&format!("{prefix}.rope.freq_base"), 1_000_000.0)? as f32,
        norm_eps: get_f64(&format!("{prefix}.attention.layer_norm_rms_epsilon"))? as f32,
    })
}

impl TensorSource for GGUFLoader {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        GGUFLoader::metadata(self, key)
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        GGUFLoader::tensor_info(self, name)
    }

    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        GGUFLoader::tensor_slice(self, name)
    }
}

/// Architecture-specific knobs derived from `general.architecture` for the
/// Qwen3 model family (Qwen2/Qwen3/Qwen3-VL/Qwen3.5/LLaMA/Hunyuan-Dense).
///
/// Phase 4c: extracted from `models::qwen3::qwen3_multimodal::Qwen3Config::from_source` so that
/// architecture dispatch lives next to the GGUF metadata it interprets, rather
/// than inside a model implementation that historically knew too much about
/// other architectures.
pub struct Qwen3ArchKnobs {
    /// Canonical architecture name (`"qwen3"`, `"qwen3vl"`, ...).
    pub arch: String,
    /// Whether per-head Q/K RMSNorm is applied.
    pub has_qk_norm: bool,
    /// Multi-modal rope sections for Qwen3-VL. `None` means Neox rope.
    /// Callers using the Interleaved variant must combine these sections
    /// with the model's `n_embd_head_k` to build the final rope flavor.
    pub rope_sections: Option<[i32; 4]>,
    /// Optional whitelist of acceptable dimensional configurations per arch.
    /// `None` means any configuration derived from `model_config_from_source`
    /// is accepted (e.g. Qwen3 base).
    pub allowed_dimensions: Option<Qwen3AllowedDimensions>,
}

/// Concrete dimensional constraints for an architecture whose variants are
/// not yet enumerated individually. When present, configs that do not match
/// are rejected — this surfaces the fact that we only support a fixed set of
/// model sizes for that architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3AllowedDimensions {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub n_ff: usize,
    pub n_ctx: usize,
    /// f32 stored as bits to avoid float-equality noise on cross-platform
    /// builds.
    pub norm_eps_bits: u32,
    pub freq_base_bits: u32,
}

/// The single Qwen3-VL main-model configuration currently supported
/// (1024-dim, 28-layer, 65 k context, 1e-6 / 1e6 RoPE base).
const KNOWN_QWEN3VL_DIMENSIONS: Qwen3AllowedDimensions = Qwen3AllowedDimensions {
    n_embd: 1024,
    n_layer: 28,
    n_head: 16,
    n_head_kv: 8,
    n_embd_head_k: 128,
    n_embd_head_v: 128,
    n_ff: 3072,
    n_ctx: 65_536,
    norm_eps_bits: 1e-6_f32.to_bits(),
    freq_base_bits: 1_000_000_f32.to_bits(),
};

/// Resolve the Qwen3-family knobs from `general.architecture`.
///
/// This is the **single** place where architecture dispatch happens. It is
/// called by `models::qwen3::qwen3_multimodal::Qwen3Config::from_source` after
/// `model_config_from_source` has produced the dimension set; together they
/// replace the older `Qwen3Config::from_source` that hardcoded the arch list
/// inside the model file.
pub fn qwen3_arch_knobs<S: TensorSource + ?Sized>(
    source: &S,
) -> Result<Qwen3ArchKnobs, String> {
    let arch = source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .ok_or_else(|| "Missing metadata: general.architecture".to_string())?
        .to_string();
    if !matches!(
        arch.as_str(),
        "qwen2" | "qwen3" | "qwen3vl" | "qwen35" | "qwen3tts" | "llama" | "hunyuan-dense" | "lfm2"
    ) {
        return Err(format!("Unsupported Qwen3-family architecture: {arch}"));
    }

    let has_qk_norm = matches!(arch.as_str(), "qwen3" | "qwen3vl" | "hunyuan-dense" | "lfm2");

    let rope_sections = if arch == "qwen3vl" {
        let sections = read_i32_array(source, "qwen3vl.rope.dimension_sections")?;
        if sections != [24, 20, 20, 0] {
            return Err(format!(
                "Unsupported qwen3vl.rope.dimension_sections: {sections:?}"
            ));
        }
        Some(sections)
    } else {
        None
    };

    let allowed_dimensions = if arch == "qwen3vl" {
        Some(KNOWN_QWEN3VL_DIMENSIONS)
    } else {
        None
    };

    Ok(Qwen3ArchKnobs {
        arch,
        has_qk_norm,
        rope_sections,
        allowed_dimensions,
    })
}

/// Read an `[i32; 4]` array from a GGUF metadata key.
pub(crate) fn read_i32_array<S: TensorSource + ?Sized>(
    source: &S,
    key: &str,
) -> Result<[i32; 4], String> {
    let value = source
        .metadata(key)
        .ok_or_else(|| format!("Missing metadata: {key}"))?;
    let MetaValue::Array(_, items) = value else {
        return Err(format!("{key} is not an array"));
    };
    if items.len() != 4 {
        return Err(format!("{key} expected 4 entries, got {}", items.len()));
    }
    let mut out = [0i32; 4];
    for (i, item) in items.iter().enumerate() {
        out[i] = item
            .to_u64()
            .ok_or_else(|| format!("{key}[{i}] is not an integer"))?
            as i32;
    }
    Ok(out)
}

/// Check that the supplied dimensions match the allowed whitelist. Returns
/// `Ok(())` if `allowed` is `None` or if the values all match.
pub(crate) fn check_qwen3_allowed_dimensions(
    allowed: Qwen3AllowedDimensions,
    config: &crate::core::traits::ModelConfig,
    n_embd_head_k: usize,
    n_embd_head_v: usize,
) -> Result<(), String> {
    if config.n_embd == allowed.n_embd
        && config.n_layer == allowed.n_layer
        && config.n_head == allowed.n_head
        && config.n_head_kv == allowed.n_head_kv
        && n_embd_head_k == allowed.n_embd_head_k
        && n_embd_head_v == allowed.n_embd_head_v
        && config.n_ff == allowed.n_ff
        && config.n_ctx == allowed.n_ctx
        && config.norm_eps.to_bits() == allowed.norm_eps_bits
        && config.rope_freq_base.to_bits() == allowed.freq_base_bits
    {
        Ok(())
    } else {
        Err(format!(
            "Unsupported main-model configuration: expected {:?}, got n_embd={} n_layer={} n_head={} n_head_kv={} n_embd_head_k={} n_embd_head_v={} n_ff={} n_ctx={} norm_eps={} freq_base={}",
            allowed,
            config.n_embd,
            config.n_layer,
            config.n_head,
            config.n_head_kv,
            n_embd_head_k,
            n_embd_head_v,
            config.n_ff,
            config.n_ctx,
            config.norm_eps,
            config.rope_freq_base
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{GGMLType, MetaValueType, TensorInfo, TensorSource};

    fn push_u8(buf: &mut Vec<u8>, v: u8) {
        buf.push(v);
    }
    fn push_u16(buf: &mut Vec<u8>, v: u16) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn push_i16(buf: &mut Vec<u8>, v: i16) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn push_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn push_i32(buf: &mut Vec<u8>, v: i32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn push_u64(buf: &mut Vec<u8>, v: u64) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn push_i64(buf: &mut Vec<u8>, v: i64) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn push_f32(buf: &mut Vec<u8>, v: f32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn push_f64(buf: &mut Vec<u8>, v: f64) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn push_str(buf: &mut Vec<u8>, s: &str) {
        push_u64(buf, s.len() as u64);
        buf.extend_from_slice(s.as_bytes());
    }

    fn push_kv(
        buf: &mut Vec<u8>,
        key: &str,
        vtype: MetaValueType,
        write_val: impl FnOnce(&mut Vec<u8>),
    ) {
        push_str(buf, key);
        push_i32(buf, vtype as i32);
        write_val(buf);
    }

    fn build_minimal_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        push_u32(&mut b, u32::from_le_bytes(*b"GGUF"));
        push_u32(&mut b, 3);
        push_u64(&mut b, 2);
        push_u64(&mut b, 11);

        push_kv(&mut b, "general.architecture", MetaValueType::String, |b| {
            push_str(b, "qwen2")
        });
        push_kv(
            &mut b,
            "qwen2.embedding_length",
            MetaValueType::Uint32,
            |b| push_u32(b, 1024),
        );
        push_kv(&mut b, "qwen2.block_count", MetaValueType::Uint32, |b| {
            push_u32(b, 24)
        });
        push_kv(
            &mut b,
            "qwen2.attention.head_count",
            MetaValueType::Uint32,
            |b| push_u32(b, 16),
        );
        push_kv(&mut b, "general.alignment", MetaValueType::Uint64, |b| {
            push_u64(b, 32)
        });
        push_kv(
            &mut b,
            "qwen2.attention.head_count_kv",
            MetaValueType::Uint32,
            |b| push_u32(b, 16),
        );
        push_kv(
            &mut b,
            "qwen2.feed_forward_length",
            MetaValueType::Uint32,
            |b| push_u32(b, 2816),
        );
        push_kv(&mut b, "qwen2.context_length", MetaValueType::Uint32, |b| {
            push_u32(b, 4096)
        });
        push_kv(&mut b, "qwen2.vocab_size", MetaValueType::Uint32, |b| {
            push_u32(b, 151936)
        });
        push_kv(
            &mut b,
            "qwen2.rope.freq_base",
            MetaValueType::Float32,
            |b| push_f32(b, 1_000_000.0),
        );
        push_kv(
            &mut b,
            "qwen2.attention.layer_norm_rms_epsilon",
            MetaValueType::Float32,
            |b| push_f32(b, 1e-6),
        );

        push_str(&mut b, "token_embd.weight");
        push_u32(&mut b, 2);
        push_u64(&mut b, 1024);
        push_u64(&mut b, 151936);
        push_i32(&mut b, GGMLType::Q4K as i32);
        push_u64(&mut b, 0);

        push_str(&mut b, "blk.0.attn_q.weight");
        push_u32(&mut b, 2);
        push_u64(&mut b, 1024);
        push_u64(&mut b, 1024);
        push_i32(&mut b, GGMLType::Q4K as i32);
        let embd_nbytes = GGMLType::Q4K.nbytes(1024 * 151936);
        let padded = ((embd_nbytes as u64 + 31) / 32 * 32) as u64;
        push_u64(&mut b, padded);

        while b.len() % 32 != 0 {
            b.push(0);
        }
        let data_start = b.len();
        let total_data = padded as usize + GGMLType::Q4K.nbytes(1024 * 1024);
        b.resize(data_start + total_data, 0xAB);
        b
    }

    fn build_all_meta_types_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        push_u32(&mut b, u32::from_le_bytes(*b"GGUF"));
        push_u32(&mut b, 3);
        push_u64(&mut b, 0);
        push_u64(&mut b, 13);

        push_kv(&mut b, "test.uint8", MetaValueType::Uint8, |b| {
            push_u8(b, 42)
        });
        push_kv(&mut b, "test.int8", MetaValueType::Int8, |b| {
            push_u8(b, (-1i8) as u8)
        });
        push_kv(&mut b, "test.uint16", MetaValueType::Uint16, |b| {
            push_u16(b, 1000)
        });
        push_kv(&mut b, "test.int16", MetaValueType::Int16, |b| {
            push_i16(b, -100)
        });
        push_kv(&mut b, "test.uint32", MetaValueType::Uint32, |b| {
            push_u32(b, 1024)
        });
        push_kv(&mut b, "test.int32", MetaValueType::Int32, |b| {
            push_i32(b, -24)
        });
        push_kv(&mut b, "test.float32", MetaValueType::Float32, |b| {
            push_f32(b, 3.14)
        });
        push_kv(&mut b, "test.bool", MetaValueType::Bool, |b| push_u8(b, 1));
        push_kv(&mut b, "test.string", MetaValueType::String, |b| {
            push_str(b, "hello")
        });
        push_kv(&mut b, "test.uint64", MetaValueType::Uint64, |b| {
            push_u64(b, 999999)
        });
        push_kv(&mut b, "test.int64", MetaValueType::Int64, |b| {
            push_i64(b, -123456)
        });
        push_kv(&mut b, "test.float64", MetaValueType::Float64, |b| {
            push_f64(b, 2.71828)
        });
        push_kv(&mut b, "test.array", MetaValueType::Array, |b| {
            push_i32(b, MetaValueType::Uint32 as i32);
            push_u64(b, 3);
            push_u32(b, 10);
            push_u32(b, 20);
            push_u32(b, 30);
        });

        while b.len() % 32 != 0 {
            b.push(0);
        }
        b
    }

    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

    fn parse_temp(data: &[u8]) -> Result<GGUFLoader, String> {
        let dir = std::env::temp_dir().join("rust_model_inference_test");
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = dir.join(format!("test_{}_{}.gguf", std::process::id(), id));
        std::fs::write(&path, data).map_err(|e| e.to_string())?;
        GGUFLoader::from_file(path.to_str().unwrap())
    }

    #[test]
    fn test_gguf_parse_minimal() {
        let data = build_minimal_gguf();
        let loader = parse_temp(&data).expect("parse minimal GGUF");

        assert_eq!(loader.version, 3);
        assert_eq!(loader.alignment, 32);
        assert_eq!(loader.n_tensors(), 2);
        assert_eq!(loader.n_kv(), 11);

        let arch = loader
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val())
            .unwrap();
        assert_eq!(arch, "qwen2");
        assert_eq!(
            loader
                .metadata("qwen2.embedding_length")
                .and_then(|v| v.to_u64()),
            Some(1024)
        );
        assert_eq!(
            loader
                .metadata("qwen2.block_count")
                .and_then(|v| v.to_u64()),
            Some(24)
        );

        let ti0 = loader.tensor_info("token_embd.weight").expect("tensor 0");
        assert_eq!(ti0.dims, vec![1024, 151936]);
        assert_eq!(ti0.ggml_type, GGMLType::Q4K);
        assert_eq!(ti0.offset, 0);

        let ti1 = loader.tensor_info("blk.0.attn_q.weight").expect("tensor 1");
        assert_eq!(ti1.dims, vec![1024, 1024]);

        let s0 = loader.tensor_slice("token_embd.weight").expect("slice 0");
        assert_eq!(s0.len(), ti0.nbytes());
        let s1 = loader.tensor_slice("blk.0.attn_q.weight").expect("slice 1");
        assert_eq!(s1.len(), ti1.nbytes());
    }

    struct DelegatingSource<'a>(&'a GGUFLoader);

    impl TensorSource for DelegatingSource<'_> {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.0.metadata(key)
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.0.tensor_info(name)
        }

        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.0.tensor_slice(name)
        }
    }

    #[test]
    fn model_config_and_linear_accept_a_tensor_source() {
        let loader = parse_temp(&build_minimal_gguf()).unwrap();
        let source = DelegatingSource(&loader);
        assert_eq!(source.model_config().unwrap().n_embd, 1024);
        assert!(crate::core::model::QuantizedLinear::from_source(
            &source,
            "blk.0.attn_q.weight",
            None,
            1024,
            1024,
            "attn_q",
        )
        .is_some());
    }

    #[derive(Default)]
    struct MapTensorSource {
        metadata: std::collections::HashMap<String, MetaValue>,
        tensors: std::collections::HashMap<String, TensorInfo>,
    }

    impl TensorSource for MapTensorSource {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.metadata.get(key)
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.tensors.get(name)
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    #[test]
    fn qwen3vl_uses_its_own_metadata_prefix() {
        let metadata = std::collections::HashMap::from([
            (
                "general.architecture".into(),
                MetaValue::String("qwen3vl".into()),
            ),
            ("qwen3vl.embedding_length".into(), MetaValue::Uint32(1024)),
            ("qwen3vl.block_count".into(), MetaValue::Uint32(28)),
            ("qwen3vl.attention.head_count".into(), MetaValue::Uint32(16)),
            (
                "qwen3vl.attention.head_count_kv".into(),
                MetaValue::Uint32(8),
            ),
            (
                "qwen3vl.feed_forward_length".into(),
                MetaValue::Uint32(3072),
            ),
            ("qwen3vl.context_length".into(), MetaValue::Uint32(65536)),
            (
                "qwen3vl.rope.freq_base".into(),
                MetaValue::Float32(1_000_000.0),
            ),
            (
                "qwen3vl.attention.layer_norm_rms_epsilon".into(),
                MetaValue::Float32(1e-6),
            ),
            ("qwen3vl.vocab_size".into(), MetaValue::Uint32(151_936)),
        ]);
        let config = model_config_from_source(&MapTensorSource {
            metadata,
            tensors: std::collections::HashMap::new(),
        })
        .unwrap();
        assert_eq!(
            (
                config.n_embd,
                config.n_layer,
                config.n_head,
                config.n_head_kv
            ),
            (1024, 28, 16, 8)
        );
        assert_eq!((config.n_ff, config.n_ctx), (3072, 65536));
        assert_eq!(config.rope_freq_base, 1_000_000.0);
        assert_eq!(config.norm_eps, 1e-6);
    }

    #[test]
    fn test_gguf_all_meta_types() {
        let data = build_all_meta_types_gguf();
        let loader = parse_temp(&data).expect("parse all meta types");

        assert_eq!(
            loader.metadata("test.uint8").and_then(|v| v.to_u64()),
            Some(42)
        );
        assert_eq!(
            loader.metadata("test.int8").and_then(|v| v.to_u64()),
            Some((-1i8) as u64)
        );
        assert_eq!(
            loader.metadata("test.uint16").and_then(|v| v.to_u64()),
            Some(1000)
        );
        assert_eq!(
            loader.metadata("test.int16").and_then(|v| v.to_u64()),
            Some((-100i16) as u64)
        );
        assert_eq!(
            loader.metadata("test.uint32").and_then(|v| v.to_u64()),
            Some(1024)
        );
        assert_eq!(
            loader.metadata("test.int32").and_then(|v| v.to_u64()),
            Some((-24i32) as u64)
        );

        let f32v = loader
            .metadata("test.float32")
            .and_then(|v| v.to_f64())
            .unwrap();
        assert!((f32v - 3.14).abs() < 0.01);

        match loader.metadata("test.bool") {
            Some(MetaValue::Bool(true)) => {}
            other => panic!("expected Bool(true), got {:?}", other),
        }

        assert_eq!(
            loader
                .metadata("test.string")
                .and_then(|v| v.to_string_val()),
            Some("hello")
        );
        assert_eq!(
            loader.metadata("test.uint64").and_then(|v| v.to_u64()),
            Some(999999)
        );
        assert_eq!(
            loader.metadata("test.int64").and_then(|v| v.to_u64()),
            Some((-123456i64) as u64)
        );

        let f64v = loader
            .metadata("test.float64")
            .and_then(|v| v.to_f64())
            .unwrap();
        assert!((f64v - 2.71828).abs() < 0.001);

        match loader.metadata("test.array") {
            Some(MetaValue::Array(et, vals)) => {
                assert_eq!(*et, MetaValueType::Uint32);
                assert_eq!(vals.len(), 3);
                assert_eq!(vals[0].to_u64(), Some(10));
                assert_eq!(vals[1].to_u64(), Some(20));
                assert_eq!(vals[2].to_u64(), Some(30));
            }
            other => panic!("expected Array, got {:?}", other),
        }
    }

    #[test]
    fn malformed_bool_and_utf8_are_rejected() {
        let mut bad_bool = ByteReader::new(&[2]);
        assert!(bad_bool.read_meta_value(MetaValueType::Bool).is_err());

        let mut encoded = Vec::new();
        push_u64(&mut encoded, 1);
        encoded.push(0xFF);
        let mut bad_string = ByteReader::new(&encoded);
        assert!(bad_string.read_string().is_err());
    }

    #[test]
    fn metadata_array_count_must_fit_remaining_data() {
        let mut encoded = Vec::new();
        push_i32(&mut encoded, MetaValueType::Uint8 as i32);
        push_u64(&mut encoded, 2);
        let err = ByteReader::new(&encoded)
            .read_meta_value(MetaValueType::Array)
            .unwrap_err();
        assert!(err.contains("array count exceeds remaining bytes"), "{err}");
    }

    #[test]
    fn nested_metadata_arrays_are_rejected_before_recursive_decode() {
        let mut encoded = Vec::new();
        push_u32(&mut encoded, u32::from_le_bytes(*b"GGUF"));
        push_u32(&mut encoded, 3);
        push_u64(&mut encoded, 0);
        push_u64(&mut encoded, 1);
        push_kv(
            &mut encoded,
            "test.nested",
            MetaValueType::Array,
            |encoded| {
                push_i32(encoded, MetaValueType::Array as i32);
                push_u64(encoded, 1);
                push_i32(encoded, MetaValueType::Uint32 as i32);
                push_u64(encoded, 1);
                push_u32(encoded, 7);
            },
        );
        while encoded.len() % 32 != 0 {
            encoded.push(0);
        }

        assert_eq!(
            parse_temp(&encoded).unwrap_err(),
            "Nested metadata arrays are not supported"
        );
    }

    #[test]
    fn metadata_count_must_fit_remaining_data() {
        let mut encoded = Vec::new();
        push_u32(&mut encoded, u32::from_le_bytes(*b"GGUF"));
        push_u32(&mut encoded, 3);
        push_u64(&mut encoded, 0);
        push_u64(&mut encoded, 1);
        let err = parse_temp(&encoded).unwrap_err();
        assert!(
            err.contains("metadata count exceeds remaining bytes"),
            "{err}"
        );
    }

    #[test]
    fn tensor_count_must_fit_remaining_data() {
        let mut encoded = Vec::new();
        push_u32(&mut encoded, u32::from_le_bytes(*b"GGUF"));
        push_u32(&mut encoded, 3);
        push_u64(&mut encoded, 1);
        push_u64(&mut encoded, 0);
        let err = parse_temp(&encoded).unwrap_err();
        assert!(
            err.contains("tensor count exceeds remaining bytes"),
            "{err}"
        );
    }

    #[test]
    fn tensor_dimension_count_must_fit_remaining_data() {
        let mut encoded = Vec::new();
        push_u32(&mut encoded, u32::from_le_bytes(*b"GGUF"));
        push_u32(&mut encoded, 3);
        push_u64(&mut encoded, 1);
        push_u64(&mut encoded, 0);
        push_str(&mut encoded, "");
        push_u32(&mut encoded, 2);
        encoded.extend_from_slice(&[0; 12]);
        let err = parse_temp(&encoded).unwrap_err();
        assert!(
            err.contains("tensor dimension count exceeds remaining bytes"),
            "{err}"
        );
    }

    #[test]
    fn test_gguf_invalid_magic() {
        let mut b = Vec::new();
        push_u32(&mut b, u32::from_le_bytes(*b"GGML"));
        push_u32(&mut b, 3);
        push_u64(&mut b, 0);
        push_u64(&mut b, 0);
        let err = parse_temp(&b).unwrap_err();
        assert!(err.contains("Invalid GGUF magic"), "got: {}", err);
    }

    #[test]
    fn test_gguf_bad_version() {
        let mut b = Vec::new();
        push_u32(&mut b, u32::from_le_bytes(*b"GGUF"));
        push_u32(&mut b, 1);
        push_u64(&mut b, 0);
        push_u64(&mut b, 0);
        let err = parse_temp(&b).unwrap_err();
        assert!(err.contains("Unsupported GGUF version"), "got: {}", err);
    }
}
