//! Read BF16/F16/F32 safetensors without copying model weights.

use crate::core::tensor::{GGMLType, MetaValue, TensorInfo, TensorSource};
use memmap2::{Mmap, MmapOptions};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

struct Entry {
    info: TensorInfo,
    file: usize,
    start: usize,
    end: usize,
}

pub struct SafetensorSource {
    files: Vec<Mmap>,
    tensors: HashMap<String, Entry>,
}

impl SafetensorSource {
    pub fn open(paths: &[impl AsRef<Path>]) -> Result<Self, String> {
        if paths.is_empty() {
            return Err("No safetensors files provided".into());
        }
        let mut source = Self {
            files: Vec::with_capacity(paths.len()),
            tensors: HashMap::new(),
        };
        for path in paths {
            let path = path.as_ref();
            let file = File::open(path).map_err(|e| format!("Open {}: {e}", path.display()))?;
            // SAFETY: the mapping is read-only and remains owned by Self.
            let map = unsafe { MmapOptions::new().map(&file) }
                .map_err(|e| format!("Map {}: {e}", path.display()))?;
            if map.len() < 8 {
                return Err(format!("Invalid safetensors header: {}", path.display()));
            }
            let header_len = usize::try_from(u64::from_le_bytes(map[..8].try_into().unwrap()))
                .map_err(|_| format!("Oversized safetensors header: {}", path.display()))?;
            if header_len > 100 * 1024 * 1024 || header_len > map.len() - 8 {
                return Err(format!(
                    "Invalid safetensors header length: {}",
                    path.display()
                ));
            }
            let header: Value = serde_json::from_slice(&map[8..8 + header_len])
                .map_err(|e| format!("Parse {}: {e}", path.display()))?;
            let entries = header
                .as_object()
                .ok_or_else(|| format!("Invalid safetensors table: {}", path.display()))?;
            let data_start = 8 + header_len;
            let file_index = source.files.len();
            for (name, tensor) in entries {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = match tensor.get("dtype").and_then(Value::as_str) {
                    Some("BF16") => GGMLType::BF16,
                    Some("F16") => GGMLType::F16,
                    Some("F32") => GGMLType::F32,
                    other => return Err(format!("Unsupported {name} dtype {other:?}")),
                };
                let shape = tensor
                    .get("shape")
                    .and_then(Value::as_array)
                    .ok_or_else(|| format!("Missing {name} shape"))?;
                let dims: Vec<u64> = shape
                    .iter()
                    .rev()
                    .map(|v| v.as_u64().ok_or_else(|| format!("Invalid {name} shape")))
                    .collect::<Result<_, _>>()?;
                let offsets = tensor
                    .get("data_offsets")
                    .and_then(Value::as_array)
                    .filter(|v| v.len() == 2)
                    .ok_or_else(|| format!("Missing {name} offsets"))?;
                let start = usize::try_from(offsets[0].as_u64().ok_or("Invalid tensor offset")?)
                    .map_err(|_| format!("Oversized {name} offset"))?;
                let end = usize::try_from(offsets[1].as_u64().ok_or("Invalid tensor offset")?)
                    .map_err(|_| format!("Oversized {name} offset"))?;
                let absolute_start = data_start
                    .checked_add(start)
                    .ok_or_else(|| format!("Overflowed {name} offset"))?;
                let absolute_end = data_start
                    .checked_add(end)
                    .ok_or_else(|| format!("Overflowed {name} offset"))?;
                let info = TensorInfo {
                    name: name.clone(),
                    dims,
                    ggml_type: dtype,
                    offset: start as u64,
                };
                if start > end
                    || absolute_end > map.len()
                    || info.checked_nbytes() != Some((end - start) as u64)
                {
                    return Err(format!("Invalid {name} tensor range or shape"));
                }
                if source
                    .tensors
                    .insert(
                        name.clone(),
                        Entry {
                            info,
                            file: file_index,
                            start: absolute_start,
                            end: absolute_end,
                        },
                    )
                    .is_some()
                {
                    return Err(format!("Duplicate safetensors tensor: {name}"));
                }
            }
            source.files.push(map);
        }
        Ok(source)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl TensorSource for SafetensorSource {
    fn metadata(&self, _key: &str) -> Option<&MetaValue> {
        None
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name).map(|entry| &entry.info)
    }

    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        let entry = self.tensors.get(name)?;
        self.files[entry.file].get(entry.start..entry.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_tensor_and_rejects_bad_range() {
        let path = std::env::temp_dir().join(format!(
            "rmi-safetensors-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let header = br#"{"weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend(0..12);
        std::fs::write(&path, &bytes).unwrap();
        let source = SafetensorSource::open(&[&path]).unwrap();
        assert_eq!(source.tensor_info("weight").unwrap().dims, [3, 2]);
        assert_eq!(
            source.tensor_slice("weight").unwrap(),
            &(0..12).collect::<Vec<_>>()
        );
        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(SafetensorSource::open(&[&path]).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "requires RMI_LONGCAT_COMPONENT_ROOT and local LongCat weights"]
    fn opens_real_longcat_encoder_and_vae() {
        let root = std::path::PathBuf::from(std::env::var("RMI_LONGCAT_COMPONENT_ROOT").unwrap());
        let text_files: Vec<_> = (1..=5)
            .map(|part| root.join(format!("text_encoder/model-{part:05}-of-00005.safetensors")))
            .collect();
        let text = SafetensorSource::open(&text_files).unwrap();
        assert_eq!(text.len(), 729);
        assert_eq!(
            text.tensor_info("model.embed_tokens.weight").unwrap().dims,
            [3584, 152064]
        );
        let vae = SafetensorSource::open(&[root.join("vae/diffusion_pytorch_model.safetensors")])
            .unwrap();
        assert_eq!(vae.len(), 244);
        assert_eq!(
            vae.tensor_info("encoder.conv_in.weight").unwrap().dims,
            [3, 3, 3, 128]
        );
    }
}
