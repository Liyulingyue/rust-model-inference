use super::config::BASE_KV_LAYERS;
use super::scratch::Gemma4Scratch;
use super::weights::{head_dim, Gemma4Model};
use crate::core::scratchpad::KvFormat;

pub struct Gemma4Session<'model> {
    pub(super) model: &'model Gemma4Model,
    pub(super) kv: Vec<KvLayer>,
    pub(super) scratch: Gemma4Scratch,
    pub(super) seq_len: usize,
}

pub(super) struct KvLayer {
    pub(super) head_dim: usize,
    pub(super) keys: Vec<f32>,
    pub(super) values: Vec<f32>,
}

impl<'model> Gemma4Session<'model> {
    pub fn new(model: &'model Gemma4Model, kv_format: KvFormat) -> Result<Self, String> {
        require_f32_kv(kv_format)?;
        let kv = (0..BASE_KV_LAYERS)
            .map(|layer| KvLayer {
                head_dim: head_dim(layer),
                keys: Vec::new(),
                values: Vec::new(),
            })
            .collect();
        Ok(Self {
            model,
            kv,
            scratch: Gemma4Scratch::new(),
            seq_len: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.seq_len
    }
}

impl KvLayer {
    pub(super) fn append(
        &mut self,
        layer: usize,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), String> {
        if key.len() != self.head_dim || value.len() != self.head_dim {
            return Err(format!(
                "blk.{layer} KV row length mismatch: key {}, value {}, expected {}",
                key.len(),
                value.len(),
                self.head_dim
            ));
        }
        let expected = position
            .checked_mul(self.head_dim)
            .ok_or_else(|| format!("blk.{layer} KV length overflow"))?;
        if self.keys.len() != expected || self.values.len() != expected {
            return Err(format!(
                "blk.{layer} KV context mismatch at position {position}: key {}, value {}, expected {expected}",
                self.keys.len(),
                self.values.len()
            ));
        }
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        Ok(())
    }
}

pub(super) fn require_f32_kv(kv_format: KvFormat) -> Result<(), String> {
    if kv_format != KvFormat::F32 {
        return Err("Gemma4 incremental session requires an F32 KV cache".into());
    }
    Ok(())
}
