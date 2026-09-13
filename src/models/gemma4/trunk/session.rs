use super::config::HEADS;
use super::scratch::Gemma4Scratch;
use super::weights::Gemma4Model;
use crate::core::scratchpad::KvFormat;

pub struct Gemma4Session<'model> {
    pub(super) model: &'model Gemma4Model,
    pub(super) kv: Vec<KvLayer>,
    pub(super) scratch: Gemma4Scratch,
    pub(super) seq_len: usize,
}

pub(super) struct KvLayer {
    /// Per-head dimension (single K/V head).
    pub(super) head_dim: usize,
    /// Per-position storage size: `kv_heads * head_dim` (GQA-aware).
    pub(super) row_width: usize,
    /// Q heads per KV head (= `q_heads / kv_heads`). For E2B: 8/1 = 8.
    /// For E4B: 8/2 = 4.
    pub(super) group_size: usize,
    pub(super) keys: Vec<f32>,
    pub(super) values: Vec<f32>,
}

impl<'model> Gemma4Session<'model> {
    pub fn new(model: &'model Gemma4Model, kv_format: KvFormat) -> Result<Self, String> {
        require_f32_kv(kv_format)?;
        let cfg = &model.config;
        let base = cfg.base_kv_layers();
        let kv = (0..base)
            .map(|layer| {
                let head_dim = cfg.head_dim(layer);
                KvLayer {
                    head_dim,
                    row_width: cfg.kv_heads * head_dim,
                    group_size: HEADS / cfg.kv_heads,
                    keys: Vec::new(),
                    values: Vec::new(),
                }
            })
            .collect();
        Ok(Self {
            model,
            kv,
            scratch: Gemma4Scratch::new(cfg),
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
        if key.len() != self.row_width || value.len() != self.row_width {
            return Err(format!(
                "blk.{layer} KV row length mismatch: key {}, value {}, expected {}",
                key.len(),
                value.len(),
                self.row_width
            ));
        }
        let expected = position
            .checked_mul(self.row_width)
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
