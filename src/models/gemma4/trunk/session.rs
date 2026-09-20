use super::forward::Gemma4InputRow;
use super::scratch::Gemma4Scratch;
use super::weights::Gemma4Model;
use crate::core::prefill::{checked_prefill_batch_size, DEFAULT_PREFILL_BATCH_SIZE};
use crate::core::scratchpad::KvFormat;

pub struct Gemma4Session<'model> {
    pub(super) model: &'model Gemma4Model,
    pub(super) kv: Vec<KvLayer>,
    pub(super) scratch: Gemma4Scratch,
    pub(super) seq_len: usize,
    pub(super) prefill_batch_size: usize,
    pub(super) prefill_linear: Gemma4PrefillLinear,
}

#[derive(Default)]
pub(super) struct Gemma4PrefillLinear {
    #[cfg(feature = "vulkan")]
    pub(super) runtime: Option<crate::vulkan::ops::BatchedLinearRuntime>,
    #[cfg(all(test, feature = "vulkan"))]
    pub(super) dispatcher: Option<std::sync::Arc<std::sync::Mutex<super::tests::LinearDispatcher>>>,
}

impl Gemma4PrefillLinear {
    fn new(_model: &Gemma4Model, _rows: usize) -> Self {
        #[cfg(feature = "vulkan")]
        {
            use crate::vulkan::{ops::BatchedLinearRuntime, VulkanError};
            if !crate::ops::gpu_requested()
                || crate::core::thread_pool::gpu_matmul_disabled()
                || crate::vulkan::gpu_broken()
            {
                return Self::default();
            }
            let Some(context) = crate::ops::get_vulkan_context() else {
                crate::vulkan::mark_gpu_broken("Gemma4 prefill Vulkan initialization failed");
                return Self::default();
            };
            let n_ctx = _model.config.n_ctx;
            let result = Self::limits(_model).and_then(|(n_in, n_out, descriptors)| {
                BatchedLinearRuntime::new(context, _rows.min(n_ctx), n_in, n_out, descriptors)
            });
            return match result {
                Ok(runtime) => Self {
                    runtime: Some(runtime),
                    ..Self::default()
                },
                Err(VulkanError::UnsupportedShape(_)) => Self::default(),
                Err(error) => {
                    crate::vulkan::mark_gpu_broken(&error.to_string());
                    Self::default()
                }
            };
        }
        #[cfg(not(feature = "vulkan"))]
        Self::default()
    }

    #[cfg(feature = "vulkan")]
    pub(super) fn limits(
        model: &Gemma4Model,
    ) -> Result<(usize, usize, usize), crate::vulkan::VulkanError> {
        // Each visited projection has a distinct tensor label. Shared-KV layers
        // never project K/V; embeddings and the tied vocabulary output stay CPU.
        let per_layer_iter: Box<dyn Iterator<Item = &Weight<'static>>> =
            if let Some(p) = model.per_layer_model_proj.as_ref() {
                Box::new(std::iter::once(p))
            } else {
                Box::new(std::iter::empty())
            };
        per_layer_iter
            .chain(model.layers.iter().enumerate().flat_map(|(index, layer)| {
                let base_kv = model.config.base_kv_layers();
                let mut items: Vec<&Weight<'static>> = vec![
                    &layer.attn_q,
                    &layer.attn_output,
                    &layer.ffn_gate,
                    &layer.ffn_up,
                    &layer.ffn_down,
                ];
                if let Some(ig) = layer.inp_gate.as_ref() {
                    items.push(ig);
                }
                if let Some(pj) = layer.proj.as_ref() {
                    items.push(pj);
                }
                if index < base_kv {
                    items.push(&layer.attn_k);
                    if let Some(v) = layer.attn_v.as_ref() {
                        items.push(v);
                    }
                }
                items.into_iter()
            }))
            .try_fold((0, 0, 1usize), |(n_in, n_out, descriptors), weight| {
                Ok((
                    n_in.max(weight.n_in),
                    n_out.max(weight.n_out),
                    descriptors
                        .checked_add(1)
                        .ok_or(crate::vulkan::VulkanError::OutOfMemory)?,
                ))
            })
    }

    pub(super) fn active(&self) -> bool {
        #[cfg(feature = "vulkan")]
        {
            if crate::core::thread_pool::gpu_matmul_disabled() || crate::vulkan::gpu_broken() {
                return false;
            }
            #[cfg(test)]
            if self.dispatcher.is_some() {
                return true;
            }
            return self.runtime.is_some();
        }
        #[cfg(not(feature = "vulkan"))]
        false
    }

    #[cfg(feature = "vulkan")]
    pub(super) fn finish_dispatch(
        &mut self,
        result: Result<(), crate::vulkan::VulkanError>,
    ) -> bool {
        match result {
            Ok(()) => true,
            Err(crate::vulkan::VulkanError::UnsupportedShape(_)) => false,
            Err(error) => {
                #[cfg(test)]
                let injected = if let Some(dispatcher) = self.dispatcher.take() {
                    dispatcher.lock().unwrap().failures += 1;
                    true
                } else {
                    false
                };
                #[cfg(not(test))]
                let injected = false;
                if !injected {
                    crate::vulkan::mark_gpu_broken(&error.to_string());
                }
                self.runtime = None;
                false
            }
        }
    }
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
        Self::new_with_prefill_batch_size(model, kv_format, DEFAULT_PREFILL_BATCH_SIZE)
    }

    pub fn new_with_prefill_batch_size(
        model: &'model Gemma4Model,
        kv_format: KvFormat,
        prefill_batch_size: usize,
    ) -> Result<Self, String> {
        require_f32_kv(kv_format)?;
        let prefill_batch_size = checked_prefill_batch_size(Some(prefill_batch_size))?;
        let cfg = &model.config;
        let base = cfg.base_kv_layers();
        let kv = (0..base)
            .map(|layer| {
                let head_dim = cfg.head_dim(layer);
                let kv_heads = cfg.kv_heads(layer);
                KvLayer {
                    head_dim,
                    row_width: kv_heads * head_dim,
                    group_size: cfg.n_heads / kv_heads,
                    keys: Vec::new(),
                    values: Vec::new(),
                }
            })
            .collect();
        Ok(Self {
            model,
            kv,
            scratch: Gemma4Scratch::new(cfg, prefill_batch_size.min(cfg.n_ctx)),
            seq_len: 0,
            prefill_batch_size,
            prefill_linear: Gemma4PrefillLinear::new(model, prefill_batch_size),
        })
    }

    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Reuse the scratch and projection runtime for an unrelated prompt.
    pub fn reset(&mut self) {
        self.seq_len = 0;
        for layer in &mut self.kv {
            layer.keys.clear();
            layer.values.clear();
        }
    }

    pub fn scratch_bytes(&self) -> usize {
        self.scratch.bytes()
    }

    /// Single forward pass: prefill a token list and return the
    /// last-position logits. Used by JEV / classification modes that do
    /// not need autoregressive decoding.
    pub fn forward_logits(&mut self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() {
            return Err("Gemma4 prompt must contain at least one token".into());
        }
        let rows: Vec<Gemma4InputRow> = token_ids
            .iter()
            .map(|&t| Gemma4InputRow::Token(t))
            .collect();
        self.forward_rows(&rows)
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
