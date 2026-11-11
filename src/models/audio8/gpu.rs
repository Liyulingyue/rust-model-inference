//! GPU compute operations for the Audio8 decoder: attention, RMS norm, SiLU.
//!
//! The matmul projections run through `LinearExecutor` / `BatchedLinearRuntime`.
//! This module handles the non-matmul operations that used to run on CPU:
//! attention (~260ms/step), RMS norm (~2ms/layer), and SiLU (~1ms/layer).

use crate::vulkan::ops::{
    ArenaRegion, AttentionRuntime, OperatorBindings, Qwen3Ops, TokenCommands,
};
use crate::vulkan::{GpuBuffer, VulkanContext, VulkanError};
use std::collections::HashMap;

const WIDTH: usize = 2048;
const HEAD_WIDTH: usize = 128;
const QUERY_HEADS: usize = 16;
const KV_HEADS: usize = 2;
const FFN_WIDTH: usize = 11008;

/// GPU compute context for Audio8 non-matmul operations.
pub(crate) struct Audio8GpuCompute {
    ops: Qwen3Ops<'static>,
    attention: AttentionRuntime,
    // Arena regions for RMS norm.
    norm_input: ArenaRegion,
    norm_output: ArenaRegion,
    norm_weight_region: ArenaRegion,
    // Arena regions for SiLU.
    silu_gate: ArenaRegion,
    silu_up: ArenaRegion,
    // Cached norm weight bindings, keyed by weight data pointer.
    // Avoids creating a new descriptor set (from a fixed-size pool) per call.
    norm_bindings_cache: HashMap<usize, (GpuBuffer, OperatorBindings)>,
}

impl Audio8GpuCompute {
    pub(crate) fn new(context: &'static VulkanContext) -> Result<Self, String> {
        let arena_size = (3 * WIDTH + 2 * FFN_WIDTH) * 4;
        let ops = Qwen3Ops::new_with_size(context, arena_size, 128)
            .map_err(|e| format!("failed to create Audio8 GPU compute arena: {e}"))?;

        let mut cursor = 0usize;
        let norm_input = f32_region(&mut cursor, WIDTH).map_err(|e| e.to_string())?;
        let norm_output = f32_region(&mut cursor, WIDTH).map_err(|e| e.to_string())?;
        let norm_weight_region = f32_region(&mut cursor, WIDTH).map_err(|e| e.to_string())?;
        let silu_gate = f32_region(&mut cursor, FFN_WIDTH).map_err(|e| e.to_string())?;
        let silu_up = f32_region(&mut cursor, FFN_WIDTH).map_err(|e| e.to_string())?;

        const MAX_TOKENS: usize = 1500;
        let attn_q_capacity = WIDTH * MAX_TOKENS;
        let attn_kv_capacity = KV_HEADS * HEAD_WIDTH * MAX_TOKENS;
        let attention = AttentionRuntime::new(context, (attn_q_capacity, attn_kv_capacity))
            .map_err(|e| format!("failed to create Audio8 attention runtime: {e}"))?;

        Ok(Self {
            ops,
            attention,
            norm_input,
            norm_output,
            norm_weight_region,
            silu_gate,
            silu_up,
            norm_bindings_cache: HashMap::new(),
        })
    }

    // ── GPU ops (begin + submit per op) ─────────────────────────────────

    /// Run GPU RMS norm: output[i] = weight[i] * (input[i] * inv_rms)
    ///
    /// Caches the weight buffer and descriptor bindings per unique weight
    /// slice, so the fixed-size descriptor pool is not exhausted by repeated
    /// calls with the same weights (e.g. Audio8's 72 norm weights).
    pub(crate) fn rms_norm(
        &mut self,
        input: &[f32],
        weight: &[f32],
        output: &mut [f32],
        eps: f32,
    ) -> Result<(), String> {
        let count = input.len();
        if weight.len() != count {
            return Err("Audio8 GPU RMS norm: weight length mismatch".into());
        }
        let _ = eps;
        self.ops
            .write_f32(self.norm_input, input)
            .map_err(|e| e.to_string())?;
        // Cache key: weight data pointer (stable for the model's lifetime).
        let key = weight.as_ptr() as usize;
        let bindings = if let Some((_buffer, bindings)) = self.norm_bindings_cache.get(&key) {
            *bindings
        } else {
            let weight_buffer = unsafe {
                let weight_bytes: Vec<u8> =
                    weight.iter().flat_map(|w| f32::to_le_bytes(*w)).collect();
                self.ops
                    .context()
                    .upload_static(&weight_bytes)
                    .map_err(|e| e.to_string())?
            };
            let bindings = self
                .ops
                .bind_buffers(&[weight_buffer])
                .map_err(|e| e.to_string())?;
            self.norm_bindings_cache
                .insert(key, (weight_buffer, bindings));
            bindings
        };
        let commands = TokenCommands::begin(self.ops.context()).map_err(|e| e.to_string())?;
        self.ops
            .record_rms_norm_rows(
                &commands,
                bindings,
                self.norm_input,
                self.norm_output,
                count,
                1e-6,
                1,
                count,
                count,
            )
            .map_err(|e| e.to_string())?;
        commands.submit_and_wait().map_err(|e| e.to_string())?;
        output.copy_from_slice(
            self.ops
                .read_f32(self.norm_output, count)
                .map_err(|e| e.to_string())?,
        );
        Ok(())
    }

    /// Run GPU SiLU(gate) * up, storing result in gate (in-place).
    pub(crate) fn silu_mul(&self, gate: &mut [f32], up: &[f32]) -> Result<(), String> {
        let count = gate.len();
        if gate.len() != up.len() {
            return Err("Audio8 GPU SiLU: dimension mismatch".into());
        }
        self.ops
            .write_f32(self.silu_gate, gate)
            .map_err(|e| e.to_string())?;
        self.ops
            .write_f32(self.silu_up, up)
            .map_err(|e| e.to_string())?;
        let commands = TokenCommands::begin(self.ops.context()).map_err(|e| e.to_string())?;
        self.ops
            .record_silu_mul_rows(&commands, self.silu_gate, self.silu_up, count, 1)
            .map_err(|e| e.to_string())?;
        commands.submit_and_wait().map_err(|e| e.to_string())?;
        gate.copy_from_slice(
            self.ops
                .read_f32(self.silu_gate, count)
                .map_err(|e| e.to_string())?,
        );
        Ok(())
    }

    /// Run GPU causal attention with GQA.
    pub(crate) fn attention(
        &mut self,
        query: &[f32],
        kv_cache_keys: &[f32],
        kv_cache_values: &[f32],
        position: usize,
        output: &mut [f32],
    ) -> Result<(), String> {
        use crate::ops::nn::AttentionSpec;
        let spec = AttentionSpec {
            query_tokens: 1,
            key_tokens: position + 1,
            query_heads: QUERY_HEADS,
            key_value_heads: KV_HEADS,
            head_dim: HEAD_WIDTH,
            causal: true,
            scale: 1.0 / (HEAD_WIDTH as f32).sqrt(),
        };
        self.attention
            .forward(query, kv_cache_keys, kv_cache_values, output, spec)
            .map_err(|e| e.to_string())?;
        Ok(())
    }
}

fn f32_region(cursor: &mut usize, elements: usize) -> Result<ArenaRegion, VulkanError> {
    let size = elements
        .checked_mul(4)
        .ok_or_else(|| VulkanError::UnsupportedShape("region size overflow".into()))?;
    let offset = cursor
        .checked_add(15)
        .map(|v| v & !15)
        .ok_or_else(|| VulkanError::UnsupportedShape("region offset overflow".into()))?;
    *cursor = offset
        .checked_add(size)
        .ok_or_else(|| VulkanError::UnsupportedShape("region end overflow".into()))?;
    Ok(ArenaRegion { offset, size })
}
