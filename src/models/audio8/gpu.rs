//! GPU compute operations for the Audio8 decoder: attention, RMS norm, SiLi.
//!
//! The matmul projections run through `LinearExecutor` / `BatchedLinearRuntime`.
//! This module handles the non-matmul operations that currently run on CPU:
//! attention (~260ms/step), RMS norm (~2ms/layer), and SiLU (~1ms/layer).

use crate::vulkan::ops::{
    ArenaRegion, AttentionRuntime, Qwen3Ops, TokenCommands, VulkanError,
};
use crate::vulkan::{GpuBuffer, VulkanContext};

const WIDTH: usize = 2048;
const HEAD_WIDTH: usize = 128;
const QUERY_HEADS: usize = 16;
const KV_HEADS: usize = 2;
const FFN_WIDTH: usize = 11008;

/// GPU compute context for Audio8 non-matmul operations.
///
/// Owns a `Qwen3Ops` arena and an `AttentionRuntime`. The arena holds the
//! intermediate buffers for RMS norm and SiLU; the attention runtime has its
/// own arena for Q/K/V.
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
    // Scratch for uploading norm weights.
    norm_weight_buffer: Option<GpuBuffer>,
}

impl Audio8GpuCompute {
    pub(crate) fn new(context: &'static VulkanContext) -> Result<Self, String> {
        // Arena size: norm_input (WIDTH) + norm_output (WIDTH) + norm_weight (WIDTH)
        // + silu_gate (FFN_WIDTH) + silu_up (FFN_WIDTH)
        let arena_size = (3 * WIDTH + 2 * FFN_WIDTH) * 4;
        let ops = Qwen3Ops::new_with_size(context, arena_size, 2)
            .map_err(|e| format!("failed to create Audio8 GPU compute arena: {e}"))?;

        let mut cursor = 0usize;
        let norm_input = f32_region(&mut cursor, WIDTH).map_err(|e| e.to_string())?;
        let norm_output = f32_region(&mut cursor, WIDTH).map_err(|e| e.to_string())?;
        let norm_weight_region = f32_region(&mut cursor, WIDTH).map_err(|e| e.to_string())?;
        let silu_gate = f32_region(&mut cursor, FFN_WIDTH).map_err(|e| e.to_string())?;
        let silu_up = f32_region(&mut cursor, FFN_WIDTH).map_err(|e| e.to_string())?;

        // Attention runtime: Q capacity = WIDTH * max_tokens, KV capacity = KV_HEADS * HEAD_WIDTH * max_tokens.
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
            norm_weight_buffer: None,
        })
    }

    /// Run GPU RMS norm: output[i] = weight[i] * (input[i] * inv_rms)
    ///
    /// Uses the `rms_norm` shader with a 64-lane tree reduction and Kahan
    /// correction, matching the CPU `audio_rms_norm` arithmetic order.
    pub(crate) fn rms_norm(
        &mut self,
        input: &[f32],
        weight: &[f32],
        output: &mut [f32],
        eps: f32,
    ) -> Result<(), String> {
        let count = input.len();
        if output.len() != count || weight.len() != count {
            return Err(format!(
                "Audio8 GPU RMS norm: dimension mismatch input={} output={} weight={}",
                input.len(),
                output.len(),
                weight.len(),
            ));
        }
        // Write input activation to the arena.
        self.ops
            .write_f32(self.norm_input, input)
            .map_err(|e| e.to_string())?;
        // Write norm weight to the arena (upload_f32 writes to arena via staging or mapping).
        self.ops
            .write_f32(self.norm_weight_region, weight)
            .map_err(|e| e.to_string())?;
        // The RMS norm shader expects binding[0] = arena, binding[1] = weight.
        // We need to create a descriptor set that includes both.
        // Since both are in the same arena buffer, we can use the arena_bindings
        // and pass the weight region offset in push constants.
        // Actually, the rms_norm shader has `binding = 1 readonly buffer Weight`.
        // We need a separate buffer for the weight. Let's upload it as a static buffer.
        let weight_bytes: Vec<u8> = weight
            .iter()
            .flat_map(|w| f32::to_le_bytes(*w))
            .collect();
        let weight_buffer = unsafe {
            self.ops
                .context
                .upload_static(&weight_bytes)
                .map_err(|e| e.to_string())?
        };
        self.norm_weight_buffer = Some(weight_buffer);
        let bindings = self
            .ops
            .bind_buffers(&[weight_buffer])
            .map_err(|e| e.to_string())?;
        let commands = TokenCommands::begin(self.ops.context).map_err(|e| e.to_string())?;
        self.ops
            .record_rms_norm_rows(
                &commands,
                bindings,
                self.norm_input,
                self.norm_output,
                count,
                eps,
                1,
                count,
                count,
            )
            .map_err(|e| e.to_string())?;
        commands
            .submit_and_wait()
            .map_err(|e| e.to_string())?;
        output.copy_from_slice(
            self.ops
                .read_f32(self.norm_output, count)
                .map_err(|e| e.to_string())?,
        );
        Ok(())
    }

    /// Run GPU SiLU(gate) * up, storing result in gate (in-place).
    pub(crate) fn silu_mul(
        &self,
        gate: &mut [f32],
        up: &[f32],
    ) -> Result<(), String> {
        let count = gate.len();
        if up.len() != count {
            return Err(format!(
                "Audio8 GPU SiLU: dimension mismatch gate={} up={}",
                gate.len(),
                up.len(),
            ));
        }
        self.ops
            .write_f32(self.silu_gate, gate)
            .map_err(|e| e.to_string())?;
        self.ops
            .write_f32(self.silu_up, up)
            .map_err(|e| e.to_string())?;
        let commands = TokenCommands::begin(self.ops.context).map_err(|e| e.to_string())?;
        self.ops
            .record_silu_mul_rows(&commands, self.silu_gate, self.silu_up, count, 1)
            .map_err(|e| e.to_string())?;
        commands
            .submit_and_wait()
            .map_err(|e| e.to_string())?;
        gate.copy_from_slice(
            self.ops
                .read_f32(self.silu_gate, count)
                .map_err(|e| e.to_string())?,
        );
        Ok(())
    }

    /// Run GPU causal attention with GQA.
    ///
    /// `query` is [QUERY_HEADS * HEAD_WIDTH] f32 values for the current token.
    /// `kv_cache_keys` is [(position+1) * KV_HEADS * HEAD_WIDTH] accumulated keys.
    /// `kv_cache_values` is [(position+1) * KV_HEADS * HEAD_WIDTH] accumulated values.
    /// `output` is [QUERY_HEADS * HEAD_WIDTH] f32 values (written).
    pub(crate) fn attention(
        &mut self,
        query: &[f32],
        kv_cache_keys: &[f32],
        kv_cache_values: &[f32],
        position: usize,
        output: &mut [f32],
    ) -> Result<(), String> {
        use crate::models::diffusion::dreamx::kernels::AttentionSpec;
        let key_tokens = position + 1;
        let spec = AttentionSpec {
            query_tokens: 1,
            key_tokens,
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
