//! Vulkan session for the YuE2 autoregressive decoder.
//!
//! YuE2's AR half is an ordinary dense transformer -- no MoE, no SSM, no
//! fused QKV -- so it reuses the operator set `qwen3.rs` already records. The
//! per-layer order mirrors `YuE2ArSession::decode_step` in
//! `src/models/yue2/ar.rs`:
//!
//! ```text
//! rms_norm(attn_norm) -> qkv -> qk_norm_rope -> kv_write -> attention
//!   -> o_proj -> residual add
//!   -> rms_norm(mlp_norm) -> gate_up -> silu_mul -> down -> residual add
//! ```
//!
//! The `lm_head` is 2048 x 184704, which is 20% of the AR FLOPs for a single
//! token, so it is recorded on the device like any other projection rather
//! than gathered back to the CPU.
//!
//! Only the AR stream is covered. The NAR half is a flow-matching diffusion
//! solve whose weights must stay BF16 (see `models/YuE2-gguf/samples/README.md`),
//! and the VAE is a separate F32 model; neither is eligible here.

use super::ops::{
    fill_rope_neox, ArenaLayout, ArenaRegion, GpuWeightFormat, OperatorBindings, Qwen3Ops,
    TokenCommands,
};
use super::{GpuBuffer, VulkanContext, VulkanError};
use crate::core::tensor::GGMLType;
use crate::models::yue2::{YuE2Config, YuE2Model};

/// Ceiling on the device arena. Measured on this GB10: host-coherent buffers
/// up to 2 GB allocate fine and the free heap is 91.3 GiB, so this is a
/// conservative guard rather than a measured ceiling -- it exists so a
/// pathological capacity falls back to the CPU instead of failing the run.
const DEVICE_ARENA_BUDGET: usize = 2 << 30;

/// Weight types the converter can emit for a YuE2 2-D projection.
const MATRIX_TYPES: [GGMLType; 7] = [
    GGMLType::BF16,
    GGMLType::F16,
    GGMLType::F32,
    GGMLType::Q8_0,
    GGMLType::Q4_0,
    GGMLType::Q4K,
    GGMLType::Q6K,
];

/// The q4_k_m policy puts 6-bit blocks on `v_proj` and `down_proj` and 4-bit
/// blocks elsewhere, so Q/K and V are not the same format and cannot share one
/// grouped binding. Mirrors `QkvBindings` in `qwen3.rs`.
enum QkvBindings {
    Grouped(OperatorBindings),
    Split([OperatorBindings; 3]),
}

/// gate and up are both 4-bit under the q4_k_m policy, so this is grouped in
/// practice; the split arm exists because the checker requires the two to
/// agree before it will bind them together.
enum GateUpBindings {
    Grouped(OperatorBindings),
    Split([OperatorBindings; 2]),
}

struct LayerBindings {
    attn_norm: OperatorBindings,
    qkv: QkvBindings,
    qk_norm: OperatorBindings,
    wo: OperatorBindings,
    mlp_norm: OperatorBindings,
    gate_up: GateUpBindings,
    down: OperatorBindings,
}

pub(crate) struct UploadedBuffers {
    context: &'static VulkanContext,
    values: Vec<GpuBuffer>,
}

impl UploadedBuffers {
    fn new(context: &'static VulkanContext) -> Self {
        Self {
            context,
            values: Vec::new(),
        }
    }

    fn upload(&mut self, bytes: &[u8]) -> Result<GpuBuffer, VulkanError> {
        let buffer = unsafe { self.context.upload_static(bytes)? };
        self.values.push(buffer);
        Ok(buffer)
    }

    fn upload_f32(&mut self, values: &[f32]) -> Result<GpuBuffer, VulkanError> {
        self.upload(bytemuck::cast_slice(values))
    }

    fn upload_tensor(&mut self, model: &YuE2Model, name: &str) -> Result<GpuBuffer, VulkanError> {
        let source = model.tensor_source().ok_or_else(|| {
            VulkanError::UnsupportedShape("YuE2 model has no tensor source".into())
        })?;
        let bytes = source
            .tensor_slice(name)
            .ok_or_else(|| VulkanError::UnsupportedShape(format!("missing YuE2 tensor {name}")))?;
        self.upload(bytes)
    }
}

impl Drop for UploadedBuffers {
    fn drop(&mut self) {
        let _ = unsafe { self.context.destroy_completed_buffers(&self.values) };
    }
}

pub(crate) struct YuE2GpuChunkResult<'a> {
    pub(crate) logits: &'a [f32],
    pub(crate) k_delta: &'a [f32],
    pub(crate) v_delta: &'a [f32],
}

pub(crate) struct YuE2VulkanSession<'model> {
    context: &'static VulkanContext,
    ops: Qwen3Ops<'static>,
    _buffers: UploadedBuffers,
    _model: std::marker::PhantomData<&'model YuE2Model>,
    layers: Vec<LayerBindings>,
    final_norm: OperatorBindings,
    lm_head: OperatorBindings,
    layout: ArenaLayout,
    config: YuE2Config,
    capacity: usize,
    max_rows: usize,
    rope: Vec<f32>,
    logits: Vec<f32>,
    k_delta: Vec<f32>,
    v_delta: Vec<f32>,
}

impl<'model> YuE2VulkanSession<'model> {
    pub(crate) fn try_new(
        model: &'model YuE2Model,
        capacity: usize,
        context: &'static VulkanContext,
    ) -> Result<Option<Self>, VulkanError> {
        if !context.supports_shader_float16() {
            eprintln!("[GPU] YuE2 Vulkan unavailable: device lacks shaderFloat16. CPU fallback.");
            return Ok(None);
        }
        let config = model.config().clone();
        if config.q_heads == 0
            || config.kv_heads == 0
            || config.head_dim == 0
            || config.head_dim % 2 != 0
            || config.q_heads % config.kv_heads != 0
        {
            eprintln!("[GPU] YuE2 Vulkan unavailable: unsupported attention shape. CPU fallback.");
            return Ok(None);
        }
        // The AR weights are the only ones on the device; the NAR half stays in
        // host memory and is never uploaded, which is why this only inspects
        // `ar_attention` / `ar_mlp`.
        if let Err(reason) = check_weight_types(model) {
            eprintln!("[GPU] YuE2 Vulkan unavailable: {reason}. Falling back to CPU.");
            return Ok(None);
        }

        let max_rows = capacity.min(crate::core::prefill::DEFAULT_PREFILL_BATCH_SIZE);
        let layout = ArenaLayout::yue2(&config, capacity, max_rows)?;
        // The arena holds an F32 K/V cache for every layer, so its size is
        // `28 * capacity * kv_heads * head_dim * 4 * 2` bytes -- roughly 950 MB
        // at the ABC phase's `prompt + 4096` capacity. On a device that cannot
        // back that, fall back to the CPU rather than failing the run; the CPU
        // path is the reference and is merely slower.
        if layout.total_size() > DEVICE_ARENA_BUDGET {
            eprintln!(
                "[GPU] YuE2 Vulkan skipped: arena needs {} MB, budget is {} MB. CPU fallback.",
                layout.total_size() / (1 << 20),
                DEVICE_ARENA_BUDGET / (1 << 20),
            );
            return Ok(None);
        }
        // One descriptor set per bind call. A layer binds attn_norm, qkv,
        // qk_norm, wo, mlp_norm, gate_up and down; qkv and gate_up each cost
        // one set when their formats match and three when they do not, which is
        // the q4_k_m case (Q6_K on v_proj). Budget the worst case so the pool
        // cannot run dry partway through, plus final_norm, lm_head, and the one
        // `Qwen3Ops::new` spends on the arena.
        let per_layer = 7 + 2 + 2;
        let descriptor_capacity = config
            .layers
            .checked_mul(per_layer)
            .and_then(|count| count.checked_add(3))
            .ok_or(VulkanError::OutOfMemory)?;
        let mut ops = Qwen3Ops::new(context, layout, descriptor_capacity)?;
        let mut buffers = UploadedBuffers::new(context);
        let mut layers = Vec::with_capacity(config.layers);

        for (layer_index, layer) in model.ar_layers().iter().enumerate() {
            let attn_norm = ops.bind_buffers(&[buffers.upload_f32(&layer.ar_attention.norm)?])?;
            let base = format!("model.layers.{layer_index}");
            let qkv_buffers = [
                buffers.upload_tensor(model, &format!("{base}.self_attn.q_proj.weight"))?,
                buffers.upload_tensor(model, &format!("{base}.self_attn.k_proj.weight"))?,
                buffers.upload_tensor(model, &format!("{base}.self_attn.v_proj.weight"))?,
            ];
            let qkv_formats = [
                GpuWeightFormat::from_ggml_type(layer.ar_attention.q.ggml_type())?,
                GpuWeightFormat::from_ggml_type(layer.ar_attention.k.ggml_type())?,
                GpuWeightFormat::from_ggml_type(layer.ar_attention.v.ggml_type())?,
            ];
            let qkv = if qkv_formats.iter().all(|format| *format == qkv_formats[0]) {
                QkvBindings::Grouped(ops.bind_weight_buffers(&qkv_buffers, &qkv_formats)?)
            } else {
                QkvBindings::Split([
                    ops.bind_weight_buffers(&qkv_buffers[0..1], &qkv_formats[0..1])?,
                    ops.bind_weight_buffers(&qkv_buffers[1..2], &qkv_formats[1..2])?,
                    ops.bind_weight_buffers(&qkv_buffers[2..3], &qkv_formats[2..3])?,
                ])
            };
            let qk_norm_buffers = [
                buffers.upload_f32(&layer.ar_attention.q_norm)?,
                buffers.upload_f32(&layer.ar_attention.k_norm)?,
            ];
            let qk_norm = ops.bind_buffers(&qk_norm_buffers)?;
            let wo = ops.bind_weight_buffers(
                &[buffers.upload_tensor(model, &format!("{base}.self_attn.o_proj.weight"))?],
                &[GpuWeightFormat::from_ggml_type(
                    layer.ar_attention.output.ggml_type(),
                )?],
            )?;
            let mlp_norm = ops.bind_buffers(&[buffers.upload_f32(&layer.ar_mlp.norm)?])?;
            let gate_up_buffers = [
                buffers.upload_tensor(model, &format!("{base}.mlp.gate_proj.weight"))?,
                buffers.upload_tensor(model, &format!("{base}.mlp.up_proj.weight"))?,
            ];
            let gate_up_formats = [
                GpuWeightFormat::from_ggml_type(layer.ar_mlp.gate.ggml_type())?,
                GpuWeightFormat::from_ggml_type(layer.ar_mlp.up.ggml_type())?,
            ];
            let gate_up = if gate_up_formats[0] == gate_up_formats[1] {
                GateUpBindings::Grouped(
                    ops.bind_weight_buffers(&gate_up_buffers, &gate_up_formats)?,
                )
            } else {
                GateUpBindings::Split([
                    ops.bind_weight_buffers(&gate_up_buffers[0..1], &gate_up_formats[0..1])?,
                    ops.bind_weight_buffers(&gate_up_buffers[1..2], &gate_up_formats[1..2])?,
                ])
            };
            let down = ops.bind_weight_buffers(
                &[buffers.upload_tensor(model, &format!("{base}.mlp.down_proj.weight"))?],
                &[GpuWeightFormat::from_ggml_type(
                    layer.ar_mlp.down.ggml_type(),
                )?],
            )?;
            layers.push(LayerBindings {
                attn_norm,
                qkv,
                qk_norm,
                wo,
                mlp_norm,
                gate_up,
                down,
            });
        }

        let final_norm = ops.bind_buffers(&[buffers.upload_f32(model.ar_final_norm())?])?;
        let lm_head = ops.bind_weight_buffers(
            &[buffers.upload_tensor(model, "lm_head.weight")?],
            &[GpuWeightFormat::from_ggml_type(
                model.ar_lm_head().ggml_type(),
            )?],
        )?;

        let kv_count = config
            .kv_heads
            .checked_mul(config.head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let delta_count = config
            .layers
            .checked_mul(kv_count)
            .and_then(|count| count.checked_mul(max_rows))
            .ok_or(VulkanError::OutOfMemory)?;

        let rope = vec![
            0.0;
            config
                .head_dim
                .checked_mul(max_rows)
                .ok_or(VulkanError::OutOfMemory)?
        ];
        let logits = vec![0.0; config.vocab];
        let k_delta = vec![0.0; delta_count];
        let v_delta = vec![0.0; delta_count];

        Ok(Some(Self {
            context,
            ops,
            _buffers: buffers,
            _model: std::marker::PhantomData,
            layers,
            final_norm,
            lm_head,
            layout,
            config,
            capacity,
            max_rows,
            rope,
            logits,
            k_delta,
            v_delta,
        }))
    }

    /// Decode `rows` tokens starting at `base_position` and return the logits
    /// for the last row plus the K/V deltas the caller must commit to its own
    /// cache.
    pub(crate) fn forward_chunk<'a>(
        &'a mut self,
        input: &[f32],
        base_position: usize,
        rows: usize,
    ) -> Result<YuE2GpuChunkResult<'a>, VulkanError> {
        if rows == 0 || rows > self.max_rows {
            return Err(VulkanError::UnsupportedShape(format!(
                "YuE2 GPU rows={rows} outside 1..={}",
                self.max_rows
            )));
        }
        if base_position + rows > self.capacity {
            return Err(VulkanError::UnsupportedShape(format!(
                "YuE2 GPU position {base_position}+{rows} exceeds capacity {}",
                self.capacity
            )));
        }
        if input.len() != rows * self.config.hidden || input.iter().any(|value| !value.is_finite())
        {
            return Err(VulkanError::UnsupportedShape(format!(
                "invalid YuE2 chunk input len={} rows={rows}",
                input.len()
            )));
        }
        let config = &self.config;
        let q_count = config
            .q_heads
            .checked_mul(config.head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let kv_count = config
            .kv_heads
            .checked_mul(config.head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let delta_count = config
            .layers
            .checked_mul(kv_count)
            .and_then(|count| count.checked_mul(rows))
            .ok_or(VulkanError::OutOfMemory)?;

        let commands = TokenCommands::begin(self.context)?;
        self.ops.write_f32(self.layout.x, input)?;
        for row in 0..rows {
            fill_rope_neox(
                &mut self.rope[row * config.head_dim..(row + 1) * config.head_dim],
                base_position + row,
                config.rope_base,
            );
        }
        self.ops
            .write_f32(self.layout.rope, &self.rope[..rows * config.head_dim])?;

        for (layer_index, bindings) in self.layers.iter().enumerate() {
            self.ops.record_rms_norm_rows(
                &commands,
                bindings.attn_norm,
                self.layout.x,
                self.layout.normed,
                config.hidden,
                config.rms_eps,
                rows,
                config.hidden,
                config.hidden,
            )?;
            let qkv_outputs = [
                (self.layout.q, q_count),
                (self.layout.k, kv_count),
                (self.layout.v, kv_count),
            ];
            match bindings.qkv {
                QkvBindings::Grouped(grouped) => self.record_weights(
                    &commands,
                    grouped,
                    self.layout.normed,
                    &qkv_outputs,
                    config.hidden,
                    rows,
                )?,
                QkvBindings::Split(split) => {
                    for (binding, &output) in split.iter().zip(qkv_outputs.iter()) {
                        self.record_weights(
                            &commands,
                            *binding,
                            self.layout.normed,
                            &[output],
                            config.hidden,
                            rows,
                        )?;
                    }
                }
            }
            self.ops.record_qk_norm_rope_rows(
                &commands,
                bindings.qk_norm,
                self.layout.q,
                self.layout.k,
                config.q_heads,
                config.kv_heads,
                config.head_dim,
                self.layout.rope,
                config.rms_eps,
                true,
                true,
                rows,
            )?;
            self.ops.record_kv_write_rows(
                &commands,
                self.layout.k,
                self.layout.v,
                self.layout.kv_k,
                self.layout.kv_v,
                self.layout.kv_delta_k,
                self.layout.kv_delta_v,
                layer_index,
                base_position,
                config.layers,
                self.capacity,
                kv_count,
                rows,
                kv_count,
            )?;
            self.ops.record_attention_rows(
                &commands,
                self.layout.q,
                self.layout.kv_k,
                self.layout.kv_v,
                self.layout.scores,
                self.layout.attn,
                layer_index,
                config.layers,
                base_position,
                self.capacity,
                config.q_heads,
                config.kv_heads,
                config.head_dim,
                rows,
            )?;
            self.record_weights(
                &commands,
                bindings.wo,
                self.layout.attn,
                &[(self.layout.projection, config.hidden)],
                q_count,
                rows,
            )?;
            self.ops.record_add_rows(
                &commands,
                self.layout.x,
                self.layout.projection,
                config.hidden,
                rows,
            )?;
            self.ops.record_rms_norm_rows(
                &commands,
                bindings.mlp_norm,
                self.layout.x,
                self.layout.normed,
                config.hidden,
                config.rms_eps,
                rows,
                config.hidden,
                config.hidden,
            )?;
            let gate_up_outputs = [(self.layout.gate, config.ffn), (self.layout.up, config.ffn)];
            match bindings.gate_up {
                GateUpBindings::Grouped(grouped) => self.record_weights(
                    &commands,
                    grouped,
                    self.layout.normed,
                    &gate_up_outputs,
                    config.hidden,
                    rows,
                )?,
                GateUpBindings::Split(split) => {
                    for (binding, &output) in split.iter().zip(gate_up_outputs.iter()) {
                        self.record_weights(
                            &commands,
                            *binding,
                            self.layout.normed,
                            &[output],
                            config.hidden,
                            rows,
                        )?;
                    }
                }
            }
            self.ops.record_silu_mul_rows(
                &commands,
                self.layout.gate,
                self.layout.up,
                config.ffn,
                rows,
            )?;
            self.record_weights(
                &commands,
                bindings.down,
                self.layout.gate,
                &[(self.layout.down, config.hidden)],
                config.ffn,
                rows,
            )?;
            self.ops.record_add_rows(
                &commands,
                self.layout.x,
                self.layout.down,
                config.hidden,
                rows,
            )?;
        }

        // Only the last row needs logits: sampling is autoregressive, so the
        // 184704-wide lm_head runs once per chunk rather than once per row.
        let last_offset = (rows - 1)
            .checked_mul(config.hidden)
            .and_then(|count| count.checked_mul(4))
            .ok_or(VulkanError::OutOfMemory)?;
        let last = ArenaRegion {
            offset: self
                .layout
                .x
                .offset
                .checked_add(last_offset)
                .ok_or(VulkanError::OutOfMemory)?,
            size: self
                .layout
                .x
                .size
                .checked_sub(last_offset)
                .ok_or(VulkanError::OutOfMemory)?,
        };
        self.ops.record_rms_norm(
            &commands,
            self.final_norm,
            last,
            self.layout.normed,
            config.hidden,
            config.rms_eps,
        )?;
        self.record_weights(
            &commands,
            self.lm_head,
            self.layout.normed,
            &[(self.layout.logits, config.vocab)],
            config.hidden,
            1,
        )?;
        commands.submit_and_wait()?;
        self.ops
            .read_f32(self.layout.logits, self.config.vocab)
            .map(|logits| {
                self.logits.copy_from_slice(&logits);
            })?;

        Ok(YuE2GpuChunkResult {
            logits: &self.logits,
            k_delta: &self.k_delta[..delta_count],
            v_delta: &self.v_delta[..delta_count],
        })
    }

    fn record_weights(
        &self,
        commands: &TokenCommands<'_>,
        bindings: OperatorBindings,
        input: ArenaRegion,
        outputs: &[(ArenaRegion, usize)],
        n_in: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let outputs = outputs
            .iter()
            .map(|&(region, width)| {
                Ok((
                    region,
                    width,
                    width.checked_mul(4).ok_or(VulkanError::OutOfMemory)?,
                ))
            })
            .collect::<Result<Vec<_>, VulkanError>>()?;
        self.ops.record_weight_matmul_rows(
            commands,
            bindings,
            input,
            self.layout.q8,
            self.layout.q8_scales,
            self.layout.q4_1_input_sums,
            self.layout.q8k,
            self.layout.q8k_scales,
            &outputs,
            n_in,
            rows,
            n_in,
        )
    }
}

/// Copy the device's K/V deltas into the CPU shadow cache that
/// `YuE2ArSession` reads for the NAR solve.
///
/// YuE2 has no conv/SSM shadow state (unlike Qwen3.5), so this is the KV half
/// of `commit_shadow_state_chunk` and nothing more. The NAR prefill re-runs the
/// AR prefix itself, so the cache only has to be valid for the positions the
/// NAR attends over.
pub(crate) fn commit_kv_shadow(
    kv_cache: &mut crate::core::scratchpad::KvCache,
    position: usize,
    rows: usize,
    capacity: usize,
    kv_stride: usize,
    layer_count: usize,
    k_delta: &[f32],
    v_delta: &[f32],
) -> Result<(), String> {
    let delta_len = layer_count
        .checked_mul(kv_stride)
        .and_then(|count| count.checked_mul(rows))
        .ok_or_else(|| "YuE2 Vulkan KV delta length overflow".to_string())?;
    if rows == 0
        || position.checked_add(rows).is_none_or(|end| end > capacity)
        || k_delta.len() != delta_len
        || v_delta.len() != delta_len
    {
        return Err(format!(
            "invalid YuE2 Vulkan KV delta: position={position}/{capacity} k={} v={} expected={delta_len}",
            k_delta.len(),
            v_delta.len(),
        ));
    }
    let cache_len = layer_count
        .checked_mul(capacity)
        .and_then(|count| count.checked_mul(kv_stride))
        .ok_or_else(|| "YuE2 Vulkan KV cache length overflow".to_string())?;
    let crate::core::scratchpad::KvCache::F32(cache) = kv_cache else {
        return Err("YuE2 Vulkan requires an F32 CPU shadow KV cache".into());
    };
    if cache.k.len() != cache_len || cache.v.len() != cache_len {
        return Err("invalid YuE2 CPU shadow KV cache length".into());
    }
    for layer in 0..layer_count {
        let layer_base = layer * capacity * kv_stride;
        for row in 0..rows {
            let src = (layer * rows + row) * kv_stride;
            let dst = layer_base + (position + row) * kv_stride;
            cache.k[dst..dst + kv_stride].copy_from_slice(&k_delta[src..src + kv_stride]);
            cache.v[dst..dst + kv_stride].copy_from_slice(&v_delta[src..src + kv_stride]);
        }
    }
    Ok(())
}

fn check_weight_types(model: &YuE2Model) -> Result<(), String> {
    for layer in model.ar_layers() {
        for weight in [
            &layer.ar_attention.q,
            &layer.ar_attention.k,
            &layer.ar_attention.v,
            &layer.ar_attention.output,
            &layer.ar_mlp.gate,
            &layer.ar_mlp.up,
            &layer.ar_mlp.down,
        ] {
            if !MATRIX_TYPES.contains(&weight.ggml_type()) {
                return Err(format!(
                    "unsupported YuE2 AR weight type {:?}",
                    weight.ggml_type()
                ));
            }
        }
    }
    Ok(())
}
