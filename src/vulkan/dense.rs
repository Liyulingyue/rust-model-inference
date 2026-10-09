//! Dense command recording shared by model adapters. No step submits or reads back.
use super::ops::{fill_rope_coefficients, GpuWeightFormat, RopeLayout};
use super::ops::{ArenaLayout, ArenaRegion, OperatorBindings, Qwen3Ops, TokenCommands};
use super::{GpuBuffer, VulkanContext, VulkanError};
use crate::compute::dense::{run_dense_layer, DenseMatrix, DenseNorm, DenseOp, DenseTensor};
use crate::compute::state::TokenCommitState;
use crate::core::tensor::GGMLType;

#[derive(Clone, Copy)]
pub(crate) struct DenseShape {
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub eps: f32,
    pub has_qk_norm: bool,
    pub vocab: usize,
    pub freq_base: f32,
    pub rope_layout: RopeLayout,
    pub approximate_silu_multiline: bool,
    pub attention_mode: super::ops::AttentionMode,
}

#[derive(Clone, Copy)]
pub(crate) enum QkvBindings {
    Grouped(OperatorBindings),
    Split([OperatorBindings; 3]),
}

#[derive(Clone, Copy)]
pub(crate) struct LayerBindings {
    pub(crate) attn_norm: OperatorBindings,
    pub(crate) qkv: QkvBindings,
    pub(crate) qk_norm: OperatorBindings,
    pub(crate) wo: OperatorBindings,
    pub(crate) ffn_norm: OperatorBindings,
    pub(crate) gate_up: OperatorBindings,
    pub(crate) down: OperatorBindings,
}

pub(crate) fn record_weights(
    ops: &Qwen3Ops<'_>,
    layout: &ArenaLayout,
    commands: &TokenCommands<'_>,
    bindings: OperatorBindings,
    input: ArenaRegion,
    outputs: &[(ArenaRegion, usize)],
    n_in: usize,
    rows: usize,
) -> Result<(), VulkanError> {
    let outputs: Vec<_> = outputs
        .iter()
        .map(|&(region, width)| {
            Ok((
                region,
                width,
                width.checked_mul(4).ok_or(VulkanError::OutOfMemory)?,
            ))
        })
        .collect::<Result<_, VulkanError>>()?;
    ops.record_weight_matmul_rows(
        commands,
        bindings,
        input,
        layout.q8,
        layout.q8_scales,
        layout.q4_1_input_sums,
        layout.q8k,
        layout.q8k_scales,
        &outputs,
        n_in,
        rows,
        n_in,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record_dense_layer(
    ops: &Qwen3Ops<'_>,
    commands: &TokenCommands<'_>,
    layout: &ArenaLayout,
    bindings: &LayerBindings,
    config: &DenseShape,
    layer_index: usize,
    capacity: usize,
    base_position: usize,
    rows: usize,
) -> Result<(), VulkanError> {
    let q_count = config
        .n_head
        .checked_mul(config.n_embd_head_k)
        .ok_or(VulkanError::OutOfMemory)?;
    let kv_count = config
        .n_head_kv
        .checked_mul(config.n_embd_head_k)
        .ok_or(VulkanError::OutOfMemory)?;
    let attn_count = config
        .n_head
        .checked_mul(config.n_embd_head_v)
        .ok_or(VulkanError::OutOfMemory)?;
    let regions = [
        layout.x,
        layout.normed,
        layout.q,
        layout.k,
        layout.v,
        layout.attn,
        layout.projection,
        layout.gate,
        layout.up,
        layout.down,
    ];
    let widths = [
        config.n_embd,
        config.n_embd,
        q_count,
        kv_count,
        kv_count,
        attn_count,
        config.n_embd,
        config.n_ff,
        config.n_ff,
        config.n_embd,
    ];
    let region = |tensor: DenseTensor| regions[tensor as usize];
    let width = |tensor: DenseTensor| widths[tensor as usize];
    run_dense_layer(
        &mut |_: usize, op| -> Result<(), VulkanError> {
            match op {
                DenseOp::RmsNorm {
                    input,
                    weight,
                    output,
                } => {
                    let binding = match weight {
                        DenseNorm::Attn => bindings.attn_norm,
                        DenseNorm::Ffn => bindings.ffn_norm,
                    };
                    ops.record_rms_norm_rows(
                        commands,
                        binding,
                        region(input),
                        region(output),
                        width(input),
                        config.eps,
                        rows,
                        width(input),
                        width(output),
                    )?;
                }
                DenseOp::Linear { input, projections } => {
                    let mut outputs = [(layout.x, 0); 3];
                    for ((_, output), slot) in projections.iter().zip(&mut outputs) {
                        *slot = (region(*output), width(*output));
                    }
                    let outputs = &outputs[..projections.len()];
                    let binding = match projections {
                        [(DenseMatrix::Q, _), (DenseMatrix::K, _), (DenseMatrix::V, _)] => {
                            bindings.qkv
                        }
                        [(DenseMatrix::Gate, _), (DenseMatrix::Up, _)] => {
                            QkvBindings::Grouped(bindings.gate_up)
                        }
                        [(DenseMatrix::AttnOut, _)] => QkvBindings::Grouped(bindings.wo),
                        [(DenseMatrix::Down, _)] => QkvBindings::Grouped(bindings.down),
                        _ => {
                            return Err(VulkanError::UnsupportedShape(
                                "unsupported dense projection group".into(),
                            ))
                        }
                    };
                    match binding {
                        QkvBindings::Grouped(binding) => record_weights(
                            ops,
                            layout,
                            commands,
                            binding,
                            region(input),
                            outputs,
                            width(input),
                            rows,
                        )?,
                        QkvBindings::Split(split) => {
                            for (binding, output) in split.iter().zip(outputs) {
                                record_weights(
                                    ops,
                                    layout,
                                    commands,
                                    *binding,
                                    region(input),
                                    &[*output],
                                    width(input),
                                    rows,
                                )?;
                            }
                        }
                    }
                }
                DenseOp::QkNormRope => {
                    ops.record_qk_norm_rope_layout_rows(
                        &commands,
                        bindings.qk_norm,
                        layout.q,
                        layout.k,
                        config.n_head,
                        config.n_head_kv,
                        config.n_embd_head_k,
                        layout.rope,
                        config.eps,
                        config.has_qk_norm,
                        config.has_qk_norm,
                        rows,
                        config.rope_layout,
                    )?;
                }
                DenseOp::AppendKv => {
                    ops.record_kv_write_rows(
                        &commands,
                        layout.k,
                        layout.v,
                        layout.kv_k,
                        layout.kv_v,
                        layout.kv_delta_k,
                        layout.kv_delta_v,
                        layer_index,
                        base_position,
                        config.n_layer,
                        capacity,
                        kv_count,
                        rows,
                        kv_count,
                    )?;
                }
                DenseOp::Attention => {
                    ops.record_attention_rows(
                        &commands,
                        layout.q,
                        layout.kv_k,
                        layout.kv_v,
                        layout.scores,
                        layout.attn,
                        layer_index,
                        config.n_layer,
                        base_position,
                        capacity,
                        config.n_head,
                        config.n_head_kv,
                        config.n_embd_head_k,
                        rows,
                        config.attention_mode,
                    )?;
                }
                DenseOp::Add { input, output } => {
                    ops.record_add_rows(
                        commands,
                        region(output),
                        region(input),
                        width(output),
                        rows,
                    )?;
                }
                DenseOp::SiluMul { gate, up } => {
                    ops.record_silu_mul_rows_into(
                        commands,
                        region(gate),
                        region(up),
                        region(up),
                        width(gate),
                        rows,
                        rows == 1 || config.approximate_silu_multiline,
                    )?;
                }
                DenseOp::Moe { .. } => {
                    return Err(VulkanError::UnsupportedShape(
                        "dense Vulkan MoE is unsupported".into(),
                    ))
                }
            }
            Ok(())
        },
        layer_index,
        false,
    )
}

pub(crate) struct GpuChunkResult<'a> {
    pub(crate) logits: &'a [f32],
    pub(crate) k_delta: &'a [f32],
    pub(crate) v_delta: &'a [f32],
}

pub(crate) struct UploadedBuffers {
    context: &'static VulkanContext,
    values: Vec<GpuBuffer>,
}

impl UploadedBuffers {
    pub(crate) fn new(context: &'static VulkanContext) -> Self {
        Self {
            context,
            values: Vec::new(),
        }
    }

    pub(crate) fn upload(&mut self, bytes: &[u8]) -> Result<GpuBuffer, VulkanError> {
        let buffer = unsafe { self.context.upload_static(bytes)? };
        self.values.push(buffer);
        Ok(buffer)
    }

    pub(crate) fn upload_f32(&mut self, values: &[f32]) -> Result<GpuBuffer, VulkanError> {
        self.upload(bytemuck::cast_slice(values))
    }
}

impl Drop for UploadedBuffers {
    fn drop(&mut self) {
        let _ = unsafe { self.context.destroy_completed_buffers(&self.values) };
    }
}

pub(crate) struct DenseVulkanSession {
    context: &'static VulkanContext,
    ops: Qwen3Ops<'static>,
    _buffers: UploadedBuffers,
    layers: Vec<LayerBindings>,
    output_norm: OperatorBindings,
    output: OperatorBindings,
    layout: ArenaLayout,
    config: DenseShape,
    capacity: usize,
    max_rows: usize,
    #[cfg(test)]
    pub(crate) fail_after_row: Option<usize>,
    commit_state: TokenCommitState,
    rope: Vec<f32>,
    logits: Vec<f32>,
    k_delta: Vec<f32>,
    v_delta: Vec<f32>,
}

impl DenseVulkanSession {
    pub(crate) fn new(
        config: DenseShape,
        weights: &DenseWeights<'_>,
        capacity: usize,
        max_rows: usize,
        context: &'static VulkanContext,
    ) -> Result<Self, VulkanError> {
        weights.validate(&config)?;
        if !context.supports_shader_float16() {
            return Err(VulkanError::UnsupportedShape(
                "dense Vulkan requires shaderFloat16".into(),
            ));
        }
        let layout = ArenaLayout::build_rows(
            config.n_embd,
            config.n_ff,
            config.n_head,
            config.n_head_kv,
            config.n_embd_head_k,
            config.vocab,
            config.n_layer,
            capacity,
            max_rows,
        )?;
        let descriptor_capacity = config
            .n_layer
            .checked_mul(9)
            .and_then(|count| count.checked_add(3))
            .ok_or(VulkanError::OutOfMemory)?;
        let mut ops = Qwen3Ops::new(context, layout, descriptor_capacity)?;
        let mut buffers = UploadedBuffers::new(context);
        let mut layers = Vec::with_capacity(config.n_layer);

        for (layer_index, layer) in weights.layers.iter().enumerate() {
            let attn_norm_buffer = buffers.upload_f32(&layer.attn_norm)?;
            let attn_norm = ops.bind_buffers(&[attn_norm_buffer])?;

            let qkv_buffers = [
                buffers.upload(layer.wq.bytes)?,
                buffers.upload(layer.wk.bytes)?,
                buffers.upload(layer.wv.bytes)?,
            ];
            let qkv_formats = [
                GpuWeightFormat::from_ggml_type(layer.wq.ggml_type)?,
                GpuWeightFormat::from_ggml_type(layer.wk.ggml_type)?,
                GpuWeightFormat::from_ggml_type(layer.wv.ggml_type)?,
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

            let qk_norm = match (layer.q_norm, layer.k_norm) {
                (Some(q_norm), Some(k_norm)) => {
                    let q_norm = buffers.upload_f32(q_norm)?;
                    let k_norm = buffers.upload_f32(k_norm)?;
                    ops.bind_buffers(&[q_norm, k_norm])?
                }
                (None, None) => ops.bind_buffers(&[])?,
                _ => {
                    return Err(VulkanError::UnsupportedShape(format!(
                        "layer {layer_index} has incomplete Q/K norm weights"
                    )))
                }
            };

            let wo_buffer = buffers.upload(layer.wo.bytes)?;
            let wo = ops.bind_weight_buffers(
                &[wo_buffer],
                &[GpuWeightFormat::from_ggml_type(layer.wo.ggml_type)?],
            )?;

            let ffn_norm_buffer = buffers.upload_f32(&layer.ffn_norm)?;
            let ffn_norm = ops.bind_buffers(&[ffn_norm_buffer])?;

            let gate_up_buffers = [
                buffers.upload(layer.w_gate.bytes)?,
                buffers.upload(layer.w_up.bytes)?,
            ];
            let gate_up = ops.bind_weight_buffers(
                &gate_up_buffers,
                &[
                    GpuWeightFormat::from_ggml_type(layer.w_gate.ggml_type)?,
                    GpuWeightFormat::from_ggml_type(layer.w_up.ggml_type)?,
                ],
            )?;

            let down_buffer = buffers.upload(layer.w_down.bytes)?;
            let down = ops.bind_weight_buffers(
                &[down_buffer],
                &[GpuWeightFormat::from_ggml_type(layer.w_down.ggml_type)?],
            )?;
            layers.push(LayerBindings {
                attn_norm,
                qkv,
                qk_norm,
                wo,
                ffn_norm,
                gate_up,
                down,
            });
        }

        let output_norm_buffer = buffers.upload_f32(weights.output_norm)?;
        let output_norm = ops.bind_buffers(&[output_norm_buffer])?;
        let output_buffer = buffers.upload(weights.output.bytes)?;
        let output = ops.bind_weight_buffers(
            &[output_buffer],
            &[GpuWeightFormat::from_ggml_type(weights.output.ggml_type)?],
        )?;
        let kv_count = config
            .n_head_kv
            .checked_mul(config.n_embd_head_k)
            .ok_or(VulkanError::OutOfMemory)?;
        let delta_count = config
            .n_layer
            .checked_mul(kv_count)
            .and_then(|count| count.checked_mul(max_rows))
            .ok_or(VulkanError::OutOfMemory)?;
        let vocab = config.vocab;
        let head_dim = config.n_embd_head_k;

        Ok(Self {
            context,
            ops,
            _buffers: buffers,
            layers,
            output_norm,
            output,
            layout,
            config,
            capacity,
            max_rows,
            #[cfg(test)]
            fail_after_row: None,
            commit_state: TokenCommitState::new(0, capacity),
            rope: vec![
                0.0;
                head_dim
                    .checked_mul(max_rows)
                    .ok_or(VulkanError::OutOfMemory)?
            ],
            logits: vec![0.0; vocab],
            k_delta: vec![0.0; delta_count],
            v_delta: vec![0.0; delta_count],
        })
    }

    pub(crate) fn reserve_dense_rows(
        &mut self,
        weights: &DenseWeights<'_>,
        rows: usize,
    ) -> Result<(), VulkanError> {
        if rows <= self.max_rows {
            return Ok(());
        }
        // shortcut: growing beyond the initial chunk capacity reuploads weights;
        // retain uploads separately if resizing becomes frequent.
        let mut replacement = Self::new(self.config, weights, self.capacity, rows, self.context)?;
        let count = self.layout.kv_k.size / 4;
        replacement.ops.write_f32(
            replacement.layout.kv_k,
            self.ops.read_f32(self.layout.kv_k, count)?,
        )?;
        replacement.ops.write_f32(
            replacement.layout.kv_v,
            self.ops.read_f32(self.layout.kv_v, count)?,
        )?;
        replacement.commit_state = self.commit_state;
        #[cfg(test)]
        {
            replacement.fail_after_row = self.fail_after_row;
        }
        *self = replacement;
        Ok(())
    }

    pub(crate) fn forward_token<'a>(
        &'a mut self,
        input: &[f32],
        position: usize,
    ) -> Result<GpuChunkResult<'a>, VulkanError> {
        self.forward_chunk(input, position, 1, true)
    }

    pub(crate) fn forward_chunk<'a>(
        &'a mut self,
        input: &[f32],
        base_position: usize,
        rows: usize,
        project_logits: bool,
    ) -> Result<GpuChunkResult<'a>, VulkanError> {
        if let Err(error) = self.commit_state.begin(base_position, rows) {
            self.commit_state.abort();
            return Err(VulkanError::UnsupportedShape(error));
        }
        if let Err(error) = self.forward_chunk_inner(input, base_position, rows, project_logits) {
            self.commit_state.abort();
            return Err(error);
        }
        let count = self.config.n_layer * rows * self.config.n_head_kv * self.config.n_embd_head_k;
        Ok(GpuChunkResult {
            logits: if project_logits { &self.logits } else { &[] },
            k_delta: &self.k_delta[..count],
            v_delta: &self.v_delta[..count],
        })
    }

    pub(crate) fn forward_hidden_token<'a>(
        &'a mut self,
        input: &[f32],
        position: usize,
    ) -> Result<&'a [f32], VulkanError> {
        self.forward_chunk(input, position, 1, false)?;
        match self.ops.read_f32(self.layout.normed, self.config.n_embd) {
            Ok(hidden) => Ok(hidden),
            Err(error) => {
                self.commit_state.abort();
                Err(error)
            }
        }
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
        super::dense::record_weights(
            &self.ops,
            &self.layout,
            commands,
            bindings,
            input,
            outputs,
            n_in,
            rows,
        )
    }

    fn forward_chunk_inner(
        &mut self,
        input: &[f32],
        base_position: usize,
        rows: usize,
        project_logits: bool,
    ) -> Result<(), VulkanError> {
        let config = &self.config;
        let input_count = rows
            .checked_mul(config.n_embd)
            .ok_or(VulkanError::OutOfMemory)?;
        if input.len() != input_count
            || rows > self.max_rows
            || input.iter().any(|value| !value.is_finite())
        {
            return Err(VulkanError::UnsupportedShape(format!(
                "invalid dense chunk input={} rows={rows}/{}",
                input.len(),
                self.max_rows
            )));
        }
        let kv_count = config
            .n_head_kv
            .checked_mul(config.n_embd_head_k)
            .ok_or(VulkanError::OutOfMemory)?;
        // Acquire the shared submission guard before touching mapped arena bytes.
        let commands = TokenCommands::begin(self.context)?;
        self.ops.write_f32(self.layout.x, input)?;
        #[cfg(test)]
        let failure = match self.fail_after_row.take() {
            Some(row) if row < rows => Some(row),
            Some(row) => {
                self.fail_after_row = Some(row - rows);
                None
            }
            None => None,
        };
        #[cfg(test)]
        let rows = failure.map_or(rows, |row| row + 1);
        for row in 0..rows {
            fill_rope_coefficients(
                &mut self.rope[row * config.n_embd_head_k..(row + 1) * config.n_embd_head_k],
                base_position + row,
                config.freq_base,
                config.rope_layout,
            );
        }
        self.ops
            .write_f32(self.layout.rope, &self.rope[..rows * config.n_embd_head_k])?;
        for (layer_index, bindings) in self.layers.iter().enumerate() {
            record_dense_layer(
                &self.ops,
                &commands,
                &self.layout,
                bindings,
                config,
                layer_index,
                self.capacity,
                base_position,
                rows,
            )?;
            #[cfg(test)]
            if let Some(row) = failure {
                commands.submit_and_wait()?;
                return Err(VulkanError::UnsupportedShape(format!(
                    "injected Qwen3 GPU failure after row {row}"
                )));
            }
        }
        let last_offset = (rows - 1)
            .checked_mul(config.n_embd)
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
            self.output_norm,
            last,
            self.layout.normed,
            config.n_embd,
            config.eps,
        )?;
        if project_logits {
            self.record_weights(
                &commands,
                self.output,
                self.layout.normed,
                &[(self.layout.logits, config.vocab)],
                config.n_embd,
                1,
            )?;
        }
        commands.submit_and_wait()?;
        if project_logits {
            self.logits
                .copy_from_slice(self.ops.read_f32(self.layout.logits, config.vocab)?);
        }
        let delta_count = config
            .n_layer
            .checked_mul(rows)
            .and_then(|count| count.checked_mul(kv_count))
            .ok_or(VulkanError::OutOfMemory)?;
        self.k_delta[..delta_count]
            .copy_from_slice(self.ops.read_f32(self.layout.kv_delta_k, delta_count)?);
        self.v_delta[..delta_count]
            .copy_from_slice(self.ops.read_f32(self.layout.kv_delta_v, delta_count)?);
        if self.k_delta[..delta_count]
            .iter()
            .chain(&self.v_delta[..delta_count])
            .any(|value| !value.is_finite())
            || self
                .ops
                .read_f32(self.layout.x, input_count)?
                .iter()
                .any(|value| !value.is_finite())
            || self
                .ops
                .read_f32(self.layout.normed, config.n_embd)?
                .iter()
                .any(|value| !value.is_finite())
            || (project_logits && self.logits.iter().any(|value| !value.is_finite()))
        {
            return Err(VulkanError::UnsupportedShape(
                "Qwen3 Vulkan chunk produced non-finite output".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn commit_token(&mut self) {
        self.commit_state.commit();
    }

    pub(crate) fn abort_token(&mut self) {
        self.commit_state.abort();
    }

    pub(crate) fn reset(&mut self) {
        self.commit_state.reset();
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DenseWeight<'a> {
    pub bytes: &'a [u8],
    pub ggml_type: GGMLType,
    pub n_in: usize,
    pub n_out: usize,
}
impl DenseWeight<'_> {
    fn validate(&self, n_in: usize, n_out: usize) -> Result<(), VulkanError> {
        GpuWeightFormat::from_ggml_type(self.ggml_type)?;
        let (block, size) = self.ggml_type.type_traits();
        let length = n_in
            .checked_div(block)
            .and_then(|n| n.checked_mul(n_out))
            .and_then(|n| n.checked_mul(size));
        if self.n_in != n_in
            || self.n_out != n_out
            || n_in == 0
            || n_out == 0
            || !n_in.is_multiple_of(block)
            || (self.ggml_type == GGMLType::F16
                && crate::ops::f16_uses_half_accumulators(n_in)
                && !n_in.is_multiple_of(32))
            || length != Some(self.bytes.len())
        {
            return Err(VulkanError::UnsupportedShape(
                "dense weight shape or storage mismatch".into(),
            ));
        }
        Ok(())
    }
}
pub(crate) struct DenseLayer<'a> {
    pub attn_norm: &'a [f32],
    pub ffn_norm: &'a [f32],
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub wq: DenseWeight<'a>,
    pub wk: DenseWeight<'a>,
    pub wv: DenseWeight<'a>,
    pub wo: DenseWeight<'a>,
    pub w_gate: DenseWeight<'a>,
    pub w_up: DenseWeight<'a>,
    pub w_down: DenseWeight<'a>,
}
pub(crate) struct DenseWeights<'a> {
    pub layers: Vec<DenseLayer<'a>>,
    pub output_norm: &'a [f32],
    pub output: DenseWeight<'a>,
}
impl DenseWeights<'_> {
    fn validate(&self, c: &DenseShape) -> Result<(), VulkanError> {
        let invalid =
            || VulkanError::UnsupportedShape("invalid dense geometry or norm weights".into());
        if c.n_layer == 0
            || self.layers.len() != c.n_layer
            || c.n_head == 0
            || c.n_head_kv == 0
            || !c.n_head.is_multiple_of(c.n_head_kv)
            || c.n_embd_head_k == 0
            || c.n_embd_head_k != c.n_embd_head_v
            || !c.n_embd_head_k.is_multiple_of(2)
            || !c.eps.is_finite()
            || c.eps <= 0.0
            || !c.freq_base.is_finite()
            || c.freq_base <= 0.0
        {
            return Err(invalid());
        }
        let q = c.n_head.checked_mul(c.n_embd_head_k).ok_or_else(invalid)?;
        let kv = c
            .n_head_kv
            .checked_mul(c.n_embd_head_k)
            .ok_or_else(invalid)?;
        let norm = |w: &[f32], n| -> Result<(), VulkanError> {
            if w.len() != n || w.iter().any(|v| !v.is_finite()) {
                Err(invalid())
            } else {
                Ok(())
            }
        };
        norm(self.output_norm, c.n_embd)?;
        self.output.validate(c.n_embd, c.vocab)?;
        for l in &self.layers {
            norm(l.attn_norm, c.n_embd)?;
            norm(l.ffn_norm, c.n_embd)?;
            match (c.has_qk_norm, l.q_norm, l.k_norm) {
                (false, None, None) => (),
                (true, Some(q), Some(k)) => {
                    norm(q, c.n_embd_head_k)?;
                    norm(k, c.n_embd_head_k)?;
                }
                _ => return Err(invalid()),
            }
            for (w, ni, no) in [
                (&l.wq, c.n_embd, q),
                (&l.wk, c.n_embd, kv),
                (&l.wv, c.n_embd, kv),
                (&l.wo, q, c.n_embd),
                (&l.w_gate, c.n_embd, c.n_ff),
                (&l.w_up, c.n_embd, c.n_ff),
                (&l.w_down, c.n_ff, c.n_embd),
            ] {
                w.validate(ni, no)?;
            }
            if l.w_gate.ggml_type != l.w_up.ggml_type {
                return Err(VulkanError::UnsupportedShape(
                    "heterogeneous gate/up formats".into(),
                ));
            }
        }
        Ok(())
    }
}
