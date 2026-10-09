//! Dense command recording shared by model adapters. No step submits or reads back.
use super::ops::{ArenaLayout, ArenaRegion, OperatorBindings, Qwen3Ops, TokenCommands};
use super::VulkanError;
use crate::compute::dense::{run_dense_layer, DenseStep};

#[derive(Clone, Copy)]
pub(super) struct DenseShape {
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub eps: f32,
    pub has_qk_norm: bool,
}

#[derive(Clone, Copy)]
pub(super) enum QkvBindings {
    Grouped(OperatorBindings),
    Split([OperatorBindings; 3]),
}

#[derive(Clone, Copy)]
pub(super) struct LayerBindings {
    pub(super) attn_norm: OperatorBindings,
    pub(super) qkv: QkvBindings,
    pub(super) qk_norm: OperatorBindings,
    pub(super) wo: OperatorBindings,
    pub(super) ffn_norm: OperatorBindings,
    pub(super) gate_up: OperatorBindings,
    pub(super) down: OperatorBindings,
}

pub(super) fn record_weights(
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
pub(super) fn record_dense_layer(
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
    run_dense_layer(
        &mut |_: usize, step| -> Result<(), VulkanError> {
            match step {
                DenseStep::AttnNorm => {
                    ops.record_rms_norm_rows(
                        &commands,
                        bindings.attn_norm,
                        layout.x,
                        layout.normed,
                        config.n_embd,
                        config.eps,
                        rows,
                        config.n_embd,
                        config.n_embd,
                    )?;
                }
                DenseStep::Qkv => {
                    let outputs = [
                        (layout.q, q_count),
                        (layout.k, kv_count),
                        (layout.v, kv_count),
                    ];
                    match bindings.qkv {
                        QkvBindings::Grouped(grouped) => record_weights(
                            ops,
                            layout,
                            &commands,
                            grouped,
                            layout.normed,
                            &outputs,
                            config.n_embd,
                            rows,
                        )?,
                        QkvBindings::Split(split) => {
                            for (binding, output) in split.iter().zip(outputs) {
                                record_weights(
                                    ops,
                                    layout,
                                    &commands,
                                    *binding,
                                    layout.normed,
                                    &[output],
                                    config.n_embd,
                                    rows,
                                )?;
                            }
                        }
                    }
                }
                DenseStep::QkNormRope => {
                    ops.record_qk_norm_rope_rows(
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
                    )?;
                }
                DenseStep::AppendKv => {
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
                DenseStep::Attention => {
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
                    )?;
                }
                DenseStep::AttnOut => {
                    record_weights(
                        ops,
                        layout,
                        &commands,
                        bindings.wo,
                        layout.attn,
                        &[(layout.projection, config.n_embd)],
                        attn_count,
                        rows,
                    )?;
                }
                DenseStep::AttnResidual => {
                    ops.record_add_rows(
                        &commands,
                        layout.x,
                        layout.projection,
                        config.n_embd,
                        rows,
                    )?;
                }
                DenseStep::FfnNorm => {
                    ops.record_rms_norm_rows(
                        &commands,
                        bindings.ffn_norm,
                        layout.x,
                        layout.normed,
                        config.n_embd,
                        config.eps,
                        rows,
                        config.n_embd,
                        config.n_embd,
                    )?;
                }
                DenseStep::GateUp => {
                    record_weights(
                        ops,
                        layout,
                        &commands,
                        bindings.gate_up,
                        layout.normed,
                        &[(layout.gate, config.n_ff), (layout.up, config.n_ff)],
                        config.n_embd,
                        rows,
                    )?;
                }
                DenseStep::SiluMul => {
                    ops.record_silu_mul_rows(&commands, layout.gate, layout.up, config.n_ff, rows)?;
                }
                DenseStep::Down => {
                    record_weights(
                        ops,
                        layout,
                        &commands,
                        bindings.down,
                        layout.gate,
                        &[(layout.down, config.n_embd)],
                        config.n_ff,
                        rows,
                    )?;
                }
                DenseStep::FfnResidual => {
                    ops.record_add_rows(&commands, layout.x, layout.down, config.n_embd, rows)?;
                }
            }
            Ok(())
        },
        layer_index,
    )
}
