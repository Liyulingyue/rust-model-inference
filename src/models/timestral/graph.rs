//! Backend-neutral Timestral transformer graph.
//!
//! The 24-layer Timestral transformer is written **once**, here, as an
//! ordered list of [`TimestralOp`] values. Two executors consume that same
//! list:
//!
//! * the CPU executor in `network.rs`, which evaluates every op with the
//!   scalar/SIMD operators in `crate::ops`;
//! * the Vulkan executor in `crate::vulkan::timestral`, which records the
//!   matching device commands.
//!
//! Keeping a single op list is what makes `--compute cpu|vulkan|auto` a
//! backend switch instead of two independent implementations that can drift.
//!
//! This is the same shape as `crate::compute::dense` (an op vocabulary plus
//! a slot vocabulary, executed by a CPU and a Vulkan executor), but the
//! vocabulary is Timestral-local: partial RoPE, the time/variate dual
//! attention axes, the sigmoid attention gate and the absence of a KV cache
//! all differ from the standard decoder layer `compute::dense` describes, so
//! widening that shared vocabulary would distort it. The numeric kernels
//! themselves are not defined here; they stay in `crate::ops`.

use super::{D, F};

/// Intermediate activation slots.
///
/// The CPU executor maps each slot to an owned `Vec<f32>`; the Vulkan
/// executor maps it to an arena region. The slot names mirror the tensors
/// both backends already produce, so the op list reads the same on each side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum Slot {
    /// Residual stream.
    X,
    /// RMSNorm output.
    Norm,
    /// Fused projection output (`wQKV` or `mlp.w12`).
    Fused,
    /// Query tensor.
    Q,
    /// Key tensor.
    K,
    /// Value tensor.
    V,
    /// Attention output before the output projection.
    Attn,
    /// Sigmoid attention gate.
    Gate,
    /// Attention output projection / MLP output.
    Projection,
    /// SwiGLU hidden activation.
    Hidden,
    /// Per-token observation mask.
    Masks,
    /// Per-position partial-RoPE coefficients.
    Rope,
    /// Per-target role tags.
    ///
    /// The Vulkan backend keeps the role tags inside the mask region
    /// (`masks.shifted(count * 4)`), so it is the only slot without a
    /// dedicated arena region; the ordering above keeps the first twelve
    /// slots aligned with the GPU region table.
    Roles,
}

impl Slot {
    /// Dense index into the executor's slot table.
    pub(crate) fn index(self) -> usize {
        self as usize
    }

    /// Number of distinct slots.
    pub(crate) const COUNT: usize = 13;
}

/// One backend-neutral transformer operation.
///
/// Weight names are fully qualified (`transformer.layers.3.attns.1.wO`) so
/// an executor never needs to know the layer scoping rules; it only has to
/// look the name up and run the op.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TimestralOp {
    pub kind: TimestralOpKind,
    /// Parity checkpoints, recorded once the op completes.
    ///
    /// The Oracle comparison pins these intermediate tensors, so both
    /// backends must record them at the same point in the op list. Each
    /// entry names a checkpoint and the slot it reads; the shape follows
    /// from the slot (`[rows, seq, d_model]`, or
    /// `[rows, seq, heads, head_dim]` for the per-head Q/K/V tensors).
    pub trace: Vec<(String, Slot)>,
}

/// The operation itself.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TimestralOpKind {
    /// `output = rms_norm(input, weight) * eps`
    RmsNorm {
        input: Slot,
        weight: String,
        output: Slot,
    },
    /// `output = input @ weight^T` (+ bias when the tensor carries one).
    Linear {
        input: Slot,
        weight: String,
        output: Slot,
        n_in: usize,
        n_out: usize,
    },
    /// Split the fused QKV projection into Q/K/V, then apply the per-head
    /// Q/K RMSNorm and the partial RoPE.
    ///
    /// `time` selects the time axis, which is the only axis that carries
    /// positional RoPE; the variate axis uses the same weights without it.
    Qkv {
        fused: Slot,
        q: Slot,
        k: Slot,
        v: Slot,
        rope: Slot,
        time: bool,
    },
    /// Dual-axis masked attention with online softmax.
    Attention {
        q: Slot,
        k: Slot,
        v: Slot,
        out: Slot,
        masks: Slot,
        roles: Slot,
        time: bool,
    },
    /// `out = attn * sigmoid(gate)`
    ///
    /// The gate runs once per attention axis, so it carries the same axis
    /// flag as the attention it gates.
    Gate { attn: Slot, gate: Slot, time: bool },
    /// `hidden = silu(fused[..half]) * fused[half..]`
    SwiGlu { fused: Slot, hidden: Slot },
    /// `output += input`
    Add { input: Slot, output: Slot },
}

/// Number of transformer layers in the published Timestral-209M checkpoint.
pub(crate) const LAYERS: usize = 24;

/// Every fourth layer adds a variate attention axis next to the time axis.
pub(crate) fn axes_for_layer(layer: usize) -> usize {
    if layer % 4 == 3 {
        2
    } else {
        1
    }
}

impl TimestralOpKind {
    /// Whether this op runs on the time axis (the only axis with RoPE).
    pub(crate) fn is_time_axis(&self) -> bool {
        match self {
            TimestralOpKind::Qkv { time, .. }
            | TimestralOpKind::Attention { time, .. }
            | TimestralOpKind::Gate { time, .. } => *time,
            _ => false,
        }
    }
}

/// The op list for one attention axis of one transformer layer.
fn attention_ops(layer: usize, axis: usize, time: bool) -> Vec<TimestralOp> {
    let layer_scope = format!("transformer.layers.{layer}");
    let attn_scope = format!("{layer_scope}.attns.{axis}");
    let label = format!("layer.{layer}.attn.{axis}");
    vec![
        TimestralOp {
            kind: TimestralOpKind::RmsNorm {
                input: Slot::X,
                weight: format!("{layer_scope}.attn_norms.{axis}.scale"),
                output: Slot::Norm,
            },
            trace: vec![(format!("{label}.norm"), Slot::Norm)],
        },
        TimestralOp {
            kind: TimestralOpKind::Linear {
                input: Slot::Norm,
                weight: format!("{attn_scope}.wQKV.weight"),
                output: Slot::Fused,
                n_in: D,
                n_out: 3 * D,
            },
            trace: Vec::new(),
        },
        TimestralOp {
            kind: TimestralOpKind::Qkv {
                fused: Slot::Fused,
                q: Slot::Q,
                k: Slot::K,
                v: Slot::V,
                rope: Slot::Rope,
                time,
            },
            trace: vec![
                (format!("{label}.q"), Slot::Q),
                (format!("{label}.k"), Slot::K),
                (format!("{label}.v"), Slot::V),
            ],
        },
        TimestralOp {
            kind: TimestralOpKind::Attention {
                q: Slot::Q,
                k: Slot::K,
                v: Slot::V,
                out: Slot::Attn,
                masks: Slot::Masks,
                roles: Slot::Roles,
                time,
            },
            trace: Vec::new(),
        },
        TimestralOp {
            kind: TimestralOpKind::Linear {
                input: Slot::Norm,
                weight: format!("{attn_scope}._gate.weight"),
                output: Slot::Gate,
                n_in: D,
                n_out: D,
            },
            trace: Vec::new(),
        },
        TimestralOp {
            kind: TimestralOpKind::Gate {
                attn: Slot::Attn,
                gate: Slot::Gate,
                time,
            },
            trace: Vec::new(),
        },
        TimestralOp {
            kind: TimestralOpKind::Linear {
                input: Slot::Attn,
                weight: format!("{attn_scope}.wO.weight"),
                output: Slot::Projection,
                n_in: D,
                n_out: D,
            },
            trace: vec![(format!("{label}.output"), Slot::Projection)],
        },
        TimestralOp {
            kind: TimestralOpKind::Add {
                input: Slot::Projection,
                output: Slot::X,
            },
            trace: Vec::new(),
        },
    ]
}

/// The op list for the MLP part of one transformer layer.
fn mlp_ops(layer: usize) -> Vec<TimestralOp> {
    let layer_scope = format!("transformer.layers.{layer}");
    vec![
        TimestralOp {
            kind: TimestralOpKind::RmsNorm {
                input: Slot::X,
                weight: format!("{layer_scope}.mlp_norm.scale"),
                output: Slot::Norm,
            },
            trace: Vec::new(),
        },
        TimestralOp {
            kind: TimestralOpKind::Linear {
                input: Slot::Norm,
                weight: format!("{layer_scope}.mlp.w12.weight"),
                output: Slot::Fused,
                n_in: D,
                n_out: 2 * F,
            },
            trace: Vec::new(),
        },
        TimestralOp {
            kind: TimestralOpKind::SwiGlu {
                fused: Slot::Fused,
                hidden: Slot::Hidden,
            },
            trace: Vec::new(),
        },
        TimestralOp {
            kind: TimestralOpKind::Linear {
                input: Slot::Hidden,
                weight: format!("{layer_scope}.mlp.w3.weight"),
                output: Slot::Projection,
                n_in: F,
                n_out: D,
            },
            trace: Vec::new(),
        },
        TimestralOp {
            kind: TimestralOpKind::Add {
                input: Slot::Projection,
                output: Slot::X,
            },
            trace: vec![(format!("layer.{layer}.output"), Slot::X)],
        },
    ]
}

/// The op list of every transformer layer, one group per layer.
///
/// The grouping is what lets the Vulkan executor flush its command buffer
/// at the same layer boundaries the original implementation used, without
/// the graph being described twice.
pub(crate) fn transformer_layer_ops() -> Vec<Vec<TimestralOp>> {
    let mut layers = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        let mut ops = Vec::new();
        for axis in 0..axes_for_layer(layer) {
            ops.extend(attention_ops(layer, axis, axis == 0));
        }
        ops.extend(mlp_ops(layer));
        layers.push(ops);
    }
    layers
}

/// The final RMSNorm, applied after the last layer.
pub(crate) fn final_norm_op() -> TimestralOp {
    TimestralOp {
        kind: TimestralOpKind::RmsNorm {
            input: Slot::X,
            weight: "transformer.out_norm.scale".to_string(),
            output: Slot::Norm,
        },
        trace: vec![("out_norm".to_string(), Slot::Norm)],
    }
}

/// The full op list for the transformer: every layer, every attention axis,
/// then the final RMSNorm.
pub(crate) fn transformer_ops() -> Vec<TimestralOp> {
    let mut ops = Vec::with_capacity(LAYERS * 13 + 1);
    for layer in transformer_layer_ops() {
        ops.extend(layer);
    }
    ops.push(final_norm_op());
    ops
}
