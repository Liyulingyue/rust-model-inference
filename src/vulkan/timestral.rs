//! Resident F32 Transformer; scaling, patch/head projections and forecast folding stay on CPU.
use crate::models::timestral::graph::Slot;
use crate::models::timestral::network::Layout;
#[cfg(feature = "parity-trace")]
use crate::models::timestral::{trace, HD, HEADS};
use crate::models::timestral::{D, F};
use crate::vulkan::dense::UploadedBuffers;
use crate::vulkan::ops::{
    ArenaRegion, GpuWeightFormat, OperatorBindings, Qwen3Ops, TimestralOp, TokenCommands,
};
use crate::vulkan::{VulkanContext, VulkanError};
use std::collections::HashMap;

pub(crate) struct Transformer {
    ops: Qwen3Ops<'static>,
    _buffers: UploadedBuffers,
    bindings: HashMap<String, OperatorBindings>,
    regions: [ArenaRegion; 12],
    capacity: usize,
    context: &'static VulkanContext,
}

impl Transformer {
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn new(
        context: &'static VulkanContext,
        weights: &HashMap<String, &[f32]>,
        capacity: usize,
    ) -> Result<Self, VulkanError> {
        let mut size = 0usize;
        let mut regions = [ArenaRegion { offset: 0, size: 0 }; 12];
        // x, norm, fused, q, k, v, attention, gate, projection, hidden, valid/roles, rope.
        for (region, width) in regions
            .iter_mut()
            .zip([D, D, 2 * F, D, D, D, D, D, D, F, 2, 32])
        {
            let bytes = capacity
                .checked_mul(width)
                .and_then(|n| n.checked_mul(4))
                .ok_or(VulkanError::OutOfMemory)?;
            *region = ArenaRegion {
                offset: size,
                size: bytes,
            };
            size = size.checked_add(bytes).ok_or(VulkanError::OutOfMemory)?;
        }
        let mut ops = Qwen3Ops::new_device_local_with_size(context, size, weights.len() + 1)?;
        let mut buffers = UploadedBuffers::new(context);
        let mut bindings = HashMap::new();
        // Stable order keeps descriptor/dispatch traces reproducible.
        let mut names: Vec<_> = weights
            .keys()
            .filter(|n| n.starts_with("transformer."))
            .collect();
        names.sort();
        for name in names {
            let matrix = name.ends_with(".weight");
            let buffer = if matrix {
                let n_in = if name.ends_with(".mlp.w3.weight") {
                    F
                } else {
                    D
                };
                buffers.upload_f32_transposed(weights[name], n_in)?
            } else {
                buffers.upload_f32(weights[name])?
            };
            let binding = if matrix {
                ops.bind_weight_buffers(&[buffer], &[GpuWeightFormat::F32])?
            } else {
                ops.bind_buffers(&[buffer])?
            };
            bindings.insert(name.clone(), binding);
        }
        Ok(Self {
            ops,
            _buffers: buffers,
            bindings,
            regions,
            capacity,
            context,
        })
    }

    pub(crate) fn linear(
        &self,
        commands: &TokenCommands<'_>,
        name: &str,
        input: ArenaRegion,
        output: ArenaRegion,
        n_in: usize,
        n_out: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        self.ops.record_transposed_float_matmul_rows(
            commands,
            self.bindings[name],
            input,
            &[(output, n_out, n_out * 4)],
            n_in,
            rows,
            n_in,
        )
    }

    pub(crate) fn norm(
        &self,
        commands: &TokenCommands<'_>,
        name: &str,
        input: ArenaRegion,
        output: ArenaRegion,
        rows: usize,
    ) -> Result<(), VulkanError> {
        self.ops.record_rms_norm_rows(
            commands,
            self.bindings[name],
            input,
            output,
            D,
            1e-6,
            rows,
            D,
            D,
        )
    }

    /// Arena region backing a slot.
    ///
    /// The first twelve slots line up with the region table; the role tags
    /// live inside the mask region.
    fn region(&self, slot: Slot, count: usize) -> ArenaRegion {
        match slot {
            // The role tags follow the mask values inside the same region.
            Slot::Roles => self.regions[Slot::Masks as usize].shifted(count * 4),
            other => self.regions[other as usize],
        }
    }

    #[cfg(feature = "parity-trace")]
    pub(crate) fn checkpoint<'a>(
        &self,
        commands: TokenCommands<'a>,
        name: &str,
        shape: &[usize],
        region: ArenaRegion,
    ) -> Result<TokenCommands<'a>, VulkanError> {
        #[cfg(feature = "parity-trace")]
        if std::env::var_os("RMI_PARITY_TRACE").is_some() {
            commands.submit_and_wait()?;
            let mut values = vec![0.0; shape.iter().product()];
            self.ops.read_f32_into(region, &mut values)?;
            trace(name, shape, &values).map_err(VulkanError::InitFailed)?;
            return TokenCommands::begin(self.context);
        }
        let _ = (name, shape, region);
        Ok(commands)
    }

    pub(crate) fn forward(&self, input: &[f32], layout: &Layout) -> Result<Vec<f32>, VulkanError> {
        let (r, s) = (layout.rows, layout.seq());
        let count = r.checked_mul(s).ok_or(VulkanError::OutOfMemory)?;
        if count > self.capacity || input.len() != count * D || input.iter().any(|v| !v.is_finite())
        {
            return Err(VulkanError::UnsupportedShape(
                "invalid Timestral GPU input".into(),
            ));
        }
        let x = self.region(Slot::X, count);
        let roles = self.region(Slot::Roles, count);
        let before = self.ops.recorded_dispatch_count();
        self.ops.write_f32(x, input)?;
        self.ops.write_f32(
            self.region(Slot::Masks, count),
            &layout
                .valid
                .iter()
                .map(|&b| if b { 1. } else { 0. })
                .collect::<Vec<_>>(),
        )?;
        self.ops.write_f32(
            roles,
            &layout
                .roles
                .iter()
                .map(|&b| f32::from(b))
                .collect::<Vec<_>>(),
        )?;
        let mut coefficients = Vec::with_capacity(s * 32);
        for slot in 0..s {
            for i in 0..16 {
                let theta = layout.position(slot) / 10000f32.powf((2 * i) as f32 / 32.);
                let (sin, cos) = theta.sin_cos();
                coefficients.extend([cos, sin]);
            }
        }
        self.ops
            .write_f32(self.region(Slot::Rope, count), &coefficients)?;
        let mut commands = TokenCommands::begin(self.context)?;
        // The op list is shared with the CPU backend; only the execution
        // differs. The command buffer is flushed at layer boundaries to
        // bound its length on drivers that truncate long recordings.
        for layer in crate::models::timestral::graph::transformer_layer_ops() {
            for op in layer {
                let geometry = [
                    r,
                    layout.enc,
                    s,
                    layout.context,
                    usize::from(layout.covariates),
                    usize::from(op.kind.is_time_axis()),
                ];
                self.execute(&commands, &op, geometry, count)?;
                #[cfg(feature = "parity-trace")]
                if std::env::var_os("RMI_PARITY_TRACE").is_some() {
                    for (name, slot) in op.trace {
                        let shape = match slot {
                            Slot::Q | Slot::K | Slot::V => {
                                vec![geometry[0], geometry[2], HEADS, HD]
                            }
                            _ => vec![geometry[0], geometry[2], D],
                        };
                        commands =
                            self.checkpoint(commands, &name, &shape, self.region(slot, count))?;
                    }
                }
            }
            commands = commands.flush()?;
        }
        let geometry = [
            r,
            layout.enc,
            s,
            layout.context,
            usize::from(layout.covariates),
            0,
        ];
        let final_norm = crate::models::timestral::graph::final_norm_op();
        self.execute(&commands, &final_norm, geometry, count)?;
        #[cfg(feature = "parity-trace")]
        if std::env::var_os("RMI_PARITY_TRACE").is_some() {
            for (name, slot) in final_norm.trace {
                let shape = vec![geometry[0], geometry[2], D];
                commands = self.checkpoint(commands, &name, &shape, self.region(slot, count))?;
            }
        }
        commands.submit_and_wait()?;
        let mut output = vec![0.0; count * D];
        self.ops
            .read_f32_into(self.region(Slot::Norm, count), &mut output)?;
        if output.iter().any(|v| !v.is_finite()) {
            return Err(VulkanError::UnsupportedShape(
                "non-finite Timestral GPU output".into(),
            ));
        }
        // `out_norm` is traced by the final RMSNorm op in the shared op list.
        if std::env::var_os("RMI_COMPUTE_TRACE").is_some() {
            eprintln!(
                "compute scope=timestral.transformer gpu_dispatches={}",
                self.ops.recorded_dispatch_count() - before
            );
        }
        Ok(output)
    }

    /// Record one shared op on the device.
    fn execute(
        &self,
        commands: &TokenCommands<'_>,
        op: &crate::models::timestral::graph::TimestralOp,
        geometry: [usize; 6],
        count: usize,
    ) -> Result<(), VulkanError> {
        use crate::models::timestral::graph::TimestralOpKind;
        let empty = ArenaRegion { offset: 0, size: 0 };
        match &op.kind {
            TimestralOpKind::RmsNorm {
                input,
                weight,
                output,
            } => {
                self.norm(
                    commands,
                    weight,
                    self.region(*input, count),
                    self.region(*output, count),
                    count,
                )?;
            }
            TimestralOpKind::Linear {
                input,
                weight,
                output,
                n_in,
                n_out,
            } => {
                self.linear(
                    commands,
                    weight,
                    self.region(*input, count),
                    self.region(*output, count),
                    *n_in,
                    *n_out,
                    count,
                )?;
            }
            TimestralOpKind::Qkv {
                fused,
                q,
                k,
                v,
                rope,
                ..
            } => {
                self.ops.record_timestral(
                    commands,
                    TimestralOp::Qkv,
                    [
                        self.region(*fused, count),
                        self.region(*q, count),
                        self.region(*k, count),
                        self.region(*v, count),
                        self.region(*rope, count),
                        empty,
                    ],
                    geometry,
                )?;
            }
            TimestralOpKind::Attention {
                q,
                k,
                v,
                out,
                masks,
                roles,
                ..
            } => {
                self.ops.record_timestral(
                    commands,
                    TimestralOp::Attention,
                    [
                        self.region(*q, count),
                        self.region(*k, count),
                        self.region(*v, count),
                        self.region(*out, count),
                        self.region(*masks, count),
                        self.region(*roles, count),
                    ],
                    geometry,
                )?;
            }
            TimestralOpKind::Gate { attn, gate, .. } => {
                self.ops.record_timestral(
                    commands,
                    TimestralOp::Gate,
                    [
                        self.region(*attn, count),
                        self.region(*gate, count),
                        empty,
                        empty,
                        empty,
                        empty,
                    ],
                    geometry,
                )?;
            }
            TimestralOpKind::SwiGlu { fused, hidden } => {
                self.ops.record_timestral(
                    commands,
                    TimestralOp::SwiGlu,
                    [
                        self.region(*fused, count),
                        self.region(*hidden, count),
                        empty,
                        empty,
                        empty,
                        empty,
                    ],
                    geometry,
                )?;
            }
            TimestralOpKind::Add { input, output } => {
                self.ops.record_add_rows(
                    commands,
                    self.region(*output, count),
                    self.region(*input, count),
                    D,
                    count,
                )?;
            }
        }
        Ok(())
    }
}
