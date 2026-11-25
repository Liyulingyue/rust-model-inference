//! Shared Vulkan projection storage for the fixed Tessera and CLIP graphs.
use crate::ops::nn::Linear;
use crate::vulkan::{
    dense::UploadedBuffers,
    ops::{
        ArenaRegion, GpuMatmulMode, GpuMatmulSpec, GpuWeightFormat, OperatorBindings, Qwen3Ops,
        TokenCommands,
    },
    VulkanContext, VulkanError,
};

#[derive(Default)]
pub(super) struct Layout(usize);

impl Layout {
    /// All extents are bounded by the checked, fixed small-model configuration.
    pub(super) fn floats(&mut self, elements: usize) -> ArenaRegion {
        let offset = self.0;
        self.0 += elements * size_of::<f32>();
        ArenaRegion {
            offset,
            size: self.0 - offset,
        }
    }
}

pub(super) fn word(region: ArenaRegion) -> u32 {
    (region.offset / 4) as u32
}

pub(super) struct Matrix {
    binding: OperatorBindings,
    spec: GpuMatmulSpec,
    n_in: usize,
    n_out: usize,
    transposed: bool,
}

pub(super) struct Graph {
    pub(super) ops: Qwen3Ops<'static>,
    weights: UploadedBuffers,
    pub(super) projection_count: std::cell::Cell<usize>,
    kernel: usize,
    q8: ArenaRegion,
    scales: ArenaRegion,
    sums: ArenaRegion,
}

impl Graph {
    pub(super) fn new(
        context: &'static VulkanContext,
        mut layout: Layout,
        max_rows: usize,
        max_input_width: usize,
        projection_count: usize,
        shader: &[u8],
    ) -> Result<Self, VulkanError> {
        crate::vulkan::ops::require_online_attention(context)?;
        // Q8_0 and Q8_K activation quantizers can use the same scratch because
        // every projection completes before the next quantizer is dispatched.
        let q8 = layout.floats((max_rows * max_input_width).div_ceil(4));
        let blocks = max_rows * max_input_width.div_ceil(32);
        let scales = layout.floats(blocks);
        let sums = layout.floats(blocks);
        let mut ops =
            Qwen3Ops::new_device_local_with_size(context, layout.0, projection_count + 1)?;
        let kernel = ops.add_shader(shader)?;
        Ok(Self {
            ops,
            kernel,
            weights: UploadedBuffers::new(context),
            projection_count: std::cell::Cell::new(0),
            q8,
            scales,
            sums,
        })
    }

    pub(super) fn bind(&mut self, linear: &Linear<'_>) -> Result<Matrix, VulkanError> {
        let weight = linear.binding().weight;
        let spec = GpuMatmulSpec::prepared(weight)?;
        let bytes = weight.kernel.weight_bytes().ok_or_else(|| {
            VulkanError::UnsupportedShape("missing Tessera matrix storage".into())
        })?;
        let transposed = spec.mode == GpuMatmulMode::Prepared
            && linear.n_in().is_multiple_of(2)
            && matches!(spec.format, GpuWeightFormat::F32 | GpuWeightFormat::BF16);
        let buffer = match (transposed, spec.format) {
            (true, GpuWeightFormat::F32) => {
                let values = weight.kernel.f32_slice().ok_or_else(|| {
                    VulkanError::UnsupportedShape("missing dense F32 rows".into())
                })?;
                self.weights.upload_f32_transposed(values, linear.n_in())?
            }
            (true, GpuWeightFormat::BF16) => {
                self.weights.upload_half_transposed(bytes, linear.n_in())?
            }
            _ => self.weights.upload(bytes)?,
        };
        let binding = self
            .ops
            .bind_weight_buffers_mode(&[buffer], &[spec.format], spec.mode)?;
        Ok(Matrix {
            binding,
            spec,
            n_in: linear.n_in(),
            n_out: linear.n_out(),
            transposed,
        })
    }

    pub(super) fn project(
        &self,
        commands: &TokenCommands<'_>,
        matrix: &Matrix,
        input: ArenaRegion,
        output: ArenaRegion,
        rows: usize,
    ) -> Result<(), VulkanError> {
        self.projection_count.set(self.projection_count.get() + 1);
        let targets = [(output, matrix.n_out, matrix.n_out * 4)];
        if matrix.transposed {
            return self.ops.record_transposed_float_matmul_rows(
                commands,
                matrix.binding,
                input,
                &targets,
                matrix.n_in,
                rows,
                matrix.n_in,
            );
        }
        let record = if matrix.spec.uses_tiled_f16() && rows >= 32 && matrix.n_in.is_multiple_of(2)
        {
            Qwen3Ops::record_weight_matmul_tiled_rows
        } else {
            Qwen3Ops::record_weight_matmul_rows
        };
        record(
            &self.ops,
            commands,
            matrix.binding,
            input,
            self.q8,
            self.scales,
            self.sums,
            self.q8,
            self.scales,
            &targets,
            matrix.n_in,
            rows,
            matrix.n_in,
        )
    }

    pub(super) fn read(&self, region: ArenaRegion, len: usize) -> Result<Vec<f32>, VulkanError> {
        let mut values = vec![0.; len];
        self.ops.read_f32_into(region, &mut values)?;
        Ok(values)
    }

    pub(super) fn custom(
        &self,
        commands: &TokenCommands<'_>,
        push: &[u32],
        dispatch: [u32; 3],
    ) -> Result<(), VulkanError> {
        self.ops
            .record_custom(commands, self.kernel, None, push, dispatch)
    }
}
