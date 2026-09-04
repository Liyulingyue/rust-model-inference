use super::{GpuBuffer, VulkanContext, VulkanError};
use crate::models::qwen3::trunk::Qwen3Config;
use ash::vk;
use std::sync::atomic::Ordering;
use std::sync::MutexGuard;

const QUANTIZE_Q8_0_SHADER: &[u8] = include_bytes!("../../shaders/bin/quantize_q8_0.spv");
const QUANTIZE_Q8_K_SHADER: &[u8] = include_bytes!("../../shaders/bin/quantize_q8_k.spv");
const Q8_MATMUL_GROUPED_SHADER: &[u8] = include_bytes!("../../shaders/bin/q8_matmul_grouped.spv");
const Q4_0_MATMUL_SHADER: &[u8] = include_bytes!("../../shaders/bin/q4_0_matmul.spv");
const Q4_1_MATMUL_SHADER: &[u8] = include_bytes!("../../shaders/bin/q4_1_matmul.spv");
const Q6_K_MATMUL_SHADER: &[u8] = include_bytes!("../../shaders/bin/q6_k_matmul.spv");
const RMS_NORM_SHADER: &[u8] = include_bytes!("../../shaders/bin/rms_norm.spv");
const QK_NORM_ROPE_SHADER: &[u8] = include_bytes!("../../shaders/bin/qk_norm_rope.spv");
const KV_WRITE_SHADER: &[u8] = include_bytes!("../../shaders/bin/kv_write.spv");
const ATTENTION_SCORES_SHADER: &[u8] = include_bytes!("../../shaders/bin/attention_scores.spv");
const SOFTMAX_SHADER: &[u8] = include_bytes!("../../shaders/bin/softmax.spv");
const ATTENTION_VALUES_SHADER: &[u8] = include_bytes!("../../shaders/bin/attention_values.spv");
const SILU_MUL_SHADER: &[u8] = include_bytes!("../../shaders/bin/silu_mul.spv");
const ADD_SHADER: &[u8] = include_bytes!("../../shaders/bin/add.spv");

const QUANTIZE: usize = 0;
const QUANTIZE_K: usize = 1;
const Q8_MATMUL_GROUPED: usize = 2;
const Q4_0_MATMUL: usize = 3;
const Q4_1_MATMUL: usize = 4;
const Q6_K_MATMUL: usize = 5;
const RMS_NORM: usize = 6;
const QK_NORM_ROPE: usize = 7;
const KV_WRITE: usize = 8;
const ATTENTION_SCORES: usize = 9;
const SOFTMAX: usize = 10;
const ATTENTION_VALUES: usize = 11;
const SILU_MUL: usize = 12;
const ADD: usize = 13;
const OPERATOR_SHADERS: [&[u8]; 14] = [
    QUANTIZE_Q8_0_SHADER,
    QUANTIZE_Q8_K_SHADER,
    Q8_MATMUL_GROUPED_SHADER,
    Q4_0_MATMUL_SHADER,
    Q4_1_MATMUL_SHADER,
    Q6_K_MATMUL_SHADER,
    RMS_NORM_SHADER,
    QK_NORM_ROPE_SHADER,
    KV_WRITE_SHADER,
    ATTENTION_SCORES_SHADER,
    SOFTMAX_SHADER,
    ATTENTION_VALUES_SHADER,
    SILU_MUL_SHADER,
    ADD_SHADER,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuWeightFormat {
    Q8_0,
    Q4_0,
    Q4_1,
    Q6_K,
}

impl GpuWeightFormat {
    pub(crate) fn from_ggml_type(
        value: crate::core::tensor::GGMLType,
    ) -> Result<Self, VulkanError> {
        match value {
            crate::core::tensor::GGMLType::Q8_0 => Ok(Self::Q8_0),
            crate::core::tensor::GGMLType::Q4_0 => Ok(Self::Q4_0),
            crate::core::tensor::GGMLType::Q4_1 => Ok(Self::Q4_1),
            crate::core::tensor::GGMLType::Q6K => Ok(Self::Q6_K),
            value => Err(VulkanError::UnsupportedShape(format!(
                "unsupported Vulkan weight format {value:?}"
            ))),
        }
    }

    fn layout(self) -> (usize, usize, usize) {
        match self {
            Self::Q8_0 => (32, 34, Q8_MATMUL_GROUPED),
            Self::Q4_0 => (32, 18, Q4_0_MATMUL),
            Self::Q4_1 => (32, 20, Q4_1_MATMUL),
            Self::Q6_K => (256, 210, Q6_K_MATMUL),
        }
    }
}

pub(crate) fn fill_rope_neox(coefficients: &mut [f32], position: usize, freq_base: f32) {
    debug_assert!(!coefficients.is_empty() && coefficients.len() % 2 == 0);
    let half = coefficients.len() / 2;
    let theta_scale = freq_base.powf(-2.0f32 / coefficients.len() as f32);
    let mut theta = position as f32;
    for index in 0..half {
        let (cosine, sine) = crate::ops::rope_sin_cos(theta);
        coefficients[index] = cosine;
        coefficients[index + half] = sine;
        theta *= theta_scale;
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ArenaRegion {
    pub(crate) offset: usize,
    pub(crate) size: usize,
}

impl ArenaRegion {
    pub(crate) fn end(self) -> usize {
        self.offset + self.size
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ArenaLayout {
    pub(crate) x: ArenaRegion,
    pub(crate) normed: ArenaRegion,
    pub(crate) q: ArenaRegion,
    pub(crate) k: ArenaRegion,
    pub(crate) v: ArenaRegion,
    pub(crate) attn: ArenaRegion,
    pub(crate) projection: ArenaRegion,
    pub(crate) gate: ArenaRegion,
    pub(crate) up: ArenaRegion,
    pub(crate) down: ArenaRegion,
    pub(crate) logits: ArenaRegion,
    pub(crate) q8: ArenaRegion,
    pub(crate) q8_scales: ArenaRegion,
    pub(crate) q4_1_input_sums: ArenaRegion,
    pub(crate) q8k: ArenaRegion,
    pub(crate) q8k_scales: ArenaRegion,
    pub(crate) scores: ArenaRegion,
    pub(crate) kv_k: ArenaRegion,
    pub(crate) kv_v: ArenaRegion,
    pub(crate) kv_delta_k: ArenaRegion,
    pub(crate) kv_delta_v: ArenaRegion,
    total_size: usize,
}

impl ArenaLayout {
    pub(crate) fn for_dims(
        n_embd: usize,
        n_ff: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
    ) -> Result<Self, VulkanError> {
        Self::build(n_embd, n_ff, n_head, n_head_kv, head_dim, n_embd, 1, 1)
    }

    pub(crate) fn qwen3(config: &Qwen3Config, capacity: usize) -> Result<Self, VulkanError> {
        if config.n_embd_head_k != config.n_embd_head_v {
            return Err(VulkanError::UnsupportedShape(format!(
                "different Qwen3 key/value head dimensions: {}/{}",
                config.n_embd_head_k, config.n_embd_head_v
            )));
        }
        Self::build(
            config.n_embd,
            config.n_ff,
            config.n_head,
            config.n_head_kv,
            config.n_embd_head_k,
            config.vocab,
            config.n_layer,
            capacity,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        n_embd: usize,
        n_ff: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        vocab: usize,
        n_layer: usize,
        capacity: usize,
    ) -> Result<Self, VulkanError> {
        if [
            n_embd, n_ff, n_head, n_head_kv, head_dim, vocab, n_layer, capacity,
        ]
        .contains(&0)
        {
            return Err(VulkanError::UnsupportedShape(
                "Vulkan arena dimensions must be nonzero".into(),
            ));
        }

        let q_len = product("Q length", &[n_head, head_dim])?;
        let kv_len = product("KV length", &[n_head_kv, head_dim])?;
        let q8_len = n_embd.max(n_ff).max(q_len);
        let score_len = product("attention scores", &[n_head, capacity])?;
        let kv_cache_len = product("KV cache", &[n_layer, capacity, kv_len])?;
        let kv_delta_len = product("KV delta", &[n_layer, kv_len])?;
        let mut cursor = 0usize;

        let x = f32_region(&mut cursor, n_embd)?;
        let normed = f32_region(&mut cursor, n_embd)?;
        let q = f32_region(&mut cursor, q_len)?;
        let k = f32_region(&mut cursor, kv_len)?;
        let v = f32_region(&mut cursor, kv_len)?;
        let attn = f32_region(&mut cursor, q_len)?;
        let projection = f32_region(&mut cursor, n_embd)?;
        let gate = f32_region(&mut cursor, n_ff)?;
        let up = f32_region(&mut cursor, n_ff)?;
        let down = f32_region(&mut cursor, n_embd)?;
        let logits = f32_region(&mut cursor, vocab)?;
        let q8 = region(&mut cursor, q8_len)?;
        let q8_scales = f32_region(&mut cursor, q8_len.div_ceil(32))?;
        let q4_1_input_sums = f32_region(&mut cursor, q8_len.div_ceil(32))?;
        let q8k = region(&mut cursor, q8_len)?;
        let q8k_scales = f32_region(&mut cursor, q8_len.div_ceil(256))?;
        let scores = f32_region(&mut cursor, score_len)?;
        let kv_k = f32_region(&mut cursor, kv_cache_len)?;
        let kv_v = f32_region(&mut cursor, kv_cache_len)?;
        let kv_delta_k = f32_region(&mut cursor, kv_delta_len)?;
        let kv_delta_v = f32_region(&mut cursor, kv_delta_len)?;

        Ok(Self {
            x,
            normed,
            q,
            k,
            v,
            attn,
            projection,
            gate,
            up,
            down,
            logits,
            q8,
            q8_scales,
            q4_1_input_sums,
            q8k,
            q8k_scales,
            scores,
            kv_k,
            kv_v,
            kv_delta_k,
            kv_delta_v,
            total_size: cursor,
        })
    }

    pub(crate) fn regions(&self) -> [ArenaRegion; 21] {
        [
            self.x,
            self.normed,
            self.q,
            self.k,
            self.v,
            self.attn,
            self.projection,
            self.gate,
            self.up,
            self.down,
            self.logits,
            self.q8,
            self.q8_scales,
            self.q4_1_input_sums,
            self.q8k,
            self.q8k_scales,
            self.scores,
            self.kv_k,
            self.kv_v,
            self.kv_delta_k,
            self.kv_delta_v,
        ]
    }

    pub(crate) fn total_size(&self) -> usize {
        self.total_size
    }
}

fn product(label: &str, values: &[usize]) -> Result<usize, VulkanError> {
    values.iter().try_fold(1usize, |product, &value| {
        product
            .checked_mul(value)
            .ok_or_else(|| VulkanError::UnsupportedShape(format!("{label} size overflows usize")))
    })
}

fn f32_region(cursor: &mut usize, elements: usize) -> Result<ArenaRegion, VulkanError> {
    let size = elements.checked_mul(4).ok_or_else(|| {
        VulkanError::UnsupportedShape("Vulkan arena byte size overflows usize".into())
    })?;
    region(cursor, size)
}

fn region(cursor: &mut usize, size: usize) -> Result<ArenaRegion, VulkanError> {
    let offset = cursor
        .checked_add(15)
        .map(|value| value & !15)
        .ok_or_else(|| VulkanError::UnsupportedShape("Vulkan arena offset overflow".into()))?;
    *cursor = offset
        .checked_add(size)
        .ok_or_else(|| VulkanError::UnsupportedShape("Vulkan arena size overflow".into()))?;
    Ok(ArenaRegion { offset, size })
}

pub(crate) struct TokenDispatchPlan {
    pub(crate) dispatches: usize,
    pub(crate) queue_submissions: usize,
    pub(crate) fence_waits: usize,
}

impl TokenDispatchPlan {
    pub(crate) fn qwen3_dense(layer_count: usize) -> Self {
        Self {
            dispatches: layer_count.saturating_mul(18).saturating_add(3),
            queue_submissions: 1,
            fence_waits: 1,
        }
    }
}

pub(crate) struct TokenCommands<'a> {
    context: &'a VulkanContext,
    command: vk::CommandBuffer,
    _guard: MutexGuard<'a, ()>,
}

impl<'a> TokenCommands<'a> {
    pub(crate) fn begin(context: &'a VulkanContext) -> Result<Self, VulkanError> {
        let guard = context
            .mutex
            .lock()
            .map_err(|_| VulkanError::InitFailed("Vulkan command mutex poisoned".into()))?;
        unsafe {
            context
                .device
                .reset_command_buffer(
                    context.command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .map_err(|error| VulkanError::InitFailed(error.to_string()))?;
            context
                .device
                .begin_command_buffer(
                    context.command_buffer,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|error| VulkanError::InitFailed(error.to_string()))?;
        }
        Ok(Self {
            context,
            command: context.command_buffer,
            _guard: guard,
        })
    }

    pub(crate) unsafe fn bind(
        &self,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        descriptor_sets: &[vk::DescriptorSet],
        push_constants: &[u8],
    ) {
        self.context.device.cmd_bind_pipeline(
            self.command,
            vk::PipelineBindPoint::COMPUTE,
            pipeline,
        );
        self.context.device.cmd_bind_descriptor_sets(
            self.command,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            descriptor_sets,
            &[],
        );
        if !push_constants.is_empty() {
            debug_assert_eq!(push_constants.len() % 4, 0);
            self.context.device.cmd_push_constants(
                self.command,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constants,
            );
        }
    }

    pub(crate) unsafe fn barrier(&self) {
        self.context.compute_barrier(self.command);
    }

    pub(crate) unsafe fn dispatch(&self, x: u32, y: u32, z: u32) {
        self.context.device.cmd_dispatch(self.command, x, y, z);
    }

    pub(crate) fn submit_and_wait(self) -> Result<(), VulkanError> {
        unsafe {
            self.context
                .device
                .end_command_buffer(self.command)
                .map_err(|error| VulkanError::InitFailed(error.to_string()))?;
            self.context
                .device
                .reset_fences(std::slice::from_ref(&self.context.fence))
                .map_err(|error| VulkanError::InitFailed(error.to_string()))?;
            let submit = vk::SubmitInfo::builder()
                .command_buffers(std::slice::from_ref(&self.command))
                .build();
            self.context
                .device
                .queue_submit(self.context.queue, &[submit], self.context.fence)
                .map_err(|error| VulkanError::InitFailed(error.to_string()))?;
            self.context
                .submission_count
                .fetch_add(1, Ordering::Relaxed);
            self.context
                .device
                .wait_for_fences(
                    std::slice::from_ref(&self.context.fence),
                    true,
                    60_000_000_000,
                )
                .map_err(|_| VulkanError::Timeout)
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct OperatorBindings {
    descriptor_set: vk::DescriptorSet,
    sizes: [u64; 3],
    weight_formats: [Option<GpuWeightFormat>; 3],
}

impl OperatorBindings {
    fn require(&self, index: usize, bytes: usize, label: &str) -> Result<(), VulkanError> {
        let bytes = u64::try_from(bytes).map_err(|_| VulkanError::OutOfMemory)?;
        if self.sizes[index] < bytes {
            return Err(VulkanError::UnsupportedShape(format!(
                "{label} buffer has {} bytes, needs {bytes}",
                self.sizes[index]
            )));
        }
        Ok(())
    }

    fn weight_format(&self, count: usize) -> Result<GpuWeightFormat, VulkanError> {
        let format = self.weight_formats[0].ok_or_else(|| {
            VulkanError::UnsupportedShape("missing Vulkan weight format metadata".into())
        })?;
        if self.weight_formats[..count]
            .iter()
            .any(|&value| value != Some(format))
        {
            return Err(VulkanError::UnsupportedShape(
                "heterogeneous grouped Vulkan weight formats".into(),
            ));
        }
        Ok(format)
    }
}

pub(crate) struct Qwen3Ops<'a> {
    context: &'a VulkanContext,
    arena: GpuBuffer,
    descriptor_pool: vk::DescriptorPool,
    arena_bindings: OperatorBindings,
    pipelines: [vk::Pipeline; OPERATOR_SHADERS.len()],
}

impl<'a> Qwen3Ops<'a> {
    pub(crate) fn new(
        context: &'a VulkanContext,
        layout: ArenaLayout,
        descriptor_capacity: usize,
    ) -> Result<Self, VulkanError> {
        let descriptor_capacity = descriptor_capacity.max(1);
        let descriptor_count = descriptor_capacity
            .checked_mul(4)
            .and_then(|count| u32::try_from(count).ok())
            .ok_or(VulkanError::OutOfMemory)?;
        let max_sets = u32::try_from(descriptor_capacity).map_err(|_| VulkanError::OutOfMemory)?;
        let mut pipelines = Vec::with_capacity(OPERATOR_SHADERS.len());
        for shader in OPERATOR_SHADERS {
            match context.create_pipeline(context.pipeline_layout, shader) {
                Ok(pipeline) => pipelines.push(pipeline),
                Err(error) => {
                    unsafe {
                        for pipeline in pipelines {
                            context.device.destroy_pipeline(pipeline, None);
                        }
                    }
                    return Err(error);
                }
            }
        }
        let pipelines: [vk::Pipeline; OPERATOR_SHADERS.len()] = pipelines.try_into().unwrap();

        let arena = match unsafe { context.allocate_session_buffer(layout.total_size()) } {
            Ok(arena) => arena,
            Err(error) => {
                unsafe {
                    for pipeline in pipelines {
                        context.device.destroy_pipeline(pipeline, None);
                    }
                }
                return Err(error);
            }
        };
        unsafe { std::ptr::write_bytes(arena.mapped, 0, layout.total_size()) };

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count,
        }];
        let descriptor_pool = match unsafe {
            context.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::builder()
                    .max_sets(max_sets)
                    .pool_sizes(&pool_sizes),
                None,
            )
        } {
            Ok(pool) => pool,
            Err(error) => {
                unsafe {
                    context.destroy_buffer(&arena);
                    for pipeline in pipelines {
                        context.device.destroy_pipeline(pipeline, None);
                    }
                }
                return Err(VulkanError::InitFailed(error.to_string()));
            }
        };
        let arena_bindings = match allocate_bindings(context, descriptor_pool, arena, &[], &[]) {
            Ok(bindings) => bindings,
            Err(error) => {
                unsafe {
                    context
                        .device
                        .destroy_descriptor_pool(descriptor_pool, None);
                    context.destroy_buffer(&arena);
                    for pipeline in pipelines {
                        context.device.destroy_pipeline(pipeline, None);
                    }
                }
                return Err(error);
            }
        };

        Ok(Self {
            context,
            arena,
            descriptor_pool,
            arena_bindings,
            pipelines,
        })
    }

    pub(crate) fn bind_buffers(
        &mut self,
        buffers: &[GpuBuffer],
    ) -> Result<OperatorBindings, VulkanError> {
        if buffers.len() > 3 {
            return Err(VulkanError::UnsupportedShape(format!(
                "operator descriptor set accepts at most 3 buffers, got {}",
                buffers.len()
            )));
        }
        allocate_bindings(self.context, self.descriptor_pool, self.arena, buffers, &[])
    }

    pub(crate) fn bind_weight_buffers(
        &mut self,
        buffers: &[GpuBuffer],
        formats: &[GpuWeightFormat],
    ) -> Result<OperatorBindings, VulkanError> {
        if buffers.len() != formats.len() {
            return Err(VulkanError::UnsupportedShape(
                "Vulkan weight buffers and formats differ in count".into(),
            ));
        }
        allocate_bindings(
            self.context,
            self.descriptor_pool,
            self.arena,
            buffers,
            formats,
        )
    }

    pub(crate) fn write_f32(&self, region: ArenaRegion, values: &[f32]) -> Result<(), VulkanError> {
        self.f32_word(region, values.len(), "host write")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.arena.mapped.add(region.offset).cast::<f32>(),
                values.len(),
            );
        }
        Ok(())
    }

    pub(crate) fn read_f32(
        &self,
        region: ArenaRegion,
        count: usize,
    ) -> Result<&[f32], VulkanError> {
        self.f32_word(region, count, "host read")?;
        Ok(unsafe {
            std::slice::from_raw_parts(self.arena.mapped.add(region.offset).cast::<f32>(), count)
        })
    }

    pub(crate) fn read_bytes(
        &self,
        region: ArenaRegion,
        count: usize,
    ) -> Result<&[u8], VulkanError> {
        self.byte_word(region, count, "host byte read")?;
        Ok(unsafe { std::slice::from_raw_parts(self.arena.mapped.add(region.offset), count) })
    }

    pub(crate) fn record_quantize_q8_0(
        &self,
        commands: &TokenCommands<'_>,
        input: ArenaRegion,
        q8: ArenaRegion,
        scales: ArenaRegion,
        q4_1_input_sums: ArenaRegion,
        count: usize,
    ) -> Result<(), VulkanError> {
        if count == 0 || count % 32 != 0 {
            return Err(VulkanError::UnsupportedShape(format!(
                "Q8_0 activation length {count} is not a positive multiple of 32"
            )));
        }
        let push = [
            self.f32_word(input, count, "quantize input")?,
            self.byte_word(q8, count, "quantize output")?,
            self.f32_word(scales, count / 32, "quantize scales")?,
            self.f32_word(q4_1_input_sums, count / 32, "Q4_1 input sums")?,
            as_u32(count, "quantize length")?,
        ];
        let (x, y) = super::dispatch_grid(count / 32, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[QUANTIZE],
                self.context.pipeline_layout,
                &[self.arena_bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, 1);
            commands.barrier();
        }
        Ok(())
    }

    pub(crate) fn record_quantize_q8_k(
        &self,
        commands: &TokenCommands<'_>,
        input: ArenaRegion,
        q8: ArenaRegion,
        scales: ArenaRegion,
        count: usize,
    ) -> Result<(), VulkanError> {
        if count == 0 || count % 256 != 0 {
            return Err(VulkanError::UnsupportedShape(format!(
                "Q8_K activation length {count} is not a positive multiple of 256"
            )));
        }
        let push = [
            self.f32_word(input, count, "Q8_K quantize input")?,
            self.byte_word(q8, count, "Q8_K quantize output")?,
            self.f32_word(scales, count / 256, "Q8_K quantize scales")?,
            as_u32(count, "Q8_K quantize length")?,
        ];
        let (x, y) = super::dispatch_grid(count / 256, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[QUANTIZE_K],
                self.context.pipeline_layout,
                &[self.arena_bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, 1);
            commands.barrier();
        }
        Ok(())
    }

    pub(crate) fn record_rms_norm(
        &self,
        commands: &TokenCommands<'_>,
        bindings: OperatorBindings,
        input: ArenaRegion,
        output: ArenaRegion,
        count: usize,
        eps: f32,
    ) -> Result<(), VulkanError> {
        bindings.require(0, f32_bytes(count)?, "RMS norm weight")?;
        let push = [
            self.f32_word(input, count, "RMS norm input")?,
            0,
            self.f32_word(output, count, "RMS norm output")?,
            as_u32(count, "RMS norm length")?,
            1,
            eps.to_bits(),
        ];
        unsafe {
            commands.bind(
                self.pipelines[RMS_NORM],
                self.context.pipeline_layout,
                &[bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(1, 1, 1);
            commands.barrier();
        }
        Ok(())
    }

    pub(crate) fn record_q8_matvec(
        &self,
        commands: &TokenCommands<'_>,
        bindings: OperatorBindings,
        q8: ArenaRegion,
        scales: ArenaRegion,
        output: ArenaRegion,
        n_in: usize,
        n_out: usize,
    ) -> Result<(), VulkanError> {
        self.record_q8_matvec_group(commands, bindings, q8, scales, &[(output, n_out)], n_in, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_weight_matvec_group(
        &self,
        commands: &TokenCommands<'_>,
        bindings: OperatorBindings,
        input: ArenaRegion,
        q8: ArenaRegion,
        q8_scales: ArenaRegion,
        q4_1_input_sums: ArenaRegion,
        q8k: ArenaRegion,
        q8k_scales: ArenaRegion,
        outputs: &[(ArenaRegion, usize)],
        n_in: usize,
    ) -> Result<(), VulkanError> {
        let (activation, scales) =
            if bindings.weight_format(outputs.len())? == GpuWeightFormat::Q6_K {
                self.record_quantize_q8_k(commands, input, q8k, q8k_scales, n_in)?;
                (q8k, q8k_scales)
            } else {
                self.record_quantize_q8_0(
                    commands,
                    input,
                    q8,
                    q8_scales,
                    q4_1_input_sums,
                    n_in,
                )?;
                (q8, q8_scales)
            };
        self.record_q8_matvec_group(
            commands,
            bindings,
            activation,
            scales,
            outputs,
            n_in,
            Some(q4_1_input_sums),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_weight_matvec(
        &self,
        commands: &TokenCommands<'_>,
        bindings: OperatorBindings,
        input: ArenaRegion,
        q8: ArenaRegion,
        q8_scales: ArenaRegion,
        q4_1_input_sums: ArenaRegion,
        q8k: ArenaRegion,
        q8k_scales: ArenaRegion,
        output: ArenaRegion,
        n_in: usize,
        n_out: usize,
    ) -> Result<(), VulkanError> {
        self.record_weight_matvec_group(
            commands,
            bindings,
            input,
            q8,
            q8_scales,
            q4_1_input_sums,
            q8k,
            q8k_scales,
            &[(output, n_out)],
            n_in,
        )
    }

    pub(crate) fn record_q8_matvec_group(
        &self,
        commands: &TokenCommands<'_>,
        bindings: OperatorBindings,
        q8: ArenaRegion,
        scales: ArenaRegion,
        outputs: &[(ArenaRegion, usize)],
        n_in: usize,
        q4_1_input_sums: Option<ArenaRegion>,
    ) -> Result<(), VulkanError> {
        let format = bindings.weight_format(outputs.len())?;
        let (block_elements, block_bytes, pipeline) = format.layout();
        if n_in == 0 || n_in % block_elements != 0 {
            return Err(VulkanError::UnsupportedShape(format!(
                "{format:?} matvec input length {n_in} is not a positive multiple of {block_elements}"
            )));
        }
        if outputs.is_empty() || outputs.len() > 3 {
            return Err(VulkanError::UnsupportedShape(format!(
                "Q8_0 grouped matvec needs 1 to 3 outputs, got {}",
                outputs.len()
            )));
        }
        let blocks_per_row = n_in / block_elements;
        if n_in / 32 > 512 {
            return Err(VulkanError::UnsupportedShape(format!(
                "n_in {n_in} exceeds shader shared-memory capacity"
            )));
        }
        self.byte_word(q8, n_in, "Q8_0 matvec input")?;
        self.f32_word(scales, blocks_per_row, "Q8_0 matvec scales")?;
        let q4_1_input_sum_word = if format == GpuWeightFormat::Q4_1 {
            self.f32_word(
                q4_1_input_sums.ok_or_else(|| {
                    VulkanError::UnsupportedShape("Q4_1 input sums are required".into())
                })?,
                blocks_per_row,
                "Q4_1 input sums",
            )?
        } else {
            0
        };
        let row_bytes = blocks_per_row
            .checked_mul(block_bytes)
            .ok_or(VulkanError::OutOfMemory)?;
        let mut output_words = [0u32; 3];
        let mut rows = [0u32; 3];
        let mut max_rows = 0usize;
        for (index, &(region, row_count)) in outputs.iter().enumerate() {
            if row_count == 0 {
                return Err(VulkanError::UnsupportedShape(
                    "Q8_0 matvec output rows must be nonzero".into(),
                ));
            }
            bindings.require(
                index,
                row_count
                    .checked_mul(row_bytes)
                    .ok_or(VulkanError::OutOfMemory)?,
                "Vulkan weight",
            )?;
            output_words[index] = self.f32_word(region, row_count, "Q8_0 matvec output")?;
            rows[index] = as_u32(row_count, "Q8_0 output rows")?;
            max_rows = max_rows.max(row_count);
        }
        let push = [
            self.byte_word(q8, n_in, "Q8_0 matvec input")?,
            self.f32_word(scales, blocks_per_row, "Q8_0 matvec scales")?,
            as_u32(n_in, "Q8_0 input length")?,
            as_u32(blocks_per_row, "Q8_0 blocks per row")?,
            output_words[0],
            rows[0],
            output_words[1],
            rows[1],
            output_words[2],
            rows[2],
            as_u32(outputs.len(), "Q8_0 group count")?,
            q4_1_input_sum_word,
        ];
        let (x, y) = super::dispatch_grid(max_rows, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[pipeline],
                self.context.pipeline_layout,
                &[bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, outputs.len() as u32);
            commands.barrier();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_qk_norm_rope(
        &self,
        commands: &TokenCommands<'_>,
        bindings: OperatorBindings,
        q: ArenaRegion,
        k: ArenaRegion,
        q_heads: usize,
        k_heads: usize,
        head_dim: usize,
        rope: ArenaRegion,
        eps: f32,
        normalize_q: bool,
        normalize_k: bool,
    ) -> Result<(), VulkanError> {
        if q_heads == 0 || k_heads == 0 || head_dim == 0 || head_dim % 2 != 0 {
            return Err(VulkanError::UnsupportedShape(
                "Q/K heads and even head dimension must be nonzero".into(),
            ));
        }
        let q_count = q_heads
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let k_count = k_heads
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        if normalize_q {
            bindings.require(0, f32_bytes(head_dim)?, "Q norm weight")?;
        }
        if normalize_k {
            bindings.require(1, f32_bytes(head_dim)?, "K norm weight")?;
        }
        let push = [
            self.f32_word(q, q_count, "Q vector")?,
            self.f32_word(k, k_count, "K vector")?,
            0,
            0,
            as_u32(q_heads, "Q head count")?,
            as_u32(k_heads, "K head count")?,
            as_u32(head_dim, "Q/K head dimension")?,
            self.f32_word(rope, head_dim, "RoPE coefficients")?,
            u32::from(normalize_q) | (u32::from(normalize_k) << 1),
            eps.to_bits(),
        ];
        if q_heads > self.context.limits.max_compute_work_group_count[0] as usize
            || k_heads > self.context.limits.max_compute_work_group_count[0] as usize
            || self.context.limits.max_compute_work_group_count[1] < 2
        {
            return Err(VulkanError::UnsupportedShape(
                "Q/K head count exceeds device dispatch limits".into(),
            ));
        }
        unsafe {
            commands.bind(
                self.pipelines[QK_NORM_ROPE],
                self.context.pipeline_layout,
                &[bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(q_heads.max(k_heads) as u32, 2, 1);
            commands.barrier();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_kv_write(
        &self,
        commands: &TokenCommands<'_>,
        k: ArenaRegion,
        v: ArenaRegion,
        cache_k: ArenaRegion,
        cache_v: ArenaRegion,
        delta_k: ArenaRegion,
        delta_v: ArenaRegion,
        layer: usize,
        position: usize,
        layer_count: usize,
        capacity: usize,
        kv_count: usize,
    ) -> Result<(), VulkanError> {
        if layer >= layer_count || position >= capacity || kv_count == 0 {
            return Err(VulkanError::UnsupportedShape(format!(
                "invalid KV write layer={layer}/{layer_count} position={position}/{capacity} width={kv_count}"
            )));
        }
        let cache_count = layer_count
            .checked_mul(capacity)
            .and_then(|value| value.checked_mul(kv_count))
            .ok_or(VulkanError::OutOfMemory)?;
        let delta_count = layer_count
            .checked_mul(kv_count)
            .ok_or(VulkanError::OutOfMemory)?;
        let push = [
            self.f32_word(k, kv_count, "new K")?,
            self.f32_word(v, kv_count, "new V")?,
            self.f32_word(cache_k, cache_count, "K cache")?,
            self.f32_word(cache_v, cache_count, "V cache")?,
            self.f32_word(delta_k, delta_count, "K delta")?,
            self.f32_word(delta_v, delta_count, "V delta")?,
            as_u32(layer, "KV layer")?,
            as_u32(position, "KV position")?,
            as_u32(capacity, "KV capacity")?,
            as_u32(kv_count, "KV width")?,
        ];
        let (x, y) = dispatch_invocations(kv_count, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[KV_WRITE],
                self.context.pipeline_layout,
                &[self.arena_bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, 1);
            commands.barrier();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_attention_scores(
        &self,
        commands: &TokenCommands<'_>,
        q: ArenaRegion,
        cache_k: ArenaRegion,
        scores: ArenaRegion,
        layer: usize,
        layer_count: usize,
        sequence_length: usize,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), VulkanError> {
        validate_attention_shape(
            layer,
            layer_count,
            sequence_length,
            capacity,
            q_heads,
            kv_heads,
            head_dim,
        )?;
        let q_count = q_heads
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let kv_count = kv_heads
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let cache_count = layer_count
            .checked_mul(capacity)
            .and_then(|value| value.checked_mul(kv_count))
            .ok_or(VulkanError::OutOfMemory)?;
        let score_count = q_heads
            .checked_mul(sequence_length)
            .ok_or(VulkanError::OutOfMemory)?;
        let push = [
            self.f32_word(q, q_count, "attention query")?,
            self.f32_word(cache_k, cache_count, "attention K cache")?,
            self.f32_word(scores, score_count, "attention scores")?,
            as_u32(layer, "attention layer")?,
            as_u32(sequence_length, "attention sequence length")?,
            as_u32(capacity, "attention capacity")?,
            as_u32(q_heads, "attention Q heads")?,
            as_u32(kv_heads, "attention KV heads")?,
            as_u32(head_dim, "attention head dimension")?,
            (1.0 / (head_dim as f32).sqrt()).to_bits(),
        ];
        let (x, y) = dispatch_invocations(score_count, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[ATTENTION_SCORES],
                self.context.pipeline_layout,
                &[self.arena_bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, 1);
            commands.barrier();
        }
        Ok(())
    }

    fn record_softmax(
        &self,
        commands: &TokenCommands<'_>,
        scores: ArenaRegion,
        heads: usize,
        sequence_length: usize,
    ) -> Result<(), VulkanError> {
        if heads == 0 || sequence_length == 0 {
            return Err(VulkanError::UnsupportedShape(
                "softmax heads and sequence length must be nonzero".into(),
            ));
        }
        let count = heads
            .checked_mul(sequence_length)
            .ok_or(VulkanError::OutOfMemory)?;
        let push = [
            self.f32_word(scores, count, "softmax scores")?,
            as_u32(heads, "softmax heads")?,
            as_u32(sequence_length, "softmax sequence length")?,
        ];
        let (x, y) = super::dispatch_grid(heads, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[SOFTMAX],
                self.context.pipeline_layout,
                &[self.arena_bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, 1);
            commands.barrier();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_attention_values(
        &self,
        commands: &TokenCommands<'_>,
        scores: ArenaRegion,
        cache_v: ArenaRegion,
        output: ArenaRegion,
        layer: usize,
        layer_count: usize,
        sequence_length: usize,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), VulkanError> {
        validate_attention_shape(
            layer,
            layer_count,
            sequence_length,
            capacity,
            q_heads,
            kv_heads,
            head_dim,
        )?;
        let score_count = q_heads
            .checked_mul(sequence_length)
            .ok_or(VulkanError::OutOfMemory)?;
        let output_count = q_heads
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let kv_count = kv_heads
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let cache_count = layer_count
            .checked_mul(capacity)
            .and_then(|value| value.checked_mul(kv_count))
            .ok_or(VulkanError::OutOfMemory)?;
        let push = [
            self.f32_word(scores, score_count, "attention probabilities")?,
            self.f32_word(cache_v, cache_count, "attention V cache")?,
            self.f32_word(output, output_count, "attention output")?,
            as_u32(layer, "attention layer")?,
            as_u32(sequence_length, "attention sequence length")?,
            as_u32(capacity, "attention capacity")?,
            as_u32(q_heads, "attention Q heads")?,
            as_u32(kv_heads, "attention KV heads")?,
            as_u32(head_dim, "attention head dimension")?,
        ];
        let (x, y) = dispatch_invocations(output_count, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[ATTENTION_VALUES],
                self.context.pipeline_layout,
                &[self.arena_bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, 1);
            commands.barrier();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_attention(
        &self,
        commands: &TokenCommands<'_>,
        q: ArenaRegion,
        cache_k: ArenaRegion,
        cache_v: ArenaRegion,
        scores: ArenaRegion,
        output: ArenaRegion,
        layer: usize,
        layer_count: usize,
        sequence_length: usize,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), VulkanError> {
        self.record_attention_scores(
            commands,
            q,
            cache_k,
            scores,
            layer,
            layer_count,
            sequence_length,
            capacity,
            q_heads,
            kv_heads,
            head_dim,
        )?;
        self.record_softmax(commands, scores, q_heads, sequence_length)?;
        self.record_attention_values(
            commands,
            scores,
            cache_v,
            output,
            layer,
            layer_count,
            sequence_length,
            capacity,
            q_heads,
            kv_heads,
            head_dim,
        )
    }

    pub(crate) fn record_silu_mul(
        &self,
        commands: &TokenCommands<'_>,
        gate: ArenaRegion,
        up: ArenaRegion,
        count: usize,
    ) -> Result<(), VulkanError> {
        let push = [
            self.f32_word(gate, count, "SiLU gate")?,
            self.f32_word(up, count, "SiLU multiplier")?,
            as_u32(count, "SiLU length")?,
        ];
        let (x, y) = dispatch_invocations(count, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[SILU_MUL],
                self.context.pipeline_layout,
                &[self.arena_bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, 1);
            commands.barrier();
        }
        Ok(())
    }

    pub(crate) fn record_add(
        &self,
        commands: &TokenCommands<'_>,
        target: ArenaRegion,
        addition: ArenaRegion,
        count: usize,
    ) -> Result<(), VulkanError> {
        let push = [
            self.f32_word(target, count, "add target")?,
            self.f32_word(addition, count, "add source")?,
            as_u32(count, "add length")?,
        ];
        let (x, y) = dispatch_invocations(count, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[ADD],
                self.context.pipeline_layout,
                &[self.arena_bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, 1);
            commands.barrier();
        }
        Ok(())
    }

    fn f32_word(&self, region: ArenaRegion, count: usize, label: &str) -> Result<u32, VulkanError> {
        self.byte_word(region, f32_bytes(count)?, label)
    }

    fn byte_word(
        &self,
        region: ArenaRegion,
        count: usize,
        label: &str,
    ) -> Result<u32, VulkanError> {
        let end = region
            .offset
            .checked_add(count)
            .ok_or(VulkanError::OutOfMemory)?;
        if region.offset % 4 != 0 || count > region.size || end > self.arena.size as usize {
            return Err(VulkanError::UnsupportedShape(format!(
                "{label} region offset={} size={} cannot hold {count} bytes",
                region.offset, region.size
            )));
        }
        as_u32(region.offset / 4, label)
    }
}

impl Drop for Qwen3Ops<'_> {
    fn drop(&mut self) {
        let _guard = self.context.mutex.lock().ok();
        unsafe {
            self.context
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for pipeline in self.pipelines {
                self.context.device.destroy_pipeline(pipeline, None);
            }
            self.context.destroy_buffer(&self.arena);
        }
    }
}

fn allocate_bindings(
    context: &VulkanContext,
    descriptor_pool: vk::DescriptorPool,
    arena: GpuBuffer,
    extras: &[GpuBuffer],
    formats: &[GpuWeightFormat],
) -> Result<OperatorBindings, VulkanError> {
    if extras.len() != formats.len() && !formats.is_empty() {
        return Err(VulkanError::UnsupportedShape(
            "Vulkan weight buffer and format counts differ".into(),
        ));
    }
    let layouts = [context.descriptor_set_layout];
    let descriptor_set = unsafe {
        context
            .device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
            .map_err(|error| VulkanError::InitFailed(error.to_string()))?[0]
    };
    let mut buffers = [arena; 4];
    let mut sizes = [arena.size; 3];
    for (index, &buffer) in extras.iter().enumerate() {
        buffers[index + 1] = buffer;
        sizes[index] = buffer.size;
    }
    let infos = buffers.map(|buffer| vk::DescriptorBufferInfo {
        buffer: buffer.buffer,
        offset: 0,
        range: vk::WHOLE_SIZE,
    });
    let writes: [vk::WriteDescriptorSet; 4] = std::array::from_fn(|index| {
        vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(index as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&infos[index]))
            .build()
    });
    unsafe { context.device.update_descriptor_sets(&writes, &[]) };
    Ok(OperatorBindings {
        descriptor_set,
        sizes,
        weight_formats: std::array::from_fn(|index| formats.get(index).copied()),
    })
}

fn as_u32(value: usize, label: &str) -> Result<u32, VulkanError> {
    u32::try_from(value)
        .map_err(|_| VulkanError::UnsupportedShape(format!("{label} value {value} exceeds u32")))
}

fn f32_bytes(count: usize) -> Result<usize, VulkanError> {
    count.checked_mul(4).ok_or(VulkanError::OutOfMemory)
}

fn dispatch_invocations(
    count: usize,
    limits: &vk::PhysicalDeviceLimits,
) -> Result<(u32, u32), VulkanError> {
    if count == 0 {
        return Err(VulkanError::UnsupportedShape(
            "cannot dispatch zero invocations".into(),
        ));
    }
    super::dispatch_grid(count.div_ceil(64), limits)
}

fn validate_attention_shape(
    layer: usize,
    layer_count: usize,
    sequence_length: usize,
    capacity: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<(), VulkanError> {
    if layer >= layer_count
        || sequence_length == 0
        || sequence_length > capacity
        || q_heads == 0
        || kv_heads == 0
        || q_heads % kv_heads != 0
        || head_dim == 0
    {
        return Err(VulkanError::UnsupportedShape(format!(
            "invalid attention shape layer={layer}/{layer_count} sequence={sequence_length}/{capacity} heads={q_heads}/{kv_heads} dim={head_dim}"
        )));
    }
    Ok(())
}

pub fn run_qwen3_operator_check(context: &VulkanContext, formats: &[&str]) -> Result<(), String> {
    check_quantize_tie_even(context)?;
    check_attention_f16_fma(context)?;
    check_softmax_f16_rounding(context)?;
    check_attention_value_reduction(context)?;
    for &format in formats {
        check_weight_format(context, format)?;
    }
    if formats.contains(&"q6_k") {
        check_quantize_q8_k_exact(context)?;
    }

    const N_EMBD: usize = 64;
    const N_FF: usize = 96;
    const Q_HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    const CAPACITY: usize = 4;
    const POSITION: usize = 3;
    const EPS: f32 = 1e-6;
    const FREQ_BASE: f32 = 10_000.0;

    let layout = ArenaLayout::build(
        N_EMBD, N_FF, Q_HEADS, KV_HEADS, HEAD_DIM, N_EMBD, 1, CAPACITY,
    )
    .map_err(|error| error.to_string())?;
    let input: Vec<f32> = (0..N_EMBD)
        .map(|index| ((index * 13 % 37) as f32 - 18.0) * 0.07)
        .collect();
    let norm_weight: Vec<f32> = (0..N_EMBD)
        .map(|index| 0.8 + (index % 11) as f32 * 0.025)
        .collect();
    let q_norm: Vec<f32> = (0..HEAD_DIM)
        .map(|index| 0.9 + index as f32 * 0.01)
        .collect();
    let k_norm: Vec<f32> = (0..HEAD_DIM)
        .map(|index| 1.05 - index as f32 * 0.008)
        .collect();
    let q_weight = synthetic_q8_weight(N_EMBD, Q_HEADS * HEAD_DIM, 3);
    let k_weight = synthetic_q8_weight(N_EMBD, KV_HEADS * HEAD_DIM, 7);
    let v_weight = synthetic_q8_weight(N_EMBD, KV_HEADS * HEAD_DIM, 11);
    let gate: Vec<f32> = (0..N_FF)
        .map(|index| ((index * 5 % 29) as f32 - 14.0) * 0.09)
        .collect();
    let up: Vec<f32> = (0..N_FF)
        .map(|index| ((index * 7 % 31) as f32 - 15.0) * 0.04)
        .collect();
    let kv_count = KV_HEADS * HEAD_DIM;
    let mut initial_k = vec![0.0f32; CAPACITY * kv_count];
    let mut initial_v = vec![0.0f32; CAPACITY * kv_count];
    for (index, value) in initial_k.iter_mut().enumerate() {
        *value = crate::ops::f16_to_f32(crate::ops::f32_to_f16((index as f32 * 0.071).sin() * 0.5));
    }
    for (index, value) in initial_v.iter_mut().enumerate() {
        *value = crate::ops::f16_to_f32(crate::ops::f32_to_f16((index as f32 * 0.053).cos() * 0.4));
    }

    let uploads: [&[u8]; 6] = [
        bytemuck::cast_slice(&norm_weight),
        bytemuck::cast_slice(&q_norm),
        bytemuck::cast_slice(&k_norm),
        &q_weight,
        &k_weight,
        &v_weight,
    ];
    let mut allocations = Vec::with_capacity(uploads.len());
    for upload in uploads {
        match unsafe { context.upload_static(upload) } {
            Ok(buffer) => allocations.push(buffer),
            Err(error) => {
                unsafe {
                    for buffer in &allocations {
                        context.destroy_buffer(buffer);
                    }
                }
                return Err(error.to_string());
            }
        }
    }

    let result = (|| -> Result<(), String> {
        let mut ops = Qwen3Ops::new(context, layout, 5).map_err(|error| error.to_string())?;
        let rms_bindings = ops
            .bind_buffers(&allocations[0..1])
            .map_err(|error| error.to_string())?;
        let qk_bindings = ops
            .bind_buffers(&allocations[1..3])
            .map_err(|error| error.to_string())?;
        let grouped_bindings = ops
            .bind_weight_buffers(&allocations[3..6], &[GpuWeightFormat::Q8_0; 3])
            .map_err(|error| error.to_string())?;
        let single_bindings = ops
            .bind_weight_buffers(&allocations[3..4], &[GpuWeightFormat::Q8_0])
            .map_err(|error| error.to_string())?;

        ops.write_f32(layout.x, &input)
            .map_err(|error| error.to_string())?;
        ops.write_f32(layout.gate, &gate)
            .map_err(|error| error.to_string())?;
        ops.write_f32(layout.up, &up)
            .map_err(|error| error.to_string())?;
        ops.write_f32(layout.kv_k, &initial_k)
            .map_err(|error| error.to_string())?;
        ops.write_f32(layout.kv_v, &initial_v)
            .map_err(|error| error.to_string())?;
        let mut rope = vec![0.0; HEAD_DIM];
        fill_rope_neox(&mut rope, POSITION, FREQ_BASE);
        ops.write_f32(layout.logits, &rope)
            .map_err(|error| error.to_string())?;

        let mut expected_normed = vec![0.0f32; N_EMBD];
        crate::ops::rms_norm(&input, &norm_weight, &mut expected_normed, EPS);
        let mut expected_q8 = vec![0u8; N_EMBD];
        let mut expected_scales = vec![0.0f32; N_EMBD / 32];
        crate::ops::quantize_q8_0_into(
            &expected_normed,
            N_EMBD,
            &mut expected_q8,
            &mut expected_scales,
        );
        let mut expected_q = cpu_q8_matvec(&q_weight, &expected_q8, &expected_scales, N_EMBD);
        let mut expected_k = cpu_q8_matvec(&k_weight, &expected_q8, &expected_scales, N_EMBD);
        let expected_v = cpu_q8_matvec(&v_weight, &expected_q8, &expected_scales, N_EMBD);
        let mut expected_projection =
            cpu_q8_matvec(&q_weight, &expected_q8, &expected_scales, N_EMBD);
        expected_projection.truncate(N_EMBD);
        for head in expected_q.chunks_exact_mut(HEAD_DIM) {
            crate::ops::rms_norm_inplace(head, &q_norm, EPS);
            crate::ops::rope_neox(head, POSITION, HEAD_DIM, FREQ_BASE);
        }
        for head in expected_k.chunks_exact_mut(HEAD_DIM) {
            crate::ops::rms_norm_inplace(head, &k_norm, EPS);
            crate::ops::rope_neox(head, POSITION, HEAD_DIM, FREQ_BASE);
        }
        let expected_k_f16: Vec<f32> = expected_k
            .iter()
            .map(|&value| crate::ops::f16_to_f32(crate::ops::f32_to_f16(value)))
            .collect();
        let expected_v_f16: Vec<f32> = expected_v
            .iter()
            .map(|&value| crate::ops::f16_to_f32(crate::ops::f32_to_f16(value)))
            .collect();
        let mut expected_k_cache = initial_k.clone();
        let mut expected_v_cache = initial_v.clone();
        expected_k_cache[POSITION * kv_count..(POSITION + 1) * kv_count]
            .copy_from_slice(&expected_k_f16);
        expected_v_cache[POSITION * kv_count..(POSITION + 1) * kv_count]
            .copy_from_slice(&expected_v_f16);
        let mut expected_scores = cpu_attention_scores(
            &expected_q,
            &expected_k_cache,
            POSITION + 1,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
        );
        let expected_raw_scores = expected_scores.clone();
        for row in expected_scores.chunks_exact_mut(POSITION + 1) {
            crate::ops::softmax_inplace(row);
            for value in row {
                *value = crate::ops::f16_to_f32(crate::ops::f32_to_f16(*value));
            }
        }
        let expected_attention = cpu_attention_values(
            &expected_scores,
            &expected_v_cache,
            POSITION + 1,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
        );
        let expected_gate: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| crate::ops::silu(gate) * up)
            .collect();
        let expected_add: Vec<f32> = input
            .iter()
            .zip(&expected_projection)
            .map(|(&left, &right)| left + right)
            .collect();

        let before = context.submission_count();
        let commands = TokenCommands::begin(context).map_err(|error| error.to_string())?;
        ops.record_rms_norm(
            &commands,
            rms_bindings,
            layout.x,
            layout.normed,
            N_EMBD,
            EPS,
        )
        .map_err(|error| error.to_string())?;
        ops.record_quantize_q8_0(
            &commands,
            layout.normed,
            layout.q8,
            layout.q8_scales,
            layout.q4_1_input_sums,
            N_EMBD,
        )
        .map_err(|error| error.to_string())?;
        ops.record_q8_matvec_group(
            &commands,
            grouped_bindings,
            layout.q8,
            layout.q8_scales,
            &[
                (layout.q, Q_HEADS * HEAD_DIM),
                (layout.k, KV_HEADS * HEAD_DIM),
                (layout.v, KV_HEADS * HEAD_DIM),
            ],
            N_EMBD,
            None,
        )
        .map_err(|error| error.to_string())?;
        ops.record_q8_matvec(
            &commands,
            single_bindings,
            layout.q8,
            layout.q8_scales,
            layout.projection,
            N_EMBD,
            N_EMBD,
        )
        .map_err(|error| error.to_string())?;
        ops.record_qk_norm_rope(
            &commands,
            qk_bindings,
            layout.q,
            layout.k,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            layout.logits,
            EPS,
            true,
            true,
        )
        .map_err(|error| error.to_string())?;
        ops.record_kv_write(
            &commands,
            layout.k,
            layout.v,
            layout.kv_k,
            layout.kv_v,
            layout.kv_delta_k,
            layout.kv_delta_v,
            0,
            POSITION,
            1,
            CAPACITY,
            kv_count,
        )
        .map_err(|error| error.to_string())?;
        ops.record_attention_scores(
            &commands,
            layout.q,
            layout.kv_k,
            layout.down,
            0,
            1,
            POSITION + 1,
            CAPACITY,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
        )
        .map_err(|error| error.to_string())?;
        ops.record_attention(
            &commands,
            layout.q,
            layout.kv_k,
            layout.kv_v,
            layout.scores,
            layout.attn,
            0,
            1,
            POSITION + 1,
            CAPACITY,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
        )
        .map_err(|error| error.to_string())?;
        ops.record_silu_mul(&commands, layout.gate, layout.up, N_FF)
            .map_err(|error| error.to_string())?;
        ops.record_add(&commands, layout.x, layout.projection, N_EMBD)
            .map_err(|error| error.to_string())?;
        commands
            .submit_and_wait()
            .map_err(|error| error.to_string())?;
        let submissions = context.submission_count() - before;
        if submissions != 1 {
            return Err(format!(
                "operator chain used {submissions} queue submissions, expected 1"
            ));
        }

        let gpu_q8 = ops
            .read_bytes(layout.q8, N_EMBD)
            .map_err(|error| error.to_string())?;
        if gpu_q8 != expected_q8 {
            let index = gpu_q8
                .iter()
                .zip(&expected_q8)
                .position(|(gpu, cpu)| gpu != cpu)
                .unwrap();
            return Err(format!(
                "quantize_q8_0 mismatch at {index}: gpu={} cpu={}",
                gpu_q8[index] as i8, expected_q8[index] as i8
            ));
        }
        println!("operator=quantize_q8_0 exact=true");
        check_close(
            "quantize_scales",
            ops.read_f32(layout.q8_scales, N_EMBD / 32)
                .map_err(|error| error.to_string())?,
            &expected_scales,
            0.0,
            0.0,
        )?;
        check_close(
            "rms_norm",
            ops.read_f32(layout.normed, N_EMBD)
                .map_err(|error| error.to_string())?,
            &expected_normed,
            2e-5,
            2e-5,
        )?;
        check_close(
            "q8_matmul",
            ops.read_f32(layout.projection, N_EMBD)
                .map_err(|error| error.to_string())?,
            &expected_projection,
            1e-4,
            1e-4,
        )?;
        check_close(
            "q8_matmul_grouped_v",
            ops.read_f32(layout.v, kv_count)
                .map_err(|error| error.to_string())?,
            &expected_v,
            1e-4,
            1e-4,
        )?;
        check_close(
            "qk_norm_rope_q",
            ops.read_f32(layout.q, Q_HEADS * HEAD_DIM)
                .map_err(|error| error.to_string())?,
            &expected_q,
            3e-5,
            3e-5,
        )?;
        check_close(
            "qk_norm_rope_k",
            ops.read_f32(layout.k, kv_count)
                .map_err(|error| error.to_string())?,
            &expected_k,
            3e-5,
            3e-5,
        )?;
        check_close(
            "kv_write_k",
            ops.read_f32(layout.kv_delta_k, kv_count)
                .map_err(|error| error.to_string())?,
            &ops.read_f32(layout.k, kv_count)
                .map_err(|error| error.to_string())?
                .into_iter()
                .map(|&value| crate::ops::f16_to_f32(crate::ops::f32_to_f16(value)))
                .collect::<Vec<_>>(),
            0.0,
            0.0,
        )?;
        check_close(
            "kv_write_v",
            ops.read_f32(layout.kv_delta_v, kv_count)
                .map_err(|error| error.to_string())?,
            &ops.read_f32(layout.v, kv_count)
                .map_err(|error| error.to_string())?
                .into_iter()
                .map(|&value| crate::ops::f16_to_f32(crate::ops::f32_to_f16(value)))
                .collect::<Vec<_>>(),
            0.0,
            0.0,
        )?;
        check_close(
            "attention_scores",
            ops.read_f32(layout.down, Q_HEADS * (POSITION + 1))
                .map_err(|error| error.to_string())?,
            &expected_raw_scores,
            3e-5,
            3e-5,
        )?;
        check_close(
            "softmax",
            ops.read_f32(layout.scores, Q_HEADS * (POSITION + 1))
                .map_err(|error| error.to_string())?,
            &expected_scores,
            3e-5,
            3e-5,
        )?;
        check_close(
            "attention_values",
            ops.read_f32(layout.attn, Q_HEADS * HEAD_DIM)
                .map_err(|error| error.to_string())?,
            &expected_attention,
            4e-5,
            4e-5,
        )?;
        check_close(
            "silu_mul",
            ops.read_f32(layout.gate, N_FF)
                .map_err(|error| error.to_string())?,
            &expected_gate,
            3e-5,
            3e-5,
        )?;
        check_close(
            "residual_add",
            ops.read_f32(layout.x, N_EMBD)
                .map_err(|error| error.to_string())?,
            &expected_add,
            1e-6,
            1e-6,
        )?;
        println!("device={} submissions={submissions}", context.device_name());
        Ok(())
    })();

    unsafe {
        for buffer in &allocations {
            context.destroy_buffer(buffer);
        }
    }
    result
}

fn check_weight_format(context: &VulkanContext, name: &str) -> Result<(), String> {
    let format = match name {
        "q4_0" => GpuWeightFormat::Q4_0,
        "q4_1" => GpuWeightFormat::Q4_1,
        "q6_k" => GpuWeightFormat::Q6_K,
        _ => return Err(format!("unsupported Vulkan weight format {name}")),
    };
    let n_in = match format {
        GpuWeightFormat::Q4_0 => 1024,
        GpuWeightFormat::Q4_1 => 3072,
        GpuWeightFormat::Q6_K => 1024,
        GpuWeightFormat::Q8_0 => unreachable!(),
    };
    let n_out = 65;
    let layout = ArenaLayout::for_dims(n_in.max(n_out), n_out, 1, 1, n_in)
        .map_err(|error| error.to_string())?;
    let input: Vec<f32> = (0..n_in)
        .map(|index| ((index * 29 % 251) as f32 - 125.0) / 97.0)
        .collect();
    if format == GpuWeightFormat::Q4_1 {
        check_q4_1_input_sums(context, &input)?;
    }
    let weight = synthetic_weight(format, n_in, n_out);
    let mut q8 = vec![0; n_in];
    let mut scales = vec![0.0; n_in / 32];
    crate::ops::quantize_q8_0_into(&input, n_in, &mut q8, &mut scales);
    let expected = cpu_weight_matvec(format, &weight, &input, &q8, &scales, n_in, n_out);
    let buffer = unsafe { context.upload_static(&weight) }.map_err(|error| error.to_string())?;
    let result = (|| -> Result<(), String> {
        let mut ops = Qwen3Ops::new(context, layout, 2).map_err(|error| error.to_string())?;
        let bindings = ops
            .bind_weight_buffers(&[buffer], &[format])
            .map_err(|error| error.to_string())?;
        ops.write_f32(layout.x, &input)
            .map_err(|error| error.to_string())?;
        let commands = TokenCommands::begin(context).map_err(|error| error.to_string())?;
        ops.record_weight_matvec(
            &commands,
            bindings,
            layout.x,
            layout.q8,
            layout.q8_scales,
            layout.q4_1_input_sums,
            layout.q8k,
            layout.q8k_scales,
            layout.projection,
            n_in,
            n_out,
        )
        .map_err(|error| error.to_string())?;
        commands
            .submit_and_wait()
            .map_err(|error| error.to_string())?;
        check_close(
            name,
            ops.read_f32(layout.projection, n_out)
                .map_err(|error| error.to_string())?,
            &expected,
            2e-3,
            2e-3,
        )
    })();
    unsafe { context.destroy_buffer(&buffer) };
    result
}

fn check_q4_1_input_sums(context: &VulkanContext, input: &[f32]) -> Result<(), String> {
    let count = input.len();
    let layout = ArenaLayout::for_dims(count, count, 1, 1, count).map_err(|error| error.to_string())?;
    let ops = Qwen3Ops::new(context, layout, 1).map_err(|error| error.to_string())?;
    let mut expected_q8 = vec![0; count];
    let mut expected_scales = vec![0.0; count / 32];
    crate::ops::quantize_q8_0_into(input, count, &mut expected_q8, &mut expected_scales);
    let expected: Vec<f32> = input
        .chunks_exact(32)
        .zip(expected_q8.chunks_exact(32))
        .map(|(values, quantized)| {
            let amax = values
                .iter()
                .fold(0.0f32, |current, value| current.max(value.abs()));
            let raw_scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
            let sum = quantized.iter().map(|&value| i32::from(value as i8)).sum::<i32>();
            crate::ops::f16_to_f32(crate::ops::f32_to_f16(sum as f32 * raw_scale))
        })
        .collect();
    let stored_scale_terms: Vec<f32> = expected_q8
        .chunks_exact(32)
        .zip(&expected_scales)
        .map(|(quantized, &scale)| {
            let sum = quantized.iter().map(|&value| i32::from(value as i8)).sum::<i32>();
            crate::ops::f16_to_f32(crate::ops::f32_to_f16(sum as f32 * scale))
        })
        .collect();
    if expected == stored_scale_terms {
        return Err("Q4_1 input-sum fixture does not distinguish raw and stored scales".into());
    }
    ops.write_f32(layout.x, input).map_err(|error| error.to_string())?;
    let commands = TokenCommands::begin(context).map_err(|error| error.to_string())?;
    ops.record_quantize_q8_0(
        &commands,
        layout.x,
        layout.q8,
        layout.q8_scales,
        layout.q4_1_input_sums,
        count,
    )
    .map_err(|error| error.to_string())?;
    commands.submit_and_wait().map_err(|error| error.to_string())?;
    let actual = ops
        .read_f32(layout.q4_1_input_sums, count / 32)
        .map_err(|error| error.to_string())?;
    if actual != expected {
        let index = actual
            .iter()
            .zip(&expected)
            .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
            .unwrap_or(0);
        return Err(format!(
            "Q4_1 input sum mismatch at {index}: gpu={} cpu={} stored_scale={}",
            actual[index], expected[index], stored_scale_terms[index]
        ));
    }
    println!("operator=q4_1_input_sums exact=true");
    Ok(())
}

fn synthetic_weight(format: GpuWeightFormat, n_in: usize, n_out: usize) -> Vec<u8> {
    let (elements, bytes, _) = format.layout();
    let blocks = n_in / elements;
    let mut data = vec![0; n_out * blocks * bytes];
    for row in 0..n_out {
        for block in 0..blocks {
            let offset = (row * blocks + block) * bytes;
            match format {
                GpuWeightFormat::Q4_0 => {
                    data[offset..offset + 2].copy_from_slice(
                        &crate::ops::f32_to_f16(if row == 0 && block == 0 {
                            0.0
                        } else {
                            (row as f32 + 3.0) / (block as f32 + 11.0)
                        })
                        .to_le_bytes(),
                    );
                    for (index, value) in data[offset + 2..offset + 18].iter_mut().enumerate() {
                        let low = ((row * 11 + block * 7 + index * 3 + 1) & 15) as u8;
                        let high = ((row * 5 + block * 13 + index * 9 + 6) & 15) as u8;
                        *value = low | (high << 4);
                    }
                }
                GpuWeightFormat::Q4_1 => {
                    data[offset..offset + 2]
                        .copy_from_slice(&crate::ops::f32_to_f16((row as f32 + 1.0) / 91.0).to_le_bytes());
                    data[offset + 2..offset + 4].copy_from_slice(
                        &crate::ops::f32_to_f16(if row == 0 && block == 0 {
                            0.0
                        } else {
                            -(block as f32 + 2.0) / 53.0
                        })
                        .to_le_bytes(),
                    );
                    for (index, value) in data[offset + 4..offset + 20].iter_mut().enumerate() {
                        let low = ((row * 3 + block * 5 + index * 7 + 2) & 15) as u8;
                        let high = ((row * 13 + block * 11 + index * 2 + 4) & 15) as u8;
                        *value = low | (high << 4);
                    }
                }
                GpuWeightFormat::Q6_K => {
                    for index in 0..128 {
                        let low = ((row * 17 + block * 29 + index * 7) & 15) as u8;
                        let high = ((row * 11 + block * 13 + index * 5) & 15) as u8;
                        data[offset + index] = low | (high << 4);
                    }
                    for index in 0..64 {
                        data[offset + 128 + index] =
                            ((row * 19 + block * 23 + index * 37) & 255) as u8;
                    }
                    for index in 0..16 {
                        data[offset + 192 + index] =
                            (row as i32 * 7 + block as i32 * 11 + index as i32 * 3 - 23) as i8
                                as u8;
                    }
                    data[offset + 208..offset + 210].copy_from_slice(
                        &crate::ops::f32_to_f16(if row == 0 && block == 0 {
                            0.0
                        } else {
                            (row as f32 + block as f32 + 1.0) / 64.0
                        })
                        .to_le_bytes(),
                    );
                }
                GpuWeightFormat::Q8_0 => unreachable!(),
            }
        }
    }
    data
}

fn cpu_weight_matvec(
    format: GpuWeightFormat,
    weight: &[u8],
    input: &[f32],
    q8: &[u8],
    scales: &[f32],
    n_in: usize,
    n_out: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; n_out];
    match format {
        GpuWeightFormat::Q4_0 => {
            let kernel = crate::ops::kernel::q4_0::Q4_0Kernel::new(weight, n_in, n_out);
            crate::ops::kernel::Kernel::forward_prepared(
                &kernel,
                input,
                q8,
                scales,
                None,
                &mut output,
                n_in,
                n_out,
                0,
                1,
            );
        }
        GpuWeightFormat::Q4_1 => {
            let kernel = crate::ops::kernel::q4_1::Q4_1Kernel::new(weight, n_in, n_out);
            crate::ops::kernel::Kernel::forward_prepared(
                &kernel,
                input,
                q8,
                scales,
                None,
                &mut output,
                n_in,
                n_out,
                0,
                1,
            );
        }
        GpuWeightFormat::Q6_K => {
            let q8k = crate::ops::quant::quantize_row_q8_k(input);
            for (row, output) in output.iter_mut().enumerate() {
                let row_bytes = n_in / crate::ops::quant::QK_K * crate::ops::quant::BLOCK_Q6K_SIZE;
                *output = crate::ops::quant::vec_dot_q6k_q8k(
                    &weight[row * row_bytes..(row + 1) * row_bytes],
                    &q8k,
                );
            }
        }
        GpuWeightFormat::Q8_0 => unreachable!(),
    }
    output
}

fn check_quantize_q8_k_exact(context: &VulkanContext) -> Result<(), String> {
    const COUNT: usize = 512;
    let layout = ArenaLayout::for_dims(COUNT, COUNT, 1, 1, COUNT).map_err(|error| error.to_string())?;
    let ops = Qwen3Ops::new(context, layout, 1).map_err(|error| error.to_string())?;
    let input: Vec<f32> = (0..COUNT)
        .map(|index| {
            if index == 0 {
                -3.75
            } else if index == 255 {
                3.75
            } else if index == 256 {
                0.0
            } else if index == 511 {
                -2.5
            } else {
                ((index * 47 % 509) as f32 - 254.0) / 83.0
            }
        })
        .collect();
    let expected = crate::ops::quant::quantize_row_q8_k(&input);
    ops.write_f32(layout.x, &input)
        .map_err(|error| error.to_string())?;
    let commands = TokenCommands::begin(context).map_err(|error| error.to_string())?;
    ops.record_quantize_q8_k(
        &commands,
        layout.x,
        layout.q8k,
        layout.q8k_scales,
        COUNT,
    )
    .map_err(|error| error.to_string())?;
    commands
        .submit_and_wait()
        .map_err(|error| error.to_string())?;
    let actual_qs = ops
        .read_bytes(layout.q8k, COUNT)
        .map_err(|error| error.to_string())?;
    let actual_scales = ops
        .read_f32(layout.q8k_scales, COUNT / 256)
        .map_err(|error| error.to_string())?;
    let expected_qs: Vec<u8> = expected
        .iter()
        .flat_map(|block| block.qs.map(|value| value as u8))
        .collect();
    if actual_qs != expected_qs {
        let index = actual_qs
            .iter()
            .zip(&expected_qs)
            .position(|(actual, expected)| actual != expected)
            .expect("different byte vectors have a differing index");
        return Err(format!(
            "quantize_q8_k byte mismatch at {index}: input={} gpu={} cpu={} gpu_scale_bits={:#010x} cpu_scale_bits={:#010x}",
            input[index],
            actual_qs[index] as i8,
            expected_qs[index] as i8,
            actual_scales[index / 256].to_bits(),
            expected[index / 256].d.to_bits(),
        ));
    }
    let max_scale_ulps = actual_scales
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| actual.to_bits().abs_diff(expected.d.to_bits()))
        .max()
        .unwrap_or(0);
    if max_scale_ulps > 1 {
        return Err(format!(
            "quantize_q8_k scale differs by {max_scale_ulps} ULPs (expected at most 1)"
        ));
    }
    println!("operator=quantize_q8_k bytes_exact=true max_scale_ulps={max_scale_ulps}");
    Ok(())
}

fn check_quantize_tie_even(context: &VulkanContext) -> Result<(), String> {
    let layout = ArenaLayout::for_dims(32, 32, 1, 1, 32).map_err(|error| error.to_string())?;
    let ops = Qwen3Ops::new(context, layout, 1).map_err(|error| error.to_string())?;
    let mut input = [0.0f32; 32];
    input[0] = f32::from_bits(0xbdbf0aec);
    input[1] = f32::from_bits(0x3f13f10a);
    ops.write_f32(layout.x, &input)
        .map_err(|error| error.to_string())?;

    let commands = TokenCommands::begin(context).map_err(|error| error.to_string())?;
    ops.record_quantize_q8_0(
        &commands,
        layout.x,
        layout.q8,
        layout.q8_scales,
        layout.q4_1_input_sums,
        input.len(),
    )
    .map_err(|error| error.to_string())?;
    commands
        .submit_and_wait()
        .map_err(|error| error.to_string())?;

    let mut expected = [0u8; 32];
    expected[0] = (-20i8) as u8;
    expected[1] = 127;
    let actual = ops
        .read_bytes(layout.q8, input.len())
        .map_err(|error| error.to_string())?;
    if actual != expected {
        let index = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| *actual != expected)
            .unwrap();
        return Err(format!(
            "quantize tie-even mismatch at {index}: gpu={} cpu={}",
            actual[index] as i8, expected[index] as i8
        ));
    }
    println!("operator=quantize_tie_even exact=true");
    Ok(())
}

fn check_attention_f16_fma(context: &VulkanContext) -> Result<(), String> {
    const HEAD_DIM: usize = 64;
    let layout = ArenaLayout::build(HEAD_DIM, HEAD_DIM, 1, 1, HEAD_DIM, 1, 1, 1)
        .map_err(|error| error.to_string())?;
    let ops = Qwen3Ops::new(context, layout, 1).map_err(|error| error.to_string())?;
    let mut query = vec![0.0f32; HEAD_DIM];
    let mut key = vec![0.0f32; HEAD_DIM];
    query[5] = -0.91552734375;
    key[5] = 1.0;
    query[37] = -1.5693359375;
    key[37] = 0.05413818359375;
    ops.write_f32(layout.q, &query)
        .map_err(|error| error.to_string())?;
    ops.write_f32(layout.kv_k, &key)
        .map_err(|error| error.to_string())?;

    let commands = TokenCommands::begin(context).map_err(|error| error.to_string())?;
    ops.record_attention_scores(
        &commands,
        layout.q,
        layout.kv_k,
        layout.scores,
        0,
        1,
        1,
        1,
        1,
        1,
        HEAD_DIM,
    )
    .map_err(|error| error.to_string())?;
    commands
        .submit_and_wait()
        .map_err(|error| error.to_string())?;

    let actual = ops
        .read_f32(layout.scores, 1)
        .map_err(|error| error.to_string())?[0];
    let expected = -1.0009765625f32 / (HEAD_DIM as f32).sqrt();
    if actual.to_bits() != expected.to_bits() {
        return Err(format!(
            "attention F16 FMA mismatch: gpu={actual} cpu={expected} gpu_bits={:#010x} cpu_bits={:#010x}",
            actual.to_bits(),
            expected.to_bits()
        ));
    }
    println!("operator=attention_f16_fma exact=true");
    Ok(())
}

fn check_attention_value_reduction(context: &VulkanContext) -> Result<(), String> {
    const SEQUENCE: usize = 12;
    let layout =
        ArenaLayout::build(32, 32, 1, 1, 1, 1, 1, SEQUENCE).map_err(|error| error.to_string())?;
    let ops = Qwen3Ops::new(context, layout, 1).map_err(|error| error.to_string())?;
    let probabilities = [
        0x39224000, 0x38354000, 0x3b39a000, 0x37ae8000, 0x380cc000, 0x37838000, 0x3958c000,
        0x38350000, 0x3d014000, 0x36000000, 0x37b00000, 0x3f770000,
    ]
    .map(f32::from_bits);
    let values = [
        0x3c58a000, 0xbef2e000, 0xbf102000, 0xbd760000, 0xbf250000, 0xbec3c000, 0x3f07e000,
        0x3f0de000, 0xbefca000, 0x3f98a000, 0xbf0d8000, 0xbf870000,
    ]
    .map(f32::from_bits);
    ops.write_f32(layout.scores, &probabilities)
        .map_err(|error| error.to_string())?;
    ops.write_f32(layout.kv_v, &values)
        .map_err(|error| error.to_string())?;

    let commands = TokenCommands::begin(context).map_err(|error| error.to_string())?;
    ops.record_attention_values(
        &commands,
        layout.scores,
        layout.kv_v,
        layout.attn,
        0,
        1,
        SEQUENCE,
        SEQUENCE,
        1,
        1,
        1,
    )
    .map_err(|error| error.to_string())?;
    commands
        .submit_and_wait()
        .map_err(|error| error.to_string())?;

    let actual = ops
        .read_f32(layout.attn, 1)
        .map_err(|error| error.to_string())?[0];
    let expected = f32::from_bits(0xbf847001);
    if actual.to_bits() != expected.to_bits() {
        return Err(format!(
            "attention value reduction mismatch: gpu={actual} cpu={expected} gpu_bits={:#010x} cpu_bits={:#010x}",
            actual.to_bits(),
            expected.to_bits()
        ));
    }
    println!("operator=attention_value_reduction exact=true");
    Ok(())
}

fn check_softmax_f16_rounding(context: &VulkanContext) -> Result<(), String> {
    const SEQUENCE: usize = 12;
    let layout =
        ArenaLayout::build(32, 32, 1, 1, 1, 1, 1, 16).map_err(|error| error.to_string())?;
    let ops = Qwen3Ops::new(context, layout, 1).map_err(|error| error.to_string())?;
    let scores = [
        0x4143b9da, 0x410c1683, 0x4106b621, 0x40e5fc8a, 0x40d57043, 0x40ca2cae, 0x40d35ea3,
        0x40b7deaf, 0x4108ed67, 0x4048c13a, 0x40f77174, 0x412972e1,
    ]
    .map(f32::from_bits);
    let expected = [
        0x3f44c000, 0x3cc28000, 0x3c8b0000, 0x3ba22000, 0x3b414000, 0x3b07e000, 0x3b354000,
        0x3a998000, 0x3c9fa000, 0x38b4a000, 0x3c0be000, 0x3e184000,
    ]
    .map(f32::from_bits);
    ops.write_f32(layout.scores, &scores)
        .map_err(|error| error.to_string())?;

    let commands = TokenCommands::begin(context).map_err(|error| error.to_string())?;
    ops.record_softmax(&commands, layout.scores, 1, SEQUENCE)
        .map_err(|error| error.to_string())?;
    commands
        .submit_and_wait()
        .map_err(|error| error.to_string())?;

    let actual = ops
        .read_f32(layout.scores, SEQUENCE)
        .map_err(|error| error.to_string())?;
    if actual != expected {
        let index = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| *actual != expected)
            .unwrap();
        return Err(format!(
            "softmax F16 rounding mismatch at {index}: gpu={} cpu={} gpu_bits={:#010x} cpu_bits={:#010x}",
            actual[index],
            expected[index],
            actual[index].to_bits(),
            expected[index].to_bits()
        ));
    }

    const SECOND_SEQUENCE: usize = 16;
    let scores = [
        0x415fd96f, 0x41292d3b, 0x40fb16d5, 0x413296b2, 0x40b14294, 0x40fc57dc, 0x40c872bd,
        0x41318456, 0x40e65438, 0x413f4013, 0x4119b514, 0x40b72e9d, 0x41436352, 0x410ba4d8,
        0x411e924b, 0x410111af,
    ]
    .map(f32::from_bits);
    let expected = [
        0x3f2bc000, 0x3cb46000, 0x3abcc000, 0x3d226000, 0x39166000, 0x3ac46000, 0x399b2000,
        0x3d17e000, 0x3a456000, 0x3db32000, 0x3c094000, 0x3934e000, 0x3de80000, 0x3b63e000,
        0x3c3a0000, 0x3aeb6000,
    ]
    .map(f32::from_bits);
    ops.write_f32(layout.scores, &scores)
        .map_err(|error| error.to_string())?;
    let commands = TokenCommands::begin(context).map_err(|error| error.to_string())?;
    ops.record_softmax(&commands, layout.scores, 1, SECOND_SEQUENCE)
        .map_err(|error| error.to_string())?;
    commands
        .submit_and_wait()
        .map_err(|error| error.to_string())?;
    let actual = ops
        .read_f32(layout.scores, SECOND_SEQUENCE)
        .map_err(|error| error.to_string())?;
    if actual != expected {
        let index = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| *actual != expected)
            .unwrap();
        return Err(format!(
            "softmax F16 rounding mismatch in second case at {index}: gpu={} cpu={} gpu_bits={:#010x} cpu_bits={:#010x}",
            actual[index],
            expected[index],
            actual[index].to_bits(),
            expected[index].to_bits()
        ));
    }
    println!("operator=softmax_f16_rounding exact=true");
    Ok(())
}

fn synthetic_q8_weight(n_in: usize, n_out: usize, salt: usize) -> Vec<u8> {
    let blocks_per_row = n_in / 32;
    let mut bytes = vec![0u8; n_out * blocks_per_row * 34];
    for row in 0..n_out {
        for block in 0..blocks_per_row {
            let offset = (row * blocks_per_row + block) * 34;
            let scale = 0.004 + ((row + block + salt) % 13) as f32 * 0.0003;
            bytes[offset..offset + 2]
                .copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            for lane in 0..32 {
                let quant = ((row * 17 + block * 11 + lane * 5 + salt) % 41) as i8 - 20;
                bytes[offset + 2 + lane] = quant as u8;
            }
        }
    }
    bytes
}

fn cpu_q8_matvec(weight: &[u8], q8: &[u8], scales: &[f32], n_in: usize) -> Vec<f32> {
    let blocks_per_row = n_in / 32;
    let row_bytes = blocks_per_row * 34;
    let mut output = vec![0.0f32; weight.len() / row_bytes];
    for (row, value) in output.iter_mut().enumerate() {
        for block in 0..blocks_per_row {
            let offset = row * row_bytes + block * 34;
            let scale =
                half::f16::from_bits(u16::from_le_bytes([weight[offset], weight[offset + 1]]))
                    .to_f32();
            let mut dot = 0i32;
            for lane in 0..32 {
                dot +=
                    (weight[offset + 2 + lane] as i8 as i32) * (q8[block * 32 + lane] as i8 as i32);
            }
            *value += scale * scales[block] * dot as f32;
        }
    }
    output
}

fn cpu_attention_scores(
    q: &[f32],
    k: &[f32],
    sequence_length: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; q_heads * sequence_length];
    let kv_width = kv_heads * head_dim;
    let group_size = q_heads / kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for head in 0..q_heads {
        let kv_head = head / group_size;
        let query: Vec<u16> = q[head * head_dim..(head + 1) * head_dim]
            .iter()
            .map(|&value| crate::ops::f32_to_f16(value))
            .collect();
        for token in 0..sequence_length {
            let key_start = token * kv_width + kv_head * head_dim;
            let key: Vec<u16> = k[key_start..key_start + head_dim]
                .iter()
                .map(|&value| crate::ops::f32_to_f16(value))
                .collect();
            scores[head * sequence_length + token] =
                crate::ops::dot_f16(&query, &key, head_dim) * scale;
        }
    }
    scores
}

fn cpu_attention_values(
    scores: &[f32],
    values: &[f32],
    sequence_length: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; q_heads * head_dim];
    let kv_width = kv_heads * head_dim;
    let group_size = q_heads / kv_heads;
    for head in 0..q_heads {
        let kv_head = head / group_size;
        let padded = sequence_length.div_ceil(256) * 256;
        let mut weights = vec![0u16; padded];
        for (target, &value) in weights
            .iter_mut()
            .zip(&scores[head * sequence_length..(head + 1) * sequence_length])
        {
            *target = crate::ops::f32_to_f16(value);
        }
        for dimension in 0..head_dim {
            let mut column = vec![0u16; padded];
            for token in 0..sequence_length {
                column[token] = crate::ops::f32_to_f16(
                    values[token * kv_width + kv_head * head_dim + dimension],
                );
            }
            output[head * head_dim + dimension] = crate::ops::dot_f16(&column, &weights, padded);
        }
    }
    output
}

fn check_close(
    name: &str,
    gpu: &[f32],
    cpu: &[f32],
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> Result<(), String> {
    if gpu.len() != cpu.len() {
        return Err(format!(
            "{name} length mismatch: gpu={} cpu={}",
            gpu.len(),
            cpu.len()
        ));
    }
    let mut max_absolute = 0.0f32;
    let mut max_relative = 0.0f32;
    let mut first_bad = None;
    for (index, (&gpu, &cpu)) in gpu.iter().zip(cpu).enumerate() {
        let absolute = if gpu.is_finite() {
            (gpu - cpu).abs()
        } else {
            f32::INFINITY
        };
        let relative = absolute / cpu.abs().max(1e-9);
        max_absolute = max_absolute.max(absolute);
        max_relative = max_relative.max(relative);
        if first_bad.is_none()
            && (!gpu.is_finite() || absolute > absolute_tolerance + relative_tolerance * cpu.abs())
        {
            first_bad = Some((index, gpu, cpu, absolute, relative));
        }
    }
    println!(
        "operator={name} max_abs={max_absolute:.3e} max_rel={max_relative:.3e} first_bad={:?}",
        first_bad.map(|value| value.0)
    );
    if let Some((index, gpu, cpu, absolute, relative)) = first_bad {
        Err(format!(
            "{name} mismatch at {index}: gpu={gpu} cpu={cpu} abs={absolute} rel={relative}"
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ArenaLayout, TokenDispatchPlan};

    #[test]
    fn qwen3_arena_regions_are_aligned_and_disjoint() {
        let layout = ArenaLayout::for_dims(1024, 3072, 16, 2, 64).unwrap();
        let regions = layout.regions();
        assert!(regions.iter().all(|region| region.offset % 16 == 0));
        assert!(regions
            .windows(2)
            .all(|pair| pair[0].end() <= pair[1].offset));
    }

    #[test]
    fn token_command_has_one_submit_boundary() {
        let plan = TokenDispatchPlan::qwen3_dense(28);
        assert_eq!(plan.queue_submissions, 1);
        assert_eq!(plan.fence_waits, 1);
        assert!(plan.dispatches > 28);
    }
}
