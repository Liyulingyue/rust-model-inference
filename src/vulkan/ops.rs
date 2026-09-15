use super::{GpuBuffer, VulkanContext, VulkanError};
use crate::models::qwen3::trunk::Qwen3Config;
use ash::vk;
use std::collections::HashMap;
#[cfg(test)]
use std::sync::atomic::Ordering;
use std::sync::MutexGuard;

const QUANTIZE_Q8_0_SHADER: &[u8] = include_bytes!("../../shaders/bin/quantize_q8_0.spv");
const QUANTIZE_Q8_K_SHADER: &[u8] = include_bytes!("../../shaders/bin/quantize_q8_k.spv");
const Q8_MATMUL_GROUPED_SHADER: &[u8] = include_bytes!("../../shaders/bin/q8_matmul_grouped.spv");
const Q4_0_MATMUL_SHADER: &[u8] = include_bytes!("../../shaders/bin/q4_0_matmul.spv");
const Q4_1_MATMUL_SHADER: &[u8] = include_bytes!("../../shaders/bin/q4_1_matmul.spv");
const Q4_K_MATMUL_SHADER: &[u8] = include_bytes!("../../shaders/bin/q4_k_matmul.spv");
const Q5_K_MATMUL_SHADER: &[u8] = include_bytes!("../../shaders/bin/q5_k_matmul.spv");
const Q6_K_MATMUL_SHADER: &[u8] = include_bytes!("../../shaders/bin/q6_k_matmul.spv");
const F16_MATMUL_SHADER: &[u8] = include_bytes!("../../shaders/bin/f16_matmul.spv");
const BF16_MATMUL_SHADER: &[u8] = include_bytes!("../../shaders/bin/bf16_matmul.spv");
const F32_MATMUL_SHADER: &[u8] = include_bytes!("../../shaders/bin/f32_matmul.spv");
const RMS_NORM_SHADER: &[u8] = include_bytes!("../../shaders/bin/rms_norm.spv");
const QK_NORM_ROPE_SHADER: &[u8] = include_bytes!("../../shaders/bin/qk_norm_rope.spv");
const KV_WRITE_SHADER: &[u8] = include_bytes!("../../shaders/bin/kv_write.spv");
const ATTENTION_SCORES_SHADER: &[u8] = include_bytes!("../../shaders/bin/attention_scores.spv");
const SOFTMAX_SHADER: &[u8] = include_bytes!("../../shaders/bin/softmax.spv");
const ATTENTION_VALUES_SHADER: &[u8] = include_bytes!("../../shaders/bin/attention_values.spv");
const SILU_MUL_SHADER: &[u8] = include_bytes!("../../shaders/bin/silu_mul.spv");
const ADD_SHADER: &[u8] = include_bytes!("../../shaders/bin/add.spv");
const QWEN35_DENSE_PREPARE_SHADER: &[u8] =
    include_bytes!("../../shaders/bin/qwen35_dense_prepare.spv");
const QWEN35_ATTENTION_SHADER: &[u8] = include_bytes!("../../shaders/bin/qwen35_attention.spv");
const QWEN35_RECURRENT_CONV_SHADER: &[u8] =
    include_bytes!("../../shaders/bin/qwen35_recurrent_conv.spv");
const QWEN35_RECURRENT_SSM_SHADER: &[u8] =
    include_bytes!("../../shaders/bin/qwen35_recurrent_ssm.spv");

const QUANTIZE: usize = 0;
const QUANTIZE_K: usize = 1;
const Q8_MATMUL_GROUPED: usize = 2;
const Q4_0_MATMUL: usize = 3;
const Q4_1_MATMUL: usize = 4;
const Q4_K_MATMUL: usize = 5;
const Q6_K_MATMUL: usize = 6;
const F16_MATMUL: usize = 7;
const BF16_MATMUL: usize = 8;
const RMS_NORM: usize = 9;
const QK_NORM_ROPE: usize = 10;
const KV_WRITE: usize = 11;
const ATTENTION_SCORES: usize = 12;
const SOFTMAX: usize = 13;
const ATTENTION_VALUES: usize = 14;
const SILU_MUL: usize = 15;
const ADD: usize = 16;
const QWEN35_DENSE_PREPARE: usize = 17;
const QWEN35_ATTENTION: usize = 18;
const QWEN35_RECURRENT_CONV: usize = 19;
const QWEN35_RECURRENT_SSM: usize = 20;
const Q5_K_MATMUL: usize = 21;
const F32_MATMUL: usize = 22;
const OPERATOR_SHADERS: [&[u8]; 23] = [
    QUANTIZE_Q8_0_SHADER,
    QUANTIZE_Q8_K_SHADER,
    Q8_MATMUL_GROUPED_SHADER,
    Q4_0_MATMUL_SHADER,
    Q4_1_MATMUL_SHADER,
    Q4_K_MATMUL_SHADER,
    Q6_K_MATMUL_SHADER,
    F16_MATMUL_SHADER,
    BF16_MATMUL_SHADER,
    RMS_NORM_SHADER,
    QK_NORM_ROPE_SHADER,
    KV_WRITE_SHADER,
    ATTENTION_SCORES_SHADER,
    SOFTMAX_SHADER,
    ATTENTION_VALUES_SHADER,
    SILU_MUL_SHADER,
    ADD_SHADER,
    QWEN35_DENSE_PREPARE_SHADER,
    QWEN35_ATTENTION_SHADER,
    QWEN35_RECURRENT_CONV_SHADER,
    QWEN35_RECURRENT_SSM_SHADER,
    Q5_K_MATMUL_SHADER,
    F32_MATMUL_SHADER,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuWeightFormat {
    Q8_0,
    Q4_0,
    Q4_1,
    Q4_K,
    Q5_K,
    Q6_K,
    F16,
    BF16,
    F32,
}

impl GpuWeightFormat {
    pub(crate) fn from_ggml_type(
        value: crate::core::tensor::GGMLType,
    ) -> Result<Self, VulkanError> {
        match value {
            crate::core::tensor::GGMLType::Q8_0 => Ok(Self::Q8_0),
            crate::core::tensor::GGMLType::Q4_0 => Ok(Self::Q4_0),
            crate::core::tensor::GGMLType::Q4_1 => Ok(Self::Q4_1),
            crate::core::tensor::GGMLType::Q4K => Ok(Self::Q4_K),
            crate::core::tensor::GGMLType::Q5K => Ok(Self::Q5_K),
            crate::core::tensor::GGMLType::Q6K => Ok(Self::Q6_K),
            crate::core::tensor::GGMLType::F16 => Ok(Self::F16),
            crate::core::tensor::GGMLType::BF16 => Ok(Self::BF16),
            crate::core::tensor::GGMLType::F32 => Ok(Self::F32),
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
            Self::Q4_K => (256, 144, Q4_K_MATMUL),
            Self::Q5_K => (256, 176, Q5_K_MATMUL),
            Self::Q6_K => (256, 210, Q6_K_MATMUL),
            Self::F16 => (1, 2, F16_MATMUL),
            Self::BF16 => (1, 2, BF16_MATMUL),
            Self::F32 => (1, 4, F32_MATMUL),
        }
    }
}

pub(crate) fn fill_rope_neox(coefficients: &mut [f32], position: usize, freq_base: f32) {
    debug_assert!(!coefficients.is_empty() && coefficients.len() % 2 == 0);
    let half = coefficients.len() / 2;
    for index in 0..half {
        let inverse_frequency =
            1.0f32 / freq_base.powf((2 * index) as f32 / coefficients.len() as f32);
        let theta = position as f32 * inverse_frequency;
        let (cosine, sine) = crate::ops::rope_sin_cos(theta);
        coefficients[index] = cosine;
        coefficients[index + half] = sine;
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
    pub(crate) rope: ArenaRegion,
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

    pub(crate) fn qwen3(
        config: &Qwen3Config,
        capacity: usize,
        max_rows: usize,
    ) -> Result<Self, VulkanError> {
        if config.n_embd_head_k != config.n_embd_head_v {
            return Err(VulkanError::UnsupportedShape(format!(
                "different Qwen3 key/value head dimensions: {}/{}",
                config.n_embd_head_k, config.n_embd_head_v
            )));
        }
        Self::build_rows(
            config.n_embd,
            config.n_ff,
            config.n_head,
            config.n_head_kv,
            config.n_embd_head_k,
            config.vocab,
            config.n_layer,
            capacity,
            max_rows,
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
        Self::build_rows(
            n_embd, n_ff, n_head, n_head_kv, head_dim, vocab, n_layer, capacity, 1,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_rows(
        n_embd: usize,
        n_ff: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        vocab: usize,
        n_layer: usize,
        capacity: usize,
        rows: usize,
    ) -> Result<Self, VulkanError> {
        if [
            n_embd, n_ff, n_head, n_head_kv, head_dim, vocab, n_layer, capacity, rows,
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
        let score_len = product("attention scores", &[rows, n_head, capacity])?;
        let kv_cache_len = product("KV cache", &[n_layer, capacity, kv_len])?;
        let kv_delta_len = product("KV delta", &[n_layer, rows, kv_len])?;
        let mut cursor = 0usize;

        let x = f32_region(&mut cursor, product("x rows", &[rows, n_embd])?)?;
        let normed = f32_region(&mut cursor, product("normed rows", &[rows, n_embd])?)?;
        let q = f32_region(&mut cursor, product("q rows", &[rows, q_len])?)?;
        let k = f32_region(&mut cursor, product("k rows", &[rows, kv_len])?)?;
        let v = f32_region(&mut cursor, product("v rows", &[rows, kv_len])?)?;
        let attn = f32_region(&mut cursor, product("attn rows", &[rows, q_len])?)?;
        let projection = f32_region(&mut cursor, product("projection rows", &[rows, n_embd])?)?;
        let gate = f32_region(&mut cursor, product("gate rows", &[rows, n_ff])?)?;
        let up = f32_region(&mut cursor, product("up rows", &[rows, n_ff])?)?;
        let down = f32_region(&mut cursor, product("down rows", &[rows, n_embd])?)?;
        let logits = f32_region(&mut cursor, vocab)?;
        let q8 = region(&mut cursor, product("Q8 rows", &[rows, q8_len])?)?;
        let q8_scales = f32_region(
            &mut cursor,
            product("q8_scales rows", &[rows, q8_len.div_ceil(32)])?,
        )?;
        let q4_1_input_sums = f32_region(
            &mut cursor,
            product("q4_1_input_sums rows", &[rows, q8_len.div_ceil(32)])?,
        )?;
        let q8k = region(&mut cursor, product("Q8K rows", &[rows, q8_len])?)?;
        let q8k_scales = f32_region(
            &mut cursor,
            product("q8k_scales rows", &[rows, q8_len.div_ceil(256)])?,
        )?;
        let scores = f32_region(&mut cursor, score_len)?;
        let kv_k = f32_region(&mut cursor, kv_cache_len)?;
        let kv_v = f32_region(&mut cursor, kv_cache_len)?;
        let kv_delta_k = f32_region(&mut cursor, kv_delta_len)?;
        let kv_delta_v = f32_region(&mut cursor, kv_delta_len)?;
        let rope = f32_region(&mut cursor, product("RoPE rows", &[rows, head_dim])?)?;

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
            rope,
            total_size: cursor,
        })
    }

    pub(crate) fn regions(&self) -> [ArenaRegion; 22] {
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
            self.rope,
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
    guard: MutexGuard<'a, super::CommandSubmission>,
}

impl<'a> TokenCommands<'a> {
    pub(crate) fn begin(context: &'a VulkanContext) -> Result<Self, VulkanError> {
        let mut guard = context
            .mutex
            .lock()
            .map_err(|_| VulkanError::InitFailed("Vulkan command mutex poisoned".into()))?;
        context.begin_commands(&mut guard)?;
        Ok(Self {
            context,
            command: context.command_buffer,
            guard,
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

    pub(crate) fn submit_and_wait(mut self) -> Result<(), VulkanError> {
        self.context.submit_commands(&mut self.guard)
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
        if !(1..=3).contains(&count) {
            return Err(VulkanError::UnsupportedShape(
                "grouped Vulkan matmul needs 1 to 3 weights".into(),
            ));
        }
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

struct BatchedLinearLayout {
    max_rows: usize,
    max_n_in: usize,
    max_n_out: usize,
    input: ArenaRegion,
    q8: ArenaRegion,
    scales: ArenaRegion,
    sums: ArenaRegion,
    output: ArenaRegion,
    size: usize,
}

impl BatchedLinearLayout {
    fn new(max_rows: usize, max_n_in: usize, max_n_out: usize) -> Result<Self, VulkanError> {
        if max_rows == 0 || max_n_in == 0 || max_n_out == 0 {
            return Err(VulkanError::UnsupportedShape(
                "batched linear maxima must be nonzero".into(),
            ));
        }
        let inputs = max_rows
            .checked_mul(max_n_in)
            .ok_or(VulkanError::OutOfMemory)?;
        let outputs = max_rows
            .checked_mul(max_n_out)
            .ok_or(VulkanError::OutOfMemory)?;
        let blocks = max_rows
            .checked_mul(max_n_in.div_ceil(32))
            .ok_or(VulkanError::OutOfMemory)?;
        let mut size = 0;
        let input = f32_region(&mut size, inputs)?;
        // Q8_0 and Q8_K projections run separately and share packed scratch.
        let q8 = region(&mut size, inputs)?;
        let scales = f32_region(&mut size, blocks)?;
        let sums = f32_region(&mut size, blocks)?;
        let output = f32_region(&mut size, outputs)?;
        as_u32((size - 1) / 4, "batched linear arena word span")?;
        Ok(Self {
            max_rows,
            max_n_in,
            max_n_out,
            input,
            q8,
            scales,
            sums,
            output,
            size,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn validate(
        &self,
        limits: &vk::PhysicalDeviceLimits,
        weight_len: usize,
        format: GpuWeightFormat,
        input_len: usize,
        rows: usize,
        n_in: usize,
        n_out: usize,
        output_len: usize,
    ) -> Result<usize, VulkanError> {
        if rows == 0
            || rows > self.max_rows
            || n_in == 0
            || n_in > self.max_n_in
            || n_out == 0
            || n_out > self.max_n_out
        {
            return Err(VulkanError::UnsupportedShape(
                "batched linear shape exceeds constructor maxima".into(),
            ));
        }
        let inputs = rows.checked_mul(n_in).ok_or(VulkanError::OutOfMemory)?;
        let outputs = rows.checked_mul(n_out).ok_or(VulkanError::OutOfMemory)?;
        if input_len != inputs || output_len < outputs {
            return Err(VulkanError::UnsupportedShape(
                "batched linear input/output slice length mismatch".into(),
            ));
        }
        let (block, _, _) = format.layout();
        if block != 1 {
            quantize_rows_push(
                self.size,
                self.input,
                self.q8,
                self.scales,
                Some(self.sums),
                n_in,
                rows,
                n_in,
                block,
            )?;
            row_dispatch(n_in / block, rows, limits)?;
        }
        // Reuse the recorder's full shape/address checks before any upload or write.
        let bindings = OperatorBindings {
            descriptor_set: vk::DescriptorSet::null(),
            sizes: [
                u64::try_from(weight_len).map_err(|_| VulkanError::OutOfMemory)?,
                0,
                0,
            ],
            weight_formats: [Some(format), None, None],
        };
        matmul_rows_push(
            self.size,
            limits,
            bindings,
            if block == 1 { self.input } else { self.q8 },
            self.scales,
            Some(self.sums),
            &[(self.output, n_out, f32_bytes(n_out)?)],
            n_in,
            rows,
            n_in,
        )?;
        Ok(outputs)
    }
}

/// A projection runtime for immutable weight slices kept at stable addresses
/// throughout its lifetime. Changing or reusing a source allocation requires a
/// new runtime. Descriptor capacity includes the arena's descriptor set.
pub(crate) struct BatchedLinearRuntime {
    ops: std::mem::ManuallyDrop<Qwen3Ops<'static>>,
    layout: BatchedLinearLayout,
    weights: HashMap<(usize, usize), (GpuBuffer, OperatorBindings)>,
    weight_capacity: usize,
    #[cfg(test)]
    begin_commands: fn(&'static VulkanContext) -> Result<TokenCommands<'static>, VulkanError>,
}

impl BatchedLinearRuntime {
    pub(crate) fn new(
        context: &'static VulkanContext,
        max_rows: usize,
        max_n_in: usize,
        max_n_out: usize,
        descriptor_capacity: usize,
    ) -> Result<Self, VulkanError> {
        let layout = BatchedLinearLayout::new(max_rows, max_n_in, max_n_out)?;
        matmul_dispatch(max_n_out, max_rows, 1, &context.limits)?;
        let ops = Qwen3Ops::new_with_size(context, layout.size, descriptor_capacity)?;
        Ok(Self {
            ops: std::mem::ManuallyDrop::new(ops),
            layout,
            weights: HashMap::new(),
            weight_capacity: descriptor_capacity.saturating_sub(1),
            #[cfg(test)]
            begin_commands: TokenCommands::begin,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matmul_rows(
        &mut self,
        weight_bytes: &[u8],
        format: GpuWeightFormat,
        input: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        output: &mut [f32],
    ) -> Result<(), VulkanError> {
        let count = self.layout.validate(
            &self.ops.context.limits,
            weight_bytes.len(),
            format,
            input.len(),
            rows,
            n_in,
            n_out,
            output.len(),
        )?;
        let context = self.ops.context;
        #[cfg(test)]
        let commands = (self.begin_commands)(context)?;
        #[cfg(not(test))]
        let commands = TokenCommands::begin(context)?;
        let key = (weight_bytes.as_ptr() as usize, weight_bytes.len());
        let bindings = if let Some((_, bindings)) = self.weights.get(&key) {
            if bindings.weight_format(1)? != format {
                return Err(VulkanError::UnsupportedShape(
                    "cached batched linear weight format changed".into(),
                ));
            }
            *bindings
        } else {
            if self.weights.len() >= self.weight_capacity {
                return Err(VulkanError::UnsupportedShape(
                    "batched linear descriptor capacity exhausted".into(),
                ));
            }
            let buffer = unsafe { self.ops.context.upload_static(weight_bytes)? };
            let bindings = match self.ops.bind_weight_buffers(&[buffer], &[format]) {
                Ok(bindings) => bindings,
                Err(error) => {
                    unsafe { self.ops.context.destroy_buffer(&buffer) };
                    return Err(error);
                }
            };
            // Initialization succeeded. Retain this valid upload on later errors
            // so every GPU allocation stays owned and can be reused or dropped.
            self.weights.insert(key, (buffer, bindings));
            bindings
        };
        self.ops.write_f32(self.layout.input, input)?;
        self.ops.record_weight_matmul_rows(
            &commands,
            bindings,
            self.layout.input,
            self.layout.q8,
            self.layout.scales,
            self.layout.sums,
            self.layout.q8,
            self.layout.scales,
            &[(self.layout.output, n_out, f32_bytes(n_out)?)],
            n_in,
            rows,
            n_in,
        )?;
        commands.submit_and_wait()?;
        output[..count].copy_from_slice(self.ops.read_f32(self.layout.output, count)?);
        Ok(())
    }
}

impl Drop for BatchedLinearRuntime {
    fn drop(&mut self) {
        let context = self.ops.context;
        if unsafe {
            context.destroy_completed_buffers(self.weights.values().map(|(buffer, _)| buffer))
        }
        .is_err()
        {
            // Keep runtime resources alive while shared completion is unknown.
            return;
        }
        // Qwen3Ops acquires the same mutex in Drop.
        unsafe { std::mem::ManuallyDrop::drop(&mut self.ops) };
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
        Self::new_with_size(context, layout.total_size(), descriptor_capacity)
    }

    pub(crate) fn new_with_size(
        context: &'a VulkanContext,
        arena_size: usize,
        descriptor_capacity: usize,
    ) -> Result<Self, VulkanError> {
        if arena_size == 0 {
            return Err(VulkanError::UnsupportedShape(
                "Vulkan arena must not be empty".into(),
            ));
        }
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

        let arena = match unsafe { context.allocate_session_buffer(arena_size) } {
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
        unsafe { std::ptr::write_bytes(arena.mapped, 0, arena_size) };

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
        let push = quantize_rows_push(
            self.arena.size as usize,
            input,
            q8,
            scales,
            Some(q4_1_input_sums),
            count,
            1,
            count,
            32,
        )?;
        let dispatch = row_dispatch(count / 32, 1, &self.context.limits)?;
        self.record_linear_dispatch(commands, QUANTIZE, self.arena_bindings, &push, dispatch);
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
        let push = quantize_rows_push(
            self.arena.size as usize,
            input,
            q8,
            scales,
            None,
            count,
            1,
            count,
            256,
        )?;
        let dispatch = row_dispatch(count / 256, 1, &self.context.limits)?;
        self.record_linear_dispatch(commands, QUANTIZE_K, self.arena_bindings, &push, dispatch);
        Ok(())
    }

    fn record_linear_dispatch(
        &self,
        commands: &TokenCommands<'_>,
        pipeline: usize,
        bindings: OperatorBindings,
        push: &[u32],
        dispatch: [u32; 3],
    ) {
        unsafe {
            commands.bind(
                self.pipelines[pipeline],
                self.context.pipeline_layout,
                &[bindings.descriptor_set],
                bytemuck::cast_slice(push),
            );
            commands.dispatch(dispatch[0], dispatch[1], dispatch[2]);
            commands.barrier();
        }
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
        self.record_rms_norm_rows(
            commands, bindings, input, output, count, eps, 1, count, count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_rms_norm_rows(
        &self,
        commands: &TokenCommands<'_>,
        bindings: OperatorBindings,
        input: ArenaRegion,
        output: ArenaRegion,
        count: usize,
        eps: f32,
        rows: usize,
        input_stride: usize,
        output_stride: usize,
    ) -> Result<(), VulkanError> {
        bindings.require(0, f32_bytes(count)?, "RMS norm weight")?;
        let push = [
            self.f32_rows_word(input, rows, input_stride, count, "RMS norm input")?,
            0,
            self.f32_rows_word(output, rows, output_stride, count, "RMS norm output")?,
            as_u32(count, "RMS norm length")?,
            as_u32(rows, "RMS norm rows")?,
            eps.to_bits(),
            as_u32(input_stride, "RMS input stride")?,
            as_u32(output_stride, "RMS output stride")?,
        ];
        let (x, y) = super::dispatch_grid(rows, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[RMS_NORM],
                self.context.pipeline_layout,
                &[bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, 1);
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
        self.record_q8_matvec_group(
            commands,
            bindings,
            q8,
            scales,
            &[(output, n_out)],
            n_in,
            None,
        )
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
        let packed_outputs: Vec<_> = outputs
            .iter()
            .map(|&(region, n_out)| {
                Ok((
                    region,
                    n_out,
                    n_out.checked_mul(4).ok_or(VulkanError::OutOfMemory)?,
                ))
            })
            .collect::<Result<_, _>>()?;
        self.record_weight_matmul_rows(
            commands,
            bindings,
            input,
            q8,
            q8_scales,
            q4_1_input_sums,
            q8k,
            q8k_scales,
            &packed_outputs,
            n_in,
            1,
            n_in,
        )
    }

    /// Input stride is in f32 elements; output strides are in bytes. Quantized
    /// activations, scales and sums are packed independently of the input stride.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_weight_matmul_rows(
        &self,
        commands: &TokenCommands<'_>,
        bindings: OperatorBindings,
        input: ArenaRegion,
        q8: ArenaRegion,
        q8_scales: ArenaRegion,
        q4_1_input_sums: ArenaRegion,
        q8k: ArenaRegion,
        q8k_scales: ArenaRegion,
        outputs: &[(ArenaRegion, usize, usize)],
        n_in: usize,
        token_rows: usize,
        input_stride: usize,
    ) -> Result<(), VulkanError> {
        let format = bindings.weight_format(outputs.len())?;
        let (activation, scales, quantize) = match format {
            GpuWeightFormat::F16 | GpuWeightFormat::BF16 | GpuWeightFormat::F32 => {
                (input, q8_scales, None)
            }
            GpuWeightFormat::Q4_K | GpuWeightFormat::Q5_K | GpuWeightFormat::Q6_K => {
                let push = quantize_rows_push(
                    self.arena.size as usize,
                    input,
                    q8k,
                    q8k_scales,
                    None,
                    n_in,
                    token_rows,
                    input_stride,
                    256,
                )?;
                let dispatch = row_dispatch(n_in / 256, token_rows, &self.context.limits)?;
                (q8k, q8k_scales, Some((QUANTIZE_K, push, dispatch)))
            }
            _ => {
                let push = quantize_rows_push(
                    self.arena.size as usize,
                    input,
                    q8,
                    q8_scales,
                    Some(q4_1_input_sums),
                    n_in,
                    token_rows,
                    input_stride,
                    32,
                )?;
                let dispatch = row_dispatch(n_in / 32, token_rows, &self.context.limits)?;
                (q8, q8_scales, Some((QUANTIZE, push, dispatch)))
            }
        };
        let (push, dispatch) = matmul_rows_push(
            self.arena.size as usize,
            &self.context.limits,
            bindings,
            activation,
            scales,
            Some(q4_1_input_sums),
            outputs,
            n_in,
            token_rows,
            input_stride,
        )?;
        // All validation completes before the first command is recorded.
        if let Some((pipeline, push, dispatch)) = quantize {
            self.record_linear_dispatch(commands, pipeline, self.arena_bindings, &push, dispatch);
        }
        self.record_linear_dispatch(commands, format.layout().2, bindings, &push, dispatch);
        Ok(())
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
        let outputs: Vec<_> = outputs
            .iter()
            .map(|&(region, n_out)| {
                Ok((
                    region,
                    n_out,
                    n_out.checked_mul(4).ok_or(VulkanError::OutOfMemory)?,
                ))
            })
            .collect::<Result<_, _>>()?;
        let (push, dispatch) = matmul_rows_push(
            self.arena.size as usize,
            &self.context.limits,
            bindings,
            q8,
            scales,
            q4_1_input_sums,
            &outputs,
            n_in,
            1,
            n_in,
        )?;
        self.record_linear_dispatch(
            commands,
            bindings.weight_format(outputs.len())?.layout().2,
            bindings,
            &push,
            dispatch,
        );
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
        self.record_qk_norm_rope_rows(
            commands,
            bindings,
            q,
            k,
            q_heads,
            k_heads,
            head_dim,
            rope,
            eps,
            normalize_q,
            normalize_k,
            1,
        )
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
        self.record_kv_write_rows(
            commands,
            k,
            v,
            cache_k,
            cache_v,
            delta_k,
            delta_v,
            layer,
            position,
            layer_count,
            capacity,
            kv_count,
            1,
        )
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
        let base = sequence_length
            .checked_sub(1)
            .ok_or_else(|| VulkanError::UnsupportedShape("empty attention".into()))?;
        self.record_attention_scores_rows(
            commands,
            q,
            cache_k,
            scores,
            layer,
            layer_count,
            base,
            capacity,
            q_heads,
            kv_heads,
            head_dim,
            1,
        )
    }

    fn record_softmax(
        &self,
        commands: &TokenCommands<'_>,
        scores: ArenaRegion,
        heads: usize,
        sequence_length: usize,
    ) -> Result<(), VulkanError> {
        let base = sequence_length
            .checked_sub(1)
            .ok_or_else(|| VulkanError::UnsupportedShape("empty softmax".into()))?;
        self.record_softmax_rows(commands, scores, heads, base, 1)
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
        let base = sequence_length
            .checked_sub(1)
            .ok_or_else(|| VulkanError::UnsupportedShape("empty attention".into()))?;
        self.record_attention_values_rows(
            commands,
            scores,
            cache_v,
            output,
            layer,
            layer_count,
            base,
            capacity,
            q_heads,
            kv_heads,
            head_dim,
            1,
        )
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
        let base = sequence_length
            .checked_sub(1)
            .ok_or_else(|| VulkanError::UnsupportedShape("empty attention".into()))?;
        self.record_attention_rows(
            commands,
            q,
            cache_k,
            cache_v,
            scores,
            output,
            layer,
            layer_count,
            base,
            capacity,
            q_heads,
            kv_heads,
            head_dim,
            1,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_qk_norm_rope_rows(
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
        rows: usize,
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
            self.f32_rows_word(q, rows, q_count, q_count, "Q vector")?,
            self.f32_rows_word(k, rows, k_count, k_count, "K vector")?,
            0,
            0,
            as_u32(q_heads, "Q head count")?,
            as_u32(k_heads, "K head count")?,
            as_u32(head_dim, "Q/K head dimension")?,
            self.f32_rows_word(rope, rows, head_dim, head_dim, "RoPE coefficients")?,
            u32::from(normalize_q) | (u32::from(normalize_k) << 1),
            eps.to_bits(),
            as_u32(rows, "Q/K rows")?,
            as_u32(q_count, "Q stride")?,
            as_u32(k_count, "K stride")?,
            as_u32(head_dim, "RoPE stride")?,
        ];
        let z = row_dispatch(1, rows, &self.context.limits)?[2];
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
            commands.dispatch(q_heads.max(k_heads) as u32, 2, z);
            commands.barrier();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_kv_write_rows(
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
        rows: usize,
    ) -> Result<(), VulkanError> {
        if layer >= layer_count
            || rows == 0
            || position.checked_add(rows).is_none_or(|end| end > capacity)
            || kv_count == 0
        {
            return Err(VulkanError::UnsupportedShape(format!(
                "invalid KV write layer={layer}/{layer_count} position={position}/{capacity} width={kv_count}"
            )));
        }
        let cache_count = layer_count
            .checked_mul(capacity)
            .and_then(|value| value.checked_mul(kv_count))
            .ok_or(VulkanError::OutOfMemory)?;
        let delta_count = layer_count
            .checked_mul(rows)
            .and_then(|count| count.checked_mul(kv_count))
            .ok_or(VulkanError::OutOfMemory)?;
        let push = [
            self.f32_rows_word(k, rows, kv_count, kv_count, "new K")?,
            self.f32_rows_word(v, rows, kv_count, kv_count, "new V")?,
            self.f32_rows_word(cache_k, 1, cache_count, cache_count, "K cache")?,
            self.f32_rows_word(cache_v, 1, cache_count, cache_count, "V cache")?,
            self.f32_rows_word(delta_k, 1, delta_count, delta_count, "K delta")?,
            self.f32_rows_word(delta_v, 1, delta_count, delta_count, "V delta")?,
            as_u32(layer, "KV layer")?,
            as_u32(position, "KV position")?,
            as_u32(capacity, "KV capacity")?,
            as_u32(kv_count, "KV width")?,
            as_u32(rows, "KV rows")?,
        ];
        let [x, y, z] = row_dispatch(kv_count.div_ceil(64), rows, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[KV_WRITE],
                self.context.pipeline_layout,
                &[self.arena_bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, z);
            commands.barrier();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_attention_scores_rows(
        &self,
        commands: &TokenCommands<'_>,
        q: ArenaRegion,
        cache_k: ArenaRegion,
        scores: ArenaRegion,
        layer: usize,
        layer_count: usize,
        base_position: usize,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let sequence_length = base_position
            .checked_add(rows)
            .ok_or(VulkanError::OutOfMemory)?;
        if rows == 0 {
            return Err(VulkanError::UnsupportedShape(
                "empty attention chunk".into(),
            ));
        }
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
            self.f32_rows_word(q, rows, q_count, q_count, "attention query")?,
            self.f32_rows_word(cache_k, 1, cache_count, cache_count, "attention K cache")?,
            self.f32_rows_word(scores, rows, score_count, score_count, "attention scores")?,
            as_u32(layer, "attention layer")?,
            as_u32(sequence_length, "attention sequence length")?,
            as_u32(capacity, "attention capacity")?,
            as_u32(q_heads, "attention Q heads")?,
            as_u32(kv_heads, "attention KV heads")?,
            as_u32(head_dim, "attention head dimension")?,
            (1.0 / (head_dim as f32).sqrt()).to_bits(),
            as_u32(base_position, "attention base position")?,
            as_u32(rows, "attention rows")?,
        ];
        let [x, y, z] = row_dispatch(score_count.div_ceil(64), rows, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[ATTENTION_SCORES],
                self.context.pipeline_layout,
                &[self.arena_bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, z);
            commands.barrier();
        }
        Ok(())
    }

    fn record_softmax_rows(
        &self,
        commands: &TokenCommands<'_>,
        scores: ArenaRegion,
        heads: usize,
        base_position: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let sequence_length = base_position
            .checked_add(rows)
            .ok_or(VulkanError::OutOfMemory)?;
        if heads == 0 || rows == 0 {
            return Err(VulkanError::UnsupportedShape(
                "softmax heads and sequence length must be nonzero".into(),
            ));
        }
        let count = heads
            .checked_mul(sequence_length)
            .ok_or(VulkanError::OutOfMemory)?;
        let push = [
            self.f32_rows_word(scores, rows, count, count, "softmax scores")?,
            as_u32(heads, "softmax heads")?,
            as_u32(sequence_length, "softmax sequence length")?,
            as_u32(base_position, "softmax base position")?,
            as_u32(rows, "softmax rows")?,
        ];
        let [x, y, z] = row_dispatch(heads, rows, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[SOFTMAX],
                self.context.pipeline_layout,
                &[self.arena_bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, z);
            commands.barrier();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_attention_values_rows(
        &self,
        commands: &TokenCommands<'_>,
        scores: ArenaRegion,
        cache_v: ArenaRegion,
        output: ArenaRegion,
        layer: usize,
        layer_count: usize,
        base_position: usize,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let sequence_length = base_position
            .checked_add(rows)
            .ok_or(VulkanError::OutOfMemory)?;
        if rows == 0 {
            return Err(VulkanError::UnsupportedShape(
                "empty attention chunk".into(),
            ));
        }
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
            self.f32_rows_word(
                scores,
                rows,
                score_count,
                score_count,
                "attention probabilities",
            )?,
            self.f32_rows_word(cache_v, 1, cache_count, cache_count, "attention V cache")?,
            self.f32_rows_word(output, rows, output_count, output_count, "attention output")?,
            as_u32(layer, "attention layer")?,
            as_u32(sequence_length, "attention sequence length")?,
            as_u32(capacity, "attention capacity")?,
            as_u32(q_heads, "attention Q heads")?,
            as_u32(kv_heads, "attention KV heads")?,
            as_u32(head_dim, "attention head dimension")?,
            as_u32(base_position, "attention base position")?,
            as_u32(rows, "attention rows")?,
        ];
        let [x, y, z] = row_dispatch(output_count.div_ceil(64), rows, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[ATTENTION_VALUES],
                self.context.pipeline_layout,
                &[self.arena_bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, z);
            commands.barrier();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_attention_rows(
        &self,
        commands: &TokenCommands<'_>,
        q: ArenaRegion,
        cache_k: ArenaRegion,
        cache_v: ArenaRegion,
        scores: ArenaRegion,
        output: ArenaRegion,
        layer: usize,
        layer_count: usize,
        base_position: usize,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        // Validate the later stages before scores records its first dispatch.
        let output_count = product("attention output", &[q_heads, head_dim])?;
        let cache_count = product(
            "attention cache",
            &[layer_count, capacity, kv_heads, head_dim],
        )?;
        self.f32_rows_word(output, rows, output_count, output_count, "attention output")?;
        self.f32_rows_word(cache_v, 1, cache_count, cache_count, "attention V cache")?;
        row_dispatch(q_heads, rows, &self.context.limits)?;
        row_dispatch(output_count.div_ceil(64), rows, &self.context.limits)?;
        self.record_attention_scores_rows(
            commands,
            q,
            cache_k,
            scores,
            layer,
            layer_count,
            base_position,
            capacity,
            q_heads,
            kv_heads,
            head_dim,
            rows,
        )?;
        self.record_softmax_rows(commands, scores, q_heads, base_position, rows)?;
        self.record_attention_values_rows(
            commands,
            scores,
            cache_v,
            output,
            layer,
            layer_count,
            base_position,
            capacity,
            q_heads,
            kv_heads,
            head_dim,
            rows,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_qwen35_dense_prepare(
        &self,
        commands: &TokenCommands<'_>,
        bindings: OperatorBindings,
        raw_q: ArenaRegion,
        raw_k: ArenaRegion,
        v: ArenaRegion,
        q: ArenaRegion,
        gate: ArenaRegion,
        cache_k: ArenaRegion,
        cache_v: ArenaRegion,
        delta_k: ArenaRegion,
        delta_v: ArenaRegion,
        rope: ArenaRegion,
        layer: usize,
        layer_count: usize,
        position: usize,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        eps: f32,
        rows: usize,
    ) -> Result<(), VulkanError> {
        if layer >= layer_count
            || rows == 0
            || position.checked_add(rows).is_none_or(|end| end > capacity)
            || q_heads == 0
            || kv_heads == 0
            || q_heads % kv_heads != 0
            || head_dim == 0
            || rope_dim == 0
            || rope_dim > head_dim
            || rope_dim % 2 != 0
        {
            return Err(VulkanError::UnsupportedShape(format!(
                "invalid Qwen3.5 dense prepare layer={layer}/{layer_count} position={position}/{capacity} heads={q_heads}/{kv_heads} dims={head_dim}/{rope_dim}"
            )));
        }
        let q_count = q_heads
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let raw_q_count = q_count.checked_mul(2).ok_or(VulkanError::OutOfMemory)?;
        let kv_count = kv_heads
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let cache_count = layer_count
            .checked_mul(capacity)
            .and_then(|count| count.checked_mul(kv_count))
            .ok_or(VulkanError::OutOfMemory)?;
        let delta_count = layer_count
            .checked_mul(rows)
            .and_then(|count| count.checked_mul(kv_count))
            .ok_or(VulkanError::OutOfMemory)?;
        bindings.require(0, f32_bytes(head_dim)?, "Qwen3.5 Q norm")?;
        bindings.require(1, f32_bytes(head_dim)?, "Qwen3.5 K norm")?;
        let push = [
            self.f32_word(
                raw_q,
                raw_q_count
                    .checked_mul(rows)
                    .ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 raw Q/gate",
            )?,
            self.f32_word(
                raw_k,
                kv_count.checked_mul(rows).ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 raw K",
            )?,
            self.f32_word(
                v,
                kv_count.checked_mul(rows).ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 V",
            )?,
            self.f32_word(
                q,
                q_count.checked_mul(rows).ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 Q",
            )?,
            self.f32_word(
                gate,
                q_count.checked_mul(rows).ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 attention gate",
            )?,
            self.f32_word(cache_k, cache_count, "Qwen3.5 K cache")?,
            self.f32_word(cache_v, cache_count, "Qwen3.5 V cache")?,
            self.f32_word(delta_k, delta_count, "Qwen3.5 K delta")?,
            self.f32_word(delta_v, delta_count, "Qwen3.5 V delta")?,
            self.f32_word(
                rope,
                rope_dim.checked_mul(rows).ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 mRoPE coefficients",
            )?,
            as_u32(layer, "Qwen3.5 dense layer")?,
            as_u32(position, "Qwen3.5 dense position")?,
            as_u32(capacity, "Qwen3.5 dense capacity")?,
            pack_u16_pair(q_heads, kv_heads, "Qwen3.5 dense heads")?,
            pack_u16_pair(head_dim, rope_dim, "Qwen3.5 dense dimensions")?,
            eps.to_bits(),
        ];
        let group_count = q_heads.max(kv_heads);
        if group_count > self.context.limits.max_compute_work_group_count[0] as usize
            || self.context.limits.max_compute_work_group_count[1] < 2
            || rows > self.context.limits.max_compute_work_group_count[2] as usize
        {
            return Err(VulkanError::UnsupportedShape(
                "Qwen3.5 dense head count exceeds device dispatch limits".into(),
            ));
        }
        unsafe {
            commands.bind(
                self.pipelines[QWEN35_DENSE_PREPARE],
                self.context.pipeline_layout,
                &[bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(group_count as u32, 2, rows as u32);
            commands.barrier();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_qwen35_attention(
        &self,
        commands: &TokenCommands<'_>,
        q: ArenaRegion,
        gate: ArenaRegion,
        cache_k: ArenaRegion,
        cache_v: ArenaRegion,
        output: ArenaRegion,
        layer: usize,
        layer_count: usize,
        base_position: usize,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let sequence_length = base_position
            .checked_add(rows)
            .filter(|_| rows > 0)
            .ok_or(VulkanError::OutOfMemory)?;
        validate_attention_shape(
            layer,
            layer_count,
            sequence_length,
            capacity,
            q_heads,
            kv_heads,
            head_dim,
        )?;
        if capacity > 4096 {
            return Err(VulkanError::UnsupportedShape(format!(
                "Qwen3.5 Vulkan capacity {capacity} exceeds attention shader limit 4096"
            )));
        }
        let q_count = q_heads
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let kv_count = kv_heads
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let cache_count = layer_count
            .checked_mul(capacity)
            .and_then(|count| count.checked_mul(kv_count))
            .ok_or(VulkanError::OutOfMemory)?;
        let push = [
            self.f32_word(
                q,
                q_count.checked_mul(rows).ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 attention Q",
            )?,
            self.f32_word(
                gate,
                q_count.checked_mul(rows).ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 attention gate",
            )?,
            self.f32_word(cache_k, cache_count, "Qwen3.5 attention K cache")?,
            self.f32_word(cache_v, cache_count, "Qwen3.5 attention V cache")?,
            self.f32_word(
                output,
                q_count.checked_mul(rows).ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 attention output",
            )?,
            as_u32(layer, "Qwen3.5 attention layer")?,
            as_u32(base_position, "Qwen3.5 attention base position")?,
            as_u32(capacity, "Qwen3.5 attention capacity")?,
            as_u32(q_heads, "Qwen3.5 attention Q heads")?,
            as_u32(kv_heads, "Qwen3.5 attention KV heads")?,
            as_u32(head_dim, "Qwen3.5 attention head dimension")?,
        ];
        if q_heads > self.context.limits.max_compute_work_group_count[0] as usize
            || rows > self.context.limits.max_compute_work_group_count[1] as usize
        {
            return Err(VulkanError::UnsupportedShape(
                "Qwen3.5 attention head count exceeds device dispatch limits".into(),
            ));
        }
        unsafe {
            commands.bind(
                self.pipelines[QWEN35_ATTENTION],
                self.context.pipeline_layout,
                &[self.arena_bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(q_heads as u32, rows as u32, 1);
            commands.barrier();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_qwen35_recurrent_conv(
        &self,
        commands: &TokenCommands<'_>,
        bindings: OperatorBindings,
        qkv: ArenaRegion,
        q: ArenaRegion,
        k: ArenaRegion,
        v: ArenaRegion,
        state: ArenaRegion,
        layer: usize,
        layer_count: usize,
        conv_dim: usize,
        key_dim: usize,
        value_dim: usize,
        d_conv: usize,
        k_heads: usize,
        v_heads: usize,
        head_dim: usize,
        eps: f32,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let expected_key = k_heads
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let expected_value = v_heads
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let expected_conv = key_dim
            .checked_mul(2)
            .and_then(|count| count.checked_add(value_dim))
            .ok_or(VulkanError::OutOfMemory)?;
        if layer >= layer_count
            || rows == 0
            || d_conv == 0
            || k_heads == 0
            || v_heads == 0
            || head_dim == 0
            || key_dim != expected_key
            || value_dim != expected_value
            || conv_dim != expected_conv
        {
            return Err(VulkanError::UnsupportedShape(format!(
                "invalid Qwen3.5 recurrent convolution layer={layer}/{layer_count} conv={conv_dim}/{expected_conv} key={key_dim}/{expected_key} value={value_dim}/{expected_value} taps={d_conv}"
            )));
        }
        let state_count = layer_count
            .checked_mul(d_conv)
            .and_then(|count| count.checked_mul(conv_dim))
            .ok_or(VulkanError::OutOfMemory)?;
        let weight_count = conv_dim
            .checked_mul(d_conv)
            .ok_or(VulkanError::OutOfMemory)?;
        bindings.require(0, f32_bytes(weight_count)?, "Qwen3.5 convolution weight")?;
        let mut push = [
            self.f32_word(
                qkv,
                conv_dim.checked_mul(rows).ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 recurrent QKV",
            )?,
            self.f32_word(
                q,
                key_dim.checked_mul(rows).ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 recurrent Q",
            )?,
            self.f32_word(
                k,
                key_dim.checked_mul(rows).ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 recurrent K",
            )?,
            self.f32_word(
                v,
                value_dim
                    .checked_mul(rows)
                    .ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 recurrent V",
            )?,
            self.f32_word(state, state_count, "Qwen3.5 convolution state")?,
            as_u32(layer, "Qwen3.5 recurrent layer")?,
            as_u32(conv_dim, "Qwen3.5 convolution width")?,
            as_u32(key_dim, "Qwen3.5 recurrent key width")?,
            as_u32(value_dim, "Qwen3.5 recurrent value width")?,
            as_u32(d_conv, "Qwen3.5 convolution taps")?,
            as_u32(k_heads, "Qwen3.5 recurrent key heads")?,
            as_u32(v_heads, "Qwen3.5 recurrent value heads")?,
            as_u32(head_dim, "Qwen3.5 recurrent head dimension")?,
            eps.to_bits(),
            0,
            as_u32(rows, "Qwen3.5 convolution rows")?,
        ];
        let (x, y) = dispatch_invocations(conv_dim, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[QWEN35_RECURRENT_CONV],
                self.context.pipeline_layout,
                &[bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, 1);
            commands.barrier();
        }
        push[14] = 1;
        let (x, y) = super::dispatch_grid(k_heads.max(v_heads), &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[QWEN35_RECURRENT_CONV],
                self.context.pipeline_layout,
                &[bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, 1);
            commands.barrier();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_qwen35_recurrent_ssm(
        &self,
        commands: &TokenCommands<'_>,
        bindings: OperatorBindings,
        q: ArenaRegion,
        k: ArenaRegion,
        v: ArenaRegion,
        gate: ArenaRegion,
        beta: ArenaRegion,
        alpha: ArenaRegion,
        output: ArenaRegion,
        state: ArenaRegion,
        layer: usize,
        layer_count: usize,
        k_heads: usize,
        v_heads: usize,
        head_dim: usize,
        eps: f32,
        rows: usize,
    ) -> Result<(), VulkanError> {
        if rows == 0
            || layer >= layer_count
            || k_heads == 0
            || v_heads == 0
            || head_dim == 0
            || head_dim > 128
        {
            return Err(VulkanError::UnsupportedShape(format!(
                "invalid Qwen3.5 recurrent SSM layer={layer}/{layer_count} heads={k_heads}/{v_heads} dim={head_dim}"
            )));
        }
        let key_count = k_heads
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let value_count = v_heads
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let state_count = layer_count
            .checked_mul(v_heads)
            .and_then(|count| count.checked_mul(head_dim))
            .and_then(|count| count.checked_mul(head_dim))
            .ok_or(VulkanError::OutOfMemory)?;
        bindings.require(0, f32_bytes(v_heads)?, "Qwen3.5 SSM dt bias")?;
        bindings.require(1, f32_bytes(v_heads)?, "Qwen3.5 SSM A")?;
        bindings.require(2, f32_bytes(head_dim)?, "Qwen3.5 SSM norm")?;
        let push = [
            self.f32_word(
                q,
                key_count
                    .checked_mul(rows)
                    .ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 SSM Q",
            )?,
            self.f32_word(
                k,
                key_count
                    .checked_mul(rows)
                    .ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 SSM K",
            )?,
            self.f32_word(
                v,
                value_count
                    .checked_mul(rows)
                    .ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 SSM V",
            )?,
            self.f32_word(
                gate,
                value_count
                    .checked_mul(rows)
                    .ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 SSM gate",
            )?,
            self.f32_word(
                beta,
                v_heads.checked_mul(rows).ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 SSM beta",
            )?,
            self.f32_word(
                alpha,
                v_heads.checked_mul(rows).ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 SSM alpha",
            )?,
            self.f32_word(
                output,
                value_count
                    .checked_mul(rows)
                    .ok_or(VulkanError::OutOfMemory)?,
                "Qwen3.5 SSM output",
            )?,
            self.f32_word(state, state_count, "Qwen3.5 SSM state")?,
            as_u32(layer, "Qwen3.5 SSM layer")?,
            as_u32(k_heads, "Qwen3.5 SSM key heads")?,
            as_u32(v_heads, "Qwen3.5 SSM value heads")?,
            as_u32(head_dim, "Qwen3.5 SSM head dimension")?,
            eps.to_bits(),
            as_u32(rows, "Qwen3.5 SSM rows")?,
        ];
        let (x, y) = super::dispatch_grid(v_heads, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[QWEN35_RECURRENT_SSM],
                self.context.pipeline_layout,
                &[bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, 1);
            commands.barrier();
        }
        Ok(())
    }

    pub(crate) fn record_silu_mul(
        &self,
        commands: &TokenCommands<'_>,
        gate: ArenaRegion,
        up: ArenaRegion,
        count: usize,
    ) -> Result<(), VulkanError> {
        self.record_silu_mul_rows(commands, gate, up, count, 1)
    }

    pub(crate) fn record_silu_mul_rows(
        &self,
        commands: &TokenCommands<'_>,
        gate: ArenaRegion,
        up: ArenaRegion,
        count: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let push = [
            self.f32_rows_word(gate, rows, count, count, "SiLU gate")?,
            self.f32_rows_word(up, rows, count, count, "SiLU multiplier")?,
            as_u32(count, "SiLU length")?,
            as_u32(rows, "silu_mul rows")?,
            as_u32(count, "silu_mul row stride")?,
        ];
        let [x, y, z] = row_dispatch(count.div_ceil(64), rows, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[SILU_MUL],
                self.context.pipeline_layout,
                &[self.arena_bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, z);
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
        self.record_add_rows(commands, target, addition, count, 1)
    }

    pub(crate) fn record_add_rows(
        &self,
        commands: &TokenCommands<'_>,
        target: ArenaRegion,
        addition: ArenaRegion,
        count: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let push = [
            self.f32_rows_word(target, rows, count, count, "add target")?,
            self.f32_rows_word(addition, rows, count, count, "add source")?,
            as_u32(count, "add length")?,
            as_u32(rows, "add rows")?,
            as_u32(count, "add row stride")?,
        ];
        let [x, y, z] = row_dispatch(count.div_ceil(64), rows, &self.context.limits)?;
        unsafe {
            commands.bind(
                self.pipelines[ADD],
                self.context.pipeline_layout,
                &[self.arena_bindings.descriptor_set],
                bytemuck::cast_slice(&push),
            );
            commands.dispatch(x, y, z);
            commands.barrier();
        }
        Ok(())
    }

    pub(crate) fn zero_region(&self, region: ArenaRegion) -> Result<(), VulkanError> {
        self.byte_word(region, region.size, "zeroed Vulkan arena region")?;
        unsafe { std::ptr::write_bytes(self.arena.mapped.add(region.offset), 0, region.size) };
        Ok(())
    }

    fn f32_rows_word(
        &self,
        region: ArenaRegion,
        rows: usize,
        stride: usize,
        width: usize,
        label: &str,
    ) -> Result<u32, VulkanError> {
        row_word(
            self.arena.size as usize,
            region,
            rows,
            f32_bytes(stride)?,
            f32_bytes(width)?,
            label,
        )
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
        let Ok(mut submission) = self.context.mutex.lock() else {
            return;
        };
        if self.context.recover_commands(&mut submission).is_err() {
            return;
        }
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

fn pack_u16_pair(low: usize, high: usize, label: &str) -> Result<u32, VulkanError> {
    let low = u16::try_from(low).map_err(|_| {
        VulkanError::UnsupportedShape(format!("{label} low value {low} exceeds u16"))
    })?;
    let high = u16::try_from(high).map_err(|_| {
        VulkanError::UnsupportedShape(format!("{label} high value {high} exceeds u16"))
    })?;
    Ok(u32::from(low) | (u32::from(high) << 16))
}

fn f32_bytes(count: usize) -> Result<usize, VulkanError> {
    count.checked_mul(4).ok_or(VulkanError::OutOfMemory)
}

fn row_span(rows: usize, stride: usize, width: usize) -> Result<usize, VulkanError> {
    if rows == 0 || width == 0 || stride < width {
        return Err(VulkanError::UnsupportedShape(
            "row count/width must be positive and stride must cover a row".into(),
        ));
    }
    (rows - 1)
        .checked_mul(stride)
        .and_then(|size| size.checked_add(width))
        .ok_or(VulkanError::OutOfMemory)
}

fn row_word(
    arena_size: usize,
    region: ArenaRegion,
    rows: usize,
    stride_bytes: usize,
    width_bytes: usize,
    label: &str,
) -> Result<u32, VulkanError> {
    let bytes = row_span(rows, stride_bytes, width_bytes)?;
    let end = region
        .offset
        .checked_add(bytes)
        .ok_or(VulkanError::OutOfMemory)?;
    if region.offset % 4 != 0 || stride_bytes % 4 != 0 || bytes > region.size || end > arena_size {
        return Err(VulkanError::UnsupportedShape(format!(
            "{label} region cannot hold {rows} rows ({bytes} bytes)"
        )));
    }
    // Shader addressing uses uint words, including the final word of the last row.
    as_u32((end - 1) / 4, label)?;
    as_u32(region.offset / 4, label)
}

fn row_dispatch(
    groups: usize,
    rows: usize,
    limits: &vk::PhysicalDeviceLimits,
) -> Result<[u32; 3], VulkanError> {
    let z = as_u32(rows, "Vulkan dispatch z")?;
    if z == 0 || z > limits.max_compute_work_group_count[2] {
        return Err(VulkanError::UnsupportedShape(format!(
            "Vulkan dispatch z={z} exceeds device limit {}",
            limits.max_compute_work_group_count[2]
        )));
    }
    let (x, y) = super::dispatch_grid(groups, limits)?;
    Ok([x, y, z])
}

fn matmul_dispatch(
    output_rows: usize,
    token_rows: usize,
    grouped_weights: usize,
    limits: &vk::PhysicalDeviceLimits,
) -> Result<[u32; 3], VulkanError> {
    if !(1..=3).contains(&grouped_weights) {
        return Err(VulkanError::UnsupportedShape(
            "grouped Vulkan matmul needs 1 to 3 weights".into(),
        ));
    }
    let z = token_rows
        .checked_mul(grouped_weights)
        .ok_or(VulkanError::OutOfMemory)?;
    row_dispatch(output_rows, z, limits)
}

#[allow(clippy::too_many_arguments)]
fn quantize_rows_push(
    arena_size: usize,
    input: ArenaRegion,
    q8: ArenaRegion,
    scales: ArenaRegion,
    sums: Option<ArenaRegion>,
    count: usize,
    rows: usize,
    input_stride: usize,
    block_elements: usize,
) -> Result<[u32; 10], VulkanError> {
    if !matches!(block_elements, 32 | 256) || count == 0 || count % block_elements != 0 {
        return Err(VulkanError::UnsupportedShape(
            "invalid Vulkan quantization width".into(),
        ));
    }
    let blocks = count / block_elements;
    let input_word = row_word(
        arena_size,
        input,
        rows,
        f32_bytes(input_stride)?,
        f32_bytes(count)?,
        "quantize input",
    )?;
    let q8_word = row_word(arena_size, q8, rows, count, count, "quantize Q8 output")?;
    let scales_word = row_word(
        arena_size,
        scales,
        rows,
        f32_bytes(blocks)?,
        f32_bytes(blocks)?,
        "quantize scales",
    )?;
    let count = as_u32(count, "quantize width")?;
    let blocks = as_u32(blocks, "quantize blocks")?;
    let stride = as_u32(input_stride, "quantize input stride")?;
    let rows_u32 = as_u32(rows, "quantize token rows")?;
    if block_elements == 32 {
        let sums = sums.ok_or_else(|| {
            VulkanError::UnsupportedShape("Q8_0 quantization requires sums".into())
        })?;
        let sums_word = row_word(
            arena_size,
            sums,
            rows,
            f32_bytes(blocks as usize)?,
            f32_bytes(blocks as usize)?,
            "Q4_1 input sums",
        )?;
        Ok([
            input_word,
            q8_word,
            scales_word,
            sums_word,
            count,
            stride,
            count / 4,
            blocks,
            blocks,
            rows_u32,
        ])
    } else {
        Ok([
            input_word,
            q8_word,
            scales_word,
            count,
            stride,
            count / 4,
            blocks,
            rows_u32,
            0,
            0,
        ])
    }
}

#[allow(clippy::too_many_arguments)]
fn matmul_rows_push(
    arena_size: usize,
    limits: &vk::PhysicalDeviceLimits,
    bindings: OperatorBindings,
    activation: ArenaRegion,
    scales: ArenaRegion,
    sums: Option<ArenaRegion>,
    outputs: &[(ArenaRegion, usize, usize)],
    n_in: usize,
    token_rows: usize,
    input_stride: usize,
) -> Result<([u32; 22], [u32; 3]), VulkanError> {
    let format = bindings.weight_format(outputs.len())?;
    let (block_elements, block_bytes, _) = format.layout();
    if n_in == 0
        || n_in % block_elements != 0
        || input_stride < n_in
        || (format == GpuWeightFormat::Q8_0 && n_in > 16_384)
    {
        return Err(VulkanError::UnsupportedShape(
            "invalid Vulkan matmul input width/stride".into(),
        ));
    }
    let is_float = matches!(
        format,
        GpuWeightFormat::F16 | GpuWeightFormat::BF16 | GpuWeightFormat::F32
    );
    let blocks = n_in / block_elements;
    let mut push = [0; 22];
    push[0] = if is_float {
        row_word(
            arena_size,
            activation,
            token_rows,
            f32_bytes(input_stride)?,
            f32_bytes(n_in)?,
            "matmul input",
        )?
    } else {
        row_word(
            arena_size,
            activation,
            token_rows,
            n_in,
            n_in,
            "matmul Q8 input",
        )?
    };
    if !is_float {
        push[1] = row_word(
            arena_size,
            scales,
            token_rows,
            f32_bytes(blocks)?,
            f32_bytes(blocks)?,
            "matmul scales",
        )?;
        push[16] = as_u32(n_in / 4, "matmul Q8 stride")?;
        push[17] = as_u32(blocks, "matmul scale stride")?;
        push[18] = push[17];
        push[19] = push[16];
        push[20] = push[17];
    }
    if format == GpuWeightFormat::Q4_1 {
        let sums = sums
            .ok_or_else(|| VulkanError::UnsupportedShape("Q4_1 input sums are required".into()))?;
        push[14] = row_word(
            arena_size,
            sums,
            token_rows,
            f32_bytes(blocks)?,
            f32_bytes(blocks)?,
            "Q4_1 input sums",
        )?;
    }
    push[2] = as_u32(n_in, "matmul width")?;
    push[3] = as_u32(blocks, "matmul blocks")?;
    push[13] = outputs.len() as u32;
    push[15] = as_u32(input_stride, "matmul input stride")?;
    push[21] = as_u32(token_rows, "matmul token rows")?;
    let weight_row_bytes = blocks
        .checked_mul(block_bytes)
        .ok_or(VulkanError::OutOfMemory)?;
    let mut max_output_rows = 0;
    for (slot, &(region, n_out, stride_bytes)) in outputs.iter().enumerate() {
        let bytes = n_out
            .checked_mul(weight_row_bytes)
            .ok_or(VulkanError::OutOfMemory)?;
        bindings.require(slot, bytes, "Vulkan weight")?;
        // Quantized shaders form byte offsets in uint before loading words.
        as_u32(bytes, "Vulkan weight byte span")?;
        push[4 + slot * 3] = row_word(
            arena_size,
            region,
            token_rows,
            stride_bytes,
            f32_bytes(n_out)?,
            "matmul output",
        )?;
        push[5 + slot * 3] = as_u32(n_out, "matmul output rows")?;
        push[6 + slot * 3] = as_u32(stride_bytes / 4, "matmul output stride")?;
        max_output_rows = max_output_rows.max(n_out);
    }
    let dispatch = matmul_dispatch(max_output_rows, token_rows, outputs.len(), limits)?;
    Ok((push, dispatch))
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

#[cfg(test)]
pub(crate) fn matmul_dispatch_for_test(
    output_rows: usize,
    token_rows: usize,
    grouped_weights: usize,
) -> Result<[u32; 3], VulkanError> {
    let limits = vk::PhysicalDeviceLimits {
        max_compute_work_group_count: [u32::MAX; 3],
        ..Default::default()
    };
    matmul_dispatch(output_rows, token_rows, grouped_weights, &limits)
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

/// Device-only regression check for row recording, including grouped outputs and padding.
pub fn run_batched_matmul_check(
    context: &VulkanContext,
    formats: &[&str],
    rows: usize,
) -> Result<(), String> {
    for &name in formats {
        let format = match name {
            "q8_0" => GpuWeightFormat::Q8_0,
            "q4_0" => GpuWeightFormat::Q4_0,
            "q4_1" => GpuWeightFormat::Q4_1,
            "q4_k" => GpuWeightFormat::Q4_K,
            "q5_k" => GpuWeightFormat::Q5_K,
            "q6_k" => GpuWeightFormat::Q6_K,
            "f16" => GpuWeightFormat::F16,
            "bf16" => GpuWeightFormat::BF16,
            "f32" => GpuWeightFormat::F32,
            _ => return Err(format!("unsupported Vulkan weight format {name}")),
        };
        check_weight_matmul_rows(context, format, rows)
            .map_err(|error| format!("{name}: {error}"))?;
    }
    Ok(())
}

fn check_weight_matmul_rows(
    context: &VulkanContext,
    format: GpuWeightFormat,
    rows: usize,
) -> Result<(), String> {
    const N_IN: usize = 512;
    const INPUT_STRIDE: usize = 515;
    let mut buffers = Vec::new();
    let result = (|| -> Result<(), VulkanError> {
        matmul_dispatch(65, rows, 3, &context.limits)?;
        let mut cursor = 0;
        let input = f32_region(&mut cursor, row_span(rows, INPUT_STRIDE, N_IN)?)?;
        let q8 = region(&mut cursor, row_span(rows, N_IN, N_IN)?)?;
        let scales = f32_region(&mut cursor, row_span(rows, N_IN / 32, N_IN / 32)?)?;
        let sums = f32_region(&mut cursor, row_span(rows, N_IN / 32, N_IN / 32)?)?;
        let q8k = region(&mut cursor, row_span(rows, N_IN, N_IN)?)?;
        let k_scales = f32_region(&mut cursor, row_span(rows, N_IN / 256, N_IN / 256)?)?;
        let mut outputs = Vec::new();
        for (slot, n_out) in [65, 33, 17].into_iter().enumerate() {
            let stride = (n_out + (slot + 1) * 3) * 4;
            outputs.push((
                region(&mut cursor, row_span(rows, stride, n_out * 4)?)?,
                n_out,
                stride,
            ));
            let weight = if format == GpuWeightFormat::Q8_0 {
                synthetic_q8_weight(N_IN, n_out, 3 + slot * 4)
            } else {
                let (block, bytes, _) = format.layout();
                let weight = synthetic_weight(format, N_IN, n_out + slot);
                weight[slot * (N_IN / block) * bytes..].to_vec()
            };
            buffers.push(unsafe { context.upload_static(&weight)? });
        }
        let mut ops = Qwen3Ops::new_with_size(context, cursor, 5)?;
        let grouped = ops.bind_weight_buffers(&buffers, &[format; 3])?;
        let mut single = Vec::new();
        for buffer in &buffers {
            single.push(ops.bind_weight_buffers(&[*buffer], &[format])?);
        }
        let mut input_values = vec![-12345.0; input.size / 4];
        let divisor = if matches!(format, GpuWeightFormat::F16 | GpuWeightFormat::BF16) {
            131_072.0
        } else {
            97.0
        };
        for row in 0..rows {
            for index in 0..N_IN {
                input_values[row * INPUT_STRIDE + index] =
                    ((row * 53 + index * 29) % 251) as f32 / divisor - 125.0 / divisor;
            }
        }
        ops.write_f32(input, &input_values)?;
        const PADDING: f32 = 123456.0;
        for &(output, _, _) in &outputs {
            ops.write_f32(output, &vec![PADDING; output.size / 4])?;
        }
        let commands = TokenCommands::begin(context)?;
        ops.record_weight_matmul_rows(
            &commands,
            grouped,
            input,
            q8,
            scales,
            sums,
            q8k,
            k_scales,
            &outputs,
            N_IN,
            rows,
            INPUT_STRIDE,
        )?;
        commands.submit_and_wait()?;
        let mut actual = Vec::new();
        for &(output, _, _) in &outputs {
            actual.push(
                ops.read_f32(output, output.size / 4)?
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
            );
            ops.write_f32(output, &vec![PADDING; output.size / 4])?;
        }
        // Independent rows=1 calls use packed strides and one weight binding.
        let commands = TokenCommands::begin(context)?;
        for row in 0..rows {
            let row_input = ArenaRegion {
                offset: input.offset + row * INPUT_STRIDE * 4,
                size: N_IN * 4,
            };
            for (slot, &(output, n_out, stride)) in outputs.iter().enumerate() {
                let row_output = ArenaRegion {
                    offset: output.offset + row * stride,
                    size: n_out * 4,
                };
                ops.record_weight_matmul_rows(
                    &commands,
                    single[slot],
                    row_input,
                    q8,
                    scales,
                    sums,
                    q8k,
                    k_scales,
                    &[(row_output, n_out, n_out * 4)],
                    N_IN,
                    1,
                    N_IN,
                )?;
            }
        }
        commands.submit_and_wait()?;
        for (slot, &(output, n_out, stride)) in outputs.iter().enumerate() {
            let expected = ops.read_f32(output, output.size / 4)?;
            for (index, (&got, want)) in actual[slot].iter().zip(expected).enumerate() {
                if got != want.to_bits() {
                    return Err(VulkanError::UnsupportedShape(format!(
                        "rows={rows} slot={slot} index={index} batched=0x{got:08x} single=0x{:08x}",
                        want.to_bits()
                    )));
                }
                if index % (stride / 4) >= n_out && got != PADDING.to_bits() {
                    return Err(VulkanError::UnsupportedShape(
                        "matmul overwrote output padding".into(),
                    ));
                }
            }
            if rows > 1
                && expected[..n_out]
                    .iter()
                    .zip(&expected[stride / 4..stride / 4 + n_out])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                return Err(VulkanError::UnsupportedShape(
                    "row fixture does not distinguish token rows".into(),
                ));
            }
        }
        println!("operator=batched_matmul format={format:?} rows={rows} groups=3 input_stride={INPUT_STRIDE} padded_outputs=true exact_bits=true");
        Ok(())
    })();
    let cleanup = unsafe { context.destroy_completed_buffers(&buffers) };
    result.and(cleanup).map_err(|error| error.to_string())
}

pub fn run_qwen3_operator_check(context: &VulkanContext, formats: &[&str]) -> Result<(), String> {
    check_quantize_tie_even(context)?;
    check_attention_scores_match_cpu_reduction(context)?;
    check_softmax_f16_rounding(context)?;
    check_attention_value_reduction(context)?;
    for &format in formats {
        check_weight_format(context, format)?;
    }
    if formats.contains(&"q4_k") || formats.contains(&"q5_k") || formats.contains(&"q6_k") {
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
            crate::ops::rope_neox_inplace(head, POSITION, HEAD_DIM, FREQ_BASE);
        }
        for head in expected_k.chunks_exact_mut(HEAD_DIM) {
            crate::ops::rms_norm_inplace(head, &k_norm, EPS);
            crate::ops::rope_neox_inplace(head, POSITION, HEAD_DIM, FREQ_BASE);
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

    let cleanup = unsafe { context.destroy_completed_buffers(&allocations) };
    result.and(cleanup.map_err(|error| error.to_string()))
}

fn check_weight_format(context: &VulkanContext, name: &str) -> Result<(), String> {
    let format = match name {
        "q4_0" => GpuWeightFormat::Q4_0,
        "q4_1" => GpuWeightFormat::Q4_1,
        "q4_k" => GpuWeightFormat::Q4_K,
        "q5_k" => GpuWeightFormat::from_ggml_type(crate::core::tensor::GGMLType::Q5K)
            .map_err(|error| error.to_string())?,
        "q6_k" => GpuWeightFormat::Q6_K,
        "f16" => GpuWeightFormat::F16,
        "bf16" => GpuWeightFormat::from_ggml_type(crate::core::tensor::GGMLType::BF16)
            .map_err(|error| error.to_string())?,
        "f32" => GpuWeightFormat::from_ggml_type(crate::core::tensor::GGMLType::F32)
            .map_err(|error| error.to_string())?,
        _ => return Err(format!("unsupported Vulkan weight format {name}")),
    };
    let n_in = match format {
        GpuWeightFormat::Q4_0 => 1024,
        GpuWeightFormat::Q4_1 => 3072,
        GpuWeightFormat::Q4_K => 1024,
        GpuWeightFormat::Q5_K => 1024,
        GpuWeightFormat::Q6_K => 1024,
        GpuWeightFormat::F16 => 1024,
        GpuWeightFormat::BF16 => 1024,
        GpuWeightFormat::F32 => 1024,
        GpuWeightFormat::Q8_0 => unreachable!(),
    };
    check_weight_matvec(context, name, format, n_in, 65)
}

fn check_weight_matvec(
    context: &VulkanContext,
    name: &str,
    format: GpuWeightFormat,
    n_in: usize,
    n_out: usize,
) -> Result<(), String> {
    let layout = ArenaLayout::for_dims(n_in.max(n_out), n_out, 1, 1, n_in)
        .map_err(|error| error.to_string())?;
    let input: Vec<f32> = (0..n_in)
        .map(|index| {
            let divisor = if matches!(format, GpuWeightFormat::F16 | GpuWeightFormat::BF16) {
                131_072.0
            } else {
                97.0
            };
            ((index * 29 % 251) as f32 - 125.0) / divisor
        })
        .collect();
    if format == GpuWeightFormat::Q4_1 {
        check_q4_1_input_sums(context, &input)?;
    }
    let weight = synthetic_weight(format, n_in, n_out);
    if format == GpuWeightFormat::Q4_1 {
        check_q4_1_zero_scale_min_fixture(&weight, n_in, n_out)?;
    }
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
        let tolerance = match format {
            GpuWeightFormat::F32 => 0.0,
            GpuWeightFormat::Q4_K | GpuWeightFormat::Q5_K => 3e-3,
            GpuWeightFormat::F16 | GpuWeightFormat::BF16 => 2e-4,
            _ => 2e-3,
        };
        check_close(
            name,
            ops.read_f32(layout.projection, n_out)
                .map_err(|error| error.to_string())?,
            &expected,
            tolerance,
            tolerance,
        )
    })();
    let cleanup = unsafe { context.destroy_completed_buffers(&[buffer]) };
    result.and(cleanup.map_err(|error| error.to_string()))
}

fn check_q4_1_zero_scale_min_fixture(
    weight: &[u8],
    n_in: usize,
    n_out: usize,
) -> Result<(), String> {
    let blocks = n_in / 32;
    for row in 0..n_out {
        for block in 0..blocks {
            let offset = (row * blocks + block) * 20;
            let d =
                crate::ops::f16_to_f32(u16::from_le_bytes([weight[offset], weight[offset + 1]]));
            let m = crate::ops::f16_to_f32(u16::from_le_bytes([
                weight[offset + 2],
                weight[offset + 3],
            ]));
            if d == 0.0 && m != 0.0 {
                return Ok(());
            }
        }
    }
    Err("Q4_1 fixture is missing a d=0,m!=0 block".into())
}

fn check_q4_1_input_sums(context: &VulkanContext, input: &[f32]) -> Result<(), String> {
    let count = input.len();
    let layout =
        ArenaLayout::for_dims(count, count, 1, 1, count).map_err(|error| error.to_string())?;
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
            let sum = quantized
                .iter()
                .map(|&value| i32::from(value as i8))
                .sum::<i32>();
            crate::ops::f16_to_f32(crate::ops::f32_to_f16(sum as f32 * raw_scale))
        })
        .collect();
    let stored_scale_terms: Vec<f32> = expected_q8
        .chunks_exact(32)
        .zip(&expected_scales)
        .map(|(quantized, &scale)| {
            let sum = quantized
                .iter()
                .map(|&value| i32::from(value as i8))
                .sum::<i32>();
            crate::ops::f16_to_f32(crate::ops::f32_to_f16(sum as f32 * scale))
        })
        .collect();
    if expected == stored_scale_terms {
        return Err("Q4_1 input-sum fixture does not distinguish raw and stored scales".into());
    }
    ops.write_f32(layout.x, input)
        .map_err(|error| error.to_string())?;
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
    commands
        .submit_and_wait()
        .map_err(|error| error.to_string())?;
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
                    data[offset..offset + 2].copy_from_slice(
                        &crate::ops::f32_to_f16(if row == 0 && block == 0 {
                            0.0
                        } else {
                            (row as f32 + 1.0) / 91.0
                        })
                        .to_le_bytes(),
                    );
                    data[offset + 2..offset + 4].copy_from_slice(
                        &crate::ops::f32_to_f16(if row == 0 && block == 0 {
                            -0.25
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
                GpuWeightFormat::Q4_K => {
                    data[offset..offset + 2].copy_from_slice(
                        &crate::ops::f32_to_f16(if row == 0 && block == 0 {
                            0.0
                        } else {
                            (row as f32 + block as f32 + 1.0) / 64.0
                        })
                        .to_le_bytes(),
                    );
                    data[offset + 2..offset + 4].copy_from_slice(
                        &crate::ops::f32_to_f16(if row == 0 && block == 0 {
                            0.0
                        } else {
                            (row as f32 + block as f32 + 2.0) / 96.0
                        })
                        .to_le_bytes(),
                    );
                    for (index, value) in data[offset + 4..offset + 16].iter_mut().enumerate() {
                        *value = ((row * 31 + block * 17 + index * 37 + 11) & 255) as u8;
                    }
                    for (index, value) in data[offset + 16..offset + 144].iter_mut().enumerate() {
                        let low = ((row * 7 + block * 13 + index * 3 + 1) & 15) as u8;
                        let high = ((row * 11 + block * 5 + index * 9 + 6) & 15) as u8;
                        *value = low | (high << 4);
                    }
                }
                GpuWeightFormat::Q5_K => {
                    data[offset..offset + 2].copy_from_slice(
                        &crate::ops::f32_to_f16(if row == 0 && block == 0 {
                            0.0
                        } else {
                            (row as f32 + block as f32 + 1.0) / 64.0
                        })
                        .to_le_bytes(),
                    );
                    data[offset + 2..offset + 4].copy_from_slice(
                        &crate::ops::f32_to_f16(if row == 0 && block == 0 {
                            0.0
                        } else {
                            (row as f32 + block as f32 + 2.0) / 96.0
                        })
                        .to_le_bytes(),
                    );
                    for (index, value) in data[offset + 4..offset + 16].iter_mut().enumerate() {
                        *value = ((row * 31 + block * 17 + index * 37 + 11) & 255) as u8;
                    }
                    for (index, value) in data[offset + 16..offset + 48].iter_mut().enumerate() {
                        *value = ((row * 13 + block * 19 + index * 37 + 5) & 255) as u8;
                    }
                    for (index, value) in data[offset + 48..offset + 176].iter_mut().enumerate() {
                        let low = ((row * 7 + block * 13 + index * 3 + 1) & 15) as u8;
                        let high = ((row * 11 + block * 5 + index * 9 + 6) & 15) as u8;
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
                GpuWeightFormat::F16 => {
                    let bits = match block {
                        0 => 0x0000,
                        1 => 0x8000,
                        2 => 0x0001,
                        3 => 0x03ff,
                        4 => 0x0400,
                        5 => 0x3c00,
                        6 => 0x7bff,
                        7 => 0xfbff,
                        _ => crate::ops::f32_to_f16(
                            ((row * 17 + block * 31) % 257) as f32 / 64.0 - 2.0,
                        ),
                    };
                    data[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
                }
                GpuWeightFormat::BF16 => {
                    let bits: u16 = match block {
                        0 => 0x0000,
                        1 => 0x8000,
                        2 => 0x0001,
                        3 => 0x007f,
                        4 => 0x0080,
                        5 => 0x3f80,
                        6 => 0x7f7f,
                        7 => 0xff7f,
                        _ => crate::ops::f32_to_bf16(
                            ((row * 17 + block * 31) % 257) as f32 / 64.0 - 2.0,
                        ),
                    };
                    data[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
                }
                GpuWeightFormat::F32 => {
                    let value = ((row * 17 + block * 31) % 257) as f32 / 63.0 - 2.0;
                    data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
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
        GpuWeightFormat::Q4_K => {
            let q8k = crate::ops::quant::quantize_row_q8_k(input);
            for (row, output) in output.iter_mut().enumerate() {
                let row_bytes = n_in / crate::ops::quant::QK_K * crate::ops::quant::BLOCK_Q4K_SIZE;
                *output = crate::ops::quant::vec_dot_q4k_q8k(
                    &weight[row * row_bytes..(row + 1) * row_bytes],
                    &q8k,
                );
            }
        }
        GpuWeightFormat::Q5_K => {
            let q8k = crate::ops::quant::quantize_row_q8_k(input);
            for (row, output) in output.iter_mut().enumerate() {
                let row_bytes = n_in / crate::ops::quant::QK_K * crate::ops::quant::BLOCK_Q5K_SIZE;
                *output = crate::ops::quant::vec_dot_q5k_q8k(
                    &weight[row * row_bytes..(row + 1) * row_bytes],
                    &q8k,
                );
            }
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
        GpuWeightFormat::F16 => {
            let kernel = crate::ops::kernel::f16::F16Kernel::new(weight);
            crate::ops::kernel::Kernel::forward(&kernel, input, &mut output, n_in, n_out);
        }
        GpuWeightFormat::BF16 => {
            let kernel = crate::ops::kernel::bf16::BF16Kernel::new(weight);
            crate::ops::kernel::Kernel::forward(&kernel, input, &mut output, n_in, n_out);
        }
        GpuWeightFormat::F32 => {
            let values = weight
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            let kernel = crate::ops::kernel::f32::F32Kernel::new(values, 0, 0);
            crate::ops::kernel::Kernel::forward(&kernel, input, &mut output, n_in, n_out);
        }
        GpuWeightFormat::Q8_0 => unreachable!(),
    }
    output
}

fn check_quantize_q8_k_exact(context: &VulkanContext) -> Result<(), String> {
    const COUNT: usize = 16_384;
    let layout =
        ArenaLayout::for_dims(COUNT, COUNT, 1, 1, COUNT).map_err(|error| error.to_string())?;
    let ops = Qwen3Ops::new(context, layout, 1).map_err(|error| error.to_string())?;
    let input: Vec<f32> = (0..COUNT)
        .map(|index| {
            let block = index / 256;
            let local = index % 256;
            let magnitude = f32::from_bits(0x3f00_0001 + block as u32 * 0x0002_345);
            let anchor = if block % 2 == 0 {
                magnitude
            } else {
                -magnitude
            };
            if local == 0 {
                anchor
            } else {
                anchor * (((local * 47 % 251) as f32 - 125.0) / 127.0)
            }
        })
        .collect();
    let expected = crate::ops::quant::quantize_row_q8_k(&input);
    ops.write_f32(layout.x, &input)
        .map_err(|error| error.to_string())?;
    let commands = TokenCommands::begin(context).map_err(|error| error.to_string())?;
    ops.record_quantize_q8_k(&commands, layout.x, layout.q8k, layout.q8k_scales, COUNT)
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
    if max_scale_ulps != 0 {
        let index = actual_scales
            .iter()
            .zip(&expected)
            .position(|(actual, expected)| actual.to_bits() != expected.d.to_bits())
            .expect("nonzero max ULP difference has a differing scale");
        return Err(format!(
            "quantize_q8_k scale differs by {max_scale_ulps} ULPs at {index}: gpu={:#010x} cpu={:#010x} (expected exact bits)",
            actual_scales[index].to_bits(),
            expected[index].d.to_bits(),
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
    let mut expected_scales = [0.0f32; 1];
    crate::ops::quantize_q8_0_into(&input, input.len(), &mut expected, &mut expected_scales);
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

fn check_attention_scores_match_cpu_reduction(context: &VulkanContext) -> Result<(), String> {
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
    let query_f16 = query
        .iter()
        .map(|&value| crate::ops::f32_to_f16(value))
        .collect::<Vec<_>>();
    let key_f16 = key
        .iter()
        .map(|&value| crate::ops::f32_to_f16(value))
        .collect::<Vec<_>>();
    let expected = crate::ops::dot_f16(&query_f16, &key_f16, HEAD_DIM) / (HEAD_DIM as f32).sqrt();
    if actual.to_bits() != expected.to_bits() {
        return Err(format!(
            "attention score reduction mismatch: gpu={actual} cpu={expected} gpu_bits={:#010x} cpu_bits={:#010x}",
            actual.to_bits(),
            expected.to_bits()
        ));
    }
    println!("operator=attention_score_reduction exact=true");
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
    let probabilities_f16 = probabilities
        .iter()
        .map(|&value| crate::ops::f32_to_f16(value))
        .collect::<Vec<_>>();
    let values_f16 = values
        .iter()
        .map(|&value| crate::ops::f32_to_f16(value))
        .collect::<Vec<_>>();
    let expected = crate::ops::dot_f16(&probabilities_f16, &values_f16, SEQUENCE);
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
    if actual
        .iter()
        .zip(&expected)
        .any(|(actual, expected)| actual.to_bits().abs_diff(expected.to_bits()) > 0x2000)
    {
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
    if actual
        .iter()
        .zip(&expected)
        .any(|(actual, expected)| actual.to_bits().abs_diff(expected.to_bits()) > 0x2000)
    {
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
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn qwen3_chunk_attention_validates_all_spans_before_recording() {
        use super::{ArenaLayout, ArenaRegion, Qwen3Ops, TokenCommands, VulkanContext};
        let context = VulkanContext::new().unwrap();
        let layout = ArenaLayout::build_rows(32, 32, 2, 1, 16, 8, 2, 5, 3).unwrap();
        let ops = Qwen3Ops::new(&context, layout, 1).unwrap();
        ops.write_f32(layout.scores, &[9.0; 30]).unwrap();
        let before = ops.read_f32(layout.scores, 30).unwrap().to_vec();
        let mut output = layout.attn;
        output.size = 3 * 32 * 4 - 1;
        let commands = TokenCommands::begin(&context).unwrap();
        assert!(ops
            .record_attention_rows(
                &commands,
                layout.q,
                layout.kv_k,
                layout.kv_v,
                layout.scores,
                output,
                1,
                2,
                2,
                5,
                2,
                1,
                16,
                3
            )
            .is_err());
        commands.submit_and_wait().unwrap();
        assert_eq!(
            ops.read_f32(layout.scores, 30).unwrap(),
            before,
            "an invalid output span must not record scores or softmax"
        );
        let commands = TokenCommands::begin(&context).unwrap();
        for rows in [0, 4, usize::MAX] {
            assert!(ops
                .record_attention_rows(
                    &commands,
                    layout.q,
                    layout.kv_k,
                    layout.kv_v,
                    layout.scores,
                    layout.attn,
                    1,
                    2,
                    2,
                    5,
                    2,
                    1,
                    16,
                    rows
                )
                .is_err());
        }
        let mut short = layout.kv_delta_k;
        short.size = 2 * 3 * 16 * 4 - 1;
        assert!(ops
            .record_kv_write_rows(
                &commands,
                layout.k,
                layout.v,
                layout.kv_k,
                layout.kv_v,
                short,
                layout.kv_delta_v,
                1,
                2,
                2,
                5,
                16,
                3
            )
            .is_err());
        let short = ArenaRegion {
            size: 3 * 32 * 4 - 1,
            ..layout.x
        };
        assert!(ops
            .record_add_rows(&commands, short, layout.projection, 32, 3)
            .is_err());
        assert!(ops
            .record_silu_mul_rows(&commands, short, layout.up, 32, 3)
            .is_err());
    }

    use super::{fill_rope_neox, ArenaLayout, TokenDispatchPlan};

    #[test]
    fn batched_linear_submission_failure_requires_confirmed_idle() {
        use super::super::CommandSubmission;
        use super::VulkanError;
        let mut submission = CommandSubmission::default();
        submission
            .confirm_idle(|| panic!("idle work needs no wait"))
            .unwrap();
        assert!(submission.submit(|| Err(VulkanError::Timeout)).is_err());
        assert!(submission.uncertain);
        assert!(submission
            .confirm_idle(|| Err(VulkanError::Timeout))
            .is_err());
        assert!(submission.uncertain);
        submission.confirm_idle(|| Ok(())).unwrap();
        assert!(!submission.uncertain);
        submission.submit(|| Ok(())).unwrap();
        assert!(!submission.uncertain);
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn batched_linear_device_begin_failure_preserves_state() {
        use super::{BatchedLinearRuntime, GpuWeightFormat::F32, TokenCommands, VulkanError};
        let context = Box::leak(Box::new(super::VulkanContext::new().unwrap()));
        let weight = [0u8; 16];
        let mut runtime = BatchedLinearRuntime::new(context, 1, 4, 1, 2).unwrap();
        runtime.begin_commands = |_| Err(VulkanError::InitFailed("injected begin failure".into()));
        let before = runtime
            .ops
            .read_bytes(runtime.layout.input, 16)
            .unwrap()
            .to_vec();
        let mut output = [123.0];
        assert!(runtime
            .matmul_rows(&weight, F32, &[1.0; 4], 1, 4, 1, &mut output)
            .is_err());
        assert!(runtime.weights.is_empty());
        assert_eq!(
            runtime.ops.read_bytes(runtime.layout.input, 16).unwrap(),
            before
        );
        assert_eq!(output, [123.0]);
        assert_eq!(context.submission_count(), 0);
        runtime.begin_commands = TokenCommands::begin;
        // A later pre-submit failure retains an owned, reusable upload.
        let arena_size = runtime.ops.arena.size;
        runtime.ops.arena.size = 0;
        assert!(runtime
            .matmul_rows(&weight, F32, &[1.0; 4], 1, 4, 1, &mut output)
            .is_err());
        assert_eq!(runtime.weights.len(), 1);
        assert_eq!(context.submission_count(), 0);
        let key = (weight.as_ptr() as usize, weight.len());
        let buffer = runtime.weights[&key].0.buffer;
        runtime.ops.arena.size = arena_size;
        runtime
            .matmul_rows(&weight, F32, &[1.0; 4], 1, 4, 1, &mut output)
            .unwrap();
        assert_eq!(output, [0.0]);
        assert_eq!(runtime.weights.len(), 1);
        assert_eq!(runtime.weights[&key].0.buffer, buffer);
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn batched_linear_device_failed_recovery_blocks_retry_and_preserves_drop_resources() {
        use super::{BatchedLinearRuntime, GpuWeightFormat::F32, VulkanError};
        let context = Box::leak(Box::new(super::VulkanContext::new().unwrap()));
        let weight = [1.0f32; 4];
        let mut runtime = BatchedLinearRuntime::new(context, 1, 4, 1, 2).unwrap();
        let mut output = [0.0];
        runtime
            .matmul_rows(
                bytemuck::cast_slice(&weight),
                F32,
                &[1.0; 4],
                1,
                4,
                1,
                &mut output,
            )
            .unwrap();
        assert_eq!(output, [4.0]);
        // Simulate a submit/wait failure without intentionally hanging hardware.
        assert!(context
            .mutex
            .lock()
            .unwrap()
            .submit(|| Err(VulkanError::Timeout))
            .is_err());
        context.fail_wait_idle.store(true, super::Ordering::Relaxed);
        let resets = context.command_resets.load(super::Ordering::Relaxed);
        let before = runtime
            .ops
            .read_bytes(runtime.layout.input, 16)
            .unwrap()
            .to_vec();
        for _ in 0..2 {
            assert!(runtime
                .matmul_rows(
                    bytemuck::cast_slice(&weight),
                    F32,
                    &[2.0; 4],
                    1,
                    4,
                    1,
                    &mut output
                )
                .is_err());
            assert!(context.mutex.lock().unwrap().uncertain);
            assert_eq!(
                context.command_resets.load(super::Ordering::Relaxed),
                resets
            );
            assert_eq!(
                runtime.ops.read_bytes(runtime.layout.input, 16).unwrap(),
                before
            );
            assert_eq!(runtime.weights.len(), 1);
            assert_eq!(output, [4.0]);
            assert_eq!(context.submission_count(), 1);
        }
        context
            .fail_wait_idle
            .store(false, super::Ordering::Relaxed);
        runtime
            .matmul_rows(
                bytemuck::cast_slice(&weight),
                F32,
                &[2.0; 4],
                1,
                4,
                1,
                &mut output,
            )
            .unwrap();
        assert_eq!(output, [8.0]);
        assert!(!context.mutex.lock().unwrap().uncertain);

        context.mutex.lock().unwrap().uncertain = true;
        context.fail_wait_idle.store(true, super::Ordering::Relaxed);
        let waits = context.idle_waits.load(super::Ordering::Relaxed);
        // Keep copies solely to reclaim the deliberately leaked resources after
        // the real device confirms completion. The runtime must not free them.
        let retained_ops = unsafe { std::ptr::read(&*runtime.ops) };
        let retained_weights: Vec<_> = runtime
            .weights
            .values()
            .map(|(buffer, _)| *buffer)
            .collect();
        drop(runtime);
        assert_eq!(context.idle_waits.load(super::Ordering::Relaxed), waits + 1);
        context
            .fail_wait_idle
            .store(false, super::Ordering::Relaxed);
        context
            .recover_commands(&mut context.mutex.lock().unwrap())
            .unwrap();
        assert_eq!(
            retained_ops
                .read_f32(
                    super::ArenaRegion {
                        offset: 0,
                        size: 16
                    },
                    4
                )
                .unwrap(),
            &[2.0; 4]
        );
        for buffer in retained_weights {
            unsafe { context.destroy_buffer(&buffer) };
        }
        drop(retained_ops);
    }

    #[test]
    fn batched_linear_layout_bounds_and_shapes() {
        use super::{BatchedLinearLayout, GpuWeightFormat::*};
        let limits = super::vk::PhysicalDeviceLimits {
            max_compute_work_group_count: [65535; 3],
            ..Default::default()
        };
        let layout = BatchedLinearLayout::new(3, 513, 65).unwrap();
        for format in [F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K] {
            let (block, bytes, _) = format.layout();
            let weight_len = 512 / block * bytes * 65;
            for rows in 1..=3 {
                assert_eq!(
                    layout
                        .validate(
                            &limits,
                            weight_len,
                            format,
                            rows * 512,
                            rows,
                            512,
                            65,
                            rows * 65
                        )
                        .unwrap(),
                    rows * 65
                );
            }
            assert!(layout
                .validate(&limits, weight_len - 1, format, 1536, 3, 512, 65, 195)
                .is_err());
            assert!(layout
                .validate(&limits, weight_len, format, 1535, 3, 512, 65, 195)
                .is_err());
            assert!(layout
                .validate(&limits, weight_len, format, 1536, 3, 512, 65, 194)
                .is_err());
        }
        for (rows, n_in, n_out) in [
            (0, 512, 65),
            (4, 512, 65),
            (3, 0, 65),
            (3, 514, 65),
            (3, 512, 0),
            (3, 512, 66),
            (usize::MAX, 512, 65),
        ] {
            assert!(layout
                .validate(&limits, 133120, F32, 1536, rows, n_in, n_out, 195)
                .is_err());
        }
        assert!(layout
            .validate(&limits, 133120, Q8_0, 1539, 3, 513, 65, 195)
            .is_err());
        let narrow_limits = super::vk::PhysicalDeviceLimits {
            max_compute_work_group_count: [65535, 65535, 2],
            ..Default::default()
        };
        assert!(layout
            .validate(&narrow_limits, 133120, F32, 1536, 3, 512, 65, 195)
            .is_err());
        for maxima in [
            (0, 512, 65),
            (3, 0, 65),
            (3, 512, 0),
            (usize::MAX, 512, 65),
            (3, usize::MAX, 65),
            (3, 512, usize::MAX),
        ] {
            assert!(BatchedLinearLayout::new(maxima.0, maxima.1, maxima.2).is_err());
        }
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn batched_linear_device_shared_context_recovers_across_runtimes_and_legacy() {
        use super::{BatchedLinearRuntime, GpuWeightFormat::F32, Ordering};
        let context = Box::leak(Box::new(super::VulkanContext::new().unwrap()));
        let weight = [1.0f32; 32];
        let legacy_weight = super::synthetic_q8_weight(32, 1, 3);
        let mut a = BatchedLinearRuntime::new(context, 1, 32, 1, 2).unwrap();
        let mut b = BatchedLinearRuntime::new(context, 1, 32, 1, 2).unwrap();
        for legacy_producer in [false, true] {
            context.fail_fence_wait.store(true, Ordering::Relaxed);
            let result = if legacy_producer {
                unsafe { context.matmul_q8_0(&legacy_weight, &[1; 32], &[1.0], &mut [0.0], 32, 1) }
            } else {
                a.matmul_rows(
                    bytemuck::cast_slice(&weight),
                    F32,
                    &[1.0; 32],
                    1,
                    32,
                    1,
                    &mut [0.0],
                )
            };
            assert!(
                result.is_err(),
                "producer must report the injected uncertainty"
            );
            assert!(context.mutex.lock().unwrap().uncertain);
            let resets = context.command_resets.load(Ordering::Relaxed);
            let submissions = context.submission_count();
            let cache_size = b.weights.len();
            let arena = b.ops.read_bytes(b.layout.input, 128).unwrap().to_vec();
            context.fail_wait_idle.store(true, Ordering::Relaxed);
            let mut output = [123.0];
            assert!(b
                .matmul_rows(
                    bytemuck::cast_slice(&weight),
                    F32,
                    &[2.0; 32],
                    1,
                    32,
                    1,
                    &mut output
                )
                .is_err());
            assert!(unsafe {
                context.matmul_q8_0(&legacy_weight, &[1; 32], &[1.0], &mut [0.0], 32, 1)
            }
            .is_err());
            assert_eq!(context.command_resets.load(Ordering::Relaxed), resets);
            assert_eq!(context.submission_count(), submissions);
            assert_eq!(b.weights.len(), cache_size);
            assert_eq!(b.ops.read_bytes(b.layout.input, 128).unwrap(), arena);
            assert_eq!(output, [123.0]);
            assert!(context.mutex.lock().unwrap().uncertain);
            context.fail_wait_idle.store(false, Ordering::Relaxed);
            b.matmul_rows(
                bytemuck::cast_slice(&weight),
                F32,
                &[2.0; 32],
                1,
                32,
                1,
                &mut output,
            )
            .unwrap();
            assert_eq!(output, [64.0]);
            assert!(!context.mutex.lock().unwrap().uncertain);
            assert_eq!(context.command_resets.load(Ordering::Relaxed), resets + 1);
            let generation = context.current_gen();
            let mut legacy_output = [f32::NAN];
            unsafe {
                context
                    .matmul_q8_0(&legacy_weight, &[1; 32], &[1.0], &mut legacy_output, 32, 1)
                    .unwrap();
            }
            assert!(legacy_output[0].is_finite());
            assert_eq!(context.current_gen(), generation + 1);
        }
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn batched_linear_device_owned_buffers_wait_before_cleanup() {
        use super::{Ordering, VulkanContext};
        let context = Box::leak(Box::new(VulkanContext::new().unwrap()));
        let mut owner = super::super::qwen3::UploadedBuffers::new(context);
        let buffer = owner.upload(&[1, 2, 3, 4]).unwrap();
        context.mutex.lock().unwrap().uncertain = true;
        context.fail_wait_idle.store(true, Ordering::Relaxed);
        drop(owner);
        assert_eq!(context.idle_waits.load(Ordering::Relaxed), 1);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(buffer.mapped, 4) },
            &[1, 2, 3, 4]
        );
        assert!(unsafe { context.destroy_completed_buffers(&[buffer]) }.is_err());
        assert!(context.mutex.lock().unwrap().uncertain);
        context.fail_wait_idle.store(false, Ordering::Relaxed);
        unsafe { context.destroy_completed_buffers(&[buffer]).unwrap() };
        assert!(!context.mutex.lock().unwrap().uncertain);
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn batched_linear_device_rows_match_and_reuse_weights() {
        use super::{BatchedLinearRuntime, GpuWeightFormat::*};
        let context = Box::leak(Box::new(super::VulkanContext::new().unwrap()));
        let mut runtime = BatchedLinearRuntime::new(context, 3, 513, 65, 10).unwrap();
        // Keep every source allocation alive: the cache keys are stable slices.
        let weights: Vec<_> = [F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K]
            .into_iter()
            .map(|format| {
                (
                    format,
                    if format == Q8_0 {
                        super::synthetic_q8_weight(512, 65, 3)
                    } else {
                        super::synthetic_weight(format, 512, 65)
                    },
                )
            })
            .collect();
        for (slot, (format, weight)) in weights.iter().enumerate() {
            let divisor = if matches!(format, F16 | BF16) {
                131072.0
            } else {
                97.0
            };
            let input: Vec<f32> = (0..1536)
                .map(|i| ((i / 512 * 53 + i % 512 * 29) % 251) as f32 / divisor - 125.0 / divisor)
                .collect();
            let before = context.submission_count();
            let mut batched = [f32::NAN; 196];
            runtime
                .matmul_rows(weight, *format, &input, 3, 512, 65, &mut batched)
                .unwrap();
            assert_eq!(context.submission_count(), before + 1);
            assert!(batched[195].is_nan());
            assert_eq!(runtime.weights.len(), slot + 1);
            let key = (weight.as_ptr() as usize, weight.len());
            let buffer = runtime.weights[&key].0.buffer;
            let binding = runtime.weights[&key].1.descriptor_set;
            let mut singles = [0.0; 195];
            for row in 0..3 {
                runtime
                    .matmul_rows(
                        weight,
                        *format,
                        &input[row * 512..(row + 1) * 512],
                        1,
                        512,
                        65,
                        &mut singles[row * 65..(row + 1) * 65],
                    )
                    .unwrap();
            }
            assert_eq!(
                batched[..195]
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                singles.map(f32::to_bits),
                "{format:?}"
            );
            assert!(singles.iter().any(|x| *x != 0.0 && x.is_finite()));
            assert_eq!(runtime.weights.len(), slot + 1);
            assert_eq!(runtime.weights[&key].0.buffer, buffer);
            assert_eq!(runtime.weights[&key].1.descriptor_set, binding);
            let before = context.submission_count();
            let arena_before = runtime
                .ops
                .read_bytes(runtime.layout.input, 32)
                .unwrap()
                .to_vec();
            assert!(runtime
                .matmul_rows(weight, *format, &input, 4, 512, 65, &mut batched)
                .is_err());
            let other_format = if *format == F32 { Q8_0 } else { F32 };
            assert!(runtime
                .matmul_rows(weight, other_format, &input, 3, 512, 65, &mut batched)
                .is_err());
            assert_eq!(context.submission_count(), before);
            assert_eq!(
                runtime.ops.read_bytes(runtime.layout.input, 32).unwrap(),
                arena_before
            );
            assert_eq!(runtime.weights.len(), slot + 1);
            println!("batched_linear format={format:?} rows=3 exact_bits=true cached=true");
        }
        let extra_weight = vec![0u8; 512 * 65 * 4];
        assert!(runtime
            .matmul_rows(
                &extra_weight,
                F32,
                &[0.0; 1536],
                3,
                512,
                65,
                &mut [0.0; 195]
            )
            .is_err());
        assert_eq!(runtime.weights.len(), 9);
    }

    #[test]
    fn batched_matmul_quantize_validates_all_rows_without_tail_padding() {
        use super::{quantize_rows_push, ArenaRegion};
        for block in [32, 256] {
            // Three 256-element rows at stride 259 need exactly 774 floats.
            let regions = [
                ArenaRegion {
                    offset: 0,
                    size: 3096,
                },
                ArenaRegion {
                    offset: 4096,
                    size: 768,
                },
                ArenaRegion {
                    offset: 5120,
                    size: 3 * (256 / block) * 4,
                },
                ArenaRegion {
                    offset: 6144,
                    size: 96,
                },
            ];
            assert!(quantize_rows_push(
                8192,
                regions[0],
                regions[1],
                regions[2],
                Some(regions[3]),
                256,
                3,
                259,
                block
            )
            .is_ok());
            for index in 0..if block == 32 { 4 } else { 3 } {
                let mut short = regions;
                short[index].size -= 1;
                assert!(
                    quantize_rows_push(
                        8192,
                        short[0],
                        short[1],
                        short[2],
                        Some(short[3]),
                        256,
                        3,
                        259,
                        block
                    )
                    .is_err(),
                    "block={block} region={index}"
                );
            }
            assert!(quantize_rows_push(
                8192,
                regions[0],
                regions[1],
                regions[2],
                Some(regions[3]),
                256,
                0,
                259,
                block
            )
            .is_err());
            assert!(quantize_rows_push(
                8192,
                regions[0],
                regions[1],
                regions[2],
                Some(regions[3]),
                256,
                usize::MAX,
                259,
                block
            )
            .is_err());
        }
    }

    #[test]
    fn batched_matmul_dispatch_checks_device_z_limit() {
        let limits = super::vk::PhysicalDeviceLimits {
            max_compute_work_group_count: [65535, 65535, 6],
            ..Default::default()
        };
        assert_eq!(
            super::matmul_dispatch(65, 3, 2, &limits).unwrap(),
            [65, 1, 6]
        );
        assert!(super::matmul_dispatch(65, 3, 3, &limits).is_err());
        assert!(super::matmul_dispatch(65, usize::MAX, 2, &limits).is_err());
        assert_eq!(super::row_dispatch(8, 6, &limits).unwrap(), [8, 1, 6]);
        assert!(super::row_dispatch(8, 7, &limits).is_err());
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn batched_matmul_device_rows_match_single_row_bits() {
        let context = super::VulkanContext::new().unwrap();
        super::run_batched_matmul_check(
            &context,
            &[
                "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q4_k", "q5_k", "q6_k",
            ],
            3,
        )
        .unwrap();
    }

    #[test]
    fn batched_matmul_push_validates_grouped_spans_and_strides() {
        use super::{matmul_rows_push, ArenaRegion, GpuWeightFormat::*, OperatorBindings};
        let limits = super::vk::PhysicalDeviceLimits {
            max_compute_work_group_count: [65535; 3],
            ..Default::default()
        };
        for rows in [1, 2, 3] {
            for format in [F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K] {
                let (block, bytes, _) = format.layout();
                let is_float = matches!(format, F32 | F16 | BF16);
                let mut regions = [
                    ArenaRegion {
                        offset: 0,
                        size: if is_float {
                            ((rows - 1) * 259 + 256) * 4
                        } else {
                            rows * 256
                        },
                    },
                    ArenaRegion {
                        offset: 4096,
                        size: rows * (256 / block) * 4,
                    },
                    ArenaRegion {
                        offset: 8192,
                        size: rows * (256 / block) * 4,
                    },
                    ArenaRegion {
                        offset: 12288,
                        size: (rows - 1) * 20 + 12,
                    },
                    ArenaRegion {
                        offset: 13312,
                        size: (rows - 1) * 16 + 8,
                    },
                    ArenaRegion {
                        offset: 14336,
                        size: (rows - 1) * 8 + 4,
                    },
                ];
                let bindings = OperatorBindings {
                    descriptor_set: super::vk::DescriptorSet::null(),
                    sizes: [((256 / block) * bytes * 3) as u64; 3],
                    weight_formats: [Some(format); 3],
                };
                let prepare = |r: &[ArenaRegion; 6]| {
                    matmul_rows_push(
                        16384,
                        &limits,
                        bindings,
                        r[0],
                        r[1],
                        Some(r[2]),
                        &[(r[3], 3, 20), (r[4], 2, 16), (r[5], 1, 8)],
                        256,
                        rows,
                        259,
                    )
                };
                assert_eq!(prepare(&regions).unwrap().1, [3, 1, (rows * 3) as u32]);
                for index in [0, 1, 2, 3, 4, 5] {
                    if (index == 1 && is_float) || (index == 2 && format != Q4_1) {
                        continue;
                    }
                    let mut short = regions;
                    short[index].size -= 1;
                    assert!(
                        prepare(&short).is_err(),
                        "{format:?} rows={rows} region={index}"
                    );
                }
                let output = [(regions[3], 3, 8)];
                assert!(matmul_rows_push(
                    16384,
                    &limits,
                    bindings,
                    regions[0],
                    regions[1],
                    Some(regions[2]),
                    &output,
                    256,
                    rows,
                    259
                )
                .is_err());
                regions[0].offset = usize::MAX - 3;
                assert!(prepare(&regions).is_err());
            }
        }
    }

    #[test]
    fn batched_matmul_span_rejects_gpu_address_overflow() {
        use super::{row_word, ArenaRegion};
        assert!(row_word(
            usize::MAX,
            ArenaRegion {
                offset: usize::MAX - 3,
                size: 16
            },
            1,
            16,
            16,
            "input"
        )
        .is_err());
        let offset = (u32::MAX as usize) * 4;
        assert!(row_word(
            usize::MAX,
            ArenaRegion { offset, size: 8 },
            2,
            4,
            4,
            "input"
        )
        .is_err());
        assert!(row_word(
            64,
            ArenaRegion {
                offset: 4,
                size: 64
            },
            2,
            32,
            32,
            "input"
        )
        .is_err());
        assert!(row_word(
            64,
            ArenaRegion {
                offset: 0,
                size: 64
            },
            3,
            7,
            4,
            "output"
        )
        .is_err());
    }
    #[test]
    fn batched_matmul_dispatch_maps_output_weight_and_token_rows() {
        let dispatch = super::matmul_dispatch_for_test(65, 3, 2).unwrap();
        assert_eq!(dispatch, [65, 1, 6]);
    }

    #[test]
    fn batched_matmul_dispatch_rejects_empty_and_overflow_shapes() {
        assert!(super::matmul_dispatch_for_test(0, 1, 1).is_err());
        assert!(super::matmul_dispatch_for_test(1, 0, 1).is_err());
        assert!(super::matmul_dispatch_for_test(1, 1, 4).is_err());
        assert!(super::matmul_dispatch_for_test(usize::MAX, 2, 2).is_err());
        assert_eq!(super::matmul_dispatch_for_test(3, 1, 1).unwrap(), [3, 1, 1]);
        assert_eq!(super::matmul_dispatch_for_test(3, 2, 3).unwrap(), [3, 1, 6]);
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn q5_k_matvec_runs_on_vulkan() {
        let context = super::VulkanContext::new().unwrap();
        super::check_weight_format(&context, "q5_k").unwrap();
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn wide_quantized_matvec_runs_on_vulkan() {
        use super::GpuWeightFormat::*;
        let context = super::VulkanContext::new().unwrap();
        for format in [Q4_0, Q4_1, Q4_K, Q5_K, Q6_K] {
            super::check_weight_matvec(
                &context,
                &format!("{format:?} width=17408"),
                format,
                17_408,
                65,
            )
            .unwrap();
        }
    }

    #[test]
    fn vulkan_rope_coefficients_match_cpu_dimension_formula() {
        let mut actual = [0.0f32; 128];
        fill_rope_neox(&mut actual, 4, 1_000_000.0);
        for index in 0..64 {
            let theta =
                4.0 * (1.0f32 / 1_000_000.0f32.powf((2 * index) as f32 / actual.len() as f32));
            let (cosine, sine) = crate::ops::rope_sin_cos(theta);
            assert_eq!(actual[index].to_bits(), cosine.to_bits(), "cosine {index}");
            assert_eq!(actual[index + 64].to_bits(), sine.to_bits(), "sine {index}");
        }
    }

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
