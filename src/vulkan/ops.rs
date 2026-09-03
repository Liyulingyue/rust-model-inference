use super::{VulkanContext, VulkanError};
use crate::models::qwen3::trunk::Qwen3Config;
use ash::vk;
use std::sync::atomic::Ordering;
use std::sync::MutexGuard;

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
            scores,
            kv_k,
            kv_v,
            kv_delta_k,
            kv_delta_v,
            total_size: cursor,
        })
    }

    pub(crate) fn regions(&self) -> [ArenaRegion; 18] {
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
