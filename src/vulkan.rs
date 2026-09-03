//! Vulkan compute backend for Q8_0 matmul.
//!
//! v2 architecture (2026-08): weights are uploaded **once** into persistently
//! mapped device buffers and cached by `(data_ptr, len)` — mmap weight slices
//! are stable for the process lifetime, so every matmul after the first is a
//! cache hit. Input/scale/output buffers are persistent and grown on demand.
//! One dispatch covers **all** output rows (callers must route the full-range
//! matmul here from a single thread — see `parallel::matmul_q8_0_quantized_
//! parallel_rows`, where thread 0 submits and the rest return immediately).
//!
//! Memory: buffers use HOST_VISIBLE | HOST_COHERENT (preferring DEVICE_LOCAL
//! when the heap offers it). On UMA iGPUs (Intel Xe) this is zero-copy system
//! memory; discrete GPUs later want a staging → DEVICE_LOCAL upload path.

#[cfg(feature = "vulkan")]
use ash::vk;
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[cfg(feature = "vulkan")]
const SHADER: &[u8] = include_bytes!("../shaders/bin/q8_matmul.spv");

/// Set when a GPU matmul fails at runtime: all subsequent calls fall back to
/// CPU without retrying the broken path.
#[cfg(feature = "vulkan")]
static GPU_BROKEN: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "vulkan")]
pub fn gpu_broken() -> bool {
    GPU_BROKEN.load(Ordering::Relaxed)
}

#[cfg(feature = "vulkan")]
pub fn mark_gpu_broken(reason: &str) {
    if !GPU_BROKEN.swap(true, Ordering::Relaxed) {
        eprintln!("[GPU] Vulkan disabled after error: {reason}. Falling back to CPU.");
    }
}

#[cfg(feature = "vulkan")]
pub struct VulkanContext {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    _physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family: u32,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    /// Weights keyed by (data_ptr, byte_len); mmap slices are stable, so the
    /// first matmul over a tensor uploads it and every later call is a hit.
    weight_cache: Mutex<HashMap<(usize, usize), BufferInfo>>,
    /// Persistent I/O buffers, grown on demand.
    io_state: Mutex<IoState>,
    mutex: Mutex<()>,
    device_name: String,
    /// Completed-matmul generation. Thread 0 bumps it after the fence wait;
    /// other pool threads block on it before touching the matmul output
    /// (their post-matmul work — silu, residual — must see the GPU result).
    completed_gen: std::sync::atomic::AtomicU64,
}

/// Persistent activation staging buffers. Capacities grow to the largest
/// matmul seen; sizes shrink rarely in practice (vocab dominates).
#[cfg(feature = "vulkan")]
struct IoState {
    input_q8: Option<BufferInfo>,
    scales: Option<BufferInfo>,
    output: Option<BufferInfo>,
}

#[cfg(feature = "vulkan")]
impl Default for IoState {
    fn default() -> Self {
        Self {
            input_q8: None,
            scales: None,
            output: None,
        }
    }
}

#[cfg(feature = "vulkan")]
struct BufferInfo {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    mapped: *mut u8,
}

#[cfg(feature = "vulkan")]
impl VulkanContext {
    pub fn new() -> Result<Self, VulkanError> {
        unsafe {
            let entry = ash::Entry::load()
                .or_else(|error| {
                    #[cfg(target_os = "macos")]
                    for path in [
                        "/opt/homebrew/lib/libvulkan.dylib",
                        "/usr/local/lib/libvulkan.dylib",
                    ] {
                        if let Ok(entry) = ash::Entry::load_from(path) {
                            return Ok(entry);
                        }
                    }
                    Err(error)
                })
                .map_err(|e| VulkanError::InitFailed(e.to_string()))?;
            let instance = Self::create_instance(&entry)?;
            let (physical_device, queue_family, device_name) =
                Self::select_physical_device(&instance)?;
            let device = Self::create_device(&instance, physical_device, queue_family)?;
            let queue = device.get_device_queue(queue_family, 0);
            let (pipeline_layout, descriptor_set_layout, descriptor_pool, descriptor_set) =
                Self::create_compute_pipeline(&device)?;
            let pipeline = Self::create_pipeline(&device, pipeline_layout)?;
            let (command_pool, command_buffer, fence) =
                Self::create_command_pool(&device, queue_family)?;

            eprintln!("[GPU] Vulkan device: {device_name}");

            Ok(Self {
                entry,
                instance,
                device,
                _physical_device: physical_device,
                queue,
                queue_family,
                pipeline,
                pipeline_layout,
                descriptor_set_layout,
                descriptor_pool,
                descriptor_set,
                command_pool,
                command_buffer,
                fence,
                weight_cache: Mutex::new(HashMap::new()),
                io_state: Mutex::new(IoState::default()),
                mutex: Mutex::new(()),
                device_name,
                completed_gen: std::sync::atomic::AtomicU64::new(0),
            })
        }
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Run one tiny matmul and wait for it. The driver JITs the compute
    /// pipeline on first dispatch (seconds on Meteor Lake) — absorbing that
    /// here prevents the dispatch watchdog from abandoning the first real
    /// matmul of an inference run.
    pub fn warmup(&self) -> Result<(), VulkanError> {
        let n_in = 32usize;
        let n_out = 32usize;
        let mut weight = vec![0u8; n_out * (n_in / 32) * 34];
        let input = vec![0u8; n_in];
        let scales = vec![0.0f32; n_in / 32];
        let mut out = vec![0.0f32; n_out];
        unsafe { self.matmul_q8_0(&weight, &input, &scales, &mut out, n_in, n_out) }
    }

    /// Snapshot of the completed-matmul generation; pair with [`Self::wait_for_gen`].
    pub fn current_gen(&self) -> u64 {
        self.completed_gen.load(Ordering::Acquire)
    }

    /// Force-release waiters after a failed matmul (used with `mark_gpu_broken`).
    pub fn release_gen(&self) {
        self.completed_gen
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    /// Block until the completed-matmul generation reaches `gen` (or the GPU
    /// is marked broken, releasing the waiters to the CPU fallback).
    pub fn wait_for_gen(&self, gen: u64) {
        loop {
            if self.completed_gen.load(Ordering::Acquire) >= gen {
                return;
            }
            if GPU_BROKEN.load(Ordering::Relaxed) {
                return;
            }
            std::hint::spin_loop();
        }
    }

    /// True once `matmul_q8_0` succeeded at least once (smoke-tested pipeline).
    pub unsafe fn matmul_q8_0(
        &self,
        weight: &[u8],
        input_q8: &[u8],
        input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> Result<(), VulkanError> {
        let _lock = self.mutex.lock().unwrap();
        let blocks_per_row = n_in / 32;
        if blocks_per_row > 512 {
            // shader stages the input row in 4096 shared words (16 KiB);
            // UnsupportedShape lets the caller fall back to CPU for this
            // matmul while keeping the GPU alive for the rest.
            return Err(VulkanError::UnsupportedShape(format!(
                "n_in {n_in} exceeds shader shared-memory capacity"
            )));
        }
        debug_assert_eq!(weight.len(), n_out * blocks_per_row * 34);
        debug_assert_eq!(input_q8.len(), n_in);
        debug_assert_eq!(input_scales.len(), blocks_per_row);
        debug_assert_eq!(output.len(), n_out);

        // 1. Weight buffer: cache by (ptr, len); upload on first sight.
        let weight_buffer = self.weight_for(weight)?;

        // 2. Persistent I/O buffers: grow if this matmul is bigger than before.
        let mut io = self.io_state.lock().unwrap();
        let input_buffer = self.ensure_buffer(&mut io.input_q8, input_q8.len())?;
        let scale_buffer = self.ensure_buffer(&mut io.scales, input_scales.len() * 4)?;
        let output_buffer = self.ensure_buffer(&mut io.output, output.len() * 4)?;

        // 3. Stage inputs into the persistently mapped buffers.
        let in_map = input_buffer.mapped;
        std::ptr::copy_nonoverlapping(input_q8.as_ptr(), in_map, input_q8.len());
        let scale_map = scale_buffer.mapped as *mut f32;
        std::ptr::copy_nonoverlapping(input_scales.as_ptr(), scale_map, input_scales.len());

        // 4. Bind and dispatch.
        self.update_descriptor_sets(
            weight_buffer.buffer,
            input_buffer.buffer,
            scale_buffer.buffer,
            output_buffer.buffer,
        )?;

        self.device
            .reset_command_buffer(
                self.command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;
        self.device
            .begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    p_inheritance_info: std::ptr::null(),
                },
            )
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

        self.device.cmd_bind_pipeline(
            self.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline,
        );
        self.device.cmd_bind_descriptor_sets(
            self.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            0,
            &[self.descriptor_set],
            &[],
        );

        // dims = [n_in, n_out, blocks_per_row, row_stride_bytes]; the shader
        // addresses `weight` with byte offsets extracted from u32 words.
        let push_constants: [u32; 4] = [
            n_in as u32,
            n_out as u32,
            blocks_per_row as u32,
            (blocks_per_row * 34) as u32,
        ];
        self.device.cmd_push_constants(
            self.command_buffer,
            self.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push_constants),
        );
        self.device
            .cmd_dispatch(self.command_buffer, n_out as u32, 1, 1);

        self.device
            .end_command_buffer(self.command_buffer)
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

        self.device
            .reset_fences(std::slice::from_ref(&self.fence))
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;
        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffer,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
        };
        self.device
            .queue_submit(self.queue, &[submit_info], self.fence)
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;
        // 60s: the driver JITs shaders on first dispatch (observed >5 s on
        // Meteor Lake), so this timeout only catches true GPU hangs. The
        // outer watchdog (5 s) abandons wedged calls long before this fires.
        self.device
            .wait_for_fences(std::slice::from_ref(&self.fence), true, 60_000_000_000u64)
            .map_err(|_| VulkanError::Timeout)?;
        self.completed_gen
            .fetch_add(1, std::sync::atomic::Ordering::Release);

        // 5. Read results out of the persistently mapped output buffer.
        let out_map = output_buffer.mapped as *const f32;
        std::ptr::copy_nonoverlapping(out_map, output.as_mut_ptr(), n_out);

        Ok(())
    }

    // ---- buffer helpers -------------------------------------------------

    // `slot` lives in the mutex-guarded IoState, so &mut here does not
    // alias `&self` buffers.
    unsafe fn ensure_buffer(
        &self,
        slot: &mut Option<BufferInfo>,
        size: usize,
    ) -> Result<BufferInfo, VulkanError> {
        if slot.as_ref().is_some_and(|b| b.size as usize >= size) {
            return Ok(*slot.as_ref().unwrap());
        }
        // Grow: round up to the next MiB to amortize reallocation.
        let new_size = size.next_power_of_two().max(1 << 20) as u64;
        if let Some(old) = slot.take() {
            self.destroy_buffer(&old);
        }
        let buf = self.alloc_persistently_mapped(new_size)?;
        *slot = Some(buf);
        Ok(*slot.as_ref().unwrap())
    }

    unsafe fn weight_for(&self, weight: &[u8]) -> Result<BufferInfo, VulkanError> {
        let key = (weight.as_ptr() as usize, weight.len());
        if let Some(buf) = self.weight_cache.lock().unwrap().get(&key) {
            return Ok(*buf);
        }
        // +16 bytes of zero padding: the shader's speculative second-word load
        // for the last block of the last row can read one word past the tensor.
        let buf = self.alloc_persistently_mapped(weight.len() as u64 + 16)?;
        std::ptr::copy_nonoverlapping(weight.as_ptr(), buf.mapped, weight.len());
        self.weight_cache.lock().unwrap().insert(key, buf);
        Ok(*self.weight_cache.lock().unwrap().get(&key).unwrap())
    }

    unsafe fn alloc_persistently_mapped(&self, size: u64) -> Result<BufferInfo, VulkanError> {
        let buffer = self
            .device
            .create_buffer(
                &vk::BufferCreateInfo {
                    s_type: vk::StructureType::BUFFER_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: Default::default(),
                    size,
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    queue_family_index_count: 0,
                    p_queue_family_indices: std::ptr::null(),
                },
                None,
            )
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

        let mem_reqs = self.device.get_buffer_memory_requirements(buffer);
        // iGPU/UMA note: prefer plain HOST_VISIBLE|HOST_COHERENT (cached
        // system RAM). The DEVICE_LOCAL heap on Intel iGPUs is the small
        // stolen-memory carve-out — uncached and easily exhausted by
        // model-scale buffers (observed driver hangs when weights landed
        // there). For discrete GPUs revisit with a staging upload path.
        let mem_type = self
            .find_memory_type(
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or(VulkanError::OutOfMemory)?;

        let memory = self
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo {
                    s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                    p_next: std::ptr::null(),
                    allocation_size: mem_reqs.size,
                    memory_type_index: mem_type,
                },
                None,
            )
            .map_err(|e| VulkanError::OutOfMemory)?;

        self.device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

        let mapped = self
            .device
            .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
            .map_err(|e| VulkanError::OutOfMemory)? as *mut u8;

        Ok(BufferInfo {
            buffer,
            memory,
            size,
            mapped,
        })
    }

    unsafe fn destroy_buffer(&self, buf: &BufferInfo) {
        self.device.unmap_memory(buf.memory);
        self.device.destroy_buffer(buf.buffer, None);
        self.device.free_memory(buf.memory, None);
    }

    fn create_instance(entry: &ash::Entry) -> Result<ash::Instance, VulkanError> {
        let engine_name = CStr::from_bytes_with_nul(b"rust-model-inference\0").unwrap();
        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: std::ptr::null(),
            application_version: 0,
            p_engine_name: engine_name.as_ptr(),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::API_VERSION_1_3,
        };

        let portability_name = vk::KhrPortabilityEnumerationFn::name();
        let portability_available = extension_available(
            &entry
                .enumerate_instance_extension_properties(None)
                .map_err(|e| VulkanError::InitFailed(e.to_string()))?,
            portability_name,
        );
        let extension_names = [portability_name.as_ptr()];
        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            p_application_info: &app_info,
            flags: if portability_available {
                vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
            } else {
                vk::InstanceCreateFlags::empty()
            },
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: portability_available as u32,
            pp_enabled_extension_names: extension_names.as_ptr(),
        };

        unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(|e| VulkanError::InitFailed(e.to_string()))
        }
    }

    /// Pick the best compute device: discrete > integrated > software/CPU.
    fn select_physical_device(
        instance: &ash::Instance,
    ) -> Result<(vk::PhysicalDevice, u32, String), VulkanError> {
        unsafe {
            let devices = instance
                .enumerate_physical_devices()
                .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

            let mut best: Option<(u8, vk::PhysicalDevice, u32, String)> = None;
            for device in devices {
                let props = instance.get_physical_device_properties(device);
                let name = CStr::from_ptr(props.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned();
                let type_score = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 3u8,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                    _ => 0, // CPU (lavapipe) and unknown
                };
                for (i, family) in instance
                    .get_physical_device_queue_family_properties(device)
                    .iter()
                    .enumerate()
                {
                    if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                        let candidate = (type_score, device, i as u32, name.clone());
                        if best.as_ref().is_none_or(|(s, ..)| type_score > *s) {
                            best = Some(candidate);
                        }
                        break;
                    }
                }
            }

            best.map(|(_, d, q, name)| (d, q, name))
                .ok_or(VulkanError::NoComputeDevice)
        }
    }

    fn create_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
    ) -> Result<ash::Device, VulkanError> {
        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            queue_family_index: queue_family,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
        };

        let features = vk::PhysicalDeviceFeatures {
            shader_int64: 1,
            ..Default::default()
        };
        let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features {
            shader_integer_dot_product: 1,
            ..Default::default()
        };

        let portability_name = vk::KhrPortabilitySubsetFn::name();
        let portability_available = extension_available(
            &unsafe {
                instance
                    .enumerate_device_extension_properties(physical_device)
                    .map_err(|e| VulkanError::InitFailed(e.to_string()))?
            },
            portability_name,
        );
        let extension_names = [portability_name.as_ptr()];

        let device_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            p_next: &vulkan13_features as *const _ as *const std::os::raw::c_void,
            flags: Default::default(),
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: portability_available as u32,
            pp_enabled_extension_names: extension_names.as_ptr(),
            p_enabled_features: &features,
            ..Default::default()
        };

        unsafe {
            instance
                .create_device(physical_device, &device_info, None)
                .map_err(|e| VulkanError::InitFailed(e.to_string()))
        }
    }

    fn create_compute_pipeline(
        device: &ash::Device,
    ) -> Result<
        (
            vk::PipelineLayout,
            vk::DescriptorSetLayout,
            vk::DescriptorPool,
            vk::DescriptorSet,
        ),
        VulkanError,
    > {
        let binding_descs = [
            storage_binding(0),
            storage_binding(1),
            storage_binding(2),
            storage_binding(3),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            binding_count: 4,
            p_bindings: binding_descs.as_ptr(),
        };

        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(|e| VulkanError::InitFailed(e.to_string()))?
        };

        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 16, // 4 * u32 = 16 bytes
        };

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo {
                        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        set_layout_count: 1,
                        p_set_layouts: &descriptor_set_layout,
                        push_constant_range_count: 1,
                        p_push_constant_ranges: &push_constant_range,
                    },
                    None,
                )
                .map_err(|e| VulkanError::InitFailed(e.to_string()))?
        };

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 4,
        }];

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo {
                        s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        max_sets: 1,
                        pool_size_count: 1,
                        p_pool_sizes: pool_sizes.as_ptr(),
                    },
                    None,
                )
                .map_err(|e| VulkanError::InitFailed(e.to_string()))?
        };

        let descriptor_set = unsafe {
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: &descriptor_set_layout,
            };
            device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| VulkanError::InitFailed(e.to_string()))?[0]
        };

        Ok((
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
        ))
    }

    fn create_pipeline(
        device: &ash::Device,
        pipeline_layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline, VulkanError> {
        let shader_module = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        code_size: SHADER.len(),
                        p_code: SHADER.as_ptr() as *const u32,
                    },
                    None,
                )
                .map_err(|e| VulkanError::ShaderCompileFailed(e.to_string()))?
        };

        let stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: CStr::from_bytes_with_nul(b"main\0").unwrap().as_ptr(),
            p_specialization_info: std::ptr::null(),
        };

        let pipeline = unsafe {
            let create_info = vk::ComputePipelineCreateInfo {
                s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: Default::default(),
                stage,
                layout: pipeline_layout,
                base_pipeline_handle: vk::Pipeline::null(),
                base_pipeline_index: 0,
            };
            match device.create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None) {
                Ok(pipelines) => pipelines[0],
                Err((_, _)) => {
                    return Err(VulkanError::ShaderCompileFailed(
                        "Pipeline creation failed".to_string(),
                    ))
                }
            }
        };

        unsafe {
            device.destroy_shader_module(shader_module, None);
        }

        Ok(pipeline)
    }

    fn create_command_pool(
        device: &ash::Device,
        queue_family: u32,
    ) -> Result<(vk::CommandPool, vk::CommandBuffer, vk::Fence), VulkanError> {
        unsafe {
            let command_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo {
                        s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                        queue_family_index: queue_family,
                    },
                    None,
                )
                .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

            let command_buffer = device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                    p_next: std::ptr::null(),
                    command_pool,
                    level: vk::CommandBufferLevel::PRIMARY,
                    command_buffer_count: 1,
                })
                .map_err(|e| VulkanError::InitFailed(e.to_string()))?[0];

            let fence = device
                .create_fence(
                    &vk::FenceCreateInfo {
                        s_type: vk::StructureType::FENCE_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                    },
                    None,
                )
                .map_err(|e| VulkanError::InitFailed(e.to_string()))?;

            Ok((command_pool, command_buffer, fence))
        }
    }

    unsafe fn update_descriptor_sets(
        &self,
        weight_buffer: vk::Buffer,
        input_buffer: vk::Buffer,
        scale_buffer: vk::Buffer,
        output_buffer: vk::Buffer,
    ) -> Result<(), VulkanError> {
        let make_info = |buffer: vk::Buffer| vk::DescriptorBufferInfo {
            buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        };
        let weight_info = make_info(weight_buffer);
        let input_info = make_info(input_buffer);
        let scale_info = make_info(scale_buffer);
        let output_info = make_info(output_buffer);

        let mk_write = |binding: u32, info: &vk::DescriptorBufferInfo| vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: std::ptr::null(),
            dst_set: self.descriptor_set,
            dst_binding: binding,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_image_info: std::ptr::null(),
            p_buffer_info: info,
            p_texel_buffer_view: std::ptr::null(),
        };

        let desc_writes = [
            mk_write(0, &weight_info),
            mk_write(1, &input_info),
            mk_write(2, &scale_info),
            mk_write(3, &output_info),
        ];

        self.device.update_descriptor_sets(&desc_writes, &[]);

        Ok(())
    }

    fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        unsafe {
            let mem_props = self
                .instance
                .get_physical_device_memory_properties(self._physical_device);
            for i in 0..mem_props.memory_type_count {
                if (type_filter & (1 << i)) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(properties)
                {
                    return Some(i);
                }
            }
            None
        }
    }
}

// SAFETY: mapped device memory is host-addressable process-wide; all use of
// the mapped pointers is serialized by `VulkanContext::mutex`.
#[cfg(feature = "vulkan")]
unsafe impl Send for BufferInfo {}
#[cfg(feature = "vulkan")]
unsafe impl Sync for BufferInfo {}

#[cfg(feature = "vulkan")]
impl Clone for BufferInfo {
    fn clone(&self) -> Self {
        *self
    }
}

#[cfg(feature = "vulkan")]
impl Copy for BufferInfo {}

#[cfg(feature = "vulkan")]
impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            for (_, buf) in self.weight_cache.lock().unwrap().drain() {
                self.destroy_buffer(&buf);
            }
            let IoState {
                input_q8,
                scales,
                output,
            } = {
                let mut io = self.io_state.lock().unwrap();
                std::mem::take(&mut *io)
            };
            for buf in [input_q8, scales, output].into_iter().flatten() {
                self.destroy_buffer(&buf);
            }
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(feature = "vulkan")]
fn storage_binding(binding: u32) -> vk::DescriptorSetLayoutBinding {
    vk::DescriptorSetLayoutBinding {
        binding,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        p_immutable_samplers: std::ptr::null(),
    }
}

#[cfg(feature = "vulkan")]
fn extension_available(properties: &[vk::ExtensionProperties], name: &CStr) -> bool {
    properties
        .iter()
        .any(|property| unsafe { CStr::from_ptr(property.extension_name.as_ptr()) == name })
}

#[derive(Debug)]
pub enum VulkanError {
    InitFailed(String),
    NoComputeDevice,
    ShaderCompileFailed(String),
    OutOfMemory,
    /// Shape the shader cannot execute (e.g. n_in beyond shared-memory
    /// capacity) — the caller should fall back to CPU for this matmul only.
    UnsupportedShape(String),
    /// The GPU did not complete the dispatch within the timeout.
    Timeout,
}

impl std::fmt::Display for VulkanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulkanError::InitFailed(s) => write!(f, "Vulkan init failed: {}", s),
            VulkanError::NoComputeDevice => write!(f, "No compute device found"),
            VulkanError::ShaderCompileFailed(s) => write!(f, "Shader compile failed: {}", s),
            VulkanError::OutOfMemory => write!(f, "Out of GPU memory"),
            VulkanError::UnsupportedShape(s) => write!(f, "Unsupported shape: {}", s),
            VulkanError::Timeout => write!(f, "GPU dispatch timed out"),
        }
    }
}

#[cfg(all(test, feature = "vulkan", target_os = "macos"))]
mod tests {
    use super::VulkanContext;
    use std::path::Path;

    #[test]
    fn initializes_with_homebrew_moltenvk() {
        if ![
            "/opt/homebrew/lib/libvulkan.dylib",
            "/usr/local/lib/libvulkan.dylib",
        ]
        .iter()
        .any(|path| Path::new(path).exists())
        {
            eprintln!("skipping: no Homebrew Vulkan loader installed");
            return;
        }

        let context = VulkanContext::new().expect("Homebrew MoltenVK should initialize");
        assert!(!context.device_name().is_empty());
    }
}
